"""
Node Model for FEP Analysis

This module provides a clean, type-safe implementation of the node model for
Free Energy Perturbation (FEP) analysis using Pyro for probabilistic inference.

The model uses Pydantic for configuration validation and provides a simple
interface for running MAP (Maximum A Posteriori) inference on graph-structured
FEP data.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoNormal

# Import dataset classes
from ..dataset import BaseDataset
# Import configuration classes
from .model_config import NodeModelConfig, PriorType, GuideType, ErrorDistributionType


# Configuration classes have been moved to model_config.py
# Import them from there instead


@dataclass
class GraphData:
    """Container for graph-structured FEP data"""

    source_nodes: List[int]
    target_nodes: List[int]
    edge_values: List[float]
    num_nodes: int
    num_edges: int
    node_to_idx: Dict[str, int]
    idx_to_node: Dict[int, str]


class NodeModel:
    """
    Node Model for FEP Analysis

    This class implements a clean, type-safe node model for FEP analysis
    using Pyro for probabilistic inference. It provides MAP estimation
    of node values from edge-based FEP measurements.

    Attributes
    ----------
    config : NodeModelConfig
        Configuration object containing all model parameters
    dataset : BaseDataset
        Dataset containing the FEP data
    graph_data : Optional[GraphData]
        Processed graph data for inference
    node_estimates : Optional[Dict[str, float]]
        Estimated node values after training
    edge_estimates : Optional[Dict[Tuple[str, str], float]]
        Estimated edge values derived from node estimates
    """

    def __init__(self, config: NodeModelConfig, dataset: BaseDataset):
        """
        Initialize the node model with configuration and dataset.

        Parameters
        ----------
        config : NodeModelConfig
            Configuration object with all model parameters
        dataset : BaseDataset
            Dataset containing FEP data
        """
        self.config = config
        self.dataset = dataset
        self.graph_data: Optional[GraphData] = None
        self.node_estimates: Optional[Dict[str, float]] = None
        self.edge_estimates: Optional[Dict[Tuple[str, str], float]] = None
        self.node_uncertainties: Optional[Dict[str, float]] = None
        self.edge_uncertainties: Optional[Dict[Tuple[str, str], float]] = None

        # Training history
        self.loss_history: List[float] = []
        self.elbo_history: List[float] = []

        # Optionally calculate CCC values using CCC_model
        # Note: CCC calculation removed from automatic initialization
        # Use CCC_model directly if CCC values are needed as benchmark


    def _create_prior(self, num_nodes: int) -> dist.Distribution:
        """
        Create the prior distribution for node values.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph

        Returns
        -------
        dist.Distribution
            Pyro distribution for the prior
        """
        params = self.config.prior_parameters

        if self.config.prior_type == PriorType.NORMAL:
            return dist.Normal(torch.tensor(params[0]), torch.tensor(params[1]))
        elif self.config.prior_type == PriorType.GAMMA:
            return dist.Gamma(torch.tensor(params[0]), torch.tensor(params[1]))
        elif self.config.prior_type == PriorType.UNIFORM:
            return dist.Uniform(torch.tensor(params[0]), torch.tensor(params[1]))
        elif self.config.prior_type == PriorType.STUDENT_T:
            return dist.StudentT(
                torch.tensor(params[0]), torch.tensor(0.0), torch.tensor(params[1])
            )
        elif self.config.prior_type == PriorType.LAPLACE:
            return dist.Laplace(torch.tensor(params[0]), torch.tensor(params[1]))
        else:
            raise ValueError(f"Unsupported prior type: {self.config.prior_type}")

    def _extract_graph_data(self) -> GraphData:
        """
        Extract and process graph data from the dataset.

        Returns
        -------
        GraphData
            Processed graph data for inference
        """
        # Use standardized dataset attribute names
        edges_df = getattr(self.dataset, "dataset_edges", None)

        if edges_df is None:
            raise ValueError("Dataset must have 'dataset_edges' attribute")

        source_nodes = []
        target_nodes = []
        edge_values = []
        node_to_idx = {}
        idx_to_node = {}
        idx = 0

        # Process each edge
        for _, row in edges_df.iterrows():
            # Handle both old and new column names
            if "Source" in edges_df.columns and "Destination" in edges_df.columns:
                ligand1, ligand2 = row["Source"], row["Destination"]
            elif "Ligand1" in edges_df.columns and "Ligand2" in edges_df.columns:
                ligand1, ligand2 = row["Ligand1"], row["Ligand2"]
            else:
                raise ValueError(
                    "Edge data must have either 'Source'/'Destination' or 'Ligand1'/'Ligand2' columns"
                )

            # Handle both old and new column names for FEP values
            if "DeltaDeltaG" in edges_df.columns:
                fep_value = float(row["DeltaDeltaG"])
            elif "DeltaG" in edges_df.columns:
                fep_value = float(row["DeltaG"])
            elif "FEP" in edges_df.columns:
                fep_value = float(row["FEP"])
            else:
                raise ValueError(
                    "Edge data must have either 'DeltaDeltaG', 'DeltaG', or 'FEP' column"
                )

            # Add nodes to mapping if not seen before
            if ligand1 not in node_to_idx:
                node_to_idx[ligand1] = idx
                idx_to_node[idx] = ligand1
                idx += 1

            if ligand2 not in node_to_idx:
                node_to_idx[ligand2] = idx
                idx_to_node[idx] = ligand2
                idx += 1

            # Add edge data (convert to 1-indexed for Pyro)
            source_nodes.append(node_to_idx[ligand1] + 1)
            target_nodes.append(node_to_idx[ligand2] + 1)
            edge_values.append(fep_value)

        return GraphData(
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            edge_values=edge_values,
            num_nodes=len(node_to_idx),
            num_edges=len(edge_values),
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
        )

    def _node_model(self, graph_data: GraphData) -> None:
        """
        Define the probabilistic model for node values.

        This function defines the joint distribution over node values
        and observations. It uses the prior distribution for node values
        and a normal distribution for the observation noise.

        Parameters
        ----------
        graph_data : GraphData
            Graph data containing edge information
        """
        # Convert data to tensors
        source_nodes = torch.tensor(graph_data.source_nodes)
        target_nodes = torch.tensor(graph_data.target_nodes)
        edge_values = torch.tensor(graph_data.edge_values)

        # Create prior distribution
        prior = self._create_prior(graph_data.num_nodes)

        # Sample node values from prior
        with pyro.plate("nodes", graph_data.num_nodes):
            node_values = pyro.sample("node_values", prior)

        # Calculate cycle errors (difference between predicted and observed edge values)
        cycle_errors = torch.zeros(len(edge_values))
        for i in range(len(edge_values)):
            # Edge value = target_node - source_node
            predicted_edge = (
                node_values[target_nodes[i] - 1] - node_values[source_nodes[i] - 1]
            )
            cycle_errors[i] = predicted_edge - edge_values[i]

        # Observe cycle errors with error distribution
        with pyro.plate("edges", len(edge_values)):
            if self.config.error_distribution == ErrorDistributionType.NORMAL:
                # Standard normal distribution for cycle errors
                pyro.sample(
                    "cycle_errors",
                    dist.Normal(0.0, self.config.error_std),
                    obs=cycle_errors,
                )
            elif self.config.error_distribution == ErrorDistributionType.SKEWED_NORMAL:
                # Skewed normal distribution for cycle errors using mixture approximation
                # This creates a right-skewed distribution for cycle errors
                loc = 0.0
                scale = self.config.error_std
                skew = self.config.error_skew

                # Create mixture of two normal distributions for skewed normal approximation
                # Main component (80% weight) and tail component (20% weight)
                mixture_weights = torch.tensor([0.8, 0.2])
                mixture_locs = torch.tensor([loc, loc + skew * scale])
                mixture_scales = torch.tensor([scale, scale * 1.5])

                skewed_normal = dist.MixtureSameFamily(
                    dist.Categorical(mixture_weights),
                    dist.Normal(mixture_locs, mixture_scales),
                )

                pyro.sample("cycle_errors", skewed_normal, obs=cycle_errors)
            else:
                raise ValueError(
                    f"Unsupported error distribution type: {self.config.error_distribution}"
                )

    def train(self) -> Dict[str, Any]:
        """
        Train the model using variational inference.

        This method performs variational inference to estimate node values
        from the edge-based FEP measurements. The guide type determines
        whether to use MAP estimation (AutoDelta) or uncertainty quantification (AutoNormal).

        Returns
        -------
        Dict[str, Any]
            Training results including loss history and final estimates
        """
        # Extract graph data
        self.graph_data = self._extract_graph_data()

        # Clear Pyro parameter store
        pyro.clear_param_store()

        # Create guide based on configuration
        if self.config.guide_type == GuideType.AUTO_DELTA:
            guide = AutoDelta(self._node_model)
        elif self.config.guide_type == GuideType.AUTO_NORMAL:
            guide = AutoNormal(self._node_model)
        else:
            raise ValueError(f"Unsupported guide type: {self.config.guide_type}")

        # Setup optimizer
        gamma = 0.1  # Final learning rate will be gamma * initial_lr
        lrd = gamma ** (1 / self.config.num_steps)
        optimizer = pyro.optim.ClippedAdam(
            {"lr": self.config.learning_rate, "lrd": lrd}
        )

        # Setup SVI
        svi = SVI(self._node_model, guide, optimizer, loss=Trace_ELBO())

        # Training loop with early stopping
        print("Starting optimization...")
        self.loss_history = []
        self.elbo_history = []
        best_loss = float('inf')
        patience_counter = 0

        for step in range(self.config.num_steps):
            loss = svi.step(self.graph_data)
            elbo = -loss  # ELBO is negative loss

            self.loss_history.append(loss)
            self.elbo_history.append(elbo)

            # Early stopping logic
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                print(f"Early stopping at step {step}: No improvement for {self.config.patience} steps")
                break

            if step % 1000 == 0:
                print(f"Step {step}: Loss = {loss:.4f}, ELBO = {elbo:.4f}")

        print(f"Optimization completed! Final Loss: {best_loss:.4f}")

        # Extract node estimates and uncertainties
        param_store = pyro.get_param_store()

        if self.config.guide_type == GuideType.AUTO_DELTA:
            # MAP estimates (point estimates)
            node_params = list(param_store.values())[0]
            self.node_estimates = {
                self.graph_data.idx_to_node[i]: float(node_params[i].detach().numpy())
                for i in range(self.graph_data.num_nodes)
            }
            # No uncertainties for AutoDelta
            self.node_uncertainties = None

        elif self.config.guide_type == GuideType.AUTO_NORMAL:
            # Variational estimates with uncertainties
            # AutoNormal creates separate parameters for mean and scale
            param_names = list(param_store.keys())

            # Find the node parameters (they should be named like "node_values" or similar)
            node_param_name = None
            for name in param_names:
                if "node" in name.lower() or "value" in name.lower():
                    node_param_name = name
                    break

            if node_param_name is None:
                # Fallback: assume first parameter is node values
                node_param_name = param_names[0]

            node_params = param_store[node_param_name]

            # Extract means and standard deviations
            if hasattr(node_params, "shape") and len(node_params.shape) == 2:
                # If parameters are 2D, assume first dimension is mean, second is scale
                means = node_params[:, 0].detach().numpy()
                scales = node_params[:, 1].detach().numpy()
            else:
                # Single parameter case - extract mean and scale from different parameters
                means = node_params.detach().numpy()

                # Look for scale parameters
                scale_param_name = None
                for name in param_names:
                    if "scale" in name.lower() or "std" in name.lower():
                        scale_param_name = name
                        break

                if scale_param_name:
                    scales = param_store[scale_param_name].detach().numpy()
                else:
                    # If no scale parameter found, use a default uncertainty
                    scales = np.ones_like(means) * 0.1

            self.node_estimates = {
                self.graph_data.idx_to_node[i]: float(means[i])
                for i in range(self.graph_data.num_nodes)
            }

            self.node_uncertainties = {
                self.graph_data.idx_to_node[i]: float(scales[i])
                for i in range(self.graph_data.num_nodes)
            }

        # Calculate edge estimates and uncertainties
        self.edge_estimates, self.edge_uncertainties = (
            self._calculate_edge_estimates_with_uncertainty()
        )

        return {
            "final_loss": self.loss_history[-1],
            "node_estimates": self.node_estimates,
            "edge_estimates": self.edge_estimates,
            "node_uncertainties": self.node_uncertainties,
            "edge_uncertainties": self.edge_uncertainties,
            "guide_type": self.config.guide_type.value,
        }

    def _calculate_edge_estimates_with_uncertainty(
        self,
    ) -> Tuple[Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]]:
        """
        Calculate edge estimates and uncertainties from node estimates.

        Returns
        -------
        Tuple[Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]]
            Dictionary mapping (source, target) node pairs to estimated edge values,
            and optional dictionary of edge uncertainties
        """
        if self.node_estimates is None or self.graph_data is None:
            raise ValueError("Model must be trained before calculating edge estimates")

        edge_estimates = {}
        edge_uncertainties = {}

        for i in range(self.graph_data.num_edges):
            source_node = self.graph_data.idx_to_node[
                self.graph_data.source_nodes[i] - 1
            ]
            target_node = self.graph_data.idx_to_node[
                self.graph_data.target_nodes[i] - 1
            ]

            # Calculate edge value
            edge_value = (
                self.node_estimates[target_node] - self.node_estimates[source_node]
            )
            edge_estimates[(source_node, target_node)] = edge_value

            # Calculate edge uncertainty if node uncertainties are available
            if self.node_uncertainties is not None:
                source_uncertainty = self.node_uncertainties.get(source_node, 0.0)
                target_uncertainty = self.node_uncertainties.get(target_node, 0.0)

                # Propagate uncertainty: sqrt(sum of squares)
                edge_uncertainty = np.sqrt(
                    source_uncertainty**2 + target_uncertainty**2
                )
                edge_uncertainties[(source_node, target_node)] = edge_uncertainty

        return edge_estimates, (
            edge_uncertainties if self.node_uncertainties is not None else None
        )

    def get_results(self) -> Dict[str, Any]:
        """
        Get the complete results from the trained model.

        Returns
        -------
        Dict[str, Any]
            Complete results including node estimates, edge estimates, uncertainties, and metadata
        """
        if self.node_estimates is None:
            raise ValueError("Model must be trained before getting results")

        results = {
            "node_estimates": self.node_estimates,
            "edge_estimates": self.edge_estimates,
            "config": self.config.model_dump(),
            "num_nodes": self.graph_data.num_nodes if self.graph_data else 0,
            "num_edges": self.graph_data.num_edges if self.graph_data else 0,
            "guide_type": self.config.guide_type.value,
        }

        # Add uncertainties if available
        if self.node_uncertainties is not None:
            results["node_uncertainties"] = self.node_uncertainties
        if self.edge_uncertainties is not None:
            results["edge_uncertainties"] = self.edge_uncertainties

        return results

    def add_predictions_to_dataset(self) -> None:
        """
        Add model predictions to the dataset for analysis.

        This method adds the node and edge estimates to the dataset
        for further analysis and comparison with experimental data.
        If uncertainties are available, they are also added.
        """
        if self.node_estimates is None:
            raise ValueError(
                "Model must be trained before adding predictions to dataset"
            )

        # Determine the column suffix based on guide type
        if self.config.guide_type == GuideType.AUTO_DELTA:
            if self.config.prior_type == PriorType.UNIFORM:
                suffix = "MLE"
            else:
                suffix = "MAP"
        else:
            suffix = "VI"
        

        # Add node predictions to dataset
        nodes_df = getattr(self.dataset, "dataset_nodes", None)
        if nodes_df is not None:
            node_predictions = []
            node_uncertainties = []

            for ligand in nodes_df["Name"]:
                if ligand in self.node_estimates:
                    node_predictions.append(self.node_estimates[ligand])
                    if (
                        self.node_uncertainties is not None
                        and ligand in self.node_uncertainties
                    ):
                        node_uncertainties.append(self.node_uncertainties[ligand])
                    else:
                        node_uncertainties.append(np.nan)
                else:
                    node_predictions.append(np.nan)
                    node_uncertainties.append(np.nan)

            nodes_df[suffix] = node_predictions

            # Add uncertainties if available
            if self.node_uncertainties is not None:
                nodes_df[f"{suffix}_uncertainty"] = node_uncertainties

        # Add edge predictions to dataset
        edges_df = getattr(self.dataset, "dataset_edges", None)
        if edges_df is not None:
            edge_predictions = []
            edge_uncertainties = []

            for _, row in edges_df.iterrows():
                # Handle both old and new column names
                if "Source" in edges_df.columns and "Destination" in edges_df.columns:
                    edge_key = (row["Source"], row["Destination"])
                elif "Ligand1" in edges_df.columns and "Ligand2" in edges_df.columns:
                    edge_key = (row["Ligand1"], row["Ligand2"])
                else:
                    raise ValueError(
                        "Edge data must have either 'Source'/'Destination' or 'Ligand1'/'Ligand2' columns"
                    )

                if edge_key in self.edge_estimates:
                    edge_predictions.append(self.edge_estimates[edge_key])
                    if (
                        self.edge_uncertainties is not None
                        and edge_key in self.edge_uncertainties
                    ):
                        edge_uncertainties.append(self.edge_uncertainties[edge_key])
                    else:
                        edge_uncertainties.append(np.nan)
                else:
                    edge_predictions.append(np.nan)
                    edge_uncertainties.append(np.nan)

            edges_df[suffix] = edge_predictions

            # Add uncertainties if available
            if self.edge_uncertainties is not None:
                edges_df[f"{suffix}_uncertainty"] = edge_uncertainties

        # Add estimator name to dataset
        if hasattr(self.dataset, "estimators"):
            if suffix not in self.dataset.estimators:
                self.dataset.estimators.append(suffix)

    def plot_training_history(
        self,
        filename: Optional[str] = None,
        figsize: tuple = (14, 5),
        show_plot: bool = True
    ):
        """
        Plot the training history showing Loss and ELBO over optimization steps.

        This method creates two subplots side-by-side:
        1. Loss (minimized during training)
        2. ELBO (Evidence Lower Bound, maximized during training)

        Parameters
        ----------
        filename : str, optional
            If provided, saves the plot to this file (PDF format).
            If None and not in notebook, displays the plot.
        figsize : tuple, default=(14, 5)
            Figure size as (width, height) in inches
        show_plot : bool, default=True
            Whether to display the plot interactively

        Returns
        -------
        matplotlib.figure.Figure
            The created figure object

        Raises
        ------
        RuntimeError
            If the model hasn't been trained yet (no history available)

        Examples
        --------
        >>> model = NodeModel(config, dataset)
        >>> model.train()
        >>> model.plot_training_history(filename="training_history.pdf")
        """
        import matplotlib.pyplot as plt

        if not self.loss_history or not self.elbo_history:
            raise RuntimeError(
                "No training history available. Please train the model first using .train()"
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        steps = list(range(len(self.loss_history)))

        # Plot Loss
        ax1.plot(steps, self.loss_history, color='red', linewidth=2, label='Loss')
        ax1.set_xlabel('Optimization Step', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss (Minimized)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # Plot ELBO
        ax2.plot(steps, self.elbo_history, color='blue', linewidth=2, label='ELBO')
        ax2.set_xlabel('Optimization Step', fontsize=12)
        ax2.set_ylabel('ELBO', fontsize=12)
        ax2.set_title('Evidence Lower Bound (Maximized)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        # Overall title
        plt.suptitle('NodeModel Training History', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save or show
        if filename is not None:
            if not filename.endswith('.pdf'):
                filename = filename.rsplit('.', 1)[0] + '.pdf'
            plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
            print(f"✅ Training history plot saved to '{filename}'")

        if show_plot:
            plt.show()

        return fig

    @classmethod
    def calculate_ccc(
        cls, dataset: BaseDataset, config: Optional[NodeModelConfig] = None
    ) -> Dict[str, Any]:
        """
        Class method to calculate CCC values for a dataset.

        This is a convenience method that uses CCC_model to calculate
        Cycle Closure Corrected values as a benchmark.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset to calculate CCC values for
        config : NodeModelConfig, optional
            Configuration for CCC calculation. If None, uses default uniform prior config.

        Returns
        -------
        Dict[str, Any]
            Results from CCC calculation including node and edge estimates
        """
        # Import CCC_model here to avoid circular imports
        from .cycle_closure_correction import CCC_model
        
        # Use CCC_model for CCC calculation
        ccc_model = CCC_model(dataset, config)
        return ccc_model.calculate_and_add_ccc()
