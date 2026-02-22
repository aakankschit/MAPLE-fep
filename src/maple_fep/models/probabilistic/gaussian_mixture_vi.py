"""
Gaussian Mixture Variational Inference Model using Pyro

This module provides a Pyro-based implementation of full-rank Gaussian Markov
Variational Inference for FEP analysis. It combines:
- Full-rank covariance via Cholesky decomposition
- Normal prior distribution
- Mixture likelihood model (normal vs outlier edges)
- Pyro's automatic ELBO computation
- ClippedAdam optimizer with exponential decay
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoGuide
from pyro.nn import PyroParam

# Import dataset classes
from ...dataset import BaseDataset
# Import configuration classes
from ..config import GaussianMixtureVIConfig
# Import shared GraphData
from ..graph_data import GraphData
# Import base class
from ..base import BaseEstimator


class FullRankGaussianGuide(AutoGuide):
    """
    Custom guide for full-rank Gaussian variational distribution.

    Uses Cholesky decomposition to parameterize the covariance matrix:
    q(z) = N(z | mu, L L^T)
    where L is a lower triangular matrix.
    """

    def __init__(self, model, num_nodes: int, prior_std: float = 5.0):
        """
        Initialize the full-rank Gaussian guide.

        Parameters
        ----------
        model : callable
            The Pyro model function
        num_nodes : int
            Number of nodes in the graph
        prior_std : float
            Prior standard deviation (used for initialization)
        """
        super().__init__(model)
        self.num_nodes = num_nodes
        self.prior_std = prior_std

        # Initialize mean parameters
        self.loc = PyroParam(
            torch.zeros(num_nodes),
            constraint=torch.distributions.constraints.real
        )

        # Initialize Cholesky factor
        L_init = torch.eye(num_nodes) * (0.5 * prior_std)

        # Create lower triangular mask
        tril_mask = torch.tril(torch.ones(num_nodes, num_nodes))

        # Parameterize Cholesky factor (only lower triangular part)
        self.L_raw = PyroParam(
            L_init * tril_mask,
            constraint=torch.distributions.constraints.real
        )

        self.register_buffer("tril_mask", tril_mask)

    def forward(self, *args, **kwargs):
        """
        Sample from the variational distribution.

        Returns
        -------
        dict
            Dictionary with sampled node values
        """
        # Get Cholesky factor
        L = self.L_raw * self.tril_mask

        # Ensure diagonal is positive using softplus
        diag = torch.diagonal(L)
        L = L - torch.diag(diag) + torch.diag(F.softplus(diag) + 1e-4)

        # Create MultivariateNormal distribution
        q_dist = dist.MultivariateNormal(
            loc=self.loc,
            scale_tril=L
        )

        # Sample node values
        node_values = pyro.sample("node_values", q_dist)

        return {"node_values": node_values}


class GaussianMixtureVI(BaseEstimator):
    """
    Full-Rank Gaussian Mixture Variational Inference Model using Pyro.

    This class implements a Pyro-based VI model with:
    - Full-rank covariance via Cholesky decomposition
    - Normal prior distribution
    - Mixture likelihood (normal vs outlier edges)
    - Automatic ELBO computation via Pyro
    - ClippedAdam optimizer with exponential decay
    - 1-indexed nodes for Pyro compatibility
    """

    def __init__(self, config: GaussianMixtureVIConfig, dataset: BaseDataset):
        """
        Initialize the GaussianMixtureVI model.

        Parameters
        ----------
        config : GaussianMixtureVIConfig
            Configuration object containing all model parameters
        dataset : BaseDataset
            Dataset containing the FEP data
        """
        self.config = config
        self.dataset = dataset
        self.graph_data: Optional[GraphData] = None
        self.node_estimates: Optional[Dict[str, float]] = None
        self.edge_estimates: Optional[Dict[Tuple[str, str], float]] = None
        self.node_uncertainties: Optional[Dict[str, float]] = None
        self.edge_uncertainties: Optional[Dict[Tuple[str, str], float]] = None

        # Convert std parameters to tensors
        self.normal_std = torch.tensor(config.normal_std, dtype=torch.float32)
        self.outlier_std = torch.tensor(config.outlier_std, dtype=torch.float32)
        self.outlier_prob = config.outlier_prob
        self.prior_std = config.prior_std

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

    def _model(self, graph_data: GraphData) -> None:
        """
        Define the probabilistic model for node values with mixture likelihood.

        Parameters
        ----------
        graph_data : GraphData
            Graph data containing edge information
        """
        # Convert data to tensors
        source_nodes = torch.tensor(graph_data.source_nodes)
        target_nodes = torch.tensor(graph_data.target_nodes)
        edge_values = torch.tensor(graph_data.edge_values, dtype=torch.float32)

        # Normal prior for node values (multivariate for full-rank covariance)
        prior_mean = torch.zeros(graph_data.num_nodes)
        prior_cov = torch.eye(graph_data.num_nodes) * (self.prior_std ** 2)
        prior = dist.MultivariateNormal(
            loc=prior_mean,
            covariance_matrix=prior_cov
        )

        # Sample node values from prior (vector-valued, event_dim=1)
        node_values = pyro.sample("node_values", prior)

        # Compute edge predictions
        predicted_edges = node_values[target_nodes - 1] - node_values[source_nodes - 1]

        # Observe edges with mixture likelihood
        mixture_weights = torch.tensor([self.outlier_prob, 1.0 - self.outlier_prob],
                                      dtype=torch.float32)

        with pyro.plate("edges", graph_data.num_edges):
            for i in range(graph_data.num_edges):
                pred = predicted_edges[i]
                obs = edge_values[i]

                batch_locs = torch.stack([pred, pred])
                batch_scales = torch.stack([self.outlier_std, self.normal_std])

                mixture = dist.MixtureSameFamily(
                    dist.Categorical(mixture_weights),
                    dist.Normal(batch_locs, batch_scales)
                )

                pyro.sample(f"edge_{i}", mixture, obs=obs)

    def fit(self) -> None:
        """
        Fit the model using Pyro's SVI with full-rank covariance.

        This method performs variational inference to estimate node values
        from the edge-based FEP measurements using a full-rank Gaussian
        variational distribution.

        Results are accessible via ``node_estimates``, ``edge_estimates``,
        and ``get_results()`` after calling this method.
        """
        # Extract graph data
        self.graph_data = self._extract_graph_data()

        # Clear Pyro parameter store
        pyro.clear_param_store()

        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            pyro.set_rng_seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)

        # Create full-rank Gaussian guide
        guide = FullRankGaussianGuide(
            self._model,
            num_nodes=self.graph_data.num_nodes,
            prior_std=self.prior_std
        )

        # Setup optimizer with exponential decay
        gamma = 0.1  # Final learning rate will be gamma * initial_lr
        lrd = gamma ** (1 / self.config.num_steps)
        optimizer = pyro.optim.ClippedAdam(
            {"lr": self.config.learning_rate, "lrd": lrd}
        )

        # Setup SVI with Trace_ELBO
        svi = SVI(self._model, guide, optimizer, loss=Trace_ELBO())

        # Training loop
        losses = []
        for step in range(self.config.num_steps):
            loss = svi.step(self.graph_data)
            losses.append(loss)

            if step % 1000 == 0:
                print(f"Step {step}: Loss = {loss:.4f}")

        # Store training losses
        self._losses = losses

        # Extract node estimates and uncertainties
        self._extract_posterior_estimates(guide)

        # Calculate edge estimates and uncertainties
        self.edge_estimates, self.edge_uncertainties = (
            self._calculate_edge_estimates_with_uncertainty()
        )

    def _extract_posterior_estimates(self, guide: FullRankGaussianGuide) -> None:
        """
        Extract posterior estimates from the trained guide.

        Parameters
        ----------
        guide : FullRankGaussianGuide
            The trained guide containing variational parameters
        """
        # Get mean estimates
        means = guide.loc.detach().numpy()

        # Get covariance matrix from Cholesky factor
        L = guide.L_raw * guide.tril_mask
        diag = torch.diagonal(L)
        L = L - torch.diag(diag) + torch.diag(F.softplus(diag) + 1e-4)
        cov_matrix = torch.mm(L, L.t())

        # Extract standard deviations (diagonal of covariance matrix)
        stds = torch.sqrt(torch.diagonal(cov_matrix)).detach().numpy()

        # Store node estimates and uncertainties
        self.node_estimates = {
            self.graph_data.idx_to_node[i]: float(means[i])
            for i in range(self.graph_data.num_nodes)
        }

        self.node_uncertainties = {
            self.graph_data.idx_to_node[i]: float(stds[i])
            for i in range(self.graph_data.num_nodes)
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
            raise ValueError("Model must be fitted before calculating edge estimates")

        edge_estimates = {}
        edge_uncertainties = {}

        for i in range(self.graph_data.num_edges):
            source_node = self.graph_data.idx_to_node[
                self.graph_data.source_nodes[i] - 1  # Convert from 1-indexed
            ]
            target_node = self.graph_data.idx_to_node[
                self.graph_data.target_nodes[i] - 1  # Convert from 1-indexed
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
        Get the complete results from the fitted model.

        Returns
        -------
        Dict[str, Any]
            Complete results including node estimates, edge estimates,
            uncertainties, and metadata
        """
        if self.node_estimates is None:
            raise ValueError("Model must be fitted before getting results")

        results = {
            "node_estimates": self.node_estimates,
            "edge_estimates": self.edge_estimates,
            "config": self.config.model_dump(),
            "num_nodes": self.graph_data.num_nodes if self.graph_data else 0,
            "num_edges": self.graph_data.num_edges if self.graph_data else 0,
        }

        # Add training loss history if available
        if hasattr(self, "_losses") and self._losses:
            results["losses"] = self._losses
            results["final_loss"] = self._losses[-1]

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
                "Model must be fitted before adding predictions to dataset"
            )

        # Column suffix is GMVI for backward compatibility with plotting code
        suffix = "GMVI"

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

    def compute_edge_outlier_probabilities(self) -> List[float]:
        """
        Compute posterior probability that each edge is an outlier.

        Returns
        -------
        List[float]
            List of outlier probabilities for each edge
        """
        if self.node_estimates is None or self.graph_data is None:
            raise ValueError("Model must be fitted before computing outlier probabilities")

        outlier_probs = []

        for edge_idx in range(self.graph_data.num_edges):
            source_idx = self.graph_data.source_nodes[edge_idx] - 1
            target_idx = self.graph_data.target_nodes[edge_idx] - 1

            source_node = self.graph_data.idx_to_node[source_idx]
            target_node = self.graph_data.idx_to_node[target_idx]

            # Get predicted edge value
            pred = self.node_estimates[target_node] - self.node_estimates[source_node]
            obs = self.graph_data.edge_values[edge_idx]

            # Compute likelihoods for both components
            log_component1 = -0.5 * ((obs - pred) / self.outlier_std.item())**2 - np.log(
                self.outlier_std.item() * np.sqrt(2 * np.pi)
            )

            log_component2 = -0.5 * ((obs - pred) / self.normal_std.item())**2 - np.log(
                self.normal_std.item() * np.sqrt(2 * np.pi)
            )

            # Convert to probabilities
            component1_prob = np.exp(log_component1)
            component2_prob = np.exp(log_component2)

            # Compute posterior probability of belonging to component 1 (outlier)
            numerator = self.outlier_prob * component1_prob
            denominator = self.outlier_prob * component1_prob + (1 - self.outlier_prob) * component2_prob

            posterior_outlier_prob = numerator / denominator
            outlier_probs.append(float(posterior_outlier_prob))

        return outlier_probs
