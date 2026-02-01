from ..dataset import BaseDataset
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

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

class GMVI_model:
    """
    Full-Rank Gaussian Markov Variational Inference for FEP networks.
    
    Updated implementation with:
    - Clearer edge indexing (i,j)
    - Global mixture weight π
    - Simplified parameterization θ = [σ₀, σ₁, σ₂, π]
    """
    def __init__(self, dataset: BaseDataset, config: Optional['GMVIConfig'] = None,
                 prior_std=None, normal_std=None, outlier_std=None, 
                 outlier_prob=None, kl_weight=None, learning_rate=None, 
                 n_epochs=None, n_samples=None, patience=None):
        """
        Initialize the GMVI model with updated parameters.
        
        Parameters:
        -----------
        dataset : BaseDataset
            Dataset object containing the FEP data
        config : GMVIConfig, optional
            Configuration object containing model parameters.
            If provided, overrides individual parameter values.
        prior_std : float, optional
            Prior standard deviation for node values (σ₀)
        normal_std : float, optional
            Standard deviation for normal edges (σ₁)
        outlier_std : float, optional
            Standard deviation for outlier edges (σ₂)
        outlier_prob : float, optional
            Global probability of an edge being an outlier (π)
        kl_weight : float, optional
            Weight for KL divergence term in ELBO
        learning_rate : float, optional
            Learning rate for optimization
        n_epochs : int, optional
            Maximum number of training epochs
        n_samples : int, optional
            Number of Monte Carlo samples for ELBO estimation
        patience : int, optional
            Early stopping patience
        """
        self.dataset = dataset # Dataset object

        # Use config if provided, otherwise use individual parameters or defaults
        if config is not None:
            # Check if config has the expected attributes (duck typing)
            # This avoids circular import issues with isinstance checks
            required_attrs = ['prior_std', 'normal_std', 'outlier_std', 'outlier_prob',
                            'kl_weight', 'learning_rate', 'n_epochs', 'n_samples', 'patience']

            if not all(hasattr(config, attr) for attr in required_attrs):
                raise TypeError(
                    f"Config must be a GMVIConfig object with attributes: {required_attrs}, "
                    f"got {type(config)}"
                )

            self.prior_std = config.prior_std
            self.normal_std = torch.tensor(config.normal_std, dtype=torch.float32)
            self.outlier_std = torch.tensor(config.outlier_std, dtype=torch.float32)
            self.outlier_prob = config.outlier_prob
            self.kl_weight = config.kl_weight
            self.learning_rate = config.learning_rate
            self.n_epochs = config.n_epochs
            self.n_samples = config.n_samples
            self.patience = config.patience
        else:
            # Use individual parameters or defaults
            self.prior_std = prior_std if prior_std is not None else 5.0
            self.normal_std = torch.tensor(
                normal_std if normal_std is not None else 1.0, 
                dtype=torch.float32
            )
            self.outlier_std = torch.tensor(
                outlier_std if outlier_std is not None else 3.0, 
                dtype=torch.float32
            )
            self.outlier_prob = outlier_prob if outlier_prob is not None else 0.2
            self.kl_weight = kl_weight if kl_weight is not None else 0.1
            self.learning_rate = learning_rate if learning_rate is not None else 0.01
            self.n_epochs = n_epochs if n_epochs is not None else 1000
            self.n_samples = n_samples if n_samples is not None else 100
            self.patience = patience if patience is not None else 50
        
        # Variational parameters
        self.node_means = None
        self.node_cholesky = None

        # Training history
        self.loss_history = []
        self.elbo_history = []

        # Extract graph data on initialization
        self._extract_graph_data()

        # Initialize prediction attributes
        self.node_estimates = None
        self.node_uncertainties = None
        self.edge_estimates = None
        self.edge_uncertainties = None

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

            # Add edge data (keep 0-indexed for consistency)
            source_nodes.append(node_to_idx[ligand1])
            target_nodes.append(node_to_idx[ligand2])
            edge_values.append(fep_value)

        self.graph_data = GraphData(
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            edge_values=edge_values,
            num_nodes=len(node_to_idx),
            num_edges=len(edge_values),
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
        )
        
        return self.graph_data

    def initialize_parameters(self):
        """Initialize variational parameters.
        This is converting all of the input data into pytorch tensors
        
        """
        # Initialize node means using weighted least squares
        self.node_means = torch.zeros(self.graph_data.num_nodes, requires_grad=True)
        
        # Initialize covariance matrix via Cholesky decomposition
        # Start with diagonal matrix and small off-diagonal terms
        diag_vals = torch.ones(self.graph_data.num_nodes) * 0.1
        off_diag_vals = torch.ones(self.graph_data.num_nodes * (self.graph_data.num_nodes - 1) // 2) * 0.01
        
        # Build lower triangular matrix
        L = torch.zeros(self.graph_data.num_nodes, self.graph_data.num_nodes)
        torch.diagonal(L)[:] = diag_vals
        
        # Fill lower triangular part
        idx = 0
        for i in range(self.graph_data.num_nodes):
            for j in range(i):
                L[i, j] = off_diag_vals[idx]
                idx += 1
        
        self.node_cholesky = nn.Parameter(L)
        
        print("Parameters initialized")

    def get_covariance_matrix(self):
        """Get the full covariance matrix from Cholesky decomposition."""
        return torch.mm(self.node_cholesky, self.node_cholesky.t())
    
    def sample_nodes(self, n_samples=None):
        """Sample from the variational distribution q(z)."""
        if n_samples is None:
            n_samples = self.n_samples
            
        # Sample from standard normal
        eps = torch.randn(n_samples, self.graph_data.num_nodes)
        
        # Transform using Cholesky decomposition
        samples = self.node_means.unsqueeze(0) + torch.mm(eps, self.node_cholesky)
        
        return samples
    
    def compute_edge_predictions(self, node_samples):
        """
        Compute edge predictions for given node samples.
        
        Parameters:
        -----------
        node_samples : torch.Tensor
            Node samples of shape (n_samples, n_nodes)
            
        Returns:
        --------
        torch.Tensor : Edge predictions of shape (n_samples, n_edges)
        """
        predictions = torch.zeros(node_samples.shape[0], self.graph_data.num_edges)
        
        for edge_idx in range(self.graph_data.num_edges):
            # Edge value = z_j - z_i (FEP standard: final state - initial state)
            predictions[:, edge_idx] = node_samples[:, self.graph_data.target_nodes[edge_idx]] - node_samples[:, self.graph_data.source_nodes[edge_idx]]
            
        return predictions

    def compute_likelihood(self, node_samples):
        """
        Compute the likelihood p(x|z) using the updated mixture model.
        
        Parameters:
        -----------
        node_samples : torch.Tensor
            Node samples of shape (n_samples, n_nodes)
            
        Returns:
        --------
        torch.Tensor : Log-likelihood values of shape (n_samples, n_edges)
        """
        # Get edge predictions
        edge_predictions = self.compute_edge_predictions(node_samples)
        
        # Compute log-likelihood for each edge using mixture model
        log_likelihood = torch.zeros(node_samples.shape[0], self.graph_data.num_edges)
        
        for edge_idx in range(self.graph_data.num_edges):
            pred = edge_predictions[:, edge_idx]
            obs = self.graph_data.edge_values[edge_idx]
            
            # First component: p(x|z, component 1) = N(x|z_j-z_i, σ₁²) with weight π (OUTLIERS)
            log_component1 = -0.5 * ((obs - pred) / self.outlier_std)**2 - torch.log(self.outlier_std * torch.sqrt(2 * torch.tensor(torch.pi)))
            
            # Second component: p(x|z, component 2) = N(x|z_j-z_i, σ₂²) with weight (1-π) (NORMAL)
            log_component2 = -0.5 * ((obs - pred) / self.normal_std)**2 - torch.log(self.normal_std * torch.sqrt(2 * torch.tensor(torch.pi)))
            
            # Mixture: π * p(component 1) + (1-π) * p(component 2) (following professor's math)
            # Use log-sum-exp trick for numerical stability
            log_mixture = torch.logsumexp(
                torch.stack([
                    torch.log(torch.tensor(self.outlier_prob)) + log_component1,      # π * component 1
                    torch.log(torch.tensor(1.0 - self.outlier_prob)) + log_component2  # (1-π) * component 2
                ]), dim=0
            )
            
            log_likelihood[:, edge_idx] = log_mixture
            
        return log_likelihood
    
    def compute_kl_divergence(self):
        """Compute KL divergence KL[q(z)||p(z)]."""
        # Get covariance matrix
        cov_matrix = self.get_covariance_matrix()
        
        # Prior covariance
        prior_cov = torch.eye(self.graph_data.num_nodes) * (self.prior_std ** 2)
        prior_cov_inv = torch.eye(self.graph_data.num_nodes) / (self.prior_std ** 2)
        
        # KL divergence terms
        trace_term = torch.trace(torch.mm(prior_cov_inv, cov_matrix))
        mean_term = torch.mm(torch.mm(self.node_means.unsqueeze(0), prior_cov_inv), self.node_means.unsqueeze(1)).squeeze()
        det_term = torch.logdet(prior_cov) - torch.logdet(cov_matrix)
        
        kl_div = 0.5 * (trace_term + mean_term - self.graph_data.num_nodes + det_term)
        
        return kl_div
    
    def compute_elbo(self):
        """Compute the Evidence Lower BOund (ELBO)."""
        # Sample from variational distribution
        node_samples = self.sample_nodes()
        
        # Compute expected log-likelihood
        log_likelihood = self.compute_likelihood(node_samples)
        expected_log_likelihood = torch.mean(torch.sum(log_likelihood, dim=1))
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence()
        
        # ELBO = E[log p(x|z)] - KL[q(z)||p(z)]
        elbo = expected_log_likelihood - self.kl_weight * kl_div
        
        return elbo, expected_log_likelihood, kl_div
    
    def fit(self):
        """Fit the model by maximizing the ELBO."""
        print("Starting optimization...")
        
        # Initialize parameters
        self.initialize_parameters()
        
        # Setup optimizer
        optimizer = optim.Adam([self.node_means, self.node_cholesky], 
                             lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                       factor=0.5, patience=20)
        
        best_elbo = float('-inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
        
            # Compute loss (negative ELBO)
            elbo, expected_log_likelihood, kl_div = self.compute_elbo()
            loss = -elbo
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            scheduler.step(elbo)
            
            # Store history
            self.loss_history.append(loss.item())
            self.elbo_history.append(elbo.item())
            
            # Early stopping
            if elbo > best_elbo:
                best_elbo = elbo
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: ELBO = {elbo:.4f}, Loss = {loss:.4f}")
        
        print("Optimization completed!")
        print(f"Final ELBO: {best_elbo:.4f}")
        
    def get_posterior_estimates(self):
        """Get posterior estimates for node values.
        These values should be used to update the dataset object.
        To compute the node estimates 1000 samples are drawn from the posterior for each node.
        The node estimates to be added to the dataset object are the means of the samples drawn.
        The uncertainties to be added are the std values for each of the nodes estimates drawn from the posterior.
        The node estimates recorded here should be used for the other class methods
         to maintain consistency with class instances.
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'means': Mean node values
            - 'stds': Standard deviations of node values
            - 'samples': Sampled node values
        """
        # Get samples
        node_samples = self.sample_nodes(n_samples=1000)
        
        # Compute statistics
        means = torch.mean(node_samples, dim=0)
        stds = torch.std(node_samples, dim=0)
        

        # Convert all quantities to numpy arrays
        node_samples.detach().numpy()
        means = means.detach().numpy()
        stds = stds.detach().numpy()

        posterior_estimates = {
            'means': means,
            'stds': stds,
            'samples': node_samples.detach().numpy()
        }

        # Add Node estimates and uncertainties 
        self.node_estimates = {
            self.graph_data.idx_to_node[i]: float(means[i])
            for i in range(self.graph_data.num_nodes)
        }

        self.node_uncertainties = {
                self.graph_data.idx_to_node[i]: float(stds[i])
                for i in range(self.graph_data.num_nodes)
            }
        
        # Calculate edge estimates from node estimates
        self.edge_estimates = {}
        self.edge_uncertainties = {}
        
        # Get the column names from the dataset to match format
        edges_df = getattr(self.dataset, "dataset_edges", None)
        if edges_df is not None:
            for idx, row in edges_df.iterrows():
                # Handle both old and new column names
                if "Source" in edges_df.columns and "Destination" in edges_df.columns:
                    ligand1, ligand2 = row["Source"], row["Destination"]
                elif "Ligand1" in edges_df.columns and "Ligand2" in edges_df.columns:
                    ligand1, ligand2 = row["Ligand1"], row["Ligand2"]
                else:
                    continue
                
                # Calculate edge estimate as difference between nodes
                if ligand1 in self.node_estimates and ligand2 in self.node_estimates:
                    edge_key = (ligand1, ligand2)
                    self.edge_estimates[edge_key] = self.node_estimates[ligand2] - self.node_estimates[ligand1]
                    
                    # Propagate uncertainty (assuming independence)
                    if ligand1 in self.node_uncertainties and ligand2 in self.node_uncertainties:
                        self.edge_uncertainties[edge_key] = np.sqrt(
                            self.node_uncertainties[ligand1]**2 + self.node_uncertainties[ligand2]**2
                        )
        
        return posterior_estimates
    
    def compute_edge_outlier_probabilities(self):
        """
        Compute posterior probability that each edge is an outlier.
        This is similar to the E-step of EM algorithm for Gaussian mixtures.
        """
        # Get samples
        node_samples = self.sample_nodes(n_samples=1000)
        
        # Get edge predictions
        edge_predictions = self.compute_edge_predictions(node_samples)
        
        # Compute outlier probabilities for each edge
        outlier_probs = []
        
        for edge_idx in range(self.graph_data.num_edges):
            pred = edge_predictions[:, edge_idx]
            obs = self.graph_data.edge_values[edge_idx]
            
            # Compute likelihoods for both components using FEP standard
            # Component 1: N(x|z_j-z_i, σ₁²) with weight π (OUTLIERS)
            log_component1 = -0.5 * ((obs - pred) / self.outlier_std)**2 - torch.log(self.outlier_std * torch.sqrt(2 * torch.tensor(torch.pi)))
            # Component 2: N(x|z_j-z_i, σ₂²) with weight (1-π) (NORMAL)
            log_component2 = -0.5 * ((obs - pred) / self.normal_std)**2 - torch.log(self.normal_std * torch.sqrt(2 * torch.tensor(torch.pi)))
            
            # Convert to probabilities
            component1_probs = torch.exp(log_component1)
            component2_probs = torch.exp(log_component2)
            
            # Compute posterior probability of belonging to component 1 (following professor's math)
            # P(component 1 | x_ij, z) = π * p(x_ij | z, component 1) / [π * p(x_ij | z, component 1) + (1-π) * p(x_ij | z, component 2)]
            numerator = self.outlier_prob * component1_probs
            denominator = self.outlier_prob * component1_probs + (1 - self.outlier_prob) * component2_probs
            
            # Average over samples
            posterior_outlier_prob = torch.mean(numerator / denominator)
            outlier_probs.append(posterior_outlier_prob.item())
        
        return outlier_probs

    def evaluate_predictions(self):
        """Evaluate the model predictions."""
        # Get posterior estimates
        estimates = self.get_posterior_estimates()
        
        # Compute edge predictions using FEP standard: z_j - z_i (final - initial)
        edge_predictions = []
        for edge_idx in range(self.graph_data.num_edges):
            i = self.graph_data.source_nodes[edge_idx]
            j = self.graph_data.target_nodes[edge_idx]
            pred = estimates['means'][j] - estimates['means'][i]
            edge_predictions.append(pred)
        
        edge_predictions = np.array(edge_predictions)
        edge_observations = np.array(self.graph_data.edge_values)
        
        # Compute metrics
        mae = np.mean(np.abs(edge_predictions - edge_observations))
        rmse = np.sqrt(np.mean((edge_predictions - edge_observations)**2))
        correlation = np.corrcoef(edge_predictions, edge_observations)[0, 1]
        mean_uncertainty = np.mean(estimates['stds'])
        
        # Get outlier probabilities
        outlier_probs = self.compute_edge_outlier_probabilities()
        high_confidence_outliers = sum(prob > 0.7 for prob in outlier_probs)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'mean_uncertainty': mean_uncertainty,
            'outlier_probs': outlier_probs,
            'high_confidence_outliers': high_confidence_outliers
        }
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

        # Column suffix is GMVI based on the model
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

    def plot_training_history(
        self,
        filename: Optional[str] = None,
        figsize: tuple = (14, 5),
        show_plot: bool = True
    ):
        """
        Plot the training history showing Loss and ELBO over optimization epochs.

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
        >>> model = GMVI_model(dataset, config)
        >>> model.fit()
        >>> model.plot_training_history(filename="gmvi_training_history.pdf")
        """
        import matplotlib.pyplot as plt

        if not self.loss_history or not self.elbo_history:
            raise RuntimeError(
                "No training history available. Please train the model first using .fit()"
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        epochs = list(range(len(self.loss_history)))

        # Plot Loss
        ax1.plot(epochs, self.loss_history, color='red', linewidth=2, label='Loss')
        ax1.set_xlabel('Training Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss (Minimized)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # Plot ELBO
        ax2.plot(epochs, self.elbo_history, color='blue', linewidth=2, label='ELBO')
        ax2.set_xlabel('Training Epoch', fontsize=12)
        ax2.set_ylabel('ELBO', fontsize=12)
        ax2.set_title('Evidence Lower Bound (Maximized)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        # Overall title
        plt.suptitle('GMVI Model Training History', fontsize=16, fontweight='bold', y=1.02)
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
