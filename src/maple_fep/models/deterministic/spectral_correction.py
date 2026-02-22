"""
Spectral Correction (SpectralCorrection) Model for FEP Analysis

This module implements the WSFC method based on Liu et al.'s weighted graph
Laplacian approach. It solves for optimal node free energies by minimizing
weighted residuals on graph edges using the pseudoinverse of the weighted
graph Laplacian.

    z* = L_W^+ B^T W x

where:
    B = incidence matrix (M x N)
    W = diag(1/sigma_ij^2) weight matrix (M x M)
    L_W = B^T W B = weighted graph Laplacian (N x N)
    x = observed edge values (M,)

When use_weights=False (SFC), W = I and the solution is equivalent to MLE
(unweighted least-squares on the graph).
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...dataset import BaseDataset
from ..graph_data import GraphData
from ..config import SpectralCorrectionConfig
from ..base import BaseEstimator


class SpectralCorrection(BaseEstimator):
    """
    Spectral Correction Model (Weighted Spectral Free-energy Correction).

    Parameters
    ----------
    config : SpectralCorrectionConfig
        Configuration object with model parameters.
    dataset : BaseDataset
        Dataset containing FEP edge and node data.

    Attributes
    ----------
    graph_data : Optional[GraphData]
        Processed graph data.
    node_estimates : Optional[Dict[str, float]]
        Estimated node free energies after fitting.
    edge_estimates : Optional[Dict[Tuple[str, str], float]]
        Predicted edge values derived from node estimates.
    node_uncertainties : Optional[Dict[str, float]]
        Node uncertainties from diagonal of L_W pseudoinverse.
    edge_uncertainties : Optional[Dict[Tuple[str, str], float]]
        Propagated edge uncertainties.
    """

    def __init__(self, config: SpectralCorrectionConfig, dataset: BaseDataset):
        self.config = config
        self.dataset = dataset
        self.use_weights = config.use_weights

        self.graph_data: Optional[GraphData] = None
        self.node_estimates: Optional[Dict[str, float]] = None
        self.edge_estimates: Optional[Dict[Tuple[str, str], float]] = None
        self.node_uncertainties: Optional[Dict[str, float]] = None
        self.edge_uncertainties: Optional[Dict[Tuple[str, str], float]] = None

        # Internal arrays set during fit
        self._B: Optional[np.ndarray] = None  # incidence matrix (M x N)
        self._W: Optional[np.ndarray] = None  # weight diagonal (M,)
        self._z: Optional[np.ndarray] = None  # node estimates (N,)
        self._L_W_pinv: Optional[np.ndarray] = None  # pseudoinverse of L_W

    # ------------------------------------------------------------------
    # Graph data extraction
    # ------------------------------------------------------------------
    def _extract_graph_data(self) -> GraphData:
        """Extract graph data from dataset, reading ``DeltaDeltaG Error`` when available."""
        edges_df = getattr(self.dataset, "dataset_edges", None)
        if edges_df is None:
            raise ValueError("Dataset must have 'dataset_edges' attribute")

        source_nodes: List[int] = []
        target_nodes: List[int] = []
        edge_values: List[float] = []
        edge_errors: List[float] = []
        node_to_idx: Dict[str, int] = {}
        idx_to_node: Dict[int, str] = {}
        idx = 0

        has_error_col = "DeltaDeltaG Error" in edges_df.columns

        for _, row in edges_df.iterrows():
            # Node names
            if "Source" in edges_df.columns and "Destination" in edges_df.columns:
                ligand1, ligand2 = row["Source"], row["Destination"]
            elif "Ligand1" in edges_df.columns and "Ligand2" in edges_df.columns:
                ligand1, ligand2 = row["Ligand1"], row["Ligand2"]
            else:
                raise ValueError(
                    "Edge data must have 'Source'/'Destination' or 'Ligand1'/'Ligand2' columns"
                )

            # FEP value
            if "DeltaDeltaG" in edges_df.columns:
                fep_value = float(row["DeltaDeltaG"])
            elif "DeltaG" in edges_df.columns:
                fep_value = float(row["DeltaG"])
            elif "FEP" in edges_df.columns:
                fep_value = float(row["FEP"])
            else:
                raise ValueError(
                    "Edge data must have 'DeltaDeltaG', 'DeltaG', or 'FEP' column"
                )

            # Edge error
            if has_error_col:
                edge_errors.append(float(row["DeltaDeltaG Error"]))

            # Build node mapping (0-indexed)
            if ligand1 not in node_to_idx:
                node_to_idx[ligand1] = idx
                idx_to_node[idx] = ligand1
                idx += 1
            if ligand2 not in node_to_idx:
                node_to_idx[ligand2] = idx
                idx_to_node[idx] = ligand2
                idx += 1

            source_nodes.append(node_to_idx[ligand1])
            target_nodes.append(node_to_idx[ligand2])
            edge_values.append(fep_value)

        # Validate edge errors (all-or-nothing)
        final_edge_errors: Optional[List[float]] = None
        final_edge_weights: Optional[List[float]] = None
        if edge_errors and not any(np.isnan(e) for e in edge_errors):
            # Clamp sigma >= 0.01 to avoid extreme weights
            clamped = [max(e, 0.01) for e in edge_errors]
            final_edge_errors = clamped
            final_edge_weights = [1.0 / (s ** 2) for s in clamped]

        return GraphData(
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            edge_values=edge_values,
            num_nodes=len(node_to_idx),
            num_edges=len(edge_values),
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
            edge_errors=final_edge_errors,
            edge_weights=final_edge_weights,
        )

    # ------------------------------------------------------------------
    # Incidence matrix
    # ------------------------------------------------------------------
    def _build_incidence_matrix(self) -> np.ndarray:
        """Build the signed incidence matrix B (M x N).

        For edge k from source i to target j:
            B[k, i] = -1
            B[k, j] = +1

        so that B @ z gives predicted edge values (z_j - z_i).
        """
        M = self.graph_data.num_edges
        N = self.graph_data.num_nodes
        B = np.zeros((M, N))
        for k in range(M):
            i = self.graph_data.source_nodes[k]
            j = self.graph_data.target_nodes[k]
            B[k, i] = -1.0
            B[k, j] = 1.0
        return B

    # ------------------------------------------------------------------
    # Core solve
    # ------------------------------------------------------------------
    def _solve(self) -> None:
        """Solve for node values using the (weighted) graph Laplacian pseudoinverse."""
        B = self._B
        x = np.array(self.graph_data.edge_values)
        M, N = B.shape

        # Weight vector
        if self.use_weights and self.graph_data.edge_weights is not None:
            w = np.array(self.graph_data.edge_weights)
        else:
            if self.use_weights and self.graph_data.edge_weights is None:
                warnings.warn(
                    "use_weights=True but no edge errors available. "
                    "Falling back to uniform weights (SFC).",
                    UserWarning,
                )
            w = np.ones(M)

        self._W = w

        # Weighted Laplacian: L_W = B^T W B
        W_diag = np.diag(w)
        L_W = B.T @ W_diag @ B  # (N x N)

        # Right-hand side: B^T W x
        rhs = B.T @ (w * x)  # (N,)

        # Solve via pseudoinverse (lstsq handles rank-deficiency from gauge freedom)
        self._z, _, _, _ = np.linalg.lstsq(L_W, rhs, rcond=None)

        # Store pseudoinverse for uncertainty estimation
        self._L_W_pinv = np.linalg.pinv(L_W)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def fit(self) -> None:
        """Fit the model: extract data, build matrices, solve, and populate estimates."""
        self.graph_data = self._extract_graph_data()
        self._B = self._build_incidence_matrix()
        self._solve()

        # Populate node estimates
        self.node_estimates = {
            self.graph_data.idx_to_node[i]: float(self._z[i])
            for i in range(self.graph_data.num_nodes)
        }

        # Edge estimates from node differences
        self.edge_estimates = {}
        for k in range(self.graph_data.num_edges):
            src_name = self.graph_data.idx_to_node[self.graph_data.source_nodes[k]]
            dst_name = self.graph_data.idx_to_node[self.graph_data.target_nodes[k]]
            self.edge_estimates[(src_name, dst_name)] = (
                self.node_estimates[dst_name] - self.node_estimates[src_name]
            )

        # Node uncertainties from diagonal of L_W pseudoinverse
        self.node_uncertainties = {}
        diag_pinv = np.diag(self._L_W_pinv)
        for i in range(self.graph_data.num_nodes):
            name = self.graph_data.idx_to_node[i]
            self.node_uncertainties[name] = float(np.sqrt(max(diag_pinv[i], 0.0)))

        # Propagate to edge uncertainties
        self.edge_uncertainties = {}
        for k in range(self.graph_data.num_edges):
            src_name = self.graph_data.idx_to_node[self.graph_data.source_nodes[k]]
            dst_name = self.graph_data.idx_to_node[self.graph_data.target_nodes[k]]
            src_unc = self.node_uncertainties[src_name]
            dst_unc = self.node_uncertainties[dst_name]
            self.edge_uncertainties[(src_name, dst_name)] = float(
                np.sqrt(src_unc ** 2 + dst_unc ** 2)
            )

    def add_predictions_to_dataset(self) -> None:
        """Add model predictions to the dataset DataFrames."""
        if self.node_estimates is None:
            raise ValueError("Model must be fitted before adding predictions.")

        suffix = "WSFC" if self.use_weights and self.graph_data.edge_weights is not None else "SFC"

        # Node predictions
        nodes_df = getattr(self.dataset, "dataset_nodes", None)
        if nodes_df is not None:
            node_preds = []
            node_uncs = []
            for ligand in nodes_df["Name"]:
                node_preds.append(
                    self.node_estimates.get(ligand, float("nan"))
                )
                if self.node_uncertainties is not None:
                    node_uncs.append(
                        self.node_uncertainties.get(ligand, float("nan"))
                    )
                else:
                    node_uncs.append(float("nan"))
            nodes_df[suffix] = node_preds
            if self.node_uncertainties is not None:
                nodes_df[f"{suffix}_uncertainty"] = node_uncs

        # Edge predictions
        edges_df = getattr(self.dataset, "dataset_edges", None)
        if edges_df is not None:
            edge_preds = []
            edge_uncs = []
            for _, row in edges_df.iterrows():
                if "Source" in edges_df.columns and "Destination" in edges_df.columns:
                    edge_key = (row["Source"], row["Destination"])
                elif "Ligand1" in edges_df.columns and "Ligand2" in edges_df.columns:
                    edge_key = (row["Ligand1"], row["Ligand2"])
                else:
                    raise ValueError(
                        "Edge data must have 'Source'/'Destination' or 'Ligand1'/'Ligand2' columns"
                    )

                edge_preds.append(
                    self.edge_estimates.get(edge_key, float("nan"))
                )
                if self.edge_uncertainties is not None:
                    edge_uncs.append(
                        self.edge_uncertainties.get(edge_key, float("nan"))
                    )
                else:
                    edge_uncs.append(float("nan"))

            edges_df[suffix] = edge_preds
            if self.edge_uncertainties is not None:
                edges_df[f"{suffix}_uncertainty"] = edge_uncs

        # Register estimator
        if hasattr(self.dataset, "estimators"):
            if suffix not in self.dataset.estimators:
                self.dataset.estimators.append(suffix)

    def get_results(self) -> Dict[str, Any]:
        """Return results in a dict compatible with PerformanceTracker."""
        if self.node_estimates is None:
            raise ValueError("Model must be fitted before getting results.")
        suffix = "WSFC" if self.use_weights and self.graph_data.edge_weights is not None else "SFC"
        return {
            "node_estimates": self.node_estimates,
            "edge_estimates": self.edge_estimates,
            "node_uncertainties": self.node_uncertainties,
            "edge_uncertainties": self.edge_uncertainties,
            "method": suffix,
            "num_nodes": self.graph_data.num_nodes,
            "num_edges": self.graph_data.num_edges,
        }
