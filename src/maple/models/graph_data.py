"""
Shared GraphData container for MAPLE models.

This module provides the GraphData dataclass used across all graph-based
models (VariationalEstimator, GaussianMixtureVI, CycleClosureCorrection, SpectralCorrection).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GraphData:
    """Container for graph-structured FEP data.

    Parameters
    ----------
    source_nodes : List[int]
        Source node indices for each edge.
    target_nodes : List[int]
        Target node indices for each edge.
    edge_values : List[float]
        FEP values for each edge (DeltaDeltaG).
    num_nodes : int
        Total number of nodes in the graph.
    num_edges : int
        Total number of edges in the graph.
    node_to_idx : Dict[str, int]
        Mapping from node names to indices.
    idx_to_node : Dict[int, str]
        Mapping from indices to node names.
    edge_errors : Optional[List[float]]
        Per-edge simulation uncertainties (sigma_ij). None if unavailable.
    edge_weights : Optional[List[float]]
        Per-edge weights (1/sigma^2), used by WCC and WSFC. None if unavailable.
    """

    source_nodes: List[int]
    target_nodes: List[int]
    edge_values: List[float]
    num_nodes: int
    num_edges: int
    node_to_idx: Dict[str, int]
    idx_to_node: Dict[int, str]
    edge_errors: Optional[List[float]] = None
    edge_weights: Optional[List[float]] = None
