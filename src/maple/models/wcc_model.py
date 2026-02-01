"""
Weighted Cycle Closure (WCC) Model for FEP Analysis

This module implements the Weighted Cycle Closure method for correcting
free energy perturbation (FEP) calculations in molecular networks.

Reference:
Li et al. (2022) "Weighted Cycle Closure for Free Energy Calculations"
Journal of Chemical Information and Modeling
GitHub: https://github.com/zlisysu/Weighted_cc
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
import time
import warnings

import numpy as np
import pandas as pd

from ..dataset import BaseDataset


@dataclass
class GraphData:
    """Container for graph-structured FEP data"""
    source_nodes: List[int]
    target_nodes: List[int]
    edge_values: List[float]
    edge_weights: List[float]  # Weights for WCC algorithm (inverse of uncertainties)
    num_nodes: int
    num_edges: int
    node_to_idx: Dict[str, int]
    idx_to_node: Dict[int, str]


class WCC_model:
    """
    Weighted Cycle Closure Model for FEP Analysis

    This model implements an iterative cycle closure correction algorithm
    that adjusts edge values to minimize cycle closure errors in FEP networks.

    The algorithm:
    1. Detects all cycles in the graph
    2. Iteratively corrects edge values based on cycle closure errors
    3. Derives node values from corrected edges
    4. Calculates uncertainties based on path variance

    Attributes
    ----------
    dataset : BaseDataset
        Dataset containing the FEP data
    tolerance : float
        Convergence tolerance for iterative correction
    max_iterations : int
        Maximum number of correction iterations
    use_weights : bool
        Whether to use edge weights in correction
    graph_data : Optional[GraphData]
        Processed graph data for inference
    node_estimates : Optional[Dict[str, float]]
        Estimated node values after correction
    edge_estimates : Optional[Dict[Tuple[str, str], float]]
        Corrected edge values
    node_uncertainties : Optional[Dict[str, float]]
        Node value uncertainties
    edge_uncertainties : Optional[Dict[Tuple[str, str], float]]
        Edge value uncertainties
    """

    def __init__(
        self,
        dataset: BaseDataset,
        config: Optional['WCCConfig'] = None,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        use_weights: bool = True,
        max_cycle_length: int = 8,
        max_cycles: int = 100000,
        cycle_detection_timeout: float = 300.0
    ):
        """
        Initialize the WCC model.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset containing FEP data
        config : WCCConfig, optional
            Configuration object with model parameters
        tolerance : float, optional
            Convergence tolerance (default: 1e-6)
        max_iterations : int, optional
            Maximum iterations (default: 1000)
        use_weights : bool, optional
            Use edge weights in correction (default: True)
        max_cycle_length : int, optional
            Maximum cycle length to detect (default: 8)
            Larger values find more cycles but are exponentially slower
        max_cycles : int, optional
            Maximum number of cycles to find (default: 100000)
        cycle_detection_timeout : float, optional
            Timeout in seconds for cycle detection (default: 300s = 5min)
        """
        self.dataset = dataset

        # Use config if provided, otherwise use individual parameters
        if config is not None:
            # Check if config has the expected attributes (duck typing)
            # This avoids circular import issues with isinstance checks
            required_attrs = ['tolerance', 'max_iterations', 'use_weights',
                            'max_cycle_length', 'max_cycles', 'cycle_detection_timeout']

            if not all(hasattr(config, attr) for attr in required_attrs):
                raise TypeError(
                    f"Config must be a WCCConfig object with attributes: {required_attrs}, "
                    f"got {type(config)} with attributes: {dir(config)}"
                )

            self.tolerance = config.tolerance
            self.max_iterations = config.max_iterations
            self.use_weights = config.use_weights
            self.max_cycle_length = config.max_cycle_length
            self.max_cycles = config.max_cycles
            self.cycle_detection_timeout = config.cycle_detection_timeout
        else:
            self.tolerance = tolerance
            self.max_iterations = max_iterations
            self.use_weights = use_weights
            self.max_cycle_length = max_cycle_length
            self.max_cycles = max_cycles
            self.cycle_detection_timeout = cycle_detection_timeout

        # Initialize attributes
        self.graph_data: Optional[GraphData] = None
        self.node_estimates: Optional[Dict[str, float]] = None
        self.edge_estimates: Optional[Dict[Tuple[str, str], float]] = None
        self.node_uncertainties: Optional[Dict[str, float]] = None
        self.edge_uncertainties: Optional[Dict[Tuple[str, str], float]] = None

        # For cycle detection
        self.adjacency_list: Dict[int, List[int]] = defaultdict(list)
        self.cycles: List[List[int]] = []
        self.cycle_detection_stopped_early: bool = False
        self.cycle_detection_reason: str = ""
        self.used_exhaustive_search: bool = False
        self.detection_method: str = ""  # Will be "exhaustive", "bounded", or "timeout_fallback"

        # Corrected values
        self.corrected_edges: Dict[Tuple[int, int], float] = {}
        self.iteration_count: int = 0

    def _extract_graph_data(self) -> GraphData:
        """
        Extract and process graph data from the dataset.

        Returns
        -------
        GraphData
            Processed graph data for inference
        """
        edges_df = getattr(self.dataset, "dataset_edges", None)

        if edges_df is None:
            raise ValueError("Dataset must have 'dataset_edges' attribute")

        source_nodes = []
        target_nodes = []
        edge_values = []
        edge_weights = []
        node_to_idx = {}
        idx_to_node = {}
        idx = 0

        # Process each edge
        for _, row in edges_df.iterrows():
            # Handle column names
            if "Source" in edges_df.columns and "Destination" in edges_df.columns:
                ligand1, ligand2 = row["Source"], row["Destination"]
            elif "Ligand1" in edges_df.columns and "Ligand2" in edges_df.columns:
                ligand1, ligand2 = row["Ligand1"], row["Ligand2"]
            else:
                raise ValueError(
                    "Edge data must have either 'Source'/'Destination' or 'Ligand1'/'Ligand2' columns"
                )

            # Get FEP value
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

            # Get weight (inverse of uncertainty if available)
            weight = 1.0  # Default weight
            if self.use_weights:
                # Look for uncertainty/error columns
                for col in ["Uncertainty", "Error", "Std", "StdDev", "dDeltaG"]:
                    if col in edges_df.columns:
                        uncertainty = float(row[col])
                        if uncertainty > 0:
                            weight = 1.0 / (uncertainty ** 2)
                        break

            # Add nodes to mapping
            if ligand1 not in node_to_idx:
                node_to_idx[ligand1] = idx
                idx_to_node[idx] = ligand1
                idx += 1

            if ligand2 not in node_to_idx:
                node_to_idx[ligand2] = idx
                idx_to_node[idx] = ligand2
                idx += 1

            # Add edge data (0-indexed)
            source_nodes.append(node_to_idx[ligand1])
            target_nodes.append(node_to_idx[ligand2])
            edge_values.append(fep_value)
            edge_weights.append(weight)

        return GraphData(
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            edge_values=edge_values,
            edge_weights=edge_weights,
            num_nodes=len(node_to_idx),
            num_edges=len(edge_values),
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
        )

    def _build_adjacency_list(self):
        """Build adjacency list for cycle detection (bidirectional graph)."""
        self.adjacency_list = defaultdict(list)

        for i in range(self.graph_data.num_edges):
            src = self.graph_data.source_nodes[i]
            dst = self.graph_data.target_nodes[i]

            # Add both directions for undirected graph
            if dst not in self.adjacency_list[src]:
                self.adjacency_list[src].append(dst)
            if src not in self.adjacency_list[dst]:
                self.adjacency_list[dst].append(src)

    def _find_cycles_dfs(self, start_node: int, current_node: int,
                         visited: Set[int], path: List[int],
                         start_time: float, max_depth: Optional[int] = None) -> bool:
        """
        Find cycles using depth-first search.

        Parameters
        ----------
        start_node : int
            Starting node for cycle detection
        current_node : int
            Current node being visited
        visited : Set[int]
            Set of visited nodes in current path
        path : List[int]
            Current path being explored
        start_time : float
            Start time for timeout checking
        max_depth : int, optional
            Maximum cycle length to search for. If None, searches exhaustively.

        Returns
        -------
        bool
            True if should continue searching, False if limits reached
        """
        # Check timeout
        if time.time() - start_time > self.cycle_detection_timeout:
            self.cycle_detection_stopped_early = True
            self.cycle_detection_reason = f"timeout ({self.cycle_detection_timeout}s)"
            return False

        # Check max cycles limit
        if len(self.cycles) >= self.max_cycles:
            self.cycle_detection_stopped_early = True
            self.cycle_detection_reason = f"max cycles limit ({self.max_cycles})"
            return False

        # Check cycle length limit (only if max_depth is set)
        if max_depth is not None and len(path) > max_depth:
            return True  # Continue but don't go deeper on this path

        visited.add(current_node)
        path.append(current_node)

        for neighbor in self.adjacency_list[current_node]:
            # Found a cycle back to start
            if neighbor == start_node and len(path) >= 3:
                cycle = path.copy()
                # Normalize cycle representation (start with smallest node)
                min_idx = cycle.index(min(cycle))
                normalized_cycle = cycle[min_idx:] + cycle[:min_idx]

                # Check if cycle already exists (avoid duplicates)
                if normalized_cycle not in self.cycles:
                    self.cycles.append(normalized_cycle)

                    # Check limits after adding
                    if len(self.cycles) >= self.max_cycles:
                        self.cycle_detection_stopped_early = True
                        self.cycle_detection_reason = f"max cycles limit ({self.max_cycles})"
                        visited.remove(current_node)
                        path.pop()
                        return False

            # Continue DFS if not visited
            # Only check depth limit if max_depth is set
            elif neighbor not in visited:
                if max_depth is None or len(path) < max_depth:
                    should_continue = self._find_cycles_dfs(start_node, neighbor, visited, path, start_time, max_depth)
                    if not should_continue:
                        visited.remove(current_node)
                        path.pop()
                        return False

        # Backtrack
        visited.remove(current_node)
        path.pop()
        return True

    def _detect_all_cycles(self):
        """
        Adaptively detect cycles in the graph.

        Strategy:
        1. First try exhaustive search (no depth limit) with timeout
        2. If timeout occurs, fall back to bounded search with max_cycle_length
        3. This ensures we get all cycles for small/sparse graphs,
           but still work for large/dense graphs
        """
        self.cycles = []
        self.cycle_detection_stopped_early = False
        self.cycle_detection_reason = ""

        print(f"\n{'='*60}")
        print("Cycle Detection - Adaptive Strategy")
        print(f"{'='*60}")
        print(f"  Timeout: {self.cycle_detection_timeout}s")
        print(f"  Fallback max cycle length: {self.max_cycle_length}")
        print(f"  Max cycles limit: {self.max_cycles}")
        print(f"{'='*60}\n")

        # PHASE 1: Try exhaustive search first
        print("Phase 1: Attempting exhaustive cycle detection...")
        print("(Will automatically switch to bounded search if this takes too long)\n")

        start_time = time.time()
        exhaustive_succeeded = True

        # Try starting from each node with no depth limit
        for node in range(self.graph_data.num_nodes):
            should_continue = self._find_cycles_dfs(node, node, set(), [], start_time, max_depth=None)

            # Show progress for large graphs
            if node % 10 == 0 and node > 0:
                elapsed = time.time() - start_time
                print(f"  Progress: Checked {node}/{self.graph_data.num_nodes} starting nodes, "
                      f"found {len(self.cycles)} cycles (elapsed: {elapsed:.1f}s)")

            if not should_continue:
                # Check if we stopped due to timeout
                if "timeout" in self.cycle_detection_reason:
                    exhaustive_succeeded = False
                    elapsed_so_far = time.time() - start_time
                    print(f"\n  ⏱️  Exhaustive search timed out after {elapsed_so_far:.1f}s")
                    print(f"  Found {len(self.cycles)} cycles before timeout\n")
                break

        elapsed_phase1 = time.time() - start_time

        # PHASE 2: If exhaustive search timed out, restart with bounded search
        if not exhaustive_succeeded and "timeout" in self.cycle_detection_reason:
            print(f"{'='*60}")
            print(f"Phase 2: Switching to bounded search (max length={self.max_cycle_length})")
            print(f"{'='*60}\n")

            # Clear cycles and reset flags
            self.cycles = []
            self.cycle_detection_stopped_early = False
            old_reason = self.cycle_detection_reason
            self.cycle_detection_reason = ""

            # Restart with bounded search
            start_time_phase2 = time.time()

            for node in range(self.graph_data.num_nodes):
                should_continue = self._find_cycles_dfs(
                    node, node, set(), [], start_time_phase2, max_depth=self.max_cycle_length
                )

                # Show progress
                if node % 10 == 0 and node > 0:
                    elapsed = time.time() - start_time_phase2
                    print(f"  Progress: Checked {node}/{self.graph_data.num_nodes} starting nodes, "
                          f"found {len(self.cycles)} cycles (elapsed: {elapsed:.1f}s)")

                if not should_continue:
                    break

            elapsed_phase2 = time.time() - start_time_phase2
            total_elapsed = elapsed_phase1 + elapsed_phase2

            self.detection_method = "timeout_fallback"
            self.used_exhaustive_search = False

            print(f"\n{'='*60}")
            print(f"Cycle Detection Complete (took {total_elapsed:.1f}s total)")
            print(f"  Method: Bounded search (fell back from exhaustive due to timeout)")
            print(f"  Cycles found: {len(self.cycles)}")
            print(f"  Max cycle length searched: {self.max_cycle_length}")

            if self.cycle_detection_stopped_early:
                print(f"  Status: STOPPED EARLY ({self.cycle_detection_reason})")
            else:
                print(f"  Status: Complete (all cycles up to length {self.max_cycle_length})")

            print(f"{'='*60}\n")

            warnings.warn(
                f"\n⚠️  Exhaustive cycle search timed out - switched to bounded search\n"
                f"   Found all cycles up to length {self.max_cycle_length} ({len(self.cycles)} total)\n"
                f"   This is normal for dense graphs and won't affect correction quality.",
                UserWarning
            )

        else:
            # Exhaustive search completed successfully
            self.detection_method = "exhaustive"
            self.used_exhaustive_search = True

            print(f"\n{'='*60}")
            print(f"Cycle Detection Complete (took {elapsed_phase1:.1f}s)")
            print(f"  Method: Exhaustive (found ALL cycles)")
            print(f"  Cycles found: {len(self.cycles)}")

            if self.cycle_detection_stopped_early:
                print(f"  Status: STOPPED EARLY ({self.cycle_detection_reason})")
            else:
                print(f"  ✅ Status: Complete (found ALL cycles in the graph)")

            print(f"{'='*60}\n")

        # Final check
        if len(self.cycles) == 0:
            warnings.warn(
                "⚠️  No cycles detected! Graph may be a tree.\n"
                "   WCC will use original edge values without cycle correction.",
                UserWarning
            )

    def _get_edge_value(self, node1: int, node2: int) -> float:
        """
        Get edge value between two nodes (directed).

        Returns the corrected value if available, otherwise original value.
        Edge direction matters: positive if node1 -> node2, negative if node2 -> node1.
        """
        # Check corrected edges first
        if (node1, node2) in self.corrected_edges:
            return self.corrected_edges[(node1, node2)]
        elif (node2, node1) in self.corrected_edges:
            return -self.corrected_edges[(node2, node1)]

        # Fall back to original edges
        for i in range(self.graph_data.num_edges):
            if (self.graph_data.source_nodes[i] == node1 and
                self.graph_data.target_nodes[i] == node2):
                return self.graph_data.edge_values[i]
            elif (self.graph_data.source_nodes[i] == node2 and
                  self.graph_data.target_nodes[i] == node1):
                return -self.graph_data.edge_values[i]

        return 0.0

    def _get_edge_weight(self, node1: int, node2: int) -> float:
        """Get edge weight between two nodes."""
        for i in range(self.graph_data.num_edges):
            if ((self.graph_data.source_nodes[i] == node1 and
                 self.graph_data.target_nodes[i] == node2) or
                (self.graph_data.source_nodes[i] == node2 and
                 self.graph_data.target_nodes[i] == node1)):
                return self.graph_data.edge_weights[i]

        return 1.0

    def _calculate_cycle_closure_error(self, cycle: List[int]) -> float:
        """
        Calculate cycle closure error for a given cycle.

        The error is the sum of edge values around the cycle,
        which should be zero in a thermodynamically consistent network.
        """
        error = 0.0
        for i in range(len(cycle)):
            node1 = cycle[i]
            node2 = cycle[(i + 1) % len(cycle)]
            error += self._get_edge_value(node1, node2)

        return error

    def _correct_cycle(self, cycle: List[int]):
        """
        Correct edges in a cycle by distributing the closure error.

        The correction is weighted by edge weights (inverse variance).
        """
        # Calculate cycle closure error
        cycle_error = self._calculate_cycle_closure_error(cycle)

        if abs(cycle_error) < self.tolerance:
            return

        # Calculate total weight for the cycle
        total_weight = 0.0
        edge_weights_in_cycle = []

        for i in range(len(cycle)):
            node1 = cycle[i]
            node2 = cycle[(i + 1) % len(cycle)]
            weight = self._get_edge_weight(node1, node2)
            edge_weights_in_cycle.append((node1, node2, weight))
            total_weight += weight

        # Distribute correction proportional to weights
        for node1, node2, weight in edge_weights_in_cycle:
            correction = -cycle_error * weight / total_weight

            # Get current value
            current_value = self._get_edge_value(node1, node2)

            # Apply correction (maintain direction)
            # Store in canonical form (smaller index first)
            if node1 < node2:
                self.corrected_edges[(node1, node2)] = current_value + correction
            else:
                self.corrected_edges[(node2, node1)] = -(current_value + correction)

    def _iterate_cycle_closure(self):
        """
        Iteratively correct all cycles until convergence.
        """
        print(f"\n{'='*60}")
        print("Weighted Cycle Closure Iteration")
        print(f"{'='*60}")
        print(f"  Correcting {len(self.cycles)} cycles")
        print(f"  Tolerance: {self.tolerance:.2e}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"{'='*60}\n")

        # Initialize corrected edges with original values
        self.corrected_edges = {}
        for i in range(self.graph_data.num_edges):
            src = self.graph_data.source_nodes[i]
            dst = self.graph_data.target_nodes[i]
            val = self.graph_data.edge_values[i]

            # Store in canonical form (smaller index first)
            if src < dst:
                self.corrected_edges[(src, dst)] = val
            else:
                self.corrected_edges[(dst, src)] = -val

        # Calculate initial cycle errors
        initial_max_error = 0.0
        initial_mean_error = 0.0
        for cycle in self.cycles:
            error = abs(self._calculate_cycle_closure_error(cycle))
            initial_max_error = max(initial_max_error, error)
            initial_mean_error += error
        initial_mean_error /= len(self.cycles) if len(self.cycles) > 0 else 1

        print(f"Initial cycle errors:")
        print(f"  Max error:  {initial_max_error:.6e}")
        print(f"  Mean error: {initial_mean_error:.6e}\n")

        # Track convergence history
        self.convergence_history = []

        # Iterate until convergence
        for iteration in range(self.max_iterations):
            max_error = 0.0
            mean_error = 0.0

            # Correct each cycle
            for cycle in self.cycles:
                error = abs(self._calculate_cycle_closure_error(cycle))
                max_error = max(max_error, error)
                mean_error += error
                self._correct_cycle(cycle)

            mean_error /= len(self.cycles) if len(self.cycles) > 0 else 1
            self.convergence_history.append({'iteration': iteration, 'max_error': max_error, 'mean_error': mean_error})

            # Print progress
            if iteration % 50 == 0 or iteration < 10:
                print(f"  Iter {iteration:4d}: Max error = {max_error:.6e}, Mean error = {mean_error:.6e}")

            # Check convergence
            if max_error < self.tolerance:
                print(f"\n{'='*60}")
                print(f"✅ CONVERGED after {iteration + 1} iterations!")
                print(f"{'='*60}")
                print(f"Final cycle errors:")
                print(f"  Max error:  {max_error:.6e} (target: {self.tolerance:.2e})")
                print(f"  Mean error: {mean_error:.6e}")
                print(f"  Improvement: {(1 - max_error/initial_max_error)*100:.2f}% reduction in max error")
                print(f"{'='*60}\n")
                self.iteration_count = iteration + 1
                self.final_max_error = max_error
                self.final_mean_error = mean_error
                break
        else:
            warnings.warn(
                f"\n⚠️  Did not fully converge after {self.max_iterations} iterations!\n"
                f"   Final max error: {max_error:.6e} (target: {self.tolerance:.2e})\n"
                f"   Consider increasing max_iterations or relaxing tolerance.",
                UserWarning
            )
            print(f"\n{'='*60}")
            print(f"⚠️  Reached maximum iterations ({self.max_iterations})")
            print(f"{'='*60}")
            print(f"Final cycle errors:")
            print(f"  Max error:  {max_error:.6e} (target: {self.tolerance:.2e})")
            print(f"  Mean error: {mean_error:.6e}")
            print(f"  Improvement: {(1 - max_error/initial_max_error)*100:.2f}% reduction in max error")
            print(f"{'='*60}\n")
            self.iteration_count = self.max_iterations
            self.final_max_error = max_error
            self.final_mean_error = mean_error

    def _calculate_node_values(self, reference_node: Optional[str] = None):
        """
        Calculate node values from corrected edges using BFS from reference node.

        Parameters
        ----------
        reference_node : str, optional
            Reference node name. If None, uses the first node or one from dataset.
        """
        # Determine reference node
        if reference_node is None:
            # Try to get reference from dataset nodes
            nodes_df = getattr(self.dataset, "dataset_nodes", None)
            if nodes_df is not None and "Exp. DeltaG" in nodes_df.columns:
                # Use node with experimental value closest to 0 as reference
                ref_idx = nodes_df["Exp. DeltaG"].abs().idxmin()
                reference_node = nodes_df.loc[ref_idx, "Name"]
            else:
                # Use first node as reference
                reference_node = self.graph_data.idx_to_node[0]

        # Get reference node index
        ref_idx = self.graph_data.node_to_idx[reference_node]

        # Initialize node values
        node_values = {ref_idx: 0.0}  # Reference node at 0

        # BFS to calculate all node values
        queue = deque([ref_idx])
        visited = {ref_idx}

        while queue:
            current = queue.popleft()

            for neighbor in self.adjacency_list[current]:
                if neighbor not in visited:
                    # Calculate neighbor value from current value
                    edge_value = self._get_edge_value(current, neighbor)
                    node_values[neighbor] = node_values[current] + edge_value

                    visited.add(neighbor)
                    queue.append(neighbor)

        # Convert to dictionary with node names
        self.node_estimates = {
            self.graph_data.idx_to_node[idx]: value
            for idx, value in node_values.items()
        }

        # Calculate edge estimates from node values
        self.edge_estimates = {}
        for i in range(self.graph_data.num_edges):
            src_name = self.graph_data.idx_to_node[self.graph_data.source_nodes[i]]
            dst_name = self.graph_data.idx_to_node[self.graph_data.target_nodes[i]]

            edge_value = self.node_estimates[dst_name] - self.node_estimates[src_name]
            self.edge_estimates[(src_name, dst_name)] = edge_value

    def _calculate_uncertainties(self):
        """
        Calculate uncertainties based on path variance.

        For simplicity, we estimate uncertainties by:
        1. Edge uncertainties from input data or residual variance
        2. Node uncertainties from path length and edge uncertainties
        """
        # Calculate edge uncertainties
        self.edge_uncertainties = {}
        edges_df = getattr(self.dataset, "dataset_edges", None)

        # Calculate residuals
        residuals = []
        for i in range(self.graph_data.num_edges):
            src_name = self.graph_data.idx_to_node[self.graph_data.source_nodes[i]]
            dst_name = self.graph_data.idx_to_node[self.graph_data.target_nodes[i]]

            original = self.graph_data.edge_values[i]
            corrected = self.edge_estimates[(src_name, dst_name)]
            residuals.append(abs(corrected - original))

        # Use mean residual as baseline uncertainty
        mean_residual = np.mean(residuals) if residuals else 0.1

        # Set edge uncertainties
        for i in range(self.graph_data.num_edges):
            src_name = self.graph_data.idx_to_node[self.graph_data.source_nodes[i]]
            dst_name = self.graph_data.idx_to_node[self.graph_data.target_nodes[i]]

            # Use input uncertainty if available
            uncertainty = None
            if edges_df is not None:
                for col in ["Uncertainty", "Error", "Std", "StdDev", "dDeltaG"]:
                    if col in edges_df.columns:
                        row_idx = edges_df[
                            ((edges_df.get("Source", edges_df.get("Ligand1")) == src_name) &
                             (edges_df.get("Destination", edges_df.get("Ligand2")) == dst_name))
                        ].index
                        if len(row_idx) > 0:
                            uncertainty = float(edges_df.loc[row_idx[0], col])
                        break

            # Fall back to residual-based uncertainty
            if uncertainty is None:
                uncertainty = max(mean_residual, 0.1)

            self.edge_uncertainties[(src_name, dst_name)] = uncertainty

        # Calculate node uncertainties (propagate from edges)
        self.node_uncertainties = {}

        # Reference node has zero uncertainty
        ref_node = list(self.node_estimates.keys())[0]
        self.node_uncertainties[ref_node] = 0.0

        # Propagate uncertainties using BFS
        visited = {ref_node}
        queue = deque([ref_node])

        while queue:
            current = queue.popleft()
            current_idx = self.graph_data.node_to_idx[current]

            for neighbor_idx in self.adjacency_list[current_idx]:
                neighbor = self.graph_data.idx_to_node[neighbor_idx]

                if neighbor not in visited:
                    # Find edge uncertainty
                    edge_key = (current, neighbor) if (current, neighbor) in self.edge_uncertainties else (neighbor, current)
                    edge_unc = self.edge_uncertainties.get(edge_key, 0.1)

                    # Propagate uncertainty (sum of variances)
                    current_unc = self.node_uncertainties[current]
                    self.node_uncertainties[neighbor] = np.sqrt(current_unc**2 + edge_unc**2)

                    visited.add(neighbor)
                    queue.append(neighbor)

    def fit(self):
        """
        Fit the WCC model by running the cycle closure algorithm.
        """
        print("=" * 60)
        print("Weighted Cycle Closure (WCC) Model")
        print("=" * 60)

        # Extract graph data
        print("\nExtracting graph data...")
        self.graph_data = self._extract_graph_data()
        print(f"Nodes: {self.graph_data.num_nodes}, Edges: {self.graph_data.num_edges}")

        # Build adjacency list
        print("\nBuilding adjacency list...")
        self._build_adjacency_list()

        # Detect cycles
        print("\nDetecting cycles...")
        self._detect_all_cycles()

        if len(self.cycles) == 0:
            print("Warning: No cycles detected. Graph may be a tree.")
            print("Using original edge values without correction.")

            # Just calculate node values without correction
            self.corrected_edges = {}
            for i in range(self.graph_data.num_edges):
                src = self.graph_data.source_nodes[i]
                dst = self.graph_data.target_nodes[i]
                val = self.graph_data.edge_values[i]
                if src < dst:
                    self.corrected_edges[(src, dst)] = val
                else:
                    self.corrected_edges[(dst, src)] = -val
        else:
            # Perform iterative cycle closure
            print("\nPerforming cycle closure correction...")
            self._iterate_cycle_closure()

        # Calculate node values
        print("\nCalculating node values from corrected edges...")
        self._calculate_node_values()

        # Calculate uncertainties
        print("\nCalculating uncertainties...")
        self._calculate_uncertainties()

        print("\n" + "=" * 60)
        print("WCC model fitting complete!")
        print("=" * 60)

    def get_correction_quality_report(self) -> Dict[str, Any]:
        """
        Generate a quality assessment report for the WCC correction.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing quality metrics and diagnostics
        """
        if self.node_estimates is None:
            raise ValueError("Model must be fitted before generating quality report")

        report = {
            # Cycle detection info
            'num_cycles_found': len(self.cycles),
            'max_cycle_length': self.max_cycle_length,
            'detection_stopped_early': self.cycle_detection_stopped_early,
            'detection_stop_reason': self.cycle_detection_reason if self.cycle_detection_stopped_early else 'complete',
            'detection_method': self.detection_method,
            'used_exhaustive_search': self.used_exhaustive_search,

            # Convergence info
            'converged': hasattr(self, 'final_max_error') and self.final_max_error < self.tolerance,
            'num_iterations': self.iteration_count,
            'max_iterations_limit': self.max_iterations,
            'final_max_error': getattr(self, 'final_max_error', None),
            'final_mean_error': getattr(self, 'final_mean_error', None),
            'tolerance': self.tolerance,

            # Graph info
            'num_nodes': self.graph_data.num_nodes,
            'num_edges': self.graph_data.num_edges,
            'graph_density': self.graph_data.num_edges / (self.graph_data.num_nodes * (self.graph_data.num_nodes - 1) / 2),
        }

        # Calculate cycle length distribution
        cycle_lengths = [len(cycle) for cycle in self.cycles]
        if cycle_lengths:
            report['cycle_length_distribution'] = {
                'min': min(cycle_lengths),
                'max': max(cycle_lengths),
                'mean': np.mean(cycle_lengths),
                'median': np.median(cycle_lengths),
            }

        return report

    def print_quality_report(self):
        """Print a formatted quality assessment report."""
        report = self.get_correction_quality_report()

        print(f"\n{'='*70}")
        print("WCC CORRECTION QUALITY REPORT")
        print(f"{'='*70}")

        print(f"\n📊 Graph Statistics:")
        print(f"   Nodes: {report['num_nodes']}")
        print(f"   Edges: {report['num_edges']}")
        print(f"   Graph density: {report['graph_density']:.2%}")

        print(f"\n🔄 Cycle Detection:")
        print(f"   Cycles found: {report['num_cycles_found']:,}")

        # Show detection method
        if report['detection_method'] == 'exhaustive':
            print(f"   ✅ Method: EXHAUSTIVE (found ALL cycles)")
        elif report['detection_method'] == 'timeout_fallback':
            print(f"   Method: Bounded search (exhaustive timed out)")
            print(f"   Max cycle length: {report['max_cycle_length']}")
        elif report['detection_method'] == 'bounded':
            print(f"   Method: Bounded search")
            print(f"   Max cycle length: {report['max_cycle_length']}")

        if report['detection_stopped_early']:
            print(f"   ⚠️  Status: STOPPED EARLY ({report['detection_stop_reason']})")
            if not report['used_exhaustive_search']:
                print(f"   Note: Using cycles up to length {report['max_cycle_length']}")
        else:
            if report['used_exhaustive_search']:
                print(f"   ✅ Status: Complete (ALL cycles found)")
            else:
                print(f"   ✅ Status: Complete (all cycles up to length {report['max_cycle_length']})")

        if 'cycle_length_distribution' in report:
            dist = report['cycle_length_distribution']
            print(f"   Cycle lengths: min={dist['min']}, max={dist['max']}, "
                  f"mean={dist['mean']:.1f}, median={dist['median']:.1f}")

        print(f"\n🎯 Convergence:")
        if report['converged']:
            print(f"   ✅ CONVERGED in {report['num_iterations']} iterations")
        else:
            print(f"   ⚠️  DID NOT FULLY CONVERGE ({report['num_iterations']}/{report['max_iterations_limit']} iterations)")

        if report['final_max_error'] is not None:
            print(f"   Final max error: {report['final_max_error']:.6e} (target: {report['tolerance']:.2e})")
            print(f"   Final mean error: {report['final_mean_error']:.6e}")

            if report['final_max_error'] < report['tolerance']:
                print(f"   ✅ All cycles closed within tolerance!")
            else:
                print(f"   ⚠️  Max error exceeds tolerance by {report['final_max_error']/report['tolerance']:.2f}x")

        print(f"\n💡 Quality Assessment:")
        quality_score = 0
        quality_issues = []

        # Check convergence (40 points)
        if report['converged']:
            quality_score += 40
        else:
            quality_issues.append("Did not fully converge - consider increasing max_iterations")

        # Check cycle coverage (30-35 points) - BONUS for exhaustive search!
        if report['used_exhaustive_search'] and not report['detection_stopped_early']:
            # Exhaustive search completed successfully - this is the best possible outcome
            quality_score += 35
        elif not report['detection_stopped_early']:
            quality_score += 30
        elif report['num_cycles_found'] > 10000:
            quality_score += 20
            quality_issues.append("Stopped early but found many cycles - results should be good")
        else:
            quality_issues.append("Few cycles found - consider increasing max_cycle_length or max_cycles")

        # Check error levels
        if report['final_max_error'] is not None:
            if report['final_max_error'] < report['tolerance'] * 10:
                quality_score += 30
            elif report['final_max_error'] < report['tolerance'] * 100:
                quality_score += 15
                quality_issues.append("Errors moderately high - results still usable but not optimal")
            else:
                quality_issues.append("Errors very high - results may be unreliable")

        if quality_score >= 90:
            quality_status = "✅ EXCELLENT"
        elif quality_score >= 70:
            quality_status = "✅ GOOD"
        elif quality_score >= 50:
            quality_status = "⚠️  ACCEPTABLE"
        else:
            quality_status = "❌ POOR"

        print(f"   Overall quality: {quality_status} (score: {quality_score}/100)")

        if quality_issues:
            print(f"\n   Issues/Recommendations:")
            for issue in quality_issues:
                print(f"   • {issue}")

        print(f"\n{'='*70}\n")

    def add_predictions_to_dataset(self) -> None:
        """
        Add model predictions to the dataset for analysis.

        This method adds the node and edge estimates to the dataset
        for further analysis and comparison with experimental data.
        """
        if self.node_estimates is None:
            raise ValueError("Model must be fitted before adding predictions to dataset")

        suffix = "WCC"

        # Add node predictions to dataset
        nodes_df = getattr(self.dataset, "dataset_nodes", None)
        if nodes_df is not None:
            node_predictions = []
            node_uncertainties = []

            for ligand in nodes_df["Name"]:
                if ligand in self.node_estimates:
                    node_predictions.append(self.node_estimates[ligand])
                    if (self.node_uncertainties is not None and
                        ligand in self.node_uncertainties):
                        node_uncertainties.append(self.node_uncertainties[ligand])
                    else:
                        node_uncertainties.append(np.nan)
                else:
                    node_predictions.append(np.nan)
                    node_uncertainties.append(np.nan)

            nodes_df[suffix] = node_predictions

            if self.node_uncertainties is not None:
                nodes_df[f"{suffix}_uncertainty"] = node_uncertainties

        # Add edge predictions to dataset
        edges_df = getattr(self.dataset, "dataset_edges", None)
        if edges_df is not None:
            edge_predictions = []
            edge_uncertainties = []

            for _, row in edges_df.iterrows():
                # Handle column names
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
                    if (self.edge_uncertainties is not None and
                        edge_key in self.edge_uncertainties):
                        edge_uncertainties.append(self.edge_uncertainties[edge_key])
                    else:
                        edge_uncertainties.append(np.nan)
                else:
                    edge_predictions.append(np.nan)
                    edge_uncertainties.append(np.nan)

            edges_df[suffix] = edge_predictions

            if self.edge_uncertainties is not None:
                edges_df[f"{suffix}_uncertainty"] = edge_uncertainties

        # Add estimator name to dataset
        if hasattr(self.dataset, "estimators"):
            if suffix not in self.dataset.estimators:
                self.dataset.estimators.append(suffix)

        print(f"\n✅ Added WCC predictions to dataset with suffix '{suffix}'")
