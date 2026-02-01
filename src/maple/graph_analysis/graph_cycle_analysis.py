from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

from .graph_setup import GraphSetup


def _is_notebook():
    """Check if code is running in a Jupyter notebook environment."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None and 'IPKernelApp' in get_ipython().config:
            return True
    except (ImportError, AttributeError):
        pass
    return False


class GraphCycleAnalysis:
    """
    Analyze cycles and edge influence (leverages) for a graph built by GraphSetup.

    Inputs
    ------
    graph_setup : GraphSetup
        Prepared graph with standardized edge attributes (e.g., 'DeltaDeltaG', 'CCC', 'MAP').

    Outputs (via methods)
    ---------------------
    - Leverages per edge (design-matrix hat diagonals)
    - Cycle lists and cycle-closure errors for selected attributes
    """

    def __init__(self, graph_setup: GraphSetup) -> None:
        self.graph_setup = graph_setup
        self.graph: nx.DiGraph = graph_setup.graph
        # Node mapping from dataset (Name -> idx) and reverse
        self.node_to_idx, self.idx_to_node = graph_setup.node_mapping

        # Cache common structures
        self._edges_list: List[Tuple[Any, Any]] = list(self.graph.edges())
        self._nodes_list: List[Any] = list(self.graph.nodes())

    def compute_leverages(
        self,
        reference_node: Optional[Any] = None,
        ridge: float = 0.0,
        store_in_graph: bool = True,
    ) -> pd.DataFrame:
        """
        Compute leverage values for each edge using the design matrix hat matrix.

        The design matrix A has one row per directed edge and one column per node.
        For an edge (u -> v): A[row, u] = +1, A[row, v] = -1.

        To ensure full rank, an optional reference constraint row is appended where
        the reference node column is set to 1. If not provided, the first node in
        the graph is used.

        Parameters
        ----------
        reference_node : Any, optional
            Node to pin (adds a single constraint row). If None, uses first node.
        ridge : float, default 0.0
            Optional Tikhonov regularization added to (A^T A) diagonal before inverting.
        store_in_graph : bool, default True
            If True, store leverage values as 'leverage' attribute in graph edges.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ['Source', 'Destination', 'leverage'] for each edge.
        """
        if len(self._nodes_list) == 0 or len(self._edges_list) == 0:
            return pd.DataFrame(
                columns=["Source", "Destination", "leverage"]
            )  # empty graph

        num_edges = len(self._edges_list)
        num_nodes = len(self._nodes_list)

        # Map node -> column index in A
        # If nodes are already numeric indices, use them directly
        node_to_col: Dict[Any, int] = {}
        for node in self._nodes_list:
            # Check if node is already a numeric index
            if isinstance(node, (int, np.integer)):
                node_to_col[node] = int(node)
            elif node in self.node_to_idx:
                node_to_col[node] = int(self.node_to_idx[node])
            else:
                # Fallback consistent mapping
                node_to_col[node] = len(node_to_col)

        # Re-number columns to 0..N-1 if dataset mapping is sparse/non-contiguous
        # Build a compact mapping
        compact_nodes = sorted(self._nodes_list, key=lambda n: node_to_col[n])
        node_to_compact: Dict[Any, int] = {n: i for i, n in enumerate(compact_nodes)}

        A = np.zeros((num_edges, num_nodes), dtype=float)
        for row_idx, (u, v) in enumerate(self._edges_list):
            A[row_idx, node_to_compact[u]] = 1.0
            A[row_idx, node_to_compact[v]] = -1.0

        # Append reference constraint row to make (A^T A) invertible
        ref_node = reference_node if reference_node is not None else self._nodes_list[0]
        V = np.zeros((1, num_nodes), dtype=float)
        V[0, node_to_compact[ref_node]] = 1.0
        A_aug = np.vstack([A, V])

        # Hat matrix H = A * (A^T A + ridge*I)^-1 * A^T
        AtA = A_aug.T @ A_aug
        if ridge > 0.0:
            AtA = AtA + ridge * np.eye(AtA.shape[0])
        try:
            AtA_inv = np.linalg.inv(AtA)
        except np.linalg.LinAlgError:
            # As a fallback, use pseudo-inverse for stability
            AtA_inv = np.linalg.pinv(AtA)
        H = A @ (AtA_inv @ A.T)

        # Leverage values are diagonal entries of H (one per original edge row)
        leverages = np.clip(np.diag(H), 0.0, None)

        result = pd.DataFrame(
            {
                "Source": [u for (u, _v) in self._edges_list],
                "Destination": [v for (_u, v) in self._edges_list],
                "leverage": leverages,
            }
        )

        # Store leverages as edge attributes in the graph
        if store_in_graph:
            for idx, (u, v) in enumerate(self._edges_list):
                self.graph[u][v]["leverage"] = leverages[idx]

        return result

    def find_cycles(
        self,
        method: str = "basis",
        cycle_length_limit: Optional[int] = None,
    ) -> List[List[Any]]:
        """
        Find cycles in the underlying undirected graph.

        Parameters
        ----------
        method : str, default='basis'
            'basis' -> use cycle_basis (set of fundamental cycles)
            'all'   -> attempt to enumerate all simple cycles (potentially expensive)
        cycle_length_limit : int, optional
            If provided, limit the maximum length of returned cycles.

        Returns
        -------
        List[List[Any]]
            List of cycles as lists of node identifiers.
        """
        G_undirected = self.graph.to_undirected()

        def _limit(cycles: List[List[Any]]) -> List[List[Any]]:
            if cycle_length_limit is None:
                return cycles
            return [c for c in cycles if len(c) <= cycle_length_limit]

        if method == "basis":
            cycles = nx.cycle_basis(G_undirected)
            return _limit(cycles)

        # Enumerate all simple cycles in undirected graph (custom DFS-based)
        # Adapted from legacy approach
        cycles_set: set[Tuple[Any, ...]] = set()

        def canonical_cycle(cycle: List[Any]) -> Tuple[Any, ...]:
            # normalize cycle rotation/direction to canonical tuple
            m = min(cycle)
            mi = cycle.index(m)
            # rotate to start from minimal
            rot1 = cycle[mi:] + cycle[:mi]
            rot2 = list(reversed(cycle[: mi + 1])) + list(reversed(cycle[mi + 1 :]))
            t1 = tuple(rot1)
            t2 = tuple(rot2)
            return min(t1, t2)

        for comp in nx.connected_components(G_undirected):
            nodes = list(comp)
            for start in nodes:
                visited: set[Any] = set()
                parent: Dict[Any, Any] = {}

                def dfs(u: Any) -> None:
                    visited.add(u)
                    for v in G_undirected.neighbors(u):
                        if v == parent.get(u):
                            continue
                        if v in visited:
                            # found a cycle; backtrack from u to v
                            cyc = [v, u]
                            x = u
                            while parent.get(x) is not None and parent[x] != v:
                                x = parent[x]
                                cyc.append(x)
                            cycles_set.add(canonical_cycle(cyc))
                        else:
                            parent[v] = u
                            dfs(v)

                parent[start] = None  # type: ignore
                dfs(start)

        all_cycles = [list(c) for c in cycles_set]
        return _limit(all_cycles)

    def _oriented_cycle_sum(self, cycle: List[Any], attr: str) -> float:
        """
        Compute oriented sum around the cycle for the given edge attribute.

        For consecutive nodes (n0, n1, ..., nk-1, n0), each edge contributes:
        +attr if edge exists as (ni -> n(i+1)), -attr if exists as reverse.
        Missing directed edges are ignored (treated as 0 contribution).
        """
        if len(cycle) < 2:
            return 0.0
        total = 0.0
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if self.graph.has_edge(u, v) and attr in self.graph[u][v]:
                val = self.graph[u][v][attr]
                total += float(val)
            elif self.graph.has_edge(v, u) and attr in self.graph[v][u]:
                val = self.graph[v][u][attr]
                total -= float(val)
        return float(total)

    def compute_cycle_errors(
        self,
        attributes: Optional[List[str]] = None,
        method: str = "basis",
        cycle_length_limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Compute oriented cycle-closure errors for selected attributes.

        Parameters
        ----------
        attributes : list[str], optional
            Edge attributes to evaluate. Defaults to available standardized ones among
            ['DeltaDeltaG', 'CCC', 'MAP'] that are present in the graph.
        method : str, default='basis'
            Algorithm to find cycles (see find_cycles).
        cycle_length_limit : int, optional
            If provided, limit maximum cycle length considered.

        Returns
        -------
        pd.DataFrame
            One row per cycle with columns:
            - 'cycle_index'
            - 'nodes' (list of nodes in cycle order)
            - For each attribute: oriented sum and absolute sum columns
              e.g., 'DeltaDeltaG', 'abs(DeltaDeltaG)'
        """
        # Determine attributes to evaluate
        candidate_attrs = ["DeltaDeltaG", "CCC", "MAP"]
        if attributes is None:
            # infer from any edge present
            present = set()
            for _u, _v, data in self.graph.edges(data=True):
                present.update(k for k in data.keys() if k in candidate_attrs)
            attrs = [a for a in candidate_attrs if a in present]
        else:
            attrs = [a for a in attributes if a in candidate_attrs]

        cycles = self.find_cycles(method=method, cycle_length_limit=cycle_length_limit)
        records: List[Dict[str, Any]] = []
        for idx, cyc in enumerate(cycles):
            row: Dict[str, Any] = {
                "cycle_index": idx,
                "nodes": cyc,
                "length": len(cyc),
            }
            for a in attrs:
                s = self._oriented_cycle_sum(cyc, a)
                row[a] = s
                row[f"abs({a})"] = abs(s)
            records.append(row)

        return pd.DataFrame(records)

    def visualize_cycle_errors(
        self,
        attribute: str = "DeltaDeltaG",
        method: str = "basis",
        cycle_length_limit: Optional[int] = None,
        absolute: bool = False,
        bins: int = 30,
        figsize: Tuple[int, int] = (8, 5),
        save_path: Optional[str] = None,
        dpi: int = 300,
        results_dir: str = "results",
    ) -> None:
        """
        Plot a histogram with a KDE outline for cycle-closure errors of a selected attribute.

        Notes
        -----
        - The histogram and KDE are both plotted as densities (area under curve = 1).
        - Set `absolute=True` to visualize |error|.

        Parameters
        ----------
        attribute : str, default='DeltaDeltaG'
            Edge attribute to visualize (must be one of ['DeltaDeltaG', 'CCC', 'MAP']).
        method : str, default='basis'
            Cycle enumeration method (see find_cycles).
        cycle_length_limit : int, optional
            Max cycle length to include.
        absolute : bool, default=False
            If True, use absolute errors |sum| instead of oriented sums.
        bins : int, default=30
            Number of bins for histogram.
        figsize : (int, int), default=(8, 5)
            Figure size.
        save_path : str, optional
            If provided, save figure to path.
        dpi : int, default=300
            DPI when saving.
        results_dir : str, default='results'
            Directory to save temporary PDF for display.
        """
        df = self.compute_cycle_errors(
            attributes=[attribute], method=method, cycle_length_limit=cycle_length_limit
        )
        if df.empty or attribute not in df.columns:
            return

        values = df[f"abs({attribute})"].values if absolute else df[attribute].values
        values = values[np.isfinite(values)]
        if values.size == 0:
            return

        fig, ax = plt.subplots(figsize=figsize)
        title_attr = f"|{attribute}|" if absolute else attribute

        # Histogram as density
        ax.hist(
            values,
            bins=bins,
            density=True,
            color="steelblue",
            alpha=0.45,
            edgecolor="white",
        )

        # KDE outline
        try:
            kde = gaussian_kde(values)
            x_min, x_max = np.min(values), np.max(values)
            pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
            x_grid = np.linspace(x_min - pad, x_max + pad, 512)
            ax.plot(x_grid, kde(x_grid), color="black", lw=2, label="KDE")
        except Exception:
            # If KDE fails (e.g., singular data), skip
            pass

        ax.set_xlabel(f"Cycle-closure error ({title_attr})")
        ax.set_ylabel("Density")
        ax.set_title(f"Cycle errors: {title_attr}")
        if any([line.get_label() == "KDE" for line in ax.lines]):
            ax.legend()
        plt.tight_layout()

        # Handle saving and display
        if save_path is None:
            # Check if running in notebook
            if _is_notebook():
                # In Jupyter, just return figure - it will auto-display
                return fig
            else:
                # In scripts, show the plot
                plt.show()
                return fig
        else:
            # User explicitly requested saving
            # Ensure save_path has .pdf extension
            if not save_path.endswith('.pdf'):
                save_path = save_path.rsplit('.', 1)[0] + '.pdf'

            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            # Save figure as PDF
            try:
                plt.savefig(save_path, dpi=dpi, bbox_inches="tight", format='pdf')
                print(f"Figure successfully saved to: {save_path}")

                # If in notebook, return figure for display; otherwise close it
                if _is_notebook():
                    return fig
                else:
                    plt.close(fig)
                    return None
            except Exception as e:
                print(f"Error saving figure: {e}")
                # Try to close figure anyway
                try:
                    plt.close(fig)
                except:
                    pass
                return None

    def _label_point(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        labels: Union[List[str], pd.Series],
        ax: plt.Axes,
        offset: float = 0.02,
    ) -> None:
        """
        Helper method to label points on a scatter plot.

        Parameters
        ----------
        x_values : np.ndarray
            X coordinates of points to label
        y_values : np.ndarray
            Y coordinates of points to label
        labels : List[str] or pd.Series
            Labels for each point
        ax : plt.Axes
            Matplotlib axes object
        offset : float, default 0.02
            Offset for label positioning
        """
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            if i < len(labels):
                label = labels.iloc[i] if hasattr(labels, "iloc") else labels[i]
                ax.annotate(
                    str(label),
                    (x, y),
                    xytext=(offset, offset),
                    textcoords="offset fraction",
                    fontsize=8,
                    alpha=0.7,
                )

    def visualize_inference_effects(
        self,
        target_name: str,
        baseline_attribute: str = "DeltaDeltaG",
        comparison_attribute: str = "MAP",
        label_points: bool = False,
        figsize: Tuple[int, int] = (14, 12),
        save_path: Optional[str] = None,
        dpi: int = 600,
        results_dir: str = "results",
    ) -> plt.Figure:
        """
        Visualize how inference algorithms correct raw FEP values using scatter plots with quiver arrows.

        Creates a scatter plot comparing experimental values against two different estimates,
        with quiver arrows showing the shift from baseline to comparison values. Arrow colors
        represent edge leverages.

        Parameters
        ----------
        target_name : str
            Name/description for the target being analyzed (used in plot title)
        baseline_attribute : str, default 'DeltaDeltaG'
            The baseline edge attribute to compare (e.g., 'DeltaDeltaG', 'CCC')
        comparison_attribute : str, default 'MAP'
            The comparison edge attribute (e.g., 'MAP', 'CCC')
        label_points : bool, default False
            Whether to annotate points with edge labels
        figsize : Tuple[int, int], default (14, 12)
            Figure size
        save_path : str, optional
            If provided, save figure to this path
        dpi : int, default 600
            DPI when saving (higher for publication quality)
        results_dir : str, default 'results'
            Directory to save temporary PDF for display

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the visualization
        """
        # Get edge and node dataframes - these have ALL the data we need
        edges_df, nodes_df = self.graph_setup.dataset.get_dataframes()

        # Extract experimental values (standardized column names)
        exp_col_candidates = [
            "Experimental DeltaDeltaG",
            "Experimental DeltaG",
            "Exp.",
            "Experimental",
        ]
        exp_values = None

        for col in exp_col_candidates:
            if col in edges_df.columns:
                exp_values = edges_df[col].values
                break

        if exp_values is None:
            raise ValueError(
                "No experimental values found in edge data. Expected columns: "
                + ", ".join(exp_col_candidates)
            )

        # Extract baseline and comparison values directly from edges_df
        # This ensures we're using the same data source and order
        if baseline_attribute not in edges_df.columns:
            raise ValueError(
                f"Baseline attribute '{baseline_attribute}' not found in edges dataframe. "
                f"Available columns: {edges_df.columns.tolist()}"
            )
        
        if comparison_attribute not in edges_df.columns:
            raise ValueError(
                f"Comparison attribute '{comparison_attribute}' not found in edges dataframe. "
                f"Available columns: {edges_df.columns.tolist()}"
            )
        
        baseline_values = edges_df[baseline_attribute].values
        comparison_values = edges_df[comparison_attribute].values
        
        # For leverages, compute them if needed
        # First check if leverages are already computed and stored in the graph
        has_leverages = all("leverage" in data for u, v, data in self.graph.edges(data=True)) if self.graph.number_of_edges() > 0 else False
        
        if not has_leverages:
            # Compute leverages
            leverage_df = self.compute_leverages(store_in_graph=True)
            
            # Create a mapping from (source, dest) to leverage
            leverage_map = {}
            for _, row in leverage_df.iterrows():
                leverage_map[(row['Source'], row['Destination'])] = row['leverage']
            
            # Add leverages to edges_df in the same order
            leverages = []
            for _, row in edges_df.iterrows():
                key = (row['Source'], row['Destination'])
                if key in leverage_map:
                    leverages.append(leverage_map[key])
                else:
                    # Default leverage if not found
                    leverages.append(0.5)
            leverages = np.array(leverages)
        else:
            # Extract leverages from graph in the same order as edges_df
            leverages = []
            for _, row in edges_df.iterrows():
                src, dst = row['Source'], row['Destination']
                if self.graph.has_edge(src, dst):
                    leverages.append(self.graph[src][dst].get('leverage', 0.5))
                else:
                    leverages.append(0.5)
            leverages = np.array(leverages)

        # Ensure same length arrays
        min_length = min(
            len(exp_values),
            len(baseline_values),
            len(comparison_values),
            len(leverages),
        )
        exp_values = exp_values[:min_length]
        baseline_values = baseline_values[:min_length]
        comparison_values = comparison_values[:min_length]
        leverages = leverages[:min_length]

        # Create figure with manual positioning for better layout control
        # Ensure figsize is properly unpacked as a tuple
        fig = plt.figure(figsize=tuple(figsize) if figsize else (14, 12))

        # Main plot with room for legend below and colorbar on right
        ax = fig.add_axes(
            [0.1, 0.25, 0.75, 0.65]
        )  # [left, bottom, width, height] - room for legend below

        # Compute separate axis limits for x and y based on their respective data ranges
        x_padding = 0.1 * (np.max(exp_values) - np.min(exp_values)) if np.max(exp_values) != np.min(exp_values) else 0.5
        y_padding = 0.1 * (max(np.max(baseline_values), np.max(comparison_values)) - min(np.min(baseline_values), np.min(comparison_values)))
        if y_padding == 0:
            y_padding = 0.5

        xlim = [np.min(exp_values) - x_padding, np.max(exp_values) + x_padding]
        ylim = [
            min(np.min(baseline_values), np.min(comparison_values)) - y_padding,
            max(np.max(baseline_values), np.max(comparison_values)) + y_padding,
        ]

        # For shading regions, use the full range needed to cover both axes
        shade_min = min(xlim[0], ylim[0])
        shade_max = max(xlim[1], ylim[1])
        x_shade = np.linspace(shade_min, shade_max, 100)

        # Excellent region (within ±0.5 kcal/mol of X=Y line)
        ax.fill_between(
            x_shade,
            x_shade - 0.5,
            x_shade + 0.5,
            alpha=0.2,
            color="green",
            label="Excellent (±0.5 kcal/mol)",
        )

        # Good region (within ±1.0 kcal/mol of X=Y line) - darker yellow for better visibility
        ax.fill_between(
            x_shade, x_shade - 1.0, x_shade - 0.5, alpha=0.25, color="orange"
        )  # Changed from yellow to orange for visibility
        ax.fill_between(
            x_shade,
            x_shade + 0.5,
            x_shade + 1.0,
            alpha=0.25,
            color="orange",
            label="Good (±1.0 kcal/mol)",
        )

        # Calculate distance from X=Y line for color coding
        dist_baseline = np.abs(exp_values - baseline_values)
        dist_comparison = np.abs(exp_values - comparison_values)

        # Define threshold for red vs blue coloring
        threshold = 1.0  # kcal/mol

        # Color code points: red if >threshold, blue otherwise
        baseline_colors = ["red" if d > threshold else "blue" for d in dist_baseline]
        comparison_colors = [
            "red" if d > threshold else "blue" for d in dist_comparison
        ]

        # Scatter plots with distance-based coloring
        ax.scatter(
            exp_values,
            baseline_values,
            c=baseline_colors,
            label=f"{baseline_attribute}",
            alpha=0.7,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.scatter(
            exp_values,
            comparison_values,
            c=comparison_colors,
            label=f"{comparison_attribute}",
            alpha=0.7,
            s=50,
            marker="s",
            edgecolors="black",
            linewidth=0.5,
        )

        # Trend lines
        if len(exp_values) > 1:  # Need at least 2 points for polyfit
            # Baseline trend line
            m_base, b_base = np.polyfit(exp_values, baseline_values, 1)
            ax.plot(
                exp_values,
                m_base * exp_values + b_base,
                color="red",
                label=f"Best Fit Line for {baseline_attribute}",
                linewidth=2,
            )

            # Comparison trend line
            m_comp, b_comp = np.polyfit(exp_values, comparison_values, 1)
            ax.plot(
                exp_values,
                m_comp * exp_values + b_comp,
                color="blue",
                label=f"Best Fit Line for {comparison_attribute}",
                linewidth=2,
            )

        # Perfect correlation line (X=Y) - extend across the full range
        line_range = [shade_min, shade_max]
        ax.plot(
            line_range, line_range, "k-", linewidth=3, label="Perfect Correlation (X=Y)", alpha=0.9
        )

        # Set axis limits separately for x and y
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Quiver plot (arrows from baseline to comparison, colored by leverage)
        # Use only the darker portion of the Greys colormap (avoid white)
        from matplotlib.colors import LinearSegmentedColormap

        # Create a custom colormap from very light gray to black (no white)
        colors = ["#C0C0C0", "#000000"]  # Very light gray to black
        n_bins = 256
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_greys", colors, N=n_bins
        )

        norm = Normalize(vmin=np.min(leverages), vmax=np.max(leverages))
        arrow_colors = custom_cmap(norm(leverages))

        ax.quiver(
            exp_values,
            baseline_values,
            np.zeros_like(exp_values),  # dx = 0 (no horizontal shift)
            comparison_values - baseline_values,  # dy = difference in predictions
            angles="xy",
            scale_units="xy",
            scale=1,
            color=arrow_colors,
            alpha=0.8,
            width=0.004,
        )  # More visible arrows

        # Add colorbar for leverages on the right side
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes(
            [0.87, 0.25, 0.02, 0.65]
        )  # [left, bottom, width, height] - match plot position
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Edge Leverage (Network Influence)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        # Label points if requested
        if label_points:
            # Get edge dataframe for labeling
            edges_df, _ = self.graph_setup.dataset.get_dataframes()

            # Create edge labels
            edge_labels = []
            source_col = "Source" if "Source" in edges_df.columns else "Ligand1"
            dest_col = "Destination" if "Destination" in edges_df.columns else "Ligand2"

            for i in range(len(exp_values)):
                if i < len(edges_df):
                    source = edges_df.iloc[i][source_col]
                    dest = edges_df.iloc[i][dest_col]
                    edge_labels.append(f"{source}-{dest}")
                else:
                    edge_labels.append(f"Edge_{i}")

            self._label_point(exp_values, baseline_values, edge_labels, ax)
            self._label_point(exp_values, comparison_values, edge_labels, ax)

        # Formatting with publication-ready font sizes
        ax.set_xlabel("Experimental ΔΔG (kcal/mol)", fontsize=18)
        ax.set_ylabel(
            f"{baseline_attribute} and {comparison_attribute} ΔΔG (kcal/mol)",
            fontsize=18,
        )
        ax.set_title(
            f"Inference Effects Analysis for {target_name}\n"
            f"Arrows show shift from {baseline_attribute} to {comparison_attribute}",
            fontsize=20,
        )
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, alpha=0.3)

        # Add color coding legend with colored squares
        from matplotlib.patches import Patch

        color_legend_elements = [
            Patch(
                facecolor="blue",
                label=f"Within {threshold} kcal/mol error",
            ),
            Patch(
                facecolor="red",
                label=f">{threshold} kcal/mol error",
            ),
        ]

        # Get legend handles from the main plot
        main_handles, main_labels = ax.get_legend_handles_labels()

        # Combine main legend items with color legend items
        all_handles = main_handles + color_legend_elements
        all_labels = main_labels + [h.get_label() for h in color_legend_elements]

        # Create legend below the plot area
        ax.legend(
            all_handles,
            all_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            fontsize=14,
            framealpha=0.95,
            ncol=3,  # Arrange in 3 columns for compact layout
            borderaxespad=0,
        )

        # Handle saving and display
        if save_path is not None:
            # User explicitly requested saving
            # Ensure save_path has .pdf extension
            if not save_path.endswith('.pdf'):
                save_path = save_path.rsplit('.', 1)[0] + '.pdf'

            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            # Save figure as PDF
            try:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format='pdf')
                print(f"Figure successfully saved to: {save_path}")
            except Exception as e:
                print(f"Error saving figure: {e}")

        # Show plot if not in notebook environment
        if not _is_notebook() and save_path is None:
            plt.show()

        return fig
