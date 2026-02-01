from typing import Any, Dict, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset
from .FEP_benchmark_dataset import FEPBenchmarkDataset


class FEPDataset(BaseDataset):
    """
    A class to represent a dataset for graph-based analysis. This class handles
    the initialization of the dataset, including nodes and edges, and provides
    functionality to convert edge data into graph data format.

    This class can be initialized in multiple ways:
    1. From DataFrames (nodes and edges)
    2. From CSV files
    3. From benchmark datasets
    4. From edge data only (using derive_nodes_from_edges class method)

    Attributes
    ----------
    cycle_data : dict
        Graph data dictionary with keys:
            'N'   : int,   # Number of nodes
            'M'   : int,   # Number of edges
            'src' : list,  # Source node indices (0-based)
            'dst' : list,  # Destination node indices (0-based)
            'FEP' : list,  # Free Energy Perturbation values for each edge
            'CCC' : list,  # Cycle closure corrected edge values
    node2idx : dict
        Node-to-index mapping.
    idx2node : dict
        Index-to-node mapping.

    Example
    -------
        # Method 1: Initialize from benchmark dataset
        ds = FEPDataset(dataset_name="cdk8", sampling_time="5ns")

        # Method 2: Initialize with only edges (automatic node derivation)
        ds = FEPDataset(dataset_edges=edges_df)

        # Method 3: Derive nodes from edges first, then initialize
        edges, nodes = FEPDataset.derive_nodes_from_edges(edges_df)
        ds = FEPDataset(dataset_edges=edges, dataset_nodes=nodes)

        # Access graph data
        graph_data = ds.cycle_data
        node_mapping = ds.node2idx, ds.idx2node
    """

    def __init__(
        self,
        dataset_nodes: Optional[pd.DataFrame] = None,
        dataset_edges: Optional[pd.DataFrame] = None,
        dataset_name: Optional[str] = None,
        sampling_time: Optional[str] = None,
        nodes_csv_path: Optional[str] = None,
        edges_csv_path: Optional[str] = None,
    ):
        """
        Initializes the FEPDataset object. Supports multiple ways to load data:
        1. Directly from DataFrames (dataset_nodes, dataset_edges)
        2. From CSV files (nodes_csv_path, edges_csv_path)
        3. From benchmark datasets (dataset_name, sampling_time)
        4. From edges only (dataset_edges) - nodes will be derived automatically

        Parameters
        ----------
        dataset_nodes : pd.DataFrame, optional
            DataFrame containing node data.
        dataset_edges : pd.DataFrame, optional
            DataFrame containing edge data.
        dataset_name : str, optional
            Name of the benchmark dataset to fetch.
        sampling_time : str, optional
            Sampling time to fetch for benchmark dataset.
        nodes_csv_path : str, optional
            Path to CSV file containing node data.
        edges_csv_path : str, optional
            Path to CSV file containing edge data.

        Raises
        ------
        ValueError
            If no valid data source is provided.
        FileNotFoundError
            If CSV files are specified but don't exist.
        """
        # Handle different data loading methods
        if dataset_nodes is not None and dataset_edges is not None:
            # Method 1: Direct DataFrame input
            self.dataset_nodes = dataset_nodes.copy()
            self.dataset_edges = dataset_edges.copy()
            print("📊 Loaded data from provided DataFrames")

        elif nodes_csv_path is not None and edges_csv_path is not None:
            # Method 2: CSV file input
            import os

            # Check if files exist
            if not os.path.exists(nodes_csv_path):
                raise FileNotFoundError(f"Nodes CSV file not found: {nodes_csv_path}")
            if not os.path.exists(edges_csv_path):
                raise FileNotFoundError(f"Edges CSV file not found: {edges_csv_path}")

            # Load CSV files
            print(f"📊 Loading nodes from CSV: {nodes_csv_path}")
            self.dataset_nodes = pd.read_csv(nodes_csv_path)
            print(f"📊 Loading edges from CSV: {edges_csv_path}")
            self.dataset_edges = pd.read_csv(edges_csv_path)

            print(
                f"✅ Successfully loaded {len(self.dataset_nodes)} nodes and "
                f"{len(self.dataset_edges)} edges from CSV files"
            )

        elif dataset_name and sampling_time:
            # Method 3: Benchmark dataset
            print(f"📊 Loading benchmark dataset: {dataset_name} ({sampling_time})")
            self.dataset_edges, self.dataset_nodes = FEPBenchmarkDataset().get_dataset(
                dataset_name, sampling_time
            )

        elif dataset_edges is not None and dataset_nodes is None:
            # Method 4: Only edges provided - derive nodes automatically using class method
            print("📊 Only edge data provided - deriving node values from edges")

            # Get all nodes from the edges to randomly select a reference
            if "Source" in dataset_edges.columns and "Destination" in dataset_edges.columns:
                all_nodes = set(dataset_edges["Source"]) | set(dataset_edges["Destination"])
            elif "Ligand1" in dataset_edges.columns and "Ligand2" in dataset_edges.columns:
                all_nodes = set(dataset_edges["Ligand1"]) | set(dataset_edges["Ligand2"])
            else:
                raise ValueError(
                    "Edge DataFrame must have either 'Source'/'Destination' or "
                    "'Ligand1'/'Ligand2' columns"
                )

            # Randomly select a reference node
            import random
            reference_node = random.choice(list(all_nodes))

            # Use the class method to derive nodes - returns (edges, nodes) tuple
            self.dataset_edges, self.dataset_nodes = type(self).derive_nodes_from_edges(
                dataset_edges, reference_node=reference_node
            )

        else:
            raise ValueError(
                "Must provide one of the following:\n"
                "1. dataset_nodes and dataset_edges (DataFrames)\n"
                "2. nodes_csv_path and edges_csv_path (CSV file paths)\n"
                "3. dataset_name and sampling_time (benchmark dataset)\n"
                "4. dataset_edges only (nodes will be derived automatically)"
            )

        # Ensure node data has required columns regardless of input method
        if "Pred. DeltaG" not in self.dataset_nodes.columns:
            # If no predicted values, use experimental as initial predictions
            self.dataset_nodes["Pred. DeltaG"] = self.dataset_nodes["Exp. DeltaG"]
            print(
                "⚠️  Warning: No predicted DeltaG values provided in node data. "
                "Using experimental values as predicted values. "
                "Predicted and experimental node values are identical."
            )

        # Ensure experimental values are present
        if "Exp. DeltaG" not in self.dataset_nodes.columns:
            # If no experimental values, use predicted values as experimental
            self.dataset_nodes["Exp. DeltaG"] = self.dataset_nodes["Pred. DeltaG"]
            print(
                "⚠️  Warning: No experimental DeltaG values provided in node data. "
                "Using predicted values as experimental values. "
                "Predicted and experimental node values are identical."
            )

        # Store graph data as persistent attributes
        self.cycle_data, self.node2idx, self.idx2node = self._build_graph_data(
            self.dataset_edges
        )
        self.estimators = (
            []
        )  # Whenever data from an estimator is added, it updates this list
        self.prior_type = None  # Track which prior was used for inference

    @classmethod
    def derive_nodes_from_edges(
        cls,
        dataset_edges: pd.DataFrame,
        value_column: Optional[str] = None,
        reference_node: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Derive node values from edge data using graph traversal.

        This class method derives node values from only edge data through graph
        propagation. Starting from a reference node (set to 0.0), it calculates
        all connected node values using the edge differences.

        The algorithm works as follows:
        1. Set one reference node to 0.0 (arbitrary choice)
        2. For each edge (A→B) with value ΔΔG:
           - If node A is known: B = A + ΔΔG
           - If node B is known: A = B - ΔΔG
        3. Continue until all reachable nodes are calculated
        4. For disconnected components, set new reference points to 0.0

        Parameters
        ----------
        dataset_edges : pd.DataFrame
            DataFrame containing edge data with columns for source, destination,
            and edge values. Should have columns matching one of these patterns:
            - 'Source', 'Destination', and value column
            - 'Ligand1', 'Ligand2', and value column
        value_column : str, optional
            Name of the column containing edge values. If None, will attempt to
            find it automatically from common names: 'DeltaDeltaG', 'DeltaG', 'FEP'
        reference_node : str, optional
            Name of the node to use as reference (set to 0.0). If None, uses the
            first node encountered in the edge data.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (dataset_edges, dataset_nodes) where:
            - dataset_edges: The edge DataFrame (potentially with added standard column)
            - dataset_nodes: Derived node DataFrame with columns 'molecule' and 'Exp. DeltaG'

        Raises
        ------
        ValueError
            If the edge DataFrame doesn't have required columns or if value_column
            is not found.

        Examples
        --------
        >>> edges_df = pd.DataFrame({
        ...     'Source': ['mol_A', 'mol_B', 'mol_C'],
        ...     'Destination': ['mol_B', 'mol_C', 'mol_D'],
        ...     'DeltaDeltaG': [1.2, -0.8, 2.1]
        ... })
        >>> edges, nodes = FEPDataset.derive_nodes_from_edges(edges_df)
        >>> # Now you have both edge and derived node DataFrames

        Notes
        -----
        - Node values are relative to the reference node (set to 0.0)
        - For disconnected graphs, each component will have its own reference
        - The derived node values represent ΔG relative to the reference compound
        """
        # Identify column names
        if "Source" in dataset_edges.columns and "Destination" in dataset_edges.columns:
            source_col, dest_col = "Source", "Destination"
        elif (
            "Ligand1" in dataset_edges.columns and "Ligand2" in dataset_edges.columns
        ):
            source_col, dest_col = "Ligand1", "Ligand2"
        else:
            raise ValueError(
                "Edge DataFrame must have either 'Source'/'Destination' or "
                "'Ligand1'/'Ligand2' columns"
            )

        # Identify value column
        if value_column is None:
            # Try common column names
            for col_name in ["DeltaDeltaG", "FEP"]:
                if col_name in dataset_edges.columns:
                    value_column = col_name
                    break
            if value_column is None:
                raise ValueError(
                    "Could not automatically identify value column. Please specify "
                    "value_column parameter. Available columns: "
                    f"{list(dataset_edges.columns)}"
                )
        elif value_column not in dataset_edges.columns:
            raise ValueError(
                f"Specified value column '{value_column}' not found in DataFrame. "
                f"Available columns: {list(dataset_edges.columns)}"
            )

        # Build adjacency list for graph traversal
        # Store both forward and reverse edges with their values
        adjacency = {}  # node -> [(neighbor, edge_value), ...]

        for _, row in dataset_edges.iterrows():
            source = row[source_col]
            dest = row[dest_col]
            value = float(row[value_column])

            if source not in adjacency:
                adjacency[source] = []
            if dest not in adjacency:
                adjacency[dest] = []

            # Add both directions: A->B with +value, B->A with -value
            adjacency[source].append((dest, value))
            adjacency[dest].append((source, -value))

        # Calculate node values using BFS
        all_nodes = set(adjacency.keys())
        node_values = {}  # node_name -> calculated_value
        visited = set()

        # Determine reference node
        if reference_node is None:
            reference_node = next(iter(all_nodes))
        elif reference_node not in all_nodes:
            raise ValueError(
                f"Reference node '{reference_node}' not found in edge data. "
                f"Available nodes: {sorted(all_nodes)}"
            )

        # Process connected components
        while len(visited) < len(all_nodes):
            # Find an unvisited node (start with reference_node if unvisited)
            if reference_node not in visited:
                start_node = reference_node
            else:
                # Pick any unvisited node for a new component
                start_node = next(iter(all_nodes - visited))

            # Set reference value for this component
            node_values[start_node] = 0.0
            visited.add(start_node)

            # BFS to propagate values through connected component
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()
                current_value = node_values[current_node]

                # Visit all neighbors
                for neighbor, edge_value in adjacency[current_node]:
                    if neighbor not in visited:
                        # Calculate neighbor value: neighbor = current + edge_value
                        node_values[neighbor] = current_value + edge_value
                        visited.add(neighbor)
                        queue.append(neighbor)

        # Create dataset_nodes DataFrame
        dataset_nodes = pd.DataFrame(
            {
                "Name": list(node_values.keys()),
                "Exp. DeltaG": list(node_values.values()),
            }
        )

        # Report if multiple components were found
        num_components = sum(1 for node in all_nodes if node_values[node] == 0.0)
        if num_components > 1:
            print(
                f"⚠️  Warning: Graph has {num_components} disconnected components. "
                f"Each component has its own reference node set to 0.0"
            )

        print(
            f"✅ Successfully derived {len(node_values)} node values from "
            f"{len(dataset_edges)} edges"
        )

        # Ensure edge DataFrame has standard column names for FEPDataset initialization
        # If value_column is not one of the standard names, add a standard alias
        edges_for_return = dataset_edges.copy()
        standard_value_names = ["DeltaDeltaG", "DeltaG", "FEP"]
        if value_column not in standard_value_names:
            # Add the standard column name as an alias
            edges_for_return["DeltaDeltaG"] = edges_for_return[value_column]

        # Return the edge and node DataFrames
        return edges_for_return, dataset_nodes

    def check_ccc_values(self) -> bool:
        """
        Check if CCC values are present and meaningful in the dataset.

        Returns
        -------
        bool
            True if CCC values are present and not all zero, False otherwise
        """
        edges_df = getattr(self, "dataset_edges", None)

        if edges_df is None:
            return False

        # Check if CCC column exists
        if "CCC" not in edges_df.columns:
            return False

        # Check if CCC values are all zero
        ccc_values = edges_df["CCC"].values
        if len(ccc_values) == 0:
            return False

        # Check if all values are close to zero
        return not np.allclose(ccc_values, 0.0)

    def get_ccc_status(self) -> Dict[str, Any]:
        """
        Get detailed status of CCC values in the dataset.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing CCC status information:
            - 'present': bool - Whether CCC column exists
            - 'meaningful': bool - Whether CCC values are not all zero
            - 'count': int - Number of CCC values
            - 'mean': float - Mean of CCC values
            - 'std': float - Standard deviation of CCC values
        """
        edges_df = getattr(self, "dataset_edges", None)

        if edges_df is None:
            return {
                "present": False,
                "meaningful": False,
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
            }

        # Check if CCC column exists
        if "CCC" not in edges_df.columns:
            return {
                "present": False,
                "meaningful": False,
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
            }

        ccc_values = edges_df["CCC"].values

        if len(ccc_values) == 0:
            return {
                "present": True,
                "meaningful": False,
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
            }

        # Calculate statistics
        mean_val = np.mean(ccc_values)
        std_val = np.std(ccc_values)
        meaningful = not np.allclose(ccc_values, 0.0)

        return {
            "present": True,
            "meaningful": meaningful,
            "count": len(ccc_values),
            "mean": mean_val,
            "std": std_val,
        }

    def _build_graph_data(
        self, dataset_edges: Optional[pd.DataFrame] = None
    ) -> Tuple[Dict[str, Any], Dict[str, int], Dict[int, str]]:
        """
        Converts edge data into graph data format with node indices and edge attributes.
        Stores the result as attributes of the class.

        Parameters
        ----------
        dataset_edges : pd.DataFrame, optional
            DataFrame containing edge data. If None, uses self.dataset_edges.

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, int], Dict[int, str]]
            (cycle_data, node2idx, idx2node) where:
            - cycle_data: Graph data dictionary with keys: N, M, src, dst, FEP, CCC
            - node2idx: Node-to-index mapping.
            - idx2node: Index-to-node mapping.

        Raises
        ------
        ValueError
            If the input DataFrame does not have the required columns.
        """
        if dataset_edges is None:
            dataset_edges = self.dataset_edges

        # Handle both old and new column names
        column_mapping = {}

        # Check for new column names first
        if "Source" in dataset_edges.columns and "Destination" in dataset_edges.columns:
            column_mapping = {
                "source_col": "Source",
                "dest_col": "Destination",
                "fep_col": (
                    "DeltaDeltaG"
                    if "DeltaDeltaG" in dataset_edges.columns
                    else ("DeltaG" if "DeltaG" in dataset_edges.columns else "FEP")
                ),
                "ccc_col": "CCC" if "CCC" in dataset_edges.columns else None,
            }
        elif "Ligand1" in dataset_edges.columns and "Ligand2" in dataset_edges.columns:
            column_mapping = {
                "source_col": "Ligand1",
                "dest_col": "Ligand2",
                "fep_col": (
                    "FEP"
                    if "FEP" in dataset_edges.columns
                    else (
                        "DeltaDeltaG"
                        if "DeltaDeltaG" in dataset_edges.columns
                        else "DeltaG"
                    )
                ),
                "ccc_col": "CCC" if "CCC" in dataset_edges.columns else None,
            }
        else:
            raise ValueError(
                "Edge DataFrame must have either 'Source'/'Destination' or "
                "'Ligand1'/'Ligand2' columns"
            )

        # Check for required columns
        required_columns = {
            column_mapping["source_col"],
            column_mapping["dest_col"],
            column_mapping["fep_col"],
        }
        missing_columns = required_columns - set(dataset_edges.columns)
        if missing_columns:
            raise ValueError(
                f"Input edge DataFrame is missing required columns: {missing_columns}"
            )

        start: list[int] = []  # Source node indices
        end: list[int] = []  # Destination node indices
        fep: list[float] = []  # Free Energy Perturbation values
        ccc_dt: list[float] = []  # Cycle closure corrected edge values
        node2idx: Dict[str, int] = {}  # Maps nodes to unique indices
        idx2node: Dict[int, str] = {}  # Maps indices back to nodes
        idx = 0  # Counter for assigning unique indices to nodes

        # Iterate through each row in the edge data
        for _, row in dataset_edges.iterrows():
            source = row[column_mapping["source_col"]]
            dest = row[column_mapping["dest_col"]]

            for ligand in [source, dest]:
                if ligand not in node2idx:
                    node2idx[ligand] = idx
                    idx2node[idx] = ligand
                    idx += 1
            start.append(node2idx[source])  # Source node number (0-based)
            end.append(node2idx[dest])  # Destination node number (0-based)
            fep.append(float(row[column_mapping["fep_col"]]))  # FEP value for this edge

            # Handle CCC values (may be missing)
            if (
                column_mapping["ccc_col"]
                and column_mapping["ccc_col"] in dataset_edges.columns
            ):
                ccc_dt.append(float(row[column_mapping["ccc_col"]]))
            else:
                ccc_dt.append(0.0)  # Default CCC value

        # Detailed comments for each key in the dictionary
        cycle_dat = {
            "N": len(node2idx),  # Number of nodes
            "M": len(fep),  # Number of edges
            "src": start,  # Source node number (0-based)
            "dst": end,  # Destination node number (0-based)
            "FEP": fep,  # Free Energy Perturbation values for each edge
            "CCC": ccc_dt,  # Cycle closure corrected edge values
        }
        return cycle_dat, node2idx, idx2node

    def get_graph_data(self) -> Dict[str, Any]:
        """
        Get the graph data in the format required by NodeModel.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing graph data with keys: N, M, src, dst, FEP, CCC
        """
        return self.cycle_data

    def get_node_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Get the node-to-index and index-to-node mappings.

        Returns
        -------
        Tuple[Dict[str, int], Dict[int, str]]
            (node2idx, idx2node) mappings
        """
        return self.node2idx, self.idx2node

    def get_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the edge and node DataFrames.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (dataset_edges, dataset_nodes) DataFrames
        """
        return self.dataset_edges, self.dataset_nodes

    def add_estimator_results(self, estimator_metadata: Dict[str, Any]) -> None:
        """
        Add results from an estimator to the dataset.

        Parameters
        ----------
        estimator_metadata : Dict[str, Any]
            Metadata about the estimator and its results
        """
        self.estimators.append(estimator_metadata)

    def get_estimators(self) -> list:
        """
        Get the list of estimators that have been applied to this dataset.

        Returns
        -------
        list
            List of estimator metadata dictionaries
        """
        return self.estimators
