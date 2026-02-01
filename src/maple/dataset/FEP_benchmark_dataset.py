import os
from typing import Any, Dict, Tuple

import pandas as pd

from .base_dataset import BaseDataset


class FEPBenchmarkDataset(BaseDataset):
    """
    Class for accessing FEP benchmark datasets from the Schindler et al. repository.

    Supported datasets:
        - cdk8
        - cmet
        - eg5
        - hif2a
        - pfkfb3
        - shp2
        - syk
        - tnks2

    Usage:
        dataset = FEPBenchmarkDataset()
        edge_df, node_df = dataset.get_dataset("cdk8", "5ns")
    """

    SUPPORTED_DATASETS = [
        "cdk8",
        "cmet",
        "eg5",
        "hif2a",
        "pfkfb3",
        "shp2",
        "syk",
        "tnks2",
    ]

    def __init__(self, cache_dir="~/.maple_cache"):
        """
        Initialize the dataset handler.

        Parameters
        ----------
        cache_dir : str
            Directory to cache downloaded CSV files.
        """
        self.cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._edge_cache = {}
        self._node_cache = {}
        self._mapping_cache = {}  # Store node mappings per dataset

    def get_dataset(self, dataset, sampling_time):
        """
        Lazily load and return edge and node data for a given dataset and sampling time.

        Node identifiers are automatically converted to numeric indices for NodeModel compatibility.
        Original node names are preserved and can be retrieved using get_node_mapping().

        Parameters
        ----------
        dataset : str
            Name of the dataset (must be in SUPPORTED_DATASETS).
        sampling_time : str
            Sampling time, e.g., "5ns" or "20ns".

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            Tuple of (edge_data, node_data) DataFrames with numeric node indices.
        """
        if dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset '{dataset}' is not supported. Supported datasets: {self.SUPPORTED_DATASETS}"
            )

        edge_key = (dataset, sampling_time)
        node_key = (dataset, sampling_time)

        if edge_key not in self._edge_cache:
            edge_data = self._load_edge_data(dataset, sampling_time)
            node_data = self._load_node_data(dataset, sampling_time)

            # Create node mapping and apply it to both edge and node data
            edge_data, node_data = self._apply_node_mapping(edge_data, node_data)

            self._edge_cache[edge_key] = edge_data
            self._node_cache[node_key] = node_data
            # Cache the mappings for this dataset
            self._mapping_cache[edge_key] = (self.node2idx.copy(), self.idx2node.copy())
        else:
            # Restore mappings from cache for this dataset
            self.node2idx, self.idx2node = self._mapping_cache[edge_key]

        return self._edge_cache[edge_key], self._node_cache[node_key]

    def _load_edge_data(self, dataset, sampling_time):
        # Map sampling_time to file suffix
        if sampling_time not in ["5ns", "20ns"]:
            raise ValueError("sampling_time must be '5ns' or '20ns'")
        url = f"https://raw.githubusercontent.com/MCompChem/fep-benchmark/master/{dataset}/results_edges_{sampling_time}.csv"
        local_path = os.path.join(
            self.cache_dir, f"{dataset}_edges_{sampling_time}.csv"
        )
        df = self._download_and_cache(url, local_path)
        # Drop columns and add metadata as before
        df = df.drop(
            columns=[
                col for col in ["Solvation", "Solvation Error"] if col in df.columns
            ]
        )

        # Rename all Delta symbols (Δ) to 'Delta' for consistency
        delta_columns = [col for col in df.columns if "Δ" in col]
        for col in delta_columns:
            new_col = col.replace("Δ", "Delta")
            df = df.rename(columns={col: new_col})

        # Standardize column names to match MAPLE expectations
        column_mapping = {
            "Ligand1": "Source",
            "Ligand2": "Destination",
            "FEP": "DeltaDeltaG",
            "FEP Error": "DeltaDeltaG Error",
            "Exp.": "Experimental DeltaDeltaG",  # Keep experimental values
        }

        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Calculate MLE for missing CCC values if needed
        if "CCC" not in df.columns or df["CCC"].isna().any():
            if "Experimental DeltaDeltaG" in df.columns:
                df["CCC"] = df["Experimental DeltaDeltaG"]
                df["CCC Error"] = (
                    df["DeltaDeltaG Error"]
                    if "DeltaDeltaG Error" in df.columns
                    else 0.1
                )
            else:
                df["CCC"] = df["DeltaDeltaG"]
                df["CCC Error"] = (
                    df["DeltaDeltaG Error"]
                    if "DeltaDeltaG Error" in df.columns
                    else 0.1
                )

        df["Dataset"] = dataset
        df["Sampling Time"] = sampling_time
        return df

    def _load_node_data(self, dataset, sampling_time):
        url = f"https://raw.githubusercontent.com/MCompChem/fep-benchmark/master/{dataset}/results_{sampling_time}.csv"
        local_path = os.path.join(
            self.cache_dir, f"{dataset}_nodes_{sampling_time}.csv"
        )
        df = self._download_and_cache(url, local_path)
        # Drop columns and add metadata as before
        drop_cols = ["Affinity unit", " # ", "Quality", "Structure"]
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        # Rename all Delta symbols (Δ) to 'Delta' for consistency
        delta_columns = [col for col in df.columns if "Δ" in col]
        for col in delta_columns:
            new_col = col.replace("Δ", "Delta")
            df = df.rename(columns={col: new_col})

        # Standardize column names to match MAPLE expectations
        column_mapping = {
            "Ligand": "Name",
            "Exp. DeltaG": "Exp. DeltaG",  # Already correct
            "Exp. Error": "Exp. Error",  # Already correct
            "Pred. DeltaG": "Pred. DeltaG",  # Keep for reference
            "Pred. Error": "Pred. Error",  # Keep for reference
        }

        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        df["Dataset"] = dataset
        df["Sampling Time"] = sampling_time
        if "Exp. Error" not in df.columns:
            df["Exp. Error"] = 0.0
        return df

    def _download_and_cache(self, url, local_path):
        if not os.path.exists(local_path):
            try:
                df = pd.read_csv(url)
                df.to_csv(local_path, index=False)
            except Exception as e:
                raise RuntimeError(f"Failed to download or parse {url}: {e}")
        else:
            df = pd.read_csv(local_path)
        return df

    def get_graph_data(self) -> Dict[str, Any]:
        """
        Get the graph data in the format required by NodeModel.

        This method loads a default dataset (cdk8, 5ns) and converts it to graph format.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing graph data with keys: N, M, src, dst, FEP, CCC
        """
        # Load default dataset if not already loaded
        if not hasattr(self, "dataset_edges") or not hasattr(self, "dataset_nodes"):
            self.dataset_edges, self.dataset_nodes = self.get_dataset("cdk8", "5ns")

        # Build graph data
        cycle_data, node2idx, idx2node = self._build_graph_data(self.dataset_edges)
        self.cycle_data = cycle_data
        self.node2idx = node2idx
        self.idx2node = idx2node

        return self.cycle_data

    def get_node_mapping(
        self, dataset: str = None, sampling_time: str = None
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Get the node-to-index and index-to-node mappings.

        Parameters
        ----------
        dataset : str, optional
            Name of the dataset to get mappings for. If None, returns current mappings.
        sampling_time : str, optional
            Sampling time for the dataset. Required if dataset is specified.

        Returns
        -------
        Tuple[Dict[str, int], Dict[int, str]]
            (node2idx, idx2node) mappings
        """
        if dataset is not None and sampling_time is not None:
            cache_key = (dataset, sampling_time)
            if cache_key in self._mapping_cache:
                return self._mapping_cache[cache_key]
            else:
                # Load the dataset to populate the mapping
                self.get_dataset(dataset, sampling_time)
                return self._mapping_cache[cache_key]

        if not hasattr(self, "node2idx") or not hasattr(self, "idx2node"):
            self.get_graph_data()  # This will populate the mappings
        return self.node2idx, self.idx2node

    def get_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the edge and node DataFrames.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (dataset_edges, dataset_nodes) DataFrames
        """
        if not hasattr(self, "dataset_edges") or not hasattr(self, "dataset_nodes"):
            self.dataset_edges, self.dataset_nodes = self.get_dataset("cdk8", "5ns")
        return self.dataset_edges, self.dataset_nodes

    def add_estimator_results(self, estimator_metadata: Dict[str, Any]) -> None:
        """
        Add results from an estimator to the dataset.

        Parameters
        ----------
        estimator_metadata : Dict[str, Any]
            Metadata about the estimator and its results
        """
        if not hasattr(self, "estimators"):
            self.estimators = []
        self.estimators.append(estimator_metadata)

    def get_estimators(self) -> list:
        """
        Get the list of estimators that have been applied to this dataset.

        Returns
        -------
        list
            List of estimator metadata dictionaries
        """
        if not hasattr(self, "estimators"):
            self.estimators = []
        return self.estimators

    def _build_graph_data(
        self, dataset_edges: pd.DataFrame
    ) -> Tuple[Dict[str, Any], Dict[str, int], Dict[int, str]]:
        """
        Convert edge data into graph data format with node indices and edge attributes.

        Parameters
        ----------
        dataset_edges : pd.DataFrame
            DataFrame containing edge data.

        Returns
        -------
        cycle_dat : dict
            Graph data dictionary with keys: N, M, src, dst, FEP, CCC
        node2idx : dict
            Node-to-index mapping.
        idx2node : dict
            Index-to-node mapping.
        """
        start: list[int] = []  # Source node indices
        end: list[int] = []  # Destination node indices
        fep: list[float] = []  # Free Energy Perturbation values
        ccc_dt: list[float] = []  # Cycle closure corrected edge values
        node2idx: Dict[str, int] = {}  # Maps nodes to unique indices
        idx2node: Dict[int, str] = {}  # Maps indices back to nodes
        idx = 0  # Counter for assigning unique indices to nodes

        # Iterate through each row in the edge data
        for _, row in dataset_edges.iterrows():
            for ligand in [row["Source"], row["Destination"]]:
                if ligand not in node2idx:
                    node2idx[ligand] = idx
                    idx2node[idx] = ligand
                    idx += 1
            start.append(node2idx[row["Source"]])  # Source node number (0-based)
            end.append(
                node2idx[row["Destination"]]
            )  # Destination node number (0-based)
            fep.append(float(row["DeltaDeltaG"]))  # FEP value for this edge
            ccc_dt.append(
                float(row["CCC"])
            )  # Cycle closure corrected edge value for this edge

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

    def _apply_node_mapping(
        self, edge_data: pd.DataFrame, node_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply node-to-index mapping to convert string node identifiers to indices.

        This method creates a mapping from node names (like ChEMBL IDs) to indices
        and applies it to both edge and node DataFrames for NodeModel compatibility.

        Parameters
        ----------
        edge_data : pd.DataFrame
            DataFrame containing edge data with Source/Destination columns
        node_data : pd.DataFrame
            DataFrame containing node data with Name column

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Updated (edge_data, node_data) with numeric node indices
        """
        # Create copies to avoid modifying original data
        edge_data = edge_data.copy()
        node_data = node_data.copy()

        # Get all unique node names from both edge and node data
        edge_nodes = set(
            edge_data["Source"].tolist() + edge_data["Destination"].tolist()
        )
        node_names = set(node_data["Name"].tolist())
        all_nodes = edge_nodes.union(node_names)

        # Create node-to-index mapping
        node2idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}
        idx2node = {idx: node for node, idx in node2idx.items()}

        # Store mappings for later use
        self.node2idx = node2idx
        self.idx2node = idx2node

        # Apply mapping to edge data
        edge_data["Source"] = edge_data["Source"].map(node2idx)
        edge_data["Destination"] = edge_data["Destination"].map(node2idx)

        # Apply mapping to node data
        node_data["Name"] = node_data["Name"].map(node2idx)

        return edge_data, node_data
