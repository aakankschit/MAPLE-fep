from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dataset import FEPDataset


class SyntheticFEPDataset(FEPDataset):
    """
    A class to generate synthetic datasets for graph-based analysis.

    This class inherits from FEPDataset and focuses on generating synthetic data.
    Graph creation and visualization should be handled by the GraphSetup class.

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
    dataset_nodes : pd.DataFrame
        DataFrame containing node data.
    dataset_edges : pd.DataFrame
        DataFrame containing edge data.
    perturbed_edges : np.ndarray, optional
        Boolean mask indicating which edges were perturbed (only set when add_noise_to_edges is called with track_edges=True).
    num_test_data : dict, optional
        Perturbed graph data for numerical testing (only set when add_noise_to_edges is called).
    """

    def __init__(
        self,
        add_noise: bool,
        graph_dict: Optional[Dict[Any, List[Any]]] = None,
        random_seed: int = 10,
        **kwargs,
    ):
        """
        Initializes the synthetic dataset by generating nodes, edges, and graph data.

        Parameters
        ----------
        add_noise : bool
            Whether to add noise to predicted values.
        graph_dict : dict, optional
            Dictionary defining the graph topology (nodes and their neighbors).
        random_seed : int, optional
            Random seed for reproducibility.
        kwargs : dict
            Additional arguments for custom dataset initialization.
        """
        np.random.seed(random_seed)
        if graph_dict is None:
            if not kwargs:
                # Default rectangular graph topology with two diagonals
                graph_dict = {1.0: [3.0], 2.0: [1.0], 3.0: [4.0, 2.0], 4.0: [2.0, 1.0]}
                ligands_set = [
                    float(i) for i in list(range(1, len(list(graph_dict.keys())) + 1))
                ]
                self.dataset_nodes = self.get_node_data(
                    ligands=ligands_set, add_noise=add_noise
                )
                source, destination = self.generate_edges(graph_dict)
                self.dataset_edges = self.get_edge_data(
                    source_ligands=source,
                    dest_ligands=destination,
                    dataset_nodes=self.dataset_nodes,
                    add_noise=add_noise,
                )
            else:
                # Custom dataset initialization
                if "dataset_edges" not in kwargs or "dataset_nodes" not in kwargs:
                    raise ValueError(
                        "Custom initialization requires 'dataset_edges' and 'dataset_nodes' in kwargs."
                    )
                self.dataset_nodes = kwargs["dataset_nodes"]
                self.dataset_edges = kwargs["dataset_edges"]
        else:
            # Custom graph topology
            if not isinstance(graph_dict, dict):
                raise ValueError(
                    "graph_dict must be a dictionary mapping nodes to neighbor lists."
                )
            ligands_set = list(
                set(
                    [key for key in graph_dict.keys()]
                    + [lig for neighbors in graph_dict.values() for lig in neighbors]
                )
            )
            self.dataset_nodes = self.get_node_data(
                ligands=ligands_set, add_noise=add_noise
            )
            source, destination = self.generate_edges(graph_dict)
            self.dataset_edges = self.get_edge_data(
                source_ligands=source,
                dest_ligands=destination,
                dataset_nodes=self.dataset_nodes,
                add_noise=add_noise,
            )

        # Initialize the parent class with the generated data
        super().__init__(
            dataset_nodes=self.dataset_nodes, dataset_edges=self.dataset_edges
        )

        # Initialize noise testing attributes
        self.perturbed_edges = None
        self.num_test_data = None

        # Initialize estimators list for storing model results
        self.estimators = []

    def get_node_data(self, ligands: List[Any], add_noise: bool) -> pd.DataFrame:
        """
        Generates node data with predicted and experimental values.

        Parameters
        ----------
        ligands : list
            List of ligand indices.
        add_noise : bool
            Whether to add noise to predicted values.

        Returns
        -------
        dataset_nodes : pd.DataFrame
            DataFrame containing node data.
        """
        truth_vals = np.random.rand(len(ligands))
        dataset_nodes = pd.DataFrame(columns=["Name", "Pred. DeltaG", "Exp. DeltaG"])
        dataset_nodes["Name"] = ligands

        if add_noise:
            noise = np.random.normal(0, 0.1, len(ligands))
            dataset_nodes["Pred. DeltaG"] = truth_vals + noise
        else:
            dataset_nodes["Pred. DeltaG"] = truth_vals
            print(
                "⚠️  Warning: SyntheticFEPDataset created with add_noise=False. "
                "Predicted and experimental node values are identical."
            )

        dataset_nodes["Exp. DeltaG"] = truth_vals

        return dataset_nodes

    def generate_edges(
        self, graph: Dict[Any, List[Any]]
    ) -> Tuple[List[Any], List[Any]]:
        """
        Generates edges (source and destination nodes) from the graph dictionary.

        Parameters
        ----------
        graph : dict
            Dictionary defining the graph topology.

        Returns
        -------
        ligand1 : list
            Source nodes.
        ligand2 : list
            Destination nodes.
        """
        ligand1 = []  # Source nodes
        ligand2 = []  # Destination nodes
        for node in graph:
            for neighbour in graph[node]:
                ligand1.append(node)
                ligand2.append(neighbour)

        return ligand1, ligand2

    def get_edge_data(
        self,
        source_ligands: List[Any],
        dest_ligands: List[Any],
        dataset_nodes: pd.DataFrame,
        add_noise: bool = True,
    ) -> pd.DataFrame:
        """
        Generates edge data (DeltaDeltaG, CCC, experimental values) from node data.

        Parameters
        ----------
        source_ligands : list
            List of source node indices.
        dest_ligands : list
            List of destination node indices.
        dataset_nodes : pd.DataFrame
            DataFrame containing node data.
        add_noise : bool, default True
            Whether noise was added to predicted values.

        Returns
        -------
        dataset_edges : pd.DataFrame
            DataFrame containing edge data.
        """
        dataset_edges = pd.DataFrame(
            columns=[
                "Source",
                "Destination",
                "Experimental DeltaDeltaG",
                "DeltaDeltaG",
                "CCC",
            ]
        )
        dataset_edges["Source"] = source_ligands
        dataset_edges["Destination"] = dest_ligands
        dataset_edges["CCC"] = np.zeros(len(dataset_edges))
        edge_list = [
            (dataset_edges["Source"][i], dataset_edges["Destination"][i])
            for i in range(len(dataset_edges))
        ]
        DeltaDeltaG_edges = []
        exp_edges = []
        for edge in edge_list:
            source, end = edge
            DeltaDeltaG_edges.append(
                dataset_nodes[dataset_nodes["Name"] == end][
                    "Pred. DeltaG"
                ].values.item()
                - dataset_nodes[dataset_nodes["Name"] == source][
                    "Pred. DeltaG"
                ].values.item()
            )
            exp_edges.append(
                dataset_nodes[dataset_nodes["Name"] == end]["Exp. DeltaG"].values.item()
                - dataset_nodes[dataset_nodes["Name"] == source][
                    "Exp. DeltaG"
                ].values.item()
            )
        dataset_edges["DeltaDeltaG"] = DeltaDeltaG_edges
        dataset_edges["Experimental DeltaDeltaG"] = exp_edges

        # Warn if predicted and experimental edge values are the same
        if not add_noise:
            print(
                "⚠️  Warning: SyntheticFEPDataset created with add_noise=False. "
                "Predicted and experimental edge values (DeltaDeltaG) are identical."
            )

        return dataset_edges

    def add_noise_to_edges(
        self,
        noise_coeff: float,
        noise_mu: float,
        noise_sig: float,
        track_edges: bool = False,
    ) -> Dict[str, Any]:
        """
        Add noise to experimental edge data for numerical testing purposes.

        This method creates a perturbed version of the dataset by adding noise
        to the experimental edge values. This is useful for testing the robustness
        of models to noisy data.

        Parameters
        ----------
        noise_coeff : float
            Coefficient for noise scaling. Controls the magnitude of the noise.
        noise_mu : float
            Mean of the noise distribution.
        noise_sig : float
            Standard deviation of the noise distribution.
        track_edges : bool, default=False
            Whether to track which edges were perturbed. If True, creates a
            boolean mask stored in self.perturbed_edges.

        Returns
        -------
        Dict[str, Any]
            Perturbed graph data dictionary with the same structure as cycle_data
            but with noisy edge values in the 'FEP' field.

        Examples
        --------
        >>> dataset = SyntheticFEPDataset(add_noise=False)
        >>> perturbed_data = dataset.add_noise_to_edges(
        ...     noise_coeff=0.1, noise_mu=0.0, noise_sig=1.0
        ... )
        >>> print(f"Original edge values: {dataset.cycle_data['FEP']}")
        >>> print(f"Perturbed edge values: {perturbed_data['FEP']}")
        """
        if self.cycle_data is None:
            raise ValueError(
                "No cycle data available. Ensure dataset is properly initialized."
            )

        # Create a copy of the cycle data
        graph_data = self.cycle_data.copy()

        # Get the original FEP values
        original_fep_values = np.array(self.cycle_data["FEP"])
        num_edges = len(original_fep_values)

        # Generate noise based on the number of edges
        noise = noise_coeff * np.random.normal(noise_mu, noise_sig, num_edges)

        if track_edges:
            # Add initial noise to all edges
            perturbed_data = original_fep_values + noise

            # Create a random mask for additional perturbation
            mask = np.random.randint(0, 2, size=num_edges).astype(bool)

            # Add additional noise to masked edges
            additional_noise = np.random.normal(noise_mu, 10 * noise_sig, num_edges)
            # Only add additional noise to masked edges
            perturbed_data[mask] = perturbed_data[mask] + additional_noise[mask]

            # Store the perturbed edges mask
            self.perturbed_edges = mask

            # Update the graph data with perturbed values
            graph_data["FEP"] = perturbed_data.tolist()
        else:
            # Simple noise addition without tracking
            graph_data["FEP"] = (original_fep_values + noise).tolist()

        # Store the perturbed data for later use
        self.num_test_data = graph_data

        return graph_data

    def get_perturbed_dataset(self) -> Optional[Dict[str, Any]]:
        """
        Get the perturbed dataset if it exists.

        Returns
        -------
        Optional[Dict[str, Any]]
            The perturbed graph data if add_noise_to_edges has been called,
            None otherwise.
        """
        return self.num_test_data

    def get_perturbed_edges_mask(self) -> Optional[np.ndarray]:
        """
        Get the mask of perturbed edges if tracking was enabled.

        Returns
        -------
        Optional[np.ndarray]
            Boolean mask indicating which edges were perturbed if
            add_noise_to_edges was called with track_edges=True,
            None otherwise.
        """
        return self.perturbed_edges

    def reset_noise_testing(self) -> None:
        """
        Reset noise testing data and clear perturbed information.

        This method clears the perturbed edges mask and test data,
        effectively resetting the dataset to its original state.
        """
        self.perturbed_edges = None
        self.num_test_data = None

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
