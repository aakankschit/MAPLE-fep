from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import pandas as pd


class BaseDataset(ABC):
    """
    Abstract base class for all FEP dataset classes.

    This class defines the interface that all dataset classes must implement.
    It ensures that NodeModel can work with any dataset type without needing
    to know the specific implementation details.
    """

    @abstractmethod
    def get_graph_data(self) -> Dict[str, Any]:
        """
        Get the graph data in the format required by NodeModel.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'N': number of nodes
            - 'M': number of edges
            - 'src': source node indices
            - 'dst': destination node indices
            - 'FEP': FEP values
            - 'CCC': CCC data
        """
        pass

    @abstractmethod
    def get_node_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Get the node-to-index and index-to-node mappings.

        Returns
        -------
        Tuple[Dict[str, int], Dict[int, str]]
            (node2idx, idx2node) mappings
        """
        pass

    @abstractmethod
    def get_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the edge and node DataFrames.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (dataset_edges, dataset_nodes) DataFrames
        """
        pass

    @abstractmethod
    def add_estimator_results(self, estimator_metadata: Dict[str, Any]) -> None:
        """
        Add results from an estimator to the dataset.

        Parameters
        ----------
        estimator_metadata : Dict[str, Any]
            Metadata about the estimator and its results
        """
        pass

    @abstractmethod
    def get_estimators(self) -> list:
        """
        Get the list of estimators that have been applied to this dataset.

        Returns
        -------
        list
            List of estimator metadata dictionaries
        """
        pass
