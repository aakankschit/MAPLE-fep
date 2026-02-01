"""
Pytest configuration and shared fixtures for MAPLE tests.

This module provides common test fixtures and utilities that can be used
across all test modules in the MAPLE package.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def sample_edge_data():
    """
    Create sample edge data for testing.

    Returns
    -------
    pd.DataFrame
        Sample edge DataFrame with required columns
    """
    return pd.DataFrame(
        {
            "Source": ["mol_A", "mol_B", "mol_C", "mol_A"],
            "Destination": ["mol_B", "mol_C", "mol_D", "mol_C"],
            "DeltaDeltaG": [1.2, -0.8, 2.1, 0.5],
            "uncertainty": [0.1, 0.15, 0.2, 0.12],
        }
    )


@pytest.fixture
def sample_node_data():
    """
    Create sample node data for testing.

    Returns
    -------
    pd.DataFrame
        Sample node DataFrame with required columns
    """
    return pd.DataFrame(
        {
            "Name": ["mol_A", "mol_B", "mol_C", "mol_D"],
            "Exp. DeltaG": [0.0, 1.2, 0.4, 2.5],
        }
    )


@pytest.fixture
def sample_ccc_data():
    """
    Create sample CCC (Cycle Closure Consistency) data for testing.

    Returns
    -------
    pd.DataFrame
        Sample CCC DataFrame with cycle information
    """
    return pd.DataFrame(
        {
            "cycle_id": [0, 0, 0, 1, 1, 1],
            "Source": ["mol_A", "mol_B", "mol_C", "mol_B", "mol_C", "mol_D"],
            "Destination": ["mol_B", "mol_C", "mol_A", "mol_C", "mol_D", "mol_B"],
            "DeltaDeltaG": [1.2, -0.8, -0.4, -0.8, 2.1, -1.3],
            "cycle_error": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


@pytest.fixture
def sample_graph_data(sample_edge_data, sample_ccc_data):
    """
    Create sample graph data dictionary for testing NodeModel.

    Parameters
    ----------
    sample_edge_data : pd.DataFrame
        Edge data fixture
    sample_ccc_data : pd.DataFrame
        CCC data fixture

    Returns
    -------
    Dict[str, Any]
        Graph data dictionary in NodeModel format
    """
    # Create node mappings
    molecules = set(sample_edge_data["Source"]) | set(sample_edge_data["Destination"])
    node2idx = {mol: i for i, mol in enumerate(sorted(molecules))}

    # Convert to indices
    src_indices = [node2idx[mol] for mol in sample_edge_data["Source"]]
    dst_indices = [node2idx[mol] for mol in sample_edge_data["Destination"]]

    return {
        "N": len(molecules),
        "M": len(sample_edge_data),
        "src": torch.tensor(src_indices, dtype=torch.long),
        "dst": torch.tensor(dst_indices, dtype=torch.long),
        "FEP": torch.tensor(
            sample_edge_data["DeltaDeltaG"].values, dtype=torch.float32
        ),
        "CCC": sample_ccc_data,
    }


@pytest.fixture
def sample_arrays_for_stats():
    """
    Create sample arrays for testing statistical functions.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (y_true, y_pred) arrays for statistical testing
    """
    np.random.seed(42)  # For reproducible tests
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true + np.random.normal(0, 0.1, len(y_true))  # Add small noise
    return y_true, y_pred


@pytest.fixture
def sample_arrays_with_nan():
    """
    Create sample arrays with NaN values for testing edge cases.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (y_true, y_pred) arrays containing NaN values
    """
    y_true = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 3.1, np.nan, 5.1])
    return y_true, y_pred


@pytest.fixture
def mock_torch_device():
    """
    Get the appropriate torch device for testing.

    Returns
    -------
    torch.device
        CPU device for consistent testing
    """
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def reset_pyro():
    """
    Reset Pyro's global state before each test.
    This fixture runs automatically before each test.
    """
    import pyro

    pyro.clear_param_store()
    yield
    pyro.clear_param_store()


class MockDataset:
    """
    Mock dataset class for testing purposes.

    This class provides a simple implementation of the BaseDataset interface
    that can be used in tests without requiring real data files.
    """

    def __init__(
        self,
        edge_data: pd.DataFrame,
        node_data: pd.DataFrame,
        ccc_data: pd.DataFrame = None,
    ):
        """
        Initialize mock dataset.

        Parameters
        ----------
        edge_data : pd.DataFrame
            Edge data
        node_data : pd.DataFrame
            Node data
        ccc_data : pd.DataFrame, optional
            CCC data
        """
        self.edge_data = edge_data
        self.node_data = node_data
        self.ccc_data = ccc_data if ccc_data is not None else pd.DataFrame()

        # NodeModel expects these specific attribute names
        self.dataset_edges = edge_data
        self.dataset_nodes = node_data

        # Create node mappings
        molecules = set(edge_data["Source"]) | set(edge_data["Destination"])
        self.node2idx = {mol: i for i, mol in enumerate(sorted(molecules))}
        self.idx2node = {i: mol for mol, i in self.node2idx.items()}

    def get_graph_data(self) -> Dict[str, Any]:
        """Get graph data in NodeModel format."""
        src_indices = [self.node2idx[mol] for mol in self.edge_data["Source"]]
        dst_indices = [self.node2idx[mol] for mol in self.edge_data["Destination"]]

        return {
            "N": len(self.node2idx),
            "M": len(self.edge_data),
            "src": torch.tensor(src_indices, dtype=torch.long),
            "dst": torch.tensor(dst_indices, dtype=torch.long),
            "FEP": torch.tensor(
                self.edge_data["DeltaDeltaG"].values, dtype=torch.float32
            ),
            "CCC": self.ccc_data,
        }

    def get_node_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Get node mappings."""
        return self.node2idx, self.idx2node

    def get_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get edge and node DataFrames."""
        return self.edge_data, self.node_data

    def add_estimator_results(self, estimator_metadata: Dict[str, Any]) -> None:
        """Add results from an estimator to the dataset."""
        pass

    def get_estimators(self) -> list:
        """Get the list of estimators that have been applied to this dataset."""
        return []


@pytest.fixture
def mock_dataset(sample_edge_data, sample_node_data, sample_ccc_data):
    """
    Create a mock dataset for testing.

    Parameters
    ----------
    sample_edge_data : pd.DataFrame
        Edge data fixture
    sample_node_data : pd.DataFrame
        Node data fixture
    sample_ccc_data : pd.DataFrame
        CCC data fixture

    Returns
    -------
    MockDataset
        Mock dataset instance
    """
    return MockDataset(sample_edge_data, sample_node_data, sample_ccc_data)


@pytest.fixture
def temp_directory():
    """
    Create a temporary directory for testing.

    This fixture provides a clean temporary directory that is automatically
    cleaned up after the test completes.

    Returns
    -------
    Path
        Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_performance_data():
    """
    Create sample performance data for testing utils components.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing sample performance arrays
    """
    np.random.seed(42)
    return {
        "y_true": np.random.rand(20) * 10,
        "y_pred": np.random.rand(20) * 10,
        "dy_true": np.random.rand(20) * 0.5,
        "dy_pred": np.random.rand(20) * 0.5,
    }


@pytest.fixture
def sample_model_config():
    """
    Create a sample model configuration for testing.

    Returns
    -------
    Dict[str, Any]
        Sample model configuration dictionary
    """
    return {
        "learning_rate": 0.001,
        "num_steps": 1000,
        "prior_type": "normal",
        "prior_parameters": [0.0, 1.0],
        "error_std": 1.0,
    }


@pytest.fixture
def sample_dataset_info():
    """
    Create sample dataset information for testing.

    Returns
    -------
    Dict[str, Any]
        Sample dataset information dictionary
    """
    return {
        "name": "test_dataset",
        "n_nodes": 20,
        "n_edges": 35,
        "density": 0.3,
        "source": "synthetic",
    }


@pytest.fixture(scope="session", autouse=True)
def suppress_warnings():
    """
    Suppress common warnings during testing.

    This fixture runs once per test session and configures
    warning filters to reduce noise in test output.
    """
    import warnings

    # Suppress specific warnings that commonly occur during testing
    warnings.filterwarnings("ignore", category=UserWarning, message=".*scatter_.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*pandas.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*numpy.*")

    # Suppress matplotlib warnings in headless environments
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Matplotlib.*")

    yield
