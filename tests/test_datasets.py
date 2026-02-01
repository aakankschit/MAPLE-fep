"""
Unit tests for the dataset modules.

This module tests all dataset classes in the MAPLE package, including
BaseDataset interface compliance, data loading functionality, and
proper graph data generation for NodeModel compatibility.
"""

import os
import tempfile

import pandas as pd
import pytest
import torch

from maple.dataset.base_dataset import BaseDataset
from maple.dataset.dataset import FEPDataset
from maple.dataset.FEP_benchmark_dataset import FEPBenchmarkDataset
from maple.dataset.synthetic_dataset import SyntheticFEPDataset


class TestBaseDataset:
    """
    Test cases for the BaseDataset abstract base class.

    The BaseDataset class should:
    - Define the correct abstract interface
    - Prevent direct instantiation
    - Ensure subclasses implement required methods
    """

    def test_cannot_instantiate_directly(self):
        """Test that BaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDataset()

    def test_abstract_methods_defined(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = BaseDataset.__abstractmethods__
        expected_methods = {
            "get_graph_data",
            "get_node_mapping",
            "get_dataframes",
            "add_estimator_results",
            "get_estimators",
        }

        assert (
            abstract_methods == expected_methods
        ), f"Expected abstract methods {expected_methods}, got {abstract_methods}"

    def test_subclass_must_implement_methods(self):
        """Test that subclasses must implement all abstract methods."""

        class IncompleteDataset(BaseDataset):
            # Missing implementations
            pass

        with pytest.raises(TypeError):
            IncompleteDataset()


class TestFEPDataset:
    """
    Test cases for the FEPDataset class.

    The FEPDataset class should:
    - Load data from CSV files correctly
    - Generate proper graph data format
    - Handle various data formats and edge cases
    - Provide correct node mappings
    """

    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        edge_csv = """Source,Destination,DeltaDeltaG,uncertainty
mol_A,mol_B,1.2,0.1
mol_B,mol_C,-0.8,0.15
mol_C,mol_D,2.1,0.2
mol_A,mol_C,0.5,0.12"""

        node_csv = """molecule,Exp. DeltaG
mol_A,0.0
mol_B,1.2
mol_C,0.4
mol_D,2.5"""

        return edge_csv, node_csv

    @pytest.fixture
    def temp_csv_files(self, sample_csv_data):
        """Create temporary CSV files for testing."""
        edge_csv, node_csv = sample_csv_data

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f_edge:
            f_edge.write(edge_csv)
            edge_file = f_edge.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f_node:
            f_node.write(node_csv)
            node_file = f_node.name

        yield edge_file, node_file

        # Cleanup
        os.unlink(edge_file)
        os.unlink(node_file)

    def test_load_from_csv(self, temp_csv_files):
        """Test loading FEPDataset from CSV files."""
        edge_file, node_file = temp_csv_files

        dataset = FEPDataset(nodes_csv_path=node_file, edges_csv_path=edge_file)

        # Check that data was loaded
        edge_data, node_data = dataset.get_dataframes()

        assert len(edge_data) == 4, "Should load 4 edges"
        assert len(node_data) == 4, "Should load 4 nodes"

        # Check column names
        assert "Source" in edge_data.columns
        assert "Destination" in edge_data.columns
        assert "DeltaDeltaG" in edge_data.columns
        assert "molecule" in node_data.columns

    def test_get_graph_data(self, temp_csv_files):
        """Test graph data generation for NodeModel compatibility."""
        edge_file, node_file = temp_csv_files

        dataset = FEPDataset(nodes_csv_path=node_file, edges_csv_path=edge_file)
        graph_data = dataset.get_graph_data()

        # Check required keys
        required_keys = {"N", "M", "src", "dst", "FEP", "CCC"}
        assert (
            set(graph_data.keys()) == required_keys
        ), f"Missing keys: {required_keys - set(graph_data.keys())}"

        # Check data types and shapes
        assert isinstance(graph_data["N"], int)
        assert isinstance(graph_data["M"], int)
        assert isinstance(graph_data["src"], list)
        assert isinstance(graph_data["dst"], list)
        assert isinstance(graph_data["FEP"], list)

        # Check list lengths
        assert len(graph_data["src"]) == graph_data["M"]
        assert len(graph_data["dst"]) == graph_data["M"]
        assert len(graph_data["FEP"]) == graph_data["M"]

        # Check that indices are valid
        assert all(src >= 0 for src in graph_data["src"])
        assert all(src < graph_data["N"] for src in graph_data["src"])
        assert all(dst >= 0 for dst in graph_data["dst"])
        assert all(dst < graph_data["N"] for dst in graph_data["dst"])

    def test_get_node_mapping(self, temp_csv_files):
        """Test node mapping generation."""
        edge_file, node_file = temp_csv_files

        dataset = FEPDataset(nodes_csv_path=node_file, edges_csv_path=edge_file)
        node2idx, idx2node = dataset.get_node_mapping()

        # Check mapping consistency
        assert len(node2idx) == len(idx2node)

        # Check bidirectional mapping
        for node, idx in node2idx.items():
            assert idx2node[idx] == node

        # Check all molecules are included
        edge_data, node_data = dataset.get_dataframes()
        all_molecules = set(edge_data["Source"]) | set(edge_data["Destination"])

        assert set(node2idx.keys()) == all_molecules

        # Check indices are contiguous from 0
        indices = list(node2idx.values())
        assert set(indices) == set(range(len(indices)))

    def test_missing_file_error(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            FEPDataset(
                nodes_csv_path="nonexistent_node.csv",
                edges_csv_path="nonexistent_edge.csv",
            )

    def test_invalid_csv_format(self):
        """Test error handling for invalid CSV format."""
        # Create CSV with missing required columns
        invalid_csv = """wrong_col1,wrong_col2
        value1,value2"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(invalid_csv)
            invalid_file = f.name

        try:
            with pytest.raises((KeyError, ValueError)):
                FEPDataset(nodes_csv_path=invalid_file, edges_csv_path=invalid_file)
        finally:
            os.unlink(invalid_file)

    def test_derive_nodes_from_edges_simple_chain(self):
        """Test deriving node values from a simple chain graph."""
        # Create a simple chain: A -> B -> C -> D
        edge_data = pd.DataFrame(
            {
                "Source": ["mol_A", "mol_B", "mol_C"],
                "Destination": ["mol_B", "mol_C", "mol_D"],
                "DeltaDeltaG": [1.0, 2.0, 3.0],
            }
        )

        edge_df, node_df = FEPDataset.derive_nodes_from_edges(edge_data)

        # Verify we have 4 nodes
        assert len(node_df) == 4, f"Expected 4 nodes, got {len(node_df)}"

        # Check that node values are consistent with edges
        # mol_A should be 0.0 (reference), mol_B = 1.0, mol_C = 3.0, mol_D = 6.0
        node_dict = dict(zip(node_df["Name"], node_df["Exp. DeltaG"]))

        # Verify edge consistency: for each edge A->B, check that B - A = edge_value
        for _, row in edge_df.iterrows():
            src_val = node_dict[row["Source"]]
            dst_val = node_dict[row["Destination"]]
            edge_val = row["DeltaDeltaG"]

            assert abs((dst_val - src_val) - edge_val) < 1e-10, (
                f"Edge {row['Source']}->{row['Destination']}: "
                f"node difference {dst_val - src_val} != edge value {edge_val}"
            )

    def test_derive_nodes_from_edges_with_cycles(self):
        """Test deriving node values from a graph with cycles."""
        # Create a cyclic graph: A -> B -> C -> A
        edge_data = pd.DataFrame(
            {
                "Source": ["mol_A", "mol_B", "mol_C", "mol_A"],
                "Destination": ["mol_B", "mol_C", "mol_A", "mol_C"],
                "DeltaDeltaG": [1.0, 2.0, -3.0, 3.0],  # Forms a cycle
            }
        )

        edge_df, node_df = FEPDataset.derive_nodes_from_edges(edge_data)

        # Verify we have 3 nodes
        assert len(node_df) == 3, f"Expected 3 nodes, got {len(node_df)}"

        # Create node value dictionary
        node_dict = dict(zip(node_df["Name"], node_df["Exp. DeltaG"]))

        # Verify that the first 3 edges are consistent
        # (The 4th edge might have a cycle closure error)
        for i in range(3):
            row = edge_df.iloc[i]
            src_val = node_dict[row["Source"]]
            dst_val = node_dict[row["Destination"]]
            edge_val = row["DeltaDeltaG"]

            assert abs((dst_val - src_val) - edge_val) < 1e-10, (
                f"Edge {row['Source']}->{row['Destination']}: "
                f"node difference {dst_val - src_val} != edge value {edge_val}"
            )

    def test_derive_nodes_from_edges_disconnected(self):
        """Test deriving node values from disconnected components."""
        # Create two disconnected components: A-B and C-D
        edge_data = pd.DataFrame(
            {
                "Source": ["mol_A", "mol_C"],
                "Destination": ["mol_B", "mol_D"],
                "DeltaDeltaG": [1.5, 2.5],
            }
        )

        edge_df, node_df = FEPDataset.derive_nodes_from_edges(edge_data)

        # Verify we have 4 nodes
        assert len(node_df) == 4, f"Expected 4 nodes, got {len(node_df)}"

        # Create node value dictionary
        node_dict = dict(zip(node_df["Name"], node_df["Exp. DeltaG"]))

        # Count reference nodes (value = 0.0)
        reference_count = sum(1 for v in node_dict.values() if abs(v) < 1e-10)
        assert reference_count == 2, (
            f"Expected 2 reference nodes for 2 components, got {reference_count}"
        )

        # Verify each edge is consistent
        for _, row in edge_df.iterrows():
            src_val = node_dict[row["Source"]]
            dst_val = node_dict[row["Destination"]]
            edge_val = row["DeltaDeltaG"]

            assert abs((dst_val - src_val) - edge_val) < 1e-10, (
                f"Edge {row['Source']}->{row['Destination']}: "
                f"node difference {dst_val - src_val} != edge value {edge_val}"
            )

    def test_derive_nodes_custom_reference(self):
        """Test deriving nodes with a custom reference node."""
        edge_data = pd.DataFrame(
            {
                "Source": ["mol_A", "mol_B"],
                "Destination": ["mol_B", "mol_C"],
                "DeltaDeltaG": [1.0, 2.0],
            }
        )

        # Set mol_B as reference
        _, node_df = FEPDataset.derive_nodes_from_edges(
            edge_data, reference_node="mol_B"
        )

        node_dict = dict(zip(node_df["Name"], node_df["Exp. DeltaG"]))

        # mol_B should be 0.0
        assert abs(node_dict["mol_B"]) < 1e-10, (
            f"Reference node mol_B should be 0.0, got {node_dict['mol_B']}"
        )
        # mol_A should be -1.0 (reverse of edge)
        assert abs(node_dict["mol_A"] - (-1.0)) < 1e-10, (
            f"mol_A should be -1.0, got {node_dict['mol_A']}"
        )
        # mol_C should be 2.0
        assert abs(node_dict["mol_C"] - 2.0) < 1e-10, (
            f"mol_C should be 2.0, got {node_dict['mol_C']}"
        )

    def test_derive_nodes_ligand_columns(self):
        """Test deriving nodes from DataFrame with Ligand1/Ligand2 columns."""
        edge_data = pd.DataFrame(
            {
                "Ligand1": ["mol_A", "mol_B"],
                "Ligand2": ["mol_B", "mol_C"],
                "FEP": [1.5, -0.5],
            }
        )

        edge_df, node_df = FEPDataset.derive_nodes_from_edges(edge_data)

        # Verify we have 3 nodes
        assert len(node_df) == 3, f"Expected 3 nodes, got {len(node_df)}"

        # Verify edges are consistent
        node_dict = dict(zip(node_df["Name"], node_df["Exp. DeltaG"]))

        # Use Ligand1/Ligand2 column names
        for _, row in edge_df.iterrows():
            src_val = node_dict[row["Ligand1"]]
            dst_val = node_dict[row["Ligand2"]]
            edge_val = row["FEP"]

            assert abs((dst_val - src_val) - edge_val) < 1e-10, (
                f"Edge inconsistency for {row['Ligand1']}->{row['Ligand2']}"
            )

    def test_derive_nodes_custom_value_column(self):
        """Test deriving nodes with custom value column name."""
        edge_data = pd.DataFrame(
            {
                "Source": ["mol_A", "mol_B"],
                "Destination": ["mol_B", "mol_C"],
                "CustomValue": [1.0, 2.0],
            }
        )

        _, node_df = FEPDataset.derive_nodes_from_edges(
            edge_data, value_column="CustomValue"
        )

        assert len(node_df) == 3, f"Expected 3 nodes, got {len(node_df)}"

    def test_derive_nodes_invalid_columns(self):
        """Test error handling for invalid column names."""
        edge_data = pd.DataFrame(
            {
                "wrong1": ["mol_A"],
                "wrong2": ["mol_B"],
                "value": [1.0],
            }
        )

        with pytest.raises(ValueError, match="must have either"):
            FEPDataset.derive_nodes_from_edges(edge_data)

    def test_derive_nodes_invalid_value_column(self):
        """Test error handling for invalid value column."""
        edge_data = pd.DataFrame(
            {
                "Source": ["mol_A"],
                "Destination": ["mol_B"],
                "WrongValue": [1.0],
            }
        )

        with pytest.raises(ValueError, match="Could not automatically identify"):
            FEPDataset.derive_nodes_from_edges(edge_data)

    def test_derive_nodes_invalid_reference(self):
        """Test error handling for invalid reference node."""
        edge_data = pd.DataFrame(
            {
                "Source": ["mol_A"],
                "Destination": ["mol_B"],
                "DeltaDeltaG": [1.0],
            }
        )

        with pytest.raises(ValueError, match="Reference node.*not found"):
            FEPDataset.derive_nodes_from_edges(
                edge_data, reference_node="nonexistent"
            )

    def test_init_with_edges_only(self):
        """Test initializing FEPDataset with only edge data."""
        # Create edge data only
        edge_data = pd.DataFrame(
            {
                "Source": ["mol_A", "mol_B", "mol_C"],
                "Destination": ["mol_B", "mol_C", "mol_D"],
                "DeltaDeltaG": [1.0, 2.0, 3.0],
            }
        )

        # Initialize with edges only - nodes should be derived automatically
        dataset = FEPDataset(dataset_edges=edge_data)

        # Verify we have both edges and nodes
        edge_df, node_df = dataset.get_dataframes()
        assert len(edge_df) == 3, f"Expected 3 edges, got {len(edge_df)}"
        assert len(node_df) == 4, f"Expected 4 nodes, got {len(node_df)}"

        # Verify node values are consistent with edges
        node_dict = dict(zip(node_df["Name"], node_df["Exp. DeltaG"]))
        for _, row in edge_df.iterrows():
            src_val = node_dict[row["Source"]]
            dst_val = node_dict[row["Destination"]]
            edge_val = row["DeltaDeltaG"]

            assert abs((dst_val - src_val) - edge_val) < 1e-10, (
                f"Edge {row['Source']}->{row['Destination']}: "
                f"node difference {dst_val - src_val} != edge value {edge_val}"
            )

        # Verify graph data is available
        graph_data = dataset.get_graph_data()
        assert graph_data["N"] == 4
        assert graph_data["M"] == 3

    def test_init_with_edges_only_ligand_columns(self):
        """Test initializing with edges only using Ligand1/Ligand2 columns."""
        edge_data = pd.DataFrame(
            {
                "Ligand1": ["mol_A", "mol_B"],
                "Ligand2": ["mol_B", "mol_C"],
                "FEP": [1.5, -0.5],
            }
        )

        dataset = FEPDataset(dataset_edges=edge_data)
        edge_df, node_df = dataset.get_dataframes()

        assert len(node_df) == 3, f"Expected 3 nodes, got {len(node_df)}"
        assert len(edge_df) == 2, f"Expected 2 edges, got {len(edge_df)}"


class TestSyntheticFEPDataset:
    """
    Test cases for the SyntheticFEPDataset class.

    The SyntheticFEPDataset class should:
    - Generate synthetic data with specified parameters
    - Create valid graph structures
    - Add noise appropriately
    - Handle different graph topologies
    """

    def test_basic_synthetic_generation(self):
        """Test basic synthetic dataset generation."""
        dataset = SyntheticFEPDataset(add_noise=True, random_seed=42)

        graph_data = dataset.get_graph_data()

        # Check basic properties (default topology has 4 nodes, 6 edges)
        assert graph_data["N"] == 4, f"Expected 4 nodes, got {graph_data['N']}"
        assert graph_data["M"] == 6, f"Expected 6 edges, got {graph_data['M']}"

    def test_reproducible_generation(self):
        """Test that synthetic generation is reproducible with fixed seed."""
        dataset1 = SyntheticFEPDataset(add_noise=True, random_seed=42)

        dataset2 = SyntheticFEPDataset(add_noise=True, random_seed=42)

        graph_data1 = dataset1.get_graph_data()
        graph_data2 = dataset2.get_graph_data()

        # Should generate identical data
        assert graph_data1["N"] == graph_data2["N"]
        assert graph_data1["M"] == graph_data2["M"]
        assert graph_data1["FEP"] == graph_data2["FEP"]

    def test_noise_level_effect(self):
        """Test that noise setting affects the generated data appropriately."""
        # Generate datasets with and without noise
        dataset_no_noise = SyntheticFEPDataset(add_noise=False, random_seed=42)

        dataset_with_noise = SyntheticFEPDataset(add_noise=True, random_seed=42)

        # Both should generate valid data
        graph_no_noise = dataset_no_noise.get_graph_data()
        graph_with_noise = dataset_with_noise.get_graph_data()

        assert graph_no_noise["M"] > 0
        assert graph_with_noise["M"] > 0
        assert not any(v != v for v in graph_no_noise["FEP"])  # Check for NaN
        assert not any(v != v for v in graph_with_noise["FEP"])  # Check for NaN

    def test_parameter_validation(self):
        """Test validation of input parameters."""
        # Test that required parameter add_noise must be provided
        with pytest.raises(TypeError):
            SyntheticFEPDataset()  # Missing required add_noise parameter

        # Test that boolean values work for add_noise
        dataset_true = SyntheticFEPDataset(add_noise=True)
        dataset_false = SyntheticFEPDataset(add_noise=False)

        assert dataset_true.get_graph_data()["N"] > 0
        assert dataset_false.get_graph_data()["N"] > 0

    def test_node_mapping_synthetic(self):
        """Test node mapping for synthetic dataset."""
        dataset = SyntheticFEPDataset(add_noise=False, random_seed=42)

        node2idx, idx2node = dataset.get_node_mapping()

        # Check mapping properties
        assert len(node2idx) == 4
        assert len(idx2node) == 4

        # Check that mappings are consistent
        for node, idx in node2idx.items():
            assert idx2node[idx] == node

        # Check that indices are contiguous
        assert set(idx2node.keys()) == {0, 1, 2, 3}


class TestFEPBenchmarkDataset:
    """
    Test cases for the FEPBenchmarkDataset class.

    The FEPBenchmarkDataset class should:
    - Load benchmark datasets correctly
    - Handle different benchmark formats
    - Provide consistent interface with other datasets
    - Include experimental data when available
    """

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_benchmark_data(self):
        """Create mock benchmark data structure."""
        return {
            "edges": pd.DataFrame(
                {
                    "mol_i": ["compound_1", "compound_2", "compound_3"],
                    "mol_j": ["compound_2", "compound_3", "compound_1"],
                    "FEP": [1.5, -2.1, 0.6],
                    "uncertainty": [0.2, 0.3, 0.15],
                }
            ),
            "nodes": pd.DataFrame(
                {
                    "molecule": ["compound_1", "compound_2", "compound_3"],
                    "Exp. DeltaG": [-1.2, 0.3, -1.5],
                }
            ),
            "metadata": {
                "dataset_name": "test_benchmark",
                "source": "test_data",
                "n_compounds": 3,
            },
        }

    def test_load_benchmark_dataset(self, temp_cache_dir):
        """Test loading benchmark dataset."""
        # Test basic initialization with temporary cache
        dataset = FEPBenchmarkDataset(cache_dir=temp_cache_dir)

        # Test supported datasets list
        assert hasattr(dataset, "SUPPORTED_DATASETS")
        assert isinstance(dataset.SUPPORTED_DATASETS, list)
        assert len(dataset.SUPPORTED_DATASETS) > 0

    def test_benchmark_graph_data(self, temp_cache_dir):
        """Test benchmark dataset initialization and basic properties."""
        dataset = FEPBenchmarkDataset(cache_dir=temp_cache_dir)

        # Test that it has required methods for the BaseDataset interface
        assert hasattr(dataset, "get_graph_data")
        assert hasattr(dataset, "get_node_mapping")
        assert hasattr(dataset, "get_dataframes")
        assert hasattr(dataset, "add_estimator_results")
        assert hasattr(dataset, "get_estimators")

        # Test that supported datasets list contains expected entries
        assert "cdk8" in dataset.SUPPORTED_DATASETS

    def test_benchmark_metadata_access(self, temp_cache_dir):
        """Test access to benchmark metadata."""
        dataset = FEPBenchmarkDataset(cache_dir=temp_cache_dir)

        # Test that cache directory is properly initialized
        assert hasattr(dataset, "cache_dir")
        assert os.path.exists(dataset.cache_dir)

        # Test that internal caches are initialized
        assert hasattr(dataset, "_edge_cache")
        assert hasattr(dataset, "_node_cache")

    def test_invalid_benchmark_name(self, temp_cache_dir):
        """Test error handling for invalid benchmark names."""
        dataset = FEPBenchmarkDataset(cache_dir=temp_cache_dir)

        # Test that calling get_dataset with invalid name raises error
        with pytest.raises((ValueError, FileNotFoundError, KeyError)):
            dataset.get_dataset("nonexistent_benchmark", "5ns")


class TestDatasetIntegration:
    """
    Integration tests for dataset classes.

    These tests verify that all dataset classes work together
    and provide consistent interfaces for use with NodeModel.
    """

    def test_all_datasets_implement_interface(self):
        """Test that all dataset classes implement the BaseDataset interface."""
        # Create instances of all dataset types using only synthetic data
        datasets = [SyntheticFEPDataset(add_noise=False, random_seed=42)]

        for dataset in datasets:
            # Test that all required methods are implemented
            assert hasattr(dataset, "get_graph_data")
            assert hasattr(dataset, "get_node_mapping")
            assert hasattr(dataset, "get_dataframes")

            # Test that methods return expected types
            graph_data = dataset.get_graph_data()
            assert isinstance(graph_data, dict)

            node2idx, idx2node = dataset.get_node_mapping()
            assert isinstance(node2idx, dict)
            assert isinstance(idx2node, dict)

            edge_data, node_data = dataset.get_dataframes()
            assert isinstance(edge_data, pd.DataFrame)
            assert isinstance(node_data, pd.DataFrame)

    def test_graph_data_consistency(self, mock_dataset):
        """Test that graph data is consistent across calls."""
        # Test multiple calls return same data
        graph_data1 = mock_dataset.get_graph_data()
        graph_data2 = mock_dataset.get_graph_data()

        assert graph_data1["N"] == graph_data2["N"]
        assert graph_data1["M"] == graph_data2["M"]
        torch.testing.assert_close(graph_data1["src"], graph_data2["src"])
        torch.testing.assert_close(graph_data1["dst"], graph_data2["dst"])
        torch.testing.assert_close(graph_data1["FEP"], graph_data2["FEP"])

    def test_node_mapping_consistency(self, mock_dataset):
        """Test that node mappings are consistent with graph data."""
        graph_data = mock_dataset.get_graph_data()
        node2idx, idx2node = mock_dataset.get_node_mapping()

        # Check that all indices in graph data are valid
        max_idx = max(max(graph_data["src"]), max(graph_data["dst"]))
        assert max_idx < graph_data["N"]
        assert max_idx < len(idx2node)

        # Check that all indices are covered
        all_indices = set(graph_data["src"].tolist()) | set(graph_data["dst"].tolist())
        mapped_indices = set(idx2node.keys())

        assert all_indices.issubset(
            mapped_indices
        ), f"Graph uses indices {all_indices} but mapping only covers {mapped_indices}"

    def test_dataset_with_node_model_compatibility(self, mock_dataset):
        """Test that dataset output is compatible with NodeModel requirements."""
        graph_data = mock_dataset.get_graph_data()

        # Check all required keys for NodeModel
        required_keys = {"N", "M", "src", "dst", "FEP", "CCC"}
        missing_keys = required_keys - set(graph_data.keys())
        assert not missing_keys, f"Missing required keys for NodeModel: {missing_keys}"

        # Check data types
        assert isinstance(graph_data["N"], int)
        assert isinstance(graph_data["M"], int)
        assert isinstance(graph_data["src"], torch.Tensor)
        assert isinstance(graph_data["dst"], torch.Tensor)
        assert isinstance(graph_data["FEP"], torch.Tensor)

        # Check tensor dtypes
        assert graph_data["src"].dtype == torch.long
        assert graph_data["dst"].dtype == torch.long
        assert graph_data["FEP"].dtype == torch.float32

        # Check no NaN values in critical tensors
        assert not torch.any(torch.isnan(graph_data["FEP"]))

        # Check tensor shapes are consistent
        assert (
            graph_data["src"].shape
            == graph_data["dst"].shape
            == graph_data["FEP"].shape
        )
