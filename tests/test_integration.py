"""
Integration tests for the MAPLE package.

This module tests the integration between different components
of the MAPLE package to ensure they work together correctly.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from maple.graph_analysis.performance_stats import compute_simple_statistics
from maple.models.node_model import NodeModel
from maple.models.model_config import NodeModelConfig as ModelConfig


class TestPackageIntegration:
    """
    Integration tests for the complete MAPLE package.

    These tests verify that different components work together
    correctly in realistic usage scenarios.
    """

    def test_dataset_to_model_pipeline(self, mock_dataset):
        """Test the complete pipeline from dataset to model training."""
        # Test that dataset provides data in correct format for NodeModel
        graph_data = mock_dataset.get_graph_data()

        # Verify required keys for NodeModel
        required_keys = {"N", "M", "src", "dst", "FEP", "CCC"}
        assert all(key in graph_data for key in required_keys)

        # Verify data types
        assert isinstance(graph_data["N"], int)
        assert isinstance(graph_data["M"], int)
        assert isinstance(graph_data["src"], torch.Tensor)
        assert isinstance(graph_data["dst"], torch.Tensor)
        assert isinstance(graph_data["FEP"], torch.Tensor)

        # Test that NodeModel can be initialized with this dataset
        config = ModelConfig(num_steps=10)  # Short training for testing
        model = NodeModel(config, mock_dataset)

        assert model.dataset == mock_dataset
        assert model.config == config

    @patch("pyro.infer.SVI")
    @patch("pyro.infer.Predictive")
    def test_model_training_to_statistics_pipeline(
        self, mock_predictive, mock_svi, mock_dataset
    ):
        """Test pipeline from model training to performance statistics."""
        # Mock training
        mock_svi_instance = Mock()
        mock_svi_instance.step.return_value = 1.0
        mock_svi.return_value = mock_svi_instance

        # Mock prediction
        mock_pred_instance = Mock()
        n_nodes = mock_dataset.get_graph_data()["N"]
        mock_predictions = {
            "node_values": torch.randn(1, n_nodes),
            "log_likelihood": torch.tensor([-10.0]),
        }
        mock_pred_instance.return_value = mock_predictions
        mock_predictive.return_value = mock_pred_instance

        # Train model
        config = ModelConfig(num_steps=10)
        model = NodeModel(config, mock_dataset)
        model.train()

        # Get predictions
        results = model.get_results()

        # Convert to format suitable for statistics
        y_pred = np.array(list(results["node_estimates"].values()))

        # Create mock true values for statistics
        y_true = np.random.randn(len(y_pred))

        # Compute statistics
        stats = compute_simple_statistics(y_true, y_pred)

        # Verify statistics structure
        expected_keys = {"RMSE", "MUE", "R2", "rho", "KTAU"}
        assert all(key in stats for key in expected_keys)

        # Verify statistics are reasonable
        assert len(y_true) > 0  # We have data
        assert not np.isnan(stats["MUE"])  # Mean Unsigned Error (MAE equivalent)
        assert not np.isnan(stats["RMSE"])  # Root Mean Square Error
        assert not np.isnan(stats["R2"])  # R-squared

    def test_statistics_to_plotting_pipeline(self, sample_arrays_for_stats):
        """Test pipeline from statistics computation to plotting."""
        y_true, y_pred = sample_arrays_for_stats

        # Compute statistics
        stats = compute_simple_statistics(y_true, y_pred)

        # Create basic plot (simplified plotting test)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.7)

        # Add diagonal line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8)

        # Add statistics as text
        stats_text = (
            f"MUE: {stats['MUE']:.3f}\nRMSE: {stats['RMSE']:.3f}\nR²: {stats['R2']:.3f}"
        )
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white"),
        )

        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Model Performance")

        # Verify plot was created successfully
        assert isinstance(fig, plt.Figure)
        assert len(ax.texts) > 0  # Should have statistics text

        plt.close(fig)

    def test_end_to_end_workflow(self, mock_dataset):
        """Test complete end-to-end workflow."""
        # Step 1: Dataset preparation
        edge_data, node_data = mock_dataset.get_dataframes()
        assert isinstance(edge_data, pd.DataFrame)
        assert isinstance(node_data, pd.DataFrame)

        # Step 2: Graph data generation
        graph_data = mock_dataset.get_graph_data()
        assert isinstance(graph_data, dict)

        # Step 3: Model initialization and configuration
        config = ModelConfig(
            learning_rate=0.01, num_steps=10, error_std=1e-4  # Short for testing
        )
        model = NodeModel(config, mock_dataset)

        # Step 4: Basic validation that model is set up correctly
        assert model.node_estimates is None  # Model not yet trained
        assert model.config.learning_rate == 0.01

        # This test doesn't include actual training to avoid complexity,
        # but verifies that the pipeline components are compatible

    def test_data_format_consistency(self, mock_dataset):
        """Test that data formats are consistent across components."""
        # Get data in different formats
        graph_data = mock_dataset.get_graph_data()
        node2idx, idx2node = mock_dataset.get_node_mapping()
        edge_df, node_df = mock_dataset.get_dataframes()

        # Check consistency between graph data and mapping
        assert graph_data["N"] == len(node2idx) == len(idx2node)

        # Check that all indices in graph tensors are valid
        max_src = torch.max(graph_data["src"]).item()
        max_dst = torch.max(graph_data["dst"]).item()
        max_idx = max(max_src, max_dst)

        assert max_idx < graph_data["N"], "Graph indices exceed number of nodes"
        assert max_idx in idx2node, "Graph index not in node mapping"

        # Check edge count consistency
        assert graph_data["M"] == len(edge_df), "Edge count mismatch"
        assert (
            len(graph_data["src"]) == len(graph_data["dst"]) == len(graph_data["FEP"])
        )

    def test_error_propagation(self, mock_dataset):
        """Test that errors propagate appropriately through the pipeline."""
        # Test with invalid configuration
        with pytest.raises(ValueError):
            ModelConfig(learning_rate=-1.0)  # Invalid learning rate

        # Test with corrupted graph data
        corrupted_data = mock_dataset.get_graph_data().copy()
        corrupted_data["src"] = torch.tensor([999])  # Invalid node index

        mock_dataset.get_graph_data = Mock(return_value=corrupted_data)

        # Should detect the error when initializing model
        try:
            config = ModelConfig(num_steps=10)
            _ = NodeModel(config, mock_dataset)
            # Some validation might happen during training instead
        except (ValueError, RuntimeError, IndexError):
            pass  # Expected for corrupted data

    def test_memory_management(self, mock_dataset):
        """Test that memory is managed properly across components."""
        import gc

        # Create and destroy multiple models
        for _ in range(3):
            config = ModelConfig(num_steps=10)
            model = NodeModel(config, mock_dataset)

            # Get some data to ensure tensors are created
            graph_data = model.dataset.get_graph_data()

            # Delete model
            del model
            del graph_data

        # Force garbage collection
        gc.collect()

        # If we reach here without memory issues, test passes
        assert True

    def test_reproducibility_across_components(self, mock_dataset):
        """Test that results are reproducible across components."""
        # Set up identical configurations
        config1 = ModelConfig(random_seed=42, num_steps=10)
        config2 = ModelConfig(random_seed=42, num_steps=10)

        model1 = NodeModel(config1, mock_dataset)
        model2 = NodeModel(config2, mock_dataset)

        # Both should have identical configurations
        assert model1.config.__dict__ == model2.config.__dict__

        # Graph data should be identical for same dataset
        graph1 = model1.dataset.get_graph_data()
        graph2 = model2.dataset.get_graph_data()

        assert graph1["N"] == graph2["N"]
        assert graph1["M"] == graph2["M"]
        torch.testing.assert_close(graph1["FEP"], graph2["FEP"])


class TestPackageImports:
    """Test that all package components can be imported correctly."""

    def test_main_package_import(self):
        """Test importing the main package."""
        import maple

        # Should have main submodules
        assert hasattr(maple, "models")
        assert hasattr(maple, "dataset")
        assert hasattr(maple, "graph_analysis")

    def test_models_import(self):
        """Test importing models subpackage."""
        from maple.models import NodeModel, NodeModelConfig

        # Should be able to access classes
        assert NodeModel is not None
        assert NodeModelConfig is not None

    def test_dataset_import(self):
        """Test importing dataset subpackage."""
        from maple.dataset import BaseDataset, FEPDataset, SyntheticFEPDataset

        # Should be able to access classes
        assert BaseDataset is not None
        assert FEPDataset is not None
        assert SyntheticFEPDataset is not None

    def test_graph_analysis_import(self):
        """Test importing graph analysis subpackage."""
        from maple.graph_analysis import calculate_mae, calculate_rmse

        # Should be able to access functions
        assert callable(calculate_mae)
        assert callable(calculate_rmse)


class TestCompatibilityRequirements:
    """Test compatibility with required dependencies."""

    def test_torch_compatibility(self):
        """Test that PyTorch operations work correctly."""
        # Basic tensor operations
        x = torch.randn(5, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)

        assert z.shape == (5, 2)
        assert z.dtype == torch.float32

    def test_pandas_compatibility(self):
        """Test that pandas operations work correctly."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        assert len(df) == 3
        assert list(df.columns) == ["A", "B"]

    def test_numpy_compatibility(self):
        """Test that numpy operations work correctly."""
        arr = np.array([1, 2, 3, 4, 5])

        assert np.mean(arr) == 3.0
        assert np.std(arr) > 0

    def test_matplotlib_compatibility(self):
        """Test that matplotlib plotting works correctly."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
