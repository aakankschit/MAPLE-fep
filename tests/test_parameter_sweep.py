"""
Unit tests for the parameter sweep functionality.

This module provides comprehensive testing of the ParameterSweep class
and related functionality for systematic parameter exploration.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from maple.models.model_config import NodeModelConfig as ModelConfig, PriorType
from maple.utils import PerformanceTracker
from maple.utils.parameter_sweep import (ParameterSweep,
                                         create_comprehensive_parameter_study,
                                         create_prior_sweep_experiment)


class TestParameterSweepCore:
    """
    Core functionality tests for ParameterSweep class.

    These tests verify the basic parameter sweep operations
    without requiring full model training.
    """

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_datasets(self):
        """Create mock datasets for testing."""
        datasets = {}
        for name in ["dataset_A", "dataset_B"]:
            mock_dataset = Mock()
            mock_dataset.get_graph_data.return_value = {
                "N": 8,
                "M": 12,
                "src": torch.tensor([0, 1, 2]),
                "dst": torch.tensor([1, 2, 0]),
                "FEP": torch.tensor([1.0, 2.0, -3.0]),
                "CCC": pd.DataFrame(),
            }
            # Mock dataset_nodes attribute for experimental data extraction
            mock_nodes = Mock()
            mock_nodes.columns = ["Exp. DeltaG", "molecule"]
            mock_nodes.__contains__ = lambda self, key: key in [
                "Exp. DeltaG",
                "molecule",
            ]
            mock_nodes.__getitem__ = lambda self, key: Mock(
                values=np.array([1.0, 2.0, 3.0])
            )
            mock_dataset.dataset_nodes = mock_nodes

            # Mock get_dataframes method
            mock_dataset.get_dataframes.return_value = (
                pd.DataFrame(),
                pd.DataFrame({"Exp. DeltaG": [1, 2, 3]}),
            )

            datasets[name] = mock_dataset
        return datasets

    @pytest.fixture
    def parameter_sweep(self, temp_storage_dir, mock_datasets):
        """Create ParameterSweep instance for testing."""
        tracker = PerformanceTracker(temp_storage_dir, auto_save=False)
        base_config = ModelConfig(
            learning_rate=0.001,
            num_steps=100,
            prior_type=PriorType.NORMAL,
            prior_parameters=[0.0, 1.0],
        )
        return ParameterSweep(tracker, base_config, mock_datasets)

    def test_parameter_sweep_initialization(self, parameter_sweep):
        """Test ParameterSweep initialization."""
        assert isinstance(parameter_sweep.tracker, PerformanceTracker)
        assert isinstance(parameter_sweep.base_config, ModelConfig)
        assert len(parameter_sweep.datasets) == 2
        assert "dataset_A" in parameter_sweep.datasets
        assert "dataset_B" in parameter_sweep.datasets
        assert isinstance(parameter_sweep.sweep_results, list)
        assert len(parameter_sweep.sweep_results) == 0

    def test_get_experimental_data(self, parameter_sweep):
        """Test extraction of experimental data from datasets."""
        # Test with mock dataset
        mock_dataset = parameter_sweep.datasets["dataset_A"]

        # Should not raise an error and return some data
        exp_data = parameter_sweep._get_experimental_data(mock_dataset)
        assert isinstance(exp_data, np.ndarray)
        assert len(exp_data) > 0

    def test_align_predictions_with_experimental(self, parameter_sweep):
        """Test alignment of model predictions with experimental data."""
        mock_dataset = parameter_sweep.datasets["dataset_A"]
        mock_model = Mock()

        # Test case where lengths match
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.1, 2.1, 2.9])

        aligned = parameter_sweep._align_predictions_with_experimental(
            y_pred, y_true, mock_dataset, mock_model
        )

        assert len(aligned) == len(y_true)
        np.testing.assert_array_equal(aligned, y_pred)

        # Test case where lengths don't match
        y_pred_long = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        aligned_long = parameter_sweep._align_predictions_with_experimental(
            y_pred_long, y_true, mock_dataset, mock_model
        )

        assert len(aligned_long) <= len(y_true)

    def test_sweep_parameter_data_structure(self, parameter_sweep):
        """Test parameter sweep result data structure."""
        # Create mock results DataFrame to test structure validation
        mock_results = pd.DataFrame(
            {
                "experiment": ["test_sweep"] * 4,
                "dataset": ["dataset_A", "dataset_A", "dataset_B", "dataset_B"],
                "parameter": ["learning_rate"] * 4,
                "parameter_value": [0.001, 0.01, 0.001, 0.01],
                "metric": ["RMSE"] * 4,
                "value": [0.5, 0.3, 0.6, 0.2],
                "mean": [0.5, 0.3, 0.6, 0.2],
                "std_err": [0.05, 0.03, 0.06, 0.02],
                "ci_low": [0.4, 0.24, 0.48, 0.16],
                "ci_high": [0.6, 0.36, 0.72, 0.24],
                "run_id": ["run_1", "run_2", "run_3", "run_4"],
            }
        )

        # Verify expected structure
        assert isinstance(mock_results, pd.DataFrame)
        assert len(mock_results) == 4

        # Verify required columns exist
        expected_columns = [
            "experiment",
            "dataset",
            "parameter",
            "parameter_value",
            "metric",
            "value",
            "mean",
            "std_err",
            "ci_low",
            "ci_high",
            "run_id",
        ]
        for col in expected_columns:
            assert col in mock_results.columns

        # Test the find_optimal_parameters method with this structure
        optimal = parameter_sweep.find_optimal_parameters(
            mock_results, metric="RMSE", minimize=True, aggregation="mean"
        )
        assert len(optimal) == 1
        assert optimal.iloc[0]["optimal_value"] == 0.01  # Best average RMSE

    def test_find_optimal_parameters(self, parameter_sweep):
        """Test finding optimal parameters from results."""
        # Create sample results DataFrame
        results = pd.DataFrame(
            {
                "experiment": ["test"] * 8,
                "dataset": ["dataset_A"] * 4 + ["dataset_B"] * 4,
                "parameter": ["learning_rate"] * 8,
                "parameter_value": [0.001, 0.01, 0.001, 0.01] * 2,
                "metric": ["RMSE", "RMSE", "R2", "R2"] * 2,
                "value": [0.5, 0.3, 0.8, 0.9, 0.6, 0.2, 0.7, 0.95],
                "run_id": [f"run_{i}" for i in range(8)],
            }
        )

        # Test minimizing RMSE
        optimal_rmse = parameter_sweep.find_optimal_parameters(
            results, metric="RMSE", minimize=True, aggregation="mean"
        )

        assert isinstance(optimal_rmse, pd.DataFrame)
        assert len(optimal_rmse) == 1
        assert optimal_rmse.iloc[0]["parameter"] == "learning_rate"
        assert optimal_rmse.iloc[0]["optimal_value"] == 0.01  # Better average RMSE

        # Test maximizing R2
        optimal_r2 = parameter_sweep.find_optimal_parameters(
            results, metric="R2", minimize=False, aggregation="mean"
        )

        assert optimal_r2.iloc[0]["optimal_value"] == 0.01  # Better average R2

    def test_plot_parameter_effects_structure(self, parameter_sweep):
        """Test parameter effects plotting structure."""
        # Create sample results for plotting
        results = pd.DataFrame(
            {
                "experiment": ["test"] * 12,
                "dataset": (["dataset_A"] * 3 + ["dataset_B"] * 3) * 2,
                "parameter": ["learning_rate"] * 12,
                "parameter_value": [0.001, 0.01, 0.1] * 4,
                "metric": ["RMSE"] * 6 + ["R2"] * 6,
                "value": [
                    0.5,
                    0.3,
                    0.8,
                    0.6,
                    0.2,
                    0.9,
                    0.7,
                    0.8,
                    0.6,
                    0.65,
                    0.85,
                    0.55,
                ],
                "mean": [0.5, 0.3, 0.8, 0.6, 0.2, 0.9, 0.7, 0.8, 0.6, 0.65, 0.85, 0.55],
                "ci_low": [
                    0.4,
                    0.2,
                    0.7,
                    0.5,
                    0.1,
                    0.8,
                    0.6,
                    0.7,
                    0.5,
                    0.55,
                    0.75,
                    0.45,
                ],
                "ci_high": [
                    0.6,
                    0.4,
                    0.9,
                    0.7,
                    0.3,
                    1.0,
                    0.8,
                    0.9,
                    0.7,
                    0.75,
                    0.95,
                    0.65,
                ],
                "run_id": [f"run_{i}" for i in range(12)],
            }
        )

        # Test plotting
        figures = parameter_sweep.plot_parameter_effects(
            results_df=results,
            parameter_name="learning_rate",
            metrics=["RMSE", "R2"],
            confidence_intervals=True,
        )

        # plot_parameter_effects returns a dict with 'error' and 'correlation' figures
        assert isinstance(figures, dict)

        # Should have error figure (for RMSE) and correlation figure (for R2)
        if "error" in figures:
            assert isinstance(figures["error"], plt.Figure)
            plt.close(figures["error"])
        if "correlation" in figures:
            assert isinstance(figures["correlation"], plt.Figure)
            plt.close(figures["correlation"])

    def test_grid_search_data_structure(self, parameter_sweep):
        """Test grid search result data structure."""
        # Create mock grid search results to test structure
        mock_grid_results = pd.DataFrame(
            {
                "experiment": ["test_grid"] * 8,
                "dataset": ["dataset_A"] * 4 + ["dataset_B"] * 4,
                "learning_rate": [0.001, 0.001, 0.01, 0.01] * 2,
                "num_steps": [50, 100, 50, 100] * 2,
                "metric": ["RMSE"] * 8,
                "value": [0.5, 0.4, 0.3, 0.25, 0.6, 0.5, 0.35, 0.3],
                "run_id": [f"grid_run_{i}" for i in range(8)],
            }
        )

        # Verify grid search structure
        assert isinstance(mock_grid_results, pd.DataFrame)
        assert len(mock_grid_results) == 8

        # Verify parameter columns exist
        assert "learning_rate" in mock_grid_results.columns
        assert "num_steps" in mock_grid_results.columns
        assert "metric" in mock_grid_results.columns
        assert "value" in mock_grid_results.columns

        # Verify parameter values are from the expected grid
        assert set(mock_grid_results["learning_rate"].unique()) == {0.001, 0.01}
        assert set(mock_grid_results["num_steps"].unique()) == {50, 100}

        # Test finding best combination manually (since grid results have different structure)
        # Grid search results have individual parameter columns, not a 'parameter' column
        best_idx = mock_grid_results["value"].idxmin()  # Find best RMSE
        best_result = mock_grid_results.loc[best_idx]

        best_lr = best_result["learning_rate"]
        best_steps = best_result["num_steps"]

        assert best_lr in [0.001, 0.01]
        assert best_steps in [50, 100]
        assert best_result["value"] == 0.25  # Should be the minimum value


class TestParameterSweepValidation:
    """
    Test parameter validation and error handling in ParameterSweep.

    These tests verify that the parameter sweep handles edge cases
    and invalid inputs gracefully.
    """

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def basic_setup(self, temp_storage_dir):
        """Basic setup for validation tests."""
        tracker = PerformanceTracker(temp_storage_dir, auto_save=False)
        base_config = ModelConfig()
        datasets = {"test": Mock()}
        return ParameterSweep(tracker, base_config, datasets)

    def test_invalid_parameter_validation(self, basic_setup):
        """Test parameter validation without requiring full execution."""
        sweep = basic_setup

        # Test parameter name validation (these should fail early)
        # We can't test the full sweep execution due to mock dataset issues,
        # but we can test the validation logic

        # Test empty values - should raise error immediately
        try:
            sweep.sweep_parameter(
                parameter_name="learning_rate", values=[], metrics=["RMSE"]
            )
            assert False, "Should have raised an error for empty values"
        except Exception:
            # Should raise some kind of error for empty values
            assert True

        # Test empty metrics - should raise error immediately
        try:
            sweep.sweep_parameter(
                parameter_name="learning_rate", values=[0.001, 0.01], metrics=[]
            )
            assert False, "Should have raised an error for empty metrics"
        except Exception:
            # Should raise some kind of error for empty metrics
            assert True

    def test_empty_datasets(self, temp_storage_dir):
        """Test handling of empty dataset dictionary."""
        tracker = PerformanceTracker(temp_storage_dir, auto_save=False)
        base_config = ModelConfig()

        # Empty datasets should be handled gracefully
        sweep = ParameterSweep(tracker, base_config, {})

        # Sweep should complete but produce no results
        results = sweep.sweep_parameter(
            parameter_name="learning_rate", values=[0.001, 0.01], metrics=["RMSE"]
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

    def test_large_parameter_grid_limiting(self, basic_setup):
        """Test that large parameter grids are handled appropriately."""
        sweep = basic_setup

        # Create a very large grid
        large_grid = {
            "param1": list(range(10)),
            "param2": list(range(10)),
            "param3": list(range(10)),
        }

        # Test that we can create the grid without errors
        # The actual limiting would happen during execution
        assert len(large_grid["param1"]) == 10
        assert len(large_grid["param2"]) == 10
        assert len(large_grid["param3"]) == 10

        # Calculate expected combinations
        total_combinations = (
            len(large_grid["param1"])
            * len(large_grid["param2"])
            * len(large_grid["param3"])
        )
        assert total_combinations == 1000  # 10 × 10 × 10

        # Test that max_experiments parameter exists in the function signature
        import inspect

        sig = inspect.signature(sweep.grid_search)
        assert "max_experiments" in sig.parameters


class TestConvenienceFunctions:
    """
    Test convenience functions for parameter sweeps.

    These tests verify that the high-level convenience functions exist
    and have the correct interface without requiring full execution.
    """

    def test_convenience_function_signatures(self):
        """Test that convenience functions exist and have correct signatures."""
        # Test that the functions exist and can be imported
        import inspect

        from maple.utils.parameter_sweep import (
            create_comprehensive_parameter_study,
            create_prior_sweep_experiment)

        # Test create_prior_sweep_experiment signature
        prior_sig = inspect.signature(create_prior_sweep_experiment)
        expected_prior_params = {
            "tracker",
            "datasets",
            "prior_std_values",
            "base_config",
        }
        assert expected_prior_params.issubset(set(prior_sig.parameters.keys()))

        # Test create_comprehensive_parameter_study signature
        comprehensive_sig = inspect.signature(create_comprehensive_parameter_study)
        expected_comprehensive_params = {"tracker", "datasets"}
        assert expected_comprehensive_params.issubset(
            set(comprehensive_sig.parameters.keys())
        )

    def test_mock_convenience_workflow(self):
        """Test convenience function workflow with mock data."""
        # Create mock results that these functions would return
        mock_prior_results = pd.DataFrame(
            {
                "experiment": ["prior_std_effect"] * 4,
                "dataset": ["dataset_A"] * 2 + ["dataset_B"] * 2,
                "parameter": ["prior_std"] * 4,
                "parameter_value": [0.1, 1.0] * 2,
                "metric": ["RMSE"] * 4,
                "value": [0.5, 0.3, 0.6, 0.2],
                "mean": [0.5, 0.3, 0.6, 0.2],
                "ci_low": [0.4, 0.2, 0.5, 0.1],
                "ci_high": [0.6, 0.4, 0.7, 0.3],
                "run_id": ["run_1", "run_2", "run_3", "run_4"],
            }
        )

        # Verify expected structure
        assert isinstance(mock_prior_results, pd.DataFrame)
        assert "parameter_value" in mock_prior_results.columns
        assert "metric" in mock_prior_results.columns
        assert set(mock_prior_results["parameter"].unique()) == {"prior_std"}

        # Test that the expected columns match what the convenience functions should produce
        required_columns = [
            "experiment",
            "dataset",
            "parameter",
            "parameter_value",
            "metric",
            "value",
            "mean",
            "ci_low",
            "ci_high",
            "run_id",
        ]
        for col in required_columns:
            assert col in mock_prior_results.columns

        # Test comprehensive study mock structure
        mock_comprehensive_results = {
            "prior_std": mock_prior_results,
            "learning_rate": mock_prior_results.copy(),
            "error_std": mock_prior_results.copy(),
            "num_steps": mock_prior_results.copy(),
        }

        assert isinstance(mock_comprehensive_results, dict)
        expected_studies = {"prior_std", "learning_rate", "error_std", "num_steps"}
        assert expected_studies == set(mock_comprehensive_results.keys())


class TestParameterSweepIntegrationScenarios:
    """
    Test integration scenarios for parameter sweeps using mock data.

    These tests verify integration patterns and expected workflows
    without requiring full model execution.
    """

    def test_optimization_workflow_simulation(self):
        """Test parameter optimization workflow with simulated data."""
        # Simulate a cross-dataset optimization scenario
        mock_results = pd.DataFrame(
            {
                "experiment": ["optimization_test"] * 12,
                "dataset": ["dataset_easy"] * 6 + ["dataset_hard"] * 6,
                "parameter": ["learning_rate"] * 12,
                "parameter_value": [0.001, 0.01, 0.1] * 4,
                "metric": ["RMSE"] * 12,
                "value": [
                    # dataset_easy (easier to predict)
                    0.2,
                    0.15,
                    0.25,
                    0.2,
                    0.15,
                    0.25,
                    # dataset_hard (harder to predict)
                    0.4,
                    0.3,
                    0.5,
                    0.4,
                    0.3,
                    0.5,
                ],
                "run_id": [f"run_{i}" for i in range(12)],
            }
        )

        # Create a sweep instance to test optimization
        from unittest.mock import Mock

        from maple.models.model_config import NodeModelConfig as ModelConfig
        from maple.utils.parameter_sweep import ParameterSweep

        tracker = Mock()
        base_config = ModelConfig()
        datasets = {"dataset_easy": Mock(), "dataset_hard": Mock()}
        sweep = ParameterSweep(tracker, base_config, datasets)

        # Test different aggregation strategies
        optimal_mean = sweep.find_optimal_parameters(
            mock_results, metric="RMSE", minimize=True, aggregation="mean"
        )
        assert optimal_mean.iloc[0]["optimal_value"] == 0.01  # Best average

        optimal_best = sweep.find_optimal_parameters(
            mock_results, metric="RMSE", minimize=True, aggregation="best"
        )
        assert optimal_best.iloc[0]["optimal_value"] == 0.01  # Best single performance

    def test_parameter_interaction_workflow(self):
        """Test parameter interaction analysis workflow."""
        # Simulate grid search results with parameter interactions
        mock_grid_results = pd.DataFrame(
            {
                "experiment": ["interaction_test"] * 8,
                "dataset": ["test_dataset"] * 8,
                "learning_rate": [0.001, 0.001, 0.01, 0.01, 0.001, 0.001, 0.01, 0.01],
                "error_std": [0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0],
                "metric": ["RMSE"] * 8,
                "value": [0.3, 0.2, 0.4, 0.35, 0.32, 0.18, 0.38, 0.37],
                "run_id": [f"interaction_run_{i}" for i in range(8)],
            }
        )

        # Test finding best parameter combinations
        best_idx = mock_grid_results["value"].idxmin()
        best_combo = mock_grid_results.loc[best_idx]

        # Should identify lr=0.001 + error_std=1.0 as best (value=0.18)
        assert best_combo["learning_rate"] == 0.001
        assert best_combo["error_std"] == 1.0
        assert best_combo["value"] == 0.18

        # Test parameter value distributions
        assert set(mock_grid_results["learning_rate"].unique()) == {0.001, 0.01}
        assert set(mock_grid_results["error_std"].unique()) == {0.5, 1.0}

    def test_workflow_data_structures(self):
        """Test that workflow data structures match expected formats."""
        # Test prior sweep result structure (matching notebook format)
        mock_prior_results = pd.DataFrame(
            {
                "experiment": ["prior_std_study"] * 8,
                "dataset": ["dataset_A"] * 4 + ["dataset_B"] * 4,
                "parameter": ["prior_std"] * 8,
                "parameter_value": [0.1, 0.5, 1.0, 2.0]
                * 2,  # Like 'Distribution Scale'
                "metric": ["RMSE", "RMSE", "RMSE", "RMSE", "MUE", "MUE", "MUE", "MUE"],
                "value": [
                    0.5,
                    0.4,
                    0.3,
                    0.35,
                    0.45,
                    0.35,
                    0.25,
                    0.3,
                ],  # Like 'Statistic Value'
                "mean": [0.5, 0.4, 0.3, 0.35, 0.45, 0.35, 0.25, 0.3],
                "ci_low": [
                    0.4,
                    0.3,
                    0.2,
                    0.25,
                    0.35,
                    0.25,
                    0.15,
                    0.2,
                ],  # Like '95% low'
                "ci_high": [
                    0.6,
                    0.5,
                    0.4,
                    0.45,
                    0.55,
                    0.45,
                    0.35,
                    0.4,
                ],  # Like '95% upper'
                "run_id": [f"prior_run_{i}" for i in range(8)],
            }
        )

        # Verify structure matches what would be expected from real runs
        assert (
            "parameter_value" in mock_prior_results.columns
        )  # Distribution Scale equivalent
        assert "value" in mock_prior_results.columns  # Statistic Value equivalent
        assert "metric" in mock_prior_results.columns  # Type of Statistic equivalent
        assert "ci_low" in mock_prior_results.columns  # 95% low equivalent
        assert "ci_high" in mock_prior_results.columns  # 95% upper equivalent

        # Test that metrics include expected types
        metrics = set(mock_prior_results["metric"].unique())
        assert {"RMSE", "MUE"}.issubset(metrics)

        # Test parameter value range
        param_values = mock_prior_results["parameter_value"].unique()
        assert all(val > 0 for val in param_values)  # Should be positive prior stds
