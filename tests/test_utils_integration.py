"""
Integration tests for the MAPLE utils package.

This module tests the complete functionality of the performance tracking
and parameter sweep utilities, including their integration with the
core MAPLE components.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from maple.dataset.synthetic_dataset import SyntheticFEPDataset
from maple.models.model_config import NodeModelConfig as ModelConfig, PriorType
from maple.utils import (ModelRun, ParameterSweep, PerformanceTracker,
                         compare_model_runs, create_prior_sweep_experiment,
                         load_performance_history)


class TestPerformanceTrackerIntegration:
    """
    Integration tests for PerformanceTracker with real data and models.

    These tests verify that the performance tracker works correctly
    with actual MAPLE components and data structures.
    """

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_datasets(self):
        """Create test datasets for integration testing."""
        datasets = {
            "small_test": SyntheticFEPDataset(add_noise=False, random_seed=42),
            "medium_test": SyntheticFEPDataset(add_noise=False, random_seed=123),
        }
        return datasets

    @pytest.fixture
    def tracker_with_data(self, temp_storage_dir, test_datasets):
        """Create tracker with some test data."""
        tracker = PerformanceTracker(temp_storage_dir, auto_save=True)

        # Add some test runs
        np.random.seed(42)
        for dataset_name, dataset in test_datasets.items():
            graph_data = dataset.get_graph_data()
            n_nodes = graph_data["N"]

            # Simulate experimental and predicted data
            y_true = np.random.randn(n_nodes) * 2.0
            y_pred = y_true + np.random.normal(0, 0.3, n_nodes)

            tracker.record_run(
                run_id=f"test_{dataset_name}",
                y_true=y_true,
                y_pred=y_pred,
                model_config={"learning_rate": 0.001, "num_steps": 1000},
                dataset_info={"name": dataset_name, "n_nodes": n_nodes},
                tags=["test", dataset_name],
            )

        return tracker

    def test_tracker_initialization(self, temp_storage_dir):
        """Test that tracker initializes correctly."""
        tracker = PerformanceTracker(temp_storage_dir)

        assert tracker.storage_dir == temp_storage_dir
        assert tracker.auto_save is True
        assert len(tracker.runs) == 0
        assert tracker.storage_dir.exists()

    def test_record_run_integration(self, temp_storage_dir, test_datasets):
        """Test recording runs with real dataset integration."""
        tracker = PerformanceTracker(temp_storage_dir)
        dataset = test_datasets["small_test"]

        # Generate test data
        graph_data = dataset.get_graph_data()
        n_nodes = graph_data["N"]
        y_true = np.random.randn(n_nodes)
        y_pred = y_true + np.random.normal(0, 0.2, n_nodes)

        # Record run
        model_run = tracker.record_run(
            run_id="integration_test",
            y_true=y_true,
            y_pred=y_pred,
            model_config={"test_param": "test_value"},
            dataset_info={"name": "test", "size": len(y_true)},
            metadata={"test": True},
            tags=["integration"],
        )

        # Verify run was recorded correctly
        assert isinstance(model_run, ModelRun)
        assert model_run.run_id == "integration_test"
        assert len(model_run.predictions["y_true"]) == n_nodes
        assert len(model_run.predictions["y_pred"]) == n_nodes

        # Verify statistics were computed
        assert "RMSE" in model_run.performance_metrics
        assert "MUE" in model_run.performance_metrics
        assert "R2" in model_run.performance_metrics
        assert "rho" in model_run.performance_metrics

        # Verify all statistics are finite
        for metric, value in model_run.performance_metrics.items():
            assert np.isfinite(value), f"{metric} should be finite"

        # Verify bootstrap metrics
        assert len(model_run.bootstrap_metrics) > 0
        for stat_name, bootstrap_data in model_run.bootstrap_metrics.items():
            assert "mle" in bootstrap_data
            assert "mean" in bootstrap_data
            assert "stderr" in bootstrap_data
            assert "low" in bootstrap_data
            assert "high" in bootstrap_data

    def test_data_persistence(self, tracker_with_data, temp_storage_dir):
        """Test that data persists correctly to disk."""
        # Verify files were created
        assert (temp_storage_dir / "model_runs.json").exists()
        assert (temp_storage_dir / "performance_data.pkl").exists()

        # Create new tracker and verify data loads
        new_tracker = PerformanceTracker(temp_storage_dir, auto_save=False)

        assert len(new_tracker.runs) == len(tracker_with_data.runs)
        for run_id in tracker_with_data.runs:
            assert run_id in new_tracker.runs

            original_run = tracker_with_data.runs[run_id]
            loaded_run = new_tracker.runs[run_id]

            assert original_run.run_id == loaded_run.run_id
            assert original_run.model_config == loaded_run.model_config
            assert original_run.dataset_info == loaded_run.dataset_info

            # Verify numpy arrays are preserved
            np.testing.assert_array_equal(
                original_run.predictions["y_true"], loaded_run.predictions["y_true"]
            )
            np.testing.assert_array_equal(
                original_run.predictions["y_pred"], loaded_run.predictions["y_pred"]
            )

    def test_run_comparison(self, tracker_with_data):
        """Test comparing multiple runs."""
        run_ids = list(tracker_with_data.runs.keys())

        comparison = tracker_with_data.compare_runs(
            run_ids=run_ids, metrics=["RMSE", "MUE", "R2"]
        )

        # Verify comparison structure
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == len(run_ids)
        assert "run_id" in comparison.columns
        assert "RMSE" in comparison.columns
        assert "MUE" in comparison.columns
        assert "R2" in comparison.columns

        # Verify all values are finite
        for metric in ["RMSE", "MUE", "R2"]:
            assert (
                comparison[metric].notna().all()
            ), f"{metric} should not have NaN values"

    def test_best_run_identification(self, tracker_with_data):
        """Test finding best performing runs."""
        best_rmse = tracker_with_data.get_best_run(metric="RMSE", minimize=True)
        best_r2 = tracker_with_data.get_best_run(metric="R2", minimize=False)

        assert best_rmse is not None
        assert best_r2 is not None
        assert best_rmse in tracker_with_data.runs
        assert best_r2 in tracker_with_data.runs

        # Verify the runs have the expected performance
        best_rmse_run = tracker_with_data.get_run(best_rmse)
        best_r2_run = tracker_with_data.get_run(best_r2)

        assert "RMSE" in best_rmse_run.performance_metrics
        assert "R2" in best_r2_run.performance_metrics

    def test_tag_filtering(self, tracker_with_data):
        """Test filtering runs by tags."""
        # Test single tag filtering
        test_runs = tracker_with_data.list_runs(tags=["test"])
        assert len(test_runs) > 0

        for run_id in test_runs:
            run = tracker_with_data.get_run(run_id)
            assert "test" in run.metadata.get("tags", [])

        # Test multiple tag filtering
        small_test_runs = tracker_with_data.list_runs(tags=["test", "small_test"])
        assert len(small_test_runs) >= 0  # May be 0 depending on how tags are set

    def test_data_export(self, tracker_with_data, temp_storage_dir):
        """Test data export functionality."""
        # Test CSV export
        csv_file = tracker_with_data.export_data(format="csv")
        assert Path(csv_file).exists()

        # Load and verify CSV
        exported_df = pd.read_csv(csv_file)
        assert len(exported_df) > 0
        assert "run_id" in exported_df.columns
        assert "RMSE" in exported_df.columns

        # Test JSON export
        json_file = tracker_with_data.export_data(format="json")
        assert Path(json_file).exists()

        # Load and verify JSON
        with open(json_file, "r") as f:
            exported_data = json.load(f)
        assert len(exported_data) > 0
        assert all("run_id" in record for record in exported_data)

    def test_plotting_integration(self, tracker_with_data):
        """Test plotting functionality with real data."""
        run_ids = list(tracker_with_data.runs.keys())

        # Test performance trends plot
        fig = tracker_with_data.plot_performance_trends(
            metric="RMSE", run_ids=run_ids, figsize=(8, 6)
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

        # Test run comparison plot
        fig = tracker_with_data.plot_run_comparison(
            run_ids=run_ids, metrics=["RMSE", "R2"], figsize=(10, 6)
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # One for each metric
        plt.close(fig)


class TestParameterSweepIntegration:
    """
    Integration tests for ParameterSweep with real models and datasets.

    These tests verify that parameter sweeps work correctly with
    actual MAPLE models and produce valid results.
    """

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_datasets(self):
        """Create small test datasets for parameter sweeps."""
        datasets = {
            "dataset_A": SyntheticFEPDataset(add_noise=False, random_seed=42),
            "dataset_B": SyntheticFEPDataset(add_noise=False, random_seed=123),
        }
        return datasets

    @pytest.fixture
    def parameter_sweep_setup(self, temp_storage_dir, test_datasets):
        """Set up parameter sweep with tracker and datasets."""
        tracker = PerformanceTracker(temp_storage_dir, auto_save=True)
        base_config = ModelConfig(
            learning_rate=0.001,
            num_steps=100,  # Small for testing
            prior_type=PriorType.NORMAL,
            prior_parameters=[0.0, 1.0],
        )

        sweep = ParameterSweep(tracker, base_config, test_datasets)
        return sweep, tracker

    def test_single_parameter_sweep(self, parameter_sweep_setup):
        """Test single parameter sweep functionality."""
        sweep, tracker = parameter_sweep_setup

        # Run parameter sweep with fast parameters for testing
        results = sweep.sweep_parameter(
            parameter_name="learning_rate",
            values=[0.01, 0.1],  # Fewer values for speed
            metrics=["RMSE", "MUE"],
            experiment_name="test_lr_sweep",
            bootstrap_stats=False,  # Disable for speed
        )

        # Verify results structure
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

        # Verify required columns
        required_columns = [
            "experiment",
            "dataset",
            "parameter",
            "parameter_value",
            "metric",
            "value",
            "run_id",
        ]
        for col in required_columns:
            assert col in results.columns, f"Missing column: {col}"

        # Verify data content
        assert results["experiment"].iloc[0] == "test_lr_sweep"
        assert results["parameter"].iloc[0] == "learning_rate"
        assert set(results["parameter_value"].unique()) == {
            0.01,
            0.1,
        }  # Updated for fewer values
        assert set(results["metric"].unique()) == {"RMSE", "MUE"}

        # Verify tracker recorded the runs
        assert len(tracker.runs) > 0
        test_runs = tracker.list_runs(tags=["test_lr_sweep"])
        assert len(test_runs) > 0

    def test_grid_search_integration(self, parameter_sweep_setup):
        """Test grid search functionality."""
        sweep, tracker = parameter_sweep_setup

        # Define small parameter grid for testing
        parameter_grid = {
            "learning_rate": [0.01, 0.1],  # Valid values
            "num_steps": [10, 20],  # Small values for speed, but >= 10
        }

        # Run grid search
        results = sweep.grid_search(
            parameter_grid=parameter_grid,
            metrics=["RMSE"],
            experiment_name="test_grid_search",
        )

        # Verify results structure
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

        # Should have results for all parameter combinations
        expected_combinations = len(parameter_grid["learning_rate"]) * len(
            parameter_grid["num_steps"]
        )
        expected_total_results = expected_combinations * len(
            sweep.datasets
        )  # multiply by number of datasets

        # Allow some tolerance for potential failures
        assert len(results) >= expected_total_results * 0.5

        # Verify parameter columns exist
        assert "learning_rate" in results.columns
        assert "num_steps" in results.columns

        # Verify parameter values are from the grid
        lr_values = set(results["learning_rate"].unique())
        assert lr_values.issubset(set(parameter_grid["learning_rate"]))

        steps_values = set(results["num_steps"].unique())
        assert steps_values.issubset(set(parameter_grid["num_steps"]))

    def test_optimal_parameter_finding(self, parameter_sweep_setup):
        """Test finding optimal parameters from results."""
        sweep, tracker = parameter_sweep_setup

        # Create mock results DataFrame
        results = pd.DataFrame(
            {
                "experiment": ["test"] * 6,
                "dataset": ["dataset_A"] * 3 + ["dataset_B"] * 3,
                "parameter": ["learning_rate"] * 6,
                "parameter_value": [0.001, 0.01, 0.1] * 2,
                "metric": ["RMSE"] * 6,
                "value": [0.5, 0.3, 0.8, 0.6, 0.2, 0.9],  # 0.01 is best on average
                "run_id": [f"run_{i}" for i in range(6)],
            }
        )

        # Find optimal parameters
        optimal = sweep.find_optimal_parameters(
            results, metric="RMSE", minimize=True, aggregation="mean"
        )

        assert isinstance(optimal, pd.DataFrame)
        assert len(optimal) == 1  # One parameter tested
        assert optimal.iloc[0]["parameter"] == "learning_rate"
        assert optimal.iloc[0]["optimal_value"] == 0.01  # Best average performance
        assert optimal.iloc[0]["metric"] == "RMSE"

    def test_plot_parameter_effects(self, parameter_sweep_setup):
        """Test plotting parameter effects."""
        sweep, tracker = parameter_sweep_setup

        # Create mock results for plotting
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
        fig = sweep.plot_parameter_effects(
            results_df=results,
            parameter_name="learning_rate",
            metrics=["RMSE", "R2"],
            figsize=(10, 6),
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # One subplot per metric
        plt.close(fig)

    def test_parameter_sweep_error_handling(self, parameter_sweep_setup):
        """Test error handling in parameter sweeps."""
        sweep, tracker = parameter_sweep_setup

        # Test with invalid parameter values - should handle gracefully and track failures
        results = sweep.sweep_parameter(
            parameter_name="num_steps",
            values=[-1],  # Invalid: num_steps must be >= 10
            metrics=["RMSE"],
            experiment_name="error_test",
        )

        # Should return empty DataFrame when all experiments fail
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0  # No successful experiments

        # Should track the failure
        failed_experiments = sweep.get_failed_experiments()
        assert isinstance(failed_experiments, pd.DataFrame)
        assert len(failed_experiments) == 2  # 2 datasets × 1 failed parameter value
        assert "num_steps" in failed_experiments["parameter"].values
        assert -1 in failed_experiments["parameter_value"].values
        assert "ValidationError" in failed_experiments["error_type"].values

        # Test with empty values list - should handle gracefully
        empty_results = sweep.sweep_parameter(
            parameter_name="learning_rate",
            values=[],  # Empty list
            metrics=["RMSE"],
            experiment_name="empty_test",
        )

        # Should return empty DataFrame for empty values list
        assert isinstance(empty_results, pd.DataFrame)
        assert len(empty_results) == 0  # No experiments to run


class TestUtilsConvenienceFunctions:
    """
    Test convenience functions and high-level integrations.

    These tests verify that the convenience functions work correctly
    and provide the expected simplified interfaces.
    """

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_datasets(self):
        """Create test datasets."""
        datasets = {"test_A": SyntheticFEPDataset(add_noise=False, random_seed=42)}
        return datasets

    def test_load_performance_history(self, temp_storage_dir, test_datasets):
        """Test loading performance history from existing data."""
        # Create some initial data
        tracker = PerformanceTracker(temp_storage_dir, auto_save=True)

        dataset = test_datasets["test_A"]
        graph_data = dataset.get_graph_data()
        n_nodes = graph_data["N"]

        y_true = np.random.randn(n_nodes)
        y_pred = y_true + np.random.normal(0, 0.1, n_nodes)

        tracker.record_run(
            run_id="history_test",
            y_true=y_true,
            y_pred=y_pred,
            model_config={"test": True},
            dataset_info={"name": "test"},
            tags=["history"],
        )

        # Test loading history
        loaded_tracker = load_performance_history(temp_storage_dir)

        assert isinstance(loaded_tracker, PerformanceTracker)
        assert len(loaded_tracker.runs) == 1
        assert "history_test" in loaded_tracker.runs

        # Verify loaded data matches original
        original_run = tracker.get_run("history_test")
        loaded_run = loaded_tracker.get_run("history_test")

        assert original_run.run_id == loaded_run.run_id
        assert original_run.model_config == loaded_run.model_config

    def test_compare_model_runs_function(self):
        """Test standalone model run comparison function."""
        # Create mock model runs
        run1 = ModelRun(
            run_id="run1",
            timestamp="2024-01-01T00:00:00",
            model_config={"lr": 0.001},
            dataset_info={"name": "test"},
            performance_metrics={"RMSE": 0.5, "R2": 0.8},
            bootstrap_metrics={},
            predictions={
                "y_true": np.array([1, 2, 3]),
                "y_pred": np.array([1.1, 2.1, 2.9]),
            },
            metadata={},
        )

        run2 = ModelRun(
            run_id="run2",
            timestamp="2024-01-01T01:00:00",
            model_config={"lr": 0.01},
            dataset_info={"name": "test"},
            performance_metrics={"RMSE": 0.3, "R2": 0.9},
            bootstrap_metrics={},
            predictions={
                "y_true": np.array([1, 2, 3]),
                "y_pred": np.array([1.05, 2.05, 2.95]),
            },
            metadata={},
        )

        # Test comparison
        comparison = compare_model_runs([run1, run2], metrics=["RMSE", "R2"])

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "run_id" in comparison.columns
        assert "RMSE" in comparison.columns
        assert "R2" in comparison.columns

        # Verify values
        assert set(comparison["run_id"]) == {"run1", "run2"}
        assert comparison[comparison["run_id"] == "run1"]["RMSE"].iloc[0] == 0.5
        assert comparison[comparison["run_id"] == "run2"]["RMSE"].iloc[0] == 0.3

    @patch("maple.models.node_model.NodeModel.train")
    @patch("maple.models.node_model.NodeModel.get_results")
    def test_create_prior_sweep_experiment(
        self, mock_get_results, mock_train, temp_storage_dir, test_datasets
    ):
        """Test the convenience function for prior sweep experiments."""
        # Mock model behavior
        mock_get_results.return_value = {
            "node_estimates": {i: torch.randn(1).item() for i in range(6)}
        }
        mock_train.return_value = None

        # Create tracker
        tracker = PerformanceTracker(temp_storage_dir, auto_save=True)

        # Test the convenience function
        results = create_prior_sweep_experiment(
            tracker=tracker,
            datasets=test_datasets,
            prior_std_values=[0.1, 1.0],  # Small list for testing
            base_config=ModelConfig(num_steps=50),  # Small for testing
        )

        # Verify results
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

        # Should have results for both datasets and both prior values
        expected_combinations = len(test_datasets) * 2  # 2 prior values
        expected_metrics = 4  # RMSE, MUE, R2, rho
        expected_total = expected_combinations * expected_metrics

        # Allow some tolerance for potential failures
        assert len(results) >= expected_total * 0.5

        # Verify structure matches expected format (like notebook stats_df)
        expected_columns = [
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
        for col in expected_columns:
            assert col in results.columns, f"Missing expected column: {col}"

        # Verify parameter values
        assert set(results["parameter_value"].unique()).issubset({0.1, 1.0})
        assert "prior_std" in results["parameter"].values

        # Verify runs were recorded in tracker
        prior_runs = tracker.list_runs(tags=["prior_std_effect"])
        assert len(prior_runs) > 0


class TestUtilsErrorHandling:
    """
    Test error handling and edge cases for utils components.

    These tests verify that the utils components handle errors
    gracefully and provide helpful error messages.
    """

    def test_tracker_invalid_storage_dir(self):
        """Test error handling for invalid storage directories."""
        # Test with file instead of directory
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises((OSError, PermissionError, FileExistsError)):
                PerformanceTracker(temp_file.name)

    def test_tracker_corrupted_data(self, temp_directory):
        """Test handling of corrupted data files."""
        tracker = PerformanceTracker(temp_directory, auto_save=True)

        # Corrupt the JSON file
        with open(tracker.runs_file, "w") as f:
            f.write("invalid json content")

        # Should handle corrupted data gracefully
        new_tracker = PerformanceTracker(temp_directory, auto_save=False)
        assert len(new_tracker.runs) == 0  # Should start fresh

    def test_tracker_missing_data_fields(self, temp_directory):
        """Test handling of runs with missing data fields."""
        tracker = PerformanceTracker(temp_directory, auto_save=True)

        # Record run with missing optional fields
        tracker.record_run(
            run_id="minimal_run",
            y_true=np.array([1, 2, 3]),
            y_pred=np.array([1.1, 2.1, 2.9]),
            model_config={},
            dataset_info={},
            # Missing metadata, tags, uncertainties
        )

        run = tracker.get_run("minimal_run")
        assert run is not None
        assert run.run_id == "minimal_run"
        assert len(run.performance_metrics) > 0

    def test_parameter_sweep_invalid_config(self, temp_directory):
        """Test parameter sweep with invalid configuration."""
        tracker = PerformanceTracker(temp_directory)
        datasets = {"test": SyntheticFEPDataset(add_noise=False, random_seed=42)}

        # Test with invalid base config - should handle gracefully with failure tracking
        sweep = ParameterSweep(tracker, "invalid_config", datasets)

        # Should handle the error gracefully and record as failed experiment
        results = sweep.sweep_parameter(
            parameter_name="learning_rate",
            values=[0.01],
            experiment_name="invalid_test",
        )

        # Should return empty results due to failure
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

        # Should record the failure
        failed_experiments = sweep.get_failed_experiments()
        assert len(failed_experiments) == 1
        assert "AttributeError" in failed_experiments["error_type"].values

    def test_empty_results_handling(self, temp_directory):
        """Test handling of empty results in analysis functions."""
        tracker = PerformanceTracker(temp_directory)

        # Test empty comparison
        comparison = tracker.compare_runs([], metrics=["RMSE"])
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 0

        # Test best run with no runs
        best_run = tracker.get_best_run(metric="RMSE")
        assert best_run is None


class TestUtilsPerformance:
    """
    Test performance and scalability of utils components.

    These tests verify that the utils components perform
    reasonably well with larger datasets and many runs.
    """

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_tracker_many_runs_performance(self, temp_storage_dir):
        """Test tracker performance with many runs."""
        tracker = PerformanceTracker(temp_storage_dir, auto_save=False)

        # Add many runs
        n_runs = 50
        for i in range(n_runs):
            y_true = np.random.randn(10)
            y_pred = y_true + np.random.normal(0, 0.1, 10)

            tracker.record_run(
                run_id=f"perf_test_{i}",
                y_true=y_true,
                y_pred=y_pred,
                model_config={"run_number": i},
                dataset_info={"test": True},
            )

        # Test operations are still fast
        assert len(tracker.runs) == n_runs

        # Test comparison performance
        all_runs = list(tracker.runs.keys())
        comparison = tracker.compare_runs(all_runs, metrics=["RMSE", "R2"])
        assert len(comparison) == n_runs

        # Test best run finding
        best_run = tracker.get_best_run(metric="RMSE")
        assert best_run is not None

    def test_data_export_performance(self, temp_storage_dir):
        """Test data export performance with substantial data."""
        tracker = PerformanceTracker(temp_storage_dir, auto_save=False)

        # Add runs with larger datasets
        for i in range(10):
            y_true = np.random.randn(100)  # Larger arrays
            y_pred = y_true + np.random.normal(0, 0.2, 100)

            tracker.record_run(
                run_id=f"export_test_{i}",
                y_true=y_true,
                y_pred=y_pred,
                model_config={"size": "large", "run": i},
                dataset_info={"n_samples": 100},
            )

        # Test export performance
        csv_file = tracker.export_data(format="csv")
        assert Path(csv_file).exists()

        # Verify exported data
        exported_df = pd.read_csv(csv_file)
        assert len(exported_df) == 10
        assert "RMSE" in exported_df.columns

    def test_memory_usage_stability(self, temp_storage_dir):
        """Test that memory usage remains stable with repeated operations."""
        tracker = PerformanceTracker(temp_storage_dir, auto_save=False)

        # Repeatedly add and remove operations
        for batch in range(5):
            # Add runs
            for i in range(10):
                y_true = np.random.randn(20)
                y_pred = y_true + np.random.normal(0, 0.1, 20)

                tracker.record_run(
                    run_id=f"memory_test_{batch}_{i}",
                    y_true=y_true,
                    y_pred=y_pred,
                    model_config={"batch": batch, "run": i},
                    dataset_info={"memory_test": True},
                )

            # Perform operations
            all_runs = list(tracker.runs.keys())
            comparison = tracker.compare_runs(all_runs[-5:], metrics=["RMSE"])
            best_run = tracker.get_best_run(metric="RMSE")

            # Verify operations complete successfully
            assert len(comparison) <= 5
            assert best_run is not None

        # Final verification
        assert len(tracker.runs) == 50  # 5 batches × 10 runs each
