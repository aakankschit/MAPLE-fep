"""
Unit tests for the performance statistics module.

This module tests all statistical functions used for evaluating model performance
in the MAPLE package, including error metrics, correlation measures, and
bootstrap sampling functionality.
"""

import numpy as np
import pytest

from maple.graph_analysis.performance_stats import (bootstrap_statistic,
                                                    calculate_correlation,
                                                    calculate_mae,
                                                    calculate_r2,
                                                    calculate_rmse,
                                                    compute_simple_statistics)


class TestMAE:
    """
    Test cases for Mean Absolute Error (MAE) calculation.

    The MAE function should:
    - Calculate correct MAE for normal arrays
    - Handle NaN values appropriately
    - Return NaN when all values are NaN
    - Work with different array shapes
    """

    def test_mae_perfect_prediction(self):
        """Test MAE with perfect predictions (should be 0)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        mae = calculate_mae(y_true, y_pred)
        assert mae == 0.0, "MAE should be 0 for perfect predictions"

    def test_mae_known_values(self):
        """Test MAE with known expected result."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 2.5, 4.5])

        # Expected MAE = (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
        expected_mae = 0.5
        mae = calculate_mae(y_true, y_pred)

        assert (
            abs(mae - expected_mae) < 1e-10
        ), f"Expected MAE {expected_mae}, got {mae}"

    def test_mae_with_nan_values(self, sample_arrays_with_nan):
        """Test MAE handling of NaN values."""
        y_true, y_pred = sample_arrays_with_nan

        mae = calculate_mae(y_true, y_pred)

        # Should only consider non-NaN pairs: (1.0, 1.1), (2.0, 2.1), (5.0, 5.1)
        # MAE = (0.1 + 0.1 + 0.1) / 3 = 0.1
        expected_mae = 0.1
        assert (
            abs(mae - expected_mae) < 1e-10
        ), f"Expected MAE {expected_mae}, got {mae}"

    def test_mae_all_nan(self):
        """Test MAE when all values are NaN."""
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([np.nan, np.nan])

        mae = calculate_mae(y_true, y_pred)
        assert np.isnan(mae), "MAE should be NaN when all values are NaN"

    def test_mae_empty_arrays(self):
        """Test MAE with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        mae = calculate_mae(y_true, y_pred)
        assert np.isnan(mae), "MAE should be NaN for empty arrays"


class TestRMSE:
    """
    Test cases for Root Mean Square Error (RMSE) calculation.

    The RMSE function should:
    - Calculate correct RMSE for normal arrays
    - Handle NaN values appropriately
    - Return NaN when all values are NaN
    - Always be >= MAE for the same data
    """

    def test_rmse_perfect_prediction(self):
        """Test RMSE with perfect predictions (should be 0)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        rmse = calculate_rmse(y_true, y_pred)
        assert rmse == 0.0, "RMSE should be 0 for perfect predictions"

    def test_rmse_known_values(self):
        """Test RMSE with known expected result."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 1.0])

        # MSE = (1^2 + 1^2 + 2^2) / 3 = 6/3 = 2
        # RMSE = sqrt(2) ≈ 1.414
        expected_rmse = np.sqrt(2.0)
        rmse = calculate_rmse(y_true, y_pred)

        assert (
            abs(rmse - expected_rmse) < 1e-10
        ), f"Expected RMSE {expected_rmse}, got {rmse}"

    def test_rmse_with_nan_values(self, sample_arrays_with_nan):
        """Test RMSE handling of NaN values."""
        y_true, y_pred = sample_arrays_with_nan

        rmse = calculate_rmse(y_true, y_pred)

        # Should only consider non-NaN pairs: (1.0, 1.1), (2.0, 2.1), (5.0, 5.1)
        # MSE = (0.1^2 + 0.1^2 + 0.1^2) / 3 = 0.01/3 * 3 = 0.01
        # RMSE = sqrt(0.01) = 0.1
        expected_rmse = 0.1
        assert (
            abs(rmse - expected_rmse) < 1e-10
        ), f"Expected RMSE {expected_rmse}, got {rmse}"

    def test_rmse_vs_mae_relationship(self, sample_arrays_for_stats):
        """Test that RMSE >= MAE for the same data."""
        y_true, y_pred = sample_arrays_for_stats

        mae = calculate_mae(y_true, y_pred)
        rmse = calculate_rmse(y_true, y_pred)

        assert rmse >= mae, f"RMSE ({rmse}) should be >= MAE ({mae})"


class TestR2:
    """
    Test cases for R-squared (coefficient of determination) calculation.

    The R² function should:
    - Return 1.0 for perfect predictions
    - Return values <= 1.0 for all cases
    - Handle NaN values appropriately
    - Work with different data distributions
    """

    def test_r2_perfect_prediction(self):
        """Test R² with perfect predictions (should be 1.0)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        r2 = calculate_r2(y_true, y_pred)
        assert (
            abs(r2 - 1.0) < 1e-10
        ), f"R² should be 1.0 for perfect predictions, got {r2}"

    def test_r2_mean_prediction(self):
        """Test R² when predictions equal the mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, np.mean(y_true))

        r2 = calculate_r2(y_true, y_pred)
        # When predictions are constant (mean), R² may be NaN due to correlation being undefined
        # This is mathematically correct behavior
        assert (
            np.isnan(r2) or abs(r2 - 0.0) < 1e-10
        ), f"R² should be NaN or 0.0 for mean predictions, got {r2}"

    def test_r2_bounds(self, sample_arrays_for_stats):
        """Test that R² is bounded appropriately."""
        y_true, y_pred = sample_arrays_for_stats

        r2 = calculate_r2(y_true, y_pred)

        # R² can be negative for very poor predictions, but should be <= 1
        assert r2 <= 1.0, f"R² should be <= 1.0, got {r2}"

    def test_r2_with_nan_values(self, sample_arrays_with_nan):
        """Test R² handling of NaN values."""
        y_true, y_pred = sample_arrays_with_nan

        r2 = calculate_r2(y_true, y_pred)

        # Should compute R² only for non-NaN pairs
        assert not np.isnan(r2), "R² should not be NaN for partially valid data"


class TestCorrelation:
    """
    Test cases for correlation coefficient calculation.

    The correlation function should:
    - Return 1.0 for perfectly correlated data
    - Return -1.0 for perfectly anti-correlated data
    - Return 0.0 for uncorrelated data
    - Handle NaN values appropriately
    """

    def test_correlation_perfect_positive(self):
        """Test correlation with perfectly positively correlated data."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 4.0, 6.0, 8.0])  # Perfect linear relationship

        corr = calculate_correlation(y_true, y_pred)
        assert abs(corr - 1.0) < 1e-10, f"Correlation should be 1.0, got {corr}"

    def test_correlation_perfect_negative(self):
        """Test correlation with perfectly negatively correlated data."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])  # Perfect negative relationship

        corr = calculate_correlation(y_true, y_pred)
        assert abs(corr - (-1.0)) < 1e-10, f"Correlation should be -1.0, got {corr}"

    def test_correlation_bounds(self, sample_arrays_for_stats):
        """Test that correlation is bounded between -1 and 1."""
        y_true, y_pred = sample_arrays_for_stats

        corr = calculate_correlation(y_true, y_pred)

        assert (
            -1.0 <= corr <= 1.0
        ), f"Correlation should be between -1 and 1, got {corr}"

    def test_correlation_with_nan_values(self, sample_arrays_with_nan):
        """Test correlation handling of NaN values."""
        y_true, y_pred = sample_arrays_with_nan

        corr = calculate_correlation(y_true, y_pred)

        # Should compute correlation only for non-NaN pairs
        assert not np.isnan(
            corr
        ), "Correlation should not be NaN for partially valid data"


class TestBootstrapStatistic:
    """
    Test cases for bootstrap statistic calculation.

    The bootstrap function should:
    - Return statistics with correct shape
    - Provide reasonable confidence intervals
    - Handle different statistics functions
    - Work with various sample sizes
    """

    def test_bootstrap_mae_shape(self, sample_arrays_for_stats):
        """Test that bootstrap returns correct dictionary structure."""
        y_true, y_pred = sample_arrays_for_stats
        nbootstrap = 100

        bootstrap_stats = bootstrap_statistic(
            y_true, y_pred, statistic="MUE", nbootstrap=nbootstrap
        )

        expected_keys = {"mle", "mean", "stderr", "low", "high"}
        assert (
            set(bootstrap_stats.keys()) == expected_keys
        ), f"Expected keys {expected_keys}, got {set(bootstrap_stats.keys())}"

    def test_bootstrap_deterministic(self, sample_arrays_for_stats):
        """Test bootstrap reproducibility with numpy random seed."""
        y_true, y_pred = sample_arrays_for_stats

        # Run bootstrap twice with same numpy seed
        np.random.seed(42)
        bootstrap_stats1 = bootstrap_statistic(
            y_true, y_pred, statistic="MUE", nbootstrap=50
        )

        np.random.seed(42)
        bootstrap_stats2 = bootstrap_statistic(
            y_true, y_pred, statistic="MUE", nbootstrap=50
        )

        # Results should be identical
        for key in bootstrap_stats1.keys():
            assert (
                abs(bootstrap_stats1[key] - bootstrap_stats2[key]) < 1e-10
            ), f"Values for '{key}' should be identical with same seed"

    def test_bootstrap_different_statistics(self, sample_arrays_for_stats):
        """Test bootstrap with different statistic types."""
        y_true, y_pred = sample_arrays_for_stats

        mae_bootstrap = bootstrap_statistic(
            y_true, y_pred, statistic="MUE", nbootstrap=50
        )
        rmse_bootstrap = bootstrap_statistic(
            y_true, y_pred, statistic="RMSE", nbootstrap=50
        )

        # RMSE should generally be >= MAE
        assert (
            rmse_bootstrap["mean"] >= mae_bootstrap["mean"]
        ), "Mean RMSE should be >= mean MAE across bootstrap samples"

        # Results should be different for different statistics
        assert (
            mae_bootstrap["mle"] != rmse_bootstrap["mle"]
        ), "Different statistics should produce different bootstrap results"

    def test_bootstrap_confidence_interval(self, sample_arrays_for_stats):
        """Test that bootstrap provides reasonable confidence intervals."""
        y_true, y_pred = sample_arrays_for_stats

        bootstrap_stats = bootstrap_statistic(
            y_true, y_pred, statistic="MUE", nbootstrap=1000
        )

        # Check that confidence interval bounds are reasonable
        assert (
            bootstrap_stats["low"] <= bootstrap_stats["high"]
        ), "Lower CI bound should be <= upper CI bound"

        # Confidence interval should be non-degenerate
        assert (
            bootstrap_stats["high"] > bootstrap_stats["low"]
        ), f"Confidence interval should be non-degenerate: [{bootstrap_stats['low']}, {bootstrap_stats['high']}]"

        # Mean should typically be within CI bounds
        assert (
            bootstrap_stats["low"] <= bootstrap_stats["mean"] <= bootstrap_stats["high"]
        ), "Bootstrap mean should typically be within CI bounds"


class TestComputeSimpleStatistics:
    """
    Test cases for the compute_simple_statistics function.

    This function should:
    - Return a dictionary with all expected statistics
    - Handle NaN values appropriately
    - Provide consistent results with individual functions
    - Include bootstrap confidence intervals when requested
    """

    def test_simple_statistics_structure(self, sample_arrays_for_stats):
        """Test that compute_simple_statistics returns expected structure."""
        y_true, y_pred = sample_arrays_for_stats

        stats = compute_simple_statistics(y_true, y_pred)

        expected_keys = {"MUE", "RMSE", "R2", "rho", "KTAU"}
        assert (
            set(stats.keys()) == expected_keys
        ), f"Expected keys {expected_keys}, got {set(stats.keys())}"

    def test_simple_statistics_consistency(self, sample_arrays_for_stats):
        """Test that statistics match individual function calls."""
        y_true, y_pred = sample_arrays_for_stats

        stats = compute_simple_statistics(y_true, y_pred)

        # Compare with individual function calls
        assert abs(stats["MUE"] - calculate_mae(y_true, y_pred)) < 1e-10
        assert abs(stats["RMSE"] - calculate_rmse(y_true, y_pred)) < 1e-10
        assert abs(stats["R2"] - calculate_r2(y_true, y_pred)) < 1e-10
        assert abs(stats["rho"] - calculate_correlation(y_true, y_pred)) < 1e-10

        # KTAU should be a reasonable correlation value
        assert -1.0 <= stats["KTAU"] <= 1.0, "Kendall's tau should be between -1 and 1"

    def test_simple_statistics_with_bootstrap_separate(self, sample_arrays_for_stats):
        """Test combining compute_simple_statistics with bootstrap_statistic."""
        y_true, y_pred = sample_arrays_for_stats

        # Get basic statistics
        stats = compute_simple_statistics(y_true, y_pred)

        # Get bootstrap confidence intervals separately
        mae_bootstrap = bootstrap_statistic(
            y_true, y_pred, statistic="MUE", nbootstrap=100
        )
        rmse_bootstrap = bootstrap_statistic(
            y_true, y_pred, statistic="RMSE", nbootstrap=100
        )

        # Basic stats should be consistent with bootstrap MLE
        assert (
            abs(stats["MUE"] - mae_bootstrap["mle"]) < 1e-10
        ), "Simple statistics MUE should match bootstrap MLE"
        assert (
            abs(stats["RMSE"] - rmse_bootstrap["mle"]) < 1e-10
        ), "Simple statistics RMSE should match bootstrap MLE"

        # Bootstrap should provide confidence intervals
        assert (
            mae_bootstrap["low"] <= mae_bootstrap["high"]
        ), "MAE bootstrap CI should have low <= high"
        assert (
            rmse_bootstrap["low"] <= rmse_bootstrap["high"]
        ), "RMSE bootstrap CI should have low <= high"

    def test_simple_statistics_with_nan(self, sample_arrays_with_nan):
        """Test compute_simple_statistics with NaN values."""
        y_true, y_pred = sample_arrays_with_nan

        stats = compute_simple_statistics(y_true, y_pred)

        # Should handle NaN values gracefully
        assert not np.isnan(
            stats["MUE"]
        ), "MUE should not be NaN for partially valid data"
        assert not np.isnan(
            stats["RMSE"]
        ), "RMSE should not be NaN for partially valid data"
        assert not np.isnan(
            stats["R2"]
        ), "R2 should not be NaN for partially valid data"
        assert not np.isnan(
            stats["rho"]
        ), "rho should not be NaN for partially valid data"

        # KTAU might be NaN in some edge cases, so we just check it exists
        assert "KTAU" in stats, "KTAU should be present in results"


class TestEdgeCases:
    """
    Test edge cases and error conditions for performance statistics.

    These tests ensure robust handling of:
    - Empty arrays
    - Single-value arrays
    - Constant arrays
    - Very large/small values
    """

    def test_constant_true_values(self):
        """Test statistics when true values are constant."""
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 5.0, 6.0, 5.0])

        stats = compute_simple_statistics(y_true, y_pred)

        # MUE and RMSE should be computable
        assert not np.isnan(stats["MUE"])
        assert not np.isnan(stats["RMSE"])

        # Correlation is NaN when true values are constant (mathematically correct)
        assert np.isnan(
            stats["rho"]
        ), "Correlation should be NaN when true values are constant"

    def test_constant_pred_values(self):
        """Test statistics when predicted values are constant."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.5, 2.5, 2.5, 2.5])

        stats = compute_simple_statistics(y_true, y_pred)

        # MUE and RMSE should be computable
        assert not np.isnan(stats["MUE"])
        assert not np.isnan(stats["RMSE"])

        # R² should be either 0.0 or NaN when predicted values are constant
        assert stats["R2"] == 0.0 or np.isnan(stats["R2"]), "R² should be 0.0 or NaN when predicted values are constant"

    def test_single_value_arrays(self):
        """Test statistics with single-value arrays."""
        y_true = np.array([1.0])
        y_pred = np.array([1.5])

        # Most statistics should handle single values
        mae = calculate_mae(y_true, y_pred)
        rmse = calculate_rmse(y_true, y_pred)

        assert mae == 0.5
        assert rmse == 0.5

        # Correlation and R² are undefined for single values
        with pytest.warns(RuntimeWarning):
            _ = calculate_correlation(y_true, y_pred)
            _ = calculate_r2(y_true, y_pred)

    def test_very_large_values(self):
        """Test statistics with very large values."""
        y_true = np.array([1e10, 2e10, 3e10])
        y_pred = np.array([1.1e10, 2.1e10, 2.9e10])

        stats = compute_simple_statistics(y_true, y_pred)

        # Should handle large values without overflow
        assert not np.isnan(stats["MUE"])
        assert not np.isnan(stats["RMSE"])
        assert not np.isinf(stats["MUE"])
        assert not np.isinf(stats["RMSE"])

    def test_very_small_values(self):
        """Test statistics with very small values."""
        y_true = np.array([1e-10, 2e-10, 3e-10])
        y_pred = np.array([1.1e-10, 2.1e-10, 2.9e-10])

        stats = compute_simple_statistics(y_true, y_pred)

        # Should handle small values without underflow
        assert not np.isnan(stats["MUE"])
        assert not np.isnan(stats["RMSE"])
        assert stats["MUE"] > 0
        assert stats["RMSE"] > 0
