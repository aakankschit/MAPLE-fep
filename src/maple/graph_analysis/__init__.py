"""
MAPLE Graph Analysis Package

This package provides performance statistics and plotting functionality
for FEP analysis.
"""

from .performance_stats import (bootstrap_statistic, calculate_correlation,
                                calculate_mae, calculate_r2, calculate_rmse,
                                compute_simple_statistics)
from .plotting_performance import (plot_dataset_all_DDGs,
                                   plot_dataset_DDGs, plot_dataset_DGs,
                                   plot_model_comparison_bars,
                                   plot_model_comparison_correlation,
                                   plot_error_distribution)
from .graph_setup import GraphSetup
from .graph_cycle_analysis import GraphCycleAnalysis

__all__ = [
    # Performance statistics
    "calculate_mae",
    "calculate_rmse",
    "calculate_r2",
    "calculate_correlation",
    "bootstrap_statistic",
    "compute_simple_statistics",
    # Plotting functions - single model
    "plot_dataset_DDGs",
    "plot_dataset_DGs",
    "plot_dataset_all_DDGs",
    # Plotting functions - multi-model comparison
    "plot_model_comparison_bars",
    "plot_model_comparison_correlation",
    # Plotting functions - error analysis
    "plot_error_distribution",
    # Graph Setup
    "GraphSetup",
    "GraphCycleAnalysis",
]
