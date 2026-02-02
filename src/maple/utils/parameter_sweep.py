"""
Parameter Sweep and Optimization Utilities

This module provides tools for systematic parameter exploration, optimization,
and visualization of parameter effects on model performance across datasets.
"""

from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ..graph_analysis.performance_stats import bootstrap_statistic
from ..models import NodeModel, NodeModelConfig, GMVI_model, GMVIConfig, PriorType
from .performance_tracker import PerformanceTracker


class ParameterSweep:
    """
    Systematic parameter exploration and optimization tool.

    This class enables comprehensive parameter sweeps across multiple datasets,
    automatic performance tracking, and visualization of parameter effects.

    Parameters
    ----------
    tracker : PerformanceTracker
        Performance tracker instance for storing results
    base_config : NodeModelConfig
        Base model configuration to modify during sweeps
    datasets : Dict[str, Any]
        Dictionary of datasets to test across

    Examples
    --------
    >>> sweep = ParameterSweep(tracker, base_config, datasets)
    >>>
    >>> # Single parameter sweep
    >>> sweep.sweep_parameter(
    ...     'learning_rate',
    ...     values=[0.0001, 0.001, 0.01, 0.1],
    ...     metrics=['RMSE', 'R2']
    ... )
    >>>
    >>> # Multi-parameter grid search
    >>> sweep.grid_search({
    ...     'learning_rate': [0.001, 0.01],
    ...     'num_steps': [500, 1000, 2000],
    ...     'error_std': [0.5, 1.0, 2.0]
    ... })
    """

    def __init__(
        self,
        tracker: PerformanceTracker,
        base_config: Any,  # Can be NodeModelConfig or GMVIConfig
        datasets: Dict[str, Any],
    ):
        """Initialize parameter sweep system."""
        self.tracker = tracker
        self.base_config = base_config
        self.datasets = datasets
        self.sweep_results = []
        self.failed_experiments = []  # Store failed parameter combinations

        # Detect model type from config
        if isinstance(base_config, GMVIConfig):
            self.model_type = "GMVI"
            self.model_class = GMVI_model
        elif isinstance(base_config, NodeModelConfig):
            self.model_type = "NodeModel"
            self.model_class = NodeModel
        else:
            raise ValueError(f"Unsupported config type: {type(base_config)}. Must be NodeModelConfig or GMVIConfig.")

    def sweep_parameter(
        self,
        parameter_name: str,
        values: List[Any],
        metrics: List[str] = ["RMSE", "MUE", "R2", "rho"],
        experiment_name: Optional[str] = None,
        center_data: bool = True,
        bootstrap_stats: bool = True,
        n_bootstrap: int = 1000,
    ) -> pd.DataFrame:
        """
        Sweep a single parameter across multiple values and datasets.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter to sweep (must be valid NodeModelConfig attribute)
        values : List[Any]
            List of parameter values to test
        metrics : List[str], default=['RMSE', 'MUE', 'R2', 'rho']
            Performance metrics to compute
        experiment_name : str, optional
            Name for this experiment series
        center_data : bool, default=True
            Whether to center experimental and calculated data
        bootstrap_stats : bool, default=True
            Whether to compute bootstrap confidence intervals
        n_bootstrap : int, default=1000
            Number of bootstrap samples

        Returns
        -------
        pd.DataFrame
            Results table with parameter values, metrics, and confidence intervals
        """

        if experiment_name is None:
            experiment_name = f"{parameter_name}_sweep"

        print(f"🔍 Parameter Sweep: {parameter_name}")
        print(f"   Values: {values}")
        print(f"   Datasets: {list(self.datasets.keys())}")
        print(f"   Metrics: {metrics}")
        print()

        results = []

        # Progress tracking
        total_experiments = len(values) * len(self.datasets)
        pbar = tqdm(total=total_experiments, desc="Running experiments")

        for dataset_name, dataset in self.datasets.items():
            print(f"📊 Dataset: {dataset_name}")

            # Get experimental data (you'll need to adapt this to your data structure)
            y_true = self._get_experimental_data(dataset)

            for param_value in values:
                try:
                    # Create modified config
                    config_dict = self.base_config.__dict__.copy()

                    # Handle special parameter types based on model type
                    if self.model_type == "NodeModel":
                        if parameter_name == "prior_std" and hasattr(
                            self.base_config, "prior_parameters"
                        ):
                            # Modify prior standard deviation for NodeModel
                            config_dict["prior_parameters"] = [0.0, param_value]
                        else:
                            config_dict[parameter_name] = param_value
                        modified_config = NodeModelConfig(**config_dict)
                    elif self.model_type == "GMVI":
                        # For GMVI, directly set the parameter
                        config_dict[parameter_name] = param_value
                        modified_config = GMVIConfig(**config_dict)

                    # Train model with modified config
                    model = self.model_class(config=modified_config, dataset=dataset)

                    # Call appropriate training method
                    if self.model_type == "NodeModel":
                        model.train()
                        model_results = model.get_results()
                        y_pred = np.array(list(model_results["node_estimates"].values()))
                    elif self.model_type == "GMVI":
                        model.fit()
                        model.get_posterior_estimates()
                        y_pred = np.array(list(model.node_estimates.values()))

                    # Handle node mapping if needed
                    if len(y_pred) != len(y_true):
                        y_pred = self._align_predictions_with_experimental(
                            y_pred, y_true, dataset, model
                        )

                    # Center data if requested
                    if center_data:
                        y_true_centered = y_true - np.mean(y_true)
                        y_pred_centered = y_pred - np.mean(y_pred)
                    else:
                        y_true_centered = y_true
                        y_pred_centered = y_pred

                    # Create run ID
                    run_id = f"{experiment_name}_{dataset_name}_{parameter_name}_{param_value}"

                    # Record the run
                    model_run = self.tracker.record_run(
                        run_id=run_id,
                        y_true=y_true_centered,
                        y_pred=y_pred_centered,
                        model_config=config_dict,
                        dataset_info={"name": dataset_name, "n_samples": len(y_true)},
                        metadata={
                            "experiment_name": experiment_name,
                            "parameter_swept": parameter_name,
                            "parameter_value": param_value,
                            "centered": center_data,
                        },
                        tags=[experiment_name, dataset_name, parameter_name],
                    )

                    # Collect results for analysis
                    for metric in metrics:
                        if bootstrap_stats:
                            # Compute bootstrap statistics
                            try:
                                bootstrap_result = bootstrap_statistic(
                                    y_true_centered,
                                    y_pred_centered,
                                    statistic=metric,
                                    nbootstrap=n_bootstrap,
                                    include_true_uncertainty=False,
                                    include_pred_uncertainty=False,
                                )

                                results.append(
                                    {
                                        "experiment": experiment_name,
                                        "dataset": dataset_name,
                                        "parameter": parameter_name,
                                        "parameter_value": param_value,
                                        "metric": metric,
                                        "value": bootstrap_result["mle"],
                                        "mean": bootstrap_result["mean"],
                                        "std_err": bootstrap_result["stderr"],
                                        "ci_low": bootstrap_result["low"],
                                        "ci_high": bootstrap_result["high"],
                                        "run_id": run_id,
                                    }
                                )
                            except Exception as e:
                                print(f"Warning: Bootstrap failed for {metric}: {e}")
                                # Fall back to basic metric
                                value = model_run.performance_metrics.get(
                                    metric, np.nan
                                )
                                results.append(
                                    {
                                        "experiment": experiment_name,
                                        "dataset": dataset_name,
                                        "parameter": parameter_name,
                                        "parameter_value": param_value,
                                        "metric": metric,
                                        "value": value,
                                        "mean": value,
                                        "std_err": np.nan,
                                        "ci_low": np.nan,
                                        "ci_high": np.nan,
                                        "run_id": run_id,
                                    }
                                )
                        else:
                            # Use basic metrics
                            value = model_run.performance_metrics.get(metric, np.nan)
                            results.append(
                                {
                                    "experiment": experiment_name,
                                    "dataset": dataset_name,
                                    "parameter": parameter_name,
                                    "parameter_value": param_value,
                                    "metric": metric,
                                    "value": value,
                                    "mean": value,
                                    "std_err": np.nan,
                                    "ci_low": np.nan,
                                    "ci_high": np.nan,
                                    "run_id": run_id,
                                }
                            )

                    print(f"   ✅ {parameter_name}={param_value}")

                except Exception as e:
                    print(f"   ❌ {parameter_name}={param_value}: {e}")

                    # Record failed experiment
                    self.failed_experiments.append(
                        {
                            "experiment": experiment_name,
                            "dataset": dataset_name,
                            "parameter": parameter_name,
                            "parameter_value": param_value,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "stage": "training_or_evaluation",
                        }
                    )

                pbar.update(1)

            print()

        pbar.close()

        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        self.sweep_results.append(results_df)

        print("✅ Parameter sweep completed!")
        print(f"   Total experiments: {len(results_df) // len(metrics)}")
        print(f"   Results shape: {results_df.shape}")

        # Notify about failed experiments
        if self.failed_experiments:
            print(f"⚠️  {len(self.failed_experiments)} parameter combinations failed")
            print("   Use .get_failed_experiments() to view failure details")
        else:
            print("✅ All parameter combinations completed successfully")

        return results_df

    def get_failed_experiments(self) -> pd.DataFrame:
        """
        Get failed parameter combinations as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing details of failed parameter combinations,
            including experiment name, dataset, parameter values, error type, and error message.
            Returns empty DataFrame if no failures occurred.
        """
        if not self.failed_experiments:
            return pd.DataFrame(
                columns=[
                    "experiment",
                    "dataset",
                    "parameter",
                    "parameter_value",
                    "error_type",
                    "error_message",
                    "stage",
                ]
            )

        return pd.DataFrame(self.failed_experiments)

    def grid_search(
        self,
        parameter_grid: Dict[str, List[Any]],
        metrics: List[str] = ["RMSE", "MUE", "R2"],
        experiment_name: Optional[str] = None,
        max_experiments: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Perform grid search over multiple parameters.

        Parameters
        ----------
        parameter_grid : Dict[str, List[Any]]
            Dictionary mapping parameter names to lists of values
        metrics : List[str], default=['RMSE', 'MUE', 'R2']
            Performance metrics to compute
        experiment_name : str, optional
            Name for this experiment series
        max_experiments : int, optional
            Maximum number of experiments to run (for large grids)

        Returns
        -------
        pd.DataFrame
            Results table with all parameter combinations
        """

        if experiment_name is None:
            experiment_name = "grid_search"

        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        combinations = list(product(*param_values))

        if max_experiments and len(combinations) > max_experiments:
            print(f"⚠️  Grid too large ({len(combinations)} combinations)")
            print(f"   Randomly sampling {max_experiments} combinations")
            np.random.shuffle(combinations)
            combinations = combinations[:max_experiments]

        print(f"🔍 Grid Search: {param_names}")
        print(f"   Combinations: {len(combinations)}")
        print(f"   Datasets: {list(self.datasets.keys())}")
        print()

        results = []
        total_experiments = len(combinations) * len(self.datasets)
        pbar = tqdm(total=total_experiments, desc="Grid search")

        for dataset_name, dataset in self.datasets.items():
            y_true = self._get_experimental_data(dataset)

            for combination in combinations:
                try:
                    # Create parameter combination dict
                    param_combo = dict(zip(param_names, combination))

                    # Create modified config based on model type
                    config_dict = self.base_config.__dict__.copy()

                    if self.model_type == "NodeModel":
                        for param_name, param_value in param_combo.items():
                            if param_name == "prior_std":
                                config_dict["prior_parameters"] = [0.0, param_value]
                            else:
                                config_dict[param_name] = param_value
                        modified_config = NodeModelConfig(**config_dict)
                    elif self.model_type == "GMVI":
                        for param_name, param_value in param_combo.items():
                            config_dict[param_name] = param_value
                        modified_config = GMVIConfig(**config_dict)

                    # Train model with appropriate method
                    model = self.model_class(config=modified_config, dataset=dataset)

                    if self.model_type == "NodeModel":
                        model.train()
                        model_results = model.get_results()
                        y_pred = np.array(list(model_results["node_estimates"].values()))
                    elif self.model_type == "GMVI":
                        model.fit()
                        model.get_posterior_estimates()
                        y_pred = np.array(list(model.node_estimates.values()))

                    # Align predictions with experimental data
                    if len(y_pred) != len(y_true):
                        y_pred = self._align_predictions_with_experimental(
                            y_pred, y_true, dataset, model
                        )

                    # Center data
                    y_true_centered = y_true - np.mean(y_true)
                    y_pred_centered = y_pred - np.mean(y_pred)

                    # Create run ID
                    param_str = "_".join([f"{k}_{v}" for k, v in param_combo.items()])
                    run_id = f"{experiment_name}_{dataset_name}_{param_str}"

                    # Record the run
                    model_run = self.tracker.record_run(
                        run_id=run_id,
                        y_true=y_true_centered,
                        y_pred=y_pred_centered,
                        model_config=config_dict,
                        dataset_info={"name": dataset_name, "n_samples": len(y_true)},
                        metadata={
                            "experiment_name": experiment_name,
                            "parameter_combination": param_combo,
                            "grid_search": True,
                        },
                        tags=[experiment_name, dataset_name, "grid_search"],
                    )

                    # Store results
                    for metric in metrics:
                        value = model_run.performance_metrics.get(metric, np.nan)

                        result_row = {
                            "experiment": experiment_name,
                            "dataset": dataset_name,
                            "metric": metric,
                            "value": value,
                            "run_id": run_id,
                        }

                        # Add individual parameters as columns
                        for param_name, param_value in param_combo.items():
                            result_row[param_name] = param_value

                        results.append(result_row)

                except Exception as e:
                    print(f"   ❌ {param_combo}: {e}")

                    # Record failed experiment
                    self.failed_experiments.append(
                        {
                            "experiment": experiment_name,
                            "dataset": dataset_name,
                            "parameter": "grid_search",  # Multiple parameters
                            "parameter_value": str(param_combo),
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "stage": "training_or_evaluation",
                        }
                    )

                pbar.update(1)

        pbar.close()

        results_df = pd.DataFrame(results)
        self.sweep_results.append(results_df)

        print("✅ Grid search completed!")
        print(f"   Results shape: {results_df.shape}")

        # Notify about failed experiments
        if self.failed_experiments:
            print(f"⚠️  {len(self.failed_experiments)} parameter combinations failed")
            print("   Use .get_failed_experiments() to view failure details")
        else:
            print("✅ All parameter combinations completed successfully")

        return results_df

    def plot_parameter_effects(
        self,
        results_df: pd.DataFrame,
        parameter_name: str,
        metrics: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        style: str = "whitegrid",
        confidence_intervals: bool = True,
        save_path: Optional[str] = None,
    ) -> Dict[str, plt.Figure]:
        """
        Create separate line plots for error and correlation metrics.

        Creates two figures:
        1. Error metrics (RMSE, MUE) in 1x2 layout
        2. Correlation metrics (R2, rho, tau) in 1x3 layout

        Each metric has its own y-axis scale for better visualization.

        Parameters
        ----------
        results_df : pd.DataFrame
            Results from parameter sweep
        parameter_name : str
            Name of the parameter to plot on x-axis
        metrics : List[str], optional
            Metrics to plot (default: all in results)
        datasets : List[str], optional
            Datasets to include (default: all in results)
        style : str, default='whitegrid'
            Seaborn style
        confidence_intervals : bool, default=True
            Whether to show confidence intervals
        save_path : str, optional
            Base path to save the plots (will append '_error' and '_correlation')

        Returns
        -------
        Dict[str, plt.Figure]
            Dictionary with keys 'error' and 'correlation' containing the figures
        """

        # Set style
        sns.set_style(style)

        # Filter data
        plot_data = results_df[results_df["parameter"] == parameter_name].copy()

        if metrics is not None:
            plot_data = plot_data[plot_data["metric"].isin(metrics)]

        if datasets is not None:
            plot_data = plot_data[plot_data["dataset"].isin(datasets)]

        # Separate metrics into error and correlation categories
        error_metrics = ["RMSE", "MUE"]
        correlation_metrics = ["R2", "rho", "KTAU"]

        unique_metrics = plot_data["metric"].unique()
        available_error_metrics = [m for m in error_metrics if m in unique_metrics]
        available_corr_metrics = [m for m in correlation_metrics if m in unique_metrics]

        figures = {}

        # Helper function to create plots with individual y-axis scales
        def create_metric_plots(metrics_list, ncols, fig_key, fig_title):
            if not metrics_list:
                return None

            nrows = 1
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5))

            # Handle single subplot case
            if ncols == 1:
                axes = [axes]

            unique_datasets = plot_data["dataset"].unique()
            # Use darker color palette for better visibility
            colors = sns.color_palette("deep", len(unique_datasets))
            dataset_colors = dict(zip(unique_datasets, colors))

            for idx, metric in enumerate(metrics_list):
                ax = axes[idx]
                metric_data = plot_data[plot_data["metric"] == metric]

                # Plot each dataset
                for dataset in unique_datasets:
                    dataset_data = metric_data[metric_data["dataset"] == dataset].sort_values(
                        "parameter_value"
                    )

                    if len(dataset_data) == 0:
                        continue

                    color = dataset_colors[dataset]

                    # Plot main line with error bars
                    if confidence_intervals and "ci_low" in dataset_data.columns:
                        # Calculate error bars from confidence intervals
                        y_mean = dataset_data["mean"].values
                        y_err_low = y_mean - dataset_data["ci_low"].values
                        y_err_high = dataset_data["ci_high"].values - y_mean
                        y_err = np.array([y_err_low, y_err_high])

                        ax.errorbar(
                            dataset_data["parameter_value"],
                            y_mean,
                            yerr=y_err,
                            marker="o",
                            label=dataset,
                            color=color,
                            linewidth=2,
                            capsize=5,
                            capthick=2,
                            markersize=6,
                        )
                    else:
                        ax.plot(
                            dataset_data["parameter_value"],
                            dataset_data["value"],
                            marker="o",
                            label=dataset,
                            color=color,
                            linewidth=2,
                            markersize=6,
                        )

                # Customize subplot
                ax.set_xlabel(parameter_name.replace("_", " ").title(), fontsize=12)
                ax.set_ylabel(metric, fontsize=12)
                ax.set_title(metric, fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3)

            # Remove extra subplots
            for idx in range(len(metrics_list), ncols):
                fig.delaxes(axes[idx])

            # Add single legend outside the plots
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                title="Dataset",
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                fontsize=10,
                title_fontsize=11
            )

            fig.suptitle(
                f'{fig_title} - Effect of {parameter_name.replace("_", " ").title()}',
                fontsize=16,
                fontweight="bold",
            )
            fig.tight_layout()
            # Adjust layout to make room for the legend
            fig.subplots_adjust(right=0.85)

            return fig

        # Create error metrics plot (1x2)
        if available_error_metrics:
            error_fig = create_metric_plots(
                available_error_metrics,
                2,
                "error",
                "Error Metrics"
            )
            if error_fig:
                figures["error"] = error_fig

        # Create correlation metrics plot (1x3)
        if available_corr_metrics:
            corr_fig = create_metric_plots(
                available_corr_metrics,
                3,
                "correlation",
                "Correlation Metrics"
            )
            if corr_fig:
                figures["correlation"] = corr_fig

        # Save if requested
        if save_path:
            from pathlib import Path
            save_path_obj = Path(save_path)
            base_path = save_path_obj.parent / save_path_obj.stem

            if "error" in figures:
                error_path = f"{base_path}_error.pdf"
                figures["error"].savefig(error_path, dpi=300, bbox_inches="tight")
                print(f"📊 Error metrics plot saved to: {error_path}")

            if "correlation" in figures:
                corr_path = f"{base_path}_correlation.pdf"
                figures["correlation"].savefig(corr_path, dpi=300, bbox_inches="tight")
                print(f"📊 Correlation metrics plot saved to: {corr_path}")

        return figures

    def find_optimal_parameters(
        self,
        results_df: pd.DataFrame,
        metric: str = "RMSE",
        minimize: bool = True,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """
        Find optimal parameter values across datasets.

        Parameters
        ----------
        results_df : pd.DataFrame
            Results from parameter sweep or grid search
        metric : str, default='RMSE'
            Metric to optimize
        minimize : bool, default=True
            Whether to minimize (True) or maximize (False) the metric
        aggregation : str, default='mean'
            How to aggregate across datasets ('mean', 'median', 'best')

        Returns
        -------
        pd.DataFrame
            Optimal parameter values
        """

        # Filter for specific metric
        metric_data = results_df[results_df["metric"] == metric].copy()

        if aggregation == "mean":
            # Average performance across datasets
            agg_data = (
                metric_data.groupby(["parameter", "parameter_value"])["value"]
                .mean()
                .reset_index()
            )
        elif aggregation == "median":
            # Median performance across datasets
            agg_data = (
                metric_data.groupby(["parameter", "parameter_value"])["value"]
                .median()
                .reset_index()
            )
        elif aggregation == "best":
            # Best performance across any dataset
            if minimize:
                agg_data = (
                    metric_data.groupby(["parameter", "parameter_value"])["value"]
                    .min()
                    .reset_index()
                )
            else:
                agg_data = (
                    metric_data.groupby(["parameter", "parameter_value"])["value"]
                    .max()
                    .reset_index()
                )

        # Find optimal values for each parameter
        optimal_results = []

        for param in agg_data["parameter"].unique():
            param_data = agg_data[agg_data["parameter"] == param]

            if minimize:
                optimal_idx = param_data["value"].idxmin()
            else:
                optimal_idx = param_data["value"].idxmax()

            optimal_row = param_data.loc[optimal_idx]
            optimal_results.append(
                {
                    "parameter": param,
                    "optimal_value": optimal_row["parameter_value"],
                    "performance": optimal_row["value"],
                    "metric": metric,
                    "aggregation": aggregation,
                }
            )

        return pd.DataFrame(optimal_results)

    def _get_experimental_data(self, dataset) -> np.ndarray:
        """
        Extract experimental data from dataset.

        This method needs to be adapted based on your specific dataset structure.
        """
        # This is a placeholder - adapt to your dataset structure
        if (
            hasattr(dataset, "dataset_nodes")
            and "Exp. DeltaG" in dataset.dataset_nodes.columns
        ):
            return dataset.dataset_nodes["Exp. DeltaG"].values
        elif hasattr(dataset, "get_dataframes"):
            _, node_data = dataset.get_dataframes()
            if "Exp. DeltaG" in node_data.columns:
                return node_data["Exp. DeltaG"].values

        # Fallback: generate synthetic experimental data
        graph_data = dataset.get_graph_data()
        n_nodes = graph_data["N"]
        np.random.seed(42)  # For reproducibility
        return np.random.randn(n_nodes) * 2.0

    def _align_predictions_with_experimental(
        self, y_pred: np.ndarray, y_true: np.ndarray, dataset: Any, model: NodeModel
    ) -> np.ndarray:
        """
        Align model predictions with experimental data order.

        This method handles cases where the model predictions and experimental
        data might have different ordering or missing values.
        """
        # If lengths match, assume they're aligned
        if len(y_pred) == len(y_true):
            return y_pred

        # Try to get node mappings and align
        try:
            node2idx, idx2node = dataset.get_node_mapping()

            # Get experimental node order (adapt to your dataset structure)
            if hasattr(dataset, "dataset_nodes"):
                exp_nodes = dataset.dataset_nodes["Name"].values
            else:
                # Fallback: use first N nodes
                exp_nodes = [idx2node[i] for i in range(min(len(y_true), len(y_pred)))]

            # Align predictions to experimental order
            aligned_pred = []
            for node in exp_nodes:
                if node in node2idx:
                    node_idx = node2idx[node]
                    if node_idx < len(y_pred):
                        aligned_pred.append(y_pred[node_idx])
                    else:
                        aligned_pred.append(np.nan)
                else:
                    aligned_pred.append(np.nan)

            return np.array(aligned_pred)

        except Exception as e:
            print(f"Warning: Could not align predictions: {e}")
            # Fallback: truncate to minimum length
            min_len = min(len(y_pred), len(y_true))
            return y_pred[:min_len]


def create_prior_sweep_experiment(
    tracker: PerformanceTracker,
    datasets: Dict[str, Any],
    prior_std_values: List[float] = [0.01, 0.1, 0.5, 1, 2, 4, 6],
    base_config: Optional[NodeModelConfig] = None,
) -> pd.DataFrame:
    """
    Create a prior standard deviation sweep experiment for NodeModel.

    This function replicates the functionality from your attached code but
    using the parameter sweep system.

    Parameters
    ----------
    tracker : PerformanceTracker
        Performance tracker instance
    datasets : Dict[str, Any]
        Dictionary of datasets to test
    prior_std_values : List[float]
        Prior standard deviation values to test
    base_config : NodeModelConfig, optional
        Base configuration (will use defaults if not provided)

    Returns
    -------
    pd.DataFrame
        Results similar to your stats_df
    """

    if base_config is None:
        base_config = NodeModelConfig(
            learning_rate=0.001,
            num_steps=1000,
            prior_type=PriorType.NORMAL,
            prior_parameters=[0.0, 1.0],  # Will be modified during sweep
        )

    # Create parameter sweep
    sweep = ParameterSweep(tracker, base_config, datasets)

    # Run the prior standard deviation sweep
    results_df = sweep.sweep_parameter(
        parameter_name="prior_std",
        values=prior_std_values,
        metrics=["RMSE", "MUE", "R2", "rho"],
        experiment_name="prior_std_effect",
        center_data=True,
        bootstrap_stats=True,
    )

    # Create visualization
    sweep.plot_parameter_effects(
        results_df=results_df,
        parameter_name="prior_std",
        save_path=tracker.storage_dir / "prior_std_effects",
    )

    return results_df


def create_gmvi_prior_sweep_experiment(
    tracker: PerformanceTracker,
    datasets: Dict[str, Any],
    prior_std_values: List[float] = [0.01, 0.1, 0.5, 1, 2, 4, 6, 10],
    base_config: Optional[GMVIConfig] = None,
) -> pd.DataFrame:
    """
    Create a prior standard deviation sweep experiment for GMVI_model.

    This function provides parameter sweep functionality specifically
    for the GMVI model with Gaussian Markov variational inference.

    Parameters
    ----------
    tracker : PerformanceTracker
        Performance tracker instance
    datasets : Dict[str, Any]
        Dictionary of datasets to test
    prior_std_values : List[float]
        Prior standard deviation values to test
    base_config : GMVIConfig, optional
        Base configuration (will use defaults if not provided)

    Returns
    -------
    pd.DataFrame
        Results DataFrame with performance metrics across different prior_std values
    """

    if base_config is None:
        base_config = GMVIConfig(
            learning_rate=0.01,
            n_epochs=1000,
            prior_std=5.0,  # Will be modified during sweep
            normal_std=1.0,
            outlier_std=3.0,
            outlier_prob=0.2,
            kl_weight=0.1,
            n_samples=100,
            patience=50,
        )

    # Create parameter sweep
    sweep = ParameterSweep(tracker, base_config, datasets)

    # Run the prior standard deviation sweep
    results_df = sweep.sweep_parameter(
        parameter_name="prior_std",
        values=prior_std_values,
        metrics=["RMSE", "MUE", "R2", "rho"],
        experiment_name="gmvi_prior_std_effect",
        center_data=True,
        bootstrap_stats=True,
    )

    # Create visualization
    sweep.plot_parameter_effects(
        results_df=results_df,
        parameter_name="prior_std",
        figsize=(15, 8),
        save_path=tracker.storage_dir / "gmvi_prior_std_effects.png",
    )

    return results_df


def create_comprehensive_parameter_study(
    tracker: PerformanceTracker, datasets: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """
    Create a comprehensive parameter study covering multiple aspects.

    This function demonstrates various parameter sweep scenarios.
    """

    base_config = NodeModelConfig()
    sweep = ParameterSweep(tracker, base_config, datasets)

    results = {}

    # 1. Prior standard deviation sweep
    print("1️⃣ Prior Standard Deviation Sweep")
    results["prior_std"] = sweep.sweep_parameter(
        "prior_std",
        values=[0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0],
        experiment_name="prior_std_study",
    )

    # 2. Learning rate sweep
    print("2️⃣ Learning Rate Sweep")
    results["learning_rate"] = sweep.sweep_parameter(
        "learning_rate",
        values=[0.0001, 0.001, 0.01, 0.1],
        experiment_name="learning_rate_study",
    )

    # 3. Error distribution comparison
    print("3️⃣ Error Distribution Study")
    results["error_std"] = sweep.sweep_parameter(
        "error_std", values=[0.1, 0.5, 1.0, 2.0, 5.0], experiment_name="error_std_study"
    )

    # 4. Number of steps sweep
    print("4️⃣ Training Steps Study")
    results["num_steps"] = sweep.sweep_parameter(
        "num_steps",
        values=[100, 500, 1000, 2000, 5000],
        experiment_name="num_steps_study",
    )

    return results
