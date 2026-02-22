"""
Performance Tracking and Model Comparison Utilities

This module provides comprehensive tools for tracking model performance,
storing results, and comparing different model runs over time.
"""

import json
import pickle
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..graph_analysis.performance_stats import (bootstrap_statistic,
                                                compute_simple_statistics)


@dataclass
class ModelRun:
    """
    Container for a single model run's performance data.

    This class stores all relevant information about a model run including
    configuration, performance metrics, and metadata for easy comparison
    and persistence.

    Attributes
    ----------
    run_id : str
        Unique identifier for this model run
    timestamp : str
        ISO timestamp when the run was executed
    model_config : Dict[str, Any]
        Configuration parameters used for the model
    dataset_info : Dict[str, Any]
        Information about the dataset used
    performance_metrics : Dict[str, float]
        Basic performance statistics (RMSE, MUE, R2, etc.)
    bootstrap_metrics : Dict[str, Dict[str, float]]
        Bootstrap statistics with confidence intervals
    predictions : Dict[str, np.ndarray]
        Arrays of true and predicted values
    metadata : Dict[str, Any]
        Additional metadata (notes, tags, etc.)
    """

    run_id: str
    timestamp: str
    model_config: Dict[str, Any]
    dataset_info: Dict[str, Any]
    performance_metrics: Dict[str, float]
    bootstrap_metrics: Dict[str, Dict[str, float]]
    predictions: Dict[str, np.ndarray]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelRun to dictionary for serialization."""
        data = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        data["predictions"] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in data["predictions"].items()
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRun":
        """Create ModelRun from dictionary."""
        # Convert lists back to numpy arrays
        data["predictions"] = {
            key: np.array(value) if isinstance(value, list) else value
            for key, value in data["predictions"].items()
        }
        return cls(**data)

    def get_summary(self) -> str:
        """Get a human-readable summary of the model run."""
        return f"""
Model Run Summary
=================
Run ID: {self.run_id}
Timestamp: {self.timestamp}
Dataset: {self.dataset_info.get('name', 'Unknown')}
N Samples: {len(self.predictions.get('y_true', []))}

Performance Metrics:
- RMSE: {self.performance_metrics.get('RMSE', 'N/A'):.3f}
- MUE:  {self.performance_metrics.get('MUE', 'N/A'):.3f}
- R²:   {self.performance_metrics.get('R2', 'N/A'):.3f}
- ρ:    {self.performance_metrics.get('rho', 'N/A'):.3f}

Model Configuration:
{json.dumps(self.model_config, indent=2)}
        """.strip()


class PerformanceTracker:
    """
    Comprehensive performance tracking system for MAPLE models.

    This class provides functionality to:
    - Track model performance across multiple runs
    - Store results persistently
    - Compare different model configurations
    - Generate performance reports and visualizations
    - Export data for external analysis

    Parameters
    ----------
    storage_dir : str or Path
        Directory to store performance data
    auto_save : bool, default=True
        Whether to automatically save data after each run

    Examples
    --------
    >>> tracker = PerformanceTracker("./model_results")
    >>>
    >>> # Record a model run
    >>> tracker.record_run(
    ...     run_id="baseline_v1",
    ...     y_true=y_true,
    ...     y_pred=y_pred,
    ...     model_config={"prior_type": "normal", "learning_rate": 0.001},
    ...     dataset_info={"name": "synthetic_100", "n_nodes": 100}
    ... )
    >>>
    >>> # Compare multiple runs
    >>> comparison = tracker.compare_runs(["baseline_v1", "improved_v2"])
    >>> print(comparison)
    """

    def __init__(self, storage_dir: Union[str, Path], auto_save: bool = True):
        """Initialize the performance tracker."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save

        # Storage files
        self.runs_file = self.storage_dir / "model_runs.json"
        self.data_file = self.storage_dir / "performance_data.pkl"

        # In-memory storage
        self.runs: Dict[str, ModelRun] = {}

        # Load existing data
        self._load_data()

    def record_run(
        self,
        run_id: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_config: Dict[str, Any],
        dataset_info: Dict[str, Any],
        y_true_err: Optional[np.ndarray] = None,
        y_pred_err: Optional[np.ndarray] = None,
        bootstrap_stats: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> ModelRun:
        """
        Record a complete model run with performance statistics.

        Parameters
        ----------
        run_id : str
            Unique identifier for this run
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        model_config : Dict[str, Any]
            Model configuration parameters
        dataset_info : Dict[str, Any]
            Dataset information
        y_true_err : np.ndarray, optional
            Uncertainties in true values
        y_pred_err : np.ndarray, optional
            Uncertainties in predicted values
        bootstrap_stats : List[str], optional
            Statistics to compute with bootstrap (default: ['RMSE', 'MUE', 'R2'])
        metadata : Dict[str, Any], optional
            Additional metadata
        tags : List[str], optional
            Tags for categorizing runs

        Returns
        -------
        ModelRun
            The created model run object
        """

        if run_id in self.runs:
            warnings.warn(
                f"Run ID '{run_id}' already exists. Overwriting previous run."
            )

        # Default bootstrap statistics
        if bootstrap_stats is None:
            bootstrap_stats = ["RMSE", "MUE", "R2", "rho"]

        # Default metadata
        if metadata is None:
            metadata = {}

        # Add tags to metadata
        if tags:
            metadata["tags"] = tags

        # Compute basic performance metrics
        performance_metrics = compute_simple_statistics(y_true, y_pred)

        # Compute bootstrap statistics with confidence intervals
        bootstrap_metrics = {}
        for stat in bootstrap_stats:
            try:
                bootstrap_result = bootstrap_statistic(
                    y_true,
                    y_pred,
                    dy_true=y_true_err,
                    dy_pred=y_pred_err,
                    statistic=stat,
                    nbootstrap=1000,
                    include_true_uncertainty=y_true_err is not None,
                    include_pred_uncertainty=y_pred_err is not None,
                )
                bootstrap_metrics[stat] = bootstrap_result
            except Exception as e:
                print(f"Warning: Could not compute bootstrap {stat}: {e}")
                bootstrap_metrics[stat] = {
                    "mle": np.nan,
                    "mean": np.nan,
                    "stderr": np.nan,
                    "low": np.nan,
                    "high": np.nan,
                }

        # Create model run object
        model_run = ModelRun(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            model_config=model_config,
            dataset_info=dataset_info,
            performance_metrics=performance_metrics,
            bootstrap_metrics=bootstrap_metrics,
            predictions={
                "y_true": y_true,
                "y_pred": y_pred,
                "y_true_err": y_true_err if y_true_err is not None else np.array([]),
                "y_pred_err": y_pred_err if y_pred_err is not None else np.array([]),
            },
            metadata=metadata,
        )

        # Store the run
        self.runs[run_id] = model_run

        # Auto-save if enabled
        if self.auto_save:
            self.save_data()

        print(f"✅ Recorded model run '{run_id}' with {len(y_true)} samples")
        return model_run

    def get_run(self, run_id: str) -> Optional[ModelRun]:
        """Get a specific model run by ID."""
        return self.runs.get(run_id)

    def list_runs(self, tags: Optional[List[str]] = None) -> List[str]:
        """
        List all run IDs, optionally filtered by tags.

        Parameters
        ----------
        tags : List[str], optional
            Filter runs that have all specified tags

        Returns
        -------
        List[str]
            List of run IDs
        """
        if tags is None:
            return list(self.runs.keys())

        filtered_runs = []
        for run_id, run in self.runs.items():
            run_tags = run.metadata.get("tags", [])
            if all(tag in run_tags for tag in tags):
                filtered_runs.append(run_id)

        return filtered_runs

    def compare_runs(
        self, run_ids: List[str], metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare performance metrics across multiple runs.

        Parameters
        ----------
        run_ids : List[str]
            List of run IDs to compare
        metrics : List[str], optional
            Metrics to include in comparison (default: all available)

        Returns
        -------
        pd.DataFrame
            Comparison table with runs as rows and metrics as columns
        """
        if metrics is None:
            # Get all available metrics from first run
            if run_ids and run_ids[0] in self.runs:
                metrics = list(self.runs[run_ids[0]].performance_metrics.keys())
            else:
                metrics = ["RMSE", "MUE", "R2", "rho", "KTAU"]

        comparison_data = []
        for run_id in run_ids:
            if run_id not in self.runs:
                print(f"Warning: Run '{run_id}' not found")
                continue

            run = self.runs[run_id]
            row_data = {"run_id": run_id, "timestamp": run.timestamp}

            # Add performance metrics
            for metric in metrics:
                row_data[metric] = run.performance_metrics.get(metric, np.nan)

            # Add dataset info
            row_data["dataset"] = run.dataset_info.get("name", "Unknown")
            row_data["n_samples"] = len(run.predictions.get("y_true", []))

            # Add key config parameters
            for key, value in run.model_config.items():
                row_data[f"config_{key}"] = value

            comparison_data.append(row_data)

        return pd.DataFrame(comparison_data)

    def get_best_run(
        self, metric: str = "RMSE", minimize: bool = True
    ) -> Optional[str]:
        """
        Find the best performing run based on a specific metric.

        Parameters
        ----------
        metric : str, default='RMSE'
            Metric to optimize
        minimize : bool, default=True
            Whether to minimize (True) or maximize (False) the metric

        Returns
        -------
        str or None
            Run ID of the best performing run
        """
        if not self.runs:
            return None

        best_run_id = None
        best_value = float("inf") if minimize else float("-inf")

        for run_id, run in self.runs.items():
            value = run.performance_metrics.get(metric, np.nan)
            if np.isnan(value):
                continue

            if minimize and value < best_value:
                best_value = value
                best_run_id = run_id
            elif not minimize and value > best_value:
                best_value = value
                best_run_id = run_id

        return best_run_id

    def plot_performance_trends(
        self,
        metric: str = "RMSE",
        run_ids: Optional[List[str]] = None,
        figsize: tuple = (12, 6),
    ) -> plt.Figure:
        """
        Plot performance trends over time.

        Parameters
        ----------
        metric : str, default='RMSE'
            Metric to plot
        run_ids : List[str], optional
            Specific runs to include (default: all runs)
        figsize : tuple, default=(12, 6)
            Figure size

        Returns
        -------
        plt.Figure
            The created figure
        """
        if run_ids is None:
            run_ids = list(self.runs.keys())

        # Prepare data
        timestamps = []
        values = []
        labels = []

        for run_id in run_ids:
            if run_id not in self.runs:
                continue

            run = self.runs[run_id]
            value = run.performance_metrics.get(metric, np.nan)

            if not np.isnan(value):
                timestamps.append(pd.to_datetime(run.timestamp))
                values.append(value)
                labels.append(run_id)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, values, labels))
        timestamps_sorted, values_sorted, labels_sorted = zip(*sorted_data)

        ax.plot(timestamps_sorted, values_sorted, "o-", linewidth=2, markersize=8)

        # Add labels
        for i, (timestamp, value, label) in enumerate(sorted_data):
            ax.annotate(
                label,
                (timestamp, value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        ax.set_xlabel("Time")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} Performance Over Time")
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_run_comparison(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
        figsize: tuple = (10, 6),
    ) -> plt.Figure:
        """
        Create a bar plot comparing multiple runs across metrics.

        Parameters
        ----------
        run_ids : List[str]
            Runs to compare
        metrics : List[str], optional
            Metrics to include (default: ['RMSE', 'MUE', 'R2'])
        figsize : tuple, default=(10, 6)
            Figure size

        Returns
        -------
        plt.Figure
            The created figure
        """
        if metrics is None:
            metrics = ["RMSE", "MUE", "R2"]

        # Prepare data
        comparison_df = self.compare_runs(run_ids, metrics)

        # Create subplots
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharey=False)
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            values = comparison_df[metric]
            run_labels = comparison_df["run_id"]

            bars = ax.bar(range(len(values)), values)
            ax.set_title(f"{metric}")
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(run_labels, rotation=45, ha="right")

            # Color bars based on performance (lower is better for RMSE/MUE)
            if metric in ["RMSE", "MUE"]:
                colors = plt.cm.RdYlGn_r(values / values.max())
            else:  # R2, rho - higher is better
                colors = plt.cm.RdYlGn(values / values.max())

            for bar, color in zip(bars, colors):
                bar.set_color(color)

        plt.tight_layout()
        return fig

    def export_data(self, filename: Optional[str] = None, format: str = "csv") -> str:
        """
        Export performance data to external format.

        Parameters
        ----------
        filename : str, optional
            Output filename (default: auto-generated)
        format : str, default='csv'
            Export format ('csv', 'json', 'excel')

        Returns
        -------
        str
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_performance_{timestamp}.{format}"

        output_path = self.storage_dir / filename

        # Create comprehensive export data
        export_data = []
        for run_id, run in self.runs.items():
            row = {
                "run_id": run_id,
                "timestamp": run.timestamp,
                "dataset_name": run.dataset_info.get("name", "Unknown"),
                "n_samples": len(run.predictions.get("y_true", [])),
            }

            # Add performance metrics
            row.update(run.performance_metrics)

            # Add bootstrap confidence intervals
            for stat, bootstrap_data in run.bootstrap_metrics.items():
                row[f"{stat}_ci_low"] = bootstrap_data.get("low", np.nan)
                row[f"{stat}_ci_high"] = bootstrap_data.get("high", np.nan)
                row[f"{stat}_stderr"] = bootstrap_data.get("stderr", np.nan)

            # Add flattened config
            for key, value in run.model_config.items():
                row[f"config_{key}"] = value

            # Add metadata
            for key, value in run.metadata.items():
                if key != "tags":  # Handle tags separately
                    row[f"meta_{key}"] = value

            if "tags" in run.metadata:
                row["tags"] = ",".join(run.metadata["tags"])

            export_data.append(row)

        df = pd.DataFrame(export_data)

        # Export in requested format
        if format.lower() == "csv":
            df.to_csv(output_path, index=False)
        elif format.lower() == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif format.lower() == "excel":
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"✅ Exported data to {output_path}")
        return str(output_path)

    def save_data(self):
        """Save all data to disk."""
        # Save runs metadata as JSON
        runs_dict = {}
        for run_id, run in self.runs.items():
            runs_dict[run_id] = run.to_dict()

        with open(self.runs_file, "w") as f:
            json.dump(runs_dict, f, indent=2)

        # Save full data (including numpy arrays) as pickle
        with open(self.data_file, "wb") as f:
            pickle.dump(self.runs, f)

        print(f"💾 Saved {len(self.runs)} runs to {self.storage_dir}")

    def _load_data(self):
        """Load existing data from disk."""
        # Try to load from pickle first (includes numpy arrays)
        if self.data_file.exists():
            try:
                with open(self.data_file, "rb") as f:
                    self.runs = pickle.load(f)
                print(f"📂 Loaded {len(self.runs)} runs from {self.data_file}")
                return
            except Exception as e:
                print(f"Warning: Could not load pickle data: {e}")

        # Fallback to JSON (without numpy arrays)
        if self.runs_file.exists():
            try:
                with open(self.runs_file, "r") as f:
                    runs_dict = json.load(f)

                self.runs = {}
                for run_id, run_data in runs_dict.items():
                    self.runs[run_id] = ModelRun.from_dict(run_data)

                print(f"📂 Loaded {len(self.runs)} runs from {self.runs_file}")
            except Exception as e:
                print(f"Warning: Could not load JSON data: {e}")


def load_performance_history(storage_dir: Union[str, Path]) -> PerformanceTracker:
    """
    Load an existing performance tracking history.

    Parameters
    ----------
    storage_dir : str or Path
        Directory containing saved performance data

    Returns
    -------
    PerformanceTracker
        Loaded performance tracker
    """
    return PerformanceTracker(storage_dir, auto_save=False)


def compare_model_runs(
    runs: List[ModelRun],
    metrics: Optional[List[str]] = None,
    save_plot: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare multiple model runs and optionally create visualization.

    Parameters
    ----------
    runs : List[ModelRun]
        List of model runs to compare
    metrics : List[str], optional
        Metrics to compare (default: all available)
    save_plot : str, optional
        Path to save comparison plot

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    # Create temporary tracker for comparison functionality
    tracker = PerformanceTracker(
        storage_dir=Path.cwd() / "temp_comparison", auto_save=False
    )

    # Add runs to tracker
    for run in runs:
        tracker.runs[run.run_id] = run

    # Generate comparison
    comparison_df = tracker.compare_runs([run.run_id for run in runs], metrics)

    # Create plot if requested
    if save_plot:
        fig = tracker.plot_run_comparison([run.run_id for run in runs], metrics)
        fig.savefig(save_plot, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"📊 Saved comparison plot to {save_plot}")

    return comparison_df
