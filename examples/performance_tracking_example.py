"""
Example: Model Performance Tracking and Comparison

This example demonstrates how to use the MAPLE performance tracking system
to record, store, and compare different model runs over time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from maple.dataset.synthetic_dataset import SyntheticFEPDataset
from maple.models.node_model import ModelConfig
# Import MAPLE components
from maple.utils import PerformanceTracker


def run_model_comparison_example():
    """
    Example workflow showing how to track and compare model performance.

    This function demonstrates:
    1. Setting up performance tracking
    2. Running multiple model configurations
    3. Recording performance for each run
    4. Comparing and visualizing results
    5. Exporting data for further analysis
    """

    print("🚀 Starting MAPLE Model Performance Tracking Example")
    print("=" * 60)

    # ============================================================================
    # 1. Set up performance tracking
    # ============================================================================

    # Create a performance tracker that will store results in ./model_results/
    tracker = PerformanceTracker(
        storage_dir="./model_results",
        auto_save=True,  # Automatically save after each run
    )

    print("📊 Performance tracker initialized")
    print(f"   Storage directory: {tracker.storage_dir}")
    print(f"   Existing runs: {len(tracker.runs)}")
    print()

    # ============================================================================
    # 2. Generate test datasets
    # ============================================================================

    print("🔬 Generating synthetic datasets...")

    # Create different sized datasets for testing
    datasets = {
        "small": SyntheticFEPDataset(
            n_nodes=10, edge_density=0.6, noise_level=0.1, random_seed=42
        ),
        "medium": SyntheticFEPDataset(
            n_nodes=25, edge_density=0.4, noise_level=0.1, random_seed=42
        ),
        "large": SyntheticFEPDataset(
            n_nodes=50, edge_density=0.3, noise_level=0.1, random_seed=42
        ),
    }

    for name, dataset in datasets.items():
        graph_data = dataset.get_graph_data()
        print(f"   {name}: {graph_data['N']} nodes, {graph_data['M']} edges")
    print()

    # ============================================================================
    # 3. Test different model configurations
    # ============================================================================

    model_configs = [
        {
            "name": "baseline",
            "config": ModelConfig(learning_rate=0.001, num_steps=100, error_std=1.0),
            "description": "Baseline configuration with default parameters",
        },
        {
            "name": "fast_learning",
            "config": ModelConfig(learning_rate=0.01, num_steps=50, error_std=1.0),
            "description": "Higher learning rate, fewer steps",
        },
        {
            "name": "conservative",
            "config": ModelConfig(learning_rate=0.0001, num_steps=200, error_std=0.5),
            "description": "Lower learning rate, more steps, tighter error",
        },
    ]

    print(f"🔧 Testing {len(model_configs)} model configurations")
    for config in model_configs:
        print(f"   {config['name']}: {config['description']}")
    print()

    # ============================================================================
    # 4. Run experiments and record performance
    # ============================================================================

    print("⚡ Running model experiments...")
    print()

    for dataset_name, dataset in datasets.items():
        print(f"📊 Dataset: {dataset_name}")

        # Get graph data for this dataset
        graph_data = dataset.get_graph_data()

        # Generate some "experimental" values for comparison
        # In practice, these would come from your actual experimental data
        np.random.seed(42)
        n_nodes = graph_data["N"]
        y_true = np.random.randn(n_nodes) * 2.0  # "Experimental" node values

        for config_info in model_configs:
            config_name = config_info["name"]
            model_config = config_info["config"]

            print(f"   Running {config_name}...")

            try:
                # For this example, we'll simulate model predictions
                # In practice, you would initialize and train the model:
                # model = NodeModel(model_config, dataset)
                # model_predictions = model.predict(X_test)
                model_predictions = y_true + np.random.normal(0, 0.3, len(y_true))

                # Add some systematic bias for different configs to show differences
                if config_name == "fast_learning":
                    model_predictions += 0.2  # Slight positive bias
                elif config_name == "conservative":
                    model_predictions *= 0.95  # Slight scaling

                # Record this run
                run_id = f"{dataset_name}_{config_name}"
                tracker.record_run(
                    run_id=run_id,
                    y_true=y_true,
                    y_pred=model_predictions,
                    model_config=model_config.__dict__,
                    dataset_info={
                        "name": dataset_name,
                        "n_nodes": n_nodes,
                        "n_edges": graph_data["M"],
                        "edge_density": datasets[dataset_name].edge_density,
                        "noise_level": datasets[dataset_name].noise_level,
                    },
                    metadata={
                        "description": config_info["description"],
                        "experiment_type": "synthetic_comparison",
                    },
                    tags=["synthetic", "comparison", dataset_name, config_name],
                )

                print(f"     ✅ Recorded run: {run_id}")

            except Exception as e:
                print(f"     ❌ Failed: {e}")

        print()

    # ============================================================================
    # 5. Analyze and compare results
    # ============================================================================

    print("📈 Analyzing Results")
    print("=" * 40)

    # List all runs
    all_runs = tracker.list_runs()
    print(f"Total runs recorded: {len(all_runs)}")
    print(f"Run IDs: {', '.join(all_runs)}")
    print()

    # Find best performing runs overall
    best_rmse = tracker.get_best_run(metric="RMSE", minimize=True)
    best_r2 = tracker.get_best_run(metric="R2", minimize=False)

    print("🏆 Best Performing Runs:")
    print(f"   Best RMSE: {best_rmse}")
    print(f"   Best R²:   {best_r2}")
    print()

    # Compare runs by dataset
    for dataset_name in datasets.keys():
        dataset_runs = tracker.list_runs(tags=[dataset_name])
        if dataset_runs:
            print(f"📊 {dataset_name.upper()} Dataset Comparison:")
            comparison = tracker.compare_runs(
                dataset_runs, metrics=["RMSE", "MUE", "R2", "rho"]
            )
            print(
                comparison[["run_id", "RMSE", "MUE", "R2", "rho"]].to_string(
                    index=False
                )
            )
            print()

    # ============================================================================
    # 6. Generate visualizations
    # ============================================================================

    print("📊 Generating Visualizations...")

    # Create performance comparison plots
    fig1 = tracker.plot_run_comparison(
        run_ids=all_runs, metrics=["RMSE", "MUE", "R2"], figsize=(15, 5)
    )
    fig1.suptitle("Model Performance Comparison Across All Runs")
    fig1.savefig(
        "./model_results/performance_comparison.png", dpi=300, bbox_inches="tight"
    )
    print("   ✅ Saved: performance_comparison.png")

    # Create performance trends plot
    fig2 = tracker.plot_performance_trends(metric="RMSE", figsize=(12, 6))
    fig2.savefig("./model_results/rmse_trends.png", dpi=300, bbox_inches="tight")
    print("   ✅ Saved: rmse_trends.png")

    # Create detailed scatter plots for best run
    if best_rmse:
        best_run = tracker.get_run(best_rmse)
        y_true = best_run.predictions["y_true"]
        y_pred = best_run.predictions["y_pred"]

        fig3, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.7, s=50)

        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=2)

        # Add statistics text
        rmse = best_run.performance_metrics["RMSE"]
        r2 = best_run.performance_metrics["R2"]
        ax.text(
            0.05,
            0.95,
            f"RMSE: {rmse:.3f}\nR²: {r2:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Best Model Predictions: {best_rmse}")
        ax.grid(True, alpha=0.3)

        fig3.savefig(
            "./model_results/best_model_predictions.png", dpi=300, bbox_inches="tight"
        )
        print("   ✅ Saved: best_model_predictions.png")

        plt.close("all")

    # ============================================================================
    # 7. Export data for external analysis
    # ============================================================================

    print("\n💾 Exporting Data...")

    # Export to different formats
    csv_file = tracker.export_data(format="csv")
    json_file = tracker.export_data(format="json")

    print(f"   ✅ CSV export: {Path(csv_file).name}")
    print(f"   ✅ JSON export: {Path(json_file).name}")

    # ============================================================================
    # 8. Summary and recommendations
    # ============================================================================

    print("\n" + "=" * 60)
    print("📋 SUMMARY AND RECOMMENDATIONS")
    print("=" * 60)

    # Performance summary
    all_comparison = tracker.compare_runs(all_runs, metrics=["RMSE", "MUE", "R2"])

    print("\n🎯 Performance Summary:")
    print(
        f"   Average RMSE: {all_comparison['RMSE'].mean():.3f} ± {all_comparison['RMSE'].std():.3f}"
    )
    print(
        f"   Average R²:   {all_comparison['R2'].mean():.3f} ± {all_comparison['R2'].std():.3f}"
    )
    print(f"   Best RMSE:    {all_comparison['RMSE'].min():.3f} ({best_rmse})")
    print(f"   Best R²:      {all_comparison['R2'].max():.3f} ({best_r2})")

    # Configuration analysis
    print("\n🔧 Configuration Analysis:")
    config_performance = {}
    for config_info in model_configs:
        config_name = config_info["name"]
        config_runs = [run_id for run_id in all_runs if config_name in run_id]
        if config_runs:
            config_comparison = tracker.compare_runs(config_runs, metrics=["RMSE"])
            avg_rmse = config_comparison["RMSE"].mean()
            config_performance[config_name] = avg_rmse
            print(f"   {config_name}: Average RMSE = {avg_rmse:.3f}")

    best_config = min(config_performance.items(), key=lambda x: x[1])
    print(f"   🏆 Best configuration: {best_config[0]} (RMSE: {best_config[1]:.3f})")

    print(f"\n📁 All results saved to: {tracker.storage_dir}")
    print(
        f"📊 Use PerformanceTracker.load_performance_history('{tracker.storage_dir}') to reload data"
    )

    print("\n✨ Example completed successfully!")

    return tracker


def demonstrate_advanced_features():
    """Demonstrate advanced features of the performance tracking system."""

    print("\n🔬 Advanced Features Demonstration")
    print("=" * 50)

    # Load existing data
    tracker = PerformanceTracker("./model_results", auto_save=False)

    if not tracker.runs:
        print("No existing data found. Run the main example first.")
        return

    # Demonstrate filtering by tags
    print("🏷️  Filtering by Tags:")
    synthetic_runs = tracker.list_runs(tags=["synthetic"])
    baseline_runs = tracker.list_runs(tags=["baseline"])
    print(f"   Synthetic runs: {len(synthetic_runs)}")
    print(f"   Baseline runs: {len(baseline_runs)}")

    # Demonstrate detailed run information
    if synthetic_runs:
        sample_run = tracker.get_run(synthetic_runs[0])
        print("\n📋 Sample Run Details:")
        print(sample_run.get_summary())

    # Demonstrate time-based analysis
    print("\n⏱️  Performance Over Time:")
    fig = tracker.plot_performance_trends(metric="R2", figsize=(10, 6))
    fig.savefig("./model_results/r2_trends.png", dpi=300, bbox_inches="tight")
    print("   ✅ Saved R² trends plot")
    plt.close(fig)

    print("\n✨ Advanced features demonstration completed!")


if __name__ == "__main__":
    # Run the main example
    tracker = run_model_comparison_example()

    # Demonstrate advanced features
    demonstrate_advanced_features()

    print("\n🎉 All examples completed!")
    print("Check the './model_results/' directory for all generated files.")
