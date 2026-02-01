"""
Quick Performance Tracking Demo

This script demonstrates the core functionality of MAPLE's performance
tracking system with a simple, runnable example.
"""

import os

import numpy as np

from maple.utils import PerformanceTracker

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Handle OpenMP issue


def quick_demo():
    """
    Quick demonstration of performance tracking functionality.
    """

    print("🚀 MAPLE Performance Tracking Quick Demo")
    print("=" * 50)

    # Initialize tracker
    tracker = PerformanceTracker("./demo_results")
    print("📊 Performance tracker initialized")

    # Simulate some model runs with different performance
    print("\n⚡ Simulating model runs...")

    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 50

    # "True" experimental values
    y_true = np.random.randn(n_samples) * 2.0

    # Simulate different model predictions
    models = {
        "baseline": {"bias": 0.2, "noise": 0.4, "lr": 0.001},
        "improved": {"bias": 0.05, "noise": 0.25, "lr": 0.01},
        "best": {"bias": 0.01, "noise": 0.15, "lr": 0.005},
    }

    for model_name, params in models.items():
        # Simulate model predictions with different quality
        y_pred = (
            y_true + params["bias"] + np.random.normal(0, params["noise"], n_samples)
        )

        # Record the run
        tracker.record_run(
            run_id=f"demo_{model_name}",
            y_true=y_true,
            y_pred=y_pred,
            model_config={
                "learning_rate": params["lr"],
                "model_type": model_name,
                "num_steps": 1000,
            },
            dataset_info={
                "name": "demo_synthetic",
                "n_samples": n_samples,
                "type": "synthetic",
            },
            metadata={"demo": True, "quality": model_name},
            tags=["demo", model_name],
        )

        print(f"   ✅ Recorded: {model_name}")

    print("\n📈 Analysis Results:")
    print("-" * 30)

    # Compare all runs
    comparison = tracker.compare_runs(
        run_ids=["demo_baseline", "demo_improved", "demo_best"],
        metrics=["RMSE", "MUE", "R2", "rho"],
    )

    print("Model Comparison:")
    print(
        comparison[["run_id", "RMSE", "MUE", "R2", "rho"]]
        .round(3)
        .to_string(index=False)
    )

    # Find best model
    best_model = tracker.get_best_run(metric="RMSE", minimize=True)
    print(f"\n🏆 Best performing model: {best_model}")

    # Show detailed info for best model
    best_run = tracker.get_run(best_model)
    print(f"   RMSE: {best_run.performance_metrics['RMSE']:.3f}")
    print(f"   R²:   {best_run.performance_metrics['R2']:.3f}")
    print(f"   Config: {best_run.model_config}")

    # Export results
    csv_file = tracker.export_data(filename="demo_results.csv", format="csv")
    print(f"\n💾 Results exported to: {csv_file}")

    print(f"\n📁 All data saved to: {tracker.storage_dir}")
    print("✨ Demo completed successfully!")

    return tracker


if __name__ == "__main__":
    tracker = quick_demo()

    print("\n" + "=" * 50)
    print("🔍 Try these commands to explore further:")
    print("=" * 50)
    print("# Load the data")
    print("from maple.utils import PerformanceTracker")
    print("tracker = PerformanceTracker('./demo_results')")
    print()
    print("# List all runs")
    print("print(tracker.list_runs())")
    print()
    print("# Compare specific runs")
    print("comparison = tracker.compare_runs(['demo_baseline', 'demo_best'])")
    print("print(comparison)")
    print()
    print("# Create visualizations")
    print(
        "fig = tracker.plot_run_comparison("
        "['demo_baseline', 'demo_improved', 'demo_best'])"
    )
    print("fig.show()")
