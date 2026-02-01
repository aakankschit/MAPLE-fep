"""
Parameter Optimization and Sweep Example

This example demonstrates how to systematically explore parameter spaces,
find optimal configurations, and visualize parameter effects across datasets.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

from maple.dataset.synthetic_dataset import SyntheticFEPDataset
from maple.models.node_model import ModelConfig, PriorType
from maple.utils import PerformanceTracker
from maple.utils.parameter_sweep import (ParameterSweep,
                                         create_prior_sweep_experiment)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main_parameter_study():
    """
    Main parameter study demonstrating systematic parameter exploration.
    """

    print("🔬 MAPLE Parameter Optimization Study")
    print("=" * 60)

    # ============================================================================
    # 1. Setup - Create tracker and datasets
    # ============================================================================

    tracker = PerformanceTracker("./parameter_study_results")

    # Create multiple synthetic datasets with different characteristics
    datasets = {
        "small_dense": SyntheticFEPDataset(
            n_nodes=15, edge_density=0.8, noise_level=0.1, random_seed=42
        ),
        "medium_sparse": SyntheticFEPDataset(
            n_nodes=30, edge_density=0.3, noise_level=0.15, random_seed=123
        ),
        "large_moderate": SyntheticFEPDataset(
            n_nodes=50, edge_density=0.5, noise_level=0.2, random_seed=456
        ),
    }

    print(f"📊 Created {len(datasets)} test datasets:")
    for name, dataset in datasets.items():
        graph_data = dataset.get_graph_data()
        print(f"   {name}: {graph_data['N']} nodes, {graph_data['M']} edges")
    print()

    # ============================================================================
    # 2. Prior Standard Deviation Study (like your notebook)
    # ============================================================================

    print("1️⃣ Prior Standard Deviation Study")
    print("-" * 40)

    # Use the convenience function that replicates your notebook code
    prior_results = create_prior_sweep_experiment(
        tracker=tracker,
        datasets=datasets,
        prior_std_values=[0.01, 0.1, 0.5, 1, 2, 4, 6],
        base_config=ModelConfig(
            learning_rate=0.001, num_steps=1000, prior_type=PriorType.NORMAL
        ),
    )

    print(f"✅ Prior study completed: {len(prior_results)} results")
    print()

    # ============================================================================
    # 3. Learning Rate Optimization
    # ============================================================================

    print("2️⃣ Learning Rate Optimization")
    print("-" * 40)

    base_config = ModelConfig(prior_parameters=[0.0, 1.0])  # Fixed prior
    sweep = ParameterSweep(tracker, base_config, datasets)

    lr_results = sweep.sweep_parameter(
        parameter_name="learning_rate",
        values=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
        metrics=["RMSE", "MUE", "R2", "rho"],
        experiment_name="learning_rate_optimization",
    )

    # Create visualization
    fig_lr = sweep.plot_parameter_effects(
        results_df=lr_results,
        parameter_name="learning_rate",
        save_path="./parameter_study_results/learning_rate_effects.png",
    )
    plt.close(fig_lr)

    print(f"✅ Learning rate study completed: {len(lr_results)} results")
    print()

    # ============================================================================
    # 4. Error Model Investigation
    # ============================================================================

    print("3️⃣ Error Model Investigation")
    print("-" * 40)

    error_results = sweep.sweep_parameter(
        parameter_name="error_std",
        values=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        metrics=["RMSE", "MUE", "R2"],
        experiment_name="error_model_study",
    )

    # Create visualization
    fig_error = sweep.plot_parameter_effects(
        results_df=error_results,
        parameter_name="error_std",
        save_path="./parameter_study_results/error_std_effects.png",
    )
    plt.close(fig_error)

    print(f"✅ Error model study completed: {len(error_results)} results")
    print()

    # ============================================================================
    # 5. Multi-parameter Grid Search
    # ============================================================================

    print("4️⃣ Multi-parameter Grid Search")
    print("-" * 40)

    # Define a focused grid around promising regions
    parameter_grid = {
        "learning_rate": [0.001, 0.005, 0.01],
        "error_std": [0.5, 1.0, 2.0],
        "num_steps": [500, 1000, 2000],
    }

    grid_results = sweep.grid_search(
        parameter_grid=parameter_grid,
        metrics=["RMSE", "R2"],
        experiment_name="focused_grid_search",
    )

    print(f"✅ Grid search completed: {len(grid_results)} results")
    print()

    # ============================================================================
    # 6. Analysis and Optimization
    # ============================================================================

    print("📈 Analysis and Optimization")
    print("=" * 50)

    # Find optimal parameters for each study
    print("🎯 Optimal Parameters Found:")
    print()

    # Prior standard deviation optimum
    prior_optimal = sweep.find_optimal_parameters(
        prior_results, metric="RMSE", minimize=True, aggregation="mean"
    )
    print("Prior Standard Deviation Study:")
    for _, row in prior_optimal.iterrows():
        print(
            f"   Best {row['parameter']}: {row['optimal_value']} (RMSE: {row['performance']:.3f})"
        )
    print()

    # Learning rate optimum
    lr_optimal = sweep.find_optimal_parameters(
        lr_results, metric="RMSE", minimize=True, aggregation="mean"
    )
    print("Learning Rate Study:")
    for _, row in lr_optimal.iterrows():
        print(
            f"   Best {row['parameter']}: {row['optimal_value']} (RMSE: {row['performance']:.3f})"
        )
    print()

    # Error model optimum
    error_optimal = sweep.find_optimal_parameters(
        error_results, metric="RMSE", minimize=True, aggregation="mean"
    )
    print("Error Model Study:")
    for _, row in error_optimal.iterrows():
        print(
            f"   Best {row['parameter']}: {row['optimal_value']} (RMSE: {row['performance']:.3f})"
        )
    print()

    # Grid search optimum
    grid_optimal = sweep.find_optimal_parameters(
        grid_results, metric="RMSE", minimize=True, aggregation="mean"
    )
    print("Grid Search Optimization:")
    for _, row in grid_optimal.iterrows():
        print(
            f"   Best {row['parameter']}: {row['optimal_value']} (RMSE: {row['performance']:.3f})"
        )
    print()

    # ============================================================================
    # 7. Cross-dataset Performance Analysis
    # ============================================================================

    print("🔍 Cross-dataset Performance Analysis")
    print("-" * 50)

    # Analyze which datasets are easiest/hardest to predict
    all_results = tracker.compare_runs(
        tracker.list_runs(tags=["prior_std_optimization"])[:10],  # Sample of runs
        metrics=["RMSE", "R2"],
    )

    if len(all_results) > 0:
        dataset_performance = all_results.groupby("dataset")[["RMSE", "R2"]].agg(
            ["mean", "std"]
        )
        print("Average Performance by Dataset:")
        print(dataset_performance.round(3))
        print()

    # ============================================================================
    # 8. Create Comprehensive Summary Plot
    # ============================================================================

    print("📊 Creating Summary Visualizations")
    print("-" * 40)

    # Create a comprehensive summary plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Prior std effects (similar to your notebook)
    ax1 = axes[0, 0]
    for dataset in datasets.keys():
        dataset_data = prior_results[prior_results["dataset"] == dataset]
        rmse_data = dataset_data[dataset_data["metric"] == "RMSE"]
        ax1.plot(
            rmse_data["parameter_value"],
            rmse_data["mean"],
            marker="o",
            label=dataset,
            linewidth=2,
        )

        # Add confidence intervals if available
        if "ci_low" in rmse_data.columns:
            ax1.fill_between(
                rmse_data["parameter_value"],
                rmse_data["ci_low"],
                rmse_data["ci_high"],
                alpha=0.3,
            )

    ax1.set_xlabel("Prior Standard Deviation")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Prior Std Effect on RMSE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Learning rate effects
    ax2 = axes[0, 1]
    for dataset in datasets.keys():
        dataset_data = lr_results[lr_results["dataset"] == dataset]
        rmse_data = dataset_data[dataset_data["metric"] == "RMSE"]
        ax2.semilogx(
            rmse_data["parameter_value"],
            rmse_data["mean"],
            marker="o",
            label=dataset,
            linewidth=2,
        )

    ax2.set_xlabel("Learning Rate")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Learning Rate Effect on RMSE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Error std effects
    ax3 = axes[1, 0]
    for dataset in datasets.keys():
        dataset_data = error_results[error_results["dataset"] == dataset]
        rmse_data = dataset_data[dataset_data["metric"] == "RMSE"]
        ax3.plot(
            rmse_data["parameter_value"],
            rmse_data["mean"],
            marker="o",
            label=dataset,
            linewidth=2,
        )

    ax3.set_xlabel("Error Standard Deviation")
    ax3.set_ylabel("RMSE")
    ax3.set_title("Error Model Effect on RMSE")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Best parameter combinations heatmap
    ax4 = axes[1, 1]
    if len(grid_results) > 0:
        # Create heatmap of grid search results
        pivot_data = grid_results[grid_results["metric"] == "RMSE"].pivot_table(
            values="value", index="learning_rate", columns="error_std", aggfunc="mean"
        )

        sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="viridis_r", ax=ax4)
        ax4.set_title("Grid Search RMSE Results\n(Learning Rate vs Error Std)")
    else:
        ax4.text(
            0.5,
            0.5,
            "Grid search results\nnot available",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("Grid Search Results")

    plt.tight_layout()
    plt.savefig(
        "./parameter_study_results/comprehensive_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("   ✅ Saved: comprehensive_summary.png")

    # ============================================================================
    # 9. Export Results
    # ============================================================================

    print("\n💾 Exporting Results")
    print("-" * 30)

    # Export all sweep results
    prior_results.to_csv(
        "./parameter_study_results/prior_std_sweep_results.csv", index=False
    )
    lr_results.to_csv(
        "./parameter_study_results/learning_rate_sweep_results.csv", index=False
    )
    error_results.to_csv(
        "./parameter_study_results/error_std_sweep_results.csv", index=False
    )
    grid_results.to_csv(
        "./parameter_study_results/grid_search_results.csv", index=False
    )

    # Export optimization summary
    optimization_summary = {
        "prior_std_optimal": prior_optimal.to_dict("records"),
        "learning_rate_optimal": lr_optimal.to_dict("records"),
        "error_std_optimal": error_optimal.to_dict("records"),
        "grid_search_optimal": grid_optimal.to_dict("records"),
    }

    import json

    with open("./parameter_study_results/optimization_summary.json", "w") as f:
        json.dump(optimization_summary, f, indent=2)

    # Export performance tracker data
    tracker.export_data(filename="complete_parameter_study.csv", format="csv")

    print("   ✅ Exported all results to CSV files")
    print("   ✅ Exported optimization summary to JSON")
    print("   ✅ Exported complete tracker data")

    # ============================================================================
    # 10. Summary and Recommendations
    # ============================================================================

    print("\n" + "=" * 60)
    print("📋 STUDY SUMMARY AND RECOMMENDATIONS")
    print("=" * 60)

    print("\n🎯 Key Findings:")

    # Summarize key findings
    if len(prior_optimal) > 0:
        best_prior = prior_optimal.iloc[0]
        print(
            f"   • Optimal prior std: {best_prior['optimal_value']} "
            f"(RMSE: {best_prior['performance']:.3f})"
        )

    if len(lr_optimal) > 0:
        best_lr = lr_optimal.iloc[0]
        print(
            f"   • Optimal learning rate: {best_lr['optimal_value']} "
            f"(RMSE: {best_lr['performance']:.3f})"
        )

    if len(error_optimal) > 0:
        best_error = error_optimal.iloc[0]
        print(
            f"   • Optimal error std: {best_error['optimal_value']} "
            f"(RMSE: {best_error['performance']:.3f})"
        )

    print("\n📊 Experiments Completed:")
    print(f"   • Total model runs: {len(tracker.runs)}")
    print(f"   • Datasets tested: {len(datasets)}")
    print(
        f"   • Parameter combinations: {len(grid_results) // len(datasets) if len(grid_results) > 0 else 'N/A'}"
    )

    print("\n📁 Results Location: ./parameter_study_results/")
    print("   • Summary plots and detailed CSV exports available")
    print("   • Use PerformanceTracker to reload and analyze further")

    print("\n✨ Parameter study completed successfully!")

    return {
        "tracker": tracker,
        "prior_results": prior_results,
        "lr_results": lr_results,
        "error_results": error_results,
        "grid_results": grid_results,
        "optimization_summary": optimization_summary,
    }


def demonstrate_advanced_analysis():
    """
    Demonstrate advanced analysis techniques for parameter studies.
    """

    print("\n🔬 Advanced Analysis Techniques")
    print("=" * 50)

    # Load existing results
    tracker = PerformanceTracker("./parameter_study_results")

    if len(tracker.runs) == 0:
        print("No existing data found. Run main_parameter_study() first.")
        return

    # 1. Parameter interaction analysis
    print("1️⃣ Parameter Interaction Analysis")

    # Get runs with multiple parameters varied
    grid_runs = tracker.list_runs(tags=["focused_grid_search"])

    if grid_runs:
        comparison = tracker.compare_runs(grid_runs, metrics=["RMSE", "R2"])

        # Analyze parameter interactions
        print("   Parameter correlation with performance:")

        numeric_cols = []
        for col in comparison.columns:
            if col.startswith("config_") and comparison[col].dtype in [
                "float64",
                "int64",
            ]:
                numeric_cols.append(col)

        if numeric_cols:
            correlations = (
                comparison[numeric_cols + ["RMSE"]].corr()["RMSE"].sort_values()
            )
            print(correlations.to_string())

    # 2. Dataset difficulty ranking
    print("\n2️⃣ Dataset Difficulty Ranking")

    dataset_performance = {}
    for dataset_name in ["small_dense", "medium_sparse", "large_moderate"]:
        dataset_runs = tracker.list_runs(tags=[dataset_name])
        if dataset_runs:
            dataset_comparison = tracker.compare_runs(dataset_runs, metrics=["RMSE"])
            avg_rmse = dataset_comparison["RMSE"].mean()
            dataset_performance[dataset_name] = avg_rmse

    if dataset_performance:
        print("   Average RMSE by dataset:")
        for dataset, rmse in sorted(dataset_performance.items(), key=lambda x: x[1]):
            print(f"   {dataset}: {rmse:.3f}")

    # 3. Convergence analysis
    print("\n3️⃣ Convergence Analysis")

    # Analyze if more training steps improve performance
    num_steps_runs = tracker.list_runs(tags=["num_steps_study"])
    if num_steps_runs:
        print("   Training step effect analysis available in tracker data")

    print("\n✅ Advanced analysis completed!")


if __name__ == "__main__":
    # Run the main parameter study
    results = main_parameter_study()

    # Demonstrate advanced analysis
    demonstrate_advanced_analysis()

    print("\n🎉 Complete parameter optimization study finished!")
    print("Check './parameter_study_results/' for all outputs.")
