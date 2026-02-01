FEP Benchmark Tutorial
======================

This tutorial demonstrates how to use two key components of the MAPLE package:

1. **FEPBenchmarkDataset**: Loading and working with Free Energy Perturbation benchmark datasets
2. **PerformanceTracker**: Tracking, storing, and comparing model performance across runs

Setup and Imports
------------------

Let's start by importing the necessary libraries and MAPLE components:

.. code-block:: python

   # Standard libraries
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from pathlib import Path
   import warnings
   warnings.filterwarnings('ignore')

   # MAPLE components
   from maple.dataset.FEP_benchmark_dataset import FEPBenchmarkDataset
   from maple.utils.performance_tracker import PerformanceTracker, ModelRun
   from maple.graph_analysis.plotting_performance import plot_dataset_DGs, plot_dataset_DDGs

   # Set up plotting style
   plt.style.use('seaborn-v0_8')
   sns.set_palette("husl")
   np.random.seed(42)

   print("✅ Imports completed successfully")
   print("📦 MAPLE version: Available components loaded")

FEP Benchmark Dataset Tutorial
-------------------------------

The ``FEPBenchmarkDataset`` class provides access to benchmark Free Energy Perturbation datasets from the Schindler et al. repository. These datasets contain experimental and computational free energy data for various protein-ligand systems.

Available Datasets
~~~~~~~~~~~~~~~~~~

The FEP benchmark includes 8 different protein targets with molecular dynamics data at different sampling times:

.. code-block:: python

   # Initialize the dataset handler
   fep_dataset = FEPBenchmarkDataset(cache_dir="~/.maple_cache")

   print("🎯 Available FEP Benchmark Datasets:")
   for i, dataset_name in enumerate(fep_dataset.SUPPORTED_DATASETS, 1):
       print(f"   {i}. {dataset_name}")

   print("\n📊 Available sampling times: 5ns, 20ns")
   print(f"📁 Cache directory: {fep_dataset.cache_dir}")

Output::

   🎯 Available FEP Benchmark Datasets:
      1. cdk8
      2. cmet
      3. eg5
      4. hif2a
      5. pfkfb3
      6. shp2
      7. syk
      8. tnks2

   📊 Available sampling times: 5ns, 20ns
   📁 Cache directory: /Users/username/.maple_cache

Loading a Dataset
~~~~~~~~~~~~~~~~~

Let's load a specific dataset and examine its structure:

.. code-block:: python

   # Load the CDK8 dataset (5ns sampling)
   try:
       dataset = fep_dataset.load_dataset('cdk8', sampling_time='5ns')
       print(f"✅ Successfully loaded CDK8 dataset")
       
       # Get basic information about the dataset
       graph_data = dataset.get_graph_data()
       edge_df, node_df = dataset.get_dataframes()
       
       print(f"📊 Dataset Statistics:")
       print(f"   • Number of nodes: {graph_data['N']}")
       print(f"   • Number of edges: {graph_data['M']}")
       print(f"   • Edge DataFrame shape: {edge_df.shape}")
       print(f"   • Node DataFrame shape: {node_df.shape}")
       
   except Exception as e:
       print(f"❌ Failed to load dataset: {e}")
       print("💡 This may be due to network connectivity or cache issues")

Examining the Data Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's explore the loaded dataset in detail:

.. code-block:: python

   # Display first few rows of edge data
   print("🔗 Edge Data (first 5 rows):")
   print(edge_df.head())
   
   print("\n📝 Edge DataFrame columns:")
   for col in edge_df.columns:
       print(f"   • {col}: {edge_df[col].dtype}")
   
   # Display node data if available
   if not node_df.empty:
       print("\n🎯 Node Data (first 5 rows):")
       print(node_df.head())
   else:
       print("\n🎯 Node Data: No explicit node data (nodes inferred from edges)")

Dataset Visualization
~~~~~~~~~~~~~~~~~~~~~

Create visualizations to understand the data:

.. code-block:: python

   # Plot experimental vs calculated free energies
   fig, axes = plt.subplots(1, 2, figsize=(15, 6))
   
   # Plot ΔG values
   ax1 = axes[0]
   plot_dataset_DGs(dataset, ax=ax1)
   ax1.set_title('Free Energy (ΔG) Comparison')
   ax1.grid(True, alpha=0.3)
   
   # Plot ΔΔG values  
   ax2 = axes[1]
   plot_dataset_DDGs(dataset, ax=ax2)
   ax2.set_title('Relative Free Energy (ΔΔG) Comparison')
   ax2.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Performance Tracker Tutorial
-----------------------------

The ``PerformanceTracker`` class helps you systematically track and compare model performance across different runs and configurations.

Setting Up Performance Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize the performance tracker
   results_dir = Path("./tutorial_results")
   results_dir.mkdir(exist_ok=True)
   
   tracker = PerformanceTracker(storage_dir=str(results_dir))
   print(f"📊 Performance tracker initialized")
   print(f"💾 Results will be stored in: {results_dir.absolute()}")

Recording Model Runs
~~~~~~~~~~~~~~~~~~~~~

Let's simulate some model runs and track their performance:

.. code-block:: python

   # Simulate model predictions and track performance
   np.random.seed(42)
   
   # Create synthetic experimental data
   n_samples = 50
   y_true = np.random.normal(0, 2, n_samples)  # Experimental values
   
   # Simulate different model configurations
   model_configs = [
       {"learning_rate": 0.001, "prior_std": 0.5, "model_type": "baseline"},
       {"learning_rate": 0.01, "prior_std": 1.0, "model_type": "optimized"},
       {"learning_rate": 0.005, "prior_std": 2.0, "model_type": "robust"}
   ]
   
   # Record runs for each configuration
   for i, config in enumerate(model_configs):
       # Simulate model predictions (with different noise levels)
       noise_level = 0.8 - (i * 0.2)  # Decreasing noise = better models
       y_pred = y_true + np.random.normal(0, noise_level, n_samples)
       
       # Record the run
       run_id = f"experiment_{config['model_type']}"
       tracker.record_run(
           run_id=run_id,
           y_true=y_true,
           y_pred=y_pred,
           model_config=config,
           dataset_info={
               "name": "synthetic_tutorial_data",
               "n_samples": n_samples,
               "data_type": "simulated"
           }
       )
       
       print(f"✅ Recorded run: {run_id}")

Analyzing Results
~~~~~~~~~~~~~~~~~

Now let's analyze the tracked performance:

.. code-block:: python

   # Get run history
   history = tracker.get_run_history()
   print("📈 Performance Summary:")
   print(history[['run_id', 'RMSE', 'R2', 'MUE', 'correlation']].round(3))

   # Find the best performing model
   best_run = tracker.find_best_run(metric='RMSE', minimize=True)
   print(f"\n🏆 Best model: {best_run['run_id']}")
   print(f"   • RMSE: {best_run['RMSE']:.3f}")
   print(f"   • R²: {best_run['R2']:.3f}")

Comparing Models
~~~~~~~~~~~~~~~~

Create visualizations to compare model performance:

.. code-block:: python

   # Create performance comparison plots
   fig, axes = plt.subplots(2, 2, figsize=(15, 12))
   
   # Plot 1: RMSE comparison
   ax1 = axes[0, 0]
   runs = history['run_id'].tolist()
   rmse_values = history['RMSE'].tolist()
   ax1.bar(runs, rmse_values, color=['red', 'orange', 'green'])
   ax1.set_title('RMSE Comparison')
   ax1.set_ylabel('RMSE')
   ax1.tick_params(axis='x', rotation=45)
   
   # Plot 2: R² comparison
   ax2 = axes[0, 1]
   r2_values = history['R2'].tolist()
   ax2.bar(runs, r2_values, color=['red', 'orange', 'green'])
   ax2.set_title('R² Comparison')
   ax2.set_ylabel('R²')
   ax2.tick_params(axis='x', rotation=45)
   
   # Plot 3: Scatter plot for best model
   ax3 = axes[1, 0]
   best_run_data = tracker.get_run(best_run['run_id'])
   y_true_best = best_run_data.predictions['y_true']
   y_pred_best = best_run_data.predictions['y_pred']
   
   ax3.scatter(y_true_best, y_pred_best, alpha=0.6)
   min_val, max_val = min(y_true_best.min(), y_pred_best.min()), max(y_true_best.max(), y_pred_best.max())
   ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
   ax3.set_xlabel('Experimental Values')
   ax3.set_ylabel('Predicted Values')
   ax3.set_title(f'Best Model: {best_run["run_id"]}')
   ax3.legend()
   ax3.grid(True, alpha=0.3)
   
   # Plot 4: Performance evolution
   ax4 = axes[1, 1]
   metrics = ['RMSE', 'R2', 'MUE']
   for metric in metrics:
       ax4.plot(runs, history[metric], marker='o', label=metric)
   ax4.set_title('Performance Metrics Across Models')
   ax4.set_ylabel('Metric Value')
   ax4.legend()
   ax4.tick_params(axis='x', rotation=45)
   ax4.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Advanced Performance Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's perform some advanced analysis:

.. code-block:: python

   # Statistical comparison between models
   comparison = tracker.compare_runs([
       "experiment_baseline", 
       "experiment_optimized"
   ])
   
   print("📊 Statistical Comparison (Baseline vs Optimized):")
   for metric, stats in comparison.items():
       if isinstance(stats, dict):
           print(f"   {metric}:")
           for key, value in stats.items():
               print(f"     • {key}: {value:.4f}")
   
   # Export results for external analysis
   export_path = results_dir / "performance_summary.csv"
   tracker.export_results(str(export_path), format="csv")
   print(f"\n💾 Results exported to: {export_path}")

Working with Real FEP Data
---------------------------

Now let's combine both components to analyze real FEP benchmark data:

.. code-block:: python

   # Load a real dataset and analyze with performance tracking
   try:
       # Load dataset
       real_dataset = fep_dataset.load_dataset('cdk8', sampling_time='5ns')
       edge_df, _ = real_dataset.get_dataframes()
       
       # Extract experimental and calculated values
       y_exp = edge_df['exp_DDG'].values
       y_calc = edge_df['calc_DDG'].values
       
       # Remove any NaN values
       mask = ~(np.isnan(y_exp) | np.isnan(y_calc))
       y_exp_clean = y_exp[mask]
       y_calc_clean = y_calc[mask]
       
       # Record this as a model run
       tracker.record_run(
           run_id="cdk8_fep_benchmark",
           y_true=y_exp_clean,
           y_pred=y_calc_clean,
           model_config={
               "method": "FEP+",
               "sampling_time": "5ns",
               "force_field": "AMBER",
               "water_model": "TIP3P"
           },
           dataset_info={
               "name": "CDK8",
               "n_samples": len(y_exp_clean),
               "data_source": "Schindler_benchmark"
           }
       )
       
       print(f"✅ Recorded FEP benchmark run")
       print(f"📊 Samples: {len(y_exp_clean)}")
       
       # Get performance metrics for the real data
       fep_run = tracker.get_run("cdk8_fep_benchmark")
       print(f"\n📈 FEP Benchmark Performance:")
       print(f"   • RMSE: {fep_run.performance_metrics['RMSE']:.3f} kcal/mol")
       print(f"   • R²: {fep_run.performance_metrics['R2']:.3f}")
       print(f"   • MUE: {fep_run.performance_metrics['MUE']:.3f} kcal/mol")
       
   except Exception as e:
       print(f"❌ Could not process FEP data: {e}")

Summary and Best Practices
---------------------------

Key Takeaways
~~~~~~~~~~~~~

1. **FEPBenchmarkDataset**: 
   - Provides standardized access to benchmark FEP datasets
   - Supports multiple protein targets and sampling times
   - Handles data caching and preprocessing automatically

2. **PerformanceTracker**:
   - Systematic tracking of model performance across runs
   - Statistical comparison capabilities
   - Export functionality for external analysis
   - Visualization tools for performance assessment

Best Practices
~~~~~~~~~~~~~~

**For FEP Datasets:**

.. code-block:: python

   # Always check data availability
   try:
       dataset = fep_dataset.load_dataset('target', 'sampling_time')
       # Process dataset
   except Exception as e:
       print(f"Dataset not available: {e}")

   # Examine data quality
   edge_df, node_df = dataset.get_dataframes()
   print(f"Missing values in experimental data: {edge_df['exp_DDG'].isna().sum()}")

**For Performance Tracking:**

.. code-block:: python

   # Use descriptive run IDs
   run_id = f"{method}_{dataset}_{parameter_config}_{timestamp}"

   # Include comprehensive metadata
   tracker.record_run(
       run_id=run_id,
       y_true=y_true,
       y_pred=y_pred,
       model_config={...},  # Complete model configuration
       dataset_info={...},  # Dataset characteristics
       metadata={
           "notes": "Description of this run",
           "git_commit": "abc123",  # For reproducibility
           "random_seed": 42
       }
   )

   # Regular exports for backup
   tracker.export_results("backup.json", format="json")

This tutorial demonstrates the core functionality of MAPLE's FEP benchmark dataset handling and performance tracking capabilities. These tools provide a solid foundation for systematic FEP analysis and model development.
