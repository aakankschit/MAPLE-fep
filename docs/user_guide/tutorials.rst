Tutorials
=========

This section provides step-by-step tutorials for common MAPLE workflows, emphasizing the dataset-centric architecture.

Tutorial 1: Understanding the Dataset-Model Workflow
-----------------------------------------------------

This tutorial introduces MAPLE's core architecture where the ``FEPDataset`` object serves as the central hub.

The Central Concept
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   +============================================================================+
   |              MAPLE's DATASET-CENTRIC ARCHITECTURE                          |
   +============================================================================+
   
   The FEPDataset is the central hub:
   
   1. STORES original data     (dataset_nodes, dataset_edges)
   2. PROVIDES graph structure (cycle_data, node2idx)
   3. ACCUMULATES predictions  (columns added by models)
   4. TRACKS applied models    (estimators list)
   
   +------------------------------------------------------------------+
   |                          FEPDataset                               |
   +------------------------------------------------------------------+
   |                                                                   |
   |  dataset_nodes:                                                   |
   |  +----+-------------+------+------+------+-----------------+     |
   |  |Name| Exp. DeltaG | MAP  | VI   | GMVI | GMVI_uncertainty|     |
   |  +----+-------------+------+------+------+-----------------+     |
   |  |molA| -8.5        | -8.3 | -8.4 | -8.35| 0.12            |     |
   |  +----+-------------+------+------+------+-----------------+     |
   |                           ^      ^      ^                        |
   |                           |      |      |                        |
   |              NodeModel ---+      |      +--- GMVI_model          |
   |                                  |                               |
   |              NodeModel(VI) ------+                               |
   |                                                                   |
   |  estimators: ['MAP', 'VI', 'GMVI']                               |
   +------------------------------------------------------------------+

Step 1: Create Your Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.dataset import FEPDataset
   
   # Load from benchmark (easiest way to start)
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # Check initial state
   print("Initial columns:", dataset.dataset_nodes.columns.tolist())
   # Output: ['Name', 'Exp. DeltaG', 'Pred. DeltaG']
   
   print("Applied estimators:", dataset.estimators)
   # Output: []

Step 2: Train a Model and Add Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.models import NodeModel, NodeModelConfig, PriorType, GuideType
   
   # Configure MAP inference
   config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,
       num_steps=5000
   )
   
   # Create model with dataset reference
   model = NodeModel(config=config, dataset=dataset)
   
   # Train the model
   model.train()
   
   # KEY STEP: Add predictions to dataset
   model.add_predictions_to_dataset()
   
   # Verify predictions were added
   print("Columns after MAP:", dataset.dataset_nodes.columns.tolist())
   # Output: ['Name', 'Exp. DeltaG', 'Pred. DeltaG', 'MAP']
   
   print("Applied estimators:", dataset.estimators)
   # Output: ['MAP']

Step 3: Add More Models
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.models import GMVI_model, GMVIConfig
   
   # Train GMVI
   gmvi_config = GMVIConfig(prior_std=5.0, outlier_prob=0.2)
   gmvi = GMVI_model(dataset=dataset, config=gmvi_config)
   gmvi.fit()
   gmvi.get_posterior_estimates()  # Required for GMVI!
   gmvi.add_predictions_to_dataset()
   
   # Both models' predictions are now in the dataset
   print("Columns:", dataset.dataset_nodes.columns.tolist())
   # Output: ['Name', 'Exp. DeltaG', 'Pred. DeltaG', 'MAP', 'GMVI', 'GMVI_uncertainty']
   
   print("Estimators:", dataset.estimators)
   # Output: ['MAP', 'GMVI']

Step 4: Compare Models from Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.graph_analysis import compute_simple_statistics
   
   exp = dataset.dataset_nodes['Exp. DeltaG'].values
   
   print("Model Comparison:")
   for estimator in dataset.estimators:
       pred = dataset.dataset_nodes[estimator].values
       stats = compute_simple_statistics(exp, pred)
       print(f"  {estimator}: RMSE={stats['RMSE']:.3f}, R2={stats['R2']:.3f}")

Tutorial 2: Complete FEP Analysis Workflow
------------------------------------------

This tutorial walks through a complete workflow from data loading to publication-ready analysis.

Step 1: Load and Inspect Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.dataset import FEPDataset
   
   # Load benchmark dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # Inspect the graph structure
   graph_data = dataset.get_graph_data()
   print(f"Graph has {graph_data['N']} nodes and {graph_data['M']} edges")
   
   # View the data
   print("\nNode data preview:")
   print(dataset.dataset_nodes.head())
   
   print("\nEdge data preview:")
   print(dataset.dataset_edges.head())

Step 2: Train Multiple Inference Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.models import (
       NodeModel, NodeModelConfig,
       GMVI_model, GMVIConfig,
       PriorType, GuideType
   )
   
   # ============================================================
   # MAP Inference (point estimates)
   # ============================================================
   map_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,
       learning_rate=0.01,
       num_steps=5000
   )
   map_model = NodeModel(config=map_config, dataset=dataset)
   map_model.train()
   map_model.add_predictions_to_dataset()
   print(f"MAP training complete. Estimators: {dataset.estimators}")
   
   # ============================================================
   # VI Inference (with uncertainties)
   # ============================================================
   vi_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_NORMAL,
       learning_rate=0.01,
       num_steps=5000
   )
   vi_model = NodeModel(config=vi_config, dataset=dataset)
   vi_model.train()
   vi_model.add_predictions_to_dataset()
   print(f"VI training complete. Estimators: {dataset.estimators}")
   
   # ============================================================
   # GMVI (outlier-robust with full covariance)
   # ============================================================
   gmvi_config = GMVIConfig(
       prior_std=5.0,
       normal_std=1.0,
       outlier_std=3.0,
       outlier_prob=0.2,
       n_epochs=2000
   )
   gmvi_model = GMVI_model(dataset=dataset, config=gmvi_config)
   gmvi_model.fit()
   gmvi_model.get_posterior_estimates()
   gmvi_model.add_predictions_to_dataset()
   print(f"GMVI training complete. Estimators: {dataset.estimators}")

Step 3: Comprehensive Performance Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.graph_analysis import (
       compute_simple_statistics,
       compute_bootstrap_statistics
   )
   import numpy as np
   
   exp_values = dataset.dataset_nodes['Exp. DeltaG'].values
   
   print("\n" + "=" * 60)
   print("PERFORMANCE SUMMARY")
   print("=" * 60)
   
   for estimator in dataset.estimators:
       pred_values = dataset.dataset_nodes[estimator].values
       
       # Simple statistics
       simple_stats = compute_simple_statistics(exp_values, pred_values)
       
       # Bootstrap confidence intervals
       bootstrap_stats = compute_bootstrap_statistics(
           exp_values, pred_values, n_bootstrap=1000
       )
       
       print(f"\n{estimator}:")
       print(f"  RMSE: {simple_stats['RMSE']:.3f} "
             f"[{bootstrap_stats['RMSE']['ci_lower']:.3f}, "
             f"{bootstrap_stats['RMSE']['ci_upper']:.3f}]")
       print(f"  MAE:  {simple_stats['MAE']:.3f}")
       print(f"  R2:   {simple_stats['R2']:.3f}")
       print(f"  r:    {simple_stats['r']:.3f}")
       print(f"  rho:  {simple_stats['rho']:.3f}")

Step 4: Outlier Analysis (GMVI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get outlier probabilities
   outlier_probs = gmvi_model.compute_edge_outlier_probabilities()
   
   # Add to edge DataFrame for easy access
   dataset.dataset_edges['outlier_prob'] = outlier_probs
   
   # Identify high-probability outliers
   threshold = 0.5
   outliers = dataset.dataset_edges[dataset.dataset_edges['outlier_prob'] > threshold]
   
   print(f"\nEdges with >{threshold*100}% outlier probability:")
   print("-" * 50)
   for _, row in outliers.iterrows():
       print(f"  {row['Source']} -> {row['Destination']}: "
             f"DeltaDeltaG={row['DeltaDeltaG']:.2f}, "
             f"prob={row['outlier_prob']:.2f}")

Step 5: Publication-Ready Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   
   # Create figure with subplots for each model
   n_models = len(dataset.estimators)
   fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
   
   if n_models == 1:
       axes = [axes]
   
   exp = dataset.dataset_nodes['Exp. DeltaG'].values
   
   for ax, estimator in zip(axes, dataset.estimators):
       pred = dataset.dataset_nodes[estimator].values
       
       # Scatter plot
       ax.scatter(exp, pred, alpha=0.7, s=60, edgecolor='black', linewidth=0.5)
       
       # Perfect prediction line
       min_val, max_val = min(exp.min(), pred.min()) - 0.5, max(exp.max(), pred.max()) + 0.5
       ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
       
       # Error bands (+/- 1 kcal/mol)
       ax.fill_between([min_val, max_val], 
                       [min_val - 1, max_val - 1],
                       [min_val + 1, max_val + 1],
                       alpha=0.2, color='gray', label='+/- 1 kcal/mol')
       
       # Statistics annotation
       stats = compute_simple_statistics(exp, pred)
       text = f"RMSE: {stats['RMSE']:.2f}\nR$^2$: {stats['R2']:.2f}\n$r$: {stats['r']:.2f}"
       ax.text(0.05, 0.95, text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
       
       ax.set_xlabel('Experimental $\Delta$G (kcal/mol)', fontsize=12)
       ax.set_ylabel(f'{estimator} Predicted $\Delta$G (kcal/mol)', fontsize=12)
       ax.set_title(f'{estimator} Model', fontsize=14)
       ax.grid(True, alpha=0.3)
       ax.set_xlim(min_val, max_val)
       ax.set_ylim(min_val, max_val)
       ax.set_aspect('equal')
   
   plt.tight_layout()
   plt.savefig('model_comparison.pdf', dpi=300, bbox_inches='tight')
   plt.show()

Tutorial 3: Working with Your Own Data
--------------------------------------

This tutorial shows how to use MAPLE with your own FEP data.

Step 1: Prepare Your Data
~~~~~~~~~~~~~~~~~~~~~~~~~

Your data should be in two DataFrames:

**Edge DataFrame** (required):

.. code-block:: python

   import pandas as pd
   
   edges_df = pd.DataFrame({
       'Source': ['molA', 'molB', 'molA', 'molC', 'molB'],
       'Destination': ['molB', 'molC', 'molC', 'molD', 'molD'],
       'DeltaDeltaG': [-2.3, 1.1, -1.2, 0.8, 2.0],
       'DeltaDeltaG Error': [0.5, 0.4, 0.6, 0.3, 0.5]  # Optional but recommended
   })

**Node DataFrame** (required for validation):

.. code-block:: python

   nodes_df = pd.DataFrame({
       'Name': ['molA', 'molB', 'molC', 'molD'],
       'Exp. DeltaG': [-8.5, -6.2, -7.3, -6.5]
   })

Step 2: Create Dataset
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.dataset import FEPDataset
   
   dataset = FEPDataset(
       dataset_nodes=nodes_df,
       dataset_edges=edges_df
   )
   
   print(f"Created dataset with {len(dataset.dataset_nodes)} nodes "
         f"and {len(dataset.dataset_edges)} edges")

Step 3: Automatic Node Derivation (If No Experimental Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you only have edge data (no experimental node values):

.. code-block:: python

   # Option 1: Let FEPDataset derive nodes automatically
   dataset = FEPDataset(dataset_edges=edges_df)
   
   # Option 2: Use the class method for more control
   edges_processed, nodes_derived = FEPDataset.derive_nodes_from_edges(
       dataset_edges=edges_df,
       reference_node='molA'  # Set this node to DeltaG = 0.0
   )
   
   dataset = FEPDataset(
       dataset_nodes=nodes_derived,
       dataset_edges=edges_processed
   )

Step 4: Train and Compare Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.models import (
       NodeModel, NodeModelConfig,
       GMVI_model, GMVIConfig,
       PriorType, GuideType
   )
   
   # MAP
   map_model = NodeModel(
       config=NodeModelConfig(
           prior_type=PriorType.NORMAL,
           guide_type=GuideType.AUTO_DELTA
       ),
       dataset=dataset
   )
   map_model.train()
   map_model.add_predictions_to_dataset()
   
   # GMVI
   gmvi_model = GMVI_model(
       dataset=dataset,
       config=GMVIConfig(prior_std=5.0, outlier_prob=0.2)
   )
   gmvi_model.fit()
   gmvi_model.get_posterior_estimates()
   gmvi_model.add_predictions_to_dataset()
   
   # View results
   print(dataset.dataset_nodes[['Name', 'Exp. DeltaG', 'MAP', 'GMVI']])

Tutorial 4: Understanding Model Uncertainties
---------------------------------------------

This tutorial focuses on uncertainty quantification with VI and GMVI.

Variational Inference Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.models import NodeModel, NodeModelConfig, GuideType
   
   # Use AutoNormal guide for VI
   vi_config = NodeModelConfig(
       guide_type=GuideType.AUTO_NORMAL,  # This enables uncertainty estimation
       num_steps=5000
   )
   
   vi_model = NodeModel(config=vi_config, dataset=dataset)
   vi_model.train()
   vi_model.add_predictions_to_dataset()
   
   # Access uncertainties
   print("\nVI Predictions with Uncertainties:")
   print("-" * 50)
   for _, row in dataset.dataset_nodes.iterrows():
       print(f"{row['Name']:10s}: {row['VI']:.2f} +/- {row['VI_uncertainty']:.2f}")

GMVI Uncertainties and Outlier Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.models import GMVI_model, GMVIConfig
   
   gmvi = GMVI_model(
       dataset=dataset,
       config=GMVIConfig(
           prior_std=5.0,
           normal_std=1.0,
           outlier_std=3.0,  # Higher std for outliers
           outlier_prob=0.2
       )
   )
   gmvi.fit()
   gmvi.get_posterior_estimates()
   gmvi.add_predictions_to_dataset()
   
   # Node uncertainties
   print("\nGMVI Predictions with Uncertainties:")
   print("-" * 50)
   for _, row in dataset.dataset_nodes.iterrows():
       print(f"{row['Name']:10s}: {row['GMVI']:.2f} +/- {row['GMVI_uncertainty']:.2f}")
   
   # Edge outlier probabilities
   outlier_probs = gmvi.compute_edge_outlier_probabilities()
   
   print("\nEdge Outlier Probabilities:")
   print("-" * 50)
   for i, (_, row) in enumerate(dataset.dataset_edges.iterrows()):
       status = "OUTLIER" if outlier_probs[i] > 0.5 else ""
       print(f"{row['Source']:10s} -> {row['Destination']:10s}: "
             f"{outlier_probs[i]:.2f} {status}")

Visualizing Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   
   fig, ax = plt.subplots(figsize=(12, 6))
   
   # Get data
   names = dataset.dataset_nodes['Name'].values
   exp = dataset.dataset_nodes['Exp. DeltaG'].values
   gmvi_pred = dataset.dataset_nodes['GMVI'].values
   gmvi_unc = dataset.dataset_nodes['GMVI_uncertainty'].values
   
   x = np.arange(len(names))
   width = 0.35
   
   # Plot experimental values
   ax.bar(x - width/2, exp, width, label='Experimental', alpha=0.8)
   
   # Plot GMVI predictions with error bars
   ax.bar(x + width/2, gmvi_pred, width, label='GMVI', alpha=0.8,
          yerr=gmvi_unc, capsize=3)
   
   ax.set_xlabel('Compound', fontsize=12)
   ax.set_ylabel('$\Delta$G (kcal/mol)', fontsize=12)
   ax.set_title('Experimental vs GMVI Predictions with Uncertainty', fontsize=14)
   ax.set_xticks(x)
   ax.set_xticklabels(names, rotation=45, ha='right')
   ax.legend()
   ax.grid(True, alpha=0.3, axis='y')
   
   plt.tight_layout()
   plt.savefig('uncertainty_comparison.pdf', dpi=300)
   plt.show()

Tutorial 5: Batch Processing Multiple Datasets
----------------------------------------------

This tutorial shows how to analyze multiple datasets systematically.

.. code-block:: python

   from maple.dataset import FEPDataset
   from maple.models import (
       NodeModel, NodeModelConfig,
       GMVI_model, GMVIConfig,
       PriorType, GuideType
   )
   from maple.graph_analysis import compute_simple_statistics
   import pandas as pd
   
   # Define datasets and configurations
   benchmark_datasets = [
       ("cdk8", "5ns"),
       ("cmet", "5ns"),
       ("eg5", "5ns")
   ]
   
   map_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,
       num_steps=5000
   )
   
   gmvi_config = GMVIConfig(
       prior_std=5.0,
       outlier_prob=0.2,
       n_epochs=2000
   )
   
   # Collect results
   all_results = []
   all_datasets = {}
   
   for dataset_name, sampling_time in benchmark_datasets:
       print(f"\n{'='*50}")
       print(f"Processing: {dataset_name} ({sampling_time})")
       print('='*50)
       
       # Load dataset
       dataset = FEPDataset(dataset_name=dataset_name, sampling_time=sampling_time)
       all_datasets[dataset_name] = dataset
       
       # Train MAP
       print("  Training MAP...")
       map_model = NodeModel(config=map_config, dataset=dataset)
       map_model.train()
       map_model.add_predictions_to_dataset()
       
       # Train GMVI
       print("  Training GMVI...")
       gmvi_model = GMVI_model(dataset=dataset, config=gmvi_config)
       gmvi_model.fit()
       gmvi_model.get_posterior_estimates()
       gmvi_model.add_predictions_to_dataset()
       
       # Compute metrics
       exp = dataset.dataset_nodes['Exp. DeltaG'].values
       
       for estimator in dataset.estimators:
           pred = dataset.dataset_nodes[estimator].values
           stats = compute_simple_statistics(exp, pred)
           
           all_results.append({
               'Dataset': dataset_name,
               'Estimator': estimator,
               'RMSE': stats['RMSE'],
               'MAE': stats['MAE'],
               'R2': stats['R2'],
               'r': stats['r']
           })
   
   # Create summary DataFrame
   results_df = pd.DataFrame(all_results)
   
   # Pivot for easy comparison
   print("\n" + "=" * 60)
   print("SUMMARY: RMSE by Dataset and Estimator")
   print("=" * 60)
   pivot_rmse = results_df.pivot(index='Dataset', columns='Estimator', values='RMSE')
   print(pivot_rmse.to_string())
   
   print("\n" + "=" * 60)
   print("SUMMARY: R2 by Dataset and Estimator")
   print("=" * 60)
   pivot_r2 = results_df.pivot(index='Dataset', columns='Estimator', values='R2')
   print(pivot_r2.to_string())
   
   # Average across datasets
   print("\n" + "=" * 60)
   print("AVERAGE METRICS ACROSS DATASETS")
   print("=" * 60)
   avg_metrics = results_df.groupby('Estimator')[['RMSE', 'MAE', 'R2', 'r']].mean()
   print(avg_metrics.to_string())

Summary: Key Patterns to Remember
---------------------------------

.. code-block:: text

   +============================================================================+
   |                    MAPLE WORKFLOW PATTERN                                   |
   +============================================================================+
   
   1. CREATE DATASET (central hub)
      dataset = FEPDataset(...)
   
   2. CREATE & TRAIN MODELS
      model = ModelClass(config=config, dataset=dataset)
      model.train()  # or .fit()
   
   3. ADD PREDICTIONS TO DATASET
      model.add_predictions_to_dataset()
      
      NOTE: For GMVI, call get_posterior_estimates() first!
   
   4. ACCESS RESULTS FROM DATASET
      dataset.dataset_nodes['{MODEL}']      # Node predictions
      dataset.dataset_edges['{MODEL}']      # Edge predictions
      dataset.dataset_nodes['{MODEL}_uncertainty']  # Uncertainties (if available)
      dataset.estimators                    # List of applied models
   
   5. COMPARE MODELS
      for estimator in dataset.estimators:
          pred = dataset.dataset_nodes[estimator]
          # Compare with experimental values

For more examples, see :doc:`../examples/basic_usage` and the API documentation.
