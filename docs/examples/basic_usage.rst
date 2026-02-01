Basic Usage Examples
====================

This page provides comprehensive examples of MAPLE's dataset-centric workflow patterns.

Core Concept: Dataset as Central Hub
------------------------------------

MAPLE uses a dataset-centric architecture where the ``FEPDataset`` object serves as the central hub for all data and predictions:

.. code-block:: text

   +============================================================================+
   |                      MAPLE WORKFLOW OVERVIEW                                |
   +============================================================================+
   
   +-------------+      +----------------+      +------------------+
   |  FEPDataset |      |    Models      |      |  Analysis &      |
   |  (data hub) | <--> | (processors)   | <--> |  Visualization   |
   +-------------+      +----------------+      +------------------+
        ^                      |                        |
        |                      v                        |
        |            add_predictions_to_dataset()       |
        |______________________|________________________|
                               |
                               v
                      All results in dataset

Loading and Preparing Data
---------------------------

From Benchmark Dataset
~~~~~~~~~~~~~~~~~~~~~~

MAPLE includes standard FEP benchmark datasets:

.. code-block:: python

   from maple.dataset import FEPDataset
   
   # Load a benchmark dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # Inspect the data
   print(f"Loaded {len(dataset.dataset_nodes)} nodes")
   print(f"Loaded {len(dataset.dataset_edges)} edges")
   
   # View DataFrames
   print("\nNode data:")
   print(dataset.dataset_nodes.head())
   
   print("\nEdge data:")
   print(dataset.dataset_edges.head())
   
   # Check graph structure
   graph_data = dataset.get_graph_data()
   print(f"\nGraph has {graph_data['N']} nodes and {graph_data['M']} edges")

From Your Own Data
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from maple.dataset import FEPDataset
   
   # Create edge DataFrame
   edges_df = pd.DataFrame({
       'Source': ['molA', 'molB', 'molA', 'molC'],
       'Destination': ['molB', 'molC', 'molC', 'molD'],
       'DeltaDeltaG': [-2.3, 1.1, -1.2, 0.8],
       'DeltaDeltaG Error': [0.5, 0.4, 0.6, 0.3]
   })
   
   # Create node DataFrame
   nodes_df = pd.DataFrame({
       'Name': ['molA', 'molB', 'molC', 'molD'],
       'Exp. DeltaG': [-8.5, -6.2, -7.3, -6.5]
   })
   
   # Create dataset
   dataset = FEPDataset(dataset_nodes=nodes_df, dataset_edges=edges_df)
   
   print(f"Created dataset with {len(dataset.dataset_nodes)} nodes")

From CSV Files
~~~~~~~~~~~~~~

.. code-block:: python

   from maple.dataset import FEPDataset
   
   # Load from CSV files
   dataset = FEPDataset(
       nodes_csv_path="path/to/nodes.csv",
       edges_csv_path="path/to/edges.csv"
   )

From Edges Only
~~~~~~~~~~~~~~~

If you only have edge data, MAPLE can derive node values via graph traversal:

.. code-block:: python

   from maple.dataset import FEPDataset
   
   # Only provide edges - nodes will be derived automatically
   dataset = FEPDataset(dataset_edges=edges_df)
   
   # Or use the class method for more control
   edges_df, nodes_df = FEPDataset.derive_nodes_from_edges(
       dataset_edges=edges_df,
       reference_node="molA"  # Set this node to 0.0
   )
   dataset = FEPDataset(dataset_nodes=nodes_df, dataset_edges=edges_df)

Training Models and Adding Predictions
--------------------------------------

MAP Inference with NodeModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.dataset import FEPDataset
   from maple.models import NodeModel, NodeModelConfig, PriorType, GuideType
   
   # Create dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # Configure MAP inference
   config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,  # MAP
       learning_rate=0.01,
       num_steps=5000
   )
   
   # Create and train model
   model = NodeModel(config=config, dataset=dataset)
   model.train()
   
   # Add predictions to dataset
   model.add_predictions_to_dataset()
   
   # Verify predictions were added
   print("Columns after MAP training:")
   print(dataset.dataset_nodes.columns.tolist())
   # Output: ['Name', 'Exp. DeltaG', 'Pred. DeltaG', 'MAP']
   
   print(f"\nApplied estimators: {dataset.estimators}")
   # Output: ['MAP']

Variational Inference with NodeModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.models import NodeModel, NodeModelConfig, PriorType, GuideType
   
   # Configure VI (variational inference)
   vi_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_NORMAL,  # VI instead of MAP
       learning_rate=0.01,
       num_steps=5000
   )
   
   # Create and train
   vi_model = NodeModel(config=vi_config, dataset=dataset)
   vi_model.train()
   vi_model.add_predictions_to_dataset()
   
   # VI adds both predictions and uncertainties
   print("Columns after VI training:")
   print(dataset.dataset_nodes.columns.tolist())
   # Output: ['Name', 'Exp. DeltaG', 'Pred. DeltaG', 'MAP', 'VI', 'VI_uncertainty']
   
   # Access uncertainties
   print("\nVI predictions with uncertainties:")
   for _, row in dataset.dataset_nodes.iterrows():
       print(f"  {row['Name']}: {row['VI']:.2f} +/- {row['VI_uncertainty']:.2f}")

GMVI with Outlier Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.models import GMVI_model, GMVIConfig
   
   # Configure GMVI
   gmvi_config = GMVIConfig(
       prior_std=5.0,        # Prior std for node values
       normal_std=1.0,       # Std for normal edges
       outlier_std=3.0,      # Std for outlier edges
       outlier_prob=0.2,     # Prior probability of outlier
       n_epochs=2000,
       learning_rate=0.01
   )
   
   # Create and train
   gmvi_model = GMVI_model(dataset=dataset, config=gmvi_config)
   gmvi_model.fit()
   
   # IMPORTANT: Get posterior estimates before adding to dataset
   gmvi_model.get_posterior_estimates()
   
   # Add predictions to dataset
   gmvi_model.add_predictions_to_dataset()
   
   # Get outlier probabilities
   outlier_probs = gmvi_model.compute_edge_outlier_probabilities()
   
   # Find potential outliers
   print("\nPotential outlier edges:")
   for i, (_, row) in enumerate(dataset.dataset_edges.iterrows()):
       if outlier_probs[i] > 0.5:
           print(f"  {row['Source']} -> {row['Destination']}: "
                 f"prob={outlier_probs[i]:.2f}")

Complete Multi-Model Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.dataset import FEPDataset
   from maple.models import (
       NodeModel, NodeModelConfig,
       GMVI_model, GMVIConfig,
       PriorType, GuideType
   )
   from maple.graph_analysis import compute_simple_statistics
   
   # ============================================================
   # STEP 1: Create dataset
   # ============================================================
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   print("Initial state:")
   print(f"  Columns: {dataset.dataset_nodes.columns.tolist()}")
   print(f"  Estimators: {dataset.estimators}")
   
   # ============================================================
   # STEP 2: Train MAP model
   # ============================================================
   map_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,
       num_steps=5000
   )
   map_model = NodeModel(config=map_config, dataset=dataset)
   map_model.train()
   map_model.add_predictions_to_dataset()
   
   print("\nAfter MAP:")
   print(f"  Estimators: {dataset.estimators}")
   
   # ============================================================
   # STEP 3: Train VI model
   # ============================================================
   vi_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_NORMAL,
       num_steps=5000
   )
   vi_model = NodeModel(config=vi_config, dataset=dataset)
   vi_model.train()
   vi_model.add_predictions_to_dataset()
   
   print("\nAfter VI:")
   print(f"  Estimators: {dataset.estimators}")
   
   # ============================================================
   # STEP 4: Train GMVI model
   # ============================================================
   gmvi_config = GMVIConfig(prior_std=5.0, outlier_prob=0.2, n_epochs=2000)
   gmvi_model = GMVI_model(dataset=dataset, config=gmvi_config)
   gmvi_model.fit()
   gmvi_model.get_posterior_estimates()
   gmvi_model.add_predictions_to_dataset()
   
   print("\nAfter GMVI:")
   print(f"  Estimators: {dataset.estimators}")
   print(f"  Node columns: {dataset.dataset_nodes.columns.tolist()}")
   
   # ============================================================
   # STEP 5: Compare all models
   # ============================================================
   print("\n" + "=" * 50)
   print("MODEL COMPARISON")
   print("=" * 50)
   
   exp_values = dataset.dataset_nodes['Exp. DeltaG'].values
   
   for estimator in dataset.estimators:
       pred_values = dataset.dataset_nodes[estimator].values
       stats = compute_simple_statistics(exp_values, pred_values)
       print(f"\n{estimator}:")
       print(f"  RMSE: {stats['RMSE']:.3f} kcal/mol")
       print(f"  MAE:  {stats['MAE']:.3f} kcal/mol")
       print(f"  R2:   {stats['R2']:.3f}")
       print(f"  r:    {stats['r']:.3f}")

Working with Results
--------------------

Accessing Predictions from Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # All predictions are in the dataset DataFrames
   
   # Node predictions
   map_nodes = dataset.dataset_nodes['MAP']
   vi_nodes = dataset.dataset_nodes['VI']
   gmvi_nodes = dataset.dataset_nodes['GMVI']
   
   # Uncertainties (for VI and GMVI)
   vi_unc = dataset.dataset_nodes['VI_uncertainty']
   gmvi_unc = dataset.dataset_nodes['GMVI_uncertainty']
   
   # Edge predictions
   map_edges = dataset.dataset_edges['MAP']
   gmvi_edges = dataset.dataset_edges['GMVI']
   
   # Check which models have been applied
   print(f"Available estimators: {dataset.estimators}")

Creating Comparison DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   
   # Create a comparison table
   comparison = dataset.dataset_nodes[['Name', 'Exp. DeltaG'] + dataset.estimators].copy()
   
   # Add differences from experimental
   for estimator in dataset.estimators:
       comparison[f'{estimator}_diff'] = comparison[estimator] - comparison['Exp. DeltaG']
   
   print("Prediction comparison:")
   print(comparison.to_string())
   
   # Find best predictions per compound
   for estimator in dataset.estimators:
       comparison[f'{estimator}_abs_diff'] = abs(comparison[f'{estimator}_diff'])
   
   print("\nBest predictions per compound:")
   for _, row in comparison.iterrows():
       diffs = {est: abs(row[f'{est}_diff']) for est in dataset.estimators}
       best = min(diffs, key=diffs.get)
       print(f"  {row['Name']}: {best} (diff={diffs[best]:.2f})")

Visualization
-------------

Model Comparison Plot
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   
   # Create subplots for each model
   n_models = len(dataset.estimators)
   fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
   
   if n_models == 1:
       axes = [axes]
   
   exp_values = dataset.dataset_nodes['Exp. DeltaG'].values
   
   for ax, estimator in zip(axes, dataset.estimators):
       pred_values = dataset.dataset_nodes[estimator].values
       
       # Scatter plot
       ax.scatter(exp_values, pred_values, alpha=0.7, s=50)
       
       # Diagonal line
       min_val = min(exp_values.min(), pred_values.min())
       max_val = max(exp_values.max(), pred_values.max())
       ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
       
       # Labels
       ax.set_xlabel('Experimental (kcal/mol)', fontsize=12)
       ax.set_ylabel(f'{estimator} Predicted (kcal/mol)', fontsize=12)
       ax.set_title(f'{estimator} vs Experimental', fontsize=14)
       ax.grid(True, alpha=0.3)
       
       # Add statistics
       stats = compute_simple_statistics(exp_values, pred_values)
       ax.text(0.05, 0.95, f"RMSE: {stats['RMSE']:.2f}\nR2: {stats['R2']:.2f}",
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
   
   plt.tight_layout()
   plt.savefig('model_comparison.pdf', dpi=300, bbox_inches='tight')
   plt.show()

Predictions with Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   
   fig, ax = plt.subplots(figsize=(10, 6))
   
   # Get data
   names = dataset.dataset_nodes['Name'].values
   exp_values = dataset.dataset_nodes['Exp. DeltaG'].values
   
   # Plot GMVI predictions with error bars
   if 'GMVI' in dataset.estimators:
       gmvi_preds = dataset.dataset_nodes['GMVI'].values
       gmvi_unc = dataset.dataset_nodes['GMVI_uncertainty'].values
       
       x = np.arange(len(names))
       
       # Experimental values
       ax.scatter(x - 0.1, exp_values, label='Experimental', marker='o', s=80)
       
       # GMVI predictions with error bars
       ax.errorbar(x + 0.1, gmvi_preds, yerr=gmvi_unc, 
                   fmt='s', label='GMVI', capsize=3, markersize=8)
       
       ax.set_xticks(x)
       ax.set_xticklabels(names, rotation=45, ha='right')
       ax.set_ylabel('DeltaG (kcal/mol)', fontsize=12)
       ax.set_xlabel('Compound', fontsize=12)
       ax.legend()
       ax.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.savefig('gmvi_with_uncertainties.pdf', dpi=300, bbox_inches='tight')
   plt.show()

Outlier Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   
   # Get outlier probabilities from GMVI
   if hasattr(gmvi_model, 'compute_edge_outlier_probabilities'):
       outlier_probs = gmvi_model.compute_edge_outlier_probabilities()
       
       fig, ax = plt.subplots(figsize=(10, 6))
       
       # Create edge labels
       edge_labels = [f"{row['Source']}->{row['Destination']}" 
                      for _, row in dataset.dataset_edges.iterrows()]
       
       # Color bars by outlier probability
       colors = plt.cm.RdYlGn_r(np.array(outlier_probs))
       
       bars = ax.bar(range(len(outlier_probs)), outlier_probs, color=colors)
       
       ax.axhline(y=0.5, color='r', linestyle='--', label='50% threshold')
       ax.set_xticks(range(len(edge_labels)))
       ax.set_xticklabels(edge_labels, rotation=45, ha='right')
       ax.set_ylabel('Outlier Probability', fontsize=12)
       ax.set_xlabel('Edge', fontsize=12)
       ax.set_title('GMVI Edge Outlier Probabilities', fontsize=14)
       ax.legend()
       
       plt.tight_layout()
       plt.savefig('outlier_analysis.pdf', dpi=300, bbox_inches='tight')
       plt.show()

Batch Processing Multiple Datasets
----------------------------------

.. code-block:: python

   from maple.dataset import FEPDataset
   from maple.models import NodeModel, NodeModelConfig, GMVI_model, GMVIConfig
   from maple.models import PriorType, GuideType
   from maple.graph_analysis import compute_simple_statistics
   
   # Define datasets to process
   datasets_to_process = [
       ("cdk8", "5ns"),
       ("cmet", "5ns"),
       ("eg5", "5ns")
   ]
   
   # Store results
   all_results = {}
   
   # Shared configuration
   map_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,
       num_steps=5000
   )
   
   gmvi_config = GMVIConfig(prior_std=5.0, outlier_prob=0.2, n_epochs=2000)
   
   # Process each dataset
   for dataset_name, sampling_time in datasets_to_process:
       print(f"\nProcessing {dataset_name} ({sampling_time})...")
       
       # Load dataset
       dataset = FEPDataset(dataset_name=dataset_name, sampling_time=sampling_time)
       
       # Train MAP
       map_model = NodeModel(config=map_config, dataset=dataset)
       map_model.train()
       map_model.add_predictions_to_dataset()
       
       # Train GMVI
       gmvi_model = GMVI_model(dataset=dataset, config=gmvi_config)
       gmvi_model.fit()
       gmvi_model.get_posterior_estimates()
       gmvi_model.add_predictions_to_dataset()
       
       # Compute metrics
       exp_values = dataset.dataset_nodes['Exp. DeltaG'].values
       results = {}
       
       for estimator in dataset.estimators:
           pred_values = dataset.dataset_nodes[estimator].values
           stats = compute_simple_statistics(exp_values, pred_values)
           results[estimator] = stats
       
       all_results[dataset_name] = {
           'dataset': dataset,
           'metrics': results
       }
   
   # Print summary
   print("\n" + "=" * 60)
   print("SUMMARY")
   print("=" * 60)
   
   for dataset_name, data in all_results.items():
       print(f"\n{dataset_name}:")
       for estimator, metrics in data['metrics'].items():
           print(f"  {estimator}: RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}")

Best Practices Summary
----------------------

1. **Create dataset first**: Always start with ``FEPDataset``

2. **Train models**: Use ``model.train()`` or ``model.fit()``

3. **Add predictions**: Call ``model.add_predictions_to_dataset()``

4. **For GMVI**: Call ``get_posterior_estimates()`` before ``add_predictions_to_dataset()``

5. **Access results from dataset**: Use ``dataset.dataset_nodes`` and ``dataset.dataset_edges``

6. **Check estimators**: Use ``dataset.estimators`` to see applied models

.. code-block:: python

   # The pattern:
   dataset = FEPDataset(...)              # Step 1: Create dataset
   model = Model(config, dataset)         # Step 2: Create model
   model.train()                          # Step 3: Train
   model.add_predictions_to_dataset()     # Step 4: Add predictions
   
   # Results:
   dataset.dataset_nodes['{MODEL}']       # Node predictions
   dataset.dataset_edges['{MODEL}']       # Edge predictions
   dataset.estimators                     # List of applied models
