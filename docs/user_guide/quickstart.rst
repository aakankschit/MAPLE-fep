Quick Start Guide
=================

This guide will get you up and running with MAPLE in just a few minutes.

Overview
--------

MAPLE uses a **dataset-centric architecture** where the ``FEPDataset`` object serves as the central hub for all data and predictions. The main workflow involves:

1. **Creating a dataset** - Load your FEP data into an ``FEPDataset`` object
2. **Training models** - Configure and train one or more inference models
3. **Adding predictions** - Call ``add_predictions_to_dataset()`` to store results
4. **Analyzing results** - Access all predictions from the dataset

.. graphviz::
   :align: center
   :caption: MAPLE Workflow Overview

   digraph workflow {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12];
       edge [fontname="Helvetica", fontsize=10];
       compound=true;
       nodesep=0.4;
       ranksep=0.5;
       
       subgraph cluster_create {
           label="1. Create Dataset";
           style="rounded,filled";
           fillcolor="#E3F2FD";
           fontname="Helvetica-Bold";
           fontsize=13;
           
           create [label="FEPDataset(your data)", fillcolor="#BBDEFB"];
       }
       
       subgraph cluster_train {
           label="2. Train Models";
           style="rounded,filled";
           fillcolor="#E8F5E9";
           fontname="Helvetica-Bold";
           fontsize=13;
           
           subgraph cluster_train_inner {
               style=invis;
               rank=same;
               train1 [label="NodeModel", fillcolor="#C8E6C9"];
               train2 [label="GMVI_model", fillcolor="#C8E6C9"];
               train3 [label="WCC_model", fillcolor="#C8E6C9"];
           }
       }
       
       subgraph cluster_add {
           label="3. Add Predictions";
           style="rounded,filled";
           fillcolor="#FFF8E1";
           fontname="Helvetica-Bold";
           fontsize=13;
           
           add [label="add_predictions_to_dataset()", fillcolor="#FFECB3"];
       }
       
       subgraph cluster_access {
           label="4. Access Results";
           style="rounded,filled";
           fillcolor="#FCE4EC";
           fontname="Helvetica-Bold";
           fontsize=13;
           
           subgraph cluster_access_inner {
               style=invis;
               rank=same;
               access1 [label="dataset.dataset_nodes", fillcolor="#F8BBD9"];
               access2 [label="dataset.estimators", fillcolor="#F8BBD9"];
           }
       }
       
       create -> train2 [lhead=cluster_train];
       train2 -> add [ltail=cluster_train];
       add -> access1 [lhead=cluster_access];
   }

Basic Example
-------------

Here's a complete example of the MAPLE workflow:

.. code-block:: python

   from maple.dataset import FEPDataset
   from maple.models import NodeModel, NodeModelConfig, PriorType, GuideType
   from maple.models import GMVI_model, GMVIConfig
   
   # =====================================================
   # STEP 1: Create the dataset (your central data hub)
   # =====================================================
   
   # Option A: Load from benchmark dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # Option B: Load from your own DataFrames
   # dataset = FEPDataset(dataset_nodes=nodes_df, dataset_edges=edges_df)
   
   # Option C: Load from CSV files
   # dataset = FEPDataset(nodes_csv_path="nodes.csv", edges_csv_path="edges.csv")
   
   print(f"Loaded {len(dataset.dataset_nodes)} nodes and {len(dataset.dataset_edges)} edges")
   
   # =====================================================
   # STEP 2: Configure and train the MAP model
   # =====================================================
   
   map_config = NodeModelConfig(
       learning_rate=0.01,
       num_steps=5000,
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA  # MAP inference
   )
   
   map_model = NodeModel(config=map_config, dataset=dataset)
   map_model.train()
   
   # =====================================================
   # STEP 3: Add predictions to the dataset
   # =====================================================
   
   map_model.add_predictions_to_dataset()
   
   # Now dataset.dataset_nodes has a new "MAP" column!
   print(dataset.dataset_nodes[['Name', 'Exp. DeltaG', 'MAP']].head())
   
   # =====================================================
   # STEP 4: Train additional models (optional)
   # =====================================================
   
   # GMVI model with outlier detection
   gmvi_config = GMVIConfig(prior_std=5.0, outlier_prob=0.2)
   gmvi_model = GMVI_model(dataset=dataset, config=gmvi_config)
   gmvi_model.fit()
   gmvi_model.get_posterior_estimates()
   gmvi_model.add_predictions_to_dataset()
   
   # Now dataset has both "MAP" and "GMVI" columns!
   print(f"Applied estimators: {dataset.estimators}")
   # Output: ['MAP', 'GMVI']
   
   # =====================================================
   # STEP 5: Compare model performance
   # =====================================================
   
   from maple.graph_analysis import compute_simple_statistics
   
   exp_values = dataset.dataset_nodes['Exp. DeltaG'].values
   
   for estimator in dataset.estimators:
       pred_values = dataset.dataset_nodes[estimator].values
       stats = compute_simple_statistics(exp_values, pred_values)
       print(f"{estimator}: RMSE={stats['RMSE']:.3f}, R2={stats['R2']:.3f}")

Data Format
-----------

MAPLE expects FEP data in two DataFrames: **nodes** and **edges**.

Edge DataFrame
~~~~~~~~~~~~~~

Required columns:

* **Source** (or **Ligand1**): Source ligand name
* **Destination** (or **Ligand2**): Target ligand name
* **DeltaDeltaG** (or **FEP**): Relative binding free energy difference

Optional columns:

* **DeltaDeltaG Error**: Uncertainty from BAR/MBAR
* **CCC**: Cycle closure corrected values

.. code-block:: text

   +--------+-------------+-------------+------------------+-------+
   | Source | Destination | DeltaDeltaG | DeltaDeltaG Error| CCC   |
   +--------+-------------+-------------+------------------+-------+
   | molA   | molB        | -2.3        | 0.5              | -2.25 |
   | molB   | molC        | 1.1         | 0.4              | 1.08  |
   | molA   | molC        | -1.2        | 0.6              | -1.17 |
   +--------+-------------+-------------+------------------+-------+

Node DataFrame
~~~~~~~~~~~~~~

Required columns:

* **Name**: Ligand name
* **Exp. DeltaG**: Experimental binding free energy

.. code-block:: text

   +------+-------------+
   | Name | Exp. DeltaG |
   +------+-------------+
   | molA | -8.5        |
   | molB | -6.2        |
   | molC | -7.3        |
   +------+-------------+

Creating a Dataset
------------------

From Benchmark Data
~~~~~~~~~~~~~~~~~~~

MAPLE includes standard FEP benchmark datasets:

.. code-block:: python

   from maple.dataset import FEPDataset
   
   # Load a benchmark dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # Available datasets: cdk8, cmet, eg5, hif2a, pfkfb3, shp2, syk, tnks2

From DataFrames
~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from maple.dataset import FEPDataset
   
   # Create edge DataFrame
   edges_df = pd.DataFrame({
       'Source': ['molA', 'molB', 'molA'],
       'Destination': ['molB', 'molC', 'molC'],
       'DeltaDeltaG': [-2.3, 1.1, -1.2],
       'DeltaDeltaG Error': [0.5, 0.4, 0.6]
   })
   
   # Create node DataFrame
   nodes_df = pd.DataFrame({
       'Name': ['molA', 'molB', 'molC'],
       'Exp. DeltaG': [-8.5, -6.2, -7.3]
   })
   
   # Create dataset
   dataset = FEPDataset(dataset_nodes=nodes_df, dataset_edges=edges_df)

From Edges Only
~~~~~~~~~~~~~~~

If you only have edge data, MAPLE can derive node values:

.. code-block:: python

   from maple.dataset import FEPDataset
   
   # Dataset will automatically derive node values from edges
   dataset = FEPDataset(dataset_edges=edges_df)

Available Models
----------------

NodeModel (MAP/VI/MLE)
~~~~~~~~~~~~~~~~~~~~~~

The ``NodeModel`` provides Maximum A Posteriori (MAP), Variational Inference (VI), or Maximum Likelihood (MLE) estimation:

.. code-block:: python

   from maple.models import NodeModel, NodeModelConfig, PriorType, GuideType
   
   # MAP inference (point estimates)
   map_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,
       learning_rate=0.01,
       num_steps=5000
   )
   map_model = NodeModel(config=map_config, dataset=dataset)
   map_model.train()
   map_model.add_predictions_to_dataset()  # Adds "MAP" column
   
   # VI inference (with uncertainties)
   vi_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_NORMAL,  # VI instead of MAP
       learning_rate=0.01,
       num_steps=5000
   )
   vi_model = NodeModel(config=vi_config, dataset=dataset)
   vi_model.train()
   vi_model.add_predictions_to_dataset()  # Adds "VI" and "VI_uncertainty" columns

GMVI Model (Outlier-Robust)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``GMVI_model`` uses a Gaussian mixture likelihood for outlier detection:

.. code-block:: python

   from maple.models import GMVI_model, GMVIConfig
   
   gmvi_config = GMVIConfig(
       prior_std=5.0,        # Prior std for node values
       normal_std=1.0,       # Std for normal edges
       outlier_std=3.0,      # Std for outlier edges
       outlier_prob=0.2,     # Probability of outlier
       n_epochs=2000
   )
   
   gmvi_model = GMVI_model(dataset=dataset, config=gmvi_config)
   gmvi_model.fit()
   gmvi_model.get_posterior_estimates()  # Required before add_predictions
   gmvi_model.add_predictions_to_dataset()  # Adds "GMVI" and "GMVI_uncertainty"
   
   # Get outlier probabilities for each edge
   outlier_probs = gmvi_model.compute_edge_outlier_probabilities()

Key Concepts
------------

The Dataset as Central Hub
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``FEPDataset`` object serves as the central hub:

.. graphviz::
   :align: center
   :caption: FEPDataset as Central Hub

   digraph dataset_hub {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12];
       edge [fontname="Helvetica", fontsize=10];
       compound=true;
       nodesep=0.4;
       ranksep=0.5;
       
       models [label="Models (write predictions)", fillcolor="#FFF8E1", style="rounded,filled"];
       
       subgraph cluster_dataset {
           label="FEPDataset (Central Hub)";
           style="rounded,filled,bold";
           fillcolor="#E8F5E9";
           fontname="Helvetica-Bold";
           fontsize=13;
           
           nodes_df [label="dataset_nodes\nName | Exp. DeltaG | MAP | VI | GMVI", fillcolor="#C8E6C9"];
           edges_df [label="dataset_edges\nSource | Dest | DeltaDeltaG | predictions", fillcolor="#C8E6C9"];
           estimators [label="estimators = ['MAP', 'VI', 'GMVI', ...]", fillcolor="#C8E6C9"];
           
           nodes_df -> edges_df -> estimators [style=invis];
       }
       
       analysis [label="Analysis (read predictions)", fillcolor="#FCE4EC", style="rounded,filled"];
       
       models -> nodes_df [label="add_predictions_to_dataset()"];
       estimators -> analysis;
   }

The add_predictions_to_dataset() Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every model has an ``add_predictions_to_dataset()`` method that:

1. Writes node predictions to ``dataset.dataset_nodes``
2. Writes edge predictions to ``dataset.dataset_edges``
3. Adds uncertainties if available
4. Registers the model in ``dataset.estimators``

.. code-block:: python

   # This pattern is the same for all models:
   model.train()  # or model.fit() for GMVI/WCC
   model.add_predictions_to_dataset()  # Writes results to dataset
   
   # Now access results from the dataset:
   print(dataset.dataset_nodes['MAP'])  # Node predictions
   print(dataset.dataset_edges['MAP'])  # Edge predictions

Performance Analysis
--------------------

After training models, analyze performance using the dataset:

.. code-block:: python

   from maple.graph_analysis import (
       compute_simple_statistics,
       compute_bootstrap_statistics
   )
   
   # Get experimental and predicted values
   exp_values = dataset.dataset_nodes['Exp. DeltaG'].values
   map_preds = dataset.dataset_nodes['MAP'].values
   
   # Simple statistics
   stats = compute_simple_statistics(exp_values, map_preds)
   print(f"RMSE: {stats['RMSE']:.3f} kcal/mol")
   print(f"R2: {stats['R2']:.3f}")
   print(f"Pearson r: {stats['r']:.3f}")
   
   # Bootstrap confidence intervals
   bootstrap_stats = compute_bootstrap_statistics(
       exp_values, map_preds, n_bootstrap=1000
   )
   print(f"RMSE 95% CI: [{bootstrap_stats['RMSE']['ci_lower']:.3f}, "
         f"{bootstrap_stats['RMSE']['ci_upper']:.3f}]")

Visualization
-------------

Create publication-quality plots:

.. code-block:: python

   from maple.graph_analysis import plot_node_correlation
   import matplotlib.pyplot as plt
   
   # Plot model predictions vs experimental data
   fig, axes = plt.subplots(1, len(dataset.estimators), figsize=(5*len(dataset.estimators), 5))
   
   exp_values = dataset.dataset_nodes['Exp. DeltaG'].values
   
   for ax, estimator in zip(axes, dataset.estimators):
       pred_values = dataset.dataset_nodes[estimator].values
       ax.scatter(exp_values, pred_values, alpha=0.7)
       
       # Add diagonal line
       min_val = min(exp_values.min(), pred_values.min())
       max_val = max(exp_values.max(), pred_values.max())
       ax.plot([min_val, max_val], [min_val, max_val], 'r--')
       
       ax.set_xlabel('Experimental (kcal/mol)')
       ax.set_ylabel(f'{estimator} Predicted (kcal/mol)')
       ax.set_title(f'{estimator} vs Experimental')
   
   plt.tight_layout()
   plt.savefig('model_comparison.pdf')

Next Steps
----------

* See :doc:`architecture` for detailed architecture documentation
* Check :doc:`tutorials` for step-by-step tutorials
* Explore :doc:`parameter_optimization` for systematic parameter studies
* Browse the :doc:`../api/models` for complete API documentation
* Review :doc:`../examples/basic_usage` for more examples
