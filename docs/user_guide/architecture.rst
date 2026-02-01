Architecture and Design
=======================

This document explains MAPLE's architecture and design philosophy, focusing on the central role of the ``FEPDataset`` object and how models interact with it.

Design Philosophy
-----------------

MAPLE follows a **dataset-centric design** where:

1. **The dataset owns the data**: All FEP data (nodes, edges, predictions) lives in the ``FEPDataset`` object
2. **Models are processors**: Models read from the dataset, perform inference, and write results back
3. **Predictions accumulate**: Multiple models can add their predictions to the same dataset
4. **Single source of truth**: All results are accessible from one place

This design enables easy comparison of different models and methods on the same data.

Module Overview
---------------

MAPLE is organized into three main packages:

.. graphviz::
   :align: center
   :caption: MAPLE Package Structure

   digraph package_structure {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12];
       edge [fontname="Helvetica", fontsize=10];
       compound=true;
       nodesep=0.4;
       ranksep=0.6;
       
       maple [label="maple/", fillcolor="#E8F4FD", style="rounded,filled,bold", fontsize=14];
       
       subgraph cluster_packages {
           style=invis;
           
           subgraph cluster_dataset {
               label="dataset/";
               style="rounded,filled";
               fillcolor="#D4EDDA";
               fontname="Helvetica-Bold";
               fontsize=13;
               
               base_dataset [label="base_dataset.py", fillcolor="#C3E6CB"];
               dataset [label="dataset.py", fillcolor="#C3E6CB"];
               benchmark [label="FEP_benchmark_dataset.py", fillcolor="#C3E6CB"];
               synthetic [label="synthetic_dataset.py", fillcolor="#C3E6CB"];
               
               base_dataset -> dataset -> benchmark -> synthetic [style=invis];
           }
           
           subgraph cluster_models {
               label="models/";
               style="rounded,filled";
               fillcolor="#FFF3CD";
               fontname="Helvetica-Bold";
               fontsize=13;
               
               node_model [label="node_model.py", fillcolor="#FFE69C"];
               gmvi_model [label="gaussian_markov_model.py", fillcolor="#FFE69C"];
               wcc_model [label="wcc_model.py", fillcolor="#FFE69C"];
               model_config [label="model_config.py", fillcolor="#FFE69C"];
               
               node_model -> gmvi_model -> wcc_model -> model_config [style=invis];
           }
           
           subgraph cluster_analysis {
               label="graph_analysis/";
               style="rounded,filled";
               fillcolor="#F8D7DA";
               fontname="Helvetica-Bold";
               fontsize=13;
               
               perf_stats [label="performance_stats.py", fillcolor="#F5C6CB"];
               plotting [label="plotting_performance.py", fillcolor="#F5C6CB"];
               
               perf_stats -> plotting [style=invis];
           }
       }
       
       maple -> base_dataset [lhead=cluster_dataset];
       maple -> node_model [lhead=cluster_models];
       maple -> perf_stats [lhead=cluster_analysis];
   }

The Dataset as Central Hub
--------------------------

The ``FEPDataset`` object serves as the central hub in MAPLE:

.. graphviz::
   :align: center
   :caption: FEPDataset: Central Data Hub

   digraph dataset_hub {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12];
       edge [fontname="Helvetica", fontsize=10];
       compound=true;
       nodesep=0.4;
       ranksep=0.6;
       
       subgraph cluster_input {
           label="Input Sources";
           style="rounded,filled";
           fillcolor="#E3F2FD";
           fontname="Helvetica-Bold";
           fontsize=14;
           
           subgraph cluster_input_inner {
               style=invis;
               rank=same;
               csv [label="CSV Files", fillcolor="#BBDEFB"];
               df [label="DataFrames", fillcolor="#BBDEFB"];
               bench [label="Benchmarks", fillcolor="#BBDEFB"];
           }
       }
       
       subgraph cluster_dataset {
           label="FEPDataset (Central Hub)";
           style="rounded,filled,bold";
           fillcolor="#E8F5E9";
           fontname="Helvetica-Bold";
           fontsize=14;
           
           nodes_df [label="dataset_nodes", fillcolor="#C8E6C9"];
           edges_df [label="dataset_edges", fillcolor="#C8E6C9"];
           graph_data [label="cycle_data", fillcolor="#C8E6C9"];
           mappings [label="node2idx / idx2node", fillcolor="#C8E6C9"];
           est [label="estimators[]", fillcolor="#C8E6C9"];
           
           nodes_df -> edges_df -> graph_data -> mappings -> est [style=invis];
       }
       
       subgraph cluster_output {
           label="Consumers";
           style="rounded,filled";
           fillcolor="#FFF8E1";
           fontname="Helvetica-Bold";
           fontsize=14;
           
           subgraph cluster_output_inner {
               style=invis;
               rank=same;
               models [label="Models", fillcolor="#FFECB3"];
               analysis [label="Analysis", fillcolor="#FFECB3"];
               viz [label="Visualization", fillcolor="#FFECB3"];
           }
       }
       
       df -> nodes_df [lhead=cluster_dataset];
       est -> models [ltail=cluster_dataset, lhead=cluster_output];
   }

Model-Dataset Interaction Pattern
---------------------------------

All MAPLE models follow a consistent interaction pattern with the dataset:

.. graphviz::
   :align: center
   :caption: Model-Dataset Interaction Pattern

   digraph interaction {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12];
       edge [fontname="Helvetica", fontsize=10];
       nodesep=0.4;
       ranksep=0.5;
       
       subgraph cluster_step1 {
           label="Step 1: Initialize";
           style="rounded,filled";
           fillcolor="#E3F2FD";
           fontname="Helvetica-Bold";
           fontsize=13;
           
           init [label="model = NodeModel(config, dataset)", fillcolor="#BBDEFB"];
       }
       
       subgraph cluster_step2 {
           label="Step 2: Extract Data";
           style="rounded,filled";
           fillcolor="#E8F5E9";
           fontname="Helvetica-Bold";
           fontsize=13;
           
           extract [label="model._extract_graph_data()\nReads: dataset.cycle_data", fillcolor="#C8E6C9"];
       }
       
       subgraph cluster_step3 {
           label="Step 3: Train";
           style="rounded,filled";
           fillcolor="#FFF8E1";
           fontname="Helvetica-Bold";
           fontsize=13;
           
           train [label="model.train()\nProduces: node_estimates, edge_estimates", fillcolor="#FFECB3"];
       }
       
       subgraph cluster_step4 {
           label="Step 4: Add to Dataset";
           style="rounded,filled";
           fillcolor="#FCE4EC";
           fontname="Helvetica-Bold";
           fontsize=13;
           
           add [label="model.add_predictions_to_dataset()\nWrites to: dataset_nodes, dataset_edges", fillcolor="#F8BBD9"];
       }
       
       init -> extract -> train -> add;
   }

Available Models
----------------

MAPLE provides several inference methods, each adding different columns to the dataset:

.. graphviz::
   :align: center
   :caption: Available Models and Their Outputs

   digraph models {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12];
       edge [fontname="Helvetica", fontsize=10];
       compound=true;
       nodesep=0.4;
       ranksep=0.6;
       
       subgraph cluster_models {
           label="Available Models";
           style="rounded,filled";
           fillcolor="#F5F5F5";
           fontname="Helvetica-Bold";
           fontsize=14;
           
           subgraph cluster_models_inner {
               style=invis;
               rank=same;
               
               subgraph cluster_nodemodel {
                   label="NodeModel";
                   style="rounded,filled";
                   fillcolor="#E3F2FD";
                   fontname="Helvetica-Bold";
                   fontsize=12;
                   
                   map [label="MAP", fillcolor="#BBDEFB"];
                   mle [label="MLE", fillcolor="#BBDEFB"];
                   vi [label="VI", fillcolor="#BBDEFB"];
                   
                   map -> mle -> vi [style=invis];
               }
               
               subgraph cluster_gmvi {
                   label="GMVI_model";
                   style="rounded,filled";
                   fillcolor="#E8F5E9";
                   fontname="Helvetica-Bold";
                   fontsize=12;
                   
                   gmvi [label="Full-rank VI\n+ Outlier Detection", fillcolor="#C8E6C9"];
               }
               
               subgraph cluster_wcc {
                   label="WCC_model";
                   style="rounded,filled";
                   fillcolor="#FFF8E1";
                   fontname="Helvetica-Bold";
                   fontsize=12;
                   
                   wcc [label="Weighted\nCycle Closure", fillcolor="#FFECB3"];
               }
           }
       }
       
       subgraph cluster_result {
           label="Columns Added to Dataset";
           style="rounded,filled";
           fillcolor="#FCE4EC";
           fontname="Helvetica-Bold";
           fontsize=14;
           
           subgraph cluster_result_inner {
               style=invis;
               rank=same;
               col_map [label="'MAP'", fillcolor="#F8BBD9"];
               col_mle [label="'MLE'", fillcolor="#F8BBD9"];
               col_vi [label="'VI', 'VI_uncertainty'", fillcolor="#F8BBD9"];
               col_gmvi [label="'GMVI', 'GMVI_uncertainty'", fillcolor="#F8BBD9"];
               col_wcc [label="'WCC', 'WCC_uncertainty'", fillcolor="#F8BBD9"];
           }
       }
       
       map -> col_map;
       mle -> col_mle;
       vi -> col_vi;
       gmvi -> col_gmvi;
       wcc -> col_wcc;
   }

NodeModel (MAP/VI/MLE)
~~~~~~~~~~~~~~~~~~~~~~

**Usage:**

.. code-block:: python

   from maple.models import NodeModel, NodeModelConfig, PriorType, GuideType
   
   # MAP inference
   config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,
       num_steps=5000
   )
   model = NodeModel(config=config, dataset=dataset)
   model.train()
   model.add_predictions_to_dataset()  # Adds "MAP" column

GMVI_model (Gaussian Mixture Variational Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Usage:**

.. code-block:: python

   from maple.models import GMVI_model, GMVIConfig
   
   config = GMVIConfig(
       prior_std=5.0,
       normal_std=1.0,
       outlier_std=3.0,
       outlier_prob=0.2
   )
   model = GMVI_model(dataset=dataset, config=config)
   model.fit()
   model.get_posterior_estimates()  # REQUIRED before add_predictions
   model.add_predictions_to_dataset()  # Adds "GMVI" and "GMVI_uncertainty"
   
   # Get outlier probabilities
   outlier_probs = model.compute_edge_outlier_probabilities()

WCC_model (Weighted Cycle Closure)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Usage:**

.. code-block:: python

   from maple.models import WCC_model, WCCConfig
   
   config = WCCConfig(tolerance=1e-6)
   model = WCC_model(dataset=dataset, config=config)
   model.fit()
   model.add_predictions_to_dataset()  # Adds "WCC" and "WCC_uncertainty"

Complete Workflow Example
-------------------------

Here's a complete example showing the dataset-centric workflow:

.. graphviz::
   :align: center
   :caption: Complete MAPLE Workflow

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
           
           create [label="FEPDataset(dataset_name='cdk8', sampling_time='5ns')", fillcolor="#BBDEFB"];
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
               train_map [label="NodeModel (MAP)\n.train()\n.add_predictions...()", fillcolor="#C8E6C9"];
               train_vi [label="NodeModel (VI)\n.train()\n.add_predictions...()", fillcolor="#C8E6C9"];
               train_gmvi [label="GMVI_model\n.fit()\n.add_predictions...()", fillcolor="#C8E6C9"];
           }
       }
       
       subgraph cluster_result {
           label="3. Access Results";
           style="rounded,filled";
           fillcolor="#FCE4EC";
           fontname="Helvetica-Bold";
           fontsize=13;
           
           check [label="dataset.estimators = ['MAP', 'VI', 'GMVI']", fillcolor="#F8BBD9"];
           compare [label="dataset.dataset_nodes[estimator]", fillcolor="#F8BBD9"];
           
           check -> compare [style=invis];
       }
       
       create -> train_vi [lhead=cluster_train];
       train_vi -> check [ltail=cluster_train];
   }

.. code-block:: python

   from maple.dataset import FEPDataset
   from maple.models import (
       NodeModel, NodeModelConfig, 
       GMVI_model, GMVIConfig,
       PriorType, GuideType
   )
   from maple.graph_analysis import compute_simple_statistics
   
   # Step 1: Create dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # Step 2: Train MAP model
   map_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,
       num_steps=5000
   )
   map_model = NodeModel(config=map_config, dataset=dataset)
   map_model.train()
   map_model.add_predictions_to_dataset()
   
   # Step 3: Train VI model
   vi_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_NORMAL,
       num_steps=5000
   )
   vi_model = NodeModel(config=vi_config, dataset=dataset)
   vi_model.train()
   vi_model.add_predictions_to_dataset()
   
   # Step 4: Train GMVI model
   gmvi_config = GMVIConfig(prior_std=5.0, outlier_prob=0.2)
   gmvi = GMVI_model(dataset=dataset, config=gmvi_config)
   gmvi.fit()
   gmvi.get_posterior_estimates()
   gmvi.add_predictions_to_dataset()
   
   # Step 5: Compare all models
   print(f"Applied estimators: {dataset.estimators}")
   
   exp = dataset.dataset_nodes['Exp. DeltaG'].values
   for estimator in dataset.estimators:
       pred = dataset.dataset_nodes[estimator].values
       stats = compute_simple_statistics(exp, pred)
       print(f"{estimator}: RMSE={stats['RMSE']:.3f}, R2={stats['R2']:.3f}")

Key Design Patterns
-------------------

Configuration via Pydantic
~~~~~~~~~~~~~~~~~~~~~~~~~~

All model configurations use Pydantic for validation:

.. code-block:: python

   from maple.models import NodeModelConfig, GMVIConfig
   
   # Validation happens automatically
   config = NodeModelConfig(
       learning_rate=0.01,      # Must be > 0
       num_steps=5000,          # Must be > 0
       prior_type=PriorType.NORMAL
   )

Column Naming Convention
~~~~~~~~~~~~~~~~~~~~~~~~

Models add columns with consistent naming:

===============  ================================
Model Type       Columns Added
===============  ================================
MAP (NodeModel)  ``'MAP'``
MLE (NodeModel)  ``'MLE'``
VI (NodeModel)   ``'VI'``, ``'VI_uncertainty'``
GMVI_model       ``'GMVI'``, ``'GMVI_uncertainty'``
WCC_model        ``'WCC'``, ``'WCC_uncertainty'``
===============  ================================

Estimator Registry
~~~~~~~~~~~~~~~~~~

The ``dataset.estimators`` list tracks which models have been applied:

.. code-block:: python

   # Check what models have been run
   print(dataset.estimators)  # ['MAP', 'VI', 'GMVI']
   
   # Conditional logic based on available estimates
   if 'GMVI' in dataset.estimators:
       gmvi_preds = dataset.dataset_nodes['GMVI']
       gmvi_unc = dataset.dataset_nodes['GMVI_uncertainty']

Best Practices
--------------

1. **Create dataset first**: Always start by creating the ``FEPDataset`` object
2. **Train models sequentially**: Each model modifies the dataset in place
3. **Call add_predictions_to_dataset()**: Don't forget this step after training
4. **Check estimators list**: Use ``dataset.estimators`` to see what's available
5. **Access results from dataset**: All predictions are in ``dataset.dataset_nodes`` and ``dataset.dataset_edges``

Common Pitfalls
---------------

1. **Forgetting add_predictions_to_dataset()**: Predictions won't appear in the dataset
2. **GMVI requires get_posterior_estimates()**: Call this before add_predictions
3. **Model order doesn't matter**: But each model sees the same original FEP data
4. **Uncertainties may be NaN**: Not all inference methods provide uncertainties
