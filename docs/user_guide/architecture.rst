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

MAPLE is organized into four main packages, with models split into **probabilistic** and **deterministic** subpackages:

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

               base_dataset [label="base_dataset.py\n(BaseDataset ABC)", fillcolor="#C3E6CB"];
               dataset [label="dataset.py\n(FEPDataset)", fillcolor="#C3E6CB"];
               benchmark [label="FEP_benchmark_dataset.py\n(FEPBenchmarkDataset)", fillcolor="#C3E6CB"];
               synthetic [label="synthetic_dataset.py\n(SyntheticFEPDataset)", fillcolor="#C3E6CB"];

               base_dataset -> dataset -> benchmark -> synthetic [style=invis];
           }

           subgraph cluster_models {
               label="models/";
               style="rounded,filled";
               fillcolor="#FFF3CD";
               fontname="Helvetica-Bold";
               fontsize=13;

               model_base [label="base.py  (BaseEstimator ABC)\nconfig.py  (Pydantic configs + enums)\ngraph_data.py  (GraphData dataclass)", fillcolor="#FFE69C"];

               subgraph cluster_prob {
                   label="probabilistic/";
                   style="rounded,filled";
                   fillcolor="#FFECB3";
                   fontname="Helvetica";
                   fontsize=11;

                   var_est [label="variational_estimator.py\n(MAP / VI / MLE)", fillcolor="#FFE082"];
                   gmvi_model [label="gaussian_mixture_vi.py\n(GMVI + outlier detection)", fillcolor="#FFE082"];

                   var_est -> gmvi_model [style=invis];
               }

               subgraph cluster_det {
                   label="deterministic/";
                   style="rounded,filled";
                   fillcolor="#FFECB3";
                   fontname="Helvetica";
                   fontsize=11;

                   wcc_model [label="cycle_closure.py\n(WCC)", fillcolor="#FFE082"];
                   wsfc_model [label="spectral_correction.py\n(WSFC / SFC)", fillcolor="#FFE082"];

                   wcc_model -> wsfc_model [style=invis];
               }

               model_base -> var_est [style=invis];
           }

           subgraph cluster_analysis {
               label="graph_analysis/";
               style="rounded,filled";
               fillcolor="#F8D7DA";
               fontname="Helvetica-Bold";
               fontsize=13;

               perf_stats [label="performance_stats.py", fillcolor="#F5C6CB"];
               plotting [label="plotting_performance.py", fillcolor="#F5C6CB"];
               graph_setup [label="graph_setup.py", fillcolor="#F5C6CB"];
               cycle_analysis [label="graph_cycle_analysis.py", fillcolor="#F5C6CB"];

               perf_stats -> plotting -> graph_setup -> cycle_analysis [style=invis];
           }

           subgraph cluster_utils {
               label="utils/";
               style="rounded,filled";
               fillcolor="#D1ECF1";
               fontname="Helvetica-Bold";
               fontsize=13;

               param_sweep [label="parameter_sweep.py\n(ParameterSweep)", fillcolor="#BEE5EB"];
               perf_tracker [label="performance_tracker.py\n(PerformanceTracker)", fillcolor="#BEE5EB"];

               param_sweep -> perf_tracker [style=invis];
           }
       }

       maple -> base_dataset [lhead=cluster_dataset];
       maple -> model_base [lhead=cluster_models];
       maple -> perf_stats [lhead=cluster_analysis];
       maple -> param_sweep [lhead=cluster_utils];
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

           init [label="model = ModelClass(config, dataset)", fillcolor="#BBDEFB"];
       }

       subgraph cluster_step2 {
           label="Step 2: Extract Data";
           style="rounded,filled";
           fillcolor="#E8F5E9";
           fontname="Helvetica-Bold";
           fontsize=13;

           extract [label="model._extract_graph_data()\nReads: dataset.cycle_data / dataset_edges", fillcolor="#C8E6C9"];
       }

       subgraph cluster_step3 {
           label="Step 3: Fit";
           style="rounded,filled";
           fillcolor="#FFF8E1";
           fontname="Helvetica-Bold";
           fontsize=13;

           train [label="model.fit()\nProduces: node_estimates, edge_estimates", fillcolor="#FFECB3"];
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

MAPLE provides both probabilistic and deterministic inference methods:

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
                   label="VariationalEstimator\n(probabilistic/)";
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
                   label="GaussianMixtureVI\n(probabilistic/)";
                   style="rounded,filled";
                   fillcolor="#E8F5E9";
                   fontname="Helvetica-Bold";
                   fontsize=12;

                   gmvi [label="Full-rank VI\n+ Outlier Detection", fillcolor="#C8E6C9"];
               }

               subgraph cluster_wcc {
                   label="CycleClosureCorrection\n(deterministic/)";
                   style="rounded,filled";
                   fillcolor="#FFF8E1";
                   fontname="Helvetica-Bold";
                   fontsize=12;

                   wcc [label="Weighted\nCycle Closure", fillcolor="#FFECB3"];
               }

               subgraph cluster_wsfc {
                   label="SpectralCorrection\n(deterministic/)";
                   style="rounded,filled";
                   fillcolor="#F3E5F5";
                   fontname="Helvetica-Bold";
                   fontsize=12;

                   wsfc [label="Graph Laplacian\nPseudoinverse", fillcolor="#E1BEE7"];
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
               col_wsfc [label="'WSFC'/'SFC', '{}_uncertainty'", fillcolor="#F8BBD9"];
           }
       }

       map -> col_map;
       mle -> col_mle;
       vi -> col_vi;
       gmvi -> col_gmvi;
       wcc -> col_wcc;
       wsfc -> col_wsfc;
   }

Probabilistic Models (``models.probabilistic``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**VariationalEstimator** (MAP/VI/MLE):

.. code-block:: python

   from maple.models import VariationalEstimator, VariationalEstimatorConfig, PriorType, GuideType

   # MAP inference
   config = VariationalEstimatorConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,
       num_steps=5000
   )
   model = VariationalEstimator(config=config, dataset=dataset)
   model.fit()
   model.add_predictions_to_dataset()  # Adds "MAP" column

**GaussianMixtureVI** (Gaussian Mixture Variational Inference):

.. code-block:: python

   from maple.models import GaussianMixtureVI, GaussianMixtureVIConfig

   config = GaussianMixtureVIConfig(
       prior_std=5.0,
       normal_std=1.0,
       outlier_std=3.0,
       outlier_prob=0.2
   )
   model = GaussianMixtureVI(dataset=dataset, config=config)
   model.fit()
   model.get_results()  # REQUIRED before add_predictions
   model.add_predictions_to_dataset()  # Adds "GMVI" and "GMVI_uncertainty"

   # Get outlier probabilities
   outlier_probs = model.compute_edge_outlier_probabilities()

Deterministic Models (``models.deterministic``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CycleClosureCorrection** (Weighted Cycle Closure):

.. code-block:: python

   from maple.models import CycleClosureCorrection, CycleClosureCorrectionConfig

   config = CycleClosureCorrectionConfig(tolerance=1e-6)
   model = CycleClosureCorrection(dataset=dataset, config=config)
   model.fit()
   model.add_predictions_to_dataset()  # Adds "WCC" and "WCC_uncertainty"

**SpectralCorrection** (Weighted Spectral Free-energy Correction):

.. code-block:: python

   from maple.models import SpectralCorrection, SpectralCorrectionConfig

   # WSFC: weighted by edge uncertainties
   config = SpectralCorrectionConfig(use_weights=True)
   model = SpectralCorrection(config=config, dataset=dataset)
   model.fit()
   model.add_predictions_to_dataset()  # Adds "WSFC" and "WSFC_uncertainty"

   # SFC: unweighted (equivalent to MLE)
   config = SpectralCorrectionConfig(use_weights=False)
   model = SpectralCorrection(config=config, dataset=dataset)
   model.fit()
   model.add_predictions_to_dataset()  # Adds "SFC" and "SFC_uncertainty"

Key Design Patterns
-------------------

Configuration via Pydantic
~~~~~~~~~~~~~~~~~~~~~~~~~~

All model configurations use Pydantic for validation:

.. code-block:: python

   from maple.models import VariationalEstimatorConfig, GaussianMixtureVIConfig

   # Validation happens automatically
   config = VariationalEstimatorConfig(
       learning_rate=0.01,      # Must be > 0
       num_steps=5000,          # Must be > 0
       prior_type=PriorType.NORMAL
   )

Column Naming Convention
~~~~~~~~~~~~~~~~~~~~~~~~

Models add columns with consistent naming:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Model Type
     - Columns Added
   * - MAP (VariationalEstimator)
     - ``'MAP'``
   * - MLE (VariationalEstimator)
     - ``'MLE'``
   * - VI (VariationalEstimator)
     - ``'VI'``, ``'VI_uncertainty'``
   * - GaussianMixtureVI
     - ``'GMVI'``, ``'GMVI_uncertainty'``
   * - CycleClosureCorrection
     - ``'WCC'``, ``'WCC_uncertainty'``
   * - SpectralCorrection (weighted)
     - ``'WSFC'``, ``'WSFC_uncertainty'``
   * - SpectralCorrection (unweighted)
     - ``'SFC'``, ``'SFC_uncertainty'``

Estimator Registry
~~~~~~~~~~~~~~~~~~

The ``dataset.estimators`` list tracks which models have been applied:

.. code-block:: python

   # Check what models have been run
   print(dataset.estimators)  # ['MAP', 'VI', 'GMVI', 'WSFC']

   # Conditional logic based on available estimates
   if 'GMVI' in dataset.estimators:
       gmvi_preds = dataset.dataset_nodes['GMVI']
       gmvi_unc = dataset.dataset_nodes['GMVI_uncertainty']

Best Practices
--------------

1. **Create dataset first**: Always start by creating the ``FEPDataset`` object
2. **Train models sequentially**: Each model modifies the dataset in place
3. **Call add_predictions_to_dataset()**: Don't forget this step after training
4. **For GaussianMixtureVI, call get_results() first**: Required before writing predictions
5. **Check estimators list**: Use ``dataset.estimators`` to see what's available
6. **Access results from dataset**: All predictions are in ``dataset.dataset_nodes`` and ``dataset.dataset_edges``
7. **Use SpectralCorrection for fast baselines**: It provides instant results via matrix solve (no iteration)

Common Pitfalls
---------------

1. **Forgetting add_predictions_to_dataset()**: Predictions won't appear in the dataset
2. **GMVI requires get_results()**: Call this before add_predictions
3. **SpectralCorrection needs edge errors for WSFC**: Without ``DeltaDeltaG Error`` column, it falls back to SFC
4. **Model order doesn't matter**: But each model sees the same original FEP data
5. **Uncertainties may be NaN**: Not all inference methods provide uncertainties (MAP/MLE don't)
