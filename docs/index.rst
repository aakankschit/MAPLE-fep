MAPLE Documentation
===================

**MAPLE (Maximum A Posteriori Learning of Energies)** is a Python package for analyzing Free Energy Perturbation (FEP) data using probabilistic node models and Bayesian inference.

MAPLE provides tools for:

* **Bayesian Inference Models**: MAP, VI, and GMVI methods for node value estimation
* **Cycle Closure Correction**: WCC (Weighted Cycle Closure) baseline methods
* **Outlier Detection**: Probabilistic identification of problematic FEP edges
* **Uncertainty Quantification**: Full posterior distributions with confidence intervals
* **Performance Tracking**: Comprehensive tracking and comparison of model performance
* **Statistical Analysis**: Bootstrap confidence intervals and performance metrics
* **Visualization**: Professional plotting tools for FEP analysis

Core Architecture: Dataset-Centric Design
------------------------------------------

MAPLE uses a **dataset-centric architecture** where the ``FEPDataset`` object serves as the central hub for all data and predictions. Models read from the dataset, perform inference, and write their predictions back to the dataset.

.. graphviz::
   :align: center
   :caption: MAPLE Architecture: Dataset as Central Hub

   digraph maple_architecture {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12];
       edge [fontname="Helvetica", fontsize=10];
       compound=true;
       nodesep=0.4;
       ranksep=0.6;
       
       subgraph cluster_dataset {
           label="FEPDataset (Central Hub)";
           style="rounded,filled";
           fillcolor="#D4EDDA";
           fontname="Helvetica-Bold";
           fontsize=14;
           
           subgraph cluster_dataset_inner {
               style=invis;
               rank=same;
               nodes [label="dataset_nodes", fillcolor="#C3E6CB"];
               edges [label="dataset_edges", fillcolor="#C3E6CB"];
               est [label="estimators[]", fillcolor="#C3E6CB"];
               cycle [label="cycle_data", fillcolor="#C3E6CB"];
           }
       }
       
       subgraph cluster_models {
           label="Models";
           style="rounded,filled";
           fillcolor="#FFF3CD";
           fontname="Helvetica-Bold";
           fontsize=14;
           
           subgraph cluster_models_inner {
               style=invis;
               rank=same;
               nodemodel [label="NodeModel\n(MAP/VI/MLE)", fillcolor="#FFE69C"];
               gmvi [label="GMVI_model\n(outlier detection)", fillcolor="#FFE69C"];
               wcc [label="WCC_model\n(cycle closure)", fillcolor="#FFE69C"];
           }
       }
       
       subgraph cluster_results {
           label="Results Added to Dataset";
           style="rounded,filled";
           fillcolor="#F8D7DA";
           fontname="Helvetica-Bold";
           fontsize=14;
           
           subgraph cluster_results_inner {
               style=invis;
               rank=same;
               map_col [label="'MAP'", fillcolor="#F5C6CB"];
               vi_col [label="'VI'", fillcolor="#F5C6CB"];
               gmvi_col [label="'GMVI'", fillcolor="#F5C6CB"];
               wcc_col [label="'WCC'", fillcolor="#F5C6CB"];
           }
       }
       
       cycle -> nodemodel [ltail=cluster_dataset, lhead=cluster_models, label="get_graph_data()"];
       nodemodel -> map_col [ltail=cluster_models, lhead=cluster_results, label="add_predictions_to_dataset()"];
   }

This design allows you to:

1. **Train multiple models** on the same dataset
2. **Compare predictions** side-by-side in the dataset DataFrames
3. **Track which models** have been applied via ``dataset.estimators``
4. **Access all results** from a single dataset object

Quick Start
-----------

.. code-block:: python

   from maple.dataset import FEPDataset
   from maple.models import NodeModel, NodeModelConfig, GMVI_model, GMVIConfig
   from maple.models import PriorType, GuideType
   
   # Step 1: Create dataset from benchmark or your own data
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # Step 2: Configure and train MAP model
   map_config = NodeModelConfig(
       learning_rate=0.01,
       num_steps=1000,
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA  # MAP inference
   )
   map_model = NodeModel(config=map_config, dataset=dataset)
   map_model.train()
   map_model.add_predictions_to_dataset()  # Adds 'MAP' column
   
   # Step 3: Configure and train GMVI model (with outlier detection)
   gmvi_config = GMVIConfig(prior_std=5.0, outlier_prob=0.2)
   gmvi_model = GMVI_model(dataset=dataset, config=gmvi_config)
   gmvi_model.fit()
   gmvi_model.get_posterior_estimates()
   gmvi_model.add_predictions_to_dataset()  # Adds 'GMVI' column
   
   # Step 4: Access all predictions from the dataset
   print(dataset.dataset_nodes.columns)
   # ['Name', 'Exp. DeltaG', 'Pred. DeltaG', 'MAP', 'GMVI', 'GMVI_uncertainty']
   
   print(dataset.estimators)
   # ['MAP', 'GMVI']

Data Flow Pattern
-----------------

The following diagram shows the complete data flow in a typical MAPLE workflow:

.. graphviz::
   :align: center
   :caption: MAPLE Data Flow: From Data Loading to Results

   digraph data_flow {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12];
       edge [fontname="Helvetica", fontsize=10];
       compound=true;
       nodesep=0.5;
       ranksep=0.7;
       
       subgraph cluster_input {
           label="1. Input Sources";
           style="rounded,filled";
           fillcolor="#E3F2FD";
           fontname="Helvetica-Bold";
           fontsize=14;
           
           subgraph cluster_input_inner {
               style=invis;
               rank=same;
               benchmark [label="Benchmark\nDatasets", fillcolor="#BBDEFB"];
               dataframes [label="Pandas\nDataFrames", fillcolor="#BBDEFB"];
               csvfiles [label="CSV\nFiles", fillcolor="#BBDEFB"];
           }
       }
       
       subgraph cluster_dataset {
           label="2. FEPDataset (Central Hub)";
           style="rounded,filled";
           fillcolor="#E8F5E9";
           fontname="Helvetica-Bold";
           fontsize=14;
           
           subgraph cluster_dataset_inner {
               style=invis;
               rank=same;
               ds_nodes [label="dataset_nodes", fillcolor="#C8E6C9"];
               ds_edges [label="dataset_edges", fillcolor="#C8E6C9"];
               ds_graph [label="cycle_data", fillcolor="#C8E6C9"];
           }
       }
       
       subgraph cluster_train {
           label="3. Train Models";
           style="rounded,filled";
           fillcolor="#FFF8E1";
           fontname="Helvetica-Bold";
           fontsize=14;
           
           subgraph cluster_train_inner {
               style=invis;
               rank=same;
               train_map [label="NodeModel\n.train()", fillcolor="#FFECB3"];
               train_gmvi [label="GMVI_model\n.fit()", fillcolor="#FFECB3"];
               train_wcc [label="WCC_model\n.fit()", fillcolor="#FFECB3"];
           }
       }
       
       subgraph cluster_result {
           label="4. Results in Dataset";
           style="rounded,filled";
           fillcolor="#FCE4EC";
           fontname="Helvetica-Bold";
           fontsize=14;
           
           subgraph cluster_result_inner {
               style=invis;
               rank=same;
               res_cols [label="MAP | VI | GMVI | WCC\ncolumns added", fillcolor="#F8BBD9"];
               res_est [label="estimators[]\nupdated", fillcolor="#F8BBD9"];
           }
       }
       
       dataframes -> ds_nodes [lhead=cluster_dataset];
       ds_edges -> train_gmvi [ltail=cluster_dataset, lhead=cluster_train];
       train_gmvi -> res_cols [ltail=cluster_train, lhead=cluster_result, label="add_predictions_to_dataset()"];
   }

Installation
------------

.. code-block:: bash

   pip install maple-fep

Or for development:

.. code-block:: bash

   git clone https://github.com/maple-contributors/MAPLE.git
   cd MAPLE
   pip install -e .

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/architecture
   user_guide/tutorials
   user_guide/parameter_optimization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/models
   api/dataset
   api/utils
   api/graph_analysis

.. toctree::
   :maxdepth: 1
   :caption: Interactive Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic_usage
   examples/parameter_sweeps
   examples/fep_benchmark_tutorial

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer/contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
