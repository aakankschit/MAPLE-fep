Dataset Package (``maple.dataset``)
===================================

The ``maple.dataset`` package provides the central data hub for MAPLE. The ``FEPDataset`` class serves as the central object that stores all FEP data and model predictions.

Overview
--------

In MAPLE's dataset-centric architecture, the ``FEPDataset`` object:

1. **Stores FEP data** in ``dataset_nodes`` and ``dataset_edges`` DataFrames
2. **Provides graph structure** to models via ``get_graph_data()``
3. **Accumulates predictions** from multiple models
4. **Tracks applied estimators** in the ``estimators`` list

.. code-block:: text

   +------------------------------------------------------------------+
   |                    FEPDataset: Central Data Hub                   |
   +------------------------------------------------------------------+
   |                                                                   |
   |  INPUT (from user):                 OUTPUT (to models & analysis) |
   |  -----------------                  ---------------------------   |
   |  - Edge DataFrame                   - cycle_data dict             |
   |  - Node DataFrame                   - node2idx / idx2node maps    |
   |  - CSV files                        - get_graph_data()            |
   |  - Benchmark name                   - get_dataframes()            |
   |                                                                   |
   |  ACCUMULATES (from models):                                       |
   |  -------------------------                                        |
   |  - MAP predictions       -> dataset_nodes['MAP']                  |
   |  - VI predictions        -> dataset_nodes['VI']                   |
   |  - GMVI predictions      -> dataset_nodes['GMVI']                 |
   |  - WCC predictions       -> dataset_nodes['WCC']                  |
   |  - Uncertainties         -> dataset_nodes['{model}_uncertainty']  |
   |  - Estimator registry    -> estimators list                       |
   |                                                                   |
   +------------------------------------------------------------------+

Quick Example
-------------

.. code-block:: python

   from maple.dataset import FEPDataset
   from maple.models import NodeModel, NodeModelConfig, GMVI_model, GMVIConfig
   
   # Create dataset (multiple options)
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # Initial state
   print(dataset.dataset_nodes.columns.tolist())
   # ['Name', 'Exp. DeltaG', 'Pred. DeltaG']
   
   print(dataset.estimators)
   # []
   
   # Train MAP model and add predictions
   map_model = NodeModel(config=NodeModelConfig(), dataset=dataset)
   map_model.train()
   map_model.add_predictions_to_dataset()
   
   # After MAP model
   print(dataset.dataset_nodes.columns.tolist())
   # ['Name', 'Exp. DeltaG', 'Pred. DeltaG', 'MAP']
   
   print(dataset.estimators)
   # ['MAP']
   
   # Train GMVI model and add predictions
   gmvi = GMVI_model(dataset=dataset, config=GMVIConfig())
   gmvi.fit()
   gmvi.get_posterior_estimates()
   gmvi.add_predictions_to_dataset()
   
   # After GMVI model
   print(dataset.dataset_nodes.columns.tolist())
   # ['Name', 'Exp. DeltaG', 'Pred. DeltaG', 'MAP', 'GMVI', 'GMVI_uncertainty']
   
   print(dataset.estimators)
   # ['MAP', 'GMVI']

Creating a Dataset
------------------

The ``FEPDataset`` class supports multiple initialization methods:

From Benchmark Dataset
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load a standard benchmark dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # Available benchmarks: cdk8, cmet, eg5, hif2a, pfkfb3, shp2, syk, tnks2

From DataFrames
~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   
   # Create DataFrames
   edges_df = pd.DataFrame({
       'Source': ['molA', 'molB', 'molA'],
       'Destination': ['molB', 'molC', 'molC'],
       'DeltaDeltaG': [-2.3, 1.1, -1.2],
       'DeltaDeltaG Error': [0.5, 0.4, 0.6]
   })
   
   nodes_df = pd.DataFrame({
       'Name': ['molA', 'molB', 'molC'],
       'Exp. DeltaG': [-8.5, -6.2, -7.3]
   })
   
   dataset = FEPDataset(dataset_nodes=nodes_df, dataset_edges=edges_df)

From CSV Files
~~~~~~~~~~~~~~

.. code-block:: python

   dataset = FEPDataset(
       nodes_csv_path="path/to/nodes.csv",
       edges_csv_path="path/to/edges.csv"
   )

From Edges Only
~~~~~~~~~~~~~~~

.. code-block:: python

   # Node values will be derived automatically via graph traversal
   dataset = FEPDataset(dataset_edges=edges_df)

Data Flow: How Models Interact with Dataset
-------------------------------------------

The following diagram shows how models interact with the dataset:

.. code-block:: text

   MODEL-DATASET INTERACTION PATTERN
   =================================
   
   1. Model receives dataset reference
   -----------------------------------
       model = NodeModel(config=config, dataset=dataset)
                                           ^
                                           |
                                   Stored as self.dataset
   
   2. Model extracts graph data
   ----------------------------
       +-------------+                    +-------------+
       |   Model     |  model._extract_   |  Dataset    |
       |             |  graph_data()      |             |
       |  graph_data |<-------------------| cycle_data  |
       +-------------+                    +-------------+
   
   3. Model trains on extracted data
   ---------------------------------
       model.train()  # Internal optimization
       # Produces: model.node_estimates, model.edge_estimates
   
   4. Model writes predictions back
   --------------------------------
       model.add_predictions_to_dataset()
       
       +-------------+                    +-------------------+
       |   Model     |    Writes to       |     Dataset       |
       |             | ------------------>|                   |
       | node_ests   |                    | dataset_nodes     |
       | edge_ests   |                    |   ['MAP'] = ...   |
       | uncert.     |                    | dataset_edges     |
       +-------------+                    |   ['MAP'] = ...   |
                                          | estimators.append |
                                          +-------------------+

Dataset Attributes
------------------

After initialization, the dataset has these key attributes:

.. code-block:: python

   # DataFrames containing all data and predictions
   dataset.dataset_nodes   # pd.DataFrame with node data
   dataset.dataset_edges   # pd.DataFrame with edge data
   
   # Graph structure for models
   dataset.cycle_data      # Dict with 'N', 'M', 'src', 'dst', 'FEP', 'CCC'
   dataset.node2idx        # Dict mapping node names to indices
   dataset.idx2node        # Dict mapping indices to node names
   
   # Estimator tracking
   dataset.estimators      # List of applied model names: ['MAP', 'VI', 'GMVI', ...]

Column Naming Convention
------------------------

Models add columns with these names:

.. code-block:: text

   +------------------+------------------------------------------+
   | Model Type       | Columns Added to dataset_nodes/edges     |
   +------------------+------------------------------------------+
   | NodeModel (MAP)  | 'MAP'                                    |
   | NodeModel (MLE)  | 'MLE'                                    |
   | NodeModel (VI)   | 'VI', 'VI_uncertainty'                   |
   | GMVI_model       | 'GMVI', 'GMVI_uncertainty'               |
   | WCC_model        | 'WCC', 'WCC_uncertainty'                 |
   +------------------+------------------------------------------+

API Reference
-------------

Base Classes
~~~~~~~~~~~~

.. autoclass:: maple.dataset.BaseDataset
   :members:
   :undoc-members:
   :show-inheritance:

Main Dataset Classes
~~~~~~~~~~~~~~~~~~~~

FEPDataset
^^^^^^^^^^

.. autoclass:: maple.dataset.FEPDataset
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

* ``get_graph_data()`` - Returns the cycle_data dict for models
* ``get_node_mapping()`` - Returns (node2idx, idx2node) mappings
* ``get_dataframes()`` - Returns (dataset_edges, dataset_nodes)
* ``get_estimators()`` - Returns list of applied estimator names

**Key Attributes:**

* ``dataset_nodes`` - DataFrame with node data and model predictions
* ``dataset_edges`` - DataFrame with edge data and model predictions
* ``estimators`` - List tracking which models have been applied
* ``cycle_data`` - Dict containing graph structure for models
* ``node2idx`` - Mapping from node names to indices
* ``idx2node`` - Mapping from indices to node names

FEPBenchmarkDataset
^^^^^^^^^^^^^^^^^^^

.. autoclass:: maple.dataset.FEPBenchmarkDataset
   :members:
   :undoc-members:
   :show-inheritance:

**Available Benchmarks:**

* ``cdk8`` - CDK8 kinase dataset
* ``cmet`` - c-Met kinase dataset
* ``eg5`` - Eg5 kinesin dataset
* ``hif2a`` - HIF2a dataset
* ``pfkfb3`` - PFKFB3 dataset
* ``shp2`` - SHP2 dataset
* ``syk`` - Syk kinase dataset
* ``tnks2`` - TNKS2 dataset

Specialized Datasets
~~~~~~~~~~~~~~~~~~~~

SyntheticFEPDataset
^^^^^^^^^^^^^^^^^^^

.. autoclass:: maple.dataset.SyntheticFEPDataset
   :members:
   :undoc-members:
   :show-inheritance:

**Usage:**

.. code-block:: python

   from maple.dataset import SyntheticFEPDataset
   
   # Generate synthetic FEP data for testing
   synthetic = SyntheticFEPDataset(
       num_nodes=10,
       num_edges=20,
       noise_std=0.5
   )

Utility Functions
-----------------

derive_nodes_from_edges
~~~~~~~~~~~~~~~~~~~~~~~

Class method to derive node values from edge data:

.. code-block:: python

   edges_df, nodes_df = FEPDataset.derive_nodes_from_edges(
       dataset_edges=edges_df,
       reference_node="molA"  # Optional: set this node to 0.0
   )
   
   # Now use these DataFrames to create a dataset
   dataset = FEPDataset(dataset_nodes=nodes_df, dataset_edges=edges_df)

Best Practices
--------------

1. **Create dataset first**: Always start by creating the ``FEPDataset`` object
2. **Use add_predictions_to_dataset()**: Don't forget to call this after training each model
3. **Check estimators list**: Use ``dataset.estimators`` to see which models have been applied
4. **Access results from dataset**: All predictions are stored in ``dataset.dataset_nodes`` and ``dataset.dataset_edges``
5. **Compare models easily**: All model predictions are in the same DataFrame for easy comparison

Example: Multi-Model Comparison
-------------------------------

.. code-block:: python

   from maple.dataset import FEPDataset
   from maple.models import NodeModel, NodeModelConfig, GMVI_model, GMVIConfig
   from maple.models import PriorType, GuideType
   from maple.graph_analysis import compute_simple_statistics
   
   # Create dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # Train multiple models
   models_to_train = [
       ("MAP", NodeModelConfig(guide_type=GuideType.AUTO_DELTA)),
       ("VI", NodeModelConfig(guide_type=GuideType.AUTO_NORMAL)),
   ]
   
   for name, config in models_to_train:
       model = NodeModel(config=config, dataset=dataset)
       model.train()
       model.add_predictions_to_dataset()
   
   # Train GMVI
   gmvi = GMVI_model(dataset=dataset, config=GMVIConfig())
   gmvi.fit()
   gmvi.get_posterior_estimates()
   gmvi.add_predictions_to_dataset()
   
   # Compare all models from the dataset
   exp = dataset.dataset_nodes['Exp. DeltaG'].values
   
   print("Model Comparison:")
   print("-" * 40)
   for estimator in dataset.estimators:
       pred = dataset.dataset_nodes[estimator].values
       stats = compute_simple_statistics(exp, pred)
       print(f"{estimator:6s}: RMSE={stats['RMSE']:.3f}, R2={stats['R2']:.3f}")
