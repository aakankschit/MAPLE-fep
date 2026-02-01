Models Package (``maple.models``)
==================================

The ``maple.models`` package provides statistical models for FEP analysis using Bayesian inference and variational methods. All models follow the **dataset-centric pattern**: they read from a dataset, perform inference, and write predictions back to the dataset.

Overview
--------

.. code-block:: text

   +============================================================================+
   |                      MODEL-DATASET INTERACTION                              |
   +============================================================================+
   
   All models follow the same pattern:
   
   1. INITIALIZE with dataset reference
      model = ModelClass(config=config, dataset=dataset)
   
   2. TRAIN/FIT the model
      model.train()  # or model.fit()
   
   3. ADD PREDICTIONS to dataset
      model.add_predictions_to_dataset()
   
   Result: dataset.dataset_nodes now has new prediction columns!

Available Models
----------------

.. code-block:: text

   +--------------------+------------------+------------------------------------+
   | Model              | Inference Type   | Columns Added to Dataset           |
   +--------------------+------------------+------------------------------------+
   | NodeModel          | MAP              | 'MAP'                              |
   |                    | MLE              | 'MLE'                              |
   |                    | VI               | 'VI', 'VI_uncertainty'             |
   +--------------------+------------------+------------------------------------+
   | GMVI_model         | Full-rank VI     | 'GMVI', 'GMVI_uncertainty'         |
   |                    | + outlier det.   |                                    |
   +--------------------+------------------+------------------------------------+
   | WCC_model          | Weighted cycle   | 'WCC', 'WCC_uncertainty'           |
   |                    | closure          |                                    |
   +--------------------+------------------+------------------------------------+

Quick Example
-------------

.. code-block:: python

   from maple.dataset import FEPDataset
   from maple.models import (
       NodeModel, NodeModelConfig,
       GMVI_model, GMVIConfig,
       PriorType, GuideType
   )
   
   # Create dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # =====================================================
   # NodeModel: MAP Inference
   # =====================================================
   map_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,  # MAP
       learning_rate=0.01,
       num_steps=5000
   )
   map_model = NodeModel(config=map_config, dataset=dataset)
   map_model.train()
   map_model.add_predictions_to_dataset()  # Adds 'MAP' column
   
   # =====================================================
   # NodeModel: Variational Inference
   # =====================================================
   vi_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_NORMAL,  # VI
       learning_rate=0.01,
       num_steps=5000
   )
   vi_model = NodeModel(config=vi_config, dataset=dataset)
   vi_model.train()
   vi_model.add_predictions_to_dataset()  # Adds 'VI' and 'VI_uncertainty'
   
   # =====================================================
   # GMVI_model: Outlier-Robust VI
   # =====================================================
   gmvi_config = GMVIConfig(
       prior_std=5.0,
       normal_std=1.0,
       outlier_std=3.0,
       outlier_prob=0.2,
       n_epochs=2000
   )
   gmvi_model = GMVI_model(dataset=dataset, config=gmvi_config)
   gmvi_model.fit()
   gmvi_model.get_posterior_estimates()  # REQUIRED before add_predictions
   gmvi_model.add_predictions_to_dataset()  # Adds 'GMVI' and 'GMVI_uncertainty'
   
   # All predictions are now in the dataset!
   print(dataset.dataset_nodes.columns.tolist())
   # ['Name', 'Exp. DeltaG', 'Pred. DeltaG', 'MAP', 'VI', 'VI_uncertainty', 
   #  'GMVI', 'GMVI_uncertainty']
   
   print(dataset.estimators)
   # ['MAP', 'VI', 'GMVI']

The add_predictions_to_dataset() Method
---------------------------------------

This is the key method that connects models to the dataset. Every model implements it:

.. code-block:: text

   model.add_predictions_to_dataset()
   
   What it does:
   +--------------------------------------------------------------------+
   |                                                                     |
   | 1. Reads model.node_estimates                                       |
   |    -> Writes to dataset.dataset_nodes['{MODEL}']                    |
   |                                                                     |
   | 2. Reads model.edge_estimates                                       |
   |    -> Writes to dataset.dataset_edges['{MODEL}']                    |
   |                                                                     |
   | 3. If uncertainties available:                                      |
   |    -> Writes to dataset.dataset_nodes['{MODEL}_uncertainty']        |
   |    -> Writes to dataset.dataset_edges['{MODEL}_uncertainty']        |
   |                                                                     |
   | 4. Registers model in dataset.estimators                            |
   |    -> dataset.estimators.append('{MODEL}')                          |
   |                                                                     |
   +--------------------------------------------------------------------+

**Important:** 

- For ``GMVI_model``, you must call ``get_posterior_estimates()`` before ``add_predictions_to_dataset()``
- The model must be trained before calling this method
- Column names are determined automatically by the model type

NodeModel
---------

The ``NodeModel`` class provides MAP, VI, and MLE inference for FEP node values.

How NodeModel Determines Column Names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   +-------------------+----------------------+------------------+
   | guide_type        | prior_type           | Column Name      |
   +-------------------+----------------------+------------------+
   | AUTO_DELTA        | NORMAL, LAPLACE, ... | 'MAP'            |
   | AUTO_DELTA        | UNIFORM              | 'MLE'            |
   | AUTO_NORMAL       | any                  | 'VI'             |
   +-------------------+----------------------+------------------+

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   from maple.models import NodeModelConfig, PriorType, GuideType, ErrorDistributionType
   
   config = NodeModelConfig(
       # Optimization
       learning_rate=0.01,        # Step size for Adam optimizer
       num_steps=5000,            # Number of optimization steps
       
       # Prior distribution
       prior_type=PriorType.NORMAL,      # NORMAL, LAPLACE, UNIFORM
       prior_std=5.0,                     # Prior standard deviation
       
       # Inference type
       guide_type=GuideType.AUTO_DELTA,  # AUTO_DELTA (MAP) or AUTO_NORMAL (VI)
       
       # Error model
       error_distribution=ErrorDistributionType.NORMAL,
       error_std=1.0,
       
       # Other options
       center_fep_data=True,      # Mean-center FEP values
       random_seed=42             # For reproducibility
   )

Usage
~~~~~

.. code-block:: python

   from maple.models import NodeModel, NodeModelConfig, PriorType, GuideType
   from maple.dataset import FEPDataset
   
   # Create dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # MAP inference
   map_config = NodeModelConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,
       num_steps=5000
   )
   model = NodeModel(config=map_config, dataset=dataset)
   
   # Train
   model.train()
   
   # Add predictions to dataset
   model.add_predictions_to_dataset()
   
   # Access results
   map_predictions = dataset.dataset_nodes['MAP']
   print(f"Applied estimators: {dataset.estimators}")

GMVI_model
----------

The ``GMVI_model`` (Gaussian Mixture Variational Inference) provides:

- Full-rank covariance for node value posterior
- Outlier detection via Gaussian mixture likelihood
- Uncertainty quantification for all predictions

Mathematical Model
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Prior:     p(z) = N(z | 0, sigma_0^2 * I)
   
   Likelihood (mixture):
   p(y_e | z) = pi * N(y_e | z_j - z_i, sigma_2^2)      [outlier]
              + (1-pi) * N(y_e | z_j - z_i, sigma_1^2)  [normal]
   
   Parameters:
   - sigma_0 (prior_std): Prior standard deviation
   - sigma_1 (normal_std): Likelihood std for normal edges
   - sigma_2 (outlier_std): Likelihood std for outlier edges
   - pi (outlier_prob): Global outlier probability

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   from maple.models import GMVIConfig
   
   config = GMVIConfig(
       # Model parameters
       prior_std=5.0,        # sigma_0: Prior std
       normal_std=1.0,       # sigma_1: Normal edge std
       outlier_std=3.0,      # sigma_2: Outlier edge std
       outlier_prob=0.2,     # pi: Outlier probability
       
       # Training
       learning_rate=0.01,
       n_epochs=2000,
       n_samples=20,         # MC samples for ELBO
       patience=100,         # Early stopping patience
       
       # Regularization
       kl_weight=1.0         # Weight for KL divergence term
   )

Usage
~~~~~

.. code-block:: python

   from maple.models import GMVI_model, GMVIConfig
   from maple.dataset import FEPDataset
   
   # Create dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # Configure GMVI
   config = GMVIConfig(
       prior_std=5.0,
       normal_std=1.0,
       outlier_std=3.0,
       outlier_prob=0.2,
       n_epochs=2000
   )
   
   # Create and train
   model = GMVI_model(dataset=dataset, config=config)
   model.fit()
   
   # IMPORTANT: Get posterior estimates before adding to dataset
   model.get_posterior_estimates()
   
   # Add predictions to dataset
   model.add_predictions_to_dataset()
   
   # Access results
   gmvi_predictions = dataset.dataset_nodes['GMVI']
   gmvi_uncertainty = dataset.dataset_nodes['GMVI_uncertainty']
   
   # Get outlier probabilities for each edge
   outlier_probs = model.compute_edge_outlier_probabilities()
   print(f"Edges with >70% outlier probability: {sum(p > 0.7 for p in outlier_probs)}")

WCC_model
---------

The ``WCC_model`` (Weighted Cycle Closure) provides a baseline method for cycle closure correction.

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   from maple.models import WCCConfig
   
   config = WCCConfig(
       tolerance=1e-6,       # Convergence tolerance
       max_iterations=1000   # Maximum iterations
   )

Usage
~~~~~

.. code-block:: python

   from maple.models import WCC_model, WCCConfig
   from maple.dataset import FEPDataset
   
   # Create dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")
   
   # Create and train
   config = WCCConfig(tolerance=1e-6)
   model = WCC_model(dataset=dataset, config=config)
   model.fit()
   
   # Add predictions to dataset
   model.add_predictions_to_dataset()
   
   # Access results
   wcc_predictions = dataset.dataset_nodes['WCC']
   wcc_uncertainty = dataset.dataset_nodes['WCC_uncertainty']

API Reference
-------------

Models
~~~~~~

NodeModel
^^^^^^^^^

.. autoclass:: maple.models.NodeModel
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

* ``train()`` - Run inference optimization
* ``add_predictions_to_dataset()`` - Write predictions to dataset
* ``get_results()`` - Get dictionary of all results
* ``plot_training_history()`` - Visualize training convergence

**Key Attributes:**

* ``node_estimates`` - Dict mapping node names to estimated values
* ``edge_estimates`` - Dict mapping (src, dst) tuples to estimated values
* ``node_uncertainties`` - Dict of uncertainties (VI only)
* ``loss_history`` - List of loss values during training

GMVI_model
^^^^^^^^^^

.. autoclass:: maple.models.GMVI_model
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

* ``fit()`` - Train the model
* ``get_posterior_estimates()`` - Extract posterior means and stds (REQUIRED before add_predictions)
* ``add_predictions_to_dataset()`` - Write predictions to dataset
* ``compute_edge_outlier_probabilities()`` - Get per-edge outlier probabilities
* ``evaluate_predictions()`` - Get performance metrics

**Key Attributes:**

* ``node_estimates`` - Dict mapping node names to estimated values
* ``edge_estimates`` - Dict mapping (src, dst) tuples to estimated values
* ``node_uncertainties`` - Dict of posterior standard deviations
* ``edge_uncertainties`` - Dict of edge prediction uncertainties

Configuration Classes
~~~~~~~~~~~~~~~~~~~~~

NodeModelConfig
^^^^^^^^^^^^^^^

.. autoclass:: maple.models.NodeModelConfig
   :members:
   :undoc-members:
   :show-inheritance:

GMVIConfig
^^^^^^^^^^

.. autoclass:: maple.models.GMVIConfig
   :members:
   :undoc-members:
   :show-inheritance:

BaseModelConfig
^^^^^^^^^^^^^^^

.. autoclass:: maple.models.BaseModelConfig
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Enums
~~~~~~~~~~~~~~~~~~~~

PriorType
^^^^^^^^^

.. autoclass:: maple.models.PriorType
   :members:
   :undoc-members:

Available options:

* ``NORMAL`` - Normal (Gaussian) prior
* ``LAPLACE`` - Laplace (double-exponential) prior
* ``UNIFORM`` - Uniform (improper) prior

GuideType
^^^^^^^^^

.. autoclass:: maple.models.GuideType
   :members:
   :undoc-members:

Available options:

* ``AUTO_DELTA`` - Delta function guide (MAP inference)
* ``AUTO_NORMAL`` - Normal guide (variational inference)

ErrorDistributionType
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: maple.models.ErrorDistributionType
   :members:
   :undoc-members:

Available options:

* ``NORMAL`` - Normal error distribution

Data Structures
~~~~~~~~~~~~~~~

GraphData
^^^^^^^^^

.. autoclass:: maple.models.GraphData
   :members:
   :undoc-members:
   :show-inheritance:

Internal data structure used by models:

.. code-block:: python

   @dataclass
   class GraphData:
       source_nodes: List[int]      # Source node indices
       target_nodes: List[int]      # Target node indices
       edge_values: List[float]     # FEP edge values
       num_nodes: int               # Number of nodes
       num_edges: int               # Number of edges
       node_to_idx: Dict[str, int]  # Node name -> index
       idx_to_node: Dict[int, str]  # Index -> node name

Utility Functions
~~~~~~~~~~~~~~~~~

create_config
^^^^^^^^^^^^^

.. autofunction:: maple.models.create_config

Best Practices
--------------

1. **Always call add_predictions_to_dataset()**: Predictions won't appear in the dataset otherwise

2. **For GMVI, call get_posterior_estimates() first**: 

   .. code-block:: python
   
      gmvi.fit()
      gmvi.get_posterior_estimates()  # REQUIRED
      gmvi.add_predictions_to_dataset()

3. **Check dataset.estimators**: See which models have been applied:

   .. code-block:: python
   
      if 'GMVI' in dataset.estimators:
          gmvi_preds = dataset.dataset_nodes['GMVI']

4. **Compare models easily**: All predictions are in the same DataFrame:

   .. code-block:: python
   
      for estimator in dataset.estimators:
          preds = dataset.dataset_nodes[estimator]
          # Compare with experimental values
