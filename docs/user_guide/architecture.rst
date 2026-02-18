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

.. code-block:: text

   maple/
   ├── dataset/
   │   ├── base_dataset.py          (BaseDataset ABC)
   │   ├── dataset.py               (FEPDataset)
   │   ├── FEP_benchmark_dataset.py (FEPBenchmarkDataset)
   │   └── synthetic_dataset.py     (SyntheticFEPDataset)
   │
   ├── models/
   │   ├── base.py                  (BaseEstimator ABC)
   │   ├── config.py                (Pydantic configs + enums)
   │   ├── graph_data.py            (GraphData dataclass)
   │   ├── probabilistic/
   │   │   ├── variational_estimator.py  (MAP / VI / MLE)
   │   │   └── gaussian_mixture_vi.py    (GMVI + outlier detection)
   │   └── deterministic/
   │       ├── cycle_closure.py          (WCC)
   │       └── spectral_correction.py    (WSFC / SFC)
   │
   ├── graph_analysis/
   │   ├── performance_stats.py
   │   ├── plotting_performance.py
   │   ├── graph_setup.py
   │   └── graph_cycle_analysis.py
   │
   └── utils/
       ├── parameter_sweep.py       (ParameterSweep)
       └── performance_tracker.py   (PerformanceTracker)

The Dataset as Central Hub
--------------------------

The ``FEPDataset`` object serves as the central hub in MAPLE:

.. code-block:: text

   Input Sources
   CSV Files | DataFrames | Benchmarks
         |
         v
   +------------------------------------------------------------------+
   |                  FEPDataset (Central Hub)                         |
   +------------------------------------------------------------------+
   |  dataset_nodes        - Node data and model predictions           |
   |  dataset_edges        - Edge data and model predictions           |
   |  cycle_data           - Graph structure for models                |
   |  node2idx / idx2node  - Node name <-> index mappings              |
   |  estimators[]         - List of applied model names               |
   +------------------------------------------------------------------+
         |
         v
   Consumers
   Models | Analysis | Visualization

Model-Dataset Interaction Pattern
---------------------------------

All MAPLE models follow a consistent interaction pattern with the dataset:

.. code-block:: text

   Step 1: Initialize
       model = ModelClass(config, dataset)
            |
            v
   Step 2: Extract Data
       model._extract_graph_data()
       Reads: dataset.cycle_data / dataset_edges
            |
            v
   Step 3: Fit
       model.fit()
       Produces: node_estimates, edge_estimates
            |
            v
   Step 4: Add to Dataset
       model.add_predictions_to_dataset()
       Writes to: dataset_nodes, dataset_edges

Available Models
----------------

MAPLE provides both probabilistic and deterministic inference methods:

.. code-block:: text

   Available Models
   ================

   probabilistic/                          deterministic/
   +--------------------------+            +--------------------------+
   | VariationalEstimator     |            | CycleClosureCorrection   |
   |   MAP  -> 'MAP'          |            |   WCC -> 'WCC',          |
   |   MLE  -> 'MLE'          |            |          'WCC_uncertainty'|
   |   VI   -> 'VI',          |            +--------------------------+
   |          'VI_uncertainty' |            | SpectralCorrection       |
   +--------------------------+            |   WSFC -> 'WSFC',        |
   | GaussianMixtureVI        |            |           'WSFC_unc.'    |
   |   GMVI -> 'GMVI',        |            |   SFC  -> 'SFC',        |
   |           'GMVI_unc.',    |            |           'SFC_unc.'     |
   |           outlier probs   |            +--------------------------+
   +--------------------------+

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
