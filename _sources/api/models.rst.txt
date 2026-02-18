Models Package (``maple.models``)
==================================

The ``maple.models`` package provides statistical models for FEP analysis, organized into **probabilistic** (variational inference) and **deterministic** (closed-form) methods. All models follow the **dataset-centric pattern**: they read from a dataset, perform inference, and write predictions back to the dataset.

.. image:: ../MAPLE_logo.png
   :align: center
   :width: 250px

|

Overview
--------

.. code-block:: text

   +============================================================================+
   |                      MODEL-DATASET INTERACTION                              |
   +============================================================================+

   All models follow the same pattern:

   1. INITIALIZE with dataset reference
      model = ModelClass(config=config, dataset=dataset)

   2. FIT the model
      model.fit()

   3. ADD PREDICTIONS to dataset
      model.add_predictions_to_dataset()

   Result: dataset.dataset_nodes now has new prediction columns!

Package Structure
-----------------

.. code-block:: text

   maple.models/
   ├── base.py                    # BaseEstimator ABC
   ├── config.py                  # All configuration classes + enums
   ├── graph_data.py              # GraphData dataclass
   ├── probabilistic/
   │   ├── variational_estimator.py   # VariationalEstimator (MAP/VI/MLE)
   │   └── gaussian_mixture_vi.py     # GaussianMixtureVI (outlier detection)
   └── deterministic/
       ├── cycle_closure.py           # CycleClosureCorrection (WCC)
       └── spectral_correction.py     # SpectralCorrection (WSFC/SFC)

Available Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 20 44

   * - Model
     - Inference Type
     - Columns Added
     - Mathematical Formulation
   * - ``VariationalEstimator`` (MAP)
     - Probabilistic
     - ``'MAP'``
     - :math:`\mathbf{z}^* = \arg\max [\log p(\mathbf{y}|\mathbf{z}) + \log p(\mathbf{z})]`
   * - ``VariationalEstimator`` (MLE)
     - Probabilistic
     - ``'MLE'``
     - :math:`\mathbf{z}^* = \arg\max \log p(\mathbf{y}|\mathbf{z})` (uniform prior)
   * - ``VariationalEstimator`` (VI)
     - Probabilistic
     - ``'VI'``, ``'VI_uncertainty'``
     - :math:`\max_\phi \mathbb{E}_q[\log p(\mathbf{y}|\mathbf{z})] - D_{\text{KL}}[q \| p]`
   * - ``GaussianMixtureVI``
     - Probabilistic
     - ``'GMVI'``, ``'GMVI_uncertainty'``
     - Mixture likelihood: :math:`\pi \mathcal{N}(\sigma_2) + (1-\pi)\mathcal{N}(\sigma_1)`
   * - ``CycleClosureCorrection``
     - Deterministic
     - ``'WCC'``, ``'WCC_uncertainty'``
     - Iterative: :math:`y_e' = y_e - \epsilon_C \cdot w_e / W_C`
   * - ``SpectralCorrection`` (WSFC)
     - Deterministic
     - ``'WSFC'``, ``'WSFC_uncertainty'``
     - :math:`\mathbf{z}^* = L_W^+ B^T W \mathbf{x}`
   * - ``SpectralCorrection`` (SFC)
     - Deterministic
     - ``'SFC'``, ``'SFC_uncertainty'``
     - Unweighted: :math:`\mathbf{z}^* = L^+ B^T \mathbf{x}` (equivalent to MLE)

Quick Example
-------------

.. code-block:: python

   from maple.dataset import FEPDataset
   from maple.models import (
       VariationalEstimator, VariationalEstimatorConfig,
       GaussianMixtureVI, GaussianMixtureVIConfig,
       SpectralCorrection, SpectralCorrectionConfig,
       CycleClosureCorrection, CycleClosureCorrectionConfig,
       PriorType, GuideType
   )

   # Create dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")

   # =====================================================
   # VariationalEstimator: MAP Inference
   # =====================================================
   map_config = VariationalEstimatorConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,  # MAP
       learning_rate=0.01,
       num_steps=5000
   )
   map_model = VariationalEstimator(config=map_config, dataset=dataset)
   map_model.fit()
   map_model.add_predictions_to_dataset()  # Adds 'MAP' column

   # =====================================================
   # VariationalEstimator: Variational Inference
   # =====================================================
   vi_config = VariationalEstimatorConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_NORMAL,  # VI
       learning_rate=0.01,
       num_steps=5000
   )
   vi_model = VariationalEstimator(config=vi_config, dataset=dataset)
   vi_model.fit()
   vi_model.add_predictions_to_dataset()  # Adds 'VI' and 'VI_uncertainty'

   # =====================================================
   # GaussianMixtureVI: Outlier-Robust VI
   # =====================================================
   gmvi_config = GaussianMixtureVIConfig(
       prior_std=5.0,
       normal_std=1.0,
       outlier_std=3.0,
       outlier_prob=0.2,
       n_epochs=2000
   )
   gmvi_model = GaussianMixtureVI(dataset=dataset, config=gmvi_config)
   gmvi_model.fit()
   gmvi_model.get_results()  # REQUIRED before add_predictions
   gmvi_model.add_predictions_to_dataset()  # Adds 'GMVI' and 'GMVI_uncertainty'

   # =====================================================
   # SpectralCorrection: Weighted Spectral Free-energy Correction
   # =====================================================
   wsfc_config = SpectralCorrectionConfig(use_weights=True)
   wsfc_model = SpectralCorrection(config=wsfc_config, dataset=dataset)
   wsfc_model.fit()
   wsfc_model.add_predictions_to_dataset()  # Adds 'WSFC' and 'WSFC_uncertainty'

   # All predictions are now in the dataset!
   print(dataset.dataset_nodes.columns.tolist())
   print(dataset.estimators)
   # ['MAP', 'VI', 'GMVI', 'WSFC']

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

- For ``GaussianMixtureVI``, you must call ``get_results()`` before ``add_predictions_to_dataset()``
- The model must be trained before calling this method
- Column names are determined automatically by the model type

Probabilistic Models
--------------------

VariationalEstimator
~~~~~~~~~~~~~~~~~~~~

The ``VariationalEstimator`` class provides MAP, VI, and MLE inference for FEP node values
using Pyro's Stochastic Variational Inference framework.

**Mathematical formulation:**

- **Prior**: :math:`p(\mathbf{z}) = \prod_{i=1}^N p(z_i)` where :math:`p(z_i)` is configurable (Normal, Laplace, Student-t, Uniform, Gamma)
- **Likelihood**: :math:`p(y_{ij} | \mathbf{z}) = \mathcal{N}(y_{ij} \mid z_j - z_i,\; \sigma_\text{err}^2)`
- **MAP objective**: :math:`\mathbf{z}^* = \arg\max_\mathbf{z} \left[\log p(\mathbf{y}|\mathbf{z}) + \log p(\mathbf{z})\right]`, equivalent to regularized least squares with :math:`\lambda = \sigma_\text{err}^2 / \sigma_0^2`
- **VI objective**: :math:`\max_\phi\; \mathbb{E}_{q(\mathbf{z};\phi)}[\log p(\mathbf{y}|\mathbf{z})] - D_\text{KL}[q(\mathbf{z};\phi) \| p(\mathbf{z})]`

How VariationalEstimator Determines Column Names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - ``guide_type``
     - ``prior_type``
     - Column Name
   * - ``AUTO_DELTA``
     - ``NORMAL``, ``LAPLACE``, ...
     - ``'MAP'``
   * - ``AUTO_DELTA``
     - ``UNIFORM``
     - ``'MLE'``
   * - ``AUTO_NORMAL``
     - any
     - ``'VI'``

Configuration
^^^^^^^^^^^^^

.. code-block:: python

   from maple.models import VariationalEstimatorConfig, PriorType, GuideType, ErrorDistributionType

   config = VariationalEstimatorConfig(
       # Optimization
       learning_rate=0.01,        # Step size for ClippedAdam optimizer
       num_steps=5000,            # Number of SVI steps

       # Prior distribution
       prior_type=PriorType.NORMAL,      # NORMAL, LAPLACE, UNIFORM, STUDENT_T, GAMMA
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
^^^^^

.. code-block:: python

   from maple.models import VariationalEstimator, VariationalEstimatorConfig, PriorType, GuideType
   from maple.dataset import FEPDataset

   # Create dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")

   # MAP inference
   map_config = VariationalEstimatorConfig(
       prior_type=PriorType.NORMAL,
       guide_type=GuideType.AUTO_DELTA,
       num_steps=5000
   )
   model = VariationalEstimator(config=map_config, dataset=dataset)

   # Fit
   model.fit()

   # Add predictions to dataset
   model.add_predictions_to_dataset()

   # Access results
   map_predictions = dataset.dataset_nodes['MAP']
   print(f"Applied estimators: {dataset.estimators}")

GaussianMixtureVI
~~~~~~~~~~~~~~~~~

The ``GaussianMixtureVI`` (Gaussian Mixture Variational Inference) provides:

- Full-rank covariance for node value posterior via Cholesky decomposition
- Outlier detection via Gaussian mixture likelihood
- Uncertainty quantification for all predictions

**Mathematical formulation:**

- **Prior**: :math:`p(\mathbf{z}) = \mathcal{N}(\mathbf{z} \mid \mathbf{0}, \sigma_0^2 \mathbf{I}_N)`
- **Mixture likelihood**: :math:`p(y_{ij} | \mathbf{z}) = \pi \cdot \mathcal{N}(y_{ij} \mid z_j - z_i, \sigma_2^2) + (1-\pi) \cdot \mathcal{N}(y_{ij} \mid z_j - z_i, \sigma_1^2)`
- **Variational posterior**: :math:`q(\mathbf{z}; \phi) = \mathcal{N}(\mathbf{z} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})` with :math:`\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top`
- **Reparameterization**: :math:`\mathbf{z} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon},\quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})`
- **Outlier probability** (Bayes' rule): :math:`P(\text{outlier} \mid y_{ij}, \mathbf{z}) = \frac{\pi \cdot \mathcal{N}_{\sigma_2}}{\pi \cdot \mathcal{N}_{\sigma_2} + (1-\pi) \cdot \mathcal{N}_{\sigma_1}}`

**Parameters:**

.. list-table::
   :header-rows: 1

   * - Symbol
     - Code Name
     - Description
   * - :math:`\sigma_0`
     - ``prior_std``
     - Prior standard deviation
   * - :math:`\sigma_1`
     - ``normal_std``
     - Likelihood std for **normal** edges
   * - :math:`\sigma_2`
     - ``outlier_std``
     - Likelihood std for **outlier** edges
   * - :math:`\pi`
     - ``outlier_prob``
     - Global outlier probability

Configuration
^^^^^^^^^^^^^

.. code-block:: python

   from maple.models import GaussianMixtureVIConfig

   config = GaussianMixtureVIConfig(
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
       kl_weight=1.0         # Weight for KL divergence term (beta)
   )

Usage
^^^^^

.. code-block:: python

   from maple.models import GaussianMixtureVI, GaussianMixtureVIConfig
   from maple.dataset import FEPDataset

   # Create dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")

   # Configure GMVI
   config = GaussianMixtureVIConfig(
       prior_std=5.0,
       normal_std=1.0,
       outlier_std=3.0,
       outlier_prob=0.2,
       n_epochs=2000
   )

   # Create and fit
   model = GaussianMixtureVI(dataset=dataset, config=config)
   model.fit()

   # IMPORTANT: Get results before adding to dataset
   model.get_results()

   # Add predictions to dataset
   model.add_predictions_to_dataset()

   # Access results
   gmvi_predictions = dataset.dataset_nodes['GMVI']
   gmvi_uncertainty = dataset.dataset_nodes['GMVI_uncertainty']

   # Get outlier probabilities for each edge
   outlier_probs = model.compute_edge_outlier_probabilities()
   print(f"Edges with >70% outlier probability: {sum(p > 0.7 for p in outlier_probs)}")

Deterministic Models
--------------------

CycleClosureCorrection
~~~~~~~~~~~~~~~~~~~~~~

The ``CycleClosureCorrection`` (Weighted Cycle Closure) enforces thermodynamic consistency
by iteratively correcting cycle closure errors.

**Mathematical formulation:**

- **Cycle closure constraint**: :math:`\sum_{(i,j) \in C} y_{ij} = 0` for all cycles :math:`C`
- **Closure error**: :math:`\epsilon_C = \sum_{e \in C} y_e`
- **Weighted correction**: :math:`y_e' = y_e - \epsilon_C \cdot \frac{w_e}{\sum_{e' \in C} w_{e'}}` where :math:`w_e = 1/\sigma_e^2`
- **Node values** (BFS): :math:`z_j = z_i + y_{ij}`
- **Convergence**: max :math:`|\epsilon_C| <` tolerance or max iterations reached

Configuration
^^^^^^^^^^^^^

.. code-block:: python

   from maple.models import CycleClosureCorrectionConfig

   config = CycleClosureCorrectionConfig(
       tolerance=1e-6,       # Convergence tolerance
       max_iterations=1000   # Maximum iterations
   )

Usage
^^^^^

.. code-block:: python

   from maple.models import CycleClosureCorrection, CycleClosureCorrectionConfig
   from maple.dataset import FEPDataset

   # Create dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")

   # Create and fit
   config = CycleClosureCorrectionConfig(tolerance=1e-6)
   model = CycleClosureCorrection(dataset=dataset, config=config)
   model.fit()

   # Add predictions to dataset
   model.add_predictions_to_dataset()

   # Access results
   wcc_predictions = dataset.dataset_nodes['WCC']
   wcc_uncertainty = dataset.dataset_nodes['WCC_uncertainty']

SpectralCorrection
~~~~~~~~~~~~~~~~~~

The ``SpectralCorrection`` (Weighted Spectral Free-energy Correction) solves for optimal
node free energies using the pseudoinverse of the weighted graph Laplacian.

**Mathematical formulation:**

Given the FEP graph with edge observations :math:`\mathbf{x}`:

- **Incidence matrix** :math:`B \in \mathbb{R}^{M \times N}`: :math:`B_{k,i} = -1` (source), :math:`B_{k,j} = +1` (target) for edge :math:`k` from :math:`i \to j`
- **Weight matrix** :math:`W = \operatorname{diag}(1/\sigma_{ij}^2)`: precision-weighted (WSFC) or identity (SFC)
- **Weighted graph Laplacian**: :math:`L_W = B^T W B`
- **Solution**: :math:`\mathbf{z}^* = L_W^+ B^T W \mathbf{x}` via pseudoinverse (handles gauge freedom)
- **Node uncertainties**: :math:`\sigma_{z_i} = \sqrt{(L_W^+)_{ii}}` from diagonal of pseudoinverse

When ``use_weights=False`` (SFC mode), :math:`W = I` and the solution is equivalent to unweighted MLE.

Configuration
^^^^^^^^^^^^^

.. code-block:: python

   from maple.models import SpectralCorrectionConfig

   config = SpectralCorrectionConfig(
       use_weights=True   # True for WSFC (weighted), False for SFC (unweighted)
   )

Usage
^^^^^

.. code-block:: python

   from maple.models import SpectralCorrection, SpectralCorrectionConfig
   from maple.dataset import FEPDataset

   # Create dataset
   dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")

   # Weighted Spectral Free-energy Correction (WSFC)
   wsfc_config = SpectralCorrectionConfig(use_weights=True)
   wsfc_model = SpectralCorrection(config=wsfc_config, dataset=dataset)
   wsfc_model.fit()
   wsfc_model.add_predictions_to_dataset()  # Adds 'WSFC' and 'WSFC_uncertainty'

   # Unweighted Spectral Correction (SFC) - equivalent to MLE
   sfc_config = SpectralCorrectionConfig(use_weights=False)
   sfc_model = SpectralCorrection(config=sfc_config, dataset=dataset)
   sfc_model.fit()
   sfc_model.add_predictions_to_dataset()  # Adds 'SFC' and 'SFC_uncertainty'

   # Access results
   wsfc_predictions = dataset.dataset_nodes['WSFC']
   wsfc_uncertainty = dataset.dataset_nodes['WSFC_uncertainty']

API Reference
-------------

Probabilistic Models
~~~~~~~~~~~~~~~~~~~~

VariationalEstimator
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: maple.models.VariationalEstimator
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

* ``fit()`` - Run inference optimization via Pyro SVI
* ``add_predictions_to_dataset()`` - Write predictions to dataset
* ``get_results()`` - Get dictionary of all results
* ``plot_training_history()`` - Visualize training convergence

**Key Attributes:**

* ``node_estimates`` - Dict mapping node names to estimated values
* ``edge_estimates`` - Dict mapping (src, dst) tuples to estimated values
* ``node_uncertainties`` - Dict of uncertainties (VI only)
* ``loss_history`` - List of loss values during training

GaussianMixtureVI
^^^^^^^^^^^^^^^^^

.. autoclass:: maple.models.GaussianMixtureVI
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

* ``fit()`` - Train the model
* ``get_results()`` - Extract posterior means and stds (REQUIRED before add_predictions)
* ``add_predictions_to_dataset()`` - Write predictions to dataset
* ``compute_edge_outlier_probabilities()`` - Get per-edge outlier probabilities
* ``evaluate_predictions()`` - Get performance metrics

**Key Attributes:**

* ``node_estimates`` - Dict mapping node names to estimated values
* ``edge_estimates`` - Dict mapping (src, dst) tuples to estimated values
* ``node_uncertainties`` - Dict of posterior standard deviations
* ``edge_uncertainties`` - Dict of edge prediction uncertainties

Deterministic Models
~~~~~~~~~~~~~~~~~~~~

CycleClosureCorrection
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: maple.models.CycleClosureCorrection
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

* ``fit()`` - Run iterative cycle closure correction
* ``add_predictions_to_dataset()`` - Write corrected predictions to dataset
* ``get_results()`` - Get dictionary of all results

SpectralCorrection
^^^^^^^^^^^^^^^^^^

.. autoclass:: maple.models.SpectralCorrection
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

* ``fit()`` - Build incidence matrix, solve via graph Laplacian pseudoinverse
* ``add_predictions_to_dataset()`` - Write predictions to dataset (column name: ``'WSFC'`` or ``'SFC'``)
* ``get_results()`` - Get dictionary of all results

**Key Attributes:**

* ``node_estimates`` - Dict mapping node names to estimated values
* ``edge_estimates`` - Dict mapping (src, dst) tuples to estimated edge values
* ``node_uncertainties`` - Dict of uncertainties from :math:`\sqrt{(L_W^+)_{ii}}`
* ``edge_uncertainties`` - Dict of propagated edge uncertainties

Configuration Classes
~~~~~~~~~~~~~~~~~~~~~

VariationalEstimatorConfig
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: maple.models.VariationalEstimatorConfig
   :members:
   :undoc-members:
   :show-inheritance:

GaussianMixtureVIConfig
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: maple.models.GaussianMixtureVIConfig
   :members:
   :undoc-members:
   :show-inheritance:

CycleClosureCorrectionConfig
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: maple.models.CycleClosureCorrectionConfig
   :members:
   :undoc-members:
   :show-inheritance:

SpectralCorrectionConfig
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: maple.models.SpectralCorrectionConfig
   :members:
   :undoc-members:
   :show-inheritance:

BaseEstimatorConfig
^^^^^^^^^^^^^^^^^^^

.. autoclass:: maple.models.BaseEstimatorConfig
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

* ``NORMAL`` - Normal (Gaussian) prior: :math:`\mathcal{N}(\mu, \sigma^2)`
* ``LAPLACE`` - Laplace (double-exponential) prior: :math:`\text{Laplace}(\mu, b)`
* ``STUDENT_T`` - Student-t prior: :math:`t_\nu(0, s)`
* ``UNIFORM`` - Uniform (improper) prior: :math:`\mathcal{U}(a, b)`
* ``GAMMA`` - Gamma prior: :math:`\text{Gamma}(\alpha, \beta)`

GuideType
^^^^^^^^^

.. autoclass:: maple.models.GuideType
   :members:
   :undoc-members:

Available options:

* ``AUTO_DELTA`` - Delta function guide (MAP/MLE inference): :math:`q(\mathbf{z}) = \delta(\mathbf{z} - \mathbf{z}^*)`
* ``AUTO_NORMAL`` - Normal guide (variational inference): :math:`q(\mathbf{z}) = \prod_i \mathcal{N}(z_i \mid \mu_i, \sigma_i^2)`

ErrorDistributionType
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: maple.models.ErrorDistributionType
   :members:
   :undoc-members:

Available options:

* ``NORMAL`` - Normal error distribution
* ``SKEWED_NORMAL`` - Skewed normal via mixture approximation

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
       edge_values: List[float]     # FEP edge values (y_ij)
       num_nodes: int               # N
       num_edges: int               # M = |E|
       node_to_idx: Dict[str, int]  # Node name -> index
       idx_to_node: Dict[int, str]  # Index -> node name
       edge_errors: Optional[List[float]]   # Per-edge uncertainties (sigma_ij)
       edge_weights: Optional[List[float]]  # Per-edge weights (1/sigma_ij^2)

Utility Functions
~~~~~~~~~~~~~~~~~

create_config
^^^^^^^^^^^^^

.. autofunction:: maple.models.create_config

Best Practices
--------------

1. **Always call add_predictions_to_dataset()**: Predictions won't appear in the dataset otherwise

2. **For GaussianMixtureVI, call get_results() first**:

   .. code-block:: python

      gmvi.fit()
      gmvi.get_results()  # REQUIRED
      gmvi.add_predictions_to_dataset()

3. **SpectralCorrection is the fastest method** -- no iterative optimization, just a matrix solve:

   .. code-block:: python

      model = SpectralCorrection(config=SpectralCorrectionConfig(), dataset=dataset)
      model.fit()  # Instant solve
      model.add_predictions_to_dataset()

4. **Check dataset.estimators**: See which models have been applied:

   .. code-block:: python

      if 'GMVI' in dataset.estimators:
          gmvi_preds = dataset.dataset_nodes['GMVI']

5. **Compare models easily**: All predictions are in the same DataFrame:

   .. code-block:: python

      for estimator in dataset.estimators:
          preds = dataset.dataset_nodes[estimator]
          # Compare with experimental values
