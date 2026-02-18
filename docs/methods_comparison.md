# Methods Comparison

> A comprehensive reference for all inference methods available in MAPLE. Use this guide to select the right method for your FEP analysis.

---

## Overview

MAPLE provides **7 inference methods** organized into two categories:

| Category | Methods | When to Use |
|----------|---------|-------------|
| **Probabilistic** | MAP, MLE, VI, GMVI | When you need Bayesian inference, uncertainty quantification, or outlier detection |
| **Deterministic** | WCC, WSFC, SFC | When you need fast, closed-form solutions or thermodynamic consistency correction |

---

## Quick Selection Guide

```
                        Do you need outlier detection?
                              /           \
                           Yes             No
                            |               |
                          GMVI        Do you need uncertainty estimates?
                                          /           \
                                       Yes             No
                                        |               |
                                  Are edge errors    Do you want
                                    available?       regularization?
                                    /       \          /       \
                                 Yes        No      Yes        No
                                  |          |       |          |
                                WSFC        VI      MAP     MLE or SFC
```

---

## Method Details

### MAP — Maximum A Posteriori

| Property | Value |
|----------|-------|
| **Class** | `VariationalEstimator` |
| **Subpackage** | `maple.models.probabilistic` |
| **Column name** | `'MAP'` |
| **Provides uncertainties** | No |
| **Outlier detection** | No |

**Objective function:**

$$\mathbf{z}^{\text{MAP}} = \arg\max_{\mathbf{z}} \left[ \log p(\mathbf{y} \mid \mathbf{z}) + \log p(\mathbf{z}) \right]$$

Equivalent to **regularized least squares**:

$$\mathcal{L}(\mathbf{z}) = \sum_{(i,j) \in E} (z_j - z_i - y_{ij})^2 + \frac{\sigma_\text{err}^2}{\sigma_0^2} \sum_{i=1}^{N} z_i^2$$

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prior_type` | `PriorType.NORMAL` | Prior distribution family |
| `prior_std` | `5.0` | Prior standard deviation $\sigma_0$ |
| `error_std` | `1.0` | Likelihood standard deviation $\sigma_\text{err}$ |
| `guide_type` | `GuideType.AUTO_DELTA` | Must be `AUTO_DELTA` for MAP |
| `learning_rate` | `0.01` | SVI optimizer step size |
| `num_steps` | `5000` | Number of optimization steps |

**When to use:** Small or sparse graphs where regularization prevents extreme node values. Good default choice for quick point estimates.

---

### MLE — Maximum Likelihood Estimation

| Property | Value |
|----------|-------|
| **Class** | `VariationalEstimator` |
| **Subpackage** | `maple.models.probabilistic` |
| **Column name** | `'MLE'` |
| **Provides uncertainties** | No |
| **Outlier detection** | No |

**Objective function:**

$$\mathbf{z}^{\text{MLE}} = \arg\max_{\mathbf{z}} \log p(\mathbf{y} \mid \mathbf{z}) = \arg\min_{\mathbf{z}} \sum_{(i,j) \in E} (z_j - z_i - y_{ij})^2$$

This is MAP with a uniform prior ($\lambda = 0$, no regularization).

**Key parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `prior_type` | `PriorType.UNIFORM` | Must be `UNIFORM` for MLE |
| `prior_parameters` | `[-100.0, 100.0]` | Wide bounds (effectively unbounded) |
| `guide_type` | `GuideType.AUTO_DELTA` | Point estimate |

**When to use:** Large, well-connected graphs with low noise. Provides unbiased estimates without shrinkage toward zero.

---

### VI — Variational Inference

| Property | Value |
|----------|-------|
| **Class** | `VariationalEstimator` |
| **Subpackage** | `maple.models.probabilistic` |
| **Column names** | `'VI'`, `'VI_uncertainty'` |
| **Provides uncertainties** | Yes (diagonal covariance) |
| **Outlier detection** | No |

**Objective function** (ELBO maximization):

$$\mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{q(\mathbf{z};\boldsymbol{\phi})}[\log p(\mathbf{y} \mid \mathbf{z})] - D_{\text{KL}}[q(\mathbf{z};\boldsymbol{\phi}) \| p(\mathbf{z})]$$

**Variational posterior:**

$$q(\mathbf{z}; \boldsymbol{\phi}) = \prod_{i=1}^{N} \mathcal{N}(z_i \mid \mu_i, \sigma_i^2)$$

The ELBO balances three terms:
- **Data fit**: encourages predictions to match observations
- **Prior fit**: encourages node values to stay near prior mean
- **Entropy**: encourages larger uncertainties $\sigma_i$ (prevents overconfidence)

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `guide_type` | `GuideType.AUTO_NORMAL` | Must be `AUTO_NORMAL` for VI |
| `prior_type` | `PriorType.NORMAL` | Prior distribution family |
| `prior_std` | `5.0` | Prior standard deviation $\sigma_0$ |

**When to use:** When you need per-node uncertainty estimates with moderate computational cost.

---

### GMVI — Gaussian Mixture Variational Inference

| Property | Value |
|----------|-------|
| **Class** | `GaussianMixtureVI` |
| **Subpackage** | `maple.models.probabilistic` |
| **Column names** | `'GMVI'`, `'GMVI_uncertainty'` |
| **Provides uncertainties** | Yes (full-rank covariance via Cholesky) |
| **Outlier detection** | Yes (per-edge posterior outlier probabilities) |

**Mixture likelihood:**

$$p(y_{ij} \mid \mathbf{z}) = \pi \cdot \mathcal{N}(y_{ij} \mid z_j - z_i, \sigma_2^2) + (1-\pi) \cdot \mathcal{N}(y_{ij} \mid z_j - z_i, \sigma_1^2)$$

**Full-rank variational posterior:**

$$q(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad \boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$$

**Outlier detection** (Bayes' rule):

$$P(\text{outlier} \mid y_{ij}, \mathbf{z}) = \frac{\pi \cdot \mathcal{N}(y_{ij} \mid z_j - z_i, \sigma_2^2)}{\pi \cdot \mathcal{N}(y_{ij} \mid z_j - z_i, \sigma_2^2) + (1-\pi) \cdot \mathcal{N}(y_{ij} \mid z_j - z_i, \sigma_1^2)}$$

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prior_std` | `5.0` | $\sigma_0$: prior standard deviation |
| `normal_std` | `1.0` | $\sigma_1$: std for normal edges |
| `outlier_std` | `3.0` | $\sigma_2$: std for outlier edges |
| `outlier_prob` | `0.2` | $\pi$: prior outlier probability |
| `n_samples` | `20` | Monte Carlo samples for ELBO |
| `kl_weight` | `1.0` | $\beta$: KL divergence weight |
| `n_epochs` | `2000` | Training epochs |

**When to use:** When you suspect some FEP edges have large errors (outliers) and want to automatically identify them while still getting robust node estimates.

---

### WCC — Weighted Cycle Closure

| Property | Value |
|----------|-------|
| **Class** | `CycleClosureCorrection` |
| **Subpackage** | `maple.models.deterministic` |
| **Column names** | `'WCC'`, `'WCC_uncertainty'` |
| **Provides uncertainties** | Yes (BFS propagation) |
| **Outlier detection** | No |

**Cycle closure constraint:**

$$\sum_{(i,j) \in C} y_{ij} = 0 \quad \forall \text{ cycles } C$$

**Iterative correction:**

$$y_{ij}^{\text{new}} = y_{ij}^{\text{old}} - \epsilon_C \cdot \frac{w_{ij}}{\sum_{(a,b) \in C} w_{ab}}$$

where $\epsilon_C = \sum_{e \in C} y_e$ is the closure error and $w_{ij} = 1/\sigma_{ij}^2$.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tolerance` | `1e-6` | Convergence threshold for max closure error |
| `max_iterations` | `1000` | Maximum correction iterations |

**Reference:** Li, Y. *et al.* An Open Source Graph-Based Weighted Cycle Closure Method for Relative Binding Free Energy Calculations. *J. Chem. Inf. Model.* **2023**, *63*, 561--570. [DOI: 10.1021/acs.jcim.2c01076](https://doi.org/10.1021/acs.jcim.2c01076)

**When to use:** When thermodynamic consistency is the primary concern. Good baseline method that directly enforces cycle closure.

---

### WSFC — Weighted Spectral Free-energy Correction

| Property | Value |
|----------|-------|
| **Class** | `SpectralCorrection` |
| **Subpackage** | `maple.models.deterministic` |
| **Column names** | `'WSFC'`, `'WSFC_uncertainty'` |
| **Provides uncertainties** | Yes (Laplacian pseudoinverse diagonal) |
| **Outlier detection** | No |

**Solution via graph Laplacian pseudoinverse:**

$$\mathbf{z}^* = L_W^+ B^\top W \mathbf{x}$$

where:
- $B \in \mathbb{R}^{M \times N}$ is the signed incidence matrix
- $W = \text{diag}(1/\sigma_{ij}^2)$ is the precision weight matrix
- $L_W = B^\top W B$ is the weighted graph Laplacian
- $L_W^+$ is the Moore-Penrose pseudoinverse

**Node uncertainties:**

$$\sigma_{z_i} = \sqrt{(L_W^+)_{ii}}$$

**Key parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `use_weights` | `True` | Must be `True` for WSFC |

**Reference:** Liu, R. *et al.* State Function-Based Correction: A Simple and Efficient Free-Energy Correction Algorithm for Large-Scale Relative Binding Free-Energy Calculations. *J. Phys. Chem. Lett.* **2025**, *16*, 5763--5768. [DOI: 10.1021/acs.jpclett.5c01119](https://doi.org/10.1021/acs.jpclett.5c01119)

**When to use:** When you have edge uncertainties and want a fast, direct solution with analytic uncertainty estimates. No iterative optimization needed.

---

### SFC — Spectral Free-energy Correction

| Property | Value |
|----------|-------|
| **Class** | `SpectralCorrection` |
| **Subpackage** | `maple.models.deterministic` |
| **Column names** | `'SFC'`, `'SFC_uncertainty'` |
| **Provides uncertainties** | Yes (Laplacian pseudoinverse diagonal) |
| **Outlier detection** | No |

**Solution** (unweighted version of WSFC):

$$\mathbf{z}^* = L^+ B^\top \mathbf{x}, \quad L = B^\top B$$

Equivalent to unweighted MLE / ordinary least squares on the graph.

**Key parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `use_weights` | `False` | Must be `False` for SFC |

**Reference:** Liu, R. *et al.* State Function-Based Correction. *J. Phys. Chem. Lett.* **2025**, *16*, 5763--5768. [DOI: 10.1021/acs.jpclett.5c01119](https://doi.org/10.1021/acs.jpclett.5c01119)

**When to use:** When edge uncertainties are not available or not trusted. Fast baseline equivalent to MLE but with analytic uncertainties.

---

## Comparison Table

| | MAP | MLE | VI | GMVI | WCC | WSFC | SFC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Point estimates** | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **Uncertainties** | - | - | Diagonal | Full-rank | BFS | Analytic | Analytic |
| **Outlier detection** | - | - | - | Yes | - | - | - |
| **Uses edge errors** | - | - | - | - | Yes | Yes | - |
| **Regularization** | Yes ($\lambda > 0$) | - | Yes | Yes ($\beta$) | - | - | - |
| **Iterative** | Yes (SVI) | Yes (SVI) | Yes (SVI) | Yes (Adam) | Yes (cycles) | - | - |
| **Closed-form** | - | - | - | - | - | Yes | Yes |
| **Framework** | Pyro | Pyro | Pyro | PyTorch | NumPy | NumPy | NumPy |

---

## Method Relationships

```
                    ┌──────────────────────────────────────────────┐
                    │          Weighted Least Squares               │
                    │  min Σ w_e (z_j - z_i - y_e)^2              │
                    └──────────┬─────────────┬─────────────────────┘
                               │             │
                    ┌──────────▼──┐    ┌─────▼──────────┐
                    │ + Prior term │    │ Direct solve   │
                    │ (Bayesian)   │    │ (Matrix alg.)  │
                    └──────┬──────┘    └────────┬───────┘
                           │                    │
              ┌────────────┼────────────┐       │
              │            │            │       │
         ┌────▼───┐   ┌───▼────┐  ┌───▼───┐  ┌▼────┐
         │  MAP   │   │  VI    │  │ GMVI  │  │WSFC │
         │(delta) │   │(normal)│  │(mix.) │  │(L+) │
         └────┬───┘   └────────┘  └───────┘  └──┬──┘
              │                                  │
         ┌────▼───┐                         ┌───▼───┐
         │  MLE   │ (λ=0)                   │  SFC  │ (W=I)
         │(no reg)│                         │(unwtd)│
         └────────┘                         └───────┘
                         ┌────────┐
                         │  WCC   │ (iterative cycle correction)
                         │(equiv.)│ (same solution, different algorithm)
                         └────────┘
```

All methods (except GMVI with its mixture likelihood) solve variants of the same weighted least squares problem. They differ in:
1. **Regularization**: MAP/VI add a prior term; MLE/SFC/WSFC/WCC do not
2. **Solution method**: SVI optimization (MAP/MLE/VI/GMVI) vs. direct matrix solve (WSFC/SFC) vs. iterative correction (WCC)
3. **Uncertainty model**: None (MAP/MLE) vs. diagonal posterior (VI) vs. full-rank posterior (GMVI) vs. Laplacian-based (WSFC/SFC) vs. BFS propagation (WCC)

---

## Usage Examples

### Running All Methods

```python
from maple.dataset import FEPDataset
from maple.models import (
    VariationalEstimator, VariationalEstimatorConfig,
    GaussianMixtureVI, GaussianMixtureVIConfig,
    CycleClosureCorrection, CycleClosureCorrectionConfig,
    SpectralCorrection, SpectralCorrectionConfig,
    PriorType, GuideType
)
from maple.graph_analysis import compute_simple_statistics

# Load dataset
dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")

# 1. MAP
map_model = VariationalEstimator(
    config=VariationalEstimatorConfig(guide_type=GuideType.AUTO_DELTA),
    dataset=dataset
)
map_model.fit()
map_model.add_predictions_to_dataset()

# 2. VI
vi_model = VariationalEstimator(
    config=VariationalEstimatorConfig(guide_type=GuideType.AUTO_NORMAL),
    dataset=dataset
)
vi_model.fit()
vi_model.add_predictions_to_dataset()

# 3. GMVI
gmvi_model = GaussianMixtureVI(
    config=GaussianMixtureVIConfig(prior_std=5.0, outlier_prob=0.2),
    dataset=dataset
)
gmvi_model.fit()
gmvi_model.get_results()
gmvi_model.add_predictions_to_dataset()

# 4. WCC
wcc_model = CycleClosureCorrection(
    config=CycleClosureCorrectionConfig(),
    dataset=dataset
)
wcc_model.fit()
wcc_model.add_predictions_to_dataset()

# 5. WSFC
wsfc_model = SpectralCorrection(
    config=SpectralCorrectionConfig(use_weights=True),
    dataset=dataset
)
wsfc_model.fit()
wsfc_model.add_predictions_to_dataset()

# Compare all methods
exp = dataset.dataset_nodes['Exp. DeltaG'].values
print(f"{'Method':<8} {'RMSE':>8} {'R2':>8}")
print("-" * 26)
for est in dataset.estimators:
    pred = dataset.dataset_nodes[est].values
    stats = compute_simple_statistics(exp, pred)
    print(f"{est:<8} {stats['RMSE']:8.3f} {stats['R2']:8.3f}")
```

---

## References

1. Li, Y.; Liu, R.; Liu, J.; Luo, H.; Wu, C.; Li, Z. An Open Source Graph-Based Weighted Cycle Closure Method for Relative Binding Free Energy Calculations. *J. Chem. Inf. Model.* **2023**, *63*, 561--570. [DOI: 10.1021/acs.jcim.2c01076](https://doi.org/10.1021/acs.jcim.2c01076)

2. Liu, R.; Lai, Y.; Yao, Y.; Huang, W.; Zhong, Y.; Luo, H.-B.; Li, Z. State Function-Based Correction: A Simple and Efficient Free-Energy Correction Algorithm for Large-Scale Relative Binding Free-Energy Calculations. *J. Phys. Chem. Lett.* **2025**, *16*, 5763--5768. [DOI: 10.1021/acs.jpclett.5c01119](https://doi.org/10.1021/acs.jpclett.5c01119)

---

*A CSV version of this comparison table is available at `docs/methods_comparison.csv` for programmatic access.*
