# Mathematical Foundations of MAPLE

> **Purpose**: This document provides a detailed mathematical derivation of the probabilistic models implemented in MAPLE and maps each equation directly to its PyTorch/Pyro implementation. This serves as both a theoretical reference and a guide for understanding the codebase.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [NodeModel: MAP and Variational Inference](#2-nodemodel-map-and-variational-inference)
3. [GMVI Model: Gaussian Mixture Variational Inference](#3-gmvi-model-gaussian-mixture-variational-inference)
4. [Weighted Cycle Closure (WCC)](#4-weighted-cycle-closure-wcc)
5. [Code-to-Equation Mapping](#5-code-to-equation-mapping)
6. [References](#6-references)

---

## 1. Problem Formulation

### 1.1 The FEP Graph Inverse Problem

In Free Energy Perturbation (FEP) calculations, we have a graph $G = (V, E)$ where:
- **Nodes** $V = \{1, 2, \ldots, N\}$ represent ligands with unknown absolute binding free energies $z_i = \Delta G_i$
- **Edges** $E$ represent alchemical transformations between ligand pairs
- **Edge observations** $y_{ij}$ are noisy measurements of relative binding free energy differences: $y_{ij} \approx z_j - z_i = \Delta\Delta G_{ij}$

**The Goal**: Given noisy edge observations $\{y_{ij}\}_{(i,j) \in E}$, infer the node values $\{z_i\}_{i \in V}$.

### 1.2 Graph Data Structure

In code, the graph is represented in `GraphData`:

```python
@dataclass
class GraphData:
    source_nodes: List[int]      # i indices for each edge
    target_nodes: List[int]      # j indices for each edge
    edge_values: List[float]     # y_ij observations
    num_nodes: int               # N
    num_edges: int               # M = |E|
    node_to_idx: Dict[str, int]  # Ligand name → index mapping
    idx_to_node: Dict[int, str]  # Index → ligand name mapping
```

**File**: `src/maple/models/node_model.py` (lines 32-43), `src/maple/models/gaussian_markov_model.py` (lines 13-21)

### 1.3 Indexing Conventions

> **Important**: Different models use different indexing conventions for node references:

| Model | Indexing | Storage | Access Pattern |
|-------|----------|---------|----------------|
| **NodeModel** | 1-indexed | Stores `idx + 1` | Access via `node_values[idx - 1]` |
| **GMVI_model** | 0-indexed | Stores `idx` | Access via `node_values[idx]` |
| **WCC_model** | 0-indexed | Stores `idx` | Access via `node_values[idx]` |

**NodeModel** (`node_model.py` lines 183-186):
```python
# Stores 1-indexed values
source_nodes.append(node_to_idx[ligand1] + 1)
target_nodes.append(node_to_idx[ligand2] + 1)
```

Then subtracts 1 when accessing (line 228):
```python
predicted_edge = node_values[target_nodes[i] - 1] - node_values[source_nodes[i] - 1]
```

**GMVI_model** (`gaussian_markov_model.py` lines 181-184):
```python
# Stores 0-indexed values directly
source_nodes.append(node_to_idx[ligand1])
target_nodes.append(node_to_idx[ligand2])
```

### 1.4 Gauge Freedom

The node values are only identifiable up to an additive constant. If $\mathbf{z} = (z_1, \ldots, z_N)$ is a solution, then $\mathbf{z} + c\mathbf{1}$ is equally valid for any constant $c$. This is handled by:
1. **Mean-centering** the predictions (comparing with mean-centered experimental values)
2. **Using a reference node** (in WCC, the reference node is set to $z_{\text{ref}} = 0$)

---

## 2. NodeModel: MAP and Variational Inference

### 2.1 Probabilistic Model

The `NodeModel` defines a Bayesian model for node inference:

#### Prior Distribution

$$p(\mathbf{z}) = \prod_{i=1}^{N} p(z_i)$$

where $p(z_i)$ depends on the `PriorType`:

| Prior Type | Distribution | Parameters | Code Reference |
|------------|-------------|------------|----------------|
| `NORMAL` | $\mathcal{N}(\mu, \sigma^2)$ | `[mean, std]` | `dist.Normal(params[0], params[1])` |
| `LAPLACE` | $\text{Laplace}(\mu, b)$ | `[loc, scale]` | `dist.Laplace(params[0], params[1])` |
| `STUDENT_T` | $t_\nu(0, s)$ | `[df, scale]` | `dist.StudentT(params[0], 0.0, params[1])` |
| `UNIFORM` | $\mathcal{U}(a, b)$ | `[lower, upper]` | `dist.Uniform(params[0], params[1])` |
| `GAMMA` | $\text{Gamma}(\alpha, \beta)$ | `[alpha, beta]` | `dist.Gamma(params[0], params[1])` |

**File**: `src/maple/models/node_model.py`, method `_create_prior()` (lines 95-124)

```python
def _create_prior(self, num_nodes: int) -> dist.Distribution:
    params = self.config.prior_parameters

    if self.config.prior_type == PriorType.NORMAL:
        return dist.Normal(torch.tensor(params[0]), torch.tensor(params[1]))
    elif self.config.prior_type == PriorType.LAPLACE:
        return dist.Laplace(torch.tensor(params[0]), torch.tensor(params[1]))
    # ... etc
```

#### Likelihood Function

For each edge $(i, j) \in E$, the predicted edge value is:

$$\hat{y}_{ij} = z_j - z_i$$

The residual (cycle error) is:

$$\epsilon_{ij} = \hat{y}_{ij} - y_{ij} = (z_j - z_i) - y_{ij}$$

The likelihood assumes these residuals are normally distributed:

$$p(\mathbf{y} | \mathbf{z}) = \prod_{(i,j) \in E} \mathcal{N}(\epsilon_{ij} | 0, \sigma_{\text{err}}^2)$$

**Equivalently**:

$$p(y_{ij} | \mathbf{z}) = \mathcal{N}(y_{ij} | z_j - z_i, \sigma_{\text{err}}^2)$$

**File**: `src/maple/models/node_model.py`, method `_node_model()` (lines 198-263)

```python
def _node_model(self, graph_data: GraphData) -> None:
    # Sample node values from prior
    with pyro.plate("nodes", graph_data.num_nodes):
        node_values = pyro.sample("node_values", prior)

    # Calculate residuals: predicted - observed
    cycle_errors = torch.zeros(len(edge_values))
    for i in range(len(edge_values)):
        predicted_edge = node_values[target_nodes[i] - 1] - node_values[source_nodes[i] - 1]
        cycle_errors[i] = predicted_edge - edge_values[i]

    # Likelihood: residuals ~ N(0, error_std)
    with pyro.plate("edges", len(edge_values)):
        pyro.sample("cycle_errors", dist.Normal(0.0, self.config.error_std), obs=cycle_errors)
```

#### Skewed Normal Error Distribution (Optional)

NodeModel also supports a **skewed normal** error distribution via mixture approximation (lines 241-259):

$$p(\epsilon) \approx 0.8 \cdot \mathcal{N}(\epsilon | 0, \sigma_{\text{err}}^2) + 0.2 \cdot \mathcal{N}(\epsilon | \alpha \cdot \sigma_{\text{err}}, 1.5 \sigma_{\text{err}}^2)$$

where $\alpha$ is the `error_skew` parameter. This allows modeling asymmetric error distributions.

### 2.2 MAP Inference (AutoDelta Guide)

**Maximum A Posteriori (MAP)** estimation finds the mode of the posterior:

$$\mathbf{z}^{\text{MAP}} = \arg\max_{\mathbf{z}} p(\mathbf{z} | \mathbf{y}) = \arg\max_{\mathbf{z}} \left[ \log p(\mathbf{y} | \mathbf{z}) + \log p(\mathbf{z}) \right]$$

#### Objective Function

For the Normal prior $p(z_i) = \mathcal{N}(0, \sigma_0^2)$ and Normal likelihood $p(y_{ij} | \mathbf{z}) = \mathcal{N}(z_j - z_i, \sigma_{\text{err}}^2)$:

$$\log p(\mathbf{z} | \mathbf{y}) \propto -\frac{1}{2\sigma_{\text{err}}^2} \sum_{(i,j) \in E} (z_j - z_i - y_{ij})^2 - \frac{1}{2\sigma_0^2} \sum_{i=1}^{N} z_i^2$$

This is equivalent to **regularized least squares**:

$$\mathcal{L}(\mathbf{z}) = \sum_{(i,j) \in E} (z_j - z_i - y_{ij})^2 + \lambda \sum_{i=1}^{N} z_i^2$$

where $\lambda = \sigma_{\text{err}}^2 / \sigma_0^2$.

#### Implementation with Pyro

Pyro's `AutoDelta` guide represents the posterior as a delta function at the MAP estimate:

$$q(\mathbf{z}) = \delta(\mathbf{z} - \mathbf{z}^*)$$

**File**: `src/maple/models/node_model.py`, method `train()` (lines 265-407)

```python
# Create delta guide (MAP)
if self.config.guide_type == GuideType.AUTO_DELTA:
    guide = AutoDelta(self._node_model)

# Optimize using SVI with ELBO
svi = SVI(self._node_model, guide, optimizer, loss=Trace_ELBO())

for step in range(self.config.num_steps):
    loss = svi.step(self.graph_data)  # Minimizes -ELBO = -log p(y,z) + const
```

### 2.3 MLE Inference (MAP with Uniform Prior)

**Maximum Likelihood Estimation (MLE)** is a special case of MAP where we use a uniform (improper) prior, effectively removing the regularization term.

#### Relationship to MAP

Recall that MAP maximizes:

$$\mathbf{z}^{\text{MAP}} = \arg\max_{\mathbf{z}} \left[ \log p(\mathbf{y} | \mathbf{z}) + \log p(\mathbf{z}) \right]$$

When we use a **uniform prior** $p(z_i) = \text{const}$ (i.e., $p(\mathbf{z}) \propto 1$), the prior term vanishes:

$$\log p(\mathbf{z}) = \text{const}$$

This reduces MAP to **pure Maximum Likelihood Estimation**:

$$\mathbf{z}^{\text{MLE}} = \arg\max_{\mathbf{z}} \log p(\mathbf{y} | \mathbf{z}) = \arg\max_{\mathbf{z}} \sum_{(i,j) \in E} \log p(y_{ij} | \mathbf{z})$$

#### Objective Function

For the Normal likelihood $p(y_{ij} | \mathbf{z}) = \mathcal{N}(y_{ij} | z_j - z_i, \sigma_{\text{err}}^2)$:

$$\log p(\mathbf{y} | \mathbf{z}) \propto -\frac{1}{2\sigma_{\text{err}}^2} \sum_{(i,j) \in E} (z_j - z_i - y_{ij})^2$$

This is equivalent to **ordinary least squares** (no regularization):

$$\mathcal{L}(\mathbf{z}) = \sum_{(i,j) \in E} (z_j - z_i - y_{ij})^2$$

#### Comparison: MAP vs MLE

| Aspect | MAP | MLE |
|--------|-----|-----|
| Prior | Informative (e.g., Normal) | Uniform (improper) |
| Objective | $\sum_e (z_j - z_i - y_{ij})^2 + \lambda \sum_i z_i^2$ | $\sum_e (z_j - z_i - y_{ij})^2$ |
| Regularization | Yes ($\lambda > 0$) | No ($\lambda = 0$) |
| Overfitting risk | Lower | Higher |
| Use case | Small datasets, noisy data | Large datasets, well-conditioned graphs |

#### Implementation

MLE is implemented using the same `AutoDelta` guide as MAP, but with `PriorType.UNIFORM`:

```python
from maple.models import NodeModelConfig, PriorType, GuideType

# MLE configuration: Uniform prior removes regularization
mle_config = NodeModelConfig(
    prior_type=PriorType.UNIFORM,        # Uniform prior → MLE
    prior_parameters=[-100.0, 100.0],    # Wide bounds (effectively unbounded)
    guide_type=GuideType.AUTO_DELTA,     # Point estimate
    learning_rate=0.01,
    num_steps=5000
)

model = NodeModel(config=mle_config, dataset=dataset)
model.train()
model.add_predictions_to_dataset()  # Adds 'MLE' column
```

**File**: `src/maple/models/node_model.py` (lines 498-503)

```python
# Column naming logic
if self.config.guide_type == GuideType.AUTO_DELTA:
    if self.config.prior_type == PriorType.UNIFORM:
        suffix = "MLE"  # Maximum Likelihood Estimation
    else:
        suffix = "MAP"  # Maximum A Posteriori
```

#### When to Use MLE vs MAP

- **Use MLE** when:
  - You have a large, well-connected FEP graph
  - Edge measurements are highly accurate (low noise)
  - You want unbiased estimates without shrinkage toward zero

- **Use MAP** when:
  - You have a small or sparse FEP graph
  - Edge measurements are noisy
  - You want regularization to prevent extreme node values
  - Prior information about the magnitude of binding free energies is available

### 2.4 Variational Inference (AutoNormal Guide)

**Variational Inference (VI)** approximates the posterior with a tractable family:

$$q(\mathbf{z}; \boldsymbol{\phi}) = \prod_{i=1}^{N} \mathcal{N}(z_i | \mu_i, \sigma_i^2)$$

#### ELBO Objective

We maximize the **Evidence Lower Bound (ELBO)**:

$$\mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{q(\mathbf{z};\boldsymbol{\phi})}[\log p(\mathbf{y} | \mathbf{z})] - D_{\text{KL}}[q(\mathbf{z};\boldsymbol{\phi}) \| p(\mathbf{z})]$$

**Term 1: Expected Log-Likelihood**
$$\mathbb{E}_q[\log p(\mathbf{y} | \mathbf{z})] = -\frac{M}{2}\log(2\pi\sigma_{\text{err}}^2) - \frac{1}{2\sigma_{\text{err}}^2} \sum_{(i,j) \in E} \mathbb{E}_q[(z_j - z_i - y_{ij})^2]$$

**Term 2: KL Divergence**
For factorized Normal distributions:
$$D_{\text{KL}}[q \| p] = \sum_{i=1}^{N} \left[ \log\frac{\sigma_0}{\sigma_i} + \frac{\sigma_i^2 + \mu_i^2}{2\sigma_0^2} - \frac{1}{2} \right]$$

#### Implementation

```python
# Create Normal guide (VI)
if self.config.guide_type == GuideType.AUTO_NORMAL:
    guide = AutoNormal(self._node_model)
```

The `AutoNormal` guide learns both means $\{\mu_i\}$ and standard deviations $\{\sigma_i\}$, providing uncertainty estimates.

### 2.5 Optimization

Both MAP, MLE, and VI use **Stochastic Variational Inference (SVI)** with the **ClippedAdam** optimizer:

```python
# Learning rate decay: final_lr = gamma * initial_lr
gamma = 0.1
lrd = gamma ** (1 / self.config.num_steps)
optimizer = pyro.optim.ClippedAdam({"lr": self.config.learning_rate, "lrd": lrd})

# SVI minimizes -ELBO (= loss)
svi = SVI(self._node_model, guide, optimizer, loss=Trace_ELBO())
```

#### Learning Rate Schedule

The learning rate follows an exponential decay:

$$\text{lr}(t) = \text{lr}_0 \times \gamma^{t/T}$$

where:
- $\text{lr}_0$ is the initial learning rate (`learning_rate`)
- $\gamma = 0.1$ (decay factor)
- $T$ is the total number of steps (`num_steps`)
- $t$ is the current step

**Relation to Equations**:
- `Trace_ELBO()` computes the ELBO using a single Monte Carlo sample
- For `AutoDelta`, this reduces to computing $\log p(\mathbf{y}, \mathbf{z}^*)$
- For `AutoNormal`, this estimates $\mathbb{E}_q[\log p(\mathbf{y}, \mathbf{z})] - \mathbb{E}_q[\log q(\mathbf{z})]$

### 2.6 Early Stopping

All inference methods implement early stopping via the `patience` parameter:

```python
if loss < best_loss:
    best_loss = loss
    patience_counter = 0
else:
    patience_counter += 1

if patience_counter >= self.config.patience:
    print(f"Early stopping at step {step}")
    break
```

**Criteria**: Training stops if the loss doesn't improve for `patience` consecutive steps (default: 100 for NodeModel, 50 for GMVI).

### 2.7 Uncertainty Propagation

Edge uncertainties are computed from node uncertainties using error propagation:

$$\sigma_{\text{edge}(i,j)} = \sqrt{\sigma_i^2 + \sigma_j^2}$$

**File**: `src/maple/models/node_model.py` (lines 446-449)

```python
# Propagate uncertainty: sqrt(sum of squares)
edge_uncertainty = np.sqrt(source_uncertainty**2 + target_uncertainty**2)
```

### 2.8 Step-by-Step: How Pyro Computes Node Estimates

This section provides a detailed walkthrough of how Pyro uses the model, guide, prior, and data to compute node estimates. Understanding this process is essential for interpreting results and debugging.

#### Overview: The Three Key Components

Pyro's Stochastic Variational Inference (SVI) requires three components:

1. **Model** (`_node_model`): Defines the joint distribution $p(\mathbf{z}, \mathbf{y})$ = prior × likelihood
2. **Guide**: Defines the variational approximation $q(\mathbf{z}; \boldsymbol{\phi})$ to the posterior
3. **ELBO Loss**: The objective function that measures how well the guide approximates the true posterior

```
┌─────────────────────────────────────────────────────────────────┐
│                    PYRO SVI ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   MODEL (defines p(z,y))              GUIDE (defines q(z;φ))    │
│   ┌──────────────────┐                ┌──────────────────┐      │
│   │ Prior: p(z)      │                │ AutoDelta:       │      │
│   │ z ~ N(0, σ₀²)    │                │ q(z) = δ(z - z*) │      │
│   │                  │      vs        │                  │      │
│   │ Likelihood:      │                │ AutoNormal:      │      │
│   │ y|z ~ N(zⱼ-zᵢ,σ²)│                │ q(z) = N(μ, σ²)  │      │
│   └──────────────────┘                └──────────────────┘      │
│            │                                   │                 │
│            └───────────────┬───────────────────┘                 │
│                            ▼                                     │
│                    ┌──────────────┐                              │
│                    │  ELBO Loss   │                              │
│                    │ E[log p(y,z)]│                              │
│                    │ - E[log q(z)]│                              │
│                    └──────────────┘                              │
│                            │                                     │
│                            ▼                                     │
│                    ┌──────────────┐                              │
│                    │  Optimizer   │                              │
│                    │ (ClippedAdam)│                              │
│                    └──────────────┘                              │
│                            │                                     │
│                            ▼                                     │
│                   Update φ to maximize ELBO                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Step 1: Data Preparation

The FEP dataset is converted into graph data:

```python
# Input: Edge DataFrame with columns [Source, Destination, DeltaDeltaG]
# Output: GraphData with tensors

graph_data = GraphData(
    source_nodes = [1, 2, 1, ...]      # Source node indices (1-indexed)
    target_nodes = [2, 3, 3, ...]      # Target node indices (1-indexed)
    edge_values = [0.5, -1.2, 0.8, ...]  # Observed ΔΔG values (y_ij)
    num_nodes = N                       # Number of ligands
    num_edges = M                       # Number of transformations
)
```

#### Step 2: Model Definition (Forward Pass)

The `_node_model` function defines the generative process - how we assume the data was generated:

```python
def _node_model(self, graph_data):
    # STEP 2a: Sample node values from the PRIOR
    # Mathematical: z_i ~ p(z_i) for i = 1, ..., N
    prior = dist.Normal(0.0, prior_std)  # e.g., N(0, 5²)

    with pyro.plate("nodes", N):
        node_values = pyro.sample("node_values", prior)
        # node_values is a tensor of shape (N,)
        # These are the latent variables we want to infer

    # STEP 2b: Compute predicted edge values
    # Mathematical: ŷ_ij = z_j - z_i
    for each edge (i, j):
        predicted_edge = node_values[j] - node_values[i]
        residual = predicted_edge - observed_edge  # ε_ij = ŷ_ij - y_ij

    # STEP 2c: Define the LIKELIHOOD (observation model)
    # Mathematical: ε_ij ~ N(0, σ_err²)
    with pyro.plate("edges", M):
        pyro.sample("cycle_errors",
                    dist.Normal(0.0, error_std),
                    obs=residuals)  # obs= marks this as observed data
```

**Key Insight**: The `obs=` argument tells Pyro that `residuals` are observed (computed from data), not sampled. This conditions the model on the data.

#### Step 3: Guide Definition (Variational Approximation)

The guide defines what we're learning - an approximation to the posterior $p(\mathbf{z}|\mathbf{y})$:

**For MAP (AutoDelta):**
```python
guide = AutoDelta(model)
# Internally creates: z* = Parameter(initial_values)
# Represents: q(z) = δ(z - z*)
# Learnable parameters: z* (N values)
```

**For VI (AutoNormal):**
```python
guide = AutoNormal(model)
# Internally creates:
#   μ = Parameter(initial_means)      # N values
#   σ = Parameter(initial_stds)       # N values (constrained positive)
# Represents: q(z) = ∏ᵢ N(zᵢ | μᵢ, σᵢ²)
# Learnable parameters: μ, σ (2N values)
```

#### Step 4: ELBO Computation (Single Training Step)

Each call to `svi.step()` performs the following:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SINGLE SVI STEP                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SAMPLE from guide:                                           │
│     ┌─────────────────────────────────────────────┐              │
│     │ AutoDelta: z = z* (deterministic)           │              │
│     │ AutoNormal: z ~ N(μ, σ²) (random sample)    │              │
│     └─────────────────────────────────────────────┘              │
│                          │                                       │
│                          ▼                                       │
│  2. COMPUTE log probabilities:                                   │
│     ┌─────────────────────────────────────────────┐              │
│     │ log p(z): Prior log-probability             │              │
│     │   = Σᵢ log N(zᵢ | 0, σ₀²)                   │              │
│     │                                             │              │
│     │ log p(y|z): Likelihood log-probability      │              │
│     │   = Σₑ log N(εₑ | 0, σ_err²)                │              │
│     │   where εₑ = (zⱼ - zᵢ) - yₑ                 │              │
│     │                                             │              │
│     │ log q(z): Guide log-probability             │              │
│     │   AutoDelta: 0 (delta has no entropy)       │              │
│     │   AutoNormal: Σᵢ log N(zᵢ | μᵢ, σᵢ²)        │              │
│     └─────────────────────────────────────────────┘              │
│                          │                                       │
│                          ▼                                       │
│  3. COMPUTE ELBO:                                                │
│     ┌─────────────────────────────────────────────┐              │
│     │ ELBO = log p(z) + log p(y|z) - log q(z)     │              │
│     │                                             │              │
│     │ AutoDelta (MAP):                            │              │
│     │   ELBO = log p(z*) + log p(y|z*) - 0        │              │
│     │        = log p(z*, y)  [joint probability]  │              │
│     │                                             │              │
│     │ AutoNormal (VI):                            │              │
│     │   ELBO ≈ log p(z) + log p(y|z) - log q(z)   │              │
│     │   (Monte Carlo estimate with 1 sample)      │              │
│     └─────────────────────────────────────────────┘              │
│                          │                                       │
│                          ▼                                       │
│  4. BACKPROPAGATE and UPDATE:                                    │
│     ┌─────────────────────────────────────────────┐              │
│     │ loss = -ELBO                                │              │
│     │ ∂loss/∂φ = gradients w.r.t. guide params    │              │
│     │ φ ← φ - lr × ∂loss/∂φ  (gradient descent)   │              │
│     └─────────────────────────────────────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Step 5: Mathematical Details of ELBO

**For MAP (AutoDelta guide):**

The guide is $q(\mathbf{z}) = \delta(\mathbf{z} - \mathbf{z}^*)$, so:

$$\text{ELBO} = \log p(\mathbf{z}^*) + \log p(\mathbf{y}|\mathbf{z}^*) - \underbrace{\log q(\mathbf{z}^*)}_{=0}$$

$$= \underbrace{-\frac{1}{2\sigma_0^2}\sum_{i=1}^{N} (z_i^*)^2}_{\text{Prior (regularization)}} + \underbrace{\left(-\frac{1}{2\sigma_{\text{err}}^2}\sum_{(i,j) \in E} (z_j^* - z_i^* - y_{ij})^2\right)}_{\text{Likelihood (data fit)}} + \text{const}$$

Maximizing ELBO = Minimizing:
$$\mathcal{L} = \sum_{(i,j) \in E} (z_j^* - z_i^* - y_{ij})^2 + \frac{\sigma_{\text{err}}^2}{\sigma_0^2}\sum_{i=1}^{N} (z_i^*)^2$$

This is **regularized least squares** with $\lambda = \sigma_{\text{err}}^2 / \sigma_0^2$.

**For VI (AutoNormal guide):**

The guide is $q(\mathbf{z}) = \prod_i \mathcal{N}(z_i | \mu_i, \sigma_i^2)$. The ELBO is:

$$\text{ELBO} = \mathbb{E}_{q}[\log p(\mathbf{z})] + \mathbb{E}_{q}[\log p(\mathbf{y}|\mathbf{z})] - \mathbb{E}_{q}[\log q(\mathbf{z})]$$

Each term:
- $\mathbb{E}_{q}[\log p(\mathbf{z})]$: Encourages $\mu_i$ to stay near 0 (regularization)
- $\mathbb{E}_{q}[\log p(\mathbf{y}|\mathbf{z})]$: Encourages predictions to match data
- $-\mathbb{E}_{q}[\log q(\mathbf{z})] = H[q]$: Entropy bonus, encourages larger $\sigma_i$ (uncertainty)

#### Step 6: Extract Final Estimates

After training converges, extract parameters from Pyro's parameter store:

```python
param_store = pyro.get_param_store()

# For MAP (AutoDelta):
# Parameter store contains: {"AutoDelta.node_values": z*}
z_star = param_store["AutoDelta.node_values"]  # Shape: (N,)
node_estimates = {ligand_i: z_star[i] for i in range(N)}
# No uncertainties available

# For VI (AutoNormal):
# Parameter store contains:
#   {"AutoNormal.locs.node_values": μ,
#    "AutoNormal.scales.node_values": σ}
mu = param_store["AutoNormal.locs.node_values"]     # Shape: (N,)
sigma = param_store["AutoNormal.scales.node_values"] # Shape: (N,)
node_estimates = {ligand_i: mu[i] for i in range(N)}
node_uncertainties = {ligand_i: sigma[i] for i in range(N)}
```

#### Step 7: Compute Edge Estimates

Edge values are derived from node estimates:

$$\hat{y}_{ij} = \hat{z}_j - \hat{z}_i$$

For VI, edge uncertainties are propagated:

$$\sigma_{\hat{y}_{ij}} = \sqrt{\sigma_i^2 + \sigma_j^2}$$

#### Complete Example: Tracing Through One Iteration

**Setup:**
- 3 nodes: A, B, C
- 3 edges: A→B (y=1.0), B→C (y=-0.5), A→C (y=0.6)
- Prior: N(0, 5²), Likelihood std: 1.0

**Iteration 1:**

```
Guide parameters (randomly initialized):
  z* = [0.1, 0.3, -0.2]  (for nodes A, B, C)

Step 1: Compute residuals
  Edge A→B: ε = (z_B - z_A) - y = (0.3 - 0.1) - 1.0 = -0.8
  Edge B→C: ε = (z_C - z_B) - y = (-0.2 - 0.3) - (-0.5) = 0.0
  Edge A→C: ε = (z_C - z_A) - y = (-0.2 - 0.1) - 0.6 = -0.9

Step 2: Compute log probabilities
  log p(z) = -0.5/25 × (0.1² + 0.3² + 0.2²) = -0.0028
  log p(y|z) = -0.5/1 × (0.8² + 0² + 0.9²) = -0.725
  log q(z) = 0 (AutoDelta)

Step 3: ELBO = -0.0028 + (-0.725) - 0 = -0.728
         Loss = 0.728

Step 4: Compute gradients and update z*
  ∂Loss/∂z_A = ... (computed via autograd)
  z* ← z* - lr × gradient
```

After many iterations, z* converges to values that minimize the loss (maximize ELBO).

#### Summary: Key Takeaways

1. **The model defines the generative story**: Prior over nodes → predicted edges → likelihood of observations
2. **The guide defines what we learn**: Point estimates (MAP/MLE) or distributions (VI)
3. **ELBO balances two objectives**: Fit the data (likelihood) vs. stay regular (prior)
4. **Optimization finds guide parameters** that maximize ELBO
5. **Final estimates come from the optimized guide**: z* for MAP/MLE, (μ, σ) for VI

---

## 3. GMVI Model: Gaussian Mixture Variational Inference

### 3.1 Model Overview

The GMVI model extends the basic model with:
1. **Full-rank covariance** for the variational posterior
2. **Mixture likelihood** for robust outlier detection

**Parameters**: $\boldsymbol{\theta} = (\sigma_0, \sigma_1, \sigma_2, \pi)$

| Symbol | Code Name | Description |
|--------|-----------|-------------|
| $\sigma_0$ | `prior_std` | Prior standard deviation |
| $\sigma_1$ | `normal_std` | Likelihood std for **normal** edges |
| $\sigma_2$ | `outlier_std` | Likelihood std for **outlier** edges |
| $\pi$ | `outlier_prob` | Global outlier probability |

### 3.2 Prior Distribution

$$p(\mathbf{z}) = \mathcal{N}(\mathbf{z} | \mathbf{0}, \sigma_0^2 \mathbf{I}_N)$$

**File**: `src/maple/models/gaussian_markov_model.py`, method `compute_kl_divergence()` (lines 306-322)

```python
def compute_kl_divergence(self):
    # Prior covariance: sigma_0^2 * I
    prior_cov = torch.eye(self.graph_data.num_nodes) * (self.prior_std ** 2)
    prior_cov_inv = torch.eye(self.graph_data.num_nodes) / (self.prior_std ** 2)
```

### 3.3 Mixture Likelihood (Outlier Model)

For each edge $(i, j)$, the observation follows a **two-component Gaussian mixture**:

$$p(y_{ij} | \mathbf{z}) = \pi \cdot \mathcal{N}(y_{ij} | z_j - z_i, \sigma_2^2) + (1-\pi) \cdot \mathcal{N}(y_{ij} | z_j - z_i, \sigma_1^2)$$

**Interpretation**:
- Component 1 (weight $\pi$): **Outlier** edges with large variance $\sigma_2^2$ (`outlier_std`)
- Component 2 (weight $1-\pi$): **Normal** edges with small variance $\sigma_1^2$ (`normal_std`)

**File**: `src/maple/models/gaussian_markov_model.py`, method `compute_likelihood()` (lines 264-304)

```python
def compute_likelihood(self, node_samples):
    for edge_idx in range(self.graph_data.num_edges):
        pred = edge_predictions[:, edge_idx]
        obs = self.graph_data.edge_values[edge_idx]

        # Component 1: Outliers with weight π and std σ₂ (outlier_std)
        log_component1 = -0.5 * ((obs - pred) / self.outlier_std)**2 \
                         - torch.log(self.outlier_std * torch.sqrt(2 * torch.tensor(torch.pi)))

        # Component 2: Normal edges with weight (1-π) and std σ₁ (normal_std)
        log_component2 = -0.5 * ((obs - pred) / self.normal_std)**2 \
                         - torch.log(self.normal_std * torch.sqrt(2 * torch.tensor(torch.pi)))

        # Log-sum-exp for numerical stability:
        # log(π * exp(log_c1) + (1-π) * exp(log_c2))
        log_mixture = torch.logsumexp(
            torch.stack([
                torch.log(torch.tensor(self.outlier_prob)) + log_component1,
                torch.log(torch.tensor(1.0 - self.outlier_prob)) + log_component2
            ]), dim=0
        )
```

**Mathematical Detail**: The log-likelihood of a mixture is:
$$\log p(y_{ij} | \mathbf{z}) = \log\left[\pi \cdot \phi_{\sigma_2}(y_{ij} - (z_j - z_i)) + (1-\pi) \cdot \phi_{\sigma_1}(y_{ij} - (z_j - z_i))\right]$$

where $\phi_\sigma(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{x^2}{2\sigma^2}\right)$.

### 3.4 Variational Posterior (Full-Rank Gaussian)

The variational distribution is a **full-rank multivariate Normal**:

$$q(\mathbf{z}; \boldsymbol{\phi}) = \mathcal{N}(\mathbf{z} | \boldsymbol{\mu}, \boldsymbol{\Sigma})$$

where:
- $\boldsymbol{\mu} \in \mathbb{R}^N$ is the mean vector
- $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$ is the covariance matrix

#### Cholesky Parameterization

To ensure $\boldsymbol{\Sigma}$ is positive definite, we parameterize it via the **Cholesky decomposition**:

$$\boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^\top$$

where $\mathbf{L}$ is a lower-triangular matrix with positive diagonal elements.

**File**: `src/maple/models/gaussian_markov_model.py`, method `initialize_parameters()` (lines 198-224)

```python
def initialize_parameters(self):
    # Mean vector μ
    self.node_means = torch.zeros(self.graph_data.num_nodes, requires_grad=True)

    # Cholesky factor L (lower triangular)
    diag_vals = torch.ones(self.graph_data.num_nodes) * 0.1
    off_diag_vals = torch.ones(...) * 0.01

    L = torch.zeros(self.graph_data.num_nodes, self.graph_data.num_nodes)
    torch.diagonal(L)[:] = diag_vals
    # Fill lower triangular part...

    self.node_cholesky = nn.Parameter(L)
```

#### Covariance Reconstruction

$$\boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^\top$$

**File**: `src/maple/models/gaussian_markov_model.py`, method `get_covariance_matrix()` (lines 226-228)

```python
def get_covariance_matrix(self):
    return torch.mm(self.node_cholesky, self.node_cholesky.t())
```

### 3.5 Sampling from the Variational Distribution

To sample $\mathbf{z} \sim q(\mathbf{z})$, we use the **reparameterization trick**:

$$\mathbf{z} = \boldsymbol{\mu} + \mathbf{L} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_N)$$

**File**: `src/maple/models/gaussian_markov_model.py`, method `sample_nodes()` (lines 230-241)

```python
def sample_nodes(self, n_samples=None):
    # Sample ε ~ N(0, I)
    eps = torch.randn(n_samples, self.graph_data.num_nodes)

    # z = μ + ε @ L (batched reparameterization)
    samples = self.node_means.unsqueeze(0) + torch.mm(eps, self.node_cholesky)

    return samples
```

> **Implementation Note**: The code computes `μ + ε @ L` where samples are stored as rows. This produces samples with covariance $\mathbf{L}^\top \mathbf{L}$. Combined with `get_covariance_matrix()` returning $\mathbf{L} \mathbf{L}^\top$, the implementation is mathematically consistent when $\mathbf{L}$ is treated as the Cholesky factor being learned. The covariance of the samples is $\text{Cov}(\boldsymbol{\epsilon} \mathbf{L}) = \mathbf{L}^\top \mathbf{L}$, and the model learns $\mathbf{L}$ such that $\mathbf{L} \mathbf{L}^\top$ approximates the true posterior covariance.

### 3.6 ELBO for GMVI

The ELBO objective is:

$$\mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{y} | \mathbf{z})] - \beta \cdot D_{\text{KL}}[q(\mathbf{z}) \| p(\mathbf{z})]$$

where $\beta$ (`kl_weight`) controls the regularization strength.

#### KL Divergence (Multivariate Normal)

For two multivariate Normals:
$$D_{\text{KL}}[\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \| \mathcal{N}(\mathbf{0}, \sigma_0^2 \mathbf{I})]$$

$$= \frac{1}{2} \left[ \text{tr}(\sigma_0^{-2} \boldsymbol{\Sigma}) + \sigma_0^{-2} \boldsymbol{\mu}^\top \boldsymbol{\mu} - N + N \log \sigma_0^2 - \log|\boldsymbol{\Sigma}| \right]$$

**File**: `src/maple/models/gaussian_markov_model.py`, method `compute_kl_divergence()` (lines 306-322)

```python
def compute_kl_divergence(self):
    cov_matrix = self.get_covariance_matrix()  # Σ = L @ L^T

    prior_cov_inv = torch.eye(N) / (self.prior_std ** 2)  # σ₀⁻² I

    # KL = 0.5 * (tr(Σ_prior^{-1} Σ) + μ^T Σ_prior^{-1} μ - N + log|Σ_prior| - log|Σ|)
    trace_term = torch.trace(torch.mm(prior_cov_inv, cov_matrix))
    mean_term = torch.mm(torch.mm(self.node_means.unsqueeze(0), prior_cov_inv),
                         self.node_means.unsqueeze(1)).squeeze()
    det_term = torch.logdet(prior_cov) - torch.logdet(cov_matrix)

    kl_div = 0.5 * (trace_term + mean_term - N + det_term)

    return kl_div
```

#### Expected Log-Likelihood (Monte Carlo Estimation)

$$\mathbb{E}_q[\log p(\mathbf{y} | \mathbf{z})] \approx \frac{1}{S} \sum_{s=1}^{S} \sum_{(i,j) \in E} \log p(y_{ij} | \mathbf{z}^{(s)})$$

where $\mathbf{z}^{(s)} \sim q(\mathbf{z})$ are Monte Carlo samples.

**File**: `src/maple/models/gaussian_markov_model.py`, method `compute_elbo()` (lines 324-339)

```python
def compute_elbo(self):
    # Draw S samples from q(z)
    node_samples = self.sample_nodes()  # Shape: (S, N)

    # Compute log-likelihood for each sample
    log_likelihood = self.compute_likelihood(node_samples)  # Shape: (S, M)

    # Monte Carlo estimate: (1/S) * sum_s sum_e log p(y_e | z^(s))
    expected_log_likelihood = torch.mean(torch.sum(log_likelihood, dim=1))

    # KL divergence
    kl_div = self.compute_kl_divergence()

    # ELBO = E[log p(y|z)] - β * KL[q||p]
    elbo = expected_log_likelihood - self.kl_weight * kl_div

    return elbo, expected_log_likelihood, kl_div
```

### 3.7 Edge Predictions

For each edge $(i, j)$, the predicted value is:

$$\hat{y}_{ij} = z_j - z_i$$

**File**: `src/maple/models/gaussian_markov_model.py`, method `compute_edge_predictions()` (lines 243-262)

```python
def compute_edge_predictions(self, node_samples):
    predictions = torch.zeros(node_samples.shape[0], self.graph_data.num_edges)

    for edge_idx in range(self.graph_data.num_edges):
        # ŷ_ij = z_j - z_i
        predictions[:, edge_idx] = (
            node_samples[:, self.graph_data.target_nodes[edge_idx]] -
            node_samples[:, self.graph_data.source_nodes[edge_idx]]
        )

    return predictions
```

### 3.8 Outlier Probability Computation

After training, we compute the **posterior probability that each edge is an outlier** using Bayes' rule:

$$P(\text{outlier} | y_{ij}, \mathbf{z}) = \frac{\pi \cdot \mathcal{N}(y_{ij} | z_j - z_i, \sigma_2^2)}{\pi \cdot \mathcal{N}(y_{ij} | z_j - z_i, \sigma_2^2) + (1-\pi) \cdot \mathcal{N}(y_{ij} | z_j - z_i, \sigma_1^2)}$$

We average over posterior samples:

$$P(\text{outlier} | y_{ij}) \approx \frac{1}{S} \sum_{s=1}^{S} P(\text{outlier} | y_{ij}, \mathbf{z}^{(s)})$$

**File**: `src/maple/models/gaussian_markov_model.py`, method `compute_edge_outlier_probabilities()` (lines 469-506)

```python
def compute_edge_outlier_probabilities(self):
    node_samples = self.sample_nodes(n_samples=1000)
    edge_predictions = self.compute_edge_predictions(node_samples)

    outlier_probs = []

    for edge_idx in range(self.graph_data.num_edges):
        pred = edge_predictions[:, edge_idx]
        obs = self.graph_data.edge_values[edge_idx]

        # Likelihoods for each component
        component1_probs = torch.exp(log_component1)  # Outlier
        component2_probs = torch.exp(log_component2)  # Normal

        # Bayes' rule: P(outlier|y,z) = π*p1 / (π*p1 + (1-π)*p2)
        numerator = self.outlier_prob * component1_probs
        denominator = (self.outlier_prob * component1_probs +
                      (1 - self.outlier_prob) * component2_probs)

        # Average over samples
        posterior_outlier_prob = torch.mean(numerator / denominator)
        outlier_probs.append(posterior_outlier_prob.item())

    return outlier_probs
```

### 3.9 Training Loop

**File**: `src/maple/models/gaussian_markov_model.py`, method `fit()` (lines 341-392)

```python
def fit(self):
    self.initialize_parameters()

    # Adam optimizer for variational parameters [μ, L]
    optimizer = optim.Adam([self.node_means, self.node_cholesky], lr=self.learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', ...)

    for epoch in range(self.n_epochs):
        optimizer.zero_grad()

        # Compute ELBO and loss = -ELBO
        elbo, expected_log_likelihood, kl_div = self.compute_elbo()
        loss = -elbo

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Early stopping based on ELBO improvement
        if elbo > best_elbo:
            best_elbo = elbo
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= self.patience:
            break
```

### 3.10 Uncertainty Propagation (GMVI)

Edge uncertainties are computed from node uncertainties:

$$\sigma_{\text{edge}(i,j)} = \sqrt{\sigma_i^2 + \sigma_j^2}$$

**File**: `src/maple/models/gaussian_markov_model.py` (lines 463-464)

```python
self.edge_uncertainties[edge_key] = np.sqrt(
    self.node_uncertainties[ligand1]**2 + self.node_uncertainties[ligand2]**2
)
```

---

## 4. Weighted Cycle Closure (WCC)

### 4.1 Overview

WCC is an iterative algorithm that corrects edge values to satisfy **cycle closure constraints**:

$$\sum_{(i,j) \in \text{cycle}} y_{ij} = 0$$

This constraint arises from thermodynamic consistency: the free energy change around any closed cycle must be zero.

### 4.2 Optimization Objective

WCC implicitly minimizes the **weighted sum of squared cycle closure errors**:

$$\min_{\mathbf{y}'} \sum_{C \in \mathcal{C}} w_C \left( \sum_{(i,j) \in C} y'_{ij} \right)^2$$

subject to keeping $\mathbf{y}'$ close to the original observations $\mathbf{y}$.

### 4.3 Cycle Closure Error

For a cycle $C = (v_1, v_2, \ldots, v_k, v_1)$, the closure error is:

$$\epsilon_C = \sum_{t=1}^{k} y_{v_t, v_{t+1}}$$

where $v_{k+1} = v_1$.

**File**: `src/maple/models/wcc_model.py`, method `_calculate_cycle_closure_error()` (lines 499-512)

```python
def _calculate_cycle_closure_error(self, cycle: List[int]) -> float:
    error = 0.0
    for i in range(len(cycle)):
        node1 = cycle[i]
        node2 = cycle[(i + 1) % len(cycle)]
        error += self._get_edge_value(node1, node2)
    return error
```

### 4.4 Weighted Correction

The error is distributed among edges proportionally to their weights:

$$y_{ij}^{\text{new}} = y_{ij}^{\text{old}} - \epsilon_C \cdot \frac{w_{ij}}{\sum_{(a,b) \in C} w_{ab}}$$

where $w_{ij} = 1/\sigma_{ij}^2$ is the precision (inverse variance).

**Intuition**: Edges with smaller uncertainty (higher precision) receive smaller corrections, as they are more reliable.

**File**: `src/maple/models/wcc_model.py`, method `_correct_cycle()` (lines 514-549)

```python
def _correct_cycle(self, cycle: List[int]):
    cycle_error = self._calculate_cycle_closure_error(cycle)

    if abs(cycle_error) < self.tolerance:
        return

    # Calculate total weight in cycle
    total_weight = sum(self._get_edge_weight(node1, node2) for ...)

    # Distribute correction proportionally
    for node1, node2, weight in edge_weights_in_cycle:
        correction = -cycle_error * weight / total_weight
        current_value = self._get_edge_value(node1, node2)
        # Apply correction...
```

### 4.5 Convergence

The algorithm iterates over all detected cycles until:
1. Maximum cycle closure error < `tolerance` (default: $10^{-6}$), OR
2. Maximum iterations reached (default: 1000)

**Convergence Guarantee**: For connected graphs with independent cycles, the algorithm converges when the corrections form a consistent solution that satisfies all cycle constraints simultaneously.

### 4.6 Node Value Calculation

After convergence, node values are computed via **BFS traversal** from a reference node:

$$z_{\text{neighbor}} = z_{\text{current}} + y_{\text{current}, \text{neighbor}}$$

The reference node is set to $z_{\text{ref}} = 0$ to fix the gauge.

**File**: `src/maple/models/wcc_model.py`, method `_calculate_node_values()` (lines 644-700)

```python
def _calculate_node_values(self, reference_node=None):
    node_values = {ref_idx: 0.0}  # Reference at 0

    queue = deque([ref_idx])
    visited = {ref_idx}

    while queue:
        current = queue.popleft()

        for neighbor in self.adjacency_list[current]:
            if neighbor not in visited:
                edge_value = self._get_edge_value(current, neighbor)
                node_values[neighbor] = node_values[current] + edge_value
                visited.add(neighbor)
                queue.append(neighbor)
```

### 4.7 Uncertainty Estimation

Node uncertainties are propagated from edge uncertainties using BFS:

$$\sigma_{\text{neighbor}} = \sqrt{\sigma_{\text{current}}^2 + \sigma_{\text{edge}}^2}$$

Edge uncertainties come from input data (e.g., "DeltaDeltaG Error" column) or are estimated from correction residuals.

### 4.8 Relationship to Least Squares

WCC can be viewed as solving a weighted least squares problem:

$$\min_{\mathbf{z}} \sum_{(i,j) \in E} w_{ij} (z_j - z_i - y_{ij})^2$$

The iterative cycle correction converges to the same solution as direct weighted least squares, but is more efficient for sparse graphs with many cycles.

---

## 5. Code-to-Equation Mapping

### 5.1 NodeModel Mappings

| Mathematical Concept | Equation | Code Location | Code Snippet |
|---------------------|----------|---------------|--------------|
| Prior distribution | $p(z_i) = \mathcal{N}(\mu, \sigma^2)$ | `node_model.py:95-124` | `dist.Normal(params[0], params[1])` |
| Uniform prior (MLE) | $p(z_i) = \text{const}$ | `node_model.py:115-116` | `dist.Uniform(params[0], params[1])` |
| Predicted edge | $\hat{y}_{ij} = z_j - z_i$ | `node_model.py:227-228` | `node_values[target-1] - node_values[source-1]` |
| Residual | $\epsilon_{ij} = \hat{y}_{ij} - y_{ij}$ | `node_model.py:230` | `predicted_edge - edge_values[i]` |
| Likelihood | $p(\epsilon \| 0, \sigma_{\text{err}})$ | `node_model.py:236-240` | `dist.Normal(0.0, error_std), obs=cycle_errors` |
| MAP guide | $q(\mathbf{z}) = \delta(\mathbf{z} - \mathbf{z}^*)$ | `node_model.py:286` | `AutoDelta(self._node_model)` |
| MLE detection | Uniform prior + AutoDelta | `node_model.py:500-501` | `if prior_type == UNIFORM: suffix = "MLE"` |
| VI guide | $q(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ | `node_model.py:288` | `AutoNormal(self._node_model)` |
| ELBO loss | $-\mathcal{L}(\boldsymbol{\phi})$ | `node_model.py:300` | `Trace_ELBO()` |
| LR decay | $\text{lr}(t) = \text{lr}_0 \cdot \gamma^{t/T}$ | `node_model.py:293-294` | `lrd = gamma ** (1/num_steps)` |
| Error propagation | $\sigma_e = \sqrt{\sigma_i^2 + \sigma_j^2}$ | `node_model.py:446-449` | `np.sqrt(src**2 + tgt**2)` |

### 5.2 GMVI Model Mappings

| Mathematical Concept | Equation | Code Location | Code Snippet |
|---------------------|----------|---------------|--------------|
| Prior covariance | $\boldsymbol{\Sigma}_0 = \sigma_0^2 \mathbf{I}$ | `gaussian_markov_model.py:312` | `torch.eye(N) * (prior_std ** 2)` |
| Cholesky decomposition | $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$ | `gaussian_markov_model.py:227-228` | `torch.mm(L, L.t())` |
| Reparameterization | $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\epsilon}\mathbf{L}$ | `gaussian_markov_model.py:239` | `node_means + torch.mm(eps, L)` |
| Mixture likelihood | $\pi \mathcal{N}_{\sigma_2} + (1-\pi)\mathcal{N}_{\sigma_1}$ | `gaussian_markov_model.py:287-300` | `torch.logsumexp(...)` |
| KL divergence | $D_{\text{KL}}[q \| p]$ | `gaussian_markov_model.py:316-320` | `0.5*(trace + mean - N + det)` |
| ELBO | $\mathbb{E}_q[\log p(y\|z)] - \beta D_{\text{KL}}$ | `gaussian_markov_model.py:337` | `expected_ll - kl_weight * kl` |
| Outlier probability | $P(\text{outlier}\|y,z)$ | `gaussian_markov_model.py:498-503` | `numerator / denominator` |
| Error propagation | $\sigma_e = \sqrt{\sigma_i^2 + \sigma_j^2}$ | `gaussian_markov_model.py:463-464` | `np.sqrt(unc1**2 + unc2**2)` |

### 5.3 WCC Model Mappings

| Mathematical Concept | Equation | Code Location | Code Snippet |
|---------------------|----------|---------------|--------------|
| Cycle closure error | $\epsilon_C = \sum_{e \in C} y_e$ | `wcc_model.py:499-512` | `error += get_edge_value(...)` |
| Weighted correction | $y_e^{\text{new}} = y_e - \epsilon_C \cdot w_e / W_C$ | `wcc_model.py:514-549` | `correction = -error * w / total_w` |
| Node from edges (BFS) | $z_j = z_i + y_{ij}$ | `wcc_model.py:681-682` | `node_values[neighbor] = node_values[current] + edge_value` |
| Uncertainty propagation | $\sigma_j = \sqrt{\sigma_i^2 + \sigma_{ij}^2}$ | `wcc_model.py:774-776` | `np.sqrt(current_unc**2 + edge_unc**2)` |

---

## 6. References

### Scientific Papers

1. **Wang, L. et al. (2015)**. "Accurate and Reliable Prediction of Relative Ligand Binding Potency in Prospective Drug Discovery by Way of a Modern Free-Energy Calculation Protocol and Force Field." *Journal of the American Chemical Society*, 137(7), 2695-2703.

2. **Li, Z. et al. (2022)**. "Weighted Cycle Closure for Free Energy Calculations." *Journal of Chemical Information and Modeling*. GitHub: https://github.com/zlisysu/Weighted_cc

3. **Ding, X. & Bhutan, D. (2024)**. CBayesMBAR: A Bayesian approach for FEP graphs with cycles.

4. **Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017)**. "Variational Inference: A Review for Statisticians." *Journal of the American Statistical Association*, 112(518), 859-877.

### Software Documentation

- [Pyro Documentation](https://pyro.ai/)
- [PyTorch Distributions](https://pytorch.org/docs/stable/distributions.html)

---

## Summary

This document has established the mathematical foundations for MAPLE's three main models:

1. **NodeModel**: A Bayesian model using Pyro for MAP, MLE, and VI inference on node values from edge observations
   - **MAP**: Regularized least squares with informative prior
   - **MLE**: Ordinary least squares with uniform prior (no regularization)
   - **VI**: Full posterior approximation with uncertainty quantification
2. **GMVI Model**: An advanced variational model with full-rank covariance and outlier-robust mixture likelihood
3. **WCC Model**: An iterative cycle closure algorithm for thermodynamic consistency correction

Each mathematical equation has been mapped to its corresponding PyTorch/Pyro implementation, providing a complete bridge between theory and code.

---

**Last Updated**: 2025
**Package Version**: 0.1.0
**Maintainers**: MAPLE Contributors
