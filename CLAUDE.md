# CLAUDE.md - AI Assistant Developer Guide for MAPLE

> **Purpose**: This document provides context and guidelines for AI assistants (like Claude) working on the MAPLE (Maximum A Posteriori Learning of Energies) package. It helps maintain consistency, follow best practices, and understand the project's architecture and conventions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Design Philosophy](#architecture--design-philosophy)
3. [Code Organization](#code-organization)
4. [Development Guidelines](#development-guidelines)
5. [Testing Strategy](#testing-strategy)
6. [Common Patterns](#common-patterns)
7. [Performance Considerations](#performance-considerations)
8. [Documentation Standards](#documentation-standards)
9. [Integration Guidelines](#integration-guidelines)
10. [Troubleshooting & Debugging](#troubleshooting--debugging)

---

## Project Overview

### Purpose
MAPLE (Maximum A Posteriori Learning of Energies) is a Python package for analyzing Free Energy Perturbation (FEP) calculations in computational drug discovery. It provides Bayesian inference methods to correct thermodynamic inconsistencies in FEP graphs where nodes represent absolute free energies (ligand binding affinities) and edges represent relative binding free energy differences (ΔΔG) between ligand pairs.

### Key Features
- **Bayesian inference methods**: MAP (Maximum A Posteriori), VI (Variational Inference), and GMVI (Gaussian Mixture Variational Inference) for node value estimation
- **Cycle closure correction**: Implementation of CCC and weighted cycle closure (WCC) methods
- **Outlier detection**: Probabilistic identification of problematic FEP edges using mixture models
- **Uncertainty quantification**: Full posterior distribution estimation with confidence intervals
- **Graph-based analysis**: Network topology analysis for FEP perturbation maps
- **Benchmark datasets**: Integration with standard FEP benchmark datasets (CMET, Treeline, etc.)

### Target Users
Computational chemists, drug discovery researchers, and cheminformatics practitioners working with FEP calculations for predicting protein-ligand binding affinities.

### Scientific Domain
Computational chemistry, specifically:
- Free Energy Perturbation (FEP) calculations
- Relative Binding Free Energy (RBFE) predictions
- Alchemical free energy methods
- Bayesian inference for molecular simulations

---

## Architecture & Design Philosophy

### Core Principles

1. **Modularity**: Dataset handling, model implementations, and analysis tools are separated into distinct packages
2. **Extensibility**: New inference methods (e.g., new priors, guides) can be added via configuration enums
3. **Reproducibility**: All models use Pyro's probabilistic programming with explicit random seeds
4. **Type Safety**: Pydantic models for configuration validation; type hints throughout
5. **Scientific Rigor**: Bootstrap statistics for uncertainty quantification; proper treatment of gauge freedom

### Mathematical Foundation

The core problem is an inverse problem: given noisy edge measurements (ΔΔG values), infer the true node values (absolute free energies). Key mathematical concepts:

- **Cycle closure constraint**: Free energy around any closed cycle must equal zero (thermodynamic consistency)
- **Gauge freedom**: Node values are only defined up to an additive constant; results are typically mean-centered
- **Heteroscedastic likelihood**: Per-edge uncertainties from BAR/MBAR errors inform the model
- **Mixture model for outliers**: GMVI uses a Gaussian mixture to identify problematic edges

### Design Patterns

#### Configuration Management
- **Current Approach**: Pydantic `BaseModel` subclasses in `src/maple/models/model_config.py`
- **Validation**: Automatic via Pydantic's `Field` validators with range constraints
- **Enums**: `PriorType`, `GuideType`, `ErrorDistributionType` for type-safe configuration options

#### Strategy Pattern
- **Where Used**: Different inference methods (MAP vs VI vs GMVI), different prior distributions
- **How to Add New**: 
  1. Add new enum value to `model_config.py`
  2. Implement corresponding logic in model class
  3. Update tests
- **Base Classes**: `BaseModelConfig`, `BaseDataset`

#### Factory Pattern
- **Where Used**: `create_config()` factory function for model configuration
- **Registration**: Via enum-based selection rather than explicit registry

---

## Code Organization

### Directory Structure

```
MAPLE/
├── src/maple/                  # Main package source code
│   ├── __init__.py
│   ├── dataset/                # Dataset classes and data handling
│   │   ├── __init__.py
│   │   ├── dataset.py          # Base dataset class
│   │   ├── FEP_benchmark_dataset.py  # FEP-specific dataset handling
│   │   └── synthetic_dataset.py      # Synthetic data generation
│   ├── models/                 # Probabilistic models
│   │   ├── __init__.py
│   │   ├── node_model.py       # MAP/VI inference via Pyro
│   │   ├── gaussian_markov_model.py  # GMVI with full-rank covariance
│   │   ├── model_config.py     # Pydantic configuration classes
│   │   └── ccc_model.py        # Cycle closure correction baseline
│   └── graph_analysis/         # Analysis and visualization tools
│       ├── __init__.py
│       ├── performance_stats.py      # Bootstrap statistics, metrics
│       └── plotting_performance.py   # Visualization functions
├── tests/                      # Test suite
│   ├── conftest.py             # Pytest fixtures
│   ├── test_datasets.py
│   ├── test_node_model.py
│   ├── test_integration.py
│   └── test_performance_stats.py
├── docs/                       # Sphinx documentation
│   ├── conf.py
│   ├── index.rst
│   └── examples/
├── examples/                   # Example notebooks and scripts
├── .github/workflows/          # CI/CD configuration
├── pyproject.toml              # Package metadata and dependencies
├── requirements.txt
└── README.md
```

### Module Responsibilities

#### `dataset/`
- **Purpose**: Loading, validating, and transforming FEP data
- **Key Files**:
  - `dataset.py`: `BaseDataset` abstract class defining interface
  - `FEP_benchmark_dataset.py`: `FEPDataset` for benchmark datasets with node/edge DataFrames
  - `synthetic_dataset.py`: `SyntheticFEPDataset` for testing
- **Data Format**: 
  - Edge DataFrame columns: `Source`, `Destination`, `DeltaDeltaG`, `DeltaDeltaG Error`, `CCC`, `Experimental DeltaDeltaG`
  - Node DataFrame columns: `Ligand`, `DeltaG` (experimental), prediction columns (`MAP`, `VI`, `GMVI`)

#### `models/`
- **Purpose**: Probabilistic inference implementations
- **Key Files**:
  - `node_model.py`: `NodeModel` class using Pyro's SVI with AutoDelta/AutoNormal guides
  - `gaussian_markov_model.py`: `GMVI_model` with full-rank Gaussian posterior and mixture likelihood
  - `model_config.py`: `NodeModelConfig`, `GMVIConfig`, enums for configuration
  - `ccc_model.py`: Cycle closure correction baseline implementation
- **Key Parameters**:
  - `prior_std` (σ₀): Prior standard deviation for node values
  - `normal_std` (σ₁): Likelihood std for normal edges
  - `outlier_std` (σ₂): Likelihood std for outlier edges  
  - `outlier_prob` (π): Global mixture weight for outlier component

#### `graph_analysis/`
- **Purpose**: Statistical analysis and visualization
- **Key Files**:
  - `performance_stats.py`: `compute_simple_statistics()`, `compute_bootstrap_statistics()`, correlation metrics
  - `plotting_performance.py`: Scatter plots, correlation plots, edge/node comparison visualizations
- **Metrics**: RMSE, MAE/MUE, R², Pearson r, Spearman ρ, Kendall τ

---

## Development Guidelines

### Adding New Features

#### Adding a New Inference Method

1. **Add Configuration**
   ```python
   # In model_config.py
   class NewMethodConfig(BaseModelConfig):
       model_type: str = Field(default="NewMethod", frozen=True)
       custom_param: float = Field(default=1.0, ge=0.0)
   ```

2. **Implement Model Class**
   ```python
   # In models/new_method.py
   class NewMethod:
       def __init__(self, dataset: BaseDataset, config: NewMethodConfig):
           self.dataset = dataset
           self.config = config
       
       def train(self) -> Dict[str, Any]:
           # Implementation using Pyro or PyTorch
           pass
       
       def get_results(self) -> Dict[str, Any]:
           return {"node_estimates": self.node_estimates, ...}
   ```

3. **Export from `__init__.py`**
   ```python
   # In models/__init__.py
   from .new_method import NewMethod
   __all__ = [..., "NewMethod"]
   ```

4. **Add Tests**
   ```python
   # In tests/test_new_method.py
   def test_new_method_training():
       dataset = SyntheticFEPDataset()
       config = NewMethodConfig(num_steps=100)
       model = NewMethod(dataset, config)
       result = model.train()
       assert "node_estimates" in model.get_results()
   ```

### Code Style

#### General Conventions
- **Python Version**: Python 3.10+
- **Style Guide**: PEP 8, enforced via Black (88 char line length)
- **Imports**: Grouped as stdlib, third-party, local; use `isort`
- **Type Hints**: Required for all public functions and methods

#### Naming Conventions
```python
# Classes: PascalCase
class NodeModel:
    pass

# Functions/methods: snake_case
def compute_bootstrap_statistics():
    pass

# Constants: UPPER_SNAKE_CASE
DEFAULT_NUM_STEPS = 5000

# Private members: _leading_underscore
def _extract_graph_data():
    pass

# Configuration parameters: snake_case
config = NodeModelConfig(learning_rate=0.01, num_steps=1000)
```

#### FEP-Specific Naming
```python
# Use established FEP terminology
delta_delta_g = ...      # ΔΔG: relative binding free energy difference
delta_g = ...            # ΔG: absolute binding free energy
fep_values = ...         # Raw FEP simulation outputs
ccc_values = ...         # Cycle closure corrected values
bar_error = ...          # Bennett Acceptance Ratio error estimate
```

### Error Handling

#### Validation Errors
```python
from pydantic import ValidationError

# Configuration validation happens automatically
try:
    config = NodeModelConfig(learning_rate=-0.01)  # Will fail
except ValidationError as e:
    print(f"Invalid configuration: {e}")
```

#### Scientific Computation Errors
```python
def train(self) -> Dict[str, Any]:
    # Check for numerical issues
    if torch.isnan(loss).any():
        raise RuntimeError(
            f"NaN encountered in loss at step {step}. "
            "Consider reducing learning rate or checking data."
        )
    
    # Validate predictions
    node_values = pyro.get_param_store()["node_values"]
    if not torch.isfinite(node_values).all():
        raise RuntimeError("Non-finite node values in posterior")
```

---

## Testing Strategy

### Test Organization

```
tests/
├── conftest.py                 # Shared fixtures
├── test_datasets.py            # Dataset loading/validation tests
├── test_node_model.py          # NodeModel unit tests
├── test_integration.py         # End-to-end workflow tests
├── test_performance_stats.py   # Statistical function tests
└── test_parameter_sweep.py     # Hyperparameter search tests
```

### Testing Levels

#### Unit Tests
```python
# Test individual model components
def test_node_model_initialization(mock_dataset):
    config = NodeModelConfig(num_steps=10)
    model = NodeModel(config, mock_dataset)
    assert model.config.num_steps == 10
    assert model.dataset == mock_dataset

def test_prior_creation():
    model = NodeModel(config, dataset)
    prior = model._create_prior(num_nodes=4)
    assert isinstance(prior, dist.Normal)
```

#### Integration Tests
```python
def test_dataset_to_model_pipeline(mock_dataset):
    """Test complete pipeline from dataset loading to model training."""
    # Verify data format compatibility
    graph_data = mock_dataset.get_graph_data()
    assert all(key in graph_data for key in ["N", "M", "src", "dst", "FEP"])
    
    # Train model
    config = NodeModelConfig(num_steps=50)
    model = NodeModel(config, mock_dataset)
    model.train()
    
    # Verify results
    results = model.get_results()
    assert len(results["node_estimates"]) == graph_data["N"]
```

#### Property-Based Tests (Recommended for Future)
```python
from hypothesis import given, strategies as st

@given(st.floats(min_value=0.1, max_value=10.0))
def test_prior_std_affects_regularization(prior_std):
    """Higher prior_std should lead to less regularized estimates."""
    config = GMVIConfig(prior_std=prior_std)
    # ... verify relationship between prior_std and estimate variance
```

### Test Fixtures
```python
# In conftest.py
@pytest.fixture
def mock_dataset():
    """Create a minimal mock dataset for testing."""
    edge_data = pd.DataFrame({
        "Source": ["A", "B", "C", "A"],
        "Destination": ["B", "C", "D", "C"],
        "DeltaDeltaG": [1.0, -0.5, 0.3, 1.2],
        "DeltaDeltaG Error": [0.1, 0.1, 0.1, 0.1],
        "CCC": [1.0, -0.5, 0.3, 1.2],
    })
    return MockDataset(edge_data)

@pytest.fixture
def sample_arrays_for_stats():
    """Sample arrays for testing statistical functions."""
    np.random.seed(42)
    y_true = np.random.randn(20)
    y_pred = y_true + np.random.randn(20) * 0.5
    return y_true, y_pred
```

---

## Common Patterns

### Graph Data Processing

The graph data dictionary is the central data structure passed to models:

```python
# Standard graph data format
graph_data = {
    "N": int,           # Number of nodes (ligands)
    "M": int,           # Number of edges (transformations)
    "src": List[int],   # Source node indices (0-based)
    "dst": List[int],   # Destination node indices (0-based)
    "FEP": List[float], # Raw FEP ΔΔG values
    "CCC": List[float], # Cycle closure corrected values (optional)
}

# Conversion in dataset class
def get_graph_data(self) -> Dict[str, Any]:
    return {
        "N": len(self.node_to_idx),
        "M": len(self.edge_df),
        "src": torch.tensor([self.node_to_idx[s] for s in self.edge_df["Source"]]),
        "dst": torch.tensor([self.node_to_idx[d] for d in self.edge_df["Destination"]]),
        "FEP": torch.tensor(self.edge_df["DeltaDeltaG"].values),
    }
```

### Pyro Model Definition Pattern

```python
def _node_model(self, graph_data: GraphData):
    """Pyro probabilistic model for node inference."""
    # Prior over node values (with gauge freedom handling)
    with pyro.plate("nodes", graph_data.num_nodes):
        node_values = pyro.sample(
            "node_values", 
            dist.Normal(0.0, self.config.prior_std)
        )
    
    # Likelihood: observed edges should match node differences
    predicted_edges = node_values[graph_data.target_nodes] - node_values[graph_data.source_nodes]
    residuals = predicted_edges - torch.tensor(graph_data.edge_values)
    
    with pyro.plate("edges", graph_data.num_edges):
        pyro.sample(
            "observations",
            dist.Normal(0.0, self.config.error_std),
            obs=residuals
        )
```

### Bootstrap Statistics Pattern

```python
def compute_bootstrap_statistics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    n_bootstrap: int = 1000
) -> Dict[str, Dict[str, float]]:
    """Compute statistics with bootstrap confidence intervals."""
    
    def compute_metrics(y_t, y_p):
        return {
            "RMSE": np.sqrt(np.mean((y_t - y_p) ** 2)),
            "MAE": np.mean(np.abs(y_t - y_p)),
            "r": np.corrcoef(y_t, y_p)[0, 1],
        }
    
    # Bootstrap resampling
    n_samples = len(y_true)
    bootstrap_results = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        metrics = compute_metrics(y_true[indices], y_pred[indices])
        bootstrap_results.append(metrics)
    
    # Aggregate results with confidence intervals
    results = {}
    for metric in bootstrap_results[0].keys():
        values = [r[metric] for r in bootstrap_results]
        results[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "ci_lower": np.percentile(values, 2.5),
            "ci_upper": np.percentile(values, 97.5),
        }
    
    return results
```

### Mean-Centering for Gauge Freedom

```python
def mean_center_predictions(node_estimates: Dict[str, float]) -> Dict[str, float]:
    """Remove gauge freedom by mean-centering node values."""
    values = np.array(list(node_estimates.values()))
    mean_val = np.mean(values)
    return {k: v - mean_val for k, v in node_estimates.items()}
```

---

## Performance Considerations

### GPU Acceleration

MAPLE uses PyTorch tensors which automatically utilize GPU when available:

```python
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move tensors to device
source_nodes = torch.tensor(graph_data.source_nodes, device=device)
target_nodes = torch.tensor(graph_data.target_nodes, device=device)
```

### Optimization Convergence

```python
# Learning rate scheduling for stable convergence
gamma = 0.1  # Final LR = gamma * initial_lr
lrd = gamma ** (1 / self.config.num_steps)
optimizer = pyro.optim.ClippedAdam({"lr": self.config.learning_rate, "lrd": lrd})

# Early stopping pattern (recommended for GMVI)
patience = 100
best_loss = float('inf')
steps_without_improvement = 0

for step in range(self.config.num_steps):
    loss = svi.step(graph_data)
    
    if loss < best_loss - 1e-4:
        best_loss = loss
        steps_without_improvement = 0
    else:
        steps_without_improvement += 1
    
    if steps_without_improvement >= patience:
        print(f"Early stopping at step {step}")
        break
```

### Memory Management

```python
# Clear Pyro's global parameter store between runs
pyro.clear_param_store()

# For large datasets, process in batches
def process_in_batches(edges, batch_size=1000):
    for i in range(0, len(edges), batch_size):
        batch = edges[i:i + batch_size]
        yield process_batch(batch)
```

---

## Documentation Standards

### Docstring Format (Google Style)

```python
def compute_edge_estimates(
    node_estimates: Dict[str, float],
    edge_df: pd.DataFrame,
) -> Dict[Tuple[str, str], float]:
    """Compute predicted edge values from node estimates.
    
    Given estimated absolute free energies for each node (ligand), 
    compute the implied relative free energy differences for each edge.
    
    Mathematical Details:
        For an edge from node i to node j:
        ΔΔG_ij = ΔG_j - ΔG_i
        
    Args:
        node_estimates: Dictionary mapping node names to estimated ΔG values
        edge_df: DataFrame with 'Source' and 'Destination' columns
        
    Returns:
        Dictionary mapping (source, destination) tuples to predicted ΔΔG values
        
    Raises:
        KeyError: If a node in edge_df is not in node_estimates
        
    Examples:
        >>> node_est = {"A": 0.0, "B": 1.5, "C": -0.5}
        >>> edges = pd.DataFrame({"Source": ["A", "B"], "Destination": ["B", "C"]})
        >>> compute_edge_estimates(node_est, edges)
        {("A", "B"): 1.5, ("B", "C"): -2.0}
        
    Notes:
        - Results are in kcal/mol (same units as input)
        - Sign convention: positive ΔΔG means destination binds more weakly
        
    References:
        - Wang et al. (2015) Accurate and Reliable Prediction of Relative 
          Ligand Binding Potency in Prospective Drug Discovery
    """
```

### Mathematical Notation in Docstrings

Use Unicode for mathematical symbols in docstrings:

```python
"""
Parameters:
    σ₀: Prior standard deviation for node values (kcal/mol)
    σ₁: Likelihood standard deviation for normal edges (kcal/mol)  
    σ₂: Likelihood standard deviation for outlier edges (kcal/mol)
    π: Global outlier probability (0 < π < 1)
    
Mathematical Model:
    Prior: p(zᵢ) = N(zᵢ | 0, σ₀²) for each node i
    
    Likelihood (mixture):
    p(yₑ | z) = π·N(yₑ | zⱼ - zᵢ, σ₂²) + (1-π)·N(yₑ | zⱼ - zᵢ, σ₁²)
    
    where yₑ is the observed ΔΔG for edge e = (i,j)
"""
```

---

## Integration Guidelines

### Adding a New Prior Distribution

1. **Add Enum Value**
   ```python
   # In model_config.py
   class PriorType(str, Enum):
       NORMAL = "normal"
       LAPLACE = "laplace"
       STUDENT_T = "student_t"
       HORSESHOE = "horseshoe"  # New prior
   ```

2. **Implement Prior Creation**
   ```python
   # In node_model.py
   def _create_prior(self, num_nodes: int) -> dist.Distribution:
       if self.config.prior_type == PriorType.NORMAL:
           return dist.Normal(0.0, self.config.prior_std)
       elif self.config.prior_type == PriorType.HORSESHOE:
           # Horseshoe prior for sparse estimation
           tau = pyro.sample("tau", dist.HalfCauchy(1.0))
           lambda_ = pyro.sample("lambda", dist.HalfCauchy(1.0).expand([num_nodes]))
           return dist.Normal(0.0, tau * lambda_)
   ```

3. **Add Tests**
   ```python
   @pytest.mark.parametrize("prior_type", list(PriorType))
   def test_model_with_all_priors(mock_dataset, prior_type):
       config = NodeModelConfig(prior_type=prior_type, num_steps=10)
       model = NodeModel(config, mock_dataset)
       assert model.config.prior_type == prior_type
   ```

### Deprecation Process

```python
import warnings

def old_compute_statistics(y_true, y_pred):
    """Old statistics function (deprecated).
    
    .. deprecated:: 0.2.0
        Use :func:`compute_simple_statistics` instead.
        Will be removed in version 1.0.0.
    """
    warnings.warn(
        "old_compute_statistics is deprecated. Use compute_simple_statistics instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return compute_simple_statistics(y_true, y_pred)
```

---

## Troubleshooting & Debugging

### Common Issues

#### Cycle Closure Violations
**Problem**: Large cycle closure errors in FEP data
**Diagnosis**:
```python
# Check cycle closure for a simple 3-node cycle
cycle_error = ddg_AB + ddg_BC + ddg_CA
if abs(cycle_error) > 0.5:  # kcal/mol threshold
    print(f"Warning: Large cycle error ({cycle_error:.2f} kcal/mol)")
```
**Solution**: Use GMVI with outlier detection, or examine edges in problematic cycles

#### Poor Correlation Despite Good MAE
**Problem**: Low R² even when MAE is acceptable
**Diagnosis**: This often indicates "variance collapse" where predictions cluster tightly but miss the true range
```python
pred_range = np.ptp(predictions)  # Peak-to-peak range
true_range = np.ptp(experimental)
if pred_range < 0.5 * true_range:
    print("Warning: Prediction range much smaller than experimental range")
```
**Solution**: Reduce regularization (increase prior_std), or check for sparse graph topology

#### GMVI Not Converging
**Problem**: Loss oscillating or NaN values
**Solution**:
```python
# Try these adjustments
config = GMVIConfig(
    learning_rate=0.001,     # Reduce from default
    kl_weight=0.1,           # Reduce KL weight
    n_samples=50,            # Increase MC samples
    patience=200,            # Increase patience
)
```

#### Numerical Instability in Covariance
**Problem**: Non-positive-definite covariance matrix in GMVI
**Solution**:
```python
# Add jitter to diagonal
L = torch.linalg.cholesky(covariance + 1e-6 * torch.eye(N))
```

### Debugging Tips

1. **Enable Verbose Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger("maple")
   ```

2. **Visualize Loss Curves**
   ```python
   import matplotlib.pyplot as plt
   plt.plot(losses)
   plt.xlabel("Step")
   plt.ylabel("ELBO Loss")
   plt.title("Training Convergence")
   ```

3. **Check Pyro Parameter Store**
   ```python
   for name, value in pyro.get_param_store().items():
       print(f"{name}: shape={value.shape}, mean={value.mean():.3f}")
   ```

4. **Validate Graph Connectivity**
   ```python
   import networkx as nx
   G = nx.Graph()
   G.add_edges_from(zip(graph_data["src"], graph_data["dst"]))
   if not nx.is_connected(G):
       components = list(nx.connected_components(G))
       print(f"Warning: Graph has {len(components)} disconnected components")
   ```

---

## Important Reminders

### When Implementing New Features

- [ ] Follow Pydantic configuration pattern for any new parameters
- [ ] Add type hints to all public functions
- [ ] Include docstrings with mathematical notation where appropriate
- [ ] Add unit tests (target >80% coverage)
- [ ] Test with both CMET and Treeline benchmark datasets
- [ ] Update `__init__.py` exports
- [ ] Consider bootstrap uncertainty quantification for new metrics
- [ ] If writing scripts to test code then delete this script after their usage

### When Reviewing Code

- [ ] Verify Pydantic validation catches invalid inputs
- [ ] Check numerical stability (NaN handling, log-space operations)
- [ ] Ensure gauge freedom is handled (mean-centering)
- [ ] Verify units are consistent (kcal/mol throughout)
- [ ] Test with edge cases (single edge, disconnected graphs)

### Scientific Integrity

- [ ] Cite original papers (Wang et al. for FEP, Li et al. for WCC, Ding & Drohan for CBayesMBAR)
- [ ] Validate against published benchmark results
- [ ] Report confidence intervals, not just point estimates
- [ ] Document assumptions (e.g., Gaussian errors, independent edges)
- [ ] Distinguish between edge statistics (ΔΔG) and node statistics (ΔG)

---

## Additional Resources

### Internal Documentation
- API Documentation: `docs/api/`
- Example Notebooks: `examples/`
- Benchmark Results: `docs/benchmarks/`

### Key References
- Wang et al. (2015) - Accurate FEP predictions in drug discovery
- Li et al. (2023) - Open source weighted cycle closure method
- Ding & Drohan (2024) - Bayesian approach for FEP graphs with cycles
- Abel et al. - Critical review of FEP accuracy (see `A_Critical_review_2017.pdf`)
- GRAM null model framework for standardized method evaluation

### External Resources
- [Pyro Documentation](https://pyro.ai/)
- [PyTorch Distributions](https://pytorch.org/docs/stable/distributions.html)
- [FEP+ Benchmarking Papers](https://www.schrodinger.com/science-articles/fep)

---

**Last Updated**: 2025
**Package Version**: 0.1.0
**Maintainers**: MAPLE Contributors