# MAPLE-fep

**Maximum A Posteriori Learning of Energies**

[![PyPI version](https://badge.fury.io/py/maple-fep.svg)](https://badge.fury.io/py/maple-fep)
[![CI/CD Pipeline](https://github.com/aakankschit/MAPLE-fep/workflows/CI/badge.svg)](https://github.com/aakankschit/MAPLE-fep/actions)
[![Documentation Status](https://readthedocs.org/projects/maple-fep/badge/?version=latest)](https://maple-fep.readthedocs.io/en/latest/?badge=latest)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for Bayesian analysis of Free Energy Perturbation (FEP) calculations in computational drug discovery. MAPLE provides probabilistic inference methods to correct thermodynamic inconsistencies and detect outliers in FEP perturbation graphs.

## Installation

```bash
# From PyPI
pip install maple-fep

# From source (development)
git clone https://github.com/aakankschit/MAPLE-fep.git
cd MAPLE-fep
pip install -e ".[dev]"
```

## Quick Start

```python
from maple.models import GMVI_model, GMVIConfig
from maple.dataset import FEPDataset

# Load your FEP data
dataset = FEPDataset("fep_edges.csv")

# Configure the GMVI model (Gaussian Mixture Variational Inference)
config = GMVIConfig(
    prior_std=5.0,       # Prior uncertainty on node values
    normal_std=1.0,      # Expected error for normal edges
    outlier_std=3.0,     # Expected error for outlier edges
    outlier_prob=0.2,    # Prior probability of an edge being an outlier
    learning_rate=0.01,
    n_epochs=1000
)

# Train and get results
model = GMVI_model(dataset, config=config)
model.fit()

# Get node estimates (absolute free energies)
node_estimates = model.get_posterior_estimates()

# Identify problematic edges
outlier_probs = model.compute_edge_outlier_probabilities()
```

## Key Features

- **Bayesian Inference Methods**
  - **MAP**: Maximum A Posteriori estimation for quick point estimates
  - **VI**: Variational Inference with uncertainty quantification
  - **GMVI**: Gaussian Mixture VI for automatic outlier detection

- **Cycle Closure Correction**
  - Implementation of CCC and weighted cycle closure (WCC) methods
  - Thermodynamic consistency enforcement

- **Outlier Detection**
  - Probabilistic identification of problematic FEP edges
  - Mixture models for robust estimation

- **Uncertainty Quantification**
  - Full posterior distributions with confidence intervals
  - Bootstrap statistics for performance metrics

## Core Modules

| Module | Description |
|--------|-------------|
| `maple.models` | Probabilistic inference models (NodeModel, GMVI_model, WCC) |
| `maple.dataset` | Dataset loading and FEP benchmark data |
| `maple.graph_analysis` | Graph analysis, statistics, and visualization |
| `maple.utils` | Parameter optimization and performance tracking |

## Documentation

- **[User Guide](https://maple-fep.readthedocs.io/)**: Installation, tutorials, examples
- **[API Reference](https://maple-fep.readthedocs.io/api/)**: Detailed module documentation
- **[Examples](examples/)**: Jupyter notebooks and scripts

Build documentation locally:
```bash
pip install -e ".[docs]"
cd docs && make html
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src/maple --cov-report=html

# Format code
black src/ tests/
ruff check src/ tests/
```

## Citation

If you use MAPLE in your research, please cite:

```bibtex
@software{maple_fep_2025,
  title={MAPLE: Maximum A Posteriori Learning of Energies},
  author={Nandkeolyar, Aakankschit},
  year={2025},
  url={https://github.com/aakankschit/MAPLE-fep},
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [Pyro](https://pyro.ai/) - Probabilistic programming
- [OpenFE](https://openfree.energy/) - Open Free Energy
- [PyMBAR](https://pymbar.readthedocs.io/) - Multistate Bennett Acceptance Ratio

---

**MAPLE** - Advancing computational drug discovery through principled statistical methods 🍁
