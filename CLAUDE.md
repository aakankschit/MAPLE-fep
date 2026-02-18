# CLAUDE.md - MAPLE-FEP

Bayesian inference for Free Energy Perturbation (FEP) calculations. Corrects thermodynamic inconsistencies and detects outliers in FEP perturbation graphs.

## Codebase Map

```
src/maple/
├── __init__.py              # Package root (v0.1.0)
├── models/                  # Estimators for FEP analysis
│   ├── base.py              # BaseEstimator ABC (fit, get_results, add_predictions_to_dataset)
│   ├── config.py            # Pydantic configs + enums + create_config factory
│   │                        # VariationalEstimatorConfig, GaussianMixtureVIConfig,
│   │                        # CycleClosureCorrectionConfig, SpectralCorrectionConfig
│   │                        # Enums: PriorType, GuideType, ErrorDistributionType
│   ├── graph_data.py        # Shared GraphData dataclass
│   ├── probabilistic/
│   │   ├── variational_estimator.py  # VariationalEstimator - MAP/VI via Pyro SVI
│   │   └── gaussian_mixture_vi.py    # GaussianMixtureVI - outlier-robust VI
│   └── deterministic/
│       ├── cycle_closure.py          # CycleClosureCorrection - weighted cycle closure
│       └── spectral_correction.py    # SpectralCorrection - graph Laplacian (SFC/WSFC)
├── dataset/
│   ├── base_dataset.py      # BaseDataset (ABC)
│   ├── dataset.py           # FEPDataset - load from CSV/DataFrames
│   ├── FEP_benchmark_dataset.py  # FEPBenchmarkDataset - public benchmarks
│   └── synthetic_dataset.py # SyntheticFEPDataset - test data generation
├── graph_analysis/
│   ├── performance_stats.py # calculate_rmse, calculate_mae, bootstrap_statistic
│   ├── plotting_performance.py  # plot_dataset_DGs, plot_model_comparison_*
│   ├── graph_setup.py       # GraphSetup - network construction
│   └── graph_cycle_analysis.py  # GraphCycleAnalysis - cycle detection
└── utils/
    ├── parameter_sweep.py   # ParameterSweep, create_*_sweep_experiment
    └── performance_tracker.py  # PerformanceTracker, ModelRun

tests/
├── conftest.py              # Shared fixtures: mock_dataset, sample_*_data
├── test_node_model.py       # VariationalEstimator unit tests
├── test_gmvi_model.py       # GaussianMixtureVI unit tests
├── test_wsfc_model.py       # SpectralCorrection unit tests
├── test_datasets.py         # Dataset loading tests
├── test_integration.py      # End-to-end workflow tests
└── test_performance_stats.py

examples/
├── MAPLE_demo.ipynb         # Main demo notebook
├── benchmark_all_datasets.py
└── parameter_optimization_example.py
```

## Quick Reference

| Task | Location |
|------|----------|
| Add new inference method | `src/maple/models/` + config in `config.py`, inherit `BaseEstimator` |
| Add new prior type | Add to `PriorType` enum, implement in model class |
| Test fixtures | `tests/conftest.py` - MockDataset, sample_*_data |
| Benchmark datasets | `FEPBenchmarkDataset` auto-downloads to `~/.maple_cache` |

## Commands

```bash
# Install (dev)
pip install -e ".[dev]"

# Tests
pytest tests/ -v
pytest tests/ --cov=src/maple --cov-report=html
pytest tests/test_integration.py -v  # Integration only

# Format/Lint
black src/ tests/
ruff check src/ tests/

# Docs
pip install -e ".[docs]"
cd docs && make html
```

## Local Norms

- **Python**: 3.10+ required
- **Style**: Black (88 chars), ruff for linting
- **Type hints**: Required on all public functions
- **Configs**: Use Pydantic `BaseModel` subclasses with `Field` validators
- **Tests**: Pytest; use fixtures from `conftest.py`; call `pyro.clear_param_store()` between model runs
- **Docstrings**: Google style; use Unicode for math (σ₀, ΔΔG)
- **Units**: kcal/mol throughout; mean-center node values to handle gauge freedom
- **Graph data format**: `{"N": int, "M": int, "src": Tensor, "dst": Tensor, "FEP": Tensor}`
- **CI**: GitHub Actions on `main`/`develop`; tests run on Python 3.10-3.12
- **Delete test scripts**: Remove any ad-hoc test scripts after use

## Self-Correction

1. **If the code map is stale**: Update the "Codebase Map" section to reflect current file structure and exports.

2. **If the user provides a correction**: Add it to "Local Norms" (or create a new clearly labeled section) so future sessions inherit it.
