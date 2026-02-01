# MAPLE Test Suite

This directory contains the test suite for the MAPLE package.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/maple --cov-report=html

# Test specific components
pytest tests/test_datasets.py -v
pytest tests/test_node_model.py -v
pytest tests/test_performance_stats.py -v
```

## Test Files

- `conftest.py` - Test configuration and shared fixtures
- `test_datasets.py` - Dataset loading and management tests
- `test_graph_analysis.py` - Graph analysis functionality tests
- `test_integration.py` - Integration tests for component interactions
- `test_node_model.py` - Probabilistic node model tests
- `test_parameter_sweep.py` - Parameter sweep operation tests
- `test_performance_stats.py` - Statistical analysis function tests
- `test_performance_tracker.py` - Performance tracking tests
- `test_utils_integration.py` - Utils integration tests

## CI/CD Integration

Tests run automatically on GitHub Actions with Python 3.8-3.11 across multiple workflows:
- Main CI pipeline with full test suite and code quality checks
- Component-specific testing for isolated module validation
- Coverage reporting and package building validation


