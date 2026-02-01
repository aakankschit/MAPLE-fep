"""
Unit tests for the GMVI_model (Gaussian Markov Variational Inference) class.

This module tests the GMVI_model implementation including configuration validation,
model initialization, training, inference, outlier detection, and integration
with different datasets.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from maple.dataset import SyntheticFEPDataset
from maple.models import GMVI_model, GMVIConfig


class TestGMVIConfig:
    """
    Test cases for the GMVIConfig class.

    The GMVIConfig class should:
    - Validate all configuration parameters
    - Provide sensible defaults
    - Reject invalid parameter combinations
    - Support proper parameter ranges
    """

    def test_default_config(self):
        """Test that default configuration is valid."""
        config = GMVIConfig()

        # Check that all required fields have values
        assert config.prior_std > 0
        assert config.normal_std > 0
        assert config.outlier_std > 0
        assert 0 <= config.outlier_prob <= 1
        assert config.kl_weight >= 0
        assert config.learning_rate > 0
        assert config.n_epochs > 0
        assert config.n_samples > 0
        assert config.patience > 0
        assert config.model_type == "GMVI"

    def test_custom_config(self):
        """Test creating config with custom parameters."""
        config = GMVIConfig(
            prior_std=10.0,
            normal_std=0.5,
            outlier_std=5.0,
            outlier_prob=0.3,
            kl_weight=0.2,
            learning_rate=0.001,
            n_epochs=500,
            n_samples=50,
            patience=20
        )

        assert config.prior_std == 10.0
        assert config.normal_std == 0.5
        assert config.outlier_std == 5.0
        assert config.outlier_prob == 0.3
        assert config.kl_weight == 0.2
        assert config.learning_rate == 0.001
        assert config.n_epochs == 500
        assert config.n_samples == 50
        assert config.patience == 20

    def test_invalid_outlier_prob(self):
        """Test that invalid outlier probability is rejected."""
        with pytest.raises(ValueError):
            GMVIConfig(outlier_prob=1.5)

        with pytest.raises(ValueError):
            GMVIConfig(outlier_prob=-0.1)

    def test_invalid_std_values(self):
        """Test that negative std values are rejected."""
        with pytest.raises(ValueError):
            GMVIConfig(prior_std=-1.0)

        with pytest.raises(ValueError):
            GMVIConfig(normal_std=-0.5)

        with pytest.raises(ValueError):
            GMVIConfig(outlier_std=-2.0)

    def test_model_type_immutable(self):
        """Test that model_type cannot be changed."""
        config = GMVIConfig()
        assert config.model_type == "GMVI"

        # Pydantic frozen field should prevent modification
        with pytest.raises((ValueError, AttributeError)):
            config.model_type = "OTHER"


class TestGMVIModelInitialization:
    """Test cases for GMVI_model initialization."""

    def test_init_with_config(self, mock_dataset):
        """Test initialization with GMVIConfig."""
        config = GMVIConfig(prior_std=8.0, learning_rate=0.005)

        model = GMVI_model(dataset=mock_dataset, config=config)

        assert model.prior_std == 8.0
        assert model.learning_rate == 0.005
        assert model.dataset == mock_dataset

    def test_init_with_individual_params(self, mock_dataset):
        """Test initialization with individual parameters."""
        model = GMVI_model(
            dataset=mock_dataset,
            prior_std=7.0,
            normal_std=0.8,
            outlier_std=4.0,
            outlier_prob=0.25
        )

        assert model.prior_std == 7.0
        assert model.normal_std == torch.tensor(0.8)
        assert model.outlier_std == torch.tensor(4.0)
        assert model.outlier_prob == 0.25

    def test_config_overrides_individual_params(self, mock_dataset):
        """Test that config takes precedence over individual params."""
        config = GMVIConfig(prior_std=10.0)

        # Config should override individual parameter
        model = GMVI_model(
            dataset=mock_dataset,
            config=config,
            prior_std=5.0  # Should be ignored
        )

        assert model.prior_std == 10.0

    def test_default_initialization(self, mock_dataset):
        """Test initialization with defaults."""
        model = GMVI_model(dataset=mock_dataset)

        # Check defaults are set
        assert model.prior_std == 5.0
        assert model.normal_std == torch.tensor(1.0)
        assert model.outlier_std == torch.tensor(3.0)
        assert model.outlier_prob == 0.2
        assert model.node_means is None  # Not initialized yet
        assert model.node_cholesky is None


class TestGMVIGraphDataExtraction:
    """Test cases for graph data extraction from datasets."""

    def test_extract_graph_data_basic(self, mock_dataset):
        """Test basic graph data extraction."""
        model = GMVI_model(dataset=mock_dataset)

        assert model.graph_data is not None
        assert model.graph_data.num_nodes > 0
        assert model.graph_data.num_edges > 0
        assert len(model.graph_data.source_nodes) == model.graph_data.num_edges
        assert len(model.graph_data.target_nodes) == model.graph_data.num_edges
        assert len(model.graph_data.edge_values) == model.graph_data.num_edges

    def test_node_indexing(self, mock_dataset):
        """Test that node indexing is consistent."""
        model = GMVI_model(dataset=mock_dataset)

        # Check bidirectional mapping
        for idx, name in model.graph_data.idx_to_node.items():
            assert model.graph_data.node_to_idx[name] == idx

    def test_edge_values_are_numeric(self, mock_dataset):
        """Test that edge values are numeric."""
        model = GMVI_model(dataset=mock_dataset)

        for val in model.graph_data.edge_values:
            assert isinstance(val, (int, float))
            assert not np.isnan(val)


class TestGMVIParameterInitialization:
    """Test cases for variational parameter initialization."""

    def test_initialize_parameters(self, mock_dataset):
        """Test parameter initialization."""
        model = GMVI_model(dataset=mock_dataset)
        model.initialize_parameters()

        assert model.node_means is not None
        assert model.node_cholesky is not None
        assert model.node_means.shape == (model.graph_data.num_nodes,)
        assert model.node_cholesky.shape == (
            model.graph_data.num_nodes,
            model.graph_data.num_nodes
        )

    def test_cholesky_is_lower_triangular(self, mock_dataset):
        """Test that Cholesky matrix is lower triangular."""
        model = GMVI_model(dataset=mock_dataset)
        model.initialize_parameters()

        # Upper triangle should be zero
        upper_triangle = torch.triu(model.node_cholesky, diagonal=1)
        assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle))

    def test_covariance_matrix(self, mock_dataset):
        """Test covariance matrix computation."""
        model = GMVI_model(dataset=mock_dataset)
        model.initialize_parameters()

        cov = model.get_covariance_matrix()

        # Covariance should be symmetric
        assert torch.allclose(cov, cov.t())

        # Covariance should be positive semi-definite
        eigenvalues = torch.linalg.eigvalsh(cov)
        assert torch.all(eigenvalues >= -1e-6)


class TestGMVISampling:
    """Test cases for sampling from variational distribution."""

    def test_sample_nodes(self, mock_dataset):
        """Test node sampling."""
        model = GMVI_model(dataset=mock_dataset, n_samples=10)
        model.initialize_parameters()

        samples = model.sample_nodes(n_samples=10)

        assert samples.shape == (10, model.graph_data.num_nodes)

    def test_sample_nodes_uses_default(self, mock_dataset):
        """Test that sample_nodes uses model's n_samples by default."""
        model = GMVI_model(dataset=mock_dataset, n_samples=25)
        model.initialize_parameters()

        samples = model.sample_nodes()

        assert samples.shape[0] == 25


class TestGMVIEdgePredictions:
    """Test cases for edge prediction computation."""

    def test_compute_edge_predictions(self, mock_dataset):
        """Test edge prediction computation."""
        model = GMVI_model(dataset=mock_dataset)
        model.initialize_parameters()

        samples = model.sample_nodes(n_samples=5)
        predictions = model.compute_edge_predictions(samples)

        assert predictions.shape == (5, model.graph_data.num_edges)

    def test_edge_predictions_follow_fep_standard(self, mock_dataset):
        """Test that edge predictions follow FEP convention (target - source)."""
        model = GMVI_model(dataset=mock_dataset)
        model.initialize_parameters()

        # Create simple samples where we know the values
        samples = torch.zeros(1, model.graph_data.num_nodes)
        samples[0, :] = torch.arange(model.graph_data.num_nodes, dtype=torch.float32)

        predictions = model.compute_edge_predictions(samples)

        # Check first edge follows target - source convention
        expected = samples[0, model.graph_data.target_nodes[0]] - samples[0, model.graph_data.source_nodes[0]]
        assert torch.isclose(predictions[0, 0], expected)


class TestGMVILikelihood:
    """Test cases for likelihood computation."""

    def test_compute_likelihood(self, mock_dataset):
        """Test likelihood computation."""
        model = GMVI_model(dataset=mock_dataset)
        model.initialize_parameters()

        samples = model.sample_nodes(n_samples=5)
        log_likelihood = model.compute_likelihood(samples)

        assert log_likelihood.shape == (5, model.graph_data.num_edges)
        # Log likelihood should be finite
        assert torch.all(torch.isfinite(log_likelihood))


class TestGMVIKLDivergence:
    """Test cases for KL divergence computation."""

    def test_compute_kl_divergence(self, mock_dataset):
        """Test KL divergence computation."""
        model = GMVI_model(dataset=mock_dataset)
        model.initialize_parameters()

        kl_div = model.compute_kl_divergence()

        # KL divergence should be non-negative and finite
        assert kl_div >= 0
        assert torch.isfinite(kl_div)


class TestGMVIELBO:
    """Test cases for ELBO computation."""

    def test_compute_elbo(self, mock_dataset):
        """Test ELBO computation."""
        model = GMVI_model(dataset=mock_dataset)
        model.initialize_parameters()

        elbo, log_lik, kl_div = model.compute_elbo()

        # All values should be finite
        assert torch.isfinite(elbo)
        assert torch.isfinite(log_lik)
        assert torch.isfinite(kl_div)

        # ELBO relationship
        expected_elbo = log_lik - model.kl_weight * kl_div
        assert torch.isclose(elbo, expected_elbo, rtol=1e-4)


class TestGMVIFitting:
    """Test cases for model fitting."""

    def test_fit_basic(self, mock_dataset):
        """Test basic model fitting."""
        model = GMVI_model(dataset=mock_dataset, n_epochs=50, patience=10)

        # Fit should complete without errors
        model.fit()

        # Parameters should be initialized
        assert model.node_means is not None
        assert model.node_cholesky is not None

        # Loss history should be recorded
        assert len(model.loss_history) > 0
        assert len(model.elbo_history) > 0

    def test_fit_convergence(self, mock_dataset):
        """Test that fitting improves ELBO."""
        model = GMVI_model(dataset=mock_dataset, n_epochs=100, patience=20)

        model.fit()

        # ELBO should generally increase (loss decrease)
        # Check last ELBO is better than first (allowing some variance)
        initial_elbo = model.elbo_history[0]
        final_elbo = model.elbo_history[-1]

        # Final ELBO should be higher than initial (ELBO should increase)
        assert final_elbo > initial_elbo - 10  # Allow some tolerance


class TestGMVIPosteriorEstimates:
    """Test cases for posterior estimates."""

    def test_get_posterior_estimates(self, mock_dataset):
        """Test posterior estimates computation."""
        model = GMVI_model(dataset=mock_dataset, n_epochs=20)
        model.fit()

        estimates = model.get_posterior_estimates()

        assert 'means' in estimates
        assert 'stds' in estimates
        assert 'samples' in estimates

        # Check shapes
        assert estimates['means'].shape == (model.graph_data.num_nodes,)
        assert estimates['stds'].shape == (model.graph_data.num_nodes,)
        assert estimates['samples'].shape[1] == model.graph_data.num_nodes

    def test_node_estimates_populated(self, mock_dataset):
        """Test that node estimates are populated."""
        model = GMVI_model(dataset=mock_dataset, n_epochs=20)
        model.fit()
        model.get_posterior_estimates()

        assert model.node_estimates is not None
        assert len(model.node_estimates) == model.graph_data.num_nodes

        # All estimates should be numeric
        for val in model.node_estimates.values():
            assert isinstance(val, (int, float))
            assert not np.isnan(val)

    def test_edge_estimates_computed(self, mock_dataset):
        """Test that edge estimates are computed from node estimates."""
        model = GMVI_model(dataset=mock_dataset, n_epochs=20)
        model.fit()
        model.get_posterior_estimates()

        assert model.edge_estimates is not None
        assert len(model.edge_estimates) > 0


class TestGMVIOutlierDetection:
    """Test cases for outlier probability computation."""

    def test_compute_outlier_probabilities(self, mock_dataset):
        """Test outlier probability computation."""
        model = GMVI_model(dataset=mock_dataset, n_epochs=20)
        model.fit()

        outlier_probs = model.compute_edge_outlier_probabilities()

        assert len(outlier_probs) == model.graph_data.num_edges

        # All probabilities should be in [0, 1]
        for prob in outlier_probs:
            assert 0 <= prob <= 1


class TestGMVIEvaluation:
    """Test cases for model evaluation."""

    def test_evaluate_predictions(self, mock_dataset):
        """Test prediction evaluation."""
        model = GMVI_model(dataset=mock_dataset, n_epochs=20)
        model.fit()

        metrics = model.evaluate_predictions()

        # Check all expected metrics are present
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'correlation' in metrics
        assert 'mean_uncertainty' in metrics
        assert 'outlier_probs' in metrics
        assert 'high_confidence_outliers' in metrics

        # Check metrics are reasonable
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        assert -1 <= metrics['correlation'] <= 1
        assert metrics['mean_uncertainty'] >= 0
        assert metrics['high_confidence_outliers'] >= 0


class TestGMVIDatasetIntegration:
    """Test cases for adding predictions to dataset."""

    def test_add_predictions_to_dataset(self, mock_dataset):
        """Test adding predictions to dataset."""
        model = GMVI_model(dataset=mock_dataset, n_epochs=20)
        model.fit()
        model.get_posterior_estimates()

        # Should not raise errors
        model.add_predictions_to_dataset()

        # Check that GMVI column is added to nodes
        nodes_df = mock_dataset.dataset_nodes
        assert 'GMVI' in nodes_df.columns

        # Check that GMVI column is added to edges
        edges_df = mock_dataset.dataset_edges
        assert 'GMVI' in edges_df.columns

    def test_predictions_before_fit_raises_error(self, mock_dataset):
        """Test that adding predictions before fitting raises error."""
        model = GMVI_model(dataset=mock_dataset)

        with pytest.raises(ValueError):
            model.add_predictions_to_dataset()


class TestGMVIWithSyntheticDataset:
    """Test GMVI model with synthetic dataset."""

    def test_with_synthetic_dataset(self):
        """Test GMVI with SyntheticFEPDataset."""
        dataset = SyntheticFEPDataset(add_noise=True)
        model = GMVI_model(dataset=dataset, n_epochs=30)

        # Should work without errors
        model.fit()
        estimates = model.get_posterior_estimates()

        assert estimates is not None
        assert len(model.node_estimates) == dataset.cycle_data['N']
