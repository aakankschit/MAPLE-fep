"""
Unit tests for the GaussianMixtureVI (Gaussian Markov Variational Inference using Pyro).

This module tests the GaussianMixtureVI implementation including configuration validation,
model initialization, training, inference, outlier detection, and integration
with different datasets.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from maple_fep.dataset import SyntheticFEPDataset
from maple_fep.models import GaussianMixtureVI, GaussianMixtureVIConfig


class TestGaussianMixtureVIConfig:
    """
    Test cases for the GaussianMixtureVIConfig class.

    The GaussianMixtureVIConfig class should:
    - Validate all configuration parameters
    - Provide sensible defaults
    - Reject invalid parameter combinations
    - Support proper parameter ranges
    """

    def test_default_config(self):
        """Test that default configuration is valid."""
        config = GaussianMixtureVIConfig()

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
        assert config.model_type == "GaussianMixtureVI"

    def test_custom_config(self):
        """Test creating config with custom parameters."""
        config = GaussianMixtureVIConfig(
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

    def test_random_seed_config(self):
        """Test that random_seed config field works."""
        config = GaussianMixtureVIConfig(random_seed=42)
        assert config.random_seed == 42

        config_none = GaussianMixtureVIConfig()
        assert config_none.random_seed is None

    def test_invalid_outlier_prob(self):
        """Test that invalid outlier probability is rejected."""
        with pytest.raises(ValueError):
            GaussianMixtureVIConfig(outlier_prob=1.5)

        with pytest.raises(ValueError):
            GaussianMixtureVIConfig(outlier_prob=-0.1)

    def test_invalid_std_values(self):
        """Test that negative std values are rejected."""
        with pytest.raises(ValueError):
            GaussianMixtureVIConfig(prior_std=-1.0)

        with pytest.raises(ValueError):
            GaussianMixtureVIConfig(normal_std=-0.5)

        with pytest.raises(ValueError):
            GaussianMixtureVIConfig(outlier_std=-2.0)

    def test_model_type_immutable(self):
        """Test that model_type cannot be changed."""
        config = GaussianMixtureVIConfig()
        assert config.model_type == "GaussianMixtureVI"

        # Pydantic frozen field should prevent modification
        with pytest.raises((ValueError, AttributeError)):
            config.model_type = "OTHER"


class TestGMVIPyroModelInitialization:
    """Test cases for GaussianMixtureVI initialization."""

    def test_init_with_config(self, mock_dataset):
        """Test initialization with GaussianMixtureVIConfig."""
        config = GaussianMixtureVIConfig(prior_std=8.0, learning_rate=0.005)

        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        assert model.prior_std == 8.0
        assert model.config.learning_rate == 0.005
        assert model.dataset == mock_dataset

    def test_init_stores_config(self, mock_dataset):
        """Test that config is stored correctly."""
        config = GaussianMixtureVIConfig(
            prior_std=7.0,
            normal_std=0.8,
            outlier_std=4.0,
            outlier_prob=0.25
        )
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        assert model.prior_std == 7.0
        assert model.normal_std == torch.tensor(0.8)
        assert model.outlier_std == torch.tensor(4.0)
        assert model.outlier_prob == 0.25

    def test_default_initialization(self, mock_dataset):
        """Test initialization with defaults."""
        config = GaussianMixtureVIConfig()
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        # Check defaults are set
        assert model.prior_std == 5.0
        assert model.normal_std == torch.tensor(1.0)
        assert model.outlier_std == torch.tensor(3.0)
        assert model.outlier_prob == 0.2
        assert model.node_estimates is None
        assert model.edge_estimates is None

    def test_node_estimates_none_before_training(self, mock_dataset):
        """Test that estimates are None before training."""
        config = GaussianMixtureVIConfig()
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        assert model.node_estimates is None
        assert model.edge_estimates is None
        assert model.node_uncertainties is None
        assert model.edge_uncertainties is None


class TestGMVIPyroTraining:
    """Test cases for model training via .fit() API."""

    def test_train_returns_dict(self, mock_dataset):
        """Test that .fit() and .get_results() returns a results dict."""
        config = GaussianMixtureVIConfig(num_steps=50)
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        model.fit()
        result = model.get_results()

        assert isinstance(result, dict)
        assert "node_estimates" in result
        assert "edge_estimates" in result
        assert "node_uncertainties" in result
        assert "edge_uncertainties" in result

    def test_train_populates_node_estimates(self, mock_dataset):
        """Test that training populates node_estimates."""
        config = GaussianMixtureVIConfig(num_steps=50)
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        model.fit()

        assert model.node_estimates is not None
        assert len(model.node_estimates) > 0

        # All estimates should be numeric
        for val in model.node_estimates.values():
            assert isinstance(val, (int, float))
            assert not np.isnan(val)

    def test_train_populates_edge_estimates(self, mock_dataset):
        """Test that training populates edge_estimates."""
        config = GaussianMixtureVIConfig(num_steps=50)
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        model.fit()

        assert model.edge_estimates is not None
        assert len(model.edge_estimates) > 0

    def test_train_populates_uncertainties(self, mock_dataset):
        """Test that training produces uncertainties."""
        config = GaussianMixtureVIConfig(num_steps=50)
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        model.fit()

        assert model.node_uncertainties is not None
        assert model.edge_uncertainties is not None

        # Uncertainties should be positive
        for val in model.node_uncertainties.values():
            assert val > 0

    def test_train_loss_history(self, mock_dataset):
        """Test that training records losses."""
        config = GaussianMixtureVIConfig(num_steps=50)
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        model.fit()
        result = model.get_results()

        assert len(result["losses"]) == 50
        # All losses should be finite
        for loss in result["losses"]:
            assert np.isfinite(loss)

    def test_train_with_random_seed(self, mock_dataset):
        """Test that random seed produces reproducible results."""
        config = GaussianMixtureVIConfig(num_steps=50, random_seed=42)

        model1 = GaussianMixtureVI(config=config, dataset=mock_dataset)
        model1.fit()
        result1 = model1.get_results()

        model2 = GaussianMixtureVI(config=config, dataset=mock_dataset)
        model2.fit()
        result2 = model2.get_results()

        # Results should be identical with same seed
        for key in model1.node_estimates:
            assert abs(model1.node_estimates[key] - model2.node_estimates[key]) < 1e-4


class TestGMVIPyroResults:
    """Test cases for .get_results() API."""

    def test_get_results(self, mock_dataset):
        """Test that get_results returns complete results."""
        config = GaussianMixtureVIConfig(num_steps=50)
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        model.fit()
        results = model.get_results()

        assert "node_estimates" in results
        assert "edge_estimates" in results
        assert "config" in results
        assert "num_nodes" in results
        assert "num_edges" in results
        assert "node_uncertainties" in results
        assert "edge_uncertainties" in results

    def test_get_results_before_training_raises(self, mock_dataset):
        """Test that get_results before training raises error."""
        config = GaussianMixtureVIConfig()
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        with pytest.raises(ValueError):
            model.get_results()


class TestGMVIPyroOutlierDetection:
    """Test cases for outlier probability computation."""

    def test_compute_outlier_probabilities(self, mock_dataset):
        """Test outlier probability computation."""
        config = GaussianMixtureVIConfig(num_steps=50)
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        model.fit()

        outlier_probs = model.compute_edge_outlier_probabilities()

        assert len(outlier_probs) == model.graph_data.num_edges

        # All probabilities should be in [0, 1]
        for prob in outlier_probs:
            assert 0 <= prob <= 1

    def test_outlier_probabilities_before_training_raises(self, mock_dataset):
        """Test that computing outlier probs before training raises error."""
        config = GaussianMixtureVIConfig()
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        with pytest.raises(ValueError):
            model.compute_edge_outlier_probabilities()


class TestGMVIPyroDatasetIntegration:
    """Test cases for adding predictions to dataset."""

    def test_add_predictions_to_dataset(self, mock_dataset):
        """Test adding predictions to dataset."""
        config = GaussianMixtureVIConfig(num_steps=50)
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        model.fit()

        # Should not raise errors
        model.add_predictions_to_dataset()

        # Check that GMVI column is added to nodes
        nodes_df = mock_dataset.dataset_nodes
        assert 'GMVI' in nodes_df.columns

        # Check that GMVI column is added to edges
        edges_df = mock_dataset.dataset_edges
        assert 'GMVI' in edges_df.columns

    def test_add_predictions_includes_uncertainties(self, mock_dataset):
        """Test that uncertainty columns are added."""
        config = GaussianMixtureVIConfig(num_steps=50)
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        model.fit()
        model.add_predictions_to_dataset()

        nodes_df = mock_dataset.dataset_nodes
        assert 'GMVI_uncertainty' in nodes_df.columns

        edges_df = mock_dataset.dataset_edges
        assert 'GMVI_uncertainty' in edges_df.columns

    def test_predictions_before_train_raises_error(self, mock_dataset):
        """Test that adding predictions before training raises error."""
        config = GaussianMixtureVIConfig()
        model = GaussianMixtureVI(config=config, dataset=mock_dataset)

        with pytest.raises(ValueError):
            model.add_predictions_to_dataset()


class TestGMVIPyroWithSyntheticDataset:
    """Test GaussianMixtureVI with synthetic dataset."""

    def test_with_synthetic_dataset(self):
        """Test GaussianMixtureVI with SyntheticFEPDataset."""
        dataset = SyntheticFEPDataset(add_noise=True)
        config = GaussianMixtureVIConfig(num_steps=50)
        model = GaussianMixtureVI(config=config, dataset=dataset)

        # Should work without errors
        model.fit()
        result = model.get_results()

        assert result is not None
        assert len(model.node_estimates) == dataset.cycle_data['N']

    def test_end_to_end_workflow(self):
        """Test complete workflow: init, train, results, predictions."""
        dataset = SyntheticFEPDataset(add_noise=True)
        config = GaussianMixtureVIConfig(num_steps=50, random_seed=123)
        model = GaussianMixtureVI(config=config, dataset=dataset)

        # Train
        model.fit()
        train_result = model.get_results()
        assert "final_loss" in train_result

        # Get results
        results = model.get_results()
        assert "node_estimates" in results
        assert "node_uncertainties" in results

        # Compute outlier probabilities
        outlier_probs = model.compute_edge_outlier_probabilities()
        assert len(outlier_probs) > 0

        # Add to dataset
        model.add_predictions_to_dataset()
        assert "GMVI" in dataset.dataset_nodes.columns
