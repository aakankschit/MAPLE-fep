"""
Unit tests for model configuration classes.

This module tests the configuration validation for both NodeModel and GMVI_model,
including the factory function and enum types.
"""

import pytest
from pydantic import ValidationError

from maple.models.model_config import (
    BaseModelConfig,
    NodeModelConfig,
    GMVIConfig,
    PriorType,
    GuideType,
    ErrorDistributionType,
    create_config
)


class TestEnumTypes:
    """Test cases for configuration enum types."""

    def test_prior_type_enum(self):
        """Test all PriorType enum values."""
        assert PriorType.NORMAL == "normal"
        assert PriorType.GAMMA == "gamma"
        assert PriorType.UNIFORM == "uniform"
        assert PriorType.STUDENT_T == "student_t"
        assert PriorType.LAPLACE == "laplace"

    def test_guide_type_enum(self):
        """Test all GuideType enum values."""
        assert GuideType.AUTO_DELTA == "auto_delta"
        assert GuideType.AUTO_NORMAL == "auto_normal"

    def test_error_distribution_type_enum(self):
        """Test all ErrorDistributionType enum values."""
        assert ErrorDistributionType.NORMAL == "normal"
        assert ErrorDistributionType.SKEWED_NORMAL == "skewed_normal"


class TestNodeModelConfig:
    """Test cases for NodeModelConfig."""

    def test_default_config(self):
        """Test default NodeModelConfig."""
        config = NodeModelConfig()

        assert config.model_type == "NodeModel"
        assert config.prior_type == PriorType.NORMAL
        assert config.prior_parameters == [0.0, 1.0]
        assert config.error_std == 1.0
        assert config.error_distribution == ErrorDistributionType.NORMAL
        assert config.error_skew == 0.0
        assert config.guide_type == GuideType.AUTO_DELTA
        assert config.learning_rate == 0.001
        assert config.num_steps == 5000

    def test_custom_config(self):
        """Test custom NodeModelConfig."""
        config = NodeModelConfig(
            learning_rate=0.01,
            num_steps=1000,
            prior_type=PriorType.GAMMA,
            prior_parameters=[2.0, 3.0],
            error_std=0.5
        )

        assert config.learning_rate == 0.01
        assert config.num_steps == 1000
        assert config.prior_type == PriorType.GAMMA
        assert config.prior_parameters == [2.0, 3.0]
        assert config.error_std == 0.5

    def test_normal_prior_validation(self):
        """Test validation for normal prior parameters."""
        # Valid normal prior
        config = NodeModelConfig(
            prior_type=PriorType.NORMAL,
            prior_parameters=[0.0, 2.0]
        )
        assert config.prior_parameters == [0.0, 2.0]

        # Invalid: wrong number of parameters
        with pytest.raises(ValidationError):
            NodeModelConfig(
                prior_type=PriorType.NORMAL,
                prior_parameters=[0.0]
            )

        # Invalid: negative std
        with pytest.raises(ValidationError):
            NodeModelConfig(
                prior_type=PriorType.NORMAL,
                prior_parameters=[0.0, -1.0]
            )

    def test_gamma_prior_validation(self):
        """Test validation for gamma prior parameters."""
        # Valid gamma prior
        config = NodeModelConfig(
            prior_type=PriorType.GAMMA,
            prior_parameters=[2.0, 3.0]
        )
        assert config.prior_parameters == [2.0, 3.0]

        # Invalid: wrong number of parameters
        with pytest.raises(ValidationError):
            NodeModelConfig(
                prior_type=PriorType.GAMMA,
                prior_parameters=[2.0, 3.0, 4.0]
            )

        # Invalid: negative parameters
        with pytest.raises(ValidationError):
            NodeModelConfig(
                prior_type=PriorType.GAMMA,
                prior_parameters=[-1.0, 3.0]
            )

    def test_uniform_prior_validation(self):
        """Test validation for uniform prior parameters."""
        # Valid uniform prior
        config = NodeModelConfig(
            prior_type=PriorType.UNIFORM,
            prior_parameters=[-5.0, 5.0]
        )
        assert config.prior_parameters == [-5.0, 5.0]

        # Invalid: upper < lower
        with pytest.raises(ValidationError):
            NodeModelConfig(
                prior_type=PriorType.UNIFORM,
                prior_parameters=[5.0, -5.0]
            )

    def test_student_t_prior_validation(self):
        """Test validation for Student-T prior parameters."""
        # Valid Student-T prior
        config = NodeModelConfig(
            prior_type=PriorType.STUDENT_T,
            prior_parameters=[3.0, 1.0]
        )
        assert config.prior_parameters == [3.0, 1.0]

        # Invalid: negative df
        with pytest.raises(ValidationError):
            NodeModelConfig(
                prior_type=PriorType.STUDENT_T,
                prior_parameters=[-1.0, 1.0]
            )

    def test_laplace_prior_validation(self):
        """Test validation for Laplace prior parameters."""
        # Valid Laplace prior
        config = NodeModelConfig(
            prior_type=PriorType.LAPLACE,
            prior_parameters=[0.0, 1.5]
        )
        assert config.prior_parameters == [0.0, 1.5]

        # Invalid: negative scale
        with pytest.raises(ValidationError):
            NodeModelConfig(
                prior_type=PriorType.LAPLACE,
                prior_parameters=[0.0, -1.0]
            )

    def test_learning_rate_validation(self):
        """Test learning rate bounds."""
        # Valid learning rates
        config = NodeModelConfig(learning_rate=0.001)
        assert config.learning_rate == 0.001

        config = NodeModelConfig(learning_rate=1.0)
        assert config.learning_rate == 1.0

        # Invalid: too small
        with pytest.raises(ValidationError):
            NodeModelConfig(learning_rate=1e-7)

        # Invalid: too large
        with pytest.raises(ValidationError):
            NodeModelConfig(learning_rate=2.0)

    def test_num_steps_validation(self):
        """Test num_steps bounds."""
        # Valid
        config = NodeModelConfig(num_steps=1000)
        assert config.num_steps == 1000

        # Invalid: too small
        with pytest.raises(ValidationError):
            NodeModelConfig(num_steps=5)

        # Invalid: too large
        with pytest.raises(ValidationError):
            NodeModelConfig(num_steps=200000)

    def test_error_std_validation(self):
        """Test error_std must be positive."""
        # Valid
        config = NodeModelConfig(error_std=0.5)
        assert config.error_std == 0.5

        # Invalid: negative
        with pytest.raises(ValidationError):
            NodeModelConfig(error_std=-0.1)

        # Invalid: zero
        with pytest.raises(ValidationError):
            NodeModelConfig(error_std=0.0)

    def test_model_type_frozen(self):
        """Test that model_type is frozen."""
        config = NodeModelConfig()
        assert config.model_type == "NodeModel"

        # Cannot change frozen field
        with pytest.raises((ValidationError, AttributeError)):
            config.model_type = "OTHER"


class TestGMVIConfig:
    """Test cases for GMVIConfig."""

    def test_default_config(self):
        """Test default GMVIConfig."""
        config = GMVIConfig()

        assert config.model_type == "GMVI"
        assert config.prior_std == 5.0
        assert config.normal_std == 1.0
        assert config.outlier_std == 3.0
        assert config.outlier_prob == 0.2
        assert config.kl_weight == 0.1
        assert config.learning_rate == 0.01  # GMVI has different default
        assert config.n_epochs == 1000
        assert config.n_samples == 100
        assert config.patience == 50

    def test_custom_config(self):
        """Test custom GMVIConfig."""
        config = GMVIConfig(
            prior_std=10.0,
            normal_std=0.5,
            outlier_std=5.0,
            outlier_prob=0.3,
            kl_weight=0.2,
            learning_rate=0.005,
            n_epochs=500,
            n_samples=50,
            patience=20
        )

        assert config.prior_std == 10.0
        assert config.normal_std == 0.5
        assert config.outlier_std == 5.0
        assert config.outlier_prob == 0.3
        assert config.kl_weight == 0.2
        assert config.learning_rate == 0.005
        assert config.n_epochs == 500
        assert config.n_samples == 50
        assert config.patience == 20

    def test_prior_std_validation(self):
        """Test prior_std must be positive."""
        # Valid
        config = GMVIConfig(prior_std=10.0)
        assert config.prior_std == 10.0

        # Invalid: negative
        with pytest.raises(ValidationError):
            GMVIConfig(prior_std=-1.0)

        # Invalid: zero
        with pytest.raises(ValidationError):
            GMVIConfig(prior_std=0.0)

    def test_normal_std_validation(self):
        """Test normal_std must be positive."""
        # Valid
        config = GMVIConfig(normal_std=0.8)
        assert config.normal_std == 0.8

        # Invalid: negative
        with pytest.raises(ValidationError):
            GMVIConfig(normal_std=-0.5)

    def test_outlier_std_validation(self):
        """Test outlier_std must be positive."""
        # Valid
        config = GMVIConfig(outlier_std=4.0)
        assert config.outlier_std == 4.0

        # Invalid: negative
        with pytest.raises(ValidationError):
            GMVIConfig(outlier_std=-2.0)

    def test_outlier_prob_validation(self):
        """Test outlier_prob must be in [0, 1]."""
        # Valid boundary cases
        config = GMVIConfig(outlier_prob=0.0)
        assert config.outlier_prob == 0.0

        config = GMVIConfig(outlier_prob=1.0)
        assert config.outlier_prob == 1.0

        config = GMVIConfig(outlier_prob=0.5)
        assert config.outlier_prob == 0.5

        # Invalid: too small
        with pytest.raises(ValidationError):
            GMVIConfig(outlier_prob=-0.1)

        # Invalid: too large
        with pytest.raises(ValidationError):
            GMVIConfig(outlier_prob=1.5)

    def test_kl_weight_validation(self):
        """Test kl_weight must be non-negative."""
        # Valid
        config = GMVIConfig(kl_weight=0.0)
        assert config.kl_weight == 0.0

        config = GMVIConfig(kl_weight=0.5)
        assert config.kl_weight == 0.5

        # Invalid: negative
        with pytest.raises(ValidationError):
            GMVIConfig(kl_weight=-0.1)

    def test_n_epochs_validation(self):
        """Test n_epochs must be at least 10."""
        # Valid
        config = GMVIConfig(n_epochs=100)
        assert config.n_epochs == 100

        # Invalid: too small
        with pytest.raises(ValidationError):
            GMVIConfig(n_epochs=5)

    def test_n_samples_validation(self):
        """Test n_samples must be at least 1."""
        # Valid
        config = GMVIConfig(n_samples=50)
        assert config.n_samples == 50

        # Invalid: zero
        with pytest.raises(ValidationError):
            GMVIConfig(n_samples=0)

    def test_patience_validation(self):
        """Test patience must be at least 1."""
        # Valid
        config = GMVIConfig(patience=10)
        assert config.patience == 10

        # Invalid: zero
        with pytest.raises(ValidationError):
            GMVIConfig(patience=0)

    def test_model_type_frozen(self):
        """Test that model_type is frozen."""
        config = GMVIConfig()
        assert config.model_type == "GMVI"

        # Cannot change frozen field
        with pytest.raises((ValidationError, AttributeError)):
            config.model_type = "OTHER"


class TestCreateConfigFactory:
    """Test cases for the create_config factory function."""

    def test_create_node_model_config(self):
        """Test creating NodeModelConfig via factory."""
        config = create_config("NodeModel", learning_rate=0.005)

        assert isinstance(config, NodeModelConfig)
        assert config.model_type == "NodeModel"
        assert config.learning_rate == 0.005

    def test_create_gmvi_config(self):
        """Test creating GMVIConfig via factory."""
        config = create_config("GMVI", prior_std=8.0)

        assert isinstance(config, GMVIConfig)
        assert config.model_type == "GMVI"
        assert config.prior_std == 8.0

    def test_create_config_case_insensitive(self):
        """Test factory is case-insensitive."""
        config1 = create_config("nodemodel")
        config2 = create_config("NODEMODEL")
        config3 = create_config("NodeModel")

        assert all(isinstance(c, NodeModelConfig) for c in [config1, config2, config3])

        config4 = create_config("gmvi")
        config5 = create_config("GMVI")

        assert all(isinstance(c, GMVIConfig) for c in [config4, config5])

    def test_create_config_unknown_type(self):
        """Test factory raises error for unknown type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_config("UnknownModel")

    def test_create_config_with_parameters(self):
        """Test factory passes parameters correctly."""
        config = create_config(
            "NodeModel",
            learning_rate=0.02,
            num_steps=2000,
            prior_type=PriorType.UNIFORM
        )

        assert config.learning_rate == 0.02
        assert config.num_steps == 2000
        assert config.prior_type == PriorType.UNIFORM

    def test_create_config_validates_parameters(self):
        """Test factory validates parameters."""
        # Should raise ValidationError for invalid parameters
        with pytest.raises(ValidationError):
            create_config("GMVI", outlier_prob=2.0)


class TestConfigInheritance:
    """Test that configs properly inherit from BaseModelConfig."""

    def test_nodemodel_has_base_fields(self):
        """Test NodeModelConfig has base fields."""
        config = NodeModelConfig()

        assert hasattr(config, 'learning_rate')
        assert hasattr(config, 'num_steps')
        assert hasattr(config, 'model_type')

    def test_gmvi_has_base_fields(self):
        """Test GMVIConfig has base fields."""
        config = GMVIConfig()

        assert hasattr(config, 'learning_rate')
        assert hasattr(config, 'num_steps')
        assert hasattr(config, 'model_type')

    def test_gmvi_overrides_defaults(self):
        """Test that GMVI properly overrides base class defaults."""
        node_config = NodeModelConfig()
        gmvi_config = GMVIConfig()

        # GMVI has different default learning rate
        assert node_config.learning_rate != gmvi_config.learning_rate
        assert gmvi_config.learning_rate == 0.01

        # But both should have valid learning rates
        assert node_config.learning_rate > 0
        assert gmvi_config.learning_rate > 0
