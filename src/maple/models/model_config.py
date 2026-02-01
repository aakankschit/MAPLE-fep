"""
Configuration classes for MAPLE models

This module provides configuration classes for different model types in MAPLE,
including the NodeModel and GMVI_model, with proper validation using Pydantic.
"""

from abc import ABC
from enum import Enum
from typing import List, Optional, Union
import warnings

from pydantic import BaseModel, Field, field_validator


class PriorType(str, Enum):
    """Available prior distribution types for node values"""
    NORMAL = "normal"
    GAMMA = "gamma"
    UNIFORM = "uniform"
    STUDENT_T = "student_t"
    LAPLACE = "laplace"


class GuideType(str, Enum):
    """Available guide types for variational inference"""
    AUTO_DELTA = "auto_delta"
    AUTO_NORMAL = "auto_normal"


class ErrorDistributionType(str, Enum):
    """Available error distribution types for cycle errors"""
    NORMAL = "normal"
    SKEWED_NORMAL = "skewed_normal"


class BaseModelConfig(BaseModel, ABC):
    """Base configuration class for all MAPLE models"""
    
    learning_rate: float = Field(
        default=0.001, ge=1e-6, le=1.0, 
        description="Learning rate for optimization"
    )
    
    num_steps: int = Field(
        default=5000, ge=10, le=100000, 
        description="Number of optimization steps"
    )
    
    model_type: str = Field(
        description="Type of model (NodeModel or GMVI)"
    )


class NodeModelConfig(BaseModelConfig):
    """Configuration specific to NodeModel"""
    
    model_type: str = Field(default="NodeModel", frozen=True)
    
    prior_type: PriorType = Field(
        default=PriorType.NORMAL,
        description="Type of prior distribution for node values"
    )
    
    prior_parameters: List[float] = Field(
        default=[0.0, 1.0], 
        description="Parameters for the prior distribution"
    )
    
    error_std: float = Field(
        default=1.0, gt=0.0, 
        description="Standard deviation of cycle errors"
    )
    
    error_distribution: ErrorDistributionType = Field(
        default=ErrorDistributionType.NORMAL,
        description="Type of distribution for cycle errors"
    )
    
    error_skew: float = Field(
        default=0.0,
        description="Skewness parameter for skewed normal error distribution"
    )
    
    guide_type: GuideType = Field(
        default=GuideType.AUTO_DELTA,
        description="Type of guide for variational inference"
    )

    patience: int = Field(
        default=100, ge=1,
        description="Early stopping patience: stop training if no improvement after this many steps"
    )

    @field_validator("prior_parameters")
    @classmethod
    def validate_prior_parameters(cls, v, info):
        """Validate prior parameters based on prior type"""
        prior_type = info.data.get("prior_type", PriorType.NORMAL)
        
        if prior_type == PriorType.NORMAL:
            if len(v) != 2:
                raise ValueError("Normal prior requires exactly 2 parameters: [mean, std]")
            if v[1] <= 0:
                raise ValueError("Standard deviation must be positive")
                
        elif prior_type == PriorType.GAMMA:
            if len(v) != 2:
                raise ValueError("Gamma prior requires exactly 2 parameters: [alpha, beta]")
            if v[0] <= 0 or v[1] <= 0:
                raise ValueError("Gamma parameters must be positive")
                
        elif prior_type == PriorType.UNIFORM:
            if len(v) != 2:
                raise ValueError("Uniform prior requires exactly 2 parameters: [lower, upper]")
            if v[0] >= v[1]:
                raise ValueError("Uniform prior: lower bound must be less than upper bound")
                
        elif prior_type == PriorType.STUDENT_T:
            if len(v) != 2:
                raise ValueError("Student-T prior requires exactly 2 parameters: [df, scale]")
            if v[0] <= 0 or v[1] <= 0:
                raise ValueError("Student-T parameters must be positive")
                
        elif prior_type == PriorType.LAPLACE:
            if len(v) != 2:
                raise ValueError("Laplace prior requires exactly 2 parameters: [loc, scale]")
            if v[1] <= 0:
                raise ValueError("Laplace scale parameter must be positive")
                
        return v


class GMVIConfig(BaseModelConfig):
    """Configuration specific to GMVI_model"""

    model_type: str = Field(default="GMVI", frozen=True)

    prior_std: float = Field(
        default=5.0, gt=0.0,
        description="Prior standard deviation for node values (σ₀)"
    )

    normal_std: float = Field(
        default=1.0, gt=0.0,
        description="Standard deviation for normal edges (σ₁)"
    )

    outlier_std: float = Field(
        default=3.0, gt=0.0,
        description="Standard deviation for outlier edges (σ₂)"
    )

    outlier_prob: float = Field(
        default=0.2, ge=0.0, le=1.0,
        description="Global probability of an edge being an outlier (π)"
    )

    kl_weight: float = Field(
        default=0.1, ge=0.0,
        description="Weight for KL divergence term in ELBO"
    )

    n_epochs: int = Field(
        default=1000, ge=10,
        description="Maximum number of training epochs"
    )

    n_samples: int = Field(
        default=100, ge=1,
        description="Number of Monte Carlo samples for ELBO estimation"
    )

    patience: int = Field(
        default=50, ge=1,
        description="Early stopping patience"
    )

    # Override base class fields to match GMVI conventions
    learning_rate: float = Field(
        default=0.01, ge=1e-6, le=1.0,
        description="Learning rate for ADAM optimizer"
    )

    @field_validator("outlier_prob")
    @classmethod
    def validate_outlier_prob(cls, v):
        """Ensure outlier probability is valid"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Outlier probability must be between 0 and 1")
        return v

    @field_validator("normal_std", "outlier_std")
    @classmethod
    def validate_std_relationship(cls, v, info):
        """Optionally validate relationship between normal and outlier stds"""
        # This is optional but can help ensure outlier_std > normal_std
        # which is typically expected
        return v


class WCCConfig(BaseModel):
    """Configuration specific to WCC_model (Weighted Cycle Closure)"""

    model_type: str = Field(default="WCC", frozen=True)

    tolerance: float = Field(
        default=1e-6, gt=0.0,
        description="Convergence tolerance for cycle closure iteration"
    )

    max_iterations: int = Field(
        default=1000, ge=1,
        description="Maximum number of cycle closure iterations"
    )

    use_weights: bool = Field(
        default=True,
        description="Whether to use edge weights (inverse variance) in correction"
    )

    max_cycle_length: int = Field(
        default=8, ge=3,
        description="Maximum cycle length to detect (larger = more cycles but slower)"
    )

    max_cycles: int = Field(
        default=100000, ge=1,
        description="Maximum number of cycles to find before stopping"
    )

    cycle_detection_timeout: float = Field(
        default=300.0, gt=0.0,
        description="Timeout in seconds for cycle detection (default: 5 minutes)"
    )

    @field_validator("tolerance")
    @classmethod
    def validate_tolerance(cls, v):
        """Ensure tolerance is reasonable"""
        if v <= 0:
            raise ValueError("Tolerance must be positive")
        if v > 1.0:
            raise ValueError("Tolerance too large (should be << 1)")
        return v

    @field_validator("max_cycle_length")
    @classmethod
    def validate_max_cycle_length(cls, v):
        """Warn if cycle length is too large"""
        if v > 12:
            warnings.warn(
                f"max_cycle_length={v} may be very slow for dense graphs. "
                "Consider using smaller values (6-8) for initial testing.",
                UserWarning
            )
        return v


def create_config(model_type: str = "NodeModel", **kwargs) -> Union[BaseModelConfig, 'WCCConfig']:
    """
    Factory function to create appropriate config based on model type.

    Parameters
    ----------
    model_type : str
        Type of model ("NodeModel", "GMVI", or "WCC")
    **kwargs
        Configuration parameters for the specific model

    Returns
    -------
    BaseModelConfig or WCCConfig
        Appropriate configuration object for the model type

    Examples
    --------
    >>> config = create_config("NodeModel", learning_rate=0.001)
    >>> gmvi_config = create_config("GMVI", prior_std=5.0, outlier_prob=0.3)
    >>> wcc_config = create_config("WCC", tolerance=1e-6, max_iterations=1000)
    """
    if model_type.upper() == "NODEMODEL" or model_type == "NodeModel":
        return NodeModelConfig(**kwargs)
    elif model_type.upper() == "GMVI":
        return GMVIConfig(**kwargs)
    elif model_type.upper() == "WCC":
        return WCCConfig(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'NodeModel', 'GMVI', or 'WCC'")