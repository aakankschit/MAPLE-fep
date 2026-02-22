"""
MAPLE Models Package

This package provides estimators for FEP analysis, organized into
probabilistic (variational inference) and deterministic (closed-form) methods.
"""

from .base import BaseEstimator
from .graph_data import GraphData
from .config import (
    BaseEstimatorConfig,
    VariationalEstimatorConfig,
    GaussianMixtureVIConfig,
    CycleClosureCorrectionConfig,
    SpectralCorrectionConfig,
    PriorType,
    GuideType,
    ErrorDistributionType,
    create_config,
)
from .probabilistic import VariationalEstimator, GaussianMixtureVI
from .deterministic import CycleClosureCorrection, SpectralCorrection

__all__ = [
    # Base
    "BaseEstimator",
    "GraphData",
    # Estimators
    "VariationalEstimator",
    "GaussianMixtureVI",
    "CycleClosureCorrection",
    "SpectralCorrection",
    # Configs
    "BaseEstimatorConfig",
    "VariationalEstimatorConfig",
    "GaussianMixtureVIConfig",
    "CycleClosureCorrectionConfig",
    "SpectralCorrectionConfig",
    # Enums
    "PriorType",
    "GuideType",
    "ErrorDistributionType",
    # Factory
    "create_config",
]
