"""
Probabilistic estimators for MAPLE.

These estimators use variational inference (Pyro) to estimate node values
from graph-structured FEP data.
"""

from .variational_estimator import VariationalEstimator
from .gaussian_mixture_vi import GaussianMixtureVI

__all__ = ["VariationalEstimator", "GaussianMixtureVI"]
