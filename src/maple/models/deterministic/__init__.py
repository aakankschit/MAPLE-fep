"""
Deterministic estimators for MAPLE.

These estimators use closed-form or iterative algorithms (no probabilistic
inference) to estimate node values from graph-structured FEP data.
"""

from .cycle_closure import CycleClosureCorrection
from .spectral_correction import SpectralCorrection

__all__ = ["CycleClosureCorrection", "SpectralCorrection"]
