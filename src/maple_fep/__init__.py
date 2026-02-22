"""
MAPLE: Maximum A Posteriori Ligand Estimation

A Python package for analyzing Free Energy Perturbation (FEP) data using
probabilistic node models and Bayesian inference.
"""

__version__ = "0.1.0"
__author__ = "Aakankschit Nandkeolyar"
__email__ = "anandkeo@uci.edu"
__description__ = (
    "Maximum A Posteriori Learning of Energies - "
    "Tools for analyzing free energy perturbation (FEP) maps"
)

from . import dataset, graph_analysis, models, utils

__all__ = ["models", "dataset", "graph_analysis", "utils", "__version__"]
