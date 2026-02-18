"""
Base Estimator ABC for MAPLE models.

All MAPLE estimators (probabilistic and deterministic) inherit from
BaseEstimator, which defines the common interface: fit(), get_results(),
and add_predictions_to_dataset().
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from .graph_data import GraphData


class BaseEstimator(ABC):
    """Abstract base class for all MAPLE estimators.

    Subclasses must implement:
    - ``fit()`` — run the estimation algorithm
    - ``get_results()`` — return a results dict
    - ``add_predictions_to_dataset()`` — write predictions into the dataset DataFrames

    Common attributes populated by subclasses after ``fit()``:
    - ``node_estimates``
    - ``edge_estimates``
    - ``node_uncertainties``
    - ``edge_uncertainties``
    - ``graph_data``
    """

    # Subclass attributes (set in __init__)
    dataset: Any
    graph_data: Optional[GraphData]
    node_estimates: Optional[Dict[str, float]]
    edge_estimates: Optional[Dict[Tuple[str, str], float]]
    node_uncertainties: Optional[Dict[str, float]]
    edge_uncertainties: Optional[Dict[Tuple[str, str], float]]

    @abstractmethod
    def fit(self) -> None:
        """Run the estimation algorithm.

        After calling ``fit()``, results are accessible via
        ``node_estimates``, ``edge_estimates``, and ``get_results()``.
        """

    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """Return estimation results as a dictionary."""

    @abstractmethod
    def add_predictions_to_dataset(self) -> None:
        """Write predictions into the dataset DataFrames."""
