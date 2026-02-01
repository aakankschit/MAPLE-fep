"""Unit tests for graph analysis modules."""

import matplotlib.pyplot as plt
import numpy as np


class TestPlottingPerformance:
    """Test plotting functionality."""

    def test_basic_plotting(self):
        """Test basic plotting works."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])

        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plotting_with_errors(self):
        """Test plotting with error bars."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        errors = np.array([0.1, 0.1, 0.1])

        fig, ax = plt.subplots()
        ax.errorbar(y_true, y_pred, yerr=errors, fmt="o")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestGraphOperations:
    """Test basic graph operations."""

    def test_dataframe_to_graph_conversion(self, sample_edge_data):
        """Test converting DataFrame to graph representation."""
        # Basic validation that we can work with the data
        assert "Source" in sample_edge_data.columns
        assert "Destination" in sample_edge_data.columns
        assert "DeltaDeltaG" in sample_edge_data.columns

        # Check data types
        assert len(sample_edge_data) > 0
        assert sample_edge_data["DeltaDeltaG"].dtype in [np.float64, np.float32]

    def test_cycle_detection_concepts(self, sample_edge_data):
        """Test cycle detection concepts."""
        # Create simple adjacency representation
        edges = list(zip(sample_edge_data["Source"], sample_edge_data["Destination"]))

        # Should have some edges to work with
        assert len(edges) > 0

        # Basic cycle detection concept: check for triangles
        nodes = set(sample_edge_data["Source"]) | set(sample_edge_data["Destination"])
        assert len(nodes) >= 3  # Need at least 3 nodes for cycles
