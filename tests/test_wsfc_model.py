"""
Tests for the SpectralCorrection and related uncertainty-weighted inference features.

Covers:
- SpectralCorrection initialization and fit (weighted and unweighted)
- SFC === MLE validation (critical gate)
- Incidence matrix correctness
- Edge error clamping
- Suffix correctness
- VariationalEstimator heteroscedastic branch (wMAP)
- Graceful fallback when no error column is present
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from maple.models import (
    GraphData,
    VariationalEstimator,
    VariationalEstimatorConfig,
    SpectralCorrection,
    SpectralCorrectionConfig,
    GuideType,
    PriorType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleDataset:
    """Minimal dataset stub for unit tests."""

    def __init__(self, edges_df: pd.DataFrame, nodes_df: pd.DataFrame):
        self.dataset_edges = edges_df
        self.dataset_nodes = nodes_df
        self.estimators: list = []


def _make_simple_dataset(include_errors: bool = False, zero_errors: bool = False):
    """Create a tiny 4-node graph for testing."""
    edges = pd.DataFrame({
        "Source": ["A", "B", "C", "A"],
        "Destination": ["B", "C", "D", "C"],
        "DeltaDeltaG": [1.0, -0.5, 2.0, 0.6],
    })
    if include_errors:
        if zero_errors:
            edges["DeltaDeltaG Error"] = [0.0, 0.3, 0.2, 0.0]
        else:
            edges["DeltaDeltaG Error"] = [0.3, 0.5, 0.2, 0.4]
    nodes = pd.DataFrame({
        "Name": ["A", "B", "C", "D"],
        "Exp. DeltaG": [0.0, 1.0, 0.5, 2.5],
    })
    return _SimpleDataset(edges, nodes)


# ===========================================================================
# SpectralCorrection - initialization
# ===========================================================================

class TestSpectralCorrectionInit:
    """Basic initialization tests for SpectralCorrection."""

    def test_init_defaults(self):
        ds = _make_simple_dataset()
        model = SpectralCorrection(SpectralCorrectionConfig(), ds)
        assert model.use_weights is True
        assert model.node_estimates is None

    def test_init_unweighted(self):
        ds = _make_simple_dataset()
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=False), ds)
        assert model.use_weights is False


# ===========================================================================
# SpectralCorrection - incidence matrix
# ===========================================================================

class TestIncidenceMatrix:
    """Verify the incidence matrix shape and sign convention."""

    def test_shape(self):
        ds = _make_simple_dataset()
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=False), ds)
        model.graph_data = model._extract_graph_data()
        B = model._build_incidence_matrix()
        M = model.graph_data.num_edges
        N = model.graph_data.num_nodes
        assert B.shape == (M, N)

    def test_row_sums_zero(self):
        """Each row of B should have exactly one +1 and one -1."""
        ds = _make_simple_dataset()
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=False), ds)
        model.graph_data = model._extract_graph_data()
        B = model._build_incidence_matrix()
        assert np.allclose(B.sum(axis=1), 0.0)

    def test_sign_convention(self):
        """B[k, src] = -1, B[k, dst] = +1."""
        ds = _make_simple_dataset()
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=False), ds)
        model.graph_data = model._extract_graph_data()
        B = model._build_incidence_matrix()
        for k in range(model.graph_data.num_edges):
            src = model.graph_data.source_nodes[k]
            dst = model.graph_data.target_nodes[k]
            assert B[k, src] == -1.0
            assert B[k, dst] == 1.0


# ===========================================================================
# SpectralCorrection - fit (unweighted = SFC)
# ===========================================================================

class TestSpectralCorrectionFitSFC:
    """Test unweighted (SFC) fitting."""

    def test_fit_produces_estimates(self):
        ds = _make_simple_dataset()
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=False), ds)
        model.fit()
        assert model.node_estimates is not None
        assert model.edge_estimates is not None
        assert model.node_uncertainties is not None
        assert model.edge_uncertainties is not None

    def test_node_count(self):
        ds = _make_simple_dataset()
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=False), ds)
        model.fit()
        assert len(model.node_estimates) == 4

    def test_edge_count(self):
        ds = _make_simple_dataset()
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=False), ds)
        model.fit()
        assert len(model.edge_estimates) == 4

    def test_edge_consistency(self):
        """Edge estimates should equal target_node - source_node."""
        ds = _make_simple_dataset()
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=False), ds)
        model.fit()
        for (src, dst), val in model.edge_estimates.items():
            expected = model.node_estimates[dst] - model.node_estimates[src]
            assert abs(val - expected) < 1e-10


# ===========================================================================
# SFC === MLE validation (critical gate)
# ===========================================================================

class TestSFCEqualsMLE:
    """
    Verify that unweighted SFC produces the same node estimates as
    VariationalEstimator with Uniform prior (MLE). Both are unweighted least-squares
    on the graph, so mean-centered estimates must match.
    """

    def test_sfc_matches_mle(self, mock_dataset):
        # --- SFC ---
        sfc_model = SpectralCorrection(SpectralCorrectionConfig(use_weights=False), mock_dataset)
        sfc_model.fit()

        # --- MLE via VariationalEstimator ---
        mle_config = VariationalEstimatorConfig(
            prior_type=PriorType.UNIFORM,
            prior_parameters=[-100.0, 100.0],
            guide_type=GuideType.AUTO_DELTA,
            num_steps=5000,
            learning_rate=0.03,
            patience=200,
        )
        mle_model = VariationalEstimator(config=mle_config, dataset=mock_dataset)
        mle_model.fit()

        # Mean-center both
        sfc_vals = np.array([sfc_model.node_estimates[n] for n in sorted(sfc_model.node_estimates)])
        mle_vals = np.array([mle_model.node_estimates[n] for n in sorted(mle_model.node_estimates)])
        sfc_centered = sfc_vals - sfc_vals.mean()
        mle_centered = mle_vals - mle_vals.mean()

        max_diff = np.max(np.abs(sfc_centered - mle_centered))
        assert max_diff < 0.05, (
            f"SFC and MLE mean-centered estimates differ by {max_diff:.4f} "
            f"(should be < 0.05 kcal/mol)"
        )


# ===========================================================================
# SpectralCorrection - fit (weighted = WSFC)
# ===========================================================================

class TestSpectralCorrectionFitWeighted:
    """Test weighted (WSFC) fitting."""

    def test_fit_with_errors(self):
        ds = _make_simple_dataset(include_errors=True)
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=True), ds)
        model.fit()
        assert model.node_estimates is not None
        assert model.graph_data.edge_weights is not None

    def test_weighted_differs_from_unweighted(self):
        """Weighted solution should generally differ from unweighted."""
        ds_w = _make_simple_dataset(include_errors=True)
        ds_u = _make_simple_dataset(include_errors=True)
        wsfc = SpectralCorrection(SpectralCorrectionConfig(use_weights=True), ds_w)
        wsfc.fit()
        sfc = SpectralCorrection(SpectralCorrectionConfig(use_weights=False), ds_u)
        sfc.fit()

        w_vals = np.array([wsfc.node_estimates[n] for n in sorted(wsfc.node_estimates)])
        u_vals = np.array([sfc.node_estimates[n] for n in sorted(sfc.node_estimates)])
        # Mean-center
        w_vals -= w_vals.mean()
        u_vals -= u_vals.mean()
        # They should not be identical (unless errors are uniform)
        assert not np.allclose(w_vals, u_vals, atol=1e-6)

    def test_fallback_when_no_errors(self):
        """use_weights=True without error column should warn and use uniform weights."""
        ds = _make_simple_dataset(include_errors=False)
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=True), ds)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit()
            fallback_warnings = [x for x in w if "Falling back" in str(x.message)]
            assert len(fallback_warnings) >= 1
        assert model.node_estimates is not None


# ===========================================================================
# Edge error clamping
# ===========================================================================

class TestEdgeErrorClamping:
    """Zero or very small errors should be clamped to 0.01."""

    def test_zero_errors_clamped(self):
        ds = _make_simple_dataset(include_errors=True, zero_errors=True)
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=True), ds)
        model.graph_data = model._extract_graph_data()
        # Check that zero errors are clamped
        assert model.graph_data.edge_errors is not None
        for e in model.graph_data.edge_errors:
            assert e >= 0.01


# ===========================================================================
# Suffix correctness
# ===========================================================================

class TestSuffixCorrectness:
    """Verify correct suffix assignment."""

    def test_sfc_suffix(self):
        ds = _make_simple_dataset()
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=False), ds)
        model.fit()
        model.add_predictions_to_dataset()
        assert "SFC" in ds.dataset_nodes.columns
        assert "SFC" in ds.dataset_edges.columns
        assert "SFC" in ds.estimators

    def test_wsfc_suffix(self):
        ds = _make_simple_dataset(include_errors=True)
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=True), ds)
        model.fit()
        model.add_predictions_to_dataset()
        assert "WSFC" in ds.dataset_nodes.columns
        assert "WSFC" in ds.dataset_edges.columns
        assert "WSFC" in ds.estimators


# ===========================================================================
# get_results
# ===========================================================================

class TestGetResults:
    """Verify get_results returns a valid dict."""

    def test_keys(self):
        ds = _make_simple_dataset(include_errors=True)
        model = SpectralCorrection(SpectralCorrectionConfig(use_weights=True), ds)
        model.fit()
        results = model.get_results()
        assert "node_estimates" in results
        assert "edge_estimates" in results
        assert "node_uncertainties" in results
        assert "method" in results
        assert results["method"] == "WSFC"

    def test_raises_before_fit(self):
        ds = _make_simple_dataset()
        model = SpectralCorrection(SpectralCorrectionConfig(), ds)
        with pytest.raises(ValueError, match="fitted"):
            model.get_results()


# ===========================================================================
# VariationalEstimator heteroscedastic (wMAP)
# ===========================================================================

class TestVariationalEstimatorWeighted:
    """Test VariationalEstimator with use_edge_weights=True."""

    def test_wmap_suffix(self, mock_dataset_with_errors):
        config = VariationalEstimatorConfig(
            prior_type=PriorType.NORMAL,
            prior_parameters=[0.0, 1.0],
            guide_type=GuideType.AUTO_DELTA,
            use_edge_weights=True,
            num_steps=500,
            learning_rate=0.03,
            patience=100,
        )
        model = VariationalEstimator(config=config, dataset=mock_dataset_with_errors)
        model.fit()
        model.add_predictions_to_dataset()
        assert "wMAP" in mock_dataset_with_errors.dataset_nodes.columns

    def test_wvi_suffix(self, mock_dataset_with_errors):
        config = VariationalEstimatorConfig(
            prior_type=PriorType.NORMAL,
            prior_parameters=[0.0, 1.0],
            guide_type=GuideType.AUTO_NORMAL,
            use_edge_weights=True,
            num_steps=500,
            learning_rate=0.03,
            patience=100,
        )
        model = VariationalEstimator(config=config, dataset=mock_dataset_with_errors)
        model.fit()
        model.add_predictions_to_dataset()
        assert "wVI" in mock_dataset_with_errors.dataset_nodes.columns

    def test_fallback_warns_no_errors(self, mock_dataset):
        """use_edge_weights=True without error column should warn."""
        config = VariationalEstimatorConfig(
            use_edge_weights=True,
            num_steps=100,
            learning_rate=0.03,
            patience=50,
        )
        model = VariationalEstimator(config=config, dataset=mock_dataset)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit()
            fallback_warnings = [x for x in w if "Falling back" in str(x.message)]
            assert len(fallback_warnings) >= 1


# ===========================================================================
# GraphData shared module
# ===========================================================================

class TestGraphDataShared:
    """Verify GraphData from shared module works across models."""

    def test_import_from_models(self):
        from maple.models import GraphData as GD
        assert GD is GraphData

    def test_optional_fields_default_none(self):
        gd = GraphData(
            source_nodes=[0],
            target_nodes=[1],
            edge_values=[1.0],
            num_nodes=2,
            num_edges=1,
            node_to_idx={"a": 0, "b": 1},
            idx_to_node={0: "a", 1: "b"},
        )
        assert gd.edge_errors is None
        assert gd.edge_weights is None

    def test_optional_fields_populated(self):
        gd = GraphData(
            source_nodes=[0],
            target_nodes=[1],
            edge_values=[1.0],
            num_nodes=2,
            num_edges=1,
            node_to_idx={"a": 0, "b": 1},
            idx_to_node={0: "a", 1: "b"},
            edge_errors=[0.5],
            edge_weights=[4.0],
        )
        assert gd.edge_errors == [0.5]
        assert gd.edge_weights == [4.0]
