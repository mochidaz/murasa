import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

sys.path.append(".")

from murasa.core.analysis import (
    compute_gvf,
    compute_gvf_with_breaks,
    compute_spearman,
    compute_weight_sensitivity,
)

try:
    from murasa.custom_plugins.reporting import (
        VectorExplainabilityPlugin,
        _extract_time_metadata,
    )
    _HAS_CUSTOM_REPORTING = True
except ImportError:
    VectorExplainabilityPlugin = None
    _extract_time_metadata = None
    _HAS_CUSTOM_REPORTING = False


def _make_sample_gdf(n: int = 20) -> gpd.GeoDataFrame:
    """Create a minimal GeoDataFrame matching explainability post-processing inputs."""
    rng = np.random.RandomState(42)
    return gpd.GeoDataFrame(
        {
            "geometry": [Point(0, i) for i in range(n)],
            "area_ha": rng.uniform(10, 100, n),
            "risk_score": rng.uniform(2, 6, n),
            "pct_extreme": rng.uniform(0, 50, n),
            "pct_very_high": rng.uniform(0, 40, n),
            "pct_high": rng.uniform(0, 35, n),
            "composite_risk": rng.uniform(0, 100, n),
            "p_rainfall": rng.uniform(0, 1, n),
            "p_elevation": rng.uniform(0, 1, n),
            "p_slope": rng.uniform(0, 1, n),
            "p_twi": rng.uniform(0, 1, n),
            "p_landuse": rng.uniform(0, 1, n),
            "p_prox": rng.uniform(0, 1, n),
        },
        crs="EPSG:4326",
    )


class _FakeWeights:
    """Minimal stand-in for WeightsConfig."""

    rainfall = 0.30
    elevation = 0.20
    slope = 0.10
    twi = 0.15
    proximity = 0.15
    land_use = 0.10

    def as_dict(self):
        return {
            "rainfall": self.rainfall,
            "elevation": self.elevation,
            "slope": self.slope,
            "twi": self.twi,
            "proximity": self.proximity,
            "land_use": self.land_use,
        }


class TestGVFMetrics(unittest.TestCase):
    """Tests for Goodness of Variance Fit (GVF) helpers in murasa.core.analysis."""

    def test_compute_gvf_perfect_groups(self):
        """Perfectly separated groups -> GVF should be very high."""
        vals = np.array([1, 1, 5, 5, 10, 10], dtype=float)
        labels = np.array([0, 0, 1, 1, 2, 2])
        gvf = compute_gvf(vals, labels)
        self.assertGreater(gvf, 0.9, f"Expected GVF > 0.9, got {gvf}")

    def test_compute_gvf_zero_variance(self):
        """All values identical -> trivially perfect, GVF = 1.0."""
        vals = np.ones(5)
        labels = np.array([0, 1, 1, 0, 2])
        gvf = compute_gvf(vals, labels)
        self.assertEqual(gvf, 1.0)

    def test_compute_gvf_single_class(self):
        """Single class -> SDAM == SDCM -> GVF = 0.0."""
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        labels = np.array([0, 0, 0, 0])
        gvf = compute_gvf(vals, labels)
        self.assertEqual(gvf, 0.0, f"Expected 0.0, got {gvf}")

    def test_compute_gvf_with_nan(self):
        """NaN values should be filtered out."""
        vals = np.array([1, np.nan, 5, 5, 10, 10], dtype=float)
        labels = np.array([0, 0, 1, 1, 2, 2])
        gvf = compute_gvf(vals, labels)
        self.assertGreaterEqual(gvf, 0.0)
        self.assertLessEqual(gvf, 1.0)

    def test_compute_gvf_with_breaks_perfect(self):
        """Well-separated breaks -> high GVF."""
        vals = np.array([1, 2, 3, 10, 11, 12, 20, 21, 22], dtype=float)
        breaks = [1, 4, 13, 22]
        gvf = compute_gvf_with_breaks(vals, breaks)
        self.assertGreater(gvf, 0.9, f"Expected GVF > 0.9, got {gvf}")

    def test_compute_gvf_with_breaks_zero_variance(self):
        """Constant values -> GVF = 1.0."""
        vals = np.full(10, 5.0)
        breaks = [5.0, 5.0, 5.0]
        gvf = compute_gvf_with_breaks(vals, breaks)
        self.assertEqual(gvf, 1.0)

    def test_compute_gvf_with_breaks_single_break(self):
        """Single class (2 break points) -> GVF = 0.0."""
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        breaks = [1.0, 5.0]
        gvf = compute_gvf_with_breaks(vals, breaks)
        self.assertEqual(gvf, 0.0, f"Expected 0.0, got {gvf}")


class TestSpearmanCorrelation(unittest.TestCase):
    """Tests for Spearman rank correlation helper in murasa.core.analysis."""

    def test_compute_spearman_perfect(self):
        """Perfectly monotonic -> rho = 1.0."""
        rho = compute_spearman([1, 2, 3, 4], [10, 20, 30, 40])
        self.assertAlmostEqual(rho, 1.0, places=6)

    def test_compute_spearman_inverse(self):
        """Perfectly inverse monotonic -> rho = -1.0."""
        rho = compute_spearman([1, 2, 3, 4], [40, 30, 20, 10])
        self.assertAlmostEqual(rho, -1.0, places=6)

    def test_compute_spearman_insufficient(self):
        """Only 1 data point -> should return 0.0."""
        rho = compute_spearman([1], [2])
        self.assertEqual(rho, 0.0)

    def test_compute_spearman_with_nan(self):
        """NaN should be dropped, still compute on valid pairs."""
        a = [1, 2, float("nan"), 4, 5]
        b = [10, 20, 30, 40, 50]
        rho = compute_spearman(a, b)
        self.assertGreater(rho, 0.5)
        self.assertLessEqual(rho, 1.0)


class TestCoreWeightSensitivity(unittest.TestCase):
    """Domain-neutral tests for compute_weight_sensitivity in murasa.core.analysis."""

    def test_weight_sensitivity_returns_dataframe_and_valid_range(self):
        rng = np.random.RandomState(42)
        scores = {f"factor_{i}": rng.uniform(0, 1, 50) for i in range(4)}
        weights = {f"factor_{i}": 0.25 for i in range(4)}
        df_sens = compute_weight_sensitivity(scores, weights)
        self.assertEqual(len(df_sens), 4)
        self.assertEqual(list(df_sens.columns), ["factor", "stability_index", "impact"])
        for _, row in df_sens.iterrows():
            self.assertGreaterEqual(row["stability_index"], -1.0)
            self.assertLessEqual(row["stability_index"], 1.0)

    def test_weight_sensitivity_empty_returns_empty_dataframe(self):
        df_sens = compute_weight_sensitivity({}, {})
        self.assertTrue(df_sens.empty)
        self.assertEqual(list(df_sens.columns), ["factor", "stability_index", "impact"])


@unittest.skipUnless(_HAS_CUSTOM_REPORTING, "custom_plugins.reporting not present")
class TestSensitivityAnalysis(unittest.TestCase):
    """Tests for custom reporting threshold and weight sensitivity helpers."""

    def setUp(self):
        self.plugin = VectorExplainabilityPlugin()

    def test_threshold_sensitivity_returns_3_scenarios(self):
        gdf = _make_sample_gdf()
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.plugin._run_threshold_sensitivity(gdf, Path(tmpdir))
        self.assertEqual(set(results.keys()), {"Conservative", "Moderate", "Aggressive"})
        for metrics in results.values():
            self.assertIn("extreme_count", metrics)
            self.assertIn("high_risk_count", metrics)
            self.assertIn("high_risk_area_pct", metrics)

    def test_threshold_sensitivity_exports_csv(self):
        gdf = _make_sample_gdf()
        with tempfile.TemporaryDirectory() as tmpdir:
            self.plugin._run_threshold_sensitivity(gdf, Path(tmpdir))
            csv_path = Path(tmpdir) / "sensitivity_thresholds.csv"
            self.assertTrue(csv_path.exists(), "sensitivity_thresholds.csv not created")
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 3)

    def test_weight_sensitivity_returns_dataframe(self):
        gdf = _make_sample_gdf()
        weights = _FakeWeights()
        with tempfile.TemporaryDirectory() as tmpdir:
            df_sens = self.plugin._run_weight_sensitivity(gdf, weights, Path(tmpdir))
        self.assertIsInstance(df_sens, pd.DataFrame)
        self.assertIn("factor", df_sens.columns)
        self.assertIn("stability_index", df_sens.columns)
        self.assertIn("impact", df_sens.columns)
        self.assertEqual(len(df_sens), 6)

    def test_weight_sensitivity_stability_range(self):
        """All stability indices should be between -1 and 1."""
        gdf = _make_sample_gdf(50)
        weights = _FakeWeights()
        with tempfile.TemporaryDirectory() as tmpdir:
            df_sens = self.plugin._run_weight_sensitivity(gdf, weights, Path(tmpdir))
        for _, row in df_sens.iterrows():
            self.assertGreaterEqual(
                row["stability_index"],
                -1.0,
                f"{row['factor']}: stability_index out of range: {row['stability_index']}",
            )
            self.assertLessEqual(
                row["stability_index"],
                1.0,
                f"{row['factor']}: stability_index out of range: {row['stability_index']}",
            )

    def test_weight_sensitivity_exports_csv(self):
        gdf = _make_sample_gdf()
        weights = _FakeWeights()
        with tempfile.TemporaryDirectory() as tmpdir:
            self.plugin._run_weight_sensitivity(gdf, weights, Path(tmpdir))
            csv_path = Path(tmpdir) / "sensitivity_weights.csv"
            self.assertTrue(csv_path.exists(), "sensitivity_weights.csv not created")
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 6)


@unittest.skipUnless(_HAS_CUSTOM_REPORTING, "custom_plugins.reporting not present")
class TestMetadataAndGoodnessMetrics(unittest.TestCase):
    """Tests for custom reporting goodness/stability injection and time metadata extraction."""

    def setUp(self):
        self.plugin = VectorExplainabilityPlugin()

    def test_add_goodness_and_stability_adds_columns(self):
        gdf = _make_sample_gdf()
        gdf["severity"] = pd.qcut(gdf["composite_risk"], q=6, labels=False, duplicates="drop")
        self.plugin._add_goodness_and_stability(gdf, weights=None)
        self.assertIn("gvf", gdf.columns)
        self.assertIn("spearman_risk", gdf.columns)
        self.assertIn("spearman_factors", gdf.columns)
        self.assertGreaterEqual(gdf["gvf"].iloc[0], 0.0)
        self.assertLessEqual(gdf["gvf"].iloc[0], 1.0)

    def test_add_goodness_uses_jenks_breaks(self):
        """When jenks_breaks are in attrs, GVF should use breaks-based method."""
        gdf = _make_sample_gdf()
        vals = gdf["composite_risk"].values
        breaks = [
            float(np.min(vals)),
            float(np.percentile(vals, 33)),
            float(np.percentile(vals, 66)),
            float(np.max(vals)),
        ]
        gdf.attrs["jenks_breaks"] = breaks
        self.plugin._add_goodness_and_stability(gdf, weights=None)
        self.assertIn("gvf", gdf.columns)
        self.assertGreater(gdf["gvf"].iloc[0], 0.0)

    def test_extract_time_metadata(self):
        class DummyConfig:
            def __init__(self, plugin_parameters=None, parameters=None):
                self.plugin_parameters = plugin_parameters or {}
                self.parameters = parameters or {}

        cfg1 = DummyConfig(plugin_parameters={"rainfall": {"year": 2025, "month": 12}})
        y, m = _extract_time_metadata(cfg1)
        self.assertEqual(y, 2025)
        self.assertEqual(m, 12)

        cfg2 = DummyConfig(parameters={"year": 2024, "month": 6})
        y, m = _extract_time_metadata(cfg2)
        self.assertEqual(y, 2024)
        self.assertEqual(m, 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
