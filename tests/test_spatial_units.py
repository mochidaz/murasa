import tempfile
import unittest
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString, box
from rasterio.transform import from_bounds, from_origin

import murasa.core as murasa_core
from murasa.core.config import EngineConfig, SpatialConfig
from murasa.core.data_source import DataRegistry
from murasa.core.processing import (
    ProcessingContext,
    cell_size_in_meters,
    crs_equal,
    is_geographic_crs,
    meters_to_crs_units,
    resolve_metric_crs,
)
from murasa.core.terrain import calculate_aspect, calculate_curvature, calculate_slope
from murasa.engine.risk_engine import RiskEngine
from murasa.plugins.curvature import CurvaturePlugin, SlopeAspectPlugin
from murasa.plugins.slope import SlopePlugin


class TestSpatialUnitsAndCRS(unittest.TestCase):

    @staticmethod
    def _geographic_transform_60n(shape, cell_m: float = 10.0):
        rows, cols = shape
        bounds = (
            10.0,
            60.0,
            10.0 + cols * (cell_m / 55800.0),
            60.0 + rows * (cell_m / 111412.29),
        )
        return bounds, from_bounds(*bounds, cols, rows)

    def test_core_all_exports_exist(self):
        for symbol in murasa_core.__all__:
            self.assertTrue(hasattr(murasa_core, symbol), f"Missing core export: {symbol}")

    def test_resolve_metric_crs_auto_worldwide(self):
        gdf_eq = gpd.GeoDataFrame(geometry=[Point(107.6, -6.9)], crs="EPSG:4326")
        gdf_north = gpd.GeoDataFrame(geometry=[Point(10.7, 59.9)], crs="EPSG:4326")
        gdf_west = gpd.GeoDataFrame(geometry=[Point(-74.0, 40.7)], crs="EPSG:4326")

        self.assertEqual(resolve_metric_crs(gdf_eq, "auto"), "EPSG:32748")
        self.assertEqual(resolve_metric_crs(gdf_north, "auto"), "EPSG:32632")
        self.assertEqual(resolve_metric_crs(gdf_west, "auto"), "EPSG:32618")
        self.assertEqual(resolve_metric_crs(gdf_eq, "EPSG:3857"), "EPSG:3857")

        gdf_utm = gdf_eq.to_crs("EPSG:32748")
        self.assertEqual(resolve_metric_crs(gdf_utm, "auto"), "EPSG:32748")
        self.assertEqual(resolve_metric_crs(None, "auto"), "EPSG:6933")

    def test_spatial_config_explicit_resolution_unit(self):
        cfg_deg = SpatialConfig(target_crs="EPSG:4326", resolution=0.001, resolution_unit="crs")
        self.assertEqual(cfg_deg.resolution_unit, "crs")

        engine = RiskEngine(EngineConfig(spatial=cfg_deg))
        engine.setup_grid(bounds=(107.60, -6.95, 107.70, -6.85))
        self.assertEqual(engine.grid_shape, (100, 100))

        with self.assertRaises(ValueError):
            SpatialConfig(resolution=0.0)
        with self.assertRaises(ValueError):
            SpatialConfig(resolution_unit="parsecs")

    def test_unit_conversions_and_geographic_roundtrip(self):
        self.assertTrue(is_geographic_crs("EPSG:4326"))
        self.assertFalse(is_geographic_crs("EPSG:2263"))

        tf_ft = from_origin(980000.0, 200000.0, 100.0, 100.0)
        dx_ft_m, dy_ft_m = cell_size_in_meters(tf_ft, (100, 100), "EPSG:2263")
        self.assertAlmostEqual(dx_ft_m, 30.48, delta=0.01)
        self.assertAlmostEqual(dy_ft_m, 30.48, delta=0.01)

        for lat in (0.0, -6.9, 35.0, 60.0):
            bounds = (10.0, lat - 0.05, 10.1, lat + 0.05)
            res_x, res_y = meters_to_crs_units(10.0, "EPSG:4326", bounds=bounds)
            width = int(round((bounds[2] - bounds[0]) / res_x))
            height = int(round((bounds[3] - bounds[1]) / res_y))
            tf = from_bounds(*bounds, width, height)

            dx_m, dy_m = cell_size_in_meters(tf, (height, width), "EPSG:4326")
            self.assertAlmostEqual(dx_m, 10.0, delta=0.05)
            self.assertAlmostEqual(dy_m, 10.0, delta=0.05)

        tf_equal_deg = from_origin(10.0, 60.05, 0.001, 0.001)
        dx_60, dy_60 = cell_size_in_meters(tf_equal_deg, (100, 100), "EPSG:4326")
        self.assertAlmostEqual(dx_60 / dy_60, 0.5, delta=0.01)

        rx, ry = meters_to_crs_units(0.04, "EPSG:4326", lat_mid=0.0)
        self.assertLess(rx, 1e-5)
        self.assertLess(ry, 1e-5)
        with self.assertRaises(ValueError):
            meters_to_crs_units(0.0, "EPSG:4326")
        with self.assertRaises(ValueError):
            meters_to_crs_units(-5.0, "EPSG:32748")

    def test_terrain_and_plugins_consistent_between_utm_and_geographic(self):
        ramp_shape = (50, 50)
        tf_utm_ramp = from_origin(500000.0, 9200000.0, 10.0, 10.0)
        ctx_utm_ramp = ProcessingContext(ramp_shape, tf_utm_ramp, "EPSG:32748")
        cols_ramp = np.arange(ramp_shape[1], dtype=np.float64)[None, :]
        dem_utm_ramp = np.broadcast_to(cols_ramp * ctx_utm_ramp.dx_m, ramp_shape).copy()

        tf_geo_ramp = from_origin(10.0, 60.025, 0.0002, 0.0002)
        ctx_geo_ramp = ProcessingContext(ramp_shape, tf_geo_ramp, "EPSG:4326")
        dem_geo_ramp = np.broadcast_to(cols_ramp * ctx_geo_ramp.dx_m, ramp_shape).copy()

        np.testing.assert_allclose(calculate_slope(dem_utm_ramp, ctx_utm_ramp.cell_size_m, degrees=True), 45.0, atol=1e-4)
        np.testing.assert_allclose(calculate_slope(dem_geo_ramp, ctx_geo_ramp.cell_size_m, degrees=True), 45.0, atol=1e-4)
        np.testing.assert_allclose(calculate_aspect(dem_geo_ramp, ctx_geo_ramp.cell_size_m), 270.0, atol=1e-4)
        np.testing.assert_allclose(calculate_curvature(dem_geo_ramp, ctx_geo_ramp.cell_size_m, method="profile"), 0.0, atol=1e-6)

        shape = (40, 40)
        tf_utm = from_origin(500000.0, 9200400.0, 10.0, 10.0)
        _, tf_geo = self._geographic_transform_60n(shape, cell_m=10.0)

        rows = np.arange(shape[0], dtype=np.float32)[:, None]
        cols = np.arange(shape[1], dtype=np.float32)[None, :]
        dem_arr = (cols * 5.0 + (rows - 20.0) ** 2 * 0.2).astype(np.float32)

        reg_utm = DataRegistry()
        reg_utm.register_raster_array("dem", dem_arr, crs="EPSG:32748", transform=tf_utm)
        reg_geo = DataRegistry()
        reg_geo.register_raster_array("dem", dem_arr, crs="EPSG:4326", transform=tf_geo)

        np.testing.assert_allclose(
            SlopePlugin(inverse=False).process(reg_utm, shape, tf_utm, "EPSG:32748"),
            SlopePlugin(inverse=False).process(reg_geo, shape, tf_geo, "EPSG:4326"),
            atol=0.01,
        )
        np.testing.assert_allclose(
            CurvaturePlugin().process(reg_utm, shape, tf_utm, "EPSG:32748"),
            CurvaturePlugin().process(reg_geo, shape, tf_geo, "EPSG:4326"),
            atol=0.02,
        )
        np.testing.assert_allclose(
            SlopeAspectPlugin().process(reg_utm, shape, tf_utm, "EPSG:32748"),
            SlopeAspectPlugin().process(reg_geo, shape, tf_geo, "EPSG:4326"),
            atol=0.02,
        )

    def test_distance_to_features_no_overlap_returns_zero_or_inf(self):
        shape = (20, 20)
        tf_utm = from_origin(500000.0, 9200200.0, 10.0, 10.0)
        gdf_far = gpd.GeoDataFrame(
            geometry=[LineString([(600000.0, 9200000.0), (600000.0, 9200200.0)])],
            crs="EPSG:32748",
        )
        reg = DataRegistry()
        src = reg.register_vector("river", gdf_far, crs="EPSG:32748")
        ctx = ProcessingContext(shape, tf_utm, "EPSG:32748")

        np.testing.assert_array_equal(ctx.distance_to_features(src, max_distance=500.0), np.zeros(shape, dtype=np.float32))
        self.assertTrue(np.all(np.isinf(ctx.distance_to_features(src, max_distance=None))))

    def test_distance_to_features_matches_between_utm_and_geographic(self):
        shape = (101, 101)
        tf_utm = from_origin(500000.0, 9201010.0, 10.0, 10.0)
        x_mid_utm = 500000.0 + 50.5 * 10.0
        gdf_utm = gpd.GeoDataFrame(
            geometry=[LineString([(x_mid_utm, 9200000.0), (x_mid_utm, 9201010.0)])],
            crs="EPSG:32748",
        )
        reg_utm = DataRegistry()
        src_utm = reg_utm.register_vector("river", gdf_utm, crs="EPSG:32748")
        score_utm = ProcessingContext(shape, tf_utm, "EPSG:32748").distance_to_features(src_utm, max_distance=200.0)

        bounds_geo, tf_geo = self._geographic_transform_60n(shape, cell_m=10.0)
        x_mid_geo = bounds_geo[0] + 50.5 * tf_geo.a
        gdf_geo = gpd.GeoDataFrame(
            geometry=[LineString([(x_mid_geo, bounds_geo[1]), (x_mid_geo, bounds_geo[3])])],
            crs="EPSG:4326",
        )
        reg_geo = DataRegistry()
        src_geo = reg_geo.register_vector("river", gdf_geo, crs="EPSG:4326")
        score_geo = ProcessingContext(shape, tf_geo, "EPSG:4326").distance_to_features(src_geo, max_distance=200.0)

        self.assertAlmostEqual(float(score_utm[50, 60]), 0.5, delta=0.02)
        self.assertAlmostEqual(float(score_geo[50, 60]), 0.5, delta=0.02)
        np.testing.assert_allclose(score_utm[50, :], score_geo[50, :], atol=0.02)

    def test_risk_engine_setup_grid_with_geographic_crs_and_meter_resolution(self):
        engine = RiskEngine(EngineConfig(spatial=SpatialConfig(target_crs="EPSG:4326", resolution=10.0)))
        gdf_admin = gpd.GeoDataFrame(geometry=[box(107.60, -6.91, 107.61, -6.90)], crs="EPSG:4326")
        engine.registry.register_vector("admin", gdf_admin, crs="EPSG:4326")
        engine.setup_grid()

        height, width = engine.grid_shape
        self.assertGreater(width, 100)
        self.assertGreater(height, 100)
        dx_m, dy_m = cell_size_in_meters(engine.transform, engine.grid_shape, "EPSG:4326")
        self.assertAlmostEqual(dx_m, 10.0, delta=0.1)
        self.assertAlmostEqual(dy_m, 10.0, delta=0.1)

    def test_risk_engine_reprojects_mismatched_reference_raster_and_admin_crs(self):
        tf_utm = from_origin(780000.0, 9240000.0, 10.0, 10.0)
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "ref_utm.tif"
            with rasterio.open(
                ref_path,
                "w",
                driver="GTiff",
                height=20,
                width=20,
                count=1,
                dtype="float32",
                crs="EPSG:32748",
                transform=tf_utm,
            ) as dst:
                dst.write(np.zeros((20, 20), dtype=np.float32), 1)

            engine = RiskEngine(
                EngineConfig(spatial=SpatialConfig(target_crs="EPSG:4326", reference_raster=str(ref_path)))
            )
            engine.setup_grid()

            self.assertLess(abs(engine.transform.c), 180.0)
            self.assertLess(abs(engine.transform.f), 90.0)
            self.assertTrue(crs_equal(engine.crs, "EPSG:4326"))

            gdf_admin_utm = gpd.GeoDataFrame(
                geometry=[box(780020.0, 9239820.0, 780180.0, 9239980.0)],
                crs="EPSG:32748",
            )
            engine.set_study_area(gdf_admin_utm)
            self.assertTrue(engine.study_area_mask.any())
            self.assertFalse(engine.study_area_mask.all())


if __name__ == "__main__":
    unittest.main()
