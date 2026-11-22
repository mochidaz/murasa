import unittest

import numpy as np
from rasterio.transform import from_origin

from murasa.core.data_source import DataRegistry, DataSource, SourceType
from murasa.core.hydro import calculate_hand
from murasa.plugins import HANDPlugin
from murasa.plugins.hand import HANDPlugin as DirectHANDPlugin


class TestHANDCalculationAndPlugin(unittest.TestCase):

    def test_v_shaped_valley_exact_hand_and_nan_preservation(self):
        col_offsets = np.array([12.0, 5.0, 0.0, 5.0, 12.0], dtype=np.float32)
        stream_profile = np.array([10.4, 10.3, 10.2, 10.1, 10.0], dtype=np.float32)[:, None]
        dem = stream_profile + col_offsets[None, :]
        dem[0, 0] = np.nan

        stream_mask = np.zeros_like(dem, dtype=bool)
        stream_mask[:, 2] = True

        hand = calculate_hand(dem, stream_mask=stream_mask, fill=False)
        self.assertEqual(hand.dtype, np.float32)
        self.assertTrue(np.isnan(hand[0, 0]))
        np.testing.assert_allclose(hand[:, 2], 0.0, atol=1e-5)
        np.testing.assert_allclose(hand[:, 1], 5.0, atol=1e-4)
        np.testing.assert_allclose(hand[:, 3], 5.0, atol=1e-4)
        np.testing.assert_allclose(hand[1:, 0], 12.0, atol=1e-4)
        np.testing.assert_allclose(hand[:, 4], 12.0, atol=1e-4)

    def test_hand_is_invariant_to_absolute_elevation_offset(self):
        x = np.array([20.0, 10.0, 2.0, 0.0, 2.0, 10.0, 20.0], dtype=np.float32)
        slope_ns = np.array([0.4, 0.3, 0.2, 0.1, 0.0], dtype=np.float32)[:, None]
        valley_low = 200.0 + slope_ns + x[None, :]
        valley_high = 1400.0 + slope_ns + x[None, :]

        mask = np.zeros_like(valley_low, dtype=bool)
        mask[:, 3] = True

        h_low = calculate_hand(valley_low, stream_mask=mask, fill=False)
        h_high = calculate_hand(valley_high, stream_mask=mask, fill=False)
        np.testing.assert_allclose(h_low, h_high, atol=1e-4)

    def test_automatic_stream_threshold_and_all_nan_handling(self):
        rr, cc = np.mgrid[0:5, 0:5]
        dem = (4 - rr) * 0.5 + np.abs(cc - 2) * 3.0

        hand = calculate_hand(dem, stream_threshold=100.0, fill=True)
        self.assertTrue(np.all(np.isfinite(hand)))
        self.assertAlmostEqual(float(hand[4, 2]), 0.0, places=4)
        self.assertGreater(float(hand[0, 0]), float(hand[4, 2]))

        all_nan = np.full((3, 3), np.nan, dtype=np.float32)
        hand_nan = calculate_hand(all_nan)
        self.assertTrue(np.all(np.isnan(hand_nan)))
        self.assertEqual(hand_nan.dtype, np.float32)

    def test_hand_plugin_from_dem_and_precomputed_raster(self):
        self.assertIs(HANDPlugin, DirectHANDPlugin)

        tf = from_origin(500000.0, 9200050.0, 10.0, 10.0)
        rr, cc = np.mgrid[0:6, 0:6]
        dem = (200.0 + (5 - rr) * 0.2 + np.abs(cc - 2) * 5.0).astype(np.float32)

        reg = DataRegistry()
        reg.register(
            "dem",
            DataSource("dem", SourceType.RASTER, data=dem, crs="EPSG:32748", metadata={"transform": tf}),
        )

        plugin_exp = HANDPlugin(stream_threshold=5.0, scale_height=10.0)
        self.assertTrue(plugin_exp.validate_requirements(reg))
        scores = plugin_exp.process(reg, dem.shape, tf, "EPSG:32748")
        self.assertEqual(scores.shape, dem.shape)
        self.assertTrue(np.all(np.isfinite(scores)))
        np.testing.assert_allclose(scores[:, 2], 1.0, atol=1e-5)
        self.assertLess(float(scores[0, 5]), float(scores[0, 2]))

        plugin_cap = HANDPlugin(stream_threshold=5.0, max_hand=15.0, normalization="linear_cap")
        cap_scores = plugin_cap.process(reg, dem.shape, tf, "EPSG:32748")
        np.testing.assert_allclose(cap_scores[:, 2], 1.0, atol=1e-5)
        self.assertAlmostEqual(float(cap_scores[5, 5]), 0.0, places=5)

        plugin_area = HANDPlugin(stream_area_km2=0.0005, fill=True)
        area_scores = plugin_area.process(reg, dem.shape, tf, "EPSG:32748")
        self.assertEqual(area_scores.shape, dem.shape)
        self.assertTrue(np.all(np.isfinite(area_scores)))

        reg_pre = DataRegistry()
        pre_hand = np.array([[0.0, 10.0], [20.0, np.nan]], dtype=np.float32)
        tf_small = from_origin(500000.0, 9200020.0, 10.0, 10.0)
        reg_pre.register(
            "hand",
            DataSource("hand", SourceType.RASTER, data=pre_hand, crs="EPSG:32748", metadata={"transform": tf_small}),
        )
        res_pre = HANDPlugin(scale_height=10.0).process(reg_pre, pre_hand.shape, tf_small, "EPSG:32748")
        self.assertAlmostEqual(float(res_pre[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(res_pre[0, 1]), float(np.exp(-1.0)), places=5)
        self.assertTrue(np.isnan(res_pre[1, 1]))


if __name__ == "__main__":
    unittest.main()
