import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from murasa.core.data_source import DataRegistry, DataSource, SourceType
from murasa.core.resampler import RasterResampler
from murasa.plugins.rainfall import RainfallPlugin


class TestRasterNodataHandling(unittest.TestCase):

    @staticmethod
    def _write_geotiff(
        path: Path,
        data: np.ndarray,
        transform,
        crs: str = "EPSG:32748",
        nodata=None,
        dtype: str = "float32",
    ):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(data.astype(dtype), 1)

    def test_raster_resampler_unloaded_source_masks_explicit_nodata(self):
        src_tf = from_origin(500000.0, 9200100.0, 10.0, 10.0)
        dst_tf = from_origin(500005.0, 9200100.0, 10.0, 10.0)

        arr = np.full((10, 10), 100.0, dtype=np.float32)
        arr[:, :5] = -9999.0

        with tempfile.TemporaryDirectory() as tmp:
            tif_path = Path(tmp) / "explicit_nodata.tif"
            self._write_geotiff(tif_path, arr, src_tf, nodata=-9999.0)

            source = DataSource(name="dem", source_type=SourceType.RASTER, path=tif_path, crs="EPSG:32748")
            self.assertFalse(source.is_loaded())

            out = RasterResampler((10, 10), dst_tf, "EPSG:32748", method="bilinear").resample(source)

            self.assertTrue(
                np.all(np.isnan(out[:, :3])),
                f"Nodata region leaked non-NaN values: {np.unique(out[:, :3])[:5]}",
            )
            valid = out[np.isfinite(out)]
            self.assertGreater(valid.size, 0)
            self.assertGreaterEqual(
                float(valid.min()),
                99.0,
                f"Bilinear interpolation blended nodata (-9999) into valid pixels, got min={valid.min():.2f}",
            )

    def test_data_source_load_raster_masks_sentinel_when_header_nodata_unset(self):
        tf = from_origin(500000.0, 9200040.0, 10.0, 10.0)
        arr = np.full((4, 4), 50.0, dtype=np.float32)
        arr[:2, :] = -9999.0

        with tempfile.TemporaryDirectory() as tmp:
            tif_path = Path(tmp) / "unset_nodata_header.tif"
            self._write_geotiff(tif_path, arr, tf, nodata=None)

            source = DataSource(name="dem", source_type=SourceType.RASTER, path=tif_path, crs="EPSG:32748").load()

            self.assertTrue(
                np.all(np.isnan(source.data[:2, :])),
                f"Expected -9999.0 sentinel rows to become NaN, got {source.data[:2, 0]} (metadata nodata={source.metadata.get('nodata')!r})",
            )
            np.testing.assert_allclose(source.data[2:, :], 50.0)

    def test_rainfall_plugin_treats_unset_nodata_sentinel_as_nan_not_zero_mm(self):
        shape = (4, 4)
        tf = from_origin(500000.0, 9200040.0, 10.0, 10.0)
        rain_arr = np.full(shape, 400.0, dtype=np.float32)
        rain_arr[:2, :] = -9999.0

        with tempfile.TemporaryDirectory() as tmp:
            tif_path = Path(tmp) / "rain_unset_nodata.tif"
            self._write_geotiff(tif_path, rain_arr, tf, nodata=None)

            registry = DataRegistry()
            registry.register_from_path("rainfall", tif_path, SourceType.RASTER, crs="EPSG:32748")

            score = RainfallPlugin(normalize_method="bmkg_threshold").process(registry, shape, tf, "EPSG:32748")

            self.assertTrue(
                np.all(np.isnan(score[:2, :])),
                f"Expected nodata rainfall area to remain NaN, got scores {np.unique(score[:2, :])}",
            )
            np.testing.assert_allclose(score[2:, :], 0.8, atol=1e-6)

    def test_zero_uint8_int16_inmemory_nodata_and_fill_nodata(self):
        tf = from_origin(500000.0, 9200040.0, 10.0, 10.0)
        with tempfile.TemporaryDirectory() as tmp:
            zero_path = Path(tmp) / "zero_uint8.tif"
            zero_arr = np.array([[0, 5], [0, 7]], dtype=np.uint8)
            self._write_geotiff(zero_path, zero_arr, tf, nodata=0, dtype="uint8")

            src_zero = DataSource(name="z", source_type=SourceType.RASTER, path=zero_path, crs="EPSG:32748").load()
            self.assertTrue(np.all(np.isnan(src_zero.data[:, 0])))
            np.testing.assert_allclose(src_zero.data[:, 1], [5.0, 7.0])

            i16_path = Path(tmp) / "i16.tif"
            i16_arr = np.array([[-9999, 250], [-9999, 300]], dtype=np.int16)
            self._write_geotiff(i16_path, i16_arr, tf, nodata=-9999, dtype="int16")

            src_i16 = DataSource(name="dem", source_type=SourceType.RASTER, path=i16_path, crs="EPSG:32748")
            resampler = RasterResampler((2, 2), tf, "EPSG:32748", method="nearest")
            out_i16 = resampler.resample(src_i16)
            self.assertTrue(np.all(np.isnan(out_i16[:, 0])))
            np.testing.assert_allclose(out_i16[:, 1], [250.0, 300.0])

        reg = DataRegistry()
        raw = np.array([[-9999, 10], [20, 30]], dtype=np.int16)
        src_mem = reg.register_raster_array("mem", raw, crs="EPSG:32748", transform=tf, nodata=-9999)
        resampler = RasterResampler((2, 2), tf, "EPSG:32748", method="nearest")
        out_mem = resampler.resample(src_mem)
        self.assertTrue(np.isnan(out_mem[0, 0]))
        self.assertEqual(raw[0, 0], -9999)

        filled = resampler.fill_nodata(out_mem, fill_value=-1.0)
        np.testing.assert_allclose(filled, [[-1.0, 10.0], [20.0, 30.0]])


if __name__ == "__main__":
    unittest.main()
