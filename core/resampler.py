from abc import ABC, abstractmethod

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject

from murasa.core.data_source import DataSource, read_raster_band_as_float32


class BaseResampler(ABC):
    """
    Abstract base class for resampling data sources to a specified grid shape, transform, and CRS.
    """

    def __init__(self, grid_shape, transform, crs):
        self.grid_shape = grid_shape
        self.transform = transform
        self.crs = crs

    @abstractmethod
    def resample(self, source: DataSource) -> np.ndarray:
        pass

    def validate_output(self, result: np.ndarray) -> np.ndarray:
        pass

    def fill_nodata(self, data: np.ndarray, fill_value=0.0) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        return np.where(np.isfinite(arr), arr, np.float32(fill_value))


class RasterResampler(BaseResampler):
    """
    Resampler for raster data sources to a specified grid shape, transform, and CRS.
    """

    def __init__(self, grid_shape, transform, crs, method="bilinear"):
        super().__init__(grid_shape, transform, crs)
        self.method = method

    def _get_resampling_method(self):
        match self.method:
            case "bilinear":
                return Resampling.bilinear
            case "nearest":
                return Resampling.nearest
            case "cubic":
                return Resampling.cubic
            case "average":
                return Resampling.average
            case "mode":
                return Resampling.mode
            case "max":
                return Resampling.max
            case "min":
                return Resampling.min
            case "med":
                return Resampling.med
            case "q1":
                return Resampling.q1
            case "q3":
                return Resampling.q3
            case _:
                raise ValueError(f"Unknown resampling method: {self.method}")

    def resample(self, source: DataSource) -> np.ndarray:
        raster = np.full(self.grid_shape, fill_value=np.nan, dtype=np.float32)
        resampling = self._get_resampling_method()

        if source.is_loaded() and isinstance(source.data, np.ndarray):
            src_data = np.asarray(source.data, dtype=np.float32)
            nodata_val = source.metadata.get('nodata') if isinstance(source.metadata, dict) else None
            if nodata_val is not None and np.isfinite(nodata_val):
                nodata_f32 = np.float32(nodata_val)
                if np.any(src_data == nodata_f32):
                    src_data = src_data.copy()
                    src_data[src_data == nodata_f32] = np.nan
            src_transform = source.metadata.get('transform')
            src_crs = source.crs
        else:
            with rasterio.open(source.path) as src:
                src_data, _ = read_raster_band_as_float32(src, band=1)
                src_transform = src.transform
                src_crs = src.crs or source.crs or "EPSG:4326"

        reproject(
            source=src_data,
            destination=raster,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=self.transform,
            dst_crs=self.crs,
            dst_nodata=np.nan,
            resampling=resampling
        )

        return raster


class VectorResampler(BaseResampler):
    """
    Resampler for vector data sources to a specified grid shape, transform, and CRS.
    """

    def __init__(self, grid_shape, transform, crs):
        super().__init__(grid_shape, transform, crs)

    def resample(self, source: DataSource,
                 value_column: str = None,
                 all_touched: bool = True,
                 fill_value: float = 0.0) -> np.ndarray:

        gdf = source.data.to_crs(self.crs)
        if value_column and value_column in gdf.columns:
            shapes = [(geom, val) for geom, val in
                      zip(gdf.geometry, gdf[value_column])]
        else:
            shapes = [(geom, 1) for geom in gdf.geometry]
        result = rasterize(
            shapes=shapes,
            out_shape=self.grid_shape,
            transform=self.transform,
            fill=fill_value,
            dtype=np.float32,
            all_touched=all_touched
        )
        return result
