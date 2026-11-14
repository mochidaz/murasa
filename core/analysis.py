import numpy as np
from geopandas import GeoDataFrame
from rasterstats import zonal_stats
from typing import List, Dict, Any

from .logger import logger
from .data_source import DataSource


def calculate_zonal_stats(
        vector_gdf: GeoDataFrame,
        raster_source: DataSource,
        stats: List[str] = ['mean', 'count'],
        nodata: float = np.nan,
        categorical: bool = False,
        transform=None,
        crs=None
) -> List[Dict[str, Any]]:
    """
    Calculate zonal statistics for vector features against a raster source.
    """
    if hasattr(raster_source, 'data'):
        if raster_source.data is None:
            logger.warning(f"DataSource {getattr(raster_source, 'name', 'unknown')} has no data loaded.")
            return []

        data = np.asanyarray(raster_source.data)
        if hasattr(raster_source, 'metadata'):
            transform = transform or raster_source.metadata.get('transform')
            crs = crs or raster_source.crs
    elif isinstance(raster_source, np.ndarray):
        data = raster_source
    else:
        try:
            data = np.array(raster_source)
        except Exception as e:
            logger.error(f"Invalid raster source type: {type(raster_source)}. Error: {e}")
            return []

    if transform is None or crs is None:
        raise ValueError("Transform and CRS required for array-based zonal stats")

    try:
        gdf_target = (
            vector_gdf.to_crs(crs)
            if getattr(vector_gdf, "crs", None) is not None and vector_gdf.crs != crs
            else vector_gdf
        )
        return zonal_stats(
            gdf_target,
            data,
            affine=transform,
            stats=stats,
            categorical=categorical,
            nodata=nodata
        )
    except Exception as e:
        logger.error(f"Zonal stats failed: {e}")
        return []
