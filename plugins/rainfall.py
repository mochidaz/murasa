import numpy as np
from typing import Optional, Dict

from murasa.core.base import ParameterPlugin, register_plugin
from murasa.core.data_source import DataRegistry, SourceType
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger


@register_plugin("rainfall")
class RainfallPlugin(ParameterPlugin):
    """
    Rainfall intensity risk parameter supporting both raster like CHIRPS and vector inputs.
    """

    short_name = "rainfall"
    display_name = "Rainfall"

    def __init__(
        self,
        name: str = "rainfall",
        weight: float = 0.15,
        group: str = "hazard",
        column_name: str = "Band 1",
        column: Optional[str] = None,
        normalize_method: str = "bmkg_threshold",
        value_mapping: Optional[Dict[int, float]] = None,
        **kwargs
    ):
        super().__init__(name, weight, group, **kwargs)
        self.column_name = column or column_name
        self.normalize_method = normalize_method
        self.value_mapping = value_mapping

    def validate_requirements(self, registry: DataRegistry) -> bool:
        return registry.has('rainfall')

    def process(self, registry: DataRegistry, grid_shape: tuple, transform, crs):
        self.log_processing()

        rain_source = registry.get('rainfall')
        ctx = ProcessingContext(grid_shape, transform, crs)

        if rain_source.source_type == SourceType.RASTER or isinstance(rain_source.data, np.ndarray):
            logger.info(f"      Processing raw raster rainfall source: {rain_source.name}")
            rainfall_raster = ctx.resample_raster(rain_source)
            rainfall_raster = registry.apply_study_area_mask(rainfall_raster)

            if not np.any(np.isfinite(rainfall_raster)):
                logger.warning("      [WARNING] Resampled rainfall has NO valid pixels overlapping with study area!")

            rainfall_raster = np.where(np.isfinite(rainfall_raster), np.maximum(rainfall_raster, 0.0), np.nan)
        else:
            gdf_rain = rain_source.data.to_crs(crs).copy()
            col = self.column_name
            if col not in gdf_rain.columns:
                alternatives = ['intensity', 'value', 'curah_hujan', 'rainfall', 'gridcode', 'Band 1']
                col = next((c for c in alternatives if c in gdf_rain.columns), None)
                if not col:
                    logger.warning("No rainfall value column found, returning zero array")
                    return np.zeros(grid_shape, dtype=np.float32)

            if self.value_mapping and col in gdf_rain.columns:
                mapped = gdf_rain[col].map(self.value_mapping)
                if mapped.notna().any():
                    gdf_rain['_rain_val'] = mapped.fillna(gdf_rain[col])
                    col = '_rain_val'

            rain_source.data = gdf_rain
            rainfall_raster = ctx.rasterize_vector(rain_source, column=col, fill_value=np.nan)
            rainfall_raster = registry.apply_study_area_mask(rainfall_raster)
            rainfall_raster = np.where(np.isfinite(rainfall_raster), np.maximum(rainfall_raster, 0.0), np.nan)

        valid_rain = np.isfinite(rainfall_raster)
        if valid_rain.any():
            logger.info(
                f"      Raw Rainfall range: {np.nanmin(rainfall_raster):.2f} mm "
                f"to {np.nanmax(rainfall_raster):.2f} mm"
            )

        if self.normalize_method == "bmkg_threshold":
            score_raster = np.full_like(rainfall_raster, np.nan, dtype=np.float32)
            score_raster[valid_rain & (rainfall_raster < 100)] = 0.2
            score_raster[valid_rain & (rainfall_raster >= 100) & (rainfall_raster < 300)] = 0.5
            score_raster[valid_rain & (rainfall_raster >= 300) & (rainfall_raster < 500)] = 0.8
            score_raster[valid_rain & (rainfall_raster >= 500)] = 1.0
        elif self.normalize_method == "fixed_scale":
            score_raster = np.clip(rainfall_raster / 500.0, 0.0, 1.0)
        elif self.normalize_method == "minmax":
            if valid_rain.any():
                r_min, r_max = float(np.nanmin(rainfall_raster)), float(np.nanmax(rainfall_raster))
                score_raster = (
                    (rainfall_raster - r_min) / (r_max - r_min)
                    if r_max > r_min else np.where(valid_rain, 0.0, np.nan)
                )
            else:
                score_raster = np.full_like(rainfall_raster, np.nan, dtype=np.float32)
        else:
            score_raster = rainfall_raster

        self.result = score_raster.astype(np.float32)
        if np.isfinite(self.result).any():
            logger.info(
                f"      Final Rainfall score range ({self.normalize_method}): "
                f"{np.nanmin(self.result):.2f} to {np.nanmax(self.result):.2f}"
            )
        self.log_result()
        return self.result
