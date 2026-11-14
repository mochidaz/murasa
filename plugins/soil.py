import numpy as np
from typing import Optional, Dict
from murasa.core.base import ParameterPlugin, register_plugin
from murasa.core.data_source import DataRegistry, SourceType
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger


@register_plugin("soil")
class SoilPlugin(ParameterPlugin):
    """
    Soil susceptibility parameter plugin.
    Maps soil classification codes or raster soil layers into normalized 0-1 risk scores.
    """

    short_name = "soil"
    display_name = "Soil"

    def __init__(
        self,
        name: str = "soil",
        weight: float = 0.15,
        group: str = "susceptibility",
        column_name: str = "WRB2",
        column: Optional[str] = None,
        score_mapping: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__(name, weight, group, **kwargs)
        self.column_name = column or column_name
        self.score_mapping = score_mapping or {
            'AC': 0.90,
            'AN': 0.85,
            'LP': 0.80,
            'CM': 0.75,
            'NT': 0.60,
            'LX': 0.50,
            'FL': 0.20,
            'VR': 0.30,
            'default': 0.50
        }

    def validate_requirements(self, registry: DataRegistry) -> bool:
        return registry.has('soil')

    def process(self, registry: DataRegistry, grid_shape: tuple, transform, crs):
        self.log_processing()

        soil_source = registry.get('soil')
        ctx = ProcessingContext(grid_shape, transform, crs)

        default_score = self.score_mapping.get('default', 0.5)

        if soil_source.source_type == SourceType.RASTER or isinstance(soil_source.data, np.ndarray):
            logger.info(f"      Processing raster soil source: {soil_source.name}")
            soil_raster = ctx.resample_raster(soil_source)
            score_raster = np.nan_to_num(soil_raster, nan=default_score)
        else:
            gdf_soil = soil_source.data.to_crs(crs).copy()
            col = self.column_name

            if col not in gdf_soil.columns:
                alternatives = ['HWSD2_SM_7', 'WRB2', 'WRB4', 'HWSD2_SMU_WRB2', 'DOMSOI', 'FAOSOIL']
                col = next((c for c in alternatives if c in gdf_soil.columns), None)
                if not col:
                    logger.warning("      No valid soil classification column found! Returning default score 0.5")
                    return registry.apply_study_area_mask(np.full(grid_shape, 0.5, dtype=np.float32))

            gdf_soil['landslide_score'] = (
                gdf_soil[col].astype(str).str.upper().map(self.score_mapping).fillna(default_score)
            )

            logger.info(f"      Rasterizing {len(gdf_soil)} soil polygons using column '{col}'...")
            soil_source.data = gdf_soil
            score_raster = ctx.rasterize_vector(soil_source, column='landslide_score', fill_value=np.nan)
            score_raster = np.nan_to_num(score_raster, nan=default_score)

        clipped = np.clip(score_raster, 0.0, 1.0).astype(np.float32)
        self.result = registry.apply_study_area_mask(clipped)

        if np.isfinite(self.result).any():
            logger.info(f"      Soil score range: {np.nanmin(self.result):.2f} to {np.nanmax(self.result):.2f}")
        self.log_result()
        return self.result
