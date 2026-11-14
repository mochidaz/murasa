import numpy as np
from typing import Optional, Dict

from murasa.core.base import ParameterPlugin, register_plugin
from murasa.core.data_source import DataRegistry
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger


@register_plugin("land_use")
class LandUsePlugin(ParameterPlugin):
    """
    Land use susceptibility parameter plugin.
    """

    short_name = "landuse"
    display_name = "Land Use"

    def __init__(
        self,
        name: str = "land_use",
        weight: float = 0.20,
        group: str = "hazard",
        coefficients: Optional[Dict[str, float]] = None,
        priorities: Optional[Dict[str, int]] = None,
        **kwargs
    ):
        super().__init__(name, weight, group, **kwargs)
        self.coefficients = coefficients or {
            'water': 1.0,
            'settlement': 0.85,
            'building': 0.90,
            'road': 0.90,
            'paddy': 0.70,
            'plantation': 0.40,
            'forest': 0.20,
            'shrub': 0.35,
            'default': 0.50
        }
        self.priorities = priorities or {
            'water': 100,
            'building': 90,
            'road': 80,
            'settlement': 70,
            'paddy': 50,
            'plantation': 40,
            'shrub': 30,
            'forest': 20,
            'default': 0
        }

    def validate_requirements(self, registry: DataRegistry) -> bool:
        return any(k.startswith('landuse_') for k in registry.list_sources())

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()

        landuse_raster = np.full(
            grid_shape,
            self.coefficients.get('default', 0.50),
            dtype=np.float32
        )

        landuse_sources = [k for k in registry.list_sources() if k.startswith('landuse_')]
        if not landuse_sources:
            logger.warning("      No landuse data found, using default")
            return landuse_raster

        def _get_priority(src_key: str) -> int:
            lu = src_key.replace('landuse_', '')
            if lu == 'paddy' and 'paddy' not in self.priorities and 'agriculture' in self.priorities:
                return self.priorities['agriculture']
            return self.priorities.get(lu, 0)

        priority_order = sorted(landuse_sources, key=_get_priority)

        logger.info(f"      Rasterizing {len(priority_order)} landuse layers...")
        ctx = ProcessingContext(grid_shape, transform, crs)

        for name in priority_order:
            source = registry.get(name)
            lu_type = name.replace('landuse_', '')
            if lu_type == 'paddy' and 'paddy' not in self.coefficients and 'agriculture' in self.coefficients:
                coeff = self.coefficients['agriculture']
            else:
                coeff = self.coefficients.get(lu_type, self.coefficients.get('default', 0.5))

            if source.data.empty:
                continue

            mask = ctx.rasterize_vector(source).astype(bool)
            landuse_raster[mask] = coeff
            logger.debug(f"         {lu_type}: {mask.sum()} pixels")

        landuse_raster = registry.apply_study_area_mask(landuse_raster)
        self.result = landuse_raster
        self.log_result()
        return landuse_raster
