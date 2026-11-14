import numpy as np

from murasa.core.base import RasterParameterPlugin, register_plugin
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger


@register_plugin("twi")
class TWIPlugin(RasterParameterPlugin):
    """
    Topographic Wetness Index parameter.
    Resamples an existing TWI raster or derives it from DEM and slope.
    """

    source_keys = ['twi']
    inverse = False
    short_name = "twi"
    display_name = "TWI"

    def validate_requirements(self, registry):
        # For now, TWI from external raster is a MUST.
        # TWI-on-the-fly will be implemented later.
        return registry.has('twi') # or registry.has('dem')

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()
        ctx = ProcessingContext(grid_shape, transform, crs)

        if registry.has('twi'):
            twi = ctx.resample_raster(registry.get('twi'))
        else:
            # No fallbacks for now. Will be implemented later.
            logger.error("      Cannot compute TWI: No TWI raster available")
            return ctx.empty_grid()

        twi = registry.apply_study_area_mask(twi)
        valid = np.isfinite(twi)
        if not valid.any():
            return ctx.empty_grid(fill=0.5)

        normalized = self._normalize(twi)
        if self.inverse:
            normalized = 1 - normalized

        self.result = normalized.astype(np.float32)
        self.log_result()
        return self.result
