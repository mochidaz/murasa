import numpy as np
from murasa.core.base import RasterParameterPlugin, register_plugin
from murasa.core.processing import ProcessingContext


@register_plugin("slope")
class SlopePlugin(RasterParameterPlugin):
    """
    Terrain slope risk parameter.
    When inverse=True, flat terrain yields higher risk. Usually for flood.
    When inverse=False, steep terrain yields higher risk. Usually for landslide.
    """

    source_keys = ['slope']
    inverse = True
    short_name = "slope"
    display_name = "Slope"
    _VAN_ZUIDAM_DEG = [0.0, 2.0, 4.0, 8.0, 16.0, 35.0, 45.0]
    _VAN_ZUIDAM_SCORES = [0.05, 0.15, 0.30, 0.45, 0.70, 0.90, 1.00]

    def validate_requirements(self, registry):
        return registry.has('slope') or registry.has('dem')

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()
        ctx = ProcessingContext(grid_shape, transform, crs)

        if registry.has('slope'):
            slope = ctx.resample_raster(registry.get('slope'))
        else:
            dem = ctx.resample_raster(registry.get('dem'))
            from murasa.core.terrain import calculate_slope
            slope = calculate_slope(dem, ctx.resolution)

        slope = registry.apply_study_area_mask(slope)
        valid = np.isfinite(slope)

        if (
            not self.inverse
            and self.normalization in ("percentile", "van_zuidam")
            and valid.any()
            and float(np.nanmax(slope)) > 1.5
        ):
            normalized = np.full_like(slope, np.nan, dtype=np.float32)
            normalized[valid] = np.interp(
                slope[valid],
                self._VAN_ZUIDAM_DEG,
                self._VAN_ZUIDAM_SCORES,
            ).astype(np.float32)
        else:
            normalized = self._normalize(slope)
            if self.inverse:
                normalized = 1 - normalized

        self.result = normalized.astype(np.float32)
        self.log_result()
        return self.result
