import numpy as np
from murasa.core.base import ParameterPlugin, register_plugin
from murasa.core.data_source import DataRegistry
from murasa.core.processing import cell_size_in_meters
from murasa.core.resampler import RasterResampler
from murasa.core.logger import logger


@register_plugin("curvature")
class CurvaturePlugin(ParameterPlugin):
    """
    Terrain curvature risk parameter.
    """

    short_name = "curvature"
    display_name = "Curvature"

    def __init__(
        self,
        name: str = "curvature",
        weight: float = 0.10,
        group: str = "hazard",
        inverse: bool = True,
        curvature_type: str = "profile",
        **kwargs
    ):
        super().__init__(name, weight, group, **kwargs)
        self.inverse = inverse
        self.curvature_type = curvature_type

    def validate_requirements(self, registry: DataRegistry) -> bool:
        return registry.has('curvature') or registry.has('dem')

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()
        resampler = RasterResampler(grid_shape, transform, crs)

        if registry.has('curvature'):
            curvature = resampler.resample(registry.get('curvature'))
        else:
            logger.info("      Computing curvature from DEM...")
            curvature = self._compute_from_dem(registry, resampler, transform, grid_shape, crs)

        curvature = registry.apply_study_area_mask(curvature)
        normalized = self._normalize_curvature(curvature)
        if not self.inverse:
            normalized = 1.0 - normalized
        self.result = normalized.astype(np.float32)
        self.log_result()
        return self.result

    def _compute_from_dem(self, registry, resampler, transform, grid_shape=None, crs="EPSG:4326"):
        from murasa.core.terrain import calculate_curvature
        dem = resampler.resample(registry.get('dem'))
        dx_m, dy_m = cell_size_in_meters(transform, grid_shape or dem.shape, crs)
        return calculate_curvature(dem, (dy_m, dx_m), method=self.curvature_type)

    def _normalize_curvature(self, curvature):
        """Normalize curvature to 0-1 where concave (negative) maps toward 1.0."""
        valid = np.isfinite(curvature)
        if not valid.any():
            return np.full_like(curvature, np.nan, dtype=np.float32)

        p1, p99 = np.nanpercentile(curvature, [1, 99])
        curvature_clipped = np.clip(curvature, p1, p99)

        max_abs = max(abs(p1), abs(p99))
        if max_abs == 0:
            result = np.full_like(curvature, np.nan, dtype=np.float32)
            result[valid] = 0.5
            return result

        scaled = curvature_clipped / max_abs
        normalized = (1 - scaled) / 2
        return np.clip(normalized, 0, 1)


@register_plugin("aspect")
class SlopeAspectPlugin(ParameterPlugin):
    """
    Slope aspect (orientation) risk parameter.
    """

    short_name = "aspect"
    display_name = "Aspect"

    def __init__(
        self,
        name: str = "aspect",
        weight: float = 0.05,
        group: str = "hazard",
        high_risk_directions: list = None,
        **kwargs
    ):
        super().__init__(name, weight, group, **kwargs)
        self.high_risk_directions = high_risk_directions or [0, 45, 315]

    def validate_requirements(self, registry: DataRegistry) -> bool:
        return registry.has('aspect') or registry.has('dem')

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()

        if registry.has('aspect'):
            resampler = RasterResampler(grid_shape, transform, crs, method="nearest")
            aspect = resampler.resample(registry.get('aspect'))
        else:
            resampler = RasterResampler(grid_shape, transform, crs)
            aspect = self._compute_aspect(registry, resampler)

        risk = self._aspect_to_risk(aspect)
        risk = registry.apply_study_area_mask(risk)
        self.result = risk.astype(np.float32)
        self.log_result()
        return self.result

    def _compute_aspect(self, registry, resampler):
        from murasa.core.terrain import calculate_aspect
        dem = resampler.resample(registry.get('dem'))
        dx_m, dy_m = cell_size_in_meters(resampler.transform, resampler.grid_shape, resampler.crs)
        return calculate_aspect(dem, (dy_m, dx_m))

    def _aspect_to_risk(self, aspect):
        risk = np.zeros_like(aspect)
        for direction in self.high_risk_directions:
            diff = np.abs(aspect - direction)
            diff = np.minimum(diff, 360 - diff)
            contribution = np.exp(-(diff ** 2) / (2 * 30 ** 2))
            risk = np.maximum(risk, contribution)
        return risk
