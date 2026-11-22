from typing import Optional
from murasa.core.base import ParameterPlugin, register_plugin
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger


@register_plugin("proximity")
class ProximityPlugin(ParameterPlugin):
    """
    Distance-based proximity risk parameter like distance to rivers.
    Closer proximity yields higher risk scores.
    """

    short_name = "prox"
    display_name = "Proximity"

    def __init__(
        self,
        name: str = "proximity",
        weight: float = 0.10,
        group: str = "hazard",
        feature_name: Optional[str] = None,
        feature: Optional[str] = None,
        source: Optional[str] = None,
        max_distance: float = 500,
        **kwargs
    ):
        super().__init__(name, weight, group, **kwargs)
        self._explicit_feature = feature or feature_name or source
        self.feature_name = self._explicit_feature or "river"
        self.max_distance = max_distance

    def _resolve_feature_key(self, registry) -> str:
        if self._explicit_feature:
            return self._explicit_feature
        if self.name and registry.has(self.name):
            return self.name
        return self.feature_name

    def validate_requirements(self, registry):
        return registry.has(self._resolve_feature_key(registry))

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()
        ctx = ProcessingContext(grid_shape, transform, crs)

        feature_key = self._resolve_feature_key(registry)
        feature_source = registry.get(feature_key)
        if feature_source.data.empty:
            logger.warning(f"      No {feature_key} features found")
            return ctx.empty_grid()

        dist_score = ctx.distance_to_features(
            feature_source,
            max_distance=self.max_distance
        )
        self.result = registry.apply_study_area_mask(dist_score)

        self.log_result()
        return self.result
