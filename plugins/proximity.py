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
        feature_name: str = "river",
        feature: str = None,
        max_distance: float = 500,
        **kwargs
    ):
        super().__init__(name, weight, group, **kwargs)
        self.feature_name = feature or feature_name
        self.max_distance = max_distance

    def validate_requirements(self, registry):
        return registry.has(self.feature_name)

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()
        ctx = ProcessingContext(grid_shape, transform, crs)

        feature_source = registry.get(self.feature_name)
        if feature_source.data.empty:
            logger.warning(f"      No {self.feature_name} features found")
            return ctx.empty_grid()

        dist_score = ctx.distance_to_features(
            feature_source,
            max_distance=self.max_distance
        )
        self.result = registry.apply_study_area_mask(dist_score)

        self.log_result()
        return self.result
