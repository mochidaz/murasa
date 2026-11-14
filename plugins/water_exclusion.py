import numpy as np
from typing import List, Optional

from murasa.core.base import ParameterPlugin, register_plugin
from murasa.core.data_source import DataRegistry
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger


@register_plugin("water_exclusion")
@register_plugin("exclusion")
class WaterExclusionPlugin(ParameterPlugin):
    """
    Exclusion plugin that masks out permanent water bodies (or configured vector features)
    from the final risk raster.
    """

    def __init__(
        self,
        name: str = "water_exclusion",
        features: Optional[List[str]] = None,
        **kwargs
    ):
        kwargs.setdefault("weight", 0.0)
        super().__init__(name, **kwargs)
        self.is_exclusion = True
        self.features = features or ["river", "reservoir"]

    def validate_requirements(self, registry: DataRegistry) -> bool:
        return True

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()

        exclusion_mask = np.zeros(grid_shape, dtype=bool)
        ctx = ProcessingContext(grid_shape, transform, crs)

        for feat_key in self.features:
            if registry.has(feat_key):
                feat_source = registry.get(feat_key)
                if not feat_source.data.empty:
                    feat_mask = ctx.rasterize_vector(feat_source).astype(bool)
                    exclusion_mask |= feat_mask
                    logger.info(f"      Excluded {feat_key}: {feat_mask.sum()} pixels")

        self.result = exclusion_mask.astype(np.float32)
        return self.result
