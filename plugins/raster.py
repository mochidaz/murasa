from typing import Optional
import numpy as np

from murasa.core.base import RasterParameterPlugin, register_plugin


@register_plugin("raster")
class GenericRasterPlugin(RasterParameterPlugin):
    """
    Generic raster-based risk parameter plugin.

    Intended for usage via YAML plugin.
    """

    source_keys = []
    inverse = False
    normalization = "percentile"

    def __init__(
        self,
        name: str = "raster",
        weight: float = 1.0,
        group: str = "hazard",
        scale: float = 1.0,
        offset: float = 0.0,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        **kwargs
    ):
        super().__init__(name=name, weight=weight, group=group, **kwargs)
        self.scale = float(scale)
        self.offset = float(offset)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def transform_data(self, data: np.ndarray, ctx) -> np.ndarray:
        out = data
        if self.scale != 1.0 or self.offset != 0.0:
            out = out * self.scale + self.offset
        if self.clip_min is not None or self.clip_max is not None:
            out = np.clip(out, self.clip_min, self.clip_max)
        return out
