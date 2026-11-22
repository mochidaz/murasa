from typing import Optional
import numpy as np

from murasa.core.base import RasterParameterPlugin, register_plugin
from murasa.core.processing import ProcessingContext
from murasa.core.data_source import DataRegistry
from murasa.core.hydro import calculate_hand
from murasa.core.logger import logger


@register_plugin("hand")
class HANDPlugin(RasterParameterPlugin):
    """
    Height Above Nearest Drainage (HAND) risk parameter.
    """

    source_keys = ["hand"]
    inverse = True
    normalization = "exponential"
    short_name = "hand"
    display_name = "HAND"

    def __init__(
        self,
        name: str = "hand",
        weight: float = 0.16,
        group: str = "hazard",
        stream_threshold: Optional[float] = None,
        stream_area_km2: float = 0.25,
        stream_source: Optional[str] = None,
        fill: bool = True,
        scale_height: float = 10.0,
        max_hand: Optional[float] = None,
        **kwargs
    ):
        super().__init__(name=name, weight=weight, group=group, **kwargs)
        self.stream_threshold = stream_threshold
        self.stream_area_km2 = float(stream_area_km2)
        self.stream_source = stream_source
        self.fill = bool(fill)
        self.scale_height = float(scale_height)
        self.max_hand = float(max_hand) if max_hand is not None else None

    def validate_requirements(self, registry: DataRegistry) -> bool:
        return registry.has("hand") or registry.has("dem")

    def _effective_stream_threshold(self, cell_size_m) -> float:
        if self.stream_threshold is not None:
            return float(self.stream_threshold)
        if isinstance(cell_size_m, (tuple, list, np.ndarray)):
            cell_area_m2 = max(abs(float(cell_size_m[0]) * float(cell_size_m[1])), 1.0)
        else:
            cell_area_m2 = max(float(cell_size_m) ** 2, 1.0)
        return max(10.0, (self.stream_area_km2 * 1_000_000.0) / cell_area_m2)

    def _normalize_hand(self, hand: np.ndarray) -> np.ndarray:
        valid = np.isfinite(hand)
        if not valid.any():
            return np.full_like(hand, np.nan, dtype=np.float32)

        if self.normalization == "exponential":
            scale = max(self.scale_height, 1e-3)
            scores = np.full_like(hand, np.nan, dtype=np.float32)
            vals = np.exp(-np.maximum(hand[valid], 0.0) / scale)
            scores[valid] = (1.0 - vals if not self.inverse else vals).astype(np.float32)
            return scores

        if self.normalization == "linear_cap" or self.max_hand is not None:
            cap = max(self.max_hand or 25.0, 1e-3)
            scores = np.full_like(hand, np.nan, dtype=np.float32)
            vals = np.clip(hand[valid] / cap, 0.0, 1.0)
            scores[valid] = (1.0 - vals if self.inverse else vals).astype(np.float32)
            return scores

        normalized = self._normalize(hand)
        if self.inverse:
            normalized = 1.0 - normalized
        return normalized.astype(np.float32)

    def process(self, registry: DataRegistry, grid_shape, transform, crs):
        self.log_processing()
        ctx = ProcessingContext(grid_shape, transform, crs)

        if registry.has("hand"):
            hand = ctx.resample_raster(registry.get("hand"))
        else:
            dem = ctx.resample_raster(registry.get("dem"))
            stream_mask = None
            if self.stream_source and registry.has(self.stream_source):
                stream_src = registry.get(self.stream_source)
                stream_mask = ctx.rasterize_vector(stream_src).astype(bool)

            thresh = self._effective_stream_threshold(ctx.cell_size_m)
            logger.info(
                f"      Computing HAND from DEM (fill={self.fill}, stream_threshold={thresh:.0f} cells)..."
            )
            hand = calculate_hand(
                dem,
                stream_mask=stream_mask,
                stream_threshold=thresh,
                fill=self.fill,
            )

        hand = registry.apply_study_area_mask(hand)
        valid = np.isfinite(hand)
        if valid.any():
            logger.info(
                f"      Raw HAND range: {float(np.nanmin(hand)):.2f} m to {float(np.nanmax(hand)):.2f} m "
                f"(median={float(np.nanmedian(hand)):.2f} m)"
            )

        self.result = self._normalize_hand(hand)
        self.log_result()
        return self.result
