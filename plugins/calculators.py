import numpy as np
from typing import Tuple, List
from murasa.core.base import RiskCalculatorPlugin, ParameterPlugin, register_plugin
from murasa.core.logger import logger


def _weighted_nanmean(
    components: List[Tuple[np.ndarray, float]],
    grid_shape: Tuple[int, int]
) -> np.ndarray:
    total_w = sum(w for _, w in components)
    if total_w <= 0:
        return np.nanmean([np.asarray(c, dtype=np.float32) for c, _ in components], axis=0)

    v_sum = np.zeros(grid_shape, dtype=np.float32)
    w_sum = np.zeros(grid_shape, dtype=np.float32)
    for c, w in components:
        arr = np.asarray(c, dtype=np.float32)
        valid = np.isfinite(arr)
        w_norm = float(w / total_w)
        v_sum = np.where(valid, v_sum + np.where(valid, arr, 0.0) * w_norm, v_sum)
        w_sum = np.where(valid, w_sum + w_norm, w_sum)

    out = np.full(grid_shape, np.nan, dtype=np.float32)
    return np.divide(v_sum, w_sum, out=out, where=w_sum > 0)


@register_plugin("weighted_sum")
class WeightedSumCalculator(RiskCalculatorPlugin):
    """
    Standard Weighted Linear Combination
    """

    def __init__(self):
        super().__init__("weighted_sum")

    def calculate(
        self,
        parameters: list[ParameterPlugin],
        results: dict[str, np.ndarray],
        grid_shape: Tuple[int, int]
    ) -> np.ndarray:
        total_weight = sum(p.weight for p in parameters)
        if total_weight == 0:
            raise ValueError("Total weight is zero")

        components = [
            (results[p.name], p.weight)
            for p in parameters
            if p.name in results
        ]
        if not components:
            raise ValueError("No valid risk components computed!")

        return _weighted_nanmean(components, grid_shape)


@register_plugin("bnpb")
class BNPBCalculator(RiskCalculatorPlugin):
    """
    BNPB disaster risk formula

    Risk = Hazard * (Vulnerability / Capacity).

    Not normalized to 0.0-1.0 yet. This will be fixed later.
    """

    def __init__(self):
        super().__init__("bnpb")

    def calculate(
        self,
        parameters: list[ParameterPlugin],
        results: dict[str, np.ndarray],
        grid_shape: Tuple[int, int]
    ) -> np.ndarray:
        groups = {'hazard': [], 'vulnerability': [], 'capacity': []}

        for p in parameters:
            if p.name not in results:
                continue
            data = results[p.name]
            group = getattr(p, 'group', 'hazard').lower()
            if group not in groups:
                logger.warning(f"Plugin {p.name} has unknown group '{group}', defaulting to 'hazard'")
                group = 'hazard'
            groups[group].append((data, p.weight))

        scores = {}
        for group_name, components in groups.items():
            if not components:
                logger.warning(f"No parameters for group '{group_name}', defaulting to 1.0")
                scores[group_name] = np.ones(grid_shape, dtype=np.float32)
                continue

            scores[group_name] = _weighted_nanmean(components, grid_shape)
            logger.info(f"   Group '{group_name}' calculated (parameters: {len(components)})")

        H = scores['hazard']
        V = scores['vulnerability']
        C = np.clip(scores['capacity'], 0.01, 1.0)

        risk = H * (V / C)
        return risk.astype(np.float32)