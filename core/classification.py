from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _apply_breaks(
    data: np.ndarray,
    valid: np.ndarray,
    breaks: Sequence[float],
) -> Tuple[np.ndarray, List[float]]:

    classified = np.zeros_like(data, dtype=np.uint8)
    break_list = sorted(float(b) for b in breaks)
    if not valid.any() or len(break_list) < 2:
        return classified, []

    if break_list[0] == break_list[-1]:
        classified[valid] = 1
        return classified, break_list

    n_intervals = len(break_list) - 1
    for i in range(n_intervals):
        if i == 0:
            mask = valid & (data < break_list[1])
        elif i < n_intervals - 1:
            mask = valid & (data >= break_list[i]) & (data < break_list[i + 1])
        else:
            mask = valid & (data >= break_list[-2])
        classified[mask] = i + 1

    return classified, break_list


def classify_jenks(
    data: np.ndarray,
    n_classes: int = 5,
    sample_size: int = 100000,
) -> Tuple[np.ndarray, List[float]]:
    """
    Classify using Jenks Natural Breaks (falling back to quantiles if jenkspy is unavailable).

    Returns:
        (classified array (1-based integers, 0 for non-finite), break values)
    """
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")

    try:
        import jenkspy
    except ImportError:
        return classify_quantile(data, n_classes)

    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros_like(data, dtype=np.uint8), []

    values = data[valid]
    if values.size > sample_size:
        rng = np.random.default_rng(42)
        sample = rng.choice(values, sample_size, replace=False)
    else:
        sample = values

    n_unique = int(np.unique(sample).size)
    if n_unique <= 1 or n_classes == 1:
        v = float(values[0])
        return _apply_breaks(data, valid, [v, v])

    n_eff = min(n_classes, n_unique - 1)
    if n_eff < 2:
        return classify_quantile(data, n_classes)

    breaks = jenkspy.jenks_breaks(sample, n_classes=n_eff)
    return _apply_breaks(data, valid, breaks)


def classify_quantile(
    data: np.ndarray,
    n_classes: int = 5,
) -> Tuple[np.ndarray, List[float]]:
    """
    Classify using quantiles.

    Returns:
        (classified array (1-based integers, 0 for non-finite), break values)
    """
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")

    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros_like(data, dtype=np.uint8), []

    values = data[valid]
    breaks = np.percentile(values, np.linspace(0, 100, n_classes + 1))
    return _apply_breaks(data, valid, breaks.tolist())


def classify_equal_interval(
    data: np.ndarray,
    n_classes: int = 5,
) -> Tuple[np.ndarray, List[float]]:
    """
    Classify using equal intervals between min and max valid values.

    Returns:
        (classified array (1-based integers, 0 for non-finite), break values)
    """
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")

    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros_like(data, dtype=np.uint8), []

    values = data[valid]
    min_val, max_val = float(np.min(values)), float(np.max(values))
    breaks = np.linspace(min_val, max_val, n_classes + 1)
    return _apply_breaks(data, valid, breaks.tolist())


def classify_std_dev(
    data: np.ndarray,
    n_classes: int = 5,
    interval_sigma: float = 1.0,
) -> Tuple[np.ndarray, List[float]]:
    """
    Classify using symmetric standard deviation intervals centered on the mean.

    Returns:
        (classified array (1-based integers, 0 for non-finite), break values)
    """
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")

    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros_like(data, dtype=np.uint8), []

    values = data[valid]
    mean = float(np.mean(values))
    std_dev = float(np.std(values))

    offsets = np.linspace(-n_classes / 2.0, n_classes / 2.0, n_classes + 1)
    breaks = [mean + float(offset) * interval_sigma * std_dev for offset in offsets]
    return _apply_breaks(data, valid, breaks)


def classify_geometric(
    data: np.ndarray,
    n_classes: int = 5,
) -> Tuple[np.ndarray, List[float]]:
    """
    Classify using geometric intervals (handling zero/non-positive values safely).

    Returns:
        (classified array (1-based integers, 0 for non-finite), break values)
    """
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")

    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros_like(data, dtype=np.uint8), []

    values = data[valid]
    min_val, max_val = float(np.min(values)), float(np.max(values))
    if min_val == max_val:
        return _apply_breaks(data, valid, [min_val] * (n_classes + 1))

    if min_val > 0.0:
        breaks = np.geomspace(min_val, max_val, n_classes + 1).tolist()
    else:
        shift = -min_val + max((max_val - min_val) * 1e-4, 1e-6)
        shifted = np.geomspace(min_val + shift, max_val + shift, n_classes + 1) - shift
        breaks = shifted.tolist()
        breaks[0] = min_val
        breaks[-1] = max_val

    return _apply_breaks(data, valid, breaks)


def classify_manual(
    data: np.ndarray,
    thresholds: Sequence[float],
    n_classes: Optional[int] = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Classify using manual thresholds (either full break edges or internal cutpoints).

    Returns:
        (classified array (1-based integers, 0 for non-finite), break values)
    """
    if not thresholds:
        raise ValueError("Thresholds must be provided for manual classification.")

    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros_like(data, dtype=np.uint8), []

    breaks = sorted(float(t) for t in thresholds)
    if len(breaks) == 1 or (n_classes is not None and len(breaks) == n_classes - 1):
        values = data[valid]
        low = min(float(np.min(values)), breaks[0])
        high = max(float(np.max(values)), breaks[-1])
        breaks = [low, *breaks, high]

    return _apply_breaks(data, valid, breaks)


CLASSIFIERS: Dict[str, Callable[..., Tuple[np.ndarray, List[float]]]] = {
    "jenks": classify_jenks,
    "natural_breaks": classify_jenks,
    "quantile": classify_quantile,
    "equal_interval": classify_equal_interval,
    "std_dev": classify_std_dev,
    "geometric": classify_geometric,
    "manual": classify_manual,
    "thresholds": classify_manual,
}


def classify(
    data: np.ndarray,
    method: str = "jenks",
    num_classes: int = 5,
    thresholds: Optional[Sequence[float]] = None,
    **kwargs,
) -> Tuple[np.ndarray, List[float]]:
    """
    Classify data using the specified method.

    Args:
        data: The input array to classify.
        method: Classification method name.
        num_classes: Number of output classes.
        thresholds: Threshold breakpoints required when method is 'manual' or 'thresholds'.

    Returns:
        A tuple containing the 1-based classified uint8 array (0 for non-finite)
        and the break values list.
    """
    key = str(method).strip().lower()
    if key not in CLASSIFIERS:
        raise ValueError(f"Unknown classification method: {method}")

    if key in ("manual", "thresholds"):
        if not thresholds:
            raise ValueError("Thresholds must be provided for manual classification.")
        return classify_manual(data, thresholds=thresholds, n_classes=num_classes)

    return CLASSIFIERS[key](data, n_classes=num_classes, **kwargs)
