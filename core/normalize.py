import numpy as np
from typing import Optional, Tuple


def percentile(data: np.ndarray,
               lower: float = 5,
               upper: float = 95) -> np.ndarray:
    """
    Percentile-based normalization.

    Args:
        data: Input array
        lower: Lower percentile cutoff
        upper: Upper percentile cutoff

    Returns:
        Normalized array (0-1, float32)
    """
    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros_like(data, dtype=np.float32)

    d_min, d_max = np.percentile(data[valid], [lower, upper])

    if d_max == d_min:
        result = np.full_like(data, np.nan, dtype=np.float32)
        result[valid] = 0.5
        return result

    normalized = np.clip((data - d_min) / (d_max - d_min), 0, 1)
    return normalized.astype(np.float32)


def minmax(data: np.ndarray) -> np.ndarray:
    """
    Standard min-max normalization.

    Returns:
        Normalized array (0-1, float32)
    """
    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros_like(data, dtype=np.float32)

    d_min = np.nanmin(data)
    d_max = np.nanmax(data)

    if d_max == d_min:
        result = np.full_like(data, np.nan, dtype=np.float32)
        result[valid] = 0.5
        return result

    normalized = (data - d_min) / (d_max - d_min)
    return normalized.astype(np.float32)


def rank(data: np.ndarray) -> np.ndarray:
    """
    Rank-based normalization. Each value is replaced by its rank
    among valid values, then scaled to 0-1.

    Useful for highly skewed distributions.

    Returns:
        Normalized array (0-1, float32)
    """
    result = np.full_like(data, np.nan, dtype=np.float32)
    valid = np.isfinite(data)

    if not valid.any():
        return np.zeros_like(data, dtype=np.float32)

    flat = data[valid]
    ranks = flat.argsort().argsort() + 1
    n = len(ranks)
    result[valid] = (ranks / n).astype(np.float32)

    return result


def zscore(data: np.ndarray,
           clip_sigma: float = 3.0) -> np.ndarray:
    """
    Z-score normalization, clipped to [-clip_sigma, +clip_sigma]
    then scaled to 0-1.

    Args:
        data: Input array
        clip_sigma: Number of standard deviations to clip

    Returns:
        Normalized array (0-1, float32)
    """
    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros_like(data, dtype=np.float32)

    mean = np.nanmean(data)
    std = np.nanstd(data)

    if std == 0:
        result = np.full_like(data, np.nan, dtype=np.float32)
        result[valid] = 0.5
        return result

    z = (data - mean) / std
    z = np.clip(z, -clip_sigma, clip_sigma)
    normalized = (z + clip_sigma) / (2 * clip_sigma)
    return normalized.astype(np.float32)


def invert(data: np.ndarray) -> np.ndarray:
    return (1 - data).astype(np.float32)


def classify_quantile(data: np.ndarray,
                      n_classes: int = 5) -> Tuple[np.ndarray, list]:
    """Re-exported for backwards compatibility; canonical definition lives in murasa.core.classification."""
    from .classification import classify_quantile as _classify_quantile
    return _classify_quantile(data, n_classes=n_classes)


def classify_jenks(data: np.ndarray,
                   n_classes: int = 5,
                   sample_size: int = 100000) -> Tuple[np.ndarray, list]:
    """Re-exported for backwards compatibility; canonical definition lives in murasa.core.classification."""
    from .classification import classify_jenks as _classify_jenks
    return _classify_jenks(data, n_classes=n_classes, sample_size=sample_size)
