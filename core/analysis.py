import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from rasterstats import zonal_stats
from typing import List, Dict, Any

from .logger import logger
from .data_source import DataSource


def calculate_zonal_stats(
        vector_gdf: GeoDataFrame,
        raster_source: DataSource,
        stats: List[str] = ['mean', 'count'],
        nodata: float = np.nan,
        categorical: bool = False,
        transform=None,
        crs=None
) -> List[Dict[str, Any]]:
    """
    Calculate zonal statistics for vector features against a raster source.
    """
    if hasattr(raster_source, 'data'):
        if raster_source.data is None:
            logger.warning(f"DataSource {getattr(raster_source, 'name', 'unknown')} has no data loaded.")
            return []

        data = np.asanyarray(raster_source.data)
        if hasattr(raster_source, 'metadata'):
            transform = transform or raster_source.metadata.get('transform')
            crs = crs or raster_source.crs
    elif isinstance(raster_source, np.ndarray):
        data = raster_source
    else:
        try:
            data = np.array(raster_source)
        except Exception as e:
            logger.error(f"Invalid raster source type: {type(raster_source)}. Error: {e}")
            return []

    if transform is None or crs is None:
        raise ValueError("Transform and CRS required for array-based zonal stats")

    try:
        gdf_target = (
            vector_gdf.to_crs(crs)
            if getattr(vector_gdf, "crs", None) is not None and vector_gdf.crs != crs
            else vector_gdf
        )
        return zonal_stats(
            gdf_target,
            data,
            affine=transform,
            stats=stats,
            categorical=categorical,
            nodata=nodata
        )
    except Exception as e:
        logger.error(f"Zonal stats failed: {e}")
        return []


def compute_gvf(values: np.ndarray, labels: np.ndarray) -> float:
    """Compute Goodness of Variance Fit (GVF) for discrete class labels in [0, 1]."""
    vals = np.asarray(values, dtype=float)
    labs = np.asarray(labels)

    mask = np.isfinite(vals)
    if mask.sum() < 2:
        return 0.0

    vals = vals[mask]
    labs = labs[mask]

    sdam = np.sum((vals - vals.mean()) ** 2)
    if sdam == 0:
        return 1.0

    sdcm = 0.0
    for lab in np.unique(labs):
        group = vals[labs == lab]
        if group.size > 0:
            sdcm += np.sum((group - group.mean()) ** 2)

    return float(max(0.0, min(1.0, (sdam - sdcm) / sdam)))


def compute_gvf_with_breaks(values: np.ndarray, breaks: List[float]) -> float:
    """Compute Goodness of Variance Fit (GVF) given classification break intervals."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size < 2 or len(breaks) < 2:
        return 0.0

    sdam = np.sum((vals - vals.mean()) ** 2)
    if sdam == 0:
        return 1.0

    sdcm = 0.0
    for i in range(1, len(breaks)):
        if i == len(breaks) - 1:
            class_mask = vals >= breaks[i - 1]
        else:
            class_mask = (vals >= breaks[i - 1]) & (vals < breaks[i])

        class_values = vals[class_mask]
        if class_values.size > 0:
            sdcm += np.sum((class_values - class_values.mean()) ** 2)

    return float(max(0.0, min(1.0, 1.0 - (sdcm / sdam))))


def compute_spearman(a, b) -> float:
    """Compute Spearman rank correlation between two array-like sequences."""
    try:
        sa = pd.Series(a, dtype=float).rank(method='average')
        sb = pd.Series(b, dtype=float).rank(method='average')
        df = pd.concat([sa, sb], axis=1).dropna()
        if df.shape[0] < 2:
            return 0.0
        rho = df.iloc[:, 0].corr(df.iloc[:, 1], method='pearson')
        return float(rho) if pd.notnull(rho) else 0.0
    except Exception:
        return 0.0


def compute_weight_sensitivity(
    factor_scores: Dict[str, Any],
    weights: Dict[str, float],
    perturbation: float = 0.2,
    stability_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Perform One-at-a-Time (OAT) weight perturbation analysis on weighted composite scores.
    """
    active_pairs = [
        (key, pd.Series(factor_scores[key], dtype=float).fillna(0.0), float(weights[key]))
        for key in factor_scores
        if key in weights and float(weights[key]) > 0
    ]
    if not active_pairs:
        return pd.DataFrame(columns=['factor', 'stability_index', 'impact'])

    total_base = sum(w for _, _, w in active_pairs)
    baseline_score = sum(
        series * (w / total_base if total_base > 0 else 0.0)
        for _, series, w in active_pairs
    )

    rows = []
    for focus_key, _, _ in active_pairs:
        perturbed = [
            (series, w * (1.0 + perturbation) if key == focus_key else w)
            for key, series, w in active_pairs
        ]
        total_w = sum(w for _, w in perturbed)
        new_score = sum(
            series * (w / total_w if total_w > 0 else 0.0)
            for series, w in perturbed
        )
        rho = compute_spearman(baseline_score, new_score)
        rows.append({
            'factor': focus_key.replace('_', ' ').title(),
            'stability_index': round(rho, 4),
            'impact': 'Low' if rho >= stability_threshold else 'High',
        })

    return pd.DataFrame(rows)
