from typing import Any, Optional, Tuple
import numpy as np
import pyproj
from scipy.ndimage import distance_transform_edt

from .data_source import DataSource
from .resampler import RasterResampler, VectorResampler

_WGS84_GEOD = pyproj.Geod(ellps="WGS84")


def _parse_crs(crs: Any) -> Optional[pyproj.CRS]:
    """Parse a CRS input into a pyproj.CRS object, returning None if invalid/empty."""
    if not crs:
        return None
    try:
        return pyproj.CRS.from_user_input(crs)
    except Exception:
        return None


def crs_equal(crs_a: Any, crs_b: Any) -> bool:
    """Return True if two CRS representations refer to the same coordinate reference system."""
    if not crs_a or not crs_b:
        return False
    parsed_a, parsed_b = _parse_crs(crs_a), _parse_crs(crs_b)
    if parsed_a is not None and parsed_b is not None:
        return parsed_a == parsed_b
    return str(crs_a) == str(crs_b)


def is_geographic_crs(crs: Any) -> bool:
    """Return True if the given CRS uses geographic (degree) coordinates."""
    crs_obj = _parse_crs(crs)
    return bool(crs_obj.is_geographic) if crs_obj is not None else False


def resolve_metric_crs(
    gdf: Any = None,
    metric_crs: Optional[str] = "auto",
    fallback: str = "EPSG:6933",
) -> str:
    """
    Resolve a projected metric CRS for area/distance operations anywhere on Earth.

    If metric_crs is 'auto' or None, automatically estimates the local UTM zone
    from the input GeoDataFrame/GeoSeries bounds, with EPSG:6933 as a fallback if
    estimation is unavailable.
    """
    if metric_crs and str(metric_crs).strip().lower() not in ("auto", "none", ""):
        return str(metric_crs)

    if gdf is not None and hasattr(gdf, "crs") and gdf.crs is not None:
        crs_obj = _parse_crs(gdf.crs)
        if crs_obj is not None and crs_obj.is_projected:
            return str(gdf.crs)

        if not getattr(gdf, "empty", True) and hasattr(gdf, "estimate_utm_crs"):
            try:
                utm_crs = gdf.estimate_utm_crs()
                if utm_crs is not None:
                    return utm_crs.to_string()
            except Exception:
                pass

    return fallback


def _projected_unit_factors(crs_obj: pyproj.CRS) -> Tuple[float, float]:
    """Return (fx, fy) linear unit conversion factors to meters for a projected CRS."""
    if crs_obj.is_projected and crs_obj.axis_info:
        fx = float(crs_obj.axis_info[0].unit_conversion_factor or 1.0)
        fy = float(crs_obj.axis_info[1].unit_conversion_factor or fx) if len(crs_obj.axis_info) > 1 else fx
        return fx, fy
    return 1.0, 1.0


def _meters_per_degree(
    lat_deg: float,
    geod: Optional[pyproj.Geod] = None,
) -> Tuple[float, float]:
    """Compute (meters_per_deg_lon, meters_per_deg_lat) at a given latitude."""
    g = geod or _WGS84_GEOD
    lat = float(np.clip(lat_deg if np.isfinite(lat_deg) else 0.0, -89.9, 89.9))
    _, _, dist_x = g.inv(-0.0005, lat, 0.0005, lat)
    _, _, dist_y = g.inv(0.0, max(-89.999, lat - 0.0005), 0.0, min(89.999, lat + 0.0005))
    return float(abs(dist_x) * 1000.0), float(abs(dist_y) * 1000.0)


def _geod_for_crs(crs_obj: pyproj.CRS) -> pyproj.Geod:
    """Return a pyproj.Geod matching the CRS ellipsoid, falling back to WGS84."""
    try:
        geod = crs_obj.get_geod()
        if geod is not None:
            return geod
    except Exception:
        pass
    return _WGS84_GEOD


def _extract_affine_metrics(transform: Any) -> Tuple[float, float, float, float, float]:
    """Extract (dx_raw, dy_raw, y0, d, e) from an Affine object or tuple."""
    if hasattr(transform, "a"):
        dx_raw = float(np.hypot(transform.a, getattr(transform, "d", 0.0)))
        dy_raw = float(np.hypot(getattr(transform, "b", 0.0), transform.e))
        return dx_raw, dy_raw, float(transform.f), float(getattr(transform, "d", 0.0)), float(transform.e)

    dx_raw = float(abs(transform[0]))
    dy_raw = float(abs(transform[4])) if len(transform) > 4 else dx_raw
    y0 = float(transform[5]) if len(transform) > 5 else 0.0
    d = float(transform[3]) if len(transform) > 3 else 0.0
    e = float(transform[4]) if len(transform) > 4 else -dy_raw
    return dx_raw, dy_raw, y0, d, e


def cell_size_in_meters(
    transform: Any,
    grid_shape: Optional[Tuple[int, int]] = None,
    crs: Any = "EPSG:4326",
) -> Tuple[float, float]:
    """
    Compute (dx_m, dy_m) pixel resolution in meters for any projected or geographic CRS.
    """
    dx_raw, dy_raw, y0, d, e = _extract_affine_metrics(transform)
    crs_obj = _parse_crs(crs)
    if crs_obj is None:
        return dx_raw, dy_raw

    if crs_obj.is_geographic:
        rows = int(grid_shape[0]) if grid_shape and len(grid_shape) >= 2 else 0
        cols = int(grid_shape[1]) if grid_shape and len(grid_shape) >= 2 else 0
        lat_mid = y0 + d * (cols / 2.0) + e * (rows / 2.0)
        if not np.isfinite(lat_mid) or abs(lat_mid) > 90.0:
            lat_mid = 0.0
        m_per_deg_x, m_per_deg_y = _meters_per_degree(lat_mid, geod=_geod_for_crs(crs_obj))
        return dx_raw * m_per_deg_x, dy_raw * m_per_deg_y

    if crs_obj.is_projected:
        fx, fy = _projected_unit_factors(crs_obj)
        return dx_raw * fx, dy_raw * fy

    return dx_raw, dy_raw


def meters_to_crs_units(
    resolution_m: float,
    crs: Any = "EPSG:4326",
    bounds: Optional[Tuple[float, float, float, float]] = None,
    lat_mid: float = 0.0,
) -> Tuple[float, float]:
    """
    Convert a resolution in meters into (res_x, res_y) in the target CRS units.
    """
    res = float(resolution_m)
    if not np.isfinite(res) or res <= 0:
        raise ValueError(f"Resolution in meters must be positive, got {resolution_m}")

    crs_obj = _parse_crs(crs)
    if crs_obj is None:
        return res, res

    if crs_obj.is_geographic:
        if bounds is not None and len(bounds) >= 4:
            lat_mid = (float(bounds[1]) + float(bounds[3])) / 2.0
        m_per_deg_x, m_per_deg_y = _meters_per_degree(lat_mid, geod=_geod_for_crs(crs_obj))
        return res / m_per_deg_x, res / m_per_deg_y

    if crs_obj.is_projected:
        fx, fy = _projected_unit_factors(crs_obj)
        return res / fx, res / fy

    return res, res


class ProcessingContext:
    """
    Usage in a plugin:
        ctx = ProcessingContext(grid_shape, transform, crs)
        dem = ctx.resample_raster(registry.get('dem'))
        risk = ctx.distance_to_features(registry.get('river'))
    """

    def __init__(self, grid_shape, transform, crs):
        self.grid_shape = grid_shape
        self.transform = transform
        self.crs = crs
        self.is_geographic = is_geographic_crs(crs)
        self.crs_resolution = abs(transform.a)
        self.dx_m, self.dy_m = cell_size_in_meters(transform, grid_shape, crs)
        self.cell_size_m = (self.dy_m, self.dx_m)
        self.resolution_m = (self.dx_m + self.dy_m) / 2.0
        self.resolution = self.resolution_m

    def resample_raster(self, source: DataSource,
                        method: str = "bilinear") -> np.ndarray:
        """
        Resample a raster DataSource to the analysis grid.

        Args:
            source: DataSource with raster data
            method: Resampling method (bilinear, nearest, cubic, etc.)

        Returns:
            Resampled array matching grid_shape
        """
        resampler = RasterResampler(
            self.grid_shape, self.transform, self.crs, method=method
        )
        return resampler.resample(source)

    def rasterize_vector(self, source: DataSource,
                         column: str = None,
                         all_touched: bool = True,
                         fill_value: float = 0.0) -> np.ndarray:
        """
        Rasterize a vector DataSource to the analysis grid.

        Args:
            source: DataSource with vector data
            column: Column name to use for burn-in values
            all_touched: Whether to rasterize all touched pixels
            fill_value: Fill value for empty pixels

        Returns:
            Rasterized array matching grid_shape
        """
        resampler = VectorResampler(
            self.grid_shape, self.transform, self.crs
        )
        return resampler.resample(
            source,
            value_column=column,
            all_touched=all_touched,
            fill_value=fill_value
        )

    def empty_grid(self, fill: float = 0.0,
                   dtype=np.float32) -> np.ndarray:
        return np.full(self.grid_shape, fill, dtype=dtype)

    def boolean_grid(self, fill: bool = False) -> np.ndarray:
        return np.full(self.grid_shape, fill, dtype=bool)

    def distance_to_features(self, source: DataSource,
                             max_distance: Optional[float] = None) -> np.ndarray:
        """
        Compute Euclidean distance in meters from each pixel to the nearest vector feature.

        Args:
            source: DataSource with vector geometry
            max_distance: Optional distance cutoff in meters. If set, normalizes output to
                          0-1 (close=1, far=0).

        Returns:
            Distance raster in meters (or normalized 0-1 if max_distance is set).
        """
        if max_distance is not None and (not np.isfinite(max_distance) or max_distance <= 0):
            raise ValueError(f"max_distance must be positive, got {max_distance}")

        feature_mask = self.rasterize_vector(source).astype(bool)
        if not feature_mask.any():
            if max_distance is not None:
                return self.empty_grid(0.0)
            return np.full(self.grid_shape, np.inf, dtype=np.float32)

        dist_units = distance_transform_edt(
            ~feature_mask,
            sampling=(self.dy_m, self.dx_m),
        )

        if max_distance is not None:
            return np.clip(1 - (dist_units / max_distance), 0, 1).astype(np.float32)

        return dist_units.astype(np.float32)
