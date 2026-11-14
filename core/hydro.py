import heapq
import numpy as np
from scipy.ndimage import binary_erosion

from .terrain import calculate_slope


"""
Not really used yet.
Used as the base for computing TWI on-the-fly from a DEM.
"""


_D8_OFFSETS = (
    (-1, -1, np.sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, np.sqrt(2.0)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, np.sqrt(2.0)),
    (1, 0, 1.0),
    (1, 1, np.sqrt(2.0)),
)


def fill_sinks(dem: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Fill topographic depressions (sinks) in a DEM.

    Args:
        dem: Digital Elevation Model array.
        epsilon: Minimum elevation increment applied to filled cells to ensure
            strictly positive drainage gradients across flat areas.

    Returns:
        Hydrologically conditioned DEM as float64 with sinks filled and NaN
        cells preserved.
    """
    arr = np.asarray(dem, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D DEM array, got shape {arr.shape}")

    rows, cols = arr.shape
    if rows == 0 or cols == 0:
        return arr.copy()

    valid = np.isfinite(arr)
    if not valid.any():
        return arr.copy()

    interior = binary_erosion(
        valid,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    outlets = valid & ~interior

    dem_flat = arr.ravel().tolist()
    out = list(dem_flat)
    visited = ((~valid) | outlets).ravel().tolist()

    outlet_indices = np.flatnonzero(outlets).tolist()
    pq = [(dem_flat[idx], idx) for idx in outlet_indices]
    heapq.heapify(pq)

    heappop = heapq.heappop
    heappush = heapq.heappush

    while pq:
        z, idx = heappop(pq)
        r = idx // cols
        c = idx - r * cols
        for dr in (-1, 0, 1):
            nr = r + dr
            if nr < 0 or nr >= rows:
                continue
            nbase = nr * cols
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nc = c + dc
                if nc < 0 or nc >= cols:
                    continue
                nidx = nbase + nc
                if not visited[nidx]:
                    visited[nidx] = True
                    nz = dem_flat[nidx]
                    min_z = z + epsilon
                    if min_z > nz:
                        nz = min_z
                        out[nidx] = nz
                    heappush(pq, (nz, nidx))

    return np.array(out, dtype=np.float64).reshape(rows, cols)


def flow_direction(dem: np.ndarray) -> np.ndarray:
    """
    Compute D8 steepest-descent receiver index for each cell in a 2D DEM.

    Args:
        dem: 2D elevation array.

    Returns:
        1D int64 array of flattened downstream receiver indices (-1 for sinks/outlets/NaNs).
    """
    arr = np.asarray(dem, dtype=np.float64)
    rows, cols = arr.shape
    n = rows * cols
    valid = np.isfinite(arr)

    best_slope = np.zeros((rows, cols), dtype=np.float64)
    receiver = np.full((rows, cols), -1, dtype=np.int64)

    for dr, dc, dist in _D8_OFFSETS:
        r_src = slice(max(0, -dr), rows - max(0, dr))
        c_src = slice(max(0, -dc), cols - max(0, dc))
        r_dst = slice(max(0, dr), rows - max(0, -dr))
        c_dst = slice(max(0, dc), cols - max(0, -dc))

        src_z = arr[r_src, c_src]
        dst_z = arr[r_dst, c_dst]
        pair_valid = valid[r_src, c_src] & valid[r_dst, c_dst]

        drop = np.where(pair_valid, (src_z - dst_z) / dist, 0.0)
        better = pair_valid & (drop > best_slope[r_src, c_src])

        if better.any():
            best_slope[r_src, c_src] = np.where(better, drop, best_slope[r_src, c_src])
            dst_r, dst_c = np.nonzero(better)
            src_r_idx = dst_r + max(0, -dr)
            src_c_idx = dst_c + max(0, -dc)
            dst_flat = (dst_r + max(0, dr)) * cols + (dst_c + max(0, dc))
            receiver[src_r_idx, src_c_idx] = dst_flat

    return receiver.reshape(n)


def flow_accumulation(
    dem: np.ndarray,
    fill: bool = True,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """
    Compute D8 flow accumulation (contributing cell count, including self).

    Args:
        dem: Digital Elevation Model array.
        fill: Whether to fill sinks before computing flow routing.
        epsilon: Elevation increment passed to fill_sinks when fill=True.

    Returns:
        2D float32 array of upstream contributing cell counts (NaN outside valid DEM).
    """
    arr = np.asarray(dem, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D DEM array, got shape {arr.shape}")

    if arr.size == 0:
        return arr.astype(np.float32)

    conditioned = fill_sinks(arr, epsilon=epsilon) if fill else arr
    rows, cols = conditioned.shape
    n = rows * cols
    valid_flat = np.isfinite(conditioned).ravel()

    receiver = flow_direction(conditioned)
    valid_edges = receiver >= 0
    in_degree = np.bincount(receiver[valid_edges], minlength=n)

    acc = np.where(valid_flat, 1.0, np.nan)
    active = np.flatnonzero(valid_flat & (in_degree == 0))

    while active.size > 0:
        rcv = receiver[active]
        mask = rcv >= 0
        if not mask.any():
            break

        src_cells = active[mask]
        dst_cells = rcv[mask]
        np.add.at(acc, dst_cells, acc[src_cells])
        np.subtract.at(in_degree, dst_cells, 1)

        candidates = np.unique(dst_cells)
        active = candidates[in_degree[candidates] == 0]

    return acc.reshape(rows, cols).astype(np.float32)


def calculate_twi(
    dem: np.ndarray,
    cell_size: float = 1.0,
    fill: bool = True,
    min_slope: float = 1e-4, # Preventing division by zero.
    epsilon: float = 1e-5,
) -> np.ndarray:
    """
    Compute Topographic Wetness Index

    ln(a / tan(beta)).

    Args:
        dem: Digital Elevation Model array.
        cell_size: Grid cell resolution in meters.
        fill: Whether to fill sinks before computing flow accumulation and slope.
        min_slope: Minimum tan(slope) clamp to prevent division by zero on flats.
        epsilon: Elevation increment passed to fill_sinks when fill=True.

    Returns:
        2D float32 TWI raster (NaN outside valid DEM).
    """
    arr = np.asarray(dem, dtype=np.float64)
    conditioned = fill_sinks(arr, epsilon=epsilon) if fill else arr
    acc = flow_accumulation(conditioned, fill=False).astype(np.float64)
    specific_area = acc * float(cell_size)

    slope_rad = calculate_slope(conditioned, cell_size=cell_size, degrees=False)
    tan_slope = np.maximum(np.tan(slope_rad), min_slope)

    twi = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(arr) & np.isfinite(specific_area) & np.isfinite(tan_slope)
    twi[valid] = np.log(specific_area[valid] / tan_slope[valid]).astype(np.float32)
    return twi
