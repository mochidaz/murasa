from pathlib import Path
from typing import List, Optional, Sequence
import re

import geopandas as gpd
import pandas as pd

from .logger import logger, log_warning


def scan_shapefiles(directories: List[Path],
                    patterns: List[str],
                    target_crs: str = "EPSG:4326",
                    prefixes: Optional[Sequence[str]] = None) -> gpd.GeoDataFrame:
    """
    Scan directories for shapefiles matching pattern names.
    """
    frames = []

    for directory in directories:
        directory = Path(directory)
        if not directory.exists():
            continue

        for shp_file in directory.glob("*.shp"):
            if match_pattern(shp_file.name, patterns, prefixes=prefixes):
                try:
                    gdf = gpd.read_file(shp_file).to_crs(target_crs)
                    frames.append(gdf)
                    logger.debug(f"  Found: {shp_file.name} ({len(gdf)} features)")
                except Exception as e:
                    log_warning(f"Failed to load SHP {shp_file.name}: {e}")

    return _merge_frames(frames, target_crs)


def scan_gdb_layers(directories: List[Path],
                    patterns: List[str],
                    target_crs: str = "EPSG:4326",
                    prefixes: Optional[Sequence[str]] = None) -> gpd.GeoDataFrame:
    """
    Scan directories for GDB layers matching pattern names.
    """
    try:
        import fiona
    except ImportError:
        log_warning("fiona not installed, cannot scan GDB files")
        return gpd.GeoDataFrame()

    frames = []

    for directory in directories:
        directory = Path(directory)
        if not directory.exists():
            continue

        for gdb_dir in directory.glob("*.gdb"):
            try:
                layers = fiona.listlayers(str(gdb_dir))
                for layer in layers:
                    if match_pattern(layer, patterns, prefixes=prefixes):
                        try:
                            gdf = gpd.read_file(gdb_dir, layer=layer).to_crs(target_crs)
                            frames.append(gdf)
                            logger.debug(
                                f"  Found: {gdb_dir.name}/{layer} ({len(gdf)} features)"
                            )
                        except Exception as e:
                            log_warning(f"Failed to load GDB layer {layer}: {e}")
            except Exception as e:
                log_warning(f"Failed to read GDB {gdb_dir.name}: {e}")

    return _merge_frames(frames, target_crs)


def scan_all(directories: List[Path],
             patterns: List[str],
             target_crs: str = "EPSG:4326",
             prefixes: Optional[Sequence[str]] = None) -> gpd.GeoDataFrame:
    """
    Scan directories for both shapefiles and GDB layers matching pattern names.
    """
    gdf_shp = scan_shapefiles(directories, patterns, target_crs, prefixes=prefixes)
    gdf_gdb = scan_gdb_layers(directories, patterns, target_crs, prefixes=prefixes)

    frames = []
    if not gdf_shp.empty:
        frames.append(gdf_shp)
    if not gdf_gdb.empty:
        frames.append(gdf_gdb)

    return _merge_frames(frames, target_crs)


def scan_filename(directories: List[Path],
                  filename: str,
                  target_crs: str = "EPSG:4326",
                  validate_geometries: bool = True) -> gpd.GeoDataFrame:
    """
    Scan directories for a specific shapefile or GDB layer by filename.
    """
    frames = []

    for directory in directories:
        directory = Path(directory)
        file_path = directory / filename

        if file_path.exists():
            try:
                gdf = gpd.read_file(file_path).to_crs(target_crs)

                if validate_geometries:
                    invalid = ~gdf.geometry.is_valid
                    if invalid.any():
                        gdf.loc[invalid, 'geometry'] = gdf.loc[invalid].geometry.buffer(0)
                        logger.debug(f"  Fixed {invalid.sum()} invalid geometries in {file_path.name}")

                frames.append(gdf)
                logger.debug(f"  Found: {file_path.name} ({len(gdf)} features)")
            except Exception as e:
                log_warning(f"Failed to load {file_path.name}: {e}")

    return _merge_frames(frames, target_crs)


def match_pattern(
    name: str,
    patterns: Sequence[str],
    prefixes: Optional[Sequence[str]] = None,
) -> bool:
    """
    Check if a filename stem or layer name matches any pattern at a token boundary.
    """
    if not patterns:
        return False

    stem = Path(name).stem.upper()
    prefix_expr = ""
    if prefixes:
        ordered = sorted({pr.upper() for pr in prefixes if pr}, key=len, reverse=True)
        if ordered:
            prefix_expr = f"(?:{'|'.join(re.escape(pr) for pr in ordered)})?"

    pattern_expr = "|".join(re.escape(p.upper()) for p in patterns if p)
    if not pattern_expr:
        return False

    regex = re.compile(rf"(?:^|[^A-Z0-9]){prefix_expr}(?:{pattern_expr})")
    return regex.search(stem) is not None


_match_pattern = match_pattern


def _merge_frames(frames: list, target_crs: str) -> gpd.GeoDataFrame:
    """Merge list of GeoDataFrames into one."""
    if not frames:
        return gpd.GeoDataFrame()

    if len(frames) == 1:
        return frames[0]

    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        crs=target_crs
    )
