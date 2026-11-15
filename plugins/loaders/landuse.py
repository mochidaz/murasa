from pathlib import Path
from typing import Dict, List
import geopandas as gpd
import fiona
import pandas as pd

from murasa.core.base import DataLoaderPlugin
from murasa.core.data_source import DataRegistry, DataSource, SourceType
from murasa.core.file_scanner import match_pattern
from murasa.core.logger import log_success, log_warning


class LandUseLoader(DataLoaderPlugin):
    """
    Loads and standardizes land use data from RBI.
    """

    provides = [
        "landuse_settlement", "landuse_forest", "landuse_shrub",
        "landuse_plantation", "landuse_paddy", "landuse_water", "landuse_road"
    ]
    requires = []

    DEFAULT_PATTERNS = {
        'settlement': [
            'PEMUKIMAN_AR', 'BANGUNAN_AR', 'INDUSTRI_AR',
            'NIAGA_AR', 'PEMERINTAHAN_AR', 'PENDIDIKAN_AR',
            'ARENAOLAHRAGA_AR', 'SARANAIBADAH_AR', 'KESEHATAN_AR'
        ],
        'forest': [
            'HUTAN', 'HUTANLAHANRENDAH', 'HUTANKERING', 'HUTANRAWA'
        ],
        'shrub': [
            'SEMAKBELUKAR', 'HERBADANRUMPUT', 'LADANG',
            'ALANG', 'TANAMANCAMPUR', 'TEGALAN'
        ],
        'plantation': [
            'PERKEBUNAN', 'KEBUN'
        ],
        'paddy': [
            'SAWAH', 'EMPANG', 'TAMBAK'
        ],
        'water': [
            'DANAU_AR', 'WADUK_AR', 'SITU_AR', 'RAWA_AR'
        ]
    }

    ROAD_PATTERNS = ['JALAN_LN', 'JALAN']

    # BIG Rupabumi Indonesia (RBI) layers often prepend AGRI/NONAGRI to land-use names
    RBI_PREFIXES = ('NONAGRI', 'AGRI')

    ROAD_WIDTHS = {
        'tol': 12.0,
        'arteri': 8.0,
        'kolektor': 5.0,
        'lokal': 3.5,
        'default': 3.0
    }

    def __init__(self, config):
        super().__init__("LandUseLoader")
        dirs = config.paths.get("admin_dirs") or []
        self.admin_dirs = [Path(d) for d in dirs]
        self.target_crs = config.spatial.target_crs
        self.patterns = self.DEFAULT_PATTERNS
        self.load_roads = True
        self.road_widths = self.ROAD_WIDTHS

    @classmethod
    def can_handle(cls, config) -> bool:
        return bool(config.paths.get("admin_dirs"))

    def load(self, registry: DataRegistry) -> Dict[str, DataSource]:
        self.log_loading("Loading land use data...")
        results = {}

        for category, patterns in self.patterns.items():
            gdf = self._load_layers(patterns)
            if not gdf.empty:
                source_name = f"landuse_{category}"
                source = DataSource(
                    name=source_name,
                    source_type=SourceType.VECTOR,
                    data=gdf,
                    crs=self.target_crs,
                    metadata={'category': category, 'feature_count': len(gdf)}
                )
                registry.register(source_name, source)
                results[source_name] = source
                self.log_loading(f"  {category}: {len(gdf)} features")
            else:
                log_warning(f"  No data found for {category}")

        if self.load_roads:
            gdf_roads = self._load_roads_as_polygons()
            if not gdf_roads.empty:
                source = DataSource(
                    name="landuse_road",
                    source_type=SourceType.VECTOR,
                    data=gdf_roads,
                    crs=self.target_crs,
                    metadata={'category': 'road', 'feature_count': len(gdf_roads)}
                )
                registry.register("landuse_road", source)
                results["landuse_road"] = source
                self.log_loading(f"  roads: {len(gdf_roads)} features (buffered)")

        self.loaded_sources = results
        log_success(f"Loaded {len(results)} land use categories")
        return results

    def _load_layers(self, filename_patterns: List[str]) -> gpd.GeoDataFrame:
        frames = []

        for admin_dir in self.admin_dirs:
            if not admin_dir.exists():
                continue

            for shp_file in admin_dir.glob("*.shp"):
                if match_pattern(shp_file.name, filename_patterns, prefixes=self.RBI_PREFIXES):
                    try:
                        gdf = gpd.read_file(shp_file).to_crs(self.target_crs)
                        gdf = self._standardize_columns(gdf, shp_file.stem)
                        frames.append(gdf)
                    except Exception as e:
                        log_warning(f"Failed to load SHP {shp_file.name}: {e}")

            for gdb_dir in admin_dir.glob("*.gdb"):
                try:
                    for layer in fiona.listlayers(str(gdb_dir)):
                        if match_pattern(layer, filename_patterns, prefixes=self.RBI_PREFIXES):
                            try:
                                gdf = gpd.read_file(gdb_dir, layer=layer).to_crs(self.target_crs)
                                gdf = self._standardize_columns(gdf, layer)
                                frames.append(gdf)
                            except Exception as e:
                                log_warning(f"Failed to load layer {layer}: {e}")
                except Exception as e:
                    log_warning(f"Failed to read GDB {gdb_dir.name}: {e}")

        if not frames:
            return gpd.GeoDataFrame()

        return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=self.target_crs)

    def _load_roads_as_polygons(self) -> gpd.GeoDataFrame:
        gdf_lines = self._load_layers(self.ROAD_PATTERNS)
        if gdf_lines.empty:
            return gpd.GeoDataFrame()

        gdf_metric = gdf_lines.to_crs("EPSG:3857")
        widths = gdf_metric['REMARK'].map(self._get_road_width)
        gdf_metric['geometry'] = gdf_metric.geometry.buffer(widths)
        return gdf_metric.to_crs(self.target_crs)

    def _get_road_width(self, remark) -> float:
        if not isinstance(remark, str):
            return self.road_widths.get('default', 3.0)
        r = remark.lower()
        for key in ('tol', 'arteri', 'kolektor', 'lokal'):
            if key in r:
                return self.road_widths.get(key, 3.0)
        return self.road_widths.get('default', 3.0)

    def _standardize_columns(self, gdf: gpd.GeoDataFrame, source_name: str) -> gpd.GeoDataFrame:
        rename_map = {}
        for col in gdf.columns:
            c = col.upper()
            if 'NAMOBJ' in c:
                rename_map[col] = 'NAMOBJ'
            elif 'REMARK' in c or 'CATATAN' in c:
                rename_map[col] = 'REMARK'

        gdf = gdf.rename(columns=rename_map)
        if 'NAMOBJ' not in gdf.columns:
            gdf['NAMOBJ'] = 'Unknown'
        if 'REMARK' not in gdf.columns:
            gdf['REMARK'] = source_name
        return gdf
