import re
from murasa.core.base import DataLoaderPlugin
from murasa.core.file_scanner import scan_filename
from murasa.core.logger import log_success, log_error
from murasa.core.processing import resolve_metric_crs


class AdminBoundariesLoader(DataLoaderPlugin):
    """
    Loads administrative boundary polygons from RBI.
    """

    provides = ["admin"]
    requires = []

    def __init__(self, config):
        super().__init__("AdminBoundariesLoader")
        self.admin_dirs = config.paths.get("admin_dirs") or []
        self.target_crs = config.spatial.target_crs
        self.metric_crs = getattr(config.spatial, "metric_crs", "auto")
        self.filter_city = getattr(config.spatial, "filter_city", None)
        self.filter_district = getattr(config.spatial, "filter_district", None)
        self.filename = config.parameters.get("admin_filename", "ADMINISTRASIDESA_AR_25K.shp")

    @classmethod
    def can_handle(cls, config) -> bool:
        return bool(config.paths.get("admin_dirs"))

    def _clean_admin_gdf(self, gdf):
        if gdf.empty:
            return gdf

        if self.filter_city and "WADMKK" in gdf.columns:
            cities = [self.filter_city] if isinstance(self.filter_city, str) else list(self.filter_city)
            wadmkk_col = gdf["WADMKK"].astype(str).str.strip().str.upper()
            existing_wadmkk = set(wadmkk_col.unique())
            allowed = set()
            for item in cities:
                u = str(item).strip().upper()
                allowed.add(u)
                if re.match(r"^KAB(?:UPATEN)?\.?\s+", u):
                    allowed.add(re.sub(r"^KAB(?:UPATEN)?\.?\s+", "", u).strip())
                elif u.startswith("KOTA ") and u not in existing_wadmkk:
                    allowed.add(u[5:].strip())
            gdf = gdf[wadmkk_col.isin(allowed)].copy()

        if self.filter_district and "WADMKC" in gdf.columns:
            districts = [self.filter_district] if isinstance(self.filter_district, str) else list(self.filter_district)
            allowed_dist = {str(d).strip().upper() for d in districts}
            wadmkc_col = gdf["WADMKC"].astype(str).str.strip().str.upper()
            gdf = gdf[wadmkc_col.isin(allowed_dist)].copy()

        if not gdf.empty and all(c in gdf.columns for c in ("WADMKK", "WADMKC", "NAMOBJ")):
            try:
                metric_crs = resolve_metric_crs(gdf, self.metric_crs)
                areas_ha = gdf.to_crs(metric_crs).geometry.area / 10000.0
            except Exception:
                areas_ha = gdf.geometry.area
            gdf["_tmp_area"] = areas_ha
            gdf = gdf.sort_values("_tmp_area", ascending=False)
            gdf = gdf.drop_duplicates(subset=["WADMKK", "WADMKC", "NAMOBJ"], keep="first")
            gdf = gdf[gdf["_tmp_area"] >= 1.0].drop(columns=["_tmp_area"]).reset_index(drop=True)

        return gdf

    def load(self, registry):
        self.log_loading(f"Loading administrative boundaries from {len(self.admin_dirs)} directories...")
        gdf = scan_filename(self.admin_dirs, self.filename, self.target_crs)

        if gdf.empty:
            log_error(f"No {self.filename} found in provided directories")
            return {}

        gdf = self._clean_admin_gdf(gdf)

        source = registry.register_vector("admin", gdf, self.target_crs, feature_count=len(gdf))
        self.loaded_sources["admin"] = source
        log_success(f"Loaded administrative boundaries: {len(gdf)} features")
        return self.loaded_sources
