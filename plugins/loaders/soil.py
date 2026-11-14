import geopandas as gpd
from pathlib import Path
from murasa.core.base import DataLoaderPlugin
from murasa.core.logger import log_success, log_error


class SoilDataLoader(DataLoaderPlugin):
    """
    Loads soil vector datasets into the DataRegistry.
    """

    provides = ["soil"]
    requires = []

    def __init__(self, config):
        super().__init__("SoilDataLoader")
        self.soil_path = config.get_factor_path("soil")
        self.target_crs = config.spatial.target_crs

    @classmethod
    def can_handle(cls, config) -> bool:
        return config.get_factor_path("soil") is not None

    def load(self, registry):
        if registry.has("soil"):
            src = registry.get("soil", auto_load=True)
            self.loaded_sources["soil"] = src
            return self.loaded_sources

        self.log_loading(f"Loading soil dataset from {self.soil_path}...")
        soil_file = Path(self.soil_path)
        if not soil_file.exists():
            log_error(f"Soil file not found at: {self.soil_path}")
            return {}

        try:
            gdf = gpd.read_file(soil_file)
            if gdf.crs != self.target_crs:
                gdf = gdf.to_crs(self.target_crs)

            source = registry.register_vector(
                "soil",
                gdf,
                self.target_crs,
                feature_count=len(gdf)
            )
            self.loaded_sources["soil"] = source
            log_success(f"Loaded soil dataset: {len(gdf)} features")
            return self.loaded_sources
        except Exception as e:
            log_error(f"Failed to load soil dataset: {str(e)}")
            return {}
