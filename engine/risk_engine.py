from pathlib import Path
from typing import Optional, List, Tuple, Any
import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform

from murasa.core.config import EngineConfig, infer_source_type
from murasa.core.data_source import DataRegistry, DataSource, SourceType
from murasa.core.logger import logger, log_section, log_subsection, log_success
from murasa.core.processing import crs_equal, meters_to_crs_units
from murasa.core.base import (
    ParameterPlugin,
    PostProcessorPlugin,
    RiskCalculatorPlugin,
    LoaderRegistry,
    plugin_registry,
    resolve_plugin_dirs,
)


class RiskEngine:
    """
    The Engine.

    Manages the lifecycle of data sources, parameter plugins, risk calculation formulas,
    and post-processors.

    Based on the EngineConfig.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.registry = DataRegistry()
        self.parameters: List[ParameterPlugin] = []
        self.post_processors: List[PostProcessorPlugin] = []
        self.calculator: Optional[RiskCalculatorPlugin] = None

        self.grid_shape: Optional[Tuple[int, int]] = None
        self.transform: Optional[Any] = None
        self.crs: str = self.config.spatial.target_crs

        self.study_area_mask: Optional[np.ndarray] = None
        self.admin_units: Optional[gpd.GeoDataFrame] = None

        self.risk_raster: Optional[np.ndarray] = None
        self.exclusion_mask: Optional[np.ndarray] = None

    def register_calculator(self, plugin: RiskCalculatorPlugin) -> None:
        self.calculator = plugin
        log_success(f"Registered Risk Calculator: {plugin.name}")

    def register_data(self, name: str, source: DataSource) -> None:
        self.registry.register(name, source)

    def register_data_from_path(
        self,
        name: str,
        path: Path,
        source_type: SourceType
    ) -> None:
        source = DataSource(
            name=name,
            source_type=source_type,
            path=path,
            crs=self.crs
        )
        self.registry.register(name, source)

    def auto_register_from_config(self) -> None:
        """Register all file-based data sources defined in config.paths and config.factors."""
        log_subsection("Auto-registering data sources from config")

        if hasattr(self.config, "paths") and self.config.paths:
            for name, path_val in self.config.paths.items():
                if name == "output_dir" or not isinstance(path_val, Path):
                    continue
                if path_val.exists() and path_val.is_file() and not self.registry.has(name):
                    stype_str = infer_source_type(path_val, default="raster")
                    source_type = SourceType.RASTER if stype_str == "raster" else SourceType.VECTOR
                    self.register_data_from_path(name, path_val, source_type)
                    logger.info(f"  {name} ({source_type.value}): {path_val}")

        for name, factor in self.config.factors.items():
            if self.registry.has(name):
                continue
            if factor.source_path and factor.source_path.exists():
                stype_str = infer_source_type(factor.source_path, default=factor.source_type)
                source_type = SourceType.RASTER if stype_str == "raster" else SourceType.VECTOR
                self.register_data_from_path(name, factor.source_path, source_type)
                logger.info(f"  {name} ({source_type.value}): {factor.source_path}")
            elif factor.source_type == "derived":
                logger.info(f"  {name}: derived (will be computed)")
            else:
                logger.warning(f"  {name}: path not found or not specified")

        log_success(f"Registered {len(self.registry)} data sources")

    def _discover_plugins(self) -> List[Path]:
        """Auto-discover plugins across internal, custom, and configured directories."""
        extra_dirs = getattr(self.config, "plugin_dirs", None) or []
        dirs = resolve_plugin_dirs(extra_dirs=extra_dirs)
        plugin_registry.auto_discover(dirs)
        return dirs

    def _build_parameter_plugin(
        self,
        factor_name: str,
        weight: float,
        group: str = "hazard",
        explicit_processor: Optional[str] = None,
        raw_params: Optional[dict] = None,
    ) -> Optional[ParameterPlugin]:
        """Resolve and instantiate a ParameterPlugin, honoring YAML plugin/processor aliases."""
        params = dict(raw_params) if isinstance(raw_params, dict) else {}
        param_plugin = params.pop("plugin", None)
        param_processor = params.pop("processor", None)
        plugin_key = explicit_processor or param_plugin or param_processor or factor_name

        if not plugin_registry.has(plugin_key):
            return None

        aliased = plugin_key != factor_name
        if aliased and "name" not in params:
            params["name"] = factor_name

        plugin = plugin_registry.create(
            plugin_key,
            weight=weight,
            group=group,
            parameters=params,
        )
        if aliased:
            plugin.name = factor_name
            if "short_name" not in params:
                plugin.short_name = factor_name
            if "contrib_name" not in params:
                plugin.contrib_name = None
            if "display_name" not in params:
                plugin.display_name = factor_name.replace("_", " ").title()
        return plugin

    def auto_register_parameters_from_config(self) -> int:
        """Instantiate and register parameter plugins configured in weights or factors."""
        log_subsection("Auto-registering risk parameter plugins")
        self._discover_plugins()

        registered = 0
        seen_names: set[str] = set()
        params_cfg = (
            getattr(self.config, "plugin_parameters", None)
            or getattr(self.config, "parameters", None)
            or {}
        )

        if getattr(self.config, "factors", None):
            for factor_name, factor_cfg in self.config.factors.items():
                raw_params = {}
                if isinstance(params_cfg.get(factor_name), dict):
                    raw_params.update(params_cfg[factor_name])
                if isinstance(getattr(factor_cfg, "parameters", None), dict):
                    raw_params.update(factor_cfg.parameters)

                plugin = self._build_parameter_plugin(
                    factor_name=factor_name,
                    weight=factor_cfg.weight,
                    group=getattr(factor_cfg, "group", "hazard"),
                    explicit_processor=getattr(factor_cfg, "processor", None),
                    raw_params=raw_params,
                )
                if plugin is None:
                    continue
                if (factor_cfg.weight is None or factor_cfg.weight <= 0) and not getattr(plugin, "is_exclusion", False):
                    continue
                if self.register_parameter(plugin):
                    registered += 1
                    seen_names.add(plugin.name)

        elif getattr(self.config, "weights", None):
            for factor_name, weight in self.config.weights.items():
                if weight is None or weight <= 0:
                    continue
                factor_params = params_cfg.get(factor_name, {})
                group = factor_params.get("group", "hazard") if isinstance(factor_params, dict) else "hazard"
                plugin = self._build_parameter_plugin(
                    factor_name=factor_name,
                    weight=weight,
                    group=group,
                    raw_params=factor_params if isinstance(factor_params, dict) else {},
                )
                if plugin is not None and self.register_parameter(plugin):
                    registered += 1
                    seen_names.add(plugin.name)

        for param_key, param_val in params_cfg.items():
            if param_key in seen_names or (
                getattr(self.config, "weights", None) and param_key in self.config.weights
            ):
                continue
            cand = self._build_parameter_plugin(
                factor_name=param_key,
                weight=0.0,
                group=param_val.get("group", "hazard") if isinstance(param_val, dict) else "hazard",
                raw_params=param_val if isinstance(param_val, dict) else {},
            )
            if cand is not None and getattr(cand, "is_exclusion", False) and not any(
                p.name == cand.name for p in self.parameters
            ):
                self.register_parameter(cand)

        logger.info(f"Auto-registered {registered} parameter plugin(s) from config")
        return registered

    def auto_register_post_processors_from_config(self) -> int:
        """Instantiate and register post-processor plugins declared in config.post_processors."""
        if not getattr(self.config, "post_processors", None):
            return 0

        self._discover_plugins()

        registered = 0
        for item in self.config.post_processors:
            if isinstance(item, str) and plugin_registry.has(item):
                self.register_post_processor(plugin_registry.create(item))
                registered += 1
            elif isinstance(item, dict):
                for name, params in item.items():
                    if plugin_registry.has(name):
                        self.register_post_processor(
                            plugin_registry.create(name, parameters=params or {})
                        )
                        registered += 1
        return registered

    def register_parameter(self, plugin: ParameterPlugin) -> bool:
        if not plugin.validate_requirements(self.registry):
            logger.warning(f"Plugin '{plugin.name}' missing required data")
            return False

        self.parameters.append(plugin)
        log_success(f"Registered: {plugin.name} (weight={plugin.weight:.3f})")
        return True

    def register_post_processor(self, plugin: PostProcessorPlugin) -> None:
        self.post_processors.append(plugin)
        log_success(f"Registered Post-Processor: {plugin.name}")

    def clear_parameters(self) -> None:
        self.parameters.clear()
        logger.info("Cleared all parameters")

    def normalize_weights(self) -> None:
        active = [p for p in self.parameters if not p.is_exclusion]
        total = sum(p.weight for p in active)

        if total <= 0 or abs(total - 1.0) < 0.001:
            return

        logger.warning(f"Weights sum to {total:.3f}, normalizing...")
        for param in active:
            param.weight /= total

    def setup_grid(self, bounds: Optional[Tuple] = None, from_admin: bool = False) -> None:
        """Initialize the spatial grid transform and dimensions from reference raster or study area bounds."""
        ref_setting = getattr(self.config.spatial, "reference_raster", None)
        target_path: Optional[Path] = None

        if ref_setting:
            if self.registry.has(ref_setting):
                ref_source = self.registry.get(ref_setting, auto_load=False)
                if ref_source.path:
                    target_path = Path(ref_source.path)

            if not target_path:
                p = Path(ref_setting)
                if p.exists():
                    target_path = p
                else:
                    for source in self.registry.sources.values():
                        if source.path and Path(source.path).name == ref_setting:
                            target_path = Path(source.path)
                            break
        else:
            for candidate_key in self.registry.list_by_type(SourceType.RASTER):
                src = self.registry.get(candidate_key, auto_load=False)
                if src.path and Path(src.path).exists():
                    target_path = Path(src.path)
                    break

        if target_path and target_path.exists():
            with rasterio.open(target_path) as src:
                if src.crs and not crs_equal(src.crs, self.crs):
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src.crs, self.crs, src.width, src.height, *src.bounds
                    )
                    self.transform = dst_transform
                    self.grid_shape = (dst_height, dst_width)
                    logger.info(
                        f"Grid reprojected from reference raster '{target_path.name}' "
                        f"({src.crs} -> {self.crs})"
                    )
                else:
                    self.transform = src.transform
                    self.grid_shape = (src.height, src.width)
                    logger.info(f"Grid shape and transform aligned with reference raster: '{target_path.name}'")
                log_success(f"Grid: {self.grid_shape[1]} x {self.grid_shape[0]} pixels")
                return

        study_area_key = self.config.spatial.study_area_key
        if bounds is None and self.registry.has(study_area_key):
            admin_source = self.registry.get(study_area_key)
            admin_data = admin_source.data
            if getattr(admin_data, "crs", None) and not crs_equal(admin_data.crs, self.crs):
                admin_data = admin_data.to_crs(self.crs)
            bounds = admin_data.total_bounds
            logger.info(f"Grid bounds generated from vector {study_area_key} data")

        if bounds is None:
            raise ValueError("Grid bounds required (provide bounds or reference_raster)")

        resolution = self.config.spatial.resolution
        resolution_unit = getattr(self.config.spatial, "resolution_unit", "meter")
        if resolution_unit == "crs":
            res_x = res_y = float(resolution)
            unit_label = " CRS units"
        else:
            res_x, res_y = meters_to_crs_units(resolution, self.crs, bounds=bounds)
            unit_label = "m"

        width = max(1, int(round((bounds[2] - bounds[0]) / res_x)))
        height = max(1, int(round((bounds[3] - bounds[1]) / res_y)))

        self.transform = from_bounds(*bounds, width, height)
        self.grid_shape = (height, width)
        log_success(f"Grid: {width} x {height} pixels @ {resolution}{unit_label}")

    def set_study_area(self, admin_gdf: Optional[gpd.GeoDataFrame] = None) -> None:
        """Rasterize the study area boundary polygon into a boolean mask."""
        study_area_key = self.config.spatial.study_area_key
        if admin_gdf is None:
            if not self.registry.has(study_area_key):
                logger.warning(f"No study area defined using key '{study_area_key}', processing full grid")
                return
            admin_gdf = self.registry.get(study_area_key).data

        if getattr(admin_gdf, "crs", None) and not crs_equal(admin_gdf.crs, self.crs):
            admin_gdf = admin_gdf.to_crs(self.crs)

        self.admin_units = admin_gdf
        self.study_area_mask = geometry_mask(
            admin_gdf.geometry,
            transform=self.transform,
            invert=True,
            out_shape=self.grid_shape
        )
        self.registry.study_area_mask = self.study_area_mask
        log_success(f"Study area: {len(admin_gdf)} units from key '{study_area_key}'")

    def calculate_risk(self) -> np.ndarray:
        """Execute all registered parameter plugins and combine them via the active RiskCalculatorPlugin."""
        log_section("CALCULATING RISK")

        if not self.parameters:
            raise ValueError("No parameters registered")

        if self.grid_shape is None or self.transform is None:
            raise ValueError("Grid not setup (call setup_grid first)")

        if self.calculator is None:
            self._discover_plugins()
            method = getattr(self.config, "risk_method", "weighted_sum")
            if plugin_registry.has(method):
                self.register_calculator(plugin_registry.create(method))
            else:
                from murasa.plugins.calculators import WeightedSumCalculator
                self.register_calculator(WeightedSumCalculator())
            logger.info(f"Auto-selected calculator: {self.calculator.name}")

        self.normalize_weights()
        self.exclusion_mask = np.zeros(self.grid_shape, dtype=bool)

        results_map = {}
        for plugin in self.parameters:
            component = self._process_parameter(plugin)
            if component is not None:
                results_map[plugin.name] = component

        if not results_map:
            raise ValueError("No valid risk parameter results computed!")

        active_parameters = [p for p in self.parameters if not p.is_exclusion]
        self.risk_raster = self.calculator.calculate(
            active_parameters,
            results_map,
            self.grid_shape
        )

        if self.exclusion_mask.any():
            self.risk_raster[self.exclusion_mask] = np.nan
            logger.info(f"   Total excluded area: {self.exclusion_mask.sum()} pixels")

        if self.study_area_mask is not None:
            self.risk_raster[~self.study_area_mask] = np.nan

        log_section("RISK CALCULATION COMPLETE")
        logger.info(f"Range: {np.nanmin(self.risk_raster):.3f} - {np.nanmax(self.risk_raster):.3f}")
        logger.info(f"Mean: {np.nanmean(self.risk_raster):.3f}")

        return self.risk_raster

    def _process_parameter(self, plugin: ParameterPlugin) -> Optional[np.ndarray]:
        try:
            component = plugin.process(
                self.registry,
                self.grid_shape,
                self.transform,
                self.crs
            )

            if plugin.is_exclusion:
                mask = (np.nan_to_num(component, nan=0.0) > 0).astype(bool)
                self.exclusion_mask |= mask
                logger.info(f"      Added exclusion mask from {plugin.name}: {mask.sum()} pixels")
                return None

            component = self.registry.apply_study_area_mask(component)
            plugin.result = component
            output_name = f"risk_{plugin.name}"
            self.registry.register(
                output_name,
                DataSource(output_name, SourceType.RASTER, data=component, crs=self.crs)
            )

            return component
        except Exception as e:
            logger.error(f"      Failed: {plugin.name}: {e}")
            return None

    def export_raster(self, output_path: Path) -> None:
        if self.risk_raster is None:
            raise ValueError("No risk raster calculated")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            'driver': 'GTiff',
            'height': self.grid_shape[0],
            'width': self.grid_shape[1],
            'count': 1,
            'dtype': np.float32,
            'crs': self.crs,
            'transform': self.transform,
            'nodata': np.nan,
            'compress': 'lzw'
        }

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(self.risk_raster, 1)

        log_success(f"Raster saved: {output_path}")

    def export_statistics(self, output_path: Path) -> None:
        stats = {
            'parameters': {
                p.name: {
                    'weight': p.weight,
                    'statistics': p.get_statistics()
                }
                for p in self.parameters
            },
            'risk': {
                'min': float(np.nanmin(self.risk_raster)),
                'max': float(np.nanmax(self.risk_raster)),
                'mean': float(np.nanmean(self.risk_raster)),
                'std': float(np.nanstd(self.risk_raster))
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        log_success(f"Statistics saved: {output_path}")

    def export_all(self, output_dir: Optional[Path] = None) -> None:
        if output_dir is None:
            output_dir = self.config.output.output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        self.export_raster(output_dir / "risk_map.tif")
        self.export_statistics(output_dir / "statistics.json")
        log_success(f"All outputs saved to: {output_dir}")

        if self.post_processors:
            log_section("POST-PROCESSING")
            for pp in self.post_processors:
                try:
                    pp.post_process(
                        self.registry,
                        self.risk_raster,
                        self.transform,
                        self.crs,
                        output_dir,
                        config=self.config
                    )
                except Exception as e:
                    logger.error(f"Post-processor {pp.name} failed: {e}")

    def print_summary(self) -> None:
        log_section("ENGINE SUMMARY")
        logger.info(f"Data sources: {len(self.registry)}")
        self.registry.print_summary()

        logger.info(f"\nParameters: {len(self.parameters)}")
        for param in self.parameters:
            logger.info(f"  - {param}")

        if self.grid_shape:
            res_unit = "m" if getattr(self.config.spatial, "resolution_unit", "meter") == "meter" else " CRS units"
            logger.info(f"\nGrid: {self.grid_shape[1]} x {self.grid_shape[0]} pixels")
            logger.info(f"Resolution: {self.config.spatial.resolution}{res_unit}")

        if self.admin_units is not None:
            logger.info(f"\nStudy area: {len(self.admin_units)} units")

    def run(self) -> np.ndarray:
        """Execute the complete config-driven risk assessment pipeline."""
        log_section("RISK ENGINE - FULL RUN")

        plugin_dirs = self._discover_plugins()
        self.auto_register_from_config()

        loader_registry = LoaderRegistry()
        loader_registry.auto_discover(plugin_dirs)
        loader_registry.load_all(self.config, self.registry)

        if not self.parameters:
            self.auto_register_parameters_from_config()

        if not self.post_processors:
            self.auto_register_post_processors_from_config()

        self.setup_grid()
        self.set_study_area()

        risk = self.calculate_risk()
        self.export_all()

        log_section("COMPLETE")
        return risk
