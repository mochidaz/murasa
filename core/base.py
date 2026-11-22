from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Any, Optional, Dict, List, Type, Iterable, Union
import hashlib
import inspect
import numpy as np

from .data_source import DataRegistry
from .logger import logger


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
INTERNAL_PLUGINS_DIR = _PACKAGE_ROOT / "plugins"
CUSTOM_PLUGINS_DIR = _PACKAGE_ROOT / "custom_plugins"


def resolve_plugin_dirs(
    plugins_dir: Optional[Union[Path, str, Iterable[Union[Path, str]]]] = None,
    extra_dirs: Optional[Iterable[Union[Path, str]]] = None,
    include_custom: Optional[bool] = None,
) -> List[Path]:
    """
    Resolve ordered, deduplicated plugin root directories.

    When plugins_dir is omitted or resolves to Murasa's internal plugins/
    directory, the sibling custom_plugins/ directory is included automatically
    unless include_custom=False.
    """
    candidates: List[Path] = []
    if plugins_dir is None:
        candidates.append(INTERNAL_PLUGINS_DIR)
        if include_custom is not False:
            candidates.append(CUSTOM_PLUGINS_DIR)
    else:
        raw_items = (
            [plugins_dir]
            if isinstance(plugins_dir, (str, Path))
            else list(plugins_dir)
        )
        for item in raw_items:
            if not item:
                continue
            resolved = Path(item).resolve()
            candidates.append(resolved)
            if (include_custom is True) or (
                include_custom is None and resolved == INTERNAL_PLUGINS_DIR.resolve()
            ):
                candidates.append(CUSTOM_PLUGINS_DIR)

    if extra_dirs:
        for item in extra_dirs:
            if item:
                candidates.append(Path(item).resolve())

    unique: List[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(rp)
    return unique


def _plugin_module_name(base_dir: Path, py_file: Path, subpkg: Optional[str] = None) -> str:
    """Build an isolated module name for dynamically discovered plugin or loader files."""
    resolved = base_dir.resolve()
    if resolved == INTERNAL_PLUGINS_DIR.resolve():
        prefix = "murasa.plugins"
    elif resolved == CUSTOM_PLUGINS_DIR.resolve():
        prefix = "murasa.custom_plugins"
    else:
        digest = hashlib.md5(str(resolved).encode("utf-8")).hexdigest()[:8]
        prefix = f"murasa.ext_plugins_{digest}"
    if subpkg:
        return f"{prefix}.{subpkg}.{py_file.stem}"
    return f"{prefix}.{py_file.stem}"


class ParameterPlugin(ABC):
    """
    Abstract base class for spatial risk parameter plugins.
    Each plugin validates required data sources and processes spatial inputs into a normalized
    0-1 risk raster.
    """

    short_name: Optional[str] = None
    contrib_name: Optional[str] = None
    display_name: Optional[str] = None

    def __init__(self, name: str, weight: float = 1.0, group: str = "hazard", **kwargs):
        self.name = name
        self.weight = weight
        self.group = group
        self.result: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = dict(kwargs)
        self.is_exclusion: bool = False

    @property
    def column_key(self) -> str:
        """Short identifier used for parameter score columns."""
        return self.short_name or self.name

    @property
    def contrib_key(self) -> str:
        """Short identifier used for weighted contribution columns."""
        return self.contrib_name or self.column_key

    @property
    def label(self) -> str:
        """Reporting label."""
        return self.display_name or self.name.replace("_", " ").title()

    @abstractmethod
    def validate_requirements(self, registry: DataRegistry) -> bool:
        """Check whether required data sources exist in the registry."""
        pass

    @abstractmethod
    def process(
        self,
        registry: DataRegistry,
        grid_shape: Tuple[int, int],
        transform: Any,
        crs: str
    ) -> np.ndarray:
        """Compute and return the normalized risk raster (0-1, float32)."""
        pass

    def get_statistics(self) -> dict:
        """Compute descriptive statistics for the processed parameter raster."""
        if self.result is None:
            return {}

        valid = np.isfinite(self.result)
        if not valid.any():
            return {'valid_pixels': 0}

        return {
            'min': float(np.nanmin(self.result)),
            'max': float(np.nanmax(self.result)),
            'mean': float(np.nanmean(self.result)),
            'std': float(np.nanstd(self.result)),
            'valid_pixels': int(valid.sum()),
            'total_pixels': self.result.size
        }

    def normalize_percentile(
        self,
        data: np.ndarray,
        lower: float = 5,
        upper: float = 95
    ) -> np.ndarray:
        """Normalize array to 0-1 using percentile clipping."""
        valid = np.isfinite(data)
        if not valid.any():
            logger.warning(f"{self.name}: No valid data for normalization")
            return np.zeros_like(data)

        d_min, d_max = np.percentile(data[valid], [lower, upper])
        if d_max == d_min:
            logger.warning(f"{self.name}: Constant values detected")
            result = np.full_like(data, np.nan, dtype=np.float32)
            result[valid] = 0.5
            return result

        normalized = np.clip((data - d_min) / (d_max - d_min), 0, 1)
        return normalized.astype(np.float32)

    def normalize_minmax(self, data: np.ndarray) -> np.ndarray:
        """Normalize array to 0-1 using min-max scaling."""
        valid = np.isfinite(data)
        if not valid.any():
            return np.zeros_like(data)

        d_min = np.nanmin(data)
        d_max = np.nanmax(data)
        if d_max == d_min:
            result = np.full_like(data, np.nan, dtype=np.float32)
            result[valid] = 0.5
            return result

        normalized = (data - d_min) / (d_max - d_min)
        return normalized.astype(np.float32)

    def log_processing(self):
        logger.info(f"   Processing: {self.name} (weight={self.weight:.3f})")

    def log_result(self):
        stats = self.get_statistics()
        if stats and stats.get('valid_pixels', 0) > 0:
            logger.info(f"      Range: {stats['min']:.3f} - {stats['max']:.3f}")
        else:
            logger.warning("      No valid data produced")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight:.3f}, group='{self.group}')"

    def __str__(self) -> str:
        return f"{self.name} [{self.group}] ({self.weight:.3f})"


class RasterParameterPlugin(ParameterPlugin):
    """
    Base class for raster-based risk parameters with declarative source lookup and normalization.
    """

    source_keys: List[str] = []
    inverse: bool = False
    normalization: str = "percentile"

    def __init__(self, name: str = "", weight: float = 1.0, group: str = "hazard", **kwargs):
        if not name:
            name = self.__class__.__name__.replace("Plugin", "").lower()
        if "invert" in kwargs and "inverse" not in kwargs:
            kwargs["inverse"] = kwargs.pop("invert")
        super().__init__(name, weight, group, **kwargs)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def _resolve_source_keys(self, registry: Optional[DataRegistry] = None) -> List[str]:
        candidates: List[str] = []
        explicit_source = getattr(self, "source", None)
        if explicit_source:
            candidates.append(explicit_source)
        elif self.name and (
            not self.source_keys
            or (registry is not None and registry.has(self.name))
        ):
            candidates.append(self.name)

        for key in (self.source_keys or []):
            if key not in candidates:
                candidates.append(key)

        if not candidates and self.name:
            candidates.append(self.name)
        return candidates

    def validate_requirements(self, registry: DataRegistry) -> bool:
        keys = self._resolve_source_keys(registry)
        return any(registry.has(k) for k in keys)

    def _find_source(self, registry: DataRegistry):
        keys = self._resolve_source_keys(registry)
        for key in keys:
            if registry.has(key):
                return registry.get(key)
        raise KeyError(
            f"{self.name}: None of {keys} found in registry. "
            f"Available: {registry.list_sources()}"
        )

    def transform_data(self, data: np.ndarray, ctx) -> np.ndarray:
        """Apply custom transformations before normalization."""
        return data

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        from . import normalize as norm

        match self.normalization:
            case "percentile":
                return norm.percentile(data)
            case "minmax":
                return norm.minmax(data)
            case "rank":
                return norm.rank(data)
            case "zscore":
                return norm.zscore(data)
            case _:
                return norm.percentile(data)

    def process(self, registry, grid_shape, transform, crs):
        from .processing import ProcessingContext

        self.log_processing()
        ctx = ProcessingContext(grid_shape, transform, crs)

        source = self._find_source(registry)
        data = ctx.resample_raster(source)
        data = self.transform_data(data, ctx)
        data = registry.apply_study_area_mask(data)
        normalized = self._normalize(data)

        if self.inverse:
            normalized = 1 - normalized

        self.result = normalized.astype(np.float32)
        self.log_result()
        return self.result


class RiskCalculatorPlugin(ABC):
    """
    Abstract base class for risk calculation formulas.

    Like WLC, BNPB's Formula, and others.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(
        self,
        parameters: List[ParameterPlugin],
        results: Dict[str, np.ndarray],
        grid_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Combine parameter rasters into the final composite risk raster."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class PostProcessorPlugin(ABC):
    """
    Abstract base class for post-processing pipelines executed after risk raster calculation.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def post_process(
        self,
        registry: DataRegistry,
        risk_raster: np.ndarray,
        transform: Any,
        crs: str,
        output_dir: Any,
        config: Any = None
    ):
        """Execute post-processing and export derived artifacts."""
        pass


class CircularDependencyError(Exception):
    """Raised when data loaders declare a cyclic dependency graph."""
    pass


class DataLoaderPlugin(ABC):
    """
    Abstract base class for data loader plugins with dependency resolution via provides/requires.
    """

    provides: List[str] = []
    requires: List[str] = []

    def __init__(self, name: str):
        self.name = name
        self.loaded_sources: dict = {}

    @classmethod
    @abstractmethod
    def can_handle(cls, config) -> bool:
        """Determine whether this loader should run given the active configuration."""
        pass

    @abstractmethod
    def load(self, registry: DataRegistry) -> dict:
        """Load data sources and register them into the DataRegistry."""
        pass

    def log_loading(self, message: str):
        logger.info(f"[{self.name}] {message}")

    def __repr__(self) -> str:
        deps = f", requires={self.requires}" if self.requires else ""
        return f"{self.__class__.__name__}(provides={self.provides}{deps})"


class PluginRegistry:
    """
    Unified registry for parameter, calculator, and post-processor plugins with signature-safe instantiation.
    """

    def __init__(self):
        self.plugins: Dict[str, Type] = {}
        self._discovered_dirs: set[Path] = set()

    def register(self, name: str, plugin_class: Type = None):
        """Register a plugin class by canonical name or as a class decorator."""
        if plugin_class is None:
            def decorator(cls):
                self.plugins[name] = cls
                logger.debug(f"Registered plugin: {name} ({cls.__name__})")
                return cls
            return decorator

        self.plugins[name] = plugin_class
        logger.debug(f"Registered plugin: {name} ({plugin_class.__name__})")
        return plugin_class

    def has(self, name: str) -> bool:
        return name in self.plugins

    def get(self, name: str) -> Type:
        if name not in self.plugins:
            raise KeyError(
                f"Plugin '{name}' not found in registry. "
                f"Available: {list(self.plugins.keys())}"
            )
        return self.plugins[name]

    def create(
        self,
        name: str,
        weight: float = 1.0,
        group: str = "hazard",
        parameters: Optional[dict] = None
    ) -> Any:
        """Instantiate a registered plugin, matching constructor arguments safely via inspection."""
        cls = self.get(name)
        raw_params = dict(parameters or {})

        sig = inspect.signature(cls.__init__)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

        call_kwargs = {}
        extra_metadata = {}

        if issubclass(cls, ParameterPlugin):
            if "weight" in sig.parameters or accepts_kwargs:
                call_kwargs["weight"] = weight
            if "group" in sig.parameters or accepts_kwargs:
                call_kwargs["group"] = group

        for k, v in raw_params.items():
            if k in ("weight", "group") and k in call_kwargs:
                continue
            if k in sig.parameters or accepts_kwargs:
                call_kwargs[k] = v
            else:
                extra_metadata[k] = v

        instance = cls(**call_kwargs)
        if hasattr(instance, "metadata") and isinstance(instance.metadata, dict):
            instance.metadata.update(extra_metadata)
        return instance

    def auto_discover(
        self,
        plugins_dir: Optional[Union[Path, str, Iterable[Union[Path, str]]]] = None,
        extra_dirs: Optional[Iterable[Union[Path, str]]] = None,
        include_custom: Optional[bool] = None,
    ):
        """Auto-discover and import all plugin modules across the resolved plugin directories."""
        import importlib.util

        dirs = resolve_plugin_dirs(
            plugins_dir,
            extra_dirs=extra_dirs,
            include_custom=include_custom,
        )

        for pdir in dirs:
            if pdir in self._discovered_dirs or not pdir.exists():
                continue

            logger.debug(f"Auto-discovering plugins in: {pdir}")
            for py_file in sorted(pdir.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue
                try:
                    mod_name = _plugin_module_name(pdir, py_file)
                    spec = importlib.util.spec_from_file_location(mod_name, py_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as e:
                    logger.debug(f"Auto-discovery notice for {py_file.name}: {e}")

            self._discovered_dirs.add(pdir)

    def list_plugins(self, base_type: Optional[Type] = None) -> List[str]:
        """List registered plugin names, optionally filtered by base class."""
        if base_type is None:
            return list(self.plugins.keys())
        return [
            name for name, cls in self.plugins.items()
            if isinstance(cls, type) and issubclass(cls, base_type)
        ]


plugin_registry = PluginRegistry()


def register_plugin(name: str):
    """Decorator to register a ParameterPlugin, RiskCalculatorPlugin, or PostProcessorPlugin."""
    return plugin_registry.register(name)


class LoaderRegistry:
    """
    Registry for DataLoaderPlugin classes with automatic discovery and topological dependency sorting.
    """

    def __init__(self):
        self.loaders: List[Type[DataLoaderPlugin]] = []
        self._discovered_dirs: set[Path] = set()
        self._auto_discovered_called: bool = False

    @property
    def _discovered(self) -> bool:
        return self._auto_discovered_called or bool(self._discovered_dirs)

    def auto_discover(
        self,
        plugins_dir: Optional[Union[Path, str, Iterable[Union[Path, str]]]] = None,
        extra_dirs: Optional[Iterable[Union[Path, str]]] = None,
        include_custom: Optional[bool] = None,
    ):
        """Discover all DataLoaderPlugin subclasses across resolved plugin loaders directories."""
        import importlib.util

        self._auto_discovered_called = True
        dirs = resolve_plugin_dirs(
            plugins_dir,
            extra_dirs=extra_dirs,
            include_custom=include_custom,
        )

        for pdir in dirs:
            loaders_dir = pdir if pdir.name == "loaders" else (pdir / "loaders").resolve()
            if loaders_dir in self._discovered_dirs:
                continue
            if not loaders_dir.exists():
                logger.debug(f"Loaders directory not found: {loaders_dir}")
                continue

            for module_file in sorted(loaders_dir.glob("*.py")):
                if module_file.name.startswith("_"):
                    continue

                try:
                    mod_name = _plugin_module_name(pdir, module_file, subpkg="loaders")
                    spec = importlib.util.spec_from_file_location(mod_name, module_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for _, obj in inspect.getmembers(module, inspect.isclass):
                        if (
                            issubclass(obj, DataLoaderPlugin)
                            and obj is not DataLoaderPlugin
                            and obj not in self.loaders
                        ):
                            self.loaders.append(obj)
                except Exception as e:
                    logger.error(f"Failed to load loader module {module_file.name}: {e}")

            self._discovered_dirs.add(loaders_dir)

    def register(self, loader_class: Type[DataLoaderPlugin]):
        if loader_class not in self.loaders:
            self.loaders.append(loader_class)

    def _resolve_order(self, active_loaders: List[Type[DataLoaderPlugin]]) -> List[Type[DataLoaderPlugin]]:
        """Resolve loader execution order using Kahn's topological sort algorithm."""
        from collections import defaultdict, deque

        provider_map: Dict[str, Type[DataLoaderPlugin]] = {}
        for loader in active_loaders:
            for key in loader.provides:
                provider_map[key] = loader

        graph: Dict[Type, set[Type]] = defaultdict(set)
        in_degree: Dict[Type, int] = {loader: 0 for loader in active_loaders}

        for loader in active_loaders:
            for req in loader.requires:
                if req in provider_map:
                    dependency = provider_map[req]
                    if dependency is not loader:
                        graph[dependency].add(loader)
                        in_degree[loader] += 1
                else:
                    logger.warning(f"{loader.__name__} requires '{req}' but no loader provides it")

        queue = deque(
            sorted(
                [l for l in active_loaders if in_degree[l] == 0],
                key=lambda x: x.__name__
            )
        )
        ordered = []

        while queue:
            loader = queue.popleft()
            ordered.append(loader)

            for dependent in sorted(graph[loader], key=lambda x: x.__name__):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(ordered) != len(active_loaders):
            unresolved = [l.__name__ for l in active_loaders if l not in ordered]
            raise CircularDependencyError(f"Circular dependency detected among: {unresolved}")

        return ordered

    def load_all(self, config, registry: DataRegistry):
        """Execute all applicable data loaders in dependency-resolved order."""
        if not self._discovered:
            self.auto_discover()

        active = [l for l in self.loaders if l.can_handle(config)]
        ordered = self._resolve_order(active)

        if not ordered:
            return

        logger.info("\n" + "=" * 70)
        logger.info("DATA LOADING PHASE")
        logger.info("=" * 70)
        logger.info(f"Load order: {' -> '.join(l.__name__ for l in ordered)}")

        executed = 0
        for loader_class in ordered:
            provides_str = ', '.join(loader_class.provides) or 'none'
            requires_str = ', '.join(loader_class.requires) or 'none'
            logger.info(f"\n{loader_class.__name__} (provides=[{provides_str}], requires=[{requires_str}])")
            try:
                loader = loader_class(config)
                loader.load(registry)
                executed += 1
            except Exception as e:
                logger.error(f"Loader {loader_class.__name__} failed: {e}")

        logger.info("\n" + "=" * 70)
        logger.info(f"DATA LOADING COMPLETE ({executed}/{len(ordered)} loader(s) executed)")
        logger.info("=" * 70 + "\n")

    def list_loaders(self) -> List[str]:
        return [loader.__name__ for loader in self.loaders]

    def __len__(self) -> int:
        return len(self.loaders)
