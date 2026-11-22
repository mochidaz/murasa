from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Iterator, Tuple
import yaml
import json
from .logger import logger, log_success, log_error

RASTER_EXTS = (
    '.tif', '.tiff', '.bil', '.nc', '.nc4', '.hdf',
    '.img', '.asc', '.sdat', '.grd', '.bsq', '.bip'
)
VECTOR_EXTS = ('.shp', '.geojson', '.gpkg', '.kml', '.json')


def infer_source_type(path: Optional[Path], default: str = "derived") -> str:
    """Infer data source type from file extension."""
    if not path:
        return default
    ext = path.suffix.lower()
    if ext in RASTER_EXTS:
        return "raster"
    if ext in VECTOR_EXTS or path.name.lower().endswith(".gdb"):
        return "vector"
    return default


@dataclass
class FactorConfig:
    """Configuration for a single risk factor."""
    name: str
    weight: float = 0.0
    source_path: Optional[Path] = None
    source_type: str = "raster"
    group: str = "hazard"
    processor: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.source_path, str):
            self.source_path = Path(self.source_path)


def _normalize_resolution_unit(unit: str) -> str:
    """Validate and canonicalize SpatialConfig.resolution_unit ('meter' or 'crs')."""
    normalized = str(unit).strip().lower()
    if normalized in ("m", "meter", "meters"):
        return "meter"
    if normalized in ("crs",):
        return "crs"
    raise ValueError(
        f"Invalid resolution_unit '{unit}'. Expected 'meter' or 'crs'."
    )


@dataclass
class SpatialConfig:
    """Spatial grid, projection, and generic study area filter configuration."""
    target_crs: str = "EPSG:4326"
    metric_crs: str = "auto"
    resolution: float = 10.0
    resolution_unit: str = "meter"
    study_area_key: str = "admin"
    reference_raster: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        target_crs: str = "EPSG:4326",
        metric_crs: str = "auto",
        resolution: float = 10.0,
        resolution_unit: str = "meter",
        study_area_key: str = "admin",
        reference_raster: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        res_val = float(resolution)
        if res_val <= 0:
            raise ValueError(f"Spatial resolution must be positive, got {resolution}")
        self.target_crs = target_crs
        self.metric_crs = metric_crs
        self.resolution = res_val
        self.resolution_unit = _normalize_resolution_unit(resolution_unit)
        self.study_area_key = study_area_key
        self.reference_raster = reference_raster
        self.filters = dict(filters or {})
        for k, v in kwargs.items():
            if v is not None:
                self.filters[k] = v

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.filters.get(name, None)


@dataclass
class OutputConfig:
    """Output directory and export format configuration."""
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    formats: List[str] = field(default_factory=lambda: ["geojson"])
    generate_report: bool = True

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class ClassificationConfig:
    """Risk score classification configuration."""
    method: str = "jenks"
    num_classes: int = 5
    class_names: List[str] = field(default_factory=lambda: [
        "Very Low", "Low", "Moderate", "High", "Very High"
    ])
    class_colors: List[str] = field(default_factory=list)
    thresholds: Optional[List[float]] = None


class PathsConfig:
    """
    Dynamic container for data paths and directories.
    Accepts arbitrary path keys from YAML without hardcoding domain-specific attributes.
    """

    def __init__(self, base_dir: Optional[Path] = None, **kwargs):
        object.__setattr__(self, "_paths", {})
        object.__setattr__(self, "_base_dir", base_dir)
        for key, val in kwargs.items():
            self.set(key, val)

    def _resolve(self, value: Any) -> Any:
        if value is None:
            return None
        base_dir = object.__getattribute__(self, "_base_dir")
        if isinstance(value, (list, tuple)):
            return [self._resolve(v) for v in value]
        if isinstance(value, (str, Path)):
            p = Path(value)
            if base_dir and not p.is_absolute():
                return (base_dir / p).resolve()
            return p.resolve() if p.is_absolute() else p
        return value

    def set(self, key: str, value: Any) -> None:
        self._paths[key] = self._resolve(value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._paths.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._paths)

    def items(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._paths.items())

    def keys(self):
        return self._paths.keys()

    def values(self):
        return self._paths.values()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._paths.get(name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self.set(name, value)

    def __getitem__(self, key: str) -> Any:
        return self._paths[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._paths and self._paths[key] is not None

    def __bool__(self) -> bool:
        return bool(self._paths)

    def __repr__(self) -> str:
        return f"PathsConfig({self._paths})"


class WeightsConfig:
    """
    Dynamic container for factor weights.
    Accepts arbitrary factor names from YAML.
    """

    def __init__(self, **kwargs):
        object.__setattr__(self, "_weights", {
            k: float(v) for k, v in kwargs.items() if v is not None
        })

    def get(self, key: str, default: float = 0.0) -> float:
        return self._weights.get(key, default)

    def set(self, key: str, value: float) -> None:
        self._weights[key] = float(value)

    def total(self) -> float:
        return sum(self._weights.values())

    def as_dict(self) -> Dict[str, float]:
        return dict(self._weights)

    def items(self) -> Iterator[Tuple[str, float]]:
        return iter(self._weights.items())

    def keys(self):
        return self._weights.keys()

    def values(self):
        return self._weights.values()

    def __getattr__(self, name: str) -> float:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._weights.get(name, 0.0)

    def __setattr__(self, name: str, value: float) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self.set(name, value)

    def __getitem__(self, key: str) -> float:
        return self._weights[key]

    def __setitem__(self, key: str, value: float) -> None:
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._weights

    def __bool__(self) -> bool:
        return bool(self._weights)

    def __iter__(self):
        return iter(self._weights.items())

    def __repr__(self) -> str:
        return f"WeightsConfig({self._weights})"


@dataclass
class EngineConfig:
    """Main configuration object driving the RiskEngine pipeline."""
    name: str = "unnamed_analysis"
    description: str = ""
    analysis_type: str = "susceptibility"
    risk_method: str = "weighted_sum"

    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)

    factors: Dict[str, FactorConfig] = field(default_factory=dict)

    parameters: Dict[str, Any] = field(default_factory=dict)
    paths: PathsConfig = field(default_factory=PathsConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    plugin_parameters: Dict[str, Any] = field(default_factory=dict)
    post_processors: List[Any] = field(default_factory=list)
    plugin_dirs: List[Path] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.plugin_dirs, (str, Path)):
            self.plugin_dirs = [Path(self.plugin_dirs)]
        else:
            self.plugin_dirs = [Path(p) for p in self.plugin_dirs if p]

    def get_weights(self) -> Dict[str, float]:
        if self.weights:
            return {k: v for k, v in self.weights.items() if v > 0}
        return {name: f.weight for name, f in self.factors.items()}

    def get_factor_path(self, factor_name: str) -> Optional[Path]:
        if factor_name in self.paths and isinstance(self.paths.get(factor_name), Path):
            return self.paths.get(factor_name)
        if factor_name in self.factors:
            return self.factors[factor_name].source_path
        return None

    def validate_weights(self) -> None:
        if not self.factors:
            return
        total = sum(f.weight for f in self.factors.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")

    def normalize_weights(self) -> None:
        total = sum(f.weight for f in self.factors.values())
        if total > 0:
            for factor in self.factors.values():
                factor.weight /= total
            log_success(f"Weights normalized (total was {total:.3f})")

    def validate(self) -> None:
        self.validate_weights()
        self.output.output_dir.mkdir(parents=True, exist_ok=True)

        for name, factor in self.factors.items():
            if factor.source_path and not factor.source_path.exists():
                log_error(f"Factor path not found: {name}: {factor.source_path}")

        log_success("Configuration validated")

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'EngineConfig':
        """Load configuration from a YAML file supporting both paths/weights and factors schemas."""
        logger.info(f"Loading config from: {yaml_path}")
        yaml_path = Path(yaml_path)
        base_dir = yaml_path.parent.resolve()

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f) or {}

        project = data.get('project', {})
        config = cls(
            name=project.get('name', data.get('name', 'unnamed')),
            description=project.get('description', data.get('description', '')),
            analysis_type=project.get('analysis_type', data.get('analysis_type', 'susceptibility')),
            risk_method=data.get('risk_method', 'weighted_sum'),
        )

        if 'spatial' in data:
            config.spatial = SpatialConfig(**data['spatial'])

        if 'output' in data:
            out_data = data['output']
            out_dir = Path(out_data.get('dir', './output'))
            if not out_dir.is_absolute():
                out_dir = (base_dir / out_dir).resolve()
            config.output = OutputConfig(
                output_dir=out_dir,
                formats=out_data.get('formats', ['geojson']),
                generate_report=out_data.get('generate_report', True)
            )

        if 'classification' in data:
            config.classification = ClassificationConfig(**data['classification'])

        config.parameters = data.get('parameters', {})
        config.plugin_parameters = config.parameters
        config.post_processors = data.get('post_processors', [])

        raw_plugin_dirs = data.get('plugin_dirs', [])
        if isinstance(raw_plugin_dirs, (str, Path)):
            raw_plugin_dirs = [raw_plugin_dirs]
        for pdir in raw_plugin_dirs or []:
            if not pdir:
                continue
            p = Path(pdir)
            config.plugin_dirs.append(p if p.is_absolute() else (base_dir / p).resolve())

        if 'paths' in data or 'weights' in data:
            config._load_paths_weights_format(data, base_dir)
        else:
            config._load_factors_format(data, base_dir)

        config.validate()
        return config

    def _load_paths_weights_format(self, data: dict, base_dir: Path) -> None:
        """Populate dynamic paths, weights, and factor configurations from YAML."""
        paths_data = data.get('paths', {})
        self.paths = PathsConfig(base_dir=base_dir, **paths_data)

        if 'output_dir' in self.paths:
            self.output = OutputConfig(output_dir=self.paths.output_dir)

        weights_data = data.get('weights', {})
        self.weights = WeightsConfig(**weights_data)

        for factor_name, weight in self.weights.items():
            if weight <= 0:
                continue

            factor_params = self.parameters.get(factor_name, {})
            is_param_dict = isinstance(factor_params, dict)
            source_key = factor_params.get('source', factor_name) if is_param_dict else factor_name
            source_path = self.paths.get(source_key)

            stype = infer_source_type(source_path if isinstance(source_path, Path) else None)
            group = factor_params.get('group', 'hazard') if is_param_dict else 'hazard'
            processor = (factor_params.get('processor') or factor_params.get('plugin')) if is_param_dict else None

            self.factors[factor_name] = FactorConfig(
                name=factor_name,
                weight=weight,
                source_path=source_path if isinstance(source_path, Path) else None,
                source_type=stype,
                group=group,
                processor=processor,
                parameters=factor_params if is_param_dict else {}
            )

    def _load_factors_format(self, data: dict, base_dir: Path) -> None:
        """Populate configuration from explicit factors dictionary in YAML."""
        weights_dict = {}
        paths_dict = {}

        for name, fdata in data.get('factors', {}).items():
            source_path = None
            if 'path' in fdata and fdata['path']:
                p = Path(fdata['path'])
                source_path = p if p.is_absolute() else (base_dir / p).resolve()
                paths_dict[name] = source_path

            weight = float(fdata.get('weight', 0.0))
            weights_dict[name] = weight
            params = fdata.get('parameters', {}) or {}
            processor = (
                fdata.get('processor')
                or fdata.get('plugin')
                or (params.get('processor') or params.get('plugin') if isinstance(params, dict) else None)
            )

            self.factors[name] = FactorConfig(
                name=name,
                weight=weight,
                source_path=source_path,
                source_type=fdata.get('type', infer_source_type(source_path, 'raster')),
                group=fdata.get('group', 'hazard'),
                processor=processor,
                parameters=params if isinstance(params, dict) else {}
            )

        self.paths = PathsConfig(base_dir=base_dir, **paths_dict)
        self.weights = WeightsConfig(**weights_dict)

    def to_yaml(self, yaml_path: Path) -> None:
        """Serialize configuration to a YAML file."""
        data = {
            'project': {
                'name': self.name,
                'description': self.description,
                'analysis_type': self.analysis_type,
            },
            'risk_method': self.risk_method,
            'spatial': {
                'target_crs': self.spatial.target_crs,
                'metric_crs': self.spatial.metric_crs,
                'resolution': self.spatial.resolution,
                'resolution_unit': self.spatial.resolution_unit,
                'reference_raster': self.spatial.reference_raster,
                'study_area_key': self.spatial.study_area_key,
                **self.spatial.filters,
            },
            'paths': {
                k: ([str(i) for i in v] if isinstance(v, list) else str(v))
                for k, v in self.paths.items() if v is not None
            },
            'weights': self.weights.as_dict(),
            'parameters': self.parameters,
            'classification': {
                'method': self.classification.method,
                'num_classes': self.classification.num_classes,
                'class_names': self.classification.class_names,
                'class_colors': self.classification.class_colors,
                'thresholds': self.classification.thresholds,
            },
        }
        if self.plugin_dirs:
            data['plugin_dirs'] = [str(p) for p in self.plugin_dirs]

        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        log_success(f"Configuration saved to {yaml_path}")

    def to_json(self, json_path: Path) -> None:
        """Serialize configuration to a JSON file."""
        data = {
            'name': self.name,
            'description': self.description,
            'analysis_type': self.analysis_type,
            'paths': {
                k: ([str(i) for i in v] if isinstance(v, list) else str(v))
                for k, v in self.paths.items() if v is not None
            },
            'weights': self.weights.as_dict(),
            'parameters': self.parameters,
        }

        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        log_success(f"Configuration saved to {json_path}")
