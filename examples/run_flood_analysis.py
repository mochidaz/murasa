"""
Flood susceptibility analysis pipeline driven by EngineConfig and unified plugin auto-discovery.
"""

import sys
import argparse
from pathlib import Path

from murasa.core.config import EngineConfig
from murasa.engine.risk_engine import RiskEngine
from murasa.core.base import LoaderRegistry
from murasa.plugins import WaterExclusionPlugin, VectorExplainabilityPlugin
from murasa.core.logger import logger, log_section, log_success, log_error


def parse_args():
    parser = argparse.ArgumentParser(description="Flood Susceptibility Analysis")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("flood_bandung_config.yaml"),
        help="Path to YAML config file"
    )
    return parser.parse_args()


def load_config(config_path: Path) -> EngineConfig:
    log_section("Loading configuration")

    if not config_path.exists():
        log_error(f"Config not found: {config_path}")
        sys.exit(1)

    config = EngineConfig.from_yaml(config_path)

    logger.info(f"  Analysis : {config.name}")
    logger.info(f"  CRS      : {config.spatial.target_crs}")
    logger.info(f"  Resolution: {config.spatial.resolution}m")

    if config.weights:
        logger.info(f"  Weights  : {config.weights.as_dict()}")

    return config


def main():
    args = parse_args()

    log_section("Flood susceptibility pipeline")
    logger.info(f"Config: {args.config}")

    config = load_config(args.config)
    engine = RiskEngine(config)

    engine.auto_register_from_config()
    loader_registry = LoaderRegistry()
    loader_registry.auto_discover(Path(__file__).parent.parent / "plugins")
    loader_registry.load_all(config, engine.registry)

    engine.auto_register_parameters_from_config()
    if not any(p.is_exclusion for p in engine.parameters):
        engine.register_parameter(WaterExclusionPlugin())

    engine.register_post_processor(VectorExplainabilityPlugin())

    engine.print_summary()

    try:
        engine.setup_grid()
        engine.set_study_area()
        engine.calculate_risk()
        engine.export_all()
        log_success(f"Results saved to: {engine.config.output.output_dir.resolve()}")
        log_section("Completed")
    except Exception as e:
        log_error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
