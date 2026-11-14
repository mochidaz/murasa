import sys
import argparse
from pathlib import Path

from murasa.core.config import EngineConfig
from murasa.engine.risk_engine import RiskEngine
from murasa.core.base import LoaderRegistry
from murasa.plugins import VectorExplainabilityPlugin
from murasa.core.logger import logger, log_section, log_success, log_error


def parse_args():
    parser = argparse.ArgumentParser(description="Landslide Susceptibility Analysis Bandung Raya")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("examples/landslide_bandung_config.yaml"),
        help="Path to YAML config file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_section("Landslide susceptibility pipeline")
    logger.info(f"Config: {args.config}")

    if not args.config.exists():
        log_error(f"Config not found: {args.config}")
        sys.exit(1)

    config = EngineConfig.from_yaml(args.config)
    engine = RiskEngine(config)

    engine.auto_register_from_config()
    loader_registry = LoaderRegistry()
    loader_registry.auto_discover(Path(__file__).parent.parent / "plugins")
    loader_registry.load_all(config, engine.registry)

    engine.auto_register_parameters_from_config()

    engine.register_post_processor(
        VectorExplainabilityPlugin(
            labels=['Low', 'Medium', 'High', 'Very High'],
            output_prefix="landslide_risk",
            title_label="Tingkat Kerentanan Longsor",
            explain_threshold=0.60,
            run_sensitivity=False,
        )
    )

    engine.print_summary()
    try:
        engine.setup_grid()
        engine.set_study_area()
        engine.calculate_risk()
        engine.export_all()
        log_success(f"Results saved to: {config.output.output_dir.resolve()}")
    except Exception as e:
        log_error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
