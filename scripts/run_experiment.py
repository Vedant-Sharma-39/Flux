# E:/Flux/scripts/run_experiment.py
import argparse
from pathlib import Path
import traceback
import logging

# No sys.path modification needed here!

from src.utils.config_loader import load_config, ConfigurationError
from src.simulation.simulation_engine import SimulationEngine
from src.core.data_structures import SimulationConfig  # For type hinting


def main():
    parser = argparse.ArgumentParser(
        description="Run a microbial colony simulation experiment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",  # Relative to project root
        help="Path to the main/default configuration YAML file (relative to project root).",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        help="Path to an experiment-specific override configuration YAML file (relative to project root).",
    )

    args = parser.parse_args()

    try:
        # Basic logger for this script
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s - %(filename)s:%(lineno)d] %(message)s",
        )
        script_logger = logging.getLogger(__name__)
        script_logger.info("Loading configuration...")

        # Paths for load_config should be relative to the project root (E:/Flux)
        # because load_config itself determines the project root based on its own location.
        default_config_path_arg = Path(args.config)
        experiment_config_path_arg = Path(args.exp_config) if args.exp_config else None

        sim_config: SimulationConfig = load_config(
            default_config_path=default_config_path_arg,
            experiment_config_path=experiment_config_path_arg,
        )

        script_logger.info(
            f"Successfully loaded configuration for experiment: {sim_config.experiment_name}"
        )

        engine = SimulationEngine(sim_config)
        engine.run()

    except ConfigurationError as e:
        logging.error(f"Configuration error: {e}", exc_info=True)
    except ModuleNotFoundError as e:
        # This should ideally not happen now if `pip install -e .` was successful
        logging.error(
            f"Module not found error: {e}. Ensure project was installed with 'pip install -e .'",
            exc_info=True,
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
