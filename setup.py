# E:/Flux/scripts/run_experiment.py
import argparse
from pathlib import Path
import traceback
import logging  # For basic logging from this script before engine's logger takes over

# Imports will work directly if 'pip install -e .' was successful
# and your setup.py makes 'src' and its submodules discoverable.
from src.utils.config_loader import load_config, ConfigurationError
from src.simulation.simulation_engine import SimulationEngine
from src.core.data_structures import SimulationConfig  # For type hinting


def main():
    parser = argparse.ArgumentParser(
        description="Run a microbial colony simulation experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows default values in help
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",  # Default path relative to project root
        help="Path to the main/default configuration YAML file (relative to project root).",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default=None,  # No experiment override by default
        help="Path to an experiment-specific override configuration YAML file (relative to project root).",
    )
    # Example for adding a CLI override directly:
    # parser.add_argument(
    #     "--max_time",
    #     type=float,
    #     default=None,
    #     help="Override max_simulation_time from config files."
    # )

    args = parser.parse_args()

    # Basic logging setup for this script itself.
    # The SimulationEngine will set up its own more detailed logger.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    script_logger = logging.getLogger(
        __name__
    )  # Use the script's own name for its logger

    try:
        script_logger.info("Starting experiment run...")
        script_logger.info(f"Using default config: {args.config}")
        if args.exp_config:
            script_logger.info(f"Using experiment override config: {args.exp_config}")

        # Construct paths relative to the project root for load_config.
        # load_config is designed to find the project root from its own location (src/utils)
        # and interpret these paths correctly.
        default_config_path_arg = Path(args.config)
        experiment_config_path_arg = Path(args.exp_config) if args.exp_config else None

        # Placeholder for CLI overrides dictionary
        cli_overrides = {}
        # if args.max_time is not None:
        #     cli_overrides['simulation'] = {'max_simulation_time': args.max_time}
        # Note: _deep_update in load_config needs to handle nested dicts from CLI correctly.

        script_logger.info("Loading configuration...")
        sim_config: SimulationConfig = load_config(
            default_config_path=default_config_path_arg,
            experiment_config_path=experiment_config_path_arg,
            cli_overrides=cli_overrides if cli_overrides else None,
        )

        script_logger.info(
            f"Successfully loaded configuration for experiment: {sim_config.experiment_name}"
        )

        # SimulationEngine will use its own logger configured based on sim_config
        engine = SimulationEngine(sim_config)
        engine.run()

        script_logger.info(
            f"Experiment '{sim_config.experiment_name}' completed successfully."
        )

    except ConfigurationError as e:
        script_logger.error(f"Configuration error during setup: {e}", exc_info=True)
    except ModuleNotFoundError as e:
        script_logger.error(
            f"Module not found: {e}. This usually means the project was not installed correctly "
            f"in editable mode ('pip install -e .') or there's an issue with PYTHONPATH/virtual environment.",
            exc_info=True,
        )
    except Exception as e:
        script_logger.error(
            f"An unexpected error occurred during the experiment run: {e}",
            exc_info=True,
        )
        # traceback.print_exc() # exc_info=True with logger usually suffices


if __name__ == "__main__":
    main()
