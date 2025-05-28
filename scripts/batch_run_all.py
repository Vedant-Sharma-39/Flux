# E:/Flux/scripts/batch_run_all.py
import subprocess
from pathlib import Path
import os
import sys  # To get python executable more reliably
import yaml
import shutil  # For cleaning up temp configs
import itertools  # For creating parameter combinations
import logging
import numpy as np
from typing import Tuple, Optional
# Setup basic logging for this batch script
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s - BATCH_RUN] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH_STR = str(PROJECT_ROOT / "config" / "default_config.yaml")
TEMP_CONFIG_DIR = PROJECT_ROOT / "config" / "temp_experiment_configs"
RUN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_experiment.py"

# Determine Python executable (handles virtual environments)
if hasattr(sys, "real_prefix") or (
    hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
):
    # We are in a virtual environment
    PYTHON_EXE = (
        Path(sys.prefix) / "Scripts" / "python.exe"
        if os.name == "nt"
        else Path(sys.prefix) / "bin" / "python"
    )
else:
    # Not in a virtual environment, use the Python that's running this script
    PYTHON_EXE = Path(sys.executable)

if not PYTHON_EXE.exists():
    logger.error(
        f"Python executable not found at {PYTHON_EXE}. Please check your environment."
    )
    sys.exit(1)
if not RUN_SCRIPT_PATH.exists():
    logger.error(f"run_experiment.py not found at {RUN_SCRIPT_PATH}.")
    sys.exit(1)


def generate_experiment_config(
    base_config: dict, updates: dict, experiment_name: str
) -> dict:
    """
    Creates a new config dict by updating a base config.
    Deep updates nested dictionaries.
    """
    new_config = {k: v for k, v in base_config.items()}  # Shallow copy for top level

    def _deep_update_dict(target, source):
        for key, value in source.items():
            if (
                isinstance(value, dict)
                and key in target
                and isinstance(target[key], dict)
            ):
                _deep_update_dict(target[key], value)
            else:
                target[key] = value

    _deep_update_dict(new_config, updates)
    new_config["experiment_name"] = experiment_name  # Ensure experiment_name is set
    return new_config


def run_single_experiment(temp_config_filepath: Path):
    """Runs a single experiment using the generated temporary config file."""
    cmd = [
        str(PYTHON_EXE),
        str(RUN_SCRIPT_PATH),
        "--exp_config",
        str(temp_config_filepath),  # Use the temp config as the override
        "--config",
        DEFAULT_CONFIG_PATH_STR,  # Always use the default as base
    ]
    logger.info(f"Running: {' '.join([str(c) for c in cmd])}")
    try:
        process = subprocess.run(
            cmd, check=True, text=True, capture_output=True, cwd=PROJECT_ROOT
        )
        logger.info(f"Successfully ran: {temp_config_filepath.name}")
        logger.debug(f"Stdout:\n{process.stdout}")
        if process.stderr:
            logger.warning(f"Stderr for {temp_config_filepath.name}:\n{process.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(
            f"ERROR running {temp_config_filepath.name}: Return code {e.returncode}"
        )
        logger.error(f"Stdout:\n{e.stdout}")
        logger.error(f"Stderr:\n{e.stderr}")
    except FileNotFoundError:
        logger.error(
            f"ERROR: Python executable or script not found. Check PYTHON_EXE: {PYTHON_EXE}"
        )


def main():
    # --- Define Parameter Sweeps ---
    # RQ1.1: W_band sweep
    w_bands_rq1 = [5.0, 10.0, 20.0, 40.0, 80.0]
    # Strategies for RQ1.1 (prob_bet, g1, g2 - assuming g1/g2 are fixed from default for this specific RQ part)
    # For Responsive strategies, the *effective* G-growth rate is determined by the single prototype they inherit.
    # So, for Responsive HG, its g_rate is default_config's g_rate_prototype_2.
    # For Responsive LL, its g_rate is default_config's g_rate_prototype_1.
    strategies_rq1 = [
        {"label": "ResponsiveHG", "prob_bet": 0.0},  # Will use g_rate_prototype_2
        {"label": "ResponsiveLL", "prob_bet": 1.0},  # Will use g_rate_prototype_1
        {"label": "BetHedging0.5", "prob_bet": 0.5},
    ]
    num_replicates = 3  # Number of replicates for each condition

    # RQ2.1: prob_bet sweep for selected W_bands
    w_bands_for_prob_sweep_rq2 = [10.0, 40.0]  # Example W_bands
    prob_bets_rq2 = np.linspace(0.0, 1.0, 11).tolist()  # 0.0, 0.1, ..., 1.0

    # RQ2.2: Trait distinctness
    # Fixed W_band (e.g. 40.0), fixed prob_bet (e.g. 0.5)
    w_band_rq2_2 = 40.0
    prob_bet_rq2_2 = 0.5
    trait_distinctness_scenarios = [
        {
            "label": "LowDistinct",
            "g1": 0.20,
            "g2": 0.25,
        },  # Corresponding lags will be closer
        {
            "label": "HighDistinct",
            "g1": 0.1,
            "g2": 0.5,
        },  # Default, lags will be further apart
    ]

    # RQ3.1: Trade-off slope sweep
    # Fixed W_band, prob_bet (for BH), fixed g_rates
    w_band_rq3 = 40.0
    strategies_rq3 = [
        {"label": "ResponsiveHG", "prob_bet": 0.0},
        {"label": "BetHedging0.5", "prob_bet": 0.5},
    ]
    tradeoff_slopes_rq3 = [10.0, 20.0, 40.0]  # T_lag_min can be fixed from default

    # --- Load Base Config ---
    try:
        with open(DEFAULT_CONFIG_PATH_STR, "r") as f:
            base_config_content = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Could not load base config {DEFAULT_CONFIG_PATH_STR}: {e}")
        return

    # --- Create Temp Config Directory ---
    TEMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Temporary configs will be saved in: {TEMP_CONFIG_DIR}")

    configs_to_run: List[Tuple[str, dict]] = (
        []
    )  # List of (experiment_name, config_dict)

    # --- Generate Configs for RQ1.1 ---
    logger.info("Generating configs for RQ1.1 (W_band sweep)...")
    for w_band in w_bands_rq1:
        for strat_info in strategies_rq1:
            for rep in range(1, num_replicates + 1):
                exp_name = f"{strat_info['label']}_W{w_band:.0f}_Rep{rep}"
                updates = {
                    "environment": {"W_band": w_band},
                    "strategies": {
                        "prob_daughter_inherits_prototype_1": strat_info["prob_bet"]
                    },
                    "visualization": {
                        "visualization_enabled": False
                    },  # Disable for fast sweeps
                }
                # For specific W_bands, enable visualization if desired for RQ1.2 snapshots
                if w_band in [10.0, 40.0]:  # Example
                    updates["visualization"]["visualization_enabled"] = True
                    updates["visualization"][
                        "animation_color_mode"
                    ] = "REMAINING_LAG_TIME"

                cfg = generate_experiment_config(base_config_content, updates, exp_name)
                configs_to_run.append((exp_name, cfg))

    # --- Generate Configs for RQ2.1 ---
    logger.info("Generating configs for RQ2.1 (prob_bet sweep)...")
    for w_band in w_bands_for_prob_sweep_rq2:
        for prob in prob_bets_rq2:
            for rep in range(1, num_replicates + 1):
                exp_name = f"ProbSweep_W{w_band:.0f}_P{prob:.1f}_Rep{rep}"
                updates = {
                    "environment": {"W_band": w_band},
                    "strategies": {"prob_daughter_inherits_prototype_1": prob},
                    "visualization": {"visualization_enabled": False},
                }
                cfg = generate_experiment_config(base_config_content, updates, exp_name)
                configs_to_run.append((exp_name, cfg))

    # --- Generate Configs for RQ2.2 ---
    logger.info("Generating configs for RQ2.2 (Trait Distinctness)...")
    for scenario in trait_distinctness_scenarios:
        for rep in range(1, num_replicates + 1):
            exp_name = f"TraitDistinct_{scenario['label']}_W{w_band_rq2_2:.0f}_P{prob_bet_rq2_2:.1f}_Rep{rep}"
            updates = {
                "environment": {"W_band": w_band_rq2_2},
                "strategies": {
                    "prob_daughter_inherits_prototype_1": prob_bet_rq2_2,
                    "g_rate_prototype_1": scenario["g1"],
                    "g_rate_prototype_2": scenario["g2"],
                },
                "visualization": {
                    "visualization_enabled": True,
                    "animation_color_mode": "INHERENT_LAG_GL",
                },
            }
            cfg = generate_experiment_config(base_config_content, updates, exp_name)
            configs_to_run.append((exp_name, cfg))

    # --- Generate Configs for RQ3.1 ---
    logger.info("Generating configs for RQ3.1 (Trade-off Slope)...")
    for slope in tradeoff_slopes_rq3:
        for strat_info in strategies_rq3:
            for rep in range(1, num_replicates + 1):
                exp_name = f"TradeoffSlope_S{slope:.0f}_{strat_info['label']}_W{w_band_rq3:.0f}_Rep{rep}"
                updates = {
                    "environment": {"W_band": w_band_rq3},
                    "strategies": {
                        "prob_daughter_inherits_prototype_1": strat_info["prob_bet"]
                    },
                    "trade_off_params": {
                        "slope": slope
                    },  # Assumes T_lag_min comes from default
                    "visualization": {"visualization_enabled": False},
                }
                cfg = generate_experiment_config(base_config_content, updates, exp_name)
                configs_to_run.append((exp_name, cfg))

    # --- Write temp configs and run simulations ---
    logger.info(f"Total configurations to run: {len(configs_to_run)}")
    for i, (exp_name, config_dict) in enumerate(configs_to_run):
        temp_config_filename = f"temp_config_{exp_name}.yaml"
        temp_config_filepath = TEMP_CONFIG_DIR / temp_config_filename

        logger.info(f"\n--- Preparing Run {i+1}/{len(configs_to_run)}: {exp_name} ---")
        try:
            with open(temp_config_filepath, "w") as f:
                yaml.dump(config_dict, f, sort_keys=False)
            logger.info(f"Saved temporary config to {temp_config_filepath}")

            run_single_experiment(temp_config_filepath)

        except Exception as e:
            logger.error(
                f"Error processing or running experiment {exp_name}: {e}", exc_info=True
            )
        finally:
            # Optional: Clean up temp config file immediately or all at the end
            # if temp_config_filepath.exists(): temp_config_filepath.unlink()
            pass

    # Optional: Clean up the entire temp_config_dir after all runs
    # logger.info(f"Cleaning up temporary config directory: {TEMP_CONFIG_DIR}")
    # try:
    #     shutil.rmtree(TEMP_CONFIG_DIR)
    # except Exception as e:
    #     logger.error(f"Could not remove temporary config directory {TEMP_CONFIG_DIR}: {e}")

    logger.info("All batch simulation runs attempted.")


if __name__ == "__main__":
    # Make sure to activate venv and run `pip install -e .` in project root first
    # Ensure PYTHON_EXE is correctly identified or hardcode if necessary.
    main()
