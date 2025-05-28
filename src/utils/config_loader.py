# microbial_colony_sim/src/utils/config_loader.py
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from src.core.data_structures import (
    SimulationConfig,
    TradeOffParams,
    VisualizationParams,
)
from src.core.exceptions import ConfigurationError


def _deep_update(source: Dict, overrides: Dict) -> Dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            _deep_update(source[key], value)
        else:
            source[key] = value
    return source


def load_config(
    default_config_path: Path = Path("config/default_config.yaml"),
    experiment_config_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> SimulationConfig:
    project_root = Path(__file__).resolve().parent.parent.parent

    abs_default_config_path = project_root / default_config_path
    if not abs_default_config_path.exists():
        raise ConfigurationError(
            f"Default configuration file not found at: {abs_default_config_path}"
        )

    try:
        with open(abs_default_config_path, "r") as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            config_data = {}
    except Exception as e:
        raise ConfigurationError(
            f"Error loading default config {abs_default_config_path}: {e}"
        )

    if experiment_config_path:
        abs_experiment_config_path = project_root / experiment_config_path
        if not abs_experiment_config_path.exists():
            raise ConfigurationError(
                f"Experiment configuration file not found at: {abs_experiment_config_path}"
            )
        try:
            with open(abs_experiment_config_path, "r") as f:
                experiment_data = yaml.safe_load(f)
            if experiment_data:
                _deep_update(config_data, experiment_data)
        except Exception as e:
            raise ConfigurationError(
                f"Error loading experiment config {abs_experiment_config_path}: {e}"
            )

    if cli_overrides:  # CLI overrides have the highest precedence
        _deep_update(config_data, cli_overrides)

    try:
        # Start with an empty dictionary for the final config that will be passed to SimulationConfig
        final_constructor_args: Dict[str, Any] = {}

        # 1. Handle nested dataclass parameters first
        trade_off_yaml_data = config_data.pop("trade_off_params", {})
        final_constructor_args["trade_off_params"] = (
            TradeOffParams(**trade_off_yaml_data)
            if trade_off_yaml_data
            else TradeOffParams()
        )

        visualization_yaml_data = config_data.pop("visualization", {})
        final_constructor_args["visualization"] = (
            VisualizationParams(**visualization_yaml_data)
            if visualization_yaml_data
            else VisualizationParams()
        )

        # 2. Flatten known sections from YAML into final_constructor_args
        # These sections contain keys that are expected as top-level fields in SimulationConfig
        yaml_sections_to_flatten = [
            "simulation",
            "environment",
            "strategies",
            "analysis",
        ]
        for section_key in yaml_sections_to_flatten:
            section_data = config_data.pop(
                section_key, {}
            )  # Remove section from config_data
            if section_data:  # If the section existed and had content
                for key, value in section_data.items():
                    # Add to final_constructor_args. If key already exists (e.g. from a previous section or CLI override that was flat),
                    # this will overwrite it. The order of _deep_update (CLI last) and this flattening matters.
                    # Since CLI overrides are applied to config_data first, they will be part of these sections if structured that way,
                    # or directly in config_data if flat.
                    final_constructor_args[key] = value

        # 3. Add any remaining top-level keys from config_data to final_constructor_args
        # These could be experiment_name, or overrides that were already flat.
        for key, value in config_data.items():
            final_constructor_args[key] = value

        # Now, final_constructor_args should only contain keys that are actual fields
        # of SimulationConfig or its nested dataclasses (which are already handled).
        # Keys like 'simulation', 'environment' themselves should no longer be in final_constructor_args.

        return SimulationConfig(**final_constructor_args)

    except TypeError as e:
        keys_in_final_args = (
            sorted(final_constructor_args.keys())
            if "final_constructor_args" in locals()
            else "unavailable"
        )
        defined_sim_config_fields = (
            [f.name for f in fields(SimulationConfig)]
            if "fields" in locals()
            else "unavailable"
        )  # requires `from dataclasses import fields`
        raise ConfigurationError(
            f"Error creating SimulationConfig: {e}. "
            f"This often means an unexpected keyword argument was passed. "
            f"Check if all keys in your YAML configuration map to fields in SimulationConfig. "
            f"Keys passed to SimulationConfig constructor: {keys_in_final_args}. "
            f"Expected SimulationConfig fields: {defined_sim_config_fields}"
        )
    except Exception as e:
        raise ConfigurationError(f"Unexpected error creating SimulationConfig: {e}")


# Example usage (if __name__ == "__main__": block)
if __name__ == "__main__":
    from dataclasses import fields  # For debugging message

    print("--- Testing Config Loader ---")
    try:
        default_sim_config = load_config()
        print("\n--- Default Config Loaded ---")
        print(default_sim_config)
        assert default_sim_config.dt == 0.1  # From simulation section in default.yaml
        assert (
            default_sim_config.W_band == 10.0
        )  # From environment section in default.yaml

        print("\n--- Loading Experiment Config (Responsive Low Lag) ---")
        exp_path_rel_to_project_root = Path(
            "config/experiment_configs/responsive_low_lag.yaml"
        )
        exp_sim_config = load_config(
            experiment_config_path=exp_path_rel_to_project_root
        )
        print(exp_sim_config)
        assert exp_sim_config.prob_daughter_inherits_prototype_1 == 1.0
        assert exp_sim_config.experiment_name == "responsive_low_lag"

        print("\n--- Loading Default with Flat CLI Overrides ---")
        cli_data_flat = {
            "dt": 0.05,  # Overrides simulation.dt
            "lambda_L_fixed_rate": 0.15,  # Overrides strategies.lambda_L_fixed_rate
            "experiment_name": "cli_override_test",
            "W_band": 15.0,  # Overrides environment.W_band
        }
        cli_config = load_config(cli_overrides=cli_data_flat)
        print(cli_config)
        assert cli_config.dt == 0.05
        assert cli_config.lambda_L_fixed_rate == 0.15
        assert cli_config.experiment_name == "cli_override_test"
        assert cli_config.W_band == 15.0

        print(
            "\n--- Loading Default with Nested CLI Overrides (mimicking structured input) ---"
        )
        # This tests if _deep_update correctly handles nested overrides before flattening
        cli_data_nested = {
            "simulation": {"dt": 0.02},
            "strategies": {"g_rate_prototype_1": 0.11},
            "visualization": {
                "hex_pixel_size": 20.0
            },  # Override nested dataclass param
        }
        cli_nested_config = load_config(cli_overrides=cli_data_nested)
        print(cli_nested_config)
        assert cli_nested_config.dt == 0.02
        assert cli_nested_config.g_rate_prototype_1 == 0.11
        assert cli_nested_config.visualization.hex_pixel_size == 20.0

    except ConfigurationError as e:
        print(f"Config Loader Test Error: {e}")
    except Exception as e:
        print(f"Unexpected error in Config Loader Test: {e}")
        import traceback

        traceback.print_exc()
