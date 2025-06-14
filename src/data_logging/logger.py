# src/data_logging/logger.py

import csv
import json
from pathlib import Path
from typing import List, Tuple, Optional, Any
import dataclasses
from enum import Enum
import numpy as np

from ..core.shared_types import SimulationParameters, HexCoord, Nutrient, Phenotype
from ..core.cell import Cell


class DataLogger:
    """Handles logging of simulation data to files."""

    def __init__(self, output_dir: Path, sim_params: SimulationParameters):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sim_params = sim_params
        self._save_parameters()

        # Main population summary
        self.population_summary_path = self.output_dir / "population_summary.csv"
        self._population_summary_file = None
        self._population_summary_writer = None
        self._init_population_summary_log()

        # Log for nutrient band transition events
        self.transition_log_path = self.output_dir / "nutrient_transitions.csv"
        self._transition_log_file = None
        self._transition_log_writer = None
        self._init_transition_log()

        self.event_log_path = self.output_dir / "events.log"
        self.snapshot_data_dir = self.output_dir / "grid_snapshots_data"
        if sim_params.save_grid_data_interval is not None and sim_params.save_grid_data_interval > 0:  # type: ignore
            self.snapshot_data_dir.mkdir(exist_ok=True)  # Pre-create if used

    def _save_parameters(self) -> None:
        """Saves the simulation parameters to a JSON file."""
        params_path = self.output_dir / "simulation_parameters.json"
        # Use a custom encoder for Enums and other non-serializable types if needed
        # For dataclasses, asdict works well but enums become their values.
        # We want enum names.
        params_as_dict = dataclasses.asdict(self.sim_params)
        serializable_params = {}
        for key, value in params_as_dict.items():
            if isinstance(value, Enum):
                serializable_params[key] = value.name
            elif key == "nutrient_bands" and isinstance(value, list):
                serializable_params[key] = [
                    (r_sq, nt.name if isinstance(nt, Enum) else str(nt))
                    for r_sq, nt in value
                ]
            else:
                serializable_params[key] = value
        try:
            with open(params_path, "w") as f:
                json.dump(serializable_params, f, indent=4)
        except TypeError as e:
            print(
                f"Error serializing parameters to JSON: {e}. Parameters: {serializable_params}"
            )
        except IOError as e:
            print(f"Error writing parameters to {params_path}: {e}")

    def _init_population_summary_log(self) -> None:
        """Initializes the CSV file for population summary logging."""
        header = [
            "time_step",
            "simulation_time",
            "total_cells",
            "total_G_phenotype_count",
            "total_P_phenotype_count",
            "frontier_cell_count",
            "frontier_G_phenotype_count",
            "frontier_P_phenotype_count",
            "fraction_P_on_frontier",
            "max_colony_radius_cartesian",
            "observed_interfaces_frontier",
            "fmi_frontier",
            "fmi_random_baseline_frontier",
            "angular_coverage_entropy_P",
        ]
        try:
            # Open in 'w' to truncate/create, then 'a' for appending
            with open(self.population_summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
            self._population_summary_file = open(
                self.population_summary_path, "a", newline="", buffering=1
            )  # Line buffered
            self._population_summary_writer = csv.writer(self._population_summary_file)
        except IOError as e:
            print(f"Error initializing {self.population_summary_path}: {e}")
            self._population_summary_writer = None  # Ensure it's None if init fails

    def _init_transition_log(self) -> None:
        """Initializes the CSV file for nutrient transition logging."""
        header = [
            "time_step",
            "simulation_time",
            "event_type",
            "from_nutrient",
            "to_nutrient",
            "frontier_cell_count_before",
            "frontier_P_cell_count_before",
            "fraction_P_at_frontier_before",
            "observed_interfaces_before",
            "fmi_before",
            "angular_coverage_entropy_P_before",
            "time_spent_in_N2_band",
        ]
        try:
            with open(self.transition_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
            self._transition_log_file = open(
                self.transition_log_path, "a", newline="", buffering=1
            )  # Line buffered
            self._transition_log_writer = csv.writer(self._transition_log_file)
        except IOError as e:
            print(f"Error initializing {self.transition_log_path}: {e}")
            self._transition_log_writer = None

    def log_population_state(
        self, time_step: int, simulation_time: float, **kwargs: Any
    ) -> None:
        """Logs aggregate population data for the current time step."""
        if not self._population_summary_writer:
            print(
                f"Warning: Population summary log writer not initialized. Cannot log state at t={simulation_time}."
            )
            return
        try:
            # Order must match header
            row = [
                time_step,
                f"{simulation_time:.2f}",
                kwargs.get("total_cells"),
                kwargs.get("total_G_phenotype_count"),
                kwargs.get("total_P_phenotype_count"),
                kwargs.get("frontier_cell_count"),
                kwargs.get("frontier_G_phenotype_count"),
                kwargs.get("frontier_P_phenotype_count"),
                f"{kwargs.get('fraction_P_on_frontier', np.nan):.4f}",
                f"{kwargs.get('max_radius', np.nan):.2f}",
                kwargs.get("observed_interfaces"),
                f"{kwargs.get('fmi', np.nan):.4f}",
                f"{kwargs.get('fmi_random', np.nan):.4f}",
                f"{kwargs.get('angular_coverage_entropy_P', np.nan):.4f}",
            ]
            # Replace None with "NaN" for CSV if appropriate, or handle missing keys
            row = ["NaN" if v is None else v for v in row]
            self._population_summary_writer.writerow(row)
        except (
            Exception
        ) as e:  # Catch broader exceptions during row construction or writing
            print(f"Error writing to population_summary.csv: {e}. Data: {kwargs}")

    def log_nutrient_transition(
        self, time_step: int, simulation_time: float, **kwargs: Any
    ) -> None:
        """Logs data related to nutrient transitions."""
        if not self._transition_log_writer:
            print(
                f"Warning: Transition log writer not initialized. Cannot log transition at t={simulation_time}."
            )
            return
        try:
            from_nutrient_val = kwargs.get("from_nutrient")
            to_nutrient_val = kwargs.get("to_nutrient")
            row = [
                time_step,
                f"{simulation_time:.2f}",
                kwargs.get("event_type"),
                (
                    from_nutrient_val.name
                    if isinstance(from_nutrient_val, Enum)
                    else str(from_nutrient_val)
                ),
                (
                    to_nutrient_val.name
                    if isinstance(to_nutrient_val, Enum)
                    else str(to_nutrient_val)
                ),
                kwargs.get("frontier_cells_before", -1),
                kwargs.get("frontier_P_cells_before", -1),
                f"{kwargs.get('fraction_P_before', -1.0):.4f}",
                kwargs.get("interfaces_before", -1),
                f"{kwargs.get('fmi_before', -1.0):.4f}",
                f"{kwargs.get('angular_coverage_entropy_P_before', -1.0):.4f}",
                f"{kwargs.get('time_in_N2_band', -1.0):.2f}",
            ]
            row = ["NaN" if v is None else v for v in row]
            self._transition_log_writer.writerow(row)
        except Exception as e:
            print(f"Error writing to nutrient_transitions.csv: {e}. Data: {kwargs}")

    def log_event(self, message: str, simulation_time: Optional[float] = None) -> None:
        """Logs a generic event or message to events.log."""
        try:
            with open(self.event_log_path, "a", buffering=1) as f:  # Line buffered
                time_str = (
                    f"Time {simulation_time:.2f}: "
                    if simulation_time is not None
                    else ""
                )
                f.write(f"{time_str}{message}\n")
        except IOError as e:
            print(f"Error writing to event log {self.event_log_path}: {e}")

    def log_grid_snapshot_data(
        self,
        time_step: int,
        # List of (HexCoord, Cell_Object, is_frontier_bool)
        cells_data_for_snapshot: List[Tuple[HexCoord, Cell, bool]],
    ) -> None:
        """Saves the state of all cells on the grid, including frontier status."""
        if (
            not self.snapshot_data_dir.exists()
        ):  # Should have been created in __init__ if needed
            print(
                f"Warning: Snapshot data directory {self.snapshot_data_dir} does not exist. Cannot save snapshot."
            )
            return

        snapshot_file = self.snapshot_data_dir / f"grid_snapshot_t_{time_step:05d}.json"
        data_to_save = []
        for coord, cell, is_frontier in cells_data_for_snapshot:
            data_to_save.append(
                {
                    "q": coord.q,
                    "r": coord.r,
                    "id": str(cell.id),
                    "parent_id": str(cell.parent_id) if cell.parent_id else None,
                    "lineage_id": str(cell.lineage_id),
                    "phenotype": cell.phenotype.name,
                    "generation": cell.generation,
                    "birth_time": f"{cell.birth_time:.2f}",  # Consistent float formatting
                    "is_frontier": is_frontier,  # Log the passed boolean
                    "time_to_next_division": (
                        f"{cell.time_to_next_division:.2f}"
                        if cell.time_to_next_division != float("inf")
                        else "inf"
                    ),
                    "is_adapting_to_N2": cell.is_adapting_to_N2,
                    "lag_N2_remaining": f"{cell.lag_N2_remaining:.2f}",
                }
            )
        try:
            with open(snapshot_file, "w") as f:
                json.dump(data_to_save, f, indent=2)
        except IOError as e:
            print(f"Error writing grid snapshot to {snapshot_file}: {e}")
        except TypeError as e:
            print(
                f"Error serializing grid snapshot data: {e}. Data example: {data_to_save[0] if data_to_save else 'N/A'}"
            )

    def close_logs(self) -> None:
        """Closes any open log files."""
        if self._population_summary_file and not self._population_summary_file.closed:
            try:
                self._population_summary_file.close()
            except IOError as e:
                print(f"Error closing {self.population_summary_path}: {e}")
        if self._transition_log_file and not self._transition_log_file.closed:
            try:
                self._transition_log_file.close()
            except IOError as e:
                print(f"Error closing {self.transition_log_path}: {e}")
        # No need to close event_log_path as it's opened/closed per write for simplicity
        # print(f"DataLogger closed logs in {self.output_dir.resolve()}")
