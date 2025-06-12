# src/data_logging/logger.py

import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import dataclasses  # For dataclasses.asdict
from enum import Enum  # For isinstance check
import numpy as np  # For handling potential NaN from FMI

# Assuming these types are defined in shared_types
from ..core.shared_types import SimulationParameters, HexCoord, Nutrient, Phenotype
from ..core.cell import (
    Cell,
)  # For type hinting if log_grid_snapshot_data is used extensively


class DataLogger:
    """
    Handles logging of simulation data to files.
    """

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

    def _save_parameters(self):
        """Saves the simulation parameters used for this run to a JSON file."""
        params_path = self.output_dir / "simulation_parameters.json"
        params_as_dict = dataclasses.asdict(self.sim_params)
        serializable_params = {}
        for key, value in params_as_dict.items():
            if isinstance(
                value, Enum
            ):  # Handles Phenotype, Nutrient, ConflictResolutionRule
                serializable_params[key] = value.name
            elif key == "nutrient_bands" and isinstance(value, list):
                # Nutrient bands are List[Tuple[float_radius_sq, Nutrient_Enum]]
                serializable_params[key] = [
                    (r_sq, nt.name if isinstance(nt, Enum) else str(nt))
                    for r_sq, nt in value
                ]
            else:
                serializable_params[key] = value
        try:
            with open(params_path, "w") as f:
                json.dump(serializable_params, f, indent=4)
            print(f"Saved simulation parameters to {params_path}")
        except Exception as e:
            print(
                f"Error saving parameters: {e}\nProblematic serializable_params: {serializable_params}"
            )

    def _init_population_summary_log(self):
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
            "observed_interfaces_frontier",  # New
            "fmi_frontier",  # New
            "fmi_random_baseline_frontier",  # New
        ]
        try:
            with open(self.population_summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
            self._population_summary_file = open(
                self.population_summary_path, "a", newline=""
            )
            self._population_summary_writer = csv.writer(self._population_summary_file)
        except IOError as e:
            print(f"Error initializing population_summary.csv: {e}")
            self._population_summary_file = None
            self._population_summary_writer = None

    def _init_transition_log(self):
        """Initializes the CSV file for nutrient transition logging."""
        header = [
            "time_step",
            "simulation_time",
            "event_type",
            "from_nutrient",
            "to_nutrient",
            "frontier_cell_count_before_transition",
            "frontier_P_cell_count_before_transition",  # P_PREPARED cells
            "fraction_P_at_frontier_before_transition",
            "observed_interfaces_before_transition",  # New
            "fmi_before_transition",  # New
            "time_spent_in_N2_band",  # If applicable
        ]
        try:
            with open(self.transition_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
            self._transition_log_file = open(self.transition_log_path, "a", newline="")
            self._transition_log_writer = csv.writer(self._transition_log_file)
        except IOError as e:
            print(f"Error initializing nutrient_transitions.csv: {e}")
            self._transition_log_file = None
            self._transition_log_writer = None

    def log_population_state(
        self,
        time_step: int,
        simulation_time: float,
        *,  # Make subsequent arguments keyword-only for clarity
        total_cells: int,
        total_G_phenotype_count: int,
        total_P_phenotype_count: int,
        frontier_cell_count: int,
        frontier_G_phenotype_count: int,
        frontier_P_phenotype_count: int,
        fraction_P_on_frontier: float,
        max_radius: float,
        observed_interfaces: int,
        fmi: float,
        fmi_random: float,
    ):
        """Logs aggregate population data for the current time step."""
        if self._population_summary_writer:
            try:
                row = [
                    time_step,
                    f"{simulation_time:.2f}",
                    total_cells,
                    total_G_phenotype_count,
                    total_P_phenotype_count,
                    frontier_cell_count,
                    frontier_G_phenotype_count,
                    frontier_P_phenotype_count,
                    f"{fraction_P_on_frontier:.4f}",
                    f"{max_radius:.2f}",
                    observed_interfaces,
                    (
                        f"{fmi:.4f}" if not np.isnan(fmi) else "NaN"
                    ),  # Handle potential NaN
                    f"{fmi_random:.4f}" if not np.isnan(fmi_random) else "NaN",
                ]
                self._population_summary_writer.writerow(row)
                if self._population_summary_file:
                    self._population_summary_file.flush()
            except Exception as e:
                print(f"Error writing to population_summary.csv: {e}")
        else:
            print(
                f"Warning: Population summary log writer not initialized. Cannot log state at t={simulation_time}."
            )

    def log_nutrient_transition(
        self,
        time_step: int,
        simulation_time: float,
        *,  # Keyword-only arguments
        event_type: str,
        from_nutrient: Nutrient,
        to_nutrient: Nutrient,
        frontier_cells_before: int = -1,
        frontier_P_cells_before: int = -1,
        fraction_P_before: float = -1.0,
        interfaces_before: int = -1,
        fmi_before: float = -1.0,
        time_in_N2_band: float = -1.0,
    ):
        """Logs data related to the colony frontier transitioning between nutrient bands."""
        if self._transition_log_writer:
            try:
                row = [
                    time_step,
                    f"{simulation_time:.2f}",
                    event_type,
                    from_nutrient.name,
                    to_nutrient.name,  # Log Enum names
                    frontier_cells_before,
                    frontier_P_cells_before,
                    f"{fraction_P_before:.4f}" if fraction_P_before >= 0 else "-1.0",
                    interfaces_before if interfaces_before >= 0 else "-1",
                    (
                        f"{fmi_before:.4f}"
                        if fmi_before >= 0 and not np.isnan(fmi_before)
                        else ("-1.0" if fmi_before < 0 else "NaN")
                    ),
                    f"{time_in_N2_band:.2f}" if time_in_N2_band >= 0 else "-1.0",
                ]
                self._transition_log_writer.writerow(row)
                if self._transition_log_file:
                    self._transition_log_file.flush()
            except Exception as e:
                print(f"Error writing to nutrient_transitions.csv: {e}")
        else:
            print(
                f"Warning: Transition log writer not initialized. Cannot log transition at t={simulation_time}."
            )

    def log_event(self, message: str, simulation_time: Optional[float] = None):
        """Logs a generic event or message to events.log."""
        try:
            with open(self.event_log_path, "a") as f:
                time_str = (
                    f"Time {simulation_time:.2f}: "
                    if simulation_time is not None
                    else ""
                )
                f.write(f"{time_str}{message}\n")
        except IOError as e:
            print(f"Error writing to event log: {e}")

    def log_grid_snapshot_data(
        self, time_step: int, grid_cells_with_coords: List[Tuple[HexCoord, Cell]]
    ):
        """Saves the state of all cells on the grid at a given time step to a JSON file."""
        if not self.snapshot_data_dir.exists():  # Create only if used
            self.snapshot_data_dir.mkdir(exist_ok=True)

        snapshot_file = self.snapshot_data_dir / f"grid_snapshot_t_{time_step:05d}.json"
        data_to_save = []
        for coord, cell in grid_cells_with_coords:
            data_to_save.append(
                {
                    "q": coord.q,
                    "r": coord.r,
                    "id": str(cell.id),
                    "phenotype": cell.phenotype.name,
                    "generation": cell.generation,
                    "birth_time": cell.birth_time,
                    "time_to_next_division": cell.time_to_next_division,
                    "is_adapting_to_N2": cell.is_adapting_to_N2,
                    "lag_N2_remaining": cell.lag_N2_remaining,
                }
            )
        try:
            with open(snapshot_file, "w") as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            print(f"Error saving/serializing grid snapshot data: {e}")

    def close_logs(self):
        """Closes any open log files."""
        if self._population_summary_file and not self._population_summary_file.closed:
            try:
                self._population_summary_file.close()
            except Exception as e:
                print(f"Error closing population_summary.csv: {e}")
        if self._transition_log_file and not self._transition_log_file.closed:
            try:
                self._transition_log_file.close()
            except Exception as e:
                print(f"Error closing nutrient_transitions.csv: {e}")
        print(f"DataLogger closed logs in {self.output_dir.resolve()}")
