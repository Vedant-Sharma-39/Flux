# microbial_colony_sim/src/analysis/metrics.py
import time as wall_clock_time
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from dataclasses import asdict, is_dataclass

from src.core.data_structures import SimulationConfig
from src.core.enums import Phenotype, Nutrient
from src.agents.cell import Cell  # Import Cell for type hinting
from src.agents.population_manager import PopulationManager
from src.grid.hexagonal_grid import HexagonalGrid
from src.grid.nutrient_environment import NutrientEnvironment
from src.grid import coordinate_utils as coord_utils
from src.analysis.population_statistics import PopulationStatistics
from src.analysis.frontier_analysis import FrontierAnalysis


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (Phenotype, Nutrient)):
            return obj.name
        if is_dataclass(obj):
            return asdict(obj)
        try:
            return super(NpEncoder, self).default(obj)
        except TypeError:
            return str(obj)


class MetricsCollector:
    def __init__(
        self,
        config: SimulationConfig,
        population_manager: PopulationManager,
        grid: HexagonalGrid,
        nutrient_env: NutrientEnvironment,
    ):
        self.config = config
        self.population_manager = population_manager
        self.grid = grid
        self.nutrient_env = nutrient_env
        self.time_series_data: List[Dict[str, Any]] = []
        self.last_metrics_collection_time: float = -1.0
        self._initial_max_radius: Optional[float] = None
        self._initial_max_radius_time: Optional[float] = None
        self._last_max_radius: Optional[float] = None
        self._last_max_radius_time: Optional[float] = None
        self.output_path = Path(config.data_output_path) / config.experiment_name
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(
            f"sim.{config.experiment_name}.MetricsCollector"
        )
        if not self.logger.handlers:
            self.logger.debug(
                "MetricsCollector logger has no default handlers, using root if configured."
            )

        self.kymograph_perimeter_data: Dict[str, List[Tuple[float, np.ndarray]]] = {}
        self.kymograph_num_angular_bins = self.config.visualization.kymo_angular_bins
        # self.kymograph_radial_shell_width_factor = self.config.visualization.kymo_radial_shell_width_factor # Not directly used in this simple perimeter version

    def _collect_kymograph_perimeter_data(
        self, current_sim_time: float, frontier_cells: List[Cell]
    ):
        if not self.config.visualization.visualization_enabled and not getattr(
            self.config.visualization, "collect_kymograph_data", False
        ):
            # Only collect if visualization is on OR a specific kymo collection flag is set
            return

        attributes_to_track = {
            "perimeter_lag_rem": lambda c: c.remaining_lag_time,  # Shorter key
            "perimeter_lag_inh": lambda c: c.inherent_T_lag_GL,
            "perimeter_growth_g_inh": lambda c: c.inherent_growth_rate_G,
        }
        current_step_kymo_values: Dict[str, np.ndarray] = {
            attr_name: np.full(self.kymograph_num_angular_bins, np.nan)
            for attr_name in attributes_to_track
        }

        if frontier_cells:
            cells_in_angular_bins: List[List[Cell]] = [
                [] for _ in range(self.kymograph_num_angular_bins)
            ]  # Correct initialization
            for (
                cell_obj
            ) in frontier_cells:  # Use cell_obj to avoid conflict with imported Cell
                cart_x = self.nutrient_env.hex_pixel_size * (
                    np.sqrt(3) * cell_obj.coord.q + np.sqrt(3) / 2.0 * cell_obj.coord.r
                )
                cart_y = self.nutrient_env.hex_pixel_size * (
                    3.0 / 2.0 * cell_obj.coord.r
                )
                angle_rad = np.arctan2(cart_y, cart_x)
                angle_deg = (np.degrees(angle_rad) + 360) % 360
                bin_index = int(angle_deg / (360.0 / self.kymograph_num_angular_bins))
                bin_index = min(bin_index, self.kymograph_num_angular_bins - 1)
                if 0 <= bin_index < self.kymograph_num_angular_bins:
                    cells_in_angular_bins[bin_index].append(cell_obj)

            for i in range(self.kymograph_num_angular_bins):
                if cells_in_angular_bins[i]:
                    for attr_name, getter_func in attributes_to_track.items():
                        current_step_kymo_values[attr_name][i] = np.nanmean(
                            [getter_func(c) for c in cells_in_angular_bins[i]]
                        )

        for attr_name, values_array in current_step_kymo_values.items():
            if attr_name not in self.kymograph_perimeter_data:
                self.kymograph_perimeter_data[attr_name] = []
            self.kymograph_perimeter_data[attr_name].append(
                (current_sim_time, values_array.copy())
            )
        self.logger.debug(
            f"Collected kymograph data for T={current_sim_time:.2f} across {len(attributes_to_track)} attributes."
        )

    def collect_step_data(self, current_sim_time: float) -> None:
        # ... (should_collect logic remains same) ...
        should_collect = False
        if self.config.metrics_interval_time == 0:
            should_collect = True
        elif not self.time_series_data:
            should_collect = True
        elif (
            current_sim_time - self.last_metrics_collection_time
            >= self.config.metrics_interval_time - 1e-9
        ):
            should_collect = True
        if not should_collect:
            return

        self.last_metrics_collection_time = current_sim_time
        self.logger.debug(f"Collecting metrics at T={current_sim_time:.2f}")
        all_cells = self.population_manager.get_all_cells()
        step_stats = PopulationStatistics.collect_all_statistics(
            all_cells, current_sim_time, self.nutrient_env.hex_pixel_size
        )
        current_max_radius = step_stats.get("max_radial_distance", 0.0)
        if self._initial_max_radius is None and current_max_radius > 0 and all_cells:
            self._initial_max_radius = current_max_radius
            self._initial_max_radius_time = current_sim_time
        if current_max_radius > 0 and all_cells:
            self._last_max_radius = current_max_radius
            self._last_max_radius_time = current_sim_time
        frontier_cells = FrontierAnalysis.identify_frontier_cells(
            all_cells, self.grid, max_radial_distance=current_max_radius
        )
        step_stats["frontier_lag_G_to_L"] = (
            FrontierAnalysis.analyze_frontier_lag_distribution(
                frontier_cells,
                self.nutrient_env,
                target_nutrient_transition=Nutrient.GALACTOSE,
            )
        )
        step_stats["frontier_traits"] = (
            FrontierAnalysis.analyze_frontier_trait_distribution(frontier_cells)
        )

        self._collect_kymograph_perimeter_data(current_sim_time, frontier_cells)

        self.time_series_data.append(step_stats)
        pheno_counts_str = {
            (k.name if isinstance(k, Phenotype) else str(k)): v
            for k, v in step_stats["phenotype_counts"].items()
        }
        self.logger.info(
            f"Metrics @ T={current_sim_time:.2f}: Cells={step_stats['total_cell_count']}, MaxR={step_stats['max_radial_distance']:.2f}, Pheno={pheno_counts_str}"
        )

    def calculate_radial_expansion_velocity(self) -> float:
        if (
            self._last_max_radius is None
            or self._initial_max_radius is None
            or self._last_max_radius_time is None
            or self._initial_max_radius_time is None
        ):
            return 0.0
        delta_time = self._last_max_radius_time - self._initial_max_radius_time
        if delta_time < 1e-6:
            return 0.0
        delta_radius = self._last_max_radius - self._initial_max_radius
        return delta_radius / delta_time

    def get_final_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        if self.time_series_data:
            last_datapoint = self.time_series_data[-1].copy()
            if "phenotype_counts" in last_datapoint and isinstance(
                last_datapoint["phenotype_counts"], dict
            ):
                last_datapoint["phenotype_counts"] = {
                    k.name if isinstance(k, Phenotype) else str(k): v
                    for k, v in last_datapoint["phenotype_counts"].items()
                }
            summary.update(last_datapoint)
        else:
            summary["time"] = (
                self.config.max_simulation_time
            )  # Or self.current_time from engine if passed
            summary["total_cell_count"] = self.population_manager.get_cell_count()
            summary["phenotype_counts"] = {}  # Default if no data collected
        summary["overall_radial_expansion_velocity"] = (
            self.calculate_radial_expansion_velocity()
        )
        try:
            summary["config"] = asdict(self.config)
        except Exception as e:
            self.logger.warning(f"Could not serialize config via asdict: {e}")
            summary["config"] = str(self.config)
        return summary

    def export_time_series_data(self, filename: str = "time_series_data.csv") -> None:
        if not self.time_series_data:
            self.logger.info("No time series data to export.")
            return
        try:
            data_for_df = []
            for record in self.time_series_data:
                new_record = record.copy()
                if "phenotype_counts" in new_record and isinstance(
                    new_record["phenotype_counts"], dict
                ):
                    new_record["phenotype_counts"] = {
                        k.name if isinstance(k, Phenotype) else str(k): v
                        for k, v in new_record["phenotype_counts"].items()
                    }
                data_for_df.append(new_record)
            df = pd.json_normalize(data_for_df, sep="_")
            filepath = self.output_path / filename
            df.to_csv(filepath, index=False)
            self.logger.info(f"Time series data exported to {filepath}")
        except Exception as e:
            self.logger.error(
                f"Error exporting time series data to CSV: {e}", exc_info=True
            )
            try:
                json_filepath = self.output_path / filename.replace(".csv", ".json")
                with open(json_filepath, "w") as f:
                    json.dump(self.time_series_data, f, indent=2, cls=NpEncoder)
                self.logger.info(
                    f"Time series data exported as JSON to {json_filepath} due to CSV error."
                )
            except Exception as e_json:
                self.logger.error(
                    f"Error exporting time series data to JSON fallback: {e_json}",
                    exc_info=True,
                )

    def export_summary_data(
        self, summary: Dict[str, Any], filename: str = "summary.json"
    ) -> None:
        try:
            filepath = self.output_path / filename
            with open(filepath, "w") as f:
                json.dump(summary, f, indent=2, cls=NpEncoder)
            self.logger.info(f"Summary data exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Error exporting summary data: {e}", exc_info=True)

    def export_kymograph_data(
        self, filename: str = "kymograph_perimeter_raw_data.npz"
    ) -> None:
        if not self.kymograph_perimeter_data:
            self.logger.info("No kymograph data to export.")
            return
        kymo_output_file = self.output_path / filename
        try:
            save_dict = {}
            for key, series_data in self.kymograph_perimeter_data.items():
                if not series_data:
                    continue
                times = np.array([s[0] for s in series_data])
                values_matrix = np.array([s[1] for s in series_data])
                save_dict[f"{key}_times"] = times
                save_dict[f"{key}_values"] = values_matrix
            if save_dict:
                np.savez_compressed(kymo_output_file, **save_dict)
                self.logger.info(
                    f"Raw kymograph perimeter data saved to {kymo_output_file}"
                )
            else:
                self.logger.info(
                    "No valid kymograph series data was processed for saving."
                )
        except Exception as e:
            self.logger.error(f"Could not save raw kymograph data: {e}", exc_info=True)

    def finalize_data(self) -> None:
        self.logger.info("Finalizing data collection and exporting results...")
        summary = self.get_final_summary()
        log_msg_summary = "--- Simulation Summary (Log) ---\n"
        for key, value in summary.items():
            if key == "config":
                log_msg_summary += f"config_exp_name: {summary.get('config',{}).get('experiment_name','N/A')}\n"
            elif key in ["phenotype_counts"]:
                log_msg_summary += f"{key}: {value}\n"
            elif isinstance(value, (list, dict)) and len(str(value)) > 150:
                log_msg_summary += f"{key}: (details in files)\n"
            else:
                log_msg_summary += f"{key}: {value}\n"
        self.logger.info(log_msg_summary)
        self.export_time_series_data()
        self.export_summary_data(summary)
        self.export_kymograph_data()
        self.logger.info(
            f"All data export finalized for {self.config.experiment_name}."
        )
