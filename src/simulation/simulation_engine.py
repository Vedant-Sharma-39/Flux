# microbial_colony_sim/src/simulation/simulation_engine.py
import random
import time
from typing import List, Optional, Set, Callable  # Added Callable
from pathlib import Path
import logging

from src.core.data_structures import SimulationConfig, HexCoord
from src.core.enums import Phenotype
from src.agents.cell import Cell
from src.agents.cell_factory import CellFactory
from src.agents.population_manager import PopulationManager
from src.grid.hexagonal_grid import HexagonalGrid
from src.grid.nutrient_environment import NutrientEnvironment
from src.grid import coordinate_utils as coord_utils
from src.dynamics import lag_dynamics, phenotype_switching  # No reproduction directly
from src.simulation.initialization import Initializer
from src.analysis.metrics import MetricsCollector
from src.utils.logging_setup import setup_logging
from src.visualization.colony_visualizer import ColonyVisualizer, CellColorMode


class SimulationEngine:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.current_time: float = 0.0
        self._simulation_step_count: int = 0

        log_file_path = (
            Path(config.data_output_path) / config.experiment_name / "simulation.log"
        )
        self.logger = setup_logging(
            log_level=config.log_level,
            log_file=log_file_path,
            logger_name=f"sim.{config.experiment_name}",
        )
        self.logger.info(
            f"Initializing SimulationEngine for experiment: {config.experiment_name}"
        )
        self.logger.debug(f"Full configuration used: {config}")

        self.nutrient_env = NutrientEnvironment(config, hex_pixel_size=1.0)
        self.grid = HexagonalGrid()
        self.cell_factory = CellFactory(config)
        self.population_manager = PopulationManager(self.grid)
        self.initializer = Initializer(
            config, self.cell_factory, self.population_manager, self.nutrient_env
        )
        self.metrics_collector = MetricsCollector(
            config, self.population_manager, self.grid, self.nutrient_env
        )

        self.visualizer: Optional[ColonyVisualizer] = None
        if self.config.visualization.visualization_enabled:
            hex_render_size_val = self.config.visualization.hex_pixel_size
            viz_output_base_dir = (
                Path(config.data_output_path).parent / "visualizations"
            )
            self.visualizer = ColonyVisualizer(
                config=config,
                population_manager=self.population_manager,
                nutrient_env=self.nutrient_env,
                hex_render_size=hex_render_size_val,
                output_dir_base=str(viz_output_base_dir),
            )
            self.logger.info(
                f"Visualization enabled. Output directory: {self.visualizer.output_path}"
            )
        else:
            self.logger.info("Visualization disabled.")

    def _update_cell_internal_states(self) -> None:
        all_cells = list(self.population_manager.get_all_cells())
        for cell in all_cells:
            local_nutrient = self.nutrient_env.get_nutrient(cell.coord)
            lag_dynamics.process_lag_phase(cell, self.config.dt)
            if cell.current_phenotype != Phenotype.SWITCHING_GL:
                phenotype_switching.update_phenotype_based_on_nutrient(
                    cell, local_nutrient
                )
            cell.update_growth_attempt_rate(local_nutrient, self.config)

    def _process_colony_expansion(self) -> None:  # Using the verified version
        potential_parents = list(self.population_manager.get_all_cells())
        random.shuffle(potential_parents)
        newly_born_cells: List[Cell] = []
        chosen_slots_this_step: Set[HexCoord] = set()
        for parent_cell in potential_parents:
            if parent_cell.current_growth_attempt_rate <= 1e-9:
                continue
            prob_div = parent_cell.current_growth_attempt_rate * self.config.dt
            prob_div = max(0.0, min(1.0, prob_div))
            if not (random.random() < prob_div):
                continue
            available_grid_slots = self.grid.get_empty_adjacent_slots(parent_cell.coord)
            candidate_slots_for_parent = [
                s for s in available_grid_slots if s not in chosen_slots_this_step
            ]
            if not candidate_slots_for_parent:
                continue
            daughter_slot_coord: Optional[HexCoord] = None
            parent_dist_to_origin = coord_utils.euclidean_distance(
                parent_cell.coord,
                coord_utils.HexCoord(0, 0),
                self.nutrient_env.hex_pixel_size,
            )
            outward_slots, same_dist_slots, inward_slots = [], [], []
            epsilon = 1e-9
            for slot_cand in candidate_slots_for_parent:
                slot_dist = coord_utils.euclidean_distance(
                    slot_cand,
                    coord_utils.HexCoord(0, 0),
                    self.nutrient_env.hex_pixel_size,
                )
                if slot_dist > parent_dist_to_origin + epsilon:
                    outward_slots.append(slot_cand)
                elif abs(slot_dist - parent_dist_to_origin) < epsilon:
                    same_dist_slots.append(slot_cand)
                else:
                    inward_slots.append(slot_cand)
            if outward_slots:
                daughter_slot_coord = random.choice(outward_slots)
            elif same_dist_slots:
                daughter_slot_coord = random.choice(same_dist_slots)
            elif inward_slots:
                daughter_slot_coord = random.choice(inward_slots)
            if daughter_slot_coord:
                local_nutrient_at_birth = self.nutrient_env.get_nutrient(
                    daughter_slot_coord
                )
                actual_daughter_cell = self.cell_factory.create_daughter_cell(
                    parent_cell=parent_cell,
                    daughter_coord=daughter_slot_coord,
                    local_nutrient_at_birth=local_nutrient_at_birth,
                    current_time=self.current_time,
                )
                newly_born_cells.append(actual_daughter_cell)
                chosen_slots_this_step.add(daughter_slot_coord)
        for daughter in newly_born_cells:
            try:
                self.population_manager.add_cell(daughter)
            except Exception as e_add:
                self.logger.error(
                    f"CRITICAL: Error adding daughter {daughter.id} at {daughter.coord}: {e_add}",
                    exc_info=True,
                )
                continue
            nutrient_at_birth = self.nutrient_env.get_nutrient(daughter.coord)
            daughter.update_growth_attempt_rate(nutrient_at_birth, self.config)

    def _collect_data_step(self) -> None:
        try:
            self.metrics_collector.collect_step_data(self.current_time)
        except Exception as e:
            self.logger.error(
                f"Error during data collection at T={self.current_time:.2f}: {e}",
                exc_info=True,
            )

    def _record_visualization_frame_step(self) -> None:
        if self.visualizer and self.config.visualization.visualization_enabled:
            frame_interval_steps = 0
            if self.config.dt > 1e-9:
                viz_config = self.config.visualization
                if viz_config.animation_frame_interval > 0:
                    frame_interval_steps = viz_config.animation_frame_interval
                elif self.config.metrics_interval_time > 0:
                    frame_interval_steps = int(
                        self.config.metrics_interval_time / self.config.dt
                    )
            if (
                frame_interval_steps > 0
                and self._simulation_step_count % frame_interval_steps == 0
            ) or (self._simulation_step_count == 0 and self.current_time == 0):
                try:
                    anim_color_mode_str = self.config.visualization.animation_color_mode
                    try:
                        anim_color_mode = CellColorMode[anim_color_mode_str.upper()]
                    except KeyError:
                        self.logger.warning(
                            f"Invalid anim_color_mode '{anim_color_mode_str}'. Defaulting to PHENOTYPE."
                        )
                        anim_color_mode = CellColorMode.PHENOTYPE
                    self.logger.debug(
                        f"Recording anim frame T={self.current_time:.2f}, Mode={anim_color_mode.name}"
                    )
                    self.visualizer.record_animation_frame(
                        self.current_time, color_mode=anim_color_mode
                    )
                    if self.config.visualization.save_key_snapshots:
                        self.visualizer.plot_colony_state_to_file(
                            self.current_time,
                            self._simulation_step_count,
                            color_mode=anim_color_mode,
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error recording/plotting anim frame: {e}", exc_info=True
                    )

    def run(self) -> None:
        self.logger.info(f"Starting simulation: {self.config.experiment_name}")
        self.initializer.initialize_colony()
        self.logger.info(
            f"Colony initialized: {self.population_manager.get_cell_count()} cells."
        )
        self._collect_data_step()
        if self.config.visualization.visualization_enabled:
            self._record_visualization_frame_step()
        start_wall_time = time.perf_counter()
        while self.current_time < self.config.max_simulation_time:
            self.current_time += self.config.dt
            self._simulation_step_count += 1
            if (
                self._simulation_step_count > 0
                and self._simulation_step_count % 100 == 0
            ):
                self.logger.info(
                    f"Progress: T={self.current_time:.2f}, Cells={self.population_manager.get_cell_count()}"
                )
            self._update_cell_internal_states()
            self._process_colony_expansion()
            self._collect_data_step()
            if self.config.visualization.visualization_enabled:
                self._record_visualization_frame_step()
            if self.population_manager.get_cell_count() == 0:
                self.logger.info(f"Colony extinct at T={self.current_time:.2f}.")
                break
            if (
                self.population_manager.get_cell_count()
                > self.config.max_cells_safety_threshold
            ):
                self.logger.warning(
                    f"Cell count exceeded threshold at T={self.current_time:.2f}."
                )
                break
        end_wall_time = time.perf_counter()
        self.logger.info(
            f"Sim run finished: {self.config.experiment_name}. Wall-clock: {end_wall_time - start_wall_time:.3f}s."
        )
        self.logger.info(
            f"Final state: SimTime={self.current_time:.2f}, Steps={self._simulation_step_count}, Cells={self.population_manager.get_cell_count()}."
        )
        self.metrics_collector.finalize_data()
        if (
            self.visualizer
            and self.config.visualization.visualization_enabled
            and self.visualizer.animation_frames_data
        ):
            self.logger.info("Saving animation...")
            try:
                writer_to_use = self.config.visualization.animation_writer

                def progress_update_sim_engine(
                    current_frame, total_frames
                ):  # Renamed to avoid potential capture issues
                    if (
                        current_frame == 0
                        or (current_frame + 1) % max(1, total_frames // 10) == 0
                        or current_frame == total_frames - 1
                    ):
                        self.logger.info(
                            f"Animation saving: Frame {current_frame + 1}/{total_frames}"
                        )

                self.visualizer.save_animation(
                    fps=10,
                    writer_name=writer_to_use,
                    progress_callback=progress_update_sim_engine,
                )
            except Exception as e:
                self.logger.error(f"Failed to save animation: {e}", exc_info=True)
        if self.visualizer:
            self.visualizer.close_plot()
            self.logger.info("Viz resources closed.")
        self.logger.info("Simulation fully complete.")


# if __name__ == "__main__": block remains the same for direct testing
if __name__ == "__main__":
    from src.utils.config_loader import load_config  # For direct testing
    import traceback

    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        experiment_choice = "config/experiment_configs/bet_hedging_mixed.yaml"
        sim_config = load_config(
            default_config_path=project_root / "config/default_config.yaml",
            experiment_config_path=project_root / experiment_choice,
        )
        sim_config.visualization.visualization_enabled = True
        sim_config.visualization.animation_frame_interval = (
            10  # Record more often for short test
        )
        sim_config.max_simulation_time = 50.0  # Shorter for testing animation

        engine = SimulationEngine(sim_config)
        engine.run()
    except Exception as e:
        logging.basicConfig(
            level=logging.ERROR, format="[CRITICAL ERROR IN MAIN] %(message)s"
        )
        logging.error(f"Error in simulation_engine __main__: {e}", exc_info=True)
