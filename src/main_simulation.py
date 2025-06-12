# src/main_simulation.py

import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Optional

from .core.cell import Cell, Phenotype
from .core.shared_types import (
    HexCoord,
    Nutrient,
    SimulationParameters,
    ConflictResolutionRule,
)
from .environment.environment_rules import EnvironmentRules
from .grid.grid import Grid
from .grid.coordinate_utils import (
    get_filled_hexagon,
    axial_to_cartesian_sq_distance_from_origin,
)
from .evolution.conflict_resolution import resolve_reproduction_conflicts
from .evolution.phenotype_switching import attempt_phenotype_switch
from .visualization.plotter import plot_colony_snapshot
from .data_logging.logger import DataLogger


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML {config_path}: {e}")
        sys.exit(1)


def create_simulation_parameters_from_config(config: dict) -> SimulationParameters:
    try:
        parsed_bands = []
        for item in config.get("nutrient_bands_def", []):
            radius_val_str, nutrient_name_str = item[0], item[1]
            radius_val = (
                float(radius_val_str)
                if isinstance(radius_val_str, (int, float))
                else (
                    float("inf")
                    if str(radius_val_str).lower() == "inf"
                    else float(radius_val_str)
                )
            )
            radius_sq = radius_val**2 if radius_val != float("inf") else float("inf")
            parsed_bands.append((radius_sq, Nutrient[nutrient_name_str.upper()]))
        parsed_bands.sort(key=lambda x: x[0])

        return SimulationParameters(
            hex_size=float(config["hex_size"]),
            time_step_duration=float(config["time_step_duration"]),
            nutrient_bands=parsed_bands,
            lambda_G_N1=float(config["growth_rates"]["lambda_G_N1"]),
            alpha_G_N2=float(config["growth_rates"]["alpha_G_N2"]),
            lag_G_N2=float(config["growth_rates"]["lag_G_N2"]),
            lambda_G_N2_adapted=float(config["growth_rates"]["lambda_G_N2_adapted"]),
            cost_delta_P=float(config["growth_rates"]["cost_delta_P"]),
            alpha_P_N2=float(config["growth_rates"]["alpha_P_N2"]),
            lag_P_N2=float(config["growth_rates"]["lag_P_N2"]),
            lambda_P_N2=float(config["growth_rates"]["lambda_P_N2"]),
            k_GP=float(config["switching_rates"]["k_GP"]),
            k_PG=float(config["switching_rates"]["k_PG"]),
            active_conflict_rule=ConflictResolutionRule[
                config["active_conflict_rule"].upper()
            ],
            initial_colony_radius=int(config["initial_colony_radius"]),
            initial_phenotype_G_fraction=float(config["initial_phenotype_G_fraction"]),
        )
    except KeyError as e:
        print(f"Error: Missing parameter in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing configuration parameters: {e}")
        sys.exit(1)


def get_max_colony_radius(grid: Grid, sim_params: SimulationParameters) -> float:
    if not grid.occupied_sites:
        return 0.0
    max_r_sq = 0.0
    for coord in grid.occupied_sites.keys():
        dist_sq = axial_to_cartesian_sq_distance_from_origin(coord, sim_params.hex_size)
        if dist_sq > max_r_sq:
            max_r_sq = dist_sq
    return np.sqrt(max_r_sq) if max_r_sq > 0 else 0.0


def get_phenotype_counts(
    cells_with_coords_list: List[Tuple[HexCoord, Cell]],
) -> Tuple[int, int]:  # Takes list
    g_count, p_count = 0, 0
    for _, cell_obj in cells_with_coords_list:  # Iterate the provided list
        if cell_obj.phenotype == Phenotype.G_UNPREPARED:
            g_count += 1
        elif cell_obj.phenotype == Phenotype.P_PREPARED:
            p_count += 1
    return g_count, p_count


def run_simulation(
    sim_params: SimulationParameters,
    num_time_steps: int,
    output_dir: Path,
    log_interval: int,
    plot_dpi: int,
    save_grid_data_interval: Optional[int] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = EnvironmentRules(sim_params)
    grid = Grid()
    logger = DataLogger(output_dir, sim_params)
    logger.log_event("Simulation started.")

    initial_colony_cells: Dict[HexCoord, Cell] = {}
    center_coord = HexCoord(0, 0)
    coords_to_fill = get_filled_hexagon(center_coord, sim_params.initial_colony_radius)
    for init_coord in coords_to_fill:
        pheno = (
            Phenotype.G_UNPREPARED
            if random.random() < sim_params.initial_phenotype_G_fraction
            else Phenotype.P_PREPARED
        )
        cell = Cell(phenotype=pheno, birth_time=0.0, generation=0)
        nutrient = environment.get_nutrient_at_coord(init_coord)
        growth_rate = environment.get_growth_rate(cell.phenotype, nutrient)
        cell.reset_division_timer(growth_rate, sim_params.time_step_duration)
        initial_colony_cells[init_coord] = cell
    grid.initialize_colony(initial_colony_cells)

    previous_frontier_nutrient_majority: Optional[Nutrient] = None
    time_entered_N2: Optional[float] = None

    for t_step in range(num_time_steps):
        current_sim_time = t_step * sim_params.time_step_duration

        current_frontier_info_for_step = (
            grid.get_frontier_cells_with_coords()
        )  # Get once for this step's logic

        if not grid.occupied_sites:  # Extinction check moved earlier
            logger.log_event("Colony extinction.", simulation_time=current_sim_time)
            print(f"Colony went extinct at time {current_sim_time:.2f}")
            break

        # --- 0. Determine current nutrient state at the frontier (for transition logging) ---
        frontier_nutrient_counts: Dict[Nutrient, int] = {
            n: 0 for n in Nutrient
        }  # Initialize all
        if current_frontier_info_for_step:
            for coord, _ in current_frontier_info_for_step:
                frontier_nutrient_counts[environment.get_nutrient_at_coord(coord)] += 1

        current_majority_nutrient_at_frontier = Nutrient.NONE
        if sum(frontier_nutrient_counts.values()) > 0:
            current_majority_nutrient_at_frontier = max(
                frontier_nutrient_counts, key=frontier_nutrient_counts.get
            )

        if (
            previous_frontier_nutrient_majority is not None
            and current_majority_nutrient_at_frontier
            != previous_frontier_nutrient_majority
        ):
            frontier_G_before, frontier_P_before = get_phenotype_counts(
                current_frontier_info_for_step
            )
            total_frontier_before = frontier_G_before + frontier_P_before
            frac_P_before = (
                frontier_P_before / total_frontier_before
                if total_frontier_before > 0
                else 0.0
            )
            event_type, time_in_prev_band = "", -1.0
            if (
                previous_frontier_nutrient_majority == Nutrient.N1_PREFERRED
                and current_majority_nutrient_at_frontier == Nutrient.N2_CHALLENGING
            ):
                event_type = "ENTER_N2"
                time_entered_N2 = current_sim_time
                logger.log_event(
                    f"Frontier entering N2 band at t={current_sim_time:.2f}"
                )
            elif (
                previous_frontier_nutrient_majority == Nutrient.N2_CHALLENGING
                and current_majority_nutrient_at_frontier == Nutrient.N1_PREFERRED
            ):
                event_type = "EXIT_N2"
                if time_entered_N2 is not None:
                    time_in_prev_band = current_sim_time - time_entered_N2
                logger.log_event(
                    f"Frontier exiting N2 band at t={current_sim_time:.2f}, duration in N2: {time_in_prev_band:.2f}"
                )
                time_entered_N2 = None
            if event_type:
                logger.log_nutrient_transition(
                    t_step,
                    current_sim_time,
                    event_type,
                    previous_frontier_nutrient_majority,
                    current_majority_nutrient_at_frontier,
                    frontier_cells_before=total_frontier_before,
                    frontier_P_cells_before=frontier_P_before,
                    fraction_P_before=frac_P_before,
                    time_in_N2_band=time_in_prev_band,
                )
        previous_frontier_nutrient_majority = current_majority_nutrient_at_frontier

        # --- 1. UPDATE CELL STATES (Switching, N2 Adaptation Lag Management for ALL cells) ---
        all_cells_coords_this_step = (
            grid.get_all_cells_with_coords()
        )  # Get fresh list if needed
        for (
            coord,
            cell,
        ) in all_cells_coords_this_step:  # Process all cells for state changes
            local_nutrient = environment.get_nutrient_at_coord(coord)
            phenotype_before_switch_or_adapt = cell.phenotype
            adaptation_state_before = cell.is_adapting_to_N2

            switched = attempt_phenotype_switch(cell, environment)

            is_now_adapted_g = False
            if (
                cell.phenotype == Phenotype.G_UNPREPARED
                and local_nutrient == Nutrient.N2_CHALLENGING
            ):
                if not cell.is_adapting_to_N2 and cell.lag_N2_remaining <= 0:
                    alpha_g, lag_g_duration = environment.get_N2_adaptation_params(
                        Phenotype.G_UNPREPARED
                    )
                    if random.random() < alpha_g:
                        cell.initiate_N2_adaptation_lag(
                            lag_g_duration, sim_params.time_step_duration
                        )
                elif cell.is_adapting_to_N2:
                    if cell.decrement_N2_lag():
                        is_now_adapted_g = True

            # Update division timer if phenotype switched or G cell finished N2 lag
            if switched or (
                phenotype_before_switch_or_adapt == Phenotype.G_UNPREPARED
                and is_now_adapted_g
                and not adaptation_state_before
            ):
                current_growth_rate_for_timer = environment.get_growth_rate(
                    cell.phenotype,
                    local_nutrient,
                    is_cell_adapted_to_N2=is_now_adapted_g,
                )
                cell.reset_division_timer(
                    current_growth_rate_for_timer, sim_params.time_step_duration
                )

            if cell.time_to_next_division != float("inf"):
                cell.time_to_next_division -= 1

        # --- 2. IDENTIFY POTENTIAL REPRODUCTIONS (from current_frontier_info_for_step) ---
        potential_reproductions: List[Tuple[Cell, HexCoord, HexCoord, float]] = []

        random.shuffle(current_frontier_info_for_step)

        for parent_coord, parent_cell in current_frontier_info_for_step:
            if parent_cell.time_to_next_division <= 0:
                parent_local_nutrient = environment.get_nutrient_at_coord(parent_coord)
                is_parent_adapted_g_on_n2 = (
                    parent_cell.phenotype == Phenotype.G_UNPREPARED
                    and parent_local_nutrient == Nutrient.N2_CHALLENGING
                    and not parent_cell.is_adapting_to_N2
                    and parent_cell.lag_N2_remaining <= 0
                )
                can_p_grow_on_n2 = True
                if (
                    parent_cell.phenotype == Phenotype.P_PREPARED
                    and parent_local_nutrient == Nutrient.N2_CHALLENGING
                ):
                    alpha_p, _ = environment.get_N2_adaptation_params(
                        Phenotype.P_PREPARED
                    )
                    if not (random.random() < alpha_p):
                        can_p_grow_on_n2 = False

                parent_growth_rate = 0.0
                if can_p_grow_on_n2:
                    parent_growth_rate = environment.get_growth_rate(
                        parent_cell.phenotype,
                        parent_local_nutrient,
                        is_parent_adapted_g_on_n2,
                    )
                if parent_growth_rate > 0:
                    empty_neighbors = grid.get_empty_neighboring_coords(parent_coord)
                    if empty_neighbors:
                        potential_reproductions.append(
                            (
                                parent_cell,
                                parent_coord,
                                random.choice(empty_neighbors),
                                parent_growth_rate,
                            )
                        )

        # --- 3. RESOLVE CONFLICTS & 4. APPLY BIRTHS (logic as before) ---
        targeted_sites: Dict[HexCoord, List[Tuple[Cell, HexCoord, float]]] = {}
        for p_cell, p_coord, t_coord, g_rate in potential_reproductions:
            targeted_sites.setdefault(t_coord, []).append((p_cell, p_coord, g_rate))
        successful_reproduction_events = resolve_reproduction_conflicts(
            targeted_sites, sim_params.active_conflict_rule
        )
        parents_that_reproduced_ids = set()
        # ... (Birth logic and timer resets for successful parents and daughters as in previous full version) ...
        for (
            successful_parent,
            parent_coord,
            daughter_coord,
        ) in successful_reproduction_events:
            daughter_cell = Cell(
                parent_id=successful_parent.id,
                generation=successful_parent.generation + 1,
                phenotype=successful_parent.phenotype,
                birth_time=current_sim_time,
            )
            grid.place_cell(daughter_cell, daughter_coord)
            parents_that_reproduced_ids.add(successful_parent.id)
            d_nutrient = environment.get_nutrient_at_coord(daughter_coord)
            d_is_adapted_g = False
            if (
                daughter_cell.phenotype == Phenotype.G_UNPREPARED
                and d_nutrient == Nutrient.N2_CHALLENGING
            ):
                alpha_g, lag_g = environment.get_N2_adaptation_params(
                    Phenotype.G_UNPREPARED
                )
                if random.random() < alpha_g:
                    daughter_cell.initiate_N2_adaptation_lag(
                        lag_g, sim_params.time_step_duration
                    )
            d_growth_rate = environment.get_growth_rate(
                daughter_cell.phenotype, d_nutrient, d_is_adapted_g
            )
            daughter_cell.reset_division_timer(
                d_growth_rate, sim_params.time_step_duration
            )
            p_nutrient = environment.get_nutrient_at_coord(parent_coord)
            is_p_still_adapted_g = (
                successful_parent.phenotype == Phenotype.G_UNPREPARED
                and not successful_parent.is_adapting_to_N2
                and successful_parent.lag_N2_remaining <= 0
            )
            p_current_growth_rate = environment.get_growth_rate(
                successful_parent.phenotype, p_nutrient, is_p_still_adapted_g
            )
            successful_parent.reset_division_timer(
                p_current_growth_rate, sim_params.time_step_duration
            )

        for (
            parent_coord,
            parent_cell,
        ) in (
            current_frontier_info_for_step
        ):  # Use original frontier list for this check
            if (
                parent_cell.time_to_next_division <= 0
                and parent_cell.id not in parents_that_reproduced_ids
            ):
                parent_local_nutrient = environment.get_nutrient_at_coord(parent_coord)
                is_parent_adapted_g_on_n2 = (
                    parent_cell.phenotype == Phenotype.G_UNPREPARED
                    and not parent_cell.is_adapting_to_N2
                    and parent_cell.lag_N2_remaining <= 0
                )
                parent_current_growth_rate = environment.get_growth_rate(
                    parent_cell.phenotype,
                    parent_local_nutrient,
                    is_parent_adapted_g_on_n2,
                )
                parent_cell.reset_division_timer(
                    parent_current_growth_rate, sim_params.time_step_duration
                )

        # --- Data Logging & Visualization ---
        if t_step % log_interval == 0 or t_step == num_time_steps - 1:
            all_cells_for_log = grid.get_all_cells_with_coords()  # Fresh list
            frontier_info_for_log = grid.get_frontier_cells_with_coords()  # Fresh list

            total_g_overall, total_p_overall = get_phenotype_counts(all_cells_for_log)
            frontier_g_log, frontier_p_log = get_phenotype_counts(frontier_info_for_log)
            total_frontier_log = frontier_g_log + frontier_p_log
            frac_p_on_frontier_log = (
                frontier_p_log / total_frontier_log if total_frontier_log > 0 else 0.0
            )
            max_r = get_max_colony_radius(grid, sim_params)

            logger.log_population_state(
                time_step=t_step,
                simulation_time=current_sim_time,
                total_cells=len(all_cells_for_log),
                total_G_phenotype_count=total_g_overall,
                total_P_phenotype_count=total_p_overall,
                frontier_cell_count=total_frontier_log,
                frontier_G_phenotype_count=frontier_g_log,
                frontier_P_phenotype_count=frontier_p_log,
                fraction_P_on_frontier=frac_p_on_frontier_log,
                max_radius=max_r,
            )
            print(
                f"Logged T: {current_sim_time:.2f}, Cells: {len(all_cells_for_log)}, G_total: {total_g_overall}, P_total: {total_p_overall}, Frontier: {total_frontier_log}, F_P_frac: {frac_p_on_frontier_log:.3f}, MaxR: {max_r:.2f}"
            )

            snapshot_fig, _ = plot_colony_snapshot(
                grid=grid,
                environment_rules=environment,
                sim_params=sim_params,
                current_sim_time=current_sim_time,
            )
            snapshot_fig.savefig(
                output_dir / f"colony_t_{t_step:05d}.png", dpi=plot_dpi
            )
            plt.close(snapshot_fig)

            if save_grid_data_interval and (
                t_step % save_grid_data_interval == 0 or t_step == num_time_steps - 1
            ):
                logger.log_grid_snapshot_data(t_step, all_cells_for_log)

    # --- End of Loop ---
    final_sim_time = (
        (num_time_steps - 1) * sim_params.time_step_duration
        if num_time_steps > 0
        else 0.0
    )
    if grid.count_cells() > 0:  # If not extinct
        logger.log_event("Simulation finished.", simulation_time=final_sim_time)
    logger.close_logs()
    print(f"Simulation run finished. Output logged to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Run Microbial Colony Simulation from YAML config.")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    config = load_config(args.config_file)

    seed = config.get("random_seed")
    if seed is not None:
        random.seed(seed); np.random.seed(seed); print(f"Using random seed: {seed}")

    sim_params_obj = create_simulation_parameters_from_config(config)
    output_dir_run = Path(config.get("output_dir", "simulation_output_default")).resolve()
    num_steps_run = int(config.get("num_steps", 1000))
    log_interval_run = int(config.get("log_interval", 100))
    plot_dpi_run = int(config.get("plot_dpi", 150))
    save_grid_data_interval_run = config.get("save_grid_data_interval")
    if save_grid_data_interval_run is not None: save_grid_data_interval_run = int(save_grid_data_interval_run)

    run_simulation(
        sim_params=sim_params_obj, num_time_steps=num_steps_run, output_dir=output_dir_run,
        log_interval=log_interval_run, plot_dpi=plot_dpi_run,
        save_grid_data_interval=save_grid_data_interval_run
    )

if __name__ == '__main__':
    main()
