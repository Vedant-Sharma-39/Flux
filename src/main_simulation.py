# src/main_simulation.py

import random
import numpy as np
import matplotlib.pyplot as plt  # Keep for potential direct plotting, though usually done by plotter
import argparse
import yaml
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Optional, Set, Any

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
    AXIAL_DIRECTIONS,  # Import AXIAL_DIRECTIONS
)
from .evolution.conflict_resolution import resolve_reproduction_conflicts
from .evolution.phenotype_switching import attempt_phenotype_switch
from .visualization.plotter import plot_colony_snapshot
from .data_logging.logger import DataLogger
from .analysis.spatial_metrics import (
    get_ordered_frontier_cells_with_angles,
    get_phenotypes_from_ordered_data,
    get_angles_of_target_phenotype,
    calculate_observed_interfaces,
    calculate_fmi_and_random_baseline,
    calculate_angular_coverage_entropy,
)


# --- Configuration Loading ---
def load_config(config_path_str: str) -> Dict[str, Any]:
    config_path = Path(config_path_str)
    if not config_path.is_file():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            print(
                f"Error: Configuration file {config_path} did not parse into a dictionary."
            )
            sys.exit(1)
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file {config_path}: {e}")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading configuration file {config_path}: {e}")
        sys.exit(1)


def create_simulation_parameters_from_config(
    config: Dict[str, Any],
) -> SimulationParameters:
    try:
        parsed_bands = []
        for band_def in config.get("nutrient_bands_def", []):
            if not (isinstance(band_def, list) and len(band_def) == 2):
                raise ValueError(f"Invalid nutrient_bands_def item format: {band_def}")

            radius_val_str, nutrient_name_str = str(band_def[0]), str(band_def[1])
            radius_val = (
                float("inf")
                if radius_val_str.lower() == "inf"
                else float(radius_val_str)
            )
            radius_sq = radius_val**2 if radius_val != float("inf") else float("inf")

            try:
                nutrient_enum = Nutrient[nutrient_name_str.upper()]
            except KeyError:
                raise ValueError(
                    f"Unknown nutrient type '{nutrient_name_str}' in nutrient_bands_def."
                )
            parsed_bands.append((radius_sq, nutrient_enum))

        parsed_bands.sort(
            key=lambda x: x[0]
        )  # Crucial for correct nutrient determination

        # Read save_grid_data_interval from config
        save_grid_interval_conf = config.get("save_grid_data_interval")  # Can be None
        final_save_grid_interval = None
        if save_grid_interval_conf is not None:
            try:
                val = int(save_grid_interval_conf)
                if val > 0:
                    final_save_grid_interval = val
                # else: leave as None if 0 or negative, effectively disabling it
            except ValueError:
                raise ValueError(
                    f"Invalid 'save_grid_data_interval': {save_grid_interval_conf}. Must be an integer."
                )


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
            save_grid_data_interval=final_save_grid_interval,
        )
    except KeyError as e:
        print(f"Error: Missing essential parameter in configuration: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid parameter value in configuration: {e}")
        sys.exit(1)
    except Exception as e:  # Catch-all for other unexpected errors
        print(f"Unexpected error creating SimulationParameters from config: {e}")
        sys.exit(1)


# --- Helper Functions ---
def get_max_colony_radius(grid: Grid, sim_params: SimulationParameters) -> float:
    if not grid.occupied_sites:
        return 0.0
    # Using numpy for potentially faster calculation if many cells
    coords_q = np.array([coord.q for coord in grid.occupied_sites.keys()])
    coords_r = np.array([coord.r for coord in grid.occupied_sites.keys()])

    # This is a placeholder for axial_to_cartesian_sq_distance_from_origin_batch
    # For simplicity, revert to iterative if batch not fully implemented/tested elsewhere
    max_r_sq = 0.0
    for q, r in zip(coords_q, coords_r):
        dist_sq = axial_to_cartesian_sq_distance_from_origin(
            HexCoord(q, r), sim_params.hex_size
        )
        if dist_sq > max_r_sq:
            max_r_sq = dist_sq
    return np.sqrt(max_r_sq) if max_r_sq > 0 else 0.0


def get_phenotype_counts_from_cell_tuples(
    cells_with_coords_list: List[
        Tuple[HexCoord, Cell]
    ],  # Or List[Tuple[HexCoord, Cell, bool]]
) -> Tuple[int, int]:
    g_count, p_count = 0, 0
    for item in cells_with_coords_list:
        cell_obj = item[1]  # cell is the second element
        if cell_obj.phenotype == Phenotype.G_UNPREPARED:
            g_count += 1
        elif cell_obj.phenotype == Phenotype.P_PREPARED:
            p_count += 1
    return g_count, p_count


def calculate_spatial_metrics_for_logging(
    frontier_info: List[Tuple[HexCoord, Cell]],  # List of (coord, cell) for frontier
    sim_params: SimulationParameters,
) -> Dict[str, Any]:
    """Calculates all spatial metrics needed for logging for a given frontier."""
    if not frontier_info:
        return {"interfaces": 0, "fmi": np.nan, "fmi_random": np.nan, "ace_P": np.nan}

    ordered_angled_data = get_ordered_frontier_cells_with_angles(
        frontier_info, (0.0, 0.0), sim_params.hex_size
    )
    ordered_phenos = get_phenotypes_from_ordered_data(ordered_angled_data)

    interfaces = calculate_observed_interfaces(ordered_phenos)
    fmi, fmi_rand = calculate_fmi_and_random_baseline(ordered_phenos)

    p_cell_angles = get_angles_of_target_phenotype(
        ordered_angled_data, Phenotype.P_PREPARED
    )
    ace_P = calculate_angular_coverage_entropy(p_cell_angles)

    return {
        "interfaces": interfaces,
        "fmi": fmi,
        "fmi_random": fmi_rand,
        "ace_P": ace_P,
    }


# --- Main Simulation Orchestration ---
def run_simulation(
    sim_params: SimulationParameters,
    num_time_steps: int,
    output_dir: Path,
    log_interval: int,
    plot_dpi: int,
    save_grid_data_interval: Optional[int] = None,  # Explicitly Optional
):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Add save_grid_data_interval to sim_params if it's used by DataLogger constructor
    # For now, DataLogger uses it via direct param in its constructor, which is fine.

    environment = EnvironmentRules(sim_params)
    grid = Grid()
    logger = DataLogger(output_dir, sim_params)  # Pass full sim_params
    logger.log_event("Simulation started.")

    # INITIALIZATION
    initial_colony_cells: Dict[HexCoord, Cell] = {}
    center_coord = HexCoord(0, 0)
    coords_to_fill = get_filled_hexagon(center_coord, sim_params.initial_colony_radius)
    for init_coord in coords_to_fill:
        pheno = (
            Phenotype.G_UNPREPARED
            if random.random() < sim_params.initial_phenotype_G_fraction
            else Phenotype.P_PREPARED
        )
        cell = Cell(
            phenotype=pheno, birth_time=0.0, generation=0
        )  # lineage_id auto-assigned

        nutrient_at_start = environment.get_nutrient_at_coord(init_coord)
        is_initial_adapted = not (
            cell.phenotype == Phenotype.G_UNPREPARED
            and nutrient_at_start == Nutrient.N2_CHALLENGING
        )
        growth_rate_at_start = environment.get_growth_rate(
            cell.phenotype, nutrient_at_start, is_initial_adapted
        )
        cell.reset_division_timer(growth_rate_at_start, sim_params.time_step_duration)
        initial_colony_cells[init_coord] = cell
    grid.initialize_colony(initial_colony_cells)

    previous_frontier_nutrient_majority: Optional[Nutrient] = None
    time_entered_N2: Optional[float] = None
    current_sim_time: float = 0.0

    # MAIN SIMULATION LOOP
    for t_step in range(num_time_steps):
        current_sim_time = t_step * sim_params.time_step_duration

        if not grid.occupied_sites:
            logger.log_event(
                f"Colony extinction at t_step {t_step}.",
                simulation_time=current_sim_time,
            )
            print(f"Colony went extinct at step {t_step}, time {current_sim_time:.2f}")
            break

        # --- 0. Nutrient Transition Detection & Logging ---
        current_frontier_info = (
            grid.get_frontier_cells_with_coords()
        )  # (HexCoord, Cell) list

        # Determine majority nutrient at frontier
        frontier_nutrient_counts: Dict[Nutrient, int] = {n: 0 for n in Nutrient}
        if current_frontier_info:
            for coord, _ in current_frontier_info:
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

            frontier_G_before, frontier_P_before = (
                get_phenotype_counts_from_cell_tuples(current_frontier_info)
            )
            total_frontier_before = len(current_frontier_info)
            frac_P_before = (
                (frontier_P_before / total_frontier_before)
                if total_frontier_before > 0
                else 0.0
            )

            spatial_metrics_trans = calculate_spatial_metrics_for_logging(
                current_frontier_info, sim_params
            )

            event_type, time_in_prev_band = "", -1.0
            if (
                previous_frontier_nutrient_majority == Nutrient.N1_PREFERRED
                and current_majority_nutrient_at_frontier == Nutrient.N2_CHALLENGING
            ):
                event_type = "ENTER_N2"
                time_entered_N2 = current_sim_time
            elif (
                previous_frontier_nutrient_majority == Nutrient.N2_CHALLENGING
                and current_majority_nutrient_at_frontier == Nutrient.N1_PREFERRED
            ):
                event_type = "EXIT_N2"
                if time_entered_N2 is not None:
                    time_in_prev_band = current_sim_time - time_entered_N2
                time_entered_N2 = None

            if event_type:
                logger.log_event(
                    f"Frontier {event_type} (from {previous_frontier_nutrient_majority.name} to {current_majority_nutrient_at_frontier.name})",
                    simulation_time=current_sim_time,
                )
                logger.log_nutrient_transition(
                    time_step=t_step,
                    simulation_time=current_sim_time,
                    event_type=event_type,
                    from_nutrient=previous_frontier_nutrient_majority,
                    to_nutrient=current_majority_nutrient_at_frontier,
                    frontier_cells_before=total_frontier_before,
                    frontier_P_cells_before=frontier_P_before,
                    fraction_P_before=frac_P_before,
                    interfaces_before=spatial_metrics_trans["interfaces"],
                    fmi_before=spatial_metrics_trans["fmi"],
                    angular_coverage_entropy_P_before=spatial_metrics_trans["ace_P"],
                    time_in_N2_band=time_in_prev_band,
                )
        previous_frontier_nutrient_majority = current_majority_nutrient_at_frontier

        # --- 1. UPDATE ALL CELL STATES (Switching, N2 Lag Management) ---
        all_cells_coords_this_step = (
            grid.get_all_cells_with_coords()
        )  # Fresh list for state updates
        for coord, cell in all_cells_coords_this_step:
            local_nutrient = environment.get_nutrient_at_coord(coord)
            phenotype_before_switch = cell.phenotype
            adaptation_state_before_lag_decrement = cell.is_adapting_to_N2

            switched = attempt_phenotype_switch(
                cell, environment
            )  # Handles lineage_id update

            is_now_adapted_g_this_step = False
            if (
                cell.phenotype == Phenotype.G_UNPREPARED
                and local_nutrient == Nutrient.N2_CHALLENGING
            ):
                if not cell.is_adapting_to_N2 and cell.lag_N2_remaining <= 0:
                    alpha_g, lag_g_duration_nondim = (
                        environment.get_N2_adaptation_params(Phenotype.G_UNPREPARED)
                    )
                    if random.random() < alpha_g:
                        cell.initiate_N2_adaptation_lag(
                            lag_g_duration_nondim, sim_params.time_step_duration
                        )
                        cell.reset_division_timer(
                            0.0, sim_params.time_step_duration
                        )  # No growth during lag
                elif cell.is_adapting_to_N2:
                    if cell.decrement_N2_lag():
                        is_now_adapted_g_this_step = True

            # If phenotype switched OR G-cell just finished N2 lag, reset division timer
            if switched or (
                phenotype_before_switch == Phenotype.G_UNPREPARED
                and is_now_adapted_g_this_step
                and adaptation_state_before_lag_decrement
            ):
                is_adapted_for_timer = (
                    cell.phenotype == Phenotype.G_UNPREPARED
                    and is_now_adapted_g_this_step
                ) or (
                    cell.phenotype == Phenotype.P_PREPARED
                )  # P-types are "adapted" on N2 if they pass alpha
                current_growth_rate_for_timer = environment.get_growth_rate(
                    cell.phenotype, local_nutrient, is_adapted_for_timer
                )
                cell.reset_division_timer(
                    current_growth_rate_for_timer, sim_params.time_step_duration
                )

            # Decrement division timer (if not infinite and not in G-type N2 lag)
            if (
                cell.time_to_next_division != float("inf")
                and not cell.is_adapting_to_N2
            ):
                cell.time_to_next_division -= 1

        # --- 2. IDENTIFY POTENTIAL REPRODUCTIONS (from `current_frontier_info` obtained earlier) ---
        potential_reproductions: List[Tuple[Cell, HexCoord, HexCoord, float]] = []
        # random.shuffle(current_frontier_info) # Shuffle to reduce bias if multiple cells target same spot with same fitness

        for (
            parent_coord,
            parent_cell,
        ) in current_frontier_info:  # Use the list from start of step
            if parent_cell.time_to_next_division <= 0:
                parent_local_nutrient = environment.get_nutrient_at_coord(parent_coord)
                is_parent_fully_adapted_g = (
                    parent_cell.phenotype == Phenotype.G_UNPREPARED
                    and parent_local_nutrient == Nutrient.N2_CHALLENGING
                    and not parent_cell.is_adapting_to_N2
                    and parent_cell.lag_N2_remaining <= 0
                )

                can_attempt_growth = True
                if (
                    parent_cell.phenotype == Phenotype.P_PREPARED
                    and parent_local_nutrient == Nutrient.N2_CHALLENGING
                ):
                    alpha_p, _ = environment.get_N2_adaptation_params(
                        Phenotype.P_PREPARED
                    )
                    if not (random.random() < alpha_p):
                        can_attempt_growth = False

                current_parent_growth_rate = 0.0
                if can_attempt_growth:
                    current_parent_growth_rate = environment.get_growth_rate(
                        parent_cell.phenotype,
                        parent_local_nutrient,
                        is_parent_fully_adapted_g,
                    )

                if current_parent_growth_rate > 0:
                    empty_neighbors = grid.get_empty_neighboring_coords(parent_coord)
                    if empty_neighbors:
                        chosen_target_coord = random.choice(empty_neighbors)
                        potential_reproductions.append(
                            (
                                parent_cell,
                                parent_coord,
                                chosen_target_coord,
                                current_parent_growth_rate,
                            )
                        )

        # --- 3. RESOLVE CONFLICTS ---
        targeted_sites_for_conflict: Dict[
            HexCoord, List[Tuple[Cell, HexCoord, float]]
        ] = {}
        for p_cell, p_coord, t_coord, g_rate in potential_reproductions:
            targeted_sites_for_conflict.setdefault(t_coord, []).append(
                (p_cell, p_coord, g_rate)
            )

        successful_reproduction_events = resolve_reproduction_conflicts(
            targeted_sites_for_conflict, sim_params.active_conflict_rule
        )

        # --- 4. APPLY BIRTHS & RESET PARENT TIMERS ---
        parents_that_reproduced_ids: Set[str] = set()  # Store parent cell.id as string
        for (
            successful_parent,
            parent_coord,
            daughter_coord,
        ) in successful_reproduction_events:
            daughter_cell = Cell(
                parent_id=successful_parent.id,
                generation=successful_parent.generation + 1,
                phenotype=successful_parent.phenotype,
                lineage_id=successful_parent.lineage_id,  # Inherit
                birth_time=current_sim_time,
            )
            grid.place_cell(daughter_cell, daughter_coord)
            parents_that_reproduced_ids.add(str(successful_parent.id))

            # Daughter N2 adaptation initiation
            daughter_nutrient = environment.get_nutrient_at_coord(daughter_coord)
            daughter_is_adapted_g = False  # New G daughter on N2 starts unadapted
            if (
                daughter_cell.phenotype == Phenotype.G_UNPREPARED
                and daughter_nutrient == Nutrient.N2_CHALLENGING
            ):
                alpha_g_init, lag_g_init = environment.get_N2_adaptation_params(
                    Phenotype.G_UNPREPARED
                )
                if random.random() < alpha_g_init:
                    daughter_cell.initiate_N2_adaptation_lag(
                        lag_g_init, sim_params.time_step_duration
                    )

            daughter_growth_rate = environment.get_growth_rate(
                daughter_cell.phenotype, daughter_nutrient, daughter_is_adapted_g
            )
            daughter_cell.reset_division_timer(
                daughter_growth_rate, sim_params.time_step_duration
            )

            # Reset successful parent's timer
            parent_nutrient = environment.get_nutrient_at_coord(parent_coord)
            is_parent_still_adapted_g = (
                successful_parent.phenotype == Phenotype.G_UNPREPARED
                and not successful_parent.is_adapting_to_N2
                and successful_parent.lag_N2_remaining <= 0
            )
            parent_actual_growth_rate = environment.get_growth_rate(
                successful_parent.phenotype, parent_nutrient, is_parent_still_adapted_g
            )
            successful_parent.reset_division_timer(
                parent_actual_growth_rate, sim_params.time_step_duration
            )

        # Reset timers for parents from `current_frontier_info` who were due but FAILED to reproduce
        for (
            parent_coord,
            parent_cell,
        ) in current_frontier_info:  # Use same list as for identifying reproducers
            if (
                parent_cell.time_to_next_division <= 0
                and str(parent_cell.id) not in parents_that_reproduced_ids
            ):
                parent_local_nutrient = environment.get_nutrient_at_coord(parent_coord)
                is_parent_adapted_g_on_n2 = (
                    parent_cell.phenotype == Phenotype.G_UNPREPARED
                    and not parent_cell.is_adapting_to_N2
                    and parent_cell.lag_N2_remaining <= 0
                )
                parent_current_actual_growth_rate = environment.get_growth_rate(
                    parent_cell.phenotype,
                    parent_local_nutrient,
                    is_parent_adapted_g_on_n2,
                )
                parent_cell.reset_division_timer(
                    parent_current_actual_growth_rate, sim_params.time_step_duration
                )

        # --- 5. Optional: Cell Death Logic --- (Not implemented)

        # --- 6. Data Logging & Visualization ---
        if t_step % log_interval == 0 or t_step == num_time_steps - 1:
            # Get fresh data for logging
            all_cells_for_log_tuples = grid.get_all_cells_with_coords()
            frontier_info_for_log_tuples = grid.get_frontier_cells_with_coords()

            total_g_log, total_p_log = get_phenotype_counts_from_cell_tuples(
                all_cells_for_log_tuples
            )
            frontier_g_log, frontier_p_log = get_phenotype_counts_from_cell_tuples(
                frontier_info_for_log_tuples
            )

            total_frontier_log = len(frontier_info_for_log_tuples)
            frac_p_frontier_log = (
                (frontier_p_log / total_frontier_log) if total_frontier_log > 0 else 0.0
            )
            max_r = get_max_colony_radius(grid, sim_params)

            spatial_metrics_log = calculate_spatial_metrics_for_logging(
                frontier_info_for_log_tuples, sim_params
            )

            logger.log_population_state(
                time_step=t_step,
                simulation_time=current_sim_time,
                total_cells=len(all_cells_for_log_tuples),
                total_G_phenotype_count=total_g_log,
                total_P_phenotype_count=total_p_log,
                frontier_cell_count=total_frontier_log,
                frontier_G_phenotype_count=frontier_g_log,
                frontier_P_phenotype_count=frontier_p_log,
                fraction_P_on_frontier=frac_p_frontier_log,
                max_radius=max_r,
                observed_interfaces=spatial_metrics_log["interfaces"],
                fmi=spatial_metrics_log["fmi"],
                fmi_random=spatial_metrics_log["fmi_random"],
                angular_coverage_entropy_P=spatial_metrics_log["ace_P"],
            )
            print(
                f"Time: {current_sim_time:7.2f} | Cells: {len(all_cells_for_log_tuples):6} | Frontier: {total_frontier_log:4} | "
                f"fP_Front: {frac_p_frontier_log:.3f} | MaxR: {max_r:6.1f} | FMI: {spatial_metrics_log['fmi']:.3f} | ACE_P: {spatial_metrics_log['ace_P']:.3f}"
            )

            # Plotting
            if plot_dpi > 0:  # Allow disabling plotting by setting dpi to 0 or less
                try:
                    snapshot_fig, _ = plot_colony_snapshot(
                        grid, environment, sim_params, current_sim_time
                    )
                    snapshot_fig.savefig(
                        output_dir / f"colony_t_{t_step:05d}.png", dpi=plot_dpi
                    )
                    plt.close(snapshot_fig)  # Close figure to free memory
                except Exception as e:
                    print(
                        f"Warning: Failed to plot colony snapshot at t_step {t_step}: {e}"
                    )

            # Grid Snapshot Data (with is_frontier calculation)
            if (
                save_grid_data_interval
                and save_grid_data_interval > 0
                and (
                    t_step % save_grid_data_interval == 0
                    or t_step == num_time_steps - 1
                )
            ):

                # Prepare data for detailed snapshot, including frontier status
                cells_data_for_json_snapshot: List[Tuple[HexCoord, Cell, bool]] = []
                # Efficiently get all occupied coordinates for neighbor checking
                all_occupied_coords_snapshot_set = set(grid.occupied_sites.keys())

                for coord, cell_obj in all_cells_for_log_tuples:
                    is_cell_frontier = False
                    for direction in AXIAL_DIRECTIONS:
                        neighbor_coord = coord + direction
                        if neighbor_coord not in all_occupied_coords_snapshot_set:
                            is_cell_frontier = True
                            break
                    cells_data_for_json_snapshot.append(
                        (coord, cell_obj, is_cell_frontier)
                    )

                logger.log_grid_snapshot_data(t_step, cells_data_for_json_snapshot)

    # --- End of Loop ---
    final_sim_time = current_sim_time  # Use the last calculated current_sim_time
    if not grid.occupied_sites and num_time_steps > 0:
        # Extinction event already logged
        pass
    else:
        logger.log_event(
            "Simulation run finished normally.", simulation_time=final_sim_time
        )

    logger.close_logs()
    print(f"Simulation complete. Output logged to {output_dir.resolve()}")


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Run Microbial Colony Simulation from YAML config."
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the YAML configuration file."
    )
    # Could add optional override for output_dir, seed, etc. via CLI here
    args = parser.parse_args()

    config = load_config(args.config_file)

    seed_val = config.get("random_seed")
    if seed_val is not None:
        random.seed(int(seed_val))
        np.random.seed(int(seed_val))
        print(f"Using random seed: {seed_val}")
    else:
        print("No random_seed specified, using system default.")

    sim_params_obj = create_simulation_parameters_from_config(config)

    # Run control parameters from config
    output_dir_run = Path(
        config.get(
            "output_dir",
            f"simulation_output_{Path(args.config_file).stem}_{random.randint(1000,9999)}",
        )
    ).resolve()
    num_steps_run = int(config.get("num_steps", 1000))
    log_interval_run = int(config.get("log_interval", 100))
    plot_dpi_run = int(config.get("plot_dpi", 150))  # set to 0 to disable plotting
    save_grid_interval_run = config.get("save_grid_data_interval")  # Can be None or 0
    if save_grid_interval_run is not None:
        save_grid_interval_run = int(save_grid_interval_run)
        if save_grid_interval_run <= 0:
            save_grid_interval_run = None  # Treat 0 or less as None

    print(f"Output directory: {output_dir_run}")

    try:
        run_simulation(
            sim_params=sim_params_obj,
            num_time_steps=num_steps_run,
            output_dir=output_dir_run,
            log_interval=log_interval_run,
            plot_dpi=plot_dpi_run,
            save_grid_data_interval=save_grid_interval_run,
        )
    except Exception as e:
        # Catch unexpected errors during simulation run itself
        print(f"CRITICAL ERROR during simulation run: {e}")
        # Log this critical error if logger is available and functional
        # For simplicity, just printing. In robust systems, might try to log.
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    main()
