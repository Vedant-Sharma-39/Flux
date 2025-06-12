# src/main_simulation.py

import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Optional  # Ensure all needed types are here

# Assuming these are in place and updated with G_UNPREPARED, P_PREPARED
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

# Import new spatial metrics functions
from .analysis.spatial_metrics import (
    get_ordered_frontier_phenotypes,
    calculate_fmi_and_random_baseline,
    calculate_observed_interfaces,
)


def load_config(config_path: str) -> dict:
    """Loads parameters from a YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file {config_path}: {e}")
        sys.exit(1)


def create_simulation_parameters_from_config(config: dict) -> SimulationParameters:
    """Creates a SimulationParameters object from a loaded config dictionary."""
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
        # Ensure bands are sorted by their max_radius_sq for environment_rules and plotter logic
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
    """Calculates the maximum Cartesian radius of the colony from origin."""
    if not grid.occupied_sites:
        return 0.0
    max_r_sq = 0.0
    for coord in grid.occupied_sites.keys():  # Iterates over HexCoord keys
        dist_sq = axial_to_cartesian_sq_distance_from_origin(coord, sim_params.hex_size)
        if dist_sq > max_r_sq:
            max_r_sq = dist_sq
    return np.sqrt(max_r_sq) if max_r_sq > 0 else 0.0


def get_phenotype_counts_from_list(
    cells_with_coords_list: List[Tuple[HexCoord, Cell]],
) -> Tuple[int, int]:
    """Counts G_UNPREPARED and P_PREPARED phenotypes in a list of (HexCoord, Cell) tuples."""
    g_count, p_count = 0, 0
    for _, cell_obj in cells_with_coords_list:
        if cell_obj.phenotype == Phenotype.G_UNPREPARED:
            g_count += 1
        elif cell_obj.phenotype == Phenotype.P_PREPARED:
            p_count += 1
    return g_count, p_count


# --- Main Simulation Orchestration ---
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

    # INITIALIZATION
    initial_colony_cells: Dict[HexCoord, Cell] = {}
    center_coord = HexCoord(0, 0)  # Assuming colony starts centered at origin
    coords_to_fill = get_filled_hexagon(center_coord, sim_params.initial_colony_radius)
    for init_coord in coords_to_fill:
        pheno = (
            Phenotype.G_UNPREPARED
            if random.random() < sim_params.initial_phenotype_G_fraction
            else Phenotype.P_PREPARED
        )
        cell = Cell(phenotype=pheno, birth_time=0.0, generation=0)
        nutrient_at_start = environment.get_nutrient_at_coord(init_coord)
        # For initial cells, assume they are not in N2 lag if starting on N1 or P-type on N2
        is_initial_cell_adapted = not (
            cell.phenotype == Phenotype.G_UNPREPARED
            and nutrient_at_start == Nutrient.N2_CHALLENGING
        )
        growth_rate_at_start = environment.get_growth_rate(
            cell.phenotype, nutrient_at_start, is_initial_cell_adapted
        )
        cell.reset_division_timer(growth_rate_at_start, sim_params.time_step_duration)
        initial_colony_cells[init_coord] = cell
    grid.initialize_colony(initial_colony_cells)

    previous_frontier_nutrient_majority: Optional[Nutrient] = None
    time_entered_N2: Optional[float] = None
    current_sim_time: float = (
        0.0  # Initialize to ensure it's defined before final log if loop doesn't run
    )

    # MAIN SIMULATION LOOP
    for t_step in range(num_time_steps):
        current_sim_time = t_step * sim_params.time_step_duration

        if not grid.occupied_sites:  # Extinction check
            logger.log_event("Colony extinction.", simulation_time=current_sim_time)
            print(f"Colony went extinct at time {current_sim_time:.2f}")
            break

        # --- 0. Determine current nutrient state at the frontier & Log Transitions ---
        # Get frontier info ONCE for this step's logic (transition detection, reproduction attempts)
        current_frontier_info_for_step = grid.get_frontier_cells_with_coords()

        frontier_nutrient_counts: Dict[Nutrient, int] = {n: 0 for n in Nutrient}
        if current_frontier_info_for_step:
            for (
                coord,
                _,
            ) in current_frontier_info_for_step:  # Use coord from frontier info
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
            frontier_G_before, frontier_P_before = get_phenotype_counts_from_list(
                current_frontier_info_for_step
            )
            total_frontier_before = frontier_G_before + frontier_P_before
            frac_P_before = (
                frontier_P_before / total_frontier_before
                if total_frontier_before > 0
                else 0.0
            )

            ordered_phenos_trans = get_ordered_frontier_phenotypes(
                current_frontier_info_for_step, (0.0, 0.0), sim_params.hex_size
            )
            interfaces_trans = calculate_observed_interfaces(ordered_phenos_trans)
            fmi_trans, _ = calculate_fmi_and_random_baseline(ordered_phenos_trans)

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
                time_entered_N2 = None  # Reset for next N2 entry

            if event_type:
                logger.log_event(
                    f"Frontier {event_type} (from {previous_frontier_nutrient_majority.name} to {current_majority_nutrient_at_frontier.name}) at t={current_sim_time:.2f}"
                )
                logger.log_nutrient_transition(
                    t_step,
                    current_sim_time,
                    event_type,
                    previous_frontier_nutrient_majority,
                    current_majority_nutrient_at_frontier,
                    frontier_cells_before=total_frontier_before,
                    frontier_P_cells_before=frontier_P_before,
                    fraction_P_before=frac_P_before,
                    interfaces_before=interfaces_trans,
                    fmi_before=fmi_trans,
                    time_in_N2_band=time_in_prev_band,
                )
        previous_frontier_nutrient_majority = current_majority_nutrient_at_frontier

        # --- 1. UPDATE CELL STATES (Switching, N2 Lag Management for ALL cells) ---
        all_cells_coords_this_step = (
            grid.get_all_cells_with_coords()
        )  # Fresh list for state updates
        for coord, cell in all_cells_coords_this_step:
            local_nutrient = environment.get_nutrient_at_coord(coord)
            phenotype_before_switch = cell.phenotype
            adaptation_state_before_lag_decrement = cell.is_adapting_to_N2

            switched = attempt_phenotype_switch(cell, environment)  # dt is implicit

            is_now_adapted_g_this_step = (
                False  # Tracks if G cell *finishes* lag in this step
            )
            if (
                cell.phenotype == Phenotype.G_UNPREPARED
                and local_nutrient == Nutrient.N2_CHALLENGING
            ):
                if (
                    not cell.is_adapting_to_N2 and cell.lag_N2_remaining <= 0
                ):  # Not in lag, not started lag for N2 yet
                    alpha_g, lag_g_duration_nondim = (
                        environment.get_N2_adaptation_params(Phenotype.G_UNPREPARED)
                    )
                    if (
                        random.random() < alpha_g
                    ):  # Stochastic success to *begin* adaptation
                        cell.initiate_N2_adaptation_lag(
                            lag_g_duration_nondim, sim_params.time_step_duration
                        )
                        cell.reset_division_timer(
                            0.0, sim_params.time_step_duration
                        )  # No growth during lag
                elif cell.is_adapting_to_N2:  # Already in lag
                    if cell.decrement_N2_lag():  # Returns True if lag just finished
                        is_now_adapted_g_this_step = True

            # If phenotype switched OR G-cell just finished N2 lag, its growth rate changes, so reset timer
            if switched or (
                phenotype_before_switch == Phenotype.G_UNPREPARED
                and is_now_adapted_g_this_step
                and adaptation_state_before_lag_decrement
            ):
                current_growth_rate_for_timer = environment.get_growth_rate(
                    cell.phenotype,
                    local_nutrient,
                    is_cell_adapted_to_N2=(
                        cell.phenotype == Phenotype.G_UNPREPARED
                        and is_now_adapted_g_this_step
                    ),
                )
                cell.reset_division_timer(
                    current_growth_rate_for_timer, sim_params.time_step_duration
                )

            # Decrement division timer for all cells (if not infinite, and not currently in G-type N2 lag)
            if (
                cell.time_to_next_division != float("inf")
                and not cell.is_adapting_to_N2
            ):
                cell.time_to_next_division -= 1  # Each step is 1 unit for the timer

        # --- 2. IDENTIFY POTENTIAL REPRODUCTIONS (from `current_frontier_info_for_step`) ---
        potential_reproductions: List[Tuple[Cell, HexCoord, HexCoord, float]] = []

        for parent_coord, parent_cell in current_frontier_info_for_step:
            if parent_cell.time_to_next_division <= 0:  # Cell is due to divide
                parent_local_nutrient = environment.get_nutrient_at_coord(parent_coord)

                is_parent_fully_adapted_g = (
                    parent_cell.phenotype == Phenotype.G_UNPREPARED
                    and parent_local_nutrient == Nutrient.N2_CHALLENGING
                    and not parent_cell.is_adapting_to_N2
                    and parent_cell.lag_N2_remaining <= 0
                )

                # Determine if P-type can grow (passes its alpha_P_N2 check)
                can_p_type_attempt_growth_on_n2 = True
                if (
                    parent_cell.phenotype == Phenotype.P_PREPARED
                    and parent_local_nutrient == Nutrient.N2_CHALLENGING
                ):
                    alpha_p, lag_p_nondim = environment.get_N2_adaptation_params(
                        Phenotype.P_PREPARED
                    )
                    # Simplified: If P has a lag > 0, it needs to pass it. If alpha_P < 1, it needs to pass that too.
                    # For this example, assuming P's alpha and lag are handled implicitly by lambda_P_N2 being non-zero.
                    # A more rigorous P adaptation would mirror G's state machine.
                    # For now, if lambda_P_N2 is >0, it means it "passed" its alpha and lag.
                    if not (
                        random.random() < alpha_p
                    ):  # P stochastically fails its alpha check this time
                        can_p_type_attempt_growth_on_n2 = False

                current_parent_growth_rate = 0.0
                if (
                    can_p_type_attempt_growth_on_n2
                ):  # Only if P passed its alpha check (if applicable)
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
        parents_that_reproduced_ids = set()
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

            daughter_nutrient = environment.get_nutrient_at_coord(daughter_coord)
            daughter_is_adapted_g = (
                False  # New G daughter on N2 starts unadapted process
            )
            if (
                daughter_cell.phenotype == Phenotype.G_UNPREPARED
                and daughter_nutrient == Nutrient.N2_CHALLENGING
            ):
                alpha_g, lag_g_nondim_init = environment.get_N2_adaptation_params(
                    Phenotype.G_UNPREPARED
                )
                if random.random() < alpha_g:
                    daughter_cell.initiate_N2_adaptation_lag(
                        lag_g_nondim_init, sim_params.time_step_duration
                    )
            # P-type daughter would also need its alpha_P, lag_P check if explicitly modeled here.

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
            parent_current_actual_growth_rate = environment.get_growth_rate(
                successful_parent.phenotype, parent_nutrient, is_parent_still_adapted_g
            )
            successful_parent.reset_division_timer(
                parent_current_actual_growth_rate, sim_params.time_step_duration
            )

        # Reset timers for parents from `current_frontier_info_for_step` who were due but FAILED to reproduce
        for parent_coord, parent_cell in current_frontier_info_for_step:
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
                parent_current_actual_growth_rate = environment.get_growth_rate(
                    parent_cell.phenotype,
                    parent_local_nutrient,
                    is_parent_adapted_g_on_n2,
                )
                parent_cell.reset_division_timer(
                    parent_current_actual_growth_rate, sim_params.time_step_duration
                )

        # --- 5. Optional: Cell Death Logic ---

        # --- 6. Data Logging & Visualization ---
        if t_step % log_interval == 0 or t_step == num_time_steps - 1:
            all_cells_for_log = (
                grid.get_all_cells_with_coords()
            )  # Fresh list for this moment
            frontier_info_for_log = grid.get_frontier_cells_with_coords()  # Fresh list

            total_g_overall, total_p_overall = get_phenotype_counts_from_list(
                all_cells_for_log
            )
            frontier_g_log, frontier_p_log = get_phenotype_counts_from_list(
                frontier_info_for_log
            )
            total_frontier_log = frontier_g_log + frontier_p_log
            frac_p_on_frontier_log = (
                frontier_p_log / total_frontier_log if total_frontier_log > 0 else 0.0
            )
            max_r = get_max_colony_radius(grid, sim_params)

            ordered_phenos_summary = get_ordered_frontier_phenotypes(
                frontier_info_for_log, (0.0, 0.0), sim_params.hex_size
            )
            interfaces_summary = calculate_observed_interfaces(ordered_phenos_summary)
            fmi_summary, fmi_rand_summary = calculate_fmi_and_random_baseline(
                ordered_phenos_summary
            )

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
                observed_interfaces=interfaces_summary,
                fmi=fmi_summary,
                fmi_random=fmi_rand_summary,
            )
            print(
                f"Logged T: {current_sim_time:.2f}, Cells: {len(all_cells_for_log)}, TotG: {total_g_overall}, TotP: {total_p_overall}, FrontC: {total_frontier_log}, FracPFront: {frac_p_on_frontier_log:.3f}, MaxR: {max_r:.2f}, Int: {interfaces_summary}, FMI: {fmi_summary:.3f}"
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
                logger.log_grid_snapshot_data(
                    t_step, all_cells_for_log
                )  # Save detailed state

    # --- End of Loop ---
    # Ensure current_sim_time is defined for the final log message if loop didn't run (e.g. num_time_steps = 0)
    final_sim_time = current_sim_time if "current_sim_time" in locals() else 0.0
    if (
        grid.count_cells() > 0 or num_time_steps == 0
    ):  # Log finish unless extinct before first step
        logger.log_event("Simulation finished.", simulation_time=final_sim_time)
    logger.close_logs()  # Important!
    print(f"Simulation run finished. Output logged to {output_dir.resolve()}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run Microbial Colony Simulation from YAML config."
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    config = load_config(args.config_file)

    seed_val = config.get("random_seed")
    if seed_val is not None:  # Handles seed_val being 0 correctly
        random.seed(seed_val)
        np.random.seed(seed_val)  # Important for numpy's random choices too
        print(f"Using random seed: {seed_val}")

    sim_params_obj = create_simulation_parameters_from_config(config)

    output_dir_run = Path(
        config.get("output_dir", "simulation_output_default")
    ).resolve()
    num_steps_run = int(config.get("num_steps", 1000))
    log_interval_run = int(config.get("log_interval", 100))
    plot_dpi_run = int(config.get("plot_dpi", 150))
    save_grid_data_interval_run = config.get("save_grid_data_interval")  # Can be None
    if save_grid_data_interval_run is not None:
        save_grid_data_interval_run = int(save_grid_data_interval_run)

    run_simulation(
        sim_params=sim_params_obj,
        num_time_steps=num_steps_run,
        output_dir=output_dir_run,
        log_interval=log_interval_run,
        plot_dpi=plot_dpi_run,
        save_grid_data_interval=save_grid_data_interval_run,
    )


if __name__ == "__main__":
    main()
