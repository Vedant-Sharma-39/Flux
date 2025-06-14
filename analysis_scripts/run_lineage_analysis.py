# analysis_scripts/run_lineage_analysis.py

import argparse
from pathlib import Path
import json
import pandas as pd
import sys

# Adjust Python path to import from src
# This assumes run_lineage_analysis.py is in analysis_scripts/
# and src/ is one level up and then down into src/
try:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    sys.path.append(str(project_root))
    from src.analysis.lineage_analyzer import (
        parse_snapshot_data,
        calculate_P_lineage_snapshot_metrics,
        calculate_P_lineage_dynamics_between_snapshots,
    )
    from src.core.shared_types import Phenotype  # For filtering by P_PREPARED.name
except ImportError as e:
    print(
        f"Error importing project modules. Ensure script is run from project root or path is set: {e}"
    )
    sys.exit(1)


def process_simulation_run(sim_dir: Path) -> None:
    """
    Processes all snapshots in a simulation output directory to generate lineage analysis reports.
    """
    if not sim_dir.is_dir():
        print(f"Error: Directory not found: {sim_dir}")
        return

    sim_params_path = sim_dir / "simulation_parameters.json"
    if not sim_params_path.exists():
        print(f"Error: simulation_parameters.json not found in {sim_dir}")
        return
    with open(sim_params_path, "r") as f:
        sim_params_dict = json.load(f)

    hex_size = float(sim_params_dict.get("hex_size", 1.0))  # Default if somehow missing

    snapshot_data_dir = sim_dir / "grid_snapshots_data"
    if not snapshot_data_dir.is_dir():
        print(f"Error: grid_snapshots_data directory not found in {sim_dir}")
        return

    snapshot_files = sorted(snapshot_data_dir.glob("grid_snapshot_t_*.json"))
    if not snapshot_files:
        print(f"No snapshot files found in {snapshot_data_dir}")
        return

    all_snapshot_metrics_list = []
    all_dynamics_metrics_list = []

    previous_frontier_P_lineage_ids = set()
    previous_time_step = -1

    print(f"Found {len(snapshot_files)} snapshots in {sim_dir.name}. Processing...")

    for i, s_file in enumerate(snapshot_files):
        current_time_step = int(s_file.stem.split("_")[-1])
        # print(f"  Processing {s_file.name} (Time Step: {current_time_step})...")

        all_cells_df = parse_snapshot_data(s_file, hex_size)

        if all_cells_df.empty:
            print(
                f"    Snapshot {s_file.name} is empty or failed to parse. Generating empty metrics."
            )
            # Handle empty data for consistent output structure
            snapshot_metrics = calculate_P_lineage_snapshot_metrics(
                pd.DataFrame()
            )  # Get NaN structure
            current_frontier_P_lineage_ids = set()
        else:
            # Filter for P-phenotype cells on the frontier (ASSUMES 'is_frontier' logged)
            frontier_P_cells_df = all_cells_df[
                (all_cells_df["phenotype"] == Phenotype.P_PREPARED.name)
                & (all_cells_df["is_frontier"] == True)  # Critical assumption
            ].copy()
            snapshot_metrics = calculate_P_lineage_snapshot_metrics(frontier_P_cells_df)
            current_frontier_P_lineage_ids = set(
                frontier_P_cells_df["lineage_id"].unique()
            )

        snapshot_metrics_row = {"time_step": current_time_step, **snapshot_metrics}
        all_snapshot_metrics_list.append(snapshot_metrics_row)

        if i > 0:  # Cannot calculate dynamics for the very first snapshot
            dynamics_metrics = calculate_P_lineage_dynamics_between_snapshots(
                previous_frontier_P_lineage_ids, current_frontier_P_lineage_ids
            )
            dynamics_metrics_row = {
                "time_step_t1": previous_time_step,
                "time_step_t2": current_time_step,
                **dynamics_metrics,
            }
            all_dynamics_metrics_list.append(dynamics_metrics_row)

        previous_frontier_P_lineage_ids = current_frontier_P_lineage_ids
        previous_time_step = current_time_step

    print("Processing complete.")

    # Save aggregated metrics
    if all_snapshot_metrics_list:
        summary_df = pd.DataFrame(all_snapshot_metrics_list)
        summary_output_path = sim_dir / "P_lineage_snapshot_summary.csv"
        summary_df.to_csv(summary_output_path, index=False, float_format="%.4f")
        print(f"Saved P-lineage snapshot summary to {summary_output_path}")

    if all_dynamics_metrics_list:
        dynamics_df = pd.DataFrame(all_dynamics_metrics_list)
        dynamics_output_path = sim_dir / "P_lineage_dynamics.csv"
        dynamics_df.to_csv(dynamics_output_path, index=False, float_format="%.4f")
        print(f"Saved P-lineage dynamics to {dynamics_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run lineage analysis on simulation output."
    )
    parser.add_argument(
        "sim_output_dir",
        type=str,
        help="Path to the simulation output directory (e.g., results/run_xyz).",
    )
    args = parser.parse_args()
    process_simulation_run(Path(args.sim_output_dir))


if __name__ == "__main__":
    main()
