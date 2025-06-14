# analysis_scripts/plot_lineage_snapshots.py
import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Adjust Python path
try:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    sys.path.append(str(project_root))
    from src.visualization.plotter import (
        plot_colony_snapshot,
        PHENOTYPE_COLORS,
    )  # Import plotter and colors
    from src.analysis.lineage_analyzer import (
        parse_snapshot_data,
    )  # To load and prep data
    from src.core.shared_types import (
        Phenotype,
        SimulationParameters,
    )  # For Phenotype enum and SimParams structure
    from src.environment.environment_rules import EnvironmentRules  # For plotter
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)


def get_top_n_p_lineages_colormap(
    all_cells_df: pd.DataFrame,
    n: int = 5,
    default_g_color=PHENOTYPE_COLORS[Phenotype.G_UNPREPARED],
    other_p_color=PHENOTYPE_COLORS["OTHER_P_LINEAGE"],
) -> Tuple[Dict[str, Any], List[plt.Rectangle], List[str]]:
    """
    Identifies top N P-lineages by size and creates a colormap and legend info.
    Returns: (lineage_id_to_color_map, legend_handles, legend_labels)
    """
    p_cells_df = all_cells_df[all_cells_df["phenotype"] == Phenotype.P_PREPARED.name]
    if p_cells_df.empty:
        return {}, [plt.Rectangle((0, 0), 1, 1, fc=default_g_color)], ["G_UNPREPARED"]

    lineage_counts = p_cells_df["lineage_id"].value_counts()
    top_n_ids = lineage_counts.nlargest(n).index.tolist()

    # Use a perceptually distinct colormap for lineages
    # Ensure enough colors if n is large, or cycle through a smaller set
    try:
        # Try to get a good colormap, fallback if not enough distinct colors for large N
        base_colors = plt.cm.get_cmap(
            "viridis_r", max(n, 5)
        )  # Use viridis_r for P-like colors
        # Or 'tab10', 'Set1', etc. 'Accent' from previous example can be good for distinct categories
        # colors = [base_colors(i / (n -1 if n > 1 else 1) ) for i in range(n)]
        colors = [
            base_colors(i) for i in np.linspace(0.1, 0.9, n)
        ]  # Sample from colormap
    except Exception:  # Fallback if colormap issues
        colors = [
            "#FFADAD",
            "#FFC3A0",
            "#FFD6A5",
            "#FDFFB6",
            "#CAFFBF",
            "#9BF6FF",
            "#A0C4FF",
            "#BDB2FF",
            "#FFC6FF",
            "#FFFFFC",
        ][:n]

    lineage_to_color_map = {
        lineage_id: colors[i % len(colors)] for i, lineage_id in enumerate(top_n_ids)
    }

    # Prepare legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=default_g_color, label="G_UNPREPARED")
    ]
    legend_labels = ["G_UNPREPARED"]

    for i, lineage_id in enumerate(top_n_ids):
        label = f"P-Lin {i+1} ({lineage_counts.get(lineage_id, 0)} cells)"  # Add cell count to legend
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=colors[i % len(colors)], label=label)
        )
        legend_labels.append(label)

    if len(lineage_counts) > n:  # If there are "other" P lineages
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=other_p_color, label="Other P-Lineages")
        )
        legend_labels.append("Other P-Lineages")

    return lineage_to_color_map, legend_handles, legend_labels


def main():
    parser = argparse.ArgumentParser(
        description="Plot colony snapshots with P-lineage coloring."
    )
    parser.add_argument(
        "sim_output_dir", type=str, help="Path to simulation output directory."
    )
    parser.add_argument(
        "time_step", type=int, help="Time step of the snapshot to plot."
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of top P-lineages to color distinctly.",
    )
    parser.add_argument(
        "--plot_output_dir",
        type=str,
        help="Optional: Directory to save the plot. Defaults to sim_output_dir.",
    )
    args = parser.parse_args()

    sim_dir = Path(args.sim_output_dir)
    snapshot_file = (
        sim_dir / "grid_snapshots_data" / f"grid_snapshot_t_{args.time_step:05d}.json"
    )
    sim_params_file = sim_dir / "simulation_parameters.json"

    if not snapshot_file.exists():
        print(f"Error: Snapshot file not found: {snapshot_file}")
        return
    if not sim_params_file.exists():
        print(f"Error: Simulation parameters file not found: {sim_params_file}")
        return

    with open(sim_params_file, "r") as f:
        sim_params_dict_raw = json.load(f)

    # Create a proper SimulationParameters object or ensure plotter can use the dict
    # For simplicity, let's assume plotter's sim_params arg can take the dict directly if needed,
    # or construct a SimParams object if plotter strictly requires it.
    # Based on plotter's current state, it uses sim_params.hex_size etc.
    # So we need to construct a SimulationParameters object
    try:
        # A simplified constructor for plotting if some fields aren't needed by plotter
        # This assumes create_simulation_parameters_from_config is too complex/has side effects here
        # It's better if plotter can take the dict or SimulationParameters directly
        # For now, constructing a minimal SimulationParameters for plotter
        sim_params_obj = SimulationParameters(
            hex_size=float(sim_params_dict_raw["hex_size"]),
            time_step_duration=float(sim_params_dict_raw["time_step_duration"]),
            nutrient_bands=[
                (float(b[0]), Phenotype[b[1]])
                for b in sim_params_dict_raw.get("nutrient_bands", [])
            ],  # Example, needs robust parsing
            # Add other fields if plot_nutrient_background needs them from SimParams object
            # ... (fill other mandatory fields with defaults or from dict if available)
            lambda_G_N1=0,
            alpha_G_N2=0,
            lag_G_N2=0,
            lambda_G_N2_adapted=0,  # Dummies
            cost_delta_P=0,
            alpha_P_N2=0,
            lag_P_N2=0,
            lambda_P_N2=0,
            k_GP=0,
            k_PG=0,  # Dummies
            active_conflict_rule=list(PHENOTYPE_COLORS.keys())[0],
            initial_colony_radius=0,
            initial_phenotype_G_fraction=0,  # Dummies
        )
        # This is clunky. Better: modify plotter to accept sim_params_dict for hex_size
        # or ensure full SimParams object creation is easy.
        # For now, the plotter will use sim_params_obj.hex_size and sim_params_obj.nutrient_bands

    except Exception as e:
        print(f"Error creating dummy SimulationParameters for plotter: {e}")
        # Fallback: use dict for hex_size if plotter is adapted
        # For now, this will likely fail if plotter strictly expects SimParams obj with all fields.
        # Quick fix for plotter: if sim_params is dict, use sim_params['hex_size']

    # Use your robust parser
    all_cells_df = parse_snapshot_data(
        snapshot_file, float(sim_params_dict_raw["hex_size"])
    )
    if all_cells_df.empty:
        print(
            f"Failed to parse snapshot data or snapshot is empty: {snapshot_file.name}"
        )
        return

    p_lineage_color_map, legend_handles, legend_labels = get_top_n_p_lineages_colormap(
        all_cells_df, n=args.top_n
    )

    current_sim_time = float(
        args.time_step * sim_params_dict_raw.get("time_step_duration", 1.0)
    )

    # Dummy environment_rules for plotter if not doing nutrient coloring based on it
    # This part needs care if plot_nutrient_background is essential and relies on complex env_rules state
    env_rules = EnvironmentRules(
        sim_params_obj
    )  # Pass the constructed (possibly dummy) sim_params

    fig, _ = plot_colony_snapshot(
        cells_df=all_cells_df,  # Pass DataFrame of cells
        environment_rules=env_rules,
        sim_params=sim_params_obj,  # Or pass sim_params_dict_raw if plotter adapted
        current_sim_time=current_sim_time,
        color_by="lineage_p",
        p_lineage_color_map=p_lineage_color_map,
        custom_legend_handles=legend_handles,
        custom_legend_labels=legend_labels,
    )

    plot_save_dir = Path(args.plot_output_dir) if args.plot_output_dir else sim_dir
    plot_save_dir.mkdir(parents=True, exist_ok=True)
    output_plot_path = (
        plot_save_dir / f"colony_P_lineages_t_{args.time_step:05d}_top{args.top_n}.png"
    )

    try:
        fig.savefig(output_plot_path, dpi=200)  # Higher DPI for lineage plots
        plt.close(fig)
        print(f"Saved P-lineage snapshot to {output_plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")


if __name__ == "__main__":
    main()
