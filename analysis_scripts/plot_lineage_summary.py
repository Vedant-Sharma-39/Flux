# analysis_scripts/plot_lineage_summary.py

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # For formatting axes
import seaborn as sns  # Optional, for better aesthetics
import numpy as np
import ast  # For converting string representations of lists back to lists
from typing import Optional


def plot_lineage_timeseries(
    sim_output_dir_str: str, output_plot_dir_str: Optional[str] = None
):
    sim_output_dir = Path(sim_output_dir_str)
    if not sim_output_dir.is_dir():
        print(f"Error: Simulation output directory not found: {sim_output_dir}")
        return

    if output_plot_dir_str:
        output_plot_dir = Path(output_plot_dir_str)
    else:
        output_plot_dir = (
            sim_output_dir / "lineage_plots"
        )  # Default to a subfolder in sim_output_dir
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_plot_dir}")

    # --- Load Data ---
    try:
        pop_summary_df = pd.read_csv(sim_output_dir / "population_summary.csv")
        # Define converters for list-like string columns
        list_converters = {
            "P_lineage_sizes": ast.literal_eval,
            "P_lineage_angular_spans_rad": ast.literal_eval,
        }
        lineage_snapshot_df = pd.read_csv(
            sim_output_dir / "P_lineage_snapshot_summary.csv",
            converters=list_converters,
        )
        lineage_dynamics_df = pd.read_csv(sim_output_dir / "P_lineage_dynamics.csv")
    except FileNotFoundError as e:
        print(f"Error: Missing one or more summary CSV files in {sim_output_dir}: {e}")
        return
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return

    # Use simulation_time as the primary x-axis
    # Merge simulation_time from pop_summary into lineage_snapshot_df
    time_mapping = pop_summary_df.set_index("time_step")["simulation_time"]
    lineage_snapshot_df["simulation_time"] = lineage_snapshot_df["time_step"].map(
        time_mapping
    )

    # For dynamics, map t2 to simulation_time
    lineage_dynamics_df["simulation_time_t2"] = lineage_dynamics_df["time_step_t2"].map(
        time_mapping
    )

    # --- Plotting Parameters ---
    sns.set_theme(style="whitegrid")  # Optional: Use seaborn style
    plot_params = {"linewidth": 2, "markersize": 5}

    # --- Plot 1: P-Lineage Count & Size Over Time ---
    fig1, ax1_1 = plt.subplots(figsize=(12, 7))
    ax1_2 = ax1_1.twinx()

    ax1_1.plot(
        lineage_snapshot_df["simulation_time"],
        lineage_snapshot_df["num_unique_P_lineages"],
        color="tab:blue",
        marker="o",
        label="Num Unique P-Lineages (Frontier)",
        **plot_params,
    )
    ax1_1.set_xlabel("Simulation Time")
    ax1_1.set_ylabel("Number of Unique P-Lineages", color="tab:blue")
    ax1_1.tick_params(axis="y", labelcolor="tab:blue")
    ax1_1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax1_2.plot(
        lineage_snapshot_df["simulation_time"],
        lineage_snapshot_df["mean_P_lineage_size"],
        color="tab:red",
        marker="x",
        linestyle="--",
        label="Mean P-Lineage Size (Frontier)",
        **plot_params,
    )
    # Optional: Plot max size as well
    ax1_2.plot(
        lineage_snapshot_df["simulation_time"],
        lineage_snapshot_df["max_P_lineage_size"],
        color="tab:green",
        marker="s",
        linestyle=":",
        label="Max P-Lineage Size (Frontier)",
        alpha=0.7,
        **plot_params,
    )
    ax1_2.set_ylabel("P-Lineage Size (Cells)", color="tab:red")
    ax1_2.tick_params(axis="y", labelcolor="tab:red")

    fig1.suptitle(
        f"P-Lineage Count and Size Dynamics ({sim_output_dir.name})", fontsize=16
    )
    # Combine legends
    lines1, labels1 = ax1_1.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1_2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )
    fig1.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for suptitle and legend
    plt.savefig(output_plot_dir / "1_P_lineage_count_size.png", dpi=150)
    plt.close(fig1)

    # --- Plot 2: P-Lineage Age Proxies Over Time ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(
        lineage_snapshot_df["simulation_time"],
        lineage_snapshot_df["mean_P_lineage_min_gen_proxy"],
        color="tab:purple",
        marker="o",
        label="Mean Min Generation per P-Lineage",
        **plot_params,
    )
    # ax2.plot(lineage_snapshot_df['simulation_time'], lineage_snapshot_df['mean_P_lineage_min_birth_time_proxy'],
    #          color='tab:brown', marker='x', linestyle='--', label='Mean Min Birth Time per P-Lineage', **plot_params) # Can be redundant with gen
    ax2.set_xlabel("Simulation Time")
    ax2.set_ylabel('Mean "Age" Proxy')
    ax2.legend()
    ax2.set_title(
        f'P-Lineage "Age" Proxy Dynamics ({sim_output_dir.name})', fontsize=16
    )
    fig2.tight_layout()
    plt.savefig(output_plot_dir / "2_P_lineage_age_proxy.png", dpi=150)
    plt.close(fig2)

    # --- Plot 3: P-Lineage Angular Span Over Time ---
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    ax3.plot(
        lineage_snapshot_df["simulation_time"],
        lineage_snapshot_df["mean_P_lineage_angular_span_rad"],
        color="tab:orange",
        marker="o",
        label="Mean P-Lineage Angular Span (radians)",
        **plot_params,
    )
    # You could also plot max angular span or a boxplot of spans per timestep if data allows
    ax3.set_xlabel("Simulation Time")
    ax3.set_ylabel("Mean Angular Span (radians)")
    ax3.legend()
    ax3.set_title(
        f"P-Lineage Angular Span Dynamics ({sim_output_dir.name})", fontsize=16
    )
    fig3.tight_layout()
    plt.savefig(output_plot_dir / "3_P_lineage_angular_span.png", dpi=150)
    plt.close(fig3)

    # --- Plot 4: P-Lineage Origination & Survival Over Time Intervals ---
    fig4, ax4_1 = plt.subplots(figsize=(12, 7))
    ax4_2 = ax4_1.twinx()

    ax4_1.plot(
        lineage_dynamics_df["simulation_time_t2"],
        lineage_dynamics_df["num_P_lineages_originated"],
        color="tab:cyan",
        marker="o",
        label="Num P-Lineages Originated",
        **plot_params,
    )
    ax4_1.set_xlabel("Simulation Time (end of interval)")
    ax4_1.set_ylabel("Number of P-Lineages Originated", color="tab:cyan")
    ax4_1.tick_params(axis="y", labelcolor="tab:cyan")
    ax4_1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax4_2.plot(
        lineage_dynamics_df["simulation_time_t2"],
        lineage_dynamics_df["P_lineage_survival_rate"],
        color="tab:pink",
        marker="x",
        linestyle="--",
        label="P-Lineage Survival Rate",
        **plot_params,
    )
    ax4_2.set_ylabel("P-Lineage Survival Rate", color="tab:pink")
    ax4_2.tick_params(axis="y", labelcolor="tab:pink")
    ax4_2.set_ylim(0, 1.05)  # Survival rate is 0-1

    fig4.suptitle(
        f"P-Lineage Origination and Survival ({sim_output_dir.name})", fontsize=16
    )
    lines1, labels1 = ax4_1.get_legend_handles_labels()
    lines2, labels2 = ax4_2.get_legend_handles_labels()
    ax4_2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
    )
    fig4.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_plot_dir / "4_P_lineage_orig_survival.png", dpi=150)
    plt.close(fig4)

    # --- Plot 5: Combined - Num Unique P-Lineages vs. ACE_P ---
    if "ACE_P" in pop_summary_df.columns:  # Check if ACE_P was logged (it should be)
        fig5, ax5_1 = plt.subplots(figsize=(12, 7))
        ax5_2 = ax5_1.twinx()

        # Map simulation_time to pop_summary_df as well for consistent x-axis if not already there
        # (it should be there already)

        ax5_1.plot(
            pop_summary_df["simulation_time"],
            pop_summary_df["ACE_P"],
            color="tab:green",
            marker="^",
            label="ACE_P (Overall Frontier)",
            **plot_params,
        )
        ax5_1.set_xlabel("Simulation Time")
        ax5_1.set_ylabel("ACE_P (Overall Frontier)", color="tab:green")
        ax5_1.tick_params(axis="y", labelcolor="tab:green")
        ax5_1.set_ylim(-0.05, 1.05)

        ax5_2.plot(
            lineage_snapshot_df["simulation_time"],
            lineage_snapshot_df["num_unique_P_lineages"],
            color="tab:blue",
            marker="o",
            linestyle="--",
            label="Num Unique P-Lineages (Frontier)",
            **plot_params,
        )
        ax5_2.set_ylabel("Num Unique P-Lineages", color="tab:blue")
        ax5_2.tick_params(axis="y", labelcolor="tab:blue")
        ax5_2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig5.suptitle(
            f"ACE_P vs. Number of Unique P-Lineages ({sim_output_dir.name})",
            fontsize=16,
        )
        lines1, labels1 = ax5_1.get_legend_handles_labels()
        lines2, labels2 = ax5_2.get_legend_handles_labels()
        ax5_2.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=2,
        )
        fig5.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(output_plot_dir / "5_ACE_P_vs_num_lineages.png", dpi=150)
        plt.close(fig5)
    else:
        print("ACE_P column not found in population_summary.csv, skipping plot 5.")

    print("All V1 plots generated.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot lineage summary timeseries from simulation output."
    )
    parser.add_argument(
        "sim_output_dir", type=str, help="Path to the simulation output directory."
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        help="Optional: Directory to save plots. Defaults to a 'lineage_plots' subfolder in sim_output_dir.",
    )
    args = parser.parse_args()

    plot_lineage_timeseries(args.sim_output_dir, args.plot_dir)


if __name__ == "__main__":
    main()
