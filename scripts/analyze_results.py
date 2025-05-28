# microbial_colony_sim/scripts/analyze_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import argparse
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import fields

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
})

from src.core.data_structures import SimulationConfig
from src.visualization.kymograph import KymographGenerator

logging.basicConfig(level=logging.INFO, format="[%(levelname)s - %(module)s] %(message)s")
logger = logging.getLogger(__name__)

STRATEGY_COLORS = {
    "Responsive_HG": "orangered",
    "Responsive_LL": "dodgerblue", 
    "BetHedging_0.5": "forestgreen",
}
STRATEGY_MARKERS = {
    "Responsive_HG": "o",
    "Responsive_LL": "s",
    "BetHedging_0.5": "^",
}

def extract_params_from_summary(summary_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts key parameters from summary data's config section."""
    params = {}
    config = summary_data.get("config", {})

    params["experiment_name"] = config.get("experiment_name", "UnknownExp")
    params["W_band"] = config.get("W_band", np.nan)
    params["prob_bet"] = config.get("prob_daughter_inherits_prototype_1", np.nan)

    trade_off_cfg = config.get("trade_off_params", {})
    if isinstance(trade_off_cfg, dict):
        params["tradeoff_slope"] = trade_off_cfg.get("slope", np.nan)
    elif hasattr(trade_off_cfg, "slope"):
        params["tradeoff_slope"] = trade_off_cfg.slope
    else:
        params["tradeoff_slope"] = np.nan

    params["g_rate_P1"] = config.get("g_rate_prototype_1", np.nan)
    params["g_rate_P2"] = config.get("g_rate_prototype_2", np.nan)
    params["v_rad"] = summary_data.get("overall_radial_expansion_velocity", np.nan)

    if params["prob_bet"] == 0.0:
        params["strategy_label"] = "Responsive_HG"
    elif params["prob_bet"] == 1.0:
        params["strategy_label"] = "Responsive_LL"
    elif 0 < params["prob_bet"] < 1:
        params["strategy_label"] = f"BetHedging_{params['prob_bet']:.1f}"
    else:
        params["strategy_label"] = "UnknownStrategy"

    return params

def plot_vrad_vs_parameter(df_results: pd.DataFrame, param_x: str, fig_title: str, 
                          output_filename: Path, hue_param: str = "strategy_label",
                          xlabel: Optional[str] = None, xlim: Optional[tuple] = None, 
                          ylim: Optional[tuple] = None):
    if (df_results.empty or param_x not in df_results.columns or "v_rad" not in df_results.columns):
        logger.warning(f"Not enough data for '{fig_title}'. Missing '{param_x}' or 'v_rad'.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    agg_df = (df_results.groupby([param_x, hue_param])["v_rad"]
              .agg(["mean", "std", "count"]).reset_index())
    agg_df["sem"] = agg_df["std"] / np.sqrt(agg_df["count"])

    hue_values = sorted(agg_df[hue_param].unique())

    for hue_val in hue_values:
        subset_df = agg_df[agg_df[hue_param] == hue_val].sort_values(by=param_x)
        if not subset_df.empty:
            color = STRATEGY_COLORS.get(str(hue_val), None)
            marker = STRATEGY_MARKERS.get(str(hue_val), "o")

            ax.plot(subset_df[param_x], subset_df["mean"], marker=marker, linestyle="-",
                   label=str(hue_val), color=color, markersize=7)
            ax.fill_between(subset_df[param_x], subset_df["mean"] - subset_df["sem"],
                           subset_df["mean"] + subset_df["sem"], alpha=0.2, color=color)

    ax.set_xlabel(xlabel if xlabel else param_x.replace("_", " ").title())
    ax.set_ylabel("Mean Radial Expansion Velocity (v_rad)")
    ax.set_title(fig_title, fontsize=16)
    ax.legend(title=hue_param.replace("_", " ").title())
    ax.grid(True, linestyle="--", alpha=0.6)

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)
    logger.info(f"Saved plot: {output_filename}")

def analyze_strategy_crossovers(df_results: pd.DataFrame, output_dir: Path):
    """RQ1.1 Enhancement: Identify crossover points where strategies switch dominance."""
    if df_results.empty or "W_band" not in df_results.columns:
        logger.warning("No data for crossover analysis")
        return

    crossover_data = (df_results.groupby(["W_band", "strategy_label"])["v_rad"]
                     .agg(["mean", "std", "count"]).reset_index())
    crossover_data["sem"] = crossover_data["std"] / np.sqrt(crossover_data["count"])

    w_bands = sorted(crossover_data["W_band"].unique())
    strategies = sorted(crossover_data["strategy_label"].unique())
    
    crossovers = []
    for i in range(len(w_bands) - 1):
        w1, w2 = w_bands[i], w_bands[i + 1]
        
        perf_w1 = crossover_data[crossover_data["W_band"] == w1].set_index("strategy_label")["mean"]
        perf_w2 = crossover_data[crossover_data["W_band"] == w2].set_index("strategy_label")["mean"]
        
        if len(perf_w1) >= 2 and len(perf_w2) >= 2:
            rank_w1 = perf_w1.rank(ascending=False)
            rank_w2 = perf_w2.rank(ascending=False)
            
            for strategy in strategies:
                if strategy in rank_w1.index and strategy in rank_w2.index:
                    if rank_w1[strategy] != rank_w2[strategy]:
                        crossovers.append({
                            "w_band_range": f"{w1:.1f}-{w2:.1f}",
                            "strategy": strategy,
                            "rank_change": f"{int(rank_w1[strategy])} -> {int(rank_w2[strategy])}",
                            "performance_change": f"{perf_w1[strategy]:.4f} -> {perf_w2[strategy]:.4f}"
                        })

    if crossovers:
        crossover_df = pd.DataFrame(crossovers)
        crossover_file = output_dir / "RQ1_1_strategy_crossovers.csv"
        crossover_df.to_csv(crossover_file, index=False)
        logger.info(f"Strategy crossover analysis saved to {crossover_file}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy in strategies:
            strategy_data = crossover_data[crossover_data["strategy_label"] == strategy].sort_values("W_band")
            if not strategy_data.empty:
                color = STRATEGY_COLORS.get(strategy, None)
                marker = STRATEGY_MARKERS.get(strategy, "o")
                ax.plot(strategy_data["W_band"], strategy_data["mean"], 
                       marker=marker, label=strategy, color=color, linewidth=2, markersize=8)
                ax.fill_between(strategy_data["W_band"], 
                               strategy_data["mean"] - strategy_data["sem"],
                               strategy_data["mean"] + strategy_data["sem"],
                               alpha=0.2, color=color)
        
        for crossover in crossovers:
            w_range = crossover["w_band_range"].split("-")
            w_start, w_end = float(w_range[0]), float(w_range[1])
            ax.axvspan(w_start, w_end, alpha=0.1, color='red', 
                      label='Crossover Region' if crossover == crossovers[0] else "")
        
        ax.set_xlabel("Nutrient Band Width (W_band)")
        ax.set_ylabel("Mean Radial Expansion Velocity")
        ax.set_title("RQ1.1: Strategy Performance Crossovers")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        crossover_plot_file = output_dir / "RQ1_1_crossover_analysis.png"
        plt.savefig(crossover_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Crossover plot saved to {crossover_plot_file}")
    else:
        logger.info("No strategy crossovers detected in the data")

def analyze_bet_hedging_mechanisms(df_results: pd.DataFrame, output_dir: Path):
    """RQ2.3: Analyze bet-hedging advantage mechanisms."""
    if df_results.empty:
        logger.warning("No data for bet-hedging mechanism analysis")
        return
    
    bet_hedging_data = df_results[df_results["strategy_label"].str.contains("BetHedging", na=False)]
    responsive_data = df_results[df_results["strategy_label"].str.contains("Responsive", na=False)]
    
    if bet_hedging_data.empty:
        logger.warning("No bet-hedging data found for mechanism analysis")
        return
    
    comparison_results = []
    
    for w_band in bet_hedging_data["W_band"].unique():
        bh_perf = bet_hedging_data[bet_hedging_data["W_band"] == w_band]["v_rad"].mean()
        resp_hg_perf = responsive_data[
            (responsive_data["W_band"] == w_band) & 
            (responsive_data["strategy_label"] == "Responsive_HG")
        ]["v_rad"].mean()
        resp_ll_perf = responsive_data[
            (responsive_data["W_band"] == w_band) & 
            (responsive_data["strategy_label"] == "Responsive_LL")
        ]["v_rad"].mean()
        
        if not np.isnan(bh_perf) and not np.isnan(resp_hg_perf) and not np.isnan(resp_ll_perf):
            best_responsive = max(resp_hg_perf, resp_ll_perf)
            advantage = bh_perf - best_responsive
            comparison_results.append({
                "W_band": w_band,
                "bet_hedging_performance": bh_perf,
                "best_responsive_performance": best_responsive,
                "bet_hedging_advantage": advantage,
                "advantage_percentage": (advantage / best_responsive) * 100 if best_responsive > 0 else 0
            })
    
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        comparison_file = output_dir / "RQ2_3_bet_hedging_advantage.csv"
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Bet-hedging advantage analysis saved to {comparison_file}")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        ax1.plot(comparison_df["W_band"], comparison_df["bet_hedging_advantage"], 
                "o-", color="forestgreen", linewidth=2, markersize=8)
        ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Nutrient Band Width (W_band)")
        ax1.set_ylabel("Bet-Hedging Advantage (v_rad)")
        ax1.set_title("RQ2.3: Bet-Hedging Performance Advantage")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(comparison_df["W_band"], comparison_df["advantage_percentage"], 
                "o-", color="forestgreen", linewidth=2, markersize=8)
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Nutrient Band Width (W_band)")
        ax2.set_ylabel("Bet-Hedging Advantage (%)")
        ax2.set_title("RQ2.3: Bet-Hedging Percentage Advantage")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        advantage_plot_file = output_dir / "RQ2_3_bet_hedging_advantage.png"
        plt.savefig(advantage_plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Bet-hedging advantage plot saved to {advantage_plot_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze microbial colony simulation results.")
    parser.add_argument("--results_base_dir", type=str, default="results", 
                       help="Base directory for experiment results.")
    parser.add_argument("--exp_pattern", type=str, default="*", 
                       help="Pattern to match experiment dirs.")
    parser.add_argument("--output_dir", type=str, default="analysis_plots", 
                       help="Directory for output plots.")

    args = parser.parse_args()

    base_results_path = Path(args.results_base_dir)
    analysis_output_path = Path(args.output_dir)
    analysis_output_path.mkdir(parents=True, exist_ok=True)

    if not base_results_path.exists():
        logger.error(f"Results base dir not found: {base_results_path}")
        return

    experiment_dirs = sorted([d for d in base_results_path.glob(args.exp_pattern) if d.is_dir()])
    if not experiment_dirs:
        logger.warning(f"No experiment dirs found for pattern '{args.exp_pattern}'")
        return
    logger.info(f"Found {len(experiment_dirs)} experiments to analyze.")

    all_summary_data = []
    for exp_dir in experiment_dirs:
        summary_file = exp_dir / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r") as f:
                    summary_data = json.load(f)
                params = extract_params_from_summary(summary_data)
                all_summary_data.append(params)
            except Exception as e:
                logger.error(f"Error reading summary {summary_file}: {e}")
        else:
            logger.warning(f"Summary file not found: {summary_file}")

    if not all_summary_data:
        logger.error("No summary data loaded. Cannot generate aggregate plots.")
        return
    df_all_results = pd.DataFrame(all_summary_data)

    # RQ1.1: v_rad vs W_band
    plot_vrad_vs_parameter(df_all_results, "W_band", 
                          "RQ1.1: Optimal Strategy vs. Env. Fluctuation Scale",
                          analysis_output_path / "RQ1_1_vrad_vs_Wband.png",
                          xlabel="Nutrient Band Width (W_band)")

    # RQ2.1: v_rad vs prob_bet (for fixed W_bands)
    unique_w_bands_for_prob_sweep = df_all_results["W_band"].dropna().unique()
    for w_val in sorted(unique_w_bands_for_prob_sweep):
        df_filtered_by_w = df_all_results[np.isclose(df_all_results["W_band"], w_val)]
        if len(df_filtered_by_w["prob_bet"].dropna().unique()) > 2:
            plot_vrad_vs_parameter(df_filtered_by_w, "prob_bet",
                                  f"RQ2.1: Optimal Bet-Hedging Prob. (W_band={w_val:.1f})",
                                  analysis_output_path / f"RQ2_1_vrad_vs_prob_bet_W{w_val:.0f}.png",
                                  hue_param="experiment_name",
                                  xlabel="Prob. Daughter is Prototype 1")

    # RQ3.1: v_rad vs tradeoff_slope
    plot_vrad_vs_parameter(df_all_results, "tradeoff_slope", 
                          "RQ3.1: Impact of Trade-off Severity",
                          analysis_output_path / "RQ3_1_vrad_vs_tradeoff_slope.png",
                          xlabel="Trade-off Slope")

    # Enhanced Analysis Functions
    analyze_strategy_crossovers(df_all_results, analysis_output_path)
    analyze_bet_hedging_mechanisms(df_all_results, analysis_output_path)

    logger.info(f"Analysis plots saved in {analysis_output_path}")

if __name__ == "__main__":
    main()
