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
    """Extracts key parameters from summary data's config section and experiment name."""
    params = {}
    config = summary_data.get("config", {})

    params["experiment_name"] = config.get("experiment_name", "UnknownExp")
    exp_name = params["experiment_name"]
    
    # Extract parameters from experiment name (more reliable than config)
    params["W_band"] = extract_w_band_from_name(exp_name)
    params["prob_bet"] = extract_prob_bet_from_name(exp_name)
    params["tradeoff_slope"] = extract_tradeoff_slope_from_name(exp_name)
    
    # Fallback to config if not found in name
    if np.isnan(params["W_band"]):
        params["W_band"] = config.get("W_band", np.nan)
    if np.isnan(params["prob_bet"]):
        params["prob_bet"] = config.get("prob_daughter_inherits_prototype_1", np.nan)
    if np.isnan(params["tradeoff_slope"]):
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

    # Determine strategy label from experiment name and prob_bet
    params["strategy_label"] = determine_strategy_label(exp_name, params["prob_bet"])

    return params

def extract_w_band_from_name(exp_name: str) -> float:
    """Extract W_band value from experiment name."""
    import re
    # Look for patterns like W5, W10, W20, W40, W80
    match = re.search(r'_W(\d+)(?:_|$)', exp_name)
    if match:
        return float(match.group(1))
    return np.nan

def extract_prob_bet_from_name(exp_name: str) -> float:
    """Extract probability bet value from experiment name."""
    import re
    # Look for patterns like P0.5, P0.0, P1.0
    match = re.search(r'_P(\d+\.?\d*)(?:_|$)', exp_name)
    if match:
        return float(match.group(1))
    # Look for BetHedging patterns
    match = re.search(r'BetHedging(\d+\.?\d*)', exp_name)
    if match:
        return float(match.group(1))
    return np.nan

def extract_tradeoff_slope_from_name(exp_name: str) -> float:
    """Extract tradeoff slope value from experiment name."""
    import re
    # Look for patterns like S10, S20, S40
    match = re.search(r'_S(\d+)(?:_|$)', exp_name)
    if match:
        return float(match.group(1))
    return np.nan

def determine_strategy_label(exp_name: str, prob_bet: float) -> str:
    """Determine strategy label from experiment name and prob_bet."""
    if "ResponsiveHG" in exp_name or prob_bet == 0.0:
        return "Responsive_HG"
    elif "ResponsiveLL" in exp_name or prob_bet == 1.0:
        return "Responsive_LL"
    elif "BetHedging" in exp_name or (0 < prob_bet < 1):
        return f"BetHedging_{prob_bet:.1f}"
    else:
        return "UnknownStrategy"

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

def analyze_lag_dynamics(base_results_path: Path, df_results: pd.DataFrame, output_dir: Path):
    """RQ1.2: Analyze lag time distributions for different strategies."""
    logger.info("Analyzing lag dynamics...")
    
    # Get representative experiments for each strategy
    strategies_to_analyze = ["Responsive_HG", "Responsive_LL", "BetHedging_0.5"]
    w_band_to_analyze = 40.0  # Use W_band=40 as representative
    
    for strategy in strategies_to_analyze:
        strategy_data = df_results[
            (df_results["strategy_label"] == strategy) & 
            (np.isclose(df_results["W_band"], w_band_to_analyze))
        ]
        
        if strategy_data.empty:
            continue
            
        # Get the first experiment for this strategy
        exp_name = strategy_data.iloc[0]["experiment_name"]
        exp_dir = base_results_path / exp_name
        
        # Try to load time series data
        time_series_file = exp_dir / "time_series_data.csv"
        if time_series_file.exists():
            try:
                ts_data = pd.read_csv(time_series_file)
                
                # Create lag histogram if lag data exists
                if "avg_remaining_lag_at_transition" in ts_data.columns:
                    lag_data = ts_data["avg_remaining_lag_at_transition"].dropna()
                    if len(lag_data) > 0:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.hist(lag_data, bins=20, alpha=0.7, color=STRATEGY_COLORS.get(strategy, "gray"))
                        ax.set_xlabel("Remaining Lag Time at Transition")
                        ax.set_ylabel("Frequency")
                        ax.set_title(f"RQ1.2: Lag Distribution - {strategy} (W_band={w_band_to_analyze})")
                        ax.grid(True, alpha=0.3)
                        
                        lag_hist_file = output_dir / f"RQ1_2_lag_hist_{strategy}_W{w_band_to_analyze:.0f}.png"
                        plt.savefig(lag_hist_file, dpi=300, bbox_inches="tight")
                        plt.close()
                        logger.info(f"Lag histogram saved: {lag_hist_file}")
                        
            except Exception as e:
                logger.warning(f"Error processing time series for {exp_name}: {e}")

def analyze_phenotype_composition(base_results_path: Path, df_results: pd.DataFrame, output_dir: Path):
    """RQ1.2: Analyze phenotypic composition dynamics."""
    logger.info("Analyzing phenotype composition...")
    
    strategies_to_analyze = ["Responsive_HG", "Responsive_LL", "BetHedging_0.5"]
    w_band_to_analyze = 40.0
    
    for strategy in strategies_to_analyze:
        strategy_data = df_results[
            (df_results["strategy_label"] == strategy) & 
            (np.isclose(df_results["W_band"], w_band_to_analyze))
        ]
        
        if strategy_data.empty:
            continue
            
        exp_name = strategy_data.iloc[0]["experiment_name"]
        exp_dir = base_results_path / exp_name
        
        time_series_file = exp_dir / "time_series_data.csv"
        if time_series_file.exists():
            try:
                ts_data = pd.read_csv(time_series_file)
                
                # Look for phenotype columns
                phenotype_cols = [col for col in ts_data.columns if "phenotype" in col.lower() or 
                                col in ["G_SPECIALIST", "L_SPECIALIST", "SWITCHING_GL"]]
                
                if phenotype_cols and "time" in ts_data.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    for col in phenotype_cols:
                        if col in ts_data.columns:
                            ax.plot(ts_data["time"], ts_data[col], label=col, linewidth=2)
                    
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Cell Count")
                    ax.set_title(f"RQ1.2: Phenotype Composition - {strategy} (W_band={w_band_to_analyze})")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    comp_file = output_dir / f"RQ1_2_phenotype_composition_{strategy}_W{w_band_to_analyze:.0f}.png"
                    plt.savefig(comp_file, dpi=300, bbox_inches="tight")
                    plt.close()
                    logger.info(f"Phenotype composition plot saved: {comp_file}")
                    
            except Exception as e:
                logger.warning(f"Error processing phenotype composition for {exp_name}: {e}")

def analyze_trait_distributions(df_results: pd.DataFrame, output_dir: Path):
    """RQ2.2: Analyze trait diversity distributions."""
    logger.info("Analyzing trait distributions...")
    
    # Analyze trait distinctness experiments
    trait_experiments = df_results[df_results["experiment_name"].str.contains("TraitDistinct", na=False)]
    
    if trait_experiments.empty:
        logger.warning("No trait distinctness experiments found")
        return
    
    # Group by distinctness level
    high_distinct = trait_experiments[trait_experiments["experiment_name"].str.contains("HighDistinct")]
    low_distinct = trait_experiments[trait_experiments["experiment_name"].str.contains("LowDistinct")]
    
    if not high_distinct.empty and not low_distinct.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot performance comparison
        categories = ["High Distinctness", "Low Distinctness"]
        performances = [high_distinct["v_rad"].mean(), low_distinct["v_rad"].mean()]
        errors = [high_distinct["v_rad"].std(), low_distinct["v_rad"].std()]
        
        ax1.bar(categories, performances, yerr=errors, capsize=5, 
               color=["darkgreen", "lightgreen"], alpha=0.7)
        ax1.set_ylabel("Mean Radial Expansion Velocity")
        ax1.set_title("RQ2.2: Performance vs Trait Distinctness")
        ax1.grid(True, alpha=0.3)
        
        # Plot trait distribution (mock data based on g_rate values)
        g_rates_high = high_distinct["g_rate_P1"].tolist() + high_distinct["g_rate_P2"].tolist()
        g_rates_low = low_distinct["g_rate_P1"].tolist() + low_distinct["g_rate_P2"].tolist()
        
        ax2.hist([g_rates_high, g_rates_low], bins=10, alpha=0.7, 
                label=["High Distinctness", "Low Distinctness"],
                color=["darkgreen", "lightgreen"])
        ax2.set_xlabel("Growth Rate")
        ax2.set_ylabel("Frequency")
        ax2.set_title("RQ2.2: Trait Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        trait_file = output_dir / "RQ2_2_trait_hist_distinctness.png"
        plt.savefig(trait_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Trait distribution plot saved: {trait_file}")

def generate_simple_kymographs(base_results_path: Path, df_results: pd.DataFrame, output_dir: Path):
    """Generate simple kymograph visualizations for representative experiments."""
    logger.info("Generating simple kymographs...")
    
    # Select representative experiments
    representative_experiments = []
    for strategy in ["Responsive_HG", "Responsive_LL", "BetHedging_0.5"]:
        strategy_data = df_results[df_results["strategy_label"] == strategy]
        if not strategy_data.empty:
            # Get experiment with W_band=40 if available
            w40_data = strategy_data[np.isclose(strategy_data["W_band"], 40.0)]
            if not w40_data.empty:
                representative_experiments.append(w40_data.iloc[0]["experiment_name"])
    
    for exp_name in representative_experiments:
        exp_dir = base_results_path / exp_name
        kymo_data_file = exp_dir / "kymograph_perimeter_raw_data.npz"
        
        if kymo_data_file.exists():
            try:
                # Load the kymograph data directly
                loaded_data = np.load(kymo_data_file, allow_pickle=True)
                
                # Find the first available attribute to plot
                attribute_keys = set()
                for key in loaded_data.keys():
                    if key.endswith("_times"):
                        attribute_keys.add(key.replace("_times", ""))
                    elif key.endswith("_values"):
                        attribute_keys.add(key.replace("_values", ""))
                
                if attribute_keys:
                    # Use the first available attribute
                    attr_name = list(attribute_keys)[0]
                    times_key = f"{attr_name}_times"
                    values_key = f"{attr_name}_values"
                    
                    if times_key in loaded_data and values_key in loaded_data:
                        times = loaded_data[times_key]
                        values_matrix = loaded_data[values_key]
                        
                        if times.size > 0 and values_matrix.size > 0 and values_matrix.ndim == 2:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Create kymograph plot
                            im = ax.imshow(
                                values_matrix.T,
                                aspect="auto",
                                origin="lower",
                                cmap="viridis",
                                extent=[times.min(), times.max(), 0, values_matrix.shape[1]]
                            )
                            
                            ax.set_xlabel("Time")
                            ax.set_ylabel("Angular Position")
                            ax.set_title(f"RQ4.1: Kymograph - {exp_name}")
                            
                            try:
                                plt.colorbar(im, ax=ax, label=attr_name.replace("_", " ").title())
                            except:
                                pass
                            
                            plt.tight_layout()
                            kymo_output_file = output_dir / f"RQ4_1_kymograph_{exp_name}.png"
                            plt.savefig(kymo_output_file, dpi=300, bbox_inches="tight")
                            plt.close()
                            logger.info(f"Kymograph saved: {kymo_output_file}")
                        else:
                            logger.warning(f"Invalid kymograph data shape for {exp_name}")
                    else:
                        logger.warning(f"Missing time or values data for {exp_name}")
                else:
                    logger.warning(f"No valid attributes found in kymograph data for {exp_name}")
                    
            except Exception as e:
                logger.warning(f"Error generating kymograph for {exp_name}: {e}")

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
                                  analysis_output_path / f"RQ2_1_vrad_vs_prob_bet_W
