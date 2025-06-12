import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re # For parsing folder names

# --- Configuration & Constants ---
RESULTS_BASE_DIR = "results/"
CSV_FILENAME = "population_summary.csv" # Ensure this is correct
PLOTS_OUTPUT_DIR = "plots_phase1_analysis" # Specific output for these plots

# Define the time window for calculating steady-state f_P and final rate
STABLE_FP_START_TIME = 500.0

# --- Helper Functions (load_simulation_data, calculate_radial_expansion_rate, get_stable_fP_frontier - remain mostly the same) ---
def load_simulation_data(filepath):
    try:
        df = pd.read_csv(filepath)
        if not all(col in df.columns for col in ['simulation_time', 'f_P_frontier', 'max_radius']):
            print(f"Warning: Missing expected columns in {filepath}")
            return None
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def calculate_radial_expansion_rate(df):
    if df is None or df.empty:
        return np.nan
    last_row = df.iloc[-1]
    if last_row['simulation_time'] == 0:
        return 0.0
    return last_row['max_radius'] / last_row['simulation_time']

def get_stable_fP_frontier(df, start_time):
    if df is None or df.empty:
        return np.nan
    stable_df = df[df['simulation_time'] >= start_time]
    if stable_df.empty:
        return np.nan
    return stable_df['f_P_frontier'].mean()

def get_canonical_strategy_name(folder_name):
    """
    Parses a folder name to extract a canonical strategy name.
    Example: "BH_diffused_k0.01_k0.01_seed123" -> "BH_diffused_k0.01_k0.01"
    Example: "BH_Sectored_S1_kGP0.005_kPG0.0005_seed456" -> "BH_Sectored_S1_kGP0.005_kPG0.0005"
    This function is CRUCIAL and needs to be tailored to your exact naming convention.
    """
    # Attempt to remove common seed patterns
    # This regex looks for "_seed" followed by digits, or just "_s" followed by digits at the end.
    # Or common replicate patterns like "_rep" or "_r" followed by digits.
    name_without_seed = re.sub(r'(_seed\d+|[_-]s\d+|[_-]rep\d+|[_-]r\d+)$', '', folder_name, flags=re.IGNORECASE)
    
    # If your names include parameters like k_GP, k_PG, delta, those should REMAIN.
    # The goal is to group runs that only differ by seed/replicate number.
    
    # Example: If folder is "BH_diffused_k0.01_k0.01_run01"
    # name_without_seed = re.sub(r'(_run\d+)$', '', folder_name)

    # If no specific seed/rep pattern is found, return the original name (minus common suffixes if any)
    # This might need more sophisticated parsing if your naming is complex.
    return name_without_seed.strip('_') # Remove trailing underscore if any

def find_strategy_replicates_from_folders(base_dir):
    """
    Finds all replicate data by scanning subfolders in base_dir.
    Groups replicates based on a canonical strategy name derived from folder names.
    Returns a dictionary: {'canonical_strategy_name': [list_of_replicate_csv_paths]}
    """
    strategies_data_paths = {}
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return strategies_data_paths

    for folder_name in os.listdir(base_dir):
        full_folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(full_folder_path):
            csv_path = os.path.join(full_folder_path, CSV_FILENAME)
            if os.path.exists(csv_path):
                canonical_name = get_canonical_strategy_name(folder_name)
                if canonical_name not in strategies_data_paths:
                    strategies_data_paths[canonical_name] = []
                strategies_data_paths[canonical_name].append(csv_path)
            # else:
            #     print(f"Debug: No '{CSV_FILENAME}' in {full_folder_path}") # Optional debug
    
    # Filter out strategies with too few replicates if desired (e.g. less than 2)
    # strategies_data_paths = {name: paths for name, paths in strategies_data_paths.items() if len(paths) >= 1} # or >= MIN_REPLICATES

    return strategies_data_paths

# --- Plotting Functions (plot_fp_frontier_replicates, plot_mean_fp_frontier_comparison, plot_expansion_rate_comparison - minor changes for output_plot_dir) ---
def plot_fp_frontier_replicates(strategy_name, replicate_dfs_list, output_plot_dir):
    if not replicate_dfs_list: return
    plt.figure(figsize=(10, 6))
    for i, df in enumerate(replicate_dfs_list):
        if df is not None: plt.plot(df['simulation_time'], df['f_P_frontier'], label=f"Rep {i+1}", alpha=0.7)
    plt.title(f"Frontier P-Fraction ($f_P$) Dynamics for {strategy_name}\n(N1-Only, $\deltâ=0.01$ assumed)")
    plt.xlabel("Simulation Time"); plt.ylabel("$f_P$ at Frontier"); plt.ylim(0, plt.gca().get_ylim()[1] if plt.gca().get_ylim()[1] > 0.5 else 1.0) # Auto-adjust or fix ylim
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    if not os.path.exists(output_plot_dir): os.makedirs(output_plot_dir)
    plt.savefig(os.path.join(output_plot_dir, f"{strategy_name}_fp_replicates.png")); plt.close()

def plot_mean_fp_frontier_comparison(strategies_summary_data, output_plot_dir):
    plt.figure(figsize=(12, 7))
    for strategy_name, summary in strategies_summary_data.items():
        if 'mean_fp_over_time' in summary and summary['mean_fp_over_time'] is not None:
            time_points = summary['mean_fp_over_time'].index
            mean_line = summary['mean_fp_over_time'].values
            plt.plot(time_points, mean_line, label=f"{strategy_name} (n={summary.get('num_replicates',0)})", linewidth=2)
            if 'std_fp_over_time' in summary and summary['std_fp_over_time'] is not None:
                std_line = summary['std_fp_over_time'].values
                plt.fill_between(time_points, np.maximum(0, mean_line - std_line), mean_line + std_line, alpha=0.2)
    plt.title(f"Mean Frontier P-Fraction ($f_P$) Comparison (N1-Only, $\deltâ=0.01$ assumed)")
    plt.xlabel("Simulation Time"); plt.ylabel("Mean $f_P$ at Frontier"); plt.ylim(0, plt.gca().get_ylim()[1] if plt.gca().get_ylim()[1] > 0.5 else 1.0)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    if not os.path.exists(output_plot_dir): os.makedirs(output_plot_dir)
    plt.savefig(os.path.join(output_plot_dir, "ALL_STRAT_mean_fp_frontier.png")); plt.close()

def plot_expansion_rate_comparison(strategies_summary_data, output_plot_dir):
    strategy_names, mean_rates, std_rates = [], [], []
    for name, summary in strategies_summary_data.items():
        if 'mean_expansion_rate' in summary and not np.isnan(summary['mean_expansion_rate']):
            strategy_names.append(f"{name}\n(n={summary.get('num_replicates',0)})")
            mean_rates.append(summary['mean_expansion_rate'])
            std_rates.append(summary.get('std_expansion_rate', 0))
    if not strategy_names: print("No expansion rate data to plot."); return
    x = np.arange(len(strategy_names))
    plt.figure(figsize=(max(8, len(strategy_names) * 1.5), 6)) # Dynamic width
    plt.bar(x, mean_rates, yerr=std_rates, capsize=5, color='skyblue', alpha=0.8)
    plt.ylabel('Mean Radial Expansion Rate'); plt.title('N1 Radial Expansion Rates Comparison ($\deltâ=0.01$ assumed)')
    plt.xticks(x, strategy_names, rotation=45, ha="right"); plt.tight_layout(); plt.grid(axis='y', linestyle='--', alpha=0.7)
    if not os.path.exists(output_plot_dir): os.makedirs(output_plot_dir)
    plt.savefig(os.path.join(output_plot_dir, "ALL_STRAT_expansion_rates.png")); plt.close()

# --- Main Analysis Logic ---
def analyze_phase1_results_from_folders(base_results_dir, output_plot_dir):
    all_strategies_summary = {}
    
    # Find all strategies and their replicate CSV paths
    strategy_data_paths_map = find_strategy_replicates_from_folders(base_results_dir)

    if not strategy_data_paths_map:
        print("No strategy data found. Check RESULTS_BASE_DIR and folder structure.")
        return None

    for strategy_name, replicate_filepaths in strategy_data_paths_map.items():
        print(f"\nAnalyzing Strategy: {strategy_name} ({len(replicate_filepaths)} replicates found)")
        
        replicate_dfs = [load_simulation_data(p) for p in replicate_filepaths]
        replicate_dfs = [df for df in replicate_dfs if df is not None and not df.empty]

        if not replicate_dfs:
            print(f"  No valid data loaded for {strategy_name}.")
            continue

        plot_fp_frontier_replicates(strategy_name, replicate_dfs, output_plot_dir)

        expansion_rates = [calculate_radial_expansion_rate(df) for df in replicate_dfs]
        stable_fPs = [get_stable_fP_frontier(df, STABLE_FP_START_TIME) for df in replicate_dfs]
        
        min_len = min(len(df) for df in replicate_dfs) if replicate_dfs else 0
        if min_len == 0:
            print(f"  Skipping time-series aggregation for {strategy_name} due to empty dataframes or zero length.")
            mean_fp_over_time, std_fp_over_time = None, None
        else:
            # Ensure all relevant DFs for concat have the 'f_P_frontier' column and are long enough
            valid_dfs_for_concat = [df for df in replicate_dfs if 'f_P_frontier' in df.columns and len(df) >= min_len]
            if not valid_dfs_for_concat:
                 print(f"  No valid DFs for f_P time-series concat for {strategy_name}.")
                 mean_fp_over_time, std_fp_over_time = None, None
            else:
                aligned_fp_data = pd.concat([df['f_P_frontier'].iloc[:min_len].reset_index(drop=True) for df in valid_dfs_for_concat], axis=1)
                aligned_time_points = valid_dfs_for_concat[0]['simulation_time'].iloc[:min_len].reset_index(drop=True)
                mean_fp_over_time = aligned_fp_data.mean(axis=1)
                std_fp_over_time = aligned_fp_data.std(axis=1)
                mean_fp_over_time.index = aligned_time_points
                std_fp_over_time.index = aligned_time_points

        all_strategies_summary[strategy_name] = {
            'num_replicates': len(replicate_dfs),
            'mean_expansion_rate': np.nanmean(expansion_rates) if expansion_rates else np.nan,
            'std_expansion_rate': np.nanstd(expansion_rates) if expansion_rates else np.nan,
            'mean_stable_fP': np.nanmean(stable_fPs) if stable_fPs else np.nan,
            'std_stable_fP': np.nanstd(stable_fPs) if stable_fPs else np.nan,
            'mean_fp_over_time': mean_fp_over_time,
            'std_fp_over_time': std_fp_over_time
        }
        print(f"  Summary for {strategy_name}:")
        print(f"    Replicates processed: {len(replicate_dfs)}")
        print(f"    Mean Expansion Rate: {all_strategies_summary[strategy_name]['mean_expansion_rate']:.4f} +/- {all_strategies_summary[strategy_name]['std_expansion_rate']:.4f}")
        print(f"    Mean Stable fP (t>{STABLE_FP_START_TIME}): {all_strategies_summary[strategy_name]['mean_stable_fP']:.4f} +/- {all_strategies_summary[strategy_name]['std_stable_fP']:.4f}")

    if all_strategies_summary:
        plot_mean_fp_frontier_comparison(all_strategies_summary, output_plot_dir)
        plot_expansion_rate_comparison(all_strategies_summary, output_plot_dir)
    
    print(f"\nAnalysis Complete. Plots saved to '{output_plot_dir}/' directory.")
    return all_strategies_summary

# --- Example Usage ---
if __name__ == "__main__":
    print("Starting Phase 1 Analysis Script...")
    
    # Create plots directory if it doesn't exist
    if not os.path.exists(PLOTS_OUTPUT_DIR):
        os.makedirs(PLOTS_OUTPUT_DIR)
        
    summary_data = analyze_phase1_results_from_folders(RESULTS_BASE_DIR, PLOTS_OUTPUT_DIR)

    if summary_data:
        print("\n--- Overall Summary Table ---")
        summary_df_list = []
        for strat_name, data in summary_data.items():
            summary_df_list.append({
                'Strategy': strat_name,
                'N Replicates': data['num_replicates'],
                'Mean Stable fP': data['mean_stable_fP'],
                'Std Stable fP': data['std_stable_fP'],
                'Mean Exp. Rate': data['mean_expansion_rate'],
                'Std Exp. Rate': data['std_expansion_rate']
            })
        summary_table = pd.DataFrame(summary_df_list)
        print(summary_table.to_string(index=False, float_format="%.4f"))