# Project Title: Spatio-Temporal Dynamics of Pre-emptive Bet-Hedging in Microbial Colonies Under Zoned Nutrient Availability

**Last Updated:** 2024-07-27

## 1. Project Overview

This project investigates the evolutionary dynamics of microbial populations expanding in spatially structured environments with fluctuating nutrient availability. Specifically, we explore whether a pre-emptive bet-hedging strategy—where a clonal population stochastically maintains a mix of "growth-optimized/unprepared" and "costly/prepared" phenotypes—offers a fitness advantage over a homogeneous population that relies solely on reactive adaptation.

The simulation is an agent-based model implemented in Python on a 2D hexagonal lattice. Microbial colonies expand via frontier-only cell division (non-motile cells). The environment is characterized by pre-defined concentric circular bands of a preferred nutrient (N1) and a challenging nutrient (N2), with varying non-dimensionalized bandwidths ($\hat{W}$) representing different "speeds" of environmental fluctuation encountered by the expanding colony front.

All key biological rates (growth, switching, lag) and environmental parameters (bandwidths) are non-dimensionalized for generalizability, using the growth rate of the unprepared phenotype on N1 as the reference timescale and cell diameter as the reference length scale.

## 2. Core Research Question

Under what regimes of environmental structure (defined by N1/N2 non-dimensional bandwidths $\hat{W}$), cellular trade-offs (relative cost of preparedness $\hat{\delta}$, relative efficiencies of N2 adaptation), and adaptive capabilities, does a pre-emptive bet-hedging strategy (BH) provide a superior long-term fitness advantage (e.g., in average radial expansion speed $\hat{v}_{avg}$) compared to a homogeneous reactive adaptation strategy (HR)?

**Key Sub-Questions Being Addressed (aligned with Phased Research Plan):**

*   What are the optimal non-dimensional switching rates ($\hat{k}_{GP}^*$, $\hat{k}_{PG}^*$) for the BH strategy, and what is the resulting quasi-equilibrium fraction of "prepared" cells ($f_P^*$) at the frontier under various conditions (e.g., N1-only, N1/N2 alternating)? (Phase 1)
*   How does the cost of preparedness ($\hat{\delta}$) affect the viability and optimal parameters of the BH strategy? (Phase 4)
*   What is the long-term fitness ($\hat{v}_{avg,HR}$) of the HR strategy under equivalent environmental conditions? (Phase 2)
*   How do the spatio-temporal adaptation dynamics differ between optimized BH and HR strategies, particularly at nutrient interfaces? (Phase 3)
*   What are the characteristic spatio-temporal patterns (sectoring, patchiness, interface dynamics via Frontier Mixing Index - FMI, population bottlenecks) associated with each strategy? (Phase 3 & Analysis)
*   How do spatial phenomena (e.g., inflation-selection balance, genetic drift/surfing at the expansion front) influence the persistence of the costly "prepared" phenotype and the overall success of bet-hedging? (Ongoing Analysis)
*   How sensitive is the relative advantage of BH vs. HR to the "harshness" of the N2 environment for unprepared cells (defined by $\alpha_U^{N2}, \hat{L}_U^{N2}$) and the quality of N2 for prepared/adapted cells ($\hat{\lambda}_P^{N2}$, $\hat{\lambda}_U^{N2,adapted}$)? (Phase 4)

## 3. Model Strategies & Key Phenotypes (Non-Dimensional Parameters)

*   Reference growth rate: $\lambda_{ref}$ (Growth rate of G/U-type on N1). Set $\hat{\lambda}_G^{N1} = \hat{\lambda}_U^{N1} = 1$.
*   Reference length scale: $d_{cell}$ (cell diameter). Bandwidths $\hat{W} = W/d_{cell}$.
*   Reference timescale: $T_{ref} = 1/\lambda_{ref}$. Rates $\hat{k} = k \cdot T_{ref}$, Lags $\hat{L} = L \cdot \lambda_{ref}$.

**Strategy 1: Bet-Hedging (BH)**
*   Maintains two interconverting phenotypes:
    *   **Phenotype G (Growth-Optimized/Unprepared):** Growth $\hat{\lambda}_G^{N1}=1$ on N1. Poor N2 adaptation (low intrinsic probability $\alpha_G^{N2}$, long non-dimensional lag $\hat{L}_G^{N2}$).
    *   **Phenotype P (Prepared):** Slower growth $\hat{\lambda}_P^{N1}=1-\hat{\delta}$ on N1 (where $\hat{\delta}$ is the relative cost of preparedness). Efficient N2 adaptation (high $\alpha_P^{N2} \approx 1$, short $\hat{L}_P^{N2} \approx 0$). Grows at $\hat{\lambda}_P^{N2}$ on N2.
*   Stochastic switching: $G \xrightarrow{\hat{k}_{GP}} P$ and $P \xrightarrow{\hat{k}_{PG}} G$.

**Strategy 2: Homogeneous Reactive Adaptation (HR)**
*   Population consists solely of **Phenotype U (Unprepared Standard):**
    *   On N1: Growth rate $\hat{\lambda}_U^{N1}=1$.
    *   On N2: Adapts with an intrinsic probability $\alpha_U^{N2}$ after a non-dimensional lag $\hat{L}_U^{N2}$, then grows at $\hat{\lambda}_U^{N2,adapted}$.
*   **Note:** Phenotype G (BH) and Phenotype U (HR) share identical N1 growth and N2 adaptation characteristics ($\alpha_G^{N2} = \alpha_U^{N2}$, $\hat{L}_G^{N2} = \hat{L}_U^{N2}$, $\hat{\lambda}_G^{N2,adapted} = \hat{\lambda}_U^{N2,adapted}$). $\hat{\lambda}_P^{N2}$ is set equal to $\hat{\lambda}_U^{N2,adapted}$ for fair post-adaptation comparison.

## 4. Simulation Framework

*   **Core Logic:** Implemented in Python. See `src/` directory.
    *   `src/core/shared_types.py`: Defines core Enums (Phenotype, Nutrient, ConflictResolutionRule) and the `SimulationParameters` dataclass which holds all non-dimensionalized parameters.
    *   `src/core/cell.py`: Defines the Cell agent state, including phenotype, division timer, and N2 adaptation lag state.
    *   `src/grid/coordinate_utils.py`: Utilities for hexagonal grid mathematics and coordinate conversions.
    *   `src/grid/grid.py`: Manages cell positions and occupancy on the hexagonal lattice.
    *   `src/environment/environment_rules.py`: Defines nutrient zones based on radial distance and phenotype-specific rules (growth rates, N2 adaptation parameters $\alpha, \hat{L}$).
    *   `src/evolution/phenotype_switching.py`: Handles stochastic G $\leftrightarrow$ P switching based on $\hat{k}_{GP}, \hat{k}_{PG}$.
    *   `src/evolution/conflict_resolution.py`: Implements rules (e.g., RANDOM_CHOICE) for resolving competition for empty sites during reproduction.
    *   `src/data_logging/logger.py`: Manages saving simulation parameters (JSON) and time-series data (population counts, $f_P$, max radius, spatial metrics) to CSV files. Also logs nutrient transition events.
    *   `src/visualization/plotter.py`: Generates Matplotlib visualizations (colony snapshots with nutrient backgrounds, population dynamics plots).
    *   `src/analysis/spatial_metrics.py`: Calculates metrics like observed interfaces between phenotypes at the frontier and the Frontier Mixing Index (FMI).
    *   `src/main_simulation.py`: Orchestrates simulation runs, loads parameters from YAML configuration files, and manages the main simulation loop.
*   **Configuration:** Simulation parameters are managed via YAML configuration files (see `config.yaml` and `configs/` directory).
*   **Output:** Results (logs, plots, saved parameters, raw grid snapshots) are saved to the `results/` directory, organized by run.
*   **Analysis Scripts:** Standalone scripts like `cell_fraction.py` are used for post-simulation analysis of batch runs (e.g., for Phase 1).

## 5. Current Status & Next Steps (Aligning with Phased Plan)

The project is actively progressing through its phased research plan.

*   **Phase 0 (Model Finalization & Parameter Definition):** The non-dimensionalized simulation framework is implemented, capable of simulating both BH and HR strategies. Baseline parameter ranges are being refined. The `config.yaml` reflects an example parameter set where, for instance, a dimensional `lambda_G_N1 = 0.10` and `cost_delta_P = 0.01` would yield a non-dimensional cost $\hat{\delta} = 0.1$ (P-type is 10% slower on N1).
*   **Phase 1 (Optimizing the Bet-Hedging (BH) Strategy):** This phase is currently underway.
    *   Initial explorations are focusing on understanding the dynamics of the BH strategy, particularly the establishment of a quasi-steady-state fraction of Prepared cells ($f_P^*$) at the frontier.
    *   Simulations in N1-only environments are being run with varying switching rates ($\hat{k}_{GP}, \hat{k}_{PG}$) and a fixed cost of preparedness (e.g., analyses using `cell_fraction.py` examine runs where $\hat{\delta}=0.01$, i.e., P-type is 1% slower on N1). These runs help identify switching rate regimes that maintain a non-trivial $f_P^*$ despite the cost.
    *   The goal is to identify optimal or representative "tuned" BH strategies (characterized by $\hat{k}_{GP}^*$, $\hat{k}_{PG}^*$, and resulting $f_P^*$) for different costs $\hat{\delta}$.
*   **Immediate Next Steps:**
    1.  Complete systematic sweeps for Phase 1 to identify robust $\hat{k}_{GP}^*$, $\hat{k}_{PG}^*$ for a few key $\hat{\delta}$ values and a baseline $\hat{W}$.
    2.  Proceed to **Phase 2:** Evaluate the Homogeneous Reactive (HR) strategy using the same $\hat{W}$ and G-type/U-type N2 adaptation parameters.
    3.  Move to **Phase 3:** Conduct a comparative analysis of the optimized BH strategy vs. HR strategy under baseline conditions, focusing on fitness ($\hat{v}_{avg}$) and spatio-temporal dynamics.

## 6. How to Run

1.  **Prerequisites:** Ensure Python (3.8+) is installed.
2.  **Dependencies:** Create a virtual environment and install required packages:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install numpy scipy matplotlib PyYAML
    ```
    (These are also listed in `requirements.txt`).
3.  **Configuration:** Create or select a YAML configuration file (e.g., `config.yaml` or one in the `configs/` directory). Adjust parameters as needed for your experiment.
4.  **Execution:** Navigate to the root directory of the project (where this `README.md` is located). Execute the simulation using:
    ```bash
    python -m src.main_simulation path/to/your_config_file.yaml
    ```
    For example:
    ```bash
    python -m src.main_simulation config.yaml
    ```
5.  **Output:** Simulation outputs (CSV logs, JSON parameters, PNG snapshots) will be saved to the directory specified by `output_dir` in the YAML configuration file.

## 7. Future Directions (Beyond Current Phased Plan)

Upon completion of the core comparative analysis (Phases 0-3), further explorations (Phase 4 & 5) will include:

*   **Parameter Space Exploration:** Systematically vary environmental "fluctuation speed" ($\hat{W}$), cost of preparedness ($\hat{\delta}$), difficulty of reactive adaptation ($\alpha_U^{N2}, \hat{L}_U^{N2}$), and N2 quality ($\hat{\lambda}_P^{N2}$) to map out "phase diagrams" of where BH or HR is superior.
*   **Asymmetric Bandwidths:** Investigate scenarios where $\hat{W}_{N1} \neq \hat{W}_{N2}$.
*   **Mechanistic Insights:** Delve deeper into the mechanistic drivers (e.g., role of spatial persistence, bottlenecks at interfaces) for strategy success.
*   **Advanced Spatial Analysis:** Potentially incorporate more sophisticated spatial statistics to characterize colony structure and mixing.
*   **Model Extensions (Long-term):** Consider explicit nutrient diffusion/consumption or the evolution of switching rates.