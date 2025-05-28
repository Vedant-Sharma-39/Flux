# Microbial Colony Simulation

This project simulates the adaptive strategies of microbial colonies expanding in a spatially heterogeneous environment with growth-lag trade-offs. The simulation explores how different environmental fluctuation scales and internal adaptive strategies interact to determine colony success and dynamics.

## Research Questions

This simulation is designed to answer specific research questions about adaptive strategies in fluctuating environments:

### Overarching Theme
How do different environmental fluctuation scales (W_band) and internal adaptive strategies (controlled by prob_daughter_inherits_prototype_1 and the distinctness of prototype_1 vs. prototype_2 traits) interact to determine the success (radial expansion velocity) and adaptive dynamics of a microbial colony facing a growth-lag trade-off?

### Research Question Set 1: Environmental Fluctuation Scale Impact (W_band)

**RQ1.1**: How does the optimal adaptive strategy (in terms of maximizing radial expansion velocity, v_rad) change as the width of nutrient bands (W_band) varies?

**RQ1.2**: How does W_band affect the phenotypic composition and lag dynamics at the frontier for each strategy?

### Research Question Set 2: Bet-Hedging Efficacy and Optimization

**RQ2.1**: For a given environmental fluctuation scale (W_band), what is the optimal bet-hedging probability (prob_daughter_inherits_prototype_1) for maximizing v_rad?

**RQ2.2**: How does the "distinctness" of the bet-hedged traits (i.e., the difference between g_rate_prototype_1 and g_rate_prototype_2) affect the benefit of bet-hedging?

**RQ2.3**: Does bet-hedging primarily provide an advantage by (a) ensuring some pre-adapted (LL-type) cells are present at a G->L interface, or (b) by maintaining a diverse portfolio of traits that allows faster recovery/selection after a switch?

### Research Question Set 3: Growth-Lag Trade-off Nature

**RQ3.1**: How does the "steepness" or "severity" of the trade_off_lag_vs_growth() function affect the optimal strategy and the performance of bet-hedging?

### Research Question Set 4: Frontier Dynamics and Sectoring

**RQ4.1**: Can we visually demonstrate the spatial sorting and competition of different inherited traits (LL vs. HG types in Bet-Hedging) and phenotypic states (G-spec, L-spec, Lagging) at the expanding frontier?

## Project Structure

```
microbial_colony_sim/
├── README.md
├── requirements.txt                    # Python dependencies
├── setup.py                           # For packaging (optional)
├── config/                            # Configuration files
│   ├── default_config.yaml
│   └── experiment_configs/
├── src/                               # Source code
│   ├── agents/                        # Cell agents and population management
│   ├── analysis/                      # Analysis and metrics calculation
│   ├── core/                          # Core data structures and utilities
│   ├── dynamics/                      # Biological dynamics (lag, reproduction, etc.)
│   ├── grid/                          # Spatial grid and environment
│   ├── simulation/                    # Simulation engine and initialization
│   ├── utils/                         # Utility functions
│   └── visualization/                 # Visualization and animation
├── tests/                             # Unit and integration tests
├── scripts/                           # Scripts to run experiments and analyze results
│   ├── run_experiment.py              # Run single experiments
│   ├── batch_run_all.py               # Run comprehensive research question sweeps
│   └── analyze_results.py             # Analyze and plot results
├── results/                           # Experiment output data
├── analysis_plots/                    # Generated analysis plots
├── visualizations/                    # Generated animations and visualizations
└── docs/                              # Detailed documentation
```

## Quick Start

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### Running Experiments

#### Single Experiment
```bash
python scripts/run_experiment.py --config config/default_config.yaml
```

#### Comprehensive Research Question Analysis
```bash
python scripts/batch_run_all.py
```

This will run all experiments needed to answer the research questions:
- **RQ1.1**: W_band sweep [5.0, 10.0, 20.0, 40.0, 80.0] across strategies
- **RQ2.1**: prob_bet sweep [0.0 to 1.0] for selected W_bands
- **RQ2.2**: Trait distinctness scenarios (low vs high)
- **RQ3.1**: Trade-off slope sweep [10.0, 20.0, 40.0]

#### Analyzing Results
```bash
python scripts/analyze_results.py
```

This generates comprehensive analysis plots including:
- Strategy performance vs W_band with crossover analysis
- Optimal bet-hedging probability curves
- Trait distribution histograms
- Phenotypic composition dynamics
- Bet-hedging advantage quantification
- Kymograph visualizations

### Key Parameters

#### Environmental Parameters
- **W_band**: Width of nutrient bands (controls environmental fluctuation scale)
- **L_band**: Length of nutrient bands
- **nutrient_gradient_steepness**: Sharpness of transitions between bands

#### Strategy Parameters
- **prob_daughter_inherits_prototype_1**: Bet-hedging probability (0.0 = Responsive HG, 1.0 = Responsive LL, 0.5 = 50/50 bet-hedging)
- **g_rate_prototype_1**: Growth rate for prototype 1 (Low-Lag type)
- **g_rate_prototype_2**: Growth rate for prototype 2 (High-Growth type)

#### Trade-off Parameters
- **slope**: Steepness of growth-lag trade-off function
- **T_lag_min**: Minimum lag time

## Adaptive Strategies

The simulation implements three main adaptive strategies:

1. **Responsive High-Growth (HG)**: All daughters inherit high growth rate traits (prob_bet = 0.0)
2. **Responsive Low-Lag (LL)**: All daughters inherit low lag time traits (prob_bet = 1.0)
3. **Bet-Hedging**: Mixed strategy producing both trait types (0 < prob_bet < 1)

## Key Metrics

- **Radial Expansion Velocity (v_rad)**: Primary fitness measure
- **Frontier Lag Dynamics**: Distribution of remaining lag times at transitions
- **Phenotypic Composition**: Proportions of different cell types over time
- **Trait Diversity**: Distribution of inherited traits at the frontier

## Output Files

Each experiment generates:
- **summary.json**: Key metrics and configuration
- **time_series_data.csv**: Detailed temporal data
- **simulation.log**: Execution log
- **kymograph_perimeter_raw_data.npz**: Spatial-temporal frontier data
- **animations**: Visual representations (if enabled)

## Analysis Outputs

The analysis script generates publication-ready figures:
- **RQ1_1_vrad_vs_Wband.png**: Strategy performance vs environmental scale
- **RQ1_1_crossover_analysis.png**: Strategy dominance transitions
- **RQ2_1_vrad_vs_prob_bet_W*.png**: Optimal bet-hedging curves
- **RQ2_3_bet_hedging_advantage.png**: Bet-hedging performance advantage
- **RQ1_2_lag_hist_*.png**: Lag time distributions
- **RQ1_2_phenotype_composition_*.png**: Phenotypic dynamics
- **RQ2_2_trait_hist_*.png**: Trait diversity distributions

## Configuration

The simulation uses YAML configuration files. Key sections:

```yaml
environment:
  W_band: 40.0                    # Nutrient band width
  L_band: 200.0                   # Nutrient band length
  
strategies:
  prob_daughter_inherits_prototype_1: 0.5  # Bet-hedging probability
  g_rate_prototype_1: 0.1         # Low-lag growth rate
  g_rate_prototype_2: 0.5         # High-growth rate
  
trade_off_params:
  slope: 20.0                     # Trade-off steepness
  T_lag_min: 5.0                  # Minimum lag time
  
simulation:
  max_time: 1000.0               # Simulation duration
  dt: 0.1                        # Time step
  
visualization:
  visualization_enabled: true     # Enable animations
  animation_color_mode: "REMAINING_LAG_TIME"  # Color scheme
```

## Research Applications

This simulation framework is designed for studying:
- Adaptive strategies in fluctuating environments
- Bet-hedging vs responsive strategies
- Growth-lag trade-offs in microbial systems
- Spatial dynamics of expanding populations
- Environmental heterogeneity effects on evolution

## Citation

If you use this simulation in your research, please cite:
[Citation information to be added]

## License

[License information to be added]

## Contributing

[Contributing guidelines to be added]
