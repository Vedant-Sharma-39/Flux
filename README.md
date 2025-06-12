---

**Revised Research Plan: Spatio-Temporal Dynamics of Pre-emptive Bet-Hedging vs. Homogeneous Reactive Adaptation**

**Overall Goal:**
To determine the conditions under which a pre-emptive bet-hedging strategy, characterized by stochastic interconversion between two distinct phenotypes, provides a superior long-term fitness advantage over a homogeneous population relying solely on post-shift reactive adaptation, within a spatially expanding colony encountering alternating nutrient bands.

**Non-Dimensional Framework:**
*   Reference growth rate: $\lambda_{ref}$ (Growth rate of the standard/unprepared cell type on preferred Nutrient 1). Set $\hat{\lambda}_{ref}^{N1} = 1$.
*   Reference length scale: $d_{cell}$ (cell diameter). Bandwidths $\hat{W} = W/d_{cell}$.
*   Reference timescale: $T_{ref} = 1/\lambda_{ref}$. Rates $\hat{k} = k \cdot T_{ref}$, Lags $\hat{L} = L \cdot \lambda_{ref}$.

**Strategy Definitions & Key Phenotypes (Non-Dimensional Parameters):**

1.  **Strategy 1: Bet-Hedging (BH)**
    *   Maintains two interconverting phenotypes while on Nutrient 1 (N1):
        *   **Phenotype G (Growth-Optimized):** Maximizes growth on N1 ($\hat{\lambda}_G^{N1}=1$). Poor adaptation to Challenging Nutrient 2 (N2): low intrinsic adaptation probability $\alpha_G^{N2}$, long lag $\hat{L}_G^{N2}$.
        *   **Phenotype P (Prepared):** Slower growth on N1 ($\hat{\lambda}_P^{N1}=1-\hat{\delta}$, where $\hat{\delta}$ is the relative cost of preparedness). Efficient adaptation to N2: high adaptation probability $\alpha_P^{N2} \approx 1$, short lag $\hat{L}_P^{N2} \approx 0$. Grows at $\hat{\lambda}_P^{N2}$ on N2.
    *   Stochastic switching: $G \xrightarrow{\hat{k}_{GP}} P$ and $P \xrightarrow{\hat{k}_{PG}} G$.

2.  **Strategy 2: Homogeneous Reactive Adaptation (HR)**
    *   Population consists solely of **Phenotype U (Unprepared Standard):**
        *   On N1: Growth rate $\hat{\lambda}_U^{N1}=1$.
        *   On N2: Adapts with an intrinsic probability $\alpha_U^{N2}$ after a non-dimensional lag $\hat{L}_U^{N2}$, then grows at $\hat{\lambda}_U^{N2,adapted}$.
    *   **Note:** Phenotype G from BH and Phenotype U from HR share the same growth characteristics on N1 and the same *intrinsic* (poor) adaptation capability to N2 ($\alpha_G^{N2} = \alpha_U^{N2}$, $\hat{L}_G^{N2} = \hat{L}_U^{N2}$, $\hat{\lambda}_G^{N2,adapted} = \hat{\lambda}_U^{N2,adapted}$). The BH strategy's advantage comes solely from also having Phenotype P.

**Simulation Environment:**
*   2D hexagonal lattice, frontier-only cell expansion.
*   Alternating concentric bands of Nutrient N1 and N2, each of non-dimensional width $\hat{W}$ (initially $\hat{W}_{N1}=\hat{W}_{N2}=\hat{W}$).

---

**Plan of Action (Phased Approach):**

**Phase 0: Model Finalization & Parameter Definition**
1.  Implement the non-dimensionalized simulation framework capable of simulating both BH and HR strategies.
2.  Define baseline values and exploration ranges for key non-dimensional parameters:
    *   $\hat{\delta}$ (cost of P on N1).
    *   $\alpha_G^{N2}$ (low, e.g., 0.01), $\hat{L}_G^{N2}$ (long, e.g., 10-20) for G-type/U-type adaptation to N2.
    *   $\alpha_P^{N2}$ (high, e.g., 0.9-1.0), $\hat{L}_P^{N2}$ (short, e.g., 0-1) for P-type adaptation to N2.
    *   $\hat{\lambda}_P^{N2}$ (growth of P on N2), and set $\hat{\lambda}_G^{N2,adapted} = \hat{\lambda}_U^{N2,adapted} = \hat{\lambda}_P^{N2}$ for fair post-adaptation comparison.
    *   $\hat{k}_{GP}, \hat{k}_{PG}$ (switching rates for BH).
    *   $\hat{W}$ (N1/N2 bandwidth).

**Phase 1: Optimizing the Bet-Hedging (BH) Strategy**
*   **Research Question:** For a given environmental structure (fixed $\hat{W}$) and fixed phenotypic properties ($\hat{\delta}, \alpha_P^{N2}, \hat{L}_P^{N2}, \hat{\lambda}_P^{N2}$, and the G-type properties), what are the optimal non-dimensional switching rates ($\hat{k}_{GP}^*$, $\hat{k}_{PG}^*$) that maximize long-term fitness (e.g., average non-dimensional radial expansion speed $\hat{v}_{avg,BH}$)?
*   **Method:**
    1.  Select baseline values for $\hat{W}$, $\hat{\delta}$, and N2-related P-type/G-type parameters from Phase 0.
    2.  Perform a parameter sweep over $\hat{k}_{GP}$ and $\hat{k}_{PG}$.
    3.  Run multiple simulation replicates for each ($\hat{k}_{GP}, \hat{k}_{PG}$) pair, tracking $\hat{v}_{avg,BH}$.
*   **Output:** Identify $\hat{k}_{GP}^*$, $\hat{k}_{PG}^*$, the resulting optimal steady-state fraction of Prepared cells on N1 ($f_P^*$), and the maximized fitness $\hat{v}_{avg,BH}^*$.

**Phase 2: Evaluating the Homogeneous Reactive (HR) Strategy**
*   **Research Question:** What is the long-term fitness ($\hat{v}_{avg,HR}$) of the HR strategy under the same environmental structure ($\hat{W}$) and intrinsic G-type/U-type adaptation parameters used in Phase 1?
*   **Method:**
    1.  Use the same $\hat{W}$ and the defined properties for Phenotype U (which mirror Phenotype G's N1 growth and N2 adaptation characteristics: $\alpha_U^{N2}=\alpha_G^{N2}, \hat{L}_U^{N2}=\hat{L}_G^{N2}, \hat{\lambda}_U^{N2,adapted}=\hat{\lambda}_G^{N2,adapted}$).
    2.  Run multiple simulation replicates for the HR strategy.
*   **Output:** Measure $\hat{v}_{avg,HR}$.

**Phase 3: Comparative Analysis and Spatial Dynamics**
*   **Research Question:** Under baseline conditions, does the optimized BH strategy ($\hat{v}_{avg,BH}^*$) outperform the HR strategy ($\hat{v}_{avg,HR}$)? How do their spatio-temporal adaptation dynamics differ?
*   **Method:**
    1.  Compare $\hat{v}_{avg,BH}^*$ with $\hat{v}_{avg,HR}$.
    2.  Analyze saved colony snapshots for both strategies at N1/N2 interfaces and during traversal of N2 bands.
*   **Focus:** Quantify differences in population bottlenecks, speed of recovery into N1 from N2, spatial coherence of adapting fronts, and the role of $f_P^*$ in BH success.

**Phase 4: Parameter Space Exploration – Identifying Regimes of Optimality**
*   **Research Question:** How does the relative fitness advantage of the optimized BH strategy versus the HR strategy change with variations in:
    *   Environmental "fluctuation speed" (non-dimensional bandwidth $\hat{W}$)?
    *   Cost of preparedness ($\hat{\delta}$)?
    *   Difficulty of reactive adaptation for unprepared cells (e.g., increasing $\hat{L}_U^{N2}$ or decreasing $\alpha_U^{N2}$ for the HR strategy, and correspondingly for Phenotype G in BH)?
    *   Quality of the challenging nutrient N2 (varying $\hat{\lambda}_P^{N2}$ and $\hat{\lambda}_U^{N2,adapted}$)?
    *   Asymmetric bandwidths ($\hat{W}_{N1} \neq \hat{W}_{N2}$)?
*   **Method:** For each key parameter variation:
    1.  Re-optimize BH switching rates ($\hat{k}_{GP}, \hat{k}_{PG}$) if the parameter change significantly affects the balance (or use previously found optima as a starting point). This gives $\hat{v}_{avg,BH}^*(\text{new params})$.
    2.  Simulate HR with correspondingly adjusted parameters to get $\hat{v}_{avg,HR}(\text{new params})$.
    3.  Compare their performance.
*   **Output:** "Phase diagrams" or plots illustrating regions in parameter space where BH is superior, HR is superior (if ever), or their performance is comparable. Analysis of how optimal $f_P^*$ and switching rates scale with these parameters.

**Phase 5: Synthesis and Mechanistic Insights**
*   **Research Question:** What are the key mechanistic drivers (e.g., cost of preparedness, efficiency of reactive adaptation, spatial persistence of prepared cells via inflation-selection) that determine the success of each strategy across the explored parameter space?
*   **Method:** Integrate findings from all phases. Relate observed macroscopic performance differences to the underlying cellular strategies and their spatial manifestations.
*   **Output:** A comprehensive understanding of the trade-offs governing the evolution of pre-emptive bet-hedging in spatially structured, fluctuating environments.

