---

## README.md

**Project Title:** Spatio-Temporal Dynamics of Pre-emptive Bet-Hedging in Radially Expanding Populations with Zoned Nutrient Availability

**Last Updated:** [Date]

### 1. Introduction

This project investigates the evolutionary advantage and spatial dynamics of a pre-emptive bet-hedging strategy in microbial populations. We model a clonal population expanding radially on a 2D surface where it encounters sequential, concentric bands of a preferred nutrient (N1) alternating with bands of a challenging nutrient (N2). This spatial arrangement of nutrient zones serves as a model for environmental fluctuations encountered by an expanding colony.

The population is capable of stochastically switching between two phenotypic states:
*   **Phenotype A (Unprepared):** Grows faster on N1 but cannot utilize or grow on N2.
*   **Phenotype B (Prepared):** Grows slower on N1 (due to a cost of preparedness) but can immediately utilize and grow on N2.

We compare this bet-hedging strategy against a baseline strategy where the population is homogeneously unprepared (all Phenotype A). The simulation is implemented on a 2D discrete lattice (hexagonal grid) to explore the interplay between phenotypic switching, growth competition, spatial expansion, and the structure of the zoned nutrient environment.

The core hypothesis is that maintaining a "prepared" subpopulation (Phenotype B), even at a cost during growth on N1, can provide a significant long-term advantage in terms of overall radial expansion or biomass accumulation when encountering these nutrient zones, particularly for certain bandwidth configurations of N1 and N2.

### 2. Research Question

**Primary Question:**
In a spatially expanding population encountering sequential, concentric bands of preferred nutrient (N1) and challenging nutrient (N2) of varying widths (where narrow bands represent fast spatial "fluctuations" and wide bands represent slow "fluctuations"), does a pre-emptive bet-hedging strategy, defined by stochastic switching rates ($k_{AB}, k_{BA}$) between a fast-growing/unprepared phenotype (A) and a slower-growing/prepared phenotype (B), offer an advantage in overall radial expansion speed or total biomass compared to a homogeneously unprepared population?

**Sub-Questions:**
*   How do the optimal stochastic switching rates ($k_{AB}, k_{BA}$) and the resultant equilibrium fraction of prepared cells ($f_B$) at the expanding front change as a function of the *width of the N1 bands* and *width of the N2 bands*?
*   For a given N2 bandwidth, how does the tolerable cost of preparedness ($\delta = \lambda_A^{N1} - \lambda_B^{N1}$) for Phenotype B change with the width of the N1 bands?
*   What are the characteristic spatial patterns and population dynamics observed as the colony front transitions between N1 and N2 bands?
*   How effectively does the inflation-selection balance (during expansion through N1 bands) contribute to the persistence of the costly "prepared" phenotype (B), thereby enabling the bet-hedging strategy across different N1 bandwidths?
*   Is there a minimum N2 bandwidth required for the bet-hedging strategy to provide an advantage over a homogeneously unprepared strategy?

### 3. Model Description

The simulation is an agent-based model on a 2D hexagonal lattice. Each lattice site can be empty or occupied by a single cell. The nutrient type (N1 or N2) available at a given lattice site is determined by its radial distance from the origin, creating concentric bands.

**Cellular States & Parameters:**
*   **Phenotypes (Strategy 1 - Bet-Hedging):**
    *   `A` (Unprepared): Growth rate on N1 = $\lambda_A^{N1}$. Growth rate on N2 = 0.
    *   `B` (Prepared): Growth rate on N1 = $\lambda_B^{N1} = \lambda_A^{N1} - \delta$. Growth rate on N2 = $\lambda^{N2}$.
    *   Switching: $A \xrightarrow{k_{AB}} B$, $B \xrightarrow{k_{BA}} A$ (stochastic, per time step, occurs irrespective of current nutrient).
*   **Phenotypes (Strategy 2 - Homogeneous Unprepared - Baseline):**
    *   `A` (Unprepared): Growth rate on N1 = $\lambda_A^{N1}$. Growth rate on N2 = 0. (This strategy cannot grow on N2 unless a rare mutation to B is explicitly modeled for direct comparison, which should be stated if done).
*   **Reproduction:** Cells reproduce into adjacent empty sites. Probability of reproduction per time step is proportional to their current growth rate (which depends on their phenotype and the nutrient type at their current location). Contact inhibition if no empty neighbors.

**Environmental Structure:**
*   Concentric circular bands of Nutrient N1 and Nutrient N2.
*   Bandwidths ($W_{N1}, W_{N2}$) are key parameters. The sequence can be alternating (N1-N2-N1-N2...) or more complex.

### 5. Analysis & Metrics

*   **Primary Metrics:**
    *   Average radial expansion speed of the colony front.
    *   Total population size (biomass) after a fixed simulation time or after traversing a set number of nutrient bands.
*   **Secondary Metrics:**
    *   Fraction of prepared (B) cells at the expanding front as a function of radius ($f_B(r)$) and within different nutrient bands.
    *   Success rate of traversing N2 bands (i.e., does the colony go extinct or successfully bridge the N2 zone?).
    *   Spatial statistics: Patch size distribution of B cells at the N1/N2 interface and within N2 bands, sectoring patterns.
*   **Comparison:** Directly compare the primary metrics for Strategy 1 (potentially optimized over $k_{AB}, k_{BA}, \delta$) versus Strategy 2 across various N1/N2 bandwidth configurations.

---