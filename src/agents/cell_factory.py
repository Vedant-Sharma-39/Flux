# microbial_colony_sim/src/agents/cell_factory.py
import random
from typing import Optional

from src.core.enums import Phenotype, Nutrient
from src.core.data_structures import HexCoord, SimulationConfig
from src.agents.cell import Cell
from src.dynamics import trade_off_functions


class CellFactory:
    """
    Responsible for creating new Cell instances, including determining their
    inherited traits based on the simulation strategy and parent cell.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def _determine_inherited_growth_rate_G(self) -> float:
        """
        Determines the daughter's inherent_growth_rate_G based on the global
        prob_daughter_inherits_prototype_1 strategy parameter.
        """
        if random.random() < self.config.prob_daughter_inherits_prototype_1:
            return self.config.g_rate_prototype_1
        else:
            return self.config.g_rate_prototype_2

    def create_initial_cell(
        self, coord: HexCoord, initial_phenotype: Phenotype = Phenotype.G_SPECIALIST
    ) -> Cell:
        """
        Creates a cell for the initial population.
        The initial inherent_growth_rate_G is set based on prob_daughter_inherits_prototype_1:
        - If prob is 0 or 1 (Responsive run), all initial cells get the corresponding prototype rate.
        - If 0 < prob < 1 (Bet-Hedging run), initial cells can be a mix or all one prototype.
          This implementation will make each initial cell's trait stochastic for bet-hedging runs.
        """
        if (
            self.config.prob_daughter_inherits_prototype_1 == 1.0
        ):  # Deterministic Responsive Low-Lag
            inherent_g_rate_G = self.config.g_rate_prototype_1
        elif (
            self.config.prob_daughter_inherits_prototype_1 == 0.0
        ):  # Deterministic Responsive High-Growth
            inherent_g_rate_G = self.config.g_rate_prototype_2
        else:  # Stochastic Bet-Hedging for initial cells as well
            inherent_g_rate_G = self._determine_inherited_growth_rate_G()

        inherent_T_lag_GL = trade_off_functions.calculate_inherent_T_lag_GL(
            inherent_g_rate_G, self.config.trade_off_params
        )

        # Initial cells are typically not lagging.
        # Their initial growth attempt rate will be set based on their phenotype and nutrient at spawn.
        cell = Cell(
            coord=coord,
            inherent_growth_rate_G=inherent_g_rate_G,
            inherent_T_lag_GL=inherent_T_lag_GL,
            current_phenotype=initial_phenotype,
            remaining_lag_time=0.0,
        )
        # Initial growth rate update will be handled by simulation engine after placement
        # and nutrient check.
        return cell

    def create_daughter_cell(
        self,
        parent_cell: Cell,
        daughter_coord: HexCoord,
        local_nutrient_at_birth: Nutrient,
        current_time: float,
    ) -> Cell:
        """
        Creates a new daughter cell during reproduction.

        Args:
            parent_cell: The parent cell.
            daughter_coord: The coordinate where the daughter cell will be placed.
            local_nutrient_at_birth: The nutrient type at the daughter's birth location.
            current_time: The current simulation time (for birth_time attribute).

        Returns:
            A new Cell object representing the daughter.
        """
        # Daughter's Inherited Trade-off Point
        daughter_inherent_g_rate_G = self._determine_inherited_growth_rate_G()
        daughter_inherent_T_lag_GL = trade_off_functions.calculate_inherent_T_lag_GL(
            daughter_inherent_g_rate_G, self.config.trade_off_params
        )

        # Daughter's Initial Phenotype and Actual Lag (as per conceptual model)
        daughter_phenotype: Phenotype
        daughter_remaining_lag: float = 0.0
        # daughter_growth_attempt_rate will be set by cell.update_growth_attempt_rate later

        if parent_cell.current_phenotype == Phenotype.L_SPECIALIST:
            # Parent must have been on Galactose if it's L_SPECIALIST and reproducing
            if local_nutrient_at_birth == Nutrient.GALACTOSE:
                daughter_phenotype = Phenotype.L_SPECIALIST
            else:  # Born onto Glucose from an L_SPECIALIST parent
                daughter_phenotype = Phenotype.G_SPECIALIST
                # No lag G_SPECIALIST -> L_SPECIALIST is instantaneous
        else:  # Parent was G_SPECIALIST (or SWITCHING_GL, but SWITCHING_GL cannot reproduce if lagging)
            # If parent was G_SPECIALIST it must have been on Glucose to reproduce.
            if local_nutrient_at_birth == Nutrient.GLUCOSE:
                daughter_phenotype = Phenotype.G_SPECIALIST
            else:  # Born onto Galactose from a G_SPECIALIST parent
                daughter_phenotype = Phenotype.SWITCHING_GL
                daughter_remaining_lag = (
                    daughter_inherent_T_lag_GL  # Set the determined lag
                )

        daughter_cell = Cell(
            coord=daughter_coord,
            inherent_growth_rate_G=daughter_inherent_g_rate_G,
            inherent_T_lag_GL=daughter_inherent_T_lag_GL,
            current_phenotype=daughter_phenotype,
            remaining_lag_time=daughter_remaining_lag,
            parent_id=parent_cell.id,
            birth_time=current_time,
        )

        # The actual growth_attempt_rate is set after this based on the assigned phenotype
        # and local nutrient, usually by calling cell.update_growth_attempt_rate().
        return daughter_cell
