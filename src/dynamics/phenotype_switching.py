# microbial_colony_sim/src/dynamics/phenotype_switching.py
from src.agents.cell import Cell
from src.core.enums import Nutrient, Phenotype

# No need for SimulationConfig here as inherent_T_lag_GL is on the cell.


def update_phenotype_based_on_nutrient(cell: Cell, local_nutrient: Nutrient) -> bool:
    """
    Updates a cell's phenotype based on the local nutrient, if the cell is not
    currently in a lag phase (Phenotype.SWITCHING_GL).

    - A G_SPECIALIST on Galactose: current_phenotype becomes Phenotype.SWITCHING_GL,
      remaining_lag_time is set to cell.inherent_T_lag_GL.
    - An L_SPECIALIST on Glucose: current_phenotype instantaneously becomes Phenotype.G_SPECIALIST.

    Args:
        cell: The cell to update.
        local_nutrient: The nutrient type at the cell's location.

    Returns:
        True if the cell's phenotype changed, False otherwise.
    """
    phenotype_changed = False

    # This function should only apply changes if the cell is NOT ALREADY in the SWITCHING_GL lag process.
    # The SWITCHING_GL state itself is resolved by lag_dynamics.process_lag_phase.
    if cell.current_phenotype == Phenotype.SWITCHING_GL:
        return False  # No changes here; lag phase handles its own transition out.

    if (
        cell.current_phenotype == Phenotype.G_SPECIALIST
        and local_nutrient == Nutrient.GALACTOSE
    ):
        cell.current_phenotype = Phenotype.SWITCHING_GL
        cell.remaining_lag_time = cell.inherent_T_lag_GL
        # If inherent_T_lag_GL is ~0, it might immediately resolve in the same step's lag processing.
        # For clarity, the lag dynamics module is responsible for changing out of SWITCHING_GL.
        phenotype_changed = True
    elif (
        cell.current_phenotype == Phenotype.L_SPECIALIST
        and local_nutrient == Nutrient.GLUCOSE
    ):
        cell.current_phenotype = Phenotype.G_SPECIALIST
        cell.remaining_lag_time = (
            0.0  # Ensure no residual lag if any was erroneously set
        )
        phenotype_changed = True

    # Note: current_growth_attempt_rate will be updated separately after all state changes
    return phenotype_changed
