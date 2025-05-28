# microbial_colony_sim/src/dynamics/lag_dynamics.py
from src.agents.cell import Cell
from src.core.enums import Phenotype


def process_lag_phase(cell: Cell, dt: float) -> bool:
    """
    Processes the lag phase for a cell.
    If the cell is in Phenotype.SWITCHING_GL and has remaining_lag_time,
    it decrements the lag time. If lag time reaches zero, the phenotype
    is changed to Phenotype.L_SPECIALIST.

    Args:
        cell: The cell to process.
        dt: The time step duration.

    Returns:
        True if the cell's phenotype changed as a result of lag completion, False otherwise.
    """
    phenotype_changed = False
    if (
        cell.current_phenotype == Phenotype.SWITCHING_GL
        and cell.remaining_lag_time > 1e-9
    ):  # Epsilon
        cell.remaining_lag_time -= dt
        if cell.remaining_lag_time <= 1e-9:  # Epsilon
            cell.remaining_lag_time = 0.0
            cell.current_phenotype = Phenotype.L_SPECIALIST
            phenotype_changed = True
            # Note: current_growth_attempt_rate will be updated separately after all state changes
    elif (
        cell.remaining_lag_time > 1e-9
        and cell.current_phenotype != Phenotype.SWITCHING_GL
    ):
        # This case should ideally not happen if logic is correct elsewhere.
        # It means a cell has lag time but isn't in the SWITCHING_GL state.
        # For robustness, one might clear the lag time or log a warning.
        # print(f"Warning: Cell {cell.id} has lag time but is not in SWITCHING_GL state. Phenotype: {cell.current_phenotype}")
        cell.remaining_lag_time = 0.0  # Clear inconsistent lag

    return phenotype_changed
