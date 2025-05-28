# microbial_colony_sim/src/dynamics/reproduction.py
import random
from typing import Optional

from src.agents.cell import Cell
from src.agents.cell_factory import CellFactory
from src.grid.hexagonal_grid import HexagonalGrid
from src.grid.nutrient_environment import NutrientEnvironment
from src.core.data_structures import (
    SimulationConfig,
    HexCoord,
)  # Assuming HexCoord is used by grid

# from src.core.exceptions import DynamicsError # Not strictly used in this version but good for future


# Ensure this function is defined at the top level of this file
def attempt_reproduction(  # <<<< CHECK THIS NAME AND DEFINITION
    parent_cell: Cell,
    grid: HexagonalGrid,
    nutrient_env: NutrientEnvironment,
    cell_factory: CellFactory,
    config: SimulationConfig,
    current_time: float,
) -> Optional[Cell]:
    """
    Handles a single parent cell's attempt to divide.
    If successful, a new daughter cell is created and returned.
    Otherwise, returns None.
    """
    if parent_cell.current_growth_attempt_rate <= 1e-9:  # Epsilon, effectively zero
        return None

    prob_div = parent_cell.current_growth_attempt_rate * config.dt
    # Prob can be > 1 if rate*dt > 1. Cap at 1.0 for probability.
    # Or, if rate is interpreted as expected number of divisions per unit time,
    # then if rate*dt > 1, it means >1 division is expected in this dt.
    # The conceptual model says "probability P_div = ACP.current_growth_attempt_rate * dt"
    # This implies it should be capped.
    prob_div = max(0.0, min(1.0, prob_div))

    if random.random() < prob_div:
        # Division attempt is successful, now find a slot.
        daughter_slot_coord: Optional[HexCoord] = grid.get_radially_outward_empty_slot(
            parent_coord=parent_cell.coord,
            # Assuming hex_pixel_size consistent with nutrient_env for radial preference
            hex_pixel_size=nutrient_env.hex_pixel_size,
        )

        if daughter_slot_coord:
            local_nutrient_at_birth = nutrient_env.get_nutrient(daughter_slot_coord)

            daughter_cell = cell_factory.create_daughter_cell(
                parent_cell=parent_cell,
                daughter_coord=daughter_slot_coord,
                local_nutrient_at_birth=local_nutrient_at_birth,
                current_time=current_time,
            )
            return daughter_cell
        else:
            # No empty slot found
            return None
    else:
        # Division attempt failed due to probability roll
        return None
