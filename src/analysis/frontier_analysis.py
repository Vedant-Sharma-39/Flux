# microbial_colony_sim/src/analysis/frontier_analysis.py
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from src.agents.cell import Cell
from src.core.enums import Nutrient, Phenotype
from src.grid import coordinate_utils as coord_utils
from src.grid.hexagonal_grid import HexagonalGrid


class FrontierAnalysis:
    """
    Performs analyses specifically related to cells at the colony frontier.
    """

    @staticmethod
    def identify_frontier_cells(
        all_cells: List[Cell],
        grid: HexagonalGrid,
        max_radial_distance: Optional[float] = None,
        radial_threshold_factor: float = 0.90,  # Cells within this factor of max_radius are candidates
        require_empty_neighbor: bool = True,
    ) -> List[Cell]:
        """
        Identifies cells considered to be at the colony frontier.
        A cell is on the frontier if:
        1. (Optional) Its radial distance is close to the maximum radial distance of any cell.
        2. It has at least one empty adjacent hexagonal slot.

        Args:
            all_cells: List of all cells in the population.
            grid: The hexagonal grid to check for empty neighbors.
            max_radial_distance: Pre-calculated maximum radial distance of the colony.
                                 If None, it will be calculated.
            radial_threshold_factor: Defines "close" to max_radius.
            require_empty_neighbor: If True, only cells with an empty neighbor are frontier cells.

        Returns:
            A list of cells identified as being on the frontier.
        """
        if not all_cells:
            return []

        if max_radial_distance is None:
            if not all_cells:
                return []
            origin = coord_utils.HexCoord(0, 0)
            # Assuming hex_pixel_size=1.0 for these calculations unless specified otherwise
            # This scale needs to be consistent with how NutrientEnvironment interprets W_band.
            # Let's assume it is 1.0 here.
            max_rad_dist_calc = 0.0
            if all_cells:  # Ensure there are cells before trying to get max
                max_rad_dist_calc = np.max(
                    [
                        coord_utils.euclidean_distance(c.coord, origin, 1.0)
                        for c in all_cells
                    ]
                )
        else:
            max_rad_dist_calc = max_radial_distance

        frontier_candidates: List[Cell] = []
        if max_rad_dist_calc > 0:  # Only filter by radius if colony has spread
            radius_lower_bound = max_rad_dist_calc * radial_threshold_factor
            for cell in all_cells:
                # Again, assuming hex_pixel_size=1.0 for this radial check
                if (
                    coord_utils.euclidean_distance(
                        cell.coord, coord_utils.HexCoord(0, 0), 1.0
                    )
                    >= radius_lower_bound
                ):
                    frontier_candidates.append(cell)
        else:  # If colony hasn't spread, all cells are potential frontier cells (e.g. initial state)
            frontier_candidates = all_cells

        if not require_empty_neighbor:
            return frontier_candidates

        frontier_cells: List[Cell] = []
        for cell in frontier_candidates:
            if grid.get_empty_adjacent_slots(cell.coord):
                frontier_cells.append(cell)

        return frontier_cells

    @staticmethod
    def analyze_frontier_lag_distribution(
        frontier_cells: List[Cell],
        nutrient_env: "NutrientEnvironment",  # Forward reference if NutrientEnvironment is complex
        target_nutrient_transition: Nutrient = Nutrient.GALACTOSE,  # e.g. G->L transition means cells encountering Galactose
    ) -> Dict[str, Any]:
        """
        Analyzes lag times for frontier cells, especially those encountering a specific nutrient.

        Args:
            frontier_cells: List of cells identified as being on the frontier.
            nutrient_env: The nutrient environment.
            target_nutrient_transition: The nutrient type that indicates a transition of interest.
                                        For G->L, this is Galactose.

        Returns:
            A dictionary with statistics on remaining_lag_time and inherent_T_lag_GL
            for relevant frontier cells.
        """
        if not frontier_cells:
            return {
                "count_at_transition_nutrient": 0,
                "remaining_lag_times_at_transition": [],
                "avg_remaining_lag_at_transition": 0.0,
                "inherent_T_lags_GL_at_transition": [],
                "avg_inherent_T_lag_at_transition": 0.0,
            }

        cells_at_transition_nutrient: List[Cell] = []
        for cell in frontier_cells:
            local_nutrient = nutrient_env.get_nutrient(cell.coord)
            # We are interested in cells that ARE on the target nutrient AND are in SWITCHING_GL state,
            # or G_SPECIALISTS that have just encountered it and will switch next.
            if local_nutrient == target_nutrient_transition:
                if cell.current_phenotype == Phenotype.SWITCHING_GL or (
                    cell.current_phenotype == Phenotype.G_SPECIALIST
                    and target_nutrient_transition == Nutrient.GALACTOSE
                ):
                    cells_at_transition_nutrient.append(cell)

        count = len(cells_at_transition_nutrient)
        remaining_lags = [
            c.remaining_lag_time
            for c in cells_at_transition_nutrient
            if c.remaining_lag_time > 1e-9
        ]
        inherent_lags = [c.inherent_T_lag_GL for c in cells_at_transition_nutrient]

        return {
            "count_at_transition_nutrient": count,
            "remaining_lag_times_at_transition": remaining_lags,
            "avg_remaining_lag_at_transition": (
                float(np.mean(remaining_lags)) if remaining_lags else 0.0
            ),
            "inherent_T_lags_GL_at_transition": inherent_lags,
            "avg_inherent_T_lag_at_transition": (
                float(np.mean(inherent_lags)) if inherent_lags else 0.0
            ),
        }

    @staticmethod
    def analyze_frontier_trait_distribution(
        frontier_cells: List[Cell],
    ) -> Dict[str, Any]:
        """
        Analyzes the distribution of inherited traits (inherent_growth_rate_G, inherent_T_lag_GL)
        for all cells currently on the frontier.

        Args:
            frontier_cells: List of cells identified as being on the frontier.

        Returns:
            A dictionary containing lists of these traits and their averages.
        """
        if not frontier_cells:
            return {
                "count_frontier_cells": 0,
                "frontier_inherent_growth_rates_G": [],
                "avg_frontier_growth_rate_G": 0.0,
                "frontier_inherent_T_lags_GL": [],
                "avg_frontier_T_lag_GL": 0.0,
            }

        growth_rates_G = [cell.inherent_growth_rate_G for cell in frontier_cells]
        t_lags_GL = [cell.inherent_T_lag_GL for cell in frontier_cells]

        return {
            "count_frontier_cells": len(frontier_cells),
            "frontier_inherent_growth_rates_G": growth_rates_G,
            "avg_frontier_growth_rate_G": (
                float(np.mean(growth_rates_G)) if growth_rates_G else 0.0
            ),
            "frontier_inherent_T_lags_GL": t_lags_GL,
            "avg_frontier_T_lag_GL": float(np.mean(t_lags_GL)) if t_lags_GL else 0.0,
        }
