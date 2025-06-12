# src/analysis/spatial_metrics.py
import numpy as np
from typing import List, Tuple, TYPE_CHECKING

# Import necessary types from your project structure
from ..core.shared_types import Phenotype as PhenotypeEnum, HexCoord
from ..grid.coordinate_utils import axial_to_cartesian

if TYPE_CHECKING:
    from ..core.cell import Cell


def get_ordered_frontier_phenotypes(
    frontier_cells_with_coords: List[
        Tuple[HexCoord, "Cell"]
    ],  # Changed order to HexCoord, Cell
    colony_center_cartesian: Tuple[float, float],
    hex_size: float,
) -> List[PhenotypeEnum]:
    """
    Orders frontier cells angularly around a specified Cartesian center point
    and returns their phenotypes.
    """
    if not frontier_cells_with_coords:
        return []

    center_x, center_y = colony_center_cartesian
    angled_phenotypes = []

    for (
        hex_coord_obj,
        cell_obj,
    ) in frontier_cells_with_coords: 
        cart_x, cart_y = axial_to_cartesian(hex_coord_obj, hex_size)
        angle = np.arctan2(cart_y - center_y, cart_x - center_x)
        angled_phenotypes.append((angle, cell_obj.phenotype))

    angled_phenotypes.sort(key=lambda x: x[0])
    return [phenotype for angle, phenotype in angled_phenotypes]


def calculate_observed_interfaces(ordered_phenotypes_list: List[PhenotypeEnum]) -> int:
    """
    Calculates the number of phenotype switches (interfaces) in a circularly
    ordered list of phenotypes from the colony frontier.
    """
    observed_interfaces = 0
    N_total_frontier = len(ordered_phenotypes_list)

    if N_total_frontier > 1:
        for i in range(N_total_frontier):
            current_phenotype = ordered_phenotypes_list[i]
            next_phenotype = ordered_phenotypes_list[(i + 1) % N_total_frontier]
            if current_phenotype != next_phenotype:
                observed_interfaces += 1
    return observed_interfaces


def calculate_fmi_and_random_baseline(
    ordered_phenotypes_list: List[PhenotypeEnum],
) -> Tuple[float, float]:
    """
    Calculates the Frontier Mixing Index (FMI) and the FMI value expected
    for a random arrangement (FMI_random_baseline).
    FMI = 0 for maximal sectoring, FMI = 1 for maximal interspersion.
    """
    if not ordered_phenotypes_list:
        return np.nan, np.nan

    N_total_frontier = len(ordered_phenotypes_list)
    if N_total_frontier == 0:
        return np.nan, np.nan

    N_P_frontier = ordered_phenotypes_list.count(PhenotypeEnum.P_PREPARED)
    N_G_frontier = N_total_frontier - N_P_frontier

    observed_interfaces = calculate_observed_interfaces(ordered_phenotypes_list)

    if N_P_frontier == 0 or N_G_frontier == 0:
        return 0.0, 0.0

    min_int = 2.0
    max_int = 2.0 * min(N_P_frontier, N_G_frontier)

    denominator_fmi = max_int - min_int
    if abs(denominator_fmi) < 1e-9:
        fmi = 0.0
    else:
        clamped_observed_interfaces = max(min_int, float(observed_interfaces))
        fmi = (clamped_observed_interfaces - min_int) / denominator_fmi
        fmi = max(0.0, min(1.0, fmi))

    if N_total_frontier <= 1:
        e_rand_int = 0.0
    else:
        e_rand_int = (2.0 * N_P_frontier * N_G_frontier) / (N_total_frontier - 1.0)

    if abs(denominator_fmi) < 1e-9:
        fmi_random_baseline = 0.0
    else:
        clamped_e_rand_int = max(min_int, min(max_int, e_rand_int))
        fmi_random_baseline = (clamped_e_rand_int - min_int) / denominator_fmi
        fmi_random_baseline = max(0.0, min(1.0, fmi_random_baseline))

    return fmi, fmi_random_baseline
