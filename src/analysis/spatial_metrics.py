# src/analysis/spatial_metrics.py
import numpy as np
import math
from typing import List, Tuple, TYPE_CHECKING

# Import necessary types from your project structure
from ..core.shared_types import Phenotype as PhenotypeEnum, HexCoord
from ..grid.coordinate_utils import axial_to_cartesian

if TYPE_CHECKING:
    from ..core.cell import Cell  # For type hinting Cell object


def get_ordered_frontier_cells_with_angles(
    frontier_cells_with_coords: List[
        Tuple[HexCoord, "Cell"]
    ],  # Assumes (HexCoord, Cell)
    colony_center_cartesian: Tuple[float, float],
    hex_size: float,
) -> List[Tuple[float, PhenotypeEnum]]:  # Returns list of (angle_rad, phenotype)
    """
    Orders frontier cells angularly around a specified Cartesian center point
    and returns their angle and phenotype.
    Angles are in radians from -pi to pi.
    """
    if not frontier_cells_with_coords:
        return []

    center_x, center_y = colony_center_cartesian
    angled_data = []

    for hex_coord_obj, cell_obj in frontier_cells_with_coords:
        cart_x, cart_y = axial_to_cartesian(hex_coord_obj, hex_size)
        angle = np.arctan2(cart_y - center_y, cart_x - center_x)  # Output is -pi to pi
        angled_data.append((angle, cell_obj.phenotype))

    angled_data.sort(key=lambda x: x[0])  # Sort by angle
    return angled_data


def get_phenotypes_from_ordered_data(
    ordered_angled_phenotypes: List[Tuple[float, PhenotypeEnum]],
) -> List[PhenotypeEnum]:
    """Extracts just the phenotypes from the angularly sorted list of (angle, phenotype)."""
    return [phenotype for _, phenotype in ordered_angled_phenotypes]


def get_angles_of_target_phenotype(
    ordered_angled_phenotypes: List[Tuple[float, PhenotypeEnum]],
    target_phenotype: PhenotypeEnum,
) -> List[float]:  # Returns list of angles in [0, 2pi)
    """Extracts angles for a target phenotype, normalized to [0, 2*pi)."""
    angles = []
    for angle, phenotype in ordered_angled_phenotypes:
        if phenotype == target_phenotype:
            normalized_angle = (angle + 2 * np.pi) % (
                2 * np.pi
            )  # Normalize to [0, 2pi)
            angles.append(normalized_angle)
    return angles


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
            next_phenotype = ordered_phenotypes_list[
                (i + 1) % N_total_frontier
            ]  # Circular
            if current_phenotype != next_phenotype:
                observed_interfaces += 1
    return observed_interfaces


def calculate_fmi_and_random_baseline(
    ordered_phenotypes_list: List[PhenotypeEnum],
) -> Tuple[float, float]:  # Returns (FMI, FMI_random_baseline)
    """
    Calculates the Frontier Mixing Index (FMI) and the FMI value expected
    for a random arrangement (FMI_random_baseline).
    FMI = 0 for maximal sectoring (min_interfaces), FMI = 1 for maximal interspersion (max_interfaces).
    """
    if not ordered_phenotypes_list:
        return np.nan, np.nan

    N_total_frontier = len(ordered_phenotypes_list)
    if N_total_frontier <= 1:
        return 0.0, 0.0  # Or np.nan, np.nan if preferred for single/no cells

    N_P_frontier = ordered_phenotypes_list.count(
        PhenotypeEnum.P_PREPARED
    )  # Assumes P is the "minority" or focus
    N_G_frontier = N_total_frontier - N_P_frontier

    observed_interfaces = calculate_observed_interfaces(ordered_phenotypes_list)

    if N_P_frontier == 0 or N_G_frontier == 0:  # Homogeneous frontier
        return 0.0, 0.0  # FMI is 0 (perfectly sectored)

    min_interfaces = (
        2.0  # Minimum possible interfaces if both types present and perfectly sectored
    )
    max_interfaces = 2.0 * min(
        N_P_frontier, N_G_frontier
    )  # Max interfaces if perfectly alternating

    denominator_fmi = max_interfaces - min_interfaces
    fmi = 0.0
    if (
        abs(denominator_fmi) > 1e-9
    ):  # Avoid division by zero if max_interfaces equals min_interfaces (e.g. only 1 of each type)
        # Clamp observed_interfaces to be within [min_interfaces, max_interfaces]
        clamped_observed_interfaces = max(
            min_interfaces, min(max_interfaces, float(observed_interfaces))
        )
        fmi = (clamped_observed_interfaces - min_interfaces) / denominator_fmi
    # Ensure FMI is within [0,1] due to potential floating point issues or edge cases with clamping
    fmi = max(0.0, min(1.0, fmi))

    # Expected interfaces for a random arrangement
    e_rand_int = 0.0
    if N_total_frontier > 1:  # Avoid division by zero
        e_rand_int = (2.0 * N_P_frontier * N_G_frontier) / (N_total_frontier - 1.0)

    fmi_random_baseline = 0.0
    if abs(denominator_fmi) > 1e-9:
        # Clamp expected random interfaces to be within [min_interfaces, max_interfaces]
        clamped_e_rand_int = max(min_interfaces, min(max_interfaces, e_rand_int))
        fmi_random_baseline = (clamped_e_rand_int - min_interfaces) / denominator_fmi
    # Ensure FMI_random is within [0,1]
    fmi_random_baseline = max(0.0, min(1.0, fmi_random_baseline))

    return fmi, fmi_random_baseline


def calculate_angular_coverage_entropy(
    target_phenotype_angles_normalized: List[
        float
    ],  # Expects angles in [0, 2*pi) for the target phenotype
    num_bins: int = 12,  # e.g., 12 bins for 30-degree sectors
) -> float:
    """
    Calculates the normalized Shannon entropy of a target phenotype's distribution
    across angular bins on the frontier.
    Result is ~1 for uniform angular coverage, ~0 for concentration in few bins.
    """
    N_target_frontier = len(target_phenotype_angles_normalized)

    if N_target_frontier == 0:
        return np.nan  # No target cells, entropy undefined
    if N_target_frontier == 1:
        return 0.0  # Single cell is perfectly concentrated (and trivially uniform in its own bin)

    # Angles should already be normalized to [0, 2*pi) by the calling function
    bin_edges = np.linspace(0, 2 * np.pi, num_bins + 1)
    counts_in_bin, _ = np.histogram(target_phenotype_angles_normalized, bins=bin_edges)

    # Filter out zero counts before calculating probabilities for entropy formula
    probabilities = counts_in_bin[counts_in_bin > 0] / N_target_frontier

    if not probabilities.any():  # Should not happen if N_target_frontier > 0
        return 0.0

    entropy = -np.sum(probabilities * np.log2(probabilities))

    # Max entropy is log2(number of distinct states occupied).
    # If N_target_frontier < num_bins, at most N_target_frontier bins can be occupied.
    # If N_target_frontier >= num_bins, at most num_bins can be occupied.
    max_possible_occupied_bins = min(num_bins, N_target_frontier)

    if max_possible_occupied_bins <= 1:  # Avoid log2(1) or log2(0)
        return 0.0  # If all cells fall into one bin, or only one cell

    max_entropy = np.log2(max_possible_occupied_bins)

    if abs(max_entropy) < 1e-9:  # Effectively zero max entropy
        return 0.0

    normalized_entropy = entropy / max_entropy
    return max(0.0, min(1.0, normalized_entropy))  # Clamp to [0,1]
