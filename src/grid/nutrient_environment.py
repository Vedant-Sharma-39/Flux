# microbial_colony_sim/src/grid/nutrient_environment.py
from typing import List
from src.core.enums import Nutrient
from src.core.data_structures import HexCoord, SimulationConfig
from src.grid import coordinate_utils as coord_utils


class NutrientEnvironment:
    """
    Determines the nutrient type available at any given hexagonal coordinate
    based on concentric circular bands.
    """

    def __init__(self, config: SimulationConfig, hex_pixel_size: float = 1.0):
        """
        Args:
            config: SimulationConfig object containing W_band.
            hex_pixel_size: The effective size of a hexagon (center to vertex)
                            for Euclidean distance calculations. Must be consistent
                            with how radial distances are interpreted for W_band.
        """
        self.W_band: float = config.W_band
        self.hex_pixel_size: float = (
            hex_pixel_size  # Often 1.0 if W_band is in "hex units"
        )
        # or actual pixel size if W_band is in pixels.
        if self.W_band <= 0:
            raise ValueError("W_band (nutrient band width) must be positive.")

    def get_nutrient(self, coord: HexCoord) -> Nutrient:
        """
        Determines the nutrient type at the given hexagonal coordinate.
        The pattern is:
        0 <= radius < W_band: Glucose
        W_band <= radius < 2*W_band: Galactose
        2*W_band <= radius < 3*W_band: Glucose
        ...and so on.

        'radius' is the Euclidean distance from the cell's geometric center
        to the grid origin (0,0).
        """
        # Assuming origin is HexCoord(0,0) for nutrient bands
        origin = HexCoord(0, 0)
        radius = coord_utils.euclidean_distance(coord, origin, self.hex_pixel_size)

        # Determine which band the radius falls into
        # Example: W_band = 10
        # radius = 5   (5 / 10 = 0.5, band_index = 0) -> Glucose
        # radius = 15  (15 / 10 = 1.5, band_index = 1) -> Galactose
        # radius = 25  (25 / 10 = 2.5, band_index = 2) -> Glucose
        band_index = int(radius // self.W_band)

        if band_index % 2 == 0:  # Even-indexed bands (0, 2, 4, ...) are Glucose
            return Nutrient.GLUCOSE
        else:  # Odd-indexed bands (1, 3, 5, ...) are Galactose
            return Nutrient.GALACTOSE

    def get_band_transitions(self, max_radius_to_check: float) -> List[float]:
        """
        Returns a list of radial distances where nutrient bands transition,
        up to a maximum specified radius.
        Useful for visualization or analysis.
        """
        transitions: List[float] = []
        current_transition_radius = self.W_band
        while current_transition_radius <= max_radius_to_check:
            transitions.append(current_transition_radius)
            current_transition_radius += self.W_band
        return transitions
