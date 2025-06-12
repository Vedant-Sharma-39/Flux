# src/environment/environment_rules.py

from typing import Tuple
from ..core.shared_types import HexCoord, Nutrient, Phenotype, SimulationParameters
from ..grid.coordinate_utils import axial_to_cartesian_sq_distance_from_origin


class EnvironmentRules:
    def __init__(self, params: SimulationParameters):
        self.params = params
        if self.params.nutrient_bands:  # Ensure sorted only if list is not empty
            self.params.nutrient_bands.sort(key=lambda x: x[0])

    def get_nutrient_at_coord(self, coord: HexCoord) -> Nutrient:
        dist_sq_from_origin = axial_to_cartesian_sq_distance_from_origin(
            coord, self.params.hex_size
        )
        for max_radius_sq, nutrient_type in self.params.nutrient_bands:
            if dist_sq_from_origin < max_radius_sq:
                return nutrient_type
        return Nutrient.NONE

    def get_growth_rate(
        self,
        cell_phenotype: Phenotype,
        nutrient: Nutrient,
        is_cell_adapted_to_N2: bool = False,
    ) -> float:
        """
        Returns the specific growth rate (per unit time).
        is_cell_adapted_to_N2 is True if a G-type cell has finished its N2 lag.
        """
        if nutrient == Nutrient.N1_PREFERRED:
            if cell_phenotype == Phenotype.G_UNPREPARED:
                return self.params.lambda_G_N1
            elif cell_phenotype == Phenotype.P_PREPARED:
                return self.params.lambda_G_N1 - self.params.cost_delta_P
        elif nutrient == Nutrient.N2_CHALLENGING:
            if cell_phenotype == Phenotype.G_UNPREPARED:
                return (
                    self.params.lambda_G_N2_adapted if is_cell_adapted_to_N2 else 0.0
                )  # Growth only after lag & successful alpha_G_N2
            elif cell_phenotype == Phenotype.P_PREPARED:
                # Assuming P_PREPARED also needs to pass its (short) lag and alpha_P_N2 check
                # For simplicity, if alpha_P_N2 is high and lag_P_N2 is low, this rate applies quickly.
                # This will be handled by the main loop's adaptation check for P cells.
                # Here, we return the *potential* growth rate if adapted.
                return self.params.lambda_P_N2
        return 0.0

    def get_phenotype_switching_probabilities(self) -> Tuple[float, float]:
        # These are probabilities per time step (k * dt)
        return self.params.k_GP, self.params.k_PG

    def get_N2_adaptation_params(self, phenotype: Phenotype) -> Tuple[float, float]:
        """Returns (alpha_N2_adaptation_prob, characteristic_lag_N2_nondim) for the phenotype."""
        if phenotype == Phenotype.G_UNPREPARED:
            return self.params.alpha_G_N2, self.params.lag_G_N2
        elif phenotype == Phenotype.P_PREPARED:
            return self.params.alpha_P_N2, self.params.lag_P_N2
        return 0.0, float("inf")  # Should not happen for known phenotypes
