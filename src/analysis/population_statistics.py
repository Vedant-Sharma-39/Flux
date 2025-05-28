# microbial_colony_sim/src/analysis/population_statistics.py
from collections import Counter
from typing import List, Dict, Any
import numpy as np

from src.agents.cell import Cell
from src.core.enums import Phenotype
from src.grid import (
    coordinate_utils as coord_utils,
)  # For radial distance


class PopulationStatistics:
    """Calculates and stores various population-level statistics."""

    @staticmethod
    def calculate_phenotype_counts(cells: List[Cell]) -> Dict[Phenotype, int]:
        """Counts the number of cells for each phenotype."""
        counts = Counter(cell.current_phenotype for cell in cells)
        # Ensure all phenotypes are present in the output, even if count is 0
        for p_type in Phenotype:
            if p_type not in counts:
                counts[p_type] = 0
        return dict(counts)

    @staticmethod
    def calculate_total_cell_count(cells: List[Cell]) -> int:
        """Returns the total number of cells."""
        return len(cells)

    @staticmethod
    def calculate_radial_distribution_stats(
        cells: List[Cell],
        origin: coord_utils.HexCoord = coord_utils.HexCoord(0, 0),
        hex_pixel_size: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculates statistics about the radial distribution of cells.
        - Average radial distance
        - Max radial distance (colony radius)
        - Min radial distance
        - Std dev of radial distance
        """
        if not cells:
            return {
                "avg_radial_distance": 0.0,
                "max_radial_distance": 0.0,
                "min_radial_distance": 0.0,
                "std_dev_radial_distance": 0.0,
                "median_radial_distance": 0.0,
            }

        radial_distances = [
            coord_utils.euclidean_distance(cell.coord, origin, hex_pixel_size)
            for cell in cells
        ]

        return {
            "avg_radial_distance": float(np.mean(radial_distances)),
            "max_radial_distance": float(np.max(radial_distances)),
            "min_radial_distance": float(np.min(radial_distances)),
            "std_dev_radial_distance": float(np.std(radial_distances)),
            "median_radial_distance": float(np.median(radial_distances)),
        }

    @staticmethod
    def get_growth_lag_trait_distribution(cells: List[Cell]) -> Dict[str, List[float]]:
        """
        Extracts lists of inherent growth rates on G and inherent lag times on G->L.
        Useful for understanding the genetic makeup of the population.
        """
        if not cells:
            return {
                "inherent_growth_rates_G": [],
                "inherent_T_lags_GL": [],
            }

        growth_rates_G = [cell.inherent_growth_rate_G for cell in cells]
        t_lags_GL = [cell.inherent_T_lag_GL for cell in cells]

        return {
            "inherent_growth_rates_G": growth_rates_G,
            "inherent_T_lags_GL": t_lags_GL,
        }

    @staticmethod
    def get_current_lag_time_distribution(cells: List[Cell]) -> List[float]:
        """
        Extracts the distribution of current remaining_lag_time for all cells.
        Focuses on cells that are actually lagging (phenotype SWITCHING_GL or positive lag time).
        """
        if not cells:
            return []

        # Only consider cells that are actually in a lag state or have positive lag time
        lag_times = [
            cell.remaining_lag_time
            for cell in cells
            if cell.remaining_lag_time > 1e-9  # Epsilon
            # or cell.current_phenotype == Phenotype.SWITCHING_GL # Could also filter by this
        ]
        return lag_times

    @staticmethod
    def collect_all_statistics(
        cells: List[Cell],
        current_time: float,
        hex_pixel_size_for_nutrient_env: float = 1.0,  # Should match NutrientEnvironment's scale
    ) -> Dict[str, Any]:
        """Collects a comprehensive set of statistics for a given time step."""

        stats = {
            "time": current_time,
            "total_cell_count": PopulationStatistics.calculate_total_cell_count(cells),
            "phenotype_counts": PopulationStatistics.calculate_phenotype_counts(cells),
        }

        radial_stats = PopulationStatistics.calculate_radial_distribution_stats(
            cells, hex_pixel_size=hex_pixel_size_for_nutrient_env
        )
        stats.update(radial_stats)

        # Trait distributions (could be computationally intensive for very large populations every step)
        # May want to sample these less frequently.
        # trait_stats = PopulationStatistics.get_growth_lag_trait_distribution(cells)
        # stats.update(trait_stats)

        # Current lag times
        # stats["current_lag_time_distribution"] = PopulationStatistics.get_current_lag_time_distribution(cells)

        return stats
