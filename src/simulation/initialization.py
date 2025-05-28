# microbial_colony_sim/src/simulation/initialization.py
import math
import random
from typing import List

from src.core.data_structures import SimulationConfig, HexCoord
from src.core.enums import Phenotype
from src.agents.cell import Cell
from src.agents.cell_factory import CellFactory
from src.agents.population_manager import PopulationManager
from src.grid.nutrient_environment import NutrientEnvironment
from src.grid import coordinate_utils as coord_utils
from src.core.exceptions import ConfigurationError


class Initializer:
    """Handles the setup of the initial colony."""

    def __init__(
        self,
        config: SimulationConfig,
        cell_factory: CellFactory,
        population_manager: PopulationManager,
        nutrient_env: NutrientEnvironment,
    ):
        self.config = config
        self.cell_factory = cell_factory
        self.population_manager = population_manager
        self.nutrient_env = nutrient_env  # Needed to set initial growth rates correctly

    def initialize_colony(self) -> None:
        """
        Creates the initial cluster of cells at the grid origin.
        All initial cells are typically Phenotype.G_SPECIALIST.
        Their initial inherent_growth_rate_G is set by the CellFactory
        based on the simulation's prob_daughter_inherits_prototype_1 strategy.
        """
        Cell.reset_id_counter()  # Ensure fresh cell IDs for this simulation run
        self.population_manager.clear_population()  # Clear any prior state

        initial_coords: List[HexCoord] = self._get_initial_colony_coords()

        if len(initial_coords) < self.config.initial_cell_count:
            print(
                f"Warning: Could only place {len(initial_coords)} of "
                f"{self.config.initial_cell_count} initial cells due to space constraints "
                f"at radius {self.config.initial_colony_radius}."
            )
            # Potentially raise ConfigurationError if 0 cells can be placed

        created_cells: List[Cell] = []
        for i in range(min(self.config.initial_cell_count, len(initial_coords))):
            coord = initial_coords[i]
            # Initial cells typically start as G_SPECIALIST, not lagging.
            # CellFactory handles the inherent trait determination.
            new_cell = self.cell_factory.create_initial_cell(
                coord=coord,
                initial_phenotype=Phenotype.G_SPECIALIST,  # As per conceptual model
            )
            new_cell.birth_time = 0.0  # Simulation starts at t=0

            # Determine initial growth attempt rate based on phenotype and local nutrient
            local_nutrient = self.nutrient_env.get_nutrient(new_cell.coord)
            new_cell.update_growth_attempt_rate(local_nutrient, self.config)

            created_cells.append(new_cell)
            # self.population_manager.add_cell(new_cell) # Adding all at once can be slightly cleaner

        self.population_manager.add_cells(created_cells)

        if not created_cells:
            raise ConfigurationError(
                "No initial cells could be placed. Check initial_colony_radius and initial_cell_count."
            )

        print(
            f"Initialized colony with {self.population_manager.get_cell_count()} cells."
        )

    def _get_initial_colony_coords(self) -> List[HexCoord]:
        """
        Determines the coordinates for the initial cells, forming a small,
        dense, circular-ish cluster around the origin.
        """
        # For a very small number of cells, place them in a filled disk.
        # The radius of this disk should be related to initial_colony_radius config.
        # The conceptual model mentions "Small, dense, circular-ish cluster".
        # We can use get_filled_disk with a small radius.
        # The number of cells in a disk of radius R is 1 + 3*R*(R+1).
        # So, to fit initial_cell_count, we need to find an appropriate R.
        # R=0 -> 1 cell
        # R=1 -> 1 + 3*1*2 = 7 cells
        # R=2 -> 1 + 3*2*3 = 19 cells

        # We can also use the configured initial_colony_radius, which is a float.
        # This radius likely refers to the Euclidean distance for placement, not hex grid radius.
        # However, get_filled_disk uses hex grid radius.
        # Let's try to determine a hex_radius that accommodates the cells.

        hex_placement_radius = 0
        while True:
            coords = coord_utils.get_filled_disk(HexCoord(0, 0), hex_placement_radius)
            if len(coords) >= self.config.initial_cell_count:
                break
            if (
                hex_placement_radius > 5
                and len(coords) < self.config.initial_cell_count
            ):  # Safety break
                print(
                    f"Warning: Could not find enough space for {self.config.initial_cell_count} cells "
                    f"even with hex placement radius {hex_placement_radius}. Found {len(coords)} spots."
                )
                break
            hex_placement_radius += 1

        # If the number of cells is small, the above works fine.
        # If initial_colony_radius from config is meant as a constraint:
        # coords = [c for c in coords if coord_utils.euclidean_distance_to_origin(c, self.nutrient_env.hex_pixel_size) <= self.config.initial_colony_radius]
        # This filtering might reduce available spots significantly.

        # For simplicity and density, let's take the center N coords from the smallest disk that fits them.
        potential_coords = coord_utils.get_filled_disk(
            HexCoord(0, 0), hex_placement_radius
        )

        # Sort by distance to origin to pick the centermost ones first, then shuffle for randomness if needed.
        potential_coords.sort(key=lambda c: coord_utils.hex_distance(c, HexCoord(0, 0)))

        # Take up to initial_cell_count or as many as are available
        selected_coords = potential_coords[: self.config.initial_cell_count]
        random.shuffle(
            selected_coords
        )  # Shuffle to avoid biased placement if sorted list is used directly

        return selected_coords
