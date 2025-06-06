# Keep a dictionary of coordinates of all the population
# and their corresponding cell objects.

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from cell import Cell
from phenotype import Phenotype
from state import State
from grid import get_neighbors


@dataclass
class PopulationManager:
    """A class to manage the population of cells in a grid."""

    population: Dict[Tuple[int, int], Cell] = field(default_factory=dict)
    _total_cells: int = 0

    phenotype_map: Dict[Phenotype, Set[Cell]] = field(
        default_factory=lambda: {
            phenotype_enum_member: set() for phenotype_enum_member in Phenotype
        }
    )
    frontier_cells: Set[Cell] = field(default_factory=set)

    def get_empty_neighbors(self, coords: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get the coordinates of empty neighboring cells."""
        neighbors = get_neighbors(coords)
        empty_neighbors = []
        for neighbor in neighbors:
            if neighbor not in self.population:
                empty_neighbors.append(neighbor)
        return empty_neighbors

    def is_frontier_cell(self, coords: Tuple[int, int]) -> bool:
        """Check if the cell at the given coordinates is a frontier cell."""
        if coords not in self.population:
            raise ValueError(f"No cell found at {coords}.")

        empty_neighbors = self.get_empty_neighbors(coords)
        return bool(empty_neighbors)

    def update_frontier_cells(self, coords: Tuple[int, int]) -> None:
        """Update the frontier cells based on the neighbors of the cell at the given coordinates."""

        if is_frontier_cell := self.is_frontier_cell(coords):
            self.frontier_cells.add(self.population[coords])
        else:
            if self.population[coords] in self.frontier_cells:
                self.frontier_cells.remove(self.population[coords])

    def add_cell(self, coords: Tuple[int, int], cell: Cell) -> None:
        """Add a cell to the population at the specified coordinates."""
        if coords in self.population:
            raise ValueError(f"Cell already exists at {coords}.")

        # If not empty neighbor, add to frontier cells
        neighbors = get_neighbors(coords)
        empty_neighbors = [n for n in neighbors if n not in self.population]

        self.population[coords] = cell
        self._total_cells += 1
        self.phenotype_map[cell.phenotype].add(cell)
        # Check all the neighbours and if they don't have any empty neighbors, remove them from frontier cells
        self.update_frontier_cells(coords)
        for neighbor in neighbors:
            if neighbor in self.population:
                self.update_frontier_cells(neighbor)

    def remove_cell(self, coords: Tuple[int, int]) -> None:
        NotImplementedError("Not needed for current setup.")

    def get_cell_by_coord(self, coords: Tuple[int, int]) -> Cell:
        """Get the cell at the specified coordinates."""
        if coords not in self.population:
            raise ValueError(f"No cell found at {coords}.")
        return self.population[coords]

    def get_cells_by_phenotype(self, phenotype: Phenotype) -> List[Cell]:
        """Get all cells of a specific phenotype."""
        if phenotype not in self.phenotype_map:
            raise ValueError(f"Phenotype {phenotype} not recognized.")
        return list(self.phenotype_map[phenotype])

    def get_total_cells(self) -> int:
        """Get the total number of cells in the population."""
        return self._total_cells
