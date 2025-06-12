# src/grid/grid.py

from typing import Dict, List, Optional, Tuple
from ..core.shared_types import HexCoord
from ..core.cell import Cell
from .coordinate_utils import get_neighbors


class Grid:
    """
    Manages the discrete spatial grid and the cells occupying its sites.
    Focuses on occupancy, neighborhood queries, and placing/removing cells.
    """

    def __init__(self):
        self.occupied_sites: Dict[HexCoord, Cell] = {}

    def is_occupied(self, coord: HexCoord) -> bool:
        return coord in self.occupied_sites

    def get_cell_at(self, coord: HexCoord) -> Optional[Cell]:
        return self.occupied_sites.get(coord)

    def place_cell(self, cell: Cell, coord: HexCoord) -> None:
        self.occupied_sites[coord] = cell

    def remove_cell_at(self, coord: HexCoord) -> Optional[Cell]:
        return self.occupied_sites.pop(coord, None)

    def get_empty_neighboring_coords(self, coord: HexCoord) -> List[HexCoord]:
        empty_neighbors = []
        for neighbor_coord in get_neighbors(coord):
            if not self.is_occupied(neighbor_coord):
                empty_neighbors.append(neighbor_coord)
        return empty_neighbors

    def get_all_cells_with_coords(self) -> List[Tuple[HexCoord, Cell]]:
        return list(self.occupied_sites.items())

    def count_cells(self) -> int:
        return len(self.occupied_sites)

    def get_frontier_cells_with_coords(self) -> List[Tuple[HexCoord, Cell]]:
        """
        Returns a list of (coordinate, cell_object) for cells at the frontier.
        A frontier cell is an occupied cell with at least one empty neighbor.
        """
        frontier_list = []
        for coord, cell in self.occupied_sites.items():
            if self.get_empty_neighboring_coords(coord):
                frontier_list.append((coord, cell))
        return frontier_list

    def initialize_colony(self, initial_cells_at_coords: Dict[HexCoord, Cell]):
        for coord, cell in initial_cells_at_coords.items():
            if not isinstance(coord, HexCoord) or not isinstance(cell, Cell):
                raise TypeError("initialize_colony expects Dict[HexCoord, Cell]")
            if not self.is_occupied(coord):
                self.place_cell(cell, coord)
            else:
                print(
                    f"Warning: During initialization, coord {coord} was already occupied. Skipped."
                )
