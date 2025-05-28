# microbial_colony_sim/src/grid/hexagonal_grid.py
from typing import Dict, List, Optional, Set, Tuple

from src.core.data_structures import HexCoord
from src.agents.cell import (
    Cell,
)  # Forward reference, will be defined fully later
from src.grid import coordinate_utils as coord_utils
from src.core.exceptions import GridError


class HexagonalGrid:
    """
    Manages the state of the hexagonal grid, primarily cell occupancy.
    Does not inherently know about nutrients; that's the Environment's job.
    """

    def __init__(self, bounded_radius: Optional[float] = None):
        """
        Initializes the grid.
        _occupied_cells maps HexCoord to the Cell occupying it.

        Args:
            bounded_radius (Optional[float]): If provided, grid is considered bounded.
                                              Not fully implemented in this basic version.
        """
        self._occupied_cells: Dict[HexCoord, Cell] = {}
        self.bounded_radius = (
            bounded_radius  # Not used in current logic but good for future
        )

    def is_occupied(self, coord: HexCoord) -> bool:
        """Checks if a given coordinate is occupied by a cell."""
        return coord in self._occupied_cells

    def get_cell_at(self, coord: HexCoord) -> Optional[Cell]:
        """Returns the cell at a given coordinate, or None if empty."""
        return self._occupied_cells.get(coord)

    def add_cell(self, cell: Cell, coord: HexCoord) -> None:
        """Adds a cell to the specified coordinate on the grid."""
        if self.is_occupied(coord):
            # Ensure the cell being added is the one we think is there, or raise error
            existing_cell = self._occupied_cells[coord]
            if existing_cell.id != cell.id:
                raise GridError(
                    f"Coordinate {coord} is already occupied by cell {existing_cell.id}. "
                    f"Cannot add new cell {cell.id}."
                )
            # If it's the same cell (e.g. re-adding during an update), it's fine.
            self._occupied_cells[coord] = cell  # Ensure latest cell object is stored
        else:
            self._occupied_cells[coord] = cell

        # Ensure the cell's internal coordinate matches
        if cell.coord != coord:
            # This could be an assertion or a forced update depending on design.
            # For now, let's assume cell.coord is source of truth or set by caller.
            # If hexagonal_grid is responsible for setting cell.coord, it should do it here.
            # Let's assume cell.coord is correctly set before calling add_cell.
            pass

    def remove_cell(self, cell: Cell, coord: HexCoord) -> None:
        """Removes a cell from the specified coordinate on the grid."""
        if self.is_occupied(coord) and self._occupied_cells[coord].id == cell.id:
            del self._occupied_cells[coord]
        elif self.is_occupied(coord):
            # Occupied, but by a different cell. This is an issue.
            raise GridError(
                f"Attempted to remove cell {cell.id} from {coord}, "
                f"but cell {self._occupied_cells[coord].id} is there."
            )
        # If not occupied, do nothing (idempotent removal)

    def get_empty_adjacent_slots(self, coord: HexCoord) -> List[HexCoord]:
        """Finds all empty (unoccupied) hexagonal slots adjacent to the given coordinate."""
        empty_slots: List[HexCoord] = []
        for neighbor_coord in coord_utils.get_neighbors(coord):
            if not self.is_occupied(neighbor_coord):
                empty_slots.append(neighbor_coord)
        return empty_slots

    def get_all_occupied_coords(self) -> List[HexCoord]:
        """Returns a list of all currently occupied coordinates."""
        return list(self._occupied_cells.keys())

    def get_all_cells(self) -> List[Cell]:
        """Returns a list of all cells currently on the grid."""
        return list(self._occupied_cells.values())

    def get_radially_outward_empty_slot(
        self,
        parent_coord: HexCoord,
        origin: HexCoord = HexCoord(0, 0),
        hex_pixel_size: float = 1.0,
    ) -> Optional[HexCoord]:
        """
        Finds an empty adjacent slot, prioritizing radially outward ones from origin.
        If multiple equally good (same distance, outward), chooses randomly.
        If no outward, but same distance, chooses randomly from those.
        If only inward, chooses randomly from those.
        If no empty slots, returns None.

        Args:
            parent_coord: The coordinate of the parent cell.
            origin: The origin of the grid for radial distance calculation.
            hex_pixel_size: Size of hex for Euclidean distance. Consistent with nutrient env.

        Returns:
            The HexCoord of a suitable empty slot, or None.
        """
        empty_slots = self.get_empty_adjacent_slots(parent_coord)
        if not empty_slots:
            return None

        parent_dist_to_origin = coord_utils.euclidean_distance(
            parent_coord, origin, hex_pixel_size
        )

        outward_slots: List[HexCoord] = []
        same_dist_slots: List[HexCoord] = []
        inward_slots: List[HexCoord] = (
            []
        )  # Technically, all remaining are inward or same if not outward

        for slot_coord in empty_slots:
            slot_dist_to_origin = coord_utils.euclidean_distance(
                slot_coord, origin, hex_pixel_size
            )
            # Using a small epsilon for floating point comparisons of distance
            epsilon = 1e-9
            if slot_dist_to_origin > parent_dist_to_origin + epsilon:
                outward_slots.append(slot_coord)
            elif abs(slot_dist_to_origin - parent_dist_to_origin) < epsilon:
                same_dist_slots.append(slot_coord)
            else:  # slot_dist_to_origin < parent_dist_to_origin - epsilon
                inward_slots.append(slot_coord)

        import random

        if outward_slots:
            return random.choice(outward_slots)
        elif same_dist_slots:
            return random.choice(same_dist_slots)
        elif (
            inward_slots
        ):  # Should be all remaining empty_slots if no outward/same_dist
            return random.choice(inward_slots)
        else:  # Should not be reached if empty_slots is not empty
            return random.choice(empty_slots) if empty_slots else None

    def clear_grid(self):
        """Removes all cells from the grid."""
        self._occupied_cells.clear()
