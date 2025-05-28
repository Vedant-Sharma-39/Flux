# microbial_colony_sim/src/agents/population_manager.py
from typing import List, Dict, Iterable

from src.agents.cell import Cell
from src.grid.hexagonal_grid import HexagonalGrid
from src.core.data_structures import HexCoord
from src.core.exceptions import CellError


class PopulationManager:
    """
    Manages the collection of all cells in the simulation and their
    placement on the grid.
    """

    def __init__(self, grid: HexagonalGrid):
        self.grid: HexagonalGrid = grid
        self._cells: Dict[int, Cell] = (
            {}
        )  # Store cells by their unique ID for quick access

    def add_cell(self, cell: Cell) -> None:
        """Adds a new cell to the population and places it on the grid."""
        if cell.id in self._cells:
            raise CellError(
                f"Cell with ID {cell.id} already exists in PopulationManager."
            )

        self._cells[cell.id] = cell
        try:
            self.grid.add_cell(cell, cell.coord)
        except Exception as e:
            # If grid placement fails, remove from population manager to maintain consistency
            del self._cells[cell.id]
            raise e  # Re-raise the original grid error

    def add_cells(self, cells: Iterable[Cell]) -> None:
        """Adds multiple cells to the population."""
        for cell in cells:
            self.add_cell(cell)

    def remove_cell(self, cell_id: int) -> None:
        """Removes a cell from the population and the grid."""
        if cell_id not in self._cells:
            # warnings.warn(f"Cell with ID {cell_id} not found for removal.")
            return  # Cell might have already been removed or died

        cell_to_remove = self._cells[cell_id]
        self.grid.remove_cell(cell_to_remove, cell_to_remove.coord)
        del self._cells[cell_id]

    def get_cell_by_id(self, cell_id: int) -> Cell:
        """Retrieves a cell by its ID."""
        if cell_id not in self._cells:
            raise CellError(f"Cell with ID {cell_id} not found.")
        return self._cells[cell_id]

    def get_all_cells(self) -> List[Cell]:
        """Returns a list of all active cells in the population."""
        return list(self._cells.values())

    def get_cell_count(self) -> int:
        """Returns the total number of active cells."""
        return len(self._cells)

    def clear_population(self) -> None:
        """Removes all cells from the population and clears the grid."""
        # It's important to also clear the grid if the population manager is the authority
        # Or, grid should be cleared by SimulationEngine directly.
        # For now, let's assume PopulationManager also ensures grid consistency on clear.
        self.grid.clear_grid()  # Assuming HexagonalGrid has a clear_grid method
        self._cells.clear()
        Cell.reset_id_counter()  # Reset global cell ID counter for a new simulation run
