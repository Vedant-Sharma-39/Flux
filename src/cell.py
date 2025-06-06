from dataclasses import dataclass, field
from nutrients import Nutrient, nutrient_at_coords
from state import State
from grid import get_neighbors
from typing import List, Tuple, Optional
@dataclass
class Cell:
    """A class representing a cell in a grid."""

    # Working in a 2-dimensional coordinate system
    coords: tuple[int, int]
    State: State
    lag_time: float 
    lineage_id: Optional[int] = None

    def get_nutrient(self) -> Nutrient:
        """The nutrient on the given coordinates.""" 
        return nutrient_at_coords(self.coords)


