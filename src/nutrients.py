from enum import Enum

class Nutrient(Enum):
    """Nutrients available in the environment."""
    GLUCOSE = "glucose"
    GALACTOSE = "galactose"

def get_nutrient_at_coords(self, coords: tuple[int, int]) -> str:
    """The nutrient at the given coordinates."""
    raise NotImplementedError 
    pass  


