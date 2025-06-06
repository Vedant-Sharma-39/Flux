from enum import Enum
from nutrients import Nutrient

class State(Enum):
    """Phenotypes of cells."""

    GLUCOSE = "glucose"
    GALACTOSE = "galactose"


def get_growth_rate(state: State, nutrient: Nutrient ) -> float:
    """Get the growth rate of a cell based on its  current state and the nutrient available."""

    if state == State.GLUCOSE:

        if nutrient == Nutrient.GLUCOSE:
            return 1.0  # Fast growth on glucose

        elif nutrient == Nutrient.GALACTOSE:
            return 0  # No growth on galactose for glucose phenotype
        
    elif state == State.GALACTOSE:

        if nutrient == Nutrient.GLUCOSE:
            return 0.5  # Slower growth on glucose

        elif nutrient == Nutrient.GALACTOSE:
            return 1.0  # Fast growth on galactose              