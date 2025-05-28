from enum import Enum

from enum import Enum


class CellColorMode(Enum):
    PHENOTYPE = "phenotype"
    REMAINING_LAG = "remaining_lag"
    INHERENT_LAG_GL = "inherent_T_lag_GL"
    INHERENT_GROWTH_G = "inherent_growth_rate_G"


class Nutrient(Enum):
    """Nutrient types available in the environment."""

    GLUCOSE = "GLUCOSE"
    GALACTOSE = "GALACTOSE"

    def __str__(self):
        return self.value


class Phenotype(Enum):
    """Possible metabolic operational states of a cell."""

    G_SPECIALIST = "G_SPECIALIST"  # Optimized for Glucose
    L_SPECIALIST = "L_SPECIALIST"  # Optimized for Galactose
    SWITCHING_GL = (
        "SWITCHING_GL"  # Transitioning from Glucose to Galactose metabolism (in lag)
    )
    # NOTE: The conceptual model mentions "SWITCHING_LG" (L->G) is instantaneous,
    # so it doesn't need a separate phenotype state for lag.

    def __str__(self):
        return self.value
