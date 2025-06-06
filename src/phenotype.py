from enum import Enum

class Phenotype(Enum):
    """Phenotypes of cells."""

    LOW_LAG = "low_lag"
    HIGH_LAG = "high_lag" 


def get_inherent_lag(phenotype: Phenotype) -> float:
    """Get the lag time of a cell based on its phenotype."""
    if phenotype == Phenotype.LOW_LAG:
        return 0.5  # Short lag time for low lag phenotype
    elif phenotype == Phenotype.HIGH_LAG:
        return 1.0  # Longer lag time for high growth phenotype