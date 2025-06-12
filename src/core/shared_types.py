# src/core/shared_types.py

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional  # Added Optional


# --- Coordinate System ---
@dataclass(frozen=True, eq=True)
class HexCoord:
    q: int
    r: int

    def __add__(self, other: "HexCoord") -> "HexCoord":
        if not isinstance(other, HexCoord):
            return NotImplemented
        return HexCoord(self.q + other.q, self.r + other.r)

    def __sub__(self, other: "HexCoord") -> "HexCoord":
        if not isinstance(other, HexCoord):
            return NotImplemented
        return HexCoord(self.q - other.q, self.r - other.r)

    def __mul__(self, scalar: int) -> "HexCoord":
        if not isinstance(scalar, int):
            return NotImplemented
        return HexCoord(self.q * scalar, self.r * scalar)


# --- Cell Phenotypes ---
class Phenotype(Enum):
    G_UNPREPARED = (
        1  # Growth-optimized for N1, intrinsically poor/slow adaptation to N2
    )
    P_PREPARED = 2  # Slower on N1 (costly), intrinsically good/fast adaptation to N2


# --- Nutrient Types ---
class Nutrient(Enum):
    N1_PREFERRED = 1
    N2_CHALLENGING = 2
    NONE = 0


# --- Conflict Resolution Rules ---
class ConflictResolutionRule(Enum):
    RANDOM_CHOICE = 1
    FITNESS_BIASED_LOTTERY = 2
    # FIRST_COME_FIRST_SERVED = 3 # Commented out if not implemented


# --- Simulation Parameters ---
@dataclass
class SimulationParameters:
    hex_size: float
    time_step_duration: float

    lambda_G_N1: float
    alpha_G_N2: float
    lag_G_N2: float  # Non-dimensional (in units of T_ref, or steps if dt_eff = 1)
    lambda_G_N2_adapted: float

    cost_delta_P: float  # cost_delta_P for P_PREPARED on N1
    alpha_P_N2: float
    lag_P_N2: float  # Non-dimensional
    lambda_P_N2: float

    k_GP: float  # G to P switching probability per time step
    k_PG: float  # P to G switching probability per time step
    initial_colony_radius: int
    initial_phenotype_G_fraction: (
        float  # Fraction of initial cells that are G_UNPREPARED
    )
    active_conflict_rule: ConflictResolutionRule
    nutrient_bands: List[Tuple[float, Nutrient]] = field(default_factory=list)

