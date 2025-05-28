# microbial_colony_sim/src/agents/cell.py
from dataclasses import dataclass, field
from typing import Optional

from src.core.enums import Phenotype, Nutrient
from src.core.data_structures import (
    HexCoord,
    TradeOffParams,
    SimulationConfig,
)  # Added SimulationConfig
from src.core.exceptions import CellError


@dataclass
class Cell:
    """Represents a single microbial cell."""

    _next_id_counter: int = field(
        default=0,
        repr=False,
        init=False,
        compare=False,  # Class variable for ID generation
        # This should be managed/reset by the simulation context
        # For simplicity, we'll add a static method to reset it.
        # A better approach might be a dedicated ID generator service passed around.
    )

    # Identity and Location
    id: int = field(
        init=False
    )  # Will be set by factory or __post_init__ if counter is instance-based
    coord: HexCoord  # q, r hexagonal coordinates

    # Core Inherited Traits (Position on Trade-off Curve)
    inherent_growth_rate_G: float  # Max potential growth rate on Glucose (if G_SPECIALIST, on G, not lagging)
    inherent_T_lag_GL: (
        float  # Base lag time if this cell (as G_SPECIALIST) encounters Galactose
    )

    # Current State
    current_phenotype: Phenotype
    remaining_lag_time: float = 0.0  # Time left in current lag phase

    # Calculated Operational Rate (dependent on phenotype, nutrient, lag)
    current_growth_attempt_rate: float = 0.0

    # Parent ID for lineage tracking (optional)
    parent_id: Optional[int] = None

    # Simulation time of birth (optional, for analysis)
    birth_time: Optional[float] = None

    # Static variable to ensure unique IDs per simulation run
    # This should be reset at the beginning of each simulation.
    _static_next_id: int = 0

    @staticmethod
    def reset_id_counter(start_id: int = 0) -> None:
        Cell._static_next_id = start_id

    def __post_init__(self):
        # Assign ID
        self.id = Cell._static_next_id
        Cell._static_next_id += 1

        if (
            self.remaining_lag_time < 0
        ):  # Use a small epsilon for float comparison if needed
            self.remaining_lag_time = (
                0.0  # Clamp instead of raising error immediately for robustness
            )
            # raise CellError(f"Cell {self.id}: remaining_lag_time cannot be negative.")
        if self.inherent_growth_rate_G < 0:
            raise CellError(
                f"Cell {self.id}: inherent_growth_rate_G cannot be negative."
            )
        if self.inherent_T_lag_GL < 0:
            raise CellError(f"Cell {self.id}: inherent_T_lag_GL cannot be negative.")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return NotImplemented
        return self.id == other.id

    def __repr__(self):
        return (
            f"Cell(id={self.id}, coord={self.coord}, phenotype={self.current_phenotype.name}, "
            f"g_rate_G={self.inherent_growth_rate_G:.2f}, T_lag_GL={self.inherent_T_lag_GL:.2f}, "
            f"lag_rem={self.remaining_lag_time:.2f}, growth_attempt={self.current_growth_attempt_rate:.2f})"
        )

    def is_lagging(self) -> bool:
        """Checks if the cell is currently in a lag phase due to phenotype switching."""
        # A cell is lagging if its phenotype is SWITCHING_GL AND it has remaining lag time.
        # Or simply if remaining_lag_time is positive, assuming it's only set for SWITCHING_GL.
        return (
            self.remaining_lag_time > 1e-9
        )  # Use a small epsilon for float comparison

    def update_growth_attempt_rate(
        self, local_nutrient: Nutrient, config: SimulationConfig
    ):
        """
        Updates current_growth_attempt_rate based on current phenotype, local_nutrient,
        and inherent rates from config. Assumes lag has already been processed for the current step.
        This should be called after phenotype switching and lag decrement.
        """
        if self.is_lagging() or self.current_phenotype == Phenotype.SWITCHING_GL:
            # If phenotype is SWITCHING_GL, it means it's in lag phase.
            self.current_growth_attempt_rate = 0.0
            return

        if self.current_phenotype == Phenotype.G_SPECIALIST:
            if local_nutrient == Nutrient.GLUCOSE:
                self.current_growth_attempt_rate = self.inherent_growth_rate_G
            else:  # G_SPECIALIST on Galactose (should be initiating switch, so no growth this step)
                self.current_growth_attempt_rate = 0.0
        elif self.current_phenotype == Phenotype.L_SPECIALIST:
            if local_nutrient == Nutrient.GALACTOSE:
                self.current_growth_attempt_rate = (
                    config.lambda_L_fixed_rate
                )  # Use from config
            else:  # L_SPECIALIST on Glucose (should have switched instantaneously, so no growth as L_SPECIALIST)
                self.current_growth_attempt_rate = 0.0
        # else: Phenotype.SWITCHING_GL already handled by is_lagging() check
