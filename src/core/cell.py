# src/core/cell.py

from dataclasses import dataclass, field
import uuid
from typing import Optional
from .shared_types import Phenotype  # Uses updated Phenotype Enum


@dataclass
class Cell:
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    parent_id: Optional[uuid.UUID] = None
    generation: int = 0
    birth_time: float = 0.0
    phenotype: Phenotype = Phenotype.G_UNPREPARED  # Default to G_UNPREPARED
    time_to_next_division: float = field(init=False, default=float("inf"))

    # Attributes for managing lag for G-type cells if they can adapt to N2
    is_adapting_to_N2: bool = field(init=False, default=False)  # G-type specific
    lag_N2_remaining: float = field(init=False, default=0.0)  # G-type specific

    def __post_init__(self):
        pass  # Initial timer set by simulation logic

    def reset_division_timer(self, growth_rate: float, time_step_duration: float):
        if growth_rate > 0:
            # Effective number of simulation steps for one division event on average
            self.time_to_next_division = 1.0 / (growth_rate * time_step_duration)
            # For stochasticity:
            # import numpy as np
            # self.time_to_next_division = np.random.exponential(1.0 / (growth_rate * time_step_duration))
        else:
            self.time_to_next_division = float("inf")

    def initiate_N2_adaptation_lag(
        self, characteristic_lag_G_N2: float, time_step_duration: float
    ):
        """Initiates the lag phase for a G-type cell encountering N2."""
        if self.phenotype == Phenotype.G_UNPREPARED:
            self.is_adapting_to_N2 = True
            # Convert non-dimensional lag (in T_ref units) to simulation steps
            # If characteristic_lag_G_N2 is already in simulation steps, no need to divide by dt
            self.lag_N2_remaining = (
                characteristic_lag_G_N2 / time_step_duration
            )  # if lag_G_N2 is in units of time
            # Or if characteristic_lag_G_N2 is already in steps: self.lag_N2_remaining = characteristic_lag_G_N2
            if self.lag_N2_remaining <= 0:  # If lag is zero, consider it adapted
                self.is_adapting_to_N2 = False
                # Its growth rate and division timer will be set based on lambda_G_N2_adapted

    def decrement_N2_lag(self):
        """Decrements the N2 adaptation lag timer. Returns True if lag just finished."""
        if self.is_adapting_to_N2 and self.phenotype == Phenotype.G_UNPREPARED:
            self.lag_N2_remaining -= 1  # Assumes timer is in steps
            if self.lag_N2_remaining <= 0:
                self.is_adapting_to_N2 = False
                self.lag_N2_remaining = 0
                return True  # Lag finished
        return False  # Still lagging or not applicable
