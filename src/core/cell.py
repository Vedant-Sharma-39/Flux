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
    phenotype: Phenotype = Phenotype.G_UNPREPARED
    lineage_id: uuid.UUID = field(
        default_factory=uuid.uuid4
    )  # New field, defaults to a new UUID
    time_to_next_division: float = field(init=False, default=float("inf"))

    # Attributes for managing lag for G-type cells if they can adapt to N2
    is_adapting_to_N2: bool = field(init=False, default=False)
    lag_N2_remaining: float = field(init=False, default=0.0)

    # __post_init__ is not strictly needed for lineage_id if default_factory works as intended
    # for initial cell creation. If a cell is created by copying, lineage_id might need specific handling.
    # However, our main cell creation path is via direct instantiation or from parent.

    def reset_division_timer(self, growth_rate: float, time_step_duration: float):
        if growth_rate > 0:
            self.time_to_next_division = 1.0 / (growth_rate * time_step_duration)
        else:
            self.time_to_next_division = float("inf")

    def initiate_N2_adaptation_lag(
        self, characteristic_lag_G_N2: float, time_step_duration: float
    ):
        if self.phenotype == Phenotype.G_UNPREPARED:
            self.is_adapting_to_N2 = True
            self.lag_N2_remaining = characteristic_lag_G_N2 / time_step_duration
            if self.lag_N2_remaining <= 0:
                self.is_adapting_to_N2 = False

    def decrement_N2_lag(self):
        if self.is_adapting_to_N2 and self.phenotype == Phenotype.G_UNPREPARED:
            self.lag_N2_remaining -= 1
            if self.lag_N2_remaining <= 0:
                self.is_adapting_to_N2 = False
                self.lag_N2_remaining = 0
                return True
        return False
