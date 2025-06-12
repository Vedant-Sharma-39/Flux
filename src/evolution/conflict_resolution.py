# src/evolution/conflict_resolution.py

import random
from typing import List, Tuple, Dict, Optional
import numpy as np

from ..core.shared_types import (
    HexCoord,
    ConflictResolutionRule,
)  # Assuming these are in shared_types
from ..core.cell import Cell  # Assuming Cell is defined


def resolve_reproduction_conflicts(
    targeted_sites: Dict[
        HexCoord, List[Tuple[Cell, HexCoord, float]]
    ],  # target_coord -> [(parent, parent_coord, parent_growth_rate), ...]
    active_rule: ConflictResolutionRule,
) -> List[
    Tuple[Cell, HexCoord, HexCoord]
]:  # List of (successful_parent, parent_coord, daughter_coord)
    """
    Resolves conflicts when multiple parent cells target the same empty site for reproduction.
    Returns a list of successful reproduction events.
    """
    actual_new_born_events = []

    for target_coord, contenders in targeted_sites.items():
        if not contenders:
            continue

        winner_info: Optional[Tuple[Cell, HexCoord]] = (
            None  # (winning_parent, winning_parent_coord)
        )

        if len(contenders) == 1:  # No conflict
            parent_cell, parent_coord, _ = contenders[0]
            winner_info = (parent_cell, parent_coord)
        else:  # Conflict! Apply rule
            if active_rule == ConflictResolutionRule.RANDOM_CHOICE:
                chosen_parent, chosen_parent_coord, _ = random.choice(contenders)
                winner_info = (chosen_parent, chosen_parent_coord)

            elif active_rule == ConflictResolutionRule.FITNESS_BIASED_LOTTERY:
                parents_for_lottery = [c[0] for c in contenders]
                parent_coords_for_lottery = [c[1] for c in contenders]
                growth_rates = np.array([c[2] for c in contenders], dtype=float)

                if (
                    np.sum(growth_rates) <= 1e-9
                ):  # All contenders have effectively zero growth rate
                    # Fallback to random choice if all rates are zero to avoid division by zero
                    winner_idx = random.randrange(len(contenders))
                else:
                    probabilities = growth_rates / np.sum(growth_rates)
                    # Ensure probabilities sum to 1 due to potential floating point issues
                    probabilities /= np.sum(probabilities)
                    winner_idx = np.random.choice(len(contenders), p=probabilities)

                winner_info = (
                    parents_for_lottery[winner_idx],
                    parent_coords_for_lottery[winner_idx],
                )

            # Add other rules here with elif:
            # elif active_rule == ConflictResolutionRule.FIRST_COME_FIRST_SERVED:
            #     # Note: 'contenders' list would need to be ordered by processing time for this to be meaningful
            #     parent_cell, parent_coord, _ = contenders[0] # Simplistic: assumes first in list wins
            #     winner_info = (parent_cell, parent_coord)

            else:
                raise NotImplementedError(
                    f"Conflict resolution rule {active_rule} not implemented."
                )

        if winner_info:
            actual_new_born_events.append(
                (winner_info[0], winner_info[1], target_coord)
            )

    return actual_new_born_events
