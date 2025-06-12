# src/evolution/phenotype_switching.py
import random
from ..core.cell import Cell, Phenotype
from ..environment.environment_rules import EnvironmentRules


def attempt_phenotype_switch(
    cell: Cell, environment_rules: EnvironmentRules
):  # dt removed, rates are now probs
    """
    Stochastically attempts to switch the phenotype of a cell based on probabilities per step.
    Modifies cell.phenotype in place if a switch occurs.
    Returns True if a switch occurred, False otherwise.
    """
    # Assuming k_GP and k_PG in sim_params are already probabilities per time step (k*dt)
    prob_G_to_P, prob_P_to_G = environment_rules.get_phenotype_switching_probabilities()

    switched = False
    if cell.phenotype == Phenotype.G_UNPREPARED:
        if random.random() < prob_G_to_P:
            cell.phenotype = Phenotype.P_PREPARED
            # When switching to P, G-type N2 adaptation state is no longer relevant
            cell.is_adapting_to_N2 = False
            cell.lag_N2_remaining = 0
            switched = True
    elif cell.phenotype == Phenotype.P_PREPARED:
        if random.random() < prob_P_to_G:
            cell.phenotype = Phenotype.G_UNPREPARED
            # G-type adaptation state will be set if it encounters N2
            switched = True
    return switched
