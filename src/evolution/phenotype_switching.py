# src/evolution/phenotype_switching.py
import random
import uuid  # Import uuid
from ..core.cell import Cell, Phenotype
from ..environment.environment_rules import EnvironmentRules


def attempt_phenotype_switch(cell: Cell, environment_rules: EnvironmentRules):
    prob_G_to_P, prob_P_to_G = environment_rules.get_phenotype_switching_probabilities()
    switched = False

    if cell.phenotype == Phenotype.G_UNPREPARED:
        if random.random() < prob_G_to_P:
            cell.phenotype = Phenotype.P_PREPARED
            cell.lineage_id = uuid.uuid4()  # New lineage on switch
            cell.is_adapting_to_N2 = False
            cell.lag_N2_remaining = 0
            switched = True
    elif cell.phenotype == Phenotype.P_PREPARED:
        if random.random() < prob_P_to_G:
            cell.phenotype = Phenotype.G_UNPREPARED
            cell.lineage_id = uuid.uuid4()  # New lineage on switch
            # G-type adaptation state will be (re)set if it encounters N2
            switched = True
    return switched
