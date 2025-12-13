from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from holds import Hold, load_holds_from_xml, split_start_and_route_holds, sort_holds_by_height

# Les quatre membres du robot
LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]

from dataclasses import dataclass

@dataclass
class LimbState:
    limb: str     # exemple: "left_hand"
    hold: Hold    # la prise associée

@dataclass
class RobotState:
    left_hand: Hold
    right_hand: Hold
    left_foot: Hold
    right_foot: Hold

def initialize_robot_state(start_holds: List[Hold]) -> RobotState:
    """
    Cherche les prises de départ et initialise la position du robot.
    """
    lh = next(h for h in start_holds if "hand_l" in h.name)
    rh = next(h for h in start_holds if "hand_r" in h.name)
    lf = next(h for h in start_holds if "foot_l" in h.name)
    rf = next(h for h in start_holds if "foot_r" in h.name)

    return RobotState(
        left_hand=lh,
        right_hand=rh,
        left_foot=lf,
        right_foot=rf
    )

def find_next_hold_above(current_pos: np.ndarray, route_holds: List[Hold], dz_min=0.05) -> Optional[Hold]:
    """
    Trouve la prochaine prise en hauteur, au-dessus de current_pos.
    dz_min est la distance verticale minimale pour considérer une prise comme 'au-dessus'.
    """
    # on trie les prises par hauteur (z)
    sorted_holds = sort_holds_by_height(route_holds)

    for h in sorted_holds:
        if h.pos[2] > current_pos[2] + dz_min:
            return h

    return None  # aucune prise plus haute

MOVE_ORDER = [
    "right_hand",
    "left_hand",
    "right_foot",
    "left_foot"
]

def generate_climbing_plan(robot_state: RobotState, route_holds: List[Hold], max_steps=20):
    """
    Génère un plan de montée : une liste de (limb, old_hold, new_hold).
    """

    plan = []
    last_limb = None   # A) éviter deux fois le même limb

    for step in range(max_steps):

        limb = MOVE_ORDER[step % len(MOVE_ORDER)]
        current_hold = getattr(robot_state, limb)

        # A) éviter de bouger deux fois le même limb
        if limb == last_limb:
            continue

        # trouver la prochaine prise
        next_hold = find_next_hold_above(current_hold.pos, route_holds)

        if next_hold is None:
            print("No more holds above. Climbing plan complete.")
            break

        # B) éviter les prises trop proches (< 5 cm)
        dist = np.linalg.norm(next_hold.pos - current_hold.pos)
        if dist < 0.05:
            continue

        # C) vérifier que la prise est au-dessus
        if next_hold.pos[2] <= current_hold.pos[2]:
            continue

        # ajouter le mouvement au plan
        plan.append((limb, current_hold, next_hold))

        # mettre l’état du robot à jour
        setattr(robot_state, limb, next_hold)

        # mettre à jour le dernier limb bougé
        last_limb = limb

    return plan


def main():
    # Charger toutes les prises
    holds = load_holds_from_xml()
    start_holds, route_holds = split_start_and_route_holds(holds)

    # Initialiser l'état du robot
    robot = initialize_robot_state(start_holds)

    # Générer le plan de montée
    plan = generate_climbing_plan(robot, route_holds)

    print("\nGenerated climbing plan:")
    for limb, old, new in plan:
        print(f"{limb:10s} : {old.name:15s} -> {new.name}")

if __name__ == "__main__":
    main()
