from holds import load_holds_from_xml, split_start_and_route_holds
from path_planner import initialize_robot_state, generate_climbing_plan
from ik import ik
import numpy as np
from execute_motion import create_motion_step

def generate_motion_steps():
    """
    Full pipeline:
    holds → planner → IK → Motion Steps
    Returns a list of MotionStep objects.
    """

  
    holds = load_holds_from_xml()
    start_holds, route_holds = split_start_and_route_holds(holds)

    
    robot_state = initialize_robot_state(start_holds)

    plan = generate_climbing_plan(robot_state, route_holds)

    motion_steps = []

    
    mapping = {
        "right_hand": "right_arm",
        "left_hand": "left_arm",
        "right_foot": "right_leg",
        "left_foot": "left_leg",
    }

    for (limb, old_hold, new_hold) in plan:

        target_pos = new_hold.pos

        if limb not in mapping:
            continue

        ik_limb = mapping[limb]

        
        target_ori_y = 0.0
        q0 = np.zeros(4)
        T = np.eye(4)

       
        angles, success = ik(ik_limb, target_pos, target_ori_y, q0, T)

        ik_result = {
            "limb": limb,
            "ik_limb": ik_limb,
            "target": target_pos,
            "joint_angles": angles,
            "status": "success" if success else "failed",
        }

       
        motion_step = create_motion_step(ik_result)

        if motion_step is None:
            continue

        motion_steps.append(motion_step)

    return motion_steps


import json

def save_motion_steps(steps, filename="motion_steps.json"):
    data = [step.__dict__ for step in steps]
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
