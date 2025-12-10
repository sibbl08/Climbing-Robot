from holds import load_holds_from_xml, split_start_and_route_holds
from path_planner import initialize_robot_state, generate_climbing_plan
from ik import compute_ik
from execute_motion import create_motion_step

def generate_motion_steps():
    """
    Full pipeline:
    holds → planner → IK → Motion Steps
    Returns a list of MotionStep objects.
    """

    # 1️⃣ Load holds
    holds = load_holds_from_xml()
    start_holds, route_holds = split_start_and_route_holds(holds)

    # 2️⃣ Initialize robot on start holds
    robot_state = initialize_robot_state(start_holds)

    # 3️⃣ Generate the climbing plan
    plan = generate_climbing_plan(robot_state, route_holds)

    motion_steps = []

    # 4️⃣ For each step in the plan, compute IK and generate motion step
    for (limb, old_hold, new_hold) in plan:

        target_pos = new_hold.pos

        ik_result = compute_ik(limb, target_pos)

        # convert IK → motion step
        motion_step = create_motion_step(ik_result)

        if motion_step is None:
            print(f"[WARN] IK failed for {limb}, skipping movement.")
            continue

        motion_steps.append(motion_step)

    return motion_steps

import json

def save_motion_steps(steps, filename="motion_steps.json"):
    data = [step.__dict__ for step in steps]
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
