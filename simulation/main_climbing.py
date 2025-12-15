from holds import load_holds_from_xml, split_start_and_route_holds
from path_planner import initialize_robot_state, generate_climbing_plan
import ik 
from ik import ik
import numpy as np

def main():
    
    holds = load_holds_from_xml()
    start_holds, route_holds = split_start_and_route_holds(holds)

    print("[INFO] Loaded holds:", len(holds))

   
    robot = initialize_robot_state(start_holds)

    print("[INFO] Robot initialized on start holds.")

    plan = generate_climbing_plan(robot, route_holds)

    print(f"[INFO] Climbing plan generated with {len(plan)} steps.")


    mapping = {
    "right_hand": "right_arm",
    "left_hand": "left_arm",
    "right_foot": "right_leg",
    "left_foot": "left_leg",
    }

    
    for (limb, old_hold, new_hold) in plan:
        print(f"\n[STEP] Moving {limb} from {old_hold.name} → {new_hold.name}")
        
        
        target_pos = new_hold.pos
        
       
        print("[IK] Computing IK for", limb)

        if limb not in mapping:
            print(f"[IK] Unknown limb {limb}, skipping")
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
            "status": "success" if success else "failed"
        }

        print("[IK] Result:", ik_result)


        from execute_motion import create_motion_step

        motion_step = create_motion_step(ik_result)

        if motion_step is None:
            print("[MOTION] IK failed, no motion step generated.")
        else:
            print("[MOTION] Generated motion step:", motion_step)


    print("\n[INFO] Climbing simulation completed!")


if __name__ == "__main__":
    main()
