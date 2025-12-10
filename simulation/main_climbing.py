from holds import load_holds_from_xml, split_start_and_route_holds
from path_planner import initialize_robot_state, generate_climbing_plan
import ik  # we will call IK functions from here


def main():
    # 1️⃣ Charger les holds
    holds = load_holds_from_xml()
    start_holds, route_holds = split_start_and_route_holds(holds)

    print("[INFO] Loaded holds:", len(holds))

    # 2️⃣ Initialiser le robot
    robot = initialize_robot_state(start_holds)

    print("[INFO] Robot initialized on start holds.")

    # 3️⃣ Générer un plan d'escalade
    plan = generate_climbing_plan(robot, route_holds)

    print(f"[INFO] Climbing plan generated with {len(plan)} steps.")

    # 4️⃣ Exécuter le plan avec IK (simulation plus tard)
    for (limb, old_hold, new_hold) in plan:
        print(f"\n[STEP] Moving {limb} from {old_hold.name} → {new_hold.name}")
        
        # POSITION de la prise cible
        target_pos = new_hold.pos
        
        # 4️⃣ Appel IK pour obtenir les angles
        print("[IK] Computing IK for", limb)
        ik_result = ik.compute_ik(limb, target_pos)

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
