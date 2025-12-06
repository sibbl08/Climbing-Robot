import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path


from ik import ik          # inverse kinematics functions
from fk import fk          # forward kinematics function


def main():
    # -------------------------------------------------------------
    # 1. Load MuJoCo model
    # -------------------------------------------------------------
    xml_path = (Path(__file__).parent.parent / "Robot" / "Robot_V3.3.xml").resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep
    model.opt.gravity[:] = 0.0  # Disable gravity for testing


    # Set robot base pose in the world
    data.qpos[0:3] = [0, -0.3, 0.3]
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([0.0, 0.0, np.deg2rad(180)]), 'xyz')
    data.qpos[3:7] = quat
    mujoco.mj_forward(model, data)

    # -------------------------------------------------------------
    # 2. Extract chest pose in world (needed for FK/IK)
    # -------------------------------------------------------------
    T_chest_world = np.eye(4)
    T_chest_world[:3, 3]  = data.xpos[model.body('chest').id]
    T_chest_world[:3, :3] = data.xmat[model.body('chest').id].reshape(3, 3)

    # -------------------------------------------------------------
    # 3. Define IK target
    # -------------------------------------------------------------
    target_pos   = np.array([0.13, -0.27, 0.09])
    target_ori_y = np.deg2rad(0)

    # Initial guess for IK
    current_q = np.zeros(4)

    # -------------------------------------------------------------
    # 4. Compute IK
    # -------------------------------------------------------------
    q_ik, success = ik("right_leg", target_pos, target_ori_y, current_q, T_chest_world)
    print("IK joint angles:", q_ik)

    # -------------------------------------------------------------
    # 5. Apply IK result to MuJoCo model
    # -------------------------------------------------------------
    joint_names = [
        "servo_hip_yaw_r_Revolute_hip_yaw_r",
        "servo_hip_pitch_r_Revolute_hip_pitch_r",
        "servo_knee_pitch_r_Revolute_knee_pitch_r",
        "ankle_link_r_Revolute_ankle_yaw_r"
    ]
    actuator_names = [
        "hip_yaw_r",
        "hip_pitch_r",
        "knee_pitch_r",
        "ankle_yaw_r"
    ]

    for i in range(4):
        # Set joint position
        j_id = model.joint(joint_names[i]).qposadr
        data.qpos[j_id] = q_ik[i]

        # Set actuator control (important for MuJoCo)
        a_id = model.actuator(actuator_names[i]).id
        data.ctrl[a_id] = q_ik[i]

    mujoco.mj_forward(model, data)

    # -------------------------------------------------------------
    # 6. Visual marker for target
    # -------------------------------------------------------------
    target_id = model.body('target_marker').mocapid
    data.mocap_pos[target_id] = target_pos

    # -------------------------------------------------------------
    # 7. Compute FK in Python for comparison
    # -------------------------------------------------------------
    q_full = np.zeros(16)
    q_full[12:16] = q_ik      # right leg joint block

    fk_result = fk(q_full, T_chest_world)
    fk_pos    = fk_result["foot_r"][:3, 3]

    # -------------------------------------------------------------
    # 8. MuJoCo end-effector position
    # -------------------------------------------------------------
    body_id = model.body("foot_r_ref").id
    sim_pos = data.xpos[body_id]

    # -------------------------------------------------------------
    # 9. Print comparison
    # -------------------------------------------------------------
    print("\n---- Foot Position Comparison ----")
    print("Target position:   ", target_pos)
    print("FK result:         ", fk_pos)
    print("MuJoCo position:   ", sim_pos)
    print("FK error:          ", np.linalg.norm(fk_pos - target_pos))
    print("MuJoCo error:      ", np.linalg.norm(sim_pos - target_pos))
    print("IK success:", success)

    # -------------------------------------------------------------
    # 10. Optional viewer
    # -------------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, dt - (time.time() - start)))


if __name__ == "__main__":
    main()
