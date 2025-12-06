import mujoco
import numpy as np

def initialize_robot_pose(model, data):
    # --- Initial joint angles ---
    initial_angles = {
        "servo_shoulder_yaw_r_Revolute_shoulder_yaw_r": np.deg2rad(5),
        "servo_shoulder_pitch_r_Revolute_shoulder_pitch_r": 0.0,
        "servo_elbow_pitch_r_Revolute_elbow_pitch_r": 0.0,
        "servo_wrist_pitch_r_Revolute_wrist_pitch_r": 0.0,

        "servo_shoulder_yaw_l_Revolute_shoulder_yaw_l": np.deg2rad(5),
        "servo_shoulder_pitch_l_Revolute_shoulder_pitch_l": 0.0,
        "servo_elbow_pitch_l_Revolute_elbow_pitch_l": 0.0,
        "servo_wrist_pitch_l_Revolute_wrist_pitch_l": 0.0,

        "servo_hip_yaw_l_Revolute_hip_yaw_l": np.deg2rad(-90),
        "servo_hip_pitch_l_Revolute_hip_pitch_l": np.deg2rad(15),
        "servo_knee_pitch_l_Revolute_knee_pitch_l": np.deg2rad(15),
        "ankle_link_l_Revolute_ankle_yaw_l": np.deg2rad(40),

        "servo_hip_yaw_r_Revolute_hip_yaw_r": np.deg2rad(-90),
        "servo_hip_pitch_r_Revolute_hip_pitch_r": np.deg2rad(15),
        "servo_knee_pitch_r_Revolute_knee_pitch_r": np.deg2rad(15),
        "ankle_link_r_Revolute_ankle_yaw_r": np.deg2rad(40),

    }

    # set joint positions
    for joint_name, angle in initial_angles.items():
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if j_id >= 0:
            qpos_adr = model.jnt_qposadr[j_id]
            data.qpos[qpos_adr] = angle
        else:
            print(f"WARNING: Joint {joint_name} not found")

    # set position and orientation of the robot base
    data.qpos[0:3] = [0, -0.03, 0.36]
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([0.0, 0.0, np.deg2rad(180)]), 'xyz')
    data.qpos[3:7] = quat

    # set contyawer to the same values as joint positions
    for i in range(model.nu):
        j_id = model.actuator_trnid[i][0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j_id))
        if joint_name in initial_angles:
            data.ctrl[i] = initial_angles[joint_name]

    # forward the model to apply changes
    mujoco.mj_forward(model, data)
