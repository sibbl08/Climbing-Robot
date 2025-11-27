import mujoco
import numpy as np

def initialize_robot_pose(model, data):
    # --- Initial joint angles ---
    initial_angles = {
        "servo_elbow_pitch_l_Revolute_elbow_pitch_l": 0.0,
        "servo_wrist_pitch_l_Revolute_wrist_pitch_l": 0.0,
        "servo_shoulder_pitch_l_Revolute_shoulder_pitch_l": 0.0,
        "servo_shoulder_roll_l_Revolute_shoulder_roll_l": 0.0,

        "servo_elbow_pitch_r_Revolute_elbow_pitch_r": 0.0,
        "servo_wrist_pitch_r_Revolute_wrist_pitch_r": 0.0,
        "servo_shoulder_pitch_r_Revolute_shoulder_pitch_r": 0.0,
        "servo_shoulder_roll_r_Revolute_shoulder_roll_r": 0.0,

        "servo_hip_roll_l_Revolute_hip_roll_l": -1.5708,
        "servo_hip_pitch_l_Revolute_hip_pitch_l": -0.785398,
        "servo_knee_pitch_l_Revolute_knee_pitch_l": -0.785398,
        "ankle_joint_l_Revolute_ankle_roll_l": 0.436332,

        "servo_hip_roll_r_Revolute_hip_roll_r": -1.5708,
        "servo_hip_pitch_r_Revolute_hip_pitch_r": -0.785398,
        "servo_knee_pitch_r_Revolute_knee_pitch_r": 0.785398,
        "ankle_joint_r_Revolute_ankle_roll_r": -0.436332,
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
    data.qpos[0:3] = [0, -0.11, 0.03]
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([1.5708, 0.0, 0.0]), 'xyz')
    data.qpos[3:7] = quat

    # set controller to the same values as joint positions
    for i in range(model.nu):
        j_id = model.actuator_trnid[i][0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j_id))
        if joint_name in initial_angles:
            data.ctrl[i] = initial_angles[joint_name]

    # forward the model to apply changes
    mujoco.mj_forward(model, data)
