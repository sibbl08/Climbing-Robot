import numpy as np
from fk import fk


# ======================================================================
#  Inverse Kinematics (IK) dispatcher
# ======================================================================
def ik(limb, target_pos, target_ori_y, q0, T_chest_world):
    """
    Dispatch function selecting the correct IK routine for each limb.

    Parameters
    ----------
    limb : str
        One of: "right_arm", "left_arm", "right_leg", "left_leg".

    target_pos : ndarray (3,)
        Desired end-effector position in world coordinates.

    target_ori_y : float
        Desired end-effector orientation value (summed pitch for arms).
        Legs ignore this parameter.

    q0 : ndarray (4,)
        Initial joint guess for the IK iteration.

    T_chest_world : ndarray (4x4)
        Chest pose in world coordinates (base for FK).

    Returns
    -------
    q : ndarray (4,)
        Joint configuration found by IK.

    success : bool
        True if IK converged within required tolerances.
    """
    if limb == "right_arm":
        return ik_right_arm(target_pos, target_ori_y, q0, T_chest_world)

    if limb == "left_arm":
        return ik_left_arm(target_pos, target_ori_y, q0, T_chest_world)

    if limb == "right_leg":
        return ik_right_leg(target_pos, q0, T_chest_world)

    if limb == "left_leg":
        return ik_left_leg(target_pos, q0, T_chest_world)

    raise ValueError(f"Unknown limb: {limb}")


# ======================================================================
#  Right Arm IK  
# ======================================================================
def ik_right_arm(target_pos, target_ori_y, q0, T_chest_world):
    """
    Inverse kinematics for the right arm.
    Primary objective: reach desired 3-D position.
    Secondary objective: approximate orientation using (pitch+elbow+wrist).

    IK returns success only if:
        - position error < tol_pos
        - orientation error < ±10 degrees

    Orientation refinement is applied in the Jacobian nullspace.
    """

    max_iter = 80
    tol_pos = 1e-3                 # position tolerance (meters)
    tol_ori = np.deg2rad(10)       # orientation tolerance (radians)
    eps = 5e-6                     # finite difference step
    lam = 5e-4                     # Damped least squares regularizer

    # joint limits
    q_min = np.deg2rad(np.array([-90, -60,   0, -130]))
    q_max = np.deg2rad(np.array([ 90, 120, 132,   35]))

    q = q0.copy()
    prev_err = 1e9

    for _ in range(max_iter):

        # ---- Forward kinematics ----
        q_full = np.zeros(16)
        q_full[0:4] = q
        T = fk(q_full, T_chest_world)["hand_r"]
        pos = T[:3, 3]

        # orientation = sum of pitch joints
        ori_current = q[1] + q[2] + q[3]

        # errors
        e_pos = target_pos - pos
        err_pos = np.linalg.norm(e_pos)
        e_ori = ori_current - target_ori_y

        # ---- success criteria ----
        if err_pos < tol_pos and abs(e_ori) < tol_ori:
            return q, True

        # stagnation -> no convergence
        if abs(prev_err - err_pos) < 1e-7:
            return q, False
        prev_err = err_pos

        # ---- Numerical position Jacobian (3x4) ----
        J = np.zeros((3, 4))
        for i in range(4):
            dq = q.copy()
            dq[i] += eps
            q_full_eps = np.zeros(16)
            q_full_eps[0:4] = dq
            pos_eps = fk(q_full_eps, T_chest_world)["hand_r"][:3, 3]
            J[:, i] = (pos_eps - pos) / eps

        # ---- DLS inversion ----
        J_pinv = np.linalg.inv(J.T @ J + lam * np.eye(4)) @ J.T

        # primary task
        dq_pos = J_pinv @ e_pos

        # nullspace projector
        N = np.eye(4) - J_pinv @ J

        # dynamic orientation weight
        if err_pos < 0.003:
            alpha = 3.0
        elif err_pos < 0.01:
            alpha = 1.0
        else:
            alpha = 0.0

        # orientation gradient
        grad = np.array([0.0, -e_ori, -e_ori, -e_ori])

        dq_ori = alpha * (N @ grad)

        # update
        q += dq_pos + dq_ori
        q = np.clip(q, q_min, q_max)

    return q, False


# ======================================================================
#  Left Arm IK  
# ======================================================================
def ik_left_arm(target_pos, target_ori_y, q0, T_chest_world):
    """
    Inverse kinematics for the left arm.
    Same logic as right arm:
        - position is the primary task
        - orientation refined in nullspace
        - success only if both pos/orientation tolerances are met
    """

    max_iter = 80
    tol_pos = 1e-3              # position tolerance
    tol_ori = np.deg2rad(10)    # orientation tolerance
    eps = 5e-6                  # finite difference step
    lam = 5e-4                  # DLS regularizer

    q_min = np.deg2rad(np.array([-90, -60,   0, -130]))
    q_max = np.deg2rad(np.array([ 90, 120, 132,   35]))

    q = q0.copy()
    prev_err = 1e9

    for _ in range(max_iter):

        # FK
        q_full = np.zeros(16)
        q_full[4:8] = q
        T = fk(q_full, T_chest_world)["hand_l"]
        pos = T[:3, 3]

        ori_current = q[1] + q[2] + q[3]

        e_pos = target_pos - pos
        err_pos = np.linalg.norm(e_pos)
        e_ori = ori_current - target_ori_y

        if err_pos < tol_pos and abs(e_ori) < tol_ori:
            return q, True

        if abs(prev_err - err_pos) < 1e-7:
            return q, False
        prev_err = err_pos

        # Jacobian
        J = np.zeros((3, 4))
        for i in range(4):
            dq = q.copy()
            dq[i] += eps
            q_full_eps = np.zeros(16)
            q_full_eps[4:8] = dq
            pos_eps = fk(q_full_eps, T_chest_world)["hand_l"][:3, 3]
            J[:, i] = (pos_eps - pos) / eps

        J_pinv = np.linalg.inv(J.T @ J + lam * np.eye(4)) @ J.T
        dq_pos = J_pinv @ e_pos
        N = np.eye(4) - J_pinv @ J

        # dynamic orientation weight
        if err_pos < 0.003:
            alpha = 3.0
        elif err_pos < 0.01:
            alpha = 1.0
        else:
            alpha = 0.0

        grad = np.array([0.0, -e_ori, -e_ori, -e_ori])
        dq_ori = alpha * (N @ grad)

        q += dq_pos + dq_ori
        q = np.clip(q, q_min, q_max)

    return q, False


# ======================================================================
#  Right Leg IK
# ======================================================================
def ik_right_leg(target_pos, q0, T_chest_world):
    """
    Inverse kinematics for the right leg.
    Primary task: foot position.
    Two nullspace objectives:
        - keep foot level   (hpitch - kpitch ≈ 0)
        - keep hip yaw near -90°

    If q0 is near zero (unusable), a stable default configuration is used.
    """

    max_iter = 80
    tol = 2e-3         # position tolerance
    eps = 5e-6         # finite difference step
    lam = 5e-4         # DLS regularizer
    alpha_flat = 3.0   # foot leveling weight
    alpha_hyaw = 2.0   # hip yaw target weight

    q_min = np.deg2rad(np.array([-100, -20, 0, -90]))
    q_max = np.deg2rad(np.array([ -80, 120, 140, 90]))

    hyaw_target = np.deg2rad(-90)

    q0 = np.array(q0, dtype=float)

    # choose stable default if q0 does not contain meaningful pitch values
    if abs(q0[1]) < np.deg2rad(1.0) and abs(q0[2]) < np.deg2rad(1.0):
        q = np.deg2rad(np.array([-90, 20, 30, 0]))
    else:
        q = q0.copy()

    # reachability check
    hip_offset = np.array([-0.0434702141, 0.021, -0.0716202141, 1.0])
    hip_world = (T_chest_world @ hip_offset)[:3]
    if np.linalg.norm(target_pos - hip_world) > 0.32:
        return q, False

    prev_err = 1e9

    for _ in range(max_iter):

        q_full = np.zeros(16)
        q_full[12:16] = q
        T = fk(q_full, T_chest_world)["foot_r"]
        pos = T[:3, 3]

        e_pos = target_pos - pos
        err = np.linalg.norm(e_pos)

        if err < tol:
            return q, True

        if abs(prev_err - err) < 1e-7:
            return q, False

        prev_err = err

        # Jacobian
        J = np.zeros((3, 4))
        for i in range(4):
            dq = q.copy()
            dq[i] += eps
            q_full_eps = np.zeros(16)
            q_full_eps[12:16] = dq
            pos_eps = fk(q_full_eps, T_chest_world)["foot_r"][:3, 3]
            J[:, i] = (pos_eps - pos) / eps

        J_pinv = np.linalg.inv(J.T @ J + lam * np.eye(4)) @ J.T
        dq_pos = J_pinv @ e_pos

        N = np.eye(4) - J_pinv @ J

        # nullspace foot leveling
        e_flat = q[1] - q[2]
        grad_flat = np.array([0.0, -e_flat, e_flat, 0.0])
        dq_flat = alpha_flat * (N @ grad_flat)

        # nullspace hip yaw
        e_hyaw = q[0] - hyaw_target
        grad_hyaw = np.array([-e_hyaw, 0.0, 0.0, 0.0])
        dq_hyaw = alpha_hyaw * (N @ grad_hyaw)

        q += dq_pos + dq_flat + dq_hyaw
        q = np.clip(q, q_min, q_max)

    return q, False


# ======================================================================
#  Left Leg IK
# ======================================================================
def ik_left_leg(target_pos, q0, T_chest_world):
    """
    Inverse kinematics for the left leg.
    Same structure as the right leg, but mirrored FK and hip yaw sign.
    """

    max_iter = 80
    tol = 2e-3          # position tolerance
    eps = 5e-6          # finite difference step
    lam = 5e-4          # DLS regularizer
    alpha_flat = 3.0    # foot leveling weight
    alpha_hyaw = 2.0    # hip yaw target weight    

    q_min = np.deg2rad(np.array([-100, -20, 0, -90]))
    q_max = np.deg2rad(np.array([ -80, 120, 140, 90]))

    # mirrored hip yaw target (depending on robot convention)
    hyaw_target = np.deg2rad(-90)

    q0 = np.array(q0, dtype=float)

    # choose stable start
    if abs(q0[1]) < np.deg2rad(1.0) and abs(q0[2]) < np.deg2rad(1.0):
        q = np.deg2rad(np.array([-90, 20, 30, 0]))
    else:
        q = q0.copy()

    # reachability
    hip_offset = np.array([0.0434702141, 0.021, -0.0716202141, 1.0])
    hip_world = (T_chest_world @ hip_offset)[:3]
    if np.linalg.norm(target_pos - hip_world) > 0.32:
        return q, False

    prev_err = 1e9

    for _ in range(max_iter):

        q_full = np.zeros(16)
        q_full[8:12] = q
        T = fk(q_full, T_chest_world)["foot_l"]
        pos = T[:3, 3]

        e_pos = target_pos - pos
        err = np.linalg.norm(e_pos)

        if err < tol:
            return q, True

        if abs(prev_err - err) < 1e-7:
            return q, False

        prev_err = err

        # Jacobian
        J = np.zeros((3, 4))
        for i in range(4):
            dq = q.copy()
            dq[i] += eps
            q_full_eps = np.zeros(16)
            q_full_eps[8:12] = dq
            pos_eps = fk(q_full_eps, T_chest_world)["foot_l"][:3, 3]
            J[:, i] = (pos_eps - pos) / eps

        J_pinv = np.linalg.inv(J.T @ J + lam * np.eye(4)) @ J.T
        dq_pos = J_pinv @ e_pos
        N = np.eye(4) - J_pinv @ J

        # foot leveling
        e_flat = q[1] - q[2]
        grad_flat = np.array([0.0, -e_flat, e_flat, 0.0])
        dq_flat = alpha_flat * (N @ grad_flat)

        # hip yaw
        e_hyaw = q[0] - hyaw_target
        grad_hyaw = np.array([-e_hyaw, 0.0, 0.0, 0.0])
        dq_hyaw = alpha_hyaw * (N @ grad_hyaw)

        q += dq_pos + dq_flat + dq_hyaw
        q = np.clip(q, q_min, q_max)

    return q, False
