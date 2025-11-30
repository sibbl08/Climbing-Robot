import numpy as np

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s,  c, 0],
                     [0, 0,  0, 1]])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]])

def trans(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

def fk(q, T_chest_in_world):
    """
    q: list of joint angles
    T_chest_in_world: 4x4 Matrix (Pose of the chest in world coordinates).
    
    Output:
        dict with 4x4 transformationmatrixes of the endeffectors (world coordinates).
    """
    
    # Joints
    (q_syaw_r, q_spitch_r, q_epitch_r, q_wpitch_r,
     q_syaw_l, q_spitch_l, q_epitch_l, q_wpitch_l,
     q_hyaw_r, q_hpitch_r, q_kpitch_r, q_ayaw_r,
     q_hyaw_l, q_hpitch_l, q_kpitch_l, q_ayaw_l) = q

    # World Coordinates of Chest as starting point
    T = T_chest_in_world.copy()

    # ------------------- Right Arm -----------------------------------
    T_r = T.copy()
    T_r = T_r @ trans(-0.0586202141, 0.02099999999999997, 0.0330297868)
    T_r = T_r @ rot_x(q_syaw_r)
    T_r = T_r @ trans(-0.020, -0.02575, 0)
    T_r = T_r @ rot_y(q_spitch_r)
    T_r = T_r @ trans(-0.067, 0, 0)
    T_r = T_r @ rot_y(q_epitch_r)
    T_r = T_r @ trans(-0.067, 0, 0)
    T_r = T_r @ rot_y(q_wpitch_r)
    T_r = T_r @ trans(0, -0.011, 0.073)  # hand_r_ref final frame
    hand_r = T_r

    # ------------------- Left Arm -----------------------------------
    T_l = T.copy()
    T_l = T_l @ trans(0.0581797859, 0.021, 0.0330297868)
    T_l = T_l @ rot_x(q_syaw_l)
    T_l = T_l @ trans(0.020, -0.02575, 0)
    T_l = T_l @ rot_y(q_spitch_l)
    T_l = T_l @ trans(0.067, 0, 0)
    T_l = T_l @ rot_y(q_epitch_l)
    T_l = T_l @ trans(0.067, 0, 0)
    T_l = T_l @ rot_y(q_wpitch_l)
    T_l = T_l @ trans(0, -0.011, 0.073)
    hand_l = T_l

    # ------------------- Right Leg ---------------------------------
    T_br = T.copy()
    T_br = T_br @ trans(-0.0434702141, 0.021, -0.0716202141)
    T_br = T_br @ rot_z(q_hyaw_r)
    T_br = T_br @ trans(-0.02575, 0, -0.02)
    T_br = T_br @ rot_x(q_hpitch_r)
    T_br = T_br @ trans(0, 0, -0.067)
    T_br = T_br @ rot_x(q_kpitch_r)
    T_br = T_br @ trans(0, 0, 0)
    T_br = T_br @ rot_z(q_ayaw_r)
    T_br = T_br @ trans(0.018, -0.029, -0.004)
    foot_r = T_br

    # ------------------- Left Leg ----------------------------------
    T_bl = T.copy()
    T_bl = T_bl @ trans(0.0430297859, 0.021, -0.0716202141)
    T_bl = T_bl @ rot_z(q_hyaw_l)
    T_bl = T_bl @ trans(0.02575, 0, -0.02)
    T_bl = T_bl @ rot_x(q_hpitch_l)
    T_bl = T_bl @ trans(0, 0, -0.067)
    T_bl = T_bl @ rot_x(q_kpitch_l)
    T_bl = T_bl @ trans(0, 0, 0)
    T_bl = T_bl @ rot_z(q_ayaw_l)
    T_bl = T_bl @ trans(-0.018, -0.029, -0.004)
    foot_l = T_bl

    return {
        "hand_r": hand_r,
        "hand_l": hand_l,
        "foot_r": foot_r,
        "foot_l": foot_l
    }
