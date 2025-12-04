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
    Forward kinematics function for the climbing robot (Version V3.3).

    Parameters
    ----------
    q : list or array (size = 16)
        Joint angles in radians in the following order:
        [Right Arm 4 joints,
         Left Arm 4 joints,
         Left Leg 4 joints,
         Right Leg 4 joints]
    
    T_chest_in_world : 4x4 numpy array
        Homogeneous transformation matrix of the robot chest in world coordinates. (Position and orientation is important to correctly calculate fkinematics in world frame.)

    Returns
    -------
    dict
        A dictionary containing 4x4 homogeneous transformation matrices
        for each end-effector in world coordinates:
        {
            "hand_r": 4x4 matrix,
            "hand_l": 4x4 matrix,
            "foot_l": 4x4 matrix,
            "foot_r": 4x4 matrix
        }
    
    Notes
    -----
    - All rotations are applied in local end-effector frames.
    - Orientation is stored inside the rotation part of the matrix: T[:3, :3]
    - Position is inside T[:3, 3]
    
    Example
    -------
    >>> q = np.zeros(16)  # all joint angles zero
    >>> T_chest = np.eye(4)
    >>> fk_result = fk(q, T_chest)
    >>> hand_pos = fk_result["hand_r"][:3, 3]       # end-effector position
    >>> hand_rot = fk_result["hand_r"][:3, :3]      # orientation (rotation matrix)

    """
    
    # Joints
    (q_syaw_r, q_spitch_r, q_epitch_r, q_wpitch_r,
     q_syaw_l, q_spitch_l, q_epitch_l, q_wpitch_l,
     q_hyaw_l, q_hpitch_l, q_kpitch_l, q_ayaw_l,
     q_hyaw_r, q_hpitch_r, q_kpitch_r, q_ayaw_r) = q

    # World Coordinates of Chest as starting point
    T = T_chest_in_world.copy()

    # ------------------- Right Arm -----------------------------------
    T_Hr = T.copy()
    T_Hr = T_Hr @ trans(-0.0586202141, 0.02099999999999997, 0.0330297868)
    T_Hr = T_Hr @ rot_x(q_syaw_r)
    T_Hr = T_Hr @ trans(-0.020, -0.02575, 0)
    T_Hr = T_Hr @ rot_y(q_spitch_r)
    T_Hr = T_Hr @ trans(-0.067, 0, 0)
    T_Hr = T_Hr @ rot_y(q_epitch_r)
    T_Hr = T_Hr @ trans(-0.067, 0, 0)
    T_Hr = T_Hr @ rot_y(q_wpitch_r)
    T_Hr = T_Hr @ trans(0, -0.011, 0.073)  
    hand_r = T_Hr

    # ------------------- Left Arm -----------------------------------
    T_Hl = T.copy()
    T_Hl = T_Hl @ trans(0.0581797859, 0.021, 0.0330297868)
    T_Hl = T_Hl @ rot_x(q_syaw_l)
    T_Hl = T_Hl @ trans(0.020, -0.02575, 0)
    T_Hl = T_Hl @ rot_y(-q_spitch_l)
    T_Hl = T_Hl @ trans(0.067, 0, 0)
    T_Hl = T_Hl @ rot_y(-q_epitch_l)
    T_Hl = T_Hl @ trans(0.067, 0, 0)
    T_Hl = T_Hl @ rot_y(-q_wpitch_l)
    T_Hl = T_Hl @ trans(0, -0.011, 0.073)
    hand_l = T_Hl

    # ------------------- Right Leg ---------------------------------
    T_Lr = T.copy()
    T_Lr = T_Lr @ trans(-0.0434702141, 0.021, -0.0716202141) 
    T_Lr = T_Lr @ rot_z(q_hyaw_r)
    T_Lr = T_Lr @ trans(-0.02575, 0, -0.02)  
    T_Lr = T_Lr @ rot_x(-q_hpitch_r)
    T_Lr = T_Lr @ trans(0, 0, -0.067)  
    T_Lr = T_Lr @ rot_x(q_kpitch_r)
    T_Lr = T_Lr @ trans(0.02575, 0, -0.047)  
    T_Lr = T_Lr @ rot_z(q_ayaw_r)
    T_Lr = T_Lr @ trans(0, -0.015375, -0.0464)
    T_Lr = T_Lr @ trans(0.018, -0.029, -0.004) 
    foot_r = T_Lr

    # ------------------- Left Leg ----------------------------------
    T_Ll = T.copy()
    T_Ll = T_Ll @ trans(0.0430297859, 0.021, -0.0716202141)
    T_Ll = T_Ll @ rot_z(-q_hyaw_l)  
    T_Ll = T_Ll @ trans(0.02575, 0, -0.02)
    T_Ll = T_Ll @ rot_x(-q_hpitch_l)
    T_Ll = T_Ll @ trans(0, 0, -0.067)
    T_Ll = T_Ll @ rot_x(q_kpitch_l)
    T_Ll = T_Ll @ trans(-0.02575, 0, -0.047)
    T_Ll = T_Ll @ rot_z(-q_ayaw_l)
    T_Ll = T_Ll @ trans(2.12e-10, -0.015375, -0.0464)
    T_Ll = T_Ll @ trans(-0.018, -0.029, -0.004)
    foot_l = T_Ll


    return {
        "hand_r": hand_r,
        "hand_l": hand_l,
        "foot_l": foot_l,
        "foot_r": foot_r
    }
