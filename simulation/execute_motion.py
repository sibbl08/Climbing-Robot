from dataclasses import dataclass

@dataclass
class MotionStep:
    limb: str
    joint_indices: list
    target_angles: list
    duration: float = 0.5 

def get_joint_indices(limb: str):
    
    """
    Returns the list of joint indices in the MuJoCo model for each limb.
    YOU MUST ADAPT these indices based on your robot model.
    """
    mapping = {
        "right_hand": [7, 8, 9, 10],  
        "left_hand":  [11, 12, 13, 14],
        "right_foot": [3, 4, 5, 6],
        "left_foot":  [15, 16, 17, 18]
    }

    if limb not in mapping:
        raise ValueError(f"Unknown limb {limb} in get_joint_indices()")

    return mapping[limb]


def create_motion_step(ik_result):
    """
    Converts the IK result into a MotionStep instruction.
    """
    limb = ik_result["limb"]
    joint_angles = ik_result["joint_angles"]

    if joint_angles is None:
        return None 

    return MotionStep(
        limb=limb,
        joint_indices=get_joint_indices(limb),
        target_angles=list(joint_angles),
        duration=0.5
    )
