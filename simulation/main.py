"""
Climbing Robot Simulation with Body Movement
=============================================
This simulation demonstrates a humanoid robot climbing a wall by:
1. Reaching for holds with alternating hands
2. Moving the body upward after each hand placement
3. Repeating until reaching the top

Press 'c' to start climbing, 's' to stop, 'q' to quit camera view.
Press 'v' to toggle camera view mode (side/front/dual)
"""

from pathlib import Path
import mujoco  # type: ignore
import mujoco.viewer  # type: ignore
import time
import numpy as np
import cv2

from init_robot import initialize_robot_pose
from camera_vision import CameraVision
from ik import ik
from fk import fk

xml_path = (Path(__file__).parent.parent / "Mujoco" / "Robot_V3.3.xml").resolve()

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)
dt = model.opt.timestep

# set initial robot pose
initialize_robot_pose(model, data)
mujoco.mj_forward(model, data)  # Required to compute body positions

# Initialize camera vision system - main camera and side camera
camera_main = CameraVision(model, data, width=640, height=480)
camera_side = CameraVision(model, data, width=640, height=480)

# Get all hold positions
all_holds = camera_main.detect_holds_from_model()
print(f"\n🧗 Climbing Robot Simulation")
print(f"   Total holds detected: {len(all_holds)}")

# ============================================================================
# MOTION UTILITIES
# ============================================================================

def ease_in_out_cubic(t):
    """Smooth cubic easing function."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2

def ease_in_out_sine(t):
    """Smooth sine easing function."""
    return -(np.cos(np.pi * t) - 1) / 2

def ease_out_elastic(t, amplitude=1.0, period=0.3):
    """Elastic easing for more dynamic motion."""
    if t == 0 or t == 1:
        return t
    return amplitude * pow(2, -10 * t) * np.sin((t - period / 4) * (2 * np.pi) / period) + 1

# ============================================================================
# CLIMBING CONTROLLER WITH BODY MOVEMENT
# ============================================================================

class ClimbingController:
    """
    Advanced climbing controller that:
    1. Reaches for holds with IK
    2. Moves robot body upward between hand placements
    3. Alternates between left and right hands
    4. Uses smooth motion profiles
    """
    
    # States for the climbing state machine
    STATE_IDLE = 0
    STATE_REACH_HAND = 1
    STATE_MOVE_BODY = 2
    STATE_DONE = 3
    
    def __init__(self, model, data, holds):
        self.model = model
        self.data = data
        
        # Filter out foot holds (only keep hand holds)
        hand_holds = [h for h in holds if 'foot' not in h.name.lower()]
        
        # Separate and sort holds by side
        self.left_holds = sorted(
            [h for h in hand_holds if h.position[0] < 0],
            key=lambda h: h.position[2]
        )
        self.right_holds = sorted(
            [h for h in hand_holds if h.position[0] > 0],
            key=lambda h: h.position[2]
        )
        
        # Current hold indices
        self.left_index = 0
        self.right_index = 0
        
        # State machine
        self.state = self.STATE_IDLE
        self.use_right_hand = True
        
        # Motion parameters - TUNED FOR SMOOTHER MOTION
        self.reach_steps = 100      # More steps for smoother arm motion
        self.body_move_steps = 80   # More steps for smoother body lift
        self.current_step = 0
        
        # Motion easing mode
        self.motion_mode = "cubic"  # "cubic", "sine", or "elastic"
        
        # Target tracking
        self.q_start = None
        self.q_target = None
        self.body_start_z = None
        self.body_target_z = None
        
        # For coordinated arm motion (both arms move together)
        self.other_arm_start = None
        self.other_arm_target = None
        
        # Joint mappings
        self.joint_names = {
            'right_arm': [
                "servo_shoulder_yaw_r_Revolute_shoulder_yaw_r",
                "servo_shoulder_pitch_r_Revolute_shoulder_pitch_r",
                "servo_elbow_pitch_r_Revolute_elbow_pitch_r",
                "servo_wrist_pitch_r_Revolute_wrist_pitch_r"
            ],
            'left_arm': [
                "servo_shoulder_yaw_l_Revolute_shoulder_yaw_l",
                "servo_shoulder_pitch_l_Revolute_shoulder_pitch_l",
                "servo_elbow_pitch_l_Revolute_elbow_pitch_l",
                "servo_wrist_pitch_l_Revolute_wrist_pitch_l"
            ]
        }
        
        # Statistics
        self.holds_reached = 0
        self.body_moves = 0
        self.start_z = None
        self.max_z = 0  # Track maximum height reached
        
        print(f"\n📍 Hold distribution: {len(self.left_holds)} left, {len(self.right_holds)} right")
        
    def get_eased_alpha(self, t):
        """Get eased interpolation value based on motion mode."""
        t = np.clip(t, 0, 1)
        if self.motion_mode == "cubic":
            return ease_in_out_cubic(t)
        elif self.motion_mode == "elastic":
            return ease_out_elastic(t, amplitude=0.8, period=0.4)
        else:  # sine
            return ease_in_out_sine(t)
        
    def get_chest_position(self):
        """Get current chest position."""
        chest_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chest")
        return self.data.xpos[chest_id].copy()
    
    def get_chest_transform(self):
        """Get chest transformation matrix."""
        chest_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chest")
        pos = self.data.xpos[chest_id].copy()
        quat = self.data.xquat[chest_id].copy()
        
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T
    
    def get_joint_angles(self, limb):
        """Get current joint angles for a limb."""
        q = np.zeros(4)
        for i, joint_name in enumerate(self.joint_names[limb]):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if j_id >= 0:
                qpos_adr = self.model.jnt_qposadr[j_id]
                q[i] = self.data.qpos[qpos_adr]
        return q
    
    def set_joint_angles(self, limb, q):
        """Set joint control targets."""
        for i, joint_name in enumerate(self.joint_names[limb]):
            for act_id in range(self.model.nu):
                j_id = self.model.actuator_trnid[act_id][0]
                act_joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(j_id))
                if act_joint_name == joint_name:
                    self.data.ctrl[act_id] = q[i]
                    break
    
    def set_body_z(self, z):
        """Set robot body height (z position)."""
        self.data.qpos[2] = z
        # Zero out vertical velocity
        self.data.qvel[2] = 0
    
    def get_next_hold(self):
        """Get the next hold for the current hand."""
        if self.use_right_hand:
            if self.right_index < len(self.right_holds):
                return self.right_holds[self.right_index]
        else:
            if self.left_index < len(self.left_holds):
                return self.left_holds[self.left_index]
        return None
    
    def can_reach_hold(self, hold):
        """Check if a hold is within reachable range."""
        chest_z = self.get_chest_position()[2]
        hold_z = hold.position[2]
        # Can reach holds from chest_z - 0.05 to chest_z + 0.18
        return (chest_z - 0.05) <= hold_z <= (chest_z + 0.18)
    
    def start_climbing(self):
        """Start the climbing sequence."""
        self.state = self.STATE_REACH_HAND
        self.use_right_hand = True
        self.left_index = 0
        self.right_index = 0
        self.holds_reached = 0
        self.body_moves = 0
        self.current_step = 0
        self.q_start = None
        self.q_target = None
        
        self.start_z = self.get_chest_position()[2]
        self.max_z = self.start_z
        
        print("\n" + "="*50)
        print("🧗 CLIMBING STARTED!")
        print("="*50)
        print(f"   Starting height: {self.start_z:.3f}m")
        print(f"   Left holds: {len(self.left_holds)}, Right holds: {len(self.right_holds)}")
        print(f"   Motion mode: {self.motion_mode}")
        
        hold = self.get_next_hold()
        if hold:
            print(f"   First target: {hold.name} (z={hold.position[2]:.3f}m)")
        print()
    
    def stop_climbing(self):
        """Stop climbing."""
        self.state = self.STATE_IDLE
        print("\n🛑 Climbing stopped.")
    
    def update(self):
        """Update climbing state machine."""
        if self.state == self.STATE_IDLE or self.state == self.STATE_DONE:
            return
        
        # Track max height
        current_z = self.get_chest_position()[2]
        if current_z > self.max_z:
            self.max_z = current_z
        
        # REACH HAND STATE
        if self.state == self.STATE_REACH_HAND:
            self._update_reach_hand()
        
        # MOVE BODY STATE
        elif self.state == self.STATE_MOVE_BODY:
            self._update_move_body()
    
    def _update_reach_hand(self):
        """Handle hand reaching state."""
        hold = self.get_next_hold()
        
        if hold is None:
            print("\n✅ All holds reached! Climbing complete!")
            current_z = self.get_chest_position()[2]
            print(f"   Climbed from {self.start_z:.3f}m to {current_z:.3f}m")
            print(f"   Total height gained: {current_z - self.start_z:.3f}m")
            print(f"   Maximum height: {self.max_z:.3f}m")
            print(f"   Holds reached: {self.holds_reached}")
            print(f"   Body movements: {self.body_moves}")
            self.state = self.STATE_DONE
            return
        
        # Check if we need to move body up first
        if not self.can_reach_hold(hold):
            # Transition to body move state
            self.state = self.STATE_MOVE_BODY
            self.current_step = 0
            self.body_start_z = self.data.qpos[2]
            # Move body up by the step height (0.08m per hold)
            self.body_target_z = self.body_start_z + 0.08
            print(f"   📦 Moving body: {self.body_start_z:.3f}m → {self.body_target_z:.3f}m")
            return
        
        limb = 'right_arm' if self.use_right_hand else 'left_arm'
        other_limb = 'left_arm' if self.use_right_hand else 'right_arm'
        
        # Initialize IK on first step
        if self.q_start is None:
            self.q_start = self.get_joint_angles(limb)
            self.other_arm_start = self.get_joint_angles(other_limb)
            
            # Target position
            target_pos = hold.position.copy()
            target_pos[1] = -0.01  # Slightly in front of hold
            
            T_chest = self.get_chest_transform()
            
            try:
                self.q_target, success = ik(limb, target_pos, 0.0, self.q_start, T_chest)
                
                if not success:
                    print(f"   ⚠ IK failed for {hold.name}, skipping...")
                    # Skip this hold
                    if self.use_right_hand:
                        self.right_index += 1
                    else:
                        self.left_index += 1
                    self.q_start = None
                    self.use_right_hand = not self.use_right_hand
                    return
                    
            except Exception as e:
                print(f"   ❌ IK error: {e}")
                if self.use_right_hand:
                    self.right_index += 1
                else:
                    self.left_index += 1
                self.q_start = None
                self.use_right_hand = not self.use_right_hand
                return
        
        # Interpolate arm motion with smooth easing
        raw_alpha = self.current_step / self.reach_steps
        alpha = self.get_eased_alpha(raw_alpha)
        
        q_interp = self.q_start * (1 - alpha) + self.q_target * alpha
        self.set_joint_angles(limb, q_interp)
        
        # Keep other arm stable
        self.set_joint_angles(other_limb, self.other_arm_start)
        
        # Keep body stable
        self.data.qvel[0:6] = 0
        
        self.current_step += 1
        
        # Check if motion complete
        if self.current_step >= self.reach_steps:
            hand = "Right" if self.use_right_hand else "Left"
            print(f"   ✋ {hand} hand → {hold.name} (z={hold.position[2]:.3f}m)")
            
            self.holds_reached += 1
            
            # Advance hold index
            if self.use_right_hand:
                self.right_index += 1
            else:
                self.left_index += 1
            
            # Switch hands
            self.use_right_hand = not self.use_right_hand
            
            # Reset for next reach
            self.current_step = 0
            self.q_start = None
            self.q_target = None
            self.other_arm_start = None
    
    def _update_move_body(self):
        """Handle body movement state."""
        # Interpolate body position with smooth easing
        raw_alpha = self.current_step / self.body_move_steps
        alpha = self.get_eased_alpha(raw_alpha)
        
        new_z = self.body_start_z + (self.body_target_z - self.body_start_z) * alpha
        self.set_body_z(new_z)
        
        # Keep body stable (x, y, orientation)
        self.data.qpos[0] = 0  # x
        self.data.qpos[1] = -0.03  # y
        self.data.qvel[0:6] = 0
        
        self.current_step += 1
        
        # Check if motion complete
        if self.current_step >= self.body_move_steps:
            self.body_moves += 1
            current_z = self.get_chest_position()[2]
            print(f"   📦 Body at z={current_z:.3f}m (moved {self.body_moves}x)")
            
            # Transition back to reaching
            self.state = self.STATE_REACH_HAND
            self.current_step = 0
            self.q_start = None

# Initialize climbing controller
climbing_controller = ClimbingController(model, data, all_holds)

# Configuration
SHOW_CAMERA_FEED = True
CAMERA_UPDATE_INTERVAL = 5
CAMERA_VIEW_MODE = 0  # 0=side, 1=front, 2=dual view

print("\n" + "="*50)
print("CONTROLS")
print("="*50)
print("  Press 'c' - Start climbing")
print("  Press 's' - Stop climbing")
print("  Press 'v' - Toggle camera view (behind/side/dual)")
print("  Press 'q' - Close camera window")
print("="*50 + "\n")

frame_count = 0

def create_dual_view(cam_main, cam_side, chest_z):
    """Create a combined view with side and front cameras."""
    # Behind robot view - see robot back + wall with holds
    cam_main.set_camera_position(
        lookat=[0, 0.1, chest_z + 0.2],  # Look at wall, slightly above robot
        distance=0.8,
        azimuth=180,  # From behind robot looking at wall
        elevation=0
    )
    side_img, _ = cam_main.detect_holds_from_image()
    side_img = cv2.cvtColor(side_img, cv2.COLOR_RGB2BGR)
    
    # Side view - see robot profile and wall
    cam_side.set_camera_position(
        lookat=[0, 0, chest_z],
        distance=1.5,
        azimuth=250,  # Side angle
        elevation=5
    )
    front_img, _ = cam_side.detect_holds_from_image()
    front_img = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
    
    # Resize for side-by-side
    h, w = side_img.shape[:2]
    side_img = cv2.resize(side_img, (w//2, h//2))
    front_img = cv2.resize(front_img, (w//2, h//2))
    
    # Add labels
    cv2.putText(side_img, "BEHIND VIEW", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(front_img, "SIDE VIEW", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Combine horizontally
    combined = np.hstack([side_img, front_img])
    return combined

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        start = time.time()
        
        # Update climbing controller
        climbing_controller.update()
        
        mujoco.mj_step(model, data)
        viewer.sync()
        
        frame_count += 1
        
        # Update camera view
        if SHOW_CAMERA_FEED and frame_count % CAMERA_UPDATE_INTERVAL == 0:
            # Track robot with camera
            chest_pos = climbing_controller.get_chest_position()
            
            if CAMERA_VIEW_MODE == 2:  # Dual view
                display_image = create_dual_view(camera_main, camera_side, chest_pos[2])
            else:
                # Single view mode
                if CAMERA_VIEW_MODE == 0:  # Behind robot view - shows robot back + wall with holds
                    camera_main.set_camera_position(
                        lookat=[0, 0.1, chest_pos[2] + 0.2],  # Look at wall area
                        distance=0.8,
                        azimuth=180,  # Behind robot facing wall
                        elevation=0
                    )
                else:  # Side view - shows robot profile and wall
                    camera_main.set_camera_position(
                        lookat=[0, 0, chest_pos[2]],
                        distance=1.5,
                        azimuth=250,  # Side angle view
                        elevation=5
                    )
                
                annotated_image, detected = camera_main.detect_holds_from_image()
                display_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            # Status overlay
            state_names = ["IDLE", "REACHING", "MOVING BODY", "DONE"]
            state_name = state_names[climbing_controller.state]
            
            # Get image dimensions for positioning
            h, w = display_image.shape[:2]
            
            # Create info panel at top
            cv2.rectangle(display_image, (0, 0), (w, 90), (0, 0, 0), -1)
            cv2.rectangle(display_image, (0, 0), (w, 90), (0, 200, 0), 2)
            
            cv2.putText(display_image, f"State: {state_name}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            height_gained = chest_pos[2] - climbing_controller.start_z if climbing_controller.start_z else 0
            cv2.putText(display_image, 
                       f"Height: {chest_pos[2]:.2f}m (+{height_gained:.2f}m)", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.putText(display_image, 
                       f"Holds: {climbing_controller.holds_reached} | Body Moves: {climbing_controller.body_moves}", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
            
            # Progress bar
            total_holds = len(climbing_controller.left_holds) + len(climbing_controller.right_holds)
            if total_holds > 0:
                progress = climbing_controller.holds_reached / total_holds
                bar_width = 200
                bar_x = w - bar_width - 20
                cv2.rectangle(display_image, (bar_x, 30), (bar_x + bar_width, 50), (50, 50, 50), -1)
                cv2.rectangle(display_image, (bar_x, 30), (bar_x + int(bar_width * progress), 50), (0, 200, 0), -1)
                cv2.putText(display_image, f"{int(progress*100)}%", 
                           (bar_x + bar_width + 5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if climbing_controller.state == climbing_controller.STATE_IDLE:
                cv2.putText(display_image, "Press 'c' to climb | 'v' for view", 
                           (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            view_names = ["Behind", "Side", "Dual"]
            cv2.putText(display_image, f"View: {view_names[CAMERA_VIEW_MODE]}", 
                       (w - 100, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            cv2.imshow("Climbing Robot", display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                SHOW_CAMERA_FEED = False
                cv2.destroyAllWindows()
            elif key == ord('c'):
                if climbing_controller.state == climbing_controller.STATE_IDLE:
                    climbing_controller.start_climbing()
            elif key == ord('s'):
                climbing_controller.stop_climbing()
            elif key == ord('v'):
                CAMERA_VIEW_MODE = (CAMERA_VIEW_MODE + 1) % 3
                print(f"📷 Camera view: {view_names[CAMERA_VIEW_MODE]}")
        
        time.sleep(max(0, dt - (time.time() - start)))

cv2.destroyAllWindows()
