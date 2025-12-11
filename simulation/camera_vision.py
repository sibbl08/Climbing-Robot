"""
Camera Vision Module for Climbing Robot Simulation

This module provides camera functionality to detect and locate climbing holds (cribs)
on the wall and send their positions to the robot for path planning.

Features:
- Multiple camera placement options (fixed, robot-mounted)
- Climbing hold detection using color segmentation
- 3D position extraction from simulation data
- Real-time visualization of detected holds
"""

import mujoco  # type: ignore
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass


@dataclass
class ClimbingHold:
    """Represents a detected climbing hold with its properties."""
    name: str
    position: np.ndarray  # 3D world position [x, y, z]
    color: Tuple[float, float, float, float]  # RGBA color
    pixel_position: Optional[Tuple[int, int]] = None  # Position in camera image
    distance_to_robot: Optional[Union[float, np.floating]] = None


class CameraVision:
    """
    Camera system for detecting climbing holds in the MuJoCo simulation.
    
    This camera can be placed anywhere in the scene and will detect the 
    climbing holds (cribs) and provide their 3D positions to the robot.
    """
    
    def __init__(self, model: Any, data: Any,
                 width: int = 640, height: int = 480):
        """
        Initialize the camera vision system.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            width: Camera image width in pixels
            height: Camera image height in pixels
        """
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        
        # Create renderer for camera
        self.renderer = mujoco.Renderer(model, width, height)
        
        # Camera configuration
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        
        # Default camera position (can be changed)
        self.set_camera_position(
            lookat=[0, 0, 0.5],  # Look at center of wall
            distance=1.5,
            azimuth=180,  # Looking at the wall from front
            elevation=-20
        )
        
        # Hold detection parameters
        self.hold_name_patterns = ['Jug', 'hold_', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
                                   'foot_l_start', 'foot_r_start', 'hand_l_start', 'hand_r_start']
        
        # Cache for detected holds
        self._cached_holds: List[ClimbingHold] = []
        self._cache_valid = False
        
    def set_camera_position(self, lookat: List[float], distance: float,
                           azimuth: float, elevation: float):
        """
        Set the camera position and orientation.
        
        Args:
            lookat: 3D point the camera is looking at [x, y, z]
            distance: Distance from the lookat point
            azimuth: Horizontal rotation angle (degrees)
            elevation: Vertical rotation angle (degrees)
        """
        self.camera.lookat[:] = lookat
        self.camera.distance = distance
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation
        self._cache_valid = False
        
    def set_camera_on_robot(self, body_name: str = "chest", 
                           offset: np.ndarray = np.array([0, 0.1, 0.1])):
        """
        Mount the camera on the robot body.
        
        Args:
            body_name: Name of the robot body to mount camera on
            offset: Local offset from the body origin
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        
        # Get body position and orientation
        body_pos = self.data.xpos[body_id]
        body_quat = self.data.xquat[body_id]
        
        # Transform offset to world coordinates
        rotated_offset = np.zeros(3)
        mujoco.mju_rotVecQuat(rotated_offset, offset, body_quat)
        camera_pos = body_pos + rotated_offset
        
        # Set camera to look at the wall (positive y direction from robot)
        self.camera.lookat[:] = [camera_pos[0], 0.0, camera_pos[2]]
        self.camera.distance = 0.5
        self.camera.azimuth = 180
        self.camera.elevation = 0
        self._cache_valid = False
        
    def get_camera_image(self) -> np.ndarray:
        """
        Capture an image from the camera.
        
        Returns:
            RGB image as numpy array (height, width, 3)
        """
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render().copy()
    
    def get_depth_image(self) -> np.ndarray:
        """
        Capture a depth image from the camera.
        
        Returns:
            Depth image as numpy array (height, width)
        """
        self.renderer.update_scene(self.data, self.camera)
        self.renderer.enable_depth_rendering(True)
        depth = self.renderer.render().copy()
        self.renderer.enable_depth_rendering(False)
        return depth
    
    def detect_holds_from_model(self) -> List[ClimbingHold]:
        """
        Detect climbing holds directly from the MuJoCo model data.
        This is the most reliable method as it uses the actual simulation data.
        
        Returns:
            List of ClimbingHold objects with 3D positions
        """
        holds = []
        
        # Iterate through all bodies in the model
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name is None:
                continue
                
            # Check if this body is a climbing hold
            is_hold = any(pattern in body_name for pattern in self.hold_name_patterns)
            if not is_hold:
                continue
            
            # Get body position
            position = self.data.xpos[i].copy()
            
            # Get color from associated geom (if exists)
            color = (0.0, 1.0, 0.0, 1.0)  # Default green
            for j in range(self.model.nbody):
                if self.model.geom_bodyid[j] == i:
                    geom_rgba = self.model.geom_rgba[j]
                    if not np.allclose(geom_rgba, 0):
                        color = tuple(geom_rgba)
                        break
            
            hold = ClimbingHold(
                name=body_name,
                position=position,
                color=color
            )
            holds.append(hold)
        
        self._cached_holds = holds
        self._cache_valid = True
        return holds
    
    def detect_holds_from_image(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect climbing holds using computer vision on camera image.
        Uses color segmentation to find green and purple holds.
        
        Returns:
            Tuple of (annotated image, list of detected hold info)
        """
        image = self.get_camera_image()
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        detected = []
        annotated = image.copy()
        
        # Define color ranges for holds
        color_ranges = {
            'green': {
                'lower': np.array([35, 50, 50]),
                'upper': np.array([85, 255, 255]),
                'color_bgr': (0, 255, 0)
            },
            'purple': {
                'lower': np.array([125, 50, 50]),
                'upper': np.array([155, 255, 255]),
                'color_bgr': (255, 0, 255)
            }
        }
        
        for color_name, params in color_ranges.items():
            # Create mask for this color
            mask = cv2.inRange(hsv, params['lower'], params['upper'])
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Filter small noise
                    continue
                
                # Get bounding box and center
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Draw detection
                    cv2.drawContours(annotated, [contour], -1, params['color_bgr'], 2)
                    cv2.circle(annotated, (cx, cy), 5, (255, 0, 0), -1)
                    
                    detected.append({
                        'color': color_name,
                        'pixel_position': (cx, cy),
                        'area': area
                    })
        
        return annotated, detected
    
    def get_hold_positions_for_robot(self, robot_pos: Optional[np.ndarray] = None) -> List[ClimbingHold]:
        """
        Get all hold positions formatted for robot path planning.
        Optionally calculates distances from current robot position.
        
        Args:
            robot_pos: Current robot chest position (optional)
            
        Returns:
            List of ClimbingHold objects sorted by height (z coordinate)
        """
        if not self._cache_valid:
            self.detect_holds_from_model()
        
        holds = self._cached_holds.copy()
        
        # Calculate distances if robot position provided
        if robot_pos is not None:
            for hold in holds:
                hold.distance_to_robot = np.linalg.norm(hold.position - robot_pos)
        
        # Sort by height (z coordinate) - useful for climbing
        holds.sort(key=lambda h: h.position[2])
        
        return holds
    
    def get_nearest_holds(self, position: np.ndarray, n: int = 5) -> List[ClimbingHold]:
        """
        Get the n nearest holds to a given position.
        
        Args:
            position: Reference position [x, y, z]
            n: Number of holds to return
            
        Returns:
            List of n nearest ClimbingHold objects
        """
        if not self._cache_valid:
            self.detect_holds_from_model()
        
        holds = self._cached_holds.copy()
        for hold in holds:
            hold.distance_to_robot = np.linalg.norm(hold.position - position)
        
        holds.sort(key=lambda h: h.distance_to_robot or 0.0)
        return holds[:n]
    
    def get_holds_in_reach(self, position: np.ndarray, 
                          max_reach: float = 0.3) -> List[ClimbingHold]:
        """
        Get all holds within reach of a given position.
        
        Args:
            position: Reference position [x, y, z]
            max_reach: Maximum reach distance
            
        Returns:
            List of reachable ClimbingHold objects
        """
        if not self._cache_valid:
            self.detect_holds_from_model()
        
        reachable = []
        for hold in self._cached_holds:
            distance = np.linalg.norm(hold.position - position)
            if distance <= max_reach:
                hold.distance_to_robot = distance
                reachable.append(hold)
        
        reachable.sort(key=lambda h: h.distance_to_robot or 0.0)
        return reachable
    
    def get_next_hold_above(self, current_pos: np.ndarray, 
                           min_height_diff: float = 0.05) -> Optional[ClimbingHold]:
        """
        Get the next hold above the current position.
        Useful for climbing path planning.
        
        Args:
            current_pos: Current end-effector position
            min_height_diff: Minimum height difference to consider
            
        Returns:
            Next ClimbingHold above or None
        """
        if not self._cache_valid:
            self.detect_holds_from_model()
        
        candidates = []
        for hold in self._cached_holds:
            height_diff = hold.position[2] - current_pos[2]
            if height_diff >= min_height_diff:
                hold.distance_to_robot = np.linalg.norm(hold.position - current_pos)
                candidates.append(hold)
        
        if not candidates:
            return None
        
        # Return the closest hold that's above
        candidates.sort(key=lambda h: h.distance_to_robot)
        return candidates[0]
    
    def visualize_holds(self, show_labels: bool = True) -> np.ndarray:
        """
        Create a visualization of all detected holds.
        
        Args:
            show_labels: Whether to show hold names
            
        Returns:
            Annotated camera image
        """
        image = self.get_camera_image()
        
        if not self._cache_valid:
            self.detect_holds_from_model()
        
        # Project 3D positions to 2D image coordinates would require
        # camera intrinsics - for now we'll annotate with the image-based detection
        annotated, _ = self.detect_holds_from_image()
        
        return annotated
    
    def print_hold_report(self):
        """Print a formatted report of all detected holds."""
        holds = self.detect_holds_from_model()
        
        print("\n" + "=" * 60)
        print("CLIMBING HOLD DETECTION REPORT")
        print("=" * 60)
        print(f"Total holds detected: {len(holds)}")
        print("-" * 60)
        print(f"{'Name':<25} {'X':>8} {'Y':>8} {'Z':>8}")
        print("-" * 60)
        
        for hold in holds:
            print(f"{hold.name:<25} {hold.position[0]:>8.3f} "
                  f"{hold.position[1]:>8.3f} {hold.position[2]:>8.3f}")
        
        print("=" * 60 + "\n")


def add_camera_to_model_xml(camera_name: str = "overview_cam",
                            pos: str = "0 -1.5 1.0",
                            euler: str = "1.57 0 0") -> str:
    """
    Generate XML snippet to add a camera to the MuJoCo model.
    
    Args:
        camera_name: Name for the camera
        pos: Position string "x y z"
        euler: Euler angles string "rx ry rz" in radians
        
    Returns:
        XML string to add to worldbody
    """
    return f'<camera name="{camera_name}" pos="{pos}" euler="{euler}"/>'


# Utility function to get hold positions as numpy array
def get_hold_positions_array(model: Any, 
                             data: Any) -> np.ndarray:
    """
    Quick utility to get all hold positions as a numpy array.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        
    Returns:
        Array of shape (n_holds, 3) with [x, y, z] positions
    """
    camera = CameraVision(model, data)
    holds = camera.detect_holds_from_model()
    return np.array([h.position for h in holds])
