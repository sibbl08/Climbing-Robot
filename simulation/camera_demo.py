"""
Camera Demo Script for Climbing Robot Simulation

This script demonstrates different camera features:
1. Using fixed cameras defined in XML
2. Using robot-mounted camera
3. Detecting climbing holds and getting their positions
4. Saving camera images

Run this to see the camera system in action.
"""

from pathlib import Path
import mujoco  # type: ignore
import mujoco.viewer  # type: ignore
import numpy as np
import cv2
import time

from init_robot import initialize_robot_pose
from camera_vision import CameraVision, get_hold_positions_array


def main():
    # Load model
    xml_path = (Path(__file__).parent.parent / "Mujoco" / "Robot_V3.3.xml").resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # Initialize robot
    initialize_robot_pose(model, data)
    mujoco.mj_forward(model, data)
    
    # Initialize camera vision system
    camera = CameraVision(model, data, width=800, height=600)
    
    print("\n" + "="*60)
    print("CAMERA VISION SYSTEM DEMO")
    print("="*60)
    
    # 1. Detect all climbing holds
    print("\n[1] Detecting climbing holds from simulation data...")
    camera.print_hold_report()
    
    # 2. Get hold positions as numpy array (useful for path planning)
    print("[2] Getting hold positions as numpy array...")
    hold_positions = get_hold_positions_array(model, data)
    print(f"    Shape: {hold_positions.shape}")
    print(f"    Min Z (lowest hold): {hold_positions[:, 2].min():.3f}m")
    print(f"    Max Z (highest hold): {hold_positions[:, 2].max():.3f}m")
    
    # 3. Get robot's current position
    chest_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chest")
    robot_pos = data.xpos[chest_id]
    print(f"\n[3] Robot chest position: [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")
    
    # 4. Find nearest holds to robot
    print("\n[4] Nearest 5 holds to robot:")
    nearest = camera.get_nearest_holds(robot_pos, n=5)
    for i, hold in enumerate(nearest, 1):
        print(f"    {i}. {hold.name}: distance = {hold.distance_to_robot:.3f}m")
    
    # 5. Find reachable holds (within arm's reach)
    print("\n[5] Holds within reach (0.25m):")
    reachable = camera.get_holds_in_reach(robot_pos, max_reach=0.25)
    if reachable:
        for hold in reachable:
            print(f"    - {hold.name}: distance = {hold.distance_to_robot:.3f}m")
    else:
        print("    No holds within reach")
    
    # 6. Find next hold to climb to
    print("\n[6] Next hold above current position:")
    # Use hand position for more accurate climbing planning
    hand_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_r_ref")
    hand_pos = data.xpos[hand_r_id] if hand_r_id >= 0 else robot_pos
    next_hold = camera.get_next_hold_above(hand_pos)
    if next_hold:
        print(f"    {next_hold.name} at height {next_hold.position[2]:.3f}m "
              f"(distance: {next_hold.distance_to_robot:.3f}m)")
    
    # 7. Capture and save camera images from different viewpoints
    print("\n[7] Capturing camera images...")
    
    # View from front
    camera.set_camera_position(lookat=[0, 0, 0.6], distance=1.5, azimuth=180, elevation=-10)
    img_front = camera.get_camera_image()
    
    # View from side
    camera.set_camera_position(lookat=[0, 0, 0.6], distance=1.5, azimuth=225, elevation=-20)
    img_side = camera.get_camera_image()
    
    # View from top
    camera.set_camera_position(lookat=[0, 0, 0.6], distance=2.0, azimuth=180, elevation=-60)
    img_top = camera.get_camera_image()
    
    # Robot-mounted view
    camera.set_camera_on_robot("chest", offset=np.array([0, 0.15, 0.1]))
    img_robot = camera.get_camera_image()
    
    # Save images
    output_dir = Path(__file__).parent / "camera_output"
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / "view_front.png"), cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "view_side.png"), cv2.cvtColor(img_side, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "view_top.png"), cv2.cvtColor(img_top, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "view_robot.png"), cv2.cvtColor(img_robot, cv2.COLOR_RGB2BGR))
    
    print(f"    Saved 4 camera images to: {output_dir}")
    
    # 8. Show hold detection visualization
    print("\n[8] Generating hold detection visualization...")
    camera.set_camera_position(lookat=[0, 0, 0.6], distance=1.5, azimuth=180, elevation=-10)
    annotated, detected = camera.detect_holds_from_image()
    
    # Add text overlay
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.putText(annotated_bgr, f"Detected {len(detected)} hold regions", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(str(output_dir / "hold_detection.png"), annotated_bgr)
    print(f"    Saved hold detection image with {len(detected)} regions detected")
    
    # 9. Display combined view
    print("\n[9] Displaying camera views (press 'q' to close)...")
    
    # Create a combined display
    img_front_small = cv2.resize(cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR), (400, 300))
    img_side_small = cv2.resize(cv2.cvtColor(img_side, cv2.COLOR_RGB2BGR), (400, 300))
    img_robot_small = cv2.resize(cv2.cvtColor(img_robot, cv2.COLOR_RGB2BGR), (400, 300))
    annotated_small = cv2.resize(annotated_bgr, (400, 300))
    
    # Add labels
    cv2.putText(img_front_small, "Front View", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img_side_small, "Side View", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img_robot_small, "Robot Camera", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_small, "Hold Detection", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Combine into grid
    top_row = np.hstack([img_front_small, img_side_small])
    bottom_row = np.hstack([img_robot_small, annotated_small])
    combined = np.vstack([top_row, bottom_row])
    
    cv2.imshow("Camera Views - Press 'q' to close", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nThe camera system provides:")
    print("  - Climbing hold detection and position extraction")
    print("  - Multiple camera viewpoints (fixed and robot-mounted)")
    print("  - Distance calculations for path planning")
    print("  - Image capture and visualization")
    print("\nUse 'camera_vision.py' module in your climbing algorithms!")


if __name__ == "__main__":
    main()
