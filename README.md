# Climbing-Robot

## 1. Sources
 - Lecture Materials provided by Jacky Baltes
 - Language Models (i.e. ChatGPT)
 - MuJoCo Documentation 

## 2. Requirements
The project is implemented in **Python 3** and requires the following libraries:

### 2.1 Core dependencies
- `numpy` – numerical computations
- `mujoco` – physics simulation and robot model
- `mujoco.viewer` – interactive visualization

### 2.2 Additional libraries
- `opencv-python (cv2)` – camera-based hold detection
- `matplotlib` – plotting and debugging
- `dataclasses` – structured data containers (Python ≥ 3.7)
- `xml.etree.ElementTree` – parsing climbing route XML files
- `typing` – type annotations

## 3. Running our program
After installing all the required libraries you can simply run 'main.py' to see the simulation setup or run 'main_climbing.py' to get the planned motions. A fully functioning climbing simulation is not implemented yet.
