from pathlib import Path
import mujoco
import mujoco.viewer
import time

from init_robot import initialize_robot_pose

xml_path = (Path(__file__).parent.parent / "Mujoco" / "Robot_V3.3.xml").resolve()

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)
dt = model.opt.timestep

# set initial robot pose
initialize_robot_pose(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(max(0, dt - (time.time() - start)))