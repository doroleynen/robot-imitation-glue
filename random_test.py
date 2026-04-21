import numpy as np
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_robots.manipulators.position_manipulator import ManipulatorSpecs

ur3_specs = ManipulatorSpecs(
    max_joint_speeds=[3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
    max_linear_speed=1.0,
)

print("before connect")
robot = URrtde("10.42.0.163", manipulator_specs=ur3_specs)
print("connected")

print("before get joints")
q = robot.get_joint_configuration()
print("after get joints")

print(np.array(q))