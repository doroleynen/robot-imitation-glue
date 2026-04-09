import numpy as np
from airo_robots.manipulators.hardware.ur_rtde import URrtde

print("before connect")
robot = URrtde("10.42.0.163")
print("connected")

print("before get joints")
q = robot.get_joint_configuration()
print("after get joints")

print(np.array(q))