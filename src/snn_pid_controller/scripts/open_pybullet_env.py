import pybullet as p
p.connect(p.DIRECT)
robot_id = p.loadURDF(" /workspace/src/tiago_description/robots/tiago.urdf")
