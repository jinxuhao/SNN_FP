import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt


def plot_joint_angles(times, joint_angles1, joint_angles2):
    """
    绘制两台机器人关节角度随时间变化的曲线。
    """
    plt.figure()
    for i in range(joint_angles1.shape[1]):
        plt.plot(times, joint_angles1[:, i], label=f"Robot 1 Joint {i} Angle")
        plt.plot(times, joint_angles2[:, i], linestyle='--', label=f"Robot 2 Joint {i} Angle")
    plt.title("Joint Angles Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid()


def initialize_robot(urdf_path, gui=True, gravity=0, friction=0.0, base_position=(0, 0, 0)):
    """
    初始化 PyBullet 仿真环境并加载机器人。
    """
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, gravity)
    p.setPhysicsEngineParameter(solverResidualThreshold=1e-6)
    p.setPhysicsEngineParameter(numSolverIterations=150)
    robot_id = p.loadURDF(urdf_path, useFixedBase=True, basePosition=base_position)
    for joint_index in range(p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, joint_index, lateralFriction=friction)
    return physics_client, robot_id


def get_movable_joints(robot_id):
    """
    获取机器人中所有可运动的关节索引。
    """
    movable_joints = []
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        joint_type = joint_info[2]
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            movable_joints.append(i)
    return movable_joints


def get_joint_states(robot_id, movable_joints):
    """
    获取机器人可运动关节的角度和速度。
    """
    joint_states = p.getJointStates(robot_id, movable_joints)
    q = np.array([state[0] for state in joint_states])  # 关节角度
    dq = np.array([state[1] for state in joint_states])  # 关节速度
    return q, dq


def apply_joint_velocities(robot_id, movable_joints, velocities):
    """
    对机器人关节设置速度。
    """
    for joint_index, velocity in zip(movable_joints, velocities):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=velocity
        )


def pid_controller(target, current, kp, ki, kd, integral, prev_error, dt):
    """
    完整的 PID 控制器实现。
    """
    error = target - current
    integral += error * dt
    derivative = (error - prev_error) / dt
    output = kp * error + ki * integral + kd * derivative
    return output, integral, error


def main():
    urdf_path = "/workspace/src/universal_robot/ur_description/urdf/ur3.urdf"
    physics_client, robot1_id = initialize_robot(urdf_path, gui=True, base_position=(0, 0, 0))
    _, robot2_id = initialize_robot(urdf_path, gui=False, base_position=(1, 0, 0))
    movable_joints1 = get_movable_joints(robot1_id)
    movable_joints2 = get_movable_joints(robot2_id)

    p.setGravity(0, 0, 0)

    simulation_time = 20
    time_step = 0.01
    steps = int(simulation_time / time_step)

    # PID 控制器参数
    kp1, ki1, kd1 = 1.75, 0.2, 0.2  # Robot 1 的 PID 增益
    kp2, ki2, kd2 = 5, 1, 0.5  # Robot 2 的 PID 增益

    integral1 = np.zeros(len(movable_joints1))
    prev_error1 = np.zeros(len(movable_joints1))
    integral2 = np.zeros(len(movable_joints2))
    prev_error2 = np.zeros(len(movable_joints2))

    times = []
    joint_angles1 = []
    joint_angles2 = []

    first_joint_angle = 45 * (np.pi / 180)
    joint_targets = np.array([first_joint_angle] + [0] * (len(movable_joints1) - 1))

    for step in range(steps):
        q1, dq1 = get_joint_states(robot1_id, movable_joints1)
        q2, dq2 = get_joint_states(robot2_id, movable_joints2)

        velocities1 = []
        for i in range(len(movable_joints1)):
            velocity, integral1[i], prev_error1[i] = pid_controller(
                joint_targets[i], q1[i], kp1, ki1, kd1, integral1[i], prev_error1[i], time_step
            )
            velocities1.append(velocity)

        velocities2 = []
        for i in range(len(movable_joints2)):
            velocity, integral2[i], prev_error2[i] = pid_controller(
                joint_targets[i], q2[i], kp2, ki2, kd2, integral2[i], prev_error2[i], time_step
            )
            velocities2.append(velocity)

        apply_joint_velocities(robot1_id, movable_joints1, velocities1)
        apply_joint_velocities(robot2_id, movable_joints2, velocities2)

        times.append(step * time_step)
        joint_angles1.append(q1)
        joint_angles2.append(q2)

        p.stepSimulation()
        time.sleep(time_step)

    p.disconnect()

    joint_angles1 = np.array(joint_angles1)
    joint_angles2 = np.array(joint_angles2)
    plot_joint_angles(times, joint_angles1, joint_angles2)
    plt.show()


if __name__ == "__main__":
    main()
