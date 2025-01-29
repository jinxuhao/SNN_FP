import pybullet as p
import pybullet_data
import time
import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt
def plot_com_position(times, com_positions1, com_positions2):
    """
    绘制两台机器人质心位置随时间变化的曲线。
    """
    plt.figure()
    plt.plot(times, com_positions1[:, 0], label="Robot 1 COM X")
    plt.plot(times, com_positions1[:, 1], label="Robot 1 COM Y")
    plt.plot(times, com_positions1[:, 2], label="Robot 1 COM Z")
    plt.plot(times, com_positions2[:, 0], linestyle='--', label="Robot 2 COM X")
    plt.plot(times, com_positions2[:, 1], linestyle='--', label="Robot 2 COM Y")
    plt.plot(times, com_positions2[:, 2], linestyle='--', label="Robot 2 COM Z")
    plt.title("Center of Mass Position Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()

def plot_joint_velocities(times, joint_velocities1, joint_velocities2):
    """
    绘制两台机器人关节速度随时间变化的曲线。
    """
    plt.figure()
    for i in range(joint_velocities1.shape[1]):
        plt.plot(times, joint_velocities1[:, i], label=f"Robot 1 Joint {i} Velocity")
        plt.plot(times, joint_velocities2[:, i], linestyle='--', label=f"Robot 2 Joint {i} Velocity")
    plt.title("Joint Velocities Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.legend()
    plt.grid()

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

def plot_joint_torques(times, joint_torques1, joint_torques2):
    """
    绘制两台机器人关节力矩随时间变化的曲线。
    """
    plt.figure()
    for i in range(joint_torques1.shape[1]):
        plt.plot(times, joint_torques1[:, i], label=f"Robot 1 Joint {i} Torque")
        plt.plot(times, joint_torques2[:, i], linestyle='--', label=f"Robot 2 Joint {i} Torque")
    plt.title("Joint Torques Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.grid()


def initialize_robot(urdf_path, gui=True, gravity=-0, friction=0.0001, base_position=(0, 0, 0)):
    """
    初始化 PyBullet 仿真环境并加载机器人。
    """
    # 启动仿真环境
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 默认数据路径
    p.setGravity(0, 0, gravity)  # 设置重力

    # 设置物理引擎参数
    p.setPhysicsEngineParameter(solverResidualThreshold=1e-6)
    p.setPhysicsEngineParameter(numSolverIterations=150)


    # 加载机器人，并指定初始位置
    robot_id = p.loadURDF(urdf_path, useFixedBase=True, basePosition=base_position)

    # 设置关节摩擦力
    for joint_index in range(p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, joint_index, lateralFriction=friction)

    # 获取关节数
    num_joints = p.getNumJoints(robot_id)
    print(f"Robot loaded with {num_joints} joints at position {base_position}.")

    # 设置初始姿势（竖直位置：所有关节角度为 0）
    for joint_index in range(num_joints):
        p.resetJointState(robot_id, joint_index, targetValue=0)

    # 打印所有关节的最大力矩限制
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_index)
        joint_name = joint_info[1].decode('utf-8')  # 获取并解码关节名称
        max_force = joint_info[10]  # 获取关节最大力矩
        print(f"Joint {joint_index} ({joint_name}) max force: {max_force}")

    return physics_client, robot_id, num_joints

def get_movable_joints(robot_id):
    """
    获取机器人中所有可运动的关节索引。
    """
    movable_joints = []
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        joint_type = joint_info[2]  # Joint type
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:  # 只保留可运动关节
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

def remove_static_friction(robot_id):
    """
    将所有关节的静态摩擦和阻尼设置为零。
    """
    for joint_index in range(p.getNumJoints(robot_id)):
        p.changeDynamics(
            robot_id,
            joint_index,
            lateralFriction=0.0,  # 取消侧向摩擦
            spinningFriction=0.0,  # 取消旋转摩擦
            rollingFriction=0.0,  # 取消滚动摩擦
            jointDamping=0.0      # 取消关节阻尼
        )
    print("Static friction and damping removed.")


def apply_joint_torques(robot_id, movable_joints, torques):
    """
    对机器人关节施加力矩。
    """
    for joint_index, torque in zip(movable_joints, torques):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.TORQUE_CONTROL,
            force=torque
        )

def pid_controller(target, current, kp, ki, kd, integral, prev_error, dt):
    """
    简单的 PID 控制器实现。
    """
    error = target - current
    integral += error * dt
    derivative = (error - prev_error) / dt
    output = kp * error + ki * integral + kd * derivative
    return output, integral, error

def main():
    # URDF 文件路径
    urdf_path = "/workspace/src/universal_robot/ur_description/urdf/ur3.urdf"

    # 初始化机器人 1 和机器人 2，设置不同初始位置
    physics_client, robot1_id, num_joints1 = initialize_robot(urdf_path, gui=True, base_position=(0, 0, 0))
    _, robot2_id, num_joints2 = initialize_robot(urdf_path, gui=False, base_position=(1, 0, 0))  # 第二台机器人位于 (1, 0, 0)

    # 获取可运动关节
    movable_joints1 = get_movable_joints(robot1_id)
    movable_joints2 = get_movable_joints(robot2_id)
    
    remove_static_friction(robot1_id)
    remove_static_friction(robot2_id)
    p.setGravity(0, 0, 0)  # 取消重力，纯粹观察 PID 性能
    p.setPhysicsEngineParameter(contactBreakingThreshold=1e-5)


    # 仿真参数
    simulation_time = 10  # 仿真时长（秒）
    time_step = 0.01  # 时间步长（秒）
    steps = int(simulation_time / time_step)

    # PID 控制器参数（机器人 1 和机器人 2 分别使用不同参数）
    kp1, ki1, kd1 = 500,10,0  # 机器人 1 的 PID 增益
    kp2, ki2, kd2 = 1.75, 0, 0  # 机器人 2 的 PID 增益

    integral1 = np.zeros(len(movable_joints1))
    prev_error1 = np.zeros(len(movable_joints1))

    integral2 = np.zeros(len(movable_joints2))
    prev_error2 = np.zeros(len(movable_joints2))

    # 数据存储
    times = []
    joint_angles1 = []
    joint_angles2 = []
    joint_torques1 = []
    joint_torques2 = []

    # 目标关节位置
    first_joint_angle = 45 * (np.pi / 180)  # 第一个关节目标值 (弧度)
    joint_targets = np.array([first_joint_angle] + [0] * (len(movable_joints1) - 1))
    print(f"Joint targets (radians): {joint_targets}")

    p.setTimeStep(0.01)  # 设置时间步长为 0.01 秒

    print("开始仿真...")
    for step in range(steps):
        # 获取机器人 1 的关节状态
        q1, dq1 = get_joint_states(robot1_id, movable_joints1)

        # 获取机器人 2 的关节状态
        q2, dq2 = get_joint_states(robot2_id, movable_joints2)

        # PID 控制器计算力矩（机器人 1）
        torques1 = []
        for i in range(len(movable_joints1)):
            torque, integral1[i], prev_error1[i] = pid_controller(
                joint_targets[i], q1[i], kp1, ki1, kd1, integral1[i], prev_error1[i], time_step
            )
            torques1.append(torque)

        # PID 控制器计算力矩（机器人 2）
        torques2 = []
        for i in range(len(movable_joints2)):
            torque, integral2[i], prev_error2[i] = pid_controller(
                joint_targets[i], q2[i], kp2, ki2, kd2, integral2[i], prev_error2[i], time_step
            )
            torques2.append(torque)

        # 施加力矩到机器人 1 和机器人 2
        apply_joint_torques(robot1_id, movable_joints1, torques1)
        apply_joint_torques(robot2_id, movable_joints2, torques2)

        # 存储数据
        times.append(step * time_step)
        joint_angles1.append(q1)
        joint_angles2.append(q2)
        joint_torques1.append(torques1)
        joint_torques2.append(torques2)

        # 仿真步进
        p.stepSimulation()
        time.sleep(time_step)

        # 调试力矩输出
        print(f"Step {step}, Robot 1 Torques: {torques1}")
        print(f"Step {step}, Robot 2 Torques: {torques2}")

    print("仿真结束")
    p.disconnect()

    # 转换为 NumPy 数组
    joint_angles1 = np.array(joint_angles1)
    joint_angles2 = np.array(joint_angles2)
    joint_torques1 = np.array(joint_torques1)
    joint_torques2 = np.array(joint_torques2)

    # 调用绘图函数
    plot_joint_angles(times, joint_angles1, joint_angles2)
    plot_joint_torques(times, joint_torques1, joint_torques2)

    # 显示图像
    plt.show()

if __name__ == "__main__":
    main()
