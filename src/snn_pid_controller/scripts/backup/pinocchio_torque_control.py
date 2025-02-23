import pybullet as p
import pybullet_data
import time
import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt

def initialize_robot(urdf_path, gui=True, gravity=-9.81):
    """
    初始化 PyBullet 仿真环境并加载 UR3 机器人。
    """
    # 启动仿真环境
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 默认数据路径
    p.setGravity(0, 0, gravity)  # 设置重力

    # 加载 UR3 机器人
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)

    # 获取关节数
    num_joints = p.getNumJoints(robot_id)
    print(f"Robot loaded with {num_joints} joints.")

    # 设置初始姿势（竖直位置：所有关节角度为 0）
    for joint_index in range(num_joints):
        p.resetJointState(robot_id, joint_index, targetValue=0)

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

def set_joint_angle(robot_id, joint_index, target_angle, control_mode=p.POSITION_CONTROL):
    """
    设置机器人某个关节的目标角度。
    """
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=joint_index,
        controlMode=control_mode,
        targetPosition=target_angle
    )

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

def plot_com_position(times, com_positions):
    """
    绘制质心位置随时间变化的曲线。
    
    参数:
    - times: 时间序列 (N,)
    - com_positions: 质心位置数据 (Nx3)
    """
    plt.figure()
    plt.plot(times, com_positions[:, 0], label="COM X")
    plt.plot(times, com_positions[:, 1], label="COM Y")
    plt.plot(times, com_positions[:, 2], label="COM Z")
    plt.title("Center of Mass Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()

def plot_joint_velocities(times, joint_velocities):
    """
    绘制关节速度随时间变化的曲线。
    
    参数:
    - times: 时间序列 (N,)
    - joint_velocities: 关节速度数据 (NxM)
    """
    plt.figure()
    for i in range(joint_velocities.shape[1]):
        plt.plot(times, joint_velocities[:, i], label=f"Joint {i} Velocity")
    plt.title("Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.legend()
    plt.grid()

def plot_joint_angles(times, joint_angles):
    """
    绘制关节角度随时间变化的曲线。
    
    参数:
    - times: 时间序列 (N,)
    - joint_angles: 关节角度数据 (NxM)
    """
    plt.figure()
    for i in range(joint_angles.shape[1]):
        plt.plot(times, joint_angles[:, i], label=f"Joint {i} Angle")
    plt.title("Joint Angles")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid()

def plot_joint_torques(times, joint_torques):
    """
    绘制关节力矩随时间变化的曲线。

    参数:
    - times: 时间序列 (N,)
    - joint_torques: 关节力矩数据 (NxM)，其中 N 是时间步数，M 是关节数量。
    """
    plt.figure()
    for i in range(joint_torques.shape[1]):
        plt.plot(times, joint_torques[:, i], label=f"Joint {i} Torque")
    plt.title("Joint Torques")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.grid()

def main():
    # URDF 文件路径
    urdf_path = "/workspace/src/universal_robot/ur_description/urdf/ur3.urdf"

    # 初始化机器人
    physics_client, robot1_id, num_joints = initialize_robot(urdf_path)

    
    movable_joints = get_movable_joints(robot1_id)
    pin_model = pin.buildModelFromUrdf(urdf_path)
    pin_data = pin_model.createData()

    # 仿真参数
    simulation_time = 25  # 仿真时长（秒）
    time_step = 0.01  # 时间步长（秒）
    steps = int(simulation_time / time_step)

    # PID 控制器参数
    ku =17.5
    Tu = 0.25
    kp = 0.6*ku
    kp, ki, kd = 0.6*ku, 2*kp/Tu, kp*Tu/8  # PID 增益
    integral = np.zeros(len(movable_joints))
    prev_error = np.zeros(len(movable_joints))

    # 数据存储
    times = []
    com_positions = []
    joint_velocities = []
    joint_angles = []
    joint_torques = []

    # 目标关节位置，第一个关节目标为 90 度，其他为 0
    first_joint_angle = 45 * (np.pi / 180)  # 第一个关节目标值 (弧度)
    joint_targets = np.array([first_joint_angle] + [0] * (len(movable_joints) - 1))
    print(f"Joint targets (radians): {joint_targets}")

    print("开始仿真...")
    for step in range(steps):
        # 获取关节状态
        q, dq = get_joint_states(robot1_id, movable_joints)
        pin.centerOfMass(pin_model, pin_data, q, dq)
        com = pin_data.com[0]

        # PID 控制器计算力矩
        torques = []
        for i in range(len(movable_joints)):
            torque, integral[i], prev_error[i] = pid_controller(
                joint_targets[i], q[i], kp, ki, kd, integral[i], prev_error[i], time_step
            )
            torques.append(torque)

        # 施加力矩
        apply_joint_torques(robot1_id, movable_joints, torques)

        # 存储数据
        times.append(step * time_step)
        com_positions.append(com)
        joint_velocities.append(dq)
        joint_angles.append(q)
        joint_torques.append(torques)

        # 仿真步进
        p.stepSimulation()
        time.sleep(time_step)

        # 调试力矩输出
        print(f"Step {step}, Torques: {torques}")

    print("仿真结束")
    p.disconnect()


    # 转换为 numpy 数组
    com_positions = np.array(com_positions)
    joint_velocities = np.array(joint_velocities)
    joint_angles = np.array(joint_angles)
    joint_torques = np.array(joint_torques)  # 转换为 numpy 数组

    # 绘图
    plot_com_position(times, com_positions)
    plot_joint_velocities(times, joint_velocities)
    plot_joint_angles(times, joint_angles)
    plot_joint_torques(times, joint_torques)  # 绘制关节力矩曲线
    plt.show()


if __name__ == "__main__":
    main()