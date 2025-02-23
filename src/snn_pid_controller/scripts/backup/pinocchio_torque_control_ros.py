import pybullet as p
import pybullet_data
import time
import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Float64MultiArray, Float64

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


def initialize_robot(urdf_path, gui=True, gravity=-9.81, base_position=(0, 0, 0)):
    """
    初始化 PyBullet 仿真环境并加载机器人。
    """
    # 启动仿真环境
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 默认数据路径
    p.setGravity(0, 0, gravity)  # 设置重力

    # 加载机器人，并指定初始位置
    robot_id = p.loadURDF(urdf_path, useFixedBase=True, basePosition=base_position)

    # 获取关节数
    num_joints = p.getNumJoints(robot_id)
    print(f"Robot loaded with {num_joints} joints at position {base_position}.")

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

# ROS 回调函数
# 全局变量存储单个 SNN 力矩
snn_torque = 0.0

def snn_torque_callback(msg):
    global snn_torque
    snn_torque = msg.data  # 只接收单个浮点数
    rospy.loginfo(f"Received SNN torque: {snn_torque}")

def publish_joint_states(pub, robot_id, movable_joints):
    """
    发布机器人关节状态（角度）。
    """
    q, dq = get_joint_states(robot_id, movable_joints)
    msg = Float64()
    msg.data = q[0]  # 示例：只发布第一个关节的状态
    pub.publish(msg)

def main():
    rospy.init_node('pybullet_snn_interface', anonymous=True)
    rate = rospy.Rate(100)  # ROS 更新频率 100 Hz

    # ROS 发布器和订阅器
    snn_torque_sub = rospy.Subscriber('/snn_output', Float64, snn_torque_callback)
    joint_state_pub = rospy.Publisher('/current_value', Float64, queue_size=1)

    # URDF 文件路径
    urdf_path = "/workspace/src/universal_robot/ur_description/urdf/ur3.urdf"

    # 初始化机器人 1 和机器人 2，设置不同初始位置
    physics_client, robot1_id, num_joints1 = initialize_robot(urdf_path, gui=True, base_position=(0, 0, 0))
    _, robot2_id, num_joints2 = initialize_robot(urdf_path, gui=False, base_position=(1, 0, 0))  # 第二台机器人位于 (1, 0, 0)

    # 获取可运动关节
    movable_joints1 = get_movable_joints(robot1_id)
    movable_joints2 = get_movable_joints(robot2_id)

    # 仿真参数
    simulation_time = 5  # 仿真时长（秒）
    time_step = 0.01  # 时间步长（秒）
    steps = int(simulation_time / time_step)

    # PID 控制器参数（机器人 1 和机器人 2 分别使用不同参数）
    kp1, ki1, kd1 = 2000, 0, 0  # 机器人 1 的 PID 增益
    kp2, ki2, kd2 = 2000, 5, 5  # 机器人 2 的 PID 增益

    integral1 = np.zeros(len(movable_joints1))
    prev_error1 = np.zeros(len(movable_joints1))

    # integral2 = np.zeros(len(movable_joints2))
    # prev_error2 = np.zeros(len(movable_joints2))

    # 数据存储
    times = []
    joint_angles1 = []
    joint_angles2 = []
    joint_torques1 = []
    joint_torques2 = []

    # 目标关节位置
    first_joint_angle = 35 * (np.pi / 180)  # 第一个关节目标值 (弧度)
    joint_targets = np.array([first_joint_angle] + [0] * (len(movable_joints1) - 1))
    print(f"Joint targets (radians): {joint_targets}")

    print("开始仿真...")
    for step in range(steps):
        # 第一个机器人使用经典 PID
        q1, dq1 = get_joint_states(robot1_id, movable_joints1)
        torques1 = []
        for i in range(len(movable_joints1)):
            torque, integral1[i], prev_error1[i] = pid_controller(
                joint_targets[i], q1[i], kp1, ki1, kd1, integral1[i], prev_error1[i], time_step
            )
            torques1.append(torque)
        apply_joint_torques(robot1_id, movable_joints1, torques1)

        # 存储第一个机器人的数据
        joint_angles1.append(q1)
        joint_torques1.append(torques1)

        # 第二个机器人使用 SNN 控制器的输出
        if isinstance(snn_torque, float):  # 确保接收到有效的浮点数力矩
            try:
                # 将单一浮点数广播为所有关节的力矩列表
                snn_torque_list = [snn_torque] * len(movable_joints2)
                apply_joint_torques(robot2_id, movable_joints2, snn_torque_list)
                rospy.loginfo(f"Applied SNN torque: {snn_torque}")
            except Exception as e:
                rospy.logerr(f"Failed to apply torque to Robot 2: {e}")
        else:
            rospy.logwarn("SNN torque not received or invalid. Skipping this step.")
            apply_joint_torques(robot2_id, movable_joints2, [0.0] * len(movable_joints2))

        # 发布第二个机器人的关节状态
        publish_joint_states(joint_state_pub, robot2_id, movable_joints2)

        # 仿真步进
        p.stepSimulation()
        time.sleep(time_step)

        # 存储第二个机器人的数据
    q2, _ = get_joint_states(robot2_id, movable_joints2)
    joint_angles2.append(q2)
    joint_torques2.append(snn_torque_list if isinstance(snn_torque, float) else [0.0] * len(movable_joints2))

    print("仿真结束")
    # p.disconnect()

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
