import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Float64

# 绘图函数
def plot_joint_angles(times, joint_angles1, joint_angles2):
    """
    绘制两台机器人关节角度随时间变化的曲线。
    """
    plt.figure()
    for i in range(joint_angles1.shape[1]):
        plt.plot(times, joint_angles1[:, i],linestyle='--',  label=f"Robot 1 Joint {i} Angle")
        plt.plot(times, joint_angles2[:, i], label=f"Robot 2 Joint {i} Angle")
    plt.title("Joint Angles Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid()

def plot_joint_velocities(times, velocities1_data, velocities2_data):
    """
    绘制两台机器人关节速度随时间变化的曲线。
    """
    plt.figure()
    for i in range(len(velocities1_data[0])):
        plt.plot(times, [v[i] for v in velocities1_data], label=f"Robot 1 Joint {i} Velocity")
        plt.plot(times, [v[i] for v in velocities2_data], linestyle='--', label=f"Robot 2 Joint {i} Velocity")
    plt.title("Joint Velocities Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.legend()
    plt.grid()

# 初始化机器人
def initialize_robot(urdf_path, gui=True, gravity=-9.81, base_position=(0, 0, 0)):
    """
    初始化 PyBullet 仿真环境并加载机器人。
    """
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, gravity)
    robot_id = p.loadURDF(urdf_path, useFixedBase=True, basePosition=base_position)

    print(f"Robot loaded at position {base_position}.")
    return physics_client, robot_id

# 获取关节状态
def get_movable_joints(robot_id):
    """
    获取机器人中所有可运动的关节索引。
    """
    movable_joints = []
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            movable_joints.append(i)
    return movable_joints

def get_joint_states(robot_id, movable_joints):
    """
    获取机器人关节的角度和速度。
    """
    joint_states = p.getJointStates(robot_id, movable_joints)
    q = np.array([state[0] for state in joint_states])  # 关节角度
    dq = np.array([state[1] for state in joint_states])  # 关节速度
    return q, dq

# 应用关节速度
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

# PID 控制器
def pid_controller(target, current, kp, ki, kd, integral, prev_error, dt):
    """
    完整的 PID 控制器实现。
    """
    error = target - current
    integral += error * dt
    derivative = (error - prev_error) / dt
    output = kp * error + ki * integral + kd * derivative
    return output, integral, error

# ROS 回调和发布
snn_velocity = 0.0  # 全局变量存储单个 SNN 输出的速度

def snn_velocity_callback(msg):
    global snn_velocity
    snn_velocity = msg.data  # 接收单个浮点数作为速度控制输入

def publish_joint_states(pub, robot_id, movable_joints):
    """
    发布机器人关节状态（角度）。
    """
    q, dq = get_joint_states(robot_id, movable_joints)
    msg = Float64()
    msg.data = q[0]  # 示例：只发布第一个关节的状态
    pub.publish(msg)

# 主函数
def main():
    rospy.init_node('pybullet_snn_interface', anonymous=True)
    rate = rospy.Rate(10)  # ROS 更新频率 100 Hz

    # ROS 发布器和订阅器
    snn_velocity_sub = rospy.Subscriber('/snn_output', Float64, snn_velocity_callback)
    joint_state_pub = rospy.Publisher('/current_value', Float64, queue_size=5)

    # 初始化 PyBullet
    urdf_path = "/workspace/src/universal_robot/ur_description/urdf/ur3.urdf"
    physics_client, robot1_id = initialize_robot(urdf_path, gui=True, base_position=(0, 0, 0))
    _, robot2_id = initialize_robot(urdf_path, gui=False, base_position=(1, 0, 0))

    movable_joints1 = get_movable_joints(robot1_id)
    movable_joints2 = get_movable_joints(robot2_id)

    # 仿真参数
    time_step = 1  # 时间步长
    simulation_time = 1000  # 仿真时间
    steps = int(simulation_time / time_step)

    # PID 参数
    kp1, ki1, kd1 = 5.75, 1, 0  # Robot 1 的 PID 增益
    integral1 = np.zeros(len(movable_joints1))
    prev_error1 = np.zeros(len(movable_joints1))

    # 数据存储
    times = []
    joint_angles1 = []
    joint_angles2 = []
    velocities1_data = []
    velocities2_data = []

    # 目标位置
    target_angle = 90 * (np.pi / 180)  # 目标角度（弧度）
    joint_targets = np.array([target_angle] + [0] * (len(movable_joints1) - 1))

    # 开始仿真
    for step in range(steps):
        q1, dq1 = get_joint_states(robot1_id, movable_joints1)

        # PID 控制计算目标速度
        velocities1 = []
        for i in range(len(movable_joints1)):
            velocity, integral1[i], prev_error1[i] = pid_controller(
                joint_targets[i], q1[i], kp1, ki1, kd1, integral1[i], prev_error1[i], time_step
            )
            velocities1.append(velocity)

        apply_joint_velocities(robot1_id, movable_joints1, velocities1)

        # 第二个机器人基于 SNN 输出速度控制
        if isinstance(snn_velocity, float):
            velocities2 = [snn_velocity] * len(movable_joints2)
            
            apply_joint_velocities(robot2_id, movable_joints2, velocities2)
        else:
            rospy.logwarn("SNN velocity not received. Setting to zero.")
            velocities2 = [0.0] * len(movable_joints2)
            apply_joint_velocities(robot2_id, movable_joints2, [0.0] * len(movable_joints2))

        # 发布关节状态
        publish_joint_states(joint_state_pub, robot2_id, movable_joints2)

        # 存储数据
        times.append(step * time_step)
        joint_angles1.append(q1)
        q2, _ = get_joint_states(robot2_id, movable_joints2)
        joint_angles2.append(q2)

        velocities1_data.append(velocities1)
        velocities2_data.append(velocities2)

        p.stepSimulation()
        rate.sleep()

    # 仿真结束
    rospy.loginfo("Simulation finished.")
    p.disconnect()

    # 转换为 NumPy 数组
    joint_angles1 = np.array(joint_angles1)
    joint_angles2 = np.array(joint_angles2)

    # 绘图
    plot_joint_angles(times, joint_angles1, joint_angles2)
    plot_joint_velocities(times, velocities1_data, velocities2_data)
    plt.show()


if __name__ == "__main__":
    main()
