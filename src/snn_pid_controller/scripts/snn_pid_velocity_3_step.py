import pybullet as p
import pybullet_data
import torch
import numpy as np
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.topology import Connection
from layers.input_layer import InputLayer
from layers.encoding_layer import EncodingLayer
from layers.integration_layer import IntegrationLayer
from layers.output_layer import ComplexOutputLayer
from layers.P_layer import PIntermediateLayer
from layers.I_layer import IIntermediateLayer
from layers.D_layer import DIntermediateLayer

# 初始化 PyBullet 仿真
def initialize_robot(urdf_path, gui=True, gravity=-9.81, base_position=(0, 0, 0)):
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, gravity)
    robot_id = p.loadURDF(urdf_path, useFixedBase=True, basePosition=base_position)
    print(f"Robot loaded at position {base_position}.")
    return physics_client, robot_id

def get_movable_joints(robot_id):
    movable_joints = []
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            movable_joints.append(i)
    return movable_joints

def apply_joint_velocities(robot_id, movable_joints, velocities):
    for joint_index, velocity in zip(movable_joints, velocities):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=velocity
        )

def get_joint_states(robot_id, movable_joints):
    joint_states = p.getJointStates(robot_id, movable_joints)
    q = np.array([state[0] for state in joint_states])  # 关节角度
    dq = np.array([state[1] for state in joint_states])  # 关节速度
    return q, dq

# PID 控制器
def pid_controller(target, current, kp, ki, kd, integral, prev_error, dt):
    error = target - current
    integral += error * dt
    derivative = (error - prev_error) / dt
    output = kp * error + ki * integral + kd * derivative
    return output, integral, error

# 初始化 SNN
def initialize_snn(num_neurons, kp, ki, kd):
    network = Network()

    input_layer = InputLayer(num_neurons=num_neurons)
    encoding_layer = EncodingLayer(num_neurons=num_neurons)
    integration_layer = IntegrationLayer(num_neurons=num_neurons)
    output_layer = ComplexOutputLayer(num_neurons=num_neurons)
    P_layer = PIntermediateLayer(num_neurons=num_neurons)
    I_layer = IIntermediateLayer(num_neurons=num_neurons)
    D_layer = DIntermediateLayer(num_neurons=num_neurons)

    network.add_layer(input_layer, name='input')
    network.add_layer(encoding_layer, name='encoding')
    network.add_layer(integration_layer, name='integration')
    network.add_layer(output_layer, name='output')
    network.add_layer(P_layer, name='p_intermediate')
    network.add_layer(I_layer, name='i_intermediate')
    network.add_layer(D_layer, name='d_intermediate')

    def create_weight_matrix(source_size, target_size, diagonal_value=1.0):
        weight_matrix = torch.zeros(target_size, source_size)
        for i in range(min(source_size, target_size)):
            weight_matrix[i, i] = diagonal_value
        return weight_matrix

     # 创建连接并添加到网络中
    input_to_encoding = Connection(
        source=input_layer,
        target=encoding_layer,
        w=torch.eye(num_neurons),
        requires_grad=False
    )
    encoding_to_integration = Connection(
        source=encoding_layer,
        target=integration_layer,
        w=torch.eye(num_neurons),
        requires_grad=False
    )
    encoding_to_p = Connection(
        source=encoding_layer,
        target=P_layer,
        w=torch.eye(num_neurons),
        requires_grad=False
    )
    encoding_to_d = Connection(
        source=encoding_layer,
        target=D_layer,
        w=torch.eye(num_neurons),
        requires_grad=False
    )
    integration_to_i = Connection(
        source=integration_layer,
        target=I_layer,
        w=torch.eye(num_neurons),
        requires_grad=False
    )
    p_to_output = Connection(
        source=P_layer,
        target=output_layer,
        w=create_weight_matrix(num_neurons, num_neurons, kp),
        requires_grad=False
    )
    i_to_output = Connection(
        source=I_layer,
        target=output_layer,
        w=create_weight_matrix(num_neurons, num_neurons, ki),
        requires_grad=False
    )
    d_to_output = Connection(
        source=D_layer,
        target=output_layer,
        w=create_weight_matrix(num_neurons, num_neurons, kd),
        requires_grad=False
    )

    # 添加连接到网络中
    network.add_connection(input_to_encoding, source='input', target='encoding')
    network.add_connection(encoding_to_integration, source='encoding', target='integration')
    network.add_connection(encoding_to_p, source='encoding', target='p_intermediate')
    network.add_connection(encoding_to_d, source='encoding', target='d_intermediate')
    network.add_connection(integration_to_i, source='integration', target='i_intermediate')
    network.add_connection(p_to_output, source='p_intermediate', target='output')
    network.add_connection(i_to_output, source='i_intermediate', target='output')
    network.add_connection(d_to_output, source='d_intermediate', target='output')
   

    return network, input_layer, encoding_layer,output_layer

# 仿真和 SNN 集成主函数
def main():
    # 初始化 PyBullet 仿真
    urdf_path = "/workspace/src/universal_robot/ur_description/urdf/ur3.urdf"
    physics_client, robot1_id = initialize_robot(urdf_path, gui=True, base_position=(0, 0, 0))
    _, robot2_id = initialize_robot(urdf_path, gui=False, base_position=(1, 0, 0))
    movable_joints1 = get_movable_joints(robot1_id)
    movable_joints2 = get_movable_joints(robot2_id)

    # 初始化 SNN
    num_neurons = 3862+1
    kp, ki, kd = 3.75, 10/100, 0.5*100
    network, input_layer, encoding_layer, output_layer = initialize_snn(num_neurons, kp, ki, kd)

    # 仿真参数
    time_step = 0.01
    simulation_time = 30
    steps = int(simulation_time / time_step)

    # 数据存储
    times = []
    joint_angles1 = []
    joint_angles2 = []
    velocities1_data = []
    velocities2_data = []

    # PID 参数
    kp_pid, ki_pid, kd_pid = 3.75, 10, 0.5
    integral1 = np.zeros(len(movable_joints1))
    prev_error1 = np.zeros(len(movable_joints1))

    # 初始值
    current_angle = 0.0
    # target_angle =  45 * (np.pi / 180)
    target_angles = [45* (np.pi / 180), 0 * (np.pi / 180), -45 * (np.pi / 180)]  # 3个目标值
    current_target_index = 0  # 当前目标索引
    target_angle = target_angles[current_target_index]  # 初始目标值
    
    # 仿真循环
    for step in range(steps):

        current_time = step * time_step
        # 每10秒更新目标值
        if current_time >= (current_target_index + 1) * 10 and current_target_index < len(target_angles) - 1:
            current_target_index += 1
            target_angle = target_angles[current_target_index]
            print(f"Time {current_time:.2f}s: Updated target to {target_angle:.2f} rad")

        # 获取机器人状态
        q1, dq1 = get_joint_states(robot1_id, movable_joints1)
        q2, dq2 = get_joint_states(robot2_id, movable_joints2)
        joint_angles1.append(q1)
        joint_angles2.append(q2)

        # 计算 PID 控制
        velocities1 = []
        for i in range(len(movable_joints1)):
            velocity, integral1[i], prev_error1[i] = pid_controller(
                target_angle, q1[i], kp_pid, ki_pid, kd_pid, integral1[i], prev_error1[i], time_step
            )
            velocities1.append(velocity)


        encoding_layer.use_indices = True
        # 获取 InputLayer 的索引
        current_idx, target_idx = input_layer.last_indices

        # 将索引传递给 EncodingLayer
        encoding_layer.y_index = current_idx
        encoding_layer.r_index = target_idx

        # 更新 SNN 输入
        input_layer.update_input(q2[0], target_angle)
        network.run(inputs={'input': input_layer.s}, time=1)

        # 计算 SNN 控制
        output_spikes = output_layer.s
        snn_active_neuron_index = torch.argmax(output_spikes).item()
        snn_velocity = snn_active_neuron_index * (40 / (num_neurons - 1)) - 20
        velocities2 = [snn_velocity] * len(movable_joints2)

        # 应用速度控制
        apply_joint_velocities(robot1_id, movable_joints1, velocities1)
        apply_joint_velocities(robot2_id, movable_joints2, velocities2)

        # 存储速度数据
        velocities1_data.append(velocities1)
        velocities2_data.append(velocities2)

        # 仿真步进
        p.stepSimulation()
        times.append(step * time_step)

    # 仿真结束
    p.disconnect()

    # 转换数据为 NumPy 数组
    joint_angles1 = np.array(joint_angles1)
    joint_angles2 = np.array(joint_angles2)

    # 绘图：关节角度和速度
    plt.figure()
    for i in range(joint_angles1.shape[1]):
        plt.plot(times, joint_angles1[:, i], label=f"PID Joint {i} Angle")
        plt.plot(times, joint_angles2[:, i], linestyle='--', label=f"SNN Joint {i} Angle")
    plt.title("Joint Angles Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(times, velocities1_data, label="PID Velocities", linestyle='-')
    plt.plot(times, velocities2_data, label="SNN Velocities", linestyle='--')
    plt.title("Joint Velocities Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.legend()
    plt.grid()

    plt.show(block=True)


if __name__ == "__main__":
    main()
