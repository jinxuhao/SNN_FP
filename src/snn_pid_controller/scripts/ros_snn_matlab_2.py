#!/usr/bin/env python3
import torch
from bindsnet.network import Network
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from layers.input_layer import InputLayer
from layers.encoding_layer import EncodingLayer
from layers.integration_layer import IntegrationLayer
from layers.output_layer import ComplexOutputLayer
from layers.P_layer import PIntermediateLayer
from layers.I_layer import IIntermediateLayer
from layers.D_layer import DIntermediateLayer
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Header

# 工具函数：创建权重矩阵
def create_weight_matrix(source_size, target_size, diagonal_value=1.0):
    weight_matrix = torch.zeros(target_size, source_size)
    for i in range(min(source_size, target_size)):
        weight_matrix[i, i] = diagonal_value
    return weight_matrix



# 初始化网络
network = Network()

# 初始化各层
num_neurons = 6800+1#63*3##63*49
input_layer = InputLayer(num_neurons=num_neurons)
encoding_layer = EncodingLayer(num_neurons=num_neurons)
integration_layer = IntegrationLayer(num_neurons=num_neurons)
output_layer = ComplexOutputLayer(num_neurons=num_neurons)
P_layer = PIntermediateLayer(num_neurons=num_neurons)
I_layer = IIntermediateLayer(num_neurons=num_neurons)
D_layer = DIntermediateLayer(num_neurons=num_neurons)

# 添加层到网络
network.add_layer(input_layer, name='input')
network.add_layer(encoding_layer, name='encoding')
network.add_layer(integration_layer, name='integration')
network.add_layer(output_layer, name='output')
network.add_layer(P_layer, name='p_intermediate')
network.add_layer(I_layer, name='i_intermediate')
network.add_layer(D_layer, name='d_intermediate')

# 创建连接
Kp, Ki, Kd = 1.72, 0.238/10, 0.2*10  #1.72, 0.238, 0 #PI 0.5514, 0.0467, 2.654
input_to_encoding = Connection(source=input_layer, target=encoding_layer, w=torch.eye(encoding_layer.n, input_layer.n), requires_grad=False)
encoding_to_integration = Connection(source=encoding_layer, target=integration_layer, w=torch.eye(integration_layer.n, encoding_layer.n), requires_grad=False)
encoding_to_p = Connection(source=encoding_layer, target=P_layer, w=torch.eye(P_layer.n, encoding_layer.n), requires_grad=False)
integration_to_i = Connection(source=integration_layer, target=I_layer, w=torch.eye(I_layer.n, integration_layer.n), requires_grad=False)
encoding_to_d = Connection(source=encoding_layer, target=D_layer, w=torch.eye(D_layer.n, encoding_layer.n), requires_grad=False)
p_to_output = Connection(source=P_layer, target=output_layer, w=create_weight_matrix(P_layer.n, output_layer.n, Kp), requires_grad=False)
i_to_output = Connection(source=I_layer, target=output_layer, w=create_weight_matrix(I_layer.n, output_layer.n, Ki), requires_grad=False)
d_to_output = Connection(source=D_layer, target=output_layer, w=create_weight_matrix(D_layer.n, output_layer.n, Kd), requires_grad=False)

# 添加连接到网络
network.add_connection(input_to_encoding, source='input', target='encoding')
network.add_connection(encoding_to_integration, source='encoding', target='integration')
network.add_connection(encoding_to_p, source='encoding', target='p_intermediate')
network.add_connection(integration_to_i, source='integration', target='i_intermediate')
network.add_connection(encoding_to_d, source='encoding', target='d_intermediate')
network.add_connection(p_to_output, source='p_intermediate', target='output')
network.add_connection(i_to_output, source='i_intermediate', target='output')
network.add_connection(d_to_output, source='d_intermediate', target='output')

# 初始化监视器
monitors = {}
layers_to_monitor = {
    'input': input_layer,
    'encoding': encoding_layer,
    'integration': integration_layer,
    'p_intermediate': P_layer,
    'i_intermediate': I_layer,
    'd_intermediate': D_layer,
    'output': output_layer
}
# for name, layer in layers_to_monitor.items():
#     monitors[name] = Monitor(layer, state_vars=['s'], time=num_steps * time_per_step)
#     network.add_monitor(monitors[name], name=f'{name}_monitor')

def current_value_callback(msg):
    global current_angle
    current_angle = msg.data
    rospy.loginfo('Received current angle: %f', current_angle)

# ROS初始化和发布器设置
rospy.init_node('snn_output_node', anonymous=True)

# 运行网络并发布输出
rate = rospy.Rate(1000)

pub = rospy.Publisher('snn_output', Float64, queue_size=1)
sub = rospy.Subscriber('/current_value', Float64, current_value_callback)
# 初始化 ROS 时间戳发布器
time_pub = rospy.Publisher('/snn_timestamp', Float64, queue_size=100)



# 初始化输入
current_angle = 0  # 初始角度
target_angle = 5   # 目标角度
input_layer.update_input(current_angle, target_angle)
input_data = input_layer.s.clone()


# 仿真参数
num_steps = 3000
time_per_step = 1
dt = 0.1#time_per_step/1000
snn_angles = [current_angle]

input_data = input_layer.s.clone().unsqueeze(0).repeat(time_per_step, 1, 1)
for name, layer in layers_to_monitor.items():
    monitors[name] = Monitor(layer, state_vars=['s'], time=num_steps*time_per_step)
    network.add_monitor(monitors[name], name=f'{name}_monitor')

try:
    for step in range(num_steps):
        print(f"Simulation step {step + 1}/{num_steps}")



        # SNN 控制器运行
        encoding_layer.use_indices = True
        # 获取 InputLayer 的索引
        current_idx, target_idx = input_layer.last_indices

        # 将索引传递给 EncodingLayer
        encoding_layer.y_index = current_idx
        encoding_layer.r_index = target_idx
        network.run(inputs={'input': input_data}, time=time_per_step)

        # 获取 SNN 输出
        output_spikes = output_layer.s
        snn_active_neuron_index = torch.argmax(output_spikes).item()
        snn_output_value = snn_active_neuron_index * (160/ (num_neurons-1)) - 80
        print(f"called Value: {snn_active_neuron_index},  snn_output_value: {snn_output_value}")
        pub.publish(Float64(snn_output_value))
        

        # 使用动态模型更新 SNN 的角度
        current_angle = current_angle
        snn_angles.append(current_angle)


        # 更新输入
        print("TEST INPUT main:", current_angle)
        print("TEST INPUT v:", P_layer.positive)
        input_layer.update_input(current_angle, target_angle)
        # input_data = input_layer.s.clone()
        input_data = input_layer.s.clone().unsqueeze(0).repeat(time_per_step, 1, 1)
        print("TEST INPUT main:", input_data)
        
except RuntimeError as e:
    print("RuntimeError encountered:", e)


# 绘制尖峰活动
def plot_spiking_activity(monitors):
    num_layers = len(monitors)
    cols = 2
    rows = (num_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    for idx, (name, monitor) in enumerate(monitors.items()):
        recorded_spikes = [item for item in monitor.recording['s'] if isinstance(item, torch.Tensor) and item.numel() > 0]
        if recorded_spikes:
            spikes = torch.cat(recorded_spikes, dim=0).squeeze(1)
            ax = axes[idx]
            im = ax.imshow(spikes.numpy().T, aspect='auto', cmap='Reds', interpolation='nearest')
            fig.colorbar(im, ax=ax, orientation='vertical', label="Spiking Activity")
            ax.set_title(f"{name} Layer Spiking Activity")
            ax.set_xlabel("Simulation Time (Steps)")
            ax.set_ylabel("Neuron Index")
        else:
            axes[idx].axis('off')
            axes[idx].set_title(f"No Spiking Activity in {name} Layer")
    for ax in axes[len(monitors):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show(block=True)



# 加载 .mat 文件
data = loadmat('src/snn_pid_controller/scripts/matlab/simulation_results_simple001.mat/simulation_results_simple001.mat')
data01 = loadmat('src/snn_pid_controller/scripts/matlab/simulation_results_simple01.mat/simulation_results_simple01.mat')

time = data['time'].flatten()
values = data['data'].flatten()
time01 = data01['time'].flatten()
values01 = data01['data'].flatten()
# MATLAB 的时间转换为迭代次数
matlab_steps = len(time)  # MATLAB 时间步数
matlab_iterations = [int(t / 0.01) for t in time]  # 转换为迭代次数
matlab_iterations01  = [int(t / 0.01) for t in time01]

snn_x = [i * 10 for i in range(num_steps + 1)]

# 绘制对比曲线
plt.figure(figsize=(10, 6))
plt.plot(snn_x, snn_angles, label='SNN Controller', linestyle='-', marker='o')


plt.plot(matlab_iterations, values, label='matlab Controller', linestyle='-', marker='x')
plt.plot(matlab_iterations01, values01, label='matlab 01 Controller', linestyle='-', marker='o')

plt.xlabel('Simulation Step')
plt.ylabel('Current Angle')
plt.title('Comparison of SNN and PID Controllers with Mass-Spring-Damper System')
plt.legend()
plt.grid(True)

plt.show(block=True)

# 绘制尖峰活动
plot_spiking_activity(monitors)
print(f"Sampling time (dt): {network.dt} ms")