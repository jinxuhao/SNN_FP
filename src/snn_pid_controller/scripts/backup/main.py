import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from bindsnet.network import Network
from bindsnet.network.topology import Connection,MulticompartmentConnection
from bindsnet.network.monitors import Monitor
# from bindsnet.network.nodes import Input 

# from layers.input import SNNInput
from layers.input_layer import InputLayer
from layers.encoding_layer import EncodingLayer
from layers.integration_layer import IntegrationLayer
from layers.output_layer import ComplexOutputLayer
from layers.P_layer import PIntermediateLayer
from layers.I_layer import IIntermediateLayer
from layers.D_layer import DIntermediateLayer
from layers.CustomNodes import CustomNodes

from connections.identity_connection import IdentityConnection
from utils.create_weight_matrix import create_weight_matrix
import matplotlib.pyplot as plt
# 创建 SNN 网络实例
network = Network()

# 初始化各层
num_neurons = 63
# input_layer = Input(num_neurons) 
input_layer = InputLayer(num_neurons=num_neurons)
encoding_layer = EncodingLayer(num_neurons=num_neurons)
integration_layer = IntegrationLayer(num_neurons=num_neurons)
output_layer = ComplexOutputLayer(num_neurons=num_neurons)
# output_layer = CustomNodes(num_neurons=num_neurons*3)

P_layer = PIntermediateLayer(num_neurons=num_neurons)
I_layer = IIntermediateLayer(num_neurons=num_neurons)
D_layer = DIntermediateLayer(num_neurons=num_neurons)

# 将各层添加到网络中
network.add_layer(input_layer, name='input')
network.add_layer(encoding_layer, name='encoding')
network.add_layer(integration_layer, name='integration')
network.add_layer(output_layer, name='output')
network.add_layer(P_layer, name='p_intermediate')
network.add_layer(I_layer, name='i_intermediate')
network.add_layer(D_layer, name='d_intermediate')

# 设置层之间的连接
# input_to_encoding = Connection(source=network.layers['input'], target=encoding_layer)
# input_to_encoding = Connection(source=input_layer, target=encoding_layer)
identity_weights_input_to_encoding = torch.eye(encoding_layer.n, input_layer.n)
identity_weights_encoding_to_integration = torch.eye(integration_layer.n, encoding_layer.n)
identity_weights_integration_to_output = torch.eye(output_layer.n, integration_layer.n)
identity_weights_encoding_to_output = torch.eye(output_layer.n, encoding_layer.n)

Kp = 1
Ki = 0.015
Kd = 0.2
identity_weights_p_to_output = create_weight_matrix(P_layer.n, output_layer.n) *Kp
identity_weights_i_to_output = create_weight_matrix(I_layer.n, output_layer.n) *Ki
identity_weights_d_to_output = create_weight_matrix(D_layer.n, output_layer.n) *Kd

input_to_encoding = Connection(source=input_layer, target=encoding_layer, w=identity_weights_input_to_encoding, 
    requires_grad=False)
# input_to_encoding = IdentityConnection(source=input_layer, target=encoding_layer)
encoding_to_integration = Connection(source=encoding_layer, target=integration_layer,w=identity_weights_encoding_to_integration, 
    requires_grad=False)
encoding_to_p = Connection(source=encoding_layer, target=P_layer, w=torch.eye(P_layer.n, encoding_layer.n), 
    requires_grad=False)
integration_to_i = Connection(source=integration_layer, target=I_layer, w=torch.eye(I_layer.n, integration_layer.n), 
    requires_grad=False)
encoding_to_d = Connection(source=encoding_layer, target=D_layer, w=torch.eye(D_layer.n, integration_layer.n), 
    requires_grad=False)
p_to_output = Connection(source=P_layer, target=output_layer, w=identity_weights_p_to_output, 
    requires_grad=False)
i_to_output = Connection(source=I_layer, target=output_layer, w=identity_weights_i_to_output, 
    requires_grad=False)
d_to_output = Connection(source=D_layer, target=output_layer, w=identity_weights_d_to_output, 
    requires_grad=False)




network.add_connection(input_to_encoding, source='input', target='encoding')
network.add_connection(encoding_to_integration, source='encoding', target='integration')
# network.add_connection(integration_to_output, source='integration', target='output')
network.add_connection(encoding_to_p, source='encoding', target='p_intermediate')
network.add_connection(integration_to_i, source='integration', target='i_intermediate')
network.add_connection(encoding_to_d, source='encoding', target='d_intermediate')
network.add_connection(p_to_output, source='p_intermediate', target='output')
network.add_connection(i_to_output, source='i_intermediate', target='output')
network.add_connection(d_to_output, source='d_intermediate', target='output')

print(f"Connections in network: {network.connections}")

# 创建 Monitor 以观察网络的行为
monitor = Monitor(network.layers['input'], state_vars=['s'], time=100)
network.add_monitor(monitor, name='input_monitor')
layers_to_monitor = {
    'input': input_layer,
    'encoding': encoding_layer,
    'integration': integration_layer,
    'p_intermediate': P_layer,
    'i_intermediate': I_layer,
    'd_intermediate': D_layer,
    'output': output_layer
}
monitors = {}

for name, layer in layers_to_monitor.items():
    monitors[name] = Monitor(layer, state_vars=['s'], time=100)
    network.add_monitor(monitors[name], name=f'{name}_monitor')


# input_layer

# current_angle, target_angle = input_layer.generate_virtual_input()
# input_layer.update_input(current_angle, target_angle)
# input_data = input_layer.s.clone() # 获取当前的状态变量

input_layer.use_explicit_inputs = True

input_layer.explicit_current = -20  # 显式设置 current_idx
input_layer.explicit_target = 30  # 显式设置 target_idx
current_angle, target_angle = input_layer.explicit_current,input_layer.explicit_target 
current_idx, target_idx = input_layer.last_indices
input_layer.update_input()
input_data = input_layer.s.clone()

encoding_layer.use_indices = True
# 获取 InputLayer 的索引
current_idx, target_idx = input_layer.last_indices

# 将索引传递给 EncodingLayer
encoding_layer.y_index = current_idx
encoding_layer.r_index = target_idx
try:
    # 仿真参数
    num_steps = 30  # 仿真的时间步数
    time_per_step = 1  # 每步的仿真时间
    neurons = input_data.shape[1]  # 输入神经元数量

    # 循环仿真
    for step in range(num_steps):
        print(f"Simulation step {step + 1}/{num_steps}")
        
        # 更新输入数据，可以根据需要生成不同的输入

        # input_data = input_data.repeat(1, 1)  # 重复以匹配网络预期的输入格式

        print(f"Input data for step {step + 1}: {input_data.shape}")


        # input_layer.use_explicit_inputs = False
        # 显式模式，传递 y_index 和 r_index
        encoding_layer.use_indices = True
        # # 获取 InputLayer 的索引
        current_idx, target_idx = input_layer.last_indices

        # 将索引传递给 EncodingLayer
        encoding_layer.y_index = current_idx
        encoding_layer.r_index = target_idx

        # 仿真网络
        network.run(inputs={'input': input_data}, time=time_per_step)

        input_layer.use_explicit_inputs = False
        # 获取 InputLayer 的索引
        current_idx, target_idx = input_layer.last_indices

        # 将索引传递给 EncodingLayer
        encoding_layer.y_index = current_idx
        encoding_layer.r_index = target_idx



        # 检查输入层或其他层的状态
        print(f"Input layer state at step {step + 1}: {input_layer.s.shape}, Data type: {input_layer.s.dtype}")
        output_spikes = output_layer.s  # 获取 output 层的尖峰活动
        active_neuron_index = torch.argmax(output_spikes).item()  # 获取最活跃神经元的索引
        print(f"Output active neuron index at step {step + 1}: {active_neuron_index}")
        # 根据输出结果更新 y (current_angle)
        output_value = active_neuron_index * (10 / num_neurons) - 5  # 假设范围是 [-5, 5]
        current_angle += output_value  # 更新 current_angle

        input_layer.update_input(current_angle, target_angle)  # 保持 target_angle 不变
        input_data = input_layer.s.clone() 
        print(f"Updated current_angle: {current_angle}, target_angle: {target_angle}")

except RuntimeError as e:
    print("RuntimeError encountered:", e)

for name, monitor in monitors.items():
    recorded_spikes = [item for item in monitor.recording['s'] if isinstance(item, torch.Tensor) and item.numel() > 0]
    if recorded_spikes:
        spikes = torch.cat(recorded_spikes, dim=0)
        # print(f"Spiking activity in {name} layer:")
        # print(spikes)
    else:
        print(f"No spiking activity recorded in {name} layer.")



def plot_spiking_activity(monitors):
    """
    在一个窗口中绘制所有层的尖峰活动，并调整颜色和线条样式。
    """
    num_layers = len(monitors)  # 层数
    cols = 2  # 每行的子图数量
    rows = (num_layers + cols - 1) // cols  # 行数，向上取整
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # 将子图数组展开为一维列表
    
    for idx, (name, monitor) in enumerate(monitors.items()):
        # 提取有效的尖峰数据
        recorded_spikes = [item for item in monitor.recording['s'] if isinstance(item, torch.Tensor) and item.numel() > 0]
        if recorded_spikes:
            spikes = torch.cat(recorded_spikes, dim=0).squeeze(1)  # 拼接时间步的记录并去掉多余维度
            
            # 创建子图
            ax = axes[idx]
            im = ax.imshow(
                spikes.numpy().T,
                aspect='auto',
                cmap='Reds',  # 使用红色渐变
                interpolation='nearest'
            )
            
            # 添加颜色条和图例
            fig.colorbar(im, ax=ax, orientation='vertical', label="Spiking Activity")
            ax.set_title(f"{name} Layer Spiking Activity")
            ax.set_xlabel("Simulation Time (Steps)")
            ax.set_ylabel("Neuron Index")
        else:
            # 如果没有记录尖峰数据，则隐藏子图
            axes[idx].axis('off')
            axes[idx].set_title(f"No Spiking Activity in {name} Layer")
    
    # 隐藏多余的子图框
    for ax in axes[len(monitors):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show(block=True)


plot_spiking_activity(monitors)