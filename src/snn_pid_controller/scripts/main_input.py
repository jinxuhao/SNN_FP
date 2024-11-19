import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from bindsnet.network import Network
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input 

from layers.input import SNNInput
from layers.input_layer import InputLayer
from layers.encoding_layer import EncodingLayer
from layers.integration_layer import IntegrationLayer
from layers.output_layer import ComplexOutputLayer

# 创建 SNN 网络实例
network = Network()

# 初始化各层
num_neurons = 63
snn_input = SNNInput(num_neurons=num_neurons)
input_layer = Input(num_neurons) 
# input_layer = InputLayer(num_neurons=num_neurons)
encoding_layer = EncodingLayer(num_neurons=num_neurons)
integration_layer = IntegrationLayer(num_neurons=num_neurons)
output_layer = ComplexOutputLayer(num_neurons=num_neurons)

# 将各层添加到网络中
network.add_layer(input_layer, name='input')
# network.add_layer(input_layer, name='input')
network.add_layer(encoding_layer, name='encoding')
network.add_layer(integration_layer, name='integration')
network.add_layer(output_layer, name='output')

# 设置层之间的连接
input_to_encoding = Connection(source=network.layers['input'], target=encoding_layer)
# input_to_encoding = Connection(source=input_layer, target=encoding_layer)
encoding_to_integration = Connection(source=encoding_layer, target=integration_layer)
integration_to_output = Connection(source=integration_layer, target=output_layer)

network.add_connection(input_to_encoding, source='input', target='encoding')
network.add_connection(encoding_to_integration, source='encoding', target='integration')
network.add_connection(integration_to_output, source='integration', target='output')


# 创建 Monitor 以观察网络的行为
monitor = Monitor(network.layers['input'], state_vars=['s'], time=100)
network.add_monitor(monitor, name='input_monitor')


# # 生成初始输入数据
# # Generate input data as tensor
# current_angle, target_angle = input_layer.generate_virtual_input()
# input_data = {'input': input_layer.forward()}
# print(f"Input data shape for network: {input_data['input'].shape}")  # Verify shape


# # 生成虚拟输入
# current_angle, target_angle = input_layer.generate_virtual_input()
# input_layer.update_input(current_angle, target_angle)
# input_data = {'input': input_layer.s}

# print("Input data shape for network:", input_data['input'].shape)  # Should be (1, 63)
# # 使用输入层进行编码
# encoded_input = input_layer.forward()

# # 检查 encoded_input 的形状
# print("Encoded input shape:", encoded_input.shape)

# # 准备输入数据，确保形状为 (1, 63)
# input_data = {'input': encoded_input.view(1, -1)}
# print("Input data shape for network:", input_data['input'].shape)

# # 确保 input_layer 的 `s` 属性的形状符合期望
# print("Shape of source.s in input_to_encoding:", input_layer.s.shape)
# 生成随机输入数据（确保为张量，而不是嵌套字典）
# Prepare inputs without using 'x' in forward()
# current_angle, target_angle = input_layer.generate_virtual_input()
# input_layer.update_input(current_angle, target_angle)

# input_data = input_layer.forward()
# input_data = input_data.view(1, -1)  # 将 input_data 调整为 (1, 63)

# # 确保 input_data 的形状符合预期
# print(f"Input data shape for network: {input_data.shape}")
input_data = snn_input.generate_spike_input()

# Simulate network
print("Input data shape for network:", input_data.shape)
# 运行网络
try:
    # Simulate network
    network.run(inputs={'input': input_data}, time=1)
except RuntimeError as e:
    print("RuntimeError encountered:", e)

# 在运行网络仿真之后，检查并获取 Monitor 的尖峰数据
try:
    # 过滤掉空列表，只保留非空张量
    recorded_spikes = [item for item in monitor.recording['s'] if isinstance(item, torch.Tensor) and item.numel() > 0]
    
    if recorded_spikes:
        spikes = torch.cat(recorded_spikes, dim=0)
        print("Spiking activity in input layer:")
        print(spikes)
    else:
        print("No spiking activity recorded in input layer.")
except TypeError as e:
    print("TypeError encountered while processing spikes:", e)

# # 仿真步数
# num_steps = 100

# # 仿真循环
# for step in range(num_steps):
#     # 运行网络并传入当前输入数据
#     if input_data['input'] is not None :
#         network.run(inputs=input_data, time=1)
#     else:
#         print("Warning: Input data is None or has incorrect shape. Skipping this step.")
#         continue

#     # 从输出层获取结果
#     output_result = output_layer.compute_output()  # 使用 compute_output() 提取当前输出
#     print(f"Step {step}, Output: {output_result}")
    
#     # 更新输入（根据需求，可以设置动态输入）
#     current_angle, target_angle = input_layer.generate_virtual_input()
#     input_layer.set_angles(current_angle, target_angle)
#     input_data = {'input': input_layer.forward()}

# # 停止网络运行
# network.reset_()  # 使用 reset_() 来重置网络，而不是 stop()
