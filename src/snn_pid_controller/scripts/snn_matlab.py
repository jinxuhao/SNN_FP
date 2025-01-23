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
from pathlib import Path

# 工具函数：创建权重矩阵
def create_weight_matrix(source_size, target_size, diagonal_value=1.0):
    weight_matrix = torch.zeros(target_size, source_size)
    for i in range(min(source_size, target_size)):
        weight_matrix[i, i] = diagonal_value
    return weight_matrix

# 定义质量-弹簧-阻尼系统
class MassSpringDamper:
    def __init__(self, mass=1.0, damping=2.0, stiffness=20, dt=0.01):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        self.dt = dt
        self.x = 0.0  # 初始角度（位移）
        self.v = 0.0  # 初始速度

    def step(self, force):
        acceleration = (force - self.damping * self.v - self.stiffness * self.x) / self.mass
        self.v += acceleration * self.dt
        self.x += self.v * self.dt
        return self.x

# 定义 PID 控制器
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, current_angle, target_angle, dt=0.001):
        error = target_angle - current_angle
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output
class NetworkManager:
    def __init__(self, file_path='network_state.pth'):
        self.file_path = Path(file_path)
        if self.file_path.exists():
            self.network = self.load_network_state()
        else:
            self.network = self.initialize_network()
            self.save_network_state()  # 初始化后保存状态

    # 初始化网络和组件
    def initialize_network(self):
        num_neurons = 1800 + 1
        network = Network()
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
        Kp, Ki, Kd = 1.72, 0.238, 2.654
        network.add_connection(Connection(source=input_layer, target=encoding_layer), source='input', target='encoding')
        network.add_connection(Connection(source=encoding_layer, target=integration_layer), source='encoding', target='integration')
        network.add_connection(Connection(source=encoding_layer, target=P_layer), source='encoding', target='p_intermediate')
        network.add_connection(Connection(source=integration_layer, target=I_layer), source='integration', target='i_intermediate')
        network.add_connection(Connection(source=encoding_layer, target=D_layer), source='encoding', target='d_intermediate')
        network.add_connection(Connection(source=P_layer, target=output_layer, w=create_weight_matrix(P_layer.n, output_layer.n, Kp)), source='p_intermediate', target='output')
        network.add_connection(Connection(source=I_layer, target=output_layer, w=create_weight_matrix(I_layer.n, output_layer.n, Ki)), source='i_intermediate', target='output')
        network.add_connection(Connection(source=D_layer, target=output_layer, w=create_weight_matrix(D_layer.n, output_layer.n, Kd)), source='d_intermediate', target='output')

        return network
    def load_network_state(self):
        if self.file_path.exists():
            return torch.load(self.file_path)
        else:
            return self.initialize_network()

    def save_network_state(self):
        torch.save(self.network, self.file_path)

def run_simulation_step(manager, current_angle, target_angle):
    network = manager.network
    input_layer = network.layers['input']
    encoding_layer = network.layers['encoding']
    output_layer = network.layers['output']

    # 更新输入
    input_layer.update_input(current_angle, target_angle)
    input_data = input_layer.s.clone().unsqueeze(0).unsqueeze(0)

    # 手动传递输入至编码层，并处理编码层的逻辑
    encoding_layer.use_indices = True  # 确保使用索引方式处理（如果适用）
    current_idx, target_idx = input_layer.last_indices  # 假设有这样的属性记录最后的索引

    # 将索引传递给 EncodingLayer
    encoding_layer.y_index = current_idx
    encoding_layer.r_index = target_idx

    # 运行网络
    network.run(inputs={'input': input_data}, time=1)

    # 获取输出
    output_spikes = output_layer.s
    snn_output_value = torch.argmax(output_spikes).item() * (80 / (1800)) - 40
    print(f"Called Value: {snn_output_value}")

    return snn_output_value


    # result = current_angle + target_angle  # 仅为示例
    # return result

# 保存模块，以便MATLAB调用
if __name__ == "__main__":
    print("This module is ready to be imported and called from MATLAB.")
    
    # 设定运行次数
    number_of_runs = 10
    manager = NetworkManager() 

    # 运行模拟多次
    for _ in range(number_of_runs):
        output = run_simulation_step(manager,0, 15)
        print(f"Simulation output: {output}")
