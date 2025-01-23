#!/usr/bin/env python3
import torch
import rospy
from std_msgs.msg import Float64
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

# 初始化网络
network = Network()

# 初始化各层
num_neurons = 1801  # 多个神经元
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
Kp, Ki, Kd = 1.72, 0.238, 2.654  # 控制器增益
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

def current_value_callback(msg):
    global current_angle
    current_angle = msg.data
    rospy.loginfo('Received current angle: %f', current_angle)

# ROS初始化和发布器设置
rospy.init_node('snn_output_node', anonymous=True)
pub = rospy.Publisher('snn_output', Float64, queue_size=1)
sub = rospy.Subscriber('/current_value', Float64, current_value_callback)

# 运行网络并发布输出
rate = rospy.Rate(100)  # 10 Hz
current_angle = 0
while not rospy.is_shutdown():
    # 假设已有函数获取当前角度和目标角度
    # current_angle = 0
    target_angle = 15

    # 更新网络输入
    input_layer.update_input(current_angle, target_angle)
    input_data = input_layer.s.clone().unsqueeze(0).repeat(1, 1, 1)
    network.run(inputs={'input': input_data}, time=1)

    # 获取输出层的活动并计算输出值
    output_spikes = output_layer.s
    snn_output_value = torch.argmax(output_spikes).item() * (80 / (num_neurons - 1)) - 40

    # 发布SNN控制器的输出值
    rospy.loginfo('Publishing SNN output: %f', snn_output_value)
    pub.publish(Float64(snn_output_value))

    rate.sleep()