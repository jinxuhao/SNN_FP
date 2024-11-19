import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input
from bindsnet.network.monitors import Monitor

# 创建 SNN 网络实例
network = Network()

# 添加符合文章描述的输入层：63个神经元
num_neurons = 63
input_layer = Input(num_neurons)
network.add_layer(input_layer, name='I')

# 添加监视器，以观察输入层的活动
monitor = Monitor(network.layers['I'], state_vars=['s'])
network.add_monitor(monitor, 'I')

# 定义输入范围
angle_range = (-40, 40)
setpoint_range = (-25, 25)
angular_velocity_range = (-80, 80)

# 定义文章中的编码函数，将输入值映射到神经元索引
def encode_index(value, min_val, max_val, num_neurons):
    idx = int((value - min_val) / (max_val - min_val) * (num_neurons - 1))
    return max(0, min(idx, num_neurons - 1))

# 生成输入数据
def generate_input_data():
    current_angle = torch.tensor([encode_index(torch.randint(angle_range[0], angle_range[1], (1,)).item(),
                                               angle_range[0], angle_range[1], num_neurons)])
    target_angle = torch.tensor([encode_index(torch.randint(setpoint_range[0], setpoint_range[1], (1,)).item(),
                                              setpoint_range[0], setpoint_range[1], num_neurons)])
    angular_velocity = torch.tensor([encode_index(torch.randint(angular_velocity_range[0], angular_velocity_range[1], (1,)).item(),
                                                  angular_velocity_range[0], angular_velocity_range[1], num_neurons)])
    input_data = torch.zeros((1, num_neurons))
    input_data[0, current_angle] = 1
    input_data[0, target_angle] = 1
    input_data[0, angular_velocity] = 1
    return input_data

# 生成一次输入数据
input_data = generate_input_data().unsqueeze(0)  # 确保形状为 (1, 1, 63)
print(f"Generated input data shape for network: {input_data.shape}")

# 检查input_data shape是否符合预期
if input_data.shape != (1, 1, 63):
    print(f"Warning: Expected input data shape of (1, 1, 63) but got {input_data.shape}")

# 运行网络，使用生成的输入数据
try:
    network.run(inputs={'I': input_data}, time=1)
    spikes = monitor.get('s')  # 获取输入层的脉冲活动
    print(f"Spiking activity in input layer: {spikes}")
except RuntimeError as e:
    print("RuntimeError encountered:", e)
