import torch
import numpy as np
from bindsnet.network.nodes import Nodes

class InputLayer(Nodes):
    def __init__(self, num_neurons, angle_range=(-40, 40), setpoint_range=(-25, 25), angular_velocity_range=(-80, 80)):
        super(InputLayer, self).__init__(n=num_neurons, shape=(1, num_neurons))
        self.num_neurons = num_neurons
        self.angle_range = angle_range
        self.setpoint_range = setpoint_range
        self.s = torch.zeros(1, self.num_neurons)  # 初始化 s 为 (1, num_neurons)

    def encode_index(self, value, min_val, max_val):
        """根据输入范围和神经元数量进行索引编码"""
        idx = int((value - min_val) / (max_val - min_val) * (self.num_neurons - 1))
        return max(0, min(idx, self.num_neurons - 1))

    def update_input(self, current_angle, target_angle):
        current_idx = self.encode_index(current_angle, *self.angle_range)
        target_idx = self.encode_index(target_angle, *self.setpoint_range)
        self.s = torch.zeros(1, self.num_neurons)  # 确保是 (1, num_neurons)
        self.s[0, current_idx] = 1
        self.s[0, target_idx] = 1
        print("Updated input layer s shape:", self.s.shape)  # Debug print
        print(f"current_angle: {current_angle}, mapped index: {current_idx}")
        print(f"target_angle: {target_angle}, mapped index: {target_idx}")

    def forward(self , x=None ,input_1_value=None, input_2_value=None, *args, **kwargs):

        return self.s  # 或者在需要的地方使用 self.s 更新


    def generate_virtual_input(self):
        """生成并返回当前角度和目标角度的虚拟输入值"""
        current_angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        target_angle = np.random.uniform(self.setpoint_range[0], self.setpoint_range[1])
        return current_angle, target_angle


    def compute_decays(self, dt):
            """覆盖 compute_decays 方法"""
            pass  # 不执行任何操作
    
    def set_batch_size(self, batch_size):
        """覆盖 set_batch_size 方法"""
        pass  # 不执行任何操作
