import torch
import numpy as np
from bindsnet.network.nodes import Nodes

class InputLayer(Nodes):
    def __init__(self, num_neurons, angle_range=(-40, 40), setpoint_range=(-40, 40), angular_velocity_range=(-80, 80)):
        super(InputLayer, self).__init__(n=num_neurons, shape=(1, num_neurons))
        self.num_neurons = num_neurons
        self.angle_range = angle_range
        self.setpoint_range = setpoint_range
        self.s = torch.zeros(1, self.num_neurons)  # 初始化 s 为 (1, num_neurons)

        # 标志和显式输入索引
        self.use_explicit_inputs = False
        self.explicit_current = None
        self.explicit_target = None


        self.last_indices = (None, None)  # 用于存储 current_idx 和 target_idx


    def encode_index(self, value, min_val, max_val):
        """根据输入范围和神经元数量进行索引编码"""
        idx = int((value - min_val) / (max_val - min_val) * (self.num_neurons - 1))
        return max(0, min(idx, self.num_neurons - 1))

    def update_input(self, current_angle=None, target_angle=None):
        if self.use_explicit_inputs and self.explicit_current is not None and self.explicit_target is not None:
            current_idx = self.encode_index(self.explicit_current, *self.angle_range)
            target_idx = self.encode_index(self.explicit_target, *self.setpoint_range)
        else:
            current_idx = self.encode_index(current_angle, *self.angle_range)
            target_idx = self.encode_index(target_angle, *self.setpoint_range)
        self.s = torch.zeros(1, self.num_neurons)  # 确保是 (1, num_neurons)
        self.s[0, current_idx] = 1
        self.s[0, target_idx] = 1
        self.last_indices = (current_idx, target_idx)  # 保存索引
        # self.s = (self.s > 0).bool()
        print("INPUT____Updated input layer s shape:", self.s.shape)  # Debug print
        print(f"INPUT____current_angle: {current_angle}, mapped index: {current_idx}")
        print(f"INPUT____target_angle: {target_angle}, mapped index: {target_idx}")
        return current_idx, target_idx
    
    def forward(self, x, input_1_value=None, input_2_value=None, *args, **kwargs):

        # # Generate virtual input
        # current_angle, target_angle = self.generate_virtual_input()
        # print(f"DEBUG___X{x}")
        
        # # # Update the state
        # self.update_input(current_angle, target_angle)
        # self.s.view(self.s.size(0), -1)
        # # print(f"DEBUG___Updated state self.s: {self.s}")
        # print("INPUT___input_layer.forward() called",self.s)
        # self.s = (self.s > 0).bool()
        # print(f"INPUT___Shape of s (source spikes): {self.s.shape}, Data type: {self.s.dtype}")
        # print(f"DEBUG___Converted self.s to bool: {self.s}")
    # def forward(self, x, current_angle=None, target_angle=None, *args, **kwargs):
        # if current_angle is not None and target_angle is not None:
        #     self.update_input(current_angle, target_angle)
        return self.s



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
