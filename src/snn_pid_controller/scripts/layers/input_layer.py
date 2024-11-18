import numpy as np

class InputLayer:
    def __init__(self, num_neurons, angle_range=(-40, 40), setpoint_range=(-25, 25), angular_velocity_range=(-80, 80)):
        self.num_neurons = num_neurons
        self.angle_range = angle_range
        self.setpoint_range = setpoint_range
        self.angular_velocity_range = angular_velocity_range

    def generate_virtual_input(self):
        """生成当前角度和目标角度的虚拟输入"""
        current_angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        target_angle = np.random.uniform(self.setpoint_range[0], self.setpoint_range[1])
        return current_angle, target_angle

    def encode_index(self, value, min_val, max_val):
        """根据输入范围和神经元数量进行索引编码"""
        idx = int((value - min_val) / (max_val - min_val) * (self.num_neurons - 1))
        return max(0, min(idx, self.num_neurons - 1))
