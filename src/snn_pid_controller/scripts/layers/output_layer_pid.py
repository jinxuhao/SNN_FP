import torch
from bindsnet.network.nodes import DiehlAndCookNodes

class OutputLayer:
    def __init__(self, num_neurons=63, p_gain=0.1, i_gain=0.05):
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.output_neurons = DiehlAndCookNodes(n=num_neurons)
        self.output_neurons.v = torch.zeros(num_neurons)

    def compute_new_angle(self, current_angle, target_angle, integral_signal):
        # 计算误差信号
        proportional_signal = target_angle - current_angle
        # 计算控制信号
        control_signal = self.p_gain * proportional_signal + self.i_gain * integral_signal
        # 返回更新后的角度
        new_angle = current_angle + control_signal
        print(f"Control signal: {control_signal}, Updated angle: {new_angle}")
        
        return new_angle
