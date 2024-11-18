import torch
from bindsnet.network.nodes import DiehlAndCookNodes

class OutputLayer:
    def __init__(self, num_neurons=63, spike_threshold=1, base_signal=5, decay_factor=0.98):
        self.spike_threshold = spike_threshold
        self.base_signal = base_signal  # 信号增益
        self.decay_factor = decay_factor  # 电位衰减因子
        self.output_neurons = DiehlAndCookNodes(n=num_neurons)
        self.output_neurons.v = torch.zeros(num_neurons)
        
    def compute_output_from_activity(self, proportional_signal, integral_signal, target_angle, current_angle):
        # 计算误差
        error = target_angle - current_angle
        direction = 1 if error > 0 else -1  # 方向根据误差决定

        # 增加信号影响，根据误差大小动态缩放
        gain = min(abs(error), 10)  # 确保 gain 不过大
        proportional_index = min(max(int(proportional_signal * gain), 0), self.output_neurons.n - 1)
        integral_index = min(max(int(integral_signal * gain), 0), self.output_neurons.n - 1)

        # 更新神经元的电位
        self.output_neurons.v[proportional_index] += self.base_signal
        self.output_neurons.v[integral_index] += self.base_signal

        # 计算控制信号
        spike_counts = (self.output_neurons.v >= self.spike_threshold).sum().item()
        control_signal = spike_counts * 0.1 * direction  # 根据误差方向调整

        print(f"Control signal (from spikes): {control_signal}, Error: {error}")

        # 电位缓慢衰减
        self.output_neurons.v *= self.decay_factor
        return control_signal
