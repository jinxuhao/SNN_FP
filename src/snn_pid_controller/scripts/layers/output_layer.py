import torch
from bindsnet.network.nodes import DiehlAndCookNodes

class ComplexOutputLayer:
    def __init__(self, num_neurons=63, Kp=0.1, Ki=0.05, Kd=0.01):
        # 创建 P+/P-, I+/I- 群
        self.P_plus = DiehlAndCookNodes(n=num_neurons)
        self.P_plus.v = torch.zeros(num_neurons)  # 初始化 P+ 群电位
        self.P_minus = DiehlAndCookNodes(n=num_neurons)
        self.P_minus.v = torch.zeros(num_neurons)  # 初始化 P- 群电位
        self.I_plus = DiehlAndCookNodes(n=num_neurons)
        self.I_plus.v = torch.zeros(num_neurons)  # 初始化 I+ 群电位
        self.I_minus = DiehlAndCookNodes(n=num_neurons)
        self.I_minus.v = torch.zeros(num_neurons)  # 初始化 I- 群电位
        
        # 创建 B 群和 U 群，并初始化 B 群和 U 群的电位大小
        self.B = DiehlAndCookNodes(n=num_neurons)
        self.B.v = torch.zeros(num_neurons)  # 初始化 B 群电位
        self.U = DiehlAndCookNodes(n=num_neurons)
        self.U.v = torch.zeros(num_neurons)  # 初始化 U 群电位

        # 权重和阈值参数
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.thresholds = torch.linspace(0.1, 1.0, steps=num_neurons)  # B 群线性递增阈值
        self.bias_current = self.thresholds[num_neurons // 2]  # 中间神经元的阈值作为基准电流

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


    def compute_output(self):
        """
        计算输出控制信号。找到 B 群中活跃的神经元，将其映射到 U 群。
        """
        # 查找 B 群中超过阈值的神经元索引
        active_indices = (self.B.v >= self.thresholds).nonzero(as_tuple=True)[0]
        
        # 调试输出 B 群的活跃神经元索引
        print(f"Active indices in B: {active_indices}")

        if len(active_indices) > 0:
            # 找到 B 群中活跃神经元的最大索引
            max_index = active_indices.max().item()
            
            # U 群的稀疏编码，只有一个神经元活跃
            self.U.v.zero_()
            self.U.v[max_index] = 1  # 在 U 群中激活对应神经元
            
            # 返回控制信号
            return max_index  # 输出最大索引作为控制信号
        else:
            # 没有活跃神经元时返回 0
            return 0
