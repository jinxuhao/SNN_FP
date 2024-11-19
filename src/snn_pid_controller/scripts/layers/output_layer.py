import torch
from bindsnet.network.nodes import DiehlAndCookNodes

class ComplexOutputLayer(DiehlAndCookNodes):  # 继承 DiehlAndCookNodes
    def __init__(self, num_neurons=63, Kp=0.1, Ki=0.05, Kd=0.01):
        super(ComplexOutputLayer, self).__init__(n=num_neurons, thresh=1.0, reset=0)  # 初始化 DiehlAndCookNodes
        self.num_neurons = num_neurons
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # 创建 P+/P-, I+/I- 群和 B 群的电位
        self.P_plus = torch.zeros(num_neurons)
        self.P_minus = torch.zeros(num_neurons)
        self.I_plus = torch.zeros(num_neurons)
        self.I_minus = torch.zeros(num_neurons)
        self.B = torch.zeros(num_neurons)
        self.U = torch.zeros(num_neurons)

        # 设置阈值和偏置电流
        self.thresholds = torch.linspace(0.1, 1.0, steps=num_neurons)
        self.bias_current = self.thresholds[num_neurons // 2]

    def integrate_signals(self, P_signal, I_signal, D_signal=0):
        """
        将 P, I 信号分配到相应的神经元群，并更新 B 群的电位。
        """
        if P_signal > 0:
            self.P_plus += P_signal
        else:
            self.P_minus += -P_signal

        if I_signal > 0:
            self.I_plus += I_signal
        else:
            self.I_minus += -I_signal

        # 更新 B 群的电位
        self.B += abs(self.Kp) * (self.P_plus - self.P_minus)
        self.B += abs(self.Ki) * (self.I_plus - self.I_minus)
        self.B += self.bias_current

        # 限制电位范围
        self.B = torch.clamp(self.B, min=0, max=100)

    def compute_output(self):
        """
        计算输出控制信号，激活 U 群对应的神经元。
        """
        active_indices = (self.B >= self.thresholds).nonzero(as_tuple=True)[0]
        
        if len(active_indices) > 0:
            max_index = active_indices.max().item()
            self.U.zero_()  # 清除 U 群的激活状态
            self.U[max_index] = 1
            return max_index  # 返回控制信号
        else:
            return 0  # 如果没有激活神经元，则返回0

    # def forward(self, P_signal, I_signal, D_signal=0):
    #     """前向传播方法，用于更新并计算输出信号"""
    #     self.integrate_signals(P_signal, I_signal, D_signal)
    #     return self.compute_output()
    def forward(self, x):
        # 更新当前的状态变量 s
        self.s = x
        return self.s

    def compute_decays(self, dt):
        """覆盖 compute_decays 方法"""
        pass  # 不执行任何操作

    def set_batch_size(self, batch_size):
        """覆盖 set_batch_size 方法"""
        pass  # 不执行任何操作
