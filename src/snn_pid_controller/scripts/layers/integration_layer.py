from bindsnet.network.nodes import DiehlAndCookNodes
import torch
import torch.nn as nn

class IntegrationLayer(DiehlAndCookNodes):  # 改为继承 DiehlAndCookNodes
    def __init__(self, num_neurons=63, threshold=20):
        super(IntegrationLayer, self).__init__(n=num_neurons, thresh=threshold, reset=0)
        self.num_neurons = num_neurons
        self.threshold = threshold
        
        # 初始化电位
        self.v = torch.zeros(self.num_neurons)
        self.v[0] = 1  # 初始激活位置

        # 初始化 c_plus 和 c_minus
        self.c_plus = torch.zeros(1)
        self.c_minus = torch.zeros(1)

    def integrate_error(self, error_signal):
        """根据误差信号更新积分层的状态，并返回当前控制信号"""
        if error_signal > 0:
            self.c_plus += error_signal  # 累积正误差信号
        elif error_signal < 0:
            self.c_minus += abs(error_signal)  # 累积负误差信号

        # 检查是否达到阈值并触发 I_population 的激活位置更新
        if self.c_plus.item() >= self.threshold:
            self.c_plus = torch.zeros(1)  # 重置 c_plus
            self.update_integral_population(direction="up")

        if self.c_minus.item() >= self.threshold:
            self.c_minus = torch.zeros(1)  # 重置 c_minus
            self.update_integral_population(direction="down")

        # 返回当前控制信号
        return self.get_integral_state()

    def update_integral_population(self, direction):
        """根据方向更新 I_population 的激活状态"""
        active_index = torch.argmax(self.v).item()  # 当前激活的位置
        
        if direction == "up" and active_index < self.num_neurons - 1:
            self.v[active_index] = 0
            self.v[active_index + 1] = 1
        elif direction == "down" and active_index > 0:
            self.v[active_index] = 0
            self.v[active_index - 1] = 1

    def get_integral_state(self):
        """获取当前激活神经元的位置，作为控制信号"""
        return torch.argmax(self.v).item()

    # def forward(self, error_signal):
    #     """前向传播，调用 integrate_error 函数以更新状态并返回控制信号"""
    #     return self.integrate_error(error_signal)

    def forward(self, x):
        # 更新当前的状态变量 s
        self.s = x
        return self.s