from bindsnet.network.nodes import DiehlAndCookNodes
from bindsnet.network.topology import Connection
import torch

class IntegrationLayer:
    def __init__(self, num_neurons=63, threshold=20):
        self.num_neurons = num_neurons
        self.threshold = threshold
        
        # 初始化神经元群体
        self.c_plus = DiehlAndCookNodes(n=1, thresh=self.threshold, reset=0)
        self.c_minus = DiehlAndCookNodes(n=1, thresh=self.threshold, reset=0)
        self.I_population = DiehlAndCookNodes(n=self.num_neurons)
        
        # 初始化 v 张量的大小和初始值
        self.c_plus.v = torch.zeros(1)  
        self.c_minus.v = torch.zeros(1)
        self.I_population.v = torch.zeros(self.num_neurons)
        self.I_population.v[0] = 1  # 初始激活位置

    def integrate_error(self, error_signal):
        """根据误差信号更新积分层的状态，并返回当前控制信号"""
        if error_signal > 0:
            self.c_plus.v[0] += error_signal  # 累积正误差信号
        elif error_signal < 0:
            self.c_minus.v[0] += abs(error_signal)  # 累积负误差信号

        # 检查是否达到阈值并触发 I_population 的激活位置更新
        if self.c_plus.v[0].item() >= self.threshold:
            self.c_plus.v[0] = 0  # 重置 c_plus
            self.update_integral_population(direction="up")

        if self.c_minus.v[0].item() >= self.threshold:
            self.c_minus.v[0] = 0  # 重置 c_minus
            self.update_integral_population(direction="down")

        # 返回当前控制信号
        return self.get_integral_state()

    def update_integral_population(self, direction):
        """根据方向更新 I_population 的激活状态"""
        active_index = torch.argmax(self.I_population.v).item()  # 当前激活的位置
        
        if direction == "up" and active_index < self.num_neurons - 1:
            self.I_population.v[active_index] = 0
            self.I_population.v[active_index + 1] = 1
        elif direction == "down" and active_index > 0:
            self.I_population.v[active_index] = 0
            self.I_population.v[active_index - 1] = 1

    def get_integral_state(self):
        """获取当前激活神经元的位置，作为控制信号"""
        return torch.argmax(self.I_population.v).item()
