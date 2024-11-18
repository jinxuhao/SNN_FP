from bindsnet.network.nodes import DiehlAndCookNodes
from bindsnet.network.topology import Connection
import numpy as np
import torch

class IntegrationLayer:
    def __init__(self, num_neurons=63, threshold=20):
        self.num_neurons = num_neurons
        self.threshold = threshold
        
        # 创建积分层的神经元群体
        self.c_plus = DiehlAndCookNodes(n=1, thresh=self.threshold, reset=0)  # 正误差计数
        self.c_minus = DiehlAndCookNodes(n=1, thresh=self.threshold, reset=0)  # 负误差计数
        self.shift_up = DiehlAndCookNodes(n=self.num_neurons)  # 上移控制
        self.shift_down = DiehlAndCookNodes(n=self.num_neurons)  # 下移控制
        self.I_population = DiehlAndCookNodes(n=self.num_neurons)  # 积分表示层

        # 初始化 v 张量的大小和初始值
        self.c_plus.v = torch.zeros(1)  # 为单个神经元设置电位值
        self.c_minus.v = torch.zeros(1)
        self.shift_up.v = torch.zeros(self.num_neurons)
        self.shift_down.v = torch.zeros(self.num_neurons)
        self.I_population.v = torch.zeros(self.num_neurons)

        # # 初始化连接
        # self.connections = []

        # 初始化 I_population 的起始激活状态
        self.I_population.v[0] = 1  # 初始激活位置在第一个神经元
        print("Initial I_population state:", self.I_population.v)

    def add_connection(self, source, target, weight):
        """添加连接，用于链接各个神经元群体"""
        connection = Connection(source=source, target=target, w=weight)
        self.connections.append(connection)

    def integrate_error(self, error_signal):
        """根据误差信号更新积分层的状态。
        error_signal: 输入的误差值，正值增加 c_plus 的电位，负值增加 c_minus 的电位。
        """
        if error_signal > 0:
            self.c_plus.v[0] += error_signal  # 累积正误差信号
            print(f"c_plus.v: {self.c_plus.v[0].item()}")  # 调试：打印 c_plus 的电位
        elif error_signal < 0:
            self.c_minus.v[0] += abs(error_signal)  # 累积负误差信号
            print(f"c_minus.v: {self.c_minus.v[0].item()}")  # 调试：打印 c_minus 的电位

        # 检查是否达到阈值并触发 shift 位置的移动
        if self.c_plus.v[0].item() >= self.threshold:  # 获取第一个神经元的电位值
            self.c_plus.v[0] = 0  # 重置 c_plus
            self.shift_up.v = torch.roll(self.shift_up.v, 1)  # 向上移动激活位置
            print("c_plus.v 达到阈值，触发 shift_up")

        if self.c_minus.v[0].item() >= self.threshold:  # 获取第一个神经元的电位值
            self.c_minus.v[0] = 0  # 重置 c_minus
            self.shift_down.v = torch.roll(self.shift_down.v, -1)  # 向下移动激活位置
            print("c_minus.v 达到阈值，触发 shift_down")

        # 更新 I_population 的位置
        self.update_integral_population()

    def update_integral_population(self):
        """根据 shift_up 和 shift_down 的激活情况更新 I_population 的状态"""
        # 获取当前激活的位置
        active_index = torch.argmax(self.I_population.v).item()
        print(f"Current active index in I_population: {active_index}")
        
        if torch.any(self.shift_up.v > 0) and active_index < self.num_neurons - 1:
            # 向上移动激活位置，确保不越界
            self.I_population.v[active_index] = 0  # 清除当前激活
            self.I_population.v[active_index + 1] = 1  # 激活上一个神经元
            print(f"Shift up activated: moved active index to {active_index + 1}")
            
        elif torch.any(self.shift_down.v > 0) and active_index > 0:
            # 向下移动激活位置，确保不越界
            self.I_population.v[active_index] = 0  # 清除当前激活
            self.I_population.v[active_index - 1] = 1  # 激活下一个神经元
            print(f"Shift down activated: moved active index to {active_index - 1}")
        
        # 输出 I_population 的状态以调试
        print("Updated I_population state:", self.I_population.v)


    def get_integral_state(self):
        """获取 I_population 当前表示的积分状态"""
        return torch.argmax(self.I_population.v).item()  # 返回当前激活的神经元索引


    def get_integral_state(self):
        """获取 I_population 当前表示的积分状态"""
        return torch.argmax(self.I_population.v).item()
