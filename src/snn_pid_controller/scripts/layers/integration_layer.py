# from bindsnet.network.nodes import DiehlAndCookNodes
# import torch
# import torch.nn as nn

# class IntegrationLayer(DiehlAndCookNodes):  # 改为继承 DiehlAndCookNodes
#     def __init__(self, num_neurons=63, threshold=20):
#         super(IntegrationLayer, self).__init__(n=num_neurons, thresh=threshold, reset=0)
#         self.num_neurons = num_neurons
#         self.threshold = threshold
        
#         # 初始化电位
#         self.v = torch.zeros(self.num_neurons)
#         self.v[0] = 1  # 初始激活位置

#         # 初始化 c_plus 和 c_minus
#         self.c_plus = torch.zeros(1)
#         self.c_minus = torch.zeros(1)

#     def integrate_error(self, error_signal):
#         """根据误差信号更新积分层的状态，并返回当前控制信号"""
#         if error_signal > 0:
#             self.c_plus += error_signal  # 累积正误差信号
#         elif error_signal < 0:
#             self.c_minus += abs(error_signal)  # 累积负误差信号

#         # 检查是否达到阈值并触发 I_population 的激活位置更新
#         if self.c_plus.item() >= self.threshold:
#             self.c_plus = torch.zeros(1)  # 重置 c_plus
#             self.update_integral_population(direction="up")

#         if self.c_minus.item() >= self.threshold:
#             self.c_minus = torch.zeros(1)  # 重置 c_minus
#             self.update_integral_population(direction="down")

#         # 返回当前控制信号
#         return self.get_integral_state()

#     def update_integral_population(self, direction):
#         """根据方向更新 I_population 的激活状态"""
#         active_index = torch.argmax(self.v).item()  # 当前激活的位置
        
#         if direction == "up" and active_index < self.num_neurons - 1:
#             self.v[active_index] = 0
#             self.v[active_index + 1] = 1
#         elif direction == "down" and active_index > 0:
#             self.v[active_index] = 0
#             self.v[active_index - 1] = 1

#     def get_integral_state(self):
#         """获取当前激活神经元的位置，作为控制信号"""
#         return torch.argmax(self.v).item()

#     # def forward(self, error_signal):
#     #     """前向传播，调用 integrate_error 函数以更新状态并返回控制信号"""
#     #     return self.integrate_error(error_signal)

#     def forward(self, x):
#         # 更新当前的状态变量 s
#         # Ensure x is in the correct shape
#         if x.dim() == 3 and x.shape[1] == 1:
#             x = x.squeeze(1)
#         print(f"INTEGRATION received input x with shape: {x.shape}")
#         print(f"INTEGRATION___X_tensor x: {x}")
#         self.s = x
#         return self.s

import torch
from bindsnet.network.nodes import Nodes

class IntegrationLayer(Nodes):
    def __init__(self, num_neurons=63, threshold=5, scale_factor=5):
        """
        改进版积分层，接收 EncodingLayer 的输出信号，并实现误差信号积分。
        :param num_neurons: I 层的神经元数量。
        :param threshold: 计数器的阈值。
        :param scale_factor: 调整误差信号对 I 层更新影响的比例因子。
        """
        super(IntegrationLayer, self).__init__(n=num_neurons, shape=(1, num_neurons))
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.step_size =1#5 # 单次步长的基准值

        # 初始化辅助神经元群体
        self.c_plus = torch.zeros(1, 1)  # c+ 计数器神经元
        self.c_minus = torch.zeros(1, 1)  # c- 计数器神经元
        self.shift_up = torch.zeros(1, num_neurons)  # ShiftUp 层
        self.shift_down = torch.zeros(1, num_neurons)  # ShiftDown 层

        # 初始化 I 层状态
        self.I = torch.zeros(1, num_neurons)
        self.I[0, num_neurons // 2] = 1  # 初始值在中间

    def forward(self, x):
        """
        接收 EncodingLayer 的输出信号，并实现误差信号的积分。
        :param x: 输入信号，来自 EncodingLayer 的 one-hot 编码信号。
        """
        print(f"Integration___Received input x with shape: {x.shape}")

        # 确保输入形状正确
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        # 检查是否接收到有效信号
        active_indices = torch.nonzero(x, as_tuple=True)[1].tolist()
        print(f"Integration___Active indices: {active_indices}")

        if len(active_indices) < 1:
            print("DEBUG___Warning: No active neurons in the input.")
            self.s = self.I.clone()
            return self.s

        # 获取激活神经元索引
        e_t_index = active_indices[0]
        # print(f"Integration___e_t_index: {e_t_index}")
        increment = abs(e_t_index - self.num_neurons // 2) / self.scale_factor
        print(f"Integration___increment activated. increment: {increment}")

        # 判断误差信号的正负并更新计数器
        if e_t_index > self.num_neurons // 2:  # 正误差信号
            # increment = (e_t_index - self.num_neurons // 2) / self.scale_factor
            self.c_plus += increment
            self.c_minus -= increment  # 保持镜像关系
            self.c_minus = torch.clamp(self.c_minus, min=0)  # 确保 c_minus 非负
            # print(f"Integration___Positive error. Increment: {increment}, c_plus: {self.c_plus}, c_minus: {self.c_minus}")
        
        elif e_t_index < self.num_neurons // 2:  # 负误差信号
            # increment = (self.num_neurons // 2 - e_t_index) / self.scale_factor
            self.c_minus += increment
            self.c_plus -= increment  # 保持镜像关系
            self.c_plus = torch.clamp(self.c_plus, min=0)  # 确保 c_plus 非负
            # print(f"Integration___Negative error. Increment: {increment}, c_plus: {self.c_plus}, c_minus: {self.c_minus}")

        if self.c_plus >= self.threshold:
            # 软复位计数器
            self.c_plus -= self.threshold  

            # 根据增量调整 shift_index
            shift_index = torch.argmax(self.I).item()
            shift_index = min(shift_index + int(increment / self.step_size), self.num_neurons - 1)  # 增量影响步长
            # print(f"Integration___ShiftUp activated. New shift_index: {shift_index}")

            # 更新 I 层状态
            self.I.zero_()  # 清零 I 层
            self.I[0, shift_index] = 1  # 激活新位置

        if self.c_minus >= self.threshold:
            # 软复位计数器
            self.c_minus -= self.threshold  

            # 根据增量调整 shift_index
            shift_index = torch.argmax(self.I).item()
            # print(f"Integration___before ShiftDown activated. OLD shift_index: {shift_index}")
            shift_index = max(shift_index - int(increment / self.step_size), 0)  # 增量影响步长
            # print(f"Integration___ShiftDown activated. New shift_index: {shift_index}")

            # 更新 I 层状态
            self.I.zero_()  # 清零 I 层
            self.I[0, shift_index] = 1  # 激活新位置


        # print(f"Integration___Updated I state: {self.I}")


        # 获取 I 层当前激活神经元索引
        I_active_index = torch.argmax(self.I).item()
        # print(f"Integration___Active index in I: {I_active_index}")

        # 更新 self.s，确保与 BindsNET 兼容
        self.s = self.I.clone()
        return self.s
