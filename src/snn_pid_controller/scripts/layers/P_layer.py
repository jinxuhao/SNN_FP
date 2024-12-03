import torch
from bindsnet.network.nodes import Nodes


class PIntermediateLayer(Nodes):
    def __init__(self, num_neurons=63):
        """
        P 中间层，用于处理 P 信号。
        :param num_neurons: 神经元数量。
        """
        super(PIntermediateLayer, self).__init__(n=num_neurons, shape=(1, num_neurons))
        self.num_neurons = num_neurons
        self.positive = torch.zeros(1, num_neurons)
        self.negative = torch.zeros(1, num_neurons)

    def process_signal(self, x):
        """
        处理输入信号，将其拆分为正负群，并更新 self.s。
        :param x: 输入信号（稀疏编码）。
        """
        print(f"PIntermediateLayer___Received input x: {x}")
        self.positive.zero_()
        self.negative.zero_()
        self.s.zero_()  # 确保每次更新 self.s
        # Ensure x is in the correct shape
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        active_indices = torch.nonzero(x, as_tuple=True)[1].tolist()
        print(f"PIntermediateLayer___Positive: {active_indices}")
        self.s = torch.zeros(1, self.num_neurons, dtype=torch.float)

        if len(active_indices) > 0:
            index = active_indices[0]  # 获取激活神经元的索引
            if index >= self.num_neurons // 2:  # 正值处理
                self.positive[0, index - self.num_neurons // 2] = 1
                self.s[0, index] = 1  # 更新 self.s
            else:  # 负值处理
                self.negative[0, self.num_neurons // 2 - index] = 1
                self.s[0, index] = 1  # 更新 self.s

        # print(f"PIntermediateLayer___Positive: {self.positive}")
        # print(f"PIntermediateLayer___Negative: {self.negative}")
        # print(f"PIntermediateLayer___Updated self.s: {self.s}")

    def forward(self, x):
        """
        接收输入信号，调用 `process_signal` 方法，并返回更新后的 self.s。
        :param x: 输入信号。
        """
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        self.s =x
        # self.process_signal(x)
        return self.s

        #####P+ 和 P- 作为一个整体节点传递P_combined = P_plus - P_minus）######

# import torch
# from bindsnet.network.nodes import Nodes

# class PIntermediateLayer(Nodes):
#     def __init__(self, num_neurons=63, scale_factor=1.0):
#         super(PIntermediateLayer, self).__init__(n=num_neurons, shape=(1, num_neurons))
#         self.num_neurons = num_neurons
#         self.scale_factor = scale_factor
#         self.positive = torch.zeros(1, num_neurons // 2)  # P+ 部分
#         self.negative = torch.zeros(1, num_neurons // 2)  # P- 部分

#     def process_signal(self, x):
#         """
#         处理输入信号，将 P+ 和 P- 计算并合并。
#         """
#         print(f"PIntermediateLayer___Received input x: {x}")
#         if x.dim() == 3 and x.shape[1] == 1:
#             x = x.squeeze(1)

#         # 确保输入形状正确
#         active_indices = torch.nonzero(x, as_tuple=True)[1].tolist()
#         print(f"PIntermediateLayer___Active indices: {active_indices}")

#         if len(active_indices) > 0:
#             index = active_indices[0]  # 获取激活神经元的索引
#             if index >= self.num_neurons // 2:  # 正值处理
#                 self.positive[0, index - self.num_neurons // 2] = 1
#                 combined_signal = (index - self.num_neurons // 2) / self.scale_factor
#             else:  # 负值处理
#                 self.negative[0, self.num_neurons // 2 - index] = 1
#                 combined_signal = -(self.num_neurons // 2 - index) / self.scale_factor
#         else:
#             combined_signal = 0

#         print(f"PIntermediateLayer___Positive: {self.positive}")
#         print(f"PIntermediateLayer___Negative: {self.negative}")
#         print(f"PIntermediateLayer___Combined signal: {combined_signal}")

#         # 更新 self.s，与 BindsNET 兼容
#         self.s.fill_(0)  # 清空 self.s
#         self.s[0, active_indices] = 1 if len(active_indices) > 0 else 0

#         return combined_signal

#     def forward(self, x):
#         """
#         接收输入信号，并调用 `process_signal` 方法。
#         """
#         combined_signal = self.process_signal(x)
#         print(f"PIntermediateLayer___Updated self.s: {self.s}")
#         return self.s  # 保持与 BindsNET 的兼容性
