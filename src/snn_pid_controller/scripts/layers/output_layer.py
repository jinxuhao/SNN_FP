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
        self.thresholds = 0##torch.linspace(0.1, 1.0, steps=num_neurons)
        self.bias_current = 0##self.thresholds[num_neurons // 2]
        self.s = torch.zeros(self.num_neurons)  # 初始化 self.s 为全零张量

        

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

    ####paper way 
    # def forward(self, x):
    #     print(f"OUTPUT___Received B x: {x}")
    #     print(f"OUTPUT___Received of s (source spikes): {x.shape}, Data type: {x.dtype}")
    #     """
    #     处理输入 B 群的电位，并计算 U 群的输出（spike 发放信号）。
    #     :param x: 输入的 B 群电位，形状 [num_neurons]。
    #     :return: U 群的 spike 信号 self.s。
    #     """
    #     # 确保输入是 1D 张量，更新 self.B
    #     self.B = x.squeeze()

    #     # 找到所有超过阈值的神经元
    #     active_indices = (self.B >= self.thresholds).nonzero(as_tuple=True)[0]

    #     # 初始化 self.s（U 群的 spike 信号）
    #     self.s.zero_()

    #     if len(active_indices) > 0:
    #         # 找到激活的最大索引
    #         max_index = active_indices.max().item()

    #         # 激活 U 群中对应的神经元
    #         self.s[max_index] = 1  # 激活 U 群最大索引的神经元

    #         # 抑制较低索引的神经元
    #         self.s[:max_index] = 0

    #     return self.s  # 返回 U 群的 spike 输出
    def forward(self, x):
        print(f"OUTPUT___Received B x: {x}")
        print(f"OUTPUT___Received of s (source spikes): {x.shape}, Data type: {x.dtype}")
        """
        处理输入 B 群的电位，并计算 U 群的输出（spike 发放信号）。
        同时提取 B 群中所有活跃神经元的索引及其值，并计算加权平均值，再映射到新的索引。
        :param x: 输入的 B 群电位，形状 [num_neurons]。
        :return: U 群的 spike 信号 self.s，以及新的索引值。
        """
        # 确保输入是 1D 张量，更新 self.B
        self.B = x.squeeze()

        # 找到所有超过阈值的神经元
        active_indices = (self.B >= self.thresholds).nonzero(as_tuple=True)[0]
        active_values = self.B[active_indices]  # 提取对应索引的值

        # 打印活跃神经元的索引和值
        print(f"ACTIVE___Indices: {active_indices.tolist()}, Values: {active_values.tolist()}")

        # 初始化 self.s（U 群的 spike 信号）
        self.s.zero_()

        # 如果存在活跃神经元，则计算加权平均值和新索引
        if len(active_indices) > 0:
            num_neurons = self.B.shape[0]
            max_range = 40  # 设定值范围的一半
            mid_index = num_neurons // 2

            # 映射索引到值
            mapped_values = (active_indices.float() - mid_index)
            # 加权求和计算加权平均值
            weighted_value = (mapped_values * active_values).sum() 

            # 将加权值映射回索引
            new_index = mid_index + (weighted_value ) 
            print(f"CALCULATED___New Index: {new_index}")

            new_index = int(torch.round(new_index))  # 四舍五入到最近的整数索引
            new_index = max(0, min(new_index, num_neurons - 1))  # 限制索引范围
            # print(f"CALCULATED___mapped_values: {mapped_values}, num_neurons : {active_values}")

            print(f"CALCULATED___Weighted Value: {weighted_value:.2f}, New Index: {new_index}")

            # 激活 U 群中对应的神经元
            self.s[new_index] = 1  # 激活 U 群新索引的神经元

            return self.s
        else:
            # 如果没有活跃神经元，返回默认值
            print("CALCULATED___No active neurons. Returning default index: 0.")
            return self.s, 0


    
    def compute_decays(self, dt):
        """覆盖 compute_decays 方法"""
        pass  # 不执行任何操作

    def set_batch_size(self, batch_size):
        """覆盖 set_batch_size 方法"""
        pass  # 不执行任何操作
