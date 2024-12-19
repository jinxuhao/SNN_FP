import torch
from bindsnet.network.nodes import Nodes

class IIntermediateLayer(Nodes):
    def __init__(self, num_neurons=63):
        """
        I 层的中间层，用于处理积分信号。
        :param num_neurons: 神经元的数量。
        """
        super(IIntermediateLayer, self).__init__(n=num_neurons, shape=(1, num_neurons))
        self.num_neurons = num_neurons
        self.positive = torch.zeros(1, num_neurons)  # 表示 I+ 群
        self.negative = torch.zeros(1, num_neurons)  # 表示 I- 群

    def process_signal(self, x):
        """
        处理输入信号，将其拆分为正负群 I+ 和 I-，并更新 self.s。
        :param x: 输入信号（稀疏编码）。
        """
        # print(f"IIntermediateLayer___Received input x: {x}")
        
        # 清零正负信号和 self.s
        self.positive.zero_()
        self.negative.zero_()
        self.s.zero_()
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        # 获取激活神经元索引
        active_indices = (x).nonzero(as_tuple=True)[1].tolist()
        print(f"IIntermediateLayer___Active indices: {active_indices}")
        self.s = torch.zeros(1, self.num_neurons, dtype=torch.float)
        if len(active_indices) > 0:
            index = active_indices[0]  # 获取激活神经元的索引
            
            if index >= (self.num_neurons-1) // 2:  # 正值处理
                positive_index = index - (self.num_neurons-1) // 2
                self.positive[0, positive_index] = 1  # 更新 I+
                self.s[0, index] = 1  # 更新 self.s
            else:  # 负值处理
                negative_index = (self.num_neurons-1) // 2 - index
                self.negative[0, negative_index] = 1  # 更新 I-
                self.s[0, index] = 1  # 更新 self.s

        # print(f"IIntermediateLayer___Positive (I+): {self.positive}")
        # print(f"IIntermediateLayer___Negative (I-): {self.negative}")
        # print(f"IIntermediateLayer___Updated self.s: {self.s}")

    def forward(self, x):
        """
        接收输入信号，并调用 `process_signal` 方法。
        :param x: 输入信号。
        """
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        self.s =x
        # self.process_signal(x)
        return self.s
