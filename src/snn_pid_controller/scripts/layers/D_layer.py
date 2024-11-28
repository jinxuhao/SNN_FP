import torch
from bindsnet.network.nodes import Nodes

class DIntermediateLayer(Nodes):
    def __init__(self, num_neurons=63, scaling_factor=1.0):
        """
        D 层：根据误差变化率 Delta e_t 计算输出。
        :param num_neurons: 神经元数量。
        :param scaling_factor: 缩放因子，用于将 Delta e_t 映射到神经元索引。
        """
        super(DIntermediateLayer, self).__init__(n=num_neurons)
        self.num_neurons = num_neurons
        self.scaling_factor = scaling_factor
        self.previous_error = None  # 保存前一时刻误差值
        self.s = torch.zeros(1, num_neurons)  # 初始化状态
    
    def forward(self, x):
        """
        前向传播：接收输入并计算 Delta e_t。
        :param x: 输入信号（one-hot 编码）。
        """
        print(f"DIntermediateLayer___Received input x: {x}")

        # 确保输入形状正确
        if x.dim() == 3:
            x = x.squeeze(1)

        # 获取当前误差值索引
        active_indices = torch.nonzero(x, as_tuple=True)[1].tolist()
        print(f"DIntermediateLayer___Active indices: {active_indices}")

        if len(active_indices) < 1:
            print("DIntermediateLayer___Warning: No active neurons in the input.")
            return self.s  # 无输入时，返回当前状态

        # 解码当前误差值
        current_error_index = active_indices[0]
        current_error = current_error_index - self.num_neurons // 2  # 中心为 0 的对称范围
        print(f"DIntermediateLayer___Current error: {current_error}")

        # 计算 Delta e_t
        if self.previous_error is not None:
            delta_e_t = current_error - self.previous_error
        else:
            delta_e_t = 0  # 初始化时，无变化
        print(f"DIntermediateLayer___Delta e_t: {delta_e_t}")

        # 更新 previous_error
        self.previous_error = current_error

        # 将 Delta e_t 映射到神经元索引
        delta_index = int(delta_e_t * self.scaling_factor) + self.num_neurons // 2
        delta_index = max(0, min(delta_index, self.num_neurons - 1))  # 限制索引范围
        print(f"DIntermediateLayer___Delta index: {delta_index}")

        # 更新 self.s
        self.s.fill_(0)
        self.s[0, delta_index] = 1
        print(f"DIntermediateLayer___Updated self.s: {self.s}")

        return self.s

    def compute_decays(self, dt):
        """覆盖 compute_decays 方法，防止状态自动衰减。"""
        pass

    def set_batch_size(self, batch_size):
        """覆盖 set_batch_size 方法，防止批量操作的干扰。"""
        pass