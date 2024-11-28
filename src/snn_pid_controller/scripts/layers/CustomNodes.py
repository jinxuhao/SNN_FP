import torch
from bindsnet.network.nodes import DiehlAndCookNodes

class CustomNodes(DiehlAndCookNodes):
    def __init__(self,num_neurons=63, *args, **kwargs):
        self.num_neurons = num_neurons
        super().__init__(n=num_neurons,*args, **kwargs)
        self.input_storage = []  # 独立存储多个输入

    def forward(self, x: torch.Tensor,) -> None:
        """
        自定义前向传播，支持多个输入信号。
        """
        # self.inputs.append(x)  # 将每个输入缓存
        # print(f"OUTPUT___Received of input (source spikes): {self.inputs}, Data type: {self.inputs}")
        # self.input = sum(self.inputs)  # 或根据需求汇总
        print(f"OUTPUT___Received of x (source spikes): {x}, Data type: {x.dtype}")
        # # print(f"OUTPUT___Received of s (source spikes): {self.input}, Data type: {self.input}")
        # # 继续调用父类的 forward
        # super().forward(self.input)

        self.input_storage.append(x)
        # 示例：按序累加
        combined_input = torch.stack(self.input_storage, dim=0).sum(dim=0)
        self.input = combined_input


        print(f"OUTPUT___Received of input (source spikes): {self.input_storage}")
