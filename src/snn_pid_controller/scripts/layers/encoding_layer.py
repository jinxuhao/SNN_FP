import numpy as np
import torch
from bindsnet.network.nodes import Nodes
from utils.helpers import encode_1D  # 确保 `encode_1D` 已定义在 `utils/helpers.py`

class EncodingLayer(Nodes):
    def __init__(self, num_neurons=63):
        super(EncodingLayer, self).__init__(n=num_neurons, shape=(num_neurons,))
        self.num_neurons = num_neurons
        self.state = torch.zeros(num_neurons)  # 初始化状态
        self.s = self.state.clone()  # 复制状态以满足 BindsNET 的要求

    def create_2D_operation_array(self, operation="add"):
        """创建一个2D操作数组，用于表示加法或减法操作"""
        operation_array = np.zeros((self.num_neurons, self.num_neurons))
        
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if operation == "add" and i + j < self.num_neurons:
                    operation_array[i, j] = 1
                elif operation == "subtract" and abs(i - j) < self.num_neurons:
                    operation_array[i, j] = 1
        return operation_array

    def encode(self, input_1_value, input_2_value, operation="add"):
        """根据操作类型计算输入的加法或减法结果，并返回结果索引"""
        input_1_idx = encode_1D(input_1_value, -40, 40, self.num_neurons)
        input_2_idx = encode_1D(input_2_value, -40, 40, self.num_neurons)
        
        if operation == "add":
            result_index = input_1_idx + input_2_idx
        elif operation == "subtract":
            result_index = input_1_idx - input_2_idx
        else:
            raise ValueError("Unsupported operation. Use 'add' or 'subtract'.")
        
        # 确保结果索引在 [0, num_neurons - 1] 范围内
        return max(0, min(result_index, self.num_neurons - 1))

    # def forward(self, input_1_value, input_2_value, operation="add"):
    #     """执行前向操作并更新编码层的状态"""
    #     idx = self.encode(input_1_value, input_2_value, operation)
    #     self.state = torch.zeros(self.num_neurons)
    #     self.state[idx] = 1
    #     self.s = self.state.clone()  # 更新 BindsNET 的状态变量

    def forward(self, x):
        # 更新当前的状态变量 s
        self.s = x
        return self.s
