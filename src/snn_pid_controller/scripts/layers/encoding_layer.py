import numpy as np
from utils.helpers import encode_1D  # 从 utils/helpers.py 导入通用编码函数

class EncodingLayer:
    def __init__(self, num_neurons=63):
        self.num_neurons = num_neurons

    def create_2D_operation_array(self, operation="add"):
        """创建一个2D操作数组，用于表示加法或减法操作"""
        operation_array = np.zeros((self.num_neurons, self.num_neurons))
        
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if operation == "add" and i + j < self.num_neurons:
                    operation_array[i, j] = 1  # 对角线为加法
                elif operation == "subtract" and abs(i - j) < self.num_neurons:
                    operation_array[i, j] = 1  # 反对角线为减法
        return operation_array

    def perform_operation(self, input_1_value, input_2_value, operation="add"):
        """根据操作类型计算输入的加法或减法结果，并返回结果索引"""
        input_1_idx = encode_1D(input_1_value, -1, 1, self.num_neurons)
        input_2_idx = encode_1D(input_2_value, -1, 1, self.num_neurons)
        
        if operation == "add":
            result_index = input_1_idx + input_2_idx
        elif operation == "subtract":
            result_index = input_1_idx - input_2_idx
        else:
            raise ValueError("Unsupported operation. Use 'add' or 'subtract'.")
        
        # 确保结果索引在 [0, num_neurons - 1] 范围内
        return max(0, min(result_index, self.num_neurons - 1))

    def perform_operation(self, input_1_value, input_2_value, operation="add"):
        """根据操作类型计算输入的加法或减法结果，并返回结果索引"""
        # 使用实际范围进行编码
        input_1_idx = encode_1D(input_1_value, -40, 40, self.num_neurons)  # 使用角度范围
        input_2_idx = encode_1D(input_2_value, -40, 40, self.num_neurons)
        
        if operation == "add":
            result_index = input_1_idx + input_2_idx
        elif operation == "subtract":
            result_index = input_1_idx - input_2_idx
        else:
            raise ValueError("Unsupported operation. Use 'add' or 'subtract'.")
        
        # 确保结果索引在 [0, num_neurons - 1] 范围内
        return max(0, min(result_index, self.num_neurons - 1))
