import numpy as np

def encode_1D(value, min_val, max_val, num_neurons):
    """将输入值映射到神经元的激活索引"""
    idx = int((value - min_val) / (max_val - min_val) * (num_neurons - 1))
    return max(0, min(idx, num_neurons - 1))
