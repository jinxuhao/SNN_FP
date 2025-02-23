import numpy as np

# 1D 编码函数
def encode_1D(value, min_val, max_val, num_neurons):
    idx = int((value - min_val) / (max_val - min_val) * (num_neurons - 1))
    return max(0, min(idx, num_neurons - 1))

# 2D 操作数组
def create_2D_operation_array(input_1, input_2, operation="add"):
    num_neurons = len(input_1)
    operation_array = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        for j in range(num_neurons):
            if operation == "add":
                if i + j < num_neurons:
                    operation_array[i, j] = 1  # 对角线为加法连接
            elif operation == "subtract":
                if abs(i - j) < num_neurons:
                    operation_array[i, j] = 1  # 反对角线为减法连接
    return operation_array

# # 使用 2D 操作数组实现加法/减法运算
# def perform_operation(input_1_value, input_2_value, operation="add", num_neurons=63):
#     input_1_idx = encode_1D(input_1_value, -1, 1, num_neurons)
#     input_2_idx = encode_1D(input_2_value, -1, 1, num_neurons)

#     operation_array = create_2D_operation_array(range(num_neurons), range(num_neurons), operation)
#     result_index = np.argmax(operation_array[input_1_idx, :])  # 获取结果的索引
#     return result_index - (num_neurons // 2)  # 结果居中处理
def perform_operation(input_1_value, input_2_value, operation="add", num_neurons=63):
    # 将输入值编码为对应的神经元索引
    input_1_idx = encode_1D(input_1_value, -1, 1, num_neurons)
    input_2_idx = encode_1D(input_2_value, -1, 1, num_neurons)
    
    # 根据操作选择加法或减法
    if operation == "add":
        result_index = input_1_idx + input_2_idx
    elif operation == "subtract":
        result_index = input_1_idx - input_2_idx
    else:
        raise ValueError("Unsupported operation. Use 'add' or 'subtract'.")
    
    # 确保结果索引在 [0, num_neurons - 1] 范围内
    result_index = max(0, min(result_index, num_neurons - 1))
    
    return result_index

# 示例使用
angular_velocity = 0.5  # 示例输入值
angle = 0.2
setpoint = 0.8
error_value = perform_operation(setpoint, angle, "subtract")

print(f"Encoded error (e_t) value: {error_value}")
