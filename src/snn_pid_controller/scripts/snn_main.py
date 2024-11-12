# main.py

from snn_input import interactive_mode  # 导入 GUI 模式
from snn_encoding import encode_1D, perform_operation  # 导入编码和运算函数

# 配置角度范围和神经元数量
angle_range = (-40, 40)  # 当前角度范围
setpoint_range = (-25, 25)  # 目标角度范围
num_neurons = 64  # 1D和2D神经元阵列的数量

def update_error(current_angle, target_angle):
    """实时更新并打印 error 值"""
    # 对 `y` 和 `r` 进行编码
    y_encoded = encode_1D(current_angle, min_val=angle_range[0], max_val=angle_range[1], num_neurons=num_neurons)
    r_encoded = encode_1D(target_angle, min_val=setpoint_range[0], max_val=setpoint_range[1], num_neurons=num_neurons)

    # 计算误差 `e_t = r - y`
    error_index = perform_operation(target_angle, current_angle, operation="subtract", num_neurons=num_neurons)

    # 实时输出编码结果和误差
    print(f"Current Angle (y): {current_angle}, Encoded Index: {y_encoded}")
    print(f"Target Angle (r): {target_angle}, Encoded Index: {r_encoded}")
    print(f"Error (e_t = r - y) Index: {error_index}\n")

if __name__ == "__main__":
    # 启动 GUI 模式并传入回调函数
    interactive_mode(update_error, angle_range=angle_range, setpoint_range=setpoint_range)
