dt_pid = 10  # PID 采样时间 (ms)
error_signal = [1, 2, 3, 4, 5]  # 模拟误差信号
pid_integral = 0
pid_output = []

for e in error_signal:
    pid_integral += e * dt_pid
    pid_output.append(pid_integral)
print("PID Integral:", pid_output)

import torch
from bindsnet.network import Network
from layers.integration_layer import IntegrationLayer

dt_snn = 1  # SNN 采样时间 (ms)
scale_factor = 10  # 调整比例因子
integration_layer = IntegrationLayer(num_neurons=63, scale_factor=scale_factor)

snn_integral_output = []
for e in error_signal:
    inputs = torch.zeros(1, 63)
    inputs[0, int(e) + 31] = 1  # 将误差信号编码到 SNN 输入
    output = integration_layer.forward(inputs)
    snn_integral_output.append(torch.argmax(output).item())
print("SNN Integral:", snn_integral_output)
