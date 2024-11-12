import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Scale, HORIZONTAL
from bindsnet.network import Network
from bindsnet.network.nodes import Input

# 创建网络和输入层
network = Network()
num_neurons = 63  # 每个输入层有63个神经元

# 定义三个输入层
angular_velocity_input = Input(n=num_neurons, name="angular_velocity")
angle_input = Input(n=num_neurons, name="angle")
setpoint_input = Input(n=num_neurons, name="setpoint")
network.add_layer(angular_velocity_input, name="angular_velocity")
network.add_layer(angle_input, name="angle")
network.add_layer(setpoint_input, name="setpoint")

# GUI 设置
root = tk.Tk()
root.title("SNN PID 控制器输入可视化")

# 创建图形窗口
fig, axs = plt.subplots(3, 1, figsize=(8, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

def plot_layer(layer, ax, input_value, title):
    # 将输入值映射到一个神经元的激活位置
    idx = int((input_value + 1) / 2 * (num_neurons - 1))  # 假设输入值范围是 [-1, 1]
    spikes = np.zeros(num_neurons)
    spikes[idx] = 1

    # 清除并重新绘制层的结构
    ax.clear()
    ax.scatter(range(num_neurons), spikes, color="blue")
    ax.set_title(title)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Neuron Index")

# 更新图形
def update_plot(val=None):
    plot_layer(angular_velocity_input, axs[0], angular_velocity_scale.get(), "Angular Velocity Input Layer")
    plot_layer(angle_input, axs[1], angle_scale.get(), "Angle Input Layer")
    plot_layer(setpoint_input, axs[2], setpoint_scale.get(), "Setpoint Input Layer")
    canvas.draw()


# snn_input.py


def generate_virtual_input(angle_range=(-40, 40), setpoint_range=(-25, 25)):
    """生成当前角度和目标角度的虚拟输入"""
    current_angle = np.random.uniform(angle_range[0], angle_range[1])
    target_angle = np.random.uniform(setpoint_range[0], setpoint_range[1])
    return current_angle, target_angle

# 创建滑块并绑定更新事件
angular_velocity_scale = Scale(root, from_=-1, to=1, resolution=0.1, orient=HORIZONTAL, label="Angular Velocity")
angular_velocity_scale.pack()
angular_velocity_scale.bind("<Motion>", update_plot)

angle_scale = Scale(root, from_=-1, to=1, resolution=0.1, orient=HORIZONTAL, label="Angle")
angle_scale.pack()
angle_scale.bind("<Motion>", update_plot)

setpoint_scale = Scale(root, from_=-1, to=1, resolution=0.1, orient=HORIZONTAL, label="Setpoint")
setpoint_scale.pack()
setpoint_scale.bind("<Motion>", update_plot)

# 初始化图形
update_plot()

# 运行主循环
root.mainloop()
