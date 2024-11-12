import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Scale, HORIZONTAL
from tkinter import Canvas, Frame, Scrollbar
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes

# 创建网络和输入层
network = Network()
num_neurons = 63  # 每个输入层和误差层、P 层都有 63 个神经元

# 定义输入层
angular_velocity_input = Input(n=num_neurons, name="angular_velocity")
angle_input = Input(n=num_neurons, name="angle")
setpoint_input = Input(n=num_neurons, name="setpoint")
network.add_layer(angular_velocity_input, name="angular_velocity")
network.add_layer(angle_input, name="angle")
network.add_layer(setpoint_input, name="setpoint")

# 定义误差层和 P 层
error_layer = LIFNodes(n=num_neurons, name="error_layer")  # 用于展示 e(t)
p_layer = LIFNodes(n=num_neurons, name="P_layer")  # 用于展示 P 项
network.add_layer(error_layer, name="error_layer")
network.add_layer(p_layer, name="P_layer")

# GUI 设置
root = tk.Tk()
root.title("SNN PID 控制器 - 输入、误差和 P 层可视化")

# 创建滚动区域
canvas = Canvas(root)
scrollable_frame = Frame(canvas)
scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# 调整滚动区域大小
def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

scrollable_frame.bind("<Configure>", on_configure)

# 在滚动区域中创建图形
fig, axs = plt.subplots(5, 1, figsize=(8, 10))
canvas_fig = FigureCanvasTkAgg(fig, master=scrollable_frame)
canvas_fig.get_tk_widget().pack()

def plot_layer(layer, ax, input_value, title, color="blue"):
    idx = int((input_value + 1) / 2 * (num_neurons - 1))
    spikes = np.zeros(num_neurons)
    spikes[idx] = 1
    ax.clear()
    ax.scatter(range(num_neurons), spikes, color=color)
    ax.set_title(title)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Neuron Index")

# 计算误差层和 P 层的输出
def calculate_error_layer(setpoint_value, angle_value):
    error = setpoint_value - angle_value  # 计算误差 e(t)
    idx = int((error + 1) / 2 * (num_neurons - 1))  # 将误差映射到神经元索引
    idx = max(0, min(num_neurons - 1, idx))  # 限制索引在有效范围内
    spikes = np.zeros(num_neurons)
    spikes[idx] = 1
    return spikes

def calculate_p_layer(error, kp):
    p_value = kp * error  # 计算 P 项
    idx = int((p_value + 1) / 2 * (num_neurons - 1))  # 映射到神经元索引
    idx = max(0, min(num_neurons - 1, idx))  # 限制索引在有效范围内
    spikes = np.zeros(num_neurons)
    spikes[idx] = 1
    return spikes

def update_plot(val=None):
    # 更新输入层的可视化
    plot_layer(angular_velocity_input, axs[0], angular_velocity_scale.get(), "Angular Velocity Input Layer")
    plot_layer(angle_input, axs[1], angle_scale.get(), "Angle Input Layer")
    plot_layer(setpoint_input, axs[2], setpoint_scale.get(), "Setpoint Input Layer")

    # 计算并更新误差层的可视化
    setpoint_value = setpoint_scale.get()
    angle_value = angle_scale.get()
    error = setpoint_value - angle_value
    error_spikes = calculate_error_layer(setpoint_value, angle_value)
    plot_layer(error_layer, axs[3], error, "Error Layer Output e(t)", color="purple")

    # 计算并更新 P 层的可视化
    kp = kp_scale.get()
    p_spikes = calculate_p_layer(error, kp)
    axs[4].clear()
    axs[4].scatter(range(num_neurons), p_spikes, color="red")
    axs[4].set_title(f"P Layer Output (P = Kp * e(t))")
    axs[4].set_ylim(-0.5, 1.5)
    axs[4].set_yticks([])
    axs[4].set_xlabel("Neuron Index")

    # 绘制图形
    canvas_fig.draw()

# 在滚动区域中创建滑块并绑定更新事件
angular_velocity_scale = Scale(scrollable_frame, from_=-1, to=1, resolution=0.1, orient=HORIZONTAL, label="Angular Velocity")
angular_velocity_scale.pack()
angular_velocity_scale.bind("<Motion>", update_plot)

angle_scale = Scale(scrollable_frame, from_=-1, to=1, resolution=0.1, orient=HORIZONTAL, label="Angle")
angle_scale.pack()
angle_scale.bind("<Motion>", update_plot)

setpoint_scale = Scale(scrollable_frame, from_=-1, to=1, resolution=0.1, orient=HORIZONTAL, label="Setpoint")
setpoint_scale.pack()
setpoint_scale.bind("<Motion>", update_plot)

kp_scale = Scale(scrollable_frame, from_=0, to=10, resolution=0.1, orient=HORIZONTAL, label="Proportional Gain (Kp)")
kp_scale.pack()
kp_scale.bind("<Motion>", update_plot)

# 初始化图形
update_plot()

# 运行主循环
root.mainloop()
