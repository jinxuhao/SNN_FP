# snn_input.py

import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def generate_virtual_input(angle_range=(-40, 40), setpoint_range=(-25, 25)):
    """生成当前角度和目标角度的虚拟输入"""
    current_angle = np.random.uniform(angle_range[0], angle_range[1])
    target_angle = np.random.uniform(setpoint_range[0], setpoint_range[1])
    return current_angle, target_angle

def encode_index(value, min_val, max_val, num_neurons):
    """根据输入范围和神经元数量进行索引编码"""
    idx = int((value - min_val) / (max_val - min_val) * (num_neurons - 1))
    return max(0, min(idx, num_neurons - 1))

def interactive_mode(callback, angular_velocity_range=(-80, 80), angle_range=(-40, 40), setpoint_range=(-25, 25), num_neurons=63):
    """通过 GUI 滑块设置 current_angle 和 target_angle 的值，并直观展示"""
    root = tk.Tk()
    root.title("SNN Interactive Mode with Visualization")

    # 创建图形窗口
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    # 实时绘图更新函数
    def plot_layer(ax, value, title, min_val, max_val):
        idx = encode_index(value, min_val, max_val, num_neurons)
        spikes = np.zeros(num_neurons)
        spikes[idx] = 1

        ax.clear()
        ax.scatter(range(num_neurons), spikes, color="blue")
        ax.set_title(title)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([])
        ax.set_xlabel("Neuron Index")

    # 更新图形
    def update_plot(val=None):
        current_angle = angle_scale.get()
        target_angle = setpoint_scale.get()
        angular_velocity = angular_velocity_scale.get()
        
        # 调用主函数中的回调函数
        callback(current_angle, target_angle)

        # 使用正确的范围绘制每个输入层的数值
        plot_layer(axs[0], angular_velocity, "Angular Velocity Input Layer", angular_velocity_range[0], angular_velocity_range[1])
        plot_layer(axs[1], current_angle, "Angle Input Layer", angle_range[0], angle_range[1])
        plot_layer(axs[2], target_angle, "Setpoint Input Layer", setpoint_range[0], setpoint_range[1])

        canvas.draw()

    # 创建滑块并设置实际范围
    angular_velocity_scale = Scale(root, from_=angular_velocity_range[0], to=angular_velocity_range[1], resolution=0.1, orient=HORIZONTAL, label="Angular Velocity")
    angular_velocity_scale.pack()
    angular_velocity_scale.bind("<Motion>", update_plot)

    angle_scale = Scale(root, from_=angle_range[0], to=angle_range[1], resolution=0.1, orient=HORIZONTAL, label="Angle")
    angle_scale.pack()
    angle_scale.bind("<Motion>", update_plot)

    setpoint_scale = Scale(root, from_=setpoint_range[0], to=setpoint_range[1], resolution=0.1, orient=HORIZONTAL, label="Setpoint")
    setpoint_scale.pack()
    setpoint_scale.bind("<Motion>", update_plot)

    # 初始化图形
    update_plot()

    root.mainloop()
