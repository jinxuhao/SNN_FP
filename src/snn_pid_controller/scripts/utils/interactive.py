import tkinter as tk
from tkinter import Scale, HORIZONTAL
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.plotting import plot_layer

def interactive_mode(input_layer, callback):
    """通过 GUI 滑块设置 current_angle 和 target_angle 的值，并直观展示"""
    root = tk.Tk()
    root.title("SNN Interactive Mode with Visualization")

    # 创建图形窗口
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    def update_plot(val=None):
        current_angle = angle_scale.get()
        target_angle = setpoint_scale.get()
        angular_velocity = angular_velocity_scale.get()
        
        callback(current_angle, target_angle)

        # 使用绘图函数
        plot_layer(axs[0], angular_velocity, "Angular Velocity Input Layer",
                   input_layer.angular_velocity_range[0], input_layer.angular_velocity_range[1], input_layer.num_neurons)
        plot_layer(axs[1], current_angle, "Angle Input Layer",
                   input_layer.angle_range[0], input_layer.angle_range[1], input_layer.num_neurons)
        plot_layer(axs[2], target_angle, "Setpoint Input Layer",
                   input_layer.setpoint_range[0], input_layer.setpoint_range[1], input_layer.num_neurons)

        canvas.draw()

    # 创建滑块
    angular_velocity_scale = Scale(root, from_=input_layer.angular_velocity_range[0], to=input_layer.angular_velocity_range[1],
                                   resolution=0.1, orient=HORIZONTAL, label="Angular Velocity")
    angular_velocity_scale.pack()
    angular_velocity_scale.bind("<Motion>", update_plot)

    angle_scale = Scale(root, from_=input_layer.angle_range[0], to=input_layer.angle_range[1],
                        resolution=0.1, orient=HORIZONTAL, label="Angle")
    angle_scale.pack()
    angle_scale.bind("<Motion>", update_plot)

    setpoint_scale = Scale(root, from_=input_layer.setpoint_range[0], to=input_layer.setpoint_range[1],
                           resolution=0.1, orient=HORIZONTAL, label="Setpoint")
    setpoint_scale.pack()
    setpoint_scale.bind("<Motion>", update_plot)

    update_plot()
    root.mainloop()
