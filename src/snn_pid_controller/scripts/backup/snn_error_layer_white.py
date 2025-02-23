import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Scale, HORIZONTAL
from matplotlib.patches import Circle

# 创建GUI窗口
root = tk.Tk()
root.title("SNN Error Layer Visualization with 2D Coincident Spiking")

# 创建图形窗口
fig, ax = plt.subplots(figsize=(8, 8))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# 定义离散点数量
num_points = 64

# 将输入值编码为神经元索引
def encode_input(value, min_val=-1, max_val=1, num_neurons=64):
    idx = int((value - min_val) / (max_val - min_val) * (num_neurons - 1))
    return max(0, min(idx, num_neurons - 1))

# 绘制 2D 检测矩阵和 error 输出
def draw_error_matrix(angular_velocity, angle, setpoint):
    ax.clear()

    # 将 angular_velocity, angle, setpoint 的值编码到对应的索引
    av_idx = encode_input(angular_velocity, -1, 1, num_points)
    angle_idx = encode_input(angle, -1, 1, num_points)
    setpoint_idx = encode_input(setpoint, -1, 1, num_points)
    
    # 创建64x64的检测矩阵
    matrix = np.zeros((num_points, num_points))  # 初始化为0，表示未激活

    # 设置检测重叠并生成 error 信号
    if 0 <= setpoint_idx < num_points and 0 <= angle_idx < num_points:
        matrix[setpoint_idx, angle_idx] = 1  # 重叠点为1

    # 显示2D矩阵（蓝色代表重叠区域激活）
    ax.imshow(matrix, cmap="Blues", origin="lower", extent=[0, num_points, 0, num_points])

    # 在左侧、顶部、右侧分别显示 angular_velocity, angle, setpoint 的编码位置
    for i in range(num_points):
        # 显示 angular_velocity 的位置
        color = 'blue' if i == av_idx else 'lightgrey'
        ax.plot(-1, i, 'o', color=color, markersize=8)  

        # 显示 angle 的位置
        color = 'blue' if i == angle_idx else 'lightgrey'
        ax.plot(i, num_points, 'o', color=color, markersize=8) 

        # 显示 setpoint 的位置
        color = 'blue' if i == setpoint_idx else 'lightgrey'
        ax.plot(num_points, i, 'o', color=color, markersize=8)  

    # 设置显示范围和标题
    ax.set_xlim(-2, num_points + 1)
    ax.set_ylim(-2, num_points + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"SNN Error Layer (Coincident Spiking Detection)")

    # 刷新图像
    canvas.draw()

# 更新图形
def update_plot(val=None):
    angular_velocity = angular_velocity_scale.get()
    angle = angle_scale.get()
    setpoint = setpoint_scale.get()
    draw_error_matrix(angular_velocity, angle, setpoint)

# 创建 r 和 y 的滑块，并绑定更新事件，仅在释放鼠标按钮时触发更新
angular_velocity_scale = Scale(root, from_=-1, to=1, resolution=0.1, orient=HORIZONTAL, label="Angular Velocity")
angular_velocity_scale.pack()
angular_velocity_scale.bind("<ButtonRelease-1>", update_plot)  # 仅在释放时更新

angle_scale = Scale(root, from_=-1, to=1, resolution=0.1, orient=HORIZONTAL, label="Angle")
angle_scale.pack()
angle_scale.bind("<ButtonRelease-1>", update_plot)  # 仅在释放时更新

setpoint_scale = Scale(root, from_=-1, to=1, resolution=0.1, orient=HORIZONTAL, label="Setpoint")
setpoint_scale.pack()
setpoint_scale.bind("<ButtonRelease-1>", update_plot)  # 仅在释放时更新

# 初始化图形
update_plot()

# 运行主循环
root.mainloop()
