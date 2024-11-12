import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Scale, HORIZONTAL

# 创建GUI窗口
root = tk.Tk()
root.title("SNN PID 控制器 - r, y 和 e_t 矩阵表示")

# 创建图形窗口
fig, ax = plt.subplots(figsize=(8, 8))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# 定义离散点数量
num_points = 64

# 绘制 e_t 矩阵及 r 和 y 矩形
def draw_et_matrix(r_value, y_value):
    ax.clear()
    
    # 将 r 和 y 的值映射到对应的索引
    r_idx = int((r_value + 1) / 2 * (num_points - 1))
    y_idx = int((y_value + 1) / 2 * (num_points - 1))
    et_value = r_value - y_value  # 计算 e_t = r - y
    
    # 创建64x64的矩阵，仅在交叉点变色
    matrix = np.full((num_points, num_points), 0.5)  # 基础颜色为灰色（0.5）
    matrix[r_idx, y_idx] = 1.0 if et_value >= 0 else 0.0  # 蓝色表示正，红色表示负

    # 显示矩阵
    ax.imshow(matrix, cmap="coolwarm", origin="lower", extent=[0, num_points, 0, num_points])

    # 在矩阵左侧绘制 64 个离散点表示 r
    for i in range(num_points):
        color = 'blue' if i == r_idx else 'lightgrey'
        ax.plot(-1, i + 0.5, 'o', color=color, markersize=8)  # 对齐中心

    # 在矩阵上方绘制 64 个离散点表示 y
    for i in range(num_points):
        color = 'blue' if i == y_idx else 'lightgrey'
        ax.plot(i + 0.5, num_points, 'o', color=color, markersize=8)  # 对齐中心

    # 设置显示范围和标题
    ax.set_xlim(-2, num_points)
    ax.set_ylim(-2, num_points + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"e_t Matrix (e_t = r - y = {et_value:.2f})")

    # 刷新图像
    canvas.draw()

# 更新图形
def update_plot(val=None):
    r_value = r_scale.get()
    y_value = y_scale.get()
    draw_et_matrix(r_value, y_value)

# 创建 r 和 y 的滑块，并绑定更新事件，仅在释放鼠标按钮时触发更新
r_scale = Scale(root, from_=-1, to=1, resolution=0.1, orient=HORIZONTAL, label="r (Setpoint)")
r_scale.pack()
r_scale.bind("<ButtonRelease-1>", update_plot)  # 仅在释放时更新

y_scale = Scale(root, from_=-1, to=1, resolution=0.1, orient=HORIZONTAL, label="y (Feedback)")
y_scale.pack()
y_scale.bind("<ButtonRelease-1>", update_plot)  # 仅在释放时更新

# 初始化图形
update_plot()

# 运行主循环
root.mainloop()
