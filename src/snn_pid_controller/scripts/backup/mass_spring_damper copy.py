import numpy as np
import matplotlib.pyplot as plt

# 动力学模型参数
m = 1.0
c = 2.0  # 减小阻尼系数，接近欠阻尼
k = 20.0

# 动力学模型
def mass_spring_damper_dynamics(force, position, velocity, dt):
    acceleration = (force - c * velocity - k * position) / m
    new_velocity = velocity + acceleration * dt
    new_position = position + new_velocity * dt
    return new_position, new_velocity

# 仿真参数
dt = 0.001  # 时间步长
num_steps = 5000  # 仿真步数
target_position = 1.0  # 目标位置

# 寻找临界增益 Kcr
K_p_values = np.arange(0, 500, 2.0)  # 增大范围
critical_gain = None
critical_period = None

for K_p in K_p_values:
    position = 0.0
    velocity = 0.0
    positions = [position]
    for step in range(num_steps):
        error = target_position - position
        force = K_p * error
        position, velocity = mass_spring_damper_dynamics(force, position, velocity, dt)
        positions.append(position)
    
    # 检测振荡
    steady_state_positions = positions[-500:]  # 取最后 500 个位置数据
    std_dev = np.std(steady_state_positions)  # 计算标准差
    print(f"Kp = {K_p}, Std Dev = {std_dev:.3f}")
    if std_dev > 0.005:  # 标准差阈值放宽
        peaks = [i for i in range(1, len(steady_state_positions) - 1)
                 if steady_state_positions[i - 1] < steady_state_positions[i] > steady_state_positions[i + 1]]
        if len(peaks) >= 2:  # 至少有两个峰值
            critical_gain = K_p
            critical_period = (peaks[-1] - peaks[0]) * dt / (len(peaks) - 1)
            break

if critical_gain is not None:
    print(f"临界增益 Kcr = {critical_gain}, 临界周期 Tcr = {critical_period}")
else:
    print("未检测到持续振荡。请调整仿真参数或增益范围。")

# 绘图
if critical_gain is not None:
    time = [i * dt for i in range(num_steps + 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(time, positions, label=f"Kp = {critical_gain}")
    plt.axhline(target_position, color="r", linestyle="--", label="Target Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title("System Response at Critical Gain")
    plt.legend()
    plt.grid()
    plt.show()
