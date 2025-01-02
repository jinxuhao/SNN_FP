from scipy.io import loadmat
import matplotlib.pyplot as plt

# 加载 .mat 文件
# data = loadmat('src/snn_pid_controller/scripts/matlab/simulation_results_simple001.mat/simulation_results_simple001.mat')

# time = data['time'].flatten()
# values = data['data'].flatten()

data = loadmat('src/snn_pid_controller/scripts/matlab/simulation_results_simple001.mat/simulation_results_simple001.mat')
data01 = loadmat('src/snn_pid_controller/scripts/matlab/simulation_results_simple01.mat/simulation_results_simple01.mat')

time = data['time'].flatten()
values = data['data'].flatten()
time01 = data01['time'].flatten()
values01 = data01['data'].flatten()
print("MATLAB time data:", time)

intervals = time[1:] - time[:-1]  # 计算相邻时间点的差值
print("Time intervals:", intervals)

is_uniform = all(abs(intervals - intervals[0]) < 1e-6)  # 容许小误差
print("Is time uniformly sampled?:", is_uniform)

print("Number of time points:", len(time))

print("Start time:", time[0])
print("End time:", time[-1])

# 绘图
# import matplotlib.pyplot as plt
# plt.plot(time, values)
# plt.xlabel('Time (s)')
# plt.ylabel('Output')
# plt.title('Simulation Results')
# plt.grid()
# plt.show()
# 加载 .mat 文件

# MATLAB 的时间转换为迭代次数
matlab_steps = len(time)  # MATLAB 时间步数
matlab_iterations = [int(t / 0.01) for t in time]  # 转换为迭代次数
matlab_iterations01  = [int(t / 0.01) for t in time01]

# 绘制对比曲线
plt.figure(figsize=(10, 6))


plt.plot(matlab_iterations, values, label='matlab Controller', linestyle='-', marker='x')
plt.plot(matlab_iterations01, values01, label='matlab 01 Controller', linestyle='-', marker='o')

plt.xlabel('Simulation Step')
plt.ylabel('Current Angle')
plt.title('Comparison of SNN and PID Controllers with Mass-Spring-Damper System')
plt.legend()
plt.grid(True)

plt.show(block=True)