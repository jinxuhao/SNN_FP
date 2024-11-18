import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.network.monitors import Monitor

# 创建网络
network = Network(dt=1.0)

# 创建输入层
input_layer = Input(n=1)

# 创建 P、I、D 层，设置较大的 tau 以便触发
p_layer = LIFNodes(n=1, thresh=-50.0, tau=50.0)
i_layer = LIFNodes(n=1, thresh=-50.0, tau=50.0)
d_layer = LIFNodes(n=1, thresh=-50.0, tau=50.0)

# 将各层加入网络
network.add_layer(input_layer, name="input")
network.add_layer(p_layer, name="P")
network.add_layer(i_layer, name="I")
network.add_layer(d_layer, name="D")

# 增大 PID 参数
Kp, Ki, Kd = 20.0, 10.0, 5.0

# 输入层到 P、I、D 层的连接
network.add_connection(Connection(input_layer, p_layer, w=torch.tensor([Kp])), source="input", target="P")
network.add_connection(Connection(input_layer, i_layer, w=torch.tensor([Ki])), source="input", target="I")
network.add_connection(Connection(input_layer, d_layer, w=torch.tensor([Kd])), source="input", target="D")

# 创建输出层
output_layer = LIFNodes(n=1, thresh=-50.0, tau=100.0)
network.add_layer(output_layer, name="output")

# P、I、D 层到输出层的连接
network.add_connection(Connection(p_layer, output_layer, w=torch.tensor([5.0])), source="P", target="output")
network.add_connection(Connection(i_layer, output_layer, w=torch.tensor([5.0])), source="I", target="output")
network.add_connection(Connection(d_layer, output_layer, w=torch.tensor([5.0])), source="D", target="output")

# 设置监视器
time_steps = 500
input_monitor = Monitor(input_layer, ["s"], time=time_steps)
p_monitor = Monitor(p_layer, ["v", "s"], time=time_steps)
i_monitor = Monitor(i_layer, ["v", "s"], time=time_steps)
d_monitor = Monitor(d_layer, ["v", "s"], time=time_steps)
output_monitor = Monitor(output_layer, ["v", "s"], time=time_steps)

# 将监视器加入网络
network.add_monitor(input_monitor, name="input_monitor")
network.add_monitor(p_monitor, name="p_monitor")
network.add_monitor(i_monitor, name="i_monitor")
network.add_monitor(d_monitor, name="d_monitor")
network.add_monitor(output_monitor, name="output_monitor")

# 显著增大输入数据并重复
input_data = torch.tensor([[10.0]]).repeat(time_steps, 1)  # 增大输入值

# 运行仿真
network.run(inputs={"input": input_data}, time=time_steps)

# 可视化脉冲活动
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_spikes({"P": p_monitor.get("s"), "I": i_monitor.get("s"), "D": d_monitor.get("s"), "Output": output_monitor.get("s")}, axes=axes[0])
axes[0].set_title("Spiking Activity")
axes[0].legend(["P Layer", "I Layer", "D Layer", "Output Layer"])

# 可视化膜电位
plot_voltages({"P": p_monitor.get("v"), "I": i_monitor.get("v"), "D": d_monitor.get("v"), "Output": output_monitor.get("v")}, axes=axes[1])
axes[1].set_title("Membrane Potentials")
axes[1].legend(["P Layer", "I Layer", "D Layer", "Output Layer"])

plt.tight_layout()
plt.show(block=True)

# 停止网络运行，释放资源
network.reset_state_variables()
