import matplotlib.pyplot as plt
from layers.input_layer import InputLayer
from layers.integration_layer import IntegrationLayer
from layers.output_layer import OutputLayer

def main():
    num_neurons = 63
    threshold = 5
    max_iterations = 100
    target_angle = 20.0
    current_angle = 0.0

    input_layer = InputLayer(num_neurons=num_neurons)
    integration_layer = IntegrationLayer(num_neurons=num_neurons, threshold=threshold)
    output_layer = OutputLayer(num_neurons=num_neurons, p_gain=0.1, i_gain=0.05)

    # 设置实时可视化
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(0, max_iterations)
    ax.set_ylim(min(current_angle, target_angle) - 5, max(current_angle, target_angle) + 5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Angle")
    ax.plot([0, max_iterations], [target_angle, target_angle], "r--", label="Target angle")
    current_angle_line, = ax.plot([], [], "b-", label="Current angle")
    ax.legend()

    time_steps = []
    current_angles = []

    for t in range(max_iterations):
        # 计算积分信号
        error_signal = target_angle - current_angle
        integral_signal = integration_layer.integrate_error(error_signal)

        # 计算新的角度
        current_angle = output_layer.compute_new_angle(current_angle, target_angle, integral_signal)
        print(f"Time step {t+1}: Current angle: {current_angle}")

        # 存储数据用于绘图
        time_steps.append(t)
        current_angles.append(current_angle)

        # 更新图像
        current_angle_line.set_data(time_steps, current_angles)
        ax.set_xlim(0, max(t+1, max_iterations))
        ax.set_ylim(min(current_angles) - 5, max(current_angles) + 5)
        plt.pause(0.1)

        # 检查是否接近目标角度
        # if abs(current_angle - target_angle) < 0.5:
        #     print("Target angle reached!")
        #     break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
