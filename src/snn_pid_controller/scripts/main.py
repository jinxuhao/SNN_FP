import matplotlib.pyplot as plt
from layers.input_layer import InputLayer
from layers.integration_layer import IntegrationLayer
from layers.output_layer import OutputLayer

def main():
    num_neurons = 63
    threshold = 5
    max_iterations = 200
    target_angle = 10.0
    current_angle = 0.0

    input_layer = InputLayer(num_neurons=num_neurons)
    integration_layer = IntegrationLayer(num_neurons=num_neurons, threshold=threshold)
    output_layer = OutputLayer(num_neurons=num_neurons, spike_threshold=5)

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
        # 计算误差信号
        error_signal = target_angle - current_angle
        integral_signal = integration_layer.integrate_error(error_signal)

        # 通过 SNN 神经元活动计算控制信号
        control_signal = output_layer.compute_output_from_activity(error_signal, integral_signal, target_angle, current_angle)
        current_angle += control_signal  # 更新当前角度
        print(f"Time step {t+1}: Current angle: {current_angle}")

        # 存储数据用于绘图
        time_steps.append(t)
        current_angles.append(current_angle)

        # 更新图像
        current_angle_line.set_data(time_steps, current_angles)
        ax.set_xlim(0, max(t+1, max_iterations))
        ax.set_ylim(min(current_angles) - 5, max(current_angles) + 5)
        plt.pause(0.1)

        # if abs(current_angle - target_angle) < 0.5:
        #     print("Target angle reached!")
        #     break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
