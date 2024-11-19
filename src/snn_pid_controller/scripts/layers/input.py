import torch
import numpy as np

class SNNInput:
    def __init__(self, num_neurons=63, angle_range=(-40, 40), setpoint_range=(-25, 25), angular_velocity_range=(-80, 80)):
        self.num_neurons = num_neurons
        self.angle_range = angle_range
        self.setpoint_range = setpoint_range
        self.angular_velocity_range = angular_velocity_range
    
    def encode_value(self, value, min_val, max_val):
        """Maps an input value to an index within the neuron population."""
        idx = int((value - min_val) / (max_val - min_val) * (self.num_neurons - 1))
        return max(0, min(idx, self.num_neurons - 1))

    def generate_spike_input(self):
        """Generates one-hot encoded spike input for angle and setpoint."""
        # Generate random values within specified ranges
        current_angle = np.random.uniform(*self.angle_range)
        target_angle = np.random.uniform(*self.setpoint_range)

        # Encode into neuron indices
        current_idx = self.encode_value(current_angle, *self.angle_range)
        target_idx = self.encode_value(target_angle, *self.setpoint_range)

        # Create a one-hot encoded input tensor
        input_tensor = torch.zeros((1, self.num_neurons))
        input_tensor[0, current_idx] = 1
        input_tensor[0, target_idx] = 1

        return input_tensor

# Testing function for standalone use
if __name__ == "__main__":
    snn_input = SNNInput()
    input_data = snn_input.generate_spike_input()
    print(f"Generated spike input shape: {input_data.shape}")
    print(input_data)
