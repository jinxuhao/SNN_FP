import torch
import numpy as np
from bindsnet.network.nodes import Nodes

class InputLayer(Nodes):
    def __init__(self, num_neurons, angle_range=(-10, 10), setpoint_range=(-10, 10)):
        """
        Input layer for encoding angles into spike signals.
        :param num_neurons: Number of neurons in the layer.
        :param angle_range: Range of current angle values (min, max).
        :param setpoint_range: Range of target angle values (min, max).
        """
        super(InputLayer, self).__init__(n=num_neurons, shape=(1, num_neurons))
        self.num_neurons = num_neurons
        self.angle_range = angle_range
        self.setpoint_range = setpoint_range
        self.s = torch.zeros(1, self.num_neurons)  # Initialize state

        # Flags and explicit input indices
        self.use_explicit_inputs = False
        self.explicit_current = None
        self.explicit_target = None

        self.last_indices = (None, None)  # Store current and target indices

    def encode_index(self, value, min_val, max_val):
        """
        Encode a given value into a neuron index based on the defined range.
        :param value: The input value to encode.
        :param min_val: Minimum possible value in the range.
        :param max_val: Maximum possible value in the range.
        :return: Encoded neuron index.
        """
        idx = int((value - min_val) / (max_val - min_val) * (self.num_neurons - 1))
        return max(0, min(idx, self.num_neurons - 1))

    def update_input(self, current_angle=None, target_angle=None):
        """
        Update the input layer's state by encoding current and target angles.
        :param current_angle: Current angle value.
        :param target_angle: Target angle value.
        :return: Indices of current and target angles.
        """
        print("TEST INPUT layer:", current_angle)
        print("TEST INPUT target_angle:", target_angle)

        if self.use_explicit_inputs and self.explicit_current is not None and self.explicit_target is not None:
            current_idx = self.encode_index(self.explicit_current, *self.angle_range)
            target_idx = self.encode_index(self.explicit_target, *self.setpoint_range)
        else:
            current_idx = self.encode_index(current_angle, *self.angle_range)
            target_idx = self.encode_index(target_angle, *self.setpoint_range)

        print("TEST INPUT target_angle index:", target_idx)

        # Reset state and update active neurons
        self.s = torch.zeros(1, self.num_neurons)  # Ensure (1, num_neurons)
        self.s[0, current_idx] = 1
        self.s[0, target_idx] = 1

        self.last_indices = (current_idx, target_idx)  # Store indices

        return current_idx, target_idx
    
    def forward(self, x, input_1_value=None, input_2_value=None, *args, **kwargs):
        """
        Forward pass function that returns the current state of the input layer.
        :param x: Input tensor (not used in current implementation).
        """
        print("TEST INPUT:", self.s)
        return self.s

    def generate_virtual_input(self):
        """
        Generate virtual input values for current and target angles.
        :return: Tuple of (current_angle, target_angle).
        """
        current_angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        target_angle = np.random.uniform(self.setpoint_range[0], self.setpoint_range[1])
        return current_angle, target_angle

    def compute_decays(self, dt):
        """Override compute_decays method to disable decay behavior."""
        pass  
    
    def set_batch_size(self, batch_size):
        """Override set_batch_size method to disable batch processing behavior."""
        pass  
