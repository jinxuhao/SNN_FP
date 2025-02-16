import torch
from bindsnet.network.nodes import Nodes

class DIntermediateLayer(Nodes):
    def __init__(self, num_neurons=63, scaling_factor=1.0):
        """
        D Layer: Computes output based on the rate of change of error (Delta e_t).
        :param num_neurons: Number of neurons in the layer.
        :param scaling_factor: Scaling factor for mapping Delta e_t to neuron indices.
        """
        super(DIntermediateLayer, self).__init__(n=num_neurons)
        self.num_neurons = num_neurons
        self.scaling_factor = scaling_factor
        self.previous_error = None  # Stores the error value from the previous timestep
        self.s = torch.zeros(1, num_neurons)  # Initialize state
    
    def forward(self, x):
        """
        Forward propagation: Receives input and computes Delta e_t.
        :param x: Input signal (one-hot encoded).
        """
        # Ensure input shape is correct
        if x.dim() == 3:
            x = x.squeeze(1)

        # Get the current error index
        active_indices = torch.nonzero(x, as_tuple=True)[1].tolist()

        if len(active_indices) < 1:
            print("DIntermediateLayer Warning: No active neurons in the input.")
            return self.s  # Return current state if no input is detected

        # Decode the current error value
        current_error_index = active_indices[0]
        current_error = current_error_index - (self.num_neurons - 1) // 2  # Centered around 0
        print(f"DIntermediateLayer Current error: {current_error}")

        # Compute Delta e_t
        if self.previous_error is not None:
            delta_e_t = current_error - self.previous_error
            print(f"DIntermediateLayer Delta e_t: {delta_e_t}")
        else:
            delta_e_t = 0  # No change during initialization

        # Update previous_error
        self.previous_error = current_error

        # Map Delta e_t to neuron indices
        delta_index = int(delta_e_t * self.scaling_factor) + (self.num_neurons - 1) // 2
        delta_index = max(0, min(delta_index, self.num_neurons - 1))  # Ensure index stays within bounds

        # Update self.s
        self.s.fill_(0)
        self.s[0, delta_index] = 1

        return self.s

    def compute_decays(self, dt):
        """Override compute_decays to prevent automatic state decay."""
        pass

    def set_batch_size(self, batch_size):
        """Override set_batch_size to prevent batch operations from affecting behavior."""
        pass
