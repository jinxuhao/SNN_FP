import numpy as np
import torch
from bindsnet.network.nodes import Nodes

class EncodingLayer(Nodes):
    def __init__(self, num_neurons=63, e_t_range=(-20, 20)):
        """
        Encoding layer that computes mathematical operations using 2D arrays.
        :param num_neurons: Number of neurons in the 1D input and output populations.
        :param e_t_range: The range of e_t values (min, max).
        """
        super(EncodingLayer, self).__init__(n=num_neurons, shape=(1, num_neurons))
        self.num_neurons = num_neurons
        self.e_t_range = e_t_range  # (min, max)
        self.operation_array = self.create_2D_operation_array()

        # Flags for explicit indexing
        self.use_indices = False
        self.y_index = None
        self.r_index = None

    def create_2D_operation_array(self):
        """
        Creates a 2D operation array for subtraction.
        Each diagonal corresponds to a specific e_t value.
        """
        operation_array = np.zeros((self.num_neurons, self.num_neurons), dtype=int)
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                # Compute the diagonal index
                diagonal_index = j - i
                operation_array[i, j] = diagonal_index
        return operation_array

    def map_e_t_to_neuron(self, e_t):
        """
        Maps an e_t value to the corresponding neuron index in the 1D output array.
        """
        e_t_min, e_t_max = self.e_t_range
        e_t_clipped = max(e_t_min, min(e_t, e_t_max))  # Clip e_t to range
        # Convert e_t to a neuron index
        neuron_index = int(round((e_t_clipped - e_t_min) / (e_t_max - e_t_min) * (self.num_neurons - 1)))
        return neuron_index

    def forward(self, x):
        """
        Forward pass: Computes the encoded output.
        :param x: A tensor of shape (1, num_neurons), representing input spikes.
        """
        x = x.float()

        # Ensure x is in the correct shape
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        # Use explicitly passed indices
        if self.use_indices and self.y_index is not None and self.r_index is not None:
            y_index = self.y_index
            r_index = self.r_index
            print(f"Encoding - Using explicit indices: y_index={y_index}, r_index={r_index}")
        else:
            # Extract active neuron indices
            active_indices = torch.nonzero(x, as_tuple=True)[1].tolist()
            if len(active_indices) < 2:
                print("Encoding - Warning: Less than two active inputs for encoding.")
                self.s = torch.zeros(1, self.num_neurons, dtype=torch.float)  # Reset state
                return self.s

            # Assign y_index and r_index
            y_index, r_index = active_indices[:2]
            print(f"Encoding - Extracted indices: y_index={y_index}, r_index={r_index}")

        # Compute the diagonal index from the operation array
        diagonal_index = self.operation_array[y_index, r_index]

        # Map diagonal index to e_t value
        e_t_value = ((self.num_neurons - 1) + diagonal_index) / (2 * (self.num_neurons - 1)) * (self.e_t_range[1] - self.e_t_range[0])
        e_t_value += self.e_t_range[0]
        print(f"Encoding - Computed e_t value: {e_t_value}")

        # Convert e_t value to neuron index
        result_index = self.map_e_t_to_neuron(e_t_value)
        print(f"Encoding - Resulting neuron index: {result_index}")

        # Update the output state
        self.s = torch.zeros(1, self.num_neurons, dtype=torch.float)
        self.s[0, result_index] = 1

        return self.s
