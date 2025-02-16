import torch
from bindsnet.network.nodes import Nodes

class IntegrationLayer(Nodes):
    def __init__(self, num_neurons=63, scale_factor=1.0):
        """
        Integration layer that updates position based on the error signal, mimicking the integral part of a PID controller.
        :param num_neurons: Number of neurons in the I layer.
        :param scale_factor: Scaling factor adjusting the effect of the error signal on position updates.
        """
        super(IntegrationLayer, self).__init__(n=num_neurons, shape=(1, num_neurons))
        self.num_neurons = num_neurons
        self.scale_factor = scale_factor

        # Initialize I-layer state
        self.I = torch.zeros(1, num_neurons)
        self.I[0, (num_neurons - 1) // 2] = 1  # Initial activation at the center

    def forward(self, x):
        """
        Receives error signal and updates position accordingly.
        :param x: Input signal from EncodingLayer in one-hot encoding.
        """
        # Ensure correct input shape
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        # Check if input signal is valid
        active_indices = torch.nonzero(x, as_tuple=True)[1].tolist()
        if len(active_indices) < 1:
            # Maintain current state if input is empty
            self.s = self.I.clone()
            return self.s

        # Get active neuron index
        e_t_index = active_indices[0]
        middle_index = (self.num_neurons - 1) // 2

        # Compute direct index change
        increment = (e_t_index - middle_index) / self.scale_factor

        # Get current active index
        current_index = torch.argmax(self.I).item()

        # Update position while handling boundaries
        new_index = int(current_index + increment)
        new_index = max(0, min(self.num_neurons - 1, new_index))  # Restrict within valid range

        # Update I-layer state
        self.I.zero_()
        self.I[0, new_index] = 1  # Activate new position

        # Update self.s to be compatible with BindsNET
        self.s = self.I.clone()
        return self.s