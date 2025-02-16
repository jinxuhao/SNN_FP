import torch
from bindsnet.network.nodes import Nodes

class PIntermediateLayer(Nodes):
    def __init__(self, num_neurons=63):
        """
        P intermediate layer for processing P signals.
        :param num_neurons: Number of neurons in the layer.
        """
        super(PIntermediateLayer, self).__init__(n=num_neurons, shape=(1, num_neurons))
        self.num_neurons = num_neurons
        self.positive = torch.zeros(1, num_neurons)  # Represents P+ group
        self.negative = torch.zeros(1, num_neurons)  # Represents P- group

    def process_signal(self, x):
        """
        Processes the input signal, splitting it into positive and negative groups,
        and updates `self.s`.
        :param x: Input signal (sparse encoding).
        """
        print(f"PIntermediateLayer - Received input x: {x}")

        # Reset positive, negative signals and self.s
        self.positive.zero_()
        self.negative.zero_()
        self.s.zero_()

        # Ensure correct input shape
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        # Retrieve active neuron indices
        active_indices = torch.nonzero(x, as_tuple=True)[1].tolist()
        print(f"PIntermediateLayer - Active indices: {active_indices}")

        self.s = torch.zeros(1, self.num_neurons, dtype=torch.float)

        if len(active_indices) > 0:
            index = active_indices[0]  # Get active neuron index
            if index >= self.num_neurons // 2:  # Positive signal processing
                self.positive[0, index - self.num_neurons // 2] = 1
                self.s[0, index] = 1  # Update self.s
            else:  # Negative signal processing
                self.negative[0, self.num_neurons // 2 - index] = 1
                self.s[0, index] = 1  # Update self.s

    def forward(self, x):
        """
        Receives input signal, updates `self.s`, and returns it.
        :param x: Input signal.
        """
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        self.s = x
        return self.s
