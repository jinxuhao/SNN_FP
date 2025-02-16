import torch
from bindsnet.network.nodes import Nodes

class IIntermediateLayer(Nodes):
    def __init__(self, num_neurons=63):
        """
        Intermediate I layer for processing integral signals.
        :param num_neurons: Number of neurons in the layer.
        """
        super(IIntermediateLayer, self).__init__(n=num_neurons, shape=(1, num_neurons))
        self.num_neurons = num_neurons
        self.positive = torch.zeros(1, num_neurons)  # Represents I+ population
        self.negative = torch.zeros(1, num_neurons)  # Represents I- population

    def process_signal(self, x):
        """
        Processes the input signal, splitting it into positive (I+) and negative (I-) groups,
        and updates `self.s`.
        :param x: Input signal (sparse encoded).
        """
        # Reset positive, negative signals and self.s
        self.positive.zero_()
        self.negative.zero_()
        self.s.zero_()

        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        # Retrieve active neuron indices
        active_indices = torch.nonzero(x, as_tuple=True)[1].tolist()
        print(f"IIntermediateLayer - Active indices: {active_indices}")

        self.s = torch.zeros(1, self.num_neurons, dtype=torch.float)
        if len(active_indices) > 0:
            index = active_indices[0]  # Get active neuron index
            
            if index >= (self.num_neurons - 1) // 2:  # Positive signal processing
                positive_index = index - (self.num_neurons - 1) // 2
                self.positive[0, positive_index] = 1  # Update I+
                self.s[0, index] = 1  # Update self.s
            else:  # Negative signal processing
                negative_index = (self.num_neurons - 1) // 2 - index
                self.negative[0, negative_index] = 1  # Update I-
                self.s[0, index] = 1  # Update self.s

    def forward(self, x):
        """
        Receives input signal and updates `self.s`.
        :param x: Input signal.
        """
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        self.s = x
        return self.s
