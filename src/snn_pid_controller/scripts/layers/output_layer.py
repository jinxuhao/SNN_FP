import torch
from bindsnet.network.nodes import DiehlAndCookNodes

class ComplexOutputLayer(DiehlAndCookNodes):  
    def __init__(self, num_neurons=63, Kp=0.1, Ki=0.05, Kd=0.01):
        """
        Complex output layer inheriting from DiehlAndCookNodes.
        :param num_neurons: Number of neurons in the layer.
        :param Kp: Proportional gain.
        :param Ki: Integral gain.
        :param Kd: Derivative gain.
        """
        super(ComplexOutputLayer, self).__init__(n=num_neurons, thresh=1.0, reset=0)  
        self.num_neurons = num_neurons
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # Initialize potential values for P+, P-, I+, I-, and B groups
        self.P_plus = torch.zeros(num_neurons)
        self.P_minus = torch.zeros(num_neurons)
        self.I_plus = torch.zeros(num_neurons)
        self.I_minus = torch.zeros(num_neurons)
        self.B = torch.zeros(num_neurons)
        self.U = torch.zeros(num_neurons)

        # Set thresholds and bias current
        self.thresholds = 0  
        self.bias_current = 0  
        self.s = torch.zeros(self.num_neurons)  # Initialize self.s as a zero tensor

    def integrate_signals(self, P_signal, I_signal, D_signal=0):
        """
        Assigns P and I signals to respective neuron groups and updates B group potential.
        :param P_signal: Proportional signal.
        :param I_signal: Integral signal.
        :param D_signal: Derivative signal (default = 0).
        """
        if P_signal > 0:
            self.P_plus += P_signal
        else:
            self.P_minus += -P_signal

        if I_signal > 0:
            self.I_plus += I_signal
        else:
            self.I_minus += -I_signal

        # Update B group potential
        self.B += abs(self.Kp) * (self.P_plus - self.P_minus)
        self.B += abs(self.Ki) * (self.I_plus - self.I_minus)
        self.B += self.bias_current

        # Clamp potential values within range
        self.B = torch.clamp(self.B, min=0, max=100)

    def compute_output(self):
        """
        Computes the output control signal by activating the corresponding neuron in U group.
        """
        active_indices = (self.B >= self.thresholds).nonzero(as_tuple=True)[0]
        
        if len(active_indices) > 0:
            max_index = active_indices.max().item()
            self.U.zero_()  # Reset activation in U group
            self.U[max_index] = 1
            return max_index  # Return control signal
        else:
            return 0  # Return 0 if no neurons are activated

    def forward(self, x):
        """
        Processes input B group potentials and computes U group output (spiking signal).
        Extracts all active neuron indices and their values from B group, computes a weighted 
        average, and maps it to a new index.
        :param x: Input potential of B group, shape [num_neurons].
        :return: Spiking signal from U group and new index value.
        """
        # Ensure input is a 1D tensor and update self.B
        self.B = x.squeeze()

        # Identify active neurons exceeding the threshold
        active_indices = (self.B >= self.thresholds).nonzero(as_tuple=True)[0]
        active_values = self.B[active_indices]  

        # Initialize self.s (spiking signal for U group)
        self.s.zero_()

        if len(active_indices) > 0:
            num_neurons = self.B.shape[0]
            max_range = 40  # Define half of the value range
            mid_index = (num_neurons - 1) // 2

            # Map indices to values
            mapped_values = (active_indices.float() - mid_index)
            # Compute weighted sum for weighted average
            weighted_value = (mapped_values * active_values).sum()

            # Map weighted value back to index
            new_index = mid_index + weighted_value
            print(f"CALCULATED - New Index: {new_index}")

            new_index = int(torch.round(new_index))  # Round to nearest integer index
            new_index = max(0, min(new_index, num_neurons - 1))  # Clamp within valid range
            print(f"CALCULATED - Mapped values: {mapped_values}, Active values: {active_values}")
            print(f"CALCULATED - Weighted Value: {weighted_value:.2f}, Final Index: {new_index}")

            # Activate the corresponding neuron in U group
            self.s[new_index] = 1  

            return self.s
        else:
            # If no active neurons, return default index 0
            print("CALCULATED - No active neurons. Returning default index: 0.")
            return self.s, 0

    def compute_decays(self, dt):
        """Override compute_decays method to disable decay behavior."""
        pass  

    def set_batch_size(self, batch_size):
        """Override set_batch_size method to disable batch processing behavior."""
        pass  
