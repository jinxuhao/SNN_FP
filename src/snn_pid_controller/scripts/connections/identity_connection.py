from bindsnet.network.topology import AbstractConnection
import torch

class IdentityConnection(AbstractConnection):
    def __init__(self, source, target, **kwargs):
        """
        An identity connection that directly transfers source activity to the target.
        """
        super().__init__(source=source, target=target, **kwargs)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        """
        Directly pass the source activity to the target.
        :param s: Spiking activity of the source layer.
        :return: Same activity for the target layer.
        """
        return s
