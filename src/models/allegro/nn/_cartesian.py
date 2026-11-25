import torch

from e3nn.io import CartesianTensor
from e3nn.util.jit import compile_mode


@compile_mode("script")
class Cartesian_nn(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        ct = CartesianTensor(formula)
        rtp = ct.reduced_tensor_products()
        self.Q = rtp.change_of_basis
        self.last_dim = 3 ** len(self.indices)
        self.indices = ct.indices

    def forward(self, data: torch.Tensor):
        data = (data @ self.Q.flatten(-len(self.indices))).view(-1, self.last_dim)
        return data
