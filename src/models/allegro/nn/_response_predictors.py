"""Equivariant frame predictors for flat Cartesian response features."""

from typing import Sequence

import torch
from e3nn import o3
from e3nn.nn import NormActivation
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from ._fc import ScalarMLPFunction
from ._bec_readout import (
    _ambient_irreps,
    _cartesian_irreps,
    _change_of_basis,
    _symmetrize_cartesian,
)


class EquivariantDielectricPredictor(GraphModuleMixin, torch.nn.Module):
    """Scalar-gated symmetric dielectric readout, (B,9) to (B,9).

    The trace feeds a bias-free scalar MLP; its invariant gates modulate
    the expanded traceless tensor channels. Eager execution only.
    """

    def __init__(
        self,
        field: str,
        out_field: str,
        mlp_latent_dimensions: Sequence[int] = (64, 64),
        mlp_nonlinearity: str = "silu",
        tensor_hidden_channels: int = 8,
        irreps_in=None,
    ):
        super().__init__()
        if tensor_hidden_channels < 1:
            raise ValueError("tensor_hidden_channels must be positive")
        self.field = field
        self.out_field = out_field
        self.tensor_hidden_channels = tensor_hidden_channels
        self.register_buffer("_Q", _change_of_basis("ij=ji"), persistent=False)
        self.scalar_mlp = ScalarMLPFunction(
            mlp_input_dimension=1,
            mlp_latent_dimensions=list(mlp_latent_dimensions),
            mlp_output_dimension=1 + tensor_hidden_channels,
            mlp_nonlinearity=mlp_nonlinearity,
        )
        self.tensor_linear_expand = o3.Linear(
            "1x2e",
            f"{tensor_hidden_channels}x2e",
            internal_weights=True,
            shared_weights=True,
        )
        self.tensor_linear_combine = o3.Linear(
            f"{tensor_hidden_channels}x2e",
            "1x2e",
            internal_weights=True,
            shared_weights=True,
        )
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out={out_field: _ambient_irreps(2)},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        x = data[self.field]
        if x.ndim != 2 or x.shape[1] != 9:
            raise ValueError("Dielectric predictor needs flat (B, 9) input")
        irreps = x @ self._Q.T
        scalar = self.scalar_mlp(irreps[:, :1])
        expanded = self.tensor_linear_expand(irreps[:, 1:]).reshape(
            -1, self.tensor_hidden_channels, 5
        )
        gated = expanded * torch.tanh(scalar[:, 1:]).unsqueeze(-1)
        tensor = self.tensor_linear_combine(gated.reshape(x.shape[0], -1))
        data[self.out_field] = _symmetrize_cartesian(
            torch.cat([scalar[:, :1], tensor], dim=-1) @ self._Q, 2
        )
        return data


class EquivariantPiezoelectricPredictor(GraphModuleMixin, torch.nn.Module):
    """Norm-activated rank-3 readout with last-index symmetry, (B,27) to (B,27).

    Uses two vector copies, an odd rank-2 irrep and an odd rank-3 irrep.
    Eager execution only.
    """

    def __init__(self, field: str, out_field: str, irreps_in=None):
        super().__init__()
        self.field = field
        self.out_field = out_field
        self.register_buffer("_Q", _change_of_basis("ijk=ikj"), persistent=False)
        irreps = _cartesian_irreps("ijk=ikj")
        self.tensor_linear_expand = o3.Linear(
            irreps, irreps, internal_weights=True, shared_weights=True
        )
        self.tensor_norm = NormActivation(
            irreps,
            scalar_nonlinearity=torch.nn.functional.silu,
            normalize=True,
            epsilon=1e-8,
            bias=False,
        )
        self.tensor_linear_out = o3.Linear(
            irreps, irreps, internal_weights=True, shared_weights=True
        )
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out={out_field: _ambient_irreps(3)},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        x = data[self.field]
        if x.ndim != 2 or x.shape[1] != 27:
            raise ValueError("Piezoelectric predictor needs flat (B, 27) input")
        hidden = self.tensor_norm(self.tensor_linear_expand(x @ self._Q.T))
        data[self.out_field] = _symmetrize_cartesian(
            self.tensor_linear_out(hidden) @ self._Q, 3
        )
        return data
