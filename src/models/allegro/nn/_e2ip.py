"""Equivariant vector readout shared by structural and SOC e2IP cores."""
import torch
from e3nn import o3
from e3nn.nn import NormActivation
from ._strided import Linear


class EquivariantVectorHead(torch.nn.Sequential):
    def __init__(self, irreps, multiplicity, channels, alignment=1):
        if alignment != 1:
            raise ValueError("e2IP currently requires pad_to_alignment: 1")
        irreps = o3.Irreps(irreps)
        if channels < 1 or o3.Irrep("1o") not in irreps:
            raise ValueError(
                "e2IP requires positive channels and a reachable 1o path (l_max >= 1)"
            )
        base = o3.Irreps([(1, ir) for _, ir in irreps])
        super().__init__(
            Linear(
                [(multiplicity, ir) for _, ir in irreps],
                base,
                shared_weights=True,
                internal_weights=True,
                pad_to_alignment=alignment,
            ),
            NormActivation(
                base, torch.nn.functional.silu, normalize=True, epsilon=1e-8, bias=False
            ),
            Linear(
                base,
                f"{channels}x1o",
                shared_weights=True,
                internal_weights=True,
                pad_to_alignment=alignment,
            ),
        )
