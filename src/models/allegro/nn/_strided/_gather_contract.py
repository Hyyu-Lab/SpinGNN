import math
from typing import List, Optional, Sequence, Tuple

import torch
from e3nn import o3


class GatherContracter(torch.nn.Module):
    """Path-weighted strided tensor product of ``x1`` with gathered rows of ``x2``.

    ``out[e] = TP(x1[e], x2[idxs[e]])``, where both inputs and the output are
    strided ``(batch, mul, irreps.dim)`` tensors with one copy of each irrep
    per channel. The weights couple paths and channels (shape ``(mul,
    num_paths)``) or only paths (``(num_paths,)``); a single path drops the path
    dimension. Each path's Wigner-3j block is multiplied by
    ``sqrt(2 l_out + 1)`` ("component" normalization) and no other
    normalization or scatter factor is applied.

    These are the semantics of the gather-only ``Contracter`` of
    ``nequip-allegro`` 0.8.3 (same default path enumeration, weight layout,
    initialization and arithmetic order), unlike :func:`Contracter` of this
    package, which generates fused, variance-normalized FX code. The Wigner-3j
    coefficients are computed in float64 and stored in the default dtype.

    Args:
        irreps_in1, irreps_in2, irreps_out: per-channel irreps (multiplicity 1 each).
        mul: number of channels of both inputs and of the output.
        instructions: ``(i_in1, i_in2, i_out)`` paths; by default every allowed
            path, enumerated output-major, then ``i_in1``, then ``i_in2``.
        path_channel_coupling: whether the weights carry a channel index.
        irrep_normalization: ``"component"`` or ``None``.
        memory_efficient: opt into batched contraction without the output-sized
            broadcast product; changes floating-point reduction order.
    """

    mul: int
    base_dim1: int
    base_dim2: int
    base_dim_out: int
    num_paths: int
    w3j_is_ij_diagonal: bool
    path_channel_coupling: bool
    _weight_w3j_einstr: str

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        mul: int,
        instructions: Optional[Sequence[Tuple[int, int, int]]] = None,
        path_channel_coupling: bool = True,
        irrep_normalization: Optional[str] = "component",
        memory_efficient: bool = False,
    ):
        super().__init__()
        if mul < 1:
            raise ValueError(f"mul must be positive, got {mul}")
        if irrep_normalization not in ("component", None):
            raise NotImplementedError(
                f"irrep_normalization={irrep_normalization!r} is not implemented"
            )
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        for name, irreps in (
            ("irreps_in1", self.irreps_in1),
            ("irreps_in2", self.irreps_in2),
            ("irreps_out", self.irreps_out),
        ):
            if any(mul_ir.mul != 1 for mul_ir in irreps):
                raise ValueError(f"{name} must have multiplicity 1 per irrep: {irreps}")

        if instructions is None:
            instructions = [
                (i_1, i_2, i_out)
                for i_out, (_, ir_out) in enumerate(self.irreps_out)
                for i_1, (_, ir_1) in enumerate(self.irreps_in1)
                for i_2, (_, ir_2) in enumerate(self.irreps_in2)
                if ir_out in ir_1 * ir_2
            ]
        self.instructions: List[Tuple[int, int, int]] = [
            (int(a), int(b), int(c)) for a, b, c in instructions
        ]
        self.num_paths = len(self.instructions)
        if self.num_paths == 0:
            raise ValueError("No tensor-product paths available")
        self.memory_efficient = bool(memory_efficient)
        self.mul = int(mul)
        self.base_dim1 = self.irreps_in1.dim
        self.base_dim2 = self.irreps_in2.dim
        self.base_dim_out = self.irreps_out.dim
        self.irrep_normalization = irrep_normalization

        # -- Wigner 3j blocks, gathered into one dense (path, i, [j,] k) tensor --
        w3j_index: List[torch.Tensor] = []
        w3j_values: List[torch.Tensor] = []
        for i_1, i_2, i_out in self.instructions:
            ir_1 = self.irreps_in1[i_1].ir
            ir_2 = self.irreps_in2[i_2].ir
            ir_out = self.irreps_out[i_out].ir
            if ir_out not in ir_1 * ir_2:
                raise ValueError(f"Path {ir_1} x {ir_2} -> {ir_out} is not allowed")
            block = o3.wigner_3j(ir_1.l, ir_2.l, ir_out.l, dtype=torch.float64)
            index = block.nonzero()
            values = block[index[:, 0], index[:, 1], index[:, 2]]
            if irrep_normalization == "component":
                values = values * math.sqrt(2 * ir_out.l + 1)
            index[:, 0] += self.irreps_in1[:i_1].dim
            index[:, 1] += self.irreps_in2[:i_2].dim
            index[:, 2] += self.irreps_out[:i_out].dim
            w3j_index.append(index)
            w3j_values.append(values)

        # all paths diagonal in (i, j): the outer product reduces to x1 * x2
        self.w3j_is_ij_diagonal = self.base_dim1 == self.base_dim2 and all(
            bool(torch.all(index[:, 0] == index[:, 1])) for index in w3j_index
        )
        if self.w3j_is_ij_diagonal:
            w3j = torch.zeros(
                self.num_paths, self.base_dim1, self.base_dim_out, dtype=torch.float64
            )
            for p, (index, values) in enumerate(zip(w3j_index, w3j_values)):
                w3j[p, index[:, 0], index[:, 2]] = values
        else:
            w3j = torch.zeros(
                self.num_paths,
                self.base_dim1,
                self.base_dim2,
                self.base_dim_out,
                dtype=torch.float64,
            )
            for p, (index, values) in enumerate(zip(w3j_index, w3j_values)):
                w3j[p, index[:, 0], index[:, 1], index[:, 2]] = values
        if self.num_paths == 1:
            w3j = w3j.squeeze(0)
        self.register_buffer("w3j", w3j.to(torch.get_default_dtype()))

        # -- path (and channel) mixing weights --
        self.path_channel_coupling = bool(path_channel_coupling)
        weight_shape: Tuple[int, ...] = (
            (self.mul,) if self.path_channel_coupling else ()
        )
        if self.num_paths > 1:
            weight_shape = weight_shape + (self.num_paths,)
        self.weights = torch.nn.Parameter(torch.randn(weight_shape))
        torch.nn.init.uniform_(self.weights, -math.sqrt(3), math.sqrt(3))

        ij = "i" if self.w3j_is_ij_diagonal else "ij"
        p = "p" if self.num_paths > 1 else ""
        u = "u" if self.path_channel_coupling else ""
        self._weight_w3j_einstr = f"{u}{p},{p}{ij}k->{u}{ij}k"

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, idxs: torch.Tensor
    ) -> torch.Tensor:
        x1 = x1.reshape(-1, self.mul, self.base_dim1)
        x2 = x2.reshape(-1, self.mul, self.base_dim2)
        x2 = torch.index_select(x2, 0, idxs)
        ww3j = torch.einsum(self._weight_w3j_einstr, self.weights, self.w3j)
        if self.memory_efficient and self.path_channel_coupling:
            # Contract over input components without an (E, F, d1, d2, K)
            # broadcast product. Explicit bmm fixes the contraction order.
            if self.w3j_is_ij_diagonal:
                outer = x1 * x2
                return torch.bmm(outer.transpose(0, 1), ww3j).transpose(0, 1)
            outer = x1.unsqueeze(-1) * x2.unsqueeze(-2)
            return torch.bmm(
                outer.flatten(2).transpose(0, 1),
                ww3j.reshape(
                    self.mul, self.base_dim1 * self.base_dim2, self.base_dim_out
                ),
            ).transpose(0, 1)
        if self.w3j_is_ij_diagonal:
            outer = x1 * x2
            if self.path_channel_coupling:
                # zui1, uik -> zuk
                out = torch.sum(outer.unsqueeze(-1) * ww3j, 2)
            else:
                out = torch.mm(outer.view(-1, outer.size(2)), ww3j).view(
                    outer.size(0), outer.size(1), ww3j.size(1)
                )
        else:
            outer = x1.unsqueeze(-1) * x2.unsqueeze(-2)
            if self.path_channel_coupling:
                # zuij1, uijk -> zuk
                out = torch.sum(outer.unsqueeze(-1) * ww3j, (2, 3))
            else:
                out = torch.mm(
                    outer.view(
                        outer.size(0) * outer.size(1), outer.size(2) * outer.size(3)
                    ),
                    ww3j.view(-1, ww3j.size(2)),
                ).view(-1, self.mul, ww3j.size(2))
        return out

    def extra_repr(self) -> str:
        text = (
            f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out} | "
            f"{self.mul} channels | {self.num_paths} paths"
        )
        if self.memory_efficient:
            text += " | memory_efficient=True"
        return text
