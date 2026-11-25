"""Stage 0 shared geometry, bounded extrapolation, and radial frontend for SpinGNN2 v2.

All operations avoid ``abs``/``maximum`` branches whose PyTorch symbolic
derivatives (``aten.sgn``) can trigger dynamic-shape specialization during
second-order autograd under ``torch.compile(fullgraph=True, dynamic=True)``.
"""

from __future__ import annotations

import contextlib
import math
from typing import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _safe_r2_threshold(dtype: torch.dtype) -> float:
    """Dtype-safe ``r^2`` floor preventing ``1/r^5`` overflow in 2nd-order autograd."""
    if dtype == torch.float64:
        return 1e-24
    if dtype == torch.float16:
        return 1e-4
    return 1e-12


def safe_edge_geometry(
    edge_vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute edge distances and unit directions safely without ``aten.sgn``.

    Parameters
    ----------
    edge_vec : torch.Tensor
        Displacement vectors ``r_j - r_i`` with shape ``(E, 3)``.

    Returns
    -------
    r : torch.Tensor
        Euclidean lengths ``(E,)``, exact for all non-zero vectors.
    u : torch.Tensor
        Unit direction vectors ``(E, 3)``, zero for degenerate zero edges.
    """
    r_sq = (edge_vec * edge_vec).sum(dim=-1)
    nonzero = r_sq > _safe_r2_threshold(edge_vec.dtype)
    safe_r_sq = torch.where(nonzero, r_sq, torch.ones_like(r_sq))
    r_raw = torch.sqrt(safe_r_sq)
    r = torch.where(nonzero, r_raw, torch.zeros_like(r_raw))
    inv_r = torch.where(nonzero, 1.0 / r_raw, torch.zeros_like(r_raw))
    u = edge_vec * inv_r.unsqueeze(-1)
    return r, u


class C3CutoffEnvelope(nn.Module):
    r"""Cancellation-free :math:`C^3` polynomial cutoff envelope :math:`\chi_p(r)`.

    For :math:`x = r / r_{\mathrm{cut}}` and :math:`u = 1 - x`:

    .. math::

       \chi_p(r) = u^4 \sum_{k=0}^{p-1} \binom{k + 3}{3} x^k \quad (x < 1),
       \qquad \chi_p(r) = 0 \quad (x \ge 1).

    For the default ``exponent=5``:
    ``chi_5(r) = u^4 * (1 + 4x + 10x^2 + 20x^3 + 35x^4)``.
    """

    def __init__(self, r_cut: float, exponent: int = 5) -> None:
        super().__init__()
        if r_cut <= 0.0:
            raise ValueError(f"r_cut must be positive, got {r_cut}")
        if exponent < 1:
            raise ValueError(f"exponent must be >= 1, got {exponent}")
        self.r_cut = float(r_cut)
        self.exponent = int(exponent)
        self._poly_coeffs: tuple[float, ...] = tuple(
            float(math.comb(k + 3, 3)) for k in range(self.exponent)
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x_raw = r / self.r_cut
        inside = x_raw < 1.0
        x = torch.where(inside, x_raw, torch.ones_like(x_raw))
        u = torch.where(inside, 1.0 - x, torch.zeros_like(x))
        u2 = u * u
        u4 = u2 * u2

        # Horner evaluation using Python float constants (zero graph breaks)
        poly = torch.full_like(x, self._poly_coeffs[-1])
        for idx in range(self.exponent - 2, -1, -1):
            poly = poly * x + self._poly_coeffs[idx]
        return torch.where(inside, u4 * poly, torch.zeros_like(u4))


class FixedGaussianRBF(nn.Module):
    """Fixed uniform Gaussian radial basis functions on ``[0, r_cut]``.

    Registers ``centers`` in ``torch.get_default_dtype()`` (avoiding FP64->FP32
    runtime casts during FP32 training) and re-materializes exact 53-bit FP64
    centers inside :meth:`_apply` whenever transitioned to ``torch.float64``.
    """

    def __init__(self, r_cut: float, num_basis: int = 16) -> None:
        super().__init__()
        if r_cut <= 0.0:
            raise ValueError(f"r_cut must be positive, got {r_cut}")
        if num_basis < 2:
            raise ValueError(f"num_basis must be >= 2, got {num_basis}")
        self.r_cut = float(r_cut)
        self.num_basis = int(num_basis)
        centers_f64 = torch.linspace(
            0.0, self.r_cut, self.num_basis, dtype=torch.float64
        )
        spacing = float(centers_f64[1] - centers_f64[0])
        self.register_buffer("centers", centers_f64.to(dtype=torch.get_default_dtype()))
        self.inv_width_sq = 1.0 / (spacing * spacing)

    def _apply(self, fn):  # type: ignore[override]
        super()._apply(fn)
        if self.centers.dtype == torch.float64:
            exact = torch.linspace(
                0.0,
                self.r_cut,
                self.num_basis,
                dtype=torch.float64,
                device=self.centers.device,
            )
            self.centers.copy_(exact)
        return self

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        centers = (
            self.centers
            if self.centers.dtype == r.dtype
            else self.centers.to(dtype=r.dtype)
        )
        diff = r.unsqueeze(-1) - centers.unsqueeze(0)
        return torch.exp(-(diff * diff) * self.inv_width_sq)


class SwiGLUMLP(nn.Module):
    """Residual-friendly SwiGLU MLP with fused ``2*H`` gate+value GEMMs.

    Combines gate and value projections into single ``Linear(D, 2*H)`` GEMMs
    to halve CUDA GEMM launches and double Tensor Core occupancy, while omitting
    per-edge variance normalization to prevent derivative kinks at ``r -> r_cut``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)

        self.in_gate_val = nn.Linear(self.in_dim, 2 * self.hidden_dim, bias=bias)
        self.hidden_gate_vals = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, 2 * self.hidden_dim, bias=bias)
                for _ in range(self.num_layers - 1)
            ]
        )
        self.hidden_outs = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
                for _ in range(self.num_layers - 1)
            ]
        )
        self.out_proj = nn.Linear(self.hidden_dim, self.out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g0, v0 = self.in_gate_val(x).chunk(2, dim=-1)
        h = F.silu(g0) * v0
        for gv, proj in zip(self.hidden_gate_vals, self.hidden_outs):
            gk, vk = gv(h).chunk(2, dim=-1)
            h = h + proj(F.silu(gk) * vk)
        return self.out_proj(h)


class OrderedPairFiLM(nn.Module):
    """Ordered species-pair FiLM modulation with optional radial mode mixing.

    Computes ``gamma_ab * g(r) + beta_ab + sum_mu U_ab[:, mu] * q_mu(r)``
    where ``ab = type_dst * num_types + type_src`` indexes ordered pairs.
    """

    def __init__(
        self,
        num_types: int,
        channels: int,
        num_modes: int = 2,
    ) -> None:
        super().__init__()
        self.num_types = int(num_types)
        self.channels = int(channels)
        self.num_modes = int(num_modes)
        num_pairs = self.num_types * self.num_types

        self.adam_pair_gamma = nn.Parameter(torch.ones(num_pairs, self.channels))
        self.adam_pair_beta = nn.Parameter(torch.zeros(num_pairs, self.channels))
        if self.num_modes > 0:
            self.pair_mode_mix = nn.Parameter(
                torch.randn(num_pairs, self.channels, self.num_modes)
                * (1.0 / math.sqrt(float(self.num_modes * self.channels)))
            )
        else:
            self.register_parameter("pair_mode_mix", None)

    def forward(
        self,
        radial: torch.Tensor,
        pair_idx: torch.Tensor,
        radial_modes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gamma = self.adam_pair_gamma.index_select(0, pair_idx)
        beta = self.adam_pair_beta.index_select(0, pair_idx)
        if gamma.dtype != radial.dtype:
            gamma = gamma.to(dtype=radial.dtype)
            beta = beta.to(dtype=radial.dtype)
        out = gamma * radial + beta
        if self.pair_mode_mix is not None and radial_modes is not None:
            mix = self.pair_mode_mix.index_select(0, pair_idx)
            if mix.dtype != radial.dtype:
                mix = mix.to(dtype=radial.dtype)
            out = out + (mix * radial_modes.unsqueeze(1)).sum(dim=-1)
        return out


class DensityFloorNorm(nn.Module):
    r"""Measured relative density-floor envelope normalization.

    Computes per-center normalizers:

    .. math::

       n^{(0)}_i = \left(\sum_{j \in \mathcal{N}(i)} \chi_{ij}^2 + f\right)^{-1/2},
       \qquad
       n^{(+)}_i = \left(\sum_{j \in \mathcal{N}(i)} \chi_{ij}^4 + f\right)^{-1/2},

    where ``f = floor_factor * envelope_mass`` prevents isolated atoms or
    low-coordination dimers from amplifying neighbor features.
    """

    def __init__(
        self,
        floor_factor: float = 8.0,
        envelope_mass: float = 1.0,
    ) -> None:
        super().__init__()
        if floor_factor <= 0.0:
            raise ValueError("floor_factor must be positive")
        if envelope_mass <= 0.0:
            raise ValueError("envelope_mass must be positive")
        self.floor_factor = float(floor_factor)
        self.register_buffer(
            "envelope_mass",
            torch.tensor(float(envelope_mass), dtype=torch.get_default_dtype()),
        )

    @property
    def floor_value(self) -> torch.Tensor:
        return self.envelope_mass * self.floor_factor

    def forward(
        self,
        chi: torch.Tensor,
        dst: torch.Tensor,
        num_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chi2 = chi * chi
        chi4 = chi2 * chi2
        packed = torch.stack((chi2, chi4), dim=-1)
        sums = torch.zeros((num_atoms, 2), dtype=chi.dtype, device=chi.device)
        sums = sums.index_add(0, dst, packed)
        floor = self.floor_value
        if floor.dtype != chi.dtype:
            floor = floor.to(dtype=chi.dtype)
        norms = torch.rsqrt(sums + floor)
        return norms[:, 0], norms[:, 1]


class VacuumRefEnergyHead(nn.Module):
    r"""SwiGLU energy readout anchored to exact isolated-atom vacuum reference.

    Evaluates:

    .. math::

       E_i = f(D_i) - f(D_{\mathrm{vac}}(Z_i)) + E_{\mathrm{iso}}(Z_i),

    guaranteeing that an isolated atom of species :math:`Z_i` has exact energy
    :math:`E_{\mathrm{iso}}(Z_i)` with zero residual bias.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_types: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.net = SwiGLUMLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=num_layers,
        )
        self.adam_iso_energy = nn.Parameter(torch.zeros(num_types))

    def forward(
        self,
        desc: torch.Tensor,
        vac_desc_per_type: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> torch.Tensor:
        raw_e = self.net(desc).squeeze(-1)
        vac_e_per_type = self.net(vac_desc_per_type).squeeze(-1)
        vac_e = vac_e_per_type.index_select(0, atom_types)
        iso_e = self.adam_iso_energy.index_select(0, atom_types)
        if iso_e.dtype != desc.dtype:
            iso_e = iso_e.to(dtype=desc.dtype)
        return raw_e - vac_e + iso_e


@contextlib.contextmanager
def edge_bf16_autocast(enabled: bool, tensor: torch.Tensor) -> Iterator[None]:
    """Enable edge-side ``bfloat16`` AMP while keeping FP64 runs untouched."""
    use_amp = bool(enabled) and tensor.dtype != torch.float64
    if not use_amp:
        yield
        return
    device_type = "cuda" if tensor.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        yield


def maybe_edge_checkpoint(
    fn: Callable[..., torch.Tensor],
    enabled: bool,
    *args: torch.Tensor,
) -> torch.Tensor:
    """Apply selective non-reentrant activation checkpointing on edge MLPs."""
    if enabled and torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)
