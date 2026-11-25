"""Physical tensor and uncertainty readout heads for SpinGNN2 v2.

Includes:
- :class:`BornEffectiveChargeHead`: Even-parity DREAM Born effective charge
  tensor :math:`Z_i^* \\in 0ee \\oplus 1ee \\oplus 2ee` with soft/hard acoustic
  sum rule (ASR) and optional magnetoelectric spin coupling.
- :class:`EvidentialHead`: Pure-PyTorch e2SLP Normal-Inverse-Wishart evidential
  force and magnetic-torque uncertainty head (``sigma``, ``Uepi``, ``nu``,
  ``kappa``, ``uncertainty``).
- :class:`ExtrapolationGrade`: Lightweight GRACE random-projection D-optimality
  atomic extrapolation grade :math:`\\gamma_i`.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage0 import SwiGLUMLP


def _linear_tanh_damper(
    x: torch.Tensor, transition: float, ceiling: float
) -> torch.Tensor:
    span = ceiling - transition
    damped = span * torch.tanh((x - transition) / span) + transition
    return torch.where(x < transition, x, damped)


def _sym3x3_matrix_exp(
    iso_scalar: torch.Tensor,
    stf: torch.Tensor,
    eye3: torch.Tensor,
) -> torch.Tensor:
    """CUDA-Graph-safe, zero-sync, machine-precision exp(iso * I + stf) for 3x3 STF.

    Because ``iso * I`` commutes with ``stf`` and ``||stf||_F <= 8`` (enforced by
    :func:`_linear_tanh_damper`), scaling by ``2^-5 = 1/32`` gives ``||A||_F <= 0.25``;
    a 12-term Horner Taylor series has truncation error ``< 3e-18`` (< FP64 eps)
    followed by 5 exact squarings. Eliminates ``aten.linalg_matrix_exp``'s internal
    CPU-GPU synchronization so ``e2slp`` runs cleanly inside CUDA Graphs.
    """
    a = stf * (1.0 / 32.0)
    eye_b = eye3.unsqueeze(0).expand_as(a)
    res = eye_b + a * (1.0 / 12.0)
    for k in (11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1):
        res = eye_b + torch.bmm(a, res) * (1.0 / float(k))
    for _ in range(5):
        res = torch.bmm(res, res)
    return torch.exp(iso_scalar).view(-1, 1, 1) * res


def _segment_sum(
    values: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    if num_graphs == 1:
        return values.sum(dim=0, keepdim=True)
    out_shape = (num_graphs,) + tuple(values.shape[1:])
    out = torch.zeros(out_shape, dtype=values.dtype, device=values.device)
    return out.index_add(0, batch, values)


def apply_acoustic_sum_rule(
    bec: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    mode: str = "soft",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply soft, hard, or no acoustic sum rule (ASR) per structure.

    Parameters
    ----------
    bec : torch.Tensor
        Per-atom Born effective charges with shape ``(N, 3, 3)``.
    batch : torch.Tensor
        Graph membership indices ``(N,)`` in ``[0, num_graphs)``.
    num_graphs : int
        Number of structures in the batch.
    mode : str
        One of ``'soft'``, ``'hard'``, or ``'none'``.

    Returns
    -------
    bec_out : torch.Tensor
        Per-atom BEC ``(N, 3, 3)`` (zero-mean per structure when ``mode='hard'``).
    frame_mean : torch.Tensor
        Pre-ASR per-structure mean ``(num_graphs, 3, 3)`` (or post-ASR zero
        when ``mode='hard'`` — wait: we return the frame mean of ``bec_out``
        for consistency with DREAM supervision).
    """
    if mode not in ("soft", "hard", "none"):
        raise ValueError(f"Unknown ASR mode {mode!r}; expected 'soft', 'hard', 'none'")
    ones = torch.ones(bec.shape[0], dtype=bec.dtype, device=bec.device)
    counts = _segment_sum(ones, batch, num_graphs).clamp_min(1.0)
    raw_sum = _segment_sum(bec, batch, num_graphs)
    raw_mean = raw_sum / counts.view(-1, 1, 1)
    if mode == "hard":
        bec_out = bec - raw_mean.index_select(0, batch)
        post_mean = _segment_sum(bec_out, batch, num_graphs) / counts.view(-1, 1, 1)
        return bec_out, post_mean
    return bec, raw_mean


def build_l2_to_stf_matrix(
    y2_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Derive exact FP64 ``(5, 9)`` orthonormal map from ``Y^2`` to STF 3x3.

    Matches the exact ordering, phase, and normalization of ``y2_fn`` (which
    maps ``(B, 3)`` unit vectors to ``(B, 5)`` real ``l=2`` spherical harmonics)
    so that ``q @ basis`` produces a symmetric traceless ``3x3`` matrix
    satisfying ``(D^(2)(R) q) @ basis == R (q @ basis) R^T`` to machine
    precision.
    """
    dirs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [2.0, 1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    dirs = dirs / torch.linalg.vector_norm(dirs, dim=-1, keepdim=True)
    y2 = y2_fn(dirs).to(dtype=torch.float64)  # (K, 5)
    eye3 = torch.eye(3, dtype=torch.float64)
    # Canonical traceless symmetric tensor T(u) = sqrt(3/2) * (u u^T - I/3)
    stf = math.sqrt(1.5) * (
        dirs.unsqueeze(-1) * dirs.unsqueeze(-2) - (1.0 / 3.0) * eye3.unsqueeze(0)
    )
    stf_flat = stf.reshape(dirs.shape[0], 9)  # (K, 9)
    # Solve y2 @ B = stf_flat in least squares (exact rank-5 linear system)
    basis = torch.linalg.lstsq(y2, stf_flat).solution  # (5, 9)
    # Verify reconstruction and normalize rows so basis @ basis^T == c * I_5
    resid = torch.max(torch.abs(y2 @ basis - stf_flat)).item()
    if resid > 1e-12:
        raise RuntimeError(f"Failed to derive exact l=2 -> STF basis (resid={resid})")
    row_norms = torch.linalg.vector_norm(basis, dim=-1, keepdim=True)
    basis = basis / row_norms
    return basis


class BornEffectiveChargeHead(nn.Module):
    r"""Even-parity DREAM Born effective charge tensor head (:math:`0ee \oplus 1ee \oplus 2ee`).

    Maps node-level even-parity representations:
    - ``h_0e``: ``(N, C_0)`` (``0ee`` isotropic scalar)
    - ``h_1e``: ``(N, 3, C_1e)`` (``1ee`` axial vector, e.g. from ``1o x 1o``)
    - ``h_2e``: ``(N, 5, C_2e)`` (``2ee`` quadrupole)
    into the 3x3 Cartesian polarization-derivative tensor :math:`Z_i^*` via an
    exact orthogonal change of basis, followed by per-species scaling (zero
    shift for exact :math:`O(3)` equivariance) and optional soft/hard ASR.

    When magnetic spin features are supplied, supports optional zero-initialized
    magnetoelectric coupling (preserving exact nonmagnetic init and exact
    :math:`O(3) \times \mathbb{Z}_2^T` or :math:`O(3)_{\mathrm{lat}} \times
    O(3)_{\mathrm{spin}}` symmetry).
    """

    def __init__(
        self,
        c0: int,
        c1e: int,
        c2e: int,
        num_types: int,
        y2_fn: Callable[[torch.Tensor], torch.Tensor],
        asr: str = "soft",
        per_species_scale: list[float] | float = 1.0,
        per_species_scale_trainable: bool = True,
        mag_c0: int = 0,
        mag_c1e: int = 0,
        mag_c2e: int = 0,
    ) -> None:
        super().__init__()
        if asr not in ("soft", "hard", "none"):
            raise ValueError(f"Unknown asr={asr!r}")
        self.asr = str(asr)
        self.num_types = int(num_types)

        self.proj_0e = nn.Linear(c0, 1, bias=True)
        self.proj_1e = nn.Linear(c1e, 1, bias=False)
        self.proj_2e = nn.Linear(c2e, 1, bias=False) if c2e > 0 else None

        # Optional zero-initialized magnetoelectric coupling heads
        self.mag_proj_0e = nn.Linear(mag_c0, 1, bias=False) if mag_c0 > 0 else None
        self.mag_proj_1e = nn.Linear(mag_c1e, 1, bias=False) if mag_c1e > 0 else None
        self.mag_proj_2e = nn.Linear(mag_c2e, 1, bias=False) if mag_c2e > 0 else None
        for mod in (self.mag_proj_0e, self.mag_proj_1e, self.mag_proj_2e):
            if mod is not None:
                nn.init.zeros_(mod.weight)

        self._y2_fn = y2_fn
        default_dt = torch.get_default_dtype()
        b0, b1, b2 = self._build_bases_f64()

        self.register_buffer("basis_0e", b0.to(dtype=default_dt))
        self.register_buffer("basis_1e", b1.to(dtype=default_dt))
        self.register_buffer("basis_2e", b2.to(dtype=default_dt))

        if isinstance(per_species_scale, (int, float)):
            scales_t = torch.full(
                (self.num_types,), float(per_species_scale), dtype=default_dt
            )
        else:
            scales_t = torch.tensor(list(per_species_scale), dtype=default_dt)
            if scales_t.shape != (self.num_types,):
                raise ValueError(
                    f"per_species_scale must have length {self.num_types}, got {scales_t.shape}"
                )
        if per_species_scale_trainable:
            self.adam_born_scales = nn.Parameter(scales_t)
        else:
            self.register_buffer("adam_born_scales", scales_t)

    def _build_bases_f64(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inv_sqrt3 = 1.0 / math.sqrt(3.0)
        b0 = (torch.eye(3, dtype=torch.float64) * inv_sqrt3).reshape(1, 9)

        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        b1 = torch.zeros(3, 3, 3, dtype=torch.float64)
        b1[0, 1, 2] = inv_sqrt2
        b1[0, 2, 1] = -inv_sqrt2
        b1[1, 2, 0] = inv_sqrt2
        b1[1, 0, 2] = -inv_sqrt2
        b1[2, 0, 1] = inv_sqrt2
        b1[2, 1, 0] = -inv_sqrt2
        b1 = b1.reshape(3, 9)

        b2 = build_l2_to_stf_matrix(self._y2_fn)  # (5, 9)
        return b0, b1, b2

    def _apply(self, fn):  # type: ignore[override]
        super()._apply(fn)
        if self.basis_0e.dtype == torch.float64:
            b0, b1, b2 = self._build_bases_f64()
            dev = self.basis_0e.device
            self.basis_0e.copy_(b0.to(device=dev))
            self.basis_1e.copy_(b1.to(device=dev))
            self.basis_2e.copy_(b2.to(device=dev))
        return self

    def _get_bases(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.basis_0e.dtype == dtype:
            return self.basis_0e, self.basis_1e, self.basis_2e
        return (
            self.basis_0e.to(dtype=dtype),
            self.basis_1e.to(dtype=dtype),
            self.basis_2e.to(dtype=dtype),
        )

    def forward(
        self,
        h_0e: torch.Tensor,
        h_1e: torch.Tensor,
        h_2e: torch.Tensor | None,
        atom_types: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        mag_0e: torch.Tensor | None = None,
        mag_1e: torch.Tensor | None = None,
        mag_2e: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c0 = self.proj_0e(h_0e)  # (N, 1)
        c1 = self.proj_1e(h_1e).squeeze(-1)  # (N, 3)

        if self.mag_proj_0e is not None and mag_0e is not None:
            c0 = c0 + self.mag_proj_0e(mag_0e)
        if self.mag_proj_1e is not None and mag_1e is not None:
            c1 = c1 + self.mag_proj_1e(mag_1e).squeeze(-1)

        b0, b1, b2 = self._get_bases(h_0e.dtype)
        bec_flat = c0 @ b0 + c1 @ b1  # (N, 9)

        c2: torch.Tensor | None = None
        if self.proj_2e is not None and h_2e is not None:
            c2 = self.proj_2e(h_2e).squeeze(-1)  # (N, 5)
        if self.mag_proj_2e is not None and mag_2e is not None:
            m2 = self.mag_proj_2e(mag_2e).squeeze(-1)
            c2 = m2 if c2 is None else (c2 + m2)
        if c2 is not None:
            bec_flat = bec_flat + c2 @ b2

        bec = bec_flat.reshape(-1, 3, 3)

        scales = self.adam_born_scales.index_select(0, atom_types).to(dtype=bec.dtype)
        bec = bec * scales.view(-1, 1, 1)

        return apply_acoustic_sum_rule(bec, batch, num_graphs, mode=self.asr)


class _SingleEvidentialHead(nn.Module):
    """Single NIW evidential covariance head for 3D forces or 2D tangent spin torques."""

    def __init__(
        self,
        scalar_dim: int,
        vector_channels: int,
        hidden_dim: int = 32,
        is_spin: bool = False,
        spin_force_scale: float = 40.0,
    ) -> None:
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.vector_channels = int(vector_channels)
        self.is_spin = bool(is_spin)
        self.dimension = 2 if self.is_spin else 3
        self.spin_force_scale = float(spin_force_scale)

        self.nu_mlp = SwiGLUMLP(self.scalar_dim, hidden_dim, 1, num_layers=1)
        self.kappa_mlp = SwiGLUMLP(self.scalar_dim, hidden_dim, 1, num_layers=1)
        self.sigma_s_mlp = SwiGLUMLP(
            self.scalar_dim, hidden_dim, self.vector_channels, num_layers=1
        )
        inv_dim = 2 * self.vector_channels
        self.trace_linear = nn.Linear(inv_dim, 1)
        self.gate_linear = nn.Linear(inv_dim, self.vector_channels)

    def forward(
        self,
        node_scalars: torch.Tensor,
        node_vectors: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        spins: torch.Tensor | None = None,
        spin_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # node_scalars: (N, S), node_vectors: (N, 3, C_v)
        vecs = node_vectors.transpose(-1, -2)  # (N, C_v, 3)
        vec_sq = (vecs * vecs).sum(dim=-1)  # (N, C_v)
        s_raw = self.sigma_s_mlp(node_scalars)  # (N, C_v)
        invariants = torch.cat([s_raw, vec_sq], dim=-1)  # (N, 2*C_v)

        gates = self.gate_linear(invariants) / float(self.vector_channels)
        # Use bmm instead of 3-operand einsum so opt_einsum never specializes N under dynamic=True
        weighted_vecs = vecs * gates.unsqueeze(-1)  # (N, C_v, 3)
        sym = torch.bmm(weighted_vecs.transpose(-2, -1), vecs)  # (N, 3, 3)
        eye3 = torch.eye(3, dtype=sym.dtype, device=sym.device)

        trace_sym = (sym[:, 0, 0] + sym[:, 1, 1] + sym[:, 2, 2]) / 3.0
        target_trace = self.trace_linear(invariants).squeeze(-1)
        # Decompose into isotropic trace (0ee) and traceless symmetric (2ee)
        stf = sym - trace_sym.view(-1, 1, 1) * eye3.unsqueeze(0)

        # Smooth eigenvalue damping without abs/maximum (compile-safe)
        damped_iso = _linear_tanh_damper(target_trace, 5.0, 8.0)
        damped_iso = -_linear_tanh_damper(-damped_iso, 20.0, 25.0)

        stf_sq = (stf * stf).sum(dim=(-2, -1))
        small = stf_sq < 1e-16
        safe_stf_sq = torch.where(small, torch.ones_like(stf_sq), stf_sq)
        stf_norm = torch.sqrt(safe_stf_sq)
        damped_norm = _linear_tanh_damper(stf_norm, 5.0, 8.0)
        scale_factor = torch.where(
            small, torch.ones_like(stf_norm), damped_norm / stf_norm
        )
        stf_damped = stf * scale_factor.view(-1, 1, 1)
        stf_damped = 0.5 * (stf_damped + stf_damped.transpose(-1, -2))
        sigma = _sym3x3_matrix_exp(damped_iso, stf_damped, eye3)
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))

        nu = float(self.dimension + 2) + F.softplus(self.nu_mlp(node_scalars))
        kappa = F.softplus(self.kappa_mlp(node_scalars)) + 1e-6

        valid = torch.ones_like(nu)
        if self.is_spin and spins is not None:
            s_sq = (spins * spins).sum(dim=-1, keepdim=True)
            nonzero = s_sq > 1e-12
            safe_s_sq = torch.where(nonzero, s_sq, torch.ones_like(s_sq))
            s_norm = torch.sqrt(safe_s_sq)
            unit = torch.where(nonzero, spins / s_norm, torch.zeros_like(spins))
            proj = eye3.unsqueeze(0) - unit.unsqueeze(-1) * unit.unsqueeze(-2)
            valid = torch.where(nonzero, torch.ones_like(nu), torch.zeros_like(nu))
            if spin_mask is not None:
                valid = valid * spin_mask.view(-1, 1).to(dtype=sigma.dtype)
            sigma = (
                (proj @ sigma @ proj)
                * valid.unsqueeze(-1)
                / (self.spin_force_scale * self.spin_force_scale)
            )
            sigma = 0.5 * (sigma + sigma.transpose(-1, -2))

        uepi = (
            nu.unsqueeze(-1)
            * sigma
            / (kappa * (nu - float(self.dimension + 1))).unsqueeze(-1)
        )
        total_uepi = _segment_sum(uepi, batch, num_graphs)
        trace_uepi = (uepi[:, 0, 0] + uepi[:, 1, 1] + uepi[:, 2, 2]).unsqueeze(-1)
        valid_count = _segment_sum(valid, batch, num_graphs).clamp_min(1.0)
        uncertainty = _segment_sum(trace_uepi, batch, num_graphs) / valid_count

        return {
            "sigma": sigma,
            "Uepi": uepi,
            "nu": nu,
            "kappa": kappa,
            "total_Uepi": total_uepi,
            "uncertainty": uncertainty,
        }


class EvidentialHead(nn.Module):
    """Unified Stage-4 e2SLP evidential uncertainty head for forces & spin torques."""

    def __init__(
        self,
        scalar_dim: int,
        force_vec_channels: int,
        spin_vec_channels: int = 0,
        hidden_dim: int = 32,
        include_f: bool = True,
        include_sf: bool = False,
        spin_force_scale: float = 40.0,
    ) -> None:
        super().__init__()
        self.include_f = bool(include_f)
        self.include_sf = bool(include_sf)
        self.f_head = (
            _SingleEvidentialHead(
                scalar_dim=scalar_dim,
                vector_channels=force_vec_channels,
                hidden_dim=hidden_dim,
                is_spin=False,
            )
            if self.include_f
            else None
        )
        self.sf_head = (
            _SingleEvidentialHead(
                scalar_dim=scalar_dim,
                vector_channels=spin_vec_channels,
                hidden_dim=hidden_dim,
                is_spin=True,
                spin_force_scale=spin_force_scale,
            )
            if self.include_sf
            else None
        )

    def forward(
        self,
        node_scalars: torch.Tensor,
        force_vectors: torch.Tensor | None,
        batch: torch.Tensor,
        num_graphs: int,
        spin_vectors: torch.Tensor | None = None,
        spins: torch.Tensor | None = None,
        spin_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if self.f_head is not None and force_vectors is not None:
            f_out = self.f_head(node_scalars, force_vectors, batch, num_graphs)
            for k, v in f_out.items():
                out[f"f_{k}"] = v
        if self.sf_head is not None and spin_vectors is not None:
            sf_out = self.sf_head(
                node_scalars,
                spin_vectors,
                batch,
                num_graphs,
                spins=spins,
                spin_mask=spin_mask,
            )
            for k, v in sf_out.items():
                out[f"sf_{k}"] = v
        return out


class ExtrapolationGrade(nn.Module):
    """Lightweight GRACE random-projection D-optimality extrapolation grade.

    Computes ``gamma_i = max_k |(R_proj @ D_i)^T (A_active^-1)_k|`` with zero
    neighbor communication, initialized to an identity active-set inverse until
    calibrated on a training dataset.
    """

    def __init__(self, in_dim: int, proj_dim: int = 32, seed: int = 0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.proj_dim = int(proj_dim)
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        proj = torch.randn(self.in_dim, self.proj_dim, generator=gen) / math.sqrt(
            float(self.proj_dim)
        )
        self.register_buffer("rand_proj", proj)
        self.register_buffer("active_inv", torch.eye(self.proj_dim))

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        proj = self.rand_proj.to(dtype=desc.dtype)
        ainv = self.active_inv.to(dtype=desc.dtype)
        # (N, proj_dim)
        c = (desc @ proj) @ ainv
        # Avoid abs/max aten.sgn if ever differentiated; gamma is inference/monitor
        c_sq = c * c
        return torch.sqrt(torch.amax(c_sq, dim=-1) + 1e-24)
