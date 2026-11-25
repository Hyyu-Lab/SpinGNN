"""Exact FP64 equivariance and invariance verification helpers (Rule R5).

Uses exact recursive Wigner-D matrices (:func:`proper_wigner_Ds`) to test:
- Independent lattice and spin rotations ``(R_lat, R_spin)`` for decoupled blocks,
- Coupled global ``SO(3)`` rotations ``R`` + spatial inversion ``P`` + time reversal ``T``,
- Scalar energy invariance on geometric ``(pos, spins)`` inputs for both NoSOC and SOC models.
"""

from typing import Callable, Dict, Optional

import torch

from .irreps import DIrreps, Irreps
from .wigner import random_rotation_matrix

__all__ = [
    "check_decoupled_equivariance",
    "check_coupled_equivariance",
    "check_scalar_invariance",
]


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(a - b)).item())


@torch.no_grad()
def check_decoupled_equivariance(
    fn: Callable[[torch.Tensor], torch.Tensor],
    irreps_in,
    irreps_out,
    *,
    batch_size: int = 8,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Measure max absolute equivariance error under ``O(3)_lat x O(3)_spin``.

    Returns errors for:
    - ``rot_spin_only``: independent spin rotation with fixed lattice
    - ``rot_lat_only``: independent lattice rotation with fixed spin
    - ``rot_independent``: simultaneous independent ``(R_lat, R_spin)``
    - ``parity``: spatial inversion ``k_inv = 1``
    - ``time_reversal``: time reversal ``k_t = 1``
    - ``full_group``: combined ``(R_lat, R_spin, k_inv=1, k_t=1)``
    """
    ir_in = DIrreps(irreps_in)
    ir_out = DIrreps(irreps_out)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((batch_size, ir_in.dim), dtype=dtype, generator=gen).to(
        device=device
    )
    y = fn(x)

    I3 = torch.eye(3, dtype=dtype, device=device)
    R_lat = random_rotation_matrix(dtype=dtype, generator=gen).to(device=device)
    R_spin = random_rotation_matrix(dtype=dtype, generator=gen).to(device=device)

    cases = {
        "rot_spin_only": (I3, R_spin, 0, 0),
        "rot_lat_only": (R_lat, I3, 0, 0),
        "rot_independent": (R_lat, R_spin, 0, 0),
        "parity": (I3, I3, 1, 0),
        "time_reversal": (I3, I3, 0, 1),
        "full_group": (R_lat, R_spin, 1, 1),
    }
    errors: Dict[str, float] = {}
    for name, (rl, rs, k_inv, k_t) in cases.items():
        D_in = ir_in.D_from_matrices(rl, rs, k_inv=k_inv, k_t=k_t)
        D_out = ir_out.D_from_matrices(rl, rs, k_inv=k_inv, k_t=k_t)
        y_rot_in = fn(x @ D_in.T)
        y_rot_out = y @ D_out.T
        errors[name] = _max_abs_err(y_rot_in, y_rot_out)
    return errors


@torch.no_grad()
def check_coupled_equivariance(
    fn: Callable[[torch.Tensor], torch.Tensor],
    irreps_in,
    irreps_out,
    *,
    batch_size: int = 8,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Measure max absolute equivariance error under ``O(3) x Z2^T``.

    Returns errors for:
    - ``rotation``: global proper rotation ``R``
    - ``parity``: spatial inversion ``k_inv = 1``
    - ``time_reversal``: time reversal ``k_t = 1``
    - ``full_group``: combined ``(R, k_inv=1, k_t=1)``
    """
    ir_in = Irreps(irreps_in)
    ir_out = Irreps(irreps_out)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((batch_size, ir_in.dim), dtype=dtype, generator=gen).to(
        device=device
    )
    y = fn(x)

    I3 = torch.eye(3, dtype=dtype, device=device)
    R = random_rotation_matrix(dtype=dtype, generator=gen).to(device=device)

    cases = {
        "rotation": (R, 0, 0),
        "parity": (I3, 1, 0),
        "time_reversal": (I3, 0, 1),
        "full_group": (R, 1, 1),
    }
    errors: Dict[str, float] = {}
    for name, (mat, k_inv, k_t) in cases.items():
        D_in = ir_in.D_from_matrix(mat, k_inv=k_inv, k_t=k_t)
        D_out = ir_out.D_from_matrix(mat, k_inv=k_inv, k_t=k_t)
        y_rot_in = fn(x @ D_in.T)
        y_rot_out = y @ D_out.T
        errors[name] = _max_abs_err(y_rot_in, y_rot_out)
    return errors


@torch.no_grad()
def check_scalar_invariance(
    energy_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pos: torch.Tensor,
    spins: torch.Tensor,
    *,
    seed: int = 0,
) -> Dict[str, float]:
    """Measure invariance errors of ``energy_fn(pos, spins)`` under physical symmetry operations.

    Evaluates:
    - ``rot_spin_only``: ``(pos, spins @ R_spin.T)`` (invariant for NoSOC; generally non-zero for SOC)
    - ``rot_lat_only``: ``(pos @ R_lat.T, spins)`` (invariant for NoSOC; generally non-zero for SOC)
    - ``rot_joint``: ``(pos @ R.T, spins @ R.T)`` (invariant for both NoSOC and SOC)
    - ``parity``: ``(-pos, +spins)`` (polar positions flip, axial spins unchanged)
    - ``time_reversal``: ``(+pos, -spins)`` (positions unchanged, spins flip)
    - ``full_soc``: ``(-pos @ R.T, -spins @ R.T)`` (combined R + P + T)
    """
    dtype, device = pos.dtype, pos.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    R_lat = random_rotation_matrix(dtype=dtype, generator=gen).to(device=device)
    R_spin = random_rotation_matrix(dtype=dtype, generator=gen).to(device=device)

    e0 = energy_fn(pos, spins)
    return {
        "rot_spin_only": _max_abs_err(energy_fn(pos, spins @ R_spin.T), e0),
        "rot_lat_only": _max_abs_err(energy_fn(pos @ R_lat.T, spins), e0),
        "rot_joint": _max_abs_err(energy_fn(pos @ R_lat.T, spins @ R_lat.T), e0),
        "parity": _max_abs_err(energy_fn(-pos, spins), e0),
        "time_reversal": _max_abs_err(energy_fn(pos, -spins), e0),
        "full_soc": _max_abs_err(energy_fn(-pos @ R_lat.T, -spins @ R_lat.T), e0),
    }
