"""Pure-PyTorch SO(3) Wigner 3j, recursive Wigner-D, and Cartesian tensor basis."""

import contextlib
import functools
import math
from fractions import Fraction
from typing import List, Optional, Tuple

import torch

__all__ = [
    "change_basis_real_to_complex",
    "wigner_3j",
    "proper_wigner_D",
    "proper_wigner_Ds",
    "random_rotation_matrix",
    "cartesian_tensor_basis",
    "cartesian_to_irreps",
    "irreps_to_cartesian",
]


@contextlib.contextmanager
def _ordinary_tensors():
    """Build cached CPU float64 constants outside inference/meta/fake modes."""
    with contextlib.ExitStack() as stack:
        inference_mode = getattr(torch, "inference_mode", None)
        if inference_mode is not None:
            stack.enter_context(inference_mode(False))
        stack.enter_context(torch.no_grad())
        try:
            from torch.utils._python_dispatch import _disable_current_modes
        except ImportError:  # pragma: no cover
            pass
        else:
            stack.enter_context(_disable_current_modes())
        disable_fn_modes = getattr(torch._C, "DisableTorchFunction", None)
        if disable_fn_modes is not None:
            stack.enter_context(disable_fn_modes())
        yield


def _resolve_dtype_device(
    dtype: Optional[torch.dtype], device: Optional[torch.device]
) -> Tuple[torch.dtype, torch.device]:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        get_default = getattr(torch, "get_default_device", None)
        device = get_default() if get_default is not None else torch.device("cpu")
    return dtype, torch.device(device)


def _su2_cg_coeff(j1: int, m1: int, j2: int, m2: int, j3: int, m3: int) -> float:
    """Exact Racah formula for integer-spin SU(2) Clebsch-Gordan coefficient."""
    if m3 != m1 + m2:
        return 0.0
    if not (abs(j1 - j2) <= j3 <= j1 + j2):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0

    fact = math.factorial
    vmin = max(-j1 + j2 + m3, -j1 + m1, 0)
    vmax = min(j2 + j3 + m1, j3 - j1 + j2, j3 + m3)

    prefactor = math.sqrt(
        (2 * j3 + 1)
        * float(
            Fraction(
                fact(j3 + j1 - j2)
                * fact(j3 - j1 + j2)
                * fact(j1 + j2 - j3)
                * fact(j3 + m3)
                * fact(j3 - m3),
                fact(j1 + j2 + j3 + 1)
                * fact(j1 - m1)
                * fact(j1 + m1)
                * fact(j2 - m2)
                * fact(j2 + m2),
            )
        )
    )

    acc = Fraction(0)
    for v in range(vmin, vmax + 1):
        sign = -1 if ((v + j2 + m2) & 1) else 1
        acc += sign * Fraction(
            fact(j2 + j3 + m1 - v) * fact(j1 - m1 + v),
            fact(v)
            * fact(j3 - j1 + j2 - v)
            * fact(j3 + m3 - v)
            * fact(v + j1 - j2 - m3),
        )
    return prefactor * float(acc)


@functools.lru_cache(maxsize=None)
def _su2_clebsch_gordan(j1: int, j2: int, j3: int) -> torch.Tensor:
    with _ordinary_tensors():
        mat = torch.zeros((2 * j1 + 1, 2 * j2 + 1, 2 * j3 + 1), dtype=torch.float64)
        if abs(j1 - j2) <= j3 <= j1 + j2:
            for m1 in range(-j1, j1 + 1):
                for m2 in range(-j2, j2 + 1):
                    m3 = m1 + m2
                    if abs(m3) <= j3:
                        mat[j1 + m1, j2 + m2, j3 + m3] = _su2_cg_coeff(
                            j1, m1, j2, m2, j3, m3
                        )
        return mat


@functools.lru_cache(maxsize=None)
def _change_basis_real_to_complex_f64(l: int) -> torch.Tensor:
    """Unitary change of basis from real spherical harmonics to complex spherical harmonics.

    Follows the standard convention where real l=1 components are ordered (x, y, z)
    so that D^1(R) = R and real Clebsch-Gordan tensors are strictly real.
    """
    with _ordinary_tensors():
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        q = torch.zeros((2 * l + 1, 2 * l + 1), dtype=torch.complex128)
        for m in range(-l, 0):
            q[l + m, l + abs(m)] = inv_sqrt2
            q[l + m, l - abs(m)] = -1j * inv_sqrt2
        q[l, l] = 1.0
        for m in range(1, l + 1):
            phase = (-1) ** m
            q[l + m, l + abs(m)] = phase * inv_sqrt2
            q[l + m, l - abs(m)] = 1j * phase * inv_sqrt2
        q = ((-1j) ** l) * q
        return q.contiguous()


def change_basis_real_to_complex(
    l: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the unitary real-to-complex basis matrix for angular degree ``l``."""
    if not (isinstance(l, int) and not isinstance(l, bool) and l >= 0):
        raise ValueError(f"l must be a non-negative integer, got {l!r}")
    dtype, device = _resolve_dtype_device(dtype, device)
    complex_dtype = {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }.get(dtype)
    if complex_dtype is None:
        raise ValueError(f"Unsupported real dtype for complex basis: {dtype}")
    return _change_basis_real_to_complex_f64(l).to(
        dtype=complex_dtype, device=device, copy=True
    )


@functools.lru_cache(maxsize=None)
def _real_wigner_3j_f64(l1: int, l2: int, l3: int) -> torch.Tensor:
    """Cached CPU float64 real Wigner 3j tensor with Frobenius norm 1."""
    with _ordinary_tensors():
        q1 = _change_basis_real_to_complex_f64(l1)
        q2 = _change_basis_real_to_complex_f64(l2)
        q3 = _change_basis_real_to_complex_f64(l3)
        cg = _su2_clebsch_gordan(l1, l2, l3).to(dtype=torch.complex128)
        c = torch.einsum("ij,kl,mn,ikn->jlm", q1, q2, torch.conj(q3.T), cg)
        if torch.max(torch.abs(torch.imag(c))).item() >= 1e-10:
            raise RuntimeError(
                f"Real Wigner 3j ({l1}, {l2}, {l3}) acquired non-zero imaginary part"
            )
        c = torch.real(c)
        norm = torch.linalg.norm(c)
        if norm.item() > 0.0:
            c = c / norm
        return c.contiguous()


def wigner_3j(
    l1: int,
    l2: int,
    l3: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Real SO(3) Wigner 3j tensor :math:`C^{l_1 l_2 l_3}` of shape ``(2l1+1, 2l2+1, 2l3+1)``.

    Satisfies:
    - :math:`C_{lmn} = C_{ijk} D^{l_1}_{il}(R) D^{l_2}_{jm}(R) D^{l_3}_{kn}(R)` for all :math:`R \in SO(3)`,
    - Frobenius norm :math:`\|C\|_F = 1`,
    - :math:`\sum_{i,j} C_{ijk} C_{ijl} = \frac{1}{2l_3+1} \delta_{kl}`.
    """
    for name, val in (("l1", l1), ("l2", l2), ("l3", l3)):
        if not (isinstance(val, int) and not isinstance(val, bool) and val >= 0):
            raise ValueError(f"{name} must be a non-negative integer, got {val!r}")
    if not (abs(l1 - l2) <= l3 <= l1 + l2):
        raise ValueError(f"Triangle rule violated for wigner_3j({l1}, {l2}, {l3})")
    dtype, device = _resolve_dtype_device(dtype, device)
    return _real_wigner_3j_f64(l1, l2, l3).to(dtype=dtype, device=device, copy=True)


@functools.lru_cache(maxsize=None)
def _raising_isometry_f64(l: int) -> torch.Tensor:
    """Orthonormal isometry V_l = sqrt(2l+1) * C^(l-1, 1, l) of shape (3*(2l-1), 2l+1)."""
    with _ordinary_tensors():
        c = _real_wigner_3j_f64(l - 1, 1, l)
        return (
            math.sqrt(2 * l + 1) * c.reshape(3 * (2 * l - 1), 2 * l + 1)
        ).contiguous()


def proper_wigner_Ds(lmax: int, R: torch.Tensor) -> List[torch.Tensor]:
    r"""Exact recursive SO(3) Wigner-D matrices ``[D^0(R), ..., D^lmax(R)]``.

    Uses the exact raising isometry recursion
    :math:`D^l(R) = V_l^\top (D^{l-1}(R) \otimes R) V_l` with
    :math:`V_l = \sqrt{2l+1}\, C^{l-1, 1, l}`. This avoids Euler angles and
    ``torch.matrix_exp`` small-angle rounding errors, achieving machine-precision
    (< 1e-15 in float64) equivariance verification for arbitrary proper rotations.

    Parameters
    ----------
    lmax : int
        Maximum angular degree :math:`l \ge 0`.
    R : torch.Tensor
        Proper rotation matrices of shape ``(..., 3, 3)`` with :math:`\det(R) = +1`.
    """
    if not (isinstance(lmax, int) and not isinstance(lmax, bool) and lmax >= 0):
        raise ValueError(f"lmax must be a non-negative integer, got {lmax!r}")
    R = torch.as_tensor(R)
    if R.shape[-2:] != (3, 3):
        raise ValueError(
            f"Rotation matrices must have shape (..., 3, 3), got {tuple(R.shape)}"
        )
    R_det = R.float() if R.dtype in (torch.float16, torch.bfloat16) else R
    det = torch.det(R_det)
    det_atol = 1e-2 if R.dtype in (torch.float16, torch.bfloat16) else 1e-5
    if not torch.allclose(det, torch.ones_like(det), atol=det_atol, rtol=det_atol):
        raise ValueError(
            "proper_wigner_Ds requires proper rotations with det(R) = +1; "
            "pass k_inv / k_t separately for spatial inversion and time reversal."
        )
    batch = R.shape[:-2]
    Ds = [R.new_ones(batch + (1, 1)), R]
    for l in range(2, lmax + 1):
        V = _raising_isometry_f64(l).to(dtype=R.dtype, device=R.device)
        K = torch.einsum("...ij,...kl->...ikjl", Ds[-1], R).reshape(
            batch + (V.shape[0], V.shape[0])
        )
        Ds.append(V.T @ K @ V)
    return Ds[: lmax + 1]


def proper_wigner_D(l: int, R: torch.Tensor) -> torch.Tensor:
    """Exact recursive SO(3) Wigner-D matrix ``D^l(R)`` of shape ``(..., 2l+1, 2l+1)``."""
    return proper_wigner_Ds(l, R)[l]


def random_rotation_matrix(
    batch_shape: Tuple[int, ...] = (),
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample Haar-uniform proper rotation matrices in SO(3) via QR decomposition."""
    if isinstance(batch_shape, int):
        batch_shape = (batch_shape,)
    work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    mat = torch.randn(
        batch_shape + (3, 3), dtype=work_dtype, device=device, generator=generator
    )
    q, r = torch.linalg.qr(mat)
    diag = torch.diagonal(r, dim1=-2, dim2=-1)
    signs = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
    q = q * signs.unsqueeze(-2)
    det = torch.det(q)
    flip = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    q = torch.cat([q[..., :2], q[..., 2:3] * flip.unsqueeze(-1).unsqueeze(-1)], dim=-1)
    return q.to(dtype=dtype) if work_dtype != dtype else q


def _build_cartesian_tensor_basis_f64() -> torch.Tensor:
    """Exact orthogonal 9x9 matrix mapping row-major 3x3 tensors -> 0e + 1e + 2e irreps."""
    with _ordinary_tensors():
        blocks = [
            math.sqrt(2 * J + 1) * _real_wigner_3j_f64(1, 1, J).reshape(9, 2 * J + 1)
            for J in (0, 1, 2)
        ]
        return torch.cat(blocks, dim=1).contiguous()


_CARTESIAN_TENSOR_BASIS_F64: torch.Tensor = _build_cartesian_tensor_basis_f64()


def cartesian_tensor_basis(
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Return the orthogonal ``(9, 9)`` change-of-basis matrix from 3x3 Cartesian tensors to ``0e + 1e + 2e``.

    For a 3x3 tensor :math:`T` (e.g. Born effective charge :math:`Z^*_i`),
    ``irreps_vec = T.reshape(..., 9) @ Q`` decomposes :math:`T` into
    ``[0e (1), 1e (3), 2e (5)]`` transforming with ``diag(D^0(R), D^1(R), D^2(R))``.
    Conversely, ``T = (irreps_vec @ Q.T).reshape(..., 3, 3)`` reconstructs the
    Cartesian 3x3 tensor orthogonally.
    """
    dtype, device = _resolve_dtype_device(dtype, device)
    return _CARTESIAN_TENSOR_BASIS_F64.to(dtype=dtype, device=device, copy=True)


def cartesian_to_irreps(tensor_3x3: torch.Tensor) -> torch.Tensor:
    """Map ``(..., 3, 3)`` Cartesian tensor to ``(..., 9)`` in ``0e + 1e + 2e`` irrep basis."""
    if tensor_3x3.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3) tensor, got {tuple(tensor_3x3.shape)}")
    Q = _CARTESIAN_TENSOR_BASIS_F64.to(dtype=tensor_3x3.dtype, device=tensor_3x3.device)
    return tensor_3x3.reshape(tensor_3x3.shape[:-2] + (9,)) @ Q


def irreps_to_cartesian(irreps_9: torch.Tensor) -> torch.Tensor:
    """Map ``(..., 9)`` in ``0e + 1e + 2e`` irrep basis to ``(..., 3, 3)`` Cartesian tensor."""
    if irreps_9.shape[-1] != 9:
        raise ValueError(
            f"Expected trailing dimension 9 for 0e+1e+2e, got {irreps_9.shape[-1]}"
        )
    Q = _CARTESIAN_TENSOR_BASIS_F64.to(dtype=irreps_9.dtype, device=irreps_9.device)
    return (irreps_9 @ Q.T).reshape(irreps_9.shape[:-1] + (3, 3))
