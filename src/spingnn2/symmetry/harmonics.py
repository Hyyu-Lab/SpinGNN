"""Singularity-free Cartesian polynomial spherical harmonics (l=0..4) and bipolar spherical harmonics.

All polynomials follow the exact real spherical harmonic convention matching
``wigner_3j`` and ``proper_wigner_D`` (with ``Y^1(v) = sqrt(3) * (x, y, z)`` in
``component`` normalization and ``D^1(R) = R``).

Spin inputs remain unnormalized polynomials by default, eliminating the 1/theta
collinear singularity present in normalized cross-product encodings.
"""

import math
import numbers
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .irreps import Irrep, Irreps
from .wigner import _ordinary_tensors, wigner_3j

__all__ = [
    "spherical_harmonics",
    "SphericalHarmonics",
    "BipolarSphericalHarmonics",
    "bipolar_spherical_harmonics",
    "bond_spin",
    "spin_spin",
    "spin_cross_axial",
    "spin_quadrupole_symmetric",
    "bipolar_spin_pair_j1_j2",
]

_SQRT2 = math.sqrt(2.0)
_SQRT3 = math.sqrt(3.0)
_SQRT5 = math.sqrt(5.0)
_SQRT6 = math.sqrt(6.0)
_SQRT7 = math.sqrt(7.0)
_SQRT14 = math.sqrt(14.0)
_SQRT15 = math.sqrt(15.0)
_SQRT21 = math.sqrt(21.0)
_SQRT42 = math.sqrt(42.0)
_SQRT70 = math.sqrt(70.0)
_SQRT105 = math.sqrt(105.0)
_SQRT168 = math.sqrt(168.0)
_SQRT210 = math.sqrt(210.0)
_INV_SQRT_4PI = 1.0 / math.sqrt(4.0 * math.pi)
_THREE_OVER_SQRT2 = 3.0 / _SQRT2
_NORM_DEGREE_SCALES = tuple(1.0 / math.sqrt(2 * deg + 1) for deg in range(5))


def _safe_normalize(vec: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Smoothly normalize vectors of shape ``(..., 3)`` without ``abs``/``maximum`` pitfalls.

    When ``eps > 0``, divides by ``sqrt(||v||^2 + eps^2)`` (infinitely smooth everywhere).
    When ``eps == 0``, divides non-zero vectors by exact ``||v||`` and maps near-zero
    vectors to ``0`` using a dtype-safe threshold so 0th, 1st, and 2nd order derivatives
    remain strictly finite across ``float64``, ``float32``, ``bfloat16``, and ``float16``.
    """
    r2 = (vec * vec).sum(dim=-1, keepdim=True)
    if eps > 0.0:
        eps2 = eps * eps
        if vec.dtype == torch.float16:
            eps2 = max(eps2, 1e-4)
        return vec * torch.rsqrt(r2 + eps2)
    if vec.dtype == torch.float64:
        r2_floor = 1e-24
    elif vec.dtype == torch.float16:
        r2_floor = 1e-4
    else:
        # float32 / bfloat16: keeps r^-5 <= 1e30 << 3.4e38 for finite 2nd-order autograd
        r2_floor = 1e-12
    zero_mask = r2 < r2_floor
    safe_r2 = torch.where(zero_mask, torch.ones_like(r2), r2)
    inv_r = torch.where(zero_mask, torch.zeros_like(r2), torch.rsqrt(safe_r2))
    return vec * inv_r


def _poly_l2_only(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Direct 5-component real Cartesian solid harmonics for ``l = 2`` (no l=0/l=1 allocs)."""
    x2 = x * x
    y2 = y * y
    z2 = z * z
    sh_2_0 = _SQRT15 * x * z
    sh_2_1 = _SQRT15 * x * y
    sh_2_2 = _SQRT5 * (y2 - 0.5 * (x2 + z2))
    sh_2_3 = _SQRT15 * y * z
    sh_2_4 = (0.5 * _SQRT15) * (z2 - x2)
    return torch.stack((sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4), dim=-1)


def _poly_spherical_harmonics_0_to_4(
    lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """Closed-form real Cartesian solid harmonics (component normalization) for l = 0..4."""
    sh_0_0 = torch.ones_like(x)
    if lmax == 0:
        return sh_0_0.unsqueeze(-1)

    sh_1_0 = _SQRT3 * x
    sh_1_1 = _SQRT3 * y
    sh_1_2 = _SQRT3 * z
    if lmax == 1:
        return torch.stack((sh_0_0, sh_1_0, sh_1_1, sh_1_2), dim=-1)

    x2 = x * x
    y2 = y * y
    z2 = z * z
    x2z2 = x2 + z2
    sh_2_0 = _SQRT15 * x * z
    sh_2_1 = _SQRT15 * x * y
    sh_2_2 = _SQRT5 * (y2 - 0.5 * x2z2)
    sh_2_3 = _SQRT15 * y * z
    sh_2_4 = (0.5 * _SQRT15) * (z2 - x2)
    if lmax == 2:
        return torch.stack(
            (sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4),
            dim=-1,
        )

    sh_3_0 = (_SQRT42 / 6.0) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = _SQRT7 * sh_2_0 * y
    sh_3_2 = (_SQRT168 / 8.0) * (4.0 * y2 - x2z2) * x
    sh_3_3 = (0.5 * _SQRT7) * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = (_SQRT168 / 8.0) * z * (4.0 * y2 - x2z2)
    sh_3_5 = _SQRT7 * sh_2_4 * y
    sh_3_6 = (_SQRT42 / 6.0) * (sh_2_4 * z - sh_2_0 * x)
    if lmax == 3:
        return torch.stack(
            (
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
            ),
            dim=-1,
        )

    if lmax == 4:
        sh_4_0 = (0.75 * _SQRT2) * (sh_3_0 * z + sh_3_6 * x)
        sh_4_1 = (
            0.75 * sh_3_0 * y
            + (0.375 * _SQRT6) * sh_3_1 * z
            + (0.375 * _SQRT6) * sh_3_5 * x
        )
        sh_4_2 = (
            (-3.0 / 56.0 * _SQRT14) * sh_3_0 * z
            + (3.0 / 14.0 * _SQRT21) * sh_3_1 * y
            + (3.0 / 56.0 * _SQRT210) * sh_3_2 * z
            + (3.0 / 56.0 * _SQRT210) * sh_3_4 * x
            + (3.0 / 56.0 * _SQRT14) * sh_3_6 * x
        )
        sh_4_3 = (
            (-3.0 / 56.0 * _SQRT42) * sh_3_1 * z
            + (3.0 / 28.0 * _SQRT105) * sh_3_2 * y
            + (3.0 / 28.0 * _SQRT70) * sh_3_3 * x
            + (3.0 / 56.0 * _SQRT42) * sh_3_5 * x
        )
        sh_4_4 = (
            (-3.0 / 28.0 * _SQRT42) * sh_3_2 * x
            + (3.0 / 7.0 * _SQRT7) * sh_3_3 * y
            + (-3.0 / 28.0 * _SQRT42) * sh_3_4 * z
        )
        sh_4_5 = (
            (-3.0 / 56.0 * _SQRT42) * sh_3_1 * x
            + (3.0 / 28.0 * _SQRT70) * sh_3_3 * z
            + (3.0 / 28.0 * _SQRT105) * sh_3_4 * y
            + (-3.0 / 56.0 * _SQRT42) * sh_3_5 * z
        )
        sh_4_6 = (
            (-3.0 / 56.0 * _SQRT14) * sh_3_0 * x
            + (-3.0 / 56.0 * _SQRT210) * sh_3_2 * x
            + (3.0 / 56.0 * _SQRT210) * sh_3_4 * z
            + (3.0 / 14.0 * _SQRT21) * sh_3_5 * y
            + (-3.0 / 56.0 * _SQRT14) * sh_3_6 * z
        )
        sh_4_7 = (
            (-0.375 * _SQRT6) * sh_3_1 * x
            + (0.375 * _SQRT6) * sh_3_5 * z
            + 0.75 * sh_3_6 * y
        )
        sh_4_8 = (0.75 * _SQRT2) * (-sh_3_0 * x + sh_3_6 * z)
        return torch.stack(
            (
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
                sh_4_0,
                sh_4_1,
                sh_4_2,
                sh_4_3,
                sh_4_4,
                sh_4_5,
                sh_4_6,
                sh_4_7,
                sh_4_8,
            ),
            dim=-1,
        )

    raise ValueError(f"SphericalHarmonics supports degrees l = 0..4, got lmax={lmax}")


def _parse_ls(lmax_or_ls: Union[int, Sequence[int]]) -> List[int]:
    if isinstance(lmax_or_ls, int) and not isinstance(lmax_or_ls, bool):
        if not (0 <= lmax_or_ls <= 4):
            raise ValueError(f"Degree must be in 0..4, got {lmax_or_ls}")
        return list(range(lmax_or_ls + 1))
    ls = [int(v) for v in lmax_or_ls]
    if not ls or any(v < 0 or v > 4 for v in ls):
        raise ValueError(f"All degrees must be integers in 0..4, got {ls!r}")
    return ls


def _eval_component_sh(
    ls: Tuple[int, ...], lmax: int, is_contiguous: bool, vec: torch.Tensor
) -> torch.Tensor:
    """Evaluate component-normalized SH with zero-overhead single-degree fast paths."""
    if len(ls) == 1:
        deg = ls[0]
        if deg == 0:
            return torch.ones_like(vec[..., :1])
        if deg == 1:
            return vec * _SQRT3
        if deg == 2:
            return _poly_l2_only(vec[..., 0], vec[..., 1], vec[..., 2])
        full = _poly_spherical_harmonics_0_to_4(
            deg, vec[..., 0], vec[..., 1], vec[..., 2]
        )
        return full[..., deg * deg : (deg + 1) ** 2]

    full = _poly_spherical_harmonics_0_to_4(lmax, vec[..., 0], vec[..., 1], vec[..., 2])
    if is_contiguous:
        return full
    return torch.cat([full[..., deg * deg : (deg + 1) ** 2] for deg in ls], dim=-1)


def spherical_harmonics(
    l: Union[int, Sequence[int]],
    vec: torch.Tensor,
    normalize: bool = False,
    normalization: str = "component",
    eps: float = 0.0,
) -> torch.Tensor:
    """Evaluate real Cartesian polynomial spherical harmonics (l = 0..4).

    Parameters
    ----------
    l : int or sequence of int
        If ``int``, evaluates the single degree ``l`` (shape ``(..., 2l+1)``).
        If sequence of ints, concatenates all requested degrees in order.
    vec : torch.Tensor
        Vectors of shape ``(..., 3)``.
    normalize : bool
        If ``True``, projects ``vec`` onto the unit sphere first. Keep ``False``
        for classical spins to stay polynomial and singularity-free at ``s = 0``.
    normalization : {'component', 'norm', 'integral'}
        Output normalization convention.
    eps : float
        Optional smoothing epsilon when ``normalize=True``.
    """
    if normalization not in ("component", "norm", "integral"):
        raise ValueError(
            f"normalization must be 'component', 'norm', or 'integral', got {normalization!r}"
        )
    if isinstance(l, int) and not isinstance(l, bool):
        if not (0 <= l <= 4):
            raise ValueError(f"Degree l must be in 0..4, got {l}")
        ls = (l,)
    else:
        ls = tuple(_parse_ls(l))

    if normalize:
        vec = _safe_normalize(vec, eps=eps)
    lmax = max(ls)
    is_contiguous = ls == tuple(range(lmax + 1))

    if normalization == "norm" and len(ls) == 1:
        deg = ls[0]
        if deg == 1:
            return vec
        out = _eval_component_sh(ls, lmax, is_contiguous, vec)
        return out * _NORM_DEGREE_SCALES[deg]

    out = _eval_component_sh(ls, lmax, is_contiguous, vec)
    if normalization == "integral":
        return out * _INV_SQRT_4PI
    if normalization == "norm":
        parts = []
        offset = 0
        for deg in ls:
            width = 2 * deg + 1
            seg = out[..., offset : offset + width]
            scale = _NORM_DEGREE_SCALES[deg]
            parts.append(seg if deg == 0 else seg * scale)
            offset += width
        return torch.cat(parts, dim=-1)
    return out


class SphericalHarmonics(nn.Module):
    """Compile-friendly real Cartesian polynomial spherical harmonics module (l = 0..4).

    Parameters
    ----------
    lmax_or_ls : int or sequence of int
        If ``int``, evaluates all degrees ``0..lmax`` (inclusive).
        If sequence of ints, evaluates exactly those degrees in order.
    normalize : bool
        Normalize input vectors to unit length (True for bonds, False for spins).
    normalization : {'component', 'norm', 'integral'}
        Output normalization convention (default ``'component'``).
    parity : {1, -1}
        Spatial inversion parity of the input vector (``-1`` polar bond, ``+1`` axial spin).
    time_reversal : {1, -1}
        Time-reversal parity of the input vector (``+1`` bond, ``-1`` classical spin).
    eps : float
        Optional smoothing epsilon for ``normalize=True``.
    """

    def __init__(
        self,
        lmax_or_ls: Union[int, Sequence[int]],
        normalize: bool = False,
        normalization: str = "component",
        parity: int = -1,
        time_reversal: int = 1,
        eps: float = 0.0,
    ):
        super().__init__()
        if normalization not in ("component", "norm", "integral"):
            raise ValueError(
                f"normalization must be 'component', 'norm', or 'integral', got {normalization!r}"
            )
        if parity not in (-1, 1):
            raise ValueError(f"parity must be -1 or +1, got {parity!r}")
        if time_reversal not in (-1, 1):
            raise ValueError(f"time_reversal must be -1 or +1, got {time_reversal!r}")

        self._ls: Tuple[int, ...] = tuple(_parse_ls(lmax_or_ls))
        self._lmax: int = max(self._ls)
        self._is_contiguous_range: bool = self._ls == tuple(range(self._lmax + 1))
        self.normalize = bool(normalize)
        self.normalization = normalization
        self.eps = float(eps)
        self.irreps_out = Irreps(
            [(1, Irrep(l, parity**l, time_reversal**l)) for l in self._ls]
        )
        self.dim_out = self.irreps_out.dim

        if normalization == "norm" and len(self._ls) > 1:
            scales = [
                _NORM_DEGREE_SCALES[deg] for deg in self._ls for _ in range(2 * deg + 1)
            ]
            self._norm_scale_f64: Optional[Tuple[float, ...]] = tuple(scales)
            self.register_buffer(
                "_norm_scale",
                torch.tensor(scales, dtype=torch.get_default_dtype()),
                persistent=False,
            )
        else:
            self._norm_scale_f64 = None
            self.register_buffer("_norm_scale", None, persistent=False)

    def _apply(self, fn):
        ret = super()._apply(fn)
        if (
            self._norm_scale is not None
            and self._norm_scale_f64 is not None
            and self._norm_scale.dtype == torch.float64
        ):
            with _ordinary_tensors():
                exact = torch.tensor(self._norm_scale_f64, dtype=torch.float64)
            self._norm_scale.data.copy_(exact.to(device=self._norm_scale.device))
        return ret

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            vec = _safe_normalize(vec, eps=self.eps)
        if self.normalization == "norm" and len(self._ls) == 1:
            deg = self._ls[0]
            if deg == 1:
                return vec
            out = _eval_component_sh(
                self._ls, self._lmax, self._is_contiguous_range, vec
            )
            return out * _NORM_DEGREE_SCALES[deg]

        out = _eval_component_sh(self._ls, self._lmax, self._is_contiguous_range, vec)
        if self.normalization == "integral":
            return out * _INV_SQRT_4PI
        if self._norm_scale is not None:
            scale = self._norm_scale
            if scale.dtype != out.dtype:
                if out.dtype == torch.float64 and self._norm_scale_f64 is not None:
                    scale = _cached_norm_scale_f64(self._norm_scale_f64).to(
                        device=out.device
                    )
                else:
                    scale = scale.to(dtype=out.dtype)
            return out * scale
        return out


_NORM_SCALE_CACHE = {}


def _cached_norm_scale_f64(scales: Tuple[float, ...]) -> torch.Tensor:
    t = _NORM_SCALE_CACHE.get(scales)
    if t is None:
        with _ordinary_tensors():
            t = torch.tensor(scales, dtype=torch.float64)
        _NORM_SCALE_CACHE[scales] = t
    return t


_cached_norm_scale_f64._dynamo_marked_constant = True


# ============================================================================
# Bipolar Spherical Harmonics B^J_{l, lam}(a, b)
# ============================================================================


def _check_positive_scale(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not (math.isfinite(float(value)) and float(value) > 0.0)
    ):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return float(value)


def _bipolar_blocks(
    lmax_a: int, lmax_b: int, jmax: Optional[int]
) -> List[Tuple[int, int, Tuple[int, ...]]]:
    blocks = []
    for l in range(lmax_a + 1):
        for lam in range(lmax_b + 1):
            js = tuple(
                J for J in range(abs(l - lam), l + lam + 1) if jmax is None or J <= jmax
            )
            if js:
                blocks.append((l, lam, js))
    return blocks


def _build_coupling_matrix_f64(l: int, lam: int, js: Sequence[int]) -> torch.Tensor:
    with _ordinary_tensors():
        cols = [
            math.sqrt(2 * J + 1)
            * wigner_3j(l, lam, J, dtype=torch.float64, device="cpu").reshape(
                (2 * l + 1) * (2 * lam + 1), 2 * J + 1
            )
            for J in js
        ]
        return torch.cat(cols, dim=1).contiguous()


_SINGLE_COUPLING_CACHE = {}


def _cached_single_coupling_f64(l: int, lam: int, js: Tuple[int, ...]) -> torch.Tensor:
    key = (l, lam, js)
    mat = _SINGLE_COUPLING_CACHE.get(key)
    if mat is None:
        mat = _build_coupling_matrix_f64(l, lam, js)
        _SINGLE_COUPLING_CACHE[key] = mat
    return mat


_cached_single_coupling_f64._dynamo_marked_constant = True


_COUPLING_CACHE = {}


def _cached_coupling_matrices(
    lmax_a: int, lmax_b: int, jmax: Optional[int]
) -> Tuple[torch.Tensor, ...]:
    key = (lmax_a, lmax_b, jmax)
    mats = _COUPLING_CACHE.get(key)
    if mats is None:
        mats = tuple(
            _cached_single_coupling_f64(l, lam, js)
            for l, lam, js in _bipolar_blocks(lmax_a, lmax_b, jmax)
        )
        _COUPLING_CACHE[key] = mats
    return mats


_cached_coupling_matrices._dynamo_marked_constant = True


def _closed_form_b11_components(
    a: torch.Tensor, b: torch.Tensor, js: Tuple[int, ...]
) -> List[torch.Tensor]:
    """Exact closed-form B^J_{1,1}(a, b) (component normalization) matching Wigner-3j to < 1e-15."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    parts: List[torch.Tensor] = []
    for J in js:
        if J == 0:
            parts.append((_SQRT3 * (ax * bx + ay * by + az * bz)).unsqueeze(-1))
        elif J == 1:
            cx = _THREE_OVER_SQRT2 * (ay * bz - az * by)
            cy = _THREE_OVER_SQRT2 * (az * bx - ax * bz)
            cz = _THREE_OVER_SQRT2 * (ax * by - ay * bx)
            parts.append(torch.stack((cx, cy, cz), dim=-1))
        elif J == 2:
            q0 = _THREE_OVER_SQRT2 * (ax * bz + az * bx)
            q1 = _THREE_OVER_SQRT2 * (ax * by + ay * bx)
            q2 = _SQRT6 * (ay * by - 0.5 * (ax * bx + az * bz))
            q3 = _THREE_OVER_SQRT2 * (ay * bz + az * by)
            q4 = _THREE_OVER_SQRT2 * (az * bz - ax * bx)
            parts.append(torch.stack((q0, q1, q2, q3, q4), dim=-1))
    return parts


class BipolarSphericalHarmonics(nn.Module):
    r"""Singularity-free polynomial bipolar spherical harmonics :math:`B^J_{l\lambda}(a, b)`.

    Computes the orthogonal coupling

    .. math::

        B^J_{l\lambda}(a, b)_M = \sqrt{2J+1} \sum_{m, n} C^{l\lambda J}_{mnM}\,
            Y^l_m(c_a a)\, Y^\lambda_n(c_b b)

    for all :math:`0 \le l \le l_{\max,a}`, :math:`0 \le \lambda \le l_{\max,b}`,
    and :math:`|l-\lambda| \le J \le \min(l+\lambda, j_{\max})`, with exact
    :math:`O(3) \times \mathbb{Z}_2^T` labels
    :math:`(J,\, p_a^l p_b^\lambda,\, t_a^l t_b^\lambda)`.

    All coupling matrices are registered in ``torch.get_default_dtype()`` with
    exact 53-bit FP64 re-materialization in ``_apply()`` when moved to ``float64``.
    """

    def __init__(
        self,
        lmax_a: int,
        lmax_b: int,
        parity_a: int,
        parity_b: int,
        time_reversal_a: int,
        time_reversal_b: int,
        normalize_a: bool = True,
        normalize_b: bool = False,
        normalization: str = "component",
        jmax: Optional[int] = None,
        scale_a: float = 1.0,
        scale_b: float = 1.0,
        eps: float = 0.0,
    ):
        super().__init__()
        if not (isinstance(lmax_a, int) and 0 <= lmax_a <= 4):
            raise ValueError(f"lmax_a must be an integer in 0..4, got {lmax_a!r}")
        if not (isinstance(lmax_b, int) and 0 <= lmax_b <= 4):
            raise ValueError(f"lmax_b must be an integer in 0..4, got {lmax_b!r}")
        if jmax is not None and not (
            isinstance(jmax, int) and not isinstance(jmax, bool) and jmax >= 0
        ):
            raise ValueError(f"jmax must be None or non-negative int, got {jmax!r}")
        for name, val in (
            ("parity_a", parity_a),
            ("parity_b", parity_b),
            ("time_reversal_a", time_reversal_a),
            ("time_reversal_b", time_reversal_b),
        ):
            if val not in (-1, 1):
                raise ValueError(f"{name} must be -1 or +1, got {val!r}")

        self.lmax_a = lmax_a
        self.lmax_b = lmax_b
        self.jmax = jmax
        self.normalize_a = bool(normalize_a)
        self.normalize_b = bool(normalize_b)
        self.normalization = normalization
        self.eps = float(eps)
        self.scale_a = _check_positive_scale(scale_a, "scale_a")
        self.scale_b = _check_positive_scale(scale_b, "scale_b")
        self._factor_a = 1.0 if normalize_a else self.scale_a
        self._factor_b = 1.0 if normalize_b else self.scale_b

        self.sh_a = SphericalHarmonics(
            lmax_a,
            normalize=normalize_a,
            normalization=normalization,
            parity=parity_a,
            time_reversal=time_reversal_a,
            eps=eps,
        )
        self.sh_b = SphericalHarmonics(
            lmax_b,
            normalize=normalize_b,
            normalization=normalization,
            parity=parity_b,
            time_reversal=time_reversal_b,
            eps=eps,
        )

        # Fast closed-form path when lmax_a == 1, lmax_b == 1, component norm
        self._fast_11 = lmax_a == 1 and lmax_b == 1 and normalization == "component"
        self._fast_11_blocks: Tuple[Tuple[int, int, Tuple[int, ...]], ...] = tuple(
            _bipolar_blocks(lmax_a, lmax_b, jmax)
        )

        labels = []
        plan = []
        block_meta = []
        buf_spec = []
        offset = 0
        default_dtype = torch.get_default_dtype()
        for l, lam, js in self._fast_11_blocks:
            p = (parity_a**l) * (parity_b**lam)
            t = (time_reversal_a**l) * (time_reversal_b**lam)
            is_identity = (l == 0 or lam == 0) and len(js) == 1
            if is_identity:
                plan.append((l, lam, None, js, True))
            else:
                name = f"_couple_{l}_{lam}"
                U_f64 = _cached_single_coupling_f64(l, lam, js)
                self.register_buffer(
                    name, U_f64.to(dtype=default_dtype, copy=True), persistent=False
                )
                buf_spec.append((name, l, lam, js))
                plan.append((l, lam, name, js, False))
            for J in js:
                width = 2 * J + 1
                labels.append((1, Irrep(J, p, t)))
                block_meta.append((l, lam, J, slice(offset, offset + width)))
                offset += width
        self._plan = tuple(plan)
        self._buf_spec = tuple(buf_spec)
        self.block_slices = tuple(block_meta)
        self.irreps_out = Irreps(labels)
        self.dim_out = self.irreps_out.dim

    def _apply(self, fn):
        ret = super()._apply(fn)
        for name, l, lam, js in self._buf_spec:
            buf = getattr(self, name)
            if buf.dtype == torch.float64:
                exact = _cached_single_coupling_f64(l, lam, js)
                buf.data.copy_(exact.to(device=buf.device, dtype=torch.float64))
        return ret

    @classmethod
    def for_bond_spin(
        cls,
        lmax_r: int = 2,
        lmax_s: int = 1,
        jmax: Optional[int] = 2,
        spin_scale: float = 1.0,
        normalization: str = "component",
    ) -> "BipolarSphericalHarmonics":
        """Factory for bond (polar, T-even, normalized) x spin (axial, T-odd, polynomial)."""
        return cls(
            lmax_a=lmax_r,
            lmax_b=lmax_s,
            parity_a=-1,
            parity_b=1,
            time_reversal_a=1,
            time_reversal_b=-1,
            normalize_a=True,
            normalize_b=False,
            normalization=normalization,
            jmax=jmax,
            scale_a=1.0,
            scale_b=spin_scale,
        )

    @classmethod
    def for_spin_spin(
        cls,
        lmax: int = 1,
        jmax: Optional[int] = 2,
        spin_scale: float = 1.0,
        normalization: str = "component",
    ) -> "BipolarSphericalHarmonics":
        """Factory for spin_i x spin_j (both axial, T-odd, polynomial, singularity-free)."""
        return cls(
            lmax_a=lmax,
            lmax_b=lmax,
            parity_a=1,
            parity_b=1,
            time_reversal_a=-1,
            time_reversal_b=-1,
            normalize_a=False,
            normalize_b=False,
            normalization=normalization,
            jmax=jmax,
            scale_a=spin_scale,
            scale_b=spin_scale,
        )

    def get_block_slice(self, l: int, lam: int, J: int) -> slice:
        """Return the output slice for the specific ``B^J_{l, lam}`` block."""
        for bl, blam, bJ, sl in self.block_slices:
            if bl == l and blam == lam and bJ == J:
                return sl
        raise KeyError(f"Block (l={l}, lam={lam}, J={J}) not present in {self}")

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self._factor_a != 1.0:
            a = a * self._factor_a
        if self._factor_b != 1.0:
            b = b * self._factor_b

        if self._fast_11:
            if self.normalize_a:
                a = _safe_normalize(a, eps=self.eps)
            if self.normalize_b:
                b = _safe_normalize(b, eps=self.eps)
            lead = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
            out_dtype = torch.promote_types(a.dtype, b.dtype)
            a_b = a.expand(lead + (3,))
            b_b = b.expand(lead + (3,))
            pieces: List[torch.Tensor] = []
            for l, lam, js in self._fast_11_blocks:
                if l == 0 and lam == 0:
                    pieces.append(
                        torch.ones(lead + (1,), dtype=out_dtype, device=a_b.device)
                    )
                elif l == 0 and lam == 1:
                    pieces.append(b_b * _SQRT3)
                elif l == 1 and lam == 0:
                    pieces.append(a_b * _SQRT3)
                else:
                    pieces.extend(_closed_form_b11_components(a_b, b_b, js))
            return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)

        ya = self.sh_a(a)
        yb = self.sh_b(b)
        pieces = []
        for l, lam, name, js, is_identity in self._plan:
            ya_l = ya[..., l * l : (l + 1) ** 2]
            yb_lam = yb[..., lam * lam : (lam + 1) ** 2]
            if is_identity:
                pieces.append(ya_l * yb_lam)
            else:
                outer = ya_l.unsqueeze(-1) * yb_lam.unsqueeze(-2)
                U = getattr(self, name)
                if U.dtype != outer.dtype:
                    if outer.dtype == torch.float64:
                        U = _cached_single_coupling_f64(l, lam, js).to(
                            device=outer.device, dtype=torch.float64
                        )
                    else:
                        U = U.to(dtype=outer.dtype)
                pieces.append(outer.flatten(-2) @ U)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)


def bipolar_spherical_harmonics(
    lmax_a: int,
    lmax_b: int,
    a: torch.Tensor,
    b: torch.Tensor,
    parity_a: int,
    parity_b: int,
    time_reversal_a: int,
    time_reversal_b: int,
    normalize_a: bool = True,
    normalize_b: bool = False,
    normalization: str = "component",
    jmax: Optional[int] = None,
    scale_a: float = 1.0,
    scale_b: float = 1.0,
) -> Tuple[Irreps, torch.Tensor]:
    """Functional bipolar spherical harmonics returning ``(irreps_out, values)``.

    Compile-friendly under ``torch.compile(fullgraph=True, dynamic=True)``; for repeated
    GPU training calls, prefer holding a :class:`BipolarSphericalHarmonics` module so
    coupling buffers stay resident on the target device.
    """
    if not (isinstance(lmax_a, int) and 0 <= lmax_a <= 4):
        raise ValueError(f"lmax_a must be an integer in 0..4, got {lmax_a!r}")
    if not (isinstance(lmax_b, int) and 0 <= lmax_b <= 4):
        raise ValueError(f"lmax_b must be an integer in 0..4, got {lmax_b!r}")
    if jmax is not None and not (
        isinstance(jmax, int) and not isinstance(jmax, bool) and jmax >= 0
    ):
        raise ValueError(f"jmax must be None or non-negative int, got {jmax!r}")
    for name, val in (
        ("parity_a", parity_a),
        ("parity_b", parity_b),
        ("time_reversal_a", time_reversal_a),
        ("time_reversal_b", time_reversal_b),
    ):
        if val not in (-1, 1):
            raise ValueError(f"{name} must be -1 or +1, got {val!r}")
    sa = _check_positive_scale(scale_a, "scale_a")
    sb = _check_positive_scale(scale_b, "scale_b")

    if not normalize_a and sa != 1.0:
        a = a * sa
    if not normalize_b and sb != 1.0:
        b = b * sb

    ya = spherical_harmonics(
        list(range(lmax_a + 1)), a, normalize=normalize_a, normalization=normalization
    )
    yb = spherical_harmonics(
        list(range(lmax_b + 1)), b, normalize=normalize_b, normalization=normalization
    )
    blocks = _bipolar_blocks(lmax_a, lmax_b, jmax)
    mats = _cached_coupling_matrices(lmax_a, lmax_b, jmax)
    labels = []
    pieces = []
    for (l, lam, js), U_cpu in zip(blocks, mats):
        p = (parity_a**l) * (parity_b**lam)
        t = (time_reversal_a**l) * (time_reversal_b**lam)
        for J in js:
            labels.append((1, Irrep(J, p, t)))
        ya_l = ya[..., l * l : (l + 1) ** 2]
        yb_lam = yb[..., lam * lam : (lam + 1) ** 2]
        if (l == 0 or lam == 0) and len(js) == 1:
            pieces.append(ya_l * yb_lam)
        else:
            outer = ya_l.unsqueeze(-1) * yb_lam.unsqueeze(-2)
            U = U_cpu.to(device=outer.device, dtype=outer.dtype)
            pieces.append(outer.flatten(-2) @ U)
    out = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
    return Irreps(labels), out


def bond_spin(
    r: torch.Tensor,
    s: torch.Tensor,
    lmax_r: int,
    lmax_s: int,
    jmax: Optional[int] = None,
    spin_scale: float = 1.0,
) -> Tuple[Irreps, torch.Tensor]:
    """Functional bipolar harmonics for bond ``r`` (polar, T-even) and spin ``s`` (axial, T-odd)."""
    return bipolar_spherical_harmonics(
        lmax_a=lmax_r,
        lmax_b=lmax_s,
        a=r,
        b=s,
        parity_a=-1,
        parity_b=1,
        time_reversal_a=1,
        time_reversal_b=-1,
        normalize_a=True,
        normalize_b=False,
        jmax=jmax,
        scale_b=spin_scale,
    )


def spin_spin(
    si: torch.Tensor,
    sj: torch.Tensor,
    lmax: int,
    jmax: Optional[int] = None,
    spin_scale: float = 1.0,
) -> Tuple[Irreps, torch.Tensor]:
    """Functional bipolar harmonics for two classical spins (axial, T-odd, polynomial)."""
    return bipolar_spherical_harmonics(
        lmax_a=lmax,
        lmax_b=lmax,
        a=si,
        b=sj,
        parity_a=1,
        parity_b=1,
        time_reversal_a=-1,
        time_reversal_b=-1,
        normalize_a=False,
        normalize_b=False,
        jmax=jmax,
        scale_a=spin_scale,
        scale_b=spin_scale,
    )


def spin_cross_axial(si: torch.Tensor, sj: torch.Tensor) -> torch.Tensor:
    r"""Singularity-free axial T-even antisymmetric spin pair vector ``1ee``.

    Returns :math:`\frac{1}{\sqrt{2}} (\tilde{\mathbf{s}}_i \times \tilde{\mathbf{s}}_j)`
    of shape ``(..., 3)`` transforming as ``1ee`` (J=1, p=+1, t=+1).
    Smooth everywhere including collinear alignment :math:`\theta \to 0`.
    """
    ax, ay, az = si[..., 0], si[..., 1], si[..., 2]
    bx, by, bz = sj[..., 0], sj[..., 1], sj[..., 2]
    inv_sqrt2 = 1.0 / _SQRT2
    return torch.stack(
        (
            inv_sqrt2 * (ay * bz - az * by),
            inv_sqrt2 * (az * bx - ax * bz),
            inv_sqrt2 * (ax * by - ay * bx),
        ),
        dim=-1,
    )


def spin_quadrupole_symmetric(si: torch.Tensor, sj: torch.Tensor) -> torch.Tensor:
    r"""Singularity-free symmetric traceless quadrupole ``2ee`` of two spins.

    Returns the 5-component ``J=2, p=+1, t=+1`` (``2ee``) harmonic quadrupole
    constructed from the symmetric part of :math:`\tilde{\mathbf{s}}_i \otimes \tilde{\mathbf{s}}_j`,
    matching ``B^2_{1,1}(si, sj)`` up to a positive constant factor (smooth at all angles).
    """
    sx_i, sy_i, sz_i = si[..., 0], si[..., 1], si[..., 2]
    sx_j, sy_j, sz_j = sj[..., 0], sj[..., 1], sj[..., 2]
    q0 = 0.5 * _SQRT15 * (sx_i * sz_j + sz_i * sx_j)
    q1 = 0.5 * _SQRT15 * (sx_i * sy_j + sy_i * sx_j)
    q2 = _SQRT5 * (sy_i * sy_j - 0.5 * (sx_i * sx_j + sz_i * sz_j))
    q3 = 0.5 * _SQRT15 * (sy_i * sz_j + sz_i * sy_j)
    q4 = 0.5 * _SQRT15 * (sz_i * sz_j - sx_i * sx_j)
    return torch.stack((q0, q1, q2, q3, q4), dim=-1)


def bipolar_spin_pair_j1_j2(
    si: torch.Tensor,
    sj: torch.Tensor,
    spin_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fused closed-form ``(B^1_{1,1}, B^2_{1,1})`` for a pair of classical spins.

    Computes the exact ``1ee`` axial antisymmetric cross block ``(..., 3)`` and
    ``2ee`` symmetric traceless quadrupole block ``(..., 5)`` matching
    :class:`BipolarSphericalHarmonics.for_spin_spin(lmax=1, jmax=2, spin_scale=...)`
    to machine precision (``< 1e-15`` in FP64) in one register-resident pointwise pass
    with zero ``(..., 16)`` or ``(..., 9)`` outer-product materialization.
    """
    scale = _check_positive_scale(spin_scale, "spin_scale")
    scale2 = scale * scale
    c_cross = _THREE_OVER_SQRT2 * scale2
    c_quad = _THREE_OVER_SQRT2 * scale2
    c_q2 = _SQRT6 * scale2

    ax, ay, az = si[..., 0], si[..., 1], si[..., 2]
    bx, by, bz = sj[..., 0], sj[..., 1], sj[..., 2]

    ay_bz = ay * bz
    az_by = az * by
    az_bx = az * bx
    ax_bz = ax * bz
    ax_by = ax * by
    ay_bx = ay * bx
    ax_bx = ax * bx
    ay_by = ay * by
    az_bz = az * bz

    b_cross = torch.stack(
        (
            c_cross * (ay_bz - az_by),
            c_cross * (az_bx - ax_bz),
            c_cross * (ax_by - ay_bx),
        ),
        dim=-1,
    )
    b_quad = torch.stack(
        (
            c_quad * (ax_bz + az_bx),
            c_quad * (ax_by + ay_bx),
            c_q2 * (ay_by - 0.5 * (ax_bx + az_bz)),
            c_quad * (ay_bz + az_by),
            c_quad * (az_bz - ax_bx),
        ),
        dim=-1,
    )
    return b_cross, b_quad
