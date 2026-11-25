"""Parameter-free orthogonal L-S coupling layer from decoupled to coupled irreps.

Maps :math:`(l, p | \\lambda, t) \\to \\bigoplus_{J=|l-\\lambda|}^{\\min(l+\\lambda, j_{\\max})} (J, p, t)`,
strictly preserving ``(p, t)`` grading (R2) with zero learnable parameters and
static constant buffers compatible with ``torch.compile(fullgraph=True, dynamic=True)``.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .irreps import DIrrep, DIrreps, Irrep, Irreps
from .wigner import _ordinary_tensors, wigner_3j

__all__ = ["Couple", "CoupleBlock"]

_COUPLE_MAT_CACHE: Dict[Tuple[int, int, Tuple[int, ...]], torch.Tensor] = {}
_COUPLE_MAT_T_CACHE: Dict[Tuple[int, int, Tuple[int, ...]], torch.Tensor] = {}


def _cached_couple_matrix_f64(l: int, lam: int, js: Tuple[int, ...]) -> torch.Tensor:
    key = (l, lam, js)
    mat = _COUPLE_MAT_CACHE.get(key)
    if mat is None:
        with _ordinary_tensors():
            dim_in = (2 * l + 1) * (2 * lam + 1)
            cols = [
                math.sqrt(2 * J + 1)
                * wigner_3j(l, lam, J, dtype=torch.float64, device="cpu").reshape(
                    dim_in, 2 * J + 1
                )
                for J in js
            ]
            mat = torch.cat(cols, dim=1).contiguous()
        _COUPLE_MAT_CACHE[key] = mat
    return mat


_cached_couple_matrix_f64._dynamo_marked_constant = True


def _cached_couple_matrix_t_f64(l: int, lam: int, js: Tuple[int, ...]) -> torch.Tensor:
    key = (l, lam, js)
    mat_t = _COUPLE_MAT_T_CACHE.get(key)
    if mat_t is None:
        with _ordinary_tensors():
            mat_t = _cached_couple_matrix_f64(l, lam, js).T.contiguous()
        _COUPLE_MAT_T_CACHE[key] = mat_t
    return mat_t


_cached_couple_matrix_t_f64._dynamo_marked_constant = True


class Couple(nn.Module):
    r"""Parameter-free orthogonal coupling :math:`O(3)_{\mathrm{lat}} \times O(3)_{\mathrm{spin}} \to O(3) \times \mathbb{Z}_2^T`.

    For each block ``(mul, (l, p | lam, t))`` in ``irreps_in``, computes

    .. math::

        \mathrm{couple}(x)_J[u, M] = \sqrt{2J+1} \sum_{m, n} C^{l \lambda J}_{m n M}\, x[u, m, n]

    emitting blocks ``(mul, (J, p, t))`` for :math:`|l-\lambda| \le J \le \min(l+\lambda, j_{\max})`.
    Without truncation (``jmax is None`` or ``jmax >= max(l + lam)``), this is an exact
    orthogonal change of basis with exact ``inverse(y)``.

    Buffers are registered in ``torch.get_default_dtype()`` with exact 53-bit FP64
    re-materialization in ``_apply()`` when converted to ``float64``.
    """

    def __init__(self, irreps_in, jmax: Optional[int] = None):
        super().__init__()
        if jmax is not None and not (
            isinstance(jmax, int) and not isinstance(jmax, bool) and jmax >= 0
        ):
            raise ValueError(f"jmax must be None or non-negative int, got {jmax!r}")
        self.irreps_in = DIrreps(irreps_in)
        self.jmax = jmax

        out_blocks = []
        plan = []
        buf_spec = []
        truncated = False
        out_offset = 0
        default_dtype = torch.get_default_dtype()
        for (mul, ir), sl in zip(self.irreps_in, self.irreps_in.slices()):
            js_all = tuple(range(abs(ir.l - ir.lam), ir.l + ir.lam + 1))
            js = tuple(J for J in js_all if jmax is None or J <= jmax)
            if mul > 0 and len(js) < len(js_all):
                truncated = True
            if not js:
                continue
            is_identity = (ir.l == 0 or ir.lam == 0) and len(js) == 1
            if is_identity:
                buf_name = None
            else:
                buf_name = f"_couple_{ir.l}_{ir.lam}"
                if not hasattr(self, buf_name):
                    U_f64 = _cached_couple_matrix_f64(ir.l, ir.lam, js)
                    self.register_buffer(
                        buf_name,
                        U_f64.to(dtype=default_dtype, copy=True),
                        persistent=False,
                    )
                    buf_spec.append((buf_name, ir.l, ir.lam, js))
            widths = tuple(2 * J + 1 for J in js)
            total_width = sum(widths)
            mul_widths = tuple(mul * w for w in widths)
            plan.append(
                (
                    sl.start,
                    mul,
                    ir.dim,
                    buf_name,
                    ir.l,
                    ir.lam,
                    js,
                    widths,
                    mul_widths,
                    out_offset,
                    total_width,
                    is_identity,
                )
            )
            for J in js:
                out_blocks.append((mul, Irrep(J, ir.p, ir.t)))
            out_offset += mul * total_width

        self._plan = tuple(plan)
        self._buf_spec = tuple(buf_spec)
        self.irreps_out = Irreps(out_blocks)
        self.truncated = truncated
        self._in_dim = self.irreps_in.dim
        self._out_dim = self.irreps_out.dim

    def _apply(self, fn):
        ret = super()._apply(fn)
        for name, l, lam, js in self._buf_spec:
            buf = getattr(self, name)
            if buf.dtype == torch.float64:
                exact = _cached_couple_matrix_f64(l, lam, js)
                buf.data.copy_(exact.to(device=buf.device, dtype=torch.float64))
        return ret

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out})"

    def _get_U(
        self, buf_name: str, l: int, lam: int, js: Tuple[int, ...], ref: torch.Tensor
    ) -> torch.Tensor:
        U = getattr(self, buf_name)
        if U.dtype != ref.dtype:
            if ref.dtype == torch.float64:
                return _cached_couple_matrix_f64(l, lam, js).to(
                    device=ref.device, dtype=torch.float64
                )
            return U.to(dtype=ref.dtype)
        return U

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Couple ``x`` of shape ``(..., irreps_in.dim)`` to ``(..., irreps_out.dim)``."""
        if x.shape[-1] != self._in_dim:
            raise ValueError(
                f"Expected trailing dimension {self._in_dim}, got {x.shape[-1]}"
            )
        batch = x.shape[:-1]
        batch_size = 1
        for s in batch:
            batch_size = batch_size * s
        x_flat = x.reshape(batch_size, self._in_dim)
        pieces = []
        for (
            start,
            mul,
            dim,
            buf_name,
            l,
            lam,
            js,
            widths,
            _,
            _,
            _,
            is_identity,
        ) in self._plan:
            if mul == 0:
                continue
            seg = x_flat[:, start : start + mul * dim]
            if is_identity:
                pieces.append(seg)
            else:
                U = self._get_U(buf_name, l, lam, js, x)
                blk = seg.reshape(batch_size, mul, dim)
                coupled = torch.matmul(blk, U)
                if len(widths) == 1:
                    pieces.append(coupled.reshape(batch_size, mul * widths[0]))
                else:
                    for piece, w in zip(torch.split(coupled, widths, dim=-1), widths):
                        pieces.append(piece.reshape(batch_size, mul * w))
        if not pieces:
            return x.new_zeros(batch + (0,))
        out = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
        return out.reshape(batch + (self._out_dim,))

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Exact orthogonal inverse ``(..., irreps_out.dim) -> (..., irreps_in.dim)``."""
        if self.truncated:
            raise RuntimeError(
                "Couple.inverse is undefined when jmax truncated some J components"
            )
        if y.shape[-1] != self._out_dim:
            raise ValueError(
                f"Expected trailing dimension {self._out_dim}, got {y.shape[-1]}"
            )
        batch = y.shape[:-1]
        batch_size = 1
        for s in batch:
            batch_size = batch_size * s
        y_flat = y.reshape(batch_size, self._out_dim)
        blocks = []
        for (
            _,
            mul,
            dim,
            buf_name,
            l,
            lam,
            js,
            widths,
            mul_widths,
            out_offset,
            total_width,
            is_identity,
        ) in self._plan:
            if mul == 0:
                continue
            seg = y_flat[:, out_offset : out_offset + mul * total_width]
            if is_identity:
                blocks.append(seg)
            else:
                U = self._get_U(buf_name, l, lam, js, y)
                if len(widths) == 1:
                    yb = seg.reshape(batch_size, mul, widths[0])
                else:
                    parts = torch.split(seg, mul_widths, dim=-1)
                    yb = torch.cat(
                        [p.reshape(batch_size, mul, w) for p, w in zip(parts, widths)],
                        dim=-1,
                    )
                blocks.append(torch.matmul(yb, U.T).reshape(batch_size, mul * dim))
        if not blocks:
            return y.new_zeros(batch + (0,))
        out = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)
        return out.reshape(batch + (self._in_dim,))


class CoupleBlock(nn.Module):
    """Structured parameter-free L-S coupling for a single ``(..., 2l+1, 2lam+1, C)`` block.

    Returns a dictionary ``{J: Tensor(..., 2J+1, C)}`` with exact ``(J, p, t)``
    metadata stored in ``self.irreps_by_j``. Pre-stores contiguous ``_U_T`` in
    ``torch.get_default_dtype()`` with exact 53-bit FP64 re-materialization in
    ``_apply()``.
    """

    def __init__(self, ir: DIrrep, jmax: Optional[int] = None):
        super().__init__()
        self.ir_in = DIrrep(ir)
        if jmax is not None and not (
            isinstance(jmax, int) and not isinstance(jmax, bool) and jmax >= 0
        ):
            raise ValueError(f"jmax must be None or non-negative int, got {jmax!r}")
        self.jmax = jmax
        self.js: Tuple[int, ...] = tuple(
            J
            for J in range(
                abs(self.ir_in.l - self.ir_in.lam), self.ir_in.l + self.ir_in.lam + 1
            )
            if jmax is None or J <= jmax
        )
        self.irreps_by_j: Dict[int, Irrep] = {
            J: Irrep(J, self.ir_in.p, self.ir_in.t) for J in self.js
        }
        self._dim_l = 2 * self.ir_in.l + 1
        self._dim_lam = 2 * self.ir_in.lam + 1
        self._dim_in = self._dim_l * self._dim_lam
        self._widths = tuple(2 * J + 1 for J in self.js)
        self._is_identity = (self.ir_in.l == 0 or self.ir_in.lam == 0) and len(
            self.js
        ) == 1
        if self.js and not self._is_identity:
            U_T_f64 = _cached_couple_matrix_t_f64(self.ir_in.l, self.ir_in.lam, self.js)
            self.register_buffer(
                "_U_T",
                U_T_f64.to(dtype=torch.get_default_dtype(), copy=True),
                persistent=False,
            )
        else:
            self.register_buffer("_U_T", None, persistent=False)

    def _apply(self, fn):
        ret = super()._apply(fn)
        if self._U_T is not None and self._U_T.dtype == torch.float64:
            exact = _cached_couple_matrix_t_f64(self.ir_in.l, self.ir_in.lam, self.js)
            self._U_T.data.copy_(exact.to(device=self._U_T.device, dtype=torch.float64))
        return ret

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Map ``(..., 2l+1, 2lam+1, C)`` -> ``{J: (..., 2J+1, C)}``."""
        if x.shape[-3] != self._dim_l or x.shape[-2] != self._dim_lam:
            raise ValueError(
                f"Expected shape (..., {self._dim_l}, {self._dim_lam}, C), got {tuple(x.shape)}"
            )
        if not self.js:
            return {}
        lead = x.shape[:-3]
        channels = x.shape[-1]
        x_flat = x.reshape(lead + (self._dim_in, channels))
        if self._is_identity:
            return {self.js[0]: x_flat}
        U_T = self._U_T
        if U_T.dtype != x.dtype:
            if x.dtype == torch.float64:
                U_T = _cached_couple_matrix_t_f64(
                    self.ir_in.l, self.ir_in.lam, self.js
                ).to(device=x.device, dtype=torch.float64)
            else:
                U_T = U_T.to(dtype=x.dtype)
        coupled = torch.matmul(U_T, x_flat)
        if len(self._widths) == 1:
            return {self.js[0]: coupled}
        splits = torch.split(coupled, self._widths, dim=-2)
        return {J: part for J, part in zip(self.js, splits)}
