"""Compile-friendly static sparse Clebsch-Gordan and tensor-contraction builders (R1).

All contraction paths are validated strictly against full ``(l, p, t)`` or
``(l, p | lam, t)`` selection rules at construction time. Backends and forward
passes consume static constant buffers / register-fused sparse schedules and
NEVER regenerate paths from erased labels (Rule R1). Every contraction
specification carries a canonical SHA-256 hash preserving all ``(p, t)``
grading labels (Rule R4).
"""

import math
import operator
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .irreps import (
    DIrrep,
    DIrreps,
    Irrep,
    Irreps,
    decoupled_body_key,
    decoupled_clebsch_gordan,
    o3_body_key,
    symmetry_spec_hash,
)
from .wigner import _ordinary_tensors, wigner_3j

__all__ = [
    "ContractionPath",
    "SparseCGSpec",
    "build_sparse_cg",
    "validate_contraction_paths",
    "StaticTensorProduct",
    "SparseCGContract",
    "SparseBispectrumContract",
]

IrrepsLike = Union[Irreps, DIrreps]
_UNROLL_NNZ_THRESHOLD = 128


class ContractionPath(NamedTuple):
    """Validated explicit contraction instruction with final normalized path weight."""

    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: Tuple[int, ...]
    body_key: Tuple


class SparseCGSpec(NamedTuple):
    """Static sparse COO representation of a single Clebsch-Gordan coupling body."""

    src1_idx: torch.Tensor
    src2_idx: torch.Tensor
    dst_idx: torch.Tensor
    values: torch.Tensor
    shape: Tuple[int, int, int]
    body_key: Tuple
    canonical_hash: str


_PATH_SHAPE = {
    "uvw": lambda m1, m2, m3: (m1, m2, m3),
    "uvu": lambda m1, m2, m3: (m1, m2),
    "uvv": lambda m1, m2, m3: (m1, m2),
    "uuw": lambda m1, m2, m3: (m1, m3),
    "uuu": lambda m1, m2, m3: (m1,),
}

_NUM_ELEMENTS = {
    "uvw": lambda m1, m2: m1 * m2,
    "uvu": lambda m1, m2: m2,
    "uvv": lambda m1, m2: m1,
    "uuw": lambda m1, m2: m1,
    "uuu": lambda m1, m2: 1,
}

_EINSUM_WEIGHTED = {
    "uvw": "{z}uvw,zui,zvj,ijk->zwk",
    "uvu": "{z}uv,zui,zvj,ijk->zuk",
    "uvv": "{z}uv,zui,zvj,ijk->zvk",
    "uuw": "{z}uw,zui,zuj,ijk->zwk",
    "uuu": "{z}u,zui,zuj,ijk->zuk",
}

_EINSUM_UNWEIGHTED = {
    "uvu": "zui,zvj,ijk->zuk",
    "uvv": "zui,zvj,ijk->zvk",
    "uuw": "zui,zuj,ijk->zk",
    "uuu": "zui,zuj,ijk->zuk",
}


def _prod(seq: Sequence[int]) -> int:
    acc = 1
    for v in seq:
        acc *= int(v)
    return acc


def _is_decoupled_spec(spec: Any) -> bool:
    if isinstance(spec, (DIrrep, DIrreps)):
        return True
    if isinstance(spec, (Irrep, Irreps)):
        return False
    if isinstance(spec, str):
        return "|" in spec
    if isinstance(spec, (tuple, list)) and spec:
        first = spec[0]
        if isinstance(first, DIrrep):
            return True
        if isinstance(first, (tuple, list)) and len(first) == 2:
            return _is_decoupled_spec(first[1])
    return False


def _coerce_irreps(spec: Any, decoupled: bool) -> IrrepsLike:
    return DIrreps(spec) if decoupled else Irreps(spec)


def _index_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer index, got bool {value!r}")
    return operator.index(value)


_CG_BODY_CACHE: Dict[Tuple, torch.Tensor] = {}


def _cached_cg_body_f64(body_key: Tuple) -> torch.Tensor:
    mat = _CG_BODY_CACHE.get(body_key)
    if mat is None:
        with _ordinary_tensors():
            if body_key[0] == "o3":
                _, l1, l2, l3 = body_key
                mat = wigner_3j(l1, l2, l3, dtype=torch.float64, device="cpu")
            elif body_key[0] == "o3xo3":
                _, (l1, l2, l3), (lam1, lam2, lam3) = body_key
                c_lat = wigner_3j(l1, l2, l3, dtype=torch.float64, device="cpu")
                c_spin = wigner_3j(lam1, lam2, lam3, dtype=torch.float64, device="cpu")
                mat = (
                    torch.einsum("ikm,jln->ijklmn", c_lat, c_spin)
                    .reshape(
                        (2 * l1 + 1) * (2 * lam1 + 1),
                        (2 * l2 + 1) * (2 * lam2 + 1),
                        (2 * l3 + 1) * (2 * lam3 + 1),
                    )
                    .contiguous()
                )
            else:
                raise ValueError(f"Unknown body_key: {body_key!r}")
        _CG_BODY_CACHE[body_key] = mat
    return mat


_cached_cg_body_f64._dynamo_marked_constant = True


def _cg_body_f64(ir1: Any, ir2: Any, ir3: Any) -> Tuple[torch.Tensor, Tuple]:
    if isinstance(ir1, DIrrep):
        body = decoupled_clebsch_gordan(
            ir1, ir2, ir3, dtype=torch.float64, device="cpu"
        )
        return body, decoupled_body_key(ir1, ir2, ir3)
    r1, r2, r3 = Irrep(ir1), Irrep(ir2), Irrep(ir3)
    if r1.p * r2.p != r3.p:
        raise ValueError(f"{r1} x {r2} -> {r3} violates spatial parity p3 = p1 * p2")
    if r1.t * r2.t != r3.t:
        raise ValueError(
            f"{r1} x {r2} -> {r3} violates time-reversal parity t3 = t1 * t2"
        )
    if not (abs(r1.l - r2.l) <= r3.l <= r1.l + r2.l):
        raise ValueError(f"{r1} x {r2} -> {r3} violates angular momentum triangle rule")
    body = wigner_3j(r1.l, r2.l, r3.l, dtype=torch.float64, device="cpu")
    return body, o3_body_key(r1, r2, r3)


def _build_sign_factored_schedule(
    spec: SparseCGSpec,
) -> Tuple[Tuple[Tuple[float, Tuple[Tuple[int, int, int], ...]], ...], ...]:
    """Group sparse CG triplets by output index ``k`` and equal ``|C|`` magnitude.

    Returns a tuple of length ``d3``, where entry ``k`` is a tuple of
    ``(abs_coeff, ((i, j, sign), ...))`` groups so forward evaluation factors
    equal-magnitude CG entries into minimal register-resident arithmetic.
    """
    d3 = spec.shape[2]
    by_k: List[Dict[int, List[Tuple[int, int, int, float]]]] = [{} for _ in range(d3)]
    s1_list = spec.src1_idx.tolist()
    s2_list = spec.src2_idx.tolist()
    dst_list = spec.dst_idx.tolist()
    val_list = spec.values.tolist()
    for i, j, k, val in zip(s1_list, s2_list, dst_list, val_list):
        abs_v = abs(val)
        sign = 1 if val >= 0.0 else -1
        bucket_key = round(abs_v * 1e12)
        by_k[k].setdefault(bucket_key, []).append((i, j, sign, abs_v))

    schedule = []
    for k in range(d3):
        groups = []
        for _, entries in sorted(by_k[k].items()):
            mag = entries[0][3]
            terms = tuple((i, j, sign) for i, j, sign, _ in entries)
            groups.append((mag, terms))
        schedule.append(tuple(groups))
    return tuple(schedule)


def _eval_sign_factored_schedule(
    schedule: Tuple[Tuple[Tuple[float, Tuple[Tuple[int, int, int], ...]], ...], ...],
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a sign-factored sparse CG schedule into ``(..., d3, C)`` (zero atomics)."""
    zero_template: Optional[torch.Tensor] = None
    out_slices: List[torch.Tensor] = []
    for groups in schedule:
        acc: Optional[torch.Tensor] = None
        for mag, terms in groups:
            grp_acc: Optional[torch.Tensor] = None
            for i, j, sign in terms:
                prod = x1[..., i, :] * x2[..., j, :]
                if grp_acc is None:
                    grp_acc = prod if sign > 0 else -prod
                else:
                    grp_acc = grp_acc + prod if sign > 0 else grp_acc - prod
            assert grp_acc is not None
            scaled = grp_acc if mag == 1.0 else grp_acc * mag
            acc = scaled if acc is None else acc + scaled
        if acc is None:
            if zero_template is None:
                zero_template = torch.zeros_like(x1[..., 0, :])
            acc = zero_template
        out_slices.append(acc)
    return torch.stack(out_slices, dim=-2)


def build_sparse_cg(
    ir1: Any,
    ir2: Any,
    ir3: Any,
    *,
    tol: float = 1e-12,
    scale: float = 1.0,
) -> SparseCGSpec:
    """Build a static sparse COO Clebsch-Gordan buffer for ``ir1 x ir2 -> ir3`` (R1/R4).

    Works for both coupled ``Irrep`` (``O(3) x Z2^T``) and decoupled ``DIrrep``
    (``O(3)_lat x O(3)_spin``). Raises ``ValueError`` if ``(p, t)`` or triangle
    selection rules are violated.
    """
    decoupled = (
        _is_decoupled_spec(ir1) or _is_decoupled_spec(ir2) or _is_decoupled_spec(ir3)
    )
    if decoupled:
        r1, r2, r3 = DIrrep(ir1), DIrrep(ir2), DIrrep(ir3)
    else:
        r1, r2, r3 = Irrep(ir1), Irrep(ir2), Irrep(ir3)

    with _ordinary_tensors():
        body, bkey = _cg_body_f64(r1, r2, r3)
        if scale != 1.0:
            body = body * float(scale)
        mask = torch.abs(body) > tol
        coords = torch.nonzero(mask, as_tuple=False)
        src1 = coords[:, 0].to(dtype=torch.long).contiguous()
        src2 = coords[:, 1].to(dtype=torch.long).contiguous()
        dst = coords[:, 2].to(dtype=torch.long).contiguous()
        vals = body[src1, src2, dst].to(dtype=torch.float64).contiguous()

    spec_hash = symmetry_spec_hash(
        r1, r2, r3, metadata={"body_key": bkey, "scale": float(scale)}
    )
    return SparseCGSpec(
        src1_idx=src1,
        src2_idx=src2,
        dst_idx=dst,
        values=vals,
        shape=(r1.dim, r2.dim, r3.dim),
        body_key=bkey,
        canonical_hash=spec_hash,
    )


def validate_contraction_paths(
    irreps_in1: Any,
    irreps_in2: Any,
    irreps_out: Any,
    instructions: Sequence[Tuple],
    *,
    irrep_normalization: str = "component",
    path_normalization: str = "element",
) -> Tuple[IrrepsLike, IrrepsLike, IrrepsLike, Tuple[ContractionPath, ...], str]:
    """Validate every explicit contraction instruction under full ``(p, t)`` rules (R1/R4)."""
    if irrep_normalization not in ("component", "norm", "none"):
        raise ValueError(
            f"irrep_normalization must be 'component', 'norm', or 'none', got {irrep_normalization!r}"
        )
    if path_normalization not in ("element", "path", "none"):
        raise ValueError(
            f"path_normalization must be 'element', 'path', or 'none', got {path_normalization!r}"
        )

    decoupled = (
        _is_decoupled_spec(irreps_in1)
        or _is_decoupled_spec(irreps_in2)
        or _is_decoupled_spec(irreps_out)
    )
    in1 = _coerce_irreps(irreps_in1, decoupled)
    in2 = _coerce_irreps(irreps_in2, decoupled)
    out = _coerce_irreps(irreps_out, decoupled)

    raw = []
    for ins in instructions:
        tup = tuple(ins)
        if len(tup) not in (5, 6):
            raise ValueError(
                f"Instruction {tup} must be (i_in1, i_in2, i_out, mode, has_weight[, path_weight])"
            )
        i1 = _index_int(tup[0], "i_in1")
        i2 = _index_int(tup[1], "i_in2")
        io = _index_int(tup[2], "i_out")
        mode = str(tup[3])
        has_w = bool(tup[4])
        pw = float(tup[5]) if len(tup) == 6 else 1.0
        raw.append((i1, i2, io, mode, has_w, pw))

    for i1, i2, io, mode, has_w, _ in raw:
        if not (0 <= i1 < len(in1)):
            raise IndexError(f"i_in1={i1} out of range for {in1}")
        if not (0 <= i2 < len(in2)):
            raise IndexError(f"i_in2={i2} out of range for {in2}")
        if not (0 <= io < len(out)):
            raise IndexError(f"i_out={io} out of range for {out}")
        if mode not in _PATH_SHAPE:
            raise ValueError(f"Unsupported connection mode {mode!r}")

        (m1, ir1), (m2, ir2), (m3, ir3) = in1[i1], in2[i2], out[io]
        if ir3 not in tuple(ir1 * ir2):
            raise ValueError(
                f"Forbidden contraction path ({i1}, {i2}, {io}): {ir1} x {ir2} -> {ir3} "
                f"violates symmetry selection rules (R1)"
            )
        if mode == "uvw" and not has_w:
            raise ValueError(
                f"Path ({i1}, {i2}, {io}) mode 'uvw' requires has_weight=True"
            )
        compat = {
            "uvw": True,
            "uvu": m1 == m3,
            "uvv": m2 == m3,
            "uuw": m1 == m2 and (has_w or m3 == 1),
            "uuu": m1 == m2 == m3,
        }[mode]
        if not compat:
            raise ValueError(
                f"Path ({i1}, {i2}, {io}) multiplicities ({m1}, {m2}, {m3}) incompatible with mode {mode!r}"
            )

    def _num_el(entry):
        return _NUM_ELEMENTS[entry[3]](in1[entry[0]].mul, in2[entry[1]].mul)

    paths: List[ContractionPath] = []
    for entry in raw:
        i1, i2, io, mode, has_w, pw = entry
        ir1, ir2, ir3 = in1[i1].ir, in2[i2].ir, out[io].ir
        if irrep_normalization == "component":
            alpha = float(ir3.dim)
        elif irrep_normalization == "norm":
            alpha = float(ir1.dim * ir2.dim)
        else:
            alpha = 1.0

        if path_normalization == "element":
            denom = sum(_num_el(e) for e in raw if e[2] == io)
        elif path_normalization == "path":
            denom = _num_el(entry) * sum(1 for e in raw if e[2] == io)
        else:
            denom = 1
        if denom > 0:
            alpha /= float(denom)
        alpha *= pw
        pshape = _PATH_SHAPE[mode](in1[i1].mul, in2[i2].mul, out[io].mul)
        bkey = (
            decoupled_body_key(ir1, ir2, ir3)
            if decoupled
            else o3_body_key(ir1, ir2, ir3)
        )
        paths.append(
            ContractionPath(
                i_in1=i1,
                i_in2=i2,
                i_out=io,
                connection_mode=mode,
                has_weight=has_w,
                path_weight=math.sqrt(alpha),
                path_shape=pshape,
                body_key=bkey,
            )
        )

    spec_hash = symmetry_spec_hash(
        in1,
        in2,
        out,
        tuple(
            (p.i_in1, p.i_in2, p.i_out, p.connection_mode, p.has_weight, p.path_weight)
            for p in paths
        ),
        metadata={
            "irrep_normalization": irrep_normalization,
            "path_normalization": path_normalization,
        },
    )
    return in1, in2, out, tuple(paths), spec_hash


class SparseCGContract(nn.Module):
    """Compile-friendly static sparse Clebsch-Gordan contraction for a single ``(ir1, ir2 -> ir3)`` block.

    - For ``nnz <= 128``: uses a sign-factored, register-resident unrolled schedule
      with **zero** ``index_add``, **zero** atomics, **zero** ``(B, nnz, C)``
      intermediate allocation, and immediate FP literals under ``torch.compile``.
    - For large decoupled bodies (``nnz > 128``): uses exact 2-step lattice-then-spin
      factorized contraction to avoid Triton/Inductor register spills.
    """

    def __init__(
        self,
        ir1: Any,
        ir2: Any,
        ir3: Any,
        *,
        scale: float = 1.0,
        tol: float = 1e-12,
    ):
        super().__init__()
        spec = build_sparse_cg(ir1, ir2, ir3, tol=tol, scale=scale)
        self.shape_cg = spec.shape
        self.d1, self.d2, self.d3 = spec.shape
        self.nnz = int(spec.values.numel())
        self.body_key = spec.body_key
        self.canonical_hash = spec.canonical_hash
        self.scale = float(scale)

        self._use_factorized = (
            self.nnz > _UNROLL_NNZ_THRESHOLD and self.body_key[0] == "o3xo3"
        )
        self._use_unrolled = (
            not self._use_factorized and self.nnz <= _UNROLL_NNZ_THRESHOLD
        )

        if self._use_unrolled:
            self._schedule = _build_sign_factored_schedule(spec)
            self._factor_meta = None
            self.register_buffer("_c_lat", None, persistent=False)
            self.register_buffer("_c_spin", None, persistent=False)
            self.register_buffer("_src1", None, persistent=False)
            self.register_buffer("_src2", None, persistent=False)
            self.register_buffer("_dst", None, persistent=False)
            self.register_buffer("_val", None, persistent=False)
        elif self._use_factorized:
            self._schedule = None
            _, (l1, l2, l3), (lam1, lam2, lam3) = self.body_key
            self._factor_meta = (
                ("o3", l1, l2, l3),
                ("o3", lam1, lam2, lam3),
                2 * l1 + 1,
                2 * lam1 + 1,
                2 * l2 + 1,
                2 * lam2 + 1,
                2 * l3 + 1,
                2 * lam3 + 1,
            )
            default_dtype = torch.get_default_dtype()
            c_lat = _cached_cg_body_f64(("o3", l1, l2, l3))
            c_spin = _cached_cg_body_f64(("o3", lam1, lam2, lam3))
            self.register_buffer(
                "_c_lat", c_lat.to(dtype=default_dtype, copy=True), persistent=False
            )
            self.register_buffer(
                "_c_spin", c_spin.to(dtype=default_dtype, copy=True), persistent=False
            )
            self.register_buffer("_src1", None, persistent=False)
            self.register_buffer("_src2", None, persistent=False)
            self.register_buffer("_dst", None, persistent=False)
            self.register_buffer("_val", None, persistent=False)
        else:
            # High-l O(3) fallback (nnz > 128)
            self._schedule = None
            self._factor_meta = None
            self._val_f64 = spec.values.clone()
            self.register_buffer("_c_lat", None, persistent=False)
            self.register_buffer("_c_spin", None, persistent=False)
            self.register_buffer("_src1", spec.src1_idx, persistent=False)
            self.register_buffer("_src2", spec.src2_idx, persistent=False)
            self.register_buffer("_dst", spec.dst_idx, persistent=False)
            self.register_buffer(
                "_val",
                spec.values.to(dtype=torch.get_default_dtype(), copy=True),
                persistent=False,
            )

    def _apply(self, fn):
        ret = super()._apply(fn)
        if self._use_factorized and self._factor_meta is not None:
            lat_key, spin_key = self._factor_meta[0], self._factor_meta[1]
            if self._c_lat is not None and self._c_lat.dtype == torch.float64:
                self._c_lat.data.copy_(
                    _cached_cg_body_f64(lat_key).to(
                        device=self._c_lat.device, dtype=torch.float64
                    )
                )
            if self._c_spin is not None and self._c_spin.dtype == torch.float64:
                self._c_spin.data.copy_(
                    _cached_cg_body_f64(spin_key).to(
                        device=self._c_spin.device, dtype=torch.float64
                    )
                )
        elif (
            not self._use_unrolled
            and self._val is not None
            and self._val.dtype == torch.float64
        ):
            self._val.data.copy_(
                self._val_f64.to(device=self._val.device, dtype=torch.float64)
            )
        return ret

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Channel-wise sparse CG contraction: ``(..., d1, C) x (..., d2, C) -> (..., d3, C)``."""
        if x1.shape[-2] != self.d1 or x2.shape[-2] != self.d2:
            raise ValueError(
                f"Expected shapes (..., {self.d1}, C) and (..., {self.d2}, C), "
                f"got {tuple(x1.shape)} and {tuple(x2.shape)}"
            )
        lead = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
        channels = x1.shape[-1]
        x1_b = x1.expand(lead + (self.d1, channels))
        x2_b = x2.expand(lead + (self.d2, channels))

        if self._use_unrolled:
            assert self._schedule is not None
            return _eval_sign_factored_schedule(self._schedule, x1_b, x2_b)

        if self._use_factorized:
            assert self._factor_meta is not None
            (
                lat_key,
                spin_key,
                dl1,
                ds1,
                dl2,
                ds2,
                dl3,
                ds3,
            ) = self._factor_meta
            out_dtype = torch.promote_types(x1.dtype, x2.dtype)
            c_lat = self._c_lat
            c_spin = self._c_spin
            if c_lat.dtype != out_dtype:
                c_lat = (
                    _cached_cg_body_f64(lat_key).to(
                        device=x1.device, dtype=torch.float64
                    )
                    if out_dtype == torch.float64
                    else c_lat.to(dtype=out_dtype)
                )
            if c_spin.dtype != out_dtype:
                c_spin = (
                    _cached_cg_body_f64(spin_key).to(
                        device=x1.device, dtype=torch.float64
                    )
                    if out_dtype == torch.float64
                    else c_spin.to(dtype=out_dtype)
                )
            batch_size = 1
            for s in lead:
                batch_size = batch_size * s
            x1_r = x1_b.reshape(batch_size, dl1, ds1, channels)
            x2_r = x2_b.reshape(batch_size, dl2, ds2, channels)
            if x1_r.dtype != out_dtype:
                x1_r = x1_r.to(dtype=out_dtype)
            if x2_r.dtype != out_dtype:
                x2_r = x2_r.to(dtype=out_dtype)
            # Step 1: contract lattice (dl1, dl2 -> dl3), Step 2: contract spin (ds1, ds2 -> ds3)
            tmp = torch.einsum("ijk,bipc,bjqc->bkpqc", c_lat, x1_r, x2_r)
            out = torch.einsum("pqr,bkpqc->bkrc", c_spin, tmp)
            if self.scale != 1.0:
                out = out * self.scale
            return out.reshape(lead + (self.d3, channels))

        batch_size = 1
        for s in lead:
            batch_size = batch_size * s
        out_dtype = torch.promote_types(x1.dtype, x2.dtype)
        x1_f = x1_b.reshape(batch_size, self.d1, channels)
        x2_f = x2_b.reshape(batch_size, self.d2, channels)
        val = self._val
        if val.dtype != out_dtype:
            val = (
                self._val_f64.to(device=x1.device, dtype=torch.float64)
                if out_dtype == torch.float64
                else val.to(dtype=out_dtype)
            )
        prod_terms = (
            x1_f[:, self._src1, :] * x2_f[:, self._src2, :] * val[None, :, None]
        )
        out = x1.new_zeros((batch_size, self.d3, channels), dtype=out_dtype)
        out = out.index_add(1, self._dst, prod_terms)
        return out.reshape(lead + (self.d3, channels))


class SparseBispectrumContract(nn.Module):
    r"""Register-fused sparse trilinear Wigner-3j bispectrum contraction (R1).

    Evaluates:

    - ``mode="channelwise"`` (e.g. ``(2, 2, 2)``):
      :math:`\sum_{i,j,k} C^{l_1 l_2 l_3}_{ijk}\, x^{(1)}_{\dots, i, c}\, x^{(2)}_{\dots, j, c}\, x^{(3)}_{\dots, k, c}`
      returning shape ``(..., C)``.
    - ``mode="pair"`` (e.g. ``(1, 1, 2)``):
      :math:`\sum_{i,j,k} C^{l_1 l_2 l_3}_{ijk}\, x^{(1)}_{\dots, i, a}\, x^{(2)}_{\dots, j, a}\, x^{(3)}_{\dots, k, b}`
      returning shape ``(..., K_1, K_2)``.

    Eliminates 75%–80% zero multiply-adds compared to dense ``einsum("ijk,...")``
    with zero CG buffer memory loads and zero graph breaks under ``torch.compile``.
    """

    def __init__(
        self,
        l1: int,
        l2: int,
        l3: int,
        *,
        mode: str = "channelwise",
        scale: float = 1.0,
        tol: float = 1e-12,
    ):
        super().__init__()
        if mode not in ("channelwise", "pair"):
            raise ValueError(f"mode must be 'channelwise' or 'pair', got {mode!r}")
        for name, deg in (("l1", l1), ("l2", l2), ("l3", l3)):
            if not (isinstance(deg, int) and not isinstance(deg, bool) and deg >= 0):
                raise ValueError(f"{name} must be non-negative int, got {deg!r}")
        if (l1 + l2 + l3) % 2 != 0:
            raise ValueError(
                f"Parity rule violated for natural-parity bispectrum ({l1}, {l2}, {l3}): "
                "l1 + l2 + l3 must be even"
            )
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.mode = mode
        ir1 = Irrep(l1, (-1) ** l1, 1)
        ir2 = Irrep(l2, (-1) ** l2, 1)
        ir3 = Irrep(l3, (-1) ** l3, 1)
        spec = build_sparse_cg(ir1, ir2, ir3, tol=tol, scale=scale)
        self.d1, self.d2, self.d3 = spec.shape
        self.nnz = int(spec.values.numel())
        self.canonical_hash = spec.canonical_hash
        self._schedule = _build_sign_factored_schedule(spec)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> torch.Tensor:
        if (
            x1.shape[-2] != self.d1
            or x2.shape[-2] != self.d2
            or x3.shape[-2] != self.d3
        ):
            raise ValueError(
                f"Expected shapes (..., {self.d1}, C1), (..., {self.d2}, C1), (..., {self.d3}, C2), "
                f"got {tuple(x1.shape)}, {tuple(x2.shape)}, {tuple(x3.shape)}"
            )
        # Register-fused (d1, d2) -> (..., d3, C1)
        t = _eval_sign_factored_schedule(self._schedule, x1, x2)
        if self.mode == "channelwise":
            return (t * x3).sum(dim=-2)
        # mode == "pair": (..., d3, K1)^T @ (..., d3, K2) -> (..., K1, K2)
        if t.dtype != x3.dtype:
            out_dtype = torch.promote_types(t.dtype, x3.dtype)
            t = t.to(dtype=out_dtype)
            x3 = x3.to(dtype=out_dtype)
        return torch.matmul(t.transpose(-2, -1), x3)


class StaticTensorProduct(nn.Module):
    """Compile-friendly static tensor product for BOTH ``Irreps`` and ``DIrreps`` (R1/R4).

    All coupling bodies are pre-built as static non-persistent buffers in
    ``torch.get_default_dtype()`` with exact 53-bit FP64 re-materialization in
    ``_apply()``. If ``instructions is None``, creates a fully-connected ``'uvw'``
    tensor product strictly filtered by full ``(p, t)`` and triangle rules.
    """

    def __init__(
        self,
        irreps_in1: Any,
        irreps_in2: Any,
        irreps_out: Any,
        instructions: Optional[Sequence[Tuple]] = None,
        *,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        shared_weights: Optional[bool] = None,
        internal_weights: Optional[bool] = None,
    ):
        super().__init__()
        decoupled = (
            _is_decoupled_spec(irreps_in1)
            or _is_decoupled_spec(irreps_in2)
            or _is_decoupled_spec(irreps_out)
        )
        in1_tmp = _coerce_irreps(irreps_in1, decoupled)
        in2_tmp = _coerce_irreps(irreps_in2, decoupled)
        out_tmp = _coerce_irreps(irreps_out, decoupled)

        if instructions is None:
            instructions = [
                (i1, i2, io, "uvw", True, 1.0)
                for i1, (_, ir1) in enumerate(in1_tmp)
                for i2, (_, ir2) in enumerate(in2_tmp)
                for io, (_, ir3) in enumerate(out_tmp)
                if ir3 in tuple(ir1 * ir2)
            ]

        (
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            self.instructions,
            self.canonical_hash,
        ) = validate_contraction_paths(
            in1_tmp,
            in2_tmp,
            out_tmp,
            instructions,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
        )
        self.decoupled = decoupled

        if shared_weights is False and internal_weights is None:
            internal_weights = False
        if shared_weights is None:
            shared_weights = True
        if internal_weights is None:
            internal_weights = shared_weights and any(
                ins.has_weight for ins in self.instructions
            )
        if internal_weights and not shared_weights:
            raise ValueError("internal_weights=True requires shared_weights=True")
        self.shared_weights = bool(shared_weights)
        self.internal_weights = bool(internal_weights)

        self.weight_numel = sum(
            _prod(ins.path_shape) for ins in self.instructions if ins.has_weight
        )
        if self.internal_weights and self.weight_numel > 0:
            self.weight = nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.empty(0), persistent=False)

        # Register static CG buffers in default dtype; re-materialize exact FP64 in _apply()
        key_to_buf: Dict[Tuple, str] = {}
        buf_spec = []
        z = "" if self.shared_weights else "z"
        plan = []
        w_offset = 0
        default_dtype = torch.get_default_dtype()
        for ins in self.instructions:
            if ins.body_key not in key_to_buf:
                body_f64 = _cached_cg_body_f64(ins.body_key)
                buf_name = f"_cg_{len(key_to_buf)}"
                self.register_buffer(
                    buf_name,
                    body_f64.to(dtype=default_dtype, copy=True),
                    persistent=False,
                )
                key_to_buf[ins.body_key] = buf_name
                buf_spec.append((buf_name, ins.body_key))
            buf_name = key_to_buf[ins.body_key]
            n_w = _prod(ins.path_shape) if ins.has_weight else 0
            eq = (
                _EINSUM_WEIGHTED[ins.connection_mode].format(z=z)
                if ins.has_weight
                else _EINSUM_UNWEIGHTED[ins.connection_mode]
            )
            plan.append(
                (
                    ins.i_in1,
                    ins.i_in2,
                    ins.i_out,
                    buf_name,
                    ins.body_key,
                    eq,
                    ins.has_weight,
                    w_offset,
                    n_w,
                    ins.path_shape,
                    ins.path_weight,
                )
            )
            w_offset += n_w

        self._plan = tuple(plan)
        self._buf_spec = tuple(buf_spec)
        self._in1_dim = self.irreps_in1.dim
        self._in2_dim = self.irreps_in2.dim
        self._out_dim = self.irreps_out.dim
        self._in1_blocks = tuple(
            (sl.start, mul, ir.dim)
            for sl, (mul, ir) in zip(self.irreps_in1.slices(), self.irreps_in1)
        )
        self._in2_blocks = tuple(
            (sl.start, mul, ir.dim)
            for sl, (mul, ir) in zip(self.irreps_in2.slices(), self.irreps_in2)
        )
        self._out_block_dims = tuple(mul_ir.dim for mul_ir in self.irreps_out)

    def _apply(self, fn):
        ret = super()._apply(fn)
        for buf_name, bkey in self._buf_spec:
            buf = getattr(self, buf_name)
            if buf.dtype == torch.float64:
                exact = _cached_cg_body_f64(bkey)
                buf.data.copy_(exact.to(device=buf.device, dtype=torch.float64))
        return ret

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x1.shape[-1] != self._in1_dim:
            raise ValueError(
                f"Expected x1 trailing dim {self._in1_dim}, got {x1.shape[-1]}"
            )
        if x2.shape[-1] != self._in2_dim:
            raise ValueError(
                f"Expected x2 trailing dim {self._in2_dim}, got {x2.shape[-1]}"
            )
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "weight tensor must be provided when internal_weights=False"
                )
            weight = self.weight

        shapes = [x1.shape[:-1], x2.shape[:-1]]
        if not self.shared_weights and self.weight_numel > 0:
            shapes.append(weight.shape[:-1])
        batch = torch.broadcast_shapes(*shapes)
        batch_size = 1
        for s in batch:
            batch_size = batch_size * s
        out_dtype = torch.promote_types(x1.dtype, x2.dtype)
        if self.weight_numel > 0:
            out_dtype = torch.promote_types(out_dtype, weight.dtype)
        x1_f = x1.expand(batch + (self._in1_dim,)).reshape(batch_size, self._in1_dim)
        x2_f = x2.expand(batch + (self._in2_dim,)).reshape(batch_size, self._in2_dim)
        if x1_f.dtype != out_dtype:
            x1_f = x1_f.to(dtype=out_dtype)
        if x2_f.dtype != out_dtype:
            x2_f = x2_f.to(dtype=out_dtype)
        if not self.shared_weights and self.weight_numel > 0:
            weight = weight.expand(batch + (self.weight_numel,)).reshape(
                batch_size, self.weight_numel
            )
        if self.weight_numel > 0 and weight.dtype != out_dtype:
            weight = weight.to(dtype=out_dtype)

        b1 = [
            x1_f[:, start : start + mul * dim].reshape(batch_size, mul, dim)
            for start, mul, dim in self._in1_blocks
        ]
        b2 = [
            x2_f[:, start : start + mul * dim].reshape(batch_size, mul, dim)
            for start, mul, dim in self._in2_blocks
        ]

        outs: List[Optional[torch.Tensor]] = [None] * len(self._out_block_dims)
        for (
            i1,
            i2,
            io,
            buf_name,
            bkey,
            eq,
            has_w,
            w_off,
            n_w,
            pshape,
            pw,
        ) in self._plan:
            C = getattr(self, buf_name)
            if C.dtype != out_dtype:
                C = (
                    _cached_cg_body_f64(bkey).to(device=x1.device, dtype=torch.float64)
                    if out_dtype == torch.float64
                    else C.to(dtype=out_dtype)
                )
            if has_w:
                if self.shared_weights:
                    w = weight[w_off : w_off + n_w].reshape(pshape)
                else:
                    w = weight[:, w_off : w_off + n_w].reshape((batch_size,) + pshape)
                res = torch.einsum(eq, w, b1[i1], b2[i2], C)
            else:
                res = torch.einsum(eq, b1[i1], b2[i2], C)
            res = pw * res.reshape(batch_size, self._out_block_dims[io])
            outs[io] = res if outs[io] is None else outs[io] + res

        blocks = [
            x1.new_zeros((batch_size, dim), dtype=out_dtype) if o is None else o
            for o, dim in zip(outs, self._out_block_dims)
        ]
        if not blocks:
            return x1.new_zeros(batch + (0,), dtype=out_dtype)
        out = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)
        return out.reshape(batch + (self._out_dim,))
