"""Equivariant linear, RMS normalization, and gated nonlinearity layers enforcing Rules R2 & R3.

- Rule R2: ``EquivariantLinear`` and ``EquivariantRMSNorm`` operate strictly on full
  ``(l, p, t)`` or ``(l, p | lam, t)`` labels, NEVER mixing channels of different ``p`` or ``t``.
  ``EquivariantLinear`` stores per-irrep 2D ``(mul_out, mul_in)`` matrices in
  ``weight_blocks`` by default so ``Muon`` / ``PhysMuon`` orthogonalization and
  per-channel ``iso`` fluctuation metrics act independently on each irrep's
  multiplicity space while preserving exact equivariance.
- Rule R3: ``DecoupledGate`` / ``EquivariantGate`` enforce that only full-invariant
  ``0e|0e`` / ``0ee`` scalars gate non-scalar blocks (by default ``strict_invariant_gates=True``),
  and automatically probe activation functions acting on odd-parity (``p=-1``) or
  T-odd (``t=-1``) scalars to guarantee exact odd or even parity.
"""

import operator
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .irreps import DIrrep, DIrreps, Irrep, Irreps, symmetry_spec_hash

__all__ = [
    "probe_activation_parity",
    "validate_scalar_activation",
    "validate_invariant_scalar",
    "EquivariantLinear",
    "EquivariantRMSNorm",
    "DecoupledGate",
    "EquivariantGate",
]


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


def _coerce_irreps(spec: Any, decoupled: bool):
    return DIrreps(spec) if decoupled else Irreps(spec)


def _index_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer index, got bool {value!r}")
    return operator.index(value)


def _act_name(act: Any) -> str:
    return getattr(act, "__name__", None) or repr(act)


def probe_activation_parity(act: Callable, tol: float = 1e-5) -> int:
    """Probe whether scalar activation ``act`` is even (``+1``), odd (``-1``), or neither (``0``)."""
    x = torch.linspace(0.0, 10.0, 256, dtype=torch.float64)
    if isinstance(act, nn.Module):
        params = list(act.parameters())
        if params:
            x = x.to(dtype=params[0].dtype, device=params[0].device)
    with torch.no_grad():
        y_pos = act(x)
        y_neg = act(-x)
    if torch.max(torch.abs(y_pos - y_neg)).item() < tol:
        return 1
    if torch.max(torch.abs(y_pos + y_neg)).item() < tol:
        return -1
    return 0


def validate_scalar_activation(
    ir: Union[Irrep, DIrrep], act: Optional[Callable], role: str = "scalar"
) -> Union[Irrep, DIrrep]:
    """Enforce Rule R3 parity/time-reversal constraints on a scalar activation.

    - Non-scalar blocks (``l > 0`` or ``lam > 0``) cannot receive pointwise activations.
    - If ``ir.p == -1`` or ``ir.t == -1``, ``act`` MUST be strictly even or odd
      (probed via :func:`probe_activation_parity`), otherwise ``ValueError`` is raised.
    - Returns the output irrep with updated ``(p_out, t_out)``.
    """
    is_dec = isinstance(ir, DIrrep)
    if act is None:
        return ir
    if ir.l != 0 or (is_dec and ir.lam != 0):
        raise ValueError(
            f"Rule R3 violation: cannot apply scalar activation to non-scalar {role} block {ir}"
        )
    if ir.p == 1 and ir.t == 1:
        return ir

    p_act = probe_activation_parity(act)
    p_out = p_act if ir.p == -1 else ir.p
    t_out = p_act if ir.t == -1 else ir.t
    if p_out == 0 or t_out == 0:
        reasons = []
        if p_out == 0:
            reasons.append("spatial parity (p = -1)")
        if t_out == 0:
            reasons.append("time-reversal parity (t = -1)")
        raise ValueError(
            f"Rule R3 violation: {role} block {ir} is odd under {' and '.join(reasons)}, "
            f"so activation `{_act_name(act)}` must be strictly even or odd."
        )
    if is_dec:
        return DIrrep(0, p_out, 0, t_out)
    return Irrep(0, p_out, t_out)


def validate_invariant_scalar(irreps: Any, role: str = "gate") -> None:
    """Enforce Rule R3: every block in ``irreps`` must be a full invariant ``0e|0e`` or ``0ee``."""
    decoupled = _is_decoupled_spec(irreps)
    parsed = _coerce_irreps(irreps, decoupled)
    expected = "0e|0e" if decoupled else "0ee"
    for mul, ir in parsed:
        if mul > 0 and not ir.is_scalar():
            raise ValueError(
                f"Rule R3 violation: {role} must consist solely of full-invariant "
                f"{expected} scalars, got {mul}x{ir}"
            )


class EquivariantLinear(nn.Module):
    """Equivariant linear map for BOTH ``Irreps`` and ``DIrreps`` enforcing Rule R2.

    Mixes channels ONLY between identical full irreps ``(l, p, t)`` or
    ``(l, p | lam, t)``. Biases are permitted ONLY on full-invariant ``0ee`` /
    ``0e|0e`` blocks.

    When ``internal_weights=True`` and ``structured_weights=True`` (default),
    each path stores a true 2D ``(mul_out, mul_in)`` weight matrix in
    ``self.weight_blocks`` (matching ``nn.Linear`` layout for ``PhysMuon`` /
    ``Muon`` spectral orthogonalization and per-channel ``iso`` calibration).
    External flat ``weight`` tensors remain fully supported in ``forward()``.
    """

    def __init__(
        self,
        irreps_in: Any,
        irreps_out: Any,
        biases: Union[bool, Sequence[bool]] = False,
        *,
        instructions: Optional[Sequence[Tuple[int, int]]] = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        path_normalization: str = "element",
        structured_weights: bool = True,
    ):
        super().__init__()
        if path_normalization not in ("element", "path"):
            raise ValueError(
                f"path_normalization must be 'element' or 'path', got {path_normalization!r}"
            )
        decoupled = _is_decoupled_spec(irreps_in) or _is_decoupled_spec(irreps_out)
        self.decoupled = decoupled
        self.irreps_in = _coerce_irreps(irreps_in, decoupled)
        self.irreps_out = _coerce_irreps(irreps_out, decoupled)

        if instructions is None:
            pairs = [
                (i_in, i_out)
                for i_in, (_, ir_in) in enumerate(self.irreps_in)
                for i_out, (_, ir_out) in enumerate(self.irreps_out)
                if ir_in == ir_out
            ]
        else:
            pairs = []
            for ins in instructions:
                tup = tuple(ins)
                if len(tup) != 2:
                    raise ValueError(
                        f"Linear instruction must be (i_in, i_out), got {tup}"
                    )
                i_in = _index_int(tup[0], "i_in")
                i_out = _index_int(tup[1], "i_out")
                if not (0 <= i_in < len(self.irreps_in)):
                    raise IndexError(f"i_in={i_in} out of range for {self.irreps_in}")
                if not (0 <= i_out < len(self.irreps_out)):
                    raise IndexError(
                        f"i_out={i_out} out of range for {self.irreps_out}"
                    )
                if self.irreps_in[i_in].ir != self.irreps_out[i_out].ir:
                    raise ValueError(
                        f"Rule R2 violation: EquivariantLinear cannot mix "
                        f"{self.irreps_in[i_in].ir} with {self.irreps_out[i_out].ir}"
                    )
                pairs.append((i_in, i_out))

        def _norm_factor(i_in: int, i_out: int) -> float:
            denom = sum(
                self.irreps_in[j_in if path_normalization == "element" else i_in].mul
                for j_in, j_out in pairs
                if j_out == i_out
            )
            return 1.0 if denom == 0 else denom**-0.5

        if isinstance(biases, bool):
            bias_flags = [biases and ir.is_scalar() for _, ir in self.irreps_out]
        else:
            bias_flags = [bool(b) for b in biases]
            if len(bias_flags) != len(self.irreps_out):
                raise ValueError(
                    f"biases must have length {len(self.irreps_out)}, got {len(bias_flags)}"
                )
            for flag, (_, ir) in zip(bias_flags, self.irreps_out):
                if flag and not ir.is_scalar():
                    raise ValueError(
                        f"Rule R2/R3 violation: only full-invariant scalars can carry a bias, not {ir}"
                    )

        if shared_weights is False and internal_weights is None:
            internal_weights = False
        if shared_weights is None:
            shared_weights = True
        if internal_weights is None:
            internal_weights = True
        if internal_weights and not shared_weights:
            raise ValueError("internal_weights=True requires shared_weights=True")
        self.shared_weights = bool(shared_weights)
        self.internal_weights = bool(internal_weights)
        self.structured_weights = bool(structured_weights and self.internal_weights)

        plan = []
        w_offset = 0
        b_offset = 0
        weight_blocks: List[nn.Parameter] = []
        for path_idx, (i_in, i_out) in enumerate(pairs):
            m_in = self.irreps_in[i_in].mul
            m_out = self.irreps_out[i_out].mul
            n_w = m_in * m_out
            if self.structured_weights:
                weight_blocks.append(nn.Parameter(torch.randn(m_out, m_in)))
            plan.append(
                (
                    i_in,
                    i_out,
                    path_idx,
                    w_offset,
                    n_w,
                    (m_in, m_out),
                    _norm_factor(i_in, i_out),
                )
            )
            w_offset += n_w
        for i_out, (flag, mul_ir) in enumerate(zip(bias_flags, self.irreps_out)):
            if flag:
                n_b = mul_ir.dim
                plan.append((-1, i_out, -1, b_offset, n_b, (n_b,), 1.0))
                b_offset += n_b

        self.weight_numel = w_offset
        self.bias_numel = b_offset
        if self.structured_weights:
            self.weight_blocks = nn.ParameterList(weight_blocks)
            self.register_buffer("weight", torch.empty(0), persistent=False)
        elif self.internal_weights and self.weight_numel > 0:
            self.weight_blocks = nn.ParameterList()
            self.weight = nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.weight_blocks = nn.ParameterList()
            self.register_buffer("weight", torch.empty(0), persistent=False)

        if self.internal_weights and self.bias_numel > 0:
            self.bias = nn.Parameter(torch.zeros(self.bias_numel))
        else:
            self.register_buffer("bias", torch.empty(0), persistent=False)

        self._plan = tuple(plan)
        self._in_dim = self.irreps_in.dim
        self._out_dim = self.irreps_out.dim
        self._in_blocks = tuple(
            (sl.start, mul, ir.dim)
            for sl, (mul, ir) in zip(self.irreps_in.slices(), self.irreps_in)
        )
        self._out_block_dims = tuple(mul_ir.dim for mul_ir in self.irreps_out)
        self.canonical_hash = symmetry_spec_hash(
            self.irreps_in,
            self.irreps_out,
            tuple(pairs),
            tuple(bias_flags),
            metadata={"path_normalization": path_normalization},
        )

    def flat_weight(self) -> torch.Tensor:
        """Return a 1D flat view/concatenation of internal weights in ``(m_in, m_out)`` order."""
        if self.structured_weights:
            if not self.weight_blocks:
                return self.weight
            return torch.cat(
                [wb.transpose(0, 1).reshape(-1) for wb in self.weight_blocks], dim=0
            )
        return self.weight

    def zero_init_(self) -> "EquivariantLinear":
        """Zero-initialize all internal weights and biases (useful for zero-init SOC heads)."""
        with torch.no_grad():
            if self.structured_weights:
                for wb in self.weight_blocks:
                    wb.zero_()
            elif self.weight_numel > 0 and isinstance(self.weight, nn.Parameter):
                self.weight.zero_()
            if self.bias_numel > 0 and isinstance(self.bias, nn.Parameter):
                self.bias.zero_()
        return self

    def forward(
        self,
        features: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.shape[-1] != self._in_dim:
            raise ValueError(
                f"Expected trailing dimension {self._in_dim}, got {features.shape[-1]}"
            )
        use_structured = weight is None and self.structured_weights
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "weight must be provided when internal_weights=False"
                )
            weight = self.weight
        if bias is None:
            if self.bias_numel > 0 and not self.internal_weights:
                raise RuntimeError("bias must be provided when internal_weights=False")
            bias = self.bias

        shapes = [features.shape[:-1]]
        if not self.shared_weights:
            if self.weight_numel > 0:
                shapes.append(weight.shape[:-1])
            if self.bias_numel > 0:
                shapes.append(bias.shape[:-1])
        batch = torch.broadcast_shapes(*shapes)
        batch_size = 1
        for s in batch:
            batch_size = batch_size * s
        x = features.expand(batch + (self._in_dim,)).reshape(batch_size, self._in_dim)
        if not self.shared_weights and self.weight_numel > 0:
            weight = weight.expand(batch + (self.weight_numel,)).reshape(
                batch_size, self.weight_numel
            )

        in_blocks = [
            x[:, start : start + mul * dim].reshape(batch_size, mul, dim)
            for start, mul, dim in self._in_blocks
        ]
        outs: List[Optional[torch.Tensor]] = [None] * len(self._out_block_dims)
        for i_in, i_out, path_idx, offset, numel, shape, scale in self._plan:
            if i_in == -1:
                b = bias[..., offset : offset + numel]
                res = scale * b.expand(batch + (numel,)).reshape(batch_size, numel)
            elif use_structured:
                # weight_blocks[path_idx] has shape (m_out, m_in);
                # in_blocks[i_in] has shape (Z, m_in, dim) -> (Z, m_out, dim)
                w = self.weight_blocks[path_idx]
                res = torch.matmul(w, in_blocks[i_in])
                res = (scale * res).reshape(batch_size, self._out_block_dims[i_out])
            elif self.shared_weights:
                w = weight[offset : offset + numel].reshape(shape)
                res = torch.matmul(w.transpose(0, 1), in_blocks[i_in])
                res = (scale * res).reshape(batch_size, self._out_block_dims[i_out])
            else:
                w = weight[:, offset : offset + numel].reshape((batch_size,) + shape)
                res = torch.bmm(w.transpose(1, 2), in_blocks[i_in])
                res = (scale * res).reshape(batch_size, self._out_block_dims[i_out])
            outs[i_out] = res if outs[i_out] is None else outs[i_out] + res

        blocks = [
            x.new_zeros((batch_size, dim)) if o is None else o
            for o, dim in zip(outs, self._out_block_dims)
        ]
        if not blocks:
            return x.new_zeros(batch + (0,))
        out = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)
        return out.reshape(batch + (self._out_dim,))


class EquivariantRMSNorm(nn.Module):
    """Equivariant RMS normalization for BOTH ``Irreps`` and ``DIrreps`` enforcing Rule R2.

    Normalizes each full ``(l, p, t)`` or ``(l, p | lam, t)`` block isotropically
    by :math:`\\mathrm{rsqrt}(\\mathrm{mean}_{u, m}(x_{u,m}^2) + \\epsilon)`,
    never mixing distinct ``(p, t)`` blocks and avoiding ``abs``/``maximum``
    non-smooth second-order derivative pitfalls. Optional learnable affine scale
    (per channel, invariant scalar) and optional bias strictly on ``0ee`` / ``0e|0e``.
    """

    def __init__(
        self,
        irreps: Any,
        eps: float = 1e-6,
        affine: bool = True,
        center_scalars: bool = False,
    ):
        super().__init__()
        if eps <= 0.0:
            raise ValueError(f"eps must be strictly positive, got {eps}")
        decoupled = _is_decoupled_spec(irreps)
        self.decoupled = decoupled
        self.irreps = _coerce_irreps(irreps, decoupled)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.center_scalars = bool(center_scalars)

        self._in_dim = self.irreps.dim
        blocks = []
        scales = []
        shifts = []
        for sl, (mul, ir) in zip(self.irreps.slices(), self.irreps):
            is_sc = ir.is_scalar()
            if self.affine and mul > 0:
                scale_idx = len(scales)
                scales.append(nn.Parameter(torch.ones(mul)))
                if self.center_scalars and is_sc:
                    shift_idx = len(shifts)
                    shifts.append(nn.Parameter(torch.zeros(mul)))
                else:
                    shift_idx = -1
            else:
                scale_idx = -1
                shift_idx = -1
            blocks.append((sl.start, mul, ir.dim, is_sc, scale_idx, shift_idx))
        self._blocks = tuple(blocks)
        self.canonical_hash = symmetry_spec_hash(
            self.irreps,
            metadata={
                "eps": self.eps,
                "affine": self.affine,
                "center_scalars": self.center_scalars,
            },
        )
        self.scales = nn.ParameterList(scales)
        self.shifts = nn.ParameterList(shifts)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self._in_dim:
            raise ValueError(
                f"Expected trailing dimension {self._in_dim}, got {features.shape[-1]}"
            )
        batch = features.shape[:-1]
        out_parts = []
        for start, mul, dim, is_sc, scale_idx, shift_idx in self._blocks:
            if mul == 0:
                continue
            blk = features.narrow(-1, start, mul * dim).reshape(batch + (mul, dim))
            if self.center_scalars and is_sc:
                blk = blk - blk.mean(dim=-2, keepdim=True)
            # Isotropic block RMS across both channels and irrep components (exact O(3) invariant)
            mean_sq = (blk * blk).mean(dim=(-2, -1), keepdim=True)
            normed = blk * torch.rsqrt(mean_sq + self.eps)
            if scale_idx >= 0:
                normed = normed * self.scales[scale_idx].unsqueeze(-1)
            if shift_idx >= 0:
                normed = normed + self.shifts[shift_idx].unsqueeze(-1)
            out_parts.append(normed.reshape(batch + (mul * dim,)))
        if not out_parts:
            return features.new_zeros(batch + (0,))
        return out_parts[0] if len(out_parts) == 1 else torch.cat(out_parts, dim=-1)


class _ScalarActivationBlock(nn.Module):
    """Apply validated scalar activations block-wise (R3)."""

    def __init__(
        self, irreps_in, acts: Sequence[Optional[Callable]], role: str = "scalar"
    ):
        super().__init__()
        self.decoupled = _is_decoupled_spec(irreps_in)
        self.irreps_in = _coerce_irreps(irreps_in, self.decoupled)
        acts_list = list(acts)
        if len(acts_list) != len(self.irreps_in):
            raise ValueError(
                f"{len(self.irreps_in)} blocks in {self.irreps_in} but {len(acts_list)} activations"
            )
        out_blocks = []
        for (mul, ir), act in zip(self.irreps_in, acts_list):
            ir_out = validate_scalar_activation(ir, act, role=role)
            out_blocks.append((mul, ir_out))
        self.irreps_out = _coerce_irreps(out_blocks, self.decoupled)
        self._acts = tuple(acts_list)
        self._spans = tuple(
            (sl.start, mul * ir.dim)
            for sl, (mul, ir) in zip(self.irreps_in.slices(), self.irreps_in)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._spans:
            return x
        parts = []
        for (start, width), act in zip(self._spans, self._acts):
            seg = x.narrow(-1, start, width)
            parts.append(act(seg) if act is not None else seg)
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)


class _SortedSplitter(nn.Module):
    """Split a sorted+simplified concatenation of ``(scalars, gates, gated)``."""

    def __init__(self, *parts):
        super().__init__()
        decoupled = any(_is_decoupled_spec(p) for p in parts)
        self.decoupled = decoupled
        self.irreps_outs = tuple(_coerce_irreps(p, decoupled).simplify() for p in parts)
        empty = DIrreps() if decoupled else Irreps()
        combined = sum(self.irreps_outs, empty)

        cursor = 0
        groups = []
        for part in self.irreps_outs:
            groups.append(tuple(range(cursor, cursor + len(part))))
            cursor += len(part)

        sorted_ir, perm, _ = combined.sort()
        slices = sorted_ir.slices()
        self._spans = tuple(
            tuple(
                (slices[perm[k]].start, slices[perm[k]].stop - slices[perm[k]].start)
                for k in grp
            )
            for grp in groups
        )
        self.irreps_in = sorted_ir.simplify()
        self._in_dim = self.irreps_in.dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if x.shape[-1] != self._in_dim:
            raise ValueError(
                f"Expected trailing dimension {self._in_dim}, got {x.shape[-1]}"
            )
        out = []
        for spans in self._spans:
            segs = [x.narrow(-1, start, length) for start, length in spans]
            if not segs:
                out.append(x.new_zeros(x.shape[:-1] + (0,)))
            elif len(segs) == 1:
                out.append(segs[0])
            else:
                out.append(torch.cat(segs, dim=-1))
        return tuple(out)


class _BaseGate(nn.Module):
    """Shared implementation for ``DecoupledGate`` (``DIrreps``) and ``EquivariantGate`` (``Irreps``)."""

    def __init__(
        self,
        irreps_scalars: Any,
        act_scalars: Sequence[Optional[Callable]],
        irreps_gates: Any,
        act_gates: Sequence[Optional[Callable]],
        irreps_gated: Any,
        *,
        decoupled: bool,
        strict_invariant_gates: bool = True,
    ):
        super().__init__()
        self.decoupled = decoupled
        self.strict_invariant_gates = bool(strict_invariant_gates)
        sc = _coerce_irreps(irreps_scalars, decoupled)
        gt = _coerce_irreps(irreps_gates, decoupled)
        gd = _coerce_irreps(irreps_gated, decoupled)

        for name, irs in (("irreps_scalars", sc), ("irreps_gates", gt)):
            for mul, ir in irs:
                if mul > 0 and (ir.l != 0 or (decoupled and ir.lam != 0)):
                    raise ValueError(
                        f"Rule R3 violation: {name} must contain only scalar blocks, got {mul}x{ir}"
                    )

        if self.strict_invariant_gates:
            validate_invariant_scalar(gt, role="irreps_gates")

        if gt.num_irreps != gd.num_irreps:
            raise ValueError(
                f"irreps_gates.num_irreps ({gt.num_irreps}) must match "
                f"irreps_gated.num_irreps ({gd.num_irreps})"
            )

        self.splitter = _SortedSplitter(sc, gt, gd)
        (
            self.irreps_scalars,
            self.irreps_gates,
            self.irreps_gated,
        ) = self.splitter.irreps_outs
        self.irreps_in = self.splitter.irreps_in

        self.act_scalars = _ScalarActivationBlock(sc, act_scalars, role="scalar")
        self.act_gates = _ScalarActivationBlock(gt, act_gates, role="gate")

        gate_labels = [
            (ir.p, ir.t) for mul, ir in self.act_gates.irreps_out for _ in range(mul)
        ]
        gating_plan = []
        gated_out = []
        i_gate = 0
        offset = 0
        for mul, ir in self.irreps_gated:
            u = 0
            while u < mul:
                lbl = gate_labels[i_gate + u]
                v = u
                while v < mul and gate_labels[i_gate + v] == lbl:
                    v += 1
                gating_plan.append((offset + u * ir.dim, v - u, ir.dim, i_gate + u))
                if decoupled:
                    out_ir = DIrrep(ir.l, ir.p * lbl[0], ir.lam, ir.t * lbl[1])
                else:
                    out_ir = Irrep(ir.l, ir.p * lbl[0], ir.t * lbl[1])
                gated_out.append((v - u, out_ir))
                u = v
            offset += mul * ir.dim
            i_gate += mul

        self._gating = tuple(gating_plan)
        self._has_gates = self.irreps_gates.dim > 0
        self.irreps_out = self.act_scalars.irreps_out + _coerce_irreps(
            gated_out, decoupled
        )
        self.canonical_hash = symmetry_spec_hash(
            self.irreps_in,
            self.irreps_out,
            metadata={"strict_invariant_gates": self.strict_invariant_gates},
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scalars, gates, gated = self.splitter(features)
        scalars = self.act_scalars(scalars)
        if not self._has_gates:
            return scalars
        gates = self.act_gates(gates)
        parts = [scalars] if scalars.shape[-1] > 0 else []
        batch = gated.shape[:-1]
        for start, mul, dim, i_gate in self._gating:
            y = gated.narrow(-1, start, mul * dim).reshape(batch + (mul, dim))
            g = gates.narrow(-1, i_gate, mul)
            parts.append((y * g.unsqueeze(-1)).reshape(batch + (mul * dim,)))
        if not parts:
            return features.new_zeros(batch + (0,))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


class DecoupledGate(_BaseGate):
    """Gated nonlinearity for decoupled ``DIrreps`` enforcing Rule R3.

    By default (``strict_invariant_gates=True``), only ``0e|0e`` invariant scalars
    may gate non-scalar blocks. Activations on ``0o|0e``, ``0e|0o``, or ``0o|0o``
    scalars are automatically probed and must be strictly even or odd.
    """

    def __init__(
        self,
        irreps_scalars: Any,
        act_scalars: Sequence[Optional[Callable]],
        irreps_gates: Any,
        act_gates: Sequence[Optional[Callable]],
        irreps_gated: Any,
        *,
        strict_invariant_gates: bool = True,
    ):
        super().__init__(
            irreps_scalars,
            act_scalars,
            irreps_gates,
            act_gates,
            irreps_gated,
            decoupled=True,
            strict_invariant_gates=strict_invariant_gates,
        )


class EquivariantGate(_BaseGate):
    """Gated nonlinearity for coupled ``Irreps`` (``O(3) x Z2^T``) enforcing Rule R3.

    By default (``strict_invariant_gates=True``), only ``0ee`` invariant scalars
    may gate non-scalar blocks. Activations on ``0oe``, ``0eo``, or ``0oo``
    scalars are automatically probed and must be strictly even or odd.
    """

    def __init__(
        self,
        irreps_scalars: Any,
        act_scalars: Sequence[Optional[Callable]],
        irreps_gates: Any,
        act_gates: Sequence[Optional[Callable]],
        irreps_gated: Any,
        *,
        strict_invariant_gates: bool = True,
    ):
        super().__init__(
            irreps_scalars,
            act_scalars,
            irreps_gates,
            act_gates,
            irreps_gated,
            decoupled=False,
            strict_invariant_gates=strict_invariant_gates,
        )
