"""First-party PhysMuon, Muon, HybridMuon optimizers, presets, and LR schedules.

Clean pure-PyTorch implementation of ``PhysMuon`` (Hyyu-Lab/PhysMuon,
arXiv:2608.22994) with all five switchable physical mechanisms, spike guard,
micro-batch and DDP rank-variance fluctuation estimators, equivariance-safe
parameter routing, ``min(m, n)``-bucketed Newton-Schulz orthogonalization,
and optional whole-step CUDA Graph capture for sub-millisecond optimizer steps.

Exports:
- :class:`PhysMuon`: Full PhysMuon + internal AdamW optimizer.
- :class:`Muon`: Exact standard Muon + internal AdamW (all 5 PhysMuon switches off).
- :class:`HybridMuon`: Backward-compatible SpinGNN2 hybrid Muon/AdamW wrapper.
- :class:`WSDCosineScheduler`: Warmup-Stable-Decay cosine LR schedule.
- Presets: :func:`interatomic_potential_gated`, :func:`interatomic_potential`,
  :func:`language_model`, :func:`paper_bm`, :data:`PHYSMUON_PRESETS`.
- Memory helpers: :func:`age`, :func:`beta_for_age`, :func:`kernel_age`,
  :func:`fast_weight_for_age`, :func:`switch_step`.
"""

from __future__ import annotations

import bisect
import math
from typing import Any, Callable, Iterable, Sequence
import warnings

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

# Newton-Schulz iteration schedules:
# - quintic: Keller Jordan's 5-step quintic iteration (PhysMuon/Muon default)
# - cubic: 12-step cubic iteration
# - two_stage: 8 quintic + 2 exact cubic polish steps (legacy HybridMuon option)
_NS_QUINTIC_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
_NS_CUBIC_COEFFS: tuple[float, float, float] = (2.0, -1.5, 0.5)
_NS_MODES: dict[str, tuple[tuple[float, float, float], int] | None] = {
    "quintic": (_NS_QUINTIC_COEFFS, 5),
    "cubic": (_NS_CUBIC_COEFFS, 12),
    "two_stage": None,
}
_NS_EPS: float = 1e-7

# Magma-lite alignment damping constants (used only when magma_damping=True)
_MAGMA_TAU: float = 2.0
_MAGMA_EMA: float = 0.9
_MAGMA_MIN_SCALE: float = 0.1
_MAGMA_EPS: float = 1e-12
_MAGMA_SIG_MIN: float = 1.0 / (1.0 + math.exp(1.0 / _MAGMA_TAU))
_MAGMA_SIG_MAX: float = 1.0 / (1.0 + math.exp(-1.0 / _MAGMA_TAU))


# ---------------------------------------------------------------------------
# Memory-time schedule helpers (PhysMuon)
# ---------------------------------------------------------------------------


def age(beta: float) -> float:
    """Mean age (in steps) of an EMA with decay ``beta``: ``beta / (1 - beta)``."""
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must lie in [0, 1), got {beta}")
    return beta / (1.0 - beta)


def beta_for_age(mean_age: float) -> float:
    """Inverse of :func:`age`."""
    if mean_age < 0.0:
        raise ValueError(f"mean_age must be >= 0, got {mean_age}")
    return mean_age / (1.0 + mean_age)


def kernel_age(beta_fast: float, beta_slow: float, w_fast: float) -> float:
    """Mean age of ``w_fast * EMA(beta_fast) + (1 - w_fast) * EMA(beta_slow)``."""
    return w_fast * age(beta_fast) + (1.0 - w_fast) * age(beta_slow)


def fast_weight_for_age(beta_fast: float, beta_slow: float, mean_age: float) -> float:
    """Solve for ``w_fast`` giving ``mean_age`` between ``beta_fast`` and ``beta_slow``."""
    af, as_ = age(beta_fast), age(beta_slow)
    if not (min(af, as_) <= mean_age <= max(af, as_)):
        raise ValueError(
            f"mean_age={mean_age} must lie between age(beta_fast)={af} and age(beta_slow)={as_}"
        )
    if af == as_:
        return 0.0
    return (as_ - mean_age) / (as_ - af)


def switch_step(total_steps: int, decay_start_fraction: float) -> int:
    """Step index at which two-mode relaxation readout switches on."""
    if not (0.0 <= decay_start_fraction < 1.0):
        raise ValueError(
            f"decay_start_fraction must lie in [0, 1), got {decay_start_fraction}"
        )
    return int(round(total_steps * decay_start_fraction))


# ---------------------------------------------------------------------------
# Recommended presets (PhysMuon recipes)
# ---------------------------------------------------------------------------


def language_model(total_steps: int = 3250) -> dict[str, Any]:
    """All five PhysMuon changes (GPT-2 124M benchmark constants)."""
    return dict(
        momentum=0.95,
        bm_start=round(total_steps * 750 / 3250),
        bm_beta_fast=0.85,
        bm_beta_slow=0.985,
        iso=True,
        iso_lam=0.5,
        iso_clamp=2.0,
        iso_beta=0.95,
        osc_gate=True,
        osc_wmax=0.6,
        osc_beta=0.9,
        bm_shadow=True,
        fresh_cap=0.25,
    )


gpt_speedrun = language_model


def interatomic_potential(
    total_steps: int,
    memory: float = 170.0,
    switch: float = 0.3077,
) -> dict[str, Any]:
    """Changes 1, 2, 4: staged fast-to-slow switch (``w = 0``) for large-batch MLIPs."""
    return dict(
        bm_start=round(total_steps * switch),
        bm_beta_slow=memory / (1.0 + memory),
        bm_w_fast=0.0,
        iso=True,
        iso_lam=0.5,
        iso_clamp=2.0,
        iso_beta=0.95,
        bm_shadow=True,
    )


def interatomic_potential_gated(
    total_steps: int,
    beta_fast: float = 0.90,
    beta_slow: float = 0.995,
    w_max: float = 0.3,
    switch: float = 0.3077,
) -> dict[str, Any]:
    """Changes 1-4: general MLIP preset (parallel fast+slow modes + oscillation gate)."""
    return dict(
        bm_start=round(total_steps * switch),
        bm_beta_fast=beta_fast,
        bm_beta_slow=beta_slow,
        iso=True,
        iso_lam=0.5,
        iso_clamp=2.0,
        iso_beta=0.95,
        osc_gate=True,
        osc_wmax=w_max,
        osc_beta=0.9,
        bm_shadow=True,
    )


def paper_bm(total_steps: int = 3250) -> dict[str, Any]:
    """Change 1 alone with fixed readout weight (mean memory 30 steps)."""
    return dict(
        momentum=0.95,
        bm_start=round(total_steps * 1000 / 3250),
        bm_beta_fast=0.85,
        bm_beta_slow=0.98,
        bm_w_fast=0.4385,
    )


PHYSMUON_PRESETS: dict[str, Callable[..., dict[str, Any]]] = {
    "interatomic_potential_gated": interatomic_potential_gated,
    "interatomic_potential": interatomic_potential,
    "language_model": language_model,
    "gpt_speedrun": language_model,
    "paper_bm": paper_bm,
}


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization
# ---------------------------------------------------------------------------


def _run_ns_stage(
    cur: torch.Tensor, steps: int, coeffs: tuple[float, float, float]
) -> torch.Tensor:
    a, b, c = coeffs
    is_batched = cur.ndim == 3
    for _ in range(steps):
        if is_batched:
            gram = torch.bmm(cur, cur.transpose(-2, -1))
            poly = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            cur = torch.baddbmm(cur, poly, cur, beta=a, alpha=1.0)
        else:
            gram = torch.mm(cur, cur.transpose(-2, -1))
            poly = torch.addmm(gram, gram, gram, beta=b, alpha=c)
            cur = torch.addmm(cur, poly, cur, beta=a, alpha=1.0)
    return cur


def newton_schulz(
    grad_matrix: torch.Tensor,
    mode: str = "quintic",
    steps: int | None = None,
    eps: float = _NS_EPS,
) -> torch.Tensor:
    """Newton-Schulz iteration toward the polar factor of ``grad_matrix`` (last 2 dims).

    Runs in ``bfloat16`` for FP32/BF16 tensors (hardware Tensor Core fast path)
    and preserves ``float64`` for double-precision tensors.
    """
    if mode not in _NS_MODES:
        raise ValueError(
            f"Unknown ns mode {mode!r}; expected one of {sorted(_NS_MODES)}"
        )
    orig_shape = grad_matrix.shape
    if grad_matrix.ndim < 2:
        raise ValueError(
            f"Expected >=2D tensor for Newton-Schulz, got shape {tuple(orig_shape)}"
        )

    work_dtype = torch.float64 if grad_matrix.dtype == torch.float64 else torch.bfloat16
    x = grad_matrix.to(dtype=work_dtype)
    if x.ndim > 3:
        x = x.reshape(-1, orig_shape[-2], orig_shape[-1])

    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)

    norm = x.norm(dim=(-2, -1), keepdim=True) + eps
    x = x / norm

    if mode == "two_stage":
        fast_steps = 8 if steps is None else int(steps)
        x = _run_ns_stage(x, fast_steps, _NS_QUINTIC_COEFFS)
        x = _run_ns_stage(x, 2, _NS_CUBIC_COEFFS)
    else:
        spec = _NS_MODES[mode]
        assert spec is not None
        coeffs, default_steps = spec
        x = _run_ns_stage(x, default_steps if steps is None else int(steps), coeffs)

    if transposed:
        x = x.transpose(-2, -1)
    if len(orig_shape) > 3:
        x = x.reshape(orig_shape)
    return x


def newton_schulz5(
    grad_matrix: torch.Tensor, steps: int = 5, eps: float = _NS_EPS
) -> torch.Tensor:
    """Keller Jordan's 5-step quintic Newton-Schulz iteration."""
    return newton_schulz(grad_matrix, mode="quintic", steps=steps, eps=eps)


def newton_schulz_orthogonalize(
    grad_matrix: torch.Tensor,
    fast_steps: int = 8,
    polish_steps: int = 2,
) -> torch.Tensor:
    """Orthogonalize a 2D ``(m, n)`` or 3D ``(B, m, n)`` tensor via Newton-Schulz.

    Defaults to the two-stage schedule (``fast_steps=8, polish_steps=2``) for
    tight ``< 1e-3`` polar orthogonality; pass ``fast_steps=5, polish_steps=0``
    or call :func:`newton_schulz` for the 5-step quintic optimizer iteration.
    """
    if grad_matrix.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2D or 3D tensor for Newton-Schulz, got shape {tuple(grad_matrix.shape)}"
        )
    work_dtype = torch.float64 if grad_matrix.dtype == torch.float64 else torch.bfloat16
    x = grad_matrix.to(dtype=work_dtype)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    norm = x.norm(dim=(-2, -1), keepdim=True).clamp(min=_NS_EPS)
    x = x / norm
    if fast_steps > 0:
        x = _run_ns_stage(x, fast_steps, _NS_QUINTIC_COEFFS)
    if polish_steps > 0:
        x = _run_ns_stage(x, polish_steps, _NS_CUBIC_COEFFS)
    if transposed:
        x = x.transpose(-2, -1)
    return x


def _iso_metric(heat: torch.Tensor, lam: float, clamp: float) -> torch.Tensor:
    m = (heat / heat.mean().clamp_min(1e-30)).clamp_min(1e-30) ** lam
    return m.clamp(1.0 / clamp, clamp)


# ---------------------------------------------------------------------------
# Equivariance-safe parameter routing
# ---------------------------------------------------------------------------


def _effective_shape(shape: torch.Size | tuple[int, ...]) -> tuple[int, ...]:
    eff = tuple(int(d) for d in shape if int(d) != 1)
    return eff if eff else (1,)


def _classify_param_name(
    name: str | None,
    adam_patterns: Sequence[str],
) -> str:
    """Return ``'adam'``, ``'adamw'``, or ``'muon'`` from parameter name rules."""
    if name is None:
        return "muon"
    lowered = name.lower()
    segments = lowered.split(".")
    leaf_idx = len(segments) - 1
    while leaf_idx > 0 and segments[leaf_idx].isdigit():
        leaf_idx -= 1
    leaf = segments[leaf_idx]

    if leaf == "b" or "bias" in leaf or leaf.startswith("adam_"):
        return "adam"
    if leaf.startswith("adamw_"):
        return "adamw"
    if any(
        token in lowered
        for token in (
            "embedding",
            "embed",
            "norm",
            "scale",
            "shift",
            "spin_gate",
            "vacuum",
            "species",
            "awl",
        )
    ):
        return "adamw"
    if any(pat in lowered for pat in adam_patterns):
        return "adamw"
    return "muon"


def _matrix_view_shape(
    eff_shape: tuple[int, ...], muon_mode: str, min_dim: int = 2
) -> tuple[int, int, int] | None:
    if len(eff_shape) < 2:
        return None
    if muon_mode == "2d":
        if len(eff_shape) != 2:
            return None
        if min(eff_shape[0], eff_shape[1]) < min_dim:
            return None
        return (1, eff_shape[0], eff_shape[1])
    if muon_mode == "flat":
        rows = int(math.prod(eff_shape[:-1]))
        cols = eff_shape[-1]
        if min(rows, cols) < min_dim:
            return None
        return (1, rows, cols)
    if muon_mode == "slice":
        if len(eff_shape) == 2:
            if min(eff_shape[0], eff_shape[1]) < min_dim:
                return None
            return (1, eff_shape[0], eff_shape[1])
        rows, cols = eff_shape[-2], eff_shape[-1]
        if min(rows, cols) < min_dim:
            return None
        return (int(math.prod(eff_shape[:-2])), rows, cols)
    raise ValueError(f"Unknown muon_mode={muon_mode!r}; expected '2d', 'flat', 'slice'")


def _scale_factor(rows: int, cols: int, scale: str | float) -> float:
    if scale == "shape":
        return max(1.0, float(rows) / float(cols)) ** 0.5
    return float(scale) * math.sqrt(float(max(rows, cols)))


# ---------------------------------------------------------------------------
# PhysMuon / Muon / HybridMuon
# ---------------------------------------------------------------------------


class PhysMuon(Optimizer):
    """Muon with five switchable physical mechanisms + internal AdamW + CUDA Graph step.

    Equivariance-safe auto-routing (when ``use_muon=None``, the default):
    - Routes true 2D hidden/equivariant channel matrices (``len(eff_shape) == 2``
      and ``min(eff_shape) >= 2``, excluding biases, norms, embeddings, species
      tables, scales, gates, vacuum anchors, and ``adam_*``/``adamw_*`` leaves)
      through Muon / PhysMuon.
    - Protects single-row ``(1, K)`` scalar energy and zero-init SOC readout
      heads by routing them to internal AdamW.
    - Respects explicit per-group ``use_muon=True`` or ``use_muon=False`` overrides
      (passing ``check_param_group_options`` in v1 YAML configs cleanly).
    - Whole-step CUDA Graph capture (``cuda_graph=True`` on CUDA) accelerates
      ``step()`` to sub-millisecond latency with exact parity across dynamic WSD
      LR updates, ``bm_start`` transitions, ``spike_guard`` skips, ``weight_decay``,
      ``iso=True`` heat refreshes, DDP grad buckets, and ``grad=None`` subsets.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns: str = "quintic",
        ns_steps: int | None = None,
        scale: str | float = "shape",
        bm_start: int | None = None,
        bm_beta_fast: float = 0.85,
        bm_beta_slow: float = 0.98,
        bm_w_fast: float = 0.4385,
        iso: bool = False,
        iso_lam: float = 0.5,
        iso_clamp: float = 2.0,
        iso_beta: float = 0.95,
        iso_sync: bool = True,
        iso_min_dim: int = 4,
        iso_spike_skip: bool = True,
        osc_gate: bool = False,
        osc_wmax: float = 0.6,
        osc_beta: float = 0.9,
        bm_shadow: bool = False,
        fresh_cap: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        spike_guard: float = 0.0,
        spike_window: int = 1000,
        muon_mode: str = "2d",
        named_parameters: Iterable[tuple[str, torch.Tensor]] | None = None,
        adam_patterns: Sequence[str] = (),
        magma_damping: bool = False,
        cuda_graph: bool = True,
        skip_nonfinite: bool | None = None,
    ) -> None:
        muon_mode = str(muon_mode).lower()
        if muon_mode not in ("2d", "flat", "slice"):
            raise ValueError(
                f"Invalid muon_mode {muon_mode!r}; expected '2d', 'flat', or 'slice'"
            )
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not spike_guard >= 0.0 or spike_window < 10:
            raise ValueError("spike_guard must be >= 0 and spike_window >= 10")

        defaults: dict[str, Any] = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            ns=str(ns),
            ns_steps=ns_steps,
            scale=scale,
            bm_start=bm_start,
            bm_beta_fast=float(bm_beta_fast),
            bm_beta_slow=float(bm_beta_slow),
            bm_w_fast=float(bm_w_fast),
            iso=bool(iso),
            iso_lam=float(iso_lam),
            iso_clamp=float(iso_clamp),
            iso_beta=float(iso_beta),
            iso_sync=bool(iso_sync),
            iso_min_dim=int(iso_min_dim),
            iso_spike_skip=bool(iso_spike_skip),
            osc_gate=bool(osc_gate),
            osc_wmax=float(osc_wmax),
            osc_beta=float(osc_beta),
            bm_shadow=bool(bm_shadow),
            fresh_cap=float(fresh_cap),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            use_muon=None,
            muon_mode=muon_mode,
            magma_damping=bool(magma_damping),
        )
        super().__init__(params, defaults)

        self._adam_patterns = tuple(str(p).lower() for p in adam_patterns)
        self._param_names: dict[int, str] = {}
        if named_parameters is not None:
            for name, param in named_parameters:
                self._param_names[id(param)] = str(name)
        self._route_cache: dict[
            tuple[int, Any, str], tuple[str, tuple[int, int, int] | None]
        ] = {}

        for group in self.param_groups:
            if group["use_muon"] is not False:
                self._check_muon_group(group)

        self.spike_guard = float(spike_guard)
        self.spike_window = int(spike_window)
        self.skip_nonfinite = (
            bool(skip_nonfinite)
            if skip_nonfinite is not None
            else bool(self.spike_guard > 0.0)
        )
        self._spike_sorted: list[float] = []
        self._spike_fifo: list[float] = []
        self.spike_frozen_steps: int = 0
        self.nonfinite_skipped_steps: int = 0
        self._warned_iso: bool = False

        self.cuda_graph_enabled = bool(cuda_graph)
        # Cached CUDA graphs keyed by (grad_sig, phase_sig)
        self._cuda_graphs: dict[Any, dict[str, Any]] = {}
        self._ddp_iso_handles: list[Any] = []

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        # Invalidate any captured CUDA graphs and keep auxiliary stats in float32
        self._cuda_graphs.clear()
        self._route_cache.clear()
        for st in self.state.values():
            if not isinstance(st, dict):
                continue
            for key in (
                "row_heat",
                "col_heat",
                "osc",
                "drift",
                "magma_ema",
                "iso_rs2",
                "iso_cs2",
                "iso_prev",
            ):
                val = st.get(key)
                if torch.is_tensor(val) and val.dtype != torch.float32:
                    st[key] = val.to(dtype=torch.float32)

    @staticmethod
    def _check_muon_group(group: dict[str, Any]) -> None:
        if group["ns"] not in _NS_MODES:
            raise ValueError(f"ns must be one of {sorted(_NS_MODES)}")
        if not 0.0 <= group["momentum"] < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        needs_memory = [k for k in ("osc_gate", "bm_shadow") if group[k]] + (
            ["fresh_cap"] if group["fresh_cap"] else []
        )
        if group["bm_start"] is None:
            if needs_memory:
                raise ValueError(
                    f"{needs_memory} act on the two relaxation modes: set bm_start"
                )
        else:
            for k in ("bm_beta_fast", "bm_beta_slow"):
                if not 0.0 <= group[k] < 1.0:
                    raise ValueError(f"{k} must lie in [0, 1)")
            if not 0.0 <= group["bm_w_fast"] <= 1.0:
                raise ValueError("bm_w_fast must lie in [0, 1]")
            if not 0.0 < group["osc_wmax"] <= 1.0 or not 0.0 <= group["osc_beta"] < 1.0:
                raise ValueError("osc_wmax must lie in (0, 1] and osc_beta in [0, 1)")
            if not 0.0 <= group["fresh_cap"] < 1.0:
                raise ValueError("fresh_cap must lie in [0, 1)")
        if group["iso"] and (
            group["iso_lam"] < 0.0
            or group["iso_clamp"] < 1.0
            or not 0.0 <= group["iso_beta"] < 1.0
        ):
            raise ValueError(
                "iso_lam must be >= 0, iso_clamp >= 1 and iso_beta in [0, 1)"
            )

    def _resolve_route(
        self,
        param: torch.Tensor,
        group: dict[str, Any],
    ) -> tuple[str, tuple[int, int, int] | None]:
        use_muon = group["use_muon"]
        muon_mode = group["muon_mode"]
        cache_key = (id(param), use_muon, muon_mode)
        cached = self._route_cache.get(cache_key)
        if cached is not None:
            return cached

        if use_muon is False:
            name_route = _classify_param_name(
                self._param_names.get(id(param)), self._adam_patterns
            )
            res: tuple[str, tuple[int, int, int] | None] = (
                ("adam", None) if name_route == "adam" else ("adamw", None)
            )
            self._route_cache[cache_key] = res
            return res

        if use_muon is True:
            if param.ndim < 2:
                res = ("adamw", None)
            elif param.ndim == 2:
                res = ("muon", (1, int(param.shape[0]), int(param.shape[1])))
            else:
                bs = int(math.prod(param.shape[:-2]))
                res = ("muon", (bs, int(param.shape[-2]), int(param.shape[-1])))
            self._route_cache[cache_key] = res
            return res

        # Auto-routing (use_muon is None): equivariance-safe rules
        name = self._param_names.get(id(param))
        eff_shape = _effective_shape(param.shape)
        if len(eff_shape) < 2:
            res = ("adam", None)
        else:
            name_route = _classify_param_name(name, self._adam_patterns)
            if name_route in ("adam", "adamw"):
                res = (name_route, None)
            else:
                mat_shape = _matrix_view_shape(eff_shape, muon_mode, min_dim=2)
                if mat_shape is None:
                    res = ("adamw", None)
                else:
                    res = ("muon", mat_shape)

        self._route_cache[cache_key] = res
        return res

    def routing_summary(self) -> dict[str, list[str]]:
        """Return parameter names (or indices) grouped by optimizer route."""
        summary: dict[str, list[str]] = {"muon": [], "adam": [], "adamw": []}
        idx = 0
        for group in self.param_groups:
            for param in group["params"]:
                route, _ = self._resolve_route(param, group)
                label = self._param_names.get(id(param), f"param_{idx}")
                summary[route].append(label)
                idx += 1
        return summary

    def _can_use_iso_for_param(self, p: torch.Tensor, group: dict[str, Any]) -> bool:
        if not group["iso"] or p.ndim != 2:
            return False
        min_dim = 1 if group["use_muon"] is True else int(group["iso_min_dim"])
        return min(int(p.size(0)), int(p.size(1))) >= min_dim

    # -----------------------------------------------------------------------
    # Change 2: Micro-batch & DDP rank-variance fluctuation statistics
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def observe_microbatch(self) -> None:
        """Record the gradient of the micro-batch whose backward pass just finished.

        Call after every ``loss.backward()`` while gradients accumulate across
        micro-batches. Statistics are consumed and cleared by ``step()``.
        """
        for group in self.param_groups:
            if not group["iso"]:
                continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                route, _ = self._resolve_route(p, group)
                if route != "muon" or not self._can_use_iso_for_param(p, group):
                    continue
                st = self.state[p]
                if st.get("iso_count", 0) == 0:
                    st["iso_rs2"] = torch.zeros(
                        p.size(0), device=p.device, dtype=torch.float32
                    )
                    st["iso_cs2"] = torch.zeros(
                        p.size(1), device=p.device, dtype=torch.float32
                    )
                    st["iso_count"] = 0
                    st.pop("iso_prev", None)
                g_cur = p.grad.detach().float()
                gk = g_cur if "iso_prev" not in st else g_cur - st["iso_prev"]
                gk2 = gk.square()
                st["iso_rs2"].add_(gk2.sum(dim=1))
                st["iso_cs2"].add_(gk2.sum(dim=0))
                st["iso_count"] += 1
                st["iso_prev"] = g_cur.clone()

    @torch.no_grad()
    def record_ddp_rank_variance(self) -> None:
        """Estimate gradient fluctuation variance across DDP ranks (zero micro-batching).

        Call after local ``backward()`` (when ``p.grad`` holds local rank gradient
        before reduction, or via :meth:`register_ddp_iso_hook`). Computes unbiased
        cross-rank variance ``(g_r - g_mean)^2`` using standard ``dist.all_reduce``.
        """
        if not (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        ):
            return
        world = float(dist.get_world_size())
        for group in self.param_groups:
            if not group["iso"]:
                continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                route, _ = self._resolve_route(p, group)
                if route != "muon" or not self._can_use_iso_for_param(p, group):
                    continue
                g_loc = p.grad.detach().float()
                g_mean = g_loc.clone()
                dist.all_reduce(g_mean)
                g_mean.div_(world)
                diff2 = (g_loc - g_mean).square()
                rh = diff2.mean(dim=1)
                ch = diff2.mean(dim=0)
                dist.all_reduce(rh)
                dist.all_reduce(ch)
                rh.div_(max(world - 1.0, 1.0))
                ch.div_(max(world - 1.0, 1.0))
                st = self.state[p]
                st["_ddp_rh"] = rh.clamp_min(0.0) + 1e-30
                st["_ddp_ch"] = ch.clamp_min(0.0) + 1e-30

    def register_ddp_iso_hook(self, ddp_model: torch.nn.Module) -> None:
        """Attach a standard PyTorch DDP comm hook estimating cross-rank iso heat."""
        if not hasattr(ddp_model, "register_comm_hook"):
            return
        iso_params: dict[int, tuple[torch.Tensor, dict[str, Any]]] = {}
        for group in self.param_groups:
            if not group["iso"]:
                continue
            for p in group["params"]:
                route, _ = self._resolve_route(p, group)
                if route == "muon" and self._can_use_iso_for_param(p, group):
                    iso_params[id(p)] = (p, group)

        if not iso_params:
            return

        def _hook(state: Any, bucket: Any) -> torch.futures.Future:
            world = float(dist.get_world_size())
            buf = bucket.buffer()
            local_by_id: dict[int, torch.Tensor] = {}
            for p, g_view in zip(bucket.parameters(), bucket.gradients()):
                pid = id(p)
                if pid in iso_params:
                    local_by_id[pid] = g_view.detach().float().clone()

            fut = dist.all_reduce(buf, async_op=True).get_future()

            def _after(f: torch.futures.Future) -> torch.Tensor:
                t_sum = f.value()[0]
                t_sum.div_(world)
                # Store only local squared-diff means here (never invoke blocking
                # dist.all_reduce inside the DDP comm-hook callback); the main
                # thread performs one batched all_reduce inside _refresh_heats().
                for p, g_view in zip(bucket.parameters(), bucket.gradients()):
                    g_loc = local_by_id.get(id(p))
                    if g_loc is not None:
                        diff2 = (g_loc - g_view.detach().float()).square()
                        st = self.state[p]
                        st["_ddp_rh_local"] = diff2.mean(dim=1)
                        st["_ddp_ch_local"] = diff2.mean(dim=0)
                return t_sum

            return fut.then(_after)

        ddp_model.register_comm_hook(state=None, hook=_hook)

    def _refresh_heats(self, group: dict[str, Any], spike: bool) -> bool:
        """Consume micro-batch / DDP variance stats into EMA row/col heats.

        Returns ``True`` if any parameter newly created ``row_heat``/``col_heat``.
        """
        # Entry tag: "ddp_local" (needs main-thread all_reduce / (world - 1)),
        # "ddp_ready" (already reduced by record_ddp_rank_variance), or
        # "mb" (micro-batch stats, synced across ranks when iso_sync=True).
        todo: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]] = []
        created_new = False
        for p in group["params"]:
            if p.grad is None:
                continue
            route, _ = self._resolve_route(p, group)
            if route != "muon" or not self._can_use_iso_for_param(p, group):
                continue
            st = self.state[p]
            if "_ddp_rh_local" in st and "_ddp_ch_local" in st:
                rh = st.pop("_ddp_rh_local")
                ch = st.pop("_ddp_ch_local")
                st["iso_count"] = 0
                st.pop("iso_prev", None)
                todo.append((p, rh, ch, "ddp_local"))
                continue
            if "_ddp_rh" in st and "_ddp_ch" in st:
                rh = st.pop("_ddp_rh")
                ch = st.pop("_ddp_ch")
                st["iso_count"] = 0
                st.pop("iso_prev", None)
                todo.append((p, rh, ch, "ddp_ready"))
                continue
            k_mb = st.get("iso_count", 0)
            if k_mb == 0:
                continue
            if k_mb < 2:
                # 1-micro-batch tail has zero variance estimate: reset cleanly without
                # shrinking EMA heat toward 1e-30.
                st["iso_count"] = 0
                st.pop("iso_prev", None)
                continue
            gm = st["iso_prev"] / float(k_mb)
            rh = st["iso_rs2"] / float(k_mb) / float(p.size(1)) - gm.square().sum(
                dim=1
            ) / float(p.size(1))
            ch = st["iso_cs2"] / float(k_mb) / float(p.size(0)) - gm.square().sum(
                dim=0
            ) / float(p.size(0))
            todo.append((p, rh.clamp_min(0.0) + 1e-30, ch.clamp_min(0.0) + 1e-30, "mb"))
            st["iso_count"] = 0
            st.pop("iso_prev", None)

        if not todo:
            if not self._warned_iso and any(
                p.grad is not None
                and self._resolve_route(p, group)[0] == "muon"
                and self._can_use_iso_for_param(p, group)
                and "row_heat" not in self.state[p]
                for p in group["params"]
            ):
                warnings.warn(
                    "iso=True but observe_microbatch() was never called: change (2) "
                    "falls back to the plain polar factor until micro-batch or DDP "
                    "fluctuation statistics arrive.",
                    stacklevel=3,
                )
                self._warned_iso = True
            return False

        if spike and group["iso_spike_skip"]:
            return False

        dist_ready = (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        )
        # Main-thread batched all_reduce for DDP comm-hook local squared diffs
        ddp_local_items = [item for item in todo if item[3] == "ddp_local"]
        if ddp_local_items:
            flat = torch.cat([torch.cat([rh, ch]) for _, rh, ch, _ in ddp_local_items])
            if dist_ready:
                dist.all_reduce(flat)
                denom = max(float(dist.get_world_size()) - 1.0, 1.0)
                flat.div_(denom)
            reduced_map: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            offset = 0
            for p, rh, ch, _ in ddp_local_items:
                nr, nc = rh.numel(), ch.numel()
                reduced_map[id(p)] = (
                    flat[offset : offset + nr].clamp_min(0.0) + 1e-30,
                    flat[offset + nr : offset + nr + nc].clamp_min(0.0) + 1e-30,
                )
                offset += nr + nc
            todo = [
                (p, reduced_map[id(p)][0], reduced_map[id(p)][1], "ddp_ready")
                if tag == "ddp_local"
                else (p, rh, ch, tag)
                for p, rh, ch, tag in todo
            ]

        # Synchronize micro-batch heats across DDP ranks if requested
        need_sync = [item for item in todo if item[3] == "mb"]
        if need_sync and group["iso_sync"] and dist_ready:
            flat = torch.cat([torch.cat([rh, ch]) for _, rh, ch, _ in need_sync])
            dist.all_reduce(flat)
            flat.div_(float(dist.get_world_size()))
            synced_map: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            offset = 0
            for p, rh, ch, _ in need_sync:
                nr, nc = rh.numel(), ch.numel()
                synced_map[id(p)] = (
                    flat[offset : offset + nr],
                    flat[offset + nr : offset + nr + nc],
                )
                offset += nr + nc
            todo = [
                (p, synced_map[id(p)][0], synced_map[id(p)][1], tag)
                if id(p) in synced_map
                else (p, rh, ch, tag)
                for p, rh, ch, tag in todo
            ]

        iso_beta = group["iso_beta"]
        for p, rh, ch, _ in todo:
            st = self.state[p]
            if "row_heat" not in st:
                st["row_heat"] = rh.float().clone()
                st["col_heat"] = ch.float().clone()
                created_new = True
            else:
                st["row_heat"].lerp_(rh.to(dtype=st["row_heat"].dtype), 1.0 - iso_beta)
                st["col_heat"].lerp_(ch.to(dtype=st["col_heat"].dtype), 1.0 - iso_beta)
        return created_new

    # -----------------------------------------------------------------------
    # Spike guard
    # -----------------------------------------------------------------------

    def _global_grad_norm(self) -> float:
        norms = [
            p.grad.detach().float().norm()
            for g in self.param_groups
            for p in g["params"]
            if p.grad is not None
        ]
        return float(torch.stack(norms).norm()) if norms else 0.0

    def _clear_pending_iso(self) -> None:
        """Discard any pending micro-batch / DDP fluctuation accumulators."""
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                st["iso_count"] = 0
                st.pop("iso_prev", None)
                st.pop("_ddp_rh_local", None)
                st.pop("_ddp_ch_local", None)
                st.pop("_ddp_iso_rh", None)
                st.pop("_ddp_iso_ch", None)
                st.pop("_ddp_rh", None)
                st.pop("_ddp_ch", None)

    def _spike_decision(self, grad_norm: float | None) -> tuple[bool, bool]:
        """Return ``(is_spike, is_nonfinite)`` with zero sync when both guards are off."""
        if self.spike_guard <= 0.0 and not self.skip_nonfinite:
            return False, False
        v = float(grad_norm) if grad_norm is not None else self._global_grad_norm()
        if not math.isfinite(v):
            if self.spike_guard > 0.0:
                self.spike_frozen_steps += 1
            return True, True
        if self.spike_guard <= 0.0:
            return False, False
        window = self._spike_sorted
        spike = False
        if len(window) >= min(100, self.spike_window):
            spike = v > self.spike_guard * window[len(window) // 2]
        bisect.insort(window, v)
        self._spike_fifo.append(v)
        if len(self._spike_fifo) > self.spike_window:
            window.pop(bisect.bisect_left(window, self._spike_fifo.pop(0)))
        self.spike_frozen_steps += int(spike)
        return spike, False

    # -----------------------------------------------------------------------
    # Core Muon & AdamW step kernels (shared by eager & CUDA-Graph capture!)
    # -----------------------------------------------------------------------

    def _muon_step_entries(
        self,
        entries: list[
            tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], bool, bool]
        ],
        group: dict[str, Any],
        lr_t: torch.Tensor,
        wd_mul_t: torch.Tensor,
        slow_alpha_t: torch.Tensor,
    ) -> None:
        if not entries:
            return

        mu = group["momentum"]
        nesterov = group["nesterov"]
        ns_mode = group["ns"]
        ns_steps = group["ns_steps"]
        scale = group["scale"]
        bm_start = group["bm_start"]
        bm_shadow = group["bm_shadow"]
        bf = group["bm_beta_fast"]
        bs = group["bm_beta_slow"]
        w_fixed = group["bm_w_fast"]
        osc_gate = group["osc_gate"]
        osc_wmax = group["osc_wmax"]
        osc_beta = group["osc_beta"]
        fresh_cap = group["fresh_cap"]
        magma_damping = group["magma_damping"]
        iso_on = group["iso"]
        iso_lam = group["iso_lam"]
        iso_clamp = group["iso_clamp"]
        wd = group["weight_decay"]

        params: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        shapes: list[tuple[int, int, int]] = []
        u_list: list[torch.Tensor] = []
        mom_bufs: list[torch.Tensor] = []

        # Fast pure-single-EMA path when no entry needs two-mode or seeding
        all_single_ema = bm_start is None or (
            not bm_shadow and not any(is_p or sd for _p, _g, _ms, is_p, sd in entries)
        )

        if all_single_ema:
            for p, g, mat_shape, _is_p, _sd in entries:
                st = self.state[p]
                params.append(p)
                grads.append(g)
                mom_bufs.append(st["momentum"])
                shapes.append(mat_shape)
            torch._foreach_lerp_(mom_bufs, grads, 1.0 - mu)
            if nesterov:
                u_list = list(torch._foreach_lerp(grads, mom_bufs, mu))
            else:
                u_list = [m.clone() for m in mom_bufs]
        else:
            rho_f = (1.0 - bf) / (1.0 + bf)
            for p, g, mat_shape, is_post_p, seed_now_p in entries:
                st = self.state[p]
                params.append(p)
                grads.append(g)
                mom_bufs.append(st["momentum"])
                shapes.append(mat_shape)

                if seed_now_p:
                    m = st["momentum"]
                    m.lerp_(g, 1.0 - mu)
                    u = g.lerp(m, mu) if nesterov else m.clone()
                    st["m_fast"] = m.clone()
                    st["m_slow"] = m.clone()
                    u_list.append(u)
                    continue

                if not is_post_p and not bm_shadow:
                    m = st["momentum"]
                    m.lerp_(g, 1.0 - mu)
                    u = g.lerp(m, mu) if nesterov else m.clone()
                    u_list.append(u)
                    continue

                gf = g.float()
                g_norm = gf.norm().clamp_min(1e-20)
                if osc_gate:
                    mf = st["m_fast"].float()
                    osc = (
                        -(gf * mf).sum() / (g_norm * mf.norm().clamp_min(1e-20))
                    ).clamp_min(0.0)
                    drift = st["m_slow"].float().norm() / g_norm
                    if not st["osc_initialized"]:
                        st["osc"].copy_(osc.to(dtype=st["osc"].dtype))
                        st["drift"].copy_(drift.to(dtype=st["drift"].dtype))
                        st["osc_initialized"] = True
                    else:
                        st["osc"].lerp_(osc.to(dtype=st["osc"].dtype), 1.0 - osc_beta)
                        st["drift"].lerp_(
                            drift.to(dtype=st["drift"].dtype), 1.0 - osc_beta
                        )

                st["m_fast"].lerp_(g, 1.0 - bf)
                # Spike-guard safe: slow_alpha_t is 0.0 on spike, (1 - bs) otherwise
                st["m_slow"].lerp_(g, slow_alpha_t.to(dtype=g.dtype))

                if not is_post_p:
                    m = st["momentum"]
                    m.lerp_(g, 1.0 - mu)
                    u = g.lerp(m, mu) if nesterov else m.clone()
                else:
                    if osc_gate:
                        w = osc_wmax * (
                            1.0 - rho_f * st["osc"] / st["drift"].clamp_min(1e-20)
                        ).clamp(0.0, 1.0)
                    else:
                        w = w_fixed
                    m_eff = w * st["m_fast"] + (1.0 - w) * st["m_slow"]
                    if fresh_cap > 0.0:
                        mu_floor = torch.full_like(g_norm, mu)
                        mu_t = torch.maximum(
                            mu_floor,
                            1.0
                            / (
                                1.0
                                + fresh_cap
                                / (1.0 - fresh_cap)
                                * m_eff.float().norm()
                                / g_norm
                            ),
                        )
                    else:
                        mu_t = mu
                    m_eff_cast = m_eff.to(dtype=g.dtype)
                    if nesterov:
                        if torch.is_tensor(mu_t):
                            u = g.lerp(m_eff_cast, mu_t.to(dtype=g.dtype))
                        else:
                            u = g.lerp(m_eff_cast, mu_t)
                    else:
                        u = m_eff_cast
                    st["momentum"].copy_(m_eff_cast)
                u_list.append(u)

        # Separate iso-active 2D matrices vs plain bucketed matrices
        # (both use column-padded min(m, n) batched NS!)
        buckets: dict[tuple[int, torch.device, torch.dtype], list[int]] = {}
        iso_factors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for idx, (p, (_bs, rows, cols)) in enumerate(zip(params, shapes)):
            st = self.state[p]
            if iso_on and self._can_use_iso_for_param(p, group) and "row_heat" in st:
                bi = _iso_metric(st["row_heat"].float(), iso_lam, iso_clamp).rsqrt()[
                    :, None
                ]
                ai = _iso_metric(st["col_heat"].float(), iso_lam, iso_clamp).rsqrt()[
                    None, :
                ]
                iso_factors[idx] = (bi, ai)
            key = (min(rows, cols), p.device, p.dtype)
            buckets.setdefault(key, []).append(idx)

        deltas: list[torch.Tensor] = [None] * len(params)  # type: ignore[list-item]

        for (_min_dim, _dev, _dt), indices in buckets.items():
            max_cols = max(max(shapes[i][1], shapes[i][2]) for i in indices)
            wide_list: list[torch.Tensor] = []
            meta: list[tuple[int, int, int, int, int, bool]] = []
            for i in indices:
                bs, rows, cols = shapes[i]
                mat = u_list[i].reshape(bs, rows, cols)
                if i in iso_factors:
                    bi, ai = iso_factors[i]
                    mat = (mat.float() * bi.unsqueeze(0)) * ai.unsqueeze(0)
                    mat = mat.to(dtype=params[i].dtype)
                transposed = rows > cols
                if transposed:
                    mat = mat.transpose(-2, -1)
                orig_max = max(rows, cols)
                pad_w = max_cols - orig_max
                if pad_w > 0:
                    mat = F.pad(mat, (0, pad_w))
                wide_list.append(mat)
                meta.append((i, bs, rows, cols, orig_max, transposed))

            stacked = torch.cat(wide_list, dim=0)
            orth_all = newton_schulz(stacked, mode=ns_mode, steps=ns_steps)

            if magma_damping:
                g_wide: list[torch.Tensor] = []
                m_wide: list[torch.Tensor] = []
                for i in indices:
                    bs, rows, cols = shapes[i]
                    transposed = rows > cols
                    gm = grads[i].reshape(bs, rows, cols)
                    mm = mom_bufs[i].reshape(bs, rows, cols)
                    if transposed:
                        gm = gm.transpose(-2, -1)
                        mm = mm.transpose(-2, -1)
                    pad_w = max_cols - max(rows, cols)
                    if pad_w > 0:
                        gm = F.pad(gm, (0, pad_w))
                        mm = F.pad(mm, (0, pad_w))
                    g_wide.append(gm.reshape(bs, -1).float())
                    m_wide.append(mm.reshape(bs, -1).float())
                g_cat = torch.cat(g_wide, dim=0)
                m_cat = torch.cat(m_wide, dim=0)
                dot = (m_cat * g_cat).sum(dim=1)
                denom = (m_cat.norm(dim=1) * g_cat.norm(dim=1)).clamp(min=_MAGMA_EPS)
                cosine = (dot / denom).clamp(min=-1.0, max=1.0)
                sig = torch.sigmoid(cosine / _MAGMA_TAU)
                raw_scores: torch.Tensor | None = (
                    (sig - _MAGMA_SIG_MIN) / (_MAGMA_SIG_MAX - _MAGMA_SIG_MIN)
                ).clamp(0.0, 1.0)
            else:
                raw_scores = None

            offset = 0
            for i, bs, rows, cols, orig_max, transposed in meta:
                orth = orth_all[offset : offset + bs]
                if orig_max < max_cols:
                    orth = orth[:, :, :orig_max]
                if transposed:
                    orth = orth.transpose(-2, -1)

                if i in iso_factors:
                    bi, ai = iso_factors[i]
                    d_mat = (orth.float() * bi.unsqueeze(0)) * ai.unsqueeze(0)
                    frob_target = math.sqrt(float(min(rows, cols)))
                    d_norm = d_mat.norm(dim=(-2, -1), keepdim=True) + 1e-12
                    orth = (d_mat * (frob_target / d_norm)).to(dtype=params[i].dtype)
                else:
                    orth = orth.to(dtype=params[i].dtype)

                s_fac = _scale_factor(rows, cols, scale)
                orth = orth * s_fac

                if raw_scores is not None:
                    st = self.state[params[i]]
                    ema = st["magma_ema"]
                    ema.mul_(_MAGMA_EMA).add_(
                        raw_scores[offset : offset + bs].to(dtype=ema.dtype),
                        alpha=1.0 - _MAGMA_EMA,
                    )
                    damp = (_MAGMA_MIN_SCALE + (1.0 - _MAGMA_MIN_SCALE) * ema).to(
                        dtype=params[i].dtype
                    )
                    orth = orth * damp.view(bs, 1, 1)

                deltas[i] = orth.reshape_as(params[i])
                offset += bs

        # Apply decoupled weight decay & LR step using 0-d GPU scalar tensors
        # (100% CUDA Graph safe across dynamic WSD cosine LR changes!)
        if wd != 0.0:
            for dt_p, dt_list in self._group_by_dtype(params):
                torch._foreach_mul_(dt_list, wd_mul_t.to(dtype=dt_p))
        for dt_p, p_sub, d_sub in self._group_pairs_by_dtype(params, deltas):
            neg_lr = (-lr_t).to(dtype=dt_p)
            scaled = torch._foreach_mul(d_sub, neg_lr)
            torch._foreach_add_(p_sub, scaled)

    def _adamw_step_entries(
        self,
        entries: list[tuple[torch.Tensor, torch.Tensor, bool]],
        group: dict[str, Any],
        wd_mul_t: torch.Tensor,
        neg_step_size_t: torch.Tensor,
        inv_sqrt_bc2_t: torch.Tensor,
        uniform_step: bool,
    ) -> None:
        if not entries:
            return
        b1, b2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]
        lr = group["lr"]

        buckets: dict[
            tuple[torch.device, torch.dtype],
            list[tuple[torch.Tensor, torch.Tensor, bool]],
        ] = {}
        for item in entries:
            key = (item[0].device, item[0].dtype)
            buckets.setdefault(key, []).append(item)

        for (_dev, dt), items in buckets.items():
            acc_dtype = torch.float64 if dt == torch.float64 else torch.float32
            params: list[torch.Tensor] = []
            grads_acc: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            wd_params: list[torch.Tensor] = []
            steps: list[int] = []

            for p, g, use_wd in items:
                st = self.state[p]
                params.append(p)
                grads_acc.append(g if g.dtype == acc_dtype else g.to(dtype=acc_dtype))
                exp_avgs.append(st["exp_avg"])
                exp_avg_sqs.append(st["exp_avg_sq"])
                steps.append(int(st["step"]))
                if use_wd and wd != 0.0:
                    wd_params.append(p)

            if wd_params:
                torch._foreach_mul_(wd_params, wd_mul_t.to(dtype=dt))

            torch._foreach_mul_(exp_avgs, b1)
            torch._foreach_add_(exp_avgs, grads_acc, alpha=1.0 - b1)
            torch._foreach_mul_(exp_avg_sqs, b2)
            torch._foreach_addcmul_(exp_avg_sqs, grads_acc, grads_acc, value=1.0 - b2)

            if uniform_step:
                denom = torch._foreach_sqrt(exp_avg_sqs)
                torch._foreach_mul_(denom, inv_sqrt_bc2_t.to(dtype=acc_dtype))
                torch._foreach_add_(denom, eps)
                updates = torch._foreach_div(exp_avgs, denom)
                torch._foreach_mul_(updates, neg_step_size_t.to(dtype=acc_dtype))
                if acc_dtype != dt:
                    updates = [u.to(dtype=dt) for u in updates]
                torch._foreach_add_(params, updates)
            else:
                for p, ea, easq, st_k in zip(params, exp_avgs, exp_avg_sqs, steps):
                    bc1 = 1.0 - b1**st_k
                    bc2 = 1.0 - b2**st_k
                    denom = (easq.sqrt() / math.sqrt(bc2)).add_(eps)
                    p.add_((ea / denom).to(dtype=dt), alpha=-(lr / bc1))

    @staticmethod
    def _group_by_dtype(
        tensors: list[torch.Tensor],
    ) -> list[tuple[torch.dtype, list[torch.Tensor]]]:
        by_dt: dict[torch.dtype, list[torch.Tensor]] = {}
        for t in tensors:
            by_dt.setdefault(t.dtype, []).append(t)
        return list(by_dt.items())

    @staticmethod
    def _group_pairs_by_dtype(
        a_list: list[torch.Tensor], b_list: list[torch.Tensor]
    ) -> list[tuple[torch.dtype, list[torch.Tensor], list[torch.Tensor]]]:
        by_dt: dict[torch.dtype, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        for a, b in zip(a_list, b_list):
            pair = by_dt.setdefault(a.dtype, ([], []))
            pair[0].append(a)
            pair[1].append(b)
        return [(dt, ps, ds) for dt, (ps, ds) in by_dt.items()]

    # -----------------------------------------------------------------------
    # State initialization & CUDA Graph orchestration
    # -----------------------------------------------------------------------

    def _prepare_step_plan(
        self, spike: bool
    ) -> tuple[list[dict[str, Any]], bool, tuple[Any, ...], list[torch.Tensor],]:
        """Initialize any missing state tensors and build the step execution plan."""
        plan: list[dict[str, Any]] = []
        must_run_eager = False
        sig_parts: list[Any] = []
        all_live_grads: list[torch.Tensor] = []

        for g_idx, group in enumerate(self.param_groups):
            if group["iso"]:
                created_heat = self._refresh_heats(group, spike)
                if created_heat:
                    must_run_eager = True

            muon_entries: list[
                tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], bool, bool]
            ] = []
            adam_entries: list[tuple[torch.Tensor, torch.Tensor, bool]] = []
            bm_start = group["bm_start"]
            bm_shadow = group["bm_shadow"]
            osc_gate = group["osc_gate"]
            magma_damping = group["magma_damping"]

            in_post_bm_set: set[bool] = set()
            seed_bm_set: set[bool] = set()
            adam_steps: set[int] = set()

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("PhysMuon does not support sparse gradients")
                all_live_grads.append(g)
                route, mat_shape = self._resolve_route(p, group)
                st = self.state[p]

                if route == "muon" and mat_shape is not None:
                    if "step" not in st:
                        st["step"] = 0
                        st["momentum"] = torch.zeros_like(g)
                        if bm_start is not None and bm_shadow:
                            st["m_fast"] = torch.zeros_like(g)
                            st["m_slow"] = torch.zeros_like(g)
                            if osc_gate:
                                st["osc"] = torch.zeros(
                                    (), dtype=torch.float32, device=p.device
                                )
                                st["drift"] = torch.zeros(
                                    (), dtype=torch.float32, device=p.device
                                )
                                st["osc_initialized"] = False
                        must_run_eager = True
                    if magma_damping:
                        bs = mat_shape[0]
                        ema = st.get("magma_ema")
                        if ema is None or ema.numel() != bs or ema.device != p.device:
                            st["magma_ema"] = torch.full(
                                (bs,), 0.5, dtype=torch.float32, device=p.device
                            )
                            must_run_eager = True

                    t = int(st["step"])
                    st["step"] = t + 1
                    is_post = bm_start is not None and t >= bm_start
                    seed_now = (
                        bm_start is not None and t >= bm_start and "m_fast" not in st
                    )
                    if seed_now:
                        must_run_eager = True
                    if (
                        bm_start is not None
                        and osc_gate
                        and not st.get("osc_initialized", True)
                    ):
                        must_run_eager = True
                    if (
                        bm_start is not None
                        and t >= bm_start
                        and osc_gate
                        and "osc" not in st
                    ):
                        st["osc"] = torch.zeros(
                            (), dtype=torch.float32, device=p.device
                        )
                        st["drift"] = torch.zeros(
                            (), dtype=torch.float32, device=p.device
                        )
                        st["osc_initialized"] = False
                        must_run_eager = True

                    in_post_bm_set.add(is_post)
                    seed_bm_set.add(seed_now)
                    muon_entries.append((p, g, mat_shape, is_post, seed_now))
                    sig_parts.append(
                        (
                            g_idx,
                            id(p),
                            "m",
                            is_post,
                            "row_heat" in st,
                            group["weight_decay"] != 0.0,
                        )
                    )
                else:
                    acc_dtype = (
                        torch.float64 if p.dtype == torch.float64 else torch.float32
                    )
                    if "step" not in st:
                        st["step"] = 0
                        st["exp_avg"] = torch.zeros_like(p, dtype=acc_dtype)
                        st["exp_avg_sq"] = torch.zeros_like(p, dtype=acc_dtype)
                        must_run_eager = True
                    st["step"] = int(st["step"]) + 1
                    adam_steps.add(int(st["step"]))
                    use_wd = route == "adamw"
                    adam_entries.append((p, g, use_wd))
                    sig_parts.append(
                        (g_idx, id(p), "a", use_wd, group["weight_decay"] != 0.0)
                    )

            if len(in_post_bm_set) > 1 or len(seed_bm_set) > 1:
                must_run_eager = True
            uniform_adam = len(adam_steps) <= 1
            if not uniform_adam:
                must_run_eager = True

            plan.append(
                {
                    "group": group,
                    "muon_entries": muon_entries,
                    "adam_entries": adam_entries,
                    "in_post_bm": next(iter(in_post_bm_set), False),
                    "seed_bm_now": next(iter(seed_bm_set), False),
                    "adam_step": next(iter(adam_steps), 1),
                    "uniform_adam": uniform_adam,
                }
            )

        return plan, must_run_eager, tuple(sig_parts), all_live_grads

    def _compute_group_scalars(
        self, plan: list[dict[str, Any]], spike: bool
    ) -> list[float]:
        """Pack per-group step scalars (lr, wd_mul, slow_alpha, adam_neg_ss, adam_inv_s_bc2)."""
        vals: list[float] = []
        for item in plan:
            group = item["group"]
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            bs = float(group["bm_beta_slow"])
            b1, b2 = group["betas"]
            k = int(item["adam_step"])
            wd_mul = 1.0 - lr * wd
            slow_alpha = 0.0 if spike else (1.0 - bs)
            bc1 = 1.0 - b1**k
            bc2 = 1.0 - b2**k
            neg_ss = -(lr / bc1) if bc1 != 0.0 else 0.0
            inv_s_bc2 = 1.0 / math.sqrt(bc2) if bc2 > 0.0 else 1.0
            vals.extend([lr, wd_mul, slow_alpha, neg_ss, inv_s_bc2])
        return vals

    def _execute_plan(
        self,
        plan: list[dict[str, Any]],
        scalars_gpu: torch.Tensor,
        grad_list: list[torch.Tensor],
    ) -> None:
        grad_offset = 0
        for g_idx, item in enumerate(plan):
            base = 5 * g_idx
            lr_t = scalars_gpu[base]
            wd_mul_t = scalars_gpu[base + 1]
            slow_alpha_t = scalars_gpu[base + 2]
            neg_ss_t = scalars_gpu[base + 3]
            inv_s_bc2_t = scalars_gpu[base + 4]

            m_raw = item["muon_entries"]
            a_raw = item["adam_entries"]
            # Rebind gradients from grad_list (either live grads or static graph buffers)
            m_bound: list[
                tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], bool, bool]
            ] = []
            a_bound: list[tuple[torch.Tensor, torch.Tensor, bool]] = []
            # Note: all_live_grads was collected in group param iteration order;
            # reconstruct exact binding via id(p) map.
            m_map = {
                id(p): (mat_shape, is_post_p, seed_now_p)
                for p, _g, mat_shape, is_post_p, seed_now_p in m_raw
            }
            a_map = {id(p): use_wd for p, _g, use_wd in a_raw}
            for p in item["group"]["params"]:
                pid = id(p)
                if pid in m_map:
                    mat_shape, is_post_p, seed_now_p = m_map[pid]
                    m_bound.append(
                        (p, grad_list[grad_offset], mat_shape, is_post_p, seed_now_p)
                    )
                    grad_offset += 1
                elif pid in a_map:
                    a_bound.append((p, grad_list[grad_offset], a_map[pid]))
                    grad_offset += 1

            self._muon_step_entries(
                entries=m_bound,
                group=item["group"],
                lr_t=lr_t,
                wd_mul_t=wd_mul_t,
                slow_alpha_t=slow_alpha_t,
            )
            self._adamw_step_entries(
                entries=a_bound,
                group=item["group"],
                wd_mul_t=wd_mul_t,
                neg_step_size_t=neg_ss_t,
                inv_sqrt_bc2_t=inv_s_bc2_t,
                uniform_step=item["uniform_adam"],
            )

    def _can_use_cuda_graph(self, grads: list[torch.Tensor]) -> bool:
        if not self.cuda_graph_enabled or not grads:
            return False
        if not torch.cuda.is_available():
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        dev = grads[0].device
        if dev.type != "cuda":
            return False
        return all(g.device == dev for g in grads)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
        grad_norm: float | None = None,
    ) -> torch.Tensor | None:
        """Perform a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            Reevaluates the model and returns the loss.
        grad_norm : float, optional
            Pre-clipping global gradient norm (used only when ``spike_guard > 0``).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        spike, is_nonfinite = self._spike_decision(grad_norm)
        if is_nonfinite and self.skip_nonfinite:
            self._clear_pending_iso()
            self.nonfinite_skipped_steps += 1
            return loss
        plan, must_run_eager, graph_sig, live_grads = self._prepare_step_plan(spike)
        if not live_grads:
            return loss

        dev = live_grads[0].device
        scalar_vals = self._compute_group_scalars(plan, spike)

        if not must_run_eager and self._can_use_cuda_graph(live_grads):
            entry = self._cuda_graphs.get(graph_sig)
            if entry is not None:
                # Fast path: copy live grads & scalars into static buffers + replay!
                torch._foreach_copy_(entry["static_grads"], live_grads)
                entry["scalars_cpu"].copy_(
                    torch.tensor(scalar_vals, dtype=torch.float64)
                )
                entry["scalars_gpu"].copy_(entry["scalars_cpu"], non_blocking=True)
                entry["graph"].replay()
                return loss

            # Capture new CUDA Graph on a side stream (keeps current step exact!)
            try:
                static_grads = [g.detach().clone() for g in live_grads]
                scalars_cpu = torch.tensor(
                    scalar_vals, dtype=torch.float64, pin_memory=True
                )
                scalars_gpu = scalars_cpu.to(device=dev, non_blocking=False)

                # Capture directly on current step's static grads & scalars
                # (1 side-stream warmup? No — warmup would advance state twice!
                # Instead, warm up CUDA allocator on side stream with current
                # eager step on live_grads, then capture on the NEXT step.)
                entry_wait = self._cuda_graphs.get(("warmup", graph_sig))
                if entry_wait is None:
                    # Run this step eagerly to stabilize CUDA caching allocator
                    self._execute_plan(plan, scalars_gpu, live_grads)
                    self._cuda_graphs[("warmup", graph_sig)] = {"warmed": True}
                    return loss

                # Now allocator is warm: capture & replay once for this exact step!
                self._cuda_graphs.pop(("warmup", graph_sig), None)
                graph = torch.cuda.CUDAGraph()
                # Sync current stream -> capture stream -> current stream
                torch.cuda.synchronize(dev)
                with torch.cuda.graph(graph):
                    self._execute_plan(plan, scalars_gpu, static_grads)
                # Note: capturing does NOT execute the ops; replay executes this step!
                graph.replay()
                # Cap cached graphs to prevent unbounded growth across unusual subsets
                if len(self._cuda_graphs) >= 8:
                    self._cuda_graphs.clear()
                self._cuda_graphs[graph_sig] = {
                    "graph": graph,
                    "static_grads": static_grads,
                    "scalars_cpu": scalars_cpu,
                    "scalars_gpu": scalars_gpu,
                }
                return loss
            except RuntimeError:
                # Graceful fallback to eager if capture fails in exotic contexts
                self._cuda_graphs.pop(("warmup", graph_sig), None)

        scalars_gpu = torch.tensor(scalar_vals, dtype=torch.float64, device=dev)
        self._execute_plan(plan, scalars_gpu, live_grads)
        return loss


class Muon(PhysMuon):
    """Exact standard Muon + internal AdamW (all five PhysMuon switches off)."""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns: str = "quintic",
        ns_steps: int | None = None,
        scale: str | float = "shape",
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        muon_mode: str = "2d",
        named_parameters: Iterable[tuple[str, torch.Tensor]] | None = None,
        adam_patterns: Sequence[str] = (),
        cuda_graph: bool = True,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns=ns,
            ns_steps=ns_steps,
            scale=scale,
            bm_start=None,
            iso=False,
            osc_gate=False,
            bm_shadow=False,
            fresh_cap=0.0,
            betas=betas,
            eps=eps,
            spike_guard=0.0,
            muon_mode=muon_mode,
            named_parameters=named_parameters,
            adam_patterns=adam_patterns,
            magma_damping=False,
            cuda_graph=cuda_graph,
        )


class HybridMuon(PhysMuon):
    """Backward-compatible SpinGNN2 v2 HybridMuon (5-step quintic NS + CUDA Graph).

    Preserves the Phase B1 constructor signature (``adam_betas``, ``adam_eps``,
    ``lr_adjust_coeff``, ``magma_damping``, ``named_parameters``) while sharing
    the fast CUDA-Graphed 5-step quintic Newton-Schulz engine of :class:`PhysMuon`.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-12,
        lr_adjust_coeff: float = 0.18,
        muon_mode: str = "2d",
        named_parameters: Iterable[tuple[str, torch.Tensor]] | None = None,
        adam_patterns: Sequence[str] = (),
        magma_damping: bool = True,
        ns: str = "quintic",
        ns_steps: int | None = None,
        cuda_graph: bool = True,
        **physmuon_kwargs: Any,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
            ns=ns,
            ns_steps=ns_steps,
            scale=float(lr_adjust_coeff),
            betas=adam_betas,
            eps=adam_eps,
            muon_mode=muon_mode,
            named_parameters=named_parameters,
            adam_patterns=adam_patterns,
            magma_damping=magma_damping,
            cuda_graph=cuda_graph,
            **physmuon_kwargs,
        )


# ---------------------------------------------------------------------------
# WSD Cosine Learning-Rate Scheduler
# ---------------------------------------------------------------------------


class WSDCosineScheduler(LRScheduler):
    """Warmup-Stable-Decay (WSD) cosine learning-rate scheduler.

    - Steps ``[0, warmup_steps)``: linear warmup from ``min_lr_ratio`` to ``1.0``
    - Steps ``[warmup_steps, decay_start)``: constant plateau at ``1.0``
    - Steps ``[decay_start, total_steps]``: cosine decay to ``min_lr_ratio``
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int | None = None,
        warmup_steps: int = 0,
        stable_steps: int | None = None,
        decay_steps: int | None = None,
        min_lr_ratio: float = 0.01,
        warmup_init_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps is None:
            if stable_steps is None or decay_steps is None:
                raise ValueError(
                    "Provide either total_steps or both stable_steps and decay_steps"
                )
            if stable_steps < 0 or decay_steps < 0:
                raise ValueError("stable_steps and decay_steps must be >= 0")
            total_steps = int(warmup_steps) + int(stable_steps) + int(decay_steps)
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        if warmup_steps > total_steps:
            raise ValueError(
                f"warmup_steps must be in [0, {total_steps}], got {warmup_steps}"
            )
        if decay_steps is None:
            if stable_steps is not None:
                decay_steps = max(0, total_steps - warmup_steps - int(stable_steps))
            else:
                decay_steps = max(1, (total_steps - warmup_steps) // 3)
        if decay_steps < 0 or warmup_steps + decay_steps > total_steps:
            raise ValueError(
                f"warmup_steps + decay_steps must be <= total_steps ({total_steps})"
            )
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
        if not 0.0 <= warmup_init_ratio <= 1.0:
            raise ValueError(
                f"warmup_init_ratio must be in [0, 1], got {warmup_init_ratio}"
            )

        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.decay_steps = int(decay_steps)
        self.decay_start = self.total_steps - self.decay_steps
        self.stable_steps = self.decay_start - self.warmup_steps
        self.min_lr_ratio = float(min_lr_ratio)
        self.warmup_init_ratio = float(warmup_init_ratio)
        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        if step <= 0:
            return self.warmup_init_ratio if self.warmup_steps > 0 else 1.0
        if self.warmup_steps > 0 and step < self.warmup_steps:
            frac = float(step) / float(self.warmup_steps)
            return self.warmup_init_ratio + (1.0 - self.warmup_init_ratio) * frac
        if step < self.decay_start or self.decay_steps == 0:
            return 1.0
        if step >= self.total_steps:
            return self.min_lr_ratio
        progress = float(step - self.decay_start) / float(self.decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def get_lr(self) -> list[float]:
        mult = self._lr_multiplier(self.last_epoch)
        return [base_lr * mult for base_lr in self.base_lrs]
