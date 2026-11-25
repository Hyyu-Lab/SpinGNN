"""Online GPST helper & convenience facade over ``spingnn2.gpst``.

Provides LAMMPS metal-unit physical constants, online ``compute spin/gpst``
vector decoding, single-frame oracle evaluation, and lightweight pooling, while
re-exporting the full cross-fitted GPST toolkit from ``spingnn2.gpst``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .gpst import (
    CHANNELS,
    STATUS_CODES,
    FrameError,
    FrameInput,
    analyze,
    build_graph,
    compute_moments,
    fieller,
    from_lammps_fm,
    geyer,
    has_native_kernel,
    load_frame_npz,
    pooled_ratio,
    read_dump,
    validate_frame,
)

try:
    from scipy.stats import t as _student_t
except Exception:  # pragma: no cover - optional fallback if scipy absent
    _student_t = None

# LAMMPS metal-unit physical constants (matching update.cpp force->boltz & hplanck/2pi).
LAMMPS_METAL_KB: float = 8.617343e-5
LAMMPS_METAL_HBAR: float = 4.135667403e-3 / (2.0 * math.pi)
KB_EV_K: float = LAMMPS_METAL_KB
HBAR_EV_PS: float = LAMMPS_METAL_HBAR

_PAIR_ORDERS: tuple[int, ...] = (1, 2, 4, 8)


def _legendre_and_deriv(
    x: np.ndarray, max_l: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xc = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    p = [np.ones_like(xc), xc.copy()]
    dp = [np.zeros_like(xc), np.ones_like(xc)]
    for k in range(1, max_l):
        a = float(2 * k + 1)
        b = float(k)
        inv = 1.0 / float(k + 1)
        p.append((a * xc * p[-1] - b * p[-2]) * inv)
        dp.append((a * (p[k] + xc * dp[-1]) - b * dp[-2]) * inv)
    return p, dp


def _periodic_pairs(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return all directed neighbor pairs (i, j) with 0 < |r_j - r_i| < cutoff."""
    pos = np.asarray(positions, dtype=np.float64)
    box = np.asarray(cell, dtype=np.float64)
    periodic = np.asarray(pbc, dtype=bool)
    n = pos.shape[0]
    if n == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return (
            empty_i,
            empty_i,
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    inv_box = np.linalg.inv(box)
    frac = pos @ inv_box
    wrapped = (frac - np.floor(frac) * periodic) @ box

    heights = 1.0 / np.linalg.norm(inv_box, axis=0)
    reaches = [
        int(math.ceil(cutoff / heights[a])) if periodic[a] else 0 for a in range(3)
    ]
    shifts_frac = np.array(
        [
            (sx, sy, sz)
            for sx in range(-reaches[0], reaches[0] + 1)
            for sy in range(-reaches[1], reaches[1] + 1)
            for sz in range(-reaches[2], reaches[2] + 1)
        ],
        dtype=np.float64,
    )
    shifts_cart = shifts_frac @ box

    src_list, dst_list, disp_list, rsq_list = [], [], [], []
    cutsq = cutoff * cutoff
    for shift in shifts_cart:
        diff = wrapped[None, :, :] + shift[None, None, :] - wrapped[:, None, :]
        rsq = np.sum(diff * diff, axis=-1)
        hit_i, hit_j = np.nonzero((rsq > 1.0e-24) & (rsq < cutsq))
        if hit_i.size:
            src_list.append(hit_i)
            dst_list.append(hit_j)
            disp_list.append(diff[hit_i, hit_j])
            rsq_list.append(rsq[hit_i, hit_j])

    if not src_list:
        empty_i = np.empty(0, dtype=np.int64)
        return (
            empty_i,
            empty_i,
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    return (
        np.concatenate(src_list),
        np.concatenate(dst_list),
        np.concatenate(disp_list, axis=0),
        np.concatenate(rsq_list),
    )


def compute_gpst_moments(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool],
    spins: np.ndarray,
    spin_forces: np.ndarray,
    cutoff: float,
    *,
    magnetic_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute exact 7-channel GPST (N, D) sufficient moments for a single frame."""
    pos = np.asarray(positions, dtype=np.float64)
    box = np.asarray(cell, dtype=np.float64)
    periodic = np.asarray(pbc, dtype=bool)
    s_all = np.asarray(spins, dtype=np.float64)
    f_all = np.asarray(spin_forces, dtype=np.float64)

    if cutoff <= 0.0 or not math.isfinite(cutoff):
        raise ValueError("cutoff must be a finite positive scalar")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if box.shape != (3, 3) or periodic.shape != (3,):
        raise ValueError("cell must be (3, 3) and pbc must be (3,)")

    if magnetic_mask is not None:
        mask = np.asarray(magnetic_mask, dtype=bool)
        if mask.shape != (pos.shape[0],):
            raise ValueError("magnetic_mask must match positions length")
        pos = pos[mask]
        s = s_all[mask] if s_all.shape[0] == mask.shape[0] else s_all
        f = f_all[mask] if f_all.shape[0] == mask.shape[0] else f_all
    else:
        s = s_all
        f = f_all

    if s.shape != pos.shape or f.shape != pos.shape:
        raise ValueError(
            "magnetic positions, spins, and spin_forces must share shape (N_mag, 3)"
        )

    n_mag = pos.shape[0]
    numer = np.zeros(len(CHANNELS), dtype=np.float64)
    denom = np.zeros(len(CHANNELS), dtype=np.float64)
    if n_mag == 0:
        return {
            "channels": CHANNELS,
            "N": numer,
            "D": denom,
            "n_mag": 0,
            "magnetization": np.zeros(4, dtype=np.float64),
        }

    norms_sq = np.sum(s * s, axis=-1)
    if np.max(np.abs(norms_sq - 1.0)) > 1.0e-8:
        raise ValueError("Magnetic spins must be unit vectors (|s|^2 == 1)")

    ft = f - s * np.sum(s * f, axis=-1, keepdims=True)
    m_vec = np.mean(s, axis=0)
    magnetization = np.array(
        [m_vec[0], m_vec[1], m_vec[2], float(np.linalg.norm(m_vec))]
    )

    src, dst, disp, rsq = _periodic_pairs(pos, box, periodic, float(cutoff))
    if src.size:
        r = np.sqrt(rsq)
        u = disp / r[:, None]
        w = (1.0 - rsq / (cutoff * cutoff)) ** 3

        si = s[src]
        sj = s[dst]
        fti = ft[src]

        xdot = np.sum(si * sj, axis=-1)
        sj_dot_fti = np.sum(sj * fti, axis=-1)
        p, dp = _legendre_and_deriv(xdot, 8)
        for idx, ell in enumerate(_PAIR_ORDERS):
            numer[idx] = float(np.sum(w * dp[ell] * sj_dot_fti))
            denom[idx] = float(ell * (ell + 1) * np.sum(w * p[ell]))

        outer = w[:, None, None] * u[:, :, None] * u[:, None, :]
        q_mat = np.zeros((n_mag, 3, 3), dtype=np.float64)
        np.add.at(q_mat, src, outer)
        tr_q = np.zeros(n_mag, dtype=np.float64)
        np.add.at(tr_q, src, w)
        qs = np.einsum("nab,nb->na", q_mat, s)
        numer[4] = float(np.sum(qs * ft))
        denom[4] = float(np.sum(3.0 * np.sum(qs * s, axis=-1) - tr_q))

        zdot = np.sum(si * u, axis=-1)
        fti_dot_u = np.sum(fti * u, axis=-1)
        pz, dpz = _legendre_and_deriv(zdot, 4)
        numer[5] = float(np.sum(w * dpz[4] * fti_dot_u))
        denom[5] = float(20.0 * np.sum(w * pz[4]))

        numer[6] = float(np.sum(w * np.sum(fti * np.cross(sj, u), axis=-1)))
        denom[6] = float(2.0 * np.sum(w * np.sum(u * np.cross(si, sj), axis=-1)))

    return {
        "channels": CHANNELS,
        "N": numer,
        "D": denom,
        "n_mag": int(n_mag),
        "magnetization": magnetization,
    }


def _critical_t(df: int, alpha: float) -> float:
    if _student_t is not None and df >= 1:
        return float(_student_t.ppf(1.0 - 0.5 * alpha, df))
    return 1.959963984540054


def _fieller_interval(
    mean_n: float,
    mean_d: float,
    cov: np.ndarray,
    df: int,
    *,
    alpha: float,
    denom_scale: float,
) -> dict[str, Any]:
    q = _critical_t(df, alpha)
    q2 = q * q
    var_n = max(0.0, float(cov[0, 0]))
    var_d = max(0.0, float(cov[1, 1]))
    cov_nd = float(cov[0, 1])

    zero_gate_d = 1.0e-12 * max(denom_scale, 1.0e-30)
    if abs(mean_d) <= zero_gate_d:
        status = (
            "ZERO_SIGNAL"
            if abs(mean_n) <= 1.0e-13 * max(denom_scale, 1.0e-30)
            else "WEAK_DENOMINATOR"
        )
        return {
            "status": status,
            "point": None,
            "se": None,
            "ci": None,
            "mean_N": mean_n,
            "mean_D": mean_d,
        }

    point = mean_n / mean_d
    if abs(mean_n) <= 1.0e-13 * max(abs(mean_d), denom_scale, 1.0e-30):
        return {
            "status": "ZERO_SIGNAL",
            "point": 0.0,
            "se": math.sqrt(max(0.0, var_n)) / abs(mean_d),
            "ci": None,
            "mean_N": mean_n,
            "mean_D": mean_d,
        }

    var_ratio = max(
        0.0,
        (var_n - 2.0 * point * cov_nd + point * point * var_d) / (mean_d * mean_d),
    )
    se = math.sqrt(var_ratio)

    aa = mean_d * mean_d - q2 * var_d
    bb = -2.0 * mean_n * mean_d + 2.0 * q2 * cov_nd
    cc = mean_n * mean_n - q2 * var_n

    if abs(mean_d) <= q * math.sqrt(var_d) or aa <= 0.0:
        return {
            "status": "WEAK_DENOMINATOR",
            "point": point,
            "se": se,
            "ci": None,
            "mean_N": mean_n,
            "mean_D": mean_d,
        }

    disc = bb * bb - 4.0 * aa * cc
    if disc < 0.0:
        return {
            "status": "UNBOUNDED_CI",
            "point": point,
            "se": se,
            "ci": None,
            "mean_N": mean_n,
            "mean_D": mean_d,
        }

    root_s = math.sqrt(max(0.0, disc))
    lo = (-bb - root_s) / (2.0 * aa)
    hi = (-bb + root_s) / (2.0 * aa)
    if lo > hi:
        lo, hi = hi, lo

    status = "NEGATIVE_POINT" if point < 0.0 else "IDENTIFIED"
    return {
        "status": status,
        "point": point,
        "se": se,
        "ci": (float(lo), float(hi)),
        "mean_N": mean_n,
        "mean_D": mean_d,
    }


def pooled_gpst(
    numerators: np.ndarray | Sequence[np.ndarray],
    denominators: np.ndarray | Sequence[np.ndarray],
    *,
    kb: float = KB_EV_K,
    alpha: float = 0.05,
    n_blocks: int = 10,
    channel_names: Sequence[str] = CHANNELS,
) -> dict[str, dict[str, Any]]:
    """Pool GPST (N, D) trajectories into temperatures with Fieller intervals."""
    if not math.isfinite(kb) or kb <= 0.0:
        raise ValueError("kb must be finite and positive")

    if isinstance(numerators, np.ndarray) and isinstance(denominators, np.ndarray):
        n_arr = np.atleast_2d(np.asarray(numerators, dtype=np.float64))
        d_arr = np.atleast_2d(np.asarray(denominators, dtype=np.float64))
        if n_arr.shape != d_arr.shape:
            raise ValueError("numerators and denominators must have identical shapes")
        k = min(max(2, int(n_blocks)), n_arr.shape[0])
        if n_arr.shape[0] < 2:
            n_blocks_arr = n_arr
            d_blocks_arr = d_arr
            df = 1
        else:
            n_blocks_arr = np.array(
                [blk.mean(axis=0) for blk in np.array_split(n_arr, k)]
            )
            d_blocks_arr = np.array(
                [blk.mean(axis=0) for blk in np.array_split(d_arr, k)]
            )
            df = k - 1
        denom_scales = np.mean(np.abs(d_arr), axis=0)
    else:
        n_chains = [np.atleast_2d(np.asarray(x, dtype=np.float64)) for x in numerators]
        d_chains = [
            np.atleast_2d(np.asarray(x, dtype=np.float64)) for x in denominators
        ]
        if len(n_chains) != len(d_chains) or len(n_chains) < 2:
            raise ValueError("Multi-chain pooling requires at least 2 matching chains")
        for nc, dc in zip(n_chains, d_chains):
            if nc.shape != dc.shape:
                raise ValueError(
                    "Each chain's numerator and denominator shapes must match"
                )
        n_blocks_arr = np.array([nc.mean(axis=0) for nc in n_chains])
        d_blocks_arr = np.array([dc.mean(axis=0) for dc in d_chains])
        df = len(n_chains) - 1
        denom_scales = np.mean([np.mean(np.abs(dc), axis=0) for dc in d_chains], axis=0)

    m = n_blocks_arr.shape[0]
    n_channels = n_blocks_arr.shape[1]
    names = list(channel_names)
    if len(names) != n_channels:
        names = [f"channel_{c}" for c in range(n_channels)]

    results: dict[str, dict[str, Any]] = {}
    for c, name in enumerate(names):
        mean_n = float(np.mean(n_blocks_arr[:, c]))
        mean_d = float(np.mean(d_blocks_arr[:, c]))
        if m >= 2:
            paired = np.column_stack([n_blocks_arr[:, c], d_blocks_arr[:, c]])
            cov = np.cov(paired, rowvar=False, ddof=1) / float(m)
        else:
            cov = np.zeros((2, 2), dtype=np.float64)

        raw = _fieller_interval(
            mean_n,
            mean_d,
            cov,
            df,
            alpha=alpha,
            denom_scale=float(denom_scales[c]),
        )
        point_k = None if raw["point"] is None else raw["point"] / kb
        se_k = None if raw["se"] is None else raw["se"] / kb
        ci_k = None if raw["ci"] is None else (raw["ci"][0] / kb, raw["ci"][1] / kb)
        results[name] = {
            "status": raw["status"],
            "temperature_K": point_k,
            "se_K": se_k,
            "ci_K": ci_k,
            "mean_N": raw["mean_N"],
            "mean_D": raw["mean_D"],
        }
    return results


def parse_gpst_vector(vector: Mapping[str, Any] | Sequence[float]) -> dict[str, Any]:
    """Decode the 19-element vector emitted by LAMMPS ``compute spin/gpst``."""
    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape[-1] != 19:
        raise ValueError(
            f"Expected 19-element compute spin/gpst vector, got shape {arr.shape}"
        )
    numer = arr[..., 0:14:2]
    denom = arr[..., 1:14:2]
    return {
        "channels": CHANNELS,
        "N": numer,
        "D": denom,
        "n_mag": arr[..., 14],
        "magnetization": arr[..., 15:19],
    }


__all__ = [
    "CHANNELS",
    "HBAR_EV_PS",
    "KB_EV_K",
    "LAMMPS_METAL_HBAR",
    "LAMMPS_METAL_KB",
    "STATUS_CODES",
    "FrameError",
    "FrameInput",
    "analyze",
    "build_graph",
    "compute_gpst_moments",
    "compute_moments",
    "fieller",
    "from_lammps_fm",
    "geyer",
    "has_native_kernel",
    "load_frame_npz",
    "parse_gpst_vector",
    "pooled_gpst",
    "pooled_ratio",
    "read_dump",
    "validate_frame",
]
