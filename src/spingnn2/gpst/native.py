"""Optional fused C99 GPST edge kernels; degrades transparently to pure NumPy."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_LIB = None


def _load_library():
    if os.environ.get("SPINGNN2_GPST_NUMPY_ONLY") == "1":
        return None
    if os.environ.get("TSPIN_NUMPY_ONLY") == "1":
        return None
    so_path = Path(__file__).with_name("_edge_kernel.so")
    if not so_path.is_file():
        return None
    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError:
        return None
    lib.gpst_edges_exact.argtypes = [ctypes.c_int64] * 3 + [ctypes.c_void_p] * 16
    lib.gpst_edges_exact.restype = ctypes.c_int
    lib.gpst_edge_moments_fused.argtypes = [ctypes.c_int64] * 4 + [ctypes.c_void_p] * 15
    lib.gpst_edge_moments_fused.restype = ctypes.c_int
    return lib


_LIB = _load_library()


def reload_native_kernel() -> bool:
    """Reload ``_edge_kernel.so`` (useful after compiling at runtime or toggling env)."""
    global _LIB
    _LIB = _load_library()
    return _LIB is not None


def has_native_kernel() -> bool:
    """Return True when the compiled C99 GPST edge kernel is active."""
    return _LIB is not None


def edge_moments_fused(s: np.ndarray, ft: np.ndarray, graph: dict):
    """Fused shell-reducing C99 kernel returning reduced arrays directly, or None."""
    if _LIB is None:
        return None
    if s.shape[-1] != 3:
        raise ValueError("BAD_INPUT")
    t, ns, _ = s.shape
    edges = len(graph["i"])
    groups = len(graph["shell_cuts"]) + 1

    s_c = np.ascontiguousarray(s, dtype=np.float64)
    ft_c = np.ascontiguousarray(ft, dtype=np.float64)
    left = np.ascontiguousarray(graph["i"], dtype=np.int64)
    right = np.ascontiguousarray(graph["j"], dtype=np.int64)
    shell = np.ascontiguousarray(graph["shell"], dtype=np.int64)
    w = np.ascontiguousarray(graph["w"], dtype=np.float64)
    u = np.ascontiguousarray(graph["direction"], dtype=np.float64)

    if (
        left.shape != (edges,)
        or right.shape != (edges,)
        or shell.shape != (edges,)
        or w.shape != (edges,)
        or u.shape != (edges, 3)
        or (
            edges > 0
            and (
                np.any(left < 0)
                or np.any(left >= ns)
                or np.any(right < 0)
                or np.any(right >= ns)
            )
        )
    ):
        raise ValueError("ID_MISMATCH")

    shell_n = np.empty((t, 4, groups), dtype=np.float64)
    shell_d = np.empty((t, 4, groups), dtype=np.float64)
    zonal_n = np.empty(t, dtype=np.float64)
    zonal_d = np.empty(t, dtype=np.float64)
    chir_n = np.empty(t, dtype=np.float64)
    chir_d = np.empty(t, dtype=np.float64)
    shell_ss = np.empty((t, groups, 3, 3), dtype=np.float64)
    shell_an = np.empty((t, groups, 3, 3), dtype=np.float64)

    code = _LIB.gpst_edge_moments_fused(
        t,
        ns,
        edges,
        groups,
        s_c.ctypes.data,
        ft_c.ctypes.data,
        left.ctypes.data,
        right.ctypes.data,
        shell.ctypes.data,
        w.ctypes.data,
        u.ctypes.data,
        shell_n.ctypes.data,
        shell_d.ctypes.data,
        zonal_n.ctypes.data,
        zonal_d.ctypes.data,
        chir_n.ctypes.data,
        chir_d.ctypes.data,
        shell_ss.ctypes.data,
        shell_an.ctypes.data,
    )
    if code:
        raise ValueError("BAD_INPUT")
    return shell_n, shell_d, zonal_n, zonal_d, chir_n, chir_d, shell_ss, shell_an
