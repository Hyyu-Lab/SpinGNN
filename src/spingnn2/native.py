"""ctypes bindings for the standalone, versioned SpinGNN2 C ABI."""

import ctypes as ct
import json
import os
from pathlib import Path
from threading import RLock
import weakref

import numpy as np


_Handle = ct.c_void_p
_I = ct.c_int32
_U = ct.c_uint32
_PTR = ct.POINTER
_PATH = Path(__file__).resolve().parents[2]


class _Graph(ct.Structure):
    _fields_ = [
        ("struct_size", _U),
        *((k, _I) for k in ("num_atoms", "num_edges", "num_frames", "num_cells")),
        *(
            (k, _Handle)
            for k in (
                "positions",
                "spins",
                "types",
                "edge_index",
                "batch",
                "cells",
                "edge_cell_shifts",
            )
        ),
    ]


class _Options(ct.Structure):
    _fields_ = [
        ("struct_size", _U),
        ("tile_atoms", _I),
        ("radial_mode", _I),
        ("profile", _I),
        ("cache_device_limit_bytes", ct.c_uint64),
        ("eval_flags", _U),
        ("execution_mode", _I),
        ("execution_profile", _I),
        ("reserved", _U),
        ("math_mode", _U),
    ]


class _Diagnostics(ct.Structure):
    _fields_ = [
        ("struct_size", _U),
        *(
            (k, ct.c_uint64)
            for k in ("workspace_bytes", "cache_host_bytes", "peak_tile_edges")
        ),
        *(
            (k, ct.c_double)
            for k in (
                "geometry_seconds",
                "mlp_seconds",
                "tp_linear_seconds",
                "reverse_seconds",
            )
        ),
        ("cache_device_bytes", ct.c_uint64),
        ("stage1_seconds", ct.c_double),
        ("stage2_seconds", ct.c_double),
        ("stage1_cache_hits", ct.c_uint64),
        ("topology_metadata_bytes", ct.c_uint64),
    ]


def _detect_cuda_arch():
    """Detect active CUDA device compute capability via libcuda.so.1; return None if unavailable."""
    try:
        driver = ct.CDLL("libcuda.so.1")
        if driver.cuInit(0) != 0:
            return None
        dev = ct.c_int(0)
        if driver.cuDeviceGet(ct.byref(dev), 0) != 0:
            return None
        major = ct.c_int(0)
        minor = ct.c_int(0)
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75, MINOR = 76
        if (
            driver.cuDeviceGetAttribute(ct.byref(major), 75, dev) != 0
            or driver.cuDeviceGetAttribute(ct.byref(minor), 76, dev) != 0
        ):
            return None
        return f"sm_{10 * major.value + minor.value}"
    except (OSError, AttributeError):
        return None


def _load_library(path, backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be cpu or cuda")
    path = path or os.getenv("SPINGNN2_LIBRARY")
    if path is None:
        if backend == "cpu":
            path = _PATH / "validation/native/build-cpu/libspingnn2.so"
        else:
            native_root = _PATH / "validation/native"
            env_arch = os.getenv("SPINGNN2_ARCH")
            if env_arch:
                norm = env_arch.replace("_", "")
                path = native_root / f"build-cuda-{norm}" / "libspingnn2.so"
            else:
                candidates = [native_root / "build-cuda-all-gpu" / "libspingnn2.so"]
                detected = _detect_cuda_arch()
                if detected:
                    candidates.append(
                        native_root
                        / f"build-cuda-{detected.replace('_', '')}"
                        / "libspingnn2.so"
                    )
                candidates.append(native_root / "build-cuda-sm120" / "libspingnn2.so")
                path = next((c for c in candidates if c.is_file()), candidates[-1])
    lib = ct.CDLL(os.fspath(path))
    signatures = {
        "abi_version": (_U, []),
        "last_error": (ct.c_char_p, []),
        "backend": (ct.c_char_p, []),
        "model_load": (_I, [ct.c_char_p, _I, _PTR(_Handle)]),
        "model_free": (None, [_Handle]),
        "model_info": (ct.c_char_p, [_Handle]),
        "evaluate": (_I, [_Handle, _PTR(_Graph), _PTR(_Options), _PTR(_Handle)]),
        "stage1_cache_create": (
            _I,
            [_Handle, _PTR(_Graph), _PTR(_Options), _PTR(_Handle)],
        ),
        "evaluate_stage2_cached": (_I, [_Handle, _Handle, _Handle, _PTR(_Handle)]),
        "cache_free": (None, [_Handle]),
        "result_count": (_I, [_Handle]),
        "result_field": (
            _I,
            [
                _Handle,
                _I,
                _PTR(ct.c_char_p),
                _PTR(_Handle),
                _PTR(_I),
                _PTR(_PTR(ct.c_int64)),
            ],
        ),
        "result_diagnostics": (_I, [_Handle, _PTR(_Diagnostics)]),
        "result_free": (None, [_Handle]),
    }
    for name, (ret, args) in signatures.items():
        func = getattr(lib, "sp2_" + name)
        func.restype, func.argtypes = ret, args
    if hasattr(lib, "sp2_model_load_jit_artifact"):
        for name, ret, args in (
            ("sp2_model_load_jit_artifact", _I, [_Handle, ct.c_char_p]),
            ("sp2_model_execution_plan", ct.c_char_p, [_Handle]),
            ("sp2_jit_target_arch", ct.c_char_p, []),
            (
                "sp2_evaluate_stage2_cached_flags",
                _I,
                [_Handle, _Handle, _Handle, _U, _PTR(_Handle)],
            ),
        ):
            func = getattr(lib, name)
            func.restype, func.argtypes = ret, args
    if lib.sp2_abi_version() != 1 or lib.sp2_backend().decode() != backend:
        raise ValueError(f"Incompatible SpinGNN2 ABI or backend in {path}")
    return lib


def _checked(lib, status):
    if status:
        raise ValueError(lib.sp2_last_error().decode(errors="replace"))


def _array(value, dtype, shape, key):
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {arr.shape}")
    if dtype == np.int32 and (
        arr.dtype.kind not in "iu"
        or (arr.size and (arr.min() < 0 or arr.max() > np.iinfo(np.int32).max))
    ):
        raise ValueError(f"{key} must contain nonnegative int32 indices")
    return np.ascontiguousarray(arr, dtype=dtype)


def _prepare(data, dtype):
    pos = np.asarray(data["pos"])
    n = len(pos)
    edges = np.asarray(data["edge_index"])
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    e = edges.shape[1]
    if max(n, e) > np.iinfo(np.int32).max:
        raise ValueError("Graph exceeds C ABI index range")
    arrays = {
        "positions": _array(pos, dtype, (n, 3), "pos"),
        "types": _array(
            np.asarray(data["atom_types"]).reshape(-1), np.int32, (n,), "atom_types"
        ),
        "edge_index": _array(edges, np.int32, (2, e), "edge_index"),
    }
    frames = 1
    if "batch" in data:
        b = _array(data["batch"], np.int32, (n,), "batch")
        arrays["batch"] = b
        frames = int(b.max(initial=0)) + 1
    if frames > max(n, 1) or int(data.get("num_batch", frames)) != frames:
        raise ValueError("Invalid batch count")
    nc = 0
    if "cell" in data:
        cell = np.asarray(data["cell"]).reshape(-1, 3, 3)
        nc = len(cell)
        if nc not in (1, frames) or not np.all(
            np.broadcast_to(data.get("pbc", False), (frames, 3))
        ):
            raise ValueError("Stress requires fully periodic cells")
        arrays["cells"] = np.ascontiguousarray(cell, dtype=dtype)
        arrays["edge_cell_shifts"] = _array(
            data["edge_cell_shift"], dtype, (e, 3), "edge_cell_shift"
        )
    if "spin" in data:
        arrays["spins"] = _array(data["spin"], dtype, (n, 3), "spin")
    graph = _Graph(
        ct.sizeof(_Graph),
        n,
        e,
        frames,
        nc,
        **{k: v.ctypes.data for k, v in arrays.items()},
    )
    return graph, arrays


class LatticeCache:
    """Native fixed-lattice snapshot. Close explicitly to release cache storage."""

    def __init__(self, model, handle, atoms, spin_only=False):
        self._model, self._handle, self._atoms = model, handle, atoms
        self._spin_only = spin_only
        self._finalizer = weakref.finalize(self, model._lib.sp2_cache_free, handle)

    def close(self):
        with self._model._lock:
            self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class NativeModel:
    """Evaluate M1 AtomicDataDict graphs using OpenMP or CUDA without Python training packages."""

    def __init__(
        self,
        manifest,
        *,
        backend="cpu",
        precision=None,
        tile_atoms="plan",
        radial_mode="analytic",
        profile=False,
        cache_device_limit_bytes=None,
        library=None,
        execution_mode="generic",
        execution_profile="capacity",
        artifact=None,
        cache_dir=None,
        allow_tf32=False,
    ):
        if not isinstance(allow_tf32, (bool, np.bool_)):
            raise ValueError("allow_tf32 must be a boolean")
        if cache_device_limit_bytes is not None and (
            not isinstance(cache_device_limit_bytes, (int, np.integer))
            or not 0 <= cache_device_limit_bytes < 2**64 - 1
        ):
            raise ValueError(
                "cache_device_limit_bytes must be None or a nonnegative byte count"
            )
        if execution_mode not in ("auto", "generic", "direct"):
            raise ValueError("execution_mode must be auto, generic or direct")
        if execution_profile not in ("capacity", "speed"):
            raise ValueError("execution_profile must be capacity or speed")
        if execution_mode == "direct" and artifact is None:
            raise ValueError("Direct execution requires a prepared artifact manifest")
        if execution_mode == "generic" and artifact is not None:
            raise ValueError("An artifact requires auto or direct execution")
        self.execution_mode, self.execution_profile = execution_mode, execution_profile
        self.cache_device_limit_bytes = cache_device_limit_bytes
        self._lib = _load_library(library, backend)
        self._handle = _Handle()
        bits = 0
        if precision is not None:
            dtype = np.dtype(precision)
            if dtype not in (np.float32, np.float64):
                raise ValueError("precision must be float32 or float64")
            bits = dtype.itemsize * 8
        _checked(
            self._lib,
            self._lib.sp2_model_load(
                os.fsencode(manifest), bits, ct.byref(self._handle)
            ),
        )
        self._lock = RLock()
        self._finalizer = weakref.finalize(self, self._lib.sp2_model_free, self._handle)
        self.info = json.loads(self._lib.sp2_model_info(self._handle))
        self.backend = backend
        self.dtype = np.dtype(self.info["precision"])
        self.allow_tf32 = bool(allow_tf32)
        self._jit_abi = hasattr(self._lib, "sp2_model_load_jit_artifact")
        try:
            if self.allow_tf32 and (backend != "cuda" or self.dtype != np.float32):
                raise ValueError("TF32 requires a CUDA FP32 model")
            if self.allow_tf32 and not self.info.get("supports_tf32", False):
                raise ValueError(
                    "This native library does not expose explicit TF32 selection"
                )
            if execution_mode != "generic":
                if not self._jit_abi:
                    raise ValueError(
                        "This native library does not support JIT artifacts"
                    )
                if artifact is None:
                    from .jit import prepare_jit_artifact

                    target_arch = None
                    if backend == "cuda":
                        raw_target = self._lib.sp2_jit_target_arch().decode()
                        if "," in raw_target or ";" in raw_target:
                            supported = [
                                a for a in raw_target.replace(";", ",").split(",") if a
                            ]
                            detected = _detect_cuda_arch()
                            if detected not in supported:
                                raise ValueError(
                                    f"Active GPU architecture {detected!r} is not supported by "
                                    f"native library ({raw_target})"
                                )
                            target_arch = detected
                        else:
                            target_arch = raw_target
                    artifact = prepare_jit_artifact(
                        manifest,
                        backend=backend,
                        precision=self.dtype.name,
                        gpu_arch=target_arch,
                        cache_dir=cache_dir,
                        library=self._lib._name,
                    )
                _checked(
                    self._lib,
                    self._lib.sp2_model_load_jit_artifact(
                        self._handle, os.fsencode(artifact)
                    ),
                )
        except BaseException:
            self.close()
            raise
        self.tile_atoms, self.radial_mode, self.profile = (
            tile_atoms,
            radial_mode,
            profile,
        )
        self.diagnostics = {}
        self._last_tile = 512 if execution_profile == "speed" else 256

    @classmethod
    def load(cls, manifest, **options):
        return cls(manifest, **options)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._lock:
            self._finalizer()

    @property
    def execution_plan(self):
        """Inspect the selected path and effective tile width of the latest call."""
        with self._lock:
            if not self._finalizer.alive:
                raise ValueError("Native model is closed")
            if not self._jit_abi:
                return {
                    "selected_mode": "generic",
                    "mode": "generic",
                    "execution_profile": self.execution_profile,
                    "artifact_sha256": "",
                    "fallback_count": 0,
                    "math_mode": "strict_fp32" if self.dtype == np.float32 else "fp64",
                    "tile_atoms": self._last_tile,
                }
            plan = self._lib.sp2_model_execution_plan(self._handle)
            if not plan:
                _checked(self._lib, -1)
            return json.loads(plan)

    def _opts(self, tile_atoms, radial_mode, n):
        size = n if tile_atoms is None else tile_atoms
        if size == "plan":
            size = 512 if self.execution_profile == "speed" else 256
        if (
            not isinstance(size, (int, np.integer))
            or not 1 <= size <= np.iinfo(np.int32).max
        ):
            raise ValueError("tile_atoms must be a positive int32")
        mode = self.radial_mode if radial_mode is None else radial_mode
        if mode not in ("analytic", "spline"):
            raise ValueError("radial_mode must be analytic or spline")
        limit = self.cache_device_limit_bytes
        self._last_tile = min(n, size)
        return _Options(
            ct.sizeof(_Options),
            0 if tile_atoms == "plan" and self._jit_abi else size,
            int(mode == "spline"),
            self.profile,
            0 if limit is None else (limit or 2**64 - 1),
            0,
            {"auto": 0, "generic": 1, "direct": 2}[self.execution_mode],
            int(self.execution_profile == "speed"),
            0,
            int(self.allow_tf32),
        )

    def _run(self, data, tile_atoms, radial_mode, cache, spin_only=False):
        if not self._finalizer.alive:
            raise ValueError("Native model is closed")
        graph, keepalive = _prepare(data, self.dtype)
        opts = self._opts(
            self.tile_atoms if tile_atoms is ... else tile_atoms,
            radial_mode,
            graph.num_atoms,
        )
        opts.eval_flags = int(bool(spin_only))
        out = _Handle()
        func = self._lib.sp2_stage1_cache_create if cache else self._lib.sp2_evaluate
        _checked(
            self._lib,
            func(self._handle, ct.byref(graph), ct.byref(opts), ct.byref(out)),
        )
        return (
            LatticeCache(self, out, graph.num_atoms, spin_only)
            if cache
            else self._read(out)
        )

    def evaluate(self, data, *, tile_atoms=..., radial_mode=None):
        with self._lock:
            return self._run(data, tile_atoms, radial_mode, False)

    def evaluate_stage1_lattice(
        self, data, *, tile_atoms=..., radial_mode=None, spin_only=False
    ):
        """Retain only Stage-2 forward inputs when spin_only is set."""
        with self._lock:
            if spin_only and not self._jit_abi:
                raise ValueError(
                    "Frozen-lattice evaluation requires the M3 native library"
                )
            return self._run(data, tile_atoms, radial_mode, True, spin_only)

    def evaluate_stage2_spin_cached(self, cache, spin, *, spin_only=False):
        with self._lock:
            return self._cached(cache, spin, spin_only)

    def _cached(self, cache, spin, spin_only):
        if (
            not self._finalizer.alive
            or cache._model is not self
            or not cache._finalizer.alive
        ):
            raise ValueError("Invalid cache or closed model")
        s = _array(spin, self.dtype, (cache._atoms, 3), "spin")
        out = _Handle()
        if spin_only or cache._spin_only:
            if not self._jit_abi:
                raise ValueError(
                    "Frozen-lattice evaluation requires the M3 native library"
                )
            status = self._lib.sp2_evaluate_stage2_cached_flags(
                self._handle,
                cache._handle,
                s.ctypes.data,
                int(bool(spin_only)),
                ct.byref(out),
            )
        else:
            status = self._lib.sp2_evaluate_stage2_cached(
                self._handle, cache._handle, s.ctypes.data, ct.byref(out)
            )
        _checked(self._lib, status)
        return self._read(out)

    def _read(self, out):
        lib = self._lib
        result = {}
        scalar = ct.c_float if self.dtype == np.float32 else ct.c_double
        try:
            for i in range(lib.sp2_result_count(out)):
                name, addr, rank, dims = (
                    ct.c_char_p(),
                    _Handle(),
                    _I(),
                    _PTR(ct.c_int64)(),
                )
                _checked(
                    lib,
                    lib.sp2_result_field(
                        out,
                        i,
                        ct.byref(name),
                        ct.byref(addr),
                        ct.byref(rank),
                        ct.byref(dims),
                    ),
                )
                shape = tuple(dims[: rank.value])
                if 0 in shape:
                    value = np.empty(shape, dtype=self.dtype)
                else:
                    value = np.ctypeslib.as_array(
                        ct.cast(addr, _PTR(scalar)), shape=shape
                    ).copy()
                result[name.value.decode()] = value
            diag = _Diagnostics(ct.sizeof(_Diagnostics))
            _checked(lib, lib.sp2_result_diagnostics(out, ct.byref(diag)))
            self.diagnostics = {key: getattr(diag, key) for key, _ in diag._fields_[1:]}
        finally:
            lib.sp2_result_free(out)
        return result
