import contextlib
import os
import threading
import warnings

import torch

import e3nn
import e3nn.util.jit

from nequip.data import register_fields
from .misc import dtype_from_name
from .auto_init import instantiate
from .test import set_irreps_debug


# for multiprocessing, we need to keep track of our latest global options so
# that we can reload/reset them in worker processes. While we could be more careful here,
# to keep only relevant keys, configs should have only small values (no big objects)
# and those should have references elsewhere anyway, so keeping references here is fine.
_latest_global_config = {}
_GLOBAL_OPTIONS_LOCK = threading.RLock()


def _get_latest_global_options() -> dict:
    """Get the config used latest to ``_set_global_options``.

    This is useful for getting worker processes into the same state as the parent.
    """
    global _latest_global_config
    return _latest_global_config


def _set_global_options(config, warn_on_override: bool = False) -> None:
    with _GLOBAL_OPTIONS_LOCK:
        _apply_global_options(config, warn_on_override=warn_on_override)


# A module with this class or instance attribute set to ``True`` keeps the
# dtypes of its own parameters and buffers, and of all its submodules, when
# the model is cast to ``model_dtype`` (or ``default_dtype``) at build,
# restart and evaluation.
KEEP_DTYPE_ATTR = "_spingnn_keep_dtype"


def model_dtype_from_config(config) -> torch.dtype:
    """dtype of the model: ``model_dtype`` if set, else ``default_dtype``.

    Datasets, targets and losses always use ``default_dtype``; the model
    receives ``default_dtype`` inputs, so modules of a model with a different
    ``model_dtype`` must cast their inputs where needed.
    """
    name = config.get("model_dtype", None)
    key = "model_dtype"
    if name is None:
        name = config.get("default_dtype", None)
        key = "default_dtype"
    if name is None:
        return torch.get_default_dtype()
    try:
        return dtype_from_name(name)
    except KeyError:
        raise ValueError(
            f"{key} must be 'float32' or 'float64'; got {name!r}"
        ) from None


def check_deployable_model_dtype(config) -> None:
    """Reject deployment when ``model_dtype`` differs from ``default_dtype``.

    Deployed models (TorchScript/PT2 and the deployed-model ASE calculators)
    record and expect one dtype for the model and its inputs; a mixed
    model/data dtype is not supported there yet.
    """
    model_dtype = model_dtype_from_config(config)
    default_name = config.get("default_dtype", None)
    data_dtype = (
        torch.get_default_dtype()
        if default_name is None
        else dtype_from_name(default_name)
    )
    if model_dtype != data_dtype:
        raise NotImplementedError(
            f"deployment of a model with model_dtype {model_dtype} different "
            f"from default_dtype {data_dtype} (mixed model/data dtype) is not "
            "supported yet; evaluate it from the training session with "
            "nequip-evaluate, or train with model_dtype equal to default_dtype "
            "to deploy"
        )


def _keeps_dtype(module: torch.nn.Module) -> bool:
    return getattr(module, KEEP_DTYPE_ATTR, False) is True


def cast_model_dtype(
    model: torch.nn.Module, dtype: torch.dtype, device=None
) -> torch.nn.Module:
    """``model.to(device=device, dtype=dtype)`` honoring the keep-dtype marker.

    Modules marked with ``_spingnn_keep_dtype = True`` (see
    ``KEEP_DTYPE_ATTR``), including everything below them, keep their dtypes
    but are still moved to ``device``. Without marked modules this is exactly
    ``model.to(device=device, dtype=dtype)``.
    """
    if not any(_keeps_dtype(module) for module in model.modules()):
        return model.to(device=device, dtype=dtype)
    if device is not None:
        model.to(device=device)

    def convert(tensor):
        # same rule as ``torch.nn.Module.to``
        if tensor.is_floating_point() or tensor.is_complex():
            return tensor.to(dtype=dtype)
        return tensor

    def visit(module):
        if _keeps_dtype(module):
            return
        for child in module.children():
            visit(child)
        module._apply(convert, recurse=False)

    visit(model)
    return model


def prepare_deterministic_cuda(config):
    """Set/validate cuBLAS workspace before CUDA initialization for explicit mode.

    Call before set_device, hardware probes or process-group initialization.
    An already initialized process must have a supported workspace configured.
    The environment setting persists for the lifetime of its CUDA context.
    """
    enabled = config.get("deterministic_algorithms", False)
    if type(enabled) is not bool:
        raise ValueError("deterministic_algorithms must be a boolean")
    if not enabled:
        return
    with _GLOBAL_OPTIONS_LOCK:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace in (":4096:8", ":16:8"):
            return
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization for deterministic_algorithms"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _read_float32_matmul_precision():
    """Read the global float32 matmul precision, tolerating per-backend state.

    ``torch.get_float32_matmul_precision()`` asks for a single global answer. On
    torch >= 2.11 it raises ``RuntimeError`` ("checking the matmul precision
    without a specific backend name ... mix of the legacy and new APIs") once any
    per-backend fp32 precision has been configured -- which is exactly what the
    ``allow_tf32`` switch below does via ``torch.backends.cuda.matmul.allow_tf32``.

    The crash is therefore ordering-dependent and CUDA-only: within one call the
    new-API read happens *before* the legacy write, so the FIRST call succeeds and
    every SUBSEQUENT call raises. ``torch.cuda.is_available()`` gates the legacy
    write, so CPU CI can never reach this state. Observed on an H100 (torch
    2.11.0+cu128) as 13 ``ERROR at setup`` failures in the shared ``model_paths``
    fixture, which builds several models in a loop.

    The value is only used to decide whether to emit an override warning, so a
    failed read degrades to "unknown" rather than breaking global setup.
    """
    try:
        return torch.get_float32_matmul_precision()
    except RuntimeError:
        return None


def _read_legacy_allow_tf32():
    """Read the legacy TF32 backend flags, tolerating mixed-API state.

    Same hazard as :func:`_read_float32_matmul_precision`: these legacy getters
    can also refuse to answer once per-backend precision has been set. Each
    backend is read INDEPENDENTLY and degrades to ``None`` on its own -- the
    getters can disagree (only one backend's precision may have been touched),
    and collapsing both into one unknown used to discard a readable value that
    the scoped context then failed to restore (RV4A-M1). Returns a
    ``(matmul, cudnn)`` tuple whose elements are each ``bool`` or ``None``.
    """

    def _read_one(getter):
        try:
            return getter()
        except RuntimeError:
            return None

    return (
        _read_one(lambda: torch.backends.cuda.matmul.allow_tf32),
        _read_one(lambda: torch.backends.cudnn.allow_tf32),
    )


def _apply_global_options(config, warn_on_override: bool = False) -> None:
    """Configure global options of libraries like `torch` and `e3nn` based on `config`.

    Args:
        warn_on_override: if True, will try to warn if new options are inconsistant with previously set ones.
    """
    if (
        "deterministic_algorithms" in config
        and type(config["deterministic_algorithms"]) is not bool
    ):
        raise ValueError("deterministic_algorithms must be a boolean")
    # update these options into the latest global config.
    global _latest_global_config
    _latest_global_config.update(dict(config))
    # Set the float32 matmul algorithm preference first. The explicit CUDA
    # allow_tf32 switch below is applied afterwards and can veto TF32.
    k = "float32_matmul_precision"
    if k in config:
        precision = config[k]
        allowed = ("highest", "high", "medium")
        if precision not in allowed:
            raise ValueError(f"{k} must be one of {allowed}; got {precision!r}")
        old_precision = _read_float32_matmul_precision()
        if (
            warn_on_override
            and old_precision is not None
            and old_precision != precision
        ):
            warnings.warn(
                f"Setting the GLOBAL value for {k} to `{precision}` "
                f"which is different from `{old_precision}`"
            )
        torch.set_float32_matmul_precision(precision)

    # Set TF32 support
    # See https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available() and "allow_tf32" in config:
        allow_tf32 = config["allow_tf32"]
        current_matmul, current_cudnn = _read_legacy_allow_tf32()
        # An unreadable element means that backend refused to report; write
        # anyway. Warn only when BOTH old values were readable and differ.
        if current_matmul is not allow_tf32 or current_cudnn is not allow_tf32:
            if (
                warn_on_override
                and current_matmul is not None
                and current_cudnn is not None
            ):
                warnings.warn(
                    f"Setting the GLOBAL value for allow_tf32 to {allow_tf32} "
                    "which is different than the previous backend values"
                )
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32

    if int(torch.__version__.split(".")[1]) >= 11:
        # PyTorch >= 1.11
        k = "_jit_fusion_strategy"
        if k in config:
            new_strat = config.get(k)
            old_strat = torch.jit.set_fusion_strategy(new_strat)
            if warn_on_override and old_strat != new_strat:
                warnings.warn(
                    f"Setting the GLOBAL value for jit fusion strategy to `{new_strat}` which is different than the previous value of `{old_strat}`"
                )
    else:
        # For avoiding 20 steps of painfully slow JIT recompilation
        # See https://github.com/pytorch/pytorch/issues/52286
        k = "_jit_bailout_depth"
        if k in config:
            new_depth = config[k]
            old_depth = torch._C._jit_set_bailout_depth(new_depth)
            if warn_on_override and old_depth != new_depth:
                warnings.warn(
                    f"Setting the GLOBAL value for jit bailout depth to `{new_depth}` which is different than the previous value of `{old_depth}`"
                )

    # TODO: warn_on_override for the rest here?
    if config.get("model_debug_mode", False):
        set_irreps_debug(enabled=True)
    if "default_dtype" in config:
        torch.set_default_dtype(dtype_from_name(config["default_dtype"]))
    if config.get("model_dtype", None) is not None:
        # validate early; the model itself is cast where it is built/loaded
        model_dtype_from_config(config)
    if "deterministic_algorithms" in config:
        deterministic = config["deterministic_algorithms"]
        torch.use_deterministic_algorithms(deterministic, warn_only=False)
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            torch.backends.cudnn.benchmark = False
    if config.get("grad_anomaly_mode", False):
        torch.autograd.set_detect_anomaly(True)

    e3nn.set_optimization_defaults(**config.get("e3nn_optimization_defaults", {}))

    # Register fields:
    instantiate(register_fields, all_args=config)
    return


@contextlib.contextmanager
def global_options_context(config=None, warn_on_override: bool = False):
    """Hold precision globals for one operation and restore them afterwards.

    The save/restore pair uses the same tolerant readers as
    :func:`_apply_global_options`: on CUDA with torch >= 2.11 the global
    matmul-precision getter refuses to answer once per-backend precision has been
    configured. A knob whose previous value could not be read is left untouched on
    exit rather than being restored to a fabricated value -- restoring a guessed
    precision would be a silent correctness change, whereas leaving it alone keeps
    whatever the body established and is observable by the caller. Each legacy
    TF32 backend flag is restored INDEPENDENTLY, so one backend's unreadable
    getter cannot discard the other backend's readable value (RV4A-M1).

    Bookkeeping honesty (RV4A-M2): ``_latest_global_config`` initialises worker
    processes, so on exit it must describe the state that actually SURVIVES.
    For a knob that could not be restored, the surviving value is whatever the
    body established -- the pre-context dict entry would be a lie. If the body
    did not set the knob either, the surviving value is unknown and the entry
    is dropped rather than fabricated.
    """
    global _latest_global_config
    with _GLOBAL_OPTIONS_LOCK:
        previous_precision = _read_float32_matmul_precision()
        previous_matmul_tf32, previous_cudnn_tf32 = _read_legacy_allow_tf32()
        previous_default_dtype = torch.get_default_dtype()
        previous_global_config = dict(_latest_global_config)
        previous_dtype = torch.get_default_dtype()
        previous_determinism = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        )
        body_config = dict(config) if config is not None else {}
        try:
            if config is not None:
                _apply_global_options(
                    config,
                    warn_on_override=warn_on_override,
                )
            yield
        finally:
            torch.set_default_dtype(previous_dtype)
            torch.use_deterministic_algorithms(
                previous_determinism[0], warn_only=previous_determinism[1]
            )
            torch.backends.cudnn.deterministic = previous_determinism[2]
            torch.backends.cudnn.benchmark = previous_determinism[3]
            restored_config = dict(previous_global_config)

            def _reconcile_unrestorable(key):
                # The real knob keeps the body's value; make the bookkeeping
                # say so, or say nothing when even the body did not set it.
                if key in body_config:
                    restored_config[key] = body_config[key]
                else:
                    restored_config.pop(key, None)

            if previous_precision is not None:
                torch.set_float32_matmul_precision(previous_precision)
            else:
                _reconcile_unrestorable("float32_matmul_precision")
            if previous_matmul_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            if previous_cudnn_tf32 is not None:
                torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
            if previous_matmul_tf32 is None or previous_cudnn_tf32 is None:
                _reconcile_unrestorable("allow_tf32")
            # default_dtype is a process-global torch knob: leaving the body's
            # value behind leaks into unrelated suites (test_smoke_routes
            # float32 configs broke the session's float64 trainer pass).
            torch.set_default_dtype(previous_default_dtype)
            survived_dtype = (
                "float64" if previous_default_dtype is torch.float64 else "float32"
            )
            if "default_dtype" in body_config:
                restored_config["default_dtype"] = survived_dtype
            elif restored_config.get("default_dtype", survived_dtype) != survived_dtype:
                # The knob itself is always restored, so a contradicting entry
                # was already stale at entry; drop it rather than restore a lie.
                restored_config.pop("default_dtype", None)
            _latest_global_config = restored_config
