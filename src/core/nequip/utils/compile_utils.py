"""Conservative compilation helpers for SpinGNN training."""

import contextlib
from dataclasses import dataclass, field
import fnmatch
import functools
import threading
import warnings
from typing import Any, Callable, Dict

import e3nn
import torch

from nequip.data import AtomicDataDict
from nequip.utils._compile_grad_partitions import compile_gradient_partitions


DEFAULT_COMPILE_CONFIG = {
    "enabled": True,
    "scope": "full_fx",
    "dynamic": True,
    "inductor_options": {},
}

BASE_AUTOCAST_FP32_MODULES = (
    "RescaleOutput",
    "ParaStressForceSpinOutput",
    "edge_eng_soc_on",
)

# A4 CPU calibration found worst-case force relative errors of 1.7803e-2,
# 2.6487e-2, and 2.6487e-2 for current, +tensor-products, and +geometry.
# Expansion did not meet the 1e-2 force-supervision gate, so keep the least
# invasive current tier as the default and retain the larger tiers for audits.

TENSOR_PRODUCT_AUTOCAST_FP32_MODULES = (
    "TensorProduct",
    "FullyConnectedTensorProduct",
    "ElementwiseTensorProduct",
    "e3nn.o3._linear.Linear",
    "*.tps.*",
    "*.linears.*",
)

GEOMETRY_AUTOCAST_FP32_MODULES = (
    "SphericalHarmonicEdgeAttrs*",
    "SphericalHarmonics",
    "RadialBasisEdgeEncoding*",
    "BesselBasis",
    "FourierBasis",
    "NormalizedBasis",
    "PolynomialCutoff",
    "Norm",
    "NormActivation",
    "normalize2mom",
)

AUTOCAST_FP32_EXPERIMENT_TIERS = {
    "current": BASE_AUTOCAST_FP32_MODULES,
    "tensor_products": (
        BASE_AUTOCAST_FP32_MODULES + TENSOR_PRODUCT_AUTOCAST_FP32_MODULES
    ),
    "tensor_products_geometry": (
        BASE_AUTOCAST_FP32_MODULES
        + TENSOR_PRODUCT_AUTOCAST_FP32_MODULES
        + GEOMETRY_AUTOCAST_FP32_MODULES
    ),
}

DEFAULT_AUTOCAST_CONFIG = {
    "enabled": False,
    "dtype": "bfloat16",
    "fp32_modules": list(BASE_AUTOCAST_FP32_MODULES),
}

_CONSERVATIVE_INDUCTOR_OPTIONS = {
    "triton.cudagraphs": False,
    "max_autotune": False,
    "max_autotune_gemm": False,
}

_GRAD_INPUT_KEYS = (
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.EDGE_VECTORS_KEY,
    AtomicDataDict.SPIN_KEY,
)

_MAX_FULL_FX_CACHE_ENTRIES = 64


@dataclass
class _CompileState:
    scope: str
    dynamic: bool
    options: Dict[str, Any]
    compiled_by_mode: Dict[str, Callable] = field(default_factory=dict)
    input_signatures: Dict[str, tuple] = field(default_factory=dict)
    output_signatures: Dict[str, tuple] = field(default_factory=dict)
    trace_sizes: Dict[str, tuple] = field(default_factory=dict)
    capture_signature: tuple = ()
    compile_count: int = 0
    full_fx_expand_count: int = 0
    eager_grad_bridge_calls: int = 0
    compile_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )


@dataclass
class _AutocastState:
    path: str
    class_name: str
    device_type: str
    calls: int = 0
    input_dtypes: tuple = ()
    output_dtypes: tuple = ()


def _torch_version_tuple():
    version = torch.__version__.split("+", 1)[0].split(".")
    return tuple(int(part) for part in version[:2])


def _clone_tensor_dict(data):
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


def _filtered_inductor_options(user_options):
    valid_options = set(torch._inductor.list_options())
    invalid = sorted(set(user_options) - valid_options)
    if invalid:
        warnings.warn(
            "Ignoring unsupported torch 2.11 inductor options: " + ", ".join(invalid)
        )
    options = {
        key: value for key, value in user_options.items() if key in valid_options
    }
    for key, value in _CONSERVATIVE_INDUCTOR_OPTIONS.items():
        if key in valid_options:
            options[key] = value
    return options


def _normalized_compile_config(config):
    raw = config.get("spingnn_compile", {}) if hasattr(config, "get") else {}
    settings = dict(DEFAULT_COMPILE_CONFIG)
    settings.update(dict(raw or {}))
    settings["inductor_options"] = _filtered_inductor_options(
        dict(settings.get("inductor_options", {}))
    )
    if settings["scope"] not in ("energy_core", "full_fx"):
        raise ValueError(
            "spingnn_compile.scope must be 'energy_core' or 'full_fx'; "
            f"got {settings['scope']!r}"
        )
    return settings


def _normalized_autocast_config(config):
    raw = config.get("train_autocast", {}) if hasattr(config, "get") else {}
    settings = dict(DEFAULT_AUTOCAST_CONFIG)
    settings.update(dict(raw or {}))
    settings["fp32_modules"] = list(settings.get("fp32_modules", []))
    dtype = settings.get("dtype")
    if dtype not in ("bfloat16", "bf16", torch.bfloat16):
        raise ValueError("train_autocast.dtype must be bfloat16; " f"got {dtype!r}")
    settings["torch_dtype"] = torch.bfloat16
    return settings


def training_autocast_context(config, device_type):
    settings = _normalized_autocast_config(config)
    if not settings["enabled"]:
        return contextlib.nullcontext()
    return torch.autocast(
        device_type=device_type,
        dtype=settings["torch_dtype"],
        enabled=True,
    )


def _tree_to_fp32(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _tree_to_fp32(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_fp32(item) for item in value)
    if isinstance(value, list):
        return [_tree_to_fp32(item) for item in value]
    return value


def _floating_dtypes(value):
    dtypes = set()
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        dtypes.add(value.dtype)
    elif isinstance(value, dict):
        for item in value.values():
            dtypes.update(_floating_dtypes(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            dtypes.update(_floating_dtypes(item))
    return dtypes


def _matches_fp32_module(path, module, patterns):
    class_name = type(module).__name__
    qualified_name = f"{type(module).__module__}.{class_name}"
    return any(
        fnmatch.fnmatchcase(class_name, pattern)
        or fnmatch.fnmatchcase(qualified_name, pattern)
        or fnmatch.fnmatchcase(path, pattern)
        or (path and path.endswith("." + pattern))
        for pattern in patterns
    )


def _unwrap_fp32_forward(module):
    """Remove our outer precision wrapper before rebuilding either policy."""
    state = getattr(module, "_spingnn_autocast_state", None)
    original = getattr(module, "_spingnn_autocast_original_forward", None)
    if original is not None:
        object.__setattr__(module, "forward", original)
    for attribute in ("_spingnn_autocast_state", "_spingnn_autocast_original_forward"):
        if hasattr(module, attribute):
            delattr(module, attribute)
    return state


def _wrap_fp32_forward(module, path, device_type):
    previous = getattr(module, "_spingnn_autocast_state", None)
    if previous is not None and previous.device_type == device_type:
        return
    _unwrap_fp32_forward(module)
    state = _AutocastState(
        path=path or "<root>",
        class_name=type(module).__name__,
        device_type=device_type,
    )
    original_forward = module.forward

    @functools.wraps(original_forward)
    def fp32_forward(*args, **kwargs):
        fp32_args = _tree_to_fp32(args)
        fp32_kwargs = _tree_to_fp32(kwargs)
        state.input_dtypes = tuple(
            sorted((str(dtype) for dtype in _floating_dtypes((fp32_args, fp32_kwargs))))
        )
        with torch.autocast(device_type=device_type, enabled=False):
            output = original_forward(*fp32_args, **fp32_kwargs)
        output = _tree_to_fp32(output)
        state.output_dtypes = tuple(
            sorted(str(dtype) for dtype in _floating_dtypes(output))
        )
        state.calls += 1
        return output

    object.__setattr__(module, "_spingnn_autocast_state", state)
    object.__setattr__(module, "_spingnn_autocast_original_forward", original_forward)
    object.__setattr__(module, "forward", fp32_forward)


def configure_model_for_autocast(model, config, device_type):
    settings = _normalized_autocast_config(config)
    modules = list(model.named_modules())
    matches = [
        (path, module)
        for path, module in modules
        if settings["enabled"]
        and _matches_fp32_module(path, module, settings["fp32_modules"])
    ]
    if settings["enabled"] and not matches:
        raise RuntimeError(
            "train_autocast enabled but fp32_modules matched no model modules"
        )
    selected = {id(module) for _, module in matches}
    for _, module in modules:
        if id(module) not in selected:
            _unwrap_fp32_forward(module)
            if hasattr(module, "_spingnn_energy_autocast"):
                delattr(module, "_spingnn_energy_autocast")
    for path, module in matches:
        if type(module).__name__ == "ParaStressForceSpinOutput":
            object.__setattr__(
                module,
                "_spingnn_energy_autocast",
                (device_type, settings["torch_dtype"]),
            )
        _wrap_fp32_forward(module, path, device_type)
    return model


def get_autocast_stats(model):
    states = [
        getattr(module, "_spingnn_autocast_state")
        for module in model.modules()
        if hasattr(module, "_spingnn_autocast_state")
    ]
    return [
        {
            "path": state.path,
            "class_name": state.class_name,
            "calls": state.calls,
            "input_dtypes": state.input_dtypes,
            "output_dtypes": state.output_dtypes,
        }
        for state in states
    ]


def configure_model_for_compile(model, config):
    """Attach lazy compile state without changing module/state_dict structure."""
    settings = _normalized_compile_config(config)
    targets = [
        module
        for module in model.modules()
        if type(module).__name__ == "ParaStressForceSpinOutput"
    ]
    if not settings["enabled"]:
        for module in targets:
            _clear_compile_configuration(module)
        return model
    if not targets:
        warnings.warn(
            "spingnn_compile is enabled by default, but this model has no "
            "ParaStressForceSpinOutput target; continuing with eager training"
        )
        return model
    if _torch_version_tuple() < (2, 11):
        raise RuntimeError(
            "spingnn_compile.enabled requires torch>=2.11; "
            f"found {torch.__version__}"
        )
    if e3nn.get_optimization_defaults()["jit_script_fx"]:
        raise RuntimeError(
            "spingnn_compile.enabled requires "
            "e3nn_optimization_defaults.jit_script_fx=false"
        )

    import torch._functorch.config as functorch_config

    # AOT donated buffers reject the retained graph needed by force training.
    functorch_config.donated_buffer = False
    # torch_runstats.scatter derives dim_size through Tensor.item().
    torch._dynamo.config.capture_scalar_outputs = True
    # e3nn Irrep metadata deliberately does not implement sequence length.
    # Keep incompatible nn modules as conservative Dynamo graph boundaries.
    torch._dynamo.config.guard_nn_modules = False
    torch._dynamo.config.guard_nn_modules_using_dict_tags = False
    torch._dynamo.config.force_nn_module_property_static_shapes = False
    torch._dynamo.config.inline_inbuilt_nn_modules = False

    for module in targets:
        state = _CompileState(
            scope=settings["scope"],
            dynamic=bool(settings["dynamic"]),
            options=dict(settings["inductor_options"]),
        )
        object.__setattr__(module, "_spingnn_compile_state", state)
        if state.scope == "full_fx":
            _wrap_full_fx_forward(module, state)
    return model


def _wrap_full_fx_forward(output_module, state):
    if hasattr(output_module, "_spingnn_full_fx_wrapped"):
        return

    autocast_state = _unwrap_fp32_forward(output_module)
    original_forward = output_module.forward

    @functools.wraps(original_forward)
    def full_fx_forward(data):
        current_state = getattr(output_module, "_spingnn_compile_state", None)
        if current_state is None or current_state.scope != "full_fx":
            return original_forward(data)
        output = _call_full_fx(output_module, data, current_state)
        _assert_compile_state_is_current(output_module, current_state)
        return output

    object.__setattr__(
        output_module, "_spingnn_compile_original_forward", original_forward
    )
    object.__setattr__(output_module, "forward", full_fx_forward)
    object.__setattr__(output_module, "_spingnn_full_fx_wrapped", True)
    if autocast_state is not None:
        _wrap_fp32_forward(
            output_module, autocast_state.path, autocast_state.device_type
        )


def _clear_compile_configuration(output_module):
    original_forward = getattr(output_module, "_spingnn_compile_original_forward", None)
    if original_forward is not None:
        autocast_state = _unwrap_fp32_forward(output_module)
        object.__setattr__(output_module, "forward", original_forward)
        if autocast_state is not None:
            _wrap_fp32_forward(
                output_module, autocast_state.path, autocast_state.device_type
            )
    for attribute in (
        "_spingnn_compile_state",
        "_spingnn_compile_original_forward",
        "_spingnn_full_fx_wrapped",
    ):
        if hasattr(output_module, attribute):
            delattr(output_module, attribute)


def _capture_signature(output_module, data):
    """Track structure, not tensor versions: optimizer writes reuse live weights."""
    model = (
        output_module
        if isinstance(output_module, torch.nn.Module)
        else output_module.func
    )
    tensors = tuple(model.named_parameters()) + tuple(model.named_buffers())
    model_signature = tuple(
        (
            name,
            id(value),
            value.device,
            value.dtype,
            value.requires_grad,
            tuple(value.shape),
            tuple(value.stride()),
        )
        for name, value in tensors
    )
    input_signature = tuple(
        sorted(
            (key, value.device, value.dtype, value.requires_grad, value.ndim)
            for key, value in data.items()
            if isinstance(value, torch.Tensor)
        )
    )
    precision_signature = tuple(
        (name, module._spingnn_autocast_state.device_type)
        for name, module in model.named_modules()
        if hasattr(module, "_spingnn_autocast_state")
    )
    autocast = tuple(
        (device, torch.is_autocast_enabled(device), torch.get_autocast_dtype(device))
        for device in ("cpu", "cuda")
    )
    return model_signature, input_signature, precision_signature, autocast


def _compiled_full_fx_callable(output_module, state, data, n_frames=None):
    mode = "train" if output_module.training else "eval"
    cache_key = _full_fx_cache_key(state, data, mode, n_frames=n_frames)
    with state.compile_lock:
        signature = _capture_signature(output_module, data)
        if state.capture_signature != signature:
            state.compiled_by_mode.clear()
            state.input_signatures.clear()
            state.output_signatures.clear()
            state.trace_sizes.clear()
            state.capture_signature = signature
        if cache_key not in state.compiled_by_mode:
            from nequip.utils.fx_compat import expand_training_graph

            if len(state.compiled_by_mode) >= _MAX_FULL_FX_CACHE_ENTRIES:
                oldest = next(iter(state.compiled_by_mode))
                state.compiled_by_mode.pop(oldest)
                state.input_signatures.pop(oldest, None)
                state.output_signatures.pop(oldest, None)
                state.trace_sizes.pop(oldest, None)
            output_module.func.train(output_module.training)
            graph_module = expand_training_graph(
                output_module,
                data,
                create_graph=output_module.training,
            )
            state.input_signatures[cache_key] = tuple(graph_module.input_signature)
            state.output_signatures[cache_key] = tuple(graph_module.output_signature)
            state.trace_sizes[cache_key] = tuple(graph_module.trace_sizes)
            if not any(
                isinstance(value, torch.Tensor) and value.is_cuda
                for value in data.values()
            ):
                # Torch 2.11's CPU vectorizer can emit scalar blendv calls for
                # expanded higher-order graphs. Keep this CPU-only correctness
                # gate scalar; CUDA compilation does not receive this option.
                state.options["cpp.simdlen"] = 1
                state.options["cpp.enable_loop_tail_vec"] = False
            elif state.dynamic and (
                "triton.coalesce_tiling_analysis" in torch._inductor.list_options()
            ):
                # Torch 2.11 documents this analysis as unsupported for dynamic
                # shapes; grouped higher-order views can retain nested modular
                # indexing and trip its tiling solver. Keep the full graph and
                # disable only this optional analysis for this CUDA callable.
                state.options["triton.coalesce_tiling_analysis"] = False
            state.compiled_by_mode[cache_key] = compile_gradient_partitions(
                graph_module,
                functools.partial(
                    torch.compile,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=state.dynamic,
                    options=state.options,
                ),
            )
            state.compile_count += 1
            state.full_fx_expand_count += 1
        return state.compiled_by_mode[cache_key], cache_key


def _full_fx_cache_key(state, data, mode, n_frames=None):
    if n_frames is None:
        n_frames = int(data[AtomicDataDict.NUM_BATCH_KEY].item())
    n_atoms = data[AtomicDataDict.POSITIONS_KEY].shape[0]
    n_edges = data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
    if not state.dynamic:
        return f"{mode}:static:{n_frames}:{n_atoms}:{n_edges}"

    frame_class = "one" if n_frames == 1 else "many"
    if n_atoms == 1:
        atom_class = "one"
    elif n_atoms <= 16:
        atom_class = "small"
    else:
        atom_class = "large"
    if n_edges == 0:
        edge_class = "zero"
    elif n_edges == 1:
        edge_class = "one"
    elif n_edges < 25:
        edge_class = "small"
    elif n_edges < 32:
        edge_class = "medium"
    elif n_edges <= 819:
        edge_class = "standard"
    elif n_edges <= 1365:
        edge_class = "large"
    else:
        edge_class = "huge"
    return f"{mode}:{frame_class}:{atom_class}:{edge_class}"


def _mark_full_fx_dimensions(graph_input):
    # Strong dynamic markers reject accidental scalar specialization while
    # allowing Inductor to infer valid ranges from indexing and kernel limits.
    # A promised INT32_MAX dimension conflicts with guards such as 32*N <=
    # INT32_MAX even for small real batches; these are compiler bounds, not
    # data admission limits. Keep the existing cache classes independently.
    dynamic_axes = {
        AtomicDataDict.POSITIONS_KEY: {0},
        AtomicDataDict.SPIN_KEY: {0},
        AtomicDataDict.ATOM_TYPE_KEY: {0},
        AtomicDataDict.ATOMIC_NUMBERS_KEY: {0},
        AtomicDataDict.SPIN_LENGTH_KEY: {0},
        AtomicDataDict.BATCH_KEY: {0},
        AtomicDataDict.EDGE_INDEX_KEY: {1},
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: {0},
        AtomicDataDict.CELL_KEY: {0},
        AtomicDataDict.PBC_KEY: {0},
    }
    for key, value in graph_input.items():
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            continue
        variable = dynamic_axes.get(key, set())
        for dimension in range(value.ndim):
            if dimension in variable and value.shape[dimension] > 1:
                torch._dynamo.mark_dynamic(value, dimension)
            else:
                torch._dynamo.mark_static(value, dimension)


def _call_full_fx(output_module, data, state):
    from nequip.utils.fx_compat import validate_training_batch

    sample, (n_frames, _, _) = validate_training_batch(output_module, data)
    with _energy_autocast_context(output_module):
        # Snapshot the callable and its signatures together. An eviction from
        # another forward cannot remove metadata while this call is executing.
        with state.compile_lock:
            compiled, cache_key = _compiled_full_fx_callable(
                output_module, state, sample, n_frames=n_frames
            )
            graph_input = {
                key: sample[key] for key in state.input_signatures[cache_key]
            }
            output_signature = state.output_signatures[cache_key]
        _mark_full_fx_dimensions(graph_input)
        values = compiled(graph_input)
    if isinstance(values, torch.Tensor):
        values = (values,)
    output = dict(data)
    output.update(zip(output_signature, values))
    return output


def _compiled_callable(output_module, state):
    mode = "train" if output_module.training else "eval"
    if mode not in state.compiled_by_mode:
        output_module.func.train(output_module.training)
        state.compiled_by_mode[mode] = torch.compile(
            output_module.func,
            backend="inductor",
            dynamic=state.dynamic,
            options=state.options,
        )
        state.compile_count += 1
    return state.compiled_by_mode[mode]


def _cuda_rng_devices(data):
    return sorted(
        {
            value.device.index
            for value in data.values()
            if isinstance(value, torch.Tensor)
            and value.is_cuda
            and value.device.index is not None
        }
    )


def _compiled_value_eager_gradient(output_module, compiled, data, state):
    # Torch 2.11 AOTAutograd does not support the double backward required by
    # force loss. Use Inductor for the forward values and an eager recomputation
    # for the higher-order gradient graph. This is correctness-first and does
    # not claim a training speedup while the bridge is active.
    with torch.random.fork_rng(devices=_cuda_rng_devices(data), enabled=True):
        with torch.no_grad():
            compiled_output = compiled(_clone_tensor_dict(data))
    eager_output = output_module.func(data)

    for key, eager_value in eager_output.items():
        if key not in compiled_output:
            compiled_output[key] = eager_value
            continue
        if key in _GRAD_INPUT_KEYS:
            compiled_output[key] = eager_value
            continue
        compiled_value = compiled_output[key]
        if (
            isinstance(compiled_value, torch.Tensor)
            and isinstance(eager_value, torch.Tensor)
            and compiled_value.is_floating_point()
        ):
            compiled_output[key] = compiled_value.detach() + (
                eager_value - eager_value.detach()
            )
    state.eager_grad_bridge_calls += 1
    return compiled_output


def _energy_autocast_context(output_module):
    autocast_state = getattr(output_module, "_spingnn_energy_autocast", None)
    if autocast_state is None:
        return contextlib.nullcontext()
    device_type, dtype = autocast_state
    return torch.autocast(
        device_type=device_type,
        dtype=dtype,
        enabled=True,
    )


def _assert_compile_state_is_current(output_module, state):
    if getattr(output_module, "_spingnn_compile_state", None) is not state:
        raise RuntimeError("compile configuration changed during forward")


def call_compiled_energy_core(output_module, data):
    with _energy_autocast_context(output_module):
        state = getattr(output_module, "_spingnn_compile_state", None)
        if state is None:
            return output_module.func(data)
        if state.scope == "full_fx":
            return output_module.func(data)

        compiled = _compiled_callable(output_module, state)
        if torch.is_grad_enabled():
            output = _compiled_value_eager_gradient(
                output_module, compiled, data, state
            )
        else:
            output = compiled(data)
        _assert_compile_state_is_current(output_module, state)
        return output


def get_compile_stats(model):
    states = [
        getattr(module, "_spingnn_compile_state")
        for module in model.modules()
        if hasattr(module, "_spingnn_compile_state")
    ]
    return {
        "targets": len(states),
        "scopes": sorted({state.scope for state in states}),
        "compile_count": sum(state.compile_count for state in states),
        "full_fx_expand_count": sum(state.full_fx_expand_count for state in states),
        "cache_modes": sorted(
            {
                cache_key.split(":", 1)[0]
                for state in states
                for cache_key in state.compiled_by_mode
            }
        ),
        "cache_entries": sum(len(state.compiled_by_mode) for state in states),
        # A logical cached callable can contain multiple independent gradient
        # regions; expose those separately from cache construction counts.
        "compiled_regions": sum(
            getattr(compiled, "_spingnn_gradient_partition_count", 1)
            for state in states
            for compiled in state.compiled_by_mode.values()
        ),
        "eager_grad_bridge_calls": sum(
            state.eager_grad_bridge_calls for state in states
        ),
        "trace_sizes": {
            mode: sizes for state in states for mode, sizes in state.trace_sizes.items()
        },
        "options": [dict(state.options) for state in states],
    }
