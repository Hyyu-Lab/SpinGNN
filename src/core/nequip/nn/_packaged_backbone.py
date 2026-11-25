"""Run a ``torch.package``-d NequIP-family model as a feature backbone.

The packaged model is executed with the code interned in its archive. Only the
leading stages of its ``SequentialGraphNetwork`` up to ``stop_after`` are run;
selected intermediate quantities ("taps") are copied into the SpinGNN data
dictionary so that SpinGNN modules can read them.
"""

import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from e3nn import o3

from nequip.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin


# SpinGNN field -> packaged field. Keys that are absent from the SpinGNN data
# are simply not passed on (e.g. no cell for an isolated molecule).
_TRANSLATED_FIELDS: Tuple[Tuple[str, str], ...] = (
    (AtomicDataDict.POSITIONS_KEY, "pos"),
    (AtomicDataDict.EDGE_INDEX_KEY, "edge_index"),
    (AtomicDataDict.EDGE_CELL_SHIFT_KEY, "edge_cell_shift"),
    (AtomicDataDict.CELL_KEY, "cell"),
    (AtomicDataDict.ATOM_TYPE_KEY, "atom_types"),
    (AtomicDataDict.BATCH_KEY, "batch"),
)
# Derived from ``batch``: number of atoms per frame.
_PACKAGED_NUM_ATOMS_FIELD = "num_atoms"
_PROVIDED_FIELDS = frozenset(
    [packaged for _, packaged in _TRANSLATED_FIELDS] + [_PACKAGED_NUM_ATOMS_FIELD]
)
_BATCH_PTR_FIELD = "ptr"

TapSpec = Union[str, Mapping[str, Any]]

# ``torch_compile`` settings of the packaged stages (off by default).
DEFAULT_TORCH_COMPILE_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "backend": "inductor",
    "dynamic": True,
    "mode": None,
    "inductor_options": {},
}
# Input dimensions that never vary. Marking them static keeps dynamic-shape
# compilation from equating them with a varying size of the same value (e.g.
# two frames and the two rows of ``edge_index``), which would recompile.
_STATIC_INPUT_DIMS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("pos", (1,)),
    ("edge_index", (0,)),
    ("edge_cell_shift", (1,)),
    ("cell", (1, 2)),
)


def _torch_compile_kwargs(settings) -> Optional[Dict[str, Any]]:
    """Validate ``torch_compile`` settings and return ``torch.compile`` arguments.

    ``None``, ``False`` and ``{"enabled": False, ...}`` mean eager execution
    (``None`` is returned); ``True`` enables compilation with the defaults.
    """
    if settings is None or settings is False:
        return None
    if settings is True:
        raw = {"enabled": True}
    elif isinstance(settings, Mapping):
        raw = dict(settings)
    else:
        raise TypeError(
            f"torch_compile must be a bool or a mapping; got {type(settings).__name__}"
        )
    unknown = sorted(set(raw) - set(DEFAULT_TORCH_COMPILE_CONFIG))
    if unknown:
        raise ValueError(
            f"Unknown torch_compile option(s) {unknown}; valid options are "
            f"{sorted(DEFAULT_TORCH_COMPILE_CONFIG)}"
        )
    merged = dict(DEFAULT_TORCH_COMPILE_CONFIG)
    merged.update(raw)
    if not merged["enabled"]:
        return None
    if merged["dynamic"] not in (True, False, None):
        raise ValueError(
            f"torch_compile.dynamic must be true, false or null; got {merged['dynamic']!r}"
        )
    backend, mode = merged["backend"], merged["mode"]
    options = dict(merged["inductor_options"] or {})
    kwargs: Dict[str, Any] = {"backend": backend, "dynamic": merged["dynamic"]}
    if backend == "inductor":
        if mode is not None and options:
            raise ValueError(
                "torch_compile: give either `mode` or `inductor_options`, not both"
            )
        if mode is not None:
            kwargs["mode"] = str(mode)
        else:
            # Same policy as `spingnn_compile.inductor_options`: unknown options
            # are dropped with a warning; CUDA graphs and max-autotune stay off.
            from nequip.utils.compile_utils import _filtered_inductor_options

            kwargs["options"] = _filtered_inductor_options(options)
    elif mode is not None or options:
        raise ValueError(
            "torch_compile: `mode` and `inductor_options` apply to the inductor "
            f"backend only; got backend={backend!r}"
        )
    return kwargs


class _ModuleTap:
    """Forward hook of one module tap; records the output during a forward.

    It is registered once, at construction, and does nothing outside
    :meth:`PackagedBackbone.forward`. A fixed hook keeps the hooked module's
    ``_forward_hooks`` unchanged between forwards. Compiled stages may address
    a hook by its handle id: Dynamo does so with
    ``inline_inbuilt_nn_modules=False``, which ``spingnn_compile`` sets for the
    whole process. A new hook per forward would then recompile on every call;
    with ``guard_nn_modules=False`` as well, it would replay a stale handle id
    and raise ``KeyError``.
    """

    def __init__(self, field: str, path: str):
        self.field = field
        self.path = path
        self.recording = False
        self.output: Optional[torch.Tensor] = None

    def start(self):
        self.output = None
        self.recording = True

    def stop(self) -> Optional[torch.Tensor]:
        output = self.output
        self.output = None
        self.recording = False
        return output

    def __call__(self, module, inputs, output):
        if not self.recording:
            return None
        if self.output is not None:
            raise RuntimeError(
                f"Tap {self.field!r}: hooked module ran more than once in one forward"
            )
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"Tap {self.field!r}: hooked module returned {type(output).__name__}, "
                "expected a tensor"
            )
        self.output = output
        return None


def _to_irreps(value) -> Optional[o3.Irreps]:
    """Convert an irreps-like value (possibly from the packaged e3nn) to SpinGNN irreps."""
    if value is None:
        return None
    return o3.Irreps(str(value))


def _channel_irreps(irreps) -> o3.Irreps:
    """Per-channel irreps of a strided ``(n_edges, n_channels, irreps.dim)`` tensor.

    Registered irreps describe the last dimension only, so the channel
    multiplicity of a strided tensor is not part of them.
    """
    return o3.Irreps([(1, mul_ir.ir) for mul_ir in _to_irreps(irreps)])


def _resolve_sequential(graph_model: torch.nn.Module) -> torch.nn.Sequential:
    """Find the ``SequentialGraphNetwork`` inside a packaged nequip>=0.7 ``GraphModel``.

    ``GraphModel.model`` is either the sequential network itself or an output
    wrapper (e.g. ``ForceStressOutput``) that holds it as ``.func``.
    """
    module = graph_model
    for attr in ("model", "func"):
        child = getattr(module, attr, None)
        if isinstance(child, torch.nn.Module):
            module = child
    if not isinstance(module, torch.nn.Sequential):
        raise TypeError(
            "Could not locate the packaged SequentialGraphNetwork "
            f"(got {type(module).__name__}); expected GraphModel.model[.func]"
        )
    return module


def _packaged_type_names(graph_model: torch.nn.Module) -> Optional[List[str]]:
    names = getattr(graph_model, "type_names", None)
    if names:
        return [str(n) for n in names]
    metadata = getattr(graph_model, "metadata", None) or {}
    if "type_names" in metadata:
        return str(metadata["type_names"]).split()
    return None


def _packaged_r_max(graph_model: torch.nn.Module) -> Optional[float]:
    metadata = getattr(graph_model, "metadata", None) or {}
    if "r_max" in metadata:
        return float(metadata["r_max"])
    return None


class PackagedBackbone(GraphModuleMixin, torch.nn.Module):
    """Expose intermediate features of a ``torch.package``-d model as graph fields.

    The archive is loaded with :class:`torch.package.PackageImporter` and the
    resource ``model/<compile_mode>_model.pkl``; when that object is a mapping
    (nequip>=0.7 writes a ``ModuleDict``), ``model_key`` selects the model. The
    packaged ``SequentialGraphNetwork`` is run on a dictionary translated from
    the SpinGNN data (``pos``, ``edge_index``, ``edge_cell_shift``, ``cell``,
    ``atom_types``, ``batch`` and ``num_atoms`` derived from ``batch``),
    restricted to the package's ``model_input_fields``. Execution stops after
    the stage ``stop_after``; later stages are dropped unless
    ``keep_skipped_stages`` is set, and no force/stress autograd is performed.

    Args:
        package_path: path of the ``torch.package`` archive.
        taps: maps a SpinGNN output field to its source. A string, or a mapping
            ``{"key": name}``, copies an entry of the packaged data dictionary
            after ``stop_after``. A mapping ``{"module": path}`` captures the
            output tensor of the submodule ``path`` (relative to the packaged
            network, e.g. ``"allegro.tps.2"``) with a forward hook that is
            registered at construction and records only during this module's
            forward. An optional
            ``"irreps"`` entry overrides the registered irreps of the last
            dimension; by default they are taken from the stage's
            ``irreps_out`` (key taps) or from the module's ``irreps_out``. A
            strided module output ``(n_edges, mul, dim)`` (one that has a
            ``mul`` attribute) registers its per-channel irreps.
        stop_after: last stage of the packaged network to run (default: all).
        trainable_prefixes: parameter-name prefixes that stay trainable; all
            other packaged parameters are frozen. A parameter is trainable if a
            prefix matches the start of its name relative to the packaged
            network (``"<stage>.<rest>"``) or the start of that name with the
            stage component removed (``"<rest>"``). For example
            ``"allegro.latents."`` and ``"latents."`` both select
            ``allegro.latents.*``, and ``"tensor_embed."`` selects the
            ``tensor_embed`` stage. Every prefix must match at least one parameter.
        type_names: SpinGNN type names; must equal the packaged type names in order.
        r_max: SpinGNN neighbor cutoff; must equal the packaged ``r_max`` if known.
        compile_mode: selects ``model/<compile_mode>_model.pkl``.
        model_key: key of the model inside a packaged mapping.
        keep_skipped_stages: keep stages after ``stop_after`` as (unused) submodules.
        torch_compile: opt-in :func:`torch.compile` of the packaged stage
            execution (the stages up to ``stop_after``), with the keys of
            ``DEFAULT_TORCH_COMPILE_CONFIG``: ``enabled`` (default ``False``),
            ``backend`` (default ``"inductor"``), ``dynamic`` (default ``True``:
            dynamic atom, edge and frame counts), ``mode`` and
            ``inductor_options`` (inductor only, mutually exclusive; options are
            filtered like ``spingnn_compile.inductor_options``). ``True`` enables
            the defaults; ``None``/``False`` run eagerly.

    The packaged submodules always stay in eval mode, and the module keeps the
    packaged parameter and buffer dtypes: it carries ``_spingnn_keep_dtype``
    so whole-model dtype casts skip it.

    With ``torch_compile`` enabled, the input translation and the tap copies
    stay eager and only the stage loop is compiled, lazily at the first
    forward. Dynamo traces the module-tap hooks into the compiled stages;
    since the hooks never change, the compiled stages stay valid whatever
    Dynamo's nn-module settings are. Parameter and ``state_dict`` names are
    unchanged (no ``_orig_mod.`` prefix), so eager and compiled runs share
    checkpoints; copies and unpickled modules recompile on first use. Only
    first-order gradients through the compiled stages are supported (no
    force/stress double backward), and Dynamo's recompile limit applies per
    process to all packaged backbones together.
    """

    _spingnn_keep_dtype = True

    def __init__(
        self,
        package_path: str,
        taps: Mapping[str, TapSpec],
        stop_after: Optional[str] = None,
        trainable_prefixes: Sequence[str] = (),
        type_names: Optional[Sequence[str]] = None,
        r_max: Optional[float] = None,
        compile_mode: str = "eager",
        model_key: Optional[str] = "sole_model",
        keep_skipped_stages: bool = False,
        torch_compile: Union[None, bool, Mapping[str, Any]] = None,
        irreps_in=None,
    ):
        super().__init__()
        if not taps:
            raise ValueError("PackagedBackbone needs at least one tap")
        self.package_path = str(package_path)
        self.compile_mode = str(compile_mode)
        self._torch_compile_kwargs = _torch_compile_kwargs(torch_compile)
        self._compiled_stages = None

        importer = torch.package.PackageImporter(self.package_path)
        loaded = importer.load_pickle(
            "model", f"{self.compile_mode}_model.pkl", map_location="cpu"
        )
        if isinstance(loaded, (torch.nn.ModuleDict, Mapping)):
            if model_key not in loaded:
                raise KeyError(
                    f"Packaged model has no key {model_key!r}; available: {list(loaded.keys())}"
                )
            graph_model = loaded[model_key]
        else:
            graph_model = loaded

        self._check_type_names(graph_model, type_names)
        self._check_r_max(graph_model, r_max)
        self._input_fields = self._resolve_input_fields(importer, graph_model)

        sequential = _resolve_sequential(graph_model)
        stage_names = [name for name, _ in sequential.named_children()]
        if stop_after is None:
            stop_after = stage_names[-1]
        if stop_after not in stage_names:
            raise ValueError(
                f"stop_after={stop_after!r} is not a stage of the packaged network; "
                f"stages are {stage_names}"
            )
        n_run = stage_names.index(stop_after) + 1
        self.stop_after = stop_after
        self._run_stages: List[str] = stage_names[:n_run]
        kept = stage_names if keep_skipped_stages else self._run_stages
        self.func = torch.nn.ModuleDict(
            OrderedDict((name, getattr(sequential, name)) for name in kept)
        )
        del sequential, graph_model, loaded, importer

        self._key_taps: List[Tuple[str, str]] = []
        self._module_taps: List[_ModuleTap] = []
        tap_irreps = self._parse_taps(taps)

        self._set_trainable(list(trainable_prefixes))
        self.func.eval()

        self._init_irreps(irreps_in=irreps_in, irreps_out=tap_irreps)

    # -- construction helpers --

    @staticmethod
    def _check_type_names(graph_model, type_names):
        packaged = _packaged_type_names(graph_model)
        if packaged is None:
            raise ValueError("The packaged model does not declare its type names")
        if type_names is None:
            raise ValueError(
                "PackagedBackbone needs the SpinGNN type_names to check them against "
                f"the packaged type names {packaged}"
            )
        if list(type_names) != packaged:
            raise ValueError(
                "SpinGNN type names must equal the packaged type names in the same "
                f"order; got {list(type_names)}, package has {packaged}"
            )

    @staticmethod
    def _check_r_max(graph_model, r_max):
        packaged = _packaged_r_max(graph_model)
        if r_max is None or packaged is None:
            return
        if not math.isclose(float(r_max), packaged, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"SpinGNN r_max={r_max} differs from the packaged r_max={packaged}; "
                "the neighbor lists must use the packaged cutoff"
            )

    @staticmethod
    def _resolve_input_fields(importer, graph_model) -> List[str]:
        declared = getattr(graph_model, "model_input_fields", None)
        declared = list(_PROVIDED_FIELDS if declared is None else declared)
        try:
            example = importer.load_pickle(
                "model", "example_data.pkl", map_location="cpu"
            )
        except RuntimeError:  # torch.package: resource not found in the archive
            example = None
            logging.warning(
                "Packaged model has no model/example_data.pkl; input fields are not "
                "checked against example data"
            )
        if example is not None:
            needed = sorted(k for k in example if k in declared)
            unsupported = [k for k in needed if k not in _PROVIDED_FIELDS]
            if unsupported:
                raise ValueError(
                    f"The packaged model takes input fields {unsupported} (from its "
                    f"example data) that PackagedBackbone cannot provide; it provides "
                    f"{sorted(_PROVIDED_FIELDS)}"
                )
        fields = [k for k in declared if k in _PROVIDED_FIELDS]
        logging.debug(f"PackagedBackbone passes packaged input fields {fields}")
        return fields

    def _parse_taps(self, taps: Mapping[str, TapSpec]) -> Dict[str, Any]:
        stop_stage = self.func[self.stop_after]
        stage_out = getattr(stop_stage, "irreps_out", None)
        tap_irreps: Dict[str, Any] = {}
        for field, spec in taps.items():
            if isinstance(spec, str):
                spec = {"key": spec}
            spec = dict(spec)
            unknown = set(spec) - {"key", "module", "irreps"}
            if unknown or (("key" in spec) == ("module" in spec)):
                raise ValueError(
                    f"Tap {field!r} must give exactly one of 'key' or 'module' "
                    f"(and optionally 'irreps'); got {spec}"
                )
            if "key" in spec:
                key = str(spec["key"])
                if stage_out is not None and key not in stage_out:
                    raise ValueError(
                        f"Tap {field!r}: packaged field {key!r} is not produced by the "
                        f"stages up to {self.stop_after!r}"
                    )
                default = None if stage_out is None else stage_out[key]
                self._key_taps.append((field, key))
            else:
                path = str(spec["module"])
                if path.split(".")[0] not in self._run_stages:
                    raise ValueError(
                        f"Tap {field!r}: module {path!r} is not inside the stages run "
                        f"up to {self.stop_after!r} ({self._run_stages})"
                    )
                module = self.func.get_submodule(path)
                default = getattr(module, "irreps_out", None)
                if default is not None and hasattr(module, "mul"):
                    default = _channel_irreps(default)
                tap = _ModuleTap(field, path)
                module.register_forward_hook(tap)
                self._module_taps.append(tap)
            irreps = spec.get("irreps", default)
            if "irreps" not in spec and irreps is None:
                raise ValueError(
                    f"Tap {field!r}: cannot infer irreps; give them as 'irreps'"
                )
            tap_irreps[field] = _to_irreps(irreps)
        return tap_irreps

    def _set_trainable(self, prefixes: List[str]):
        used = {p: 0 for p in prefixes}
        n_train = n_frozen = 0
        for name, param in self.func.named_parameters():
            # name relative to the packaged network, and the same without its stage
            within_stage = name.split(".", 1)[1] if "." in name else name
            hits = [
                p for p in prefixes if name.startswith(p) or within_stage.startswith(p)
            ]
            for p in hits:
                used[p] += 1
            param.requires_grad_(bool(hits))
            if hits:
                n_train += param.numel()
            else:
                n_frozen += param.numel()
        unused = [p for p, n in used.items() if n == 0]
        if unused:
            raise ValueError(
                f"trainable_prefixes {unused} match no parameter of the packaged network"
            )
        logging.info(
            f"PackagedBackbone: {n_train} trainable and {n_frozen} frozen packaged parameters"
        )

    # -- runtime --

    def train(self, mode: bool = True):
        """Set this module's mode; the packaged submodules always stay in eval mode."""
        super().train(mode)
        self.func.eval()
        return self

    def __getstate__(self):
        # The compiled callable is bound to this instance; a copy or an
        # unpickled module compiles its own on first use.
        state = super().__getstate__()
        state["_compiled_stages"] = None
        return state

    def _forward_stages(self, packaged: Dict[str, torch.Tensor]):
        for name in self._run_stages:
            packaged = self.func[name](packaged)
        return packaged

    def _run_packaged_stages(self, packaged: Dict[str, torch.Tensor]):
        if self._torch_compile_kwargs is None:
            return self._forward_stages(packaged)
        if self._compiled_stages is None:
            self._compiled_stages = torch.compile(
                self._forward_stages, **self._torch_compile_kwargs
            )
        for key, dims in _STATIC_INPUT_DIMS:
            value = packaged.get(key)
            if value is not None:
                for dim in dims:
                    torch._dynamo.mark_static(value, dim)
        return self._compiled_stages(packaged)

    def _translate(self, data: AtomicDataDict.Type) -> Dict[str, torch.Tensor]:
        pos = data[AtomicDataDict.POSITIONS_KEY]
        if AtomicDataDict.BATCH_KEY in data:
            batch = data[AtomicDataDict.BATCH_KEY].reshape(-1)
        else:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        if _BATCH_PTR_FIELD in data:
            num_frames = data[_BATCH_PTR_FIELD].numel() - 1
        elif AtomicDataDict.NUM_BATCH_KEY in data:
            num_frames = int(data[AtomicDataDict.NUM_BATCH_KEY])
        else:
            num_frames = int(batch.max()) + 1 if batch.numel() > 0 else 1
        values = {
            "pos": pos,
            "edge_index": data[AtomicDataDict.EDGE_INDEX_KEY],
            "atom_types": data[AtomicDataDict.ATOM_TYPE_KEY].reshape(-1),
            "batch": batch,
            _PACKAGED_NUM_ATOMS_FIELD: torch.bincount(batch, minlength=num_frames),
        }
        if AtomicDataDict.CELL_KEY in data:
            values["cell"] = data[AtomicDataDict.CELL_KEY].reshape(-1, 3, 3)
        if AtomicDataDict.EDGE_CELL_SHIFT_KEY in data:
            values["edge_cell_shift"] = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
        return {k: values[k] for k in self._input_fields if k in values}

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        packaged = self._translate(data)
        for tap in self._module_taps:
            tap.start()
        try:
            packaged = self._run_packaged_stages(packaged)
        finally:
            # also on errors: no tap keeps recording or holds a tensor
            captured = [(tap, tap.stop()) for tap in self._module_taps]

        for field, key in self._key_taps:
            data[field] = packaged[key]
        for tap, output in captured:
            if output is None:
                raise RuntimeError(
                    f"Tap {tap.field!r}: module {tap.path!r} did not run"
                )
            data[tap.field] = output
        return data
