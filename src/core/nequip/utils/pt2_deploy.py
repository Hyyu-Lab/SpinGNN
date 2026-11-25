"""PT2/AOTInductor deployment construction for SpinGNN models."""

from __future__ import annotations

import hashlib
import pathlib
import zipfile
from typing import Optional, Union

import torch

from nequip.scripts import deploy
from nequip.utils.fx_compat import (
    PERIODIC_INPUT_SIGNATURE,
    expand_deploy_graph,
    expand_periodic_deploy_graph,
    make_deploy_sample,
    make_periodic_deploy_sample,
    strip_saved_tensor_detach,
)


PathLike = Union[pathlib.Path, str]
PT2_MANIFEST_RECORD = "spingnn_deploy.meta"
_PT2_MANIFEST_KEYS = (
    deploy.SCHEMA_VERSION_KEY,
    deploy.INPUT_SIGNATURE_KEY,
    deploy.OUTPUT_SIGNATURE_KEY,
    deploy.PROVIDED_OUTPUTS_KEY,
    deploy.REQUIRED_INPUTS_KEY,
    deploy.DEFAULT_DTYPE_KEY,
    deploy.TARGET_DEVICE_KEY,
    deploy.RESCALE_LAYERS_KEY,
    deploy.R_MAX_KEY,
    deploy.NEQUIP_NUM_INTERACTIONS_KEY,
    deploy.N_SPECIES_KEY,
    deploy.TYPE_NAMES_KEY,
    deploy.TF32_KEY,
    deploy.ENERGY_UNITS_KEY,
    deploy.LENGTH_UNITS_KEY,
    deploy.HAS_SPIN_KEY,
    deploy.HAS_BEC_KEY,
    deploy.EDGE_PRUNE_KEY,
    deploy.PACKAGE_PAYLOAD_SHA256_KEY,
    deploy.CUDA_COMPUTE_CAPABILITY_KEY,
    deploy.CUDA_CODE_KIND_KEY,
    deploy.TORCH_CUDA_VERSION_KEY,
)


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embed_pt2_manifest(path: pathlib.Path, metadata) -> None:
    manifest = {key: metadata[key] for key in _PT2_MANIFEST_KEYS if key in metadata}
    encoded = deploy.encode_deploy_metadata(manifest).encode("utf-8")
    with zipfile.ZipFile(path, "r") as archive:
        roots = {
            name.split("/", 1)[0]
            for name in archive.namelist()
            if "/" in name and name.split("/", 1)[0]
        }
        if len(roots) != 1:
            raise RuntimeError(
                f"PT2 package has ambiguous archive roots: {sorted(roots)}"
            )
        record = f"{next(iter(roots))}/{PT2_MANIFEST_RECORD}"
        if record in archive.namelist():
            raise RuntimeError(f"PT2 package already contains {PT2_MANIFEST_RECORD}")
    info = zipfile.ZipInfo(record, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o600 << 16
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(info, encoded)


# AOTInductor selects int64 indexing only when a buffer's numel has a finite
# value-range upper bound (< 1e20, see torch/_inductor/utils.py
# expr_fits_within_32bit). With an int64-scale Dim max, per-edge buffers with
# >= 11 elements/edge exceed 1e20 at the declared bound, so inductor falls back
# to the trace-sample hint and emits int32 indexing — an out-of-bounds access
# once n_edges * features_per_edge crosses 2**31 (observed at 32k atoms).
# 2**40 keeps every realistic per-atom/per-edge product below 1e20, forcing
# 64-bit-safe codegen while remaining far beyond any deployable system size.
_DYNAMIC_DIM_MAX = 2**40


def _dynamic_shapes(input_signature):
    n_atoms = torch.export.Dim("n_atoms", min=1, max=_DYNAMIC_DIM_MAX)
    n_edges = torch.export.Dim("n_edges", min=2, max=_DYNAMIC_DIM_MAX)
    shapes = {
        "pos": {0: n_atoms},
        "spin": {0: n_atoms},
        "edge_index": {1: n_edges},
        "atom_types": {0: n_atoms},
    }
    return tuple(shapes[name] for name in input_signature)


def _periodic_dynamic_shapes():
    n_atoms = torch.export.Dim("n_atoms", min=1, max=_DYNAMIC_DIM_MAX)
    n_edges = torch.export.Dim("n_edges", min=2, max=_DYNAMIC_DIM_MAX)
    return (
        {0: n_atoms},
        {0: n_atoms},
        {1: n_edges},
        {0: n_edges},
        {},
        {},
        {0: n_atoms},
    )


def _config_uses_periodic_inputs(config) -> bool:
    builders = config.get("model_builders", [])
    return not any("allegro" in str(builder).lower() for builder in builders)


def _materialize_scalar_unsqueeze_constants(graph_module):
    for node in list(graph_module.graph.nodes):
        if node.target != torch.ops.aten.unsqueeze.default:
            continue
        source, dim = node.args
        if not isinstance(source, torch.fx.Node) or source.op != "get_attr" or dim != 0:
            continue
        value = getattr(graph_module, source.target)
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            continue
        if len(source.users) != 1:
            continue
        setattr(graph_module, source.target, value.reshape(1))
        node.replace_all_uses_with(source)
        graph_module.graph.erase_node(node)
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


def build_pt2(
    config_or_traindir: PathLike,
    out_path: PathLike,
    device: Union[str, torch.device] = "cpu",
    periodic: Optional[bool] = None,
) -> pathlib.Path:
    """Build an inference-only dynamic-shape AOTI package and metadata sidecar.

    Non-Allegro model builders default to the periodic NequIP input signature.
    """
    source = pathlib.Path(config_or_traindir)
    model, config = deploy.load_model_for_deploy(
        train_dir=source if source.is_dir() else None,
        model_path=None if source.is_dir() else source,
        device=device,
        fx_compatible=True,
    )
    return build_pt2_from_model(
        model,
        config,
        out_path,
        device=device,
        periodic=periodic,
    )


def build_pt2_from_model(
    model: torch.nn.Module,
    config,
    out_path: PathLike,
    device: Union[str, torch.device] = "cpu",
    periodic: Optional[bool] = None,
) -> pathlib.Path:
    """Package an already constructed deploy model with the standard PT2 ABI."""

    output = pathlib.Path(out_path)
    deploy._enable_equal_cutoff_dense_edge_prune(
        model, config, strict_lammps_prefilter=True
    )
    target_device = torch.device(device).type
    if output.suffix != ".pt2":
        raise ValueError(f"PT2 output path must end in .pt2: {output}")
    if target_device not in ("cpu", "cuda"):
        raise ValueError(
            f"PT2 target device must be cpu or cuda, got {target_device!r}"
        )
    if target_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA PT2 export requested but CUDA is unavailable")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    if periodic is None:
        periodic = _config_uses_periodic_inputs(config)
    if periodic:
        input_signature = PERIODIC_INPUT_SIGNATURE
        sample = make_periodic_deploy_sample(model)
        graph_module = expand_periodic_deploy_graph(model, sample)
        dynamic_shapes = _periodic_dynamic_shapes()
    else:
        sample = make_deploy_sample(model)
        graph_module = expand_deploy_graph(model, sample)
        input_signature = graph_module.input_signature
        dynamic_shapes = _dynamic_shapes(input_signature)
    graph_module = strip_saved_tensor_detach(graph_module)
    graph_module = _materialize_scalar_unsqueeze_constants(graph_module)
    exported_program = torch.export.export(
        graph_module,
        sample,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    package_path = torch._inductor.aoti_compile_and_package(
        exported_program,
        package_path=str(output),
    )
    package = pathlib.Path(package_path)
    if package.resolve() != output.resolve():
        raise RuntimeError(
            f"AOTI wrote an unexpected package path: {package} != {output}"
        )

    provided_outputs = tuple(graph_module.output_signature)
    metadata = deploy.assemble_deploy_metadata(
        config,
        model,
        provided_outputs,
        input_signature,
    )
    metadata[deploy.INPUT_SIGNATURE_KEY] = " ".join(input_signature)
    metadata[deploy.OUTPUT_SIGNATURE_KEY] = " ".join(provided_outputs)
    metadata[deploy.TARGET_DEVICE_KEY] = target_device
    metadata[deploy.RESCALE_LAYERS_KEY] = str(graph_module.num_rescale_layers)
    if target_device == "cuda":
        major, minor = torch.cuda.get_device_capability(torch.device(device))
        metadata[deploy.CUDA_COMPUTE_CAPABILITY_KEY] = f"{major}.{minor}"
        # AOTInductor emits a native target for the export device. Do not claim
        # broader PTX compatibility unless package inspection proves it.
        metadata[deploy.CUDA_CODE_KIND_KEY] = "cubin"
        metadata[deploy.TORCH_CUDA_VERSION_KEY] = str(torch.version.cuda)
    metadata[deploy.PACKAGE_PAYLOAD_SHA256_KEY] = _file_sha256(output)
    _embed_pt2_manifest(output, metadata)
    metadata[deploy.PACKAGE_SHA256_KEY] = _file_sha256(output)
    deploy.write_deploy_metadata_sidecar(output, metadata)
    return output
