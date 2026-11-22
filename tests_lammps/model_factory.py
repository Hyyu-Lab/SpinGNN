"""Build deterministic random-weight deployment models without training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from helpers import CASES, CaseSpec, python_reference
from nequip.data import AtomicDataDict
from nequip.model import model_from_config
from nequip.nn import RescaleOutput
from nequip.scripts import deploy
from nequip.scripts.train import default_config
from nequip.utils import Config
from nequip.utils._global_options import _set_global_options
from nequip.utils.pt2_deploy import build_pt2, build_pt2_from_model
from nequip.utils.versions import check_code_version

MODEL_SEED = 20260827


def _deployment_metadata(config: Config, model) -> dict[str, bytes]:
    required_inputs = sorted(set(model.irreps_in) | {"atom_types"})
    provided_outputs = sorted(model.irreps_out)
    metadata = deploy.assemble_deploy_metadata(
        config, model, provided_outputs, required_inputs
    )
    return {key: value.encode("ascii") for key, value in metadata.items()}


def build_deployed_model(
    case: CaseSpec,
    output_path: Path,
    *,
    fx_compatible: bool = False,
    config_path: Path | None = None,
    config_overrides: Mapping[str, object] | None = None,
) -> Path:
    """Build, deploy, reload, and finite-check one random-weight model."""

    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
    config = Config.from_file(
        str(case.config_path if config_path is None else config_path),
        defaults=default_config,
    )
    for key, value in (config_overrides or {}).items():
        config[key] = value
    if fx_compatible:
        optimization_defaults = dict(config.get("e3nn_optimization_defaults", {}))
        optimization_defaults["jit_script_fx"] = False
        config["e3nn_optimization_defaults"] = optimization_defaults
    _set_global_options(config)
    check_code_version(config, add_to_config=True)
    model = model_from_config(config, initialize=False, deploy=True)
    metadata = _deployment_metadata(config, model)
    compiled = deploy._compile_for_deploy(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(compiled, output_path, _extra_files=metadata)

    _, loaded_metadata = deploy.load_deployed_model(
        output_path,
        device="cpu",
        freeze=False,
        set_global_options=False,
    )
    required_metadata = (
        deploy.SCHEMA_VERSION_KEY,
        deploy.PROVIDED_OUTPUTS_KEY,
        deploy.REQUIRED_INPUTS_KEY,
        deploy.DEFAULT_DTYPE_KEY,
        deploy.R_MAX_KEY,
        deploy.TYPE_NAMES_KEY,
    )
    missing = [key for key in required_metadata if not loaded_metadata[key]]
    if missing:
        raise ValueError(f"deployed model metadata missing keys: {missing}")
    python_reference(output_path, case, device="cpu")
    return output_path


def build_all_models(output_dir: Path) -> dict[str, Path]:
    return {
        case_name: build_deployed_model(case, output_dir / f"{case_name}.pth")
        for case_name, case in CASES.items()
    }


def build_pt2_model(
    case: CaseSpec,
    output_path: Path,
    *,
    target_device: str,
    config_path: Path | None = None,
) -> Path:
    """Build one deterministic PT2 package for the requested pair device."""

    if target_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA-target PT2 export is unavailable because torch.cuda is unavailable"
        )
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
    return build_pt2(
        case.config_path if config_path is None else config_path,
        output_path,
        device=target_device,
    )


def build_deployment_model_pair(
    case: CaseSpec,
    scripted_path: Path,
    pt2_path: Path,
    *,
    target_device: str,
    config_path: Path | None = None,
) -> tuple[Path, Path]:
    """Export TorchScript and PT2 from one exact model instance."""

    if target_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA-target PT2 export is unavailable because torch.cuda is unavailable"
        )
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
    config = Config.from_file(
        str(case.config_path if config_path is None else config_path),
        defaults=default_config,
    )
    optimization_defaults = dict(config.get("e3nn_optimization_defaults", {}))
    optimization_defaults["jit_script_fx"] = False
    config["e3nn_optimization_defaults"] = optimization_defaults
    _set_global_options(config)
    check_code_version(config, add_to_config=True)
    model = model_from_config(config, initialize=False, deploy=True).eval()

    build_pt2_from_model(
        model,
        config,
        pt2_path,
        device=target_device,
    )
    metadata = _deployment_metadata(config, model)
    scripted_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(
        deploy._compile_for_deploy(model),
        scripted_path,
        _extra_files=metadata,
    )
    python_reference(scripted_path, case, device="cpu")
    return scripted_path, pt2_path


def build_shifted_deployment_models(
    case: CaseSpec,
    output_dir: Path,
    *,
    target_device: str,
    shift_ev: float = 13.0,
) -> tuple[Path, Path]:
    """Build TorchScript/PT2 artifacts with a non-extensive total-energy shift."""

    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
    config = Config.from_file(str(case.config_path), defaults=default_config)
    optimization_defaults = dict(config.get("e3nn_optimization_defaults", {}))
    optimization_defaults["jit_script_fx"] = False
    config["e3nn_optimization_defaults"] = optimization_defaults
    _set_global_options(config)
    check_code_version(config, add_to_config=True)
    base = model_from_config(config, initialize=False, deploy=True)
    model = RescaleOutput(
        base,
        shift_keys=(AtomicDataDict.TOTAL_ENERGY_KEY,),
        shift_by=shift_ev,
    ).eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    scripted_path = output_dir / f"{case.name}-shifted.pth"
    pt2_path = output_dir / f"{case.name}-shifted-{target_device}.pt2"
    build_pt2_from_model(
        model,
        config,
        pt2_path,
        device=target_device,
        periodic=False,
    )
    metadata = _deployment_metadata(config, model)
    torch.jit.save(
        deploy._compile_for_deploy(model),
        scripted_path,
        _extra_files=metadata,
    )
    return scripted_path, pt2_path


def resave_with_metadata(
    source_path: Path,
    output_path: Path,
    overrides: Mapping[str, str] | None = None,
    *,
    omit_all_metadata: bool = False,
) -> Path:
    model, metadata = deploy.load_deployed_model(
        source_path,
        device="cpu",
        freeze=False,
        set_global_options=False,
    )
    extra_files: dict[str, bytes] = {}
    if not omit_all_metadata:
        extra_files = {key: value.encode("ascii") for key, value in metadata.items()}
        for key, value in (overrides or {}).items():
            extra_files[key] = value.encode("ascii")
    torch.jit.save(model, output_path, _extra_files=extra_files)
    return output_path
