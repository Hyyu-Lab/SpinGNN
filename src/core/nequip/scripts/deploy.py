import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final
from typing import Tuple, Dict, Union, Mapping
import argparse
import pathlib
import logging
import re
import yaml
import itertools

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

import torch

import ase.data

from e3nn.util.jit import script

from nequip.model import model_from_config
from nequip.train import Trainer
from nequip.utils import Config
from nequip.utils.versions import check_code_version, get_config_code_versions
from nequip.scripts.train import default_config
from nequip.utils._global_options import (
    _set_global_options,
    check_deployable_model_dtype,
    global_options_context,
)

CONFIG_KEY: Final[str] = "config"
NEQUIP_VERSION_KEY: Final[str] = "nequip_version"
TORCH_VERSION_KEY: Final[str] = "torch_version"
E3NN_VERSION_KEY: Final[str] = "e3nn_version"
CODE_COMMITS_KEY: Final[str] = "code_commits"
R_MAX_KEY: Final[str] = "r_max"
NEQUIP_NUM_INTERACTIONS_KEY: Final[str] = "nequip_num_interactions"
N_SPECIES_KEY: Final[str] = "n_species"
TYPE_NAMES_KEY: Final[str] = "type_names"
JIT_BAILOUT_KEY: Final[str] = "_jit_bailout_depth"
JIT_FUSION_STRATEGY: Final[str] = "_jit_fusion_strategy"
TF32_KEY: Final[str] = "allow_tf32"
SCHEMA_VERSION_KEY: Final[str] = "schema_version"
PROVIDED_OUTPUTS_KEY: Final[str] = "provided_outputs"
REQUIRED_INPUTS_KEY: Final[str] = "required_inputs"
DEFAULT_DTYPE_KEY: Final[str] = "default_dtype"
ENERGY_UNITS_KEY: Final[str] = "energy_units"
LENGTH_UNITS_KEY: Final[str] = "length_units"
HAS_SPIN_KEY: Final[str] = "has_spin"
HAS_BEC_KEY: Final[str] = "has_bec"
EDGE_PRUNE_KEY: Final[str] = "edge_prune"
INPUT_SIGNATURE_KEY: Final[str] = "input_signature"
OUTPUT_SIGNATURE_KEY: Final[str] = "output_signature"
TARGET_DEVICE_KEY: Final[str] = "target_device"
RESCALE_LAYERS_KEY: Final[str] = "rescale_layers"
PACKAGE_SHA256_KEY: Final[str] = "package_sha256"
PACKAGE_PAYLOAD_SHA256_KEY: Final[str] = "package_payload_sha256"
CUDA_COMPUTE_CAPABILITY_KEY: Final[str] = "cuda_compute_capability"
CUDA_CODE_KIND_KEY: Final[str] = "cuda_code_kind"
TORCH_CUDA_VERSION_KEY: Final[str] = "torch_cuda_version"

DEPLOY_SCHEMA_VERSION: Final[str] = "1"

_ALL_METADATA_KEYS = [
    CONFIG_KEY,
    NEQUIP_VERSION_KEY,
    TORCH_VERSION_KEY,
    E3NN_VERSION_KEY,
    R_MAX_KEY,
    NEQUIP_NUM_INTERACTIONS_KEY,
    N_SPECIES_KEY,
    TYPE_NAMES_KEY,
    JIT_BAILOUT_KEY,
    JIT_FUSION_STRATEGY,
    TF32_KEY,
    SCHEMA_VERSION_KEY,
    PROVIDED_OUTPUTS_KEY,
    REQUIRED_INPUTS_KEY,
    DEFAULT_DTYPE_KEY,
    ENERGY_UNITS_KEY,
    LENGTH_UNITS_KEY,
    HAS_SPIN_KEY,
    HAS_BEC_KEY,
    EDGE_PRUNE_KEY,
]


def _resolve_pt2_device(requested_device: str) -> str:
    """Resolve the CLI's current-machine PT2 target selection."""
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def _compile_for_deploy(model):
    model.eval()

    if not isinstance(model, torch.jit.ScriptModule):
        model = script(model)

    return model


def _enable_equal_cutoff_dense_edge_prune(
    model, config, *, strict_lammps_prefilter: bool
) -> bool:
    """Enable the source dense fast path only for the strict spin pair ABI."""
    if not strict_lammps_prefilter or EDGE_PRUNE_KEY in config:
        return False
    if "spin" not in model.irreps_in:
        return False

    edge_modules = [
        module for module in model.modules() if hasattr(module, EDGE_PRUNE_KEY)
    ]
    if not edge_modules:
        return False

    reference_cutoff = None
    for module in edge_modules:
        cutoffs = getattr(module, "per_layer_cutoffs", None)
        if cutoffs is None or cutoffs.numel() == 0:
            return False
        first = cutoffs.reshape(-1)[0]
        if not torch.equal(cutoffs, first.expand_as(cutoffs)):
            return False
        if reference_cutoff is not None and not torch.equal(first, reference_cutoff):
            return False
        reference_cutoff = first

    for module in edge_modules:
        module.edge_prune = "dense"
    config[EDGE_PRUNE_KEY] = "dense"
    logging.info(
        "Enabled edge_prune=dense: all per-layer cutoffs are equal and the "
        "spin LAMMPS ABI strictly prefilters neighbor edges"
    )
    return True


def assemble_deploy_metadata(
    config,
    model,
    provided_outputs,
    required_inputs,
) -> Dict[str, str]:
    """Assemble metadata shared by TorchScript and PT2 deployment."""
    # Signatures are supplied explicitly because PT2 exports a tensor-only
    # wrapper. Receptive depth comes from the original EnergyModel builder,
    # before scripting/freezing can erase its construction metadata.
    provided_outputs = tuple(provided_outputs)
    required_inputs = tuple(required_inputs)
    has_spin = "spin_forces" in provided_outputs
    signature_has_spin = "spin" in required_inputs
    if has_spin != signature_has_spin:
        raise ValueError(
            f"has_spin={int(has_spin)} is incompatible with required_inputs "
            f"{required_inputs!r}: spin input presence must match spin_forces output"
        )
    metadata: Dict[str, str] = {}
    code_versions, code_commits = get_config_code_versions(config)
    for code, version in code_versions.items():
        metadata[code + "_version"] = version
    if len(code_commits) > 0:
        metadata[CODE_COMMITS_KEY] = ";".join(
            f"{key}={value}" for key, value in code_commits.items()
        )

    metadata[R_MAX_KEY] = str(float(config["r_max"]))
    if isinstance(model, torch.nn.Module):
        depths = [
            vars(module)[NEQUIP_NUM_INTERACTIONS_KEY]
            for module in model.modules()
            if NEQUIP_NUM_INTERACTIONS_KEY in vars(module)
        ]
        if depths:
            if len(depths) != 1 or not isinstance(depths[0], int) or depths[0] <= 0:
                raise ValueError(
                    "NequIP deployment requires one EnergyModel core with a "
                    "positive nequip_num_interactions construction attribute"
                )
            metadata[NEQUIP_NUM_INTERACTIONS_KEY] = str(depths[0])
    if "allowed_species" in config:
        n_species = len(config["allowed_species"])
        type_names = [
            ase.data.chemical_symbols[atomic_num]
            for atomic_num in config["allowed_species"]
        ]
    else:
        n_species = str(config["num_types"])
        type_names = config["type_names"]
    metadata[N_SPECIES_KEY] = str(n_species)
    metadata[TYPE_NAMES_KEY] = " ".join(type_names)

    metadata[JIT_BAILOUT_KEY] = str(config[JIT_BAILOUT_KEY])
    if int(torch.__version__.split(".")[1]) >= 11 and JIT_FUSION_STRATEGY in config:
        metadata[JIT_FUSION_STRATEGY] = ";".join(
            "%s,%i" % entry for entry in config[JIT_FUSION_STRATEGY]
        )
    metadata[TF32_KEY] = str(int(config["allow_tf32"]))
    metadata[SCHEMA_VERSION_KEY] = DEPLOY_SCHEMA_VERSION
    metadata[PROVIDED_OUTPUTS_KEY] = " ".join(provided_outputs)
    metadata[REQUIRED_INPUTS_KEY] = " ".join(required_inputs)
    metadata[DEFAULT_DTYPE_KEY] = str(config[DEFAULT_DTYPE_KEY])
    metadata[ENERGY_UNITS_KEY] = "eV"
    metadata[LENGTH_UNITS_KEY] = "Angstrom"
    metadata[HAS_SPIN_KEY] = str(int(has_spin))
    metadata[HAS_BEC_KEY] = str(int("born_effective_charges" in provided_outputs))
    edge_prune = config.get(EDGE_PRUNE_KEY, "nonzero")
    if edge_prune not in ("nonzero", "dense"):
        raise ValueError(f"edge_prune must be 'nonzero' or 'dense', got {edge_prune!r}")
    metadata[EDGE_PRUNE_KEY] = edge_prune
    metadata[CONFIG_KEY] = yaml.dump(dict(config))
    return metadata


def _escape_sidecar_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def _unescape_sidecar_value(value: str) -> str:
    decoded = []
    index = 0
    while index < len(value):
        if value[index] != "\\":
            decoded.append(value[index])
            index += 1
            continue
        if index + 1 == len(value):
            raise ValueError("metadata sidecar value ends with an incomplete escape")
        escaped = value[index + 1]
        decoded.append({"n": "\n", "t": "\t", "\\": "\\"}.get(escaped, escaped))
        index += 2
    return "".join(decoded)


def encode_deploy_metadata(metadata: Mapping[str, str]) -> str:
    """Encode deployment metadata as deterministic tab-separated text."""

    lines = []
    for key in sorted(metadata):
        if not key or any(character in key for character in "\t\r\n"):
            raise ValueError(f"invalid metadata key: {key!r}")
        lines.append(f"{key}\t{_escape_sidecar_value(str(metadata[key]))}\n")
    return "".join(lines)


def write_deploy_metadata_sidecar(
    model_path: Union[pathlib.Path, str], metadata: Mapping[str, str]
) -> pathlib.Path:
    """Write ``<model_path>.meta`` as deterministic tab-separated text."""
    required = {
        INPUT_SIGNATURE_KEY,
        OUTPUT_SIGNATURE_KEY,
        TARGET_DEVICE_KEY,
        PACKAGE_SHA256_KEY,
        PACKAGE_PAYLOAD_SHA256_KEY,
        EDGE_PRUNE_KEY,
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"PT2 metadata is missing required keys: {missing}")
    if re.fullmatch(r"[0-9a-f]{64}", metadata[PACKAGE_SHA256_KEY]) is None:
        raise ValueError("PT2 metadata package_sha256 must be 64 lowercase hex digits")
    if re.fullmatch(r"[0-9a-f]{64}", metadata[PACKAGE_PAYLOAD_SHA256_KEY]) is None:
        raise ValueError(
            "PT2 metadata package_payload_sha256 must be 64 lowercase hex digits"
        )
    if metadata[EDGE_PRUNE_KEY] not in ("nonzero", "dense"):
        raise ValueError("PT2 metadata edge_prune must be 'nonzero' or 'dense'")
    if metadata[TARGET_DEVICE_KEY] == "cuda":
        cuda_required = {
            CUDA_COMPUTE_CAPABILITY_KEY,
            CUDA_CODE_KIND_KEY,
            TORCH_CUDA_VERSION_KEY,
        }
        missing_cuda = sorted(cuda_required - set(metadata))
        if missing_cuda:
            raise ValueError(
                f"CUDA PT2 metadata is missing required keys: {missing_cuda}"
            )
        if (
            re.fullmatch(r"[1-9][0-9]*\.[0-9]+", metadata[CUDA_COMPUTE_CAPABILITY_KEY])
            is None
        ):
            raise ValueError(
                "PT2 metadata cuda_compute_capability must have major.minor format"
            )
        if (
            re.fullmatch(r"[1-9][0-9]*\.[0-9]+", metadata[TORCH_CUDA_VERSION_KEY])
            is None
        ):
            raise ValueError(
                "PT2 metadata torch_cuda_version must have major.minor format"
            )
        if metadata[CUDA_CODE_KIND_KEY] != "cubin":
            raise ValueError("PT2 metadata cuda_code_kind must be 'cubin'")
    sidecar_path = pathlib.Path(str(model_path) + ".meta")
    with sidecar_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encode_deploy_metadata(metadata))
    return sidecar_path


def read_deploy_metadata_sidecar(
    model_path: Union[pathlib.Path, str],
) -> Dict[str, str]:
    """Read the Python form of the PT2 metadata sidecar contract."""
    sidecar_path = pathlib.Path(str(model_path) + ".meta")
    metadata: Dict[str, str] = {}
    with sidecar_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\n")
            if "\t" not in line:
                raise ValueError(
                    f"malformed metadata sidecar line {line_number}: missing tab"
                )
            key, value = line.split("\t", 1)
            if not key or key in metadata:
                raise ValueError(
                    f"malformed metadata sidecar line {line_number}: invalid key {key!r}"
                )
            metadata[key] = _unescape_sidecar_value(value)
    return metadata


def load_deployed_model(
    model_path: Union[pathlib.Path, str],
    device: Union[str, torch.device] = "cpu",
    freeze: bool = True,
    set_global_options: Union[str, bool] = "warn",
) -> Tuple[torch.jit.ScriptModule, Dict[str, str]]:
    r"""Load a deployed model.

    Args:
        model_path: the path to the deployed model's ``.pth`` file.

    Returns:
        model, metadata dictionary
    """
    metadata = {k: "" for k in _ALL_METADATA_KEYS}
    try:
        model = torch.jit.load(model_path, map_location=device, _extra_files=metadata)
    except RuntimeError as e:
        raise ValueError(
            f"{model_path} does not seem to be a deployed NequIP model file. Did you forget to deploy it using `spingnn-deploy`? \n\n(Underlying error: {e})"
        )
    # Confirm nequip made it
    if not metadata[NEQUIP_VERSION_KEY]:
        raise ValueError(
            f"{model_path} does not seem to be a deployed NequIP model file"
        )
    # Confirm its TorchScript
    assert isinstance(model, torch.jit.ScriptModule)
    # Make sure we're in eval mode
    model.eval()
    # Freeze on load:
    if freeze and hasattr(model, "training"):
        # hasattr is how torch checks whether model is unfrozen
        # only freeze if already unfrozen
        model = torch.jit.freeze(model)
    # Everything we store right now is ASCII, so decode for printing
    metadata = {k: v.decode("ascii") for k, v in metadata.items()}
    edge_prune = metadata[EDGE_PRUNE_KEY]
    if edge_prune != "" and edge_prune not in ("nonzero", "dense"):
        raise ValueError(
            "deployed model metadata edge_prune must be 'nonzero' or 'dense'"
        )
    if edge_prune != "":
        embedded_config = yaml.load(metadata[CONFIG_KEY], Loader=yaml.BaseLoader)
        if not isinstance(embedded_config, Mapping):
            raise ValueError("deployed model metadata config must be a mapping")
        config_edge_prune = embedded_config.get(EDGE_PRUNE_KEY, "nonzero")
        if edge_prune != config_edge_prune:
            raise ValueError(
                "deployed model metadata edge_prune does not match embedded config: "
                f"{edge_prune!r} != {config_edge_prune!r}"
            )
    # Set up global settings:
    assert set_global_options in (True, False, "warn")
    if set_global_options:
        global_config_dict = {}
        global_config_dict["allow_tf32"] = bool(int(metadata[TF32_KEY]))
        # JIT strategy
        strategy = metadata.get(JIT_FUSION_STRATEGY, "")
        if strategy != "":
            strategy = [e.split(",") for e in strategy.split(";")]
            strategy = [(e[0], int(e[1])) for e in strategy]
        else:
            strategy = default_config[JIT_FUSION_STRATEGY]
        global_config_dict["_jit_fusion_strategy"] = strategy
        # JIT bailout
        # _set_global_options will check torch version
        jit_bailout: int = metadata.get(JIT_BAILOUT_KEY, "")
        if jit_bailout == "":
            jit_bailout = default_config[JIT_BAILOUT_KEY]
        jit_bailout = int(jit_bailout)
        global_config_dict["_jit_bailout_depth"] = jit_bailout
        # call to actually set the global options
        _set_global_options(
            global_config_dict,
            warn_on_override=set_global_options == "warn",
        )
    return model, metadata


def load_model_for_deploy(
    model_path=None,
    train_dir=None,
    device: Union[str, torch.device] = "cpu",
    fx_compatible: bool = False,
):
    """Load or construct an undeployed model for a deployment backend."""
    if model_path is not None and train_dir is not None:
        raise ValueError("--model and --train-dir cannot both be specified.")
    if train_dir is not None:
        logging.info("Loading best_model from training session...")
        train_dir = pathlib.Path(train_dir)
        config = Config.from_file(str(train_dir / "config.yaml"))
    elif model_path is not None:
        logging.info("Building model from config...")
        config = Config.from_file(str(model_path), defaults=default_config)
    else:
        raise ValueError("one of --train-dir or --model must be given")

    if fx_compatible:
        optimization_defaults = dict(config.get("e3nn_optimization_defaults", {}))
        optimization_defaults["jit_script_fx"] = False
        config["e3nn_optimization_defaults"] = optimization_defaults
    _set_global_options(config)
    # Fresh models have no training provenance to retain; record the code
    # that constructs them so both deployment readers recognize the artifact.
    check_code_version(config, add_to_config=train_dir is None)
    check_deployable_model_dtype(config)

    if train_dir is not None:
        model, _ = Trainer.load_model_from_training_session(
            train_dir,
            model_name="best_model.pth",
            device=device,
            config_dictionary=dict(config) if fx_compatible else None,
        )
    else:
        model = model_from_config(config, deploy=True)
        model.to(device=torch.device(device))
    model.eval()
    return model, config


def main(args=None):
    if __package__:
        from ._source_guard import guard_source_entry
    else:
        from _source_guard import guard_source_entry
    guard_source_entry()

    with global_options_context():
        return _main(args=args)


def _main(args=None):
    parser = argparse.ArgumentParser(
        description="Deploy and view information about previously deployed NequIP models."
    )
    # backward compat for 3.6
    if sys.version_info[1] > 6:
        required = {"required": True}
    else:
        required = {}
    parser.add_argument("--verbose", help="log level", default="INFO", type=str)
    subparsers = parser.add_subparsers(dest="command", title="commands", **required)
    info_parser = subparsers.add_parser(
        "info", help="Get information from a deployed model file"
    )
    info_parser.add_argument(
        "model_path",
        help="Path to a deployed model file.",
        type=pathlib.Path,
    )
    info_parser.add_argument(
        "--print-config",
        help="Print the full config of the model.",
        action="store_true",
    )

    build_parser = subparsers.add_parser("build", help="Build a deployment model")
    build_parser.add_argument(
        "--model",
        help="Path to a YAML file defining a model to deploy. Unless you know why you need to, do not use this option.",
        type=pathlib.Path,
    )
    build_parser.add_argument(
        "--train-dir",
        help="Path to a working directory from a training session to deploy.",
        type=pathlib.Path,
    )
    build_parser.add_argument(
        "out_file",
        help="Output file for deployed model.",
        type=pathlib.Path,
    )

    build_pt2_parser = subparsers.add_parser(
        "build-pt2", help="Build a packaged PT2 deployment model"
    )
    build_pt2_parser.add_argument(
        "--model",
        help="Path to a YAML file defining a model to deploy.",
        type=pathlib.Path,
    )
    build_pt2_parser.add_argument(
        "--train-dir",
        help="Path to a working directory from a training session to deploy.",
        type=pathlib.Path,
    )
    build_pt2_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "Target device for the packaged AOTInductor model. "
            "auto selects CUDA when available, otherwise CPU."
        ),
    )
    build_pt2_parser.add_argument(
        "out_file",
        help="Output .pt2 package path.",
        type=pathlib.Path,
    )

    args = parser.parse_args(args=args)

    logging.basicConfig(level=getattr(logging, args.verbose.upper()))

    if args.command == "info":
        model, metadata = load_deployed_model(
            args.model_path, set_global_options=False, freeze=False
        )
        config = metadata.pop(CONFIG_KEY)
        if args.print_config:
            print(config)
        else:
            metadata_str = "\n".join("  %s: %s" % e for e in metadata.items())
            logging.info(f"Loaded TorchScript model with metadata:\n{metadata_str}\n")
            logging.info(
                f"Model has {sum(p.numel() for p in model.parameters())} weights"
            )
            logging.info(
                f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable weights"
            )
            logging.info(
                f"Model weights and buffers take {sum(p.numel() * p.element_size() for p in itertools.chain(model.parameters(), model.buffers())) / (1024 * 1024):.2f} MB"
            )
            logging.debug(f"Model had config:\n{config}")

    elif args.command == "build":
        model, config = load_model_for_deploy(
            model_path=args.model,
            train_dir=args.train_dir,
            device="cpu",
        )
        _enable_equal_cutoff_dense_edge_prune(
            model, config, strict_lammps_prefilter=True
        )

        required_inputs = sorted(set(model.irreps_in) | {"atom_types"})
        provided_outputs = sorted(model.irreps_out)
        # Scripting hides native construction attributes such as interaction
        # depth, so capture the deployment contract from the original model.
        metadata = assemble_deploy_metadata(
            config, model, provided_outputs, required_inputs
        )

        # -- compile --
        model = _compile_for_deploy(model)
        logging.info("Compiled & optimized model.")

        metadata = {k: v.encode("ascii") for k, v in metadata.items()}
        torch.jit.save(model, args.out_file, _extra_files=metadata)
    elif args.command == "build-pt2":
        from nequip.utils.pt2_deploy import build_pt2

        source = args.train_dir if args.train_dir is not None else args.model
        if source is None:
            raise ValueError("one of --train-dir or --model must be given")
        target_device = _resolve_pt2_device(args.device)
        output = build_pt2(source, args.out_file, device=target_device)
        output = pathlib.Path(output).resolve()
        sidecar = pathlib.Path(str(output) + ".meta")
        metadata = read_deploy_metadata_sidecar(output)
        rescale_layers = int(metadata.get(RESCALE_LAYERS_KEY, "0"))
        if args.device == "auto":
            print(f"PT2 device auto-selected: {target_device}")
            if target_device == "cpu":
                print(
                    "PT2 note: CUDA LAMMPS requires a CUDA-target package; "
                    "export again inside the target CUDA environment."
                )
        print(f"PT2 package: {output}")
        print(f"PT2 sidecar: {sidecar}")
        print(f"PT2 target_device: {target_device}")
        if rescale_layers:
            print("PT2 learned scaling: included " f"({rescale_layers} layer(s))")
        else:
            print("PT2 learned scaling: not present (0 layer(s))")
        print(f"LAMMPS pair_coeff: pair_coeff * * {output} <ELEMENTS...>")
    else:
        raise ValueError

    return


if __name__ == "__main__":
    main()
