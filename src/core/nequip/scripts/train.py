""" Train a network."""
import logging
import argparse
import warnings
import uuid
import os
import time

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

from os.path import isdir
from pathlib import Path

import torch
import torch.distributed as dist

from nequip.model import model_from_config
from nequip.utils import Config
from nequip.data import dataset_from_config
from nequip.utils import load_file
from nequip.utils.test import assert_AtomicData_equivariant
from nequip.utils.versions import check_code_version
from nequip.utils._global_options import (
    _set_global_options,
    cast_model_dtype,
    global_options_context,
    model_dtype_from_config,
)
from nequip.utils.compile_utils import (
    configure_model_for_autocast,
    configure_model_for_compile,
)
from nequip.scripts._logger import set_up_script_logger

default_config = dict(
    root="./",
    run_name="rand_" + str(uuid.uuid4()),
    wandb=False,
    wandb_project="NequIP",
    model_builders=[
        "SimpleIrrepsConfig",
        "EnergyModel",
        "PerSpeciesRescale",
        "ForceOutput",
        "RescaleEnergyEtc",
    ],
    dataset_statistics_stride=1,
    default_dtype="float32",
    # H100 calibration bounds the TF32 force relative error by 8e-3 and the
    # energy error by 5e-4 eV/atom. Use highest/False for publication parity.
    float32_matmul_precision="high",
    allow_tf32=True,
    e3nn_optimization_defaults={"jit_script_fx": False},
    spingnn_compile={
        "enabled": True,
        "scope": "full_fx",
        "dynamic": True,
        "inductor_options": {},
    },
    train_autocast={
        "enabled": False,
        "dtype": "bfloat16",
        "fp32_modules": [
            "RescaleOutput",
            "ParaStressForceSpinOutput",
            "edge_eng_soc_on",
        ],
    },
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    append=False,
    reset_best_metrics=False,
    _jit_bailout_depth=2,  # avoid 20 iters of pain, see https://github.com/pytorch/pytorch/issues/52286
    # Quote from eelison in PyTorch slack:
    # https://pytorch.slack.com/archives/CDZD1FANA/p1644259272007529?thread_ts=1644064449.039479&cid=CDZD1FANA
    # > Right now the default behavior is to specialize twice on static shapes and then on dynamic shapes.
    # > To reduce warmup time you can do something like setFusionStrartegy({{FusionBehavior::DYNAMIC, 3}})
    # > ... Although we would wouldn't really expect to recompile a dynamic shape fusion in a model,
    # > provided broadcasting patterns remain fixed
    # We default to DYNAMIC alone because the number of edges is always dynamic,
    # even if the number of atoms is fixed:
    _jit_fusion_strategy=[("DYNAMIC", 3)],
)

# Runtime-only keys that a restart (``append: true``) may change, or add when
# the checkpoint predates them: the invocation config's value replaces the
# saved one and the change is logged, as for ``max_epochs``. A key that the
# invocation config does not set keeps its saved value. None of these keys
# changes the model parameters, the optimizer state layout, the data or the
# loss, so the restored training state stays valid:
_RESTART_RUNTIME_KEYS = (
    # torch.compile of the packaged backbone stages: parameter names, the
    # state_dict keys and the optimizer/EMA parameter order are the same with
    # and without it (see nequip.nn.PackagedBackbone)
    "packaged_backbone_torch_compile",
    # cadence of per-batch metric materialization and batch CSV/log rows only;
    # epoch metrics, scheduler and early-stopping inputs are unchanged
    "batch_metrics_log_interval",
    # native DDP options: how buffers and gradients are synchronized, not the
    # model, optimizer or data. The DDP wrapper is rebuilt on restart with the
    # merged values; their preconditions are those of a fresh run (e.g. equal
    # buffers on every rank for `ddp_broadcast_buffers: false`)
    "ddp_broadcast_buffers",
    "ddp_static_graph",
    "ddp_gradient_as_bucket_view",
    "ddp_find_unused_parameters",
)


def main(args=None, running_as_script: bool = True):
    if __package__:
        from ._source_guard import guard_source_entry
    else:
        from _source_guard import guard_source_entry
    guard_source_entry()

    distributed = _initialize_distributed()
    try:
        config = parse_command_line(args)
        if distributed:
            _prepare_distributed_config(config)

        with global_options_context(config):
            return _run(config, running_as_script=running_as_script)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _initialize_distributed() -> bool:
    """Initialize torchrun's process group when distributed env is present."""

    if not dist.is_available() or "RANK" not in os.environ:
        return False
    if os.environ.get("WORLD_SIZE", "1") == "1":
        # Single-rank torchrun: no process group is needed, and NCCL
        # init_process_group with world_size=1 SIGSEGVs on this stack
        # (import_probe 2290340/2290344: set_device_ok printed, init crash).
        return False
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(local_rank)
        backend = os.environ.get("SPINGNN_DDP_BACKEND")
        if backend is None:
            backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        # Warm up the communicator before any DistributedDataParallel
        # construction: torch 2.11.0+cu128 / NCCL 2.28.9 SIGSEGVs inside DDP
        # construction on RTX 5090/4090 without it (probe jobs 2290284/2290286
        # crash vs 2290293/2290294 pass). Only the nccl branch touches CUDA;
        # gloo uses a CPU tensor so CPU-only tests are unaffected.
        from nequip.train.trainer import warmup_process_group_for_ddp

        warmup_process_group_for_ddp()
    return True


def _broadcast_object(obj) -> None:
    """Broadcast a python object list from rank 0 to every rank."""

    kwargs = {}
    if dist.get_backend() != "gloo" and torch.cuda.is_available():
        kwargs["device"] = torch.device("cuda", torch.cuda.current_device())
    dist.broadcast_object_list(obj, src=0, **kwargs)


def _distributed_invocation_id() -> str:
    """One shared identity per launch for the dataset marker namespace.

    Markers keyed by this identity cannot be satisfied or poisoned by stale
    ``.ok``/``.failed`` files from a previous invocation.
    """

    if dist.get_rank() == 0:
        job_id = os.environ.get("SLURM_JOB_ID")
        identity = f"slurm-{job_id}" if job_id else f"uuid-{uuid.uuid4()}"
    else:
        identity = None
    obj = [identity]
    _broadcast_object(obj)
    return obj[0]


def _prepare_distributed_config(config) -> None:
    """Separate shared dataset cache from rank-owned training output files."""

    rank = dist.get_rank()
    if config.get("seed") is None or config.get("dataset_seed") is None:
        raise ValueError(
            "distributed training requires explicit seed and dataset_seed "
            "to guarantee identical initial model and dataset split state"
        )
    base_root = Path(str(config["root"])).expanduser()
    if not base_root.is_absolute():
        base_root = base_root.resolve()
    config["dataset_root"] = str(
        Path(str(config.get("dataset_root", base_root / "dataset-cache")))
        .expanduser()
        .resolve()
    )
    config["root"] = str(base_root / f"rank-{rank}")
    # rank 0 owns the canonical checkpoint files; every rank records where
    config["ddp_rank0_root"] = str(base_root / "rank-0")
    # the default run_name embeds a per-process uuid; rank 0 owns one shared
    # name (explicit config run_names are identical across ranks anyway)
    run_name = [str(config["run_name"]) if rank == 0 else None]
    _broadcast_object(run_name)
    config["run_name"] = run_name[0]
    if torch.cuda.is_available() and str(config.get("device", "cuda")) == "cuda":
        config["device"] = f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}"


def _distributed_dataset_from_config(config, prefix="dataset"):
    """Build a dataset once, then let every rank load the completed cache."""

    # ``validation_dataset`` is optional in NequIP configs. Check this before
    # entering a collective so every rank follows the same no-op path.
    if config.get(prefix, None) is None:
        return None

    if not dist.is_available() or not dist.is_initialized():
        return dataset_from_config(config, prefix=prefix)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    error = None
    dataset = None
    marker_root = Path(str(config["dataset_root"]))
    marker_root.mkdir(parents=True, exist_ok=True)
    marker_key = f".spingnn-ddp-{prefix}-{world_size}-{_distributed_invocation_id()}"
    ok_marker = marker_root / f"{marker_key}-rank-{rank}.ok"
    failed_marker = marker_root / f"{marker_key}-rank-{rank}.failed"

    def write_marker(path: Path, message: str = "") -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(message)
        temporary.replace(path)

    def wait_for_rank_zero() -> None:
        deadline = time.monotonic() + float(
            os.environ.get("SPINGNN_DDP_DATASET_TIMEOUT", "900")
        )
        ready = marker_root / f"{marker_key}-rank-0.ok"
        failed = marker_root / f"{marker_key}-rank-0.failed"
        while not ready.exists() and not failed.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"timed out waiting for rank 0 dataset preparation: {marker_root}"
                )
            time.sleep(0.2)
        if failed.exists():
            raise RuntimeError(
                "rank 0 dataset preparation failed: " + failed.read_text()
            )

    def wait_for_all_ranks() -> None:
        deadline = time.monotonic() + float(
            os.environ.get("SPINGNN_DDP_DATASET_TIMEOUT", "900")
        )
        expected = [
            marker_root / f"{marker_key}-rank-{index}.ok" for index in range(world_size)
        ]
        failed = [
            marker_root / f"{marker_key}-rank-{index}.failed"
            for index in range(world_size)
        ]
        while not all(path.exists() for path in expected):
            failures = [path for path in failed if path.exists()]
            if failures:
                raise RuntimeError(
                    "distributed dataset preparation failed: "
                    + "; ".join(path.read_text() for path in failures)
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"timed out waiting for distributed dataset ranks: {marker_root}"
                )
            time.sleep(0.2)
        # failure markers are checked even when all success markers exist
        failures = [path for path in failed if path.exists()]
        if failures:
            raise RuntimeError(
                "distributed dataset preparation failed: "
                + "; ".join(path.read_text() for path in failures)
            )

    if rank == 0:
        try:
            dataset = dataset_from_config(config, prefix=prefix)
            write_marker(ok_marker)
        except Exception as exc:  # propagate the real failure to every rank
            error = f"{type(exc).__name__}: {exc}"
            write_marker(failed_marker, error)
            raise RuntimeError(f"distributed dataset setup failed on rank 0: {error}")
    else:
        wait_for_rank_zero()
        try:
            dataset = dataset_from_config(config, prefix=prefix)
            write_marker(ok_marker)
        except Exception as exc:
            error = f"rank {rank}: {type(exc).__name__}: {exc}"
            write_marker(failed_marker, error)
            raise RuntimeError(error) from exc

    wait_for_all_ranks()
    return dataset


def _run(config, running_as_script: bool = True):
    if running_as_script:
        set_up_script_logger(config.get("log", None), config.verbose)

    found_restart_file = isdir(f"{config.root}/{config.run_name}")
    if found_restart_file and not config.append:
        raise RuntimeError(
            f"Training instance exists at {config.root}/{config.run_name}; "
            "either set append to True or use a different root or runname"
        )

    # for fresh new train
    if not found_restart_file:
        trainer = fresh_start(config)
    else:
        trainer = restart(config)

    # Train
    trainer.save()
    trainer.train()

    return


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="Train (or restart training of) a NequIP model."
    )
    parser.add_argument(
        "config", help="YAML file configuring the model, dataset, and other options"
    )
    parser.add_argument(
        "--equivariance-test",
        help="test the model's equivariance before training on n (default 1) random frames from the dataset",
        const=1,
        type=int,
        nargs="?",
    )
    parser.add_argument(
        "--model-debug-mode",
        help="enable model debug mode, which can sometimes give much more useful error messages at the cost of some speed. Do not use for production training!",
        action="store_true",
    )
    parser.add_argument(
        "--grad-anomaly-mode",
        help="enable PyTorch autograd anomaly mode to debug NaN gradients. Do not use for production training!",
        action="store_true",
    )
    parser.add_argument(
        "--log",
        help="log file to store all the screen logging",
        type=Path,
        default=None,
    )
    args = parser.parse_args(args=args)

    config = Config.from_file(args.config, defaults=default_config)
    for flag in ("model_debug_mode", "equivariance_test", "grad_anomaly_mode"):
        config[flag] = getattr(args, flag) or config[flag]

    return config


def fresh_start(config):
    # we use add_to_config cause it's a fresh start and need to record it
    check_code_version(config, add_to_config=True)
    _set_global_options(config)

    # = Make the trainer =
    if config.wandb:
        import wandb  # noqa: F401
        from nequip.train.trainer_wandb import TrainerWandB

        # download parameters from wandb in case of sweeping
        from nequip.utils.wandb import init_n_update

        config = init_n_update(config)

        trainer = TrainerWandB(model=None, **dict(config))
    else:
        from nequip.train.trainer import Trainer

        trainer = Trainer(model=None, **dict(config))

    # what is this
    # to update wandb data?
    config.update(trainer.params)

    # = Load the dataset =
    dataset = _distributed_dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully loaded the data set of type {dataset}...")
    try:
        validation_dataset = _distributed_dataset_from_config(
            config, prefix="validation_dataset"
        )
        logging.info(
            f"Successfully loaded the validation data set of type {validation_dataset}..."
        )
    except KeyError:
        # It couldn't be found
        validation_dataset = None

    # = Train/test split =
    trainer.set_dataset(dataset, validation_dataset)

    # = Build model =
    final_model = model_from_config(
        config=config, initialize=True, dataset=trainer.dataset_train
    )
    # the model is built in ``default_dtype``, which data, targets and losses
    # keep; ``model_dtype`` may differ (modules marked ``_spingnn_keep_dtype``
    # keep their own dtypes)
    model_dtype = model_dtype_from_config(config)
    if model_dtype != torch.get_default_dtype():
        final_model = cast_model_dtype(final_model, model_dtype)
        logging.info(f"Cast the network to model_dtype {model_dtype}")
    logging.info("Successfully built the network...")

    # by doing this here we check also any keys custom builders may have added
    _check_old_keys(config)

    # Equivar test
    if config.equivariance_test > 0:
        n_train: int = len(trainer.dataset_train)
        assert config.equivariance_test <= n_train
        final_model.eval()
        indexes = torch.randperm(n_train)[: config.equivariance_test]
        errstr = assert_AtomicData_equivariant(
            final_model,
            [trainer.dataset_train[i] for i in indexes],
            do_parity=config["parity"],
            do_time_reversal=config["time_reversal"],
            do_only_rot_spin=(not config["soc"]),
        )
        final_model.train()
        logging.info(
            "Equivariance test passed; equivariance errors:\n"
            "   Errors are in real units, where relevant.\n"
            "   Please note that the large scale of the typical\n"
            "   shifts to the (atomic) energy can cause\n"
            "   catastrophic cancellation and give incorrectly\n"
            "   the equivariance error as zero for those fields.\n"
            f"{errstr}"
        )
        del errstr, indexes, n_train

    final_model = configure_model_for_compile(final_model, config)
    final_model = configure_model_for_autocast(
        final_model, config, trainer.torch_device.type
    )

    # Set the trainer
    trainer.model = final_model

    # Store any updated config information in the trainer
    trainer.update_kwargs(config)

    return trainer


def restart(config):
    # load the dictionary; distributed checkpoints are owned by rank 0
    canonical_root = config.get("ddp_rank0_root", config.root)
    restart_file = f"{canonical_root}/{config.run_name}/trainer.pth"
    # the file records the saving rank's CUDA device for its optimizer and EMA
    # tensors; read it on the CPU so that no rank allocates on that device,
    # the state loaders move the tensors to each rank's own device
    dictionary = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=restart_file,
        enforced_format="torch",
        map_location="cpu",
    )
    dictionary["append"] = True
    # the canonical dictionary carries rank 0's rank-local fields; remap
    # them to this rank before comparing against the invocation config
    dictionary["root"] = config["root"]
    if "device" in config:
        dictionary["device"] = config["device"]

    # compare dictionary to config and update stop condition related arguments
    # and the runtime-only keys (_RESTART_RUNTIME_KEYS). Any other differing
    # key raises if the config value has the type of the saved value; a key
    # whose saved value has another type, or that trainer.pth does not
    # contain, is ignored and the saved value (if any) is used.
    for k in config.keys():
        if not _restart_config_values_match(k, config[k], dictionary.get(k, "")):
            if k == "max_epochs":
                dictionary[k] = config[k]
                logging.info(f'Update "{k}" to {dictionary[k]}')
            elif k.startswith("early_stop"):
                dictionary[k] = config[k]
                logging.info(f'Update "{k}" to {dictionary[k]}')
            elif k in _RESTART_RUNTIME_KEYS:
                previous = dictionary[k] if k in dictionary else "(not in trainer.pth)"
                dictionary[k] = config[k]
                logging.info(f'Update "{k}" from {previous} to {dictionary[k]}')
            elif k == "append":
                pass
            elif isinstance(config[k], type(dictionary.get(k, ""))):
                raise ValueError(
                    f'Key "{k}" is different in config and the result trainer.pth file. Please double check'
                )

    # note, "trainer.pth"/dictionary also store code versions,
    # which will not be stored in config and thus not checked here
    check_code_version(config)

    # recursive loop, if same type but different value
    # raise error

    config = Config(dictionary, exclude_keys=["state_dict", "progress"])

    # dtype, etc.
    _set_global_options(config)

    # note, the from_dict method will check whether the code version
    # in trainer.pth is consistent and issue warnings
    if config.wandb:
        from nequip.train.trainer_wandb import TrainerWandB
        from nequip.utils.wandb import resume

        resume(config)
        trainer = TrainerWandB.from_dict(dictionary)
    else:
        from nequip.train.trainer import Trainer

        trainer = Trainer.from_dict(dictionary)

    # = Load the dataset =
    dataset = _distributed_dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully re-loaded the data set of type {dataset}...")
    try:
        validation_dataset = _distributed_dataset_from_config(
            config, prefix="validation_dataset"
        )
        logging.info(
            f"Successfully re-loaded the validation data set of type {validation_dataset}..."
        )
    except KeyError:
        # It couldn't be found
        validation_dataset = None
    trainer.set_dataset(dataset, validation_dataset)
    trainer.model = configure_model_for_compile(trainer.model, config)
    trainer.model = configure_model_for_autocast(
        trainer.model, config, trainer.torch_device.type
    )

    return trainer


def _restart_nested_values_match(requested, saved):
    if isinstance(requested, dict) and isinstance(saved, dict):
        return all(
            key in saved and _restart_nested_values_match(value, saved[key])
            for key, value in requested.items()
        )
    if isinstance(requested, (list, tuple)) and isinstance(saved, (list, tuple)):
        return len(requested) == len(saved) and all(
            _restart_nested_values_match(requested_value, saved_value)
            for requested_value, saved_value in zip(requested, saved)
        )
    return requested == saved


def _restart_config_values_match(key, requested, saved):
    if requested == saved:
        return True
    if key not in ("optimizer_kwargs", "lr_scheduler_kwargs"):
        return False
    if requested is None and isinstance(saved, dict):
        return True
    if not isinstance(requested, dict) or not isinstance(saved, dict):
        return False
    return _restart_nested_values_match(requested, saved)


def _check_old_keys(config) -> None:
    """check ``config`` for old/depricated keys and emit corresponding errors/warnings"""
    # compile_model
    k = "compile_model"
    if k in config:
        if config[k]:
            raise ValueError("the `compile_model` option has been removed")
        else:
            warnings.warn("the `compile_model` option has been removed")


if __name__ == "__main__":
    main(running_as_script=True)
