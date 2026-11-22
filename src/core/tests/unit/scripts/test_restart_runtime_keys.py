"""Restart (``append: true``) with changed runtime-only keys.

``nequip-train`` lets a restart change, or newly add, the keys of
``_RESTART_RUNTIME_KEYS`` (``packaged_backbone_torch_compile``,
``batch_metrics_log_interval`` and the ``ddp_*`` options): the invocation
config's value replaces the saved one and reaches the rebuilt model and
trainer. Any other changed key still raises when its type matches the saved
value; a key with another type, or one missing from ``trainer.pth``, is
still ignored.
"""

import copy
import functools
import importlib.util
import logging
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

import nequip.scripts.train as train_script
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, PackagedBackbone
from nequip.utils import Config
from nequip.utils._global_options import global_options_context

pytestmark = [
    # torch.package's exporter still pickles through TypedStorage (torch 2.11)
    pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning"),
    # emitted by nequip-train's default `_jit_fusion_strategy` global option
    pytest.mark.filterwarnings(
        "ignore:`torch.jit.set_fusion_strategy` is deprecated:DeprecationWarning"
    ),
]

_CPU_BACKEND = "aot_eager"  # Dynamo + AOTAutograd, no Inductor C++ build


@functools.lru_cache(maxsize=None)
def _packaged_helpers():
    """Toy-package helpers of ``tests/unit/nn/test_packaged_backbone.py``."""
    path = Path(__file__).parents[1] / "nn" / "test_packaged_backbone.py"
    spec = importlib.util.spec_from_file_location("_restart_packaged_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_nested_equal(reference, candidate, path="root"):
    if isinstance(reference, dict):
        assert set(reference) == set(candidate), f"{path}: key sets differ"
        for key in reference:
            _assert_nested_equal(reference[key], candidate[key], f"{path}/{key}")
    elif isinstance(reference, (list, tuple)):
        assert len(reference) == len(candidate), f"{path}: lengths differ"
        for index, (ref_item, cand_item) in enumerate(zip(reference, candidate)):
            _assert_nested_equal(ref_item, cand_item, f"{path}/{index}")
    elif isinstance(reference, torch.Tensor):
        assert torch.equal(reference, candidate), f"{path}: tensors differ"
    else:
        assert reference == candidate, f"{path}: {reference!r} != {candidate!r}"


def _snapshot(value):
    # deep copy; EMA ``collected_params`` are non-leaf clones (no deepcopy)
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _training_state(trainer):
    model = getattr(trainer.model, "module", trainer.model)
    return _snapshot(
        {
            "model": model.state_dict(),
            "optim": trainer.optim.state_dict(),
            "ema": trainer.ema.state_dict(),
        }
    )


def _load_config(path):
    return Config.from_file(str(path), defaults=train_script.default_config)


def _write_config(path, **config):
    path.write_text(yaml.safe_dump(config))
    return path


# -- single process: a toy packaged backbone gains torch_compile on restart ---


class _ToyForceHead(GraphModuleMixin, torch.nn.Module):
    """Per-atom vectors from the strided tensor tap, summed over edges."""

    def __init__(self, irreps_in, mul):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(0.5, 1.5, mul))
        self._init_irreps(
            irreps_in=irreps_in, irreps_out={AtomicDataDict.FORCE_KEY: "1x1o"}
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = data.copy()
        tensor = data["bb_tensor"].to(self.weight.dtype)  # (edges, mul, 1 + 3)
        edge_vectors = (tensor[:, :, 1:] * self.weight.unsqueeze(-1)).sum(1)
        pos = data[AtomicDataDict.POSITIONS_KEY]
        forces = torch.zeros_like(pos).index_add(
            0, data[AtomicDataDict.EDGE_INDEX_KEY][0], edge_vectors.to(pos.dtype)
        )
        data[AtomicDataDict.FORCE_KEY] = forces
        return data


def toy_force_head(model):
    model.append("head", _ToyForceHead(model.irreps_out, mul=_packaged_helpers().MUL))
    return model


def _packaged_inputs(tmp_path):
    helpers = _packaged_helpers()
    package = helpers._write_package(
        tmp_path, "toy.zip", "spingnn_toy_backbone_restart"
    )
    rng = np.random.default_rng(5)
    n_frames = 6
    np.savez(
        tmp_path / "data.npz",
        R=rng.uniform(0.0, 2.0, size=(n_frames, 4, 3)),
        F=rng.normal(size=(n_frames, 4, 3)),
        z=np.array([1, 6, 8, 1]),
    )
    return dict(
        root=str(tmp_path / "results"),
        run_name="packaged",
        seed=0,
        dataset_seed=0,
        dataset="npz",
        dataset_file_name=str(tmp_path / "data.npz"),
        key_mapping={"z": "atomic_numbers", "F": "forces", "R": "pos"},
        npz_fixed_field_keys=["atomic_numbers"],
        chemical_symbols=list(helpers.TYPE_NAMES),
        r_max=helpers.R_MAX,
        default_dtype=str(torch.get_default_dtype()).replace("torch.", ""),
        model_builders=["PackagedBackboneModel", f"{__name__}.toy_force_head"],
        packaged_backbone_path=str(package),
        packaged_backbone_stop_after="allegro",
        packaged_backbone_taps={"bb_tensor": {"module": "allegro.tps.1"}},
        packaged_backbone_trainable_prefixes=["latents.", "tps.", "bessel_weights"],
        n_train=4,
        n_val=2,
        batch_size=2,
        validation_batch_size=2,
        max_epochs=1,
        learning_rate=0.01,
        loss_coeffs={"forces": 1.0},
        use_ema=True,
        report_init_validation=False,
        spingnn_compile={"enabled": False},
        append=False,
        verbose="warning",
    )


def _backbone(model):
    return next(m for m in model.modules() if isinstance(m, PackagedBackbone))


def test_restart_adds_packaged_backbone_compile(tmp_path, caplog):
    base = _packaged_inputs(tmp_path)
    try:
        torch._dynamo.reset()
        with global_options_context(base):
            trainer = train_script.fresh_start(
                _load_config(_write_config(tmp_path / "fresh.yaml", **base))
            )
            trainer.save()
            trainer.train()
        assert trainer.iepoch == 1
        assert _backbone(trainer.model)._torch_compile_kwargs is None
        before = _training_state(trainer)
        saved = torch.load(
            Path(base["root"]) / "packaged" / "trainer.pth", weights_only=False
        )
        assert "packaged_backbone_torch_compile" not in saved
        assert saved["batch_metrics_log_interval"] == 1

        resume = dict(
            base,
            max_epochs=2,
            append=True,
            packaged_backbone_torch_compile={"enabled": True, "backend": _CPU_BACKEND},
            batch_metrics_log_interval=2,
            # not runtime-only and missing from trainer.pth: ignored as before
            unlisted_option={"enabled": True},
        )
        # a changed key outside the runtime-only list still raises
        rejected = _load_config(
            _write_config(
                tmp_path / "rejected.yaml", **dict(resume, learning_rate=0.02)
            )
        )
        with global_options_context(rejected):
            with pytest.raises(ValueError, match='Key "learning_rate" is different'):
                train_script.restart(rejected)

        config = _load_config(_write_config(tmp_path / "resume.yaml", **resume))
        with global_options_context(config), caplog.at_level(logging.INFO):
            restarted = train_script.restart(config)
            messages = [record.getMessage() for record in caplog.records]
            assert any(
                message.startswith(
                    'Update "packaged_backbone_torch_compile" from (not in trainer.pth) to'
                )
                for message in messages
            ), messages
            assert 'Update "batch_metrics_log_interval" from 1 to 2' in messages
            backbone = _backbone(restarted.model)
            assert backbone._torch_compile_kwargs == {
                "backend": _CPU_BACKEND,
                "dynamic": True,
            }
            assert backbone._compiled_stages is None  # compiled at the first forward
            assert restarted.batch_metrics_log_interval == 2
            assert "unlisted_option" not in restarted.kwargs
            assert restarted.iepoch == 1
            # parameters, optimizer and EMA state are those before the restart
            _assert_nested_equal(before, _training_state(restarted))

            restarted.save()
            restarted.train()
        assert restarted.iepoch == 2
        assert backbone._compiled_stages is not None
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
        assert all(
            torch.isfinite(p).all() for p in restarted.model.parameters()
        ), "non-finite parameters after the compiled epoch"
        assert not torch.equal(
            before["model"]["head.weight"], restarted.model.head.weight
        )
        saved = torch.load(
            Path(base["root"]) / "packaged" / "trainer.pth", weights_only=False
        )
        assert saved["packaged_backbone_torch_compile"] == {
            "enabled": True,
            "backend": _CPU_BACKEND,
        }
        assert saved["batch_metrics_log_interval"] == 2
    finally:
        torch._dynamo.reset()


# -- two-process gloo: a restart rebuilds DDP with changed G8 options ---------

_WORLD_SIZE = 2
_JOIN_TIMEOUT = 120.0
_GLOO_AVAILABLE = dist.is_available() and dist.is_gloo_available()
_CHANGED_DDP_OPTIONS = {
    "ddp_broadcast_buffers": False,
    "ddp_static_graph": True,
    "ddp_gradient_as_bucket_view": True,
    "ddp_find_unused_parameters": False,
}


class _LinearForces(GraphModuleMixin, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.register_buffer("offset", torch.full((3,), 0.25))
        self._init_irreps(
            irreps_in={"pos": "1x1o"},
            irreps_out={AtomicDataDict.FORCE_KEY: "1x1o"},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = data.copy()
        data[AtomicDataDict.FORCE_KEY] = self.linear(data["pos"]) + self.offset
        return data


def linear_forces():
    return _LinearForces()


def _ddp_worker_setup(rank, init_file):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.pop("SLURM_JOB_ID", None)  # fresh dataset-marker namespaces
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
    )


def _ddp_fresh_worker(process, init_files, base_dir):
    # processes [0, 2) train "uninterrupted" for two epochs and [2, 4) train
    # "interrupted" for one, both through nequip-train's fresh_start
    group, rank = divmod(process, _WORLD_SIZE)
    name = ("uninterrupted", "interrupted")[group]
    _ddp_worker_setup(rank, init_files[group])
    try:
        config = _load_config(Path(base_dir) / f"{name}.yaml")
        train_script._prepare_distributed_config(config)
        with global_options_context(config):
            trainer = train_script.fresh_start(config)
            trainer.save()
            trainer.train()
            torch.save(
                dict(_training_state(trainer), iepoch=trainer.iepoch),
                Path(base_dir) / f"{name}-{rank}.pth",
            )
    finally:
        dist.destroy_process_group()


def _ddp_resume_worker(rank, init_file, base_dir):
    from torch.nn.parallel import DistributedDataParallel

    _ddp_worker_setup(rank, init_file)
    try:
        config = _load_config(Path(base_dir) / "resume.yaml")
        train_script._prepare_distributed_config(config)
        with global_options_context(config):
            trainer = train_script.restart(config)
            assert isinstance(trainer.model, DistributedDataParallel)
            assert trainer._ddp_options == {
                "broadcast_buffers": False,
                "static_graph": True,
                "gradient_as_bucket_view": True,
                "find_unused_parameters": False,
            }
            ddp = trainer.model
            assert ddp.broadcast_buffers is False
            assert ddp.static_graph is True
            assert ddp.gradient_as_bucket_view is True
            assert ddp.find_unused_parameters is False
            assert trainer.batch_metrics_log_interval == 3
            restored = _training_state(trainer)
            trainer.save()
            trainer.train()
            torch.save(
                dict(
                    _training_state(trainer),
                    iepoch=trainer.iepoch,
                    restored=restored,
                ),
                Path(base_dir) / f"resumed-{rank}.pth",
            )
    finally:
        dist.destroy_process_group()


def _start(worker, args, nprocs):
    context = mp.start_processes(
        worker, args=args, nprocs=nprocs, join=False, start_method="spawn"
    )
    for process in context.processes:
        process.join(timeout=_JOIN_TIMEOUT)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("gloo worker hung; terminated after timeout")
    exitcodes = [process.exitcode for process in context.processes]
    assert exitcodes == [0] * nprocs, f"worker exitcodes: {exitcodes}"


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason="torch distributed gloo unavailable")
def test_ddp_restart_applies_changed_ddp_options(tmp_path):
    if torch.get_default_dtype() is not torch.float64:
        pytest.skip("runs once, in the float64 session (the workers set float64)")
    rng = np.random.default_rng(11)
    np.savez(
        tmp_path / "data.npz",
        R=rng.normal(size=(6, 3, 3)),
        F=rng.normal(size=(6, 3, 3)),
        z=np.ones(3, dtype=np.int64),
    )
    common = dict(
        run_name="ddp",
        seed=0,
        dataset_seed=0,
        dataset="npz",
        dataset_file_name=str(tmp_path / "data.npz"),
        key_mapping={"z": "atomic_numbers", "F": "forces", "R": "pos"},
        npz_fixed_field_keys=["atomic_numbers"],
        chemical_symbols=["H"],
        r_max=2.5,
        default_dtype="float64",
        model_builders=[f"{__name__}.linear_forces"],
        n_train=4,
        n_val=2,
        batch_size=1,
        validation_batch_size=1,
        learning_rate=0.01,
        loss_coeffs={"forces": 1.0},
        use_ema=True,
        report_init_validation=False,
        dataloader_num_workers=0,
        device="cpu",
        spingnn_compile={"enabled": False},
        verbose="warning",
    )
    for name, epochs in (("uninterrupted", 2), ("interrupted", 1)):
        _write_config(
            tmp_path / f"{name}.yaml",
            root=str(tmp_path / name),
            max_epochs=epochs,
            append=False,
            **common,
        )
    _write_config(
        tmp_path / "resume.yaml",
        root=str(tmp_path / "interrupted"),
        max_epochs=2,
        append=True,
        batch_metrics_log_interval=3,
        **_CHANGED_DDP_OPTIONS,
        **common,
    )

    _start(
        _ddp_fresh_worker,
        ((str(tmp_path / "init-a"), str(tmp_path / "init-b")), str(tmp_path)),
        2 * _WORLD_SIZE,
    )
    saved = torch.load(
        tmp_path / "interrupted" / "rank-0" / "ddp" / "trainer.pth", weights_only=False
    )
    # the fresh runs used the defaults: a null option (another type) and the
    # interval 1 (same type) were silently ignored or rejected before
    assert all(saved[key] is None for key in _CHANGED_DDP_OPTIONS)
    assert saved["batch_metrics_log_interval"] == 1
    _start(_ddp_resume_worker, (str(tmp_path / "init-c"), str(tmp_path)), _WORLD_SIZE)

    saved = torch.load(
        tmp_path / "interrupted" / "rank-0" / "ddp" / "trainer.pth", weights_only=False
    )
    assert {key: saved[key] for key in _CHANGED_DDP_OPTIONS} == _CHANGED_DDP_OPTIONS
    assert saved["batch_metrics_log_interval"] == 3
    for rank in range(_WORLD_SIZE):
        interrupted = torch.load(
            tmp_path / f"interrupted-{rank}.pth", weights_only=False
        )
        resumed = torch.load(tmp_path / f"resumed-{rank}.pth", weights_only=False)
        uninterrupted = torch.load(
            tmp_path / f"uninterrupted-{rank}.pth", weights_only=False
        )
        assert interrupted.pop("iepoch") == 1
        # the restart restores this state exactly ...
        _assert_nested_equal(interrupted, resumed.pop("restored"), f"rank-{rank}")
        # ... and continues bitwise like the uninterrupted default-option run
        assert resumed["iepoch"] == uninterrupted["iepoch"] == 2
        _assert_nested_equal(uninterrupted, resumed, f"rank-{rank}")
