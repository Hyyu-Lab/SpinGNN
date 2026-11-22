"""Behavioral two-process gloo tests for DDP checkpoint recovery and the
native-DDP default.

Covers research/ddp-contract items 5 and 6: the canonical + rank-local
checkpoint contract (rank 0 owns model/optimizer/scheduler/progress, every
rank owns its RNG state, linked by a shared generation id), exact recovery
at epoch boundaries with ``dataloader_num_workers=0``, actionable errors
for torn or incompatible checkpoints, and the ``SPINGNN_DDP_IMPL`` default
flip to native DDP. Exact recovery is intentionally scoped to epoch
boundaries with no dataloader workers; persistent workers and stochastic
in-worker transforms keep their own RNG state outside this contract.
"""

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ._multiprocessing import join_workers

_WORLD_SIZE = 2
_JOIN_TIMEOUT = 60.0

_GLOO_AVAILABLE = dist.is_available() and dist.is_gloo_available()
_SKIP_REASON = "torch distributed gloo backend unavailable"

_N_FRAMES = 6


def _worker_setup():
    # single-threaded CPU kernels: GEMM accumulation order (and thus the
    # bitwise result) must not depend on how many worker processes share the
    # machine, or interrupted and resumed runs cannot be compared exactly
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)


def _init_gloo(rank, world_size, init_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


# ---------------------------------------------------------------------------
# shared tiny model / dataset / config
# ---------------------------------------------------------------------------

from nequip.data import AtomicDataDict  # noqa: E402
from nequip.data.dataset import AtomicInMemoryDataset  # noqa: E402
from nequip.data.transforms import TypeMapper  # noqa: E402
from nequip.nn import GraphModuleMixin  # noqa: E402


class _TinyDataset(AtomicInMemoryDataset):
    """Deterministic six-frame dataset of three-atom hydrogen clusters."""

    @property
    def raw_file_names(self):
        return []

    def get_data(self):
        generator = torch.Generator().manual_seed(20240915)
        positions = torch.randn(_N_FRAMES, 3, 3, generator=generator)
        forces = torch.randn(_N_FRAMES, 3, 3, generator=generator)
        energies = torch.randn(_N_FRAMES, 1, generator=generator)
        fields = {
            AtomicDataDict.POSITIONS_KEY: positions,
            AtomicDataDict.FORCE_KEY: forces,
            AtomicDataDict.TOTAL_ENERGY_KEY: energies,
            "atomic_numbers": torch.ones(_N_FRAMES, 3, dtype=torch.long),
        }
        return fields, {"r_max": 4.0}


def _type_mapper():
    return TypeMapper(chemical_symbol_to_type={"H": 0})


class _DummyNet(GraphModuleMixin, torch.nn.Module):
    def __init__(self, ndim, nydim=1) -> None:
        super().__init__()
        self.ndim = ndim
        self.nydim = nydim
        self.linear1 = torch.nn.Linear(ndim, nydim)
        self.linear2 = torch.nn.Linear(ndim, nydim * 3)
        self._init_irreps(
            irreps_in={"pos": "1x1o"},
            irreps_out={
                AtomicDataDict.FORCE_KEY: "1x1o",
                AtomicDataDict.TOTAL_ENERGY_KEY: "1x0e",
            },
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = data.copy()
        x = data["pos"]
        total_energy = self.linear1(x)
        if AtomicDataDict.BATCH_KEY in data:
            batch = data[AtomicDataDict.BATCH_KEY].long()
            total_energy = torch.zeros(
                (int(batch.max()) + 1, self.nydim),
                dtype=total_energy.dtype,
                device=total_energy.device,
            ).index_add_(0, batch, total_energy)
        data.update(
            {
                AtomicDataDict.FORCE_KEY: self.linear2(x),
                AtomicDataDict.TOTAL_ENERGY_KEY: total_energy,
            }
        )
        return data


def _dummy_builder():
    return _DummyNet(3)


def _rng_drain_callback(trainer):
    # consume different RNG amounts per rank so a resume that restores the
    # wrong rank's RNG state (e.g. rank 0's everywhere) is detectable
    for _ in range(trainer._ddp_rank + 1):
        torch.rand(3)
        np.random.rand(2)
        random.random()
        torch.rand(2, generator=trainer.dataset_rng)


def _make_config(rank_root, max_epochs, rank0_root=None):
    return dict(
        root=str(rank_root),
        run_name="recovery",
        n_train=4,
        n_val=2,
        max_epochs=max_epochs,
        batch_size=2,
        validation_batch_size=2,
        learning_rate=1.0e-2,
        optimizer="Adam",
        lr_scheduler_name="CosineAnnealingWarmRestarts",
        T_0=50,
        T_mult=2,
        loss_coeffs={"forces": 2.0, "total_energy": 1.0},
        seed=0,
        dataset_seed=0,
        append=False,
        report_init_validation=False,
        dataloader_num_workers=0,
        device="cpu",
        default_dtype="float64",
        model_builders=[_dummy_builder],
        chemical_symbols=["H"],
        end_of_epoch_callbacks=[_rng_drain_callback],
        verbose="WARNING",
        ddp_rank0_root=str(rank0_root) if rank0_root is not None else None,
    )


def _make_dataset(root):
    return _TinyDataset(root=str(root), type_mapper=_type_mapper())


def _train_fresh(scenario_dir, rank, max_epochs):
    from nequip.model import model_from_config
    from nequip.train.trainer import Trainer

    config = _make_config(
        scenario_dir / f"rank-{rank}",
        max_epochs=max_epochs,
        rank0_root=scenario_dir / "rank-0",
    )
    config = {k: v for k, v in config.items() if v is not None}
    # Match production ordering (scripts/train.py:fresh_start): Trainer is
    # constructed before the model and seeds all three RNG streams in
    # __init__. The model build below must see the same seeds on every rank
    # group, otherwise separate groups start from different initial weights
    # and can never reproduce each other (found via replica: runA vs runB
    # diverged by 0.84 at epoch 1 before any recovery was involved).
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    model = model_from_config(config)
    trainer = Trainer(model=model, **config)
    trainer.set_dataset(_make_dataset(scenario_dir / f"dataset-{rank}"))
    trainer.train()
    return trainer


def _resume_trainer(scenario_dir, rank, max_epochs):
    from nequip.train.trainer import Trainer
    from nequip.utils.savenload import load_file

    canonical = scenario_dir / "rank-0" / "recovery" / "trainer.pth"
    dictionary = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=str(canonical),
        enforced_format="torch",
    )
    dictionary["append"] = True
    dictionary["max_epochs"] = max_epochs
    # mimic restart(): remap the rank-local root to this rank. All
    # checkpoint-internal absolute paths are remapped into scenario_dir so a
    # resume into a copy never mutates the source checkpoint (the module
    # fixture is shared between dtype parametrizations; without the copy,
    # the first resume rewrites the canonical file to a stopped state and
    # the second dtype's resume fails with "previously stopped").
    dictionary["root"] = str(scenario_dir / f"rank-{rank}")
    progress = dictionary.get("progress")
    if isinstance(progress, dict):
        for key in ("last_model_path", "best_model_path"):
            if key in progress:
                progress[key] = str(
                    scenario_dir / "rank-0" / "recovery" / Path(progress[key]).name
                )
    if dictionary.get("ddp_rank0_root"):
        dictionary["ddp_rank0_root"] = str(scenario_dir / "rank-0")
    trainer = Trainer.from_dict(dictionary)
    trainer.set_dataset(_make_dataset(scenario_dir / f"dataset-resume-{rank}"))
    return trainer


def _dump_trainer_state(trainer, path):
    state = {
        "params": {
            key: value.detach().cpu().clone()
            for key, value in trainer.model.state_dict().items()
        },
        "optim": trainer.optim.state_dict(),
        "lr_sched": trainer.lr_sched.state_dict()
        if trainer.lr_sched is not None
        else None,
        "best_metrics": trainer.best_metrics,
        "best_epoch": trainer.best_epoch,
        "iepoch": trainer.iepoch,
        # next sampled IDs from the dataset generator
        "next_ids": torch.randperm(_N_FRAMES, generator=trainer.dataset_rng),
        # subsequent per-rank RNG draws
        "torch_draw": torch.rand(4),
        "numpy_draw": np.random.rand(3),
        "python_draw": [random.random() for _ in range(2)],
    }
    torch.save(state, path)


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
    elif isinstance(reference, np.ndarray):
        assert np.array_equal(reference, candidate), f"{path}: arrays differ"
    elif isinstance(reference, float):
        assert reference == candidate, f"{path}: {reference} != {candidate}"
    else:
        assert reference == candidate, f"{path}: {reference!r} != {candidate!r}"


# ---------------------------------------------------------------------------
# recovery equality: 4 uninterrupted epochs == 2 epochs + resume to 4
# ---------------------------------------------------------------------------


def _fresh_training_worker(rank, group_size, init_file_a, init_file_b, base_dir):
    # ranks [0, group_size) train runA uninterrupted for 4 epochs;
    # ranks [group_size, 2 * group_size) train runB for 2 epochs
    if rank < group_size:
        scenario, pg_rank, init_file, epochs = "runA", rank, init_file_a, 4
    else:
        scenario, pg_rank, init_file, epochs = (
            "runB",
            rank - group_size,
            init_file_b,
            2,
        )
    _worker_setup()
    _init_gloo(pg_rank, group_size, init_file)
    try:
        scenario_dir = Path(base_dir) / scenario
        trainer = _train_fresh(scenario_dir, pg_rank, max_epochs=epochs)
        _dump_trainer_state(trainer, scenario_dir / f"dump-fresh-{pg_rank}.pth")
    finally:
        dist.destroy_process_group()


def _resume_training_worker(rank, world_size, init_file, base_dir):
    _worker_setup()
    _init_gloo(rank, world_size, init_file)
    try:
        # resume from a private copy: continued training rewrites the
        # canonical checkpoint, which must not corrupt the shared fixture
        source_dir = Path(base_dir) / "runB"
        scenario_dir = Path(base_dir) / "runB-resumed"
        if rank == 0:
            if scenario_dir.exists():
                shutil.rmtree(scenario_dir)
            shutil.copytree(source_dir, scenario_dir)
        dist.barrier()
        trainer = _resume_trainer(scenario_dir, rank, max_epochs=4)
        trainer.train()
        _dump_trainer_state(trainer, scenario_dir / f"dump-resume-{rank}.pth")
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def ddp_checkpoints(tmp_path_factory):
    if not _GLOO_AVAILABLE:
        pytest.skip(_SKIP_REASON)
    base = tmp_path_factory.mktemp("ddp-recovery")
    context = mp.start_processes(
        _fresh_training_worker,
        args=(
            _WORLD_SIZE,
            str(base / "gloo-init-a"),
            str(base / "gloo-init-b"),
            str(base),
        ),
        nprocs=2 * _WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    join_workers(context, nprocs=2 * _WORLD_SIZE, timeout=_JOIN_TIMEOUT)
    return base


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason=_SKIP_REASON)
def test_ddp_recovery_equals_uninterrupted(ddp_checkpoints):
    base = ddp_checkpoints
    # resume runB from its two-epoch checkpoint in fresh processes
    context = mp.start_processes(
        _resume_training_worker,
        args=(_WORLD_SIZE, str(base / "gloo-init-b2"), str(base)),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    join_workers(context, nprocs=_WORLD_SIZE, timeout=_JOIN_TIMEOUT)
    for rank in range(_WORLD_SIZE):
        uninterrupted = torch.load(
            base / "runA" / f"dump-fresh-{rank}.pth", weights_only=False
        )
        resumed = torch.load(
            base / "runB-resumed" / f"dump-resume-{rank}.pth", weights_only=False
        )
        assert resumed["iepoch"] == 4
        _assert_nested_equal(uninterrupted, resumed, path=f"rank-{rank}")


# ---------------------------------------------------------------------------
# recovery negatives: torn or incompatible checkpoints fail with actionable
# errors, on every rank, before DDP construction
# ---------------------------------------------------------------------------


def _tamper_missing_rng_file(dictionary, canonical_dir):
    (canonical_dir / "trainer-rank1.pth").unlink()


def _tamper_world_size(dictionary, canonical_dir):
    dictionary["recovery"]["world_size"] = 3


def _tamper_mixed_generations(dictionary, canonical_dir):
    rank_path = canonical_dir / "trainer-rank1.pth"
    rank_state = torch.load(rank_path, map_location="cpu", weights_only=False)
    rank_state["generation"] = "tampered-generation"
    torch.save(rank_state, rank_path)


def _tamper_missing_optim(dictionary, canonical_dir):
    del dictionary["state_dict"]["optim"]


def _tamper_missing_weights(dictionary, canonical_dir):
    (canonical_dir / "last_model.pth").unlink()


_NEGATIVE_SCENARIOS = [
    ("missing_rng_file", _tamper_missing_rng_file, "missing rank-local RNG state file"),
    ("world_size_change", _tamper_world_size, "world size"),
    ("mixed_generations", _tamper_mixed_generations, "generation mismatch"),
    ("missing_optim", _tamper_missing_optim, "missing optimizer state"),
    ("missing_weights", _tamper_missing_weights, "canonical model weights are missing"),
]


def _make_negative_scenario(source_dir, scenario_dir, tamper):
    shutil.copytree(source_dir, scenario_dir)
    canonical_dir = scenario_dir / "rank-0" / "recovery"
    canonical = canonical_dir / "trainer.pth"
    dictionary = torch.load(canonical, map_location="cpu", weights_only=False)
    old_dir = str(source_dir / "rank-0" / "recovery")
    new_dir = str(canonical_dir)
    for key in (
        "best_model_path",
        "last_model_path",
        "trainer_save_path",
        "config_save_path",
    ):
        if key in dictionary["progress"]:
            dictionary["progress"][key] = dictionary["progress"][key].replace(
                old_dir, new_dir
            )
    tamper(dictionary, canonical_dir)
    torch.save(dictionary, canonical)


def _negative_resume_worker(rank, world_size, init_file, base_dir, scenario_names):
    _worker_setup()
    _init_gloo(rank, world_size, init_file)
    outcomes = {}
    try:
        from nequip.train.trainer import Trainer
        from nequip.utils.savenload import load_file

        for name in scenario_names:
            scenario_dir = Path(base_dir) / name
            canonical = scenario_dir / "rank-0" / "recovery" / "trainer.pth"
            try:
                dictionary = load_file(
                    supported_formats=dict(torch=["pt", "pth"]),
                    filename=str(canonical),
                    enforced_format="torch",
                )
                dictionary["append"] = True
                dictionary["max_epochs"] = 4
                dictionary["root"] = str(scenario_dir / f"rank-{rank}")
                Trainer.from_dict(dictionary)
                outcomes[name] = "NO ERROR"
            except Exception as exc:
                outcomes[name] = f"{type(exc).__name__}: {exc}"
        (Path(base_dir) / f"negative-outcome-{rank}.json").write_text(
            json.dumps(outcomes)
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason=_SKIP_REASON)
def test_ddp_recovery_negative_checkpoints(ddp_checkpoints, tmp_path):
    source = ddp_checkpoints / "runB"
    scenario_names = []
    for name, tamper, _ in _NEGATIVE_SCENARIOS:
        _make_negative_scenario(source, tmp_path / name, tamper)
        scenario_names.append(name)
    context = mp.start_processes(
        _negative_resume_worker,
        args=(
            _WORLD_SIZE,
            str(tmp_path / "gloo-init"),
            str(tmp_path),
            scenario_names,
        ),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    join_workers(context, nprocs=_WORLD_SIZE, timeout=_JOIN_TIMEOUT)
    outcomes = [
        json.loads((tmp_path / f"negative-outcome-{rank}.json").read_text())
        for rank in range(_WORLD_SIZE)
    ]
    for name, _, expected in _NEGATIVE_SCENARIOS:
        messages = [outcome[name] for outcome in outcomes]
        for message in messages:
            assert message != "NO ERROR", f"{name}: resume unexpectedly succeeded"
        assert any(
            expected in message for message in messages
        ), f"{name}: no rank reported the expected error {expected!r}: {messages}"


# ---------------------------------------------------------------------------
# default flip: native DDP unless SPINGNN_DDP_IMPL=manual
# ---------------------------------------------------------------------------


def _impl_selection_worker(rank, world_size, init_file, out_dir):
    from torch.nn.parallel import DistributedDataParallel

    from nequip.model import model_from_config
    from nequip.train.trainer import Trainer

    _worker_setup()
    _init_gloo(rank, world_size, init_file)
    outcomes = {}
    try:
        for label, env_value in (("default", None), ("manual", "manual")):
            if env_value is None:
                os.environ.pop("SPINGNN_DDP_IMPL", None)
            else:
                os.environ["SPINGNN_DDP_IMPL"] = env_value
            config = _make_config(
                Path(out_dir) / f"impl-{label}" / f"rank-{rank}", max_epochs=1
            )
            config = {k: v for k, v in config.items() if v is not None}
            model = model_from_config(config)
            trainer = Trainer(model=model, **config)
            outcomes[label] = isinstance(trainer.model, DistributedDataParallel)
        (Path(out_dir) / f"impl-outcome-{rank}.json").write_text(json.dumps(outcomes))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason=_SKIP_REASON)
def test_ddp_impl_defaults_to_native_ddp(tmp_path):
    context = mp.start_processes(
        _impl_selection_worker,
        args=(_WORLD_SIZE, str(tmp_path / "gloo_init"), str(tmp_path)),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    join_workers(context, nprocs=_WORLD_SIZE, timeout=_JOIN_TIMEOUT)
    for rank in range(_WORLD_SIZE):
        outcomes = json.loads((tmp_path / f"impl-outcome-{rank}.json").read_text())
        assert outcomes == {"default": True, "manual": False}, outcomes


# ---------------------------------------------------------------------------
# legacy single-process checkpoints still load through the old format
# ---------------------------------------------------------------------------


def test_legacy_single_process_checkpoint_loads(tmp_path):
    from nequip.model import model_from_config
    from nequip.train.trainer import Trainer
    from nequip.utils.savenload import load_file

    config = _make_config(tmp_path / "legacy", max_epochs=1)
    config = {k: v for k, v in config.items() if v is not None}
    config["default_dtype"] = str(torch.get_default_dtype())[len("torch.") :]
    dataset = _make_dataset(tmp_path / "legacy-data")
    model = model_from_config(config)
    trainer = Trainer(model=model, **config)
    trainer.set_dataset(dataset)
    trainer.train()
    assert trainer.iepoch == 1

    dictionary = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=str(tmp_path / "legacy" / "recovery" / "trainer.pth"),
        enforced_format="torch",
    )
    # simulate the OLD single-process format: no recovery metadata
    dictionary.pop("recovery", None)
    assert "rng_state" in dictionary["state_dict"]
    dictionary["append"] = True
    dictionary["max_epochs"] = 2

    resumed = Trainer.from_dict(dictionary)
    resumed.set_dataset(_make_dataset(tmp_path / "legacy-data-resume"))
    resumed.train()
    assert resumed.iepoch == 2
