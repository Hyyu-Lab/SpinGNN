"""Two-process gloo tests for optimizer param groups and start-of-epoch
callbacks under native DDP, including a restart in fresh processes.

Param-group names are resolved on the unwrapped model (no ``module.``
prefix) and the group layout survives a restart; start-of-epoch callbacks
run on every rank with ``trainer.model`` being the DDP wrapper, and after a
restart they first run before the first resumed epoch.
"""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nequip.data import AtomicDataDict
from nequip.data.dataset import AtomicInMemoryDataset
from nequip.data.transforms import TypeMapper
from nequip.nn import GraphModuleMixin

_WORLD_SIZE = 2
_JOIN_TIMEOUT = 60.0
_N_FRAMES = 6
_LOG_ENV = "SPINGNN_TEST_START_OF_EPOCH_LOG"

_GLOO_AVAILABLE = dist.is_available() and dist.is_gloo_available()
_SKIP_REASON = "torch distributed gloo backend unavailable"


def _join_workers(context):
    for process in context.processes:
        process.join(timeout=_JOIN_TIMEOUT)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("gloo worker hung; terminated after timeout")
    exitcodes = [process.exitcode for process in context.processes]
    assert exitcodes == [0] * _WORLD_SIZE, f"worker exitcodes: {exitcodes}"


class _TinyDataset(AtomicInMemoryDataset):
    @property
    def raw_file_names(self):
        return []

    def get_data(self):
        generator = torch.Generator().manual_seed(7)
        fields = {
            AtomicDataDict.POSITIONS_KEY: torch.randn(
                _N_FRAMES, 3, 3, generator=generator
            ),
            AtomicDataDict.FORCE_KEY: torch.randn(_N_FRAMES, 3, 3, generator=generator),
            "atomic_numbers": torch.ones(_N_FRAMES, 3, dtype=torch.long),
        }
        return fields, {"r_max": 4.0}


class _ScaledNet(GraphModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.head = torch.nn.Linear(3, 3)
        self.register_buffer("scale", torch.zeros(()))
        self._init_irreps(
            irreps_in={"pos": "1x1o"},
            irreps_out={AtomicDataDict.FORCE_KEY: "1x1o"},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = data.copy()
        features = self.backbone(data["pos"])
        data[AtomicDataDict.FORCE_KEY] = self.head(features) * (1 + self.scale)
        return data


def _builder():
    return _ScaledNet()


def record_start(trainer):
    from torch.nn.parallel import DistributedDataParallel

    assert isinstance(trainer.model, DistributedDataParallel)
    module = trainer.model.module
    event = dict(
        rank=trainer._ddp_rank, iepoch=trainer.iepoch, scale=module.scale.item()
    )
    with open(os.environ[_LOG_ENV], "a") as log:
        log.write(json.dumps(event) + "\n")
    with torch.no_grad():
        module.scale.fill_(float(trainer.iepoch + 1))


def _config(rank_root, rank0_root, max_epochs):
    return dict(
        root=str(rank_root),
        run_name="hooks",
        ddp_rank0_root=str(rank0_root),
        n_train=4,
        n_val=2,
        max_epochs=max_epochs,
        batch_size=1,
        validation_batch_size=1,
        learning_rate=1.0e-2,
        seed=0,
        dataset_seed=0,
        loss_coeffs={"forces": 1.0},
        early_stopping_patiences={"validation_loss": 50},
        report_init_validation=False,
        dataloader_num_workers=0,
        device="cpu",
        default_dtype="float64",
        model_builders=[_builder],
        chemical_symbols=["H"],
        optimizer_param_groups=[
            {"name": "head", "lr": 1.0e-3},
            {"name": "backbone", "match": ["backbone."], "weight_decay": 1.0e-4},
        ],
        early_stopping_lr_group="backbone",
        start_of_epoch_callbacks=[f"{__name__}.record_start"],
        verbose="WARNING",
    )


def _check_groups(trainer, rank):
    structure = trainer._optimizer_group_structure
    assert structure == [
        {"name": "head", "params": ["head.weight", "head.bias"]},
        {"name": "backbone", "params": ["backbone.weight", "backbone.bias"]},
    ], structure
    module = trainer.model.module
    groups = trainer.optim.param_groups
    assert groups[0]["params"][0] is module.head.weight
    assert groups[1]["params"][1] is module.backbone.bias
    assert [g["lr"] for g in groups] == [1.0e-3, 1.0e-2]
    assert trainer.mae_dict["LR"] == 1.0e-2
    gathered = [None] * _WORLD_SIZE
    dist.all_gather_object(gathered, structure)
    assert all(other == structure for other in gathered), f"rank {rank}: {gathered}"


def _fresh_worker(rank, world_size, init_file, base_dir):
    from nequip.model import model_from_config
    from nequip.train.trainer import Trainer

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    os.environ[_LOG_ENV] = str(Path(base_dir) / f"fresh-{rank}.jsonl")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        base = Path(base_dir)
        config = _config(base / f"rank-{rank}", base / "rank-0", max_epochs=2)
        torch.manual_seed(0)
        trainer = Trainer(model=model_from_config(config), **config)
        trainer.set_dataset(
            _TinyDataset(
                root=str(base / f"data-{rank}"),
                type_mapper=TypeMapper(chemical_symbol_to_type={"H": 0}),
            )
        )
        trainer.train()
        _check_groups(trainer, rank)
    finally:
        dist.destroy_process_group()


def _resume_worker(rank, world_size, init_file, base_dir):
    from nequip.train.trainer import Trainer
    from nequip.utils.savenload import load_file

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    os.environ[_LOG_ENV] = str(Path(base_dir) / f"resume-{rank}.jsonl")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        base = Path(base_dir)
        dictionary = load_file(
            supported_formats=dict(torch=["pt", "pth"]),
            filename=str(base / "rank-0" / "hooks" / "trainer.pth"),
            enforced_format="torch",
        )
        dictionary["append"] = True
        dictionary["max_epochs"] = 4
        dictionary["root"] = str(base / f"rank-{rank}")
        trainer = Trainer.from_dict(dictionary)
        trainer.set_dataset(
            _TinyDataset(
                root=str(base / f"data-resume-{rank}"),
                type_mapper=TypeMapper(chemical_symbol_to_type={"H": 0}),
            )
        )
        trainer.train()
        assert trainer.iepoch == 4
        _check_groups(trainer, rank)
    finally:
        dist.destroy_process_group()


def _events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason=_SKIP_REASON)
def test_ddp_param_groups_and_start_of_epoch_callbacks(tmp_path):
    for worker, init_name in ((_fresh_worker, "init-a"), (_resume_worker, "init-b")):
        context = mp.start_processes(
            worker,
            args=(_WORLD_SIZE, str(tmp_path / init_name), str(tmp_path)),
            nprocs=_WORLD_SIZE,
            join=False,
            start_method="spawn",
        )
        _join_workers(context)

    for rank in range(_WORLD_SIZE):
        assert _events(tmp_path / f"fresh-{rank}.jsonl") == [
            dict(rank=rank, iepoch=0, scale=0.0),
            dict(rank=rank, iepoch=1, scale=1.0),
        ]
        # first call after the restart sees the restored buffer
        assert _events(tmp_path / f"resume-{rank}.jsonl") == [
            dict(rank=rank, iepoch=2, scale=2.0),
            dict(rank=rank, iepoch=3, scale=3.0),
        ]
