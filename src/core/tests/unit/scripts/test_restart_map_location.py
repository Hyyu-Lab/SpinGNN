"""A restart reads ``trainer.pth`` on the CPU, whatever device wrote it.

The canonical checkpoint of a CUDA run records the saving rank's device for
its optimizer and EMA tensors. Reading it without a ``map_location`` makes
every restarting rank allocate those tensors on the recorded device. The
fixtures here rewrite a trained toy run so that all of its storages are tagged
``cuda:0`` and the spy refuses to restore any storage on a CUDA device, so the
tests behave the same on CPU-only and GPU hosts.
"""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import yaml

import nequip.scripts.train as train_script
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.train.trainer import Trainer
from nequip.utils import Config, load_file
from nequip.utils._global_options import global_options_context

pytestmark = [
    # emitted by spingnn-train's default `_jit_fusion_strategy` global option
    pytest.mark.filterwarnings(
        "ignore:`torch.jit.set_fusion_strategy` is deprecated:DeprecationWarning"
    ),
]

_FORMATS = dict(torch=["pt", "pth"])


class _LinearForces(GraphModuleMixin, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self._init_irreps(
            irreps_in={"pos": "1x1o"},
            irreps_out={AtomicDataDict.FORCE_KEY: "1x1o"},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = data.copy()
        data[AtomicDataDict.FORCE_KEY] = self.linear(data["pos"])
        return data


def linear_forces():
    return _LinearForces()


def _write_inputs(tmp_path, max_epochs, append):
    rng = np.random.default_rng(3)
    np.savez(
        tmp_path / "data.npz",
        R=rng.normal(size=(8, 3, 3)),
        F=rng.normal(size=(8, 3, 3)),
        z=np.ones(3, dtype=np.int64),
    )
    config = dict(
        root=str(tmp_path / "results"),
        run_name="cpu_restart",
        seed=0,
        dataset_seed=0,
        dataset="npz",
        dataset_file_name=str(tmp_path / "data.npz"),
        key_mapping={"z": "atomic_numbers", "F": "forces", "R": "pos"},
        npz_fixed_field_keys=["atomic_numbers"],
        chemical_symbols=["H"],
        r_max=2.5,
        model_builders=[f"{__name__}.linear_forces"],
        default_dtype="float64",
        device="cpu",
        n_train=4,
        n_val=2,
        batch_size=2,
        validation_batch_size=2,
        max_epochs=max_epochs,
        learning_rate=0.01,
        loss_coeffs={"forces": 1.0},
        use_ema=True,
        spingnn_compile={"enabled": False},
        append=append,
        verbose="warning",
    )
    path = tmp_path / f"config-{max_epochs}.yaml"
    path.write_text(yaml.safe_dump(config))
    return Config.from_file(str(path), defaults=train_script.default_config)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def _retag_as_cuda0(path, update=None):
    """Rewrite ``path`` so that ``torch.save`` records every storage on cuda:0."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if update is not None:
        update(state)
    with mock.patch.object(torch.serialization, "location_tag", lambda _: "cuda:0"):
        torch.save(state, path)


def _extend_run(state):
    state["max_epochs"] = 2
    state["progress"]["stop_arg"] = None


@pytest.fixture
def cuda_tagged_run(tmp_path):
    """A trained toy run whose checkpoint files record ``cuda:0`` storages."""
    config = _write_inputs(tmp_path, max_epochs=1, append=False)
    with global_options_context(config):
        trainer = train_script.fresh_start(config)
        trainer.save()
        trainer.train()
        assert trainer.iepoch == 1
    run_dir = tmp_path / "results" / "cpu_restart"
    for name in ("trainer.pth", "last_model.pth"):
        _retag_as_cuda0(run_dir / name)
    saved = torch.load(run_dir / "trainer.pth", map_location="cpu", weights_only=False)
    assert saved["state_dict"]["optim"]["state"], "the test needs optimizer state"
    assert saved["state_dict"]["ema"]["shadow_params"], "the test needs EMA state"
    return tmp_path, run_dir


@pytest.fixture
def restore_locations(monkeypatch):
    """Record the location of every restored storage and refuse CUDA ones."""
    locations = []
    restore = torch.serialization.default_restore_location

    def spy(storage, location):
        locations.append(str(location))
        if str(location).startswith("cuda"):
            raise AssertionError(f"checkpoint storage restored on {location}")
        return restore(storage, location)

    monkeypatch.setattr(torch.serialization, "default_restore_location", spy)
    return locations


def test_restart_reads_checkpoint_on_cpu(
    cuda_tagged_run, restore_locations, monkeypatch
):
    tmp_path, run_dir = cuda_tagged_run
    saved = torch.load(run_dir / "trainer.pth", map_location="cpu", weights_only=False)

    loads = []
    load_file_spy = train_script.load_file

    def spy(*args, **kwargs):
        result = load_file_spy(*args, **kwargs)
        loads.append((kwargs, result))
        return result

    monkeypatch.setattr(train_script, "load_file", spy)
    config = _write_inputs(tmp_path, max_epochs=2, append=True)
    del restore_locations[:]
    with global_options_context(config):
        restarted = train_script.restart(config)

    assert restarted.iepoch == 1
    # the checkpoint was requested on the CPU ...
    [(kwargs, dictionary)] = loads
    assert kwargs["map_location"] == "cpu"
    assert dictionary["state_dict"]["optim"]["state"]
    assert all(t.device.type == "cpu" for t in _tensors(dictionary))
    # ... and no storage of any checkpoint file was restored anywhere else
    assert restore_locations and set(restore_locations) == {"cpu"}
    # the restored state is the saved state, on the trainer's own device
    assert restarted.torch_device.type == "cpu"
    for shadow, expected in zip(
        restarted.ema.shadow_params, saved["state_dict"]["ema"]["shadow_params"]
    ):
        assert shadow.device.type == "cpu" and torch.equal(shadow, expected)
    restored = restarted.optim.state_dict()["state"]
    for index, expected in saved["state_dict"]["optim"]["state"].items():
        for key, value in expected.items():
            assert torch.equal(restored[index][key], value), (index, key)


def test_trainer_from_file_reads_checkpoint_on_cpu(cuda_tagged_run, restore_locations):
    tmp_path, run_dir = cuda_tagged_run
    # Trainer.from_file cannot raise max_epochs, so extend the finished run
    _retag_as_cuda0(run_dir / "trainer.pth", update=_extend_run)
    config = _write_inputs(tmp_path, max_epochs=2, append=True)
    del restore_locations[:]
    with global_options_context(config):
        trainer = Trainer.from_file(str(run_dir / "trainer.pth"), append=True)
    assert trainer.iepoch == 1
    assert restore_locations and set(restore_locations) == {"cpu"}


def test_load_file_map_location(cuda_tagged_run, restore_locations):
    _, run_dir = cuda_tagged_run
    path = str(run_dir / "trainer.pth")

    # the default keeps restoring every storage on the device it was saved from
    del restore_locations[:]
    with pytest.raises(AssertionError, match="restored on cuda:0"):
        load_file(supported_formats=_FORMATS, filename=path, enforced_format="torch")
    assert restore_locations[0] == "cuda:0"

    del restore_locations[:]
    loaded = load_file(
        supported_formats=_FORMATS,
        filename=path,
        enforced_format="torch",
        map_location="cpu",
    )
    assert restore_locations and set(restore_locations) == {"cpu"}
    assert all(t.device.type == "cpu" for t in _tensors(loaded))
    assert Path(loaded["progress"]["trainer_save_path"]).name == "trainer.pth"
