"""float64 data and losses with a float32 model (``model_dtype``).

Covers the build cast in ``nequip-train``, training steps, a restart, and
``nequip-evaluate``; a module marked ``_spingnn_keep_dtype`` keeps float64.
Deployment of a mixed model/data dtype is rejected.
"""

import logging
import sys

import numpy as np
import pytest
import torch
import yaml

import nequip.scripts.deploy as deploy_script
import nequip.scripts.evaluate as evaluate_script
import nequip.scripts.train as train_script
from nequip.ase import NequIP_Latent_Opt_Calculator
from nequip.data import AtomicData, AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.train.trainer import Trainer
from nequip.utils import Config
from nequip.utils._global_options import (
    check_deployable_model_dtype,
    global_options_context,
)

_MIXED_DEPLOY_ERROR = "mixed model/data dtype"


class _KeepFloat64(torch.nn.Module):
    """Stands in for a submodule that manages its own dtypes."""

    _spingnn_keep_dtype = True

    def __init__(self):
        super().__init__()
        self.inner = torch.nn.Linear(3, 3, dtype=torch.float64)
        self.register_buffer("shift", torch.full((3,), 0.5, dtype=torch.float64))


class _MixedNet(GraphModuleMixin, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.keep = _KeepFloat64()
        self.linear = torch.nn.Linear(3, 3)
        self._init_irreps(
            irreps_in={"pos": "1x1o"},
            irreps_out={AtomicDataDict.FORCE_KEY: "1x1o"},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = data.copy()
        features = self.keep.inner(data["pos"]) + self.keep.shift
        data[AtomicDataDict.FORCE_KEY] = self.linear(
            features.to(self.linear.weight.dtype)
        )
        return data


def mixed_builder():
    return _MixedNet()


def _assert_model_dtypes(model):
    dtypes = {name: t.dtype for name, t in model.state_dict().items()}
    assert dtypes == {
        "keep.shift": torch.float64,
        "keep.inner.weight": torch.float64,
        "keep.inner.bias": torch.float64,
        "linear.weight": torch.float32,
        "linear.bias": torch.float32,
    }


def _write_inputs(tmp_path, max_epochs, append):
    rng = np.random.default_rng(3)
    n_frames, n_atoms = 8, 3
    np.savez(
        tmp_path / "data.npz",
        R=rng.normal(size=(n_frames, n_atoms, 3)),
        F=rng.normal(size=(n_frames, n_atoms, 3)),
        z=np.ones(n_atoms, dtype=np.int64),
    )
    config = dict(
        root=str(tmp_path / "results"),
        run_name="mixed",
        seed=0,
        dataset_seed=0,
        dataset="npz",
        dataset_file_name=str(tmp_path / "data.npz"),
        key_mapping={"z": "atomic_numbers", "F": "forces", "R": "pos"},
        npz_fixed_field_keys=["atomic_numbers"],
        chemical_symbols=["H"],
        r_max=2.5,
        model_builders=[f"{__name__}.mixed_builder"],
        default_dtype="float64",
        model_dtype="float32",
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


def test_load_keeps_marked_dtype_without_model_dtype(tmp_path):
    # no model_dtype: the load casts to default_dtype, the marked module keeps
    # its native float64
    config = dict(
        model_builders=[f"{__name__}.mixed_builder"],
        default_dtype="float32",
        chemical_symbols=["H"],
    )
    with global_options_context(config):
        torch.save(mixed_builder().state_dict(), tmp_path / "model.pth")
        model, _ = Trainer.load_model_from_training_session(
            tmp_path, "model.pth", config_dictionary=config
        )
    _assert_model_dtypes(model)


def test_float64_data_float32_model(tmp_path, monkeypatch, caplog):
    config = _write_inputs(tmp_path, max_epochs=2, append=False)
    with global_options_context(config):
        trainer = train_script.fresh_start(config)
        _assert_model_dtypes(trainer.model)
        trainer.save()
        trainer.train()
        assert trainer.iepoch == 2
        _assert_model_dtypes(trainer.model)
        # data, targets and the loss stay in default_dtype
        batch = AtomicData.to_AtomicDataDict(next(iter(trainer.dl_train)))
        assert batch[AtomicDataDict.POSITIONS_KEY].dtype == torch.float64
        assert batch[AtomicDataDict.FORCE_KEY].dtype == torch.float64
        pred = trainer.model(
            {k: v for k, v in batch.items() if k != AtomicDataDict.FORCE_KEY}
        )
        assert pred[AtomicDataDict.FORCE_KEY].dtype == torch.float32
        loss, _ = trainer.loss(pred=pred, ref=batch)
        assert loss.dtype == torch.float64
        assert all(
            shadow.dtype == param.dtype
            for shadow, param in zip(
                trainer.ema.shadow_params, trainer.model.parameters()
            )
        )

    run_dir = tmp_path / "results" / "mixed"
    saved = torch.load(run_dir / "last_model.pth", weights_only=True)
    assert saved["linear.weight"].dtype == torch.float32
    assert saved["keep.inner.weight"].dtype == torch.float64

    # restart: the model is rebuilt in default_dtype, then cast to model_dtype
    config = _write_inputs(tmp_path, max_epochs=3, append=True)
    with global_options_context(config):
        restarted = train_script.restart(config)
        assert restarted.iepoch == 2
        _assert_model_dtypes(restarted.model)
        for name, value in restarted.model.state_dict().items():
            assert torch.equal(value, saved[name]), name
        restarted.train()
        assert restarted.iepoch == 3
        _assert_model_dtypes(restarted.model)

    # evaluate loads the model through the same path
    with global_options_context(config):
        model, _ = Trainer.load_model_from_training_session(run_dir)
        _assert_model_dtypes(model)
    monkeypatch.chdir(tmp_path)
    eval_args = [
        "--train-dir",
        str(run_dir),
        "--output",
        str(tmp_path / "out.xyz"),
        "--device",
        "cpu",
    ]
    # evaluate.main prints its help and exits when sys.argv has length 1, which
    # is the case in a pytest-xdist worker (sys.argv == ["-c"])
    monkeypatch.setattr(sys, "argv", ["nequip-evaluate"] + eval_args)
    with caplog.at_level(logging.INFO):
        evaluate_script.main(args=eval_args, running_as_script=False)
    results = [r.getMessage() for r in caplog.records if "f_mae" in r.getMessage()]
    assert results, "no evaluation metrics were reported"
    mae = float(results[-1].split("f_mae")[1].split("=")[1].split()[0])
    assert np.isfinite(mae)
    assert (tmp_path / "out.xyz").is_file()

    # deployment of the mixed-dtype training session is rejected
    with global_options_context():
        with pytest.raises(NotImplementedError, match=_MIXED_DEPLOY_ERROR):
            deploy_script.load_model_for_deploy(train_dir=str(run_dir))


def test_check_deployable_model_dtype():
    check_deployable_model_dtype({"default_dtype": "float64"})
    check_deployable_model_dtype({"default_dtype": "float32", "model_dtype": None})
    check_deployable_model_dtype({"default_dtype": "float64", "model_dtype": "float64"})
    with pytest.raises(NotImplementedError, match=_MIXED_DEPLOY_ERROR):
        check_deployable_model_dtype(
            {"default_dtype": "float64", "model_dtype": "float32"}
        )


def test_deploy_paths_reject_mixed_dtype(tmp_path):
    config = dict(
        model_builders=[f"{__name__}.mixed_builder"],
        default_dtype="float64",
        model_dtype="float32",
        chemical_symbols=["H"],
        r_max=2.5,
    )
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    with global_options_context():
        # nequip-deploy build --train-dir (rejected before any weights are read)
        with pytest.raises(NotImplementedError, match=_MIXED_DEPLOY_ERROR):
            deploy_script.load_model_for_deploy(train_dir=str(tmp_path))
        # nequip-deploy build --model <config>
        with pytest.raises(NotImplementedError, match=_MIXED_DEPLOY_ERROR):
            deploy_script.load_model_for_deploy(
                model_path=str(tmp_path / "config.yaml")
            )
        # the latent-opt ASE calculator pairs a deployed model with the
        # training-session model (rejected before the deployed file is read)
        with pytest.raises(NotImplementedError, match=_MIXED_DEPLOY_ERROR):
            NequIP_Latent_Opt_Calculator.from_training_session(str(tmp_path))
