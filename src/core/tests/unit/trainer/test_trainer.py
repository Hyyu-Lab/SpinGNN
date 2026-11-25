"""
Trainer tests
"""
import pytest

import numpy as np
import tempfile
from os.path import isfile

import torch
from torch.nn import Linear

import nequip.scripts.train as train_script
from nequip.model import model_from_config
from nequip.data import AtomicDataDict
from nequip.train.trainer import Trainer
from nequip.utils.savenload import load_file
from nequip.nn import GraphModuleMixin


def dummy_builder():
    return DummyNet(3)


# set up two config to test
DEBUG = False
NATOMS = 3
NFRAMES = 10
minimal_config = dict(
    run_name="test",
    n_train=4,
    n_val=4,
    exclude_keys=["sth"],
    max_epochs=2,
    batch_size=2,
    learning_rate=1e-2,
    optimizer="Adam",
    seed=0,
    append=False,
    T_0=50,
    T_mult=2,
    loss_coeffs={"forces": 2},
    early_stopping_patiences={"loss": 50},
    early_stopping_lower_bounds={"LR": 1e-10},
    model_builders=[dummy_builder],
    chemical_symbols=["H", "C", "O"],
)


@pytest.fixture
def trainer():
    """
    Generate a class instance with minimal configurations
    """
    config = dict(minimal_config)
    config["default_dtype"] = str(torch.get_default_dtype())[len("torch.") :]
    with tempfile.TemporaryDirectory(prefix="output") as path:
        config["root"] = path
        model = model_from_config(config)
        c = Trainer(model=model, **config)
        yield c


class TestTrainerSetUp:
    """
    test initialization
    """

    def test_init(self, trainer):
        assert isinstance(trainer, Trainer)


class TestDuplicateError:
    def test_duplicate_id_2(self, temp_data):
        """
        check whether the Output class can automatically
        insert timestr when a workdir has pre-existed
        """

        config = dict(minimal_config, root=temp_data)

        model = DummyNet(3)
        Trainer(model=model, **config)

        with pytest.raises(RuntimeError):
            Trainer(model=model, **config)


class TestSaveLoad:
    def test_optimizer_kwargs_preserve_user_config(self, trainer):
        assert trainer.optimizer_kwargs is None
        assert len(trainer._resolved_optimizer_kwargs) > 1
        assert trainer.lr_scheduler_kwargs is None
        assert trainer._resolved_lr_scheduler_kwargs == {}

    @pytest.mark.parametrize("key", ("optimizer_kwargs", "lr_scheduler_kwargs"))
    def test_restart_optimizer_kwargs_accept_old_expanded_values(self, key):
        requested = {
            "amsgrad": True,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
        }
        saved = {
            **requested,
            "lr": 1.0e-3,
            "foreach": None,
            "fused": None,
        }
        assert train_script._restart_config_values_match(key, requested, saved)
        assert not train_script._restart_config_values_match(
            key, {**requested, "eps": 1.0e-6}, saved
        )

    @pytest.mark.parametrize("state_dict", [True, False])
    @pytest.mark.parametrize("training_progress", [True, False])
    def test_as_dict(self, trainer, state_dict, training_progress):

        dictionary = trainer.as_dict(
            state_dict=state_dict, training_progress=training_progress
        )

        assert "optimizer_kwargs" in dictionary
        assert state_dict == ("state_dict" in dictionary)
        assert training_progress == ("progress" in dictionary)
        assert dictionary["optimizer_kwargs"] is None

    @pytest.mark.parametrize("format, suffix", [("torch", "pth"), ("yaml", "yaml")])
    def test_save(self, trainer, format, suffix):

        with tempfile.NamedTemporaryFile(suffix="." + suffix) as tmp:
            file_name = tmp.name
            trainer.save(file_name, format=format)
            assert isfile(file_name), "fail to save to file"
            assert suffix in file_name

    @pytest.mark.parametrize("append", [True])  # , False])
    def test_from_file(self, trainer, append):

        format = "torch"

        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:

            file_name = tmp.name
            file_name = trainer.save(file_name, format=format)

            assert isfile(trainer.last_model_path)

            trainer1 = Trainer.from_file(file_name, format=format, append=append)

            for key in [
                "best_model_path",
                "last_model_path",
                "logfile",
                "epoch_log",
                "batch_log",
                "workdir",
            ]:
                v1 = getattr(trainer, key, None)
                v2 = getattr(trainer1, key, None)
                assert append == (v1 == v2)

            for iparam, group1 in enumerate(trainer.optim.param_groups):
                group2 = trainer1.optim.param_groups[iparam]

                for key in group1:
                    v1 = group1[key]
                    v2 = group2[key]
                    if key != "params":
                        assert v1 == v2


class TestAutomaticWeightedLoss:
    @staticmethod
    def make_trainer(path, lauto_loss, model_dtype=torch.float64):
        config = dict(minimal_config)
        config.update(
            root=path,
            run_name=f"awl_{lauto_loss}",
            device="cpu",
            default_dtype=str(model_dtype)[len("torch.") :],
            max_epochs=1,
            report_init_validation=False,
            lauto_loss=lauto_loss,
            loss_coeffs={
                AtomicDataDict.FORCE_KEY: 1.0,
                AtomicDataDict.TOTAL_ENERGY_KEY: 1.0,
            },
        )
        model = model_from_config(config).to(dtype=model_dtype)
        return Trainer(model=model, **config)

    def test_awl_is_optimized(self):
        with tempfile.TemporaryDirectory(prefix="awl_optim") as path:
            trainer = self.make_trainer(path, lauto_loss=True)
            awl_param = trainer.loss.awl.params

            assert awl_param.device == next(trainer.model.parameters()).device
            assert awl_param.dtype == next(trainer.model.parameters()).dtype
            assert len(trainer.optim.param_groups) == 2
            assert any(
                parameter is awl_param
                for parameter in trainer.optim.param_groups[1]["params"]
            )

            pred = {
                AtomicDataDict.FORCE_KEY: torch.full(
                    (2, 3), 2.0, dtype=awl_param.dtype, requires_grad=True
                ),
                AtomicDataDict.TOTAL_ENERGY_KEY: torch.full(
                    (1, 1), 3.0, dtype=awl_param.dtype, requires_grad=True
                ),
            }
            ref = {key: torch.zeros_like(value) for key, value in pred.items()}

            before = awl_param.detach().clone()
            trainer.optim.zero_grad(set_to_none=True)
            loss, _ = trainer.loss(pred=pred, ref=ref)
            loss.backward()
            assert awl_param.grad is not None
            assert torch.count_nonzero(awl_param.grad) == awl_param.numel()
            trainer.optim.step()

            assert not torch.equal(before, awl_param.detach())

    def test_awl_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory(prefix="awl_checkpoint") as path:
            trainer = self.make_trainer(path, lauto_loss=True)
            expected = torch.tensor([1.25, 0.75], dtype=trainer.loss.awl.params.dtype)
            with torch.no_grad():
                trainer.loss.awl.params.copy_(expected)

            checkpoint = trainer.save(f"{path}/trainer.pth")
            restored = Trainer.from_file(checkpoint, append=True)

            assert torch.equal(restored.loss.awl.params, expected)
            assert len(restored.optim.param_groups) == 2
            assert (
                restored.optim.param_groups[1]["params"][0] is restored.loss.awl.params
            )

    def test_one_epoch_toy_training_updates_awl(self, nequip_dataset):
        with tempfile.TemporaryDirectory(prefix="awl_toy_train") as path:
            trainer = self.make_trainer(
                path,
                lauto_loss=True,
                model_dtype=torch.get_default_dtype(),
            )
            before = trainer.loss.awl.params.detach().clone()

            trainer.set_dataset(nequip_dataset)
            trainer.train()

            after = trainer.loss.awl.params.detach()
            checkpoint = load_file(
                filename=trainer.trainer_save_path,
                supported_formats=dict(torch=["pt", "pth"]),
                enforced_format="torch",
            )
            saved = checkpoint["state_dict"]["loss_awl"]["params"]
            with open(trainer.logfile) as logfile:
                log = logfile.read()

            assert trainer.iepoch == 1
            assert not torch.equal(before, after)
            assert torch.equal(saved, after.cpu())
            assert "! AWL parameters [" in log

    def test_disabled_awl_keeps_model_only_optimizer_group(self):
        with tempfile.TemporaryDirectory(prefix="awl_disabled") as path:
            trainer = self.make_trainer(path, lauto_loss=False)

            assert not hasattr(trainer.loss, "awl")
            assert len(trainer.optim.param_groups) == 1
            optimizer_params = trainer.optim.param_groups[0]["params"]
            model_params = list(trainer.model.parameters())
            assert len(optimizer_params) == len(model_params)
            assert all(
                optimizer_param is model_param
                for optimizer_param, model_param in zip(
                    optimizer_params,
                    model_params,
                )
            )


class TestData:
    @pytest.mark.parametrize("mode", ["random", "sequential"])
    def test_split(self, trainer, nequip_dataset, mode):

        trainer.train_val_split = mode
        trainer.set_dataset(nequip_dataset)
        for i, batch in enumerate(trainer.dl_train):
            print(i, batch)


class TestTrain:
    def test_train(self, trainer, nequip_dataset):

        v0 = get_param(trainer.model)
        trainer.set_dataset(nequip_dataset)
        trainer.train()
        v1 = get_param(trainer.model)

        assert not np.allclose(v0, v1), "fail to train parameters"
        assert isfile(trainer.last_model_path), "fail to save best model"

    def test_batch_metrics_interval_preserves_epoch_csv(self, nequip_dataset):
        with tempfile.TemporaryDirectory(prefix="batch_metrics_interval") as path:
            epoch_rows = []
            for name, interval in (("every", 1), ("sparse", 3)):
                config = dict(
                    minimal_config,
                    root=path,
                    run_name=name,
                    max_epochs=1,
                    report_init_validation=False,
                    batch_metrics_log_interval=interval,
                    default_dtype="float64",
                )
                torch.manual_seed(0)
                trainer = Trainer(model=DummyNet(3), **config)
                trainer.set_dataset(nequip_dataset)
                logged_batches = []
                original_batch_log = trainer.end_of_batch_log

                def record_batch_log(batch_type):
                    if trainer.batch_metrics is not None:
                        logged_batches.append((batch_type, trainer.ibatch))
                    original_batch_log(batch_type)

                trainer.end_of_batch_log = record_batch_log
                trainer.train()
                run_dir = f"{path}/{name}"
                with open(f"{run_dir}/metrics_epoch.csv") as stream:
                    stream.read()
                epoch_rows.append(
                    {
                        key: value
                        for key, value in trainer.mae_dict.items()
                        if key not in {"wall", "cumulative_wall"}
                    }
                )
                with open(f"{run_dir}/metrics_batch_train.csv") as stream:
                    train_lines = stream.readlines()
                with open(f"{run_dir}/metrics_batch_val.csv") as stream:
                    validation_lines = stream.readlines()
                assert train_lines and validation_lines
                assert trainer.batch_metrics_log_interval == interval
                assert trainer._should_materialize_batch_log() is True
                expected_batches = [0, 1] if interval == 1 else [1]
                assert [
                    batch
                    for category, batch in logged_batches
                    if category == "training"
                ] == expected_batches
                assert [
                    batch
                    for category, batch in logged_batches
                    if category == "validation"
                ] == [0]
            # Wall time is intentionally variable; all epoch metric/loss values
            # that drive training decisions must be identical.
            assert epoch_rows[0] == epoch_rows[1]

    def test_load_w_revision(self, trainer):

        with tempfile.TemporaryDirectory() as folder:

            file_name = trainer.save(f"{folder}/a.pth")

            dictionary = load_file(
                supported_formats=dict(torch=["pt", "pth"]),
                filename=file_name,
                enforced_format="torch",
            )

            dictionary["root"] = folder
            dictionary["progress"]["stop_arg"] = None
            dictionary["max_epochs"] *= 2
            trainer1 = Trainer.from_dict(dictionary, append=False)
            assert trainer1.iepoch == trainer.iepoch
            assert trainer1.max_epochs == minimal_config["max_epochs"] * 2

    def test_restart_training(self, trainer, nequip_dataset):
        _ = trainer.model
        _ = trainer.device
        _ = trainer.optim
        trainer.set_dataset(nequip_dataset)
        trainer.train()

        v0 = get_param(trainer.model)
        with tempfile.TemporaryDirectory() as folder:
            file_name = trainer.save(f"{folder}/hello.pth")

            dictionary = load_file(
                filename=file_name,
                supported_formats=dict(torch=["pt", "pth"]),
                enforced_format="torch",
            )

            dictionary["progress"]["stop_arg"] = None
            dictionary["max_epochs"] *= 2
            dictionary["root"] = folder

            trainer1 = Trainer.from_dict(dictionary, append=False)
            trainer1.stop_arg = None
            trainer1.max_epochs *= 2
            trainer1.set_dataset(nequip_dataset)
            trainer1.train()
            v1 = get_param(trainer1.model)

            assert not np.allclose(v0, v1), "fail to train parameters"


class TestRescale:
    def test_scaling(self, scale_train):

        trainer = scale_train

        data = trainer.dataset_train[0]

        def get_loss_val(validation):
            trainer.ibatch = 0
            trainer.n_batches = 10
            trainer.reset_metrics()
            trainer.batch_step(
                data=data,
                validation=validation,
            )
            metrics, _ = trainer.metrics.flatten_metrics(
                trainer.batch_metrics,
            )
            return trainer.batch_losses, metrics

        ref_loss, ref_met = get_loss_val(True)
        loss, met = get_loss_val(False)

        # metrics should always be in real unit
        for k in ref_met:
            assert np.allclose(ref_met[k], met[k])
        # loss should always be in normalized unit
        for k in ref_loss:
            assert np.allclose(ref_loss[k], loss[k])

        assert np.allclose(np.sqrt(loss["loss_f"]) * trainer.scale, met["f_rmse"])


def get_param(model):
    v = []
    for p in model.parameters():
        v += [np.hstack(p.data.tolist())]
    v = np.hstack(v)
    return v


class DummyNet(GraphModuleMixin, torch.nn.Module):
    def __init__(self, ndim, nydim=1) -> None:
        super().__init__()
        self.ndim = ndim
        self.nydim = nydim
        self.linear1 = Linear(ndim, nydim)
        self.linear2 = Linear(ndim, nydim * 3)
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


class DummyScale(torch.nn.Module):
    """mimic the rescale model"""

    def __init__(self, key, scale, shift) -> None:
        super().__init__()
        self.key = key
        self.scale_by = torch.as_tensor(scale, dtype=torch.get_default_dtype())
        self.shift_by = torch.as_tensor(shift, dtype=torch.get_default_dtype())
        self.linear2 = Linear(3, 3)

    def forward(self, data):
        out = self.linear2(data["pos"])
        if not self.training:
            out = out * self.scale_by
            out = out + self.shift_by
        return {self.key: out}

    def scale(self, data, force_process=False):
        data = data.copy()
        if force_process or not self.training:
            data[self.key] = data[self.key] * self.scale_by
            data[self.key] = data[self.key] + self.shift_by
        return data

    def unscale(self, data, force_process=False):
        data = data.copy()
        if force_process or self.training:
            data[self.key] = data[self.key] - self.shift_by
            data[self.key] = data[self.key] / self.scale_by
        return data


@pytest.fixture(scope="class")
def scale_train(nequip_dataset):
    with tempfile.TemporaryDirectory(prefix="output") as path:
        trainer = Trainer(
            model=DummyScale(AtomicDataDict.FORCE_KEY, scale=1.3, shift=1),
            n_train=4,
            n_val=4,
            max_epochs=0,
            batch_size=2,
            loss_coeffs=AtomicDataDict.FORCE_KEY,
            root=path,
            run_name="test_scale",
        )
        trainer.set_dataset(nequip_dataset)
        trainer.train()
        trainer.scale = 1.3
        yield trainer
