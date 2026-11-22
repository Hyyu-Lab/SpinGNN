"""Named optimizer parameter groups (``optimizer_param_groups``)."""

import tempfile

import pytest
import torch
import yaml

from nequip.data import AtomicDataDict
from nequip.model import model_from_config
from nequip.nn import GraphModuleMixin
from nequip.train._param_groups import build_optimizer_param_groups
from nequip.train.trainer import Trainer
from nequip.utils.savenload import load_file


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = torch.nn.Linear(3, 3)
        self.expert_gate = torch.nn.Linear(3, 1)
        self.conv = torch.nn.Linear(3, 3)
        self.conv_gate = torch.nn.Parameter(torch.zeros(1))


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.frozen = torch.nn.Linear(3, 3)
        self.frozen.requires_grad_(False)
        self.head = _Head()


def _names(structure):
    return {group["name"]: group["params"] for group in structure}


class TestBuild:
    def test_first_match_catch_all_and_order(self):
        model = _Toy()
        specs = yaml.safe_load(
            """
            - {name: readout, lr: 1e-4, weight_decay: 1.0e-5}
            - {name: expert, match: ["re:\\\\.expert_"], lr: 1.0e-3, weight_decay: 0}
            - {name: conv, match: ["re:^head\\\\.conv(?!_gate)"], lr: 1e-5}
            - {name: gate, match: [head.conv_gate, head.conv], lr: 1e-5}
            - {name: backbone, match: backbone, lr: 3e-6}
            """
        )
        param_groups, structure = build_optimizer_param_groups(model, specs)

        # list order is the optimizer group order; the catch-all may come first
        assert [group["name"] for group in param_groups] == [
            "readout",
            "expert",
            "conv",
            "gate",
            "backbone",
        ]
        assert _names(structure) == {
            "readout": ["head.readout.weight", "head.readout.bias"],
            "expert": ["head.expert_gate.weight", "head.expert_gate.bias"],
            "conv": ["head.conv.weight", "head.conv.bias"],
            "gate": ["head.conv_gate"],
            "backbone": ["backbone.weight", "backbone.bias"],
        }
        named = dict(model.named_parameters())
        for group, layout in zip(param_groups, structure):
            assert all(p is named[n] for p, n in zip(group["params"], layout["params"]))
        # frozen parameters are not grouped
        assert all("frozen" not in n for g in structure for n in g["params"])
        # options are passed through; YAML strings like ``1e-4`` become floats
        assert param_groups[0]["lr"] == 1e-4
        assert param_groups[0]["weight_decay"] == 1e-5
        assert param_groups[1]["weight_decay"] == 0
        assert param_groups[4]["lr"] == 3e-6
        assert "weight_decay" not in param_groups[4]

    @pytest.mark.parametrize(
        "specs, message",
        [
            ([], "non-empty list"),
            ([{"match": ["head"]}], "non-empty string `name`"),
            ([{"name": "a"}, {"name": "a", "match": ["x"]}], "duplicate group name"),
            ([{"name": "a"}, {"name": "b"}], "only one group may omit `match`"),
            ([{"name": "a", "match": []}], "non-empty list of non-empty strings"),
            ([{"name": "a", "match": ["re:("]}], "invalid regular expression"),
            ([{"name": "a", "params": []}], "`params` is set by `match`"),
            ([{"name": "a", "match": ["head"]}], "match no optimizer_param_groups"),
            (
                [{"name": "all"}, {"name": "typo", "match": ["head.readuot"]}],
                r"\['typo'\] match no trainable parameter",
            ),
            ([{"name": "a,b"}], "may only contain letters"),
        ],
    )
    def test_errors(self, specs, message):
        with pytest.raises(ValueError, match=message):
            build_optimizer_param_groups(_Toy(), specs)


def _dummy_builder():
    return _DummyNet(3)


class _DummyNet(GraphModuleMixin, torch.nn.Module):
    def __init__(self, ndim) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(ndim, 1)
        self.linear2 = torch.nn.Linear(ndim, 3)
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
        energy = self.linear1(x)
        batch = data[AtomicDataDict.BATCH_KEY].long()
        energy = torch.zeros(
            (int(batch.max()) + 1, 1), dtype=energy.dtype, device=energy.device
        ).index_add_(0, batch, energy)
        data[AtomicDataDict.FORCE_KEY] = self.linear2(x)
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = energy
        return data


_GROUPS = [
    {"name": "energy", "match": ["linear1."], "lr": 1.0e-2, "weight_decay": 1.0e-4},
    {"name": "forces", "lr": 1.0e-3},
]


def _config(root, **overrides):
    config = dict(
        root=root,
        run_name="groups",
        n_train=4,
        n_val=4,
        max_epochs=2,
        batch_size=2,
        learning_rate=5.0e-3,
        seed=0,
        loss_coeffs={"forces": 1.0, "total_energy": 1.0},
        early_stopping_patiences={"validation_loss": 50},
        early_stopping_lower_bounds={"LR": 1e-10},
        model_builders=[_dummy_builder],
        chemical_symbols=["H", "C", "O"],
        default_dtype=str(torch.get_default_dtype())[len("torch.") :],
        optimizer_param_groups=_GROUPS,
        verbose="WARNING",
    )
    config.update(overrides)
    return config


def _trainer(root, **overrides):
    config = _config(root, **overrides)
    return Trainer(model=model_from_config(config), **config)


class TestTrainer:
    def test_groups_in_optimizer(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = _trainer(root)
            groups = trainer.optim.param_groups
            assert [g["name"] for g in groups] == ["energy", "forces"]
            assert [g["lr"] for g in groups] == [1.0e-2, 1.0e-3]
            assert [g["weight_decay"] for g in groups] == [1.0e-4, 0]
            named = dict(trainer.model.named_parameters())
            for group, names in zip(
                groups,
                (
                    ["linear1.weight", "linear1.bias"],
                    ["linear2.weight", "linear2.bias"],
                ),
            ):
                assert len(group["params"]) == len(names)
                assert all(p is named[n] for p, n in zip(group["params"], names))
            # default: the reported LR is group 0
            assert trainer._early_stopping_lr_group_index == 0

    def test_ladder_training_rejected(self):
        # ladders unfreeze parameters after the optimizer has been built
        with tempfile.TemporaryDirectory() as root:
            with pytest.raises(ValueError, match="cannot be combined"):
                _trainer(root, train_spin_ladder=True)

    def test_unknown_group_option(self):
        with tempfile.TemporaryDirectory() as root:
            groups = [{"name": "all", "weight_decy": 0.1}]
            with pytest.raises(
                ValueError, match=r"unknown option\(s\) \['weight_decy'\]"
            ):
                _trainer(root, optimizer_param_groups=groups)

    def test_lr_group_selection(self, nequip_dataset):
        with tempfile.TemporaryDirectory() as root:
            with pytest.raises(ValueError, match="no optimizer param group named"):
                _trainer(root, early_stopping_lr_group="missing")
        with tempfile.TemporaryDirectory() as root:
            with pytest.raises(ValueError, match="out of range"):
                _trainer(root, early_stopping_lr_group=2)
        with tempfile.TemporaryDirectory() as root:
            trainer = _trainer(root, early_stopping_lr_group="forces", max_epochs=1)
            trainer.set_dataset(nequip_dataset)
            trainer.train()
            assert trainer._early_stopping_lr_group_index == 1
            assert trainer.mae_dict["LR"] == 1.0e-3

    def test_per_group_lr_columns(self, nequip_dataset):
        with tempfile.TemporaryDirectory() as root:
            # the metrics CSV files are written at INFO level
            trainer = _trainer(root, max_epochs=2, verbose="INFO")
            trainer.set_dataset(nequip_dataset)
            trainer.train()
            # LR keeps its meaning (group 0); one LR_<name> column per group
            assert trainer.mae_dict["LR"] == 1.0e-2
            assert trainer.mae_dict["LR_energy"] == 1.0e-2
            assert trainer.mae_dict["LR_forces"] == 1.0e-3
            run_dir = f"{root}/groups"
            # epochs 1-2 in metrics_epoch.csv, the initial validation separately
            for name, n_rows in (
                ("metrics_epoch.csv", 2),
                ("metrics_initialization.csv", 1),
            ):
                with open(f"{run_dir}/{name}") as csv:
                    header, *rows = [line.strip() for line in csv if line.strip()]
                columns = [c.strip() for c in header.split(",")]
                assert columns[:5] == ["epoch", "wall", "LR", "LR_energy", "LR_forces"]
                for row in rows:
                    values = [float(v) for v in row.split(",")]
                    assert len(values) == len(columns)
                    assert values[2:5] == [1.0e-2, 1.0e-2, 1.0e-3]
                assert len(rows) == n_rows

    def test_no_lr_columns_without_groups(self, nequip_dataset):
        with tempfile.TemporaryDirectory() as root:
            trainer = _trainer(
                root, max_epochs=1, optimizer_param_groups=None, verbose="INFO"
            )
            trainer.set_dataset(nequip_dataset)
            trainer.train()
            assert not any(k.startswith("LR_") for k in trainer.mae_dict)
            with open(f"{root}/groups/metrics_epoch.csv") as csv:
                header = csv.readline()
            assert header.startswith("epoch, wall, LR,")
            assert "LR_" not in header

    def test_early_stopping_on_group_lr(self, nequip_dataset):
        with tempfile.TemporaryDirectory() as root:
            trainer = _trainer(
                root,
                max_epochs=5,
                early_stopping_lower_bounds={"LR_forces": 2.0e-3},
            )
            assert trainer.early_stopping_conds.lower_bounds == {"LR_forces": 2.0e-3}
            trainer.set_dataset(nequip_dataset)
            trainer.train()
            assert "LR_forces is smaller than" in trainer.stop_arg
            assert trainer.iepoch == 0  # stops after the initial validation

    def test_plateau_scales_all_groups(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = _trainer(
                root,
                lr_scheduler_name="ReduceLROnPlateau",
                lr_scheduler_patience=0,
                lr_scheduler_factor=0.5,
            )
            for value in (1.0, 2.0, 3.0):
                trainer.lr_sched.step(metrics=value)
            assert [g["lr"] for g in trainer.optim.param_groups] == [
                1.0e-2 * 0.25,
                1.0e-3 * 0.25,
            ]

    def test_resume_keeps_groups(self, nequip_dataset):
        with tempfile.TemporaryDirectory() as root:
            trainer = _trainer(root, max_epochs=1)
            trainer.set_dataset(nequip_dataset)
            trainer.train()
            saved = load_file(
                supported_formats=dict(torch=["pt", "pth"]),
                filename=trainer.trainer_save_path,
                enforced_format="torch",
            )
            assert saved["state_dict"]["optim_param_groups"] == [
                {"name": "energy", "params": ["linear1.weight", "linear1.bias"]},
                {"name": "forces", "params": ["linear2.weight", "linear2.bias"]},
            ]

            dictionary = dict(saved, max_epochs=2)
            resumed = Trainer.from_dict(dictionary, append=True)
            assert [g["name"] for g in resumed.optim.param_groups] == [
                "energy",
                "forces",
            ]
            for old, new in zip(
                trainer.optim.state_dict()["state"].values(),
                resumed.optim.state_dict()["state"].values(),
            ):
                assert torch.equal(old["exp_avg"], new["exp_avg"])
            resumed.set_dataset(nequip_dataset)
            resumed.train()
            assert resumed.iepoch == 2

            # same group sizes, different layout: torch alone would accept it
            swapped = dict(
                saved,
                max_epochs=2,
                optimizer_param_groups=[
                    {"name": "forces", "match": ["linear2."], "lr": 1.0e-3},
                    {"name": "energy", "lr": 1.0e-2, "weight_decay": 1.0e-4},
                ],
            )
            with pytest.raises(RuntimeError, match="optimizer param groups of the"):
                Trainer.from_dict(swapped, append=True)

            # a grouped checkpoint cannot resume without groups, and vice versa
            ungrouped = dict(saved, max_epochs=2, optimizer_param_groups=None)
            with pytest.raises(RuntimeError, match=r"current \[no optimizer_param"):
                Trainer.from_dict(ungrouped, append=True)
            legacy = dict(saved, max_epochs=2)
            legacy["state_dict"] = {
                k: v
                for k, v in saved["state_dict"].items()
                if k != "optim_param_groups"
            }
            with pytest.raises(RuntimeError, match=r"checkpoint \[no optimizer_param"):
                Trainer.from_dict(legacy, append=True)
