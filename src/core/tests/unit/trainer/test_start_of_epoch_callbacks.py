"""``start_of_epoch_callbacks``: call points, order, and restart behaviour."""

import tempfile

import torch

from nequip.data import AtomicDataDict
from nequip.model import model_from_config
from nequip.nn import GraphModuleMixin
from nequip.train.trainer import Trainer
from nequip.utils.savenload import load_file

EVENTS = []


def record_start(trainer):
    # state restored by a restart is visible to the first call
    EVENTS.append(("start", trainer.iepoch, trainer.model.scale.item()))
    with torch.no_grad():
        trainer.model.scale.fill_(float(trainer.iepoch + 1))


def record_batch(trainer):
    EVENTS.append(("batch", trainer.iepoch))


class _ScaledNet(GraphModuleMixin, torch.nn.Module):
    """Toy model with a persistent buffer set by a start-of-epoch callback."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.register_buffer("scale", torch.zeros(()))
        self._init_irreps(
            irreps_in={"pos": "1x1o"},
            irreps_out={AtomicDataDict.FORCE_KEY: "1x1o"},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = data.copy()
        data[AtomicDataDict.FORCE_KEY] = self.linear(data["pos"]) * (1 + self.scale)
        return data


def _builder():
    return _ScaledNet()


def _trainer(root, **overrides):
    config = dict(
        root=root,
        run_name="start_of_epoch",
        n_train=4,
        n_val=2,
        max_epochs=2,
        batch_size=2,
        validation_batch_size=2,
        seed=0,
        loss_coeffs={"forces": 1.0},
        early_stopping_patiences={"validation_loss": 50},
        model_builders=[_builder],
        chemical_symbols=["H", "C", "O"],
        default_dtype=str(torch.get_default_dtype())[len("torch.") :],
        # a direct callable here; dotted paths (plain YAML strings) for the
        # recorder below and in the DDP test of start_of_epoch_callbacks
        start_of_epoch_callbacks=[record_start],
        end_of_batch_callbacks=[f"{__name__}.record_batch"],
        verbose="WARNING",
    )
    config.update(overrides)
    return Trainer(model=model_from_config(config), **config)


def test_called_before_each_training_epoch(nequip_dataset):
    EVENTS.clear()
    with tempfile.TemporaryDirectory() as root:
        trainer = _trainer(root)
        trainer.set_dataset(nequip_dataset)
        trainer.train()
    # initial validation (iepoch -1) is not a training epoch; 2 train + 1 val
    # batches per epoch follow the start call of that epoch
    assert EVENTS == [
        ("batch", -1),
        ("start", 0, 0.0),
        ("batch", 0),
        ("batch", 0),
        ("batch", 0),
        ("start", 1, 1.0),
        ("batch", 1),
        ("batch", 1),
        ("batch", 1),
    ]


def test_called_once_before_first_resumed_epoch(nequip_dataset):
    EVENTS.clear()
    with tempfile.TemporaryDirectory() as root:
        trainer = _trainer(root, report_init_validation=False)
        trainer.set_dataset(nequip_dataset)
        trainer.train()
        assert [e for e in EVENTS if e[0] == "start"] == [
            ("start", 0, 0.0),
            ("start", 1, 1.0),
        ]

        EVENTS.clear()
        dictionary = load_file(
            supported_formats=dict(torch=["pt", "pth"]),
            filename=trainer.trainer_save_path,
            enforced_format="torch",
        )
        dictionary["max_epochs"] = 4
        resumed = Trainer.from_dict(dictionary, append=True)
        assert EVENTS == []
        resumed.set_dataset(nequip_dataset)
        resumed.train()
    # the restored buffer (2.0, set at the start of epoch 1) is seen first
    assert [e for e in EVENTS if e[0] == "start"] == [
        ("start", 2, 2.0),
        ("start", 3, 3.0),
    ]
    assert EVENTS[0] == ("start", 2, 2.0)
