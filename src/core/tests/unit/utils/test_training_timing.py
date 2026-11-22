"""CUDA timing must include completion of the measured optimizer step."""

from types import SimpleNamespace

import torch

from nequip.data import AtomicDataDict
from nequip.utils.unittests import parity_tests


def test_training_walltime_synchronizes_before_reading_clock(monkeypatch):
    events = []
    weight = torch.nn.Parameter(torch.tensor(0.5))

    class Model(torch.nn.Module):
        def forward(self, data):
            return {
                AtomicDataDict.FORCE_KEY: weight.pow(3),
                AtomicDataDict.SPIN_FORCE_VERT_KEY: weight.square(),
                AtomicDataDict.STRESS_KEY: weight,
            }

    class Optimizer:
        def zero_grad(self, set_to_none):
            weight.grad = None

        def step(self):
            events.append("step")

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))

    def clock():
        events.append("clock")
        return float(len(events))

    monkeypatch.setattr(parity_tests, "perf_counter", clock)
    data = {AtomicDataDict.POSITIONS_KEY: SimpleNamespace(device=torch.device("cuda"))}
    loss, elapsed = parity_tests._BaseParityTests._loss_training_step(
        Model(), Optimizer(), data
    )
    assert loss > 0 and elapsed > 0
    assert events == ["sync", "clock", "step", "sync", "clock"]
