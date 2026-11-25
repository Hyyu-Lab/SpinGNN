"""The smoke report must derive updates from real optimizer state."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[5] / "tools/ci/e2ip_training_smoke.py"
SPEC = importlib.util.spec_from_file_location("e2ip_training_smoke", SCRIPT)
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def adam_checkpoint(steps=3):
    used = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    optimizer = torch.optim.Adam([used, unused], lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        used.square().sum().backward()
        optimizer.step()
    return {"state_dict": {"optim": optimizer.state_dict()}}


def test_actual_adam_counters_and_unused_parameters_are_reported():
    result = SMOKE.optimizer_checkpoint_evidence(adam_checkpoint(), expected_steps=3)
    assert result["steps"] == 3
    assert result["parameter_steps"] == {"0": 3}
    assert result["initialized_parameter_states"] == 1
    assert result["uninitialized_parameter_ids"] == [1]


def test_planned_work_cannot_substitute_for_completed_updates():
    with pytest.raises(RuntimeError, match="do not match"):
        SMOKE.optimizer_checkpoint_evidence(adam_checkpoint(), expected_steps=60)


@pytest.mark.parametrize(
    "counter", [None, True, -1, 0, 1.5, float("nan"), float("inf"), torch.tensor([3.0])]
)
def test_invalid_optimizer_counters_are_rejected(counter):
    checkpoint = adam_checkpoint()
    checkpoint["state_dict"]["optim"]["state"][0]["step"] = counter
    with pytest.raises(RuntimeError, match="counter"):
        SMOKE.optimizer_checkpoint_evidence(checkpoint, expected_steps=3)


def test_partially_updated_parameters_are_not_hidden_by_the_maximum():
    checkpoint = adam_checkpoint()
    state = checkpoint["state_dict"]["optim"]["state"]
    state[1] = copy.deepcopy(state[0])
    state[1]["step"] = torch.tensor(2.0)
    with pytest.raises(RuntimeError, match="do not match"):
        SMOKE.optimizer_checkpoint_evidence(checkpoint, expected_steps=3)


@pytest.mark.parametrize("problem", ["empty", "unknown", "duplicate"])
def test_incomplete_optimizer_identity_is_rejected(problem):
    checkpoint = adam_checkpoint()
    optimizer = checkpoint["state_dict"]["optim"]
    if problem == "empty":
        optimizer["state"] = {}
    elif problem == "unknown":
        optimizer["state"][99] = optimizer["state"].pop(0)
    else:
        optimizer["param_groups"][0]["params"].append(0)
    with pytest.raises(RuntimeError):
        SMOKE.optimizer_checkpoint_evidence(checkpoint, expected_steps=3)
