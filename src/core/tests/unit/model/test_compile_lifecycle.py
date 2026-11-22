"""Warm full-FX calls preserve admission, live weights, and precision policy."""

import copy
import warnings

import e3nn
import pytest
import torch

from nequip.data import AtomicDataDict as A
from nequip.utils import compile_utils as cu, fx_compat
from nequip.utils.test import set_irreps_debug


import test_training_hardening as hardening


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def lifecycle_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA lifecycle regression requires an allocated GPU")
    return request.param


@pytest.fixture(autouse=True)
def compiler_environment(monkeypatch, lifecycle_device):
    original = e3nn.get_optimization_defaults()["jit_script_fx"]
    e3nn.set_optimization_defaults(jit_script_fx=False)
    set_irreps_debug(False)
    torch._dynamo.reset()
    real_compile = torch.compile
    monkeypatch.setattr(
        cu.torch,
        "compile",
        lambda model, **kw: real_compile(
            model,
            backend="inductor" if lifecycle_device == "cuda" else "aot_eager",
            fullgraph=True,
            dynamic=kw.get("dynamic", True),
        ),
    )
    yield
    torch._dynamo.reset()
    set_irreps_debug(True)
    e3nn.set_optimization_defaults(jit_script_fx=original)


def batch(lifecycle_device):
    data = hardening._data_to_device(
        hardening._training_batch((3,), ((0, 1, 2), (1, 2, 0))), lifecycle_device
    )
    return {
        key: value.to(torch.get_default_dtype()) if value.is_floating_point() else value
        for key, value in data.items()
    }


def configured(model):
    return cu.configure_model_for_compile(
        model,
        {"spingnn_compile": {"enabled": True, "scope": "full_fx", "dynamic": True}},
    )


@pytest.mark.parametrize("invalid", ["pbc", "volume", "batch", "edge_dtype"])
def test_warm_cache_rejects_invalid_input(invalid, lifecycle_device):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = configured(
            hardening.ParaStressForceSpinOutput(hardening._ToySpinEnergy())
            .to(lifecycle_device)
            .train()
        )
    data = batch(lifecycle_device)
    model(hardening._clone_data(data))
    if invalid == "pbc":
        data[A.PBC_KEY].fill_(False)
    elif invalid == "volume":
        data[A.CELL_KEY].zero_()
    elif invalid == "batch":
        data[A.BATCH_KEY].fill_(1)
    else:
        data[A.EDGE_INDEX_KEY] = data[A.EDGE_INDEX_KEY].int()
    with pytest.raises((ValueError, TypeError)):
        model(data)


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor(0.7))
        self.head = torch.nn.Parameter(torch.tensor(1.2))
        self.register_buffer("scale", torch.eye(3) * 1.1)

    def forward(self, data):
        feature = self.shared * (data[A.POSITIONS_KEY] @ self.scale).square()
        return feature, feature.sin() * self.head


class ParaStressForceSpinOutput(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.func = core

    def forward(self, data):
        return data


def captured_target(monkeypatch, lifecycle_device):
    model = configured(ParaStressForceSpinOutput(Core()).to(lifecycle_device))

    def capture(target, data, **kwargs):
        graph = torch.fx.symbolic_trace(target.func)
        graph.input_signature = (A.POSITIONS_KEY,)
        graph.output_signature = ("energy", "aux")
        graph.trace_sizes = (1, 3, 3)
        return graph

    monkeypatch.setattr(fx_compat, "expand_training_graph", capture)
    return model


@pytest.mark.parametrize("mutation", ["buffer", "assign", "dtype"])
def test_capture_tracks_model_replacement(monkeypatch, mutation, lifecycle_device):
    model = captured_target(monkeypatch, lifecycle_device)
    data = batch(lifecycle_device)
    model(data)
    if mutation == "buffer":
        model.func.scale = model.func.scale * 2
    elif mutation == "assign":
        state = {
            key: value.detach().clone() * 2 for key, value in model.state_dict().items()
        }
        model.load_state_dict(state, assign=True)
    else:
        dtype = (
            torch.float64
            if torch.get_default_dtype() == torch.float32
            else torch.float32
        )
        model.to(dtype=dtype)
        data = {
            key: value.to(dtype) if value.is_floating_point() else value
            for key, value in data.items()
        }
    expected = model.func(data)
    actual = model(data)
    for value, key in zip(expected, ("energy", "aux")):
        torch.testing.assert_close(actual[key], value)
    parameters = tuple(model.parameters())
    expected_gradients = torch.autograd.grad(
        expected[0].sum(), parameters, allow_unused=True
    )
    actual_gradients = torch.autograd.grad(
        actual["energy"].sum(), parameters, allow_unused=True
    )
    for reference, candidate in zip(expected_gradients, actual_gradients):
        assert (reference is None) == (candidate is None)
        if reference is not None:
            torch.testing.assert_close(candidate, reference)
    assert cu.get_compile_stats(model)["full_fx_expand_count"] == 2


def test_unfreezing_preserves_absent_head_gradients(monkeypatch, lifecycle_device):
    model = captured_target(monkeypatch, lifecycle_device)
    model.func.head.requires_grad_(False)
    data = batch(lifecycle_device)
    model(data)
    model.func.head.requires_grad_(True)
    reference = copy.deepcopy(model.func)
    for target, call in ((reference, reference), (model.func, model)):
        optimizer = torch.optim.AdamW(target.parameters(), lr=0.01, weight_decay=0.1)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = call(data)
            energy = output[0] if isinstance(output, tuple) else output["energy"]
            energy.square().sum().backward()
            assert target.head.grad is None
            optimizer.step()
    for expected, actual in zip(reference.parameters(), model.func.parameters()):
        torch.testing.assert_close(actual, expected)


AUTOCAST = {
    "train_autocast": {"enabled": True, "fp32_modules": ["ParaStressForceSpinOutput"]}
}


def test_autocast_can_be_disabled(lifecycle_device):
    model = ParaStressForceSpinOutput(Core()).to(lifecycle_device)
    cu.configure_model_for_autocast(model, AUTOCAST, lifecycle_device)
    cu.configure_model_for_autocast(
        model, {"train_autocast": {"enabled": False}}, lifecycle_device
    )
    payload = {"value": torch.ones(2, dtype=torch.bfloat16, device=lifecycle_device)}
    assert model(payload)["value"].dtype == torch.bfloat16
    assert not hasattr(model, "_spingnn_energy_autocast")


def test_disabling_compile_preserves_autocast(lifecycle_device):
    model = configured(ParaStressForceSpinOutput(Core()).to(lifecycle_device))
    cu.configure_model_for_autocast(model, AUTOCAST, lifecycle_device)
    cu.configure_model_for_compile(model, {"spingnn_compile": {"enabled": False}})
    cu.configure_model_for_autocast(model, AUTOCAST, lifecycle_device)
    payload = {"value": torch.ones(2, dtype=torch.bfloat16, device=lifecycle_device)}
    assert model(payload)["value"].dtype == torch.float32
    assert cu.get_autocast_stats(model)[0]["calls"] == 1


def test_optimizer_writes_reuse_live_capture(monkeypatch, lifecycle_device):
    model = captured_target(monkeypatch, lifecycle_device)
    data = batch(lifecycle_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        expected = model.func(data)
        torch.testing.assert_close(output["energy"], expected[0])
        output["energy"].square().sum().backward()
        optimizer.step()
    assert cu.get_compile_stats(model)["full_fx_expand_count"] == 1


def test_autocast_whitelist_removes_old_wrappers(lifecycle_device):
    model = ParaStressForceSpinOutput(Core()).to(lifecycle_device)
    cu.configure_model_for_autocast(model, AUTOCAST, lifecycle_device)
    cu.configure_model_for_autocast(
        model,
        {"train_autocast": {"enabled": True, "fp32_modules": ["Core"]}},
        lifecycle_device,
    )
    assert not hasattr(model, "_spingnn_autocast_state")
    assert not hasattr(model, "_spingnn_energy_autocast")
    assert hasattr(model.func, "_spingnn_autocast_state")


def test_compile_after_autocast_keeps_precision_boundary(monkeypatch, lifecycle_device):
    model = ParaStressForceSpinOutput(Core()).to(lifecycle_device)
    cu.configure_model_for_autocast(model, AUTOCAST, lifecycle_device)
    configured(model)
    monkeypatch.setattr(cu, "_call_full_fx", lambda module, data, state: data)
    payload = {"value": torch.ones(2, dtype=torch.bfloat16, device=lifecycle_device)}
    assert model(payload)["value"].dtype == torch.float32
    assert cu.get_autocast_stats(model)[0]["calls"] == 1
