"""Pass a local supported policy to the real full-FX compiler boundary."""

import copy

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from nequip.data import AtomicDataDict as A
from nequip.utils import compile_utils as cu, fx_compat


OPTION = "triton.coalesce_tiling_analysis"


class _Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7))

    def forward(self, data):
        return data[A.POSITIONS_KEY] * self.weight


class _Output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = _Core()


@pytest.mark.parametrize(
    "dynamic,cuda,user,supported,scope,expected",
    [
        pytest.param(
            True, True, None, True, "full_fx", False, id="dynamic-cuda-default"
        ),
        pytest.param(
            True, True, True, True, "full_fx", False, id="dynamic-cuda-override"
        ),
        pytest.param(True, True, False, True, "full_fx", False, id="explicit-false"),
        pytest.param(False, True, True, True, "full_fx", True, id="static-cuda"),
        pytest.param(True, False, True, True, "full_fx", True, id="cpu"),
        pytest.param(True, True, True, True, "energy_core", True, id="energy-core"),
        pytest.param(True, True, None, False, "full_fx", None, id="unavailable-option"),
    ],
)
def test_inductor_policy_scope_and_compiler_arguments(
    monkeypatch, dynamic, cuda, user, supported, scope, expected
):
    fake_mode = FakeTensorMode()

    def tensor(shape, dtype=None):
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if cuda:
            # Device metadata alone exercises policy dispatch; no CUDA runtime
            # or kernel is needed for this CPU contract test.
            return FakeTensor(
                fake_mode,
                torch.empty(shape, dtype=dtype, device="meta"),
                torch.device("cuda:0"),
            )
        return torch.empty(shape, dtype=dtype)

    data = {
        A.POSITIONS_KEY: tensor((4, 3)),
        A.EDGE_INDEX_KEY: tensor((2, 8), torch.long),
        A.CELL_KEY: tensor((2, 3, 3)),
    }
    model = _Output()
    graph = torch.fx.symbolic_trace(model.func)
    graph.input_signature = (A.POSITIONS_KEY,)
    graph.output_signature = (A.TOTAL_ENERGY_KEY,)
    graph.trace_sizes = (7, 17, 19)
    state = cu._CompileState(
        scope=scope, dynamic=dynamic, options={} if user is None else {OPTION: user}
    )
    object.__setattr__(model, "_spingnn_compile_state", state)
    parameter_ids = tuple(map(id, model.parameters()))
    calls = []
    valid_options = set(torch._inductor.list_options())
    if supported:
        valid_options.add(OPTION)
    else:
        valid_options.discard(OPTION)

    def compiler(module, **kwargs):
        calls.append(copy.deepcopy(kwargs))
        return module

    missing = object()
    expected_global = getattr(
        torch._inductor.config.triton, "coalesce_tiling_analysis", missing
    )
    if expected_global is not missing:
        monkeypatch.setattr(
            torch._inductor.config.triton, "coalesce_tiling_analysis", True
        )
        expected_global = True
    monkeypatch.setattr(torch._inductor, "list_options", lambda: list(valid_options))
    monkeypatch.setattr(cu.torch, "compile", compiler)
    monkeypatch.setattr(
        fx_compat, "expand_training_graph", lambda *args, **kwargs: graph
    )
    if scope == "full_fx":
        first, _ = cu._compiled_full_fx_callable(model, state, data, n_frames=2)
        second, _ = cu._compiled_full_fx_callable(model, state, data, n_frames=2)
        assert state.full_fx_expand_count == 1
    else:
        first = cu._compiled_callable(model, state)
        second = cu._compiled_callable(model, state)
    assert first is second
    assert len(calls) == 1
    assert calls[0]["backend"] == "inductor"
    assert calls[0]["dynamic"] is dynamic
    if scope == "full_fx":
        assert calls[0]["fullgraph"] is True
    if expected is None:
        assert OPTION not in calls[0]["options"]
    else:
        assert calls[0]["options"][OPTION] is expected
    assert (
        getattr(torch._inductor.config.triton, "coalesce_tiling_analysis", missing)
        is expected_global
    )
    assert tuple(map(id, model.parameters())) == parameter_ids
    assert cu.get_compile_stats(model)["options"] == [state.options]
