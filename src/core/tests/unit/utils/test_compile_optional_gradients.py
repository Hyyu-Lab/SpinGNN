"""Compiled optional outputs must preserve optimizer-visible unused gradients."""

import copy
from types import SimpleNamespace

import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.utils import compile_utils, fx_compat
from nequip.utils._compile_grad_partitions import compile_gradient_partitions
from nequip.utils.test import set_irreps_debug


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    torch._dynamo.reset()
    set_irreps_debug(False)
    try:
        yield
    finally:
        torch._dynamo.reset()
        set_irreps_debug(True)


class _OptionalHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor(0.7))
        self.head = torch.nn.Parameter(torch.tensor(1.2))

    def forward(self, data):
        features = self.shared * data["x"].square()
        return features, features.sin() * self.head


@pytest.mark.parametrize("supervise_head", [False, True])
@pytest.mark.parametrize("direct_partition", [False, True])
def test_full_fx_optional_output_preserves_adamw_update(
    monkeypatch, supervise_head, direct_partition
):
    reference = _OptionalHead()
    candidate = copy.deepcopy(reference)
    graph = torch.fx.symbolic_trace(candidate)
    graph.input_signature = ("x",)
    graph.output_signature = ("energy", "spin_length")
    graph.trace_sizes = (5, 7, 11)
    monkeypatch.setattr(
        fx_compat, "expand_training_graph", lambda *args, **kwargs: graph
    )
    compile_torch = torch.compile
    monkeypatch.setattr(
        compile_utils.torch,
        "compile",
        lambda module, **kwargs: compile_torch(
            module, backend="aot_eager", fullgraph=True
        ),
    )
    data = {
        "x": torch.tensor([0.3, 0.8]),
        AtomicDataDict.NUM_BATCH_KEY: torch.tensor(1),
        AtomicDataDict.POSITIONS_KEY: torch.zeros(2, 3),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]]),
    }
    state = compile_utils._CompileState(scope="full_fx", dynamic=True, options={})
    if direct_partition:
        compiled = compile_gradient_partitions(
            graph,
            lambda module: compile_torch(module, backend="aot_eager", fullgraph=True),
        )
    else:
        compiled, _ = compile_utils._compiled_full_fx_callable(
            SimpleNamespace(func=candidate, training=True), state, data
        )
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        for model in (reference, candidate)
    ]
    for model_call, optimizer in zip((reference, compiled), optimizers):
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs = model_call({"x": data["x"]})
            loss = outputs[0].square().sum()
            if supervise_head:
                loss = loss + outputs[1].square().sum()
            loss.backward()
            optimizer.step()
    for ref, actual in zip(reference.parameters(), candidate.parameters()):
        assert (ref.grad is None) == (actual.grad is None)
        torch.testing.assert_close(actual, ref)
    assert len(optimizers[0].state) == len(optimizers[1].state)


def test_partitioned_shared_parameters_survive_shape_and_loss_changes():
    reference = _OptionalHead()
    candidate = copy.deepcopy(reference)
    graph = torch.fx.symbolic_trace(candidate)
    compiled = compile_gradient_partitions(
        graph,
        lambda module: torch.compile(
            module, backend="aot_eager", fullgraph=True, dynamic=True
        ),
    )
    assert {id(parameter) for parameter in compiled.parameters()} == {
        id(parameter) for parameter in candidate.parameters()
    }
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        for model in (reference, candidate)
    ]
    for count, supervise_head in ((2, False), (3, True), (2, False)):
        data = {"x": torch.linspace(0.2, 0.8, count)}
        for model_call, optimizer in zip((reference, compiled), optimizers):
            optimizer.zero_grad(set_to_none=True)
            outputs = model_call(data)
            loss = outputs[0].square().sum()
            if supervise_head:
                loss = loss + outputs[1].square().sum()
            loss.backward()
            optimizer.step()
        for ref, actual in zip(reference.parameters(), candidate.parameters()):
            assert (ref.grad is None) == (actual.grad is None)
            torch.testing.assert_close(actual, ref)
        for ref_state, actual_state in zip(
            optimizers[0].state.values(), optimizers[1].state.values()
        ):
            for name in ref_state:
                torch.testing.assert_close(actual_state[name], ref_state[name])


def test_partitioned_make_fx_preserves_dictionary_boundary():
    from torch.fx.experimental.proxy_tensor import make_fx

    model = _OptionalHead()
    data = {"x": torch.tensor([0.3, 0.8]), "unused": torch.zeros(1)}
    graph = make_fx(model)(data)
    compiled = compile_gradient_partitions(
        graph,
        lambda module: torch.compile(module, backend="aot_eager", fullgraph=True),
    )
    actual = compiled(data)
    expected = model(data)
    for reference, candidate in zip(expected, actual):
        torch.testing.assert_close(candidate, reference)
    actual[0].square().sum().backward()
    assert model.shared.grad is not None
    assert model.head.grad is None


@pytest.mark.parametrize("loss_case", ["force_only", "optional_head_only"])
def test_unused_primary_parameters_do_not_decay(loss_case):
    from torch.fx.experimental.proxy_tensor import make_fx

    class IndependentOutputs(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Parameter(torch.tensor(0.7))
            self.energy_bias = torch.nn.Parameter(torch.tensor(1.2))
            self.energy_weight = torch.nn.Parameter(torch.tensor(0.9))
            self.optional_weight = torch.nn.Parameter(torch.tensor(1.1))

        def forward(self, data):
            features = self.shared * data["x"].square()
            energy = self.energy_weight * features + self.energy_bias
            force = -2 * self.energy_weight * self.shared * data["x"]
            return energy, force, self.optional_weight * features.sin()

    reference = IndependentOutputs()
    candidate = copy.deepcopy(reference)
    data = {"x": torch.tensor([0.3, 0.8])}
    compiled = compile_gradient_partitions(
        make_fx(candidate)(data),
        lambda module: torch.compile(module, backend="aot_eager", fullgraph=True),
    )
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        for model in (reference, candidate)
    ]
    selected = 1 if loss_case == "force_only" else 2
    for model_call, optimizer in zip((reference, compiled), optimizers):
        optimizer.zero_grad(set_to_none=True)
        model_call(data)[selected].square().sum().backward()
        optimizer.step()
    for expected, actual in zip(reference.parameters(), candidate.parameters()):
        assert (expected.grad is None) == (actual.grad is None)
        torch.testing.assert_close(actual, expected)
    assert candidate.energy_bias.grad is None
    if loss_case == "optional_head_only":
        assert candidate.energy_weight.grad is None
    assert len(optimizers[0].state) == len(optimizers[1].state)


@pytest.mark.parametrize("explicit_zero", [False, True])
def test_autograd_node_hook_observes_unmaterialized_sink_gradients(explicit_zero):
    class OutputSink(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *values):
            ctx.set_materialize_grads(False)
            return values

        @staticmethod
        def backward(ctx, *gradients):
            return gradients

    reference = _OptionalHead()
    candidate = copy.deepcopy(reference)
    compiled = compile_gradient_partitions(
        torch.fx.symbolic_trace(candidate),
        lambda module: torch.compile(module, backend="aot_eager", fullgraph=True),
    )
    observed = []

    def preserve_absence(gradient_inputs, gradient_outputs):
        absent = all(gradient is None for gradient in gradient_outputs)
        observed.append(absent)
        if absent:
            return tuple(None for _ in gradient_inputs)
        return gradient_inputs

    for call, model in ((reference, reference), (compiled, candidate)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        optimizer.zero_grad(set_to_none=True)
        values = call({"x": torch.tensor([0.3, 0.8])})
        if model is candidate:
            for node in {value.grad_fn for value in values}:
                node.register_hook(preserve_absence)
        values = OutputSink.apply(*values)
        loss = values[0].square().sum()
        if explicit_zero:
            loss = loss + 0 * values[1].sum()
        loss.backward()
        optimizer.step()
    assert any(observed) is (not explicit_zero)
    for expected, actual in zip(reference.parameters(), candidate.parameters()):
        assert (expected.grad is None) == (actual.grad is None)
        torch.testing.assert_close(actual, expected)
