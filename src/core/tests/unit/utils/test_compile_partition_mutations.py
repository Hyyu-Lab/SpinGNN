"""Gradient regions must retain alias writes and copy gradient participation."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from nequip.utils._compile_grad_partitions import (
    _differentiable_copy,
    compile_gradient_partitions,
)


class _MutationHeads(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.primary = torch.nn.Parameter(torch.tensor(0.7))
        self.optional = torch.nn.Parameter(torch.tensor(1.2))
        self.kind = kind

    def forward(self, data):
        x = data["x"]
        values = torch.stack((x, 2 * x), dim=-1)
        view = values[:, :1]
        if self.kind == "copy":
            view.copy_(values[:, 1:])
        elif self.kind == "add":
            view.add_(3 * x[:, None])
        else:
            values[:, 0] = values[:, 1]
        return {"main": values.sum(dim=-1) * self.primary, "head": x * self.optional}


@pytest.mark.parametrize("kind", ["copy", "add", "index"])
def test_partition_preserves_alias_write_order(kind):
    model = _MutationHeads(kind)
    data = {"x": torch.tensor([0.2, 0.4, 0.8])}
    graph = make_fx(model, tracing_mode="symbolic", _allow_non_fake_inputs=True)(data)
    partitioned = compile_gradient_partitions(graph, lambda region: region)
    reference = graph({"x": data["x"].clone()})
    actual = partitioned({"x": data["x"].clone()})
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    parameters = tuple(model.parameters())
    expected_grad = torch.autograd.grad(
        reference["main"].sum(), parameters, allow_unused=True
    )
    actual_grad = torch.autograd.grad(
        actual["main"].sum(), parameters, allow_unused=True
    )
    assert expected_grad[1] is actual_grad[1] is None
    torch.testing.assert_close(actual_grad[0], expected_grad[0], rtol=0, atol=0)


class _OverwrittenParameter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.original = torch.nn.Parameter(torch.tensor(0.7))
        self.replacement = torch.nn.Parameter(torch.tensor(1.2))
        self.optional = torch.nn.Parameter(torch.tensor(0.4))

    def forward(self, x):
        value = self.original * x
        value.copy_(self.replacement * x)
        return value.square(), self.optional * x


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_overwritten_parameter_keeps_real_zero_gradient(backend):
    model = _OverwrittenParameter()
    data = torch.tensor([0.2, 0.4, 0.8])
    graph = make_fx(model, tracing_mode="symbolic", _allow_non_fake_inputs=True)(data)
    compiler = (
        (lambda region: region)
        if backend == "eager"
        else (lambda region: torch.compile(region, backend="aot_eager", fullgraph=True))
    )
    actual = compile_gradient_partitions(graph, compiler)(data)
    expected = model(data)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    parameters = tuple(model.parameters())
    # Full-FX expands force derivatives before AOT. Test further derivatives
    # on that ordinary tensor graph; AOT itself has no double-backward API.
    create_graph = backend == "eager"
    reference_grad = torch.autograd.grad(
        expected[0].sum(), parameters, allow_unused=True, create_graph=create_graph
    )
    actual_grad = torch.autograd.grad(
        actual[0].sum(), parameters, allow_unused=True, create_graph=create_graph
    )
    assert actual_grad[0] is not None and torch.count_nonzero(actual_grad[0]) == 0
    assert actual_grad[2] is reference_grad[2] is None
    torch.testing.assert_close(actual_grad[:2], reference_grad[:2], rtol=0, atol=0)
    if create_graph:
        reference_second = torch.autograd.grad(reference_grad[1], model.replacement)[0]
        actual_second = torch.autograd.grad(actual_grad[1], model.replacement)[0]
        torch.testing.assert_close(actual_second, reference_second, rtol=0, atol=0)


@pytest.mark.parametrize("source_shape", [(), (3,), (1, 3)])
def test_copy_preserves_cast_broadcast_and_nonfinite_overwrite(source_shape):
    source_dtype = torch.get_default_dtype()
    destination_dtype = (
        torch.float64 if source_dtype == torch.float32 else torch.float32
    )
    destination = torch.tensor(
        [[float("nan"), float("inf"), -float("inf")], [1, 2, 3]],
        dtype=destination_dtype,
        requires_grad=True,
    )
    source = torch.full(source_shape, 0.7, dtype=source_dtype, requires_grad=True)
    expected = destination.clone().copy_(source)
    actual = _differentiable_copy(destination, source)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    reference_grad = torch.autograd.grad(
        expected.square().sum(), (destination, source), create_graph=True
    )
    actual_grad = torch.autograd.grad(
        actual.square().sum(), (destination, source), create_graph=True
    )
    assert actual_grad[0] is not None and torch.count_nonzero(actual_grad[0]) == 0
    torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)
    reference_second = torch.autograd.grad(reference_grad[1].sum(), source)[0]
    actual_second = torch.autograd.grad(actual_grad[1].sum(), source)[0]
    torch.testing.assert_close(actual_second, reference_second, rtol=0, atol=0)
