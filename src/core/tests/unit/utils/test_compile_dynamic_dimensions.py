"""Allow valid compiler ranges while retaining genuinely dynamic dimensions."""

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import ConstraintViolationError

from nequip.data import AtomicDataDict
from nequip.utils.compile_utils import _mark_full_fx_dimensions


@pytest.mark.parametrize(
    "key,axis,sizes,other_shape",
    [
        (AtomicDataDict.POSITIONS_KEY, 0, (64, 96, 80, 64), (3,)),
        (AtomicDataDict.SPIN_LENGTH_KEY, 0, (64, 96, 80, 64), (1,)),
        (AtomicDataDict.CELL_KEY, 0, (512, 600, 550, 512), (3, 3)),
        (AtomicDataDict.EDGE_INDEX_KEY, 1, (2048, 2304, 2176, 2048), (2,)),
    ],
)
def test_backend_refines_dynamic_range_without_recompiling(
    key, axis, sizes, other_shape
):
    captures = []
    weight = torch.nn.Parameter(torch.tensor(0.7))

    def function(data):
        values = data[key]
        # An Inductor flattened-index guard constrains the valid size range.
        # Reproduce that guard with small tensors, without a giant allocation.
        torch._check(32 * values.shape[axis] <= 2**31 - 1)
        return (values * weight).square().sum()

    def backend(graph, inputs):
        captures.append(graph)
        return graph.forward

    compiled = torch.compile(function, backend=backend, fullgraph=True, dynamic=True)
    dtype = (
        torch.long
        if key == AtomicDataDict.EDGE_INDEX_KEY
        else torch.get_default_dtype()
    )
    for size in sizes:
        shape = (size, *other_shape) if axis == 0 else (*other_shape, size)
        data = {key: torch.ones(shape, dtype=dtype)}
        _mark_full_fx_dimensions(data)
        expected = function(data)
        actual = compiled(data)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_grad = torch.autograd.grad(expected, weight)[0]
        actual_grad = torch.autograd.grad(actual, weight)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert len(captures) == 1


def test_explicit_dynamic_axis_cannot_be_silently_specialized():
    def function(data):
        values = data[AtomicDataDict.POSITIONS_KEY]
        if values.shape[0] == 64:
            return values.square().sum()
        return values.sum()

    data = {AtomicDataDict.POSITIONS_KEY: torch.ones(64, 3)}
    _mark_full_fx_dimensions(data)
    compiled = torch.compile(function, backend="eager", fullgraph=True, dynamic=True)
    with pytest.raises(ConstraintViolationError):
        compiled(data)
