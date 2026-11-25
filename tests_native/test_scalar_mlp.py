"""Exercise export of a real shifted-softplus ScalarMLPFunction."""

import e3nn
import numpy as np
import pytest
import torch

from allegro.nn._fc import ScalarMLPFunction
from spingnn2.backbone import mlp_forward
from spingnn2.backbone_export import export_mlp


@pytest.mark.parametrize("jit_script_fx", [False, True], ids=["fx", "torchscript"])
def test_shifted_softplus_export_and_pullback(jit_script_fx):
    previous = e3nn.get_optimization_defaults()["jit_script_fx"]
    try:
        e3nn.set_optimization_defaults(jit_script_fx=jit_script_fx)
        model = ScalarMLPFunction(3, [4, 5], 2, mlp_nonlinearity="ssp").double().eval()
    finally:
        e3nn.set_optimization_defaults(jit_script_fx=previous)

    x = (
        torch.linspace(-2.0, 2.0, 12, dtype=torch.float64)
        .reshape(4, 3)
        .requires_grad_()
    )
    seed = torch.linspace(-1.0, 1.0, 8, dtype=x.dtype).reshape(4, 2)
    expected = model(x)
    (expected_gradient,) = torch.autograd.grad(expected, x, seed)
    spec = export_mlp(model)
    assert [layer["activation"] for layer in spec["layers"]] == ["ssp", "ssp", None]
    actual, pullback = mlp_forward(spec, x.detach().numpy())
    np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        pullback(seed.numpy()), expected_gradient.numpy(), rtol=0.0, atol=1e-13
    )
