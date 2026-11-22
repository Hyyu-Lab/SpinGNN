import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.train import Metrics
from nequip.train.metrics import Trace


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_missing_trace_survives_epoch_device_transfer(device, distributed, scale):
    dtype = (
        torch.float64 if torch.get_default_dtype() == torch.float32 else torch.float32
    )
    pred = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.zeros(
            2, 1, dtype=torch.long, device=device
        ),
        "f_nu": torch.full((2, 1), 4.0, dtype=dtype, device=device),
    }
    key = "total_sf_Uepi"
    trace = Trace(scale=scale)
    zero = trace(pred, {}, key)
    assert zero.device == pred["f_nu"].device
    assert zero.dtype == dtype
    assert not zero.requires_grad
    assert torch.equal(zero, torch.zeros_like(zero))

    metrics = Metrics(
        components=[
            (
                key,
                "mean",
                {"functional": "Trace", "functional_kwargs": {"scale": scale}},
            )
        ],
        distributed=distributed,
    )
    # Initial validation creates stats; each training epoch resets and moves
    # them before the disabled head produces another zero trace.
    metrics(pred, {})
    for _ in range(2):
        metrics.reset()
        metrics.to(device)
        for _ in range(3):
            result = metrics(pred, {})
            assert all(value.device == zero.device for value in result.values())
            assert all(
                torch.equal(value, torch.zeros_like(value)) for value in result.values()
            )
        assert all(
            value.device == zero.device and torch.equal(value, torch.zeros_like(value))
            for value in metrics.current_result().values()
        )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_present_trace_keeps_values_and_gradient(device):
    tensor = torch.arange(18.0, device=device).reshape(2, 3, 3).requires_grad_()
    result = Trace(scale=100.0)({"covariance": tensor}, {}, "covariance")
    expected = torch.tensor([12.0, 39.0], device=device) * 100.0
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    gradient = torch.autograd.grad(result.sum(), tensor)[0]
    torch.testing.assert_close(
        gradient, torch.eye(3, device=device).expand(2, 3, 3) * 100.0, rtol=0, atol=0
    )


def test_missing_trace_with_empty_predictions_preserves_zero():
    torch.testing.assert_close(Trace()({}, {}, "covariance"), torch.zeros(1))
