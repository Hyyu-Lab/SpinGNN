import pytest
import torch

from nequip.nn import VolumeNormalization

pytestmark = [
    pytest.mark.filterwarnings("ignore:`torch.jit.*:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:The TorchScript type system doesn't support.*:UserWarning"
    ),
]


@pytest.mark.parametrize("script", [False, True])
@pytest.mark.parametrize("unbatched", [False, True])
def test_volume_normalization(script, unbatched):
    module = VolumeNormalization("total", "response", irreps_in={"total": "9x0e"})
    if script:
        module = torch.jit.script(module)
    cells = torch.diag(torch.tensor([-2.0, 3.0, 4.0], dtype=torch.float64))
    if not unbatched:
        cells = torch.stack([cells, cells * 2])
    n = 1 if unbatched else 2
    total = torch.arange(n * 9, dtype=torch.float32).reshape(n, 9).requires_grad_()
    actual = module(_data(cells, total))["response"]
    expected_volume = torch.tensor(
        [24.0] if unbatched else [24.0, 192.0], dtype=torch.float32
    )
    torch.testing.assert_close(actual, total / expected_volume[:, None])
    assert actual.dtype == torch.float32
    actual.sum().backward()
    torch.testing.assert_close(
        total.grad, (1 / expected_volume[:, None]).expand_as(total)
    )


@pytest.mark.parametrize("value", [0.0, 1e-15, float("nan"), float("inf")])
def test_volume_rejects_invalid(value):
    module = VolumeNormalization("total", "response", irreps_in={"total": "9x0e"})
    with pytest.raises(RuntimeError, match="volume"):
        module(_data(torch.diag(torch.tensor([value, 1.0, 1.0])), torch.ones(1, 9)))


def test_volume_rejects_frame_mismatch():
    module = VolumeNormalization("total", "response", irreps_in={"total": "9x0e"})
    with pytest.raises(RuntimeError, match="frame-count"):
        module(_data(torch.eye(3), torch.ones(2, 9)))


def _data(cell, total):
    return {
        "cell": cell,
        "total": total,
        "pos": torch.zeros(1, 3),
        "edge_index": torch.empty(2, 0, dtype=torch.long),
    }
