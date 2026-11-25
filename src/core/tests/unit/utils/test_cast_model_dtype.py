import pytest
import torch

from nequip.utils._global_options import (
    KEEP_DTYPE_ATTR,
    cast_model_dtype,
    model_dtype_from_config,
)


class _KeepByClass(torch.nn.Module):
    _spingnn_keep_dtype = True

    def __init__(self):
        super().__init__()
        self.inner = torch.nn.Linear(2, 2, dtype=torch.float64)
        self.register_buffer("offset", torch.zeros(2, dtype=torch.float64))


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, dtype=torch.float64)
        self.register_buffer("index", torch.arange(3))
        self.register_buffer("scale", torch.ones(2, dtype=torch.float64))
        self.by_class = _KeepByClass()
        self.by_instance = torch.nn.Sequential(
            torch.nn.Linear(2, 2, dtype=torch.float64)
        )
        setattr(self.by_instance, KEEP_DTYPE_ATTR, True)
        self.not_marked = torch.nn.Linear(2, 2, dtype=torch.float64)
        # a falsy marker does not opt out
        self.not_marked._spingnn_keep_dtype = False


def test_marked_modules_keep_their_dtypes():
    model = cast_model_dtype(_Net(), torch.float32, device=torch.device("cpu"))
    dtypes = {name: t.dtype for name, t in model.state_dict().items()}
    assert dtypes == {
        "linear.weight": torch.float32,
        "linear.bias": torch.float32,
        "index": torch.int64,
        "scale": torch.float32,
        "by_class.offset": torch.float64,
        "by_class.inner.weight": torch.float64,
        "by_class.inner.bias": torch.float64,
        "by_instance.0.weight": torch.float64,
        "by_instance.0.bias": torch.float64,
        "not_marked.weight": torch.float32,
        "not_marked.bias": torch.float32,
    }
    assert isinstance(model.linear.weight, torch.nn.Parameter)


def test_without_marker_equals_module_to():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    reference = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    reference.load_state_dict(model.state_dict())
    cast_model_dtype(model, torch.float64)
    reference.to(dtype=torch.float64)
    for (name, a), (_, b) in zip(
        model.state_dict().items(), reference.state_dict().items()
    ):
        assert a.dtype == b.dtype == torch.float64, name
        assert torch.equal(a, b), name


def test_model_dtype_from_config():
    assert model_dtype_from_config({"default_dtype": "float64"}) is torch.float64
    assert (
        model_dtype_from_config({"default_dtype": "float64", "model_dtype": "float32"})
        is torch.float32
    )
    assert (
        model_dtype_from_config({"default_dtype": "float32", "model_dtype": None})
        is torch.float32
    )
    with pytest.raises(ValueError, match="model_dtype must be 'float32' or 'float64'"):
        model_dtype_from_config({"default_dtype": "float64", "model_dtype": "half"})
