"""Checkpoint boundaries preserve higher derivatives without retaining final graphs."""

import copy
import warnings

import pytest
import torch
from packaging.version import Version

from allegro.nn._allegro_og import _ActivationCheckpointFunction
from allegro.nn._fc import ScalarMLPFunction
from nequip.utils._global_options import global_options_context


@pytest.mark.parametrize("higher_order", [False, True])
def test_checkpoint_respects_requested_derivative_order(higher_order):
    x = torch.tensor([1.5, 2.0], requires_grad=True)
    weight = torch.nn.Parameter(torch.tensor([0.7, 1.1]))
    received = []
    x.register_hook(lambda gradient: received.append(gradient.requires_grad))

    def polynomial(data):
        return dict(data, energy=data["x"].pow(3) * weight)

    _, output = _ActivationCheckpointFunction.apply(
        polynomial, {}, ("x",), False, x, weight
    )
    gradient = torch.autograd.grad(output.sum(), x, create_graph=higher_order)[0]
    torch.testing.assert_close(gradient, 3 * x.square() * weight)
    assert received == [higher_order]
    if higher_order:
        parameter_gradient = torch.autograd.grad(gradient.square().sum(), weight)[0]
        torch.testing.assert_close(parameter_gradient, 18 * weight * x.pow(4))


class _StatefulCore(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(4))
        self.layer = (
            torch.nn.AlphaDropout(0.4)
            if kind == "dropout"
            else (
                torch.nn.BatchNorm1d(4)
                if kind in ("batchnorm", "mode")
                else torch.nn.Identity()
            )
        )
        if kind == "scripted_mlp":
            with global_options_context(
                {"e3nn_optimization_defaults": {"jit_script_fx": True}}
            ):
                self.layer = ScalarMLPFunction(
                    4, [4], 4, mlp_batchnorm=True, mlp_dropout_p=0.2
                )

    def forward(self, data):
        return dict(data, energy=self.layer(data["x"] @ self.weight).sin())


@pytest.mark.parametrize(
    "kind", ["dropout", "batchnorm", "autocast", "mode", "scripted_mlp"]
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_checkpoint_preserves_execution_state(kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA checkpoint regression requires an allocated GPU")
    devices = list(range(torch.cuda.device_count())) if device == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.jit.optimized_execution(False):
        if kind == "scripted_mlp":
            # This regression deliberately exercises deprecated serialization APIs.
            # Preserve other filters so recording does not multiply unrelated warnings.
            with warnings.catch_warnings(record=True) as recorded:
                warnings.filterwarnings(
                    "always",
                    message=r"`torch\.jit\.(script|save|load)` is deprecated",
                    category=DeprecationWarning,
                )
                _assert_checkpoint_execution_state(kind, device)
            expected = {"torch.jit.script", "torch.jit.save", "torch.jit.load"}
            observed = set()
            for warning in recorded:
                api = next(
                    (
                        name
                        for name in expected
                        if str(warning.message).startswith(f"`{name}` is deprecated")
                    ),
                    None,
                )
                if issubclass(warning.category, DeprecationWarning) and api:
                    observed.add(api)
                else:
                    warnings.warn_explicit(
                        warning.message,
                        warning.category,
                        warning.filename,
                        warning.lineno,
                        source=warning.source,
                    )
            # Older supported Torch releases do not emit these deprecations.
            if Version(torch.__version__).release >= (2, 11):
                assert expected <= observed
        else:
            _assert_checkpoint_execution_state(kind, device)


def test_checkpoint_warning_contract_reemits_unrelated_warnings(monkeypatch):
    def emit_warnings(*args):
        for api in ("script", "save", "load"):
            warnings.warn(f"`torch.jit.{api}` is deprecated", DeprecationWarning)
        warnings.warn("unexpected legacy-checkpoint warning", UserWarning)

    monkeypatch.setitem(globals(), "_assert_checkpoint_execution_state", emit_warnings)
    with pytest.warns(UserWarning, match="unexpected legacy-checkpoint warning"):
        test_checkpoint_preserves_execution_state("scripted_mlp", "cpu")


def test_checkpoint_warning_contract_supports_older_torch(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.6.0")
    monkeypatch.setitem(
        globals(), "_assert_checkpoint_execution_state", lambda *args: None
    )
    test_checkpoint_preserves_execution_state("scripted_mlp", "cpu")


def _assert_checkpoint_execution_state(kind, device):
    torch.manual_seed(193)
    reference = _StatefulCore(kind).to(device)
    candidate = copy.deepcopy(reference)
    source = torch.linspace(-0.5, 0.7, 32, device=device).reshape(8, 4)
    outputs = []
    for model, checkpoint in ((reference, False), (candidate, True)):
        torch.manual_seed(317)
        x = source.clone().requires_grad_(True)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=kind == "autocast"):
            if checkpoint:
                _, output = _ActivationCheckpointFunction.apply(
                    model.forward, {}, ("x",), False, x, *model.parameters()
                )
            else:
                output = model({"x": x})["energy"]
        if kind == "mode":
            model.eval()
        derivative = torch.autograd.grad(output.sum(), x, create_graph=True)[0]
        derivative.square().sum().backward()
        outputs.append((output, derivative, torch.rand(4, device=device)))
    for actual, expected in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(candidate.parameters(), reference.parameters()):
        rtol, atol = (1e-12, 1e-14) if actual.dtype == torch.float64 else (1e-4, 1e-6)
        torch.testing.assert_close(actual.grad, expected.grad, rtol=rtol, atol=atol)
    for actual, expected in zip(candidate.buffers(), reference.buffers()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
