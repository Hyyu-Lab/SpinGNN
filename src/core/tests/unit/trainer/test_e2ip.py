"""Student-t normalization, tangent invariance, and additive reductions."""

import math

import pytest
import torch
from e3nn import o3

from nequip.train.e2ip import EvidentialLoss, e2IPLoss, tangent_basis
from nequip.train import Metrics


@pytest.mark.parametrize("spin", [False, True])
def test_nll_matches_student_t_and_rotates(spin):
    torch.manual_seed(13)
    key, prefix, dimension = (
        ("spin_forces_vert", "sf", 2) if spin else ("forces", "f", 3)
    )
    vectors = torch.randn(5, 3)
    vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    raw = torch.randn(5, 3, 3)
    sigma = raw @ raw.transpose(-1, -2) + torch.eye(3)
    delta = torch.randn(5, 3)
    nu, kappa = torch.full((5, 1), 6.0), torch.full((5, 1), 0.7)
    pred = {
        key: delta,
        prefix + "_sigma": sigma,
        prefix + "_nu": nu,
        prefix + "_kappa": kappa,
    }
    ref = {key: torch.zeros_like(delta), "spin": vectors}
    loss = EvidentialLoss(reg=0.1, spin_force_scale=1.0)
    nll, reg = loss.components(pred, ref, key)
    basis = tangent_basis(vectors) if spin else torch.eye(3).expand(5, 3, 3)
    cov = basis.transpose(-1, -2) @ sigma @ basis + 1e-6 * torch.eye(dimension)
    residual = (basis.transpose(-1, -2) @ delta.unsqueeze(-1)).squeeze(-1)
    df = nu[:, 0] - dimension + 1
    scale = cov * (nu * (1 + kappa) / (kappa * df.unsqueeze(-1))).unsqueeze(-1)
    maha = (
        residual.unsqueeze(-2) @ torch.linalg.solve(scale, residual.unsqueeze(-1))
    ).reshape(-1)
    expected = (
        torch.lgamma(df / 2)
        - torch.lgamma((df + dimension) / 2)
        + dimension / 2 * (df * math.pi).log()
        + 0.5 * torch.linalg.slogdet(scale).logabsdet
        + (df + dimension) / 2 * torch.log1p(maha / df)
    )
    torch.testing.assert_close(nll, expected)
    torch.testing.assert_close(reg, (nu[:, 0] + kappa[:, 0]) * residual.norm(dim=-1))
    rotation = o3.rand_matrix()
    rotated = dict(
        pred,
        **{key: delta @ rotation.T, prefix + "_sigma": rotation @ sigma @ rotation.T}
    )
    rotated_ref = dict(ref, spin=vectors @ rotation.T)
    torch.testing.assert_close(loss(rotated, rotated_ref, key), loss(pred, ref, key))


def test_zero_spins_mask_before_cholesky_and_additive_loss():
    pred = {
        "spin_forces_vert": torch.ones(3, 3, requires_grad=True),
        "sf_sigma": torch.stack([torch.eye(3), torch.zeros(3, 3), torch.eye(3)]),
        "sf_nu": torch.full((3, 1), 5.0),
        "sf_kappa": torch.ones(3, 1),
    }
    ref = {
        "spin_forces_vert": torch.zeros(3, 3),
        "spin": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    }
    loss = e2IPLoss(
        {"spin_forces_vert": 0.2}, evidential_keys=["spin_forces_vert"], reg=0.01
    )
    value, _ = loss(pred, ref)
    sums, counts = loss.sum_and_count(pred, ref)
    assert counts["spin_forces_vert"] == 2
    torch.testing.assert_close(
        value, 0.2 * sums["spin_forces_vert"] / counts["spin_forces_vert"]
    )
    value.backward()
    assert torch.count_nonzero(pred["spin_forces_vert"].grad[1]) == 0
    ref["spin"].zero_()
    zero, _ = loss(pred, ref)
    assert zero == 0 and torch.isfinite(zero)
    zero.backward()


@pytest.mark.parametrize("spin", [False, True])
def test_numerical_rescaling_preserves_small_covariance_loss_and_gradients(spin):
    key, prefix = ("spin_forces_vert", "sf") if spin else ("forces", "f")
    # Include jitter comparable to covariance, as in the trained SOC model.
    delta = torch.tensor([[1e-4, -2e-4, 3e-4]], requires_grad=True)
    sigma = (torch.eye(3).unsqueeze(0) * 2e-9).requires_grad_()
    nu = torch.tensor([[7.0]], requires_grad=True)
    kappa = torch.tensor([[3.0]], requires_grad=True)
    pred = {
        key: delta,
        prefix + "_sigma": sigma,
        prefix + "_nu": nu,
        prefix + "_kappa": kappa,
    }
    ref = {key: torch.zeros_like(delta), "spin": torch.tensor([[1.0, 2.0, 3.0]])}
    values, gradients = [], []
    for scale in (1.0, 1000.0):
        loss = EvidentialLoss(
            reg=0.001, likelihood_scale=scale, likelihood_dtype="float64"
        )
        nll, regularizer = loss.components(pred, ref, key)
        assert nll.dtype == torch.float64
        value = loss(pred, ref, key)
        values.append((nll, regularizer, value))
        gradients.append(torch.autograd.grad(value, (delta, sigma, nu, kappa)))
    for unscaled, scaled in zip(values[0], values[1]):
        torch.testing.assert_close(unscaled, scaled, rtol=1e-11, atol=1e-11)
    for unscaled, scaled in zip(gradients[0], gradients[1]):
        assert torch.isfinite(scaled).all()
        assert scaled.dtype == delta.dtype
        torch.testing.assert_close(unscaled, scaled)


def test_scaled_trace_keeps_physical_metric_and_names_separate():
    key = "total_sf_Uepi"
    physical = torch.eye(3).unsqueeze(0) * 1e-8
    pred = {key: physical.clone()}
    metrics = Metrics(
        components=[
            (key, "mean", {"functional": "Trace"}),
            (
                key,
                "mean",
                {
                    "functional": "Trace",
                    "functional_kwargs": {"scale": 1e6, "units": "x1e6"},
                },
            ),
        ]
    )
    result, _ = metrics.flatten_metrics(metrics(pred, {}))
    assert result[key + "_mean"] == pytest.approx(3e-8)
    assert result[key + "_x1e6_mean"] == pytest.approx(0.03)
    torch.testing.assert_close(pred[key], physical)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_likelihood_scale(scale):
    with pytest.raises(ValueError, match="likelihood_scale"):
        EvidentialLoss(likelihood_scale=scale)


@pytest.mark.parametrize("parameter", ["reg", "epsilon", "spin_force_scale"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_invalid_evidential_parameters(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        EvidentialLoss(**{parameter: value})


@pytest.mark.parametrize("parameter", ["epsilon", "spin_force_scale"])
def test_evidential_positive_parameters_reject_zero(parameter):
    with pytest.raises(ValueError, match=parameter):
        EvidentialLoss(**{parameter: 0.0})


def test_evidential_zero_regularization_is_supported():
    assert EvidentialLoss(reg=0.0).reg == 0.0
