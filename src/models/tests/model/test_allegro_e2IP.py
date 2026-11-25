"""Regression coverage for force evidence and covariance units."""

import copy

import pytest
import torch
from e3nn import o3

from nequip.model import model_from_config
from nequip.nn import RescaleOutput
from nequip.train.e2ip import EvidentialLoss, tangent_basis
from allegro.nn._nodewise import (
    DampedSymmetricOutput,
    e2IP_nodewise,
    linear_tanh_damper,
)


E2IP_WARNINGS = [
    pytest.mark.filterwarnings("ignore:`torch.jit.*:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:The TorchScript type system doesn't support.*:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:!! Stresses in NequIP are in BETA.*:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:edge_prune='dense' computes every supplied edge.*:UserWarning"
    ),
]
pytestmark = E2IP_WARNINGS


def e2ip_config(soc=False, key="forces", **kwargs):
    config = dict(
        model_builders=[
            "allegro.model.Allegro_soc_e2IP" if soc else "allegro.model.Allegro_e2IP",
            "ParaStressForceSpinOutput" if soc else "ParaStressForceOutput",
        ],
        num_types=2,
        type_names=["Fe", "O"],
        r_max=4.0,
        avg_num_neighbors=2.0,
        l_max=2,
        num_layers=2,
        parity="o3_full",
        symmetry="u2",
        env_embed_multiplicity=2,
        two_body_latent_mlp_latent_dimensions=[8],
        latent_mlp_latent_dimensions=[8],
        env_embed_mlp_latent_dimensions=[],
        edge_eng_mlp_latent_dimensions=[8],
        edge_onlyr_eng_mlp_latent_dimensions=[8],
        edge_biquad_coeff_mlp_latent_dimensions=[8],
        edge_eng_soc_on_mlp_latent_dimensions=[8],
        e2ip_latent_channels=2,
        e2ip_mlp_latent_dimensions=[8],
        evidential_key=key,
        spingnn_compile={"enabled": False},
    )
    config.update(kwargs)
    return config


def e2ip_sample():
    return {
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.3, 0.1], [0.1, 1.1, 0.4]]),
        "edge_index": torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
        "edge_cell_shift": torch.zeros(6, 3),
        "atom_types": torch.tensor([0, 0, 1]),
        "cell": torch.eye(3).unsqueeze(0) * 8.0,
        "pbc": torch.ones(1, 3, dtype=torch.bool),
        "batch": torch.zeros(3, dtype=torch.long),
        "spin": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [0.0, 0.0, 0.0]]),
    }


def check_evidence_model(soc, key):
    torch.manual_seed(42)
    model = model_from_config(e2ip_config(soc, key))
    sample = e2ip_sample()
    out = model(copy.deepcopy(sample))
    prefix = "sf" if key == "spin_forces_vert" else "f"
    sigma = out[prefix + "_sigma"]
    assert sigma.shape == (3, 3, 3)
    assert model.irreps_out[prefix + "_sigma"] is None
    valid = (
        sample["spin"].norm(dim=-1) > 1e-6
        if soc and prefix == "sf"
        else torch.ones(3, dtype=torch.bool)
    )
    basis = (
        tangent_basis(sample["spin"][valid])
        if prefix == "sf"
        else torch.eye(3).expand(3, 3, 3)
    )
    reduced = basis.transpose(-1, -2) @ sigma[valid] @ basis
    assert (torch.linalg.eigvalsh(reduced) > 0).all()
    if prefix == "sf":
        torch.testing.assert_close(
            sigma @ sample["spin"].unsqueeze(-1), torch.zeros(3, 3, 1)
        )
        assert torch.count_nonzero(sigma[~valid]) == 0
    score = out[prefix + "_Uepi"].diagonal(dim1=-2, dim2=-1).sum() / valid.sum()
    torch.testing.assert_close(out[prefix + "_uncertainty"].reshape(()), score)
    ref = {key: torch.zeros_like(out[key]), "spin": sample["spin"]}
    loss = (
        EvidentialLoss(reg=0.001)(out, ref, key) + out["total_energy"].square().mean()
    )
    loss.backward()
    gradients = {
        name: p.grad for name, p in model.named_parameters() if p.grad is not None
    }
    assert all(torch.isfinite(grad).all() for grad in gradients.values())
    assert any(
        "sigma_heads" in name and grad.abs().sum() > 0
        for name, grad in gradients.items()
    )
    assert any(
        "e2ip_vector_head" in name and grad.abs().sum() > 0
        for name, grad in gradients.items()
    )

    rotation = o3.rand_matrix()
    rotated = copy.deepcopy(sample)
    for field in ("pos", "spin"):
        rotated[field] = sample[field] @ rotation.T
    rotated["cell"] = sample["cell"] @ rotation.T
    rot_out = model(rotated)
    torch.testing.assert_close(
        rot_out[prefix + "_sigma"], rotation @ sigma @ rotation.T, rtol=2e-4, atol=2e-6
    )
    torch.testing.assert_close(
        rot_out[key], out[key] @ rotation.T, rtol=2e-4, atol=2e-6
    )
    torch.testing.assert_close(
        rot_out[prefix + "_uncertainty"],
        out[prefix + "_uncertainty"],
        rtol=2e-4,
        atol=2e-6,
    )
    for field in ("pos", "spin"):
        transformed = copy.deepcopy(sample)
        transformed[field] = -sample[field]
        transformed_out = model(transformed)
        # Spatial inversion leaves axial spins fixed; time reversal flips them.
        sign = -1 if (field == "pos") == (key == "forces") else 1
        torch.testing.assert_close(
            transformed_out[key], sign * out[key], rtol=2e-4, atol=2e-6
        )
        torch.testing.assert_close(
            transformed_out[prefix + "_sigma"], sigma, rtol=2e-4, atol=2e-6
        )


def test_allegro_e2ip_evidence_and_gradients():
    check_evidence_model(False, "forces")


@pytest.mark.parametrize(
    "soc,key", [(False, "forces"), (True, "spin_forces_vert"), (True, "forces")]
)
def test_covariance_rescaling(soc, key):
    torch.manual_seed(11)
    core = model_from_config(e2ip_config(soc, key))
    # Legacy derivative metadata describes Cartesian stress as a flat vector.
    # This test exercises covariance scaling; stress has separate parity tests.
    for field in ("stress", "virial", "atom_virial"):
        if field in core.irreps_out:
            core.irreps_out[field] = None
    model = RescaleOutput(core, scale_keys=[key], scale_by=2.5)
    sample = e2ip_sample()
    prefix = "sf" if key == "spin_forces_vert" else "f"
    training = model(copy.deepcopy(sample))
    model.eval()
    evaluation = model(copy.deepcopy(sample))
    for suffix in ("sigma", "Uepi", "uncertainty"):
        torch.testing.assert_close(
            evaluation[prefix + "_" + suffix],
            training[prefix + "_" + suffix] * 2.5**2,
        )
    torch.testing.assert_close(evaluation[key], training[key] * 2.5)
    restored = model.unscale(evaluation, force_process=True)
    for field in (
        key,
        prefix + "_sigma",
        prefix + "_Uepi",
        "total_" + prefix + "_Uepi",
        prefix + "_uncertainty",
    ):
        torch.testing.assert_close(restored[field], training[field])


@pytest.mark.parametrize("soc", [False, True])
def test_evidence_dense_prune_parity(soc):
    torch.manual_seed(1)
    first = model_from_config(e2ip_config(soc, per_layer_cutoffs=[4.0, 2.0, 1.3]))
    second = model_from_config(
        e2ip_config(soc, edge_prune="dense", per_layer_cutoffs=[4.0, 2.0, 1.3])
    )
    second.load_state_dict(first.state_dict())
    sample = e2ip_sample()
    a, b = first(copy.deepcopy(sample)), second(copy.deepcopy(sample))
    for field in ("f_sigma", "f_nu", "f_kappa", "forces"):
        torch.testing.assert_close(a[field], b[field])


def test_e2ip_rejects_unsupported_options():
    with pytest.raises(ValueError, match="pad_to_alignment"):
        model_from_config(e2ip_config(pad_to_alignment=8))
    with pytest.raises(ValueError, match="l_max"):
        model_from_config(e2ip_config(l_max=0))
    with pytest.raises(ValueError, match="compile"):
        model_from_config(e2ip_config(spingnn_compile={"enabled": True}))
    with pytest.raises(ValueError, match="evidential_key"):
        model_from_config(e2ip_config(key="spin_forces_vert"))


@pytest.mark.parametrize(
    "soc,key", [(False, "forces"), (True, "spin_forces_vert"), (True, "forces")]
)
@pytest.mark.parametrize("prune", ["nonzero", "dense"])
def test_empty_edges(soc, key, prune):
    model = model_from_config(e2ip_config(soc, key, edge_prune=prune))
    sample = e2ip_sample()
    sample["edge_index"] = torch.empty(2, 0, dtype=torch.long)
    sample["edge_cell_shift"] = torch.empty(0, 3)
    out = model(sample)
    prefix = "sf" if key == "spin_forces_vert" else "f"
    assert torch.isfinite(out[prefix + "_sigma"]).all()
    assert out[prefix + "_sigma"].shape == (3, 3, 3)


@pytest.mark.parametrize(
    "value", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
def test_evidence_rejects_invalid_spin_scale(value):
    with pytest.raises(ValueError, match="e2ip_spin_force_scale"):
        e2IP_nodewise(e2ip_spin_force_scale=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_evidence_rejects_nonfinite_soc_coefficient(value):
    with pytest.raises(ValueError, match="soc_f_coeff"):
        e2IP_nodewise(soc_f_coeff=value)


@pytest.mark.parametrize("value", [-0.5, 0.0, 0.5])
def test_evidence_supports_finite_soc_coefficients(value):
    head = e2IP_nodewise(
        irreps_in={"f_sigma_s_raw": "2x0e", "f_sigma_v_raw": "2x1o"},
        soc_f_coeff=value,
    )
    assert head.soc_f_coeff == value


@pytest.mark.parametrize("magnitude", [0.0, 1e-30, 1e-10, 1e-9])
@pytest.mark.parametrize("scripted", [False, True])
def test_damper_linear_region_derivatives(magnitude, scripted):
    damper = DampedSymmetricOutput()
    if scripted:
        damper = torch.jit.script(damper)
    x = torch.full((1, 6), magnitude, requires_grad=True)
    torch.testing.assert_close(damper(x), x, rtol=0.0, atol=0.0)
    jacobian = torch.autograd.functional.jacobian(damper, x).reshape(6, 6)
    torch.testing.assert_close(jacobian, torch.eye(6), rtol=0.0, atol=0.0)
    # Check each output, not just their sum, so cancellation cannot hide NaNs.
    for component in range(6):
        hessian = torch.autograd.functional.hessian(
            lambda value: damper(value)[0, component], x
        )
        torch.testing.assert_close(
            hessian, torch.zeros_like(hessian), rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("scripted", [False, True])
def test_damper_nonzero_values_and_derivatives_unchanged(scripted):
    def reference(x):
        scalar = linear_tanh_damper(x[:, :1], 5.0, 8.0)
        scalar = -linear_tanh_damper(-scalar, 20.0, 25.0)
        nonscalar = x[:, 1:]
        norm = torch.linalg.vector_norm(nonscalar, dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.cat(
            (scalar, nonscalar * (linear_tanh_damper(norm, 5.0, 8.0) / norm)),
            dim=-1,
        )

    damper = DampedSymmetricOutput()
    if scripted:
        damper = torch.jit.script(damper)
        reference = torch.jit.script(reference)
    x = torch.tensor(
        [
            [1.0, 1e-7, 2e-7, -1e-7, 1e-7, 3e-7],
            [2.0, 1.0, 0.0, -1.0, 0.5, 2.0],
            [6.0, 4.0, -3.0, 4.0, 5.0, -6.0],
            [-23.0, -30.0, 15.0, 20.0, 10.0, -5.0],
        ],
        requires_grad=True,
    )
    torch.testing.assert_close(damper(x), reference(x), rtol=0.0, atol=0.0)
    # Safe branching changes backward accumulation order; use the model's
    # existing derivative tolerances while keeping forward values bitwise exact.
    rtol, atol = (1e-4, 1e-6) if x.dtype == torch.float32 else (1e-12, 1e-14)
    actual = torch.autograd.grad(damper(x).square().sum(), x, create_graph=True)[0]
    expected = torch.autograd.grad(reference(x).square().sum(), x, create_graph=True)[0]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    actual_second = torch.autograd.grad(actual.sum(), x)[0]
    expected_second = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(actual_second, expected_second, rtol=rtol, atol=atol)
