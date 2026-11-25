"""``allegro_activation_checkpoint`` must be numerically transparent.

The flag trades compute for memory by recomputing Allegro module activations
during backward. Forward energy outputs must match the plain path bitwise;
autograd-derived forces/spin-forces and parameter gradients must match to
dtype roundoff, since recomputation (and host staging under
``allegro_checkpoint_offload``) can reassociate reductions by a few ULPs. On a
CPU-only host the pinned staging degrades to plain host copies.
"""

import numpy as np
import pytest
import torch
from torch.jit import script

from nequip.data import AtomicData, AtomicDataDict
from nequip.model import model_from_config
from nequip.utils.unittests.parity_tests import BaseCompileParityTests

BASE_CONFIG = dict(
    l_max=2,
    lnosoc_layer=True,
    lte3nn=True,
    ene_coeff_trainable=False,
    avg_num_neighbors=5.0,
    r_max=4,
    num_types=3,
    type_names=["H", "C", "O"],
    allegro_onlyr_num_layers=2,
    allegro_nosoc_num_layers=2,
    env_embed_multiplicity=7,
    two_body_latent_mlp_latent_dimensions=[32],
    latent_mlp_latent_dimensions=[32],
    env_embed_mlp_latent_dimensions=[],
    edge_onlyr_eng_mlp_latent_dimensions=[16],
    edge_hb_coeff_mlp_latent_dimensions=[16],
    edge_biquad_coeff_mlp_latent_dimensions=[16],
    edge_eng_nosoc_mlp_latent_dimensions=[16],
    edge_onlyr_eng_coeff=1.0,
    edge_hb_eng_coeff=0.1,
    edge_eng_nosoc_coeff=0.05,
    parity="o3_full",
    model_builders=["allegro.model.Allegro_nosoc", "ParaStressForceSpinOutput"],
)

COMPARE_FIELDS = [
    AtomicDataDict.TOTAL_ENERGY_KEY,
    AtomicDataDict.PER_ATOM_ENERGY_KEY,
    AtomicDataDict.FORCE_KEY,
    AtomicDataDict.SPIN_FORCE_KEY,
]


def _build(extra):
    torch.manual_seed(0)
    np.random.seed(0)
    return model_from_config(dict(BASE_CONFIG, **extra))


def _loss(output):
    return (
        output[AtomicDataDict.TOTAL_ENERGY_KEY].square().sum()
        + output[AtomicDataDict.FORCE_KEY].square().sum()
        + output[AtomicDataDict.SPIN_FORCE_KEY].square().sum()
    )


def _allegro_modules(model):
    return [m for m in model.modules() if type(m).__name__ == "Allegro_og_Module"]


@pytest.mark.parametrize("offload", [False, True], ids=["device", "offload"])
def test_activation_checkpoint_matches_baseline(
    atomic_batch_spin, monkeypatch, offload
):
    offload_scopes = []
    save_on_cpu = torch.autograd.graph.save_on_cpu

    def counting_save_on_cpu(*args, **kwargs):
        offload_scopes.append(kwargs)
        return save_on_cpu(*args, **kwargs)

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", counting_save_on_cpu)

    reference = _build({})
    checkpointed = _build(
        {"allegro_activation_checkpoint": True, "allegro_checkpoint_offload": offload}
    )
    reference.train()
    checkpointed.train()

    modules = _allegro_modules(checkpointed)
    assert len(modules) == 2  # allegro_onlyr + allegro_nosoc
    assert all(m.allegro_activation_checkpoint for m in modules)
    assert all(m.allegro_checkpoint_offload is offload for m in modules)
    assert not any(m.allegro_activation_checkpoint for m in _allegro_modules(reference))

    data = AtomicData.to_AtomicDataDict(atomic_batch_spin)
    out_ref = reference(dict(data))
    out_ckpt = checkpointed(dict(data))
    # the checkpoint path must actually have engaged (its grad_fn is the marker)
    engaged = {
        type(value.grad_fn).__name__
        for value in out_ckpt.values()
        if torch.is_tensor(value) and value.grad_fn is not None
    }
    assert "_ActivationCheckpointFunctionBackward" in engaged
    # force/spin-force autograd.grad ran the checkpoint backward, which must
    # have entered the host-offload scope exactly when offload is enabled
    assert bool(offload_scopes) is offload
    dtype = out_ref[AtomicDataDict.TOTAL_ENERGY_KEY].dtype
    deriv_rtol, deriv_atol = (1e-11, 1e-14) if dtype == torch.float64 else (1e-5, 1e-6)
    exact_fields = {
        AtomicDataDict.TOTAL_ENERGY_KEY,
        AtomicDataDict.PER_ATOM_ENERGY_KEY,
    }
    compared = 0
    for field in COMPARE_FIELDS:
        if field in out_ref:
            if field in exact_fields:
                torch.testing.assert_close(
                    out_ckpt[field], out_ref[field], rtol=0, atol=0
                )
            else:
                # Forces and spin-forces are computed via autograd.grad through
                # the recomputed backward graph; save_on_cpu staging on CPU can
                # shift BLAS/SIMD reduction alignment by 1-2 ULPs.
                torch.testing.assert_close(
                    out_ckpt[field],
                    out_ref[field],
                    rtol=deriv_rtol,
                    atol=deriv_atol,
                    msg=field,
                )
            compared += 1
    assert compared >= 3

    loss_ref = _loss(out_ref)
    loss_ckpt = _loss(out_ckpt)
    torch.testing.assert_close(loss_ckpt, loss_ref, rtol=deriv_rtol, atol=deriv_atol)
    loss_ref.backward()
    loss_ckpt.backward()
    ref_params = dict(reference.named_parameters())
    # Parameter gradients are compared at dtype roundoff level, not bitwise:
    # checkpointing recomputes the backward graph, so autograd reassociates
    # sums differently (measured <= 6e-7 relative in float32, <= 4e-16 in
    # float64).
    rtol, atol = (1e-12, 1e-14) if dtype == torch.float64 else (1e-4, 1e-6)
    checked = 0
    for name, param in checkpointed.named_parameters():
        ref_grad = ref_params[name].grad
        if ref_grad is None:
            assert param.grad is None, name
            continue
        assert param.grad is not None, name
        torch.testing.assert_close(param.grad, ref_grad, rtol=rtol, atol=atol, msg=name)
        checked += 1
    assert checked > 0


def test_activation_checkpoint_scriptable(atomic_batch_spin):
    model = _build({"allegro_activation_checkpoint": True})
    model.eval()
    scripted = script(model)
    data = AtomicData.to_AtomicDataDict(atomic_batch_spin)
    out_eager = model(dict(data))
    out_script = scripted(dict(data))
    for field in COMPARE_FIELDS:
        if field in out_eager:
            assert torch.allclose(out_eager[field], out_script[field], atol=1e-6)


class TestCheckpointCompileParity(BaseCompileParityTests):
    """Checkpoint recomputation and full-FX capture must compose correctly."""

    @pytest.fixture(scope="class")
    @classmethod
    def config(cls):
        return dict(BASE_CONFIG, allegro_activation_checkpoint=True), COMPARE_FIELDS
