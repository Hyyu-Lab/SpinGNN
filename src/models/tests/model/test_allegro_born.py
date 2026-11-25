from collections import OrderedDict

import pytest
import torch

from nequip.data import AtomicData, AtomicDataDict
from nequip.model import ParaStressForceOutput, model_from_config
from nequip.nn import AtomwiseReduce, PerSpeciesScaleShift, SequentialGraphNetwork
from nequip.utils.unittests.model_tests import BaseEnergyModelTests

from allegro.nn import CartesianBECReduce, EdgewiseReduce

from _class_scoped_fixtures import ClassScopedModelFixtures

COMMON_CONFIG = {
    "avg_num_neighbors": 5.0,  # very approximate to keep numerics sane
    "r_max": 4,
    "num_types": 3,
    "type_names": ["H", "C", "O"],
    "env_embed_multiplicity": 4,
    "two_body_latent_mlp_latent_dimensions": [32],
    "latent_mlp_latent_dimensions": [32],
    "env_embed_mlp_latent_dimensions": [],
    "edge_eng_mlp_latent_dimensions": [16],
    "born_scale_list": [-1.0, 4.0, -2.0],
}
# TODO: test so3 mode when can pass down option to assert equivariance to ignore parity
minimal_config1 = dict(
    l_max=2,
    parity="o3_full",
    num_layers=2,
    **COMMON_CONFIG,
)
# minimal_config2 = dict(
#     l_max=1,
#     parity="o3_restricted",
#     num_layers=2,
#     born_scale_trainable=False,
#     **COMMON_CONFIG,
# )


class TestAllegro_BEC(ClassScopedModelFixtures, BaseEnergyModelTests):
    @pytest.fixture
    def strict_locality(self):
        return True

    @pytest.fixture(
        params=[
            minimal_config1,
            # minimal_config2,
        ],
        scope="class",
    )
    @classmethod
    def base_config(cls, request):
        return request.param

    @pytest.fixture(
        params=[
            (
                [
                    "allegro.model.Allegro_born",
                    "ParaStressForceOutput",
                ],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                    AtomicDataDict.STRESS_KEY,
                    AtomicDataDict.VIRIAL_KEY,
                    AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY,
                    AtomicDataDict.TOTAL_BORN_EFFECTIVE_CHARGE_KEY,
                ],
            ),
        ],
        scope="class",
    )
    @classmethod
    def config(cls, request, base_config, float_tolerance):
        config = base_config.copy()
        builder, out_fields = request.param
        config = config.copy()
        config["model_builders"] = builder
        return config, out_fields

    def test_equivariance(self, model, atomic_batch, device):
        from nequip.utils.test import assert_AtomicData_equivariant

        instance, out_fields = model
        instance = instance.to(device=device)
        atomic_batch = atomic_batch.to(device=device)
        assert_AtomicData_equivariant(func=instance, data_in=atomic_batch)

    def test_no_piezo_output(self, model, atomic_batch, device):
        instance, _ = model
        instance = instance.to(device=device)
        data = AtomicData.to_AtomicDataDict(atomic_batch.to(device=device))
        data.pop(AtomicDataDict.PIEZOELECTRIC_KEY, None)
        assert AtomicDataDict.PIEZOELECTRIC_KEY not in instance(data)


# -- behaviour preservation of the shared BEC reduction --------------------------
# Allegro_born used three layers for the BEC reduction (edge_born_sum ->
# born_scale -> total_born_sum). It now uses one CartesianBECReduce layer that
# keeps the name `born_scale`. The tests rebuild the old layer stack inline and
# require identical state_dict keys and bitwise-identical outputs and gradients.
# The two models are separate instances called in the same sequence, because
# TorchScript-compiled submodules may change numerics after their first calls.

BORN = AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY
TOTAL = AtomicDataDict.TOTAL_BORN_EFFECTIVE_CHARGE_KEY

PRESERVATION_CONFIGS = {
    "default": {},
    "dense": {"edge_prune": "dense"},
    "frozen_scalar_scale": {"born_scale_list": 1.5, "born_scale_trainable": False},
}


def _born_model(overrides):
    config = dict(minimal_config1, model_builders=["allegro.model.Allegro_born"])
    config.update(overrides)
    torch.manual_seed(1234)
    return model_from_config(dict(config)), config


def _legacy_reduction_stack(model, config):
    """The pre-refactor layer stack around the other layers of ``model``."""
    layers = OrderedDict()
    previous = None
    for name, module in model.named_children():
        if name != "born_scale":
            layers[name] = module
            previous = module
            continue
        edge = EdgewiseReduce(
            field=BORN,
            out_field=BORN,
            normalize_edge_reduce=True,
            avg_num_neighbors=config["avg_num_neighbors"],
            reduce="sum",
            irreps_in=previous.irreps_out,
        )
        scale = PerSpeciesScaleShift(
            field=BORN,
            out_field=BORN,
            num_types=config["num_types"],
            type_names=config["type_names"],
            shifts=0.0,
            shifts_trainable=False,
            scales=config.get("born_scale_list", 1.0),
            scales_trainable=config.get("born_scale_trainable", True),
            arguments_in_dataset_units=False,
            irreps_in=edge.irreps_out,
        )
        total = AtomwiseReduce(
            reduce="mean", field=BORN, out_field=TOTAL, irreps_in=scale.irreps_out
        )
        layers.update(edge_born_sum=edge, born_scale=scale, total_born_sum=total)
        previous = total
    return SequentialGraphNetwork(layers)


def _bitwise_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


@pytest.mark.filterwarnings("ignore:`torch.jit.*:DeprecationWarning")
@pytest.mark.filterwarnings(
    "ignore:The TorchScript type system doesn't support.*:UserWarning"
)
@pytest.mark.filterwarnings("ignore:edge_prune='dense' computes.*:UserWarning")
@pytest.mark.filterwarnings("ignore:!! Stresses in NequIP are in BETA.*:UserWarning")
@pytest.mark.parametrize("variant", sorted(PRESERVATION_CONFIGS))
def test_born_reduction_preserves_legacy_layers(variant, atomic_batch):
    model, config = _born_model(PRESERVATION_CONFIGS[variant])
    assert isinstance(model.born_scale, CartesianBECReduce)
    legacy = _legacy_reduction_stack(
        _born_model(PRESERVATION_CONFIGS[variant])[0], config
    )

    # identical checkpoint layout; old checkpoints load (and vice versa)
    assert list(model.state_dict()) == list(legacy.state_dict())
    assert sorted(k for k in model.state_dict() if k.startswith("born_scale")) == [
        "born_scale.scales",
        "born_scale.shifts",
    ]
    assert [n for n, _ in model.named_parameters()] == [
        n for n, _ in legacy.named_parameters()
    ]
    with torch.no_grad():
        model.born_scale.scales.mul_(1.25).add_(0.375)
    legacy.load_state_dict(model.state_dict(), strict=True)
    model.load_state_dict(legacy.state_dict(), strict=True)

    data = AtomicData.to_AtomicDataDict(atomic_batch)
    out_new = model({k: v.clone() for k, v in data.items()})
    out_old = legacy({k: v.clone() for k, v in data.items()})
    assert out_new.keys() == out_old.keys()
    for key, value in out_old.items():
        assert _bitwise_equal(out_new[key], value), key

    # forces/stress and parameter gradients through the wrapped models
    def run(module):
        wrapped = ParaStressForceOutput(module).train()
        module.zero_grad(set_to_none=True)
        out = wrapped({k: v.clone() for k, v in data.items()})
        objective = sum(
            (
                out[key] * torch.linspace(0.5, 1.5, out[key].numel()).view_as(out[key])
            ).sum()
            for key in (
                AtomicDataDict.FORCE_KEY,
                AtomicDataDict.STRESS_KEY,
                BORN,
                TOTAL,
            )
        )
        objective.backward()
        grads = {
            n: p.grad.detach().clone()
            for n, p in module.named_parameters()
            if p.grad is not None
        }
        return {k: v.detach() for k, v in out.items()}, grads

    out_new, grads_new = run(model)
    out_old, grads_old = run(legacy)
    for key in (AtomicDataDict.FORCE_KEY, AtomicDataDict.STRESS_KEY, BORN, TOTAL):
        assert _bitwise_equal(out_new[key], out_old[key]), key
    assert grads_new.keys() == grads_old.keys()
    for name, grad in grads_old.items():
        assert _bitwise_equal(grads_new[name], grad), name


@pytest.mark.filterwarnings("ignore:`torch.jit.*:DeprecationWarning")
@pytest.mark.filterwarnings(
    "ignore:The TorchScript type system doesn't support.*:UserWarning"
)
@pytest.mark.parametrize(
    "key, replacement",
    [
        ("edge_born_sum_avg_num_neighbors", "born_scale_avg_num_neighbors"),
        ("EdgewiseReduce_avg_num_neighbors", "born_scale_avg_num_neighbors"),
        ("edge_born_sum_reduce", None),
        ("total_born_sum_reduce", None),
        ("born_scale_scales", "born_scale_list"),
        ("born_scale_scales_trainable", "born_scale_trainable"),
        ("born_scale_arguments_in_dataset_units", None),
    ],
)
def test_removed_reduction_layer_options_raise(key, replacement):
    config = dict(minimal_config1, model_builders=["allegro.model.Allegro_born"])
    config[key] = 1.0
    with pytest.raises(ValueError, match="merged into the `born_scale` layer") as error:
        model_from_config(config)
    assert f"`{key}`" in str(error.value)
    if replacement is not None:
        assert f"`{replacement}`" in str(error.value)


@pytest.mark.filterwarnings("ignore:`torch.jit.*:DeprecationWarning")
@pytest.mark.filterwarnings(
    "ignore:The TorchScript type system doesn't support.*:UserWarning"
)
def test_born_scale_avg_num_neighbors_replaces_edge_sum_option():
    config = dict(minimal_config1, model_builders=["allegro.model.Allegro_born"])
    assert model_from_config(dict(config)).born_scale._factor == 5.0**-0.5
    config["born_scale_avg_num_neighbors"] = 20.0
    assert model_from_config(config).born_scale._factor == 20.0**-0.5
