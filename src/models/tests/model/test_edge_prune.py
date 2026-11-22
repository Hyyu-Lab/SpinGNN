"""Regression tests for the source-level Allegro edge pruning modes."""

from __future__ import annotations

import copy

import pytest
import torch

from nequip.utils.fx_compat import _training_energy_core_for_trace

from nequip.data import AtomicData, AtomicDataDict
from nequip.model import model_from_config
from nequip.scripts import deploy
from nequip.scripts.deploy import _compile_for_deploy
from nequip.utils.test import assert_AtomicData_equivariant

from test_allegro_born import minimal_config1 as BORN_CONFIG
from test_allegro_spin_nosoc import minimal_config3 as NOSOC_CONFIG
from test_allegro_spin_soc import minimal_config2 as SOC_CONFIG
from test_allegro_universal import minimal_config2 as UNIVERSAL_CONFIG


pytestmark = [
    pytest.mark.filterwarnings("ignore:`torch.jit.*:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:The TorchScript type system doesn't support.*:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:edge_prune='dense' computes every supplied edge.*:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:!! Stresses in NequIP are in BETA.*:UserWarning"
    ),
]


COMMON_CONFIG = {
    "avg_num_neighbors": 5.0,
    "r_max": 4.0,
    "num_types": 3,
    "type_names": ["H", "C", "O"],
    "env_embed_multiplicity": 4,
    "two_body_latent_mlp_latent_dimensions": [16],
    "latent_mlp_latent_dimensions": [16],
    "env_embed_mlp_latent_dimensions": [],
    "edge_eng_mlp_latent_dimensions": [8],
}

CORE_CASES = (
    pytest.param(
        None,
        "allegro.model.Allegro",
        ("Allegro_og_Module",),
        False,
        id="og",
    ),
    pytest.param(
        BORN_CONFIG,
        "allegro.model.Allegro_born",
        ("Allegro_born_Module",),
        False,
        id="born",
    ),
    pytest.param(
        SOC_CONFIG,
        "allegro.model.Allegro_soc",
        ("Allegro_spin_Module", "Allegro_og_Module"),
        True,
        id="spin",
    ),
    pytest.param(
        UNIVERSAL_CONFIG,
        "allegro.model.Allegro_universal",
        ("Allegro_Module",),
        False,
        id="generic",
    ),
)


@pytest.mark.parametrize(
    ("base_config", "builder", "core_names", "uses_spin"), CORE_CASES
)
@pytest.mark.parametrize("per_layer_cutoffs", (None, [4.0, 3.0, 2.0]))
def test_training_trace_dense_copy_preserves_second_derivatives(
    base_config, builder, core_names, uses_spin, per_layer_cutoffs
):
    del core_names
    torch.manual_seed(1234)
    model = model_from_config(
        _case_config(base_config, builder, "nonzero", per_layer_cutoffs)
    ).train()
    traced_core = _training_energy_core_for_trace(model)
    assert traced_core is not model
    assert all(
        module.edge_prune == "nonzero"
        for module in model.modules()
        if hasattr(module, "edge_prune")
    )
    assert all(
        module.edge_prune == "dense"
        for module in traced_core.modules()
        if hasattr(module, "edge_prune")
    )
    assert tuple(map(id, model.parameters())) == tuple(
        map(id, traced_core.parameters())
    )
    assert tuple(map(id, model.buffers())) == tuple(map(id, traced_core.buffers()))

    def derivatives(core):
        data = {key: value.clone() for key, value in _mixed_cutoff_data().items()}
        positions = data[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
        inputs = [positions]
        if uses_spin:
            inputs.append(data[AtomicDataDict.SPIN_KEY].requires_grad_(True))
        output = core(data)
        gradients = torch.autograd.grad(
            output[AtomicDataDict.TOTAL_ENERGY_KEY].sum(), inputs, create_graph=True
        )
        loss = sum(gradient.square().sum() for gradient in gradients)
        parameter_gradients = torch.autograd.grad(
            loss,
            tuple(
                parameter for parameter in core.parameters() if parameter.requires_grad
            ),
            allow_unused=True,
        )
        return gradients, parameter_gradients

    reference = derivatives(model)
    candidate = derivatives(traced_core)
    rtol, atol = (
        (1e-4, 1e-6) if torch.get_default_dtype() == torch.float32 else (1e-11, 1e-13)
    )
    for expected_values, actual_values in zip(reference, candidate):
        for expected, actual in zip(expected_values, actual_values):
            assert (expected is None) == (actual is None)
            if expected is not None:
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _config(builder="allegro.model.Allegro", **overrides):
    config = dict(
        COMMON_CONFIG,
        l_max=2,
        parity="o3_full",
        num_layers=2,
        model_builders=[builder],
    )
    config.update(overrides)
    return config


def _case_config(base_config, builder, edge_prune, per_layer_cutoffs=None):
    config = _config(builder) if base_config is None else copy.deepcopy(base_config)
    config["model_builders"] = [builder]
    config["edge_prune"] = edge_prune
    if per_layer_cutoffs is not None:
        config["per_layer_cutoffs"] = per_layer_cutoffs
    return config


def test_edge_prune_invalid_value_fails_at_model_build():
    with pytest.raises(ValueError, match="edge_prune"):
        model_from_config(_config(edge_prune="invalid"))


def test_edge_prune_deploy_auto_dense_is_fail_closed():
    soc_config = _case_config(SOC_CONFIG, "allegro.model.Allegro_soc", "nonzero")
    soc_config.pop("edge_prune")
    soc_model = model_from_config(soc_config)
    assert deploy._enable_equal_cutoff_dense_edge_prune(
        soc_model, soc_config, strict_lammps_prefilter=True
    )
    assert soc_config["edge_prune"] == "dense"
    assert {
        module.edge_prune
        for module in soc_model.modules()
        if hasattr(module, "edge_prune")
    } == {"dense"}

    explicit_config = _case_config(SOC_CONFIG, "allegro.model.Allegro_soc", "nonzero")
    explicit_model = model_from_config(explicit_config)
    assert not deploy._enable_equal_cutoff_dense_edge_prune(
        explicit_model, explicit_config, strict_lammps_prefilter=True
    )

    decreasing_config = _case_config(
        SOC_CONFIG,
        "allegro.model.Allegro_soc",
        "nonzero",
        per_layer_cutoffs=[4.0, 3.0, 2.0],
    )
    decreasing_config.pop("edge_prune")
    decreasing_model = model_from_config(decreasing_config)
    assert not deploy._enable_equal_cutoff_dense_edge_prune(
        decreasing_model, decreasing_config, strict_lammps_prefilter=True
    )

    born_config = _case_config(BORN_CONFIG, "allegro.model.Allegro_born", "nonzero")
    born_config.pop("edge_prune")
    born_model = model_from_config(born_config)
    assert not deploy._enable_equal_cutoff_dense_edge_prune(
        born_model, born_config, strict_lammps_prefilter=True
    )

    no_prefilter_config = _case_config(
        SOC_CONFIG, "allegro.model.Allegro_soc", "nonzero"
    )
    no_prefilter_config.pop("edge_prune")
    no_prefilter_model = model_from_config(no_prefilter_config)
    assert not deploy._enable_equal_cutoff_dense_edge_prune(
        no_prefilter_model,
        no_prefilter_config,
        strict_lammps_prefilter=False,
    )


@pytest.mark.parametrize(
    ("base_config", "builder", "core_names", "uses_spin"), CORE_CASES
)
def test_edge_prune_default_and_dense_are_propagated_to_allegro_core(
    base_config, builder, core_names, uses_spin
):
    del uses_spin
    default_config = _case_config(base_config, builder, "nonzero")
    default_config.pop("edge_prune")
    default_model = model_from_config(default_config)
    dense_model = model_from_config(_case_config(base_config, builder, "dense"))

    for model, expected_mode in (
        (default_model, "nonzero"),
        (dense_model, "dense"),
    ):
        modes = {
            type(module).__name__: module.edge_prune
            for module in model.modules()
            if type(module).__name__ in core_names
        }
        assert modes == {name: expected_mode for name in core_names}


def _mixed_cutoff_data():
    dtype = torch.get_default_dtype()
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    return {
        AtomicDataDict.POSITIONS_KEY: positions,
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor(
            [[0, 0, 0, 0, 0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]],
            dtype=torch.long,
        ),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.zeros((10, 3), dtype=dtype),
        AtomicDataDict.CELL_KEY: torch.eye(3, dtype=dtype).unsqueeze(0) * 20.0,
        AtomicDataDict.PBC_KEY: torch.zeros((1, 3), dtype=torch.bool),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1, 2, 0, 1, 2]),
        AtomicDataDict.SPIN_KEY: torch.tensor(
            [
                [0.2, -0.1, 0.4],
                [0.3, 0.5, -0.2],
                [-0.4, 0.1, 0.2],
                [0.6, -0.3, 0.1],
                [0.2, 0.1, -0.5],
                [-0.1, 0.4, 0.3],
            ],
            dtype=dtype,
        ),
    }


def _empty_edge_data():
    dtype = torch.get_default_dtype()
    return {
        AtomicDataDict.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=dtype
        ),
        AtomicDataDict.EDGE_INDEX_KEY: torch.empty((2, 0), dtype=torch.long),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.empty((0, 3), dtype=dtype),
        AtomicDataDict.CELL_KEY: torch.eye(3, dtype=dtype).unsqueeze(0) * 20.0,
        AtomicDataDict.PBC_KEY: torch.zeros((1, 3), dtype=torch.bool),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1]),
        AtomicDataDict.SPIN_KEY: torch.tensor(
            [[0.2, -0.1, 0.4], [0.3, 0.5, -0.2]], dtype=dtype
        ),
    }


def _assert_output_close(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if not isinstance(actual[key], torch.Tensor):
            continue
        if actual[key].is_floating_point():
            tolerance = 128 * torch.finfo(actual[key].dtype).eps
            torch.testing.assert_close(
                actual[key], expected[key], rtol=tolerance, atol=tolerance
            )
        else:
            assert torch.equal(actual[key], expected[key]), key


def _run_with_input_gradients(model, data, uses_spin):
    sample = {key: value.detach().clone() for key, value in data.items()}
    sample[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
    inputs = [sample[AtomicDataDict.POSITIONS_KEY]]
    if uses_spin:
        sample[AtomicDataDict.SPIN_KEY].requires_grad_(True)
        inputs.append(sample[AtomicDataDict.SPIN_KEY])
    output = model(sample)
    gradients = torch.autograd.grad(
        output[AtomicDataDict.TOTAL_ENERGY_KEY].sum(), inputs
    )
    return output, gradients


@pytest.mark.parametrize(
    ("base_config", "builder", "core_names", "uses_spin"), CORE_CASES
)
@pytest.mark.parametrize("per_layer_cutoffs", (None, [4.0, 3.0, 2.0]))
def test_edge_prune_nonzero_dense_output_and_gradient_parity(
    base_config, builder, core_names, uses_spin, per_layer_cutoffs
):
    del core_names
    torch.manual_seed(1234)
    nonzero = model_from_config(
        _case_config(base_config, builder, "nonzero", per_layer_cutoffs)
    )
    dense = model_from_config(
        _case_config(base_config, builder, "dense", per_layer_cutoffs)
    )
    dense.load_state_dict(copy.deepcopy(nonzero.state_dict()))
    nonzero.eval()
    dense.eval()

    # Active, later-pruned, exact-boundary, and outside-cutoff edges coexist.
    data = _mixed_cutoff_data()
    nonzero_out, nonzero_grad = _run_with_input_gradients(nonzero, data, uses_spin)
    dense_out, dense_grad = _run_with_input_gradients(dense, data, uses_spin)
    _assert_output_close(nonzero_out, dense_out)
    for actual, expected in zip(nonzero_grad, dense_grad):
        tolerance = 128 * torch.finfo(actual.dtype).eps
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    ("base_config", "builder", "core_names", "uses_spin"), CORE_CASES
)
def test_edge_prune_empty_edge_output_and_parameter_gradient_parity(
    base_config, builder, core_names, uses_spin
):
    del core_names, uses_spin
    torch.manual_seed(4321)
    nonzero = model_from_config(_case_config(base_config, builder, "nonzero"))
    dense = model_from_config(_case_config(base_config, builder, "dense"))
    dense.load_state_dict(copy.deepcopy(nonzero.state_dict()))
    data = _empty_edge_data()

    def run(model):
        model.zero_grad(set_to_none=True)
        output = model({key: value.clone() for key, value in data.items()})
        output[AtomicDataDict.TOTAL_ENERGY_KEY].sum().backward()
        gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        return output, gradients

    nonzero_out, nonzero_grad = run(nonzero)
    dense_out, dense_grad = run(dense)
    _assert_output_close(nonzero_out, dense_out)
    assert nonzero_grad.keys() == dense_grad.keys()
    for key in nonzero_grad:
        assert (nonzero_grad[key] is None) == (dense_grad[key] is None), key
        if nonzero_grad[key] is not None:
            torch.testing.assert_close(
                nonzero_grad[key], dense_grad[key], rtol=0.0, atol=0.0
            )


@pytest.mark.parametrize(
    ("base_config", "builder", "core_names", "uses_spin"), CORE_CASES
)
def test_edge_prune_dense_forward_does_not_call_tensor_nonzero(
    monkeypatch, base_config, builder, core_names, uses_spin
):
    del core_names, uses_spin
    model = model_from_config(_case_config(base_config, builder, "dense")).eval()

    def forbidden_nonzero(*args, **kwargs):
        raise AssertionError("dense edge pruning called Tensor.nonzero")

    monkeypatch.setattr(torch.Tensor, "nonzero", forbidden_nonzero)
    model(_mixed_cutoff_data())


@pytest.mark.parametrize(
    ("base_config", "builder", "core_names", "uses_spin"), CORE_CASES
)
def test_edge_prune_dense_torchscript_graph_and_eager_parity(
    base_config, builder, core_names, uses_spin
):
    del core_names, uses_spin
    model = model_from_config(_case_config(base_config, builder, "dense")).eval()
    data = _mixed_cutoff_data()
    expected = model({key: value.clone() for key, value in data.items()})
    scripted = _compile_for_deploy(model)
    actual = scripted({key: value.clone() for key, value in data.items()})
    _assert_output_close(actual, expected)

    graphs = []
    for module in scripted.modules():
        try:
            graphs.append(str(module.graph))
        except (AttributeError, RuntimeError):
            pass
    graph_text = "\n".join(graphs)
    assert "aten::nonzero" not in graph_text
    assert "aten::index_copy" not in graph_text
    assert "aten::index_add" not in graph_text


def _dense_physics_model(config, builder, output_builder):
    config = copy.deepcopy(config)
    config["edge_prune"] = "dense"
    config["model_builders"] = [builder, output_builder]
    torch.manual_seed(2468)
    return model_from_config(config).eval()


def test_edge_prune_dense_nosoc_rotation_and_time_reversal(atomic_batch_spin):
    model = _dense_physics_model(
        NOSOC_CONFIG,
        "allegro.model.Allegro_nosoc",
        "ParaStressForceSpinOutput",
    )
    assert_AtomicData_equivariant(
        func=model,
        data_in=atomic_batch_spin,
        do_parity=True,
        do_time_reversal=True,
        do_only_rot_spin=True,
    )


def test_edge_prune_dense_soc_joint_rotation_and_spin_only_distinction(
    atomic_batch_spin,
):
    model = _dense_physics_model(
        SOC_CONFIG,
        "allegro.model.Allegro_soc",
        "ParaStressForceSpinOutput",
    )
    assert_AtomicData_equivariant(
        func=model,
        data_in=atomic_batch_spin,
        do_parity=True,
        do_time_reversal=True,
        do_only_rot_spin=False,
    )

    data = AtomicData.to_AtomicDataDict(atomic_batch_spin)
    reference = {key: value.clone() for key, value in data.items()}
    spin_only = {key: value.clone() for key, value in data.items()}
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.get_default_dtype(),
    )
    spin_only[AtomicDataDict.SPIN_KEY] = spin_only[AtomicDataDict.SPIN_KEY] @ rotation.T
    reference_energy = model(reference)[AtomicDataDict.TOTAL_ENERGY_KEY]
    spin_only_energy = model(spin_only)[AtomicDataDict.TOTAL_ENERGY_KEY]
    difference = (reference_energy - spin_only_energy).abs().max()
    assert difference > 100.0 * torch.finfo(reference_energy.dtype).eps


def test_edge_prune_dense_bec_equivariance_and_no_piezo(atomic_batch):
    model = _dense_physics_model(
        BORN_CONFIG,
        "allegro.model.Allegro_born",
        "ParaStressForceOutput",
    )
    assert_AtomicData_equivariant(func=model, data_in=atomic_batch)
    output = model(AtomicData.to_AtomicDataDict(atomic_batch))
    assert AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY in output
    assert AtomicDataDict.PIEZOELECTRIC_KEY not in output
