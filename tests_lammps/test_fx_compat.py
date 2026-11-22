"""Inference-only FX graph expansion tests."""

from __future__ import annotations

import math

import e3nn
import pytest
import torch
from allegro.nn.cutoffs import polynomial_cutoff
from torch.fx.experimental.proxy_tensor import make_fx

from helpers import CASES, cloned
from model_factory import MODEL_SEED
from nequip.data import AtomicDataDict
from nequip.model import model_from_config
from nequip.nn import RescaleOutput
from nequip.nn.cutoffs import PolynomialCutoff, _poly_cutoff
from nequip.scripts.train import default_config
from nequip.utils import fx_compat
from nequip.utils import Config
from nequip.utils._global_options import _set_global_options
from nequip.utils.fx_compat import (
    INPUT_SIGNATURE,
    _allegro_polynomial_cutoff,
    _nequip_polynomial_cutoff,
    expand_deploy_graph,
    make_deploy_sample,
    strip_saved_tensor_detach,
)
from tolerances import PT2_ATOL, PT2_RTOL


# Factorized cutoff changed the float32 operation order. The observed
# eager-vs-FX envelope was 1.275e-6 relative / 5.07e-7 absolute on a nonzero
# virial element; keep measured headroom without approaching deployment gates.
FX_RTOL = 2.0e-6
FX_ATOL = 2.0e-7


def test_float32_allegro_cutoff_is_positive_strictly_inside_rmax():
    distances = torch.tensor([3.999, 4.0, 4.001], dtype=torch.float32)
    r_max = torch.tensor([4.0], dtype=torch.float32)
    scripted = polynomial_cutoff(distances, r_max, 6.0)
    traced = _allegro_polynomial_cutoff(distances, r_max, 6.0)
    assert scripted[0, 0] > 0.0
    assert scripted[0, 1] == 0.0
    assert scripted[0, 2] == 0.0
    torch.testing.assert_close(traced, scripted, rtol=0.0, atol=0.0)


def _float64_cutoff_reference(x, p):
    work = x.to(torch.float64)
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(work, p))
    out = out + (p * (p + 2.0) * torch.pow(work, p + 1.0))
    out = out - ((p * (p + 1.0) / 2.0) * torch.pow(work, p + 2.0))
    return (out * (work < 1.0)).to(x.dtype)


def test_float32_nequip_cutoff_is_positive_strictly_inside_rmax():
    distances = torch.tensor([3.999, 4.0, 4.001], dtype=torch.float32)
    scripted = _poly_cutoff(distances, 0.25, 6.0)
    traced = _nequip_polynomial_cutoff(distances, 0.25, 6.0)
    assert scripted[0] > 0.0
    assert scripted[1] == 0.0
    assert scripted[2] == 0.0
    torch.testing.assert_close(traced, scripted, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("p", (2.0, 6.0, 8.0))
def test_factorized_cutoff_paths_match_float64_value_and_gradient(p):
    normalized = torch.linspace(0.0, 1.001, 2003, dtype=torch.float32)
    reference_input = normalized.detach().clone().requires_grad_(True)
    reference = _float64_cutoff_reference(reference_input, p)
    reference_gradient = torch.autograd.grad(reference.sum(), reference_input)[0]

    candidates = (
        lambda value: polynomial_cutoff(value * 4.0, torch.tensor([4.0]), p)[0],
        lambda value: _allegro_polynomial_cutoff(value * 4.0, torch.tensor([4.0]), p)[
            0
        ],
        lambda value: _poly_cutoff(value * 4.0, 0.25, p),
        lambda value: _nequip_polynomial_cutoff(value * 4.0, 0.25, p),
    )
    outputs = []
    for candidate in candidates:
        candidate_input = normalized.detach().clone().requires_grad_(True)
        output = candidate(candidate_input)
        gradient = torch.autograd.grad(output.sum(), candidate_input)[0]
        torch.testing.assert_close(output, reference, rtol=2.0e-6, atol=2.0e-7)
        torch.testing.assert_close(
            gradient, reference_gradient, rtol=2.0e-6, atol=2.0e-6
        )
        outputs.append(output)
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=0.0, atol=0.0)


def test_cutoff_graphs_use_factorized_horner_without_pow_or_float64():
    distances = torch.tensor([1.0, 3.999, 4.0], dtype=torch.float32)
    r_max = torch.tensor([4.0], dtype=torch.float32)
    graphs = (
        str(polynomial_cutoff.graph),
        str(_poly_cutoff.graph),
        str(
            make_fx(lambda x, cutoff: _allegro_polynomial_cutoff(x, cutoff, 6.0))(
                distances, r_max
            ).graph
        ),
        str(
            make_fx(lambda x: _nequip_polynomial_cutoff(x, 0.25, 6.0))(distances).graph
        ),
    )
    for graph in graphs:
        assert "aten::pow" not in graph and "aten.pow" not in graph
        assert "float64" not in graph and "Double" not in graph


@pytest.mark.parametrize("p", (2.5, 6.5))
def test_cutoff_rejects_noninteger_power(p):
    distances = torch.tensor([1.0], dtype=torch.float32)
    r_max = torch.tensor([4.0], dtype=torch.float32)
    for cutoff_call in (
        lambda: polynomial_cutoff(distances, r_max, p),
        lambda: _allegro_polynomial_cutoff(distances, r_max, p),
        lambda: _poly_cutoff(distances, 0.25, p),
        lambda: _nequip_polynomial_cutoff(distances, 0.25, p),
        lambda: PolynomialCutoff(4.0, p),
    ):
        with pytest.raises((AssertionError, torch.jit.Error)):
            cutoff_call()


def _build_fx_model(case_name: str, edge_prune: str = "nonzero"):
    torch.manual_seed(MODEL_SEED)
    config = Config.from_file(
        str(CASES[case_name].config_path), defaults=default_config
    )
    config["e3nn_optimization_defaults"] = {"jit_script_fx": False}
    config["edge_prune"] = edge_prune
    _set_global_options(config)
    model = model_from_config(config, initialize=False, deploy=True)
    model.eval()
    return model


def _rescale_keys(model):
    candidates = (
        AtomicDataDict.TOTAL_ENERGY_KEY,
        AtomicDataDict.PER_ATOM_ENERGY_KEY,
        AtomicDataDict.FORCE_KEY,
        AtomicDataDict.SPIN_FORCE_KEY,
        AtomicDataDict.VIRIAL_KEY,
        AtomicDataDict.ATOM_VIRIAL_KEY,
    )
    return tuple(key for key in candidates if key in model.irreps_out)


def _wrap_rescale(model, *, scale_by, shift_by):
    return RescaleOutput(
        model,
        scale_keys=_rescale_keys(model),
        shift_keys=(AtomicDataDict.TOTAL_ENERGY_KEY,),
        related_scale_keys=(AtomicDataDict.PER_ATOM_ENERGY_KEY,),
        scale_by=scale_by,
        shift_by=shift_by,
    ).eval()


def _eager_input(sample):
    return {
        AtomicDataDict.POSITIONS_KEY: sample[0].clone(),
        AtomicDataDict.SPIN_KEY: sample[1].clone(),
        AtomicDataDict.EDGE_INDEX_KEY: sample[2].clone(),
        AtomicDataDict.ATOM_TYPE_KEY: sample[3].clone(),
    }


def _assert_outputs_close(graph_module, eager_output, sample):
    graph_output = graph_module(*sample)
    eager_tuple = tuple(eager_output[key] for key in graph_module.output_signature)
    assert len(graph_output) == len(eager_tuple)
    for eager, traced in zip(eager_tuple, graph_output):
        torch.testing.assert_close(traced, eager, rtol=FX_RTOL, atol=FX_ATOL)


def _is_prime(value: int) -> bool:
    return value >= 2 and all(
        value % divisor for divisor in range(2, math.isqrt(value) + 1)
    )


@pytest.mark.parametrize("case_name", ("nosoc", "soc"))
def test_expand_deploy_graph_matches_eager(case_name):
    model = _build_fx_model(case_name)
    assert e3nn.get_optimization_defaults()["jit_script_fx"] is False
    sample = make_deploy_sample(model)
    forbidden = {
        int(size)
        for tensor in list(model.parameters()) + list(model.buffers())
        for size in tensor.shape
    }
    assert sample[0].shape[0] != sample[2].shape[1]
    assert _is_prime(sample[0].shape[0]) and sample[0].shape[0] not in forbidden
    assert _is_prime(sample[2].shape[1]) and sample[2].shape[1] not in forbidden
    assert all(tensor.is_contiguous() for tensor in sample)

    eager_output = model(cloned(_eager_input(sample)))
    graph_module = expand_deploy_graph(model, sample)
    assert graph_module.input_signature == INPUT_SIGNATURE
    assert graph_module.removed_edge_filters > 0
    _assert_outputs_close(graph_module, eager_output, sample)
    assert all("autograd" not in str(node.target) for node in graph_module.graph.nodes)


def test_expand_bec_deploy_graph_uses_nonspin_signature():
    model = _build_fx_model("bec")
    sample = make_deploy_sample(model)
    expected_signature = (
        AtomicDataDict.POSITIONS_KEY,
        AtomicDataDict.EDGE_INDEX_KEY,
        AtomicDataDict.ATOM_TYPE_KEY,
    )

    assert len(sample) == len(expected_signature)
    eager_output = model(cloned(dict(zip(expected_signature, sample))))
    graph_module = expand_deploy_graph(model, sample)

    assert graph_module.input_signature == expected_signature
    _assert_outputs_close(graph_module, eager_output, sample)
    assert all("autograd" not in str(node.target) for node in graph_module.graph.nodes)


@pytest.mark.parametrize(
    ("irreps_in", "irreps_out"),
    (
        ({AtomicDataDict.SPIN_KEY: object()}, {}),
        ({}, {AtomicDataDict.SPIN_FORCE_KEY: object()}),
    ),
)
def test_pt2_model_spin_input_and_output_capabilities_must_match(irreps_in, irreps_out):
    model = type(
        "DeclaredSpinContract",
        (),
        {"irreps_in": irreps_in, "irreps_out": irreps_out},
    )()

    with pytest.raises(ValueError, match="spin input and spin_forces output"):
        fx_compat._model_has_spin(model)


def test_dropping_rescale_exceeds_pt2_parity_gate():
    wrapped = _wrap_rescale(_build_fx_model("nosoc"), scale_by=7.25, shift_by=13.0)
    sample = make_deploy_sample(wrapped)
    eager_output = wrapped(cloned(_eager_input(sample)))
    dropped_graph = expand_deploy_graph(wrapped.model, sample)
    dropped_output = dict(zip(dropped_graph.output_signature, dropped_graph(*sample)))

    for key in _rescale_keys(wrapped):
        if key not in dropped_output:
            continue
        reference = eager_output[key]
        threshold = PT2_ATOL + PT2_RTOL * torch.max(torch.abs(reference)).item()
        max_abs_error = torch.max(torch.abs(dropped_output[key] - reference)).item()
        assert max_abs_error > threshold, (
            f"dropping learned scaling did not trip the PT2 gate for {key}: "
            f"max_abs_error={max_abs_error}, threshold={threshold}"
        )


def test_expand_deploy_graph_replays_rescale_after_derived_outputs():
    wrapped = _wrap_rescale(_build_fx_model("nosoc"), scale_by=7.25, shift_by=13.0)
    sample = make_deploy_sample(wrapped)
    eager_output = wrapped(cloned(_eager_input(sample)))
    graph_module = expand_deploy_graph(wrapped, sample)

    _assert_outputs_close(graph_module, eager_output, sample)
    assert graph_module.num_rescale_layers == 1
    targets = tuple(str(node.target) for node in graph_module.graph.nodes)
    assert any("aten.mul" in target for target in targets)
    assert any("aten.add" in target for target in targets)


def test_expand_deploy_graph_replays_nested_rescale_inner_to_outer():
    base = _build_fx_model("nosoc")
    inner = _wrap_rescale(base, scale_by=2.0, shift_by=3.0)
    outer = _wrap_rescale(inner, scale_by=5.0, shift_by=7.0)
    sample = make_deploy_sample(outer)
    base_graph = expand_deploy_graph(base, sample)
    graph_module = expand_deploy_graph(outer, sample)
    eager_output = outer(cloned(_eager_input(sample)))

    _assert_outputs_close(graph_module, eager_output, sample)
    assert graph_module.num_rescale_layers == 2
    base_output = dict(zip(base_graph.output_signature, base_graph(*sample)))
    traced_output = dict(zip(graph_module.output_signature, graph_module(*sample)))
    expected_energy = (
        base_output[AtomicDataDict.TOTAL_ENERGY_KEY] * 2.0 + 3.0
    ) * 5.0 + 7.0
    torch.testing.assert_close(
        traced_output[AtomicDataDict.TOTAL_ENERGY_KEY],
        expected_energy,
        rtol=FX_RTOL,
        atol=FX_ATOL,
    )


def test_expand_deploy_graph_rejects_unknown_output_wrapper():
    class UnsupportedOutputWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.irreps_in = model.irreps_in
            self.irreps_out = model.irreps_out

        def forward(self, data):
            return self.model(data)

    model = UnsupportedOutputWrapper(_build_fx_model("nosoc")).eval()
    with pytest.raises(TypeError, match="UnsupportedOutputWrapper"):
        expand_deploy_graph(model, make_deploy_sample(model))


def test_strip_saved_tensor_detach_rebuilds_equivalent_graph():
    def with_detach(value):
        return value.detach().sin()

    sample = torch.linspace(-1.0, 1.0, 7)
    graph_module = make_fx(with_detach)(sample)
    assert any("detach" in str(node.target) for node in graph_module.graph.nodes)
    stripped = strip_saved_tensor_detach(graph_module)
    assert stripped is not graph_module
    assert not any("detach" in str(node.target) for node in stripped.graph.nodes)
    torch.testing.assert_close(stripped(sample), graph_module(sample))


def test_expand_rejects_noncontiguous_sample():
    model = _build_fx_model("nosoc")
    sample = list(make_deploy_sample(model))
    sample[0] = sample[0].transpose(0, 1)
    with pytest.raises(ValueError, match="contiguous"):
        expand_deploy_graph(model, tuple(sample))


def test_expand_rejects_decreasing_per_layer_cutoffs():
    model = _build_fx_model("nosoc")
    target = next(
        module
        for module in model.modules()
        if hasattr(module, "per_layer_cutoffs") and module.per_layer_cutoffs.numel() > 1
    )
    target.per_layer_cutoffs[-1] *= 0.5
    with pytest.raises(ValueError, match="decreasing per-layer cutoffs"):
        expand_deploy_graph(model, make_deploy_sample(model))


@pytest.mark.parametrize("case_name", ("nosoc", "soc"))
def test_source_dense_expand_does_not_use_export_only_edge_rewrites(
    monkeypatch, case_name
):
    model = _build_fx_model(case_name, edge_prune="dense")
    sample = make_deploy_sample(model)
    eager_output = model(cloned(_eager_input(sample)))

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("source dense graph used an export-only edge rewrite")

    monkeypatch.setattr(fx_compat, "_all_indices_nonzero", forbidden)
    monkeypatch.setattr(fx_compat, "_strip_deploy_edge_filters", forbidden)
    graph_module = expand_deploy_graph(model, sample)

    _assert_outputs_close(graph_module, eager_output, sample)
    targets = tuple(str(node.target) for node in graph_module.graph.nodes)
    assert all("nonzero" not in target for target in targets)
    assert all("index_copy" not in target for target in targets)
    assert all("index_add" not in target for target in targets)


def test_source_dense_expand_accepts_decreasing_per_layer_cutoffs():
    model = _build_fx_model("nosoc", edge_prune="dense")
    for module in model.modules():
        if (
            hasattr(module, "per_layer_cutoffs")
            and module.per_layer_cutoffs.numel() > 1
        ):
            module.per_layer_cutoffs[-1] *= 0.5
    sample = make_deploy_sample(model)
    eager_output = model(cloned(_eager_input(sample)))
    graph_module = expand_deploy_graph(model, sample)
    _assert_outputs_close(graph_module, eager_output, sample)
