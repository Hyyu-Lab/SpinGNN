"""Spin edge harmonics retain finite higher derivatives and dynamic graph support."""

import io

import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.nn.embedding import (
    SphericalHarmonicEdgeAttrs_spin_nosoc,
    SphericalHarmonicEdgeAttrs_spin_soc,
    SphericalHarmonicEdgeAttrs_spin_soc_simple,
)


MODULES = {
    "nosoc": SphericalHarmonicEdgeAttrs_spin_nosoc,
    "simple_soc": SphericalHarmonicEdgeAttrs_spin_soc_simple,
    "full_soc": SphericalHarmonicEdgeAttrs_spin_soc,
}


@pytest.fixture(autouse=True)
def _stable_jit_execution():
    # Cold ScriptFunction profiling otherwise changes derivative reduction order.
    with torch.jit.optimized_execution(False):
        yield


def _module(kind, lmax=2, normalization="component", symmetry=True):
    options = dict(
        l_max=lmax,
        edge_sh_normalization=normalization,
        edge_sh_normalize=True,
        time_reversal=symmetry,
    )
    if kind != "nosoc":
        options["parity"] = symmetry
    module = MODULES[kind](**options)
    assert module.sh_si.normalize is False
    return module


def _data(case):
    spin = torch.tensor(
        [[0.3, -0.4, 0.9], [-0.7, 0.8, 0.1], [1.4, 0.2, -0.6], [0.0, 0.0, 0.0]]
    )
    positions = torch.tensor(
        [[0.2, 0.5, -0.1], [1.2, 0.1, 0.8], [-0.6, 1.8, 0.4], [3.0, 0.2, 2.0]]
    )
    edges = torch.tensor([[0, 1, 2, 0, 2, 1], [1, 2, 0, 2, 1, 0]])
    if case == "empty_nodes":
        spin, positions, edges = spin[:0], positions[:0], edges[:, :0]
    elif case == "empty_edges":
        edges = edges[:, :0]
    elif case == "duplicate_edges":
        edges = edges.repeat(1, 5)
    elif case == "nonmagnetic_node":
        edges = torch.tensor([[0, 3, 3, 2], [3, 0, 2, 3]])
    elif case == "self_edge":
        edges = torch.tensor([[0, 1], [0, 1]])
    elif case == "all_zero":
        spin.zero_()
    elif case == "extra_nodes":
        spin = torch.cat([spin, torch.tensor([[0.2, -0.5, 0.1]])])
        positions = torch.cat([positions, torch.tensor([[3.0, -0.5, 1.0]])])
    else:
        assert case == "isolated_zero"
    return {
        AtomicDataDict.SPIN_KEY: spin,
        AtomicDataDict.SPIN_NONLEAF_KEY: spin,
        AtomicDataDict.POSITIONS_KEY: positions,
        AtomicDataDict.EDGE_INDEX_KEY: edges,
    }


def _edge_evaluated_reference(module, kind, data):
    """The original edge endpoint expressions, including full SOC's row convention."""
    spin = data[AtomicDataDict.SPIN_NONLEAF_KEY]
    row, col = data[AtomicDataDict.EDGE_INDEX_KEY]
    if kind == "nosoc":
        parts = [module.sh_si(spin[col]), module.sh_si(spin[row])]
    else:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
        vectors = data[AtomicDataDict.EDGE_VECTORS_KEY]
        if kind == "simple_soc":
            parts = [
                module.sh_rij(vectors),
                module.sh_si(spin[col]),
                module.sh_si(spin[row]),
            ]
        else:
            cross = torch.cross(spin[row], spin[col], dim=-1)
            parts = [
                module.sh_rij(vectors),
                module.sh_si_sj(cross),
                module.sh_si(spin[row]),
                module.sh_rij_si(torch.cross(vectors, spin[row], dim=-1)),
                (vectors * spin[row]).sum(dim=-1, keepdim=True),
                module.sh_rij_si_sj_vector(torch.cross(vectors, cross, dim=-1)),
                (vectors * cross).sum(dim=-1, keepdim=True),
            ]
    result = torch.cat(parts, dim=-1)[:, module.sort_list]
    if kind == "full_soc":
        result[:, :7] = result[:, module.scalar_sort]
    return result


def _values_and_derivatives(function, data):
    spin = data[AtomicDataDict.SPIN_NONLEAF_KEY].clone().requires_grad_()
    positions = data[AtomicDataDict.POSITIONS_KEY].clone().requires_grad_()
    inputs = dict(data)
    inputs[AtomicDataDict.SPIN_KEY] = spin
    inputs[AtomicDataDict.SPIN_NONLEAF_KEY] = spin
    inputs[AtomicDataDict.POSITIONS_KEY] = positions
    attributes = function(inputs)
    weight = torch.linspace(-0.16, 0.21, attributes.shape[-1]).requires_grad_()
    unused = torch.tensor(0.7, requires_grad=True)
    projection = attributes @ weight
    energy = (torch.nn.functional.softplus(projection) + 0.03 * projection.sin()).sum()
    parameters = (spin, positions, weight, unused)
    first = torch.autograd.grad(
        energy, parameters, create_graph=True, allow_unused=True
    )
    loss = 0.07 * energy.square()
    for gradient in first[:2]:
        if gradient is not None:
            loss = loss + 0.03 * gradient.square().sum()
    second = torch.autograd.grad(loss, parameters, allow_unused=True)
    assert first[-1] is None and second[-1] is None
    return (attributes, energy, *first, *second)


def _assert_values_match(actual, expected, tolerance):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert (left is None) == (right is None)
        if left is None:
            continue
        assert left.shape == right.shape
        assert left.dtype == right.dtype and left.device == right.device
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        assert torch.allclose(left, right, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("kind", MODULES)
@pytest.mark.parametrize("lmax", [0, 1, 3])
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize(
    "case",
    [
        "empty_nodes",
        "empty_edges",
        "isolated_zero",
        "duplicate_edges",
        "nonmagnetic_node",
        "all_zero",
        "self_edge",
    ],
)
def test_spin_embeddings_match_edge_outputs_and_force_gradients(
    kind, lmax, normalization, case, float_tolerance
):
    module = _module(kind, lmax=lmax, normalization=normalization)
    data = _data(case)
    actual = _values_and_derivatives(
        lambda value: module(value)[AtomicDataDict.EDGE_ATTRS_KEY], data
    )
    expected = _values_and_derivatives(
        lambda value: _edge_evaluated_reference(module, kind, value), data
    )
    _assert_values_match(actual, expected, float_tolerance)


@pytest.mark.parametrize("kind", MODULES)
@pytest.mark.parametrize("symmetry", [True, False])
def test_script_roundtrip_handles_dynamic_graph_sizes(kind, symmetry, float_tolerance):
    module = _module(kind, symmetry=symmetry)
    scripted = torch.jit.script(module)
    stream = io.BytesIO()
    torch.jit.save(scripted, stream)
    stream.seek(0)
    restored = torch.jit.load(stream)
    assert set(restored.state_dict()) == set(module.state_dict())
    for case in [
        "empty_nodes",
        "isolated_zero",
        "empty_edges",
        "duplicate_edges",
        "extra_nodes",
    ]:
        data = _data(case)
        expected = _values_and_derivatives(
            lambda value: _edge_evaluated_reference(module, kind, value), data
        )
        actual = _values_and_derivatives(
            lambda value: restored(value)[AtomicDataDict.EDGE_ATTRS_KEY], data
        )
        _assert_values_match(actual, expected, float_tolerance)
