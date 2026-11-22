"""GatherContracter: golden tensors from nequip-allegro 0.8.3 and equivariance.

The fixture ``allegro083_contracter_golden.npz`` was generated once, outside
this repository, with torch 2.12.1, e3nn 0.6.0, nequip 0.18.0 and
nequip-allegro 0.8.3 (``allegro.nn._strided.Contracter`` and
``MakeWeightedChannels``, built under float64 and evaluated in float64 and,
after ``.to(float32)``, in float32). Inputs and weights are
float32-representable. The w3j buffers are pinned by float64 checksums.
"""

import json
import pathlib

import numpy as np
import pytest
import torch
from e3nn import o3

from allegro.nn._strided import GatherContracter, MakeWeightedChannels

pytestmark = pytest.mark.filterwarnings("ignore:`torch.jit.*:DeprecationWarning")

GOLDEN = pathlib.Path(__file__).with_name("allegro083_contracter_golden.npz")
SH = "0e+1o+2e+3o"
CASES = {"lift": "0e+1e+2e", "conv": SH, "anti": "1e"}


@pytest.fixture(scope="module")
def golden():
    with np.load(GOLDEN, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(str(arrays.pop("meta")))
    return arrays, meta


def _with_default_dtype(dtype, fn):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        return fn()
    finally:
        torch.set_default_dtype(previous)


def _build(case, meta):
    return GatherContracter(
        irreps_in1=SH,
        irreps_in2=SH,
        irreps_out=CASES[case],
        mul=meta[case]["mul"],
        path_channel_coupling=True,
    )


def _checksums(w3j):
    w = w3j.detach().double().flatten()
    ramp = torch.linspace(-1.0, 1.0, w.numel(), dtype=torch.float64)
    return [float(w.sum()), float((w * w).sum()), float((w * ramp).sum())]


@pytest.mark.parametrize("case", sorted(CASES))
def test_structure_matches_allegro_083(golden, case):
    _, meta = golden
    module = _with_default_dtype(torch.float64, lambda: _build(case, meta))
    assert module.num_paths == meta[case]["num_paths"]
    assert [list(i) for i in module.instructions] == meta[case]["instructions"]
    assert list(module.w3j.shape) == meta[case]["w3j_shape"]
    assert module.w3j_is_ij_diagonal == meta[case]["w3j_is_ij_diagonal"]
    assert tuple(module.weights.shape) == (meta[case]["mul"], module.num_paths)
    # T-e3nn Wigner 3j (real basis, signs) agrees with e3nn 0.6
    np.testing.assert_allclose(
        _checksums(module.w3j), meta[case]["w3j_checksums_float64"], rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_outputs_match_allegro_083(golden, case):
    arrays, meta = golden
    # build in float64 (as the original head was), then evaluate in both dtypes
    module = _with_default_dtype(torch.float64, lambda: _build(case, meta))
    with torch.no_grad():
        module.weights.copy_(torch.as_tensor(arrays[f"{case}/weights"]))
    x1 = torch.as_tensor(arrays[f"{case}/x1"])
    x2 = torch.as_tensor(arrays[f"{case}/x2"]) if f"{case}/x2" in arrays else x1
    idxs = torch.as_tensor(arrays[f"{case}/idxs"])

    out64 = module(x1.double(), x2.double(), idxs)
    np.testing.assert_allclose(
        out64.detach().numpy(), arrays[f"{case}/out_float64"], rtol=0, atol=1e-12
    )
    module.to(torch.float32)
    out32 = module(x1, x2, idxs)
    assert out32.dtype == torch.float32
    np.testing.assert_allclose(
        out32.detach().numpy(), arrays[f"{case}/out_float32"], rtol=0, atol=1e-6
    )


def test_weighted_channels_match_allegro_083(golden):
    arrays, meta = golden
    weighter = MakeWeightedChannels(
        irreps_in=o3.Irreps(SH), multiplicity_out=meta["mwc"]["multiplicity_out"]
    )
    assert weighter.weight_numel == meta["mwc"]["weight_numel"]
    edge_attr = torch.as_tensor(arrays["mwc/edge_attr"])
    weights = torch.as_tensor(arrays["mwc/weights"])
    out64 = weighter.to(torch.float64)(edge_attr.double(), weights.double())
    np.testing.assert_allclose(
        out64.numpy(), arrays["mwc/out_float64"], rtol=0, atol=1e-12
    )
    out32 = weighter.to(torch.float32)(edge_attr, weights)
    np.testing.assert_allclose(
        out32.numpy(), arrays["mwc/out_float32"], rtol=0, atol=1e-6
    )


def _block_rotate(x, D):
    return x @ D.to(x.dtype).T


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("coupling", [True, False])
def test_gather_semantics_and_equivariance(case, coupling):
    torch.manual_seed(0)
    module = GatherContracter(
        irreps_in1=SH,
        irreps_in2=SH,
        irreps_out=CASES[case],
        mul=3,
        path_channel_coupling=coupling,
    )
    dtype = torch.get_default_dtype()
    x1 = torch.randn(5, 3, 16, dtype=dtype)
    x2 = torch.randn(2, 3, 16, dtype=dtype)
    idxs = torch.tensor([1, 0, 1, 1, 0])
    out = module(x1, x2, idxs)
    assert out.shape == (5, 3, o3.Irreps(CASES[case]).dim)
    # gather-only: identical to an explicit per-edge copy of x2, no scatter
    torch.testing.assert_close(out, module(x1, x2[idxs], torch.arange(5)))

    # O(3) equivariance (rotation with and without inversion)
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    for sign in (1.0, -1.0):
        R = sign * o3.rand_matrix(dtype=torch.float64)
        D_in = o3.Irreps(SH).D_from_matrix(R)
        D_out = o3.Irreps(CASES[case]).D_from_matrix(R)
        rotated = module(_block_rotate(x1, D_in), _block_rotate(x2, D_in), idxs)
        torch.testing.assert_close(
            rotated, _block_rotate(out, D_out), atol=tol, rtol=tol
        )


def test_single_path_and_validation():
    module = GatherContracter("1o", "1o", "0e", mul=2)
    assert module.num_paths == 1 and tuple(module.weights.shape) == (2,)
    assert module.w3j_is_ij_diagonal
    x = torch.randn(4, 2, 3)
    out = module(x, x, torch.arange(4))
    # 1o x 1o -> 0e is a weighted dot product, sqrt(1) * w3j = 1/sqrt(3)
    expected = module.weights.view(1, 2, 1) * (x * x).sum(-1, keepdim=True) / 3**0.5
    torch.testing.assert_close(out, expected)
    with pytest.raises(ValueError, match="multiplicity 1"):
        GatherContracter("2x0e", "0e", "0e", mul=2)
    with pytest.raises(ValueError, match="No tensor-product paths"):
        GatherContracter("1o", "1o", "1o", mul=2)
    with pytest.raises(ValueError, match="not allowed"):
        GatherContracter("1o", "1o", "1o", mul=2, instructions=[(0, 0, 0)])


def test_torchscript():
    module = GatherContracter(SH, SH, "0e+1e+2e", mul=2)
    scripted = torch.jit.script(module)
    x = torch.randn(3, 2, 16)
    idxs = torch.tensor([0, 2, 1])
    torch.testing.assert_close(scripted(x, x, idxs), module(x, x, idxs))
