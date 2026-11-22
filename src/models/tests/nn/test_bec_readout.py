"""CartesianBECReduce, TensorFeatureBECReadout and the TensorFeatureBEC builder.

``tensor_feature_bec_golden.npz`` was generated once, outside this repository,
from the reference implementation of this head (torch 2.12.1, e3nn 0.6.0,
nequip 0.18.0, nequip-allegro 0.8.3; gated convolution + hidden readout +
antisymmetric branch, mean reduction, hard ASR), with small widths and random
float32-representable weights, evaluated in float64 and float32.
``allegro083_contracter_golden.npz`` (next to the strided tests) holds the
``CartesianTensor`` change-of-basis matrices of e3nn 0.6.
"""

import json
import pathlib

import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn.io import CartesianTensor

from nequip.data import AtomicDataDict
from nequip.model import model_from_config
from nequip.nn import GraphModuleMixin, PerSpeciesScaleShift, SequentialGraphNetwork

from allegro.nn import (
    CartesianBECReduce,
    TensorFeatureBECReadout,
    hard_acoustic_sum_rule,
)

# e3nn scripts its generated modules on construction (torch>=2.10 deprecation).
pytestmark = [
    pytest.mark.filterwarnings("ignore:`torch.jit.*:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:The TorchScript type system doesn't support.*:UserWarning"
    ),
]

HERE = pathlib.Path(__file__).parent
HEAD_GOLDEN = HERE / "tensor_feature_bec_golden.npz"
BASIS_GOLDEN = HERE / "strided" / "allegro083_contracter_golden.npz"
SH = "0e+1o+2e+3o"
BORN = AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY
TOTAL = AtomicDataDict.TOTAL_BORN_EFFECTIVE_CHARGE_KEY


def _graph(n_edges=11, seed=0):
    """Two frames (3 + 4 atoms); atom 6 is the centre of no edge."""
    g = torch.Generator().manual_seed(seed)
    centre = torch.tensor([0, 0, 1, 2, 2, 3, 4, 5, 5, 3, 4])[:n_edges]
    neighbour = torch.tensor([1, 2, 0, 0, 1, 4, 3, 6, 3, 5, 6])[:n_edges]
    dtype = torch.get_default_dtype()
    return {
        AtomicDataDict.POSITIONS_KEY: torch.randn(7, 3, generator=g).to(dtype),
        AtomicDataDict.EDGE_INDEX_KEY: torch.stack([centre, neighbour]),
        AtomicDataDict.BATCH_KEY: torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([[0], [1], [2], [1], [0], [2], [1]]),
    }


def _randn(*shape, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(torch.get_default_dtype())


def _frame_mean(x, batch):
    out = torch.zeros(int(batch.max()) + 1, x.shape[1], dtype=x.dtype)
    out.index_add_(0, batch, x)
    return out / torch.bincount(batch).to(x.dtype).unsqueeze(-1)


def _to_cartesian(formula, x):
    return CartesianTensor(formula).to_cartesian(x).reshape(x.shape[0], 9)


# --------------------------------------------------------------------------
# CartesianBECReduce
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reduce", ["sum", "mean", "normalized_sum"])
def test_edge_to_atom_reductions(reduce):
    data = _graph()
    edge = _randn(11, 9)
    data["edge_bec"] = edge
    module = CartesianBECReduce(
        fields={"edge_bec": "cartesian"},
        reduce=reduce,
        avg_num_neighbors=4.0,
        irreps_in={"edge_bec": "0e+1e+2e"},
    )
    out = module(dict(data))[BORN]
    centre = data[AtomicDataDict.EDGE_INDEX_KEY][0]
    expected = torch.zeros(7, 9, dtype=edge.dtype).index_add_(0, centre, edge)
    if reduce == "mean":
        degree = torch.bincount(centre, minlength=7).clamp_min(1).to(edge.dtype)
        expected = expected / degree.unsqueeze(-1)
    elif reduce == "normalized_sum":
        expected = expected * 0.5
    torch.testing.assert_close(out, expected)
    assert torch.equal(out[6], torch.zeros(9, dtype=edge.dtype))  # degree 0


def test_irreps_bases_match_cartesian_tensor():
    data = _graph()
    data["sym"] = _randn(11, 9, seed=2)
    data["anti"] = _randn(11, 3, seed=3)
    module = CartesianBECReduce(
        fields={"sym": "ij", "anti": "ij=-ji"},
        reduce="mean",
        irreps_in={"sym": "0e+1e+2e", "anti": "1e"},
    )
    out = module(dict(data))[BORN]
    centre = data[AtomicDataDict.EDGE_INDEX_KEY][0]
    degree = torch.bincount(centre, minlength=7).clamp_min(1).unsqueeze(-1)

    def mean(x):
        return (
            torch.zeros(7, x.shape[1], dtype=x.dtype).index_add_(0, centre, x) / degree
        )

    expected = _to_cartesian("ij", mean(data["sym"])) + _to_cartesian(
        "ij=-ji", mean(data["anti"])
    )
    torch.testing.assert_close(out, expected)
    assert str(module.irreps_out[BORN]) == str(o3.Irreps(str(CartesianTensor("ij"))))


def test_change_of_basis_matches_e3nn_06():
    with np.load(BASIS_GOLDEN, allow_pickle=False) as golden:
        reference = {
            "ij": golden["cartesian/ij"],
            "ij=-ji": golden["cartesian/antisym"],
        }
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        module = CartesianBECReduce(
            fields={"a": "ij", "b": "ij=-ji"}, irreps_in={"a": "0e+1e+2e", "b": "1e"}
        )
    finally:
        torch.set_default_dtype(previous)
    for converter in module.converters:
        expected = reference[converter.basis].reshape(-1, 9)
        # CartesianTensor basis derivation can round differently by 1 ULP across CPUs and BLAS builds.
        np.testing.assert_allclose(
            converter.change_of_basis.numpy(), expected, rtol=0.0, atol=1e-15
        )
    # derived constants are not checkpointed
    assert list(module.state_dict()) == []


def test_soft_and_hard_asr():
    data = _graph()
    data["edge_bec"] = _randn(11, 9)
    batch = data[AtomicDataDict.BATCH_KEY]
    kwargs = dict(fields={"edge_bec": "cartesian"}, irreps_in={"edge_bec": "0e+1e+2e"})
    soft = CartesianBECReduce(asr="soft", pre_asr_out_field="bec_delta", **kwargs)
    soft_out = soft(dict(data))
    torch.testing.assert_close(soft_out[TOTAL], _frame_mean(soft_out[BORN], batch))
    assert torch.equal(soft_out["bec_delta"], soft_out[BORN])

    hard = CartesianBECReduce(asr="hard", pre_asr_out_field="bec_delta", **kwargs)
    hard_out = hard(dict(data))
    delta = hard_out["bec_delta"]
    assert torch.equal(delta, soft_out[BORN])
    torch.testing.assert_close(hard_out[BORN], delta - _frame_mean(delta, batch)[batch])
    tol = 1e-6 if delta.dtype == torch.float32 else 1e-12
    assert hard_out[TOTAL].abs().max() < tol
    assert set(hard.irreps_out) >= {BORN, TOTAL, "bec_delta"}

    hard_no_total = CartesianBECReduce(asr="hard", total_out_field=None, **kwargs)
    out = hard_no_total(dict(data))
    assert TOTAL not in out and not any(k.endswith("_frame_mean") for k in out)
    torch.testing.assert_close(out[BORN], hard_out[BORN])
    no_asr = CartesianBECReduce(asr="soft", total_out_field=None, **kwargs)
    assert set(no_asr(dict(data))) == set(data) | {BORN}


def test_hard_acoustic_sum_rule_helper():
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    bec = _randn(7, 9, seed=8).requires_grad_(True)
    out = hard_acoustic_sum_rule(bec, batch)
    torch.testing.assert_close(out, bec - _frame_mean(bec, batch)[batch])
    tol = 1e-6 if bec.dtype == torch.float32 else 1e-12
    assert _frame_mean(out, batch).abs().max() < tol  # every component, per structure
    # the (n_atoms, 3, 3) layout and an explicit (larger) frame count agree bitwise
    assert torch.equal(
        hard_acoustic_sum_rule(bec.view(-1, 3, 3), batch), out.view(-1, 3, 3)
    )
    assert torch.equal(hard_acoustic_sum_rule(bec, batch, num_frames=3), out)
    single = hard_acoustic_sum_rule(bec[:3], torch.zeros(3, dtype=torch.long))
    torch.testing.assert_close(single, bec[:3] - bec[:3].mean(0, keepdim=True))
    # linear projection: the gradient of sum(w * out) is w minus its frame mean
    weights = _randn(7, 9, seed=9)
    (grad,) = torch.autograd.grad((weights * out).sum(), bec)
    torch.testing.assert_close(grad, weights - _frame_mean(weights, batch)[batch])
    # TorchScript-compatible
    scripted = torch.jit.script(hard_acoustic_sum_rule)
    assert torch.equal(scripted(bec.detach(), batch), out.detach())

    # CartesianBECReduce(asr="hard") is this helper applied to the pre-ASR charges
    data = _graph()
    data["edge_bec"] = _randn(11, 9)
    module = CartesianBECReduce(
        fields={"edge_bec": "cartesian"},
        asr="hard",
        pre_asr_out_field="bec_delta",
        irreps_in={"edge_bec": "0e+1e+2e"},
    )
    result = module(dict(data))
    assert torch.equal(
        result[BORN],
        hard_acoustic_sum_rule(result["bec_delta"], data[AtomicDataDict.BATCH_KEY]),
    )


def test_per_species_scale_and_zero_shift_layout():
    data = _graph()
    data["edge_bec"] = _randn(11, 9)
    module = CartesianBECReduce(
        fields={"edge_bec": "cartesian"},
        reduce="sum",
        num_types=3,
        per_species_scale=[-1.0, 4.0, -2.0],
        per_species_scale_trainable=True,
        per_species_shift=0.0,
        irreps_in={"edge_bec": "0e+1e+2e"},
    )
    legacy = PerSpeciesScaleShift(
        field=BORN,
        num_types=3,
        type_names=["H", "C", "O"],
        shifts=0.0,
        scales=[-1.0, 4.0, -2.0],
        arguments_in_dataset_units=False,
        out_field=BORN,
        scales_trainable=True,
        shifts_trainable=False,
        irreps_in={BORN: "0e+1e+2e"},
    )
    # same state layout as PerSpeciesScaleShift: keys, order, values, trainability
    assert (
        list(module.state_dict()) == list(legacy.state_dict()) == ["scales", "shifts"]
    )
    for key, value in legacy.state_dict().items():
        assert torch.equal(module.state_dict()[key], value)
    assert [n for n, _ in module.named_parameters()] == ["scales"]
    plain = CartesianBECReduce(
        fields={"edge_bec": "cartesian"},
        reduce="sum",
        irreps_in={"edge_bec": "0e+1e+2e"},
    )
    reduced = plain(dict(data))[BORN]
    expected = legacy({**data, BORN: reduced})[BORN]
    assert torch.equal(module(dict(data))[BORN], expected)
    x = {"irreps_in": {"x": "0e+1e+2e"}}
    with pytest.raises(ValueError, match="must be zero"):
        CartesianBECReduce(
            fields={"x": "cartesian"}, num_types=3, per_species_shift=0.1, **x
        )
    with pytest.raises(ValueError, match="num_types"):
        CartesianBECReduce(fields={"x": "cartesian"}, per_species_scale=2.0, **x)


def test_reduce_validation():
    x = {"irreps_in": {"x": "0e+1e+2e"}}
    with pytest.raises(ValueError, match="reduce"):
        CartesianBECReduce(fields={"x": "cartesian"}, reduce="max", **x)
    with pytest.raises(ValueError, match="asr"):
        CartesianBECReduce(fields={"x": "cartesian"}, asr="exact", **x)
    with pytest.raises(ValueError, match="avg_num_neighbors"):
        CartesianBECReduce(fields={"x": "cartesian"}, reduce="normalized_sum", **x)
    with pytest.raises(ValueError, match="dimension"):
        CartesianBECReduce(fields={"x": "ij=-ji"}, **x)
    with pytest.raises(ValueError, match="requires field"):
        CartesianBECReduce(fields={"x": "cartesian"})


def _rotation_pairs():
    R = o3.rand_matrix(dtype=torch.float64)
    return [R, -R]


@pytest.mark.parametrize("asr", ["soft", "hard"])
def test_reduce_equivariance(asr):
    data = _graph()
    data["sym"] = _randn(11, 9, seed=4)
    data["anti"] = _randn(11, 3, seed=5)
    module = CartesianBECReduce(
        fields={"sym": "ij", "anti": "ij=-ji"},
        asr=asr,
        irreps_in={"sym": "0e+1e+2e", "anti": "1e"},
    )
    out = module(dict(data))[BORN].reshape(-1, 3, 3)
    tol = 1e-5 if out.dtype == torch.float32 else 1e-10
    for R in _rotation_pairs():
        rotated = dict(data)
        rotated["sym"] = (
            data["sym"] @ o3.Irreps("0e+1e+2e").D_from_matrix(R).to(out.dtype).T
        )
        rotated["anti"] = (
            data["anti"] @ o3.Irreps("1e").D_from_matrix(R).to(out.dtype).T
        )
        Rt = R.to(out.dtype)
        expected = Rt @ out @ Rt.T  # rank-2 polar-polar tensor: even under inversion
        torch.testing.assert_close(
            module(rotated)[BORN].reshape(-1, 3, 3), expected, atol=tol, rtol=tol
        )


@pytest.mark.parametrize("asr", ["soft", "hard"])
def test_reduce_torchscript(asr):
    data = _graph()
    data["sym"] = _randn(11, 9, seed=6)
    data["edge_bec"] = _randn(11, 9, seed=7)
    for fields, irreps in (
        ({"edge_bec": "cartesian"}, {"edge_bec": "0e+1e+2e"}),
        ({"sym": "ij"}, {"sym": "0e+1e+2e"}),
    ):
        module = CartesianBECReduce(
            fields=fields,
            asr=asr,
            reduce="normalized_sum",
            avg_num_neighbors=3.0,
            num_types=3,
            per_species_scale=[1.0, 2.0, 3.0],
            per_species_shift=0.0,
            irreps_in=irreps,
        )
        scripted = torch.jit.script(module)
        expected = module(dict(data))
        actual = scripted(dict(data))
        for key in (BORN, TOTAL):
            assert torch.equal(actual[key], expected[key])


# --------------------------------------------------------------------------
# TensorFeatureBECReadout
# --------------------------------------------------------------------------

F, H, A, S = 4, 3, 2, 8
READOUT_IRREPS = {"bb_tensor": SH, "bb_scalars": f"{S}x0e", "bb_basis": SH}


def _readout(**kwargs):
    kwargs.setdefault("tensor_field", "bb_tensor")
    kwargs.setdefault("scalar_field", "bb_scalars")
    kwargs.setdefault("tensor_basis_field", "bb_basis")
    kwargs.setdefault("num_tensor_features", F)
    kwargs.setdefault("hidden_channels", H)
    kwargs.setdefault("antisymmetric_channels", A)
    kwargs.setdefault("irreps_in", dict(READOUT_IRREPS))
    return TensorFeatureBECReadout(**kwargs)


def _readout_inputs(n_edges=11, seed=10):
    data = _graph(n_edges=n_edges)
    data["bb_tensor"] = _randn(n_edges, F, 16, seed=seed)
    data["bb_scalars"] = _randn(n_edges, S, seed=seed + 1)
    data["bb_basis"] = _randn(n_edges, 16, seed=seed + 2)
    return data


def _rename_reference_key(key):
    return key.replace("tensor_convolution.tensor_convolution.", "tensor_convolution.")


def test_readout_matches_reference_head():
    with np.load(HEAD_GOLDEN, allow_pickle=False) as golden:
        arrays = {k: golden[k] for k in golden.files}
    meta = json.loads(str(arrays.pop("meta")))
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        readout = _readout(
            num_tensor_features=meta["num_tensor_features"],
            hidden_channels=meta["hidden_channels"],
            antisymmetric_channels=meta["antisymmetric_channels"],
            reduce="mean",
            asr="hard",
            pre_asr_out_field="bec_delta",
        )
    finally:
        torch.set_default_dtype(previous)

    # the intended weight-name mapping from the reference head is a single rename;
    # SpinGNN's MakeWeightedChannels additionally keeps its index buffer
    ours = set(readout.state_dict())
    mapped = {_rename_reference_key(k) for k in meta["state_keys"]}
    assert ours == mapped | {"tensor_convolution.env_weighter._w_index"}
    state = {
        _rename_reference_key(k[len("state/") :]): torch.as_tensor(v)
        for k, v in arrays.items()
        if k.startswith("state/")
    }
    assert set(state) == {n for n, _ in readout.named_parameters()}
    readout.load_state_dict(state, strict=False)

    edge_index = torch.as_tensor(arrays["inputs/edge_index"])
    batch = torch.as_tensor(arrays["inputs/batch"])

    def run(dtype):
        data = {
            AtomicDataDict.POSITIONS_KEY: torch.zeros(len(batch), 3, dtype=dtype),
            AtomicDataDict.EDGE_INDEX_KEY: edge_index,
            AtomicDataDict.BATCH_KEY: batch,
            "bb_tensor": torch.as_tensor(arrays["inputs/tensor_features"]).to(dtype),
            "bb_scalars": torch.as_tensor(arrays["inputs/edge_features"]).to(dtype),
            "bb_basis": torch.as_tensor(arrays["inputs/edge_attrs"]).to(dtype),
        }
        return readout(data)

    out64 = run(torch.float64)
    np.testing.assert_allclose(
        out64[BORN].detach().numpy(), arrays["out_float64"], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        out64["bec_delta"].detach().numpy(), arrays["delta_float64"], rtol=0, atol=1e-12
    )
    readout.to(torch.float32)
    out32 = run(torch.float32)
    assert out32[BORN].dtype == torch.float32
    np.testing.assert_allclose(
        out32[BORN].detach().numpy(), arrays["out_float32"], rtol=0, atol=1e-6
    )


def test_readout_zero_init_and_options():
    data = _readout_inputs()
    readout = _readout(pre_asr_out_field="bec_delta")
    out = readout(dict(data))
    assert torch.equal(out[BORN], torch.zeros_like(out[BORN]))
    assert torch.equal(out["bec_delta"], torch.zeros_like(out["bec_delta"]))
    assert out["edge_bec_symmetric"].shape == (11, 9)
    assert out["edge_bec_antisymmetric"].shape == (11, 3)
    assert float(readout.tensor_convolution.residual_gate.detach()) == 0.0

    minimal = _readout(
        gated_convolution=False,
        hidden_channels=None,
        antisymmetric_channels=None,
        zero_init_final_linear=False,
        irreps_in={"bb_tensor": SH},
    )
    names = {n.split(".")[0] for n, _ in minimal.named_parameters()}
    assert names == {"tensor_product", "channel_weights", "final_linear"}
    out = minimal(
        {k: v for k, v in data.items() if k not in ("bb_scalars", "bb_basis")}
    )
    assert "edge_bec_antisymmetric" not in out
    assert out[BORN].abs().max() > 0
    with pytest.raises(ValueError, match="irreps of field"):
        _readout(irreps_in={"bb_tensor": SH})
    with pytest.raises(ValueError, match="multiplicity 1"):
        _readout(irreps_in={**READOUT_IRREPS, "bb_tensor": f"{F}x0e+{F}x1o"})
    with pytest.raises(ValueError, match="shape"):
        readout({**data, "bb_tensor": data["bb_tensor"][:, :2]})


def _randomize(module, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(0.5 * torch.randn(param.shape, generator=g).to(param.dtype))


@pytest.mark.parametrize("asr", ["soft", "hard"])
def test_readout_equivariance_and_asr(asr):
    data = _readout_inputs()
    readout = _readout(asr=asr)
    _randomize(readout)
    out = readout(dict(data))[BORN].reshape(-1, 3, 3)
    assert out.abs().max() > 1e-3
    tol = 1e-4 if out.dtype == torch.float32 else 1e-9
    if asr == "hard":
        sums = _frame_mean(out.reshape(-1, 9), data[AtomicDataDict.BATCH_KEY])
        assert sums.abs().max() < tol
    for R in _rotation_pairs():
        D = o3.Irreps(SH).D_from_matrix(R).to(out.dtype)
        rotated = dict(data)
        rotated["bb_tensor"] = data["bb_tensor"] @ D.T
        rotated["bb_basis"] = data["bb_basis"] @ D.T
        Rt = R.to(out.dtype)
        torch.testing.assert_close(
            readout(rotated)[BORN].reshape(-1, 3, 3),
            Rt @ out @ Rt.T,
            atol=tol,
            rtol=tol,
        )


class _FeatureSource(GraphModuleMixin, torch.nn.Module):
    """Stand-in backbone writing fixed tensor features (for the builder test)."""

    def __init__(self):
        super().__init__()
        self._init_irreps(irreps_out=dict(READOUT_IRREPS))

    def forward(self, data):
        n_edges = data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        source = _readout_inputs(n_edges=n_edges)
        for key in READOUT_IRREPS:
            data[key] = source[key].to(data[AtomicDataDict.POSITIONS_KEY].dtype)
        return data


def _feature_source_builder(config):
    return SequentialGraphNetwork({"features": _FeatureSource()})


def _builder_config(**extra):
    config = {
        "model_builders": [_feature_source_builder, "allegro.model.TensorFeatureBEC"],
        "num_types": 3,
        "type_names": ["H", "C", "O"],
        "bec_readout_tensor_field": "bb_tensor",
        "bec_readout_scalar_field": "bb_scalars",
        "bec_readout_tensor_basis_field": "bb_basis",
        "bec_readout_num_tensor_features": F,
        "bec_readout_hidden_channels": H,
        "bec_readout_antisymmetric_channels": A,
        "bec_readout_pre_asr_out_field": "bec_delta",
    }
    config.update(extra)
    return config


def test_builder_appends_readout():
    model = model_from_config(_builder_config())
    assert list(dict(model.named_children())) == ["features", "bec_readout"]
    readout = model.bec_readout
    assert isinstance(readout, TensorFeatureBECReadout)
    assert readout.hidden_channels == H and readout.antisymmetric_channels == A
    out = model(_graph())
    assert {BORN, TOTAL, "bec_delta"} <= set(out) and {BORN, "bec_delta"} <= set(
        model.irreps_out
    )


def test_builder_construct_dtype():
    default = torch.get_default_dtype()
    torch.manual_seed(5)
    model = model_from_config(_builder_config(bec_readout_construct_dtype="float64"))
    state = model.bec_readout.state_dict()
    assert all(
        v.dtype == default for v in state.values() if v.is_floating_point()
    ), "cast back to the build dtype"

    # identical to constructing under float64 and casting by hand
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(5)
        reference = model_from_config(_builder_config()).bec_readout
    finally:
        torch.set_default_dtype(previous)
    reference.to(default)
    for key, value in reference.state_dict().items():
        assert torch.equal(state[key], value), key
