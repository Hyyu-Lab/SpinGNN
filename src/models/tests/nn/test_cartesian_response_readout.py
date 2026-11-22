"""Cartesian response contracts against independent tensor transformations."""

import inspect
import json
from pathlib import Path

import numpy as np

import pytest
import torch
from e3nn import o3
from e3nn.io import CartesianTensor

from allegro.nn import (
    CartesianBECReduce,
    CartesianTensorReduce,
    TensorFeatureBECReadout,
    TensorFeatureCartesianReadout,
)
from allegro.nn._bec_readout import (
    _ambient_irreps,
    _cartesian_irreps,
    _change_of_basis,
    _default_dtype,
    _lift_irreps,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore:`torch.jit.*:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:The TorchScript type system doesn't support.*:UserWarning"
    ),
]

SH = "0e+1o+2e+3o"
IRREPS = {"tensor": SH, "scalar": "4x0e", "basis": SH}


def graph():
    return {
        "pos": torch.randn(5, 3),
        "edge_index": torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 0, 0, 4, 3]]),
        "batch": torch.tensor([0, 0, 0, 1, 1]),
        "atom_types": torch.tensor([[0], [1], [0], [1], [0]]),
        "tensor": torch.randn(6, 2, 16),
        "scalar": torch.randn(6, 4),
        "basis": torch.randn(6, 16),
    }


def rotate_cartesian(x, matrix, rank):
    if rank == 2:
        return torch.einsum(
            "ia,bac,jc->bij", matrix, x.reshape(-1, 3, 3), matrix
        ).flatten(1)
    return torch.einsum(
        "ia,jc,kd,bacd->bijk", matrix, matrix, matrix, x.reshape(-1, 3, 3, 3)
    ).flatten(1)


def rotate_graph(data, matrix):
    rotated = dict(data)
    d = o3.Irreps(SH).D_from_matrix(matrix)
    rotated["tensor"] = data["tensor"] @ d.T
    rotated["basis"] = data["basis"] @ d.T
    rotated["pos"] = data["pos"] @ matrix.T
    if "cell" in data:
        rotated["cell"] = data["cell"] @ matrix.T
    return rotated


def readout(formula, **kwargs):
    options = dict(
        cartesian_formula=formula,
        out_field="out",
        total_out_field="total",
        total_reduce="sum",
        tensor_field="tensor",
        scalar_field="scalar",
        tensor_basis_field="basis",
        num_tensor_features=2,
        hidden_channels=2,
        gate_init=0.3,
        irreps_in=IRREPS,
    )
    options.update(kwargs)
    return TensorFeatureCartesianReadout(**options)


def randomize(module):
    with torch.no_grad():
        for p in module.parameters():
            p.uniform_(-0.4, 0.4)


@pytest.mark.parametrize("formula", ["ij", "ij=ji", "ij=-ji", "ijk=ikj", "ijk"])
def test_basis_against_e3nn(formula):
    assert _cartesian_irreps(formula) == CartesianTensor(formula)
    with _default_dtype(torch.float64):
        expected = (
            CartesianTensor(formula)
            .reduced_tensor_products()
            .change_of_basis.flatten(1)
        )
        q = _change_of_basis(formula)
        assert torch.equal(q, expected)
        torch.testing.assert_close(
            q @ q.T, torch.eye(q.shape[0]), atol=1e-14, rtol=1e-14
        )
        coefficients = torch.randn(4, q.shape[0])
        torch.testing.assert_close(
            coefficients @ q @ q.T, coefficients, atol=1e-14, rtol=1e-14
        )
        q.zero_()
        assert torch.equal(_change_of_basis(formula), expected)
    expected_lifts = {
        "ij": "0e+1e+2e",
        "ij=ji": "0e+2e",
        "ij=-ji": "1e",
        "ijk=ikj": "1o+1o+2o+3o",
        "ijk": "0o+1o+1o+1o+2o+2o+3o",
    }
    assert list(_lift_irreps(formula)) == list(o3.Irreps(expected_lifts[formula]))


@pytest.mark.parametrize("formula", ["ij", "ij=ji", "ijk=ikj"])
@pytest.mark.parametrize("memory_efficient", [False, True])
def test_readout_equivariance_symmetry_and_gradients(formula, memory_efficient):
    torch.manual_seed(28)
    module = readout(
        formula,
        antisymmetric_channels=2 if formula == "ij" else None,
        memory_efficient_contraction=memory_efficient,
    )
    data = graph()
    initial = module(dict(data))["out"]
    assert torch.count_nonzero(initial) == 0
    initial.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()
    )
    randomize(module)
    original = module(dict(data))["out"]
    assert original.abs().max() > 1e-8
    rank = 3 if formula == "ijk=ikj" else 2
    if formula != "ij":
        shaped = original.reshape((-1,) + (3,) * rank)
        assert torch.equal(shaped, shaped.transpose(-1, -2))
    for matrix in (o3.rand_matrix(), -o3.rand_matrix()):
        actual = module(rotate_graph(data, matrix))["out"]
        torch.testing.assert_close(
            actual,
            rotate_cartesian(original, matrix, rank),
            atol=3e-5 if original.dtype == torch.float32 else 1e-10,
            rtol=3e-5 if original.dtype == torch.float32 else 1e-10,
        )


def test_validation_does_not_consume_rng():
    for options, match in [
        ({"antisymmetric_channels": 2}, "antisymmetric"),
        ({"irreps_in": dict(IRREPS, tensor="0e+1e+2e")}, "path"),
    ]:
        torch.manual_seed(42)
        before = torch.get_rng_state().clone()
        with pytest.raises(ValueError, match=match):
            readout("ijk=ikj", **options)
        assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match="cartesian_formula"):
        readout("ijk")


@pytest.mark.parametrize("cartesian", [False, True])
def test_rank3_reduce(cartesian):
    basis = "cartesian" if cartesian else "ijk=ikj"
    width = 27 if cartesian else 18
    irreps = _ambient_irreps(3) if cartesian else _cartesian_irreps(basis)
    options = dict(
        fields={"edge": basis},
        out_field="out",
        cartesian_rank=3,
        total_out_field="total",
        total_reduce="sum",
        reduce="sum",
        num_types=2,
        per_species_scale=[2.0, 3.0],
        irreps_in={"edge": irreps},
    )
    module = CartesianTensorReduce(**options)
    data = graph()
    data["edge"] = torch.randn(6, width)
    out = module(dict(data))
    assert out["out"].shape == (5, 27)
    torch.testing.assert_close(
        out["total"], torch.stack([out["out"][:3].sum(0), out["out"][3:].sum(0)])
    )
    expected = torch.zeros(5, 27)
    edge_cart = (
        data["edge"]
        if cartesian
        else CartesianTensor(basis).to_cartesian(data["edge"]).flatten(1)
    )
    expected.index_add_(0, data["edge_index"][0], edge_cart)
    expected *= torch.tensor([2.0, 3.0, 2.0, 3.0, 2.0])[:, None]
    torch.testing.assert_close(out["out"], expected)
    scripted = torch.jit.script(module)
    assert torch.equal(scripted(dict(data))["out"], out["out"])
    soft = CartesianTensorReduce(**dict(options, asr="soft", total_reduce="mean"))
    assert torch.equal(soft(dict(data))["out"], out["out"])
    for matrix in (o3.rand_matrix(), -o3.rand_matrix()):
        rotated = dict(data)
        rotated["edge"] = (
            rotate_cartesian(data["edge"], matrix, 3)
            if cartesian
            else data["edge"] @ irreps.D_from_matrix(matrix).T
        )
        torch.testing.assert_close(
            module(rotated)["out"],
            rotate_cartesian(out["out"], matrix, 3),
            atol=2e-5,
            rtol=2e-5,
        )


@pytest.mark.parametrize(
    "options,match",
    [
        ({"cartesian_rank": 2}, "rank"),
        ({"total_reduce": "max"}, "total_reduce"),
        ({"asr": "soft", "total_reduce": "sum"}, "soft"),
        ({"irreps_in": {"edge": "9x0e"}}, "dimension"),
    ],
)
def test_reduce_errors(options, match):
    params = dict(
        fields={"edge": "ijk=ikj"},
        out_field="out",
        cartesian_rank=3,
        irreps_in={"edge": "2x1o+2o+3o"},
    )
    params.update(options)
    with pytest.raises(ValueError, match=match):
        CartesianTensorReduce(**params)


def test_bec_subclass_bitwise_and_signature():
    options = dict(
        tensor_field="tensor",
        scalar_field="scalar",
        tensor_basis_field="basis",
        num_tensor_features=2,
        hidden_channels=2,
        antisymmetric_channels=2,
        irreps_in=IRREPS,
        zero_init_final_linear=False,
    )
    torch.manual_seed(19)
    bec = TensorFeatureBECReadout(**options)
    state_rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    generic = TensorFeatureCartesianReadout(
        cartesian_formula="ij",
        out_field="born_effective_charges",
        total_out_field="total_born_effective_charges",
        total_reduce="mean",
        asr="hard",
        edge_symmetric_field="edge_bec_symmetric",
        edge_antisymmetric_field="edge_bec_antisymmetric",
        **options,
    )
    assert torch.equal(state_rng, torch.get_rng_state())
    assert list(bec.state_dict()) == list(generic.state_dict())
    for key, value in bec.state_dict().items():
        assert torch.equal(value, generic.state_dict()[key])
    data = graph()
    assert torch.equal(
        bec(dict(data))["born_effective_charges"],
        generic(dict(data))["born_effective_charges"],
    )
    assert not TensorFeatureBECReadout.__dict__.get("__annotations__")
    assert not CartesianBECReduce.__dict__.get("__annotations__")
    params = inspect.signature(TensorFeatureBECReadout).parameters
    assert list(params) == [
        "tensor_field",
        "num_tensor_features",
        "scalar_field",
        "tensor_basis_field",
        "out_field",
        "total_out_field",
        "pre_asr_out_field",
        "edge_symmetric_field",
        "edge_antisymmetric_field",
        "gated_convolution",
        "conv_normalize_by_degree",
        "gate_init",
        "hidden_channels",
        "antisymmetric_channels",
        "zero_init_final_linear",
        "reduce",
        "asr",
        "avg_num_neighbors",
        "num_types",
        "per_species_scale",
        "per_species_scale_trainable",
        "irreps_in",
        "memory_efficient_contraction",
    ]
    assert params["asr"].default == "hard"
    assert inspect.signature(CartesianBECReduce).parameters["asr"].default == "soft"


@pytest.mark.parametrize("rank", [2, 3])
def test_predictor_equivariance_zero_and_checkpoint(rank):
    from allegro.nn import (
        EquivariantDielectricPredictor,
        EquivariantPiezoelectricPredictor,
    )

    cls = (
        EquivariantDielectricPredictor
        if rank == 2
        else EquivariantPiezoelectricPredictor
    )
    module = cls("features", "response", irreps_in={"features": _ambient_irreps(rank)})
    assert "_Q" not in module.state_dict()
    names = [name for name, _ in module.named_parameters()]
    expected = (
        [
            "scalar_mlp._forward._weight_0",
            "scalar_mlp._forward._weight_1",
            "scalar_mlp._forward._weight_2",
            "tensor_linear_expand.weight",
            "tensor_linear_combine.weight",
        ]
        if rank == 2
        else ["tensor_linear_expand.weight", "tensor_linear_out.weight"]
    )
    assert names == expected
    data = graph()
    data["features"] = torch.zeros(2, 3**rank, requires_grad=True)
    zero = module(dict(data))["response"]
    assert torch.count_nonzero(zero) == 0
    zero.sum().backward()
    assert torch.isfinite(data["features"].grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()
    )
    data["features"] = torch.randn(2, 3**rank)
    original = module(dict(data))["response"]
    assert original.shape == (2, 3**rank)
    shaped = original.reshape((-1,) + (3,) * rank)
    assert torch.equal(shaped, shaped.transpose(-1, -2))
    for matrix in (o3.rand_matrix(), -o3.rand_matrix()):
        rotated = dict(data, features=rotate_cartesian(data["features"], matrix, rank))
        torch.testing.assert_close(
            module(rotated)["response"],
            rotate_cartesian(original, matrix, rank),
            atol=3e-5,
            rtol=3e-5,
        )
    with pytest.raises(ValueError, match="flat"):
        module.forward(dict(data, features=torch.zeros(2, 3, 3)))


def test_bec_constructor_snapshots():
    required = inspect.Parameter.empty
    expected = [
        ("fields", required),
        ("out_field", "born_effective_charges"),
        ("reduce", "mean"),
        ("asr", "soft"),
        ("total_out_field", "total_born_effective_charges"),
        ("pre_asr_out_field", None),
        ("avg_num_neighbors", None),
        ("num_types", None),
        ("per_species_scale", None),
        ("per_species_scale_trainable", False),
        ("per_species_shift", None),
        ("irreps_in", None),
    ]
    assert [
        (n, p.default)
        for n, p in inspect.signature(CartesianBECReduce).parameters.items()
    ] == expected
    expected = [
        ("tensor_field", required),
        ("num_tensor_features", required),
        ("scalar_field", "edge_features"),
        ("tensor_basis_field", "edge_attrs"),
        ("out_field", "born_effective_charges"),
        ("total_out_field", "total_born_effective_charges"),
        ("pre_asr_out_field", None),
        ("edge_symmetric_field", "edge_bec_symmetric"),
        ("edge_antisymmetric_field", "edge_bec_antisymmetric"),
        ("gated_convolution", True),
        ("conv_normalize_by_degree", True),
        ("gate_init", 0.0),
        ("hidden_channels", None),
        ("antisymmetric_channels", None),
        ("zero_init_final_linear", True),
        ("reduce", "mean"),
        ("asr", "hard"),
        ("avg_num_neighbors", None),
        ("num_types", None),
        ("per_species_scale", None),
        ("per_species_scale_trainable", False),
        ("irreps_in", None),
        ("memory_efficient_contraction", False),
    ]
    assert [
        (n, p.default)
        for n, p in inspect.signature(TensorFeatureBECReadout).parameters.items()
    ] == expected


def test_bec_scalar_only_tap_constructs():
    module = TensorFeatureBECReadout(
        tensor_field="t",
        num_tensor_features=2,
        gated_convolution=False,
        irreps_in={"t": "0e"},
    )
    assert module.tensor_irreps == o3.Irreps("0e")


@pytest.mark.parametrize("formula,tap", [("ij=ji", "0e"), ("ijk=ikj", "0e+2e")])
def test_missing_response_lift_paths(formula, tap):
    with pytest.raises(ValueError, match="No tensor-product path"):
        TensorFeatureCartesianReadout(
            cartesian_formula=formula,
            out_field="out",
            tensor_field="t",
            num_tensor_features=2,
            gated_convolution=False,
            irreps_in={"t": tap},
        )


def test_bec_construction_against_baseline_golden():
    path = Path(__file__).with_name("tensor_feature_bec_construction_golden.npz")
    with np.load(path, allow_pickle=False) as golden:
        metadata = json.loads(str(golden["metadata"]))
        assert metadata["base"] == "e20d1b8e320667b757af2f5f7c348b8fc80fea56"
        with torch.random.fork_rng(devices=[]), _default_dtype(torch.float64):
            for i, config in enumerate(metadata["configs"]):
                torch.manual_seed(metadata["seed"])
                module = TensorFeatureBECReadout(**config["kwargs"])
                state = module.state_dict()
                assert list(state) == config["keys"]
                assert torch.equal(
                    torch.get_rng_state(), torch.from_numpy(golden[f"rng_{i}"])
                )
                for j, (key, value) in enumerate(state.items()):
                    assert torch.equal(
                        value, torch.from_numpy(golden[f"state_{i}_{j}"])
                    ), (i, key)


@pytest.mark.parametrize("formula", ["ij", "ij=ji", "ijk=ikj"])
def test_readout_memory_contraction_parity(formula):
    models, rng = [], []
    for efficient in (False, True):
        torch.manual_seed(318)
        models.append(
            readout(
                formula,
                antisymmetric_channels=2 if formula == "ij" else None,
                memory_efficient_contraction=efficient,
                zero_init_final_linear=False,
                num_types=2,
                per_species_scale=[1.2, 0.8],
                per_species_scale_trainable=True,
            )
        )
        rng.append(torch.get_rng_state())
    assert torch.equal(*rng)
    reference, efficient = models
    assert list(reference.state_dict()) == list(efficient.state_dict())
    for key, value in reference.state_dict().items():
        assert torch.equal(value, efficient.state_dict()[key])
    for model, flag in zip(models, (False, True)):
        assert model.tensor_product.memory_efficient is flag
        assert model.tensor_convolution.tensor_product.memory_efficient is flag
        if formula == "ij":
            assert model.antisymmetric_tensor_product.memory_efficient is flag
    # Activate hidden and antisymmetric branches as well as the main branch.
    torch.manual_seed(97)
    randomize(reference)
    # Check strict checkpoint interchange in both directions.
    efficient.load_state_dict(reference.state_dict(), strict=True)
    reference.load_state_dict(efficient.state_dict(), strict=True)
    data = graph()
    width = 27 if formula == "ijk=ikj" else 9
    atom_probe, frame_probe = torch.randn(5, width), torch.randn(2, width)
    results = []
    for model in models:
        inputs = {
            key: data[key].detach().clone().requires_grad_()
            for key in ("tensor", "scalar", "basis")
        }
        result = model(dict(data, **inputs))
        loss = (result["out"] * atom_probe).sum() + (
            result["total"] * frame_probe
        ).sum()
        gradients = torch.autograd.grad(loss, (*inputs.values(), *model.parameters()))
        assert all(torch.isfinite(g).all() for g in gradients)
        named_grads = dict(zip(inputs, gradients[:3]))
        named_grads.update(zip(dict(model.named_parameters()), gradients[3:]))
        for name in (
            "tensor",
            "scalar",
            "basis",
            "tensor_product.weights",
            "tensor_convolution.tensor_product.weights",
        ):
            assert named_grads[name].abs().max() > 1e-12, name
        if formula == "ij":
            assert (
                named_grads["antisymmetric_tensor_product.weights"].abs().max() > 1e-12
            )
        assert result["out"].abs().max() > 1e-8
        results.append((result["out"], result["total"], *gradients))
    tolerance = 1e-12 if data["tensor"].dtype == torch.float64 else 1e-5
    for expected, actual in zip(*results):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
