"""End-to-end response heads on a small settable tensor-feature backbone."""

import pytest
import torch
from e3nn import o3

from nequip.model import model_from_config
from nequip.nn import GraphModuleMixin, SequentialGraphNetwork
from allegro.nn._bec_readout import _default_dtype

pytestmark = [
    pytest.mark.filterwarnings("ignore:`torch.jit.*:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:The TorchScript type system doesn't support.*:UserWarning"
    ),
]

SH = "0e+1o+2e+3o"


class FeatureSource(GraphModuleMixin, torch.nn.Module):
    """Settable fake backbone; tensor features retain a real gradient path."""

    def __init__(self):
        super().__init__()
        self.tensor = torch.nn.Parameter(torch.randn(6, 2, 16))
        self.register_buffer("scalar", torch.randn(6, 4))
        self.register_buffer("basis", torch.randn(6, 16))
        self._init_irreps(irreps_out={"tensor": SH, "scalar": "4x0e", "basis": SH})

    def forward(self, data):
        data.update(tensor=self.tensor, scalar=self.scalar, basis=self.basis)
        return data


def source_builder(config):
    return SequentialGraphNetwork({"features": FeatureSource()})


def config(kind):
    return dict(
        model_builders=[source_builder, "allegro.model.TensorFeature" + kind],
        num_types=2,
        type_names=["H", "O"],
        TensorFeatureCartesianReadout_tensor_field="tensor",
        TensorFeatureCartesianReadout_scalar_field="scalar",
        TensorFeatureCartesianReadout_tensor_basis_field="basis",
        TensorFeatureCartesianReadout_num_tensor_features=2,
        TensorFeatureCartesianReadout_hidden_channels=2,
        TensorFeatureCartesianReadout_gate_init=0.4,
        TensorFeatureCartesianReadout_per_species_scale=2.0,
        TensorFeatureCartesianReadout_per_species_scale_trainable=True,
        ipv_readout_antisymmetric_channels=2,
    )


def graph():
    return {
        "pos": torch.randn(5, 3),
        "edge_index": torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 0, 0, 4, 3]]),
        "batch": torch.tensor([0, 0, 0, 1, 1]),
        "atom_types": torch.tensor([[0], [1], [0], [1], [0]]),
        "cell": torch.stack(
            [
                torch.diag(torch.tensor([2.0, 3.0, 4.0])),
                torch.diag(torch.tensor([-3.0, 4.0, 5.0])),
            ]
        ),
        "pbc": torch.ones(2, 3, dtype=torch.bool),
    }


def rotate(x, matrix, rank):
    if rank == 2:
        return torch.einsum(
            "ia,bac,jc->bij", matrix, x.reshape(-1, 3, 3), matrix
        ).flatten(1)
    return torch.einsum(
        "ia,jc,kd,bacd->bijk", matrix, matrix, matrix, x.reshape(-1, 3, 3, 3)
    ).flatten(1)


@pytest.mark.parametrize("kind", ["Dielectric", "Piezo"])
def test_response_builder_contract(kind):
    torch.manual_seed(31)
    cfg = config(kind)
    # Conflicting shared/layer values must not override physical semantics.
    cfg.update(
        asr="hard",
        total_reduce="mean",
        cartesian_formula="ij",
        out_field="wrong",
        antisymmetric_channels=2,
    )
    cfg["eps_el_readout_cartesian_formula"] = "ij"
    model = model_from_config(cfg)
    rank = 2 if kind == "Dielectric" else 3
    node, ion, electronic = (
        ("ipv", "epsilon_ion", "epsilon_el")
        if rank == 2
        else ("ipd", "piezo_ion", "piezo_el")
    )
    layers = (
        [
            "features",
            "ipv_readout",
            "epsilon_ion_calc",
            "eps_el_readout",
            "system_eps_el_predict",
        ]
        if rank == 2
        else [
            "features",
            "ipd_readout",
            "piezo_ion_calc",
            "piezo_el_readout",
            "system_piezo_el_predict",
        ]
    )
    assert list(dict(model.named_children())) == layers
    assert model[1].cartesian_reduce.asr == "none"
    assert model[3].antisymmetric_channels is None
    data = graph()
    for randomized in (False, True):
        if randomized:
            with torch.no_grad():
                for p in model.parameters():
                    p.uniform_(-0.5, 0.5)
        model.zero_grad(set_to_none=True)
        out = model(dict(data))
        assert out[node].shape == (5, 3**rank)
        assert out[ion].shape == out[electronic].shape == (2, 3**rank)
        for field in (node, ion, electronic):
            assert model.irreps_out[field].dim == 3**rank
        total = torch.stack([out[node][:3].sum(0), out[node][3:].sum(0)])
        torch.testing.assert_close(out["total_" + node], total)
        torch.testing.assert_close(
            out[ion], total / torch.linalg.det(data["cell"]).abs()[:, None]
        )
        atomic_key, system_key = (
            ("atomic_eps_el", "system_eps_el_features")
            if rank == 2
            else ("atomic_piezo_el", "system_piezo_el_features")
        )
        mean = torch.stack([out[atomic_key][:3].mean(0), out[atomic_key][3:].mean(0)])
        torch.testing.assert_close(out[system_key], mean)
        if randomized:
            assert out[ion].abs().max() > 1e-8
        sum(
            (out[k] - torch.randn_like(out[k])).square().mean()
            for k in (node, ion, electronic)
        ).backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )
    original = {key: out[key].detach() for key in (node, ion, electronic)}
    source = model.features
    tensor, basis = source.tensor.detach().clone(), source.basis.clone()
    for matrix in (o3.rand_matrix(), -o3.rand_matrix()):
        d = o3.Irreps(SH).D_from_matrix(matrix)
        with torch.no_grad():
            source.tensor.copy_(tensor @ d.T)
            source.basis.copy_(basis @ d.T)
        rotated = dict(data, pos=data["pos"] @ matrix.T, cell=data["cell"] @ matrix.T)
        actual = model(rotated)
        for key in original:
            tolerance = 3e-5 if actual[key].dtype == torch.float32 else 1e-10
            torch.testing.assert_close(
                actual[key],
                rotate(original[key], matrix, rank),
                rtol=tolerance,
                atol=tolerance,
            )


@pytest.mark.parametrize("kind", ["Dielectric", "Piezo"])
def test_response_construct_dtype(kind):
    cfg = config(kind)
    name = "ipv_readout" if kind == "Dielectric" else "ipd_readout"
    cfg[name + "_construct_dtype"] = "float64"
    default = torch.get_default_dtype()
    torch.manual_seed(7)
    model = model_from_config(cfg)
    assert all(p.dtype == default for p in model.parameters())
    # Isolate the head's RNG sequence from the fake backbone's dtype draws.
    from allegro.model._tensor_feature_bec import append_layer
    from allegro.nn import TensorFeatureCartesianReadout

    cfg = config(kind)
    source = source_builder(cfg)
    params = dict(cartesian_formula="ij", out_field="out", asr="none")
    torch.manual_seed(123)
    append_layer(
        dict(cfg, head_construct_dtype="float64"),
        source,
        "head",
        TensorFeatureCartesianReadout,
        params,
    )
    other = source_builder(cfg)
    torch.manual_seed(123)
    with _default_dtype(torch.float64):
        append_layer(cfg, other, "head", TensorFeatureCartesianReadout, params)
    other.head.to(default)
    for key, value in source.head.state_dict().items():
        assert torch.equal(value, other.head.state_dict()[key])


@pytest.mark.parametrize("channels", [None, 2])
def test_symmetric_ipv_warning(caplog, channels):
    cfg = config("Dielectric")
    cfg["ipv_readout_antisymmetric_channels"] = channels
    with caplog.at_level("WARNING"):
        model = model_from_config(cfg)
    assert model.ipv_readout.antisymmetric_channels == channels
    assert ("same-edge ij lift has no antisymmetric (1e) part" in caplog.text) == (
        channels is None
    )


@pytest.mark.parametrize("kind", ["Dielectric", "Piezo"])
def test_memory_contraction_config_prefixes(kind):
    cfg = config(kind)
    cfg["TensorFeatureCartesianReadout_memory_efficient_contraction"] = True
    electronic = "eps_el_readout" if kind == "Dielectric" else "piezo_el_readout"
    cfg[electronic + "_memory_efficient_contraction"] = False
    model = model_from_config(cfg)
    ionic = model.ipv_readout if kind == "Dielectric" else model.ipd_readout
    assert ionic.tensor_product.memory_efficient
    assert ionic.tensor_convolution.tensor_product.memory_efficient
    if kind == "Dielectric":
        assert ionic.antisymmetric_tensor_product.memory_efficient
    head = getattr(model, electronic)
    assert not head.tensor_product.memory_efficient
    assert not head.tensor_convolution.tensor_product.memory_efficient
