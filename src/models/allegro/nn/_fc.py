from typing import List, Optional
import math
import logging

import torch
from torch import fx
import torch.nn.functional

import opt_einsum_fx

from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.util.codegen import CodeGenMixin
from e3nn.math import normalize2mom

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.nn.nonlinearities import ShiftedSoftPlus


@compile_mode("script")
class HB_Energy(GraphModuleMixin, torch.nn.Module):
    """
    Calculate Heisenberg energy.
    E_HB_ij = J_ij * S_i * S_j
    """

    field: str
    out_field: str

    def __init__(
        self,
        field: str = AtomicDataDict.EDGE_HB_COEFF_KEY,
        out_field: str = AtomicDataDict.EDGE_ENERGY_KEY,
        irreps_in=None,
        add_collect: bool = True,
        coeff: float = 1.0,
        coeff_trainable: bool = False,
    ):
        super().__init__()
        self.add_collect = add_collect
        self.coeff_og = coeff
        # coeff_og = torch.as_tensor(coeff, dtype=torch.get_default_dtype())
        # self.register_buffer("coeff_og", coeff_og)
        coeff = torch.as_tensor([coeff], dtype=torch.get_default_dtype())
        if coeff_trainable:
            self.coeff = torch.nn.Parameter(coeff)
        else:
            self.register_buffer("coeff", coeff)
        self.field = field
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field],
        )
        assert len(self.irreps_in[self.field]) == 1
        assert self.irreps_in[self.field][0].ir == (0, 1, 1)  # scalars
        self.irreps_out[self.out_field] = o3.Irreps([(1, (0, 1))])
        self.irreps_out[AtomicDataDict.EDGE_SISJ_KEY] = o3.Irreps([(1, (0, 1))])
        logging.info(
            "Building HB energy with coeff: {} and trainable status: {}".format(
                self.coeff, coeff_trainable
            )
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_sisj(data)
        data[self.field] = data[self.field] * self.coeff
        ene = data[self.field] * data[AtomicDataDict.EDGE_SISJ_KEY]
        if self.add_collect:
            data[self.out_field] = ene + data[self.out_field]
        else:
            data[self.out_field] = ene
        return data


@compile_mode("script")
class Biquad_Energy(GraphModuleMixin, torch.nn.Module):
    """
    Calculate Heisenberg energy.
    E_biquad_ij = K_ij * (S_i * S_j)^2
    """

    field: str
    out_field: str

    def __init__(
        self,
        field: str = AtomicDataDict.EDGE_HB_COEFF_KEY,
        out_field: str = AtomicDataDict.EDGE_ENERGY_KEY,
        irreps_in=None,
        add_collect: bool = True,
        coeff: float = 1.0,
        coeff_trainable: bool = False,
    ):
        super().__init__()
        self.add_collect = add_collect
        self.coeff_og = coeff
        # coeff_og = torch.as_tensor(coeff, dtype=torch.get_default_dtype())
        # self.register_buffer("coeff_og", coeff_og)
        coeff = torch.as_tensor([coeff], dtype=torch.get_default_dtype())
        if coeff_trainable:
            self.coeff = torch.nn.Parameter(coeff)
        else:
            self.register_buffer("coeff", coeff)
        self.field = field
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field],
        )
        assert len(self.irreps_in[self.field]) == 1
        assert self.irreps_in[self.field][0].ir == (0, 1, 1)  # scalars
        self.irreps_out[self.out_field] = o3.Irreps([(1, (0, 1))])
        self.irreps_out[AtomicDataDict.EDGE_SISJ_KEY] = o3.Irreps([(1, (0, 1))])
        logging.info(
            "Building HB energy with coeff: {} and trainable status: {}".format(
                self.coeff, coeff_trainable
            )
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_sisj(data)
        data[self.field] = data[self.field] * self.coeff
        ene = (
            data[self.field]
            * data[AtomicDataDict.EDGE_SISJ_KEY]
            * data[AtomicDataDict.EDGE_SISJ_KEY]
        )
        if self.add_collect:
            data[self.out_field] = ene + data[self.out_field]
        else:
            data[self.out_field] = ene
        return data


def einvecmatvecmul(v1, mat, v2):
    return torch.einsum("ik,ikj,ij->i", v1, mat, v2)


@compile_mode("script")
class Spin_JMatrix_Energy(GraphModuleMixin, torch.nn.Module):
    """
    Calculate J Matrix energy.
    Spin_JMatrix_Energy = S_i.T X J_ij X S_j
    """

    field: str
    out_field: str

    def __init__(
        self,
        field: str = AtomicDataDict.EDGE_HB_MATRIX_KEY,
        out_field: str = AtomicDataDict.EDGE_ENERGY_KEY,
        irreps_in=None,
        add_collect: bool = True,
        coeff: float = 1.0,
        coeff_trainable: bool = False,
    ):
        super().__init__()
        self.add_collect = add_collect
        self.coeff_og = coeff
        coeff = torch.as_tensor([coeff], dtype=torch.get_default_dtype())
        if coeff_trainable:
            self.coeff = torch.nn.Parameter(coeff)
        else:
            self.register_buffer("coeff", coeff)
        self.field = field
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field],
        )
        self.irreps_out[self.out_field] = o3.Irreps([(1, (0, 1))])

        graph_mod = torch.fx.symbolic_trace(einvecmatvecmul)
        graph_opt = opt_einsum_fx.optimize_einsums_full(
            model=graph_mod,
            example_inputs=(torch.zeros(5, 3), torch.zeros(5, 3, 3), torch.zeros(5, 3)),
        )
        self.graph_script_hb = opt_einsum_fx.jitable(graph_opt)

        logging.info(
            "Building HB energy with coeff: {} and trainable status: {}".format(
                self.coeff, coeff_trainable
            )
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_sisj(data)
        spin = data[AtomicDataDict.SPIN_NONLEAF_KEY]
        col = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        row = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        data[self.field] = data[self.field] * self.coeff

        ene = self.graph_script_hb(
            spin[row],
            data[self.field],
            spin[col],
        ).view(-1, 1)

        if self.add_collect:
            data[self.out_field] = ene + data[self.out_field]
        else:
            data[self.out_field] = ene
        return data


@compile_mode("script")
class Spin_ani_Energy(GraphModuleMixin, torch.nn.Module):
    """
    Calculate Anisotropy energy.
    Spin_ani_Energy = S_i.T X A_ij X S_i
    """

    field: str
    out_field: str

    def __init__(
        self,
        field: str = AtomicDataDict.NODE_SPINANI_MATRIX_KEY,
        out_field: str = AtomicDataDict.EDGE_ENERGY_KEY,
        irreps_in=None,
        add_collect: bool = True,
        coeff: float = 1.0,
        coeff_trainable: bool = False,
    ):
        super().__init__()
        self.add_collect = add_collect
        self.coeff_og = coeff
        coeff = torch.as_tensor([coeff], dtype=torch.get_default_dtype())
        if coeff_trainable:
            self.coeff = torch.nn.Parameter(coeff)
        else:
            self.register_buffer("coeff", coeff)
        self.field = field
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field],
        )
        self.irreps_out[self.out_field] = o3.Irreps([(1, (0, 1))])

        graph_mod = torch.fx.symbolic_trace(einvecmatvecmul)
        graph_opt = opt_einsum_fx.optimize_einsums_full(
            model=graph_mod,
            example_inputs=(torch.zeros(5, 3), torch.zeros(5, 3, 3), torch.zeros(5, 3)),
        )
        self.graph_script_ani = opt_einsum_fx.jitable(graph_opt)

        logging.info(
            "Building HB energy with coeff: {} and trainable status: {}".format(
                self.coeff, coeff_trainable
            )
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_sisj(data)
        spin = data[AtomicDataDict.SPIN_NONLEAF_KEY]
        row = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        data[self.field] = data[self.field] * self.coeff

        ene = self.graph_script_ani(
            spin[row],
            data[self.field],
            spin[row],
        ).view(-1, 1)

        if self.add_collect:
            data[self.out_field] = ene + data[self.out_field]
        else:
            data[self.out_field] = ene
        return data


@compile_mode("script")
class ScalarMLP(GraphModuleMixin, torch.nn.Module):
    """Apply an MLP to some scalar field."""

    field: str
    out_field: str

    def __init__(
        self,
        mlp_latent_dimensions: List[int],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_initialization: str = "uniform",
        mlp_dropout_p: float = 0.0,
        mlp_batchnorm: bool = False,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
        add_collect: bool = False,
        cat_collect: bool = False,
        cat_collect_origin_dim: int = 0,
        coeff: float = 1.0,
        coeff_trainable: bool = False,
        normalize: bool = False,
    ):
        super().__init__()
        self.add_collect = add_collect
        self.cat_collect = cat_collect
        self.normalize = normalize
        self.cat_collect_origin_dim = cat_collect_origin_dim
        # coeff_og = torch.as_tensor(coeff, dtype=torch.get_default_dtype())
        # self.register_buffer("coeff_og", coeff_og)
        coeff = torch.as_tensor(coeff, dtype=torch.get_default_dtype())
        self.coeff_og = coeff
        if coeff_trainable:
            self.coeff = torch.nn.Parameter(coeff)
        else:
            self.register_buffer("coeff", coeff)
        self.field = field
        self.out_field = out_field if out_field is not None else field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field],
        )
        assert len(self.irreps_in[self.field]) == 1
        assert self.irreps_in[self.field][0].ir == (0, 1, 1)  # scalars
        in_dim = self.irreps_in[self.field][0].mul
        self._module = ScalarMLPFunction(
            mlp_input_dimension=in_dim,
            mlp_latent_dimensions=mlp_latent_dimensions,
            mlp_output_dimension=mlp_output_dimension,
            mlp_nonlinearity=mlp_nonlinearity,
            mlp_initialization=mlp_initialization,
            mlp_dropout_p=mlp_dropout_p,
            mlp_batchnorm=mlp_batchnorm,
        )
        if self.cat_collect:
            self.irreps_out[self.out_field] = o3.Irreps(
                [(self._module.out_features + self.cat_collect_origin_dim, (0, 1))]
            )
        else:
            self.irreps_out[self.out_field] = o3.Irreps(
                [(self._module.out_features, (0, 1))]
            )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.add_collect:
            data[self.out_field] = (
                torch.einsum("ik,->ik", [self._module(data[self.field]), self.coeff])
                + data[self.out_field]
            )
        elif self.cat_collect:
            if self.normalize:
                out = torch.einsum(
                    "ik,->ik", [self._module(data[self.field]), self.coeff]
                )
                out = torch.nn.functional.normalize(out, p=2.0, dim=-1)
            else:
                out = torch.einsum(
                    "ik,->ik", [self._module(data[self.field]), self.coeff]
                )
            data[self.out_field] = torch.cat((data[self.out_field], out), dim=-1)
        else:
            if self.normalize:
                data[self.out_field] = torch.nn.functional.normalize(
                    torch.einsum(
                        "ik,->ik", [self._module(data[self.field]), self.coeff]
                    ),
                    p=2.0,
                    dim=-1,
                )
            else:
                data[self.out_field] = torch.einsum(
                    "ik,->ik", [self._module(data[self.field]), self.coeff]
                )
        return data


class ScalarMLPFunction(CodeGenMixin, torch.nn.Module):
    """Module implementing an MLP according to provided options."""

    in_features: int
    out_features: int

    def __init__(
        self,
        mlp_input_dimension: Optional[int],
        mlp_latent_dimensions: List[int],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_initialization: str = "normal",
        mlp_dropout_p: float = 0.0,
        mlp_batchnorm: bool = False,
    ):
        super().__init__()
        nonlinearity = {
            None: None,
            "silu": torch.nn.functional.silu,
            "ssp": ShiftedSoftPlus,
        }[mlp_nonlinearity]
        if nonlinearity is not None:
            nonlin_const = normalize2mom(nonlinearity).cst
        else:
            nonlin_const = 1.0

        dimensions = (
            ([mlp_input_dimension] if mlp_input_dimension is not None else [])
            + mlp_latent_dimensions
            + ([mlp_output_dimension] if mlp_output_dimension is not None else [])
        )
        assert len(dimensions) >= 2  # Must have input and output dim
        num_layers = len(dimensions) - 1

        self.in_features = dimensions[0]
        self.out_features = dimensions[-1]

        # Code
        params = {}
        graph = fx.Graph()
        tracer = fx.proxy.GraphAppendingTracer(graph)

        def Proxy(n):
            return fx.Proxy(n, tracer=tracer)

        features = Proxy(graph.placeholder("x"))
        norm_from_last: float = 1.0

        base = torch.nn.Module()

        for layer, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            # do dropout
            if mlp_dropout_p > 0:
                # only dropout if it will do something
                # dropout before linear projection- https://stats.stackexchange.com/a/245137
                features = Proxy(graph.call_module("_dropout", (features.node,)))

            # make weights
            w = torch.empty(h_in, h_out)

            if mlp_initialization == "normal":
                w.normal_()
            elif mlp_initialization == "uniform":
                # these values give < x^2 > = 1
                w.uniform_(-math.sqrt(3), math.sqrt(3))
            elif mlp_initialization == "orthogonal":
                # this rescaling gives < x^2 > = 1
                torch.nn.init.orthogonal_(w, gain=math.sqrt(max(w.shape)))
            else:
                raise NotImplementedError(
                    f"Invalid mlp_initialization {mlp_initialization}"
                )

            # generate code
            params[f"_weight_{layer}"] = w
            w = Proxy(graph.get_attr(f"_weight_{layer}"))
            w = w * (
                norm_from_last / math.sqrt(float(h_in))
            )  # include any nonlinearity normalization from previous layers
            features = torch.matmul(features, w)

            if mlp_batchnorm:
                # if we call batchnorm, do it after the nonlinearity
                features = Proxy(graph.call_module(f"_bn_{layer}", (features.node,)))
                setattr(base, f"_bn_{layer}", torch.nn.BatchNorm1d(h_out))

            # generate nonlinearity code
            if nonlinearity is not None and layer < num_layers - 1:
                features = nonlinearity(features)
                # add the normalization const in next layer
                norm_from_last = nonlin_const

        graph.output(features.node)

        for pname, p in params.items():
            setattr(base, pname, torch.nn.Parameter(p))

        if mlp_dropout_p > 0:
            # with normal dropout everything blows up
            base._dropout = torch.nn.AlphaDropout(p=mlp_dropout_p)

        self._codegen_register({"_forward": fx.GraphModule(base, graph)})

    def forward(self, x):
        return self._forward(x)


@compile_mode("script")
class Delta_spin_energy_sum(GraphModuleMixin, torch.nn.Module):
    """
    Sum over TOTAL_ENERGY_FIX_SPIN_KEY and TOTAL_ENERGY_DELTA_SPIN_KEY
    """

    field: str
    out_field: str

    def __init__(
        self,
        irreps_in=None,
    ):
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={
                AtomicDataDict.TOTAL_ENERGY_FIX_SPIN_KEY: "0e",
                AtomicDataDict.TOTAL_ENERGY_DELTA_SPIN_KEY: "0e",
            },
            irreps_out={AtomicDataDict.TOTAL_ENERGY_KEY: "0e"},
        )
        logging.info(
            "Sum over TOTAL_ENERGY_FIX_SPIN_KEY and TOTAL_ENERGY_DELTA_SPIN_KEY"
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = (
            data[AtomicDataDict.TOTAL_ENERGY_FIX_SPIN_KEY]
            + data[AtomicDataDict.TOTAL_ENERGY_DELTA_SPIN_KEY]
        )
        return data


@compile_mode("script")
class DipoleAdd(GraphModuleMixin, torch.nn.Module):
    """
    Add two dipole fields together: out_field += field_to_add
    Used to combine dipole contributions from onlyr and nosoc stages.
    """

    field_to_add: str
    out_field: str

    def __init__(
        self,
        field_to_add: str = "atomic_dipole_onlyr",
        out_field: str = AtomicDataDict.ATOMIC_DIPOLE_KEY,
        irreps_in=None,
    ):
        super().__init__()
        self.field_to_add = field_to_add
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={
                AtomicDataDict.ATOMIC_DIPOLE_KEY: "1o",
                self.field_to_add: "1o",
            },
            irreps_out={self.out_field: "1o"},
        )
        logging.info(f"DipoleAdd: {self.out_field} += {self.field_to_add}")

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.out_field] = (
            data[AtomicDataDict.ATOMIC_DIPOLE_KEY] + data[self.field_to_add]
        )
        return data
