"""Formula-generic Cartesian response readouts and reductions.

TensorFeatureCartesianReadout builds edge contributions from tensor features;
CartesianTensorReduce converts them to flat atomic and frame responses.
The BEC specializations preserve the rank-2 constructor and ASR contracts.
"""

import contextlib
import functools
import math
from typing import Dict, List, Optional, Union

import torch
from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.io import CartesianTensor
from e3nn.nn import NormActivation

from nequip.data import AtomicDataDict
from nequip.nn import AtomwiseReduce, GraphModuleMixin

from ._strided import GatherContracter, MakeWeightedChannels

# Input "basis" of a per-edge field already holding flat Cartesian components.
CARTESIAN_BASIS = "cartesian"
_REDUCTIONS = ("mean", "sum", "normalized_sum")
_ASR_MODES = ("soft", "hard", "none")
_READOUT_FORMULAS = ("ij", "ij=ji", "ijk=ikj")


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


# Irreps of the CartesianTensor formulas used here; constructing a CartesianTensor
# builds (and scripts) e3nn tensor products, which only the change of basis needs.
_CARTESIAN_IRREPS = {
    "ij": "1x0e+1x1e+1x2e",
    "ij=-ji": "1x1e",
    "ij=ji": "1x0e+1x2e",
    "ijk=ikj": "2x1o+1x2o+1x3o",
    "ijk": "1x0o+3x1o+2x2o+1x3o",
}


def _cartesian_irreps(formula: str) -> o3.Irreps:
    if formula in _CARTESIAN_IRREPS:
        return o3.Irreps(_CARTESIAN_IRREPS[formula])
    return o3.Irreps(str(CartesianTensor(formula)))


def _formula_rank(formula: str) -> int:
    return len(formula.split("=")[0].replace("-", ""))


def _ambient_irreps(rank: int) -> o3.Irreps:
    if rank not in (2, 3):
        raise ValueError("cartesian_rank must be 2 or 3")
    return _cartesian_irreps("ij" if rank == 2 else "ijk")


def _lift_irreps(formula: str) -> o3.Irreps:
    return o3.Irreps(
        [(1, ir) for mul, ir in _cartesian_irreps(formula) for _ in range(mul)]
    )


@functools.lru_cache(None)
def _change_of_basis_float64(formula: str) -> torch.Tensor:
    """Cache the float64 ``(irreps.dim, 3**rank)`` Cartesian basis matrix."""
    with _default_dtype(torch.float64):
        q = CartesianTensor(formula).reduced_tensor_products().change_of_basis
    return q.flatten(1).detach()


def _change_of_basis(formula: str) -> torch.Tensor:
    return _change_of_basis_float64(formula).clone().to(torch.get_default_dtype())


def hard_acoustic_sum_rule(
    bec: torch.Tensor, batch: torch.Tensor, num_frames: Optional[int] = None
) -> torch.Tensor:
    """Impose the acoustic sum rule exactly by removing each structure's mean.

    ``bec`` holds per-atom responses, ``(n_atoms, ...)``, e.g. ``(n_atoms, 9)``
    or ``(n_atoms, 3, 3)``. Every component is centred separately over the
    atoms of its structure (``batch`` maps atoms to structures), so the
    responses of each structure sum to zero. ``num_frames`` defaults to
    ``batch.max() + 1``.
    """
    batch = batch.reshape(-1)
    total = scatter(bec, batch, dim=0, dim_size=num_frames)
    counts = scatter(
        torch.ones_like(batch, dtype=bec.dtype),
        batch,
        dim=0,
        dim_size=total.shape[0],
    )
    shape = [total.shape[0]] + [1] * (bec.dim() - 1)
    mean = total / counts.clamp_min(1.0).view(shape)
    return bec - torch.index_select(mean, 0, batch)


def _symmetrize_cartesian(x: torch.Tensor, rank: int) -> torch.Tensor:
    """Remove roundoff asymmetry in the derived Cartesian basis projection.

    Rank zero leaves unrestricted tensors bitwise unchanged. The e3nn rank-3
    basis can differ at roundoff between mathematically identical columns.
    """
    if rank == 2:
        tensor = x.reshape(-1, 3, 3)
        return (0.5 * (tensor + tensor.transpose(-1, -2))).flatten(1)
    if rank == 3:
        tensor = x.reshape(-1, 3, 3, 3)
        return (0.5 * (tensor + tensor.transpose(-1, -2))).flatten(1)
    return x


class _ToCartesian(torch.nn.Module):
    """Change of basis of one reduced field to flattened Cartesian components."""

    field: str
    identity: bool
    symmetric_rank: int

    def __init__(self, field: str, basis: str):
        super().__init__()
        self.field = field
        self.basis = basis
        self.symmetric_rank = (
            2 if basis == "ij=ji" else (3 if basis == "ijk=ikj" else 0)
        )
        self.identity = basis == CARTESIAN_BASIS
        q = torch.zeros(0) if self.identity else _change_of_basis(basis)
        # derived constant: not part of the checkpoint
        self.register_buffer("change_of_basis", q, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return x
        result = x @ self.change_of_basis
        return _symmetrize_cartesian(result, self.symmetric_rank)


class CartesianTensorReduce(GraphModuleMixin, torch.nn.Module):
    """Reduce per-edge contributions to flat per-atom Cartesian tensors.

    ``fields`` maps each edge field to a ``CartesianTensor`` formula or
    ``"cartesian"``. Formula rank must equal ``cartesian_rank`` (2 or 3);
    Cartesian input and atomic output widths are ``3**cartesian_rank``.
    Contributions are reduced to edge centres and summed in field order.

    ``reduce`` selects edge-to-atom mean, sum, or normalized_sum (division by
    sqrt(avg_num_neighbors)). Optional per-species scales apply before ASR;
    per_species_shift accepts only zero for checkpoint compatibility.
    ``asr="none"`` leaves responses uncorrected; ``soft`` also leaves them
    uncorrected but requires mean frame totals to supervise the ASR;
    ``hard`` subtracts the per-frame atomic mean componentwise.
    ``total_reduce`` selects mean or sum for ``total_out_field`` (if present).
    ``pre_asr_out_field`` optionally exposes the uncorrected atomic tensor.
    Scale/shift checkpoint names match ``PerSpeciesScaleShift``.
    """

    out_field: str
    reduce: str
    asr: str
    _factor: Optional[float]
    has_scales: bool
    has_shifts: bool
    _write_total: bool
    _frame_mean_field: str
    _pre_asr_field: str
    _write_pre_asr: bool

    def __init__(
        self,
        fields: Dict[str, str],
        out_field: str,
        reduce: str = "mean",
        asr: str = "none",
        total_out_field: Optional[str] = None,
        cartesian_rank: int = 2,
        total_reduce: str = "mean",
        pre_asr_out_field: Optional[str] = None,
        avg_num_neighbors: Optional[float] = None,
        num_types: Optional[int] = None,
        per_species_scale: Optional[Union[float, List[float]]] = None,
        per_species_scale_trainable: bool = False,
        per_species_shift: Optional[Union[float, List[float]]] = None,
        irreps_in=None,
    ):
        super().__init__()
        if not fields:
            raise ValueError("CartesianTensorReduce needs at least one input field")
        if reduce not in _REDUCTIONS:
            raise ValueError(f"reduce must be one of {_REDUCTIONS}, got {reduce!r}")
        if asr not in _ASR_MODES:
            raise ValueError(f"asr must be one of {_ASR_MODES}, got {asr!r}")
        if total_reduce not in ("mean", "sum"):
            raise ValueError("total_reduce must be mean or sum")
        if asr == "soft" and total_reduce != "mean":
            raise ValueError("soft asr requires total_reduce=mean")
        self.out_field = out_field
        self.reduce = reduce
        self.asr = asr
        self._factor = None
        if reduce == "normalized_sum":
            if avg_num_neighbors is None:
                raise ValueError("reduce='normalized_sum' needs avg_num_neighbors")
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

        irreps_in = {} if irreps_in is None else irreps_in
        out_irreps = _ambient_irreps(cartesian_rank)
        self.converters = torch.nn.ModuleList()
        for field, basis in fields.items():
            if basis != CARTESIAN_BASIS and _formula_rank(basis) != cartesian_rank:
                raise ValueError("basis formula rank does not match cartesian_rank")
            converter = _ToCartesian(field, basis)
            if converter.identity:
                expected_dim = 3**cartesian_rank
                if len(fields) == 1 and field in irreps_in:
                    out_irreps = irreps_in[field]
            else:
                expected_dim = _cartesian_irreps(basis).dim
            if field in irreps_in and irreps_in[field] is not None:
                if o3.Irreps(irreps_in[field]).dim != expected_dim:
                    raise ValueError(
                        f"Field {field!r} has irreps {irreps_in[field]}, but basis "
                        f"{basis!r} needs dimension {expected_dim}"
                    )
            self.converters.append(converter)

        # per-species scale / zero shift, laid out like PerSpeciesScaleShift
        self.has_scales = per_species_scale is not None
        self.has_shifts = per_species_shift is not None
        if (self.has_scales or self.has_shifts) and num_types is None:
            raise ValueError("per-species scales/shifts need num_types")
        n = 1 if num_types is None else int(num_types)
        if self.has_shifts:
            shifts = self._per_type(per_species_shift, n, "per_species_shift")
            if bool((shifts != 0).any()):
                raise ValueError(
                    "per_species_shift must be zero: an isotropic shift of all Cartesian "
                    "components breaks rotation equivariance"
                )
        else:
            shifts = torch.zeros(n, dtype=torch.get_default_dtype())
        self.register_buffer("shifts", shifts, persistent=self.has_shifts)
        if self.has_scales:
            scales = self._per_type(per_species_scale, n, "per_species_scale")
        else:
            scales = torch.ones(n, dtype=torch.get_default_dtype())
        if self.has_scales and per_species_scale_trainable:
            self.scales = torch.nn.Parameter(scales)
        else:
            self.register_buffer("scales", scales, persistent=self.has_scales)

        self._write_total = total_out_field is not None
        self._frame_mean_field = (
            total_out_field
            if total_out_field is not None
            else f"_{out_field}_frame_mean"
        )
        self._write_pre_asr = pre_asr_out_field is not None
        self._pre_asr_field = (
            pre_asr_out_field if pre_asr_out_field is not None else out_field
        )

        irreps_out = {out_field: out_irreps}
        if self._write_total:
            irreps_out[self._frame_mean_field] = out_irreps
        if self._write_pre_asr:
            irreps_out[self._pre_asr_field] = out_irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=list(fields),
            irreps_out=irreps_out,
        )
        # per-structure mean over atoms; shares the batch reduction (and its
        # symbolic-tracing support) with the other atomwise reductions
        self.frame_mean = AtomwiseReduce(
            field=out_field,
            out_field=self._frame_mean_field,
            reduce=total_reduce,
            irreps_in={out_field: out_irreps},
        )

    @staticmethod
    def _per_type(value, num_types: int, name: str) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.get_default_dtype())
        if len(value.reshape([-1])) == 1:
            value = torch.ones(num_types) * value
        if value.shape != (num_types,):
            raise ValueError(f"{name} must have one entry per type, got {value}")
        return value

    def _edge_to_atom(
        self, x: torch.Tensor, edge_center: torch.Tensor, num_atoms: int
    ) -> torch.Tensor:
        out = scatter(x, edge_center, dim=0, dim_size=num_atoms)
        if self.reduce == "mean":
            counts = scatter(
                torch.ones_like(edge_center, dtype=x.dtype),
                edge_center,
                dim=0,
                dim_size=num_atoms,
            )
            out = out / counts.clamp_min(1.0).unsqueeze(-1)
        factor: Optional[float] = self._factor
        if factor is not None:
            out = out * factor
        return out

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        num_atoms = data[AtomicDataDict.POSITIONS_KEY].shape[0]
        bec = torch.zeros(0)
        first = True
        for converter in self.converters:
            atom = converter(
                self._edge_to_atom(data[converter.field], edge_center, num_atoms)
            )
            if first:
                bec = atom
                first = False
            else:
                bec = bec + atom

        if self.has_scales or self.has_shifts:
            species = data[AtomicDataDict.ATOM_TYPE_KEY]
            if self.has_scales:
                bec = self.scales[species].view(-1, 1) * bec
            if self.has_shifts:
                bec = self.shifts[species].view(-1, 1) + bec
        if self._write_pre_asr:
            data[self._pre_asr_field] = bec

        data[self.out_field] = bec
        if self.asr == "hard":
            if AtomicDataDict.BATCH_KEY in data:
                batch = data[AtomicDataDict.BATCH_KEY]
            else:
                batch = torch.zeros(bec.shape[0], dtype=torch.long, device=bec.device)
            data[self.out_field] = hard_acoustic_sum_rule(bec, batch)
        if self._write_total:
            data = self.frame_mean(data)
        return data


class GatedTensorFeatureConvolution(torch.nn.Module):
    """Residual tensor convolution over edge tensor features, gated by a scalar.

    ``T'_e = T_e + g * TP(T_e, env[c(e)])`` with
    ``env_n = sum_{c(e)=n} MWC(Y_e, W h_e) / sqrt(max(deg_n, 1))`` (the
    degree normalization is optional), where ``T`` are strided edge tensor
    features, ``h`` edge scalars, ``Y`` an edge tensor basis, ``c(e)`` the
    centre atom of edge ``e`` and ``g`` the ``residual_gate`` (``gate_init``).
    """

    def __init__(
        self,
        tensor_irreps: o3.Irreps,
        num_tensor_features: int,
        tensor_basis_irreps: o3.Irreps,
        scalar_dim: int,
        normalize_by_degree: bool = True,
        gate_init: float = 0.0,
        memory_efficient_contraction: bool = False,
    ):
        super().__init__()
        self.num_tensor_features = int(num_tensor_features)
        self.normalize_by_degree = bool(normalize_by_degree)
        self.env_weighter = MakeWeightedChannels(
            irreps_in=o3.Irreps(tensor_basis_irreps),
            multiplicity_out=self.num_tensor_features,
        )
        self.env_weight_projection = torch.nn.Linear(
            int(scalar_dim), self.env_weighter.weight_numel
        )
        self.tensor_product = GatherContracter(
            irreps_in1=tensor_irreps,
            irreps_in2=tensor_basis_irreps,
            irreps_out=tensor_irreps,
            mul=self.num_tensor_features,
            path_channel_coupling=True,
            memory_efficient=memory_efficient_contraction,
        )
        self.residual_gate = torch.nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        tensor_features: torch.Tensor,
        scalar_features: torch.Tensor,
        tensor_basis: torch.Tensor,
        edge_center: torch.Tensor,
        num_atoms: int,
    ) -> torch.Tensor:
        env_weights = self.env_weight_projection(scalar_features)
        env_edges = self.env_weighter(tensor_basis, env_weights)
        env_nodes = scatter(env_edges, edge_center, dim=0, dim_size=num_atoms)
        if self.normalize_by_degree:
            counts = scatter(
                torch.ones_like(edge_center, dtype=env_nodes.dtype),
                edge_center,
                dim=0,
                dim_size=num_atoms,
            )
            env_nodes = env_nodes / counts.clamp_min(1.0).sqrt().view(-1, 1, 1)
        conv = self.tensor_product(tensor_features, env_nodes, edge_center)
        return tensor_features + self.residual_gate.to(conv.dtype) * conv


class TensorFeatureCartesianReadout(GraphModuleMixin, torch.nn.Module):
    """Cartesian responses from strided edge tensor features of any backbone.

    Supports ``ij``, ``ij=ji`` and ``ijk=ikj``; lift entries have multiplicity
    one, including both vector copies of ``ijk=ikj``. ``asr="none"`` leaves
    responses uncorrected and ``total_reduce`` controls frame sum or mean.

    Inputs: ``tensor_field`` ``T`` of shape ``(n_edges, num_tensor_features,
    irreps.dim)`` (per-channel irreps registered for the field, e.g.
    ``0e+1o+2e+3o``), and for the convolution ``scalar_field`` ``h`` (edge
    scalars) and ``tensor_basis_field`` ``Y`` (edge spherical harmonics).

    1. optional :class:`GatedTensorFeatureConvolution` ``T -> T'``;
    2. same-edge lift ``L_e = TP(T'_e, T'_e)`` to the requested formula;
    3. main readout: per-irrep channel weights (``1/sqrt(F)``), a normalized
       SiLU ``NormActivation`` and an e3nn ``Linear`` (zero-initialized by
       default, so the responses start at zero);
    4. optional hidden readout added to 3: per-irrep channel mixes to
       ``hidden_channels`` (Xavier), NormActivation, Linear, NormActivation and
       a zero-initialized Linear back to the lift irreps;
    5. optional antisymmetric branch: two channel mixes of ``T'`` to
       ``antisymmetric_channels`` (Xavier), their ``1e`` tensor product,
       channel weights, NormActivation and a zero-initialized Linear;
    6. :class:`CartesianTensorReduce` of the lifted per-edge tensor and
       optional ``1e`` ("ij=-ji") field, with ``reduce``, ``asr`` and the optional
       per-species scale; ``pre_asr_out_field`` exposes the responses before the ASR.

    The per-edge results are left in ``edge_symmetric_field`` and
    ``edge_antisymmetric_field``. All tensor products use
    :class:`allegro.nn._strided.GatherContracter`. The readout runs in eager
    mode only; it is not TorchScript-compatible, so models using it cannot be
    deployed with ``nequip-deploy``.
    """

    def __init__(
        self,
        cartesian_formula: str,
        out_field: str,
        tensor_field: str,
        num_tensor_features: int,
        scalar_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
        tensor_basis_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        total_out_field: Optional[str] = None,
        total_reduce: str = "mean",
        pre_asr_out_field: Optional[str] = None,
        edge_symmetric_field: Optional[str] = None,
        edge_antisymmetric_field: Optional[str] = None,
        gated_convolution: bool = True,
        conv_normalize_by_degree: bool = True,
        gate_init: float = 0.0,
        hidden_channels: Optional[int] = None,
        antisymmetric_channels: Optional[int] = None,
        zero_init_final_linear: bool = True,
        reduce: str = "mean",
        asr: str = "none",
        avg_num_neighbors: Optional[float] = None,
        num_types: Optional[int] = None,
        per_species_scale: Optional[Union[float, List[float]]] = None,
        per_species_scale_trainable: bool = False,
        irreps_in=None,
        memory_efficient_contraction: bool = False,
    ):
        super().__init__()
        if cartesian_formula not in _READOUT_FORMULAS:
            raise ValueError(f"cartesian_formula must be one of {_READOUT_FORMULAS}")
        if antisymmetric_channels and cartesian_formula != "ij":
            raise ValueError("antisymmetric branch requires cartesian_formula='ij'")
        if reduce not in _REDUCTIONS:
            raise ValueError(f"reduce must be one of {_REDUCTIONS}")
        if asr not in _ASR_MODES:
            raise ValueError(f"asr must be one of {_ASR_MODES}")
        if total_reduce not in ("sum", "mean") or (
            asr == "soft" and total_reduce != "mean"
        ):
            raise ValueError("total_reduce must be mean or sum; soft asr requires mean")
        edge_symmetric_field = edge_symmetric_field or f"edge_{out_field}_symmetric"
        edge_antisymmetric_field = (
            edge_antisymmetric_field or f"edge_{out_field}_antisymmetric"
        )
        self.tensor_field = tensor_field
        self.scalar_field = scalar_field
        self.tensor_basis_field = tensor_basis_field
        self.edge_symmetric_field = edge_symmetric_field
        self.edge_antisymmetric_field = edge_antisymmetric_field
        self.num_tensor_features = int(num_tensor_features)
        if self.num_tensor_features < 1:
            raise ValueError("num_tensor_features must be positive")
        required = [tensor_field] + (
            [scalar_field, tensor_basis_field] if gated_convolution else []
        )
        irreps_in = {} if irreps_in is None else irreps_in
        for field in required:
            if irreps_in.get(field, None) is None:
                raise ValueError(
                    f"TensorFeatureCartesianReadout needs the irreps of field {field!r}"
                )
        self.tensor_irreps = o3.Irreps(irreps_in[tensor_field])
        if any(mul_ir.mul != 1 for mul_ir in self.tensor_irreps):
            raise ValueError(
                f"{tensor_field!r} must be registered with per-channel irreps "
                f"(multiplicity 1 each), got {self.tensor_irreps}"
            )
        self.symmetric_irreps = _lift_irreps(cartesian_formula)
        if cartesian_formula != "ij":
            for _, output in self.symmetric_irreps:
                if not any(
                    output in ir1 * ir2
                    for _, ir1 in self.tensor_irreps
                    for _, ir2 in self.tensor_irreps
                ):
                    raise ValueError(f"No tensor-product path to lift output {output}")
        F = self.num_tensor_features

        # Construction order follows the reference head (global-RNG order).
        self.tensor_product = GatherContracter(
            irreps_in1=self.tensor_irreps,
            irreps_in2=self.tensor_irreps,
            irreps_out=self.symmetric_irreps,
            mul=F,
            path_channel_coupling=True,
            memory_efficient=memory_efficient_contraction,
        )
        self.channel_weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.full((F,), 1.0 / math.sqrt(F)))
                for _ in self.symmetric_irreps
            ]
        )
        self.nonscalar_activation = self._norm_activation(self.symmetric_irreps)
        self.final_linear = o3.Linear(
            self.symmetric_irreps,
            self.symmetric_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        self.hidden_channels = None if not hidden_channels else int(hidden_channels)
        if self.hidden_channels is not None:
            H = self.hidden_channels
            self.hidden_irreps = o3.Irreps(
                [(mul_ir.mul * H, mul_ir.ir) for mul_ir in self.symmetric_irreps]
            )
            self.hidden_channel_weights = torch.nn.ParameterList()
            for _ in self.symmetric_irreps:
                weights = torch.nn.Parameter(torch.empty(H, F))
                torch.nn.init.xavier_uniform_(weights)
                self.hidden_channel_weights.append(weights)
            self.hidden_nonscalar_activation = self._norm_activation(self.hidden_irreps)
            self.hidden_linear = o3.Linear(
                self.hidden_irreps,
                self.hidden_irreps,
                internal_weights=True,
                shared_weights=True,
            )
            self.hidden_post_activation = self._norm_activation(self.hidden_irreps)
            self.hidden_final_linear = o3.Linear(
                self.hidden_irreps,
                self.symmetric_irreps,
                internal_weights=True,
                shared_weights=True,
            )
            _zero_linear(self.hidden_final_linear)

        if gated_convolution:
            if any(mul_ir.ir.l != 0 for mul_ir in o3.Irreps(irreps_in[scalar_field])):
                raise ValueError(f"{scalar_field!r} must hold scalars only")
            self.tensor_convolution = GatedTensorFeatureConvolution(
                tensor_irreps=self.tensor_irreps,
                num_tensor_features=F,
                tensor_basis_irreps=o3.Irreps(irreps_in[tensor_basis_field]),
                scalar_dim=o3.Irreps(irreps_in[scalar_field]).dim,
                normalize_by_degree=conv_normalize_by_degree,
                gate_init=gate_init,
                memory_efficient_contraction=memory_efficient_contraction,
            )
        else:
            self.tensor_convolution = None

        self.antisymmetric_channels = (
            None if not antisymmetric_channels else int(antisymmetric_channels)
        )
        fields = {edge_symmetric_field: cartesian_formula}
        irreps_out = {edge_symmetric_field: self.symmetric_irreps}
        if self.antisymmetric_channels is not None:
            A = self.antisymmetric_channels
            self.antisymmetric_irreps = _cartesian_irreps("ij=-ji")
            self.antisymmetric_left_mix = torch.nn.Parameter(torch.empty(A, F))
            self.antisymmetric_right_mix = torch.nn.Parameter(torch.empty(A, F))
            torch.nn.init.xavier_uniform_(self.antisymmetric_left_mix)
            torch.nn.init.xavier_uniform_(self.antisymmetric_right_mix)
            self.antisymmetric_tensor_product = GatherContracter(
                irreps_in1=self.tensor_irreps,
                irreps_in2=self.tensor_irreps,
                irreps_out=self.antisymmetric_irreps,
                mul=A,
                path_channel_coupling=True,
                memory_efficient=memory_efficient_contraction,
            )
            self.antisymmetric_channel_weights = torch.nn.Parameter(
                torch.full((A,), 1.0 / math.sqrt(A))
            )
            self.antisymmetric_activation = self._norm_activation(
                self.antisymmetric_irreps
            )
            self.antisymmetric_linear = o3.Linear(
                self.antisymmetric_irreps,
                self.antisymmetric_irreps,
                internal_weights=True,
                shared_weights=True,
            )
            _zero_linear(self.antisymmetric_linear)
            fields[edge_antisymmetric_field] = "ij=-ji"
            irreps_out[edge_antisymmetric_field] = self.antisymmetric_irreps

        if zero_init_final_linear:
            _zero_linear(self.final_linear)

        self._init_irreps(
            irreps_in=irreps_in, required_irreps_in=required, irreps_out=irreps_out
        )
        self.cartesian_reduce = CartesianTensorReduce(
            cartesian_rank=_formula_rank(cartesian_formula),
            total_reduce=total_reduce,
            fields=fields,
            out_field=out_field,
            reduce=reduce,
            asr=asr,
            total_out_field=total_out_field,
            pre_asr_out_field=pre_asr_out_field,
            avg_num_neighbors=avg_num_neighbors,
            num_types=num_types,
            per_species_scale=per_species_scale,
            per_species_scale_trainable=per_species_scale_trainable,
            irreps_in=self.irreps_out,
        )
        self.irreps_out.update(self.cartesian_reduce.irreps_out)

    @staticmethod
    def _norm_activation(irreps: o3.Irreps) -> NormActivation:
        return NormActivation(
            irreps_in=irreps,
            scalar_nonlinearity=torch.nn.functional.silu,
            normalize=True,
            epsilon=1e-8,
            bias=False,
        )

    def _reduce_channels(self, edge_tensor: torch.Tensor) -> torch.Tensor:
        pieces = []
        for weights, tensor_slice in zip(
            self.channel_weights, self.symmetric_irreps.slices()
        ):
            pieces.append(
                torch.einsum("eud,u->ed", edge_tensor[:, :, tensor_slice], weights)
            )
        return torch.cat(pieces, dim=-1)

    def _hidden_readout(self, edge_tensor: torch.Tensor) -> torch.Tensor:
        pieces = []
        n_edges = edge_tensor.shape[0]
        for weights, tensor_slice, mul_ir in zip(
            self.hidden_channel_weights,
            self.symmetric_irreps.slices(),
            self.symmetric_irreps,
        ):
            piece = edge_tensor[:, :, tensor_slice].reshape(
                n_edges, self.num_tensor_features, mul_ir.mul, mul_ir.ir.dim
            )
            piece = torch.einsum("eumi,hu->ehmi", piece, weights)
            pieces.append(piece.reshape(n_edges, -1))
        hidden = torch.cat(pieces, dim=-1)
        hidden = self.hidden_nonscalar_activation(hidden)
        hidden = self.hidden_linear(hidden)
        hidden = self.hidden_post_activation(hidden)
        return self.hidden_final_linear(hidden)

    def _antisymmetric(self, tensor_features: torch.Tensor) -> torch.Tensor:
        left = torch.einsum("eud,hu->ehd", tensor_features, self.antisymmetric_left_mix)
        right = torch.einsum(
            "eud,hu->ehd", tensor_features, self.antisymmetric_right_mix
        )
        same_edge = torch.arange(
            tensor_features.shape[0], device=tensor_features.device
        )
        edge_tensor = self.antisymmetric_tensor_product(left, right, same_edge)
        edge_tensor = torch.einsum(
            "eud,u->ed", edge_tensor, self.antisymmetric_channel_weights
        )
        edge_tensor = self.antisymmetric_activation(edge_tensor)
        return self.antisymmetric_linear(edge_tensor)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        weight = self.tensor_product.weights
        tensor_features = data[self.tensor_field].to(
            device=weight.device, dtype=weight.dtype
        )
        if tensor_features.ndim != 3 or tuple(tensor_features.shape[1:]) != (
            self.num_tensor_features,
            self.tensor_irreps.dim,
        ):
            raise ValueError(
                f"{self.tensor_field!r} must have shape (n_edges, "
                f"{self.num_tensor_features}, {self.tensor_irreps.dim}), got "
                f"{tuple(tensor_features.shape)}"
            )
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        if self.tensor_convolution is not None:
            tensor_features = self.tensor_convolution(
                tensor_features,
                data[self.scalar_field].to(device=weight.device, dtype=weight.dtype),
                data[self.tensor_basis_field].to(
                    device=weight.device, dtype=weight.dtype
                ),
                edge_center,
                data[AtomicDataDict.POSITIONS_KEY].shape[0],
            )

        same_edge = torch.arange(
            tensor_features.shape[0], device=tensor_features.device
        )
        lifted = self.tensor_product(tensor_features, tensor_features, same_edge)
        symmetric = self.final_linear(
            self.nonscalar_activation(self._reduce_channels(lifted))
        )
        if self.hidden_channels is not None:
            symmetric = symmetric + self._hidden_readout(lifted)
        data[self.edge_symmetric_field] = symmetric
        if self.antisymmetric_channels is not None:
            data[self.edge_antisymmetric_field] = self._antisymmetric(tensor_features)
        return self.cartesian_reduce(data)


def _zero_linear(linear: o3.Linear) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias.numel() > 0:
            linear.bias.zero_()


class CartesianBECReduce(CartesianTensorReduce):
    """Backward-compatible rank-2 BEC reduction with mean frame totals."""

    def __init__(
        self,
        fields: Dict[str, str],
        out_field: str = AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY,
        reduce: str = "mean",
        asr: str = "soft",
        total_out_field: Optional[str] = AtomicDataDict.TOTAL_BORN_EFFECTIVE_CHARGE_KEY,
        pre_asr_out_field: Optional[str] = None,
        avg_num_neighbors: Optional[float] = None,
        num_types: Optional[int] = None,
        per_species_scale: Optional[Union[float, List[float]]] = None,
        per_species_scale_trainable: bool = False,
        per_species_shift: Optional[Union[float, List[float]]] = None,
        irreps_in=None,
    ):
        super().__init__(
            fields=fields,
            cartesian_rank=2,
            total_reduce="mean",
            out_field=out_field,
            reduce=reduce,
            asr=asr,
            total_out_field=total_out_field,
            pre_asr_out_field=pre_asr_out_field,
            avg_num_neighbors=avg_num_neighbors,
            num_types=num_types,
            per_species_scale=per_species_scale,
            per_species_scale_trainable=per_species_scale_trainable,
            per_species_shift=per_species_shift,
            irreps_in=irreps_in,
        )


class TensorFeatureBECReadout(TensorFeatureCartesianReadout):
    """BEC specialization: ``cartesian_formula="ij"``, hard ASR by default."""

    def __init__(
        self,
        tensor_field: str,
        num_tensor_features: int,
        scalar_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
        tensor_basis_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        out_field: str = AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY,
        total_out_field: Optional[str] = AtomicDataDict.TOTAL_BORN_EFFECTIVE_CHARGE_KEY,
        pre_asr_out_field: Optional[str] = None,
        edge_symmetric_field: str = "edge_bec_symmetric",
        edge_antisymmetric_field: str = "edge_bec_antisymmetric",
        gated_convolution: bool = True,
        conv_normalize_by_degree: bool = True,
        gate_init: float = 0.0,
        hidden_channels: Optional[int] = None,
        antisymmetric_channels: Optional[int] = None,
        zero_init_final_linear: bool = True,
        reduce: str = "mean",
        asr: str = "hard",
        avg_num_neighbors: Optional[float] = None,
        num_types: Optional[int] = None,
        per_species_scale: Optional[Union[float, List[float]]] = None,
        per_species_scale_trainable: bool = False,
        irreps_in=None,
        memory_efficient_contraction: bool = False,
    ):
        super().__init__(
            cartesian_formula="ij",
            total_reduce="mean",
            tensor_field=tensor_field,
            num_tensor_features=num_tensor_features,
            scalar_field=scalar_field,
            tensor_basis_field=tensor_basis_field,
            out_field=out_field,
            total_out_field=total_out_field,
            pre_asr_out_field=pre_asr_out_field,
            edge_symmetric_field=edge_symmetric_field,
            edge_antisymmetric_field=edge_antisymmetric_field,
            gated_convolution=gated_convolution,
            conv_normalize_by_degree=conv_normalize_by_degree,
            gate_init=gate_init,
            hidden_channels=hidden_channels,
            antisymmetric_channels=antisymmetric_channels,
            zero_init_final_linear=zero_init_final_linear,
            reduce=reduce,
            asr=asr,
            avg_num_neighbors=avg_num_neighbors,
            num_types=num_types,
            per_species_scale=per_species_scale,
            per_species_scale_trainable=per_species_scale_trainable,
            irreps_in=irreps_in,
            memory_efficient_contraction=memory_efficient_contraction,
        )
