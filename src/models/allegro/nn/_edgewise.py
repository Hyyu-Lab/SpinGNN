from typing import Optional
import math
import torch
from torch_runstats.scatter import scatter

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.nn.cutoffs import PolynomialCutoff
from e3nn.o3 import Irreps


class EdgewiseReduce(GraphModuleMixin, torch.nn.Module):
    """Like ``nequip.nn.AtomwiseReduce``, but accumulating per-edge data into per-atom data."""

    _factor: Optional[float]

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        normalize_edge_reduce: bool = True,
        avg_num_neighbors: Optional[float] = None,
        reduce="sum",
        irreps_in={},
    ):
        """Sum edges into nodes."""
        super().__init__()
        assert reduce in ("sum", "mean", "min", "max")
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.field]}
            if self.field in irreps_in
            else {},
        )
        self._factor = None
        if normalize_edge_reduce and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get destination nodes 🚂
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        out = scatter(
            data[self.field],
            edge_dst,
            dim=0,
            dim_size=len(data[AtomicDataDict.POSITIONS_KEY]),
            reduce=self.reduce,
        )

        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            out = out * factor

        data[self.out_field] = out

        return data


class EdgewiseEnergySum(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise energies.

    Includes optional per-species-pair edgewise energy scales.
    """

    _factor: Optional[float]

    def __init__(
        self,
        num_types: int,
        avg_num_neighbors: Optional[float] = None,
        normalize_edge_energy_sum: bool = True,
        per_edge_species_scale: bool = False,
        ledge_cutoff: bool = False,
        cutoff=PolynomialCutoff,
        cutoff_kwargs={},
        lrescale_spin: bool = True,
        ldelta_spin: bool = False,
        irreps_in={},
        edgewise_feature_sum: bool = False,  # for rnd workflow, sum feature for the structure
        edgewise_feature_dim: int = 0,
    ):
        """Sum edges into nodes."""
        super().__init__()
        if not edgewise_feature_sum:
            my_irreps_in = {AtomicDataDict.EDGE_ENERGY_KEY: "0e"}
            irreps_out = {AtomicDataDict.PER_ATOM_ENERGY_KEY: "0e"}
        else:
            my_irreps_in = {}
            irreps_out = {
                AtomicDataDict.PER_ATOM_ENERGY_KEY: str(edgewise_feature_dim) + "x0e"
            }
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in=my_irreps_in,
            irreps_out=irreps_out,
        )
        self.edgewise_feature_sum = edgewise_feature_sum
        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

        self.per_edge_species_scale = per_edge_species_scale
        if self.per_edge_species_scale:
            self.per_edge_scales = torch.nn.Parameter(torch.ones(num_types, num_types))
        else:
            self.register_buffer("per_edge_scales", torch.Tensor())
        self.ledge_cutoff = ledge_cutoff
        self.cutoff = cutoff(**cutoff_kwargs)
        # self.register_buffer("_empty", torch.Tensor())
        self.rescale_key = [
            AtomicDataDict.EDGE_HB_COEFF_KEY,
            AtomicDataDict.EDGE_BIQUAD_COEFF_KEY,
            AtomicDataDict.EDGE_SPIN_MATRIX_KEY,
            AtomicDataDict.NODE_SPINANI_MATRIX_KEY,
            AtomicDataDict.EDGE_HB_MATRIX_KEY,
        ]
        self.lrescale_spin = lrescale_spin
        self.ldelta_spin = ldelta_spin

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        edge_eng = data[AtomicDataDict.EDGE_ENERGY_KEY]
        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        center_species = species[edge_center]
        neighbor_species = species[edge_neighbor]

        if self.per_edge_species_scale:
            edge_eng = edge_eng * self.per_edge_scales[
                center_species, neighbor_species
            ].unsqueeze(-1)

        if self.ledge_cutoff:
            data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
            edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
            coeff = self.cutoff(edge_length).unsqueeze(-1)
            edge_eng = edge_eng * coeff
            for key in self.rescale_key:
                if key in data:
                    data[key] = data[key] * coeff

        if self.ldelta_spin:
            data = AtomicDataDict.with_edge_sisj(data)
            spin = torch.nn.functional.normalize(
                data[AtomicDataDict.SPIN_NONLEAF_KEY], dim=-1
            )
            spin_norm = torch.sum(spin * spin, dim=-1).unsqueeze(-1)
            edge_sisj = torch.sum(spin[edge_center] * spin[edge_neighbor], dim=-1).view(
                -1, 1
            )
            coeff = (
                (1.0 - edge_sisj) * spin_norm[edge_center] * spin_norm[edge_neighbor]
            )
            edge_eng = edge_eng * coeff
            for key in self.rescale_key:
                if key in data:
                    data[key] = data[key] * coeff

        if self.edgewise_feature_sum:
            atom_eng = scatter(edge_eng, edge_center, dim=0, dim_size=len(species))
            # edge_center_num = scatter(torch.ones_like(edge_center), edge_center, dim=0, dim_size=len(species))

            # atom_eng = atom_eng / edge_center_num.reshape(-1, 1)
            # logging.info(f"edge_center_num: {edge_center_num.shape}, edge_center: {edge_center[-30:]}, atom_eng: {atom_eng.shape}")
        else:
            atom_eng = scatter(edge_eng, edge_center, dim=0, dim_size=len(species))
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            atom_eng = atom_eng * factor
            if self.lrescale_spin:
                for key in self.rescale_key:
                    if key in data:
                        data[key] = data[key] * factor
                        if key == AtomicDataDict.NODE_SPINANI_MATRIX_KEY:
                            data[key] = scatter(data[key], edge_center, dim=0)
        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = atom_eng
        return data


class Edge_ETGNN(GraphModuleMixin, torch.nn.Module):
    """"""

    _factor: Optional[float]

    def __init__(
        self,
        field: str = AtomicDataDict.ATOMIC_DIPOLE_KEY,
        out_field: Optional[str] = AtomicDataDict.ATOMIC_DIPOLE_KEY,
        irreps_in={},
    ):
        """Sum edges into nodes."""
        super().__init__()
        self.field = field
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in, irreps_out={self.out_field: Irreps("1o")}
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get destination nodes 🚂

        data[self.out_field] = data[self.field] * data[AtomicDataDict.EDGE_VECTORS_KEY]

        return data
