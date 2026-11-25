import logging
from typing import Optional, List

import torch
import torch.nn.functional
from torch_runstats.scatter import scatter, scatter_mean

from e3nn.o3 import Linear
from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.data.transforms import TypeMapper
from ._graph_mixin import GraphModuleMixin


class AtomwiseOperation(GraphModuleMixin, torch.nn.Module):
    def __init__(self, operation, field: str, irreps_in=None):
        super().__init__()
        self.operation = operation
        self.field = field
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={field: operation.irreps_in},
            irreps_out={field: operation.irreps_out},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.field] = self.operation(data[self.field])
        return data


class AtomwiseLinear(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
        irreps_out=None,
    ):
        super().__init__()
        self.field = field
        out_field = out_field if out_field is not None else field
        self.out_field = out_field
        if irreps_out is None:
            irreps_out = irreps_in[field]

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out={out_field: irreps_out},
        )
        self.linear = Linear(
            irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.out_field] = self.linear(data[self.field])
        return data


class AtomwiseReduce(GraphModuleMixin, torch.nn.Module):
    constant: float

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        reduce="sum",
        avg_num_atoms=None,
        irreps_in={},
    ):
        super().__init__()
        assert reduce in ("sum", "mean", "normalized_sum")
        self.constant = 1.0
        if reduce == "normalized_sum":
            assert avg_num_atoms is not None
            self.constant = float(avg_num_atoms) ** -0.5
            reduce = "sum"
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.field]}
            if self.field in irreps_in
            else {},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_batch(data)
        if self.reduce == "sum":
            data[self.out_field] = scatter(
                data[self.field],
                data[AtomicDataDict.BATCH_KEY],
                dim=0,
                reduce=self.reduce,
            )
        else:
            data[self.out_field] = scatter_mean(
                data[self.field], data[AtomicDataDict.BATCH_KEY], dim=0
            )
        if self.constant != 1.0:
            data[self.out_field] = data[self.out_field] * self.constant
        return data


class AtomwisefeatureReduce(GraphModuleMixin, torch.nn.Module):
    constant: float

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        reduce="sum",
        avg_num_atoms=None,
        normalize: bool = False,
        center_normalize: bool = False,
        irreps_in={},
        constant=1.0,
    ):
        super().__init__()
        assert reduce in ("sum", "mean", "abssum", "squaredsum", "hybrid")
        self.constant = constant
        self.normalize = normalize
        self.center_normalize = center_normalize
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        irreps_out_temp = (
            o3.Irreps([(irreps_in[self.field][0].mul * 2, (0, 1))])
            if self.reduce == "hybrid"
            else irreps_in[self.field]
        )
        self._init_irreps(
            irreps_in=irreps_in,
            # irreps_out={self.out_field: irreps_in[self.field]}
            irreps_out={self.out_field: irreps_out_temp}
            if self.field in irreps_in
            else {},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_batch(data)
        if self.reduce == "sum":
            data[self.out_field] = scatter(
                data[self.field],
                data[AtomicDataDict.BATCH_KEY],
                dim=0,
                reduce="sum",
            )
        elif self.reduce == "mean":
            data[self.out_field] = scatter_mean(
                data[self.field], data[AtomicDataDict.BATCH_KEY], dim=0
            )
        elif self.reduce == "abssum":
            data[self.out_field] = scatter(
                torch.abs(data[self.field]),
                data[AtomicDataDict.BATCH_KEY],
                dim=0,
                reduce="sum",
            )
        elif self.reduce == "squaredsum":
            data[self.out_field] = scatter(
                torch.pow(data[self.field], 2),
                data[AtomicDataDict.BATCH_KEY],
                dim=0,
                reduce="sum",
            )
        elif self.reduce == "hybrid":
            # use hybrid method
            data[self.out_field] = torch.cat(
                (
                    scatter(
                        data[self.field],
                        data[AtomicDataDict.BATCH_KEY],
                        dim=0,
                        reduce="sum",
                    ),
                    scatter(
                        torch.pow(data[self.field], 2),
                        data[AtomicDataDict.BATCH_KEY],
                        dim=0,
                        reduce="sum",
                    ),
                ),
                dim=-1,
            )

        if self.normalize:
            data[self.out_field] = torch.nn.functional.normalize(
                data[self.out_field], p=2, dim=-1
            )
        if self.center_normalize:
            data[self.out_field] = data[self.out_field] - data[self.out_field].mean(
                dim=-2, keepdim=True
            )

        if self.constant != 1.0:
            data[self.out_field] = data[self.out_field] * self.constant
        return data


class PerSpeciesScaleShift(GraphModuleMixin, torch.nn.Module):
    """Scale and/or shift a predicted per-atom property based on (learnable) per-species/type parameters.

    Args:
        field: the per-atom field to scale/shift.
        num_types: the number of types in the model.
        shifts: the initial shifts to use, one per atom type.
        scales: the initial scales to use, one per atom type.
        arguments_in_dataset_units: if ``True``, says that the provided shifts/scales are in dataset
            units (in which case they will be rescaled appropriately by any global rescaling later
            applied to the model); if ``False``, the provided shifts/scales will be used without modification.

            For example, if identity shifts/scales of zeros and ones are provided, this should be ``False``.
            But if scales/shifts computed from the training data are used, and are thus in dataset units,
            this should be ``True``.
        out_field: the output field; defaults to ``field``.
    """

    field: str
    out_field: str
    scales_trainable: bool
    shifts_trainable: bool
    has_scales: bool
    has_shifts: bool

    def __init__(
        self,
        field: str,
        num_types: int,
        type_names: List[str],
        shifts: Optional[List[float]],
        scales: Optional[List[float]],
        arguments_in_dataset_units: bool,
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        irreps_in={},
    ):
        super().__init__()
        self.num_types = num_types
        self.type_names = type_names
        self.field = field
        self.out_field = f"shifted_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            # my_irreps_in={self.field: "0e"},  # input to shift must be a single scalar
            irreps_out={self.out_field: irreps_in[self.field]},
        )

        # TorchScript requires both tensor attributes on every instance. Keep
        # disabled identity buffers non-persistent to preserve checkpoint keys.
        self.has_shifts = shifts is not None
        self.shifts_trainable = shifts_trainable and self.has_shifts
        if self.has_shifts:
            shifts = torch.as_tensor(shifts, dtype=torch.get_default_dtype())
            if len(shifts.reshape([-1])) == 1:
                shifts = torch.ones(num_types) * shifts
            assert shifts.shape == (num_types,), f"Invalid shape of shifts {shifts}"
        else:
            shifts = torch.zeros(num_types, dtype=torch.get_default_dtype())
        if self.shifts_trainable:
            self.shifts = torch.nn.Parameter(shifts)
        else:
            self.register_buffer("shifts", shifts, persistent=self.has_shifts)

        self.has_scales = scales is not None
        self.scales_trainable = scales_trainable and self.has_scales
        if self.has_scales:
            scales = torch.as_tensor(scales, dtype=torch.get_default_dtype())
            if len(scales.reshape([-1])) == 1:
                scales = torch.ones(num_types) * scales
            assert scales.shape == (num_types,), f"Invalid shape of scales {scales}"
        else:
            scales = torch.ones(num_types, dtype=torch.get_default_dtype())
        if self.scales_trainable:
            self.scales = torch.nn.Parameter(scales)
        else:
            self.register_buffer("scales", scales, persistent=self.has_scales)

        self.arguments_in_dataset_units = arguments_in_dataset_units

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if not (self.has_scales or self.has_shifts):
            return data

        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY]
        in_field = data[self.field]
        assert len(in_field) == len(
            species_idx
        ), "in_field doesnt seem to have correct per-atom shape"
        if self.has_scales:
            in_field = self.scales[species_idx].view(-1, 1) * in_field
        if self.has_shifts:
            in_field = self.shifts[species_idx].view(-1, 1) + in_field
        data[self.out_field] = in_field
        return data

    def update_for_rescale(self, rescale_module):
        if hasattr(rescale_module, "related_scale_keys"):
            if self.out_field not in rescale_module.related_scale_keys:
                return
        if self.arguments_in_dataset_units and rescale_module.has_scale:
            logging.debug(
                f"PerSpeciesScaleShift's arguments were in dataset units; rescaling:\n  "
                f"Original scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
                f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
            )
            with torch.no_grad():
                if self.has_scales:
                    self.scales.div_(rescale_module.scale_by)
                if self.has_shifts:
                    self.shifts.div_(rescale_module.scale_by)
            logging.debug(
                f"  New scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
                f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
            )
