"""Convert extensive frame responses to volume-normalized responses."""

import torch

from nequip.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin


class VolumeNormalization(GraphModuleMixin, torch.nn.Module):
    """Divide a flat frame tensor by the absolute cell volume, with no constants.

    Cells may be unbatched or batched. Volumes at or below 1e-12 in the
    supplied length units cubed are rejected, as are nonfinite volumes.
    """

    def __init__(self, field: str, out_field: str, irreps_in=None):
        super().__init__()
        self.field = field
        self.out_field = out_field
        self._init_irreps(irreps_in=irreps_in, required_irreps_in=[field])
        self.irreps_out[out_field] = self.irreps_in[field]

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        cell = data[AtomicDataDict.CELL_KEY].reshape(-1, 3, 3)
        volume = torch.sum(
            cell[:, 0] * torch.cross(cell[:, 1], cell[:, 2], dim=-1), dim=-1
        ).abs()
        bad = (~torch.isfinite(volume)) | (volume <= 1e-12)
        if bool(bad.any()):
            raise RuntimeError("VolumeNormalization needs finite, nonzero cell volume")
        target = data[self.field]
        if target.dim() != 2 or target.shape[0] != volume.shape[0]:
            raise RuntimeError(
                "VolumeNormalization frame-count mismatch or non-flat target"
            )
        volume = volume.to(dtype=target.dtype, device=target.device)
        data[self.out_field] = target / volume.unsqueeze(-1)
        return data
