"""BEC inference does not retain the graph needed only by BEC supervision."""

import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.nn._grad_output import ParaStressBECOutput


class _PolarPolynomial(GraphModuleMixin, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7))
        self._init_irreps(
            irreps_in={AtomicDataDict.POSITIONS_KEY: "1o"},
            irreps_out={
                AtomicDataDict.TOTAL_ENERGY_KEY: "0e",
                AtomicDataDict.TOTAL_DIPOLE_KEY: "1o",
            },
        )

    def forward(self, data):
        positions = data[AtomicDataDict.POSITIONS_KEY]
        vectors = data[AtomicDataDict.EDGE_VECTORS_KEY]
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = self.weight * (
            positions.square().sum() + vectors.square().sum()
        ).reshape(1, 1)
        data[AtomicDataDict.TOTAL_DIPOLE_KEY] = self.weight * positions.pow(3).sum(
            dim=0, keepdim=True
        )
        return data


@pytest.mark.parametrize("training", [False, True])
def test_bec_retains_graph_only_for_supervision(training):
    core = _PolarPolynomial()
    model = ParaStressBECOutput(core).train(training)
    positions = torch.tensor([[0.2, 0.3, 0.4], [1.0, 0.6, 0.8]])
    data = {
        AtomicDataDict.POSITIONS_KEY: positions,
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]]),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.zeros(2, 3),
        AtomicDataDict.CELL_KEY: torch.eye(3) * 10,
        AtomicDataDict.PBC_KEY: torch.ones(3, dtype=torch.bool),
        AtomicDataDict.BATCH_KEY: torch.zeros(2, dtype=torch.long),
        AtomicDataDict.NUM_BATCH_KEY: torch.tensor(1),
    }
    output = model(data)
    bec = output[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY]
    expected = torch.diag_embed(3 * core.weight * positions.square()).reshape(2, 9)
    torch.testing.assert_close(bec, expected)
    assert bec.requires_grad is training
    if training:
        bec.square().sum().backward()
        torch.testing.assert_close(
            core.weight.grad, 18 * core.weight * positions.pow(4).sum()
        )
