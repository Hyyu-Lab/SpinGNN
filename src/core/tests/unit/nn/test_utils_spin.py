import torch

from nequip.data import AtomicDataDict
from nequip.nn.embedding import OneHotAtomEncoding_spin
from nequip.nn import SequentialGraphNetwork, SaveForOutput, AtomwiseLinear


def test_spin_basic():
    sgn = SequentialGraphNetwork.from_parameters(
        shared_params={"num_types": 4, "include_spin": True},
        layers={
            "one_hot": OneHotAtomEncoding_spin,
            "save": (
                SaveForOutput,
                dict(field=AtomicDataDict.NODE_FEATURES_KEY, out_field="saved"),
            ),
            "linear": AtomwiseLinear,
        },
    )
    spin = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    spin[0] = 0.0
    out = sgn(
        {
            AtomicDataDict.POSITIONS_KEY: torch.randn(5, 3),
            AtomicDataDict.SPIN_KEY: spin,
            AtomicDataDict.EDGE_INDEX_KEY: torch.LongTensor([[0, 1], [1, 0]]),
            AtomicDataDict.ATOM_TYPE_KEY: torch.LongTensor([0, 0, 1, 2, 0]),
        }
    )
    saved = out["saved"]
    assert saved.shape == (5, 4 + 1)  # plus one dimension of spin norm
    assert torch.all(saved[0] == torch.as_tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(saved[:, -1], torch.as_tensor([0.0, 1.0, 1.0, 1.0, 1.0]))
