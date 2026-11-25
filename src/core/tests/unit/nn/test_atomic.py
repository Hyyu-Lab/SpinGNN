import pytest
import torch

from nequip.data import AtomicData, AtomicDataDict
from nequip.nn import (
    AtomwiseLinear,
    AtomwiseReduce,
    PerSpeciesScaleShift,
    SequentialGraphNetwork,
)
from nequip.nn.embedding import (
    OneHotAtomEncoding,
)
from nequip.utils.torch_geometric import Batch


@pytest.fixture(scope="class", params=[0, 1, 2])
def model(float_tolerance, request):
    zero_species = request.param
    shifts = [3, 5, 7]
    shifts[zero_species] = 0
    params = dict(
        num_types=3,
        type_names=["A", "B", "C"],
        total_shift=1.0,
        shifts=shifts,
    )
    return SequentialGraphNetwork.from_parameters(
        shared_params=params,
        layers={
            "one_hot": OneHotAtomEncoding,
            "linear": (
                AtomwiseLinear,
                dict(irreps_out="1x0e", out_field="e"),
            ),
            "shift": (
                PerSpeciesScaleShift,
                dict(
                    field="e",
                    out_field="shifted",
                    scales=1.0,
                    shifts=0.0,
                    arguments_in_dataset_units=False,
                ),
            ),
            "sum": (
                AtomwiseReduce,
                dict(reduce="sum", field="shifted", out_field="sum"),
            ),
        },
    )


@pytest.fixture(scope="class")
def batches(float_tolerance, nequip_dataset):
    b = []
    for idx in [[0], [1], [0, 1]]:
        b += [AtomicData.to_AtomicDataDict(Batch.from_data_list(nequip_dataset[idx]))]
    return b


def test_per_species_shift(nequip_dataset, batches, model):
    batch1, batch2, batch12 = batches
    result1 = model(batch1)
    result2 = model(batch2)
    result12 = model(batch12)

    assert torch.isclose(result1["sum"], result12["sum"][0])
    assert torch.isclose(result2["sum"], result12["sum"][1])
    print(result1["shifted"], result2["shifted"], result12["shifted"])


@pytest.mark.parametrize(
    "scales, shifts",
    [
        (None, None),
        ([2.0, 3.0], None),
        (None, [0.5, -0.25]),
        ([2.0, 3.0], [0.5, -0.25]),
    ],
    ids=["identity", "scale", "shift", "scale-shift"],
)
def test_per_species_scale_shift_torchscript(float_tolerance, scales, shifts):
    field = "per_atom_value"
    module = PerSpeciesScaleShift(
        field=field,
        out_field=field,
        num_types=2,
        type_names=["A", "B"],
        scales=scales,
        shifts=shifts,
        arguments_in_dataset_units=False,
        irreps_in={field: "1x0e"},
    )
    scripted = torch.jit.script(module)
    assert ("scales" in module.state_dict()) is (scales is not None)
    assert ("shifts" in module.state_dict()) is (shifts is not None)

    atom_types = torch.tensor([[0], [1], [0]], dtype=torch.long)
    values = torch.tensor([[1.0], [-2.0], [4.0]])
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.zeros((3, 3)),
        AtomicDataDict.EDGE_INDEX_KEY: torch.empty((2, 0), dtype=torch.long),
        AtomicDataDict.ATOM_TYPE_KEY: atom_types,
        field: values,
    }
    eager = module({key: value.clone() for key, value in data.items()})[field]
    scripted_out = scripted({key: value.clone() for key, value in data.items()})[field]

    expected = values.clone()
    flat_types = atom_types.view(-1)
    if scales is not None:
        expected = torch.as_tensor(scales)[flat_types].view(-1, 1) * expected
    if shifts is not None:
        expected = torch.as_tensor(shifts)[flat_types].view(-1, 1) + expected

    assert eager.dtype == torch.get_default_dtype()
    assert scripted_out.dtype == torch.get_default_dtype()
    torch.testing.assert_close(eager, expected)
    torch.testing.assert_close(scripted_out, eager)
