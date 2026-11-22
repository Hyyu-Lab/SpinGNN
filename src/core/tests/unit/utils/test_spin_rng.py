"""Spin fixtures can draw reproducibly without consuming the global RNG."""

import numpy as np
import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.utils.test import add_spin


def test_add_spin_uses_local_generator():
    before = torch.random.get_rng_state().clone()
    outputs = []
    for _ in range(2):
        data = {AtomicDataDict.POSITIONS_KEY: torch.zeros(6, 3)}
        result = add_spin(
            [data],
            spin_precent=0.5,
            generator=torch.Generator().manual_seed(193),
        )[0]
        outputs.append(result[AtomicDataDict.SPIN_KEY])
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    torch.testing.assert_close(outputs[0][:3].norm(dim=-1), torch.ones(3))
    assert torch.count_nonzero(outputs[0][3:]) == 0


@pytest.mark.parametrize("fixture_name", ["molecules", "bulks"])
def test_atomic_fixtures_use_local_numpy_rng(fixture_name):
    from nequip.utils.unittests import conftest as shared

    before = np.random.get_state()
    fixture = getattr(shared, fixture_name).__wrapped__
    first, second = fixture(), fixture()
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    for reference, candidate in zip(first, second):
        np.testing.assert_array_equal(
            reference.get_positions(), candidate.get_positions()
        )
        np.testing.assert_array_equal(reference.get_forces(), candidate.get_forces())
        assert reference.get_potential_energy() == candidate.get_potential_energy()


def test_species_fixture_uses_local_torch_rng():
    from nequip.utils.unittests import conftest as shared

    before = torch.random.get_rng_state().clone()
    first = shared.per_species_set.__wrapped__()
    second = shared.per_species_set.__wrapped__()
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    for reference, candidate in zip(first[:3], second[:3]):
        torch.testing.assert_close(reference, candidate, rtol=0, atol=0)
