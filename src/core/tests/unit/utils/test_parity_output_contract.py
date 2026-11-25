"""Parity gates must detect dropped outputs and broadcast-compatible shape drift."""

import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.utils.unittests.parity_tests import _BaseParityTests


@pytest.mark.parametrize(
    "field", [AtomicDataDict.STRESS_KEY, AtomicDataDict.SPIN_LENGTH_KEY]
)
def test_parity_rejects_dropped_optional_output(field):
    reference = {
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.ones(1, 1),
        AtomicDataDict.FORCE_KEY: torch.ones(2, 3),
        field: torch.ones(2, 1),
    }
    candidate = {key: value for key, value in reference.items() if key != field}
    with pytest.raises(AssertionError, match="output presence"):
        _BaseParityTests._compare_outputs(reference, candidate, 1e-5)


def test_parity_rejects_output_broadcasting():
    reference = {
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.ones(1, 1),
        AtomicDataDict.FORCE_KEY: torch.ones(2, 3),
    }
    candidate = dict(reference, **{AtomicDataDict.FORCE_KEY: torch.ones(1, 3)})
    with pytest.raises(AssertionError, match=AtomicDataDict.FORCE_KEY):
        _BaseParityTests._compare_outputs(reference, candidate, 1e-5)
