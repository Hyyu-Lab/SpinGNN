"""Deployed ASE models may use type names that are not chemical symbols."""

import pytest
import torch

from nequip.ase import NequIPCalculator
from nequip.scripts import deploy


@pytest.mark.parametrize(
    "names,mapping",
    [
        ("iron oxygen", {"Fe": "iron", "O": "oxygen"}),
        ("O Fe", {"Fe": "O", "O": "Fe"}),
    ],
)
def test_explicit_species_names_map_to_deployed_types(monkeypatch, names, mapping):
    monkeypatch.setattr(
        deploy,
        "load_deployed_model",
        lambda **kwargs: (
            torch.nn.Identity(),
            {deploy.R_MAX_KEY: "4.0", deploy.TYPE_NAMES_KEY: names},
        ),
    )
    calculator = NequIPCalculator.from_deployed_model(
        "unused-model.pth", species_to_type_name=mapping
    )
    order = names.split()
    assert calculator.transform.chemical_symbol_to_type == {
        species: order.index(name) for species, name in mapping.items()
    }
