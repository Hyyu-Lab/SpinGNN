import numpy as np
import torch
from ase import Atoms

from nequip.ase import NequIPCalculator
from nequip.data import AtomicDataDict


class CustomPropertyModel(torch.nn.Module):
    def forward(self, data):
        dtype = data[AtomicDataDict.POSITIONS_KEY].dtype
        device = data[AtomicDataDict.POSITIONS_KEY].device
        num_atoms = data[AtomicDataDict.POSITIONS_KEY].shape[0]
        return {
            AtomicDataDict.SPIN_FORCE_KEY: torch.ones(
                (num_atoms, 3), dtype=dtype, device=device
            ),
            AtomicDataDict.SPIN_FORCE_VERT_KEY: torch.ones(
                (num_atoms, 3), dtype=dtype, device=device
            ),
            AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY: torch.ones(
                (num_atoms, 9), dtype=dtype, device=device
            ),
            AtomicDataDict.PIEZOELECTRIC_KEY: torch.ones(
                (1, 3, 3, 3), dtype=dtype, device=device
            ),
            AtomicDataDict.ATOMIC_DIPOLE_KEY: torch.ones(
                (num_atoms, 3), dtype=dtype, device=device
            ),
            AtomicDataDict.TOTAL_DIPOLE_KEY: torch.ones(
                (1, 3), dtype=dtype, device=device
            ),
        }


def make_calculator(**kwargs):
    return NequIPCalculator(
        model=CustomPropertyModel(),
        r_max=2.0,
        device="cpu",
        **kwargs,
    )


def test_custom_properties_use_charge_and_length_units():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]])
    calculator = make_calculator(
        energy_units_to_eV=7.0,
        length_units_to_A=3.0,
        charge_units_to_e=2.5,
    )

    for property_name in [
        "spin_forces",
        "spin_forces_vert",
        "born_effective_charges",
        "piezoelectric",
        "atomic_dipole",
        "dipole",
    ]:
        assert property_name in calculator.implemented_properties
        calculator.get_property(property_name, atoms)

    assert np.allclose(calculator.results["spin_forces"], 7.0)
    assert np.allclose(calculator.results["spin_forces_vert"], 7.0)
    assert np.allclose(calculator.results["born_effective_charges"], 2.5)
    assert np.allclose(calculator.results["piezoelectric"], 7.5)
    assert np.allclose(calculator.results["atomic_dipole"], 7.5)
    assert np.allclose(calculator.results["dipole"], 7.5)


def test_custom_property_default_units_are_unchanged():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]])
    calculator = make_calculator()

    calculator.calculate(atoms)

    for property_name in [
        "spin_forces",
        "spin_forces_vert",
        "born_effective_charges",
        "piezoelectric",
        "atomic_dipole",
        "dipole",
    ]:
        assert np.allclose(calculator.results[property_name], 1.0)
