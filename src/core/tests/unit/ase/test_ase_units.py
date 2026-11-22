"""ASE observables must obey the chain rule across model unit systems."""

import numpy as np
import pytest
import torch
from ase import Atoms
from types import SimpleNamespace

from nequip.ase import NequIPCalculator, NequIP_Latent_Opt_Calculator
from nequip.data import AtomicDataDict


class HarmonicModel(torch.nn.Module):
    """A geometric harmonic trap with known forces and strain derivative."""

    def forward(self, data):
        positions = data[AtomicDataDict.POSITIONS_KEY]
        energy_per_atom = 0.5 * positions.square().sum(dim=1)
        cell = data[AtomicDataDict.CELL_KEY].reshape(3, 3)
        self.inputs = {key: value.detach().clone() for key, value in data.items()}
        return {
            AtomicDataDict.TOTAL_ENERGY_KEY: energy_per_atom.sum(),
            AtomicDataDict.PER_ATOM_ENERGY_KEY: energy_per_atom[:, None],
            AtomicDataDict.FORCE_KEY: -positions,
            AtomicDataDict.STRESS_KEY: positions.T @ positions / cell.det().abs(),
        }


def make_atoms():
    atoms = Atoms(
        "H2",
        positions=[[0.2, -0.3, 0.5], [1.4, 0.4, -0.2]],
        cell=[6.0, 7.0, 8.0],
        pbc=False,
    )
    atoms.set_initial_magnetic_moments([[1.0, 0.2, -0.4], [-0.3, 0.6, 0.8]])
    return atoms


def make_calculator(**kwargs):
    return NequIPCalculator(HarmonicModel(), r_max=4.0, device="cpu", **kwargs)


@pytest.mark.parametrize("length", [1.0, 2.5, 0.4])
def test_energy_force_and_stress_use_model_length_units(length):
    atoms = make_atoms()
    atoms.calc = make_calculator(energy_units_to_eV=7.0, length_units_to_A=length)
    positions = atoms.positions
    expected_per_atom = 3.5 * np.square(positions / length).sum(axis=1)
    np.testing.assert_allclose(
        atoms.get_potential_energies(), expected_per_atom, rtol=2e-6
    )
    np.testing.assert_allclose(
        atoms.get_potential_energy(), expected_per_atom.sum(), rtol=2e-6
    )
    np.testing.assert_allclose(
        atoms.get_forces(), -7.0 * positions / length**2, rtol=2e-6
    )
    np.testing.assert_allclose(
        atoms.get_stress(voigt=False),
        7.0 * positions.T @ positions / (length**2 * atoms.get_volume()),
        rtol=2e-6,
    )


def test_forces_are_negative_energy_derivatives_in_ase_units():
    atoms = make_atoms()
    atoms.calc = make_calculator(
        energy_units_to_eV=7.0, length_units_to_A=2.5, spin_units_to_ub=3.0
    )
    force = atoms.get_forces()
    step = 1e-3 if torch.get_default_dtype() == torch.float32 else 1e-5
    numerical = np.empty_like(force)
    for atom in range(len(atoms)):
        for axis in range(3):
            energies = []
            for sign in (-1, 1):
                displaced = atoms.copy()
                displaced.positions[atom, axis] += sign * step
                displaced.calc = atoms.calc
                energies.append(displaced.get_potential_energy())
            numerical[atom, axis] = -(energies[1] - energies[0]) / (2 * step)
    rtol, atol = (
        (3e-3, 3e-4) if torch.get_default_dtype() == torch.float32 else (1e-8, 1e-10)
    )
    np.testing.assert_allclose(force, numerical, rtol=rtol, atol=atol)


def test_stress_matches_homogeneous_strain_energy_derivative():
    atoms = make_atoms()
    atoms.calc = make_calculator(energy_units_to_eV=7.0, length_units_to_A=2.5)
    stress = atoms.get_stress(voigt=False)
    step = 1e-3 if torch.get_default_dtype() == torch.float32 else 1e-5
    for axis in range(3):
        energies = []
        for sign in (-1, 1):
            strained = atoms.copy()
            deformation = np.eye(3)
            deformation[axis, axis] += sign * step
            strained.set_cell(atoms.cell @ deformation, scale_atoms=True)
            strained.calc = atoms.calc
            energies.append(strained.get_potential_energy())
        numerical = (energies[1] - energies[0]) / (2 * step * atoms.get_volume())
        rtol, atol = (
            (4e-3, 1e-5)
            if torch.get_default_dtype() == torch.float32
            else (1e-8, 1e-10)
        )
        np.testing.assert_allclose(stress[axis, axis], numerical, rtol=rtol, atol=atol)


@pytest.mark.parametrize("periodic", [False, True])
def test_neighbor_graph_and_cell_shifts_use_converted_cutoff(periodic):
    positions = [[0.0, 0.0, 0.0], [3.5 if periodic else 1.5, 0.0, 0.0]]
    atoms = Atoms(
        "H2", positions=positions, cell=[4.0, 6.0, 6.0], pbc=[periodic, False, False]
    )
    model = HarmonicModel()
    calculator = NequIPCalculator(model, r_max=1.0, device="cpu", length_units_to_A=2.0)
    calculator.calculate(atoms)
    data = model.inputs
    assert data[AtomicDataDict.EDGE_INDEX_KEY].shape[1] == 2
    src, dst = data[AtomicDataDict.EDGE_INDEX_KEY]
    vectors = (
        data[AtomicDataDict.POSITIONS_KEY][dst]
        - data[AtomicDataDict.POSITIONS_KEY][src]
        + data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] @ data[AtomicDataDict.CELL_KEY]
    )
    expected_length = 0.25 if periodic else 0.75
    np.testing.assert_allclose(vectors.norm(dim=1).numpy(), expected_length)
    np.testing.assert_allclose(
        data[AtomicDataDict.CELL_KEY].numpy(), atoms.cell.array / 2.0
    )


def test_unit_conversion_does_not_mutate_atoms_or_auxiliary_spin_length():
    atoms = make_atoms()
    moments = atoms.get_initial_magnetic_moments()
    atoms.arrays["spin_length"] = np.linalg.norm(moments, axis=1)
    original = {key: value.copy() for key, value in atoms.arrays.items()}
    original_cell = atoms.cell.array.copy()
    calculator = make_calculator(length_units_to_A=2.5, spin_units_to_ub=3.0)
    calculator.calculate(atoms)
    for key, values in original.items():
        np.testing.assert_array_equal(atoms.arrays[key], values)
    np.testing.assert_array_equal(atoms.cell.array, original_cell)
    np.testing.assert_allclose(
        calculator.model.inputs[AtomicDataDict.SPIN_LENGTH_KEY].numpy().reshape(-1),
        original["spin_length"] / 3.0,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        calculator.model.inputs[AtomicDataDict.SPIN_KEY].norm(dim=1).numpy(),
        1.0,
        rtol=2e-6,
    )


@pytest.mark.parametrize(
    "name",
    [
        "energy_units_to_eV",
        "length_units_to_A",
        "spin_units_to_ub",
        "charge_units_to_e",
    ],
)
@pytest.mark.parametrize(
    "value", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_unit_factors_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        make_calculator(**{name: value})


class EdgeLengthLatentModel(torch.nn.Module):
    """Expose a differentiable geometric feature through the latent API."""

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(func=self.forward)

    def forward(self, data):
        data = AtomicDataDict.with_edge_vectors(data)
        data[AtomicDataDict.EDGE_FEATURES_KEY] = (
            data[AtomicDataDict.EDGE_VECTORS_KEY].square().sum(dim=1, keepdim=True)
        )
        return data


def make_latent_atoms():
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[6.0, 6.0, 6.0],
        pbc=False,
    )
    atoms.calc = NequIP_Latent_Opt_Calculator(
        EdgeLengthLatentModel(),
        r_max=2.0,
        device="cpu",
        latent_target=np.zeros(1),
        energy_units_to_eV=3.0,
        length_units_to_A=2.0,
    )
    return atoms


def test_latent_features_use_model_length_units():
    atoms = make_latent_atoms()
    np.testing.assert_allclose(atoms.calc.generate_latent_vec(atoms), 0.25)


def test_latent_forces_match_ase_energy_derivative():
    atoms = make_latent_atoms()
    force = atoms.get_forces()[1, 0]
    step = 1e-3 if torch.get_default_dtype() == torch.float32 else 1e-5
    energies = []
    for sign in (-1, 1):
        displaced = atoms.copy()
        displaced.positions[1, 0] += sign * step
        displaced.calc = atoms.calc
        energies.append(displaced.get_potential_energy())
    numerical = -(energies[1] - energies[0]) / (2 * step)
    rtol, atol = (
        (3e-3, 3e-4) if torch.get_default_dtype() == torch.float32 else (1e-8, 1e-10)
    )
    np.testing.assert_allclose(force, numerical, rtol=rtol, atol=atol)
    np.testing.assert_allclose(force, -1.5, rtol=2e-6)
