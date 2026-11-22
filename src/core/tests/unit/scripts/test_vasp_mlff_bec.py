import numpy as np
import pytest
from ase import Atoms

from nequip.scripts.vasp_mlff_bec import judge_same_atoms


def make_atoms(*, pbc=True):
    return Atoms(
        "HHe",
        positions=[[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]],
        cell=np.diag([10.0, 11.0, 12.0]),
        pbc=pbc,
    )


def test_large_negative_position_delta_is_not_equal():
    atoms1 = make_atoms(pbc=False)
    atoms2 = atoms1.copy()
    atoms2.positions[0, 0] += 10.0

    assert atoms1.positions[0, 0] - atoms2.positions[0, 0] == -10.0
    assert not judge_same_atoms(atoms1, atoms2)


def test_periodic_wrapped_positions_are_equal():
    atoms1 = make_atoms(pbc=True)
    atoms2 = atoms1.copy()
    atoms2.positions[0] += atoms1.cell[0]
    atoms2.positions[1] -= atoms1.cell[2]

    assert judge_same_atoms(atoms1, atoms2)


@pytest.mark.parametrize("mismatch", ["cell", "species_order", "pbc"])
def test_structure_contract_rejects_mismatch(mismatch):
    atoms1 = make_atoms(pbc=True)
    atoms2 = atoms1.copy()

    if mismatch == "cell":
        atoms2.cell[0, 0] += 0.1
    elif mismatch == "species_order":
        atoms2.numbers[:] = atoms2.numbers[::-1]
    else:
        atoms2.pbc[2] = False

    assert not judge_same_atoms(atoms1, atoms2)
