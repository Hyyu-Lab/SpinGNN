import pytest

from nequip.data import AtomicDataDict
from nequip.utils.unittests.model_tests import BaseEnergySpinModelTests
from nequip.utils.unittests.parity_tests import (
    BaseCompileParityTests,
    BasePrecisionLadderTests,
)
from nequip.utils.test import assert_AtomicData_equivariant

from _class_scoped_fixtures import ClassScopedSpinModelFixtures

from typing import List

import numpy as np
from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator


@pytest.fixture(scope="session")
def molecules_spin() -> List[Atoms]:
    atoms_list = []
    for i in range(8):
        atoms = molecule("CH3CHO" if i % 2 == 0 else "H2")
        atoms.set_cell([100.0, 100.0, 100.0])
        atoms.set_pbc(True)
        atoms.set_initial_magnetic_moments(np.random.random((len(atoms), 3)))
        atoms.rattle()
        atoms.calc = SinglePointCalculator(
            energy=np.random.random(),
            forces=np.random.random((len(atoms), 3)),
            stress=None,
            atoms=atoms,
        )
        atoms_list.append(atoms)
    return atoms_list


COMMON_CONFIG = {
    "avg_num_neighbors": 5.0,  # very approximate to keep numerics sane
    "r_max": 4,
    "num_types": 3,
    "type_names": ["H", "C", "O"],
    "env_embed_multiplicity": 7,  # Have to be here, features debug more
    "two_body_latent_mlp_latent_dimensions": [32],
    "mlp_latent_dimensions": [32],
    "latent_mlp_latent_dimensions": [32],
    "env_embed_mlp_latent_dimensions": [],
    "edge_onlyr_eng_mlp_latent_dimensions": [16],
    "edge_biquad_coeff_mlp_latent_dimensions": [16],
    "edge_eng_soc_on_mlp_latent_dimensions": [16],
    "edge_onlyr_eng_coeff": 1.0,
    "edge_hb_ani_eng_coeff": 0.1,
    "edge_biquad_eng_coeff": 0.1,
    "edge_eng_soc_on_coeff": 0.01,
    # "nonscalars_include_parity": False,
}
minimal_config1 = dict(
    l_max=2,
    num_layers=2,
    lbiquadratic=True,
    lsoc_layer=False,
    symmetry="u2",
    **COMMON_CONFIG
)
minimal_config2 = dict(
    l_max=2,
    num_layers=2,
    symmetry="u2",
    lsoc_layer=True,
    ledge_cutoff=False,
    **COMMON_CONFIG
)
minimal_config3 = dict(
    l_max=2,
    num_layers=2,
    symmetry="u2",
    lsoc_layer=True,
    ldelta_learning=True,
    lpredict_spin_length=True,
    ledge_cutoff=False,
    **COMMON_CONFIG
)


class _AllegroSOCConfig(ClassScopedSpinModelFixtures):
    @pytest.fixture(
        params=[
            minimal_config1,
            minimal_config2,
            minimal_config3,
        ],
        scope="class",
    )
    @classmethod
    def base_config(cls, request):
        return request.param

    @pytest.fixture
    def strict_locality(self):
        return True

    @pytest.fixture(
        params=[
            (
                ["allegro.model.Allegro_soc", "ParaStressForceSpinOutput"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                    AtomicDataDict.STRESS_KEY,
                    AtomicDataDict.VIRIAL_KEY,
                    AtomicDataDict.SPIN_FORCE_KEY,
                ],
            ),
        ],
        scope="class",
    )
    @classmethod
    def config(cls, request, base_config):
        config = base_config.copy()
        builder, out_fields = request.param
        config = config.copy()
        config["model_builders"] = builder
        return config, out_fields


class TestAllegro_spin_soc(_AllegroSOCConfig, BaseEnergySpinModelTests):
    def test_equivariance(self, model_config, atomic_batch_spin, device):
        instance, out_field, config = model_config
        parity = config["symmetry"] in ("o3", "u2", "o3_full", "o3_restricted")
        time_reversal = config["symmetry"] in ("su2", "u2")
        soc = True

        instance = instance.to(device=device)
        atomic_batch_spin = atomic_batch_spin.to(device=device)
        assert_AtomicData_equivariant(
            func=instance,
            data_in=atomic_batch_spin,
            do_parity=parity,
            do_time_reversal=time_reversal,
            do_only_rot_spin=(not soc),
        )

    def test_jit(self, model, atomic_batch_spin, device):
        super().test_jit(model, atomic_batch_spin, device)

    def test_forward(self, model, atomic_batch_spin, device):
        super().test_forward(model, atomic_batch_spin, device)


class TestAllegroSOCParity(
    _AllegroSOCConfig, BaseCompileParityTests, BasePrecisionLadderTests
):
    pass


class TestAllegroSOCSpinLengthParity(_AllegroSOCConfig, BaseCompileParityTests):
    """Regression gate for an optional head unused by the force loss.

    With ``lpredict_spin_length=True`` and no delta energy head, the
    spin-length branch never reaches the force loss. The compiled graph must
    preserve eager's ``.grad is None`` for that head, otherwise weight decay
    and optimizer state differ. Separate gradient regions keep the head
    available for supervision without materializing unused gradients.
    """

    @pytest.fixture(
        params=[
            dict(minimal_config2, lpredict_spin_length=True),
        ],
        scope="class",
    )
    @classmethod
    def base_config(cls, request):
        return request.param
