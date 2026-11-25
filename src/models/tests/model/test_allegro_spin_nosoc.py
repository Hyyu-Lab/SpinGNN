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
    "allegro_onlyr_num_layers": 2,
    "allegro_nosoc_num_layers": 2,
    "env_embed_multiplicity": 7,  # Have to be here, features debug more
    "two_body_latent_mlp_latent_dimensions": [32],
    "latent_mlp_latent_dimensions": [32],
    "env_embed_mlp_latent_dimensions": [],
    "edge_onlyr_eng_mlp_latent_dimensions": [16],
    "edge_hb_coeff_mlp_latent_dimensions": [16],
    "edge_biquad_coeff_mlp_latent_dimensions": [16],
    "edge_eng_nosoc_mlp_latent_dimensions": [16],
    "edge_onlyr_eng_coeff": 1.0,
    "edge_hb_eng_coeff": 0.1,
    "edge_eng_nosoc_coeff": 0.05,
    "parity": "o3_full",
    # "nonscalars_include_parity": False,
}
minimal_config1 = dict(
    l_max=1, lnosoc_layer=False, ene_coeff_trainable=True, **COMMON_CONFIG
)
minimal_config2 = dict(
    l_max=2, ene_coeff_trainable=False, lte3nn=False, **COMMON_CONFIG
)
minimal_config3 = dict(l_max=2, ene_coeff_trainable=False, lte3nn=True, **COMMON_CONFIG)


class _AllegroNoSOCConfig(ClassScopedSpinModelFixtures):
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
                ["allegro.model.Allegro_nosoc", "ParaStressForceSpinOutput"],
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


class TestAllegro_spin_nosoc(_AllegroNoSOCConfig, BaseEnergySpinModelTests):
    def test_equivariance(self, model_config, atomic_batch_spin, device):
        instance, out_field, config = model_config
        parity = config["parity"] in ("o3", "u2", "o3_full", "o3_restricted")
        time_reversal = True
        soc = False

        instance = instance.to(device=device)
        atomic_batch_spin = atomic_batch_spin.to(device=device)
        assert_AtomicData_equivariant(
            func=instance,
            data_in=atomic_batch_spin,
            do_parity=parity,
            do_time_reversal=time_reversal,
            do_only_rot_spin=(not soc),
        )

    # def test_ladder(self, model_config, atomic_batch_spin, device):
    #     # Debug motion only
    #     instance, out_field, config = model_config
    #     instance = instance.to(device=device)
    #     data = atomic_batch_spin.to(device=device)
    #     output = instance(AtomicData.to_AtomicDataDict(data))

    #     for i in range(5):
    #         instance = tune_model_coeff_nosoc(instance, i)
    #         print(i)


class TestAllegroNoSOCParity(
    _AllegroNoSOCConfig, BaseCompileParityTests, BasePrecisionLadderTests
):
    pass
