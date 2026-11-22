import pytest

from nequip.data import AtomicDataDict
from nequip.utils.unittests.model_tests import BaseEnergyModelTests

from _class_scoped_fixtures import ClassScopedModelFixtures

COMMON_CONFIG = {
    "avg_num_neighbors": 5.0,  # very approximate to keep numerics sane
    "r_max": 4,
    "num_types": 3,
    "type_names": ["H", "C", "O"],
    "env_embed_multiplicity": 4,
    "two_body_latent_mlp_latent_dimensions": [32],
    "latent_mlp_latent_dimensions": [32],
    "env_embed_mlp_latent_dimensions": [],
    "edge_eng_mlp_latent_dimensions": [16],
    # Layer-prefixed key for the builder's spin-length head (ScalarMLP).
    "node_spin_length_mlp_latent_dimensions": [16],
}
minimal_config1 = dict(
    l_max=2,
    parity="o3_full",
    num_layers=2,
    **COMMON_CONFIG,
)


class TestAllegro_magmom(ClassScopedModelFixtures, BaseEnergyModelTests):
    @pytest.fixture
    def strict_locality(self):
        return True

    @pytest.fixture(
        params=[
            minimal_config1,
        ],
        scope="class",
    )
    @classmethod
    def base_config(cls, request):
        return request.param

    @pytest.fixture(
        params=[
            (
                ["allegro.model.Allegro_magmom", "ForceOutput"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                    AtomicDataDict.EDGE_FEATURES_KEY,
                    AtomicDataDict.SPIN_LENGTH_KEY,
                ],
            ),
            (
                ["allegro.model.Allegro_magmom", "ParaStressForceOutput"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                    AtomicDataDict.STRESS_KEY,
                    AtomicDataDict.VIRIAL_KEY,
                    AtomicDataDict.SPIN_LENGTH_KEY,
                ],
            ),
        ],
        scope="class",
    )
    @classmethod
    def config(cls, request, base_config):
        config = base_config.copy()
        builder, out_fields = request.param
        config["model_builders"] = builder
        return config, out_fields
