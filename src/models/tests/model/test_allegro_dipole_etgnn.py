import pytest
import torch

from nequip.data import AtomicData, AtomicDataDict
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
    # Layer-prefixed key for the builder's per-edge dipole magnitude head (ScalarMLP).
    "edge_dipole_mlp_mlp_latent_dimensions": [16],
}
minimal_config1 = dict(
    l_max=2,
    parity="o3_full",
    num_layers=2,
    **COMMON_CONFIG,
)


class TestAllegro_dipole_etgnn(ClassScopedModelFixtures, BaseEnergyModelTests):
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
                ["allegro.model.Allegro_dipole_etgnn", "ForceOutput"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                    AtomicDataDict.EDGE_FEATURES_KEY,
                    AtomicDataDict.ATOMIC_DIPOLE_KEY,
                    AtomicDataDict.TOTAL_DIPOLE_KEY,
                ],
            ),
            (
                ["allegro.model.Allegro_dipole_etgnn", "ParaStressForceOutput"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                    AtomicDataDict.STRESS_KEY,
                    AtomicDataDict.VIRIAL_KEY,
                    AtomicDataDict.ATOMIC_DIPOLE_KEY,
                    AtomicDataDict.TOTAL_DIPOLE_KEY,
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

    def test_dipole_outputs(self, model, atomic_batch, device, float_tolerance):
        instance, _ = model
        out = instance(AtomicData.to_AtomicDataDict(atomic_batch.to(device=device)))
        atomic = out[AtomicDataDict.ATOMIC_DIPOLE_KEY]
        # An all-zero dipole would make the shared 1o equivariance check vacuous.
        assert atomic.abs().max() > 0
        total = out[AtomicDataDict.TOTAL_DIPOLE_KEY]
        summed = torch.zeros_like(total).index_add_(
            0, out[AtomicDataDict.BATCH_KEY], atomic
        )
        assert torch.allclose(total, summed, atol=float(float_tolerance))
