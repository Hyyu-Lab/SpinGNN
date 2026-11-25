import pytest
import torch

from nequip.data import AtomicData, AtomicDataDict
from nequip.model import model_from_config
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
}
minimal_config1 = dict(
    l_max=2,
    parity="o3_full",
    num_layers=2,
    **COMMON_CONFIG,
)


class TestAllegro_dipole(ClassScopedModelFixtures, BaseEnergyModelTests):
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
                ["allegro.model.Allegro_dipole", "ForceOutput"],
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
                ["allegro.model.Allegro_dipole", "ParaStressForceOutput"],
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


def test_dipole_so3_rejected():
    # so3 restricts features to natural-parity irreps, leaving no path for the
    # 1o dipole head; the builder must reject the combination with a clear
    # error instead of silently returning an all-zero dipole.
    config = dict(
        minimal_config1,
        parity="so3",
        model_builders=["allegro.model.Allegro_dipole", "ForceOutput"],
    )
    with pytest.raises(ValueError, match="so3"):
        model_from_config(config, initialize=True)


def test_bec_from_dipole(atomic_batch, float_tolerance):
    config = dict(
        minimal_config1,
        model_builders=["allegro.model.Allegro_dipole", "ParaStressForceBECOutput"],
    )
    model = model_from_config(config, initialize=True)
    data = AtomicData.to_AtomicDataDict(atomic_batch)
    bec = model(data)[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY]
    assert bec.shape == (len(data[AtomicDataDict.POSITIONS_KEY]), 9)
    # The dipole depends only on edge vectors, so BECs obey the acoustic sum rule.
    assert torch.allclose(bec.sum(dim=0), torch.zeros(9), atol=float(float_tolerance))
