"""``Allegro_nosoc_delta`` is ``Allegro_nosoc`` without the structure-only
``edge_onlyr_eng`` head, so the NoSOC configs, checks and equivariance flags
apply unchanged; only the builder differs.
"""

import pytest
from nequip.data import AtomicDataDict
from nequip.utils.unittests.parity_samples import deterministic_spin_batch

import test_allegro_spin_nosoc as nosoc

# module fixtures resolve per file: re-export the one test_large_separation uses
molecules_spin = nosoc.molecules_spin


class _AllegroNoSOCDeltaConfig(nosoc._AllegroNoSOCConfig):
    @pytest.fixture(
        params=[
            (
                ["allegro.model.Allegro_nosoc_delta", "ParaStressForceSpinOutput"],
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
        builder, out_fields = request.param
        return dict(base_config, model_builders=builder), out_fields


class TestAllegro_spin_nosoc_delta(
    _AllegroNoSOCDeltaConfig, nosoc.TestAllegro_spin_nosoc
):
    pass


class TestAllegroNoSOCDeltaParity(
    _AllegroNoSOCDeltaConfig, nosoc.TestAllegroNoSOCParity
):
    require_precision_signal = True

    @staticmethod
    def _batch_dict(atomic_batch_spin, device):
        return deterministic_spin_batch(atomic_batch_spin, device)
