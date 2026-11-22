"""``Allegro_soc_delta`` is ``Allegro_soc`` without the structure-only
``edge_onlyr_eng`` head and without the ``ldelta_learning`` branch, so the SOC
checks and ``symmetry``-derived equivariance flags apply unchanged.
"""

import pytest

from nequip.data import AtomicDataDict
from nequip.utils.unittests.parity_samples import deterministic_spin_batch

import test_allegro_spin_soc as soc

# module fixtures resolve per file: re-export the one test_large_separation uses
molecules_spin = soc.molecules_spin


class _AllegroSOCDeltaConfig(soc._AllegroSOCConfig):
    @pytest.fixture(
        params=[
            soc.minimal_config1,
            soc.minimal_config2,
            # spin-only branch: no Allegro_spin_Module, the SOC layer embeds r
            dict(soc.minimal_config2, lonly_soc_layer=True),
        ],
        scope="class",
    )
    @classmethod
    def base_config(cls, request):
        return request.param

    @pytest.fixture(
        params=[
            (
                ["allegro.model.Allegro_soc_delta", "ParaStressForceSpinOutput"],
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


class TestAllegro_spin_soc_delta(_AllegroSOCDeltaConfig, soc.TestAllegro_spin_soc):
    pass


class TestAllegroSOCDeltaParity(_AllegroSOCDeltaConfig, soc.TestAllegroSOCParity):
    require_precision_signal = True

    @staticmethod
    def _batch_dict(atomic_batch_spin, device):
        return deterministic_spin_batch(atomic_batch_spin, device)
