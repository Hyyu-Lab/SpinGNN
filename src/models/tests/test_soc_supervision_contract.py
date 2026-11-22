from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from nequip.data import AtomicData, AtomicDataDict
from nequip.train import Loss, Metrics
from nequip.utils import Config


REPO_ROOT = Path(__file__).resolve().parents[3]
SOC_CONFIGS = (
    REPO_ROOT / "src/models/configs/spin_soc_config/fast.yaml",
    REPO_ROOT / "example/02_spin_nosoc.yaml",
    REPO_ROOT / "example/03_spin_soc.yaml",
)


@pytest.mark.parametrize("config_path", SOC_CONFIGS, ids=lambda path: path.name)
def test_oszicar_spin_force_contract_reaches_loss_and_metrics(config_path):
    atoms = Atoms(
        "CrI",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    atoms.set_array(
        AtomicDataDict.SPIN_FORCE_VERT_KEY,
        np.asarray([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]),
    )
    reference = AtomicData.to_AtomicDataDict(AtomicData.from_ase(atoms, r_max=2.0))

    assert AtomicDataDict.SPIN_FORCE_VERT_KEY in reference
    assert AtomicDataDict.SPIN_FORCE_KEY not in reference

    config = Config.from_file(str(config_path))
    loss_coeffs = config["loss_coeffs"]
    metric_components = config["metrics_components"]

    assert AtomicDataDict.SPIN_FORCE_VERT_KEY in loss_coeffs
    assert AtomicDataDict.SPIN_FORCE_KEY not in loss_coeffs
    spin_metrics = [
        component
        for component in metric_components
        if component[0] == AtomicDataDict.SPIN_FORCE_VERT_KEY
    ]
    assert spin_metrics
    assert all(
        component[0] != AtomicDataDict.SPIN_FORCE_KEY for component in metric_components
    )

    prediction = {
        AtomicDataDict.SPIN_FORCE_VERT_KEY: torch.zeros_like(
            reference[AtomicDataDict.SPIN_FORCE_VERT_KEY]
        )
    }
    loss, contributions = Loss(
        {
            AtomicDataDict.SPIN_FORCE_VERT_KEY: loss_coeffs[
                AtomicDataDict.SPIN_FORCE_VERT_KEY
            ]
        }
    )(prediction, reference)
    metrics = Metrics(spin_metrics)(prediction, reference)

    assert torch.isfinite(loss)
    assert AtomicDataDict.SPIN_FORCE_VERT_KEY in contributions
    assert metrics
    assert all(torch.isfinite(value).all() for value in metrics.values())
