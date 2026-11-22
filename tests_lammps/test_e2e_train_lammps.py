"""End-to-end pipeline tests: dataset -> training -> deploy -> LAMMPS.

These are the only tests that exercise the full production pipeline as one
chain, one per supported Allegro line (nosoc, soc, bec; the NequIP-spin line
is deliberately out of scope):

    build labelled extxyz dataset (vector magmoms / spin_forces / BEC)
      -> ``nequip-train`` subprocess (real config file, real epochs)
      -> ``nequip-deploy build --train-dir`` on the training output
      -> the trained artifact evaluated by LAMMPS versus Python

Everything else in the suite covers the two halves separately: the
``src/core`` integration tests train on non-spin aspirin data and never
reach LAMMPS, while ``tests_lammps`` deliberately uses untrained random-weight
models (see ``model_factory.py``). The claim added here is that a checkpoint
produced by the actual trainer on actual spin/BEC data survives deployment and
matches in LAMMPS -- covering dataset ingestion of ``initial_magmoms`` /
``spin_forces`` / ``born_effective_charges`` (extxyz roundtrip asserted),
trainer state, and the train-dir deployment path, none of which the
random-weight factory touches.

Labels come from a frozen teacher: the deterministic random-weight deployment
model of the same case. That makes the dataset physically consistent and
learnable without shipping reference data, and keeps the tests self-contained.

Marked ``slow``: each case runs a real training subprocess for several epochs.
CI runs this module as an explicit step in the serial LAMMPS leg.
"""

from __future__ import annotations

import contextlib
import csv
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from ase import Atoms
import ase.io

from helpers import CASES, atoms_from_fixture, python_reference, run_case
from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper
from nequip.scripts.deploy import load_deployed_model
from tolerances import (
    BEC_ATOL_E,
    E_ATOL_EV,
    F_ATOL_EV_A,
    SERIAL_RTOL,
    SPIN_F_ATOL,
    VIRIAL_ATOL_EV,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
E2E_CASE_NAMES = ("nosoc", "soc", "bec")
DATASET_SEED = 20260831
N_TRAIN = 8
N_VAL = 2
MAX_EPOCHS = 4
POSITION_JITTER_A = 0.03
SPIN_JITTER = 0.15


def _teacher_labels(model, case, atoms: Atoms) -> dict[str, np.ndarray | float]:
    data = AtomicData.from_ase(atoms, r_max=case.r_max)
    mapper = TypeMapper(
        chemical_symbol_to_type={
            symbol: index for index, symbol in enumerate(case.type_names)
        }
    )
    output = model(mapper(data).to("cpu").to_dict())
    labels: dict[str, np.ndarray | float] = {
        "energy": float(
            output[AtomicDataDict.TOTAL_ENERGY_KEY].detach().reshape(-1)[0]
        ),
        "forces": output[AtomicDataDict.FORCE_KEY]
        .detach()
        .reshape(-1, 3)
        .numpy()
        .copy(),
    }
    if case.spin:
        labels["spin_forces"] = (
            output[AtomicDataDict.SPIN_FORCE_KEY].detach().reshape(-1, 3).numpy().copy()
        )
    if case.bec:
        # The data pipeline stores per-node BEC tensors flattened to (N, 9).
        labels["born_effective_charges"] = (
            output[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY]
            .detach()
            .reshape(-1, 9)
            .numpy()
            .copy()
        )
    return labels


def _build_dataset(model_paths, case, path: Path) -> None:
    teacher, _ = load_deployed_model(
        model_paths[case.name],
        device="cpu",
        freeze=True,
        set_global_options=True,
    )
    base = atoms_from_fixture(case)
    rng = np.random.default_rng(DATASET_SEED)

    base_spins = None
    spin_lengths = None
    if case.spin:
        base_spins = base.get_initial_magnetic_moments()
        assert (
            base_spins.ndim == 2 and base_spins.shape[1] == 3
        ), "the spin fixture must carry vector magmoms"
        spin_lengths = np.linalg.norm(base_spins, axis=1, keepdims=True)

    frames = []
    for _ in range(N_TRAIN + N_VAL):
        atoms = base.copy()
        atoms.positions = base.positions + rng.uniform(
            -POSITION_JITTER_A, POSITION_JITTER_A, size=base.positions.shape
        )
        if case.spin:
            spins = base_spins + rng.uniform(
                -SPIN_JITTER, SPIN_JITTER, size=base_spins.shape
            )
            norms = np.linalg.norm(spins, axis=1, keepdims=True)
            assert np.all(norms > 0.0)
            spins = spins / norms * spin_lengths
            atoms.set_initial_magnetic_moments(spins)

        labels = _teacher_labels(teacher, case, atoms)
        assert math.isfinite(labels["energy"])
        atoms.info["energy"] = labels["energy"]
        for key, value in labels.items():
            if key == "energy":
                continue
            assert np.all(np.isfinite(value)), key
            atoms.set_array(key, value)
        frames.append(atoms)

    ase.io.write(path, frames, format="extxyz")

    # The reader must give back exactly what the trainer will consume, so the
    # roundtrip is asserted through the trainer's actual ingestion path
    # (``AtomicData.from_ase``), not through ``atoms.arrays``: ASE's extxyz
    # reader parses non-standard per-atom columns into ``atoms.calc.results``
    # rather than ``atoms.arrays`` (a plain ``get_array`` KeyErrors for BEC),
    # and ``from_ase`` is documented to prefer calc.results. Tolerances are
    # extxyz text-serialisation precision (~1e-9 observed), not bitwise: this
    # guards the KEY MAPPING, which fails by orders of magnitude, not digits.
    reread = ase.io.read(path, index=":", format="extxyz")
    assert len(reread) == N_TRAIN + N_VAL
    ingested = AtomicData.from_ase(reread[0], r_max=case.r_max)
    if case.spin:
        np.testing.assert_allclose(
            ingested[AtomicDataDict.SPIN_FORCE_KEY].numpy(),
            frames[0].get_array("spin_forces"),
            rtol=0.0,
            atol=1e-7,
        )
        # SPIN_KEY is normalised on ingestion (torch F.normalize semantics:
        # zero-length spins -- e.g. the Ge sublattice -- stay zero, they do
        # not become NaN). Compare directions with the same convention.
        expected = frames[0].get_initial_magnetic_moments()
        norms = np.linalg.norm(expected, axis=1, keepdims=True)
        expected = np.divide(
            expected, norms, out=np.zeros_like(expected), where=norms > 0.0
        )
        np.testing.assert_allclose(
            ingested[AtomicDataDict.SPIN_KEY].numpy(),
            expected,
            rtol=0.0,
            atol=1e-7,
        )
    if case.bec:
        np.testing.assert_allclose(
            ingested[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY].numpy(),
            frames[0].get_array("born_effective_charges"),
            rtol=0.0,
            atol=1e-6,
        )


def _write_train_config(case, dataset_path: Path, root: Path, path: Path) -> None:
    config = yaml.safe_load(case.config_path.read_text())
    loss_coeffs: dict[str, object] = {
        "total_energy": [1.0, "PerAtomMSELoss"],
        "forces": 1.0,
    }
    metrics = [["forces", "mae"]]
    if case.spin:
        loss_coeffs["spin_forces"] = 1.0
        metrics.append(["spin_forces", "mae"])
    if case.bec:
        loss_coeffs["born_effective_charges"] = 1.0
        metrics.append(["born_effective_charges", "mae"])
    config.update(
        {
            "root": str(root),
            "run_name": "e2e",
            "seed": 123,
            "dataset_seed": 456,
            "append": False,
            "wandb": False,
            "dataset": "ase",
            "ase_args": {"format": "extxyz"},
            "dataset_file_name": str(dataset_path),
            "train_val_split": "random",
            "shuffle": True,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "batch_size": 2,
            "validation_batch_size": N_VAL,
            "max_epochs": MAX_EPOCHS,
            "learning_rate": 5.0e-3,
            "optimizer_name": "Adam",
            "ldetach_method": False,
            "lauto_loss": False,
            "loss_coeffs": loss_coeffs,
            "metrics_components": metrics,
        }
    )
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def _run(
    cmd: list[str], cwd: Path, env: dict[str, str]
) -> "subprocess.CompletedProcess[str]":
    # start_new_session puts the child in its own process group so a timeout
    # can kill the WHOLE tree: subprocess.run's own timeout kills only the
    # direct child, and a detached descendant (e.g. a dataloader worker that
    # re-sessioned) would survive it (RV4D-05).
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=1800)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)
        raise
    result = subprocess.CompletedProcess(cmd, process.returncode, stdout, None)
    assert result.returncode == 0, f"{' '.join(cmd)} failed:\n{result.stdout[-4000:]}"
    return result


def _checkpoint_energy(train_dir: Path, case) -> float:
    """Evaluate best_model.pth on the fixture frame WITHOUT the deploy path.

    Rebuilds the model from the training session's own config.yaml and loads
    the raw state dict directly, so this prediction cannot be produced by a
    deploy that regressed to a fresh initialisation.

    The evaluation runs on the SAME device `python_reference` uses (cuda when
    available): the binding assertion compares this value against the deployed
    artifact's prediction at 1e-5 eV, and cuda-vs-cpu float32 evaluation of
    the identical weights differs by more than that -- the first 4090 run
    failed the bec case at 2.29e-5 eV purely from this device mismatch while
    nosoc/soc passed by luck. Same defect family as GV3-F10: both sides of a
    tight numeric assertion must run on one device.
    """
    import torch

    from helpers import atomic_input
    from nequip.model import model_from_config
    from nequip.utils import Config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config.from_file(str(train_dir / "config.yaml"))
    model = model_from_config(config=config, initialize=False)
    state_dict = torch.load(
        str(train_dir / "best_model.pth"), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    output = model({k: v.clone() for k, v in atomic_input(case, device=device).items()})
    from nequip.data import AtomicDataDict

    return float(output[AtomicDataDict.TOTAL_ENERGY_KEY].detach().reshape(-1)[0].cpu())


def _epoch_training_losses(train_dir: Path) -> list[float]:
    metrics = train_dir / "metrics_epoch.csv"
    assert metrics.is_file(), f"missing {metrics}"
    losses = []
    with metrics.open() as stream:
        for row in csv.DictReader(stream):
            row = {key.strip(): value.strip() for key, value in row.items()}
            value = float(row["training_loss"])
            assert math.isfinite(value), row
            losses.append(value)
    return losses


@pytest.mark.lammps
@pytest.mark.slow
@pytest.mark.parametrize("case_name", E2E_CASE_NAMES)
def test_dataset_training_deploys_and_matches_lammps(
    case_name,
    model_paths,
    lammps_bin,
    lammps_env,
    tmp_path,
):
    case = CASES[case_name]

    # Stage 1: data processing -- labelled dataset on disk.
    dataset_path = tmp_path / f"e2e_{case_name}.extxyz"
    _build_dataset(model_paths, case, dataset_path)

    # Stage 2: real training through the production entry point.
    config_path = tmp_path / "e2e_train.yaml"
    _write_train_config(case, dataset_path, tmp_path / "results", config_path)
    _run(
        [sys.executable, "-m", "nequip.scripts.train", str(config_path)],
        cwd=tmp_path,
        env=lammps_env,
    )
    train_dir = tmp_path / "results" / "e2e"
    losses = _epoch_training_losses(train_dir)
    assert len(losses) == MAX_EPOCHS
    # Teacher-labelled data is learnable; the trainer must actually have
    # optimised, not merely iterated.
    assert losses[-1] < losses[0], losses

    # Stage 3: deployment from the training directory.
    deployed_path = tmp_path / "e2e_trained.pth"
    _run(
        [
            sys.executable,
            "-m",
            "nequip.scripts.deploy",
            "build",
            "--train-dir",
            str(train_dir),
            str(deployed_path),
        ],
        cwd=tmp_path,
        env=lammps_env,
    )
    assert deployed_path.is_file()

    # Bind the deployed artifact to the TRAINED checkpoint (RV4D-01). The old
    # guard only required the prediction to differ from the teacher -- but a
    # fresh, untrained student also differs from the teacher (different seeds),
    # so a deploy regression that exported an initial model would have passed
    # every assertion (executed reproduction in REPORT-RV4D). Load
    # best_model.pth INDEPENDENTLY of the deploy code path and require the
    # deployed artifact to reproduce its prediction on the fixture frame.
    python = python_reference(deployed_path, case)
    checkpoint_energy = _checkpoint_energy(train_dir, case)
    assert abs(python.energy_ev - checkpoint_energy) <= 1.0e-5, (
        "deployed artifact does not reproduce the trained best_model.pth "
        f"prediction: deployed={python.energy_ev} checkpoint={checkpoint_energy}"
    )
    teacher_python = python_reference(model_paths[case.name], case)
    assert abs(python.energy_ev - teacher_python.energy_ev) > 1.0e-7

    # Stage 4: the trained artifact inside LAMMPS versus Python, with the
    # exact production parity gates.
    lammps = run_case(
        case=case,
        model_path=deployed_path,
        binary=lammps_bin,
        workdir=tmp_path / "lammps",
        env=lammps_env,
    )
    assert abs(python.energy_ev - lammps.energy_ev) <= E_ATOL_EV
    np.testing.assert_allclose(
        lammps.forces_ev_a,
        python.forces_ev_a,
        rtol=SERIAL_RTOL,
        atol=F_ATOL_EV_A,
    )
    if case.spin:
        assert python.spin_forces_ev is not None
        assert lammps.spin_forces_ev is not None
        np.testing.assert_allclose(
            lammps.spin_forces_ev,
            python.spin_forces_ev,
            rtol=SERIAL_RTOL,
            atol=SPIN_F_ATOL,
        )
    if case.bec:
        assert python.bec_e is not None
        assert lammps.bec_e is not None
        np.testing.assert_allclose(
            lammps.bec_e,
            python.bec_e,
            rtol=SERIAL_RTOL,
            atol=BEC_ATOL_E,
        )
    if python.atom_virial_ev is not None and lammps.atom_virial_ev is not None:
        np.testing.assert_allclose(
            lammps.atom_virial_ev,
            python.atom_virial_ev,
            rtol=SERIAL_RTOL,
            atol=VIRIAL_ATOL_EV,
        )
