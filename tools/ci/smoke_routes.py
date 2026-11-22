#!/usr/bin/env python3
"""Generate deterministic configs and datasets for training smoke tests.

The supported training routes are:

- ``nequip_emt``: ``src/core/configs/minimal_toy_emt.yaml`` + the on-the-fly
  ``EMTTest`` dataset (labels computed by the ASE EMT calculator). This route
  reuses the EMT fixture directly.
- ``allegro_nosoc`` / ``allegro_soc`` / ``allegro_bec`` / ``nequip_spin``: the
  matching ``tests_lammps`` fixture model plus a teacher-labelled extxyz
  dataset generated on demand. The teacher is the route's own model built
  with a fixed seed (random weights), evaluated on jittered fixture frames --
  the same deterministic scheme as ``tests_lammps/test_e2e_train_lammps.py``.
  No external data files are required.

Commands:

    smoke_routes.py config  <route> <out.yaml> <run_dir>
    smoke_routes.py dataset <route> <out.extxyz>
    smoke_routes.py verify  <metrics_dir> <expected_epochs>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = "smoke-dataset.extxyz"

SPIN_ROUTES = ("allegro_nosoc", "allegro_soc", "nequip_spin")
ROUTE_FIXTURES = {
    "allegro_nosoc": "nosoc",
    "allegro_soc": "soc",
    "allegro_bec": "bec",
    "nequip_spin": "nequip_spin",
}
ROUTES = ("nequip_emt", *ROUTE_FIXTURES)

# Fixed seeds: dataset_seed is required by the distributed dataset loader.
SEED = 20260915
DATASET_SEED = 20260916
TEACHER_SEED = 20260917
POSITION_JITTER_A = 0.03
SPIN_JITTER = 0.15


def build_config(
    route: str,
    run_dir,
    *,
    n_train: int = 32,
    n_val: int = 8,
    max_epochs: int = 1,
    verbose: str = "WARNING",
) -> dict:
    """Return the training config dict for one smoke route."""
    if route not in ROUTES:
        raise ValueError(f"unknown smoke route {route!r}; expected one of {ROUTES}")
    run_dir = Path(run_dir)
    if route == "nequip_emt":
        src = yaml.safe_load(
            (REPO_ROOT / "src/core/configs/minimal_toy_emt.yaml").read_text()
        )
        src.update(
            {
                "root": str(run_dir),
                "dataset_root": str(run_dir / "dataset-cache"),
                "run_name": "run",
                "n_train": 16,
                "n_val": 4,
                "batch_size": 1,
                "max_epochs": int(max_epochs),
                "report_init_validation": False,
                "dataloader_num_workers": 0,
                "screen": False,
                "verbose": verbose,
                "dataset_num_frames": 20,
                "dataset_seed": 20260914,
            }
        )
        src["spingnn_compile"] = {
            "enabled": False,
            "scope": "full_fx",
            "dynamic": False,
        }
        return src

    src = yaml.safe_load(
        (
            REPO_ROOT
            / "tests_lammps/fixtures/configs"
            / f"{ROUTE_FIXTURES[route]}.yaml"
        ).read_text()
    )
    loss_coeffs: dict[str, object] = {
        "total_energy": [1.0, "PerAtomMSELoss"],
        "forces": 1.0,
    }
    metrics = [["forces", "mae"]]
    if route in SPIN_ROUTES:
        loss_coeffs["spin_forces"] = 1.0
        metrics.append(["spin_forces", "mae"])
    if route == "allegro_bec":
        loss_coeffs["born_effective_charges"] = 1.0
        metrics.append(["born_effective_charges", "mae"])
    src.update(
        {
            "root": str(run_dir),
            "run_name": "run",
            "seed": SEED,
            "dataset_seed": DATASET_SEED,
            "append": False,
            "wandb": False,
            "screen": False,
            "verbose": verbose,
            "dataset": "ase",
            "ase_args": {"format": "extxyz"},
            "dataset_file_name": str(run_dir / DATASET_FILE),
            "train_val_split": "random",
            "shuffle": True,
            "n_train": int(n_train),
            "n_val": int(n_val),
            # batch_size is per rank; the effective global batch is
            # batch_size * SMOKE_GPUS.
            "batch_size": int(os.environ.get("SMOKE_BATCH_SIZE", 1)),
            "max_epochs": int(max_epochs),
            "report_init_validation": False,
            "dataloader_num_workers": 0,
            "learning_rate": 5.0e-3,
            "optimizer_name": "Adam",  # never SAM in the smoke
            "ldetach_method": False,
            "lauto_loss": False,  # AWL stays off under DDP
            "loss_coeffs": loss_coeffs,
            "metrics_components": metrics,
            "default_dtype": "float32",
            "allow_tf32": False,
            "train_autocast": {"enabled": False},
        }
    )
    src["spingnn_compile"] = {"enabled": False, "scope": "full_fx", "dynamic": False}
    return src


def _ensure_source_paths() -> None:
    for root in (
        REPO_ROOT / "src/core",
        REPO_ROOT / "src/models",
        REPO_ROOT / "tests_lammps",
    ):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


def generate_dataset(
    route: str, path, n_frames: int, *, seed: int = TEACHER_SEED
) -> Path:
    """Write a deterministic teacher-labelled extxyz dataset for one route.

    Frames: the tests_lammps fixture structure with seeded uniform jitter on
    positions (and spin directions for the magnetic routes). Labels: forward
    passes of the route's own model built with a fixed seed -- energy, forces,
    spin_forces (magnetic routes), born_effective_charges (BEC route), all
    asserted finite.
    """
    if route not in ROUTE_FIXTURES:
        raise ValueError(f"route {route!r} uses an on-the-fly dataset, not extxyz")
    _ensure_source_paths()
    import math

    import ase.io
    import numpy as np
    import torch
    from helpers import CASES, atoms_from_fixture
    from nequip.data import AtomicData, AtomicDataDict
    from nequip.data.transforms import TypeMapper
    from nequip.model import model_from_config
    from nequip.scripts.train import default_config
    from nequip.utils import Config
    from nequip.utils._global_options import global_options_context

    case = CASES[ROUTE_FIXTURES[route]]
    torch.manual_seed(seed)
    config = Config.from_file(str(case.config_path), defaults=default_config)
    # Teacher build and evaluation share the config dtype: during the test
    # session's float64 pass the process default dtype is float64, which
    # otherwise produces float64 inputs for the float32 teacher (full-suite
    # float64 failure, 2026-09-16).
    with global_options_context(config):
        teacher = model_from_config(config=config, initialize=False).eval()
    mapper = TypeMapper(
        chemical_symbol_to_type={
            symbol: index for index, symbol in enumerate(case.type_names)
        }
    )
    base = atoms_from_fixture(case)
    rng = np.random.default_rng(seed)
    base_spins = spin_lengths = None
    if case.spin:
        base_spins = base.get_initial_magnetic_moments()
        assert base_spins.ndim == 2 and base_spins.shape[1] == 3
        spin_lengths = np.linalg.norm(base_spins, axis=1, keepdims=True)

    frames = []
    for _ in range(int(n_frames)):
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
            atoms.set_initial_magnetic_moments(spins / norms * spin_lengths)

        with global_options_context(config):
            data = mapper(AtomicData.from_ase(atoms, r_max=case.r_max)).to_dict()
            output = teacher(data)
        energy = float(output[AtomicDataDict.TOTAL_ENERGY_KEY].detach().reshape(-1)[0])
        forces = output[AtomicDataDict.FORCE_KEY].detach().reshape(-1, 3).numpy().copy()
        assert math.isfinite(energy) and np.isfinite(forces).all()
        atoms.info["energy"] = energy
        atoms.set_array("forces", forces)
        if case.spin:
            spin_forces = (
                output[AtomicDataDict.SPIN_FORCE_KEY]
                .detach()
                .reshape(-1, 3)
                .numpy()
                .copy()
            )
            assert np.isfinite(spin_forces).all()
            atoms.set_array("spin_forces", spin_forces)
        if case.bec:
            # The data pipeline stores per-node BEC tensors flattened to (N, 9).
            bec = (
                output[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY]
                .detach()
                .reshape(-1, 9)
                .numpy()
                .copy()
            )
            assert np.isfinite(bec).all()
            atoms.set_array("born_effective_charges", bec)
        frames.append(atoms)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(path, frames, format="extxyz")
    return path


def verify_metrics(metrics_dir, expected_epochs: int) -> None:
    """Fail unless metrics_epoch.csv shows exactly expected_epochs finite losses."""
    import csv
    import math

    path = Path(metrics_dir) / "metrics_epoch.csv"
    if not path.is_file():
        raise SystemExit(f"SMOKE_VERIFY missing {path}")
    losses = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            row = {key.strip(): value.strip() for key, value in row.items()}
            value = float(row["training_loss"])
            if not math.isfinite(value):
                raise SystemExit(f"SMOKE_VERIFY nonfinite training_loss: {row}")
            losses.append(value)
    if len(losses) != int(expected_epochs):
        raise SystemExit(
            f"SMOKE_VERIFY expected {expected_epochs} epochs, found {len(losses)} in {path}"
        )
    print(f"SMOKE_VERIFY status=PASS epochs={len(losses)} metrics={path}")


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit(__doc__)
    command, args = argv[0], argv[1:]
    if command == "config" and len(args) == 3:
        route, out_path, run_dir = args
        config = build_config(
            route,
            run_dir,
            n_train=int(os.environ.get("SMOKE_N_TRAIN", 32)),
            n_val=int(os.environ.get("SMOKE_N_VAL", 8)),
            max_epochs=int(os.environ.get("SMOKE_MAX_EPOCHS", 1)),
            verbose=os.environ.get("SMOKE_VERBOSE", "WARNING"),
        )
        Path(out_path).write_text(yaml.safe_dump(config, sort_keys=False))
    elif command == "dataset" and len(args) == 2:
        route, out_path = args
        n_frames = int(os.environ.get("SMOKE_N_TRAIN", 32)) + int(
            os.environ.get("SMOKE_N_VAL", 8)
        )
        generate_dataset(route, out_path, n_frames)
    elif command == "verify" and len(args) == 2:
        verify_metrics(args[0], int(args[1]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
