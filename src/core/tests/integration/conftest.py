import os
import pathlib
import sys

import numpy as np
import pytest


@pytest.fixture(scope="session")
def BENCHMARK_ROOT(tmp_path_factory):
    """Build a deterministic local replacement for the external aspirin NPZ."""
    benchmark_root = tmp_path_factory.mktemp("benchmark_data")
    n_frames = 1000
    n_atoms = 21

    atom_index = np.arange(n_atoms)
    base_positions = np.column_stack(
        (
            atom_index % 3,
            (atom_index // 3) % 3,
            atom_index // 9,
        )
    ).astype(np.float64)
    base_positions *= 1.5
    base_positions -= base_positions.mean(axis=0, keepdims=True)

    rng = np.random.default_rng(20260901)
    pair_displacements = rng.uniform(0.05, 0.1, size=(n_frames, 10, 3))
    pair_displacements *= rng.choice((-1.0, 1.0), size=pair_displacements.shape)
    displacements = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)
    displacements[:, 0:20:2, :] = pair_displacements
    displacements[:, 1:20:2, :] = -pair_displacements
    positions = base_positions[None, :, :] + displacements

    spring_constant = 400.0
    forces = -spring_constant * displacements
    energies = -405765.0 + 0.5 * spring_constant * np.sum(
        displacements**2, axis=(1, 2)
    )

    benchmark_path = benchmark_root / "aspirin_ccsd-train.npz"
    np.savez(
        benchmark_path,
        z=np.asarray([6] * 9 + [1] * 8 + [8] * 4, dtype=np.int64),
        E=energies,
        F=forces,
        R=positions,
        spingnn_fixture_version=np.asarray(1, dtype=np.int64),
    )
    return benchmark_root


@pytest.fixture(scope="session")
def integration_subprocess_env():
    """Run CLI subprocesses against this checkout, even from temporary cwd."""
    source_root = pathlib.Path(__file__).resolve().parents[2]
    integration_root = pathlib.Path(__file__).resolve().parent
    python_bin = pathlib.Path(sys.executable).parent

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        entry for entry in (str(python_bin), env.get("PATH", "")) if entry
    )
    env["PYTHONPATH"] = os.pathsep.join(
        entry
        for entry in (
            str(source_root),
            str(source_root.parent / "models"),
            str(integration_root),
            env.get("PYTHONPATH", ""),
        )
        if entry
    )
    return env
