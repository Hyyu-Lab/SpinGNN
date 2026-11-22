# LAMMPS pytest suite

This directory turns the SpinGNN Python/LAMMPS deployment contract into pytest tests. Models are built from the four small YAML configurations with deterministic random weights; no dataset or training run is required.

## Environment contract

Use the validated interpreter:

```bash
python -m pytest tests_lammps
```

The shared environment contains non-editable package snapshots. `conftest.py` always places the current checkout's `src/core` and `src/models` directories first on `sys.path` and subprocess `PYTHONPATH`, then rejects collection unless `nequip.__file__` and `allegro.__file__` resolve inside this checkout. It does not modify the shared environment.

Runtime variables:

- `SPINGNN_LAMMPS_BIN`: serial LAMMPS executable. Tests that request it fail closed when it is absent or invalid.
- `SPINGNN_LAMMPS_MPI_BIN`: MPI LAMMPS executable. MPI tests fail closed when it is absent or invalid, including under `SPINGNN_REQUIRE_LAMMPS=1`.
- `SPINGNN_REQUIRE_LAMMPS=1`: make a missing or invalid serial binary an early nonzero pytest configuration error.
- `SPINGNN_ALLOW_LAMMPS_SKIP=1`: explicit developer/collection opt-out that turns an unavailable requested LAMMPS contract into a skip. Do not set it in acceptance CI.
- `SPINGNN_MPIRUN`: MPI launcher command, default `mpirun`.
- `SPINGNN_MPIRUN_EXTRA_ARGS`: shell-split launcher arguments for containers or CI.
- `SPINGNN_MPIRUN_EXTRA_ENV`: shell-split `NAME=VALUE` assignments selected by
  `tools/ci/mpi_probe.sh`; they are applied to its pytest probe and LAMMPS runs.
  The probe compiles and launches a real MPI payload (`MPI_Init` plus an
  `MPI_Allreduce`, which forces inter-rank endpoint setup) and so needs `mpicc`;
  without one it reports `status=UNAVAILABLE` and fails closed. It must never go
  back to launching a non-MPI command such as `hostname`: that only proves the
  launcher can spawn processes, makes the `default` combination pass
  unconditionally, and leaves the UCX-avoiding fallbacks unreachable when
  inter-rank endpoint setup fails inside `MPI_Init`.

The subprocess environment derives `LD_LIBRARY_PATH` from the active `torch` package's `lib` directory. It also preserves existing `PYTHONPATH` and `LD_LIBRARY_PATH` entries after the current source and libtorch paths.

## Local commands

From the SpinGNN repository root, with its Python environment active:

```bash
PYTHONPATH="$PWD/src/core:$PWD/src/models${PYTHONPATH:+:$PYTHONPATH}" \
SPINGNN_LAMMPS_BIN="$PWD/build/lammps-serial/lmp" \
python -m pytest tests_lammps \
  -m "lammps and not mpi and not gpu" -p no:cacheprovider
```

MPI lane:

```bash
PYTHONPATH="$PWD/src/core:$PWD/src/models${PYTHONPATH:+:$PYTHONPATH}" \
SPINGNN_LAMMPS_BIN="$PWD/build/lammps-serial/lmp" \
SPINGNN_LAMMPS_MPI_BIN="$PWD/build/lammps-mpi/lmp" \
python -m pytest tests_lammps -m mpi \
  -p no:cacheprovider
```

Contract-only checks:

```bash
env -u SPINGNN_LAMMPS_BIN SPINGNN_ALLOW_LAMMPS_SKIP=1 \
  python -m pytest tests_lammps -p no:cacheprovider

env -u SPINGNN_LAMMPS_BIN SPINGNN_REQUIRE_LAMMPS=1 \
  python -m pytest tests_lammps -p no:cacheprovider
```

M4 and M5 placeholders are explicitly skipped with `enabled in M4` and `enabled in M5`. All numerical gates are defined in `tolerances.py`; tests do not carry private thresholds.

## End-to-end pipeline test

`test_e2e_train_lammps.py` (`lammps` + `slow`) covers the full chain for the
three Allegro lines -- nosoc, soc, and bec (NequIP-spin is deliberately out of
scope): a teacher-labelled extxyz dataset (vector `initial_magmoms` /
`spin_forces` / `born_effective_charges`) -> a real `nequip-train` subprocess
(loss must decrease) -> `nequip-deploy build --train-dir` -> the trained
artifact in LAMMPS versus Python under the production parity gates. The
dataset roundtrip is asserted through `AtomicData.from_ase`, the trainer's
actual ingestion path: ASE's extxyz reader parses non-standard per-atom
columns (such as BEC) into `atoms.calc.results`, not `atoms.arrays`, and
`from_ase` prefers calc.results -- checking `atoms.get_array` gives a false
negative there. Everything else covers the halves separately (integration
tests train but never reach LAMMPS; `model_factory.py` models are
deliberately untrained). CI runs it as an explicit step on the serial leg
(~4-5 min for all three cases).
