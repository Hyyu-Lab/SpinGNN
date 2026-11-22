# SpinGNN CI scripts

Run the commands below from the repository root. See [INSTALL.md](../../INSTALL.md)
for environment requirements and the pinned dependency versions.

## Install and run the CPU checks

```bash
uv venv --seed --python 3.10 .venv
source .venv/bin/activate
tools/ci/install.sh --gpu cpu
SPINGNN_ENV="$PWD/.venv" tools/ci/cpu_unit_tests.sh
SPINGNN_ENV="$PWD/.venv" tools/ci/train_dry_check.sh
```

`install.sh` dispatches to the CPU or CUDA 12.8 installer. `--gpu auto`
(default) uses the first visible GPU, or selects CPU if no GPU is visible.
`--gpu 4090`, `5090`, `a100`, and `h100` select the target explicitly, including
on an install host with no visible GPU. A failed `nvidia-smi` query or unusable
CUDA on a detected GPU fails the check. The CPU installer uses the active
virtual environment or the interpreter supplied in `PYTHON`. The GPU
installer creates `validation/env/spingnn-cu128` and verifies the pinned
CPython tarball and the [T-e3nn v1.0 commit](../../docs/TE3NN_INTEGRATION.md).
Set `SPINGNN_ENV_REBUILD=1` to rebuild an existing GPU environment.

`source_python.py` puts both SpinGNN package roots first when running console
commands, modules, and Python files. It rejects already imported modules from
other checkouts. The root pytest `conftest.py` enforces the same source
selection. See [SOURCE_CHECKOUT.md](../../docs/SOURCE_CHECKOUT.md).

`cpu_unit_tests.sh` prints the import origins and checks each package's pytest
warning count against `warnings-budget.json`. Update the [warnings inventory](warnings-inventory.md)
when changing a budget. `train_dry_check.sh` parses and evaluates the public
example configurations without training data. `smoke_routes.py` generates
small training configurations and teacher-labelled datasets for NequIP and
Allegro checks.

## GitHub Actions test matrix

`test-lanes.json` defines the Python test matrix. `test_inventory.py --matrix`
checks the file selections and rejects unowned or multiply owned tests.
The control-plane shard installs pytest only. The model shards use the pinned
CPU environment and run with the worker counts in the manifest. The deployment
contract shard runs CPU tests that do not need LAMMPS. Separate LAMMPS checks build
serial and MPI binaries, run marked parity tests, and exercise the end-to-end
training and deployment chain. Slow Python and PT2 tests also run on the
scheduled or manually dispatched workflow.

The public-example job evaluates all public configurations, trains the three
evidential targets for ten epochs, and verifies their checkpoint reloads.
CI uploads JUnit and JSON records including test selection, outcomes, and
import origins. `--require-test-passes=1` keeps all-skipped suites from passing.

`process_thp.py` applies an explicitly requested Linux transparent huge page
policy for one command and records the policy before replacing itself with the
command. See [TE3NN_INTEGRATION.md](../../docs/TE3NN_INTEGRATION.md).

Reproduce the Python tests with the source launcher. The workflow's pinned
Black, Flake8, and shell checks are listed in
[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).

```bash
python tools/ci/test_inventory.py --matrix
python tools/ci/source_python.py -m pytest tests_ci -q
python tools/ci/source_python.py -m pytest src/core/tests/unit src/models/tests \
  -q -m "not slow and not gpu and not lammps and not mpi"
```

## GPU checks

Run the GPU scripts in a CUDA 12.8 environment with the corresponding GPU
allocation. `SPINGNN_ENV` selects the Python environment and may be an absolute
path. The default is `validation/env/spingnn-cu128` inside the checkout.

```bash
SPINGNN_GPU_EXPECT=h100 tools/ci/gpu_full_tests.sh
tools/ci/gpu_lammps_tests.sh
# With at least two visible GPUs and an NCCL-enabled PyTorch:
tools/ci/gpu_nccl_tests.sh
```

- `gpu_full_tests.sh` checks the name and compute capability of one visible
  GPU, then runs precision, compilation, and numerical tests. Its
  GPU selector accepts `h100` (sm90), `a100` (sm80), `4090` (sm89), and `5090`
  (sm120). The slow tests require CUDA cases to pass. BF16 results are
  informational.
  `SPINGNN_GPU_BENCH_CONFIG` adds an optional `nequip-benchmark` run.
- Each full-FX compilation test checks training loss and cache reuse with a
  fixture model. H100 and RTX 4090 also check measured speed thresholds for
  NequIP and Allegro. The scripts report timing on other architectures.
  `compile_cache.sh` isolates the Inductor and Triton caches.
  Set `SPINGNN_COMPILE_CACHE_ROOT` to reuse a cache deliberately.
- `gpu_lammps_tests.sh` builds and tests a CUDA LAMMPS binary. It checks
  CUDA LibTorch linkage before testing, including for a reused binary. Set
  `SPINGNN_LAMMPS_MPI_BIN` to also run MPI parity. The script saves JUnit and
  CUDA runtime logs under `SPINGNN_GPU_EVIDENCE_DIR`.
- `gpu_nccl_tests.sh` runs the compiled DDP contract on a single host with
  at least two visible CUDA devices. `NCCL_IB_DISABLE` defaults to `1` and
  `TORCHINDUCTOR_COMPILE_THREADS` defaults to `1`.

`SPINGNN_CI_EVIDENCE_DIR` selects where GPU Python and NCCL tests store JSON
and JUnit reports. A test failure returns `1`. An
invalid environment, configuration, or requirement returns `2`.

## LAMMPS build environments

`build_lammps.sh` selects MPI and Kokkos build features. A CUDA PyTorch wheel
requires the complete toolkit even in `serial` mode. To build with CPU-only
PyTorch, activate its environment and run:

```bash
PYTHON=/path/to/cpu-env/bin/python tools/lammps/build_lammps.sh serial
```

For CUDA, provide the development toolkit and a compatible host compiler.
`CUDA_HOME` and `GCC_HOME` can select installations outside the default search
paths. `CUDACXX` can select a compiler command with optional arguments.

PT2 is the default GPU inference path. Set `SPINGNN_REQUIRE_KOKKOS=1` to
include the Kokkos CUDA builds and parity tests. Set `SPINGNN_KOKKOS_ARCH`
to `HOPPER90` (the default), `AMPERE80`, `ADA89`, or `BLACKWELL120`. The first
two architectures use Kokkos 3.7.02, and the latter two use 4.7.04. An existing
installation can be supplied as `KOKKOS_ROOT`. `SPINGNN_KOKKOS_TARBALL` allows
an offline install from a compatible local archive. The build checks that the
selected prefix matches the target architecture.
