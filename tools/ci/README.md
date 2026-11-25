# SpinGNN2 CI Scripts

Run the commands below from the repository root. See [INSTALL.md](../../INSTALL.md)
for environment requirements and pinned dependency versions.

## Install and Run CPU Checks

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

`source_python.py` puts the checkout package roots (`src/core`, `src/models`,
and `src`) first when running console commands, modules, and Python files. It
rejects already imported modules from other checkouts. The root pytest
`conftest.py` enforces the same source selection. See
[SOURCE_CHECKOUT.md](../../docs/SOURCE_CHECKOUT.md).

`cpu_unit_tests.sh` prints the import origins and checks each package's pytest
warning count against `warnings-budget.json`. Update the
[warnings inventory](warnings-inventory.md) when changing a budget.
`train_dry_check.sh` parses and evaluates the public example configurations
without training data.

## GitHub Actions Test Matrix

`test-lanes.json` defines the Python test matrix. `test_inventory.py --matrix`
checks the file selections and rejects unowned or multiply owned tests.
The control-plane shard installs pytest only. The model shards use the pinned
CPU environment and run with the worker counts in the manifest. The native
contract shard builds `libspingnn2.so` on CPU and validates `tests_native/`.

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

## GPU Checks and Modern LAMMPS Builds

Run the GPU scripts in a CUDA 12.8 environment with a GPU allocation:

```bash
SPINGNN_GPU_EXPECT=5090 tools/ci/gpu_full_tests.sh
# With at least two visible GPUs and an NCCL-enabled PyTorch:
tools/ci/gpu_nccl_tests.sh
```

Build the native C ABI library (`libspingnn2.so`) and modern LAMMPS
(`lmp_spingnn2` on `stable_30Sep2026_update1`) via Slurm:

```bash
python tools/native/build_native.py --prepare-dependencies
python tools/spingnn2_build.py --target cpu
python tools/spingnn2_build.py --target sm120
python tools/spingnn2_build.py --target sm89
```

See [INSTALL.md](../../INSTALL.md) and [lammps_spingnn2/README.md](../../lammps_spingnn2/README.md)
for full native and LAMMPS build details.
