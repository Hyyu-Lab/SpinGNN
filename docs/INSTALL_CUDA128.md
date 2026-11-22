# Install SpinGNN for CUDA 12.8

This guide installs the CUDA environment for SpinGNN v0.1.0. The examples
use an NVIDIA H100. The installer also recognizes A100, RTX 4090, and RTX 5090.
See the [installation guide](../INSTALL.md) for host requirements and
[the CI guide](../tools/ci/README.md) for the CPU environment.

## Validated versions

| Component | Validated version |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| [T-e3nn](https://github.com/Hongyu-yu/T-e3nn) | 0.5.0+spin, v1.0 (`75334027fffc38f2268db8ab1eed281a18ffdbf2`) |
| Spin-NequIP | 0.5.6+spin |
| Spin-Allegro | 0.2.0+spin |
| NumPy | 1.26.4 |
| ASE | 3.29.0 |
| pandas | 2.3.3 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |

The host NVIDIA driver must support the CUDA 12.8 runtime shipped with the
PyTorch wheel. The package metadata currently requires Python earlier than
3.11.

## Shared-filesystem environment

For a shared environment, keep the virtual environment and its base
interpreter on paths visible to every host or container that uses them.
Install uv-managed CPython on shared storage and build the environment from
that interpreter.

`tools/ci/install_gpu_env.sh` builds the pinned stack under
`SpinGNN/validation/` and verifies the CPython tarball SHA-256 and T-e3nn
commit. Set `SPINGNN_ENV` to that checkout's `validation/env/spingnn-cu128`
when using the installer, or to the workspace-level environment created
below.

The commands below assume this layout:

```text
/shared/path/spingnn/
|-- SpinGNN/               # this repository
`-- validation/
    |-- env/
    |-- python/
    `-- vendor/
```

Set the workspace path and create the interpreter and environment:

```bash
export SPINGNN_WORKSPACE=/shared/path/spingnn
cd "$SPINGNN_WORKSPACE"

uv python install 3.10.20 --install-dir validation/python --no-bin
uv venv --clear \
  --python validation/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10 \
  validation/env/spingnn-cu128

export SPINGNN_ENV="$SPINGNN_WORKSPACE/validation/env/spingnn-cu128"
readlink -f "$SPINGNN_ENV/bin/python"
```

Check that the path printed by `readlink` is visible in the runtime
environment.

## Install PyTorch and direct dependencies

Install the CUDA 12.8 PyTorch wheel first, then the validated direct package
set:

```bash
uv pip install --python "$SPINGNN_ENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.11.0+cu128'

uv pip install --python "$SPINGNN_ENV/bin/python" \
  'numpy==1.26.4' 'ase==3.29.0' 'pandas==2.3.3' 'PyYAML==6.0.3' \
  'tqdm==4.70.0' 'torch-runstats==0.2.0' 'torch-ema==0.3' \
  'opt-einsum==3.4.0' 'opt-einsum-fx==0.1.4' 'packaging==26.3' \
  'pytest==9.1.1' 'scikit-learn==1.7.2' 'matplotlib==3.10.9' \
  'scipy==1.15.3' 'pip==26.2.1' 'setuptools==78.1.0'
```

## Install the spin forks in the required order

Clone T-e3nn beside the environment if it is not already available:

```bash
cd "$SPINGNN_WORKSPACE"
git clone https://github.com/Hongyu-yu/T-e3nn.git validation/vendor/T-e3nn
git -C validation/vendor/T-e3nn checkout --detach 75334027fffc38f2268db8ab1eed281a18ffdbf2  # v1.0
```

The following order and flags are a hard requirement:

```bash
"$SPINGNN_ENV/bin/python" -m pip install \
  --no-build-isolation ./validation/vendor/T-e3nn

"$SPINGNN_ENV/bin/python" -m pip install --no-deps \
  ./SpinGNN/src/core ./SpinGNN/src/models
```

Do not replace the second command with a normal dependency-resolving install.
The setup metadata uses public version ranges such as
`e3nn>=0.5.0,<0.5.1` and `nequip>=0.5.6,<0.6`. A resolver can satisfy those
ranges with the official e3nn or NequIP distributions, which do not provide
the SpinGNN time-reversal and spin APIs. Installing T-e3nn first establishes
`e3nn==0.5.0+spin`; installing both repository packages with `--no-deps`
prevents the resolver from replacing that fork or pulling an official NequIP.

T-e3nn uses `--no-build-isolation` so its build sees the environment's
installed PyTorch and setuptools. This also avoids fetching a separate build
environment on offline compute systems.

The commands above install package snapshots. Use the checkout launcher below
for current-source training and validation. See
[the source checkout guide](SOURCE_CHECKOUT.md) for editable installations.

## Verify the installation

Verify the interpreter, CUDA wheel, and the three spin package imports:

```bash
"$SPINGNN_ENV/bin/python" - <<'PY'
import e3nn
import nequip
import allegro
import torch

assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert e3nn.__version__ == "0.5.0+spin", e3nn.__version__
assert nequip.__version__ == "0.5.6+spin", nequip.__version__
assert allegro.__version__ == "0.2.0+spin", allegro.__version__
print(
    "IMPORT_OK",
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
    f"e3nn={e3nn.__version__}",
    f"nequip={nequip.__version__}",
    f"allegro={allegro.__version__}",
)
PY
```

On an H100, check that PyTorch can access the expected CUDA device:

```bash
"$SPINGNN_ENV/bin/python" - <<'PY'
import torch

assert torch.cuda.is_available()
assert "H100" in torch.cuda.get_device_name(0)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

## LAMMPS builds and optional Kokkos CUDA

Use `tools/lammps/build_lammps.sh` with `PYTHON` set to this environment's
interpreter; see [the build tutorial](../INSTALL.md#4-build-the-lammps-fork).
The preflight checks Python/torch and, for CUDA torch or `kokkos-cuda`, runs
the selected CUDA compiler's `--version`, honoring `CUDACXX`. Failures direct
CPU builds to a separate virtual environment prepared with
`tools/ci/install_cpu_env.sh`. Both serial and MPI builds against CUDA
PyTorch need the CUDA 12.8 toolkit headers and libraries.

External Kokkos **3.7.02** with CUDA **12.8** and **HOPPER90/sm_90** passed all
seven host/Kokkos parity tests on one H100, including the full 21-frame SIB
trace, with `SPINGNN_DETERMINISTIC=1` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Kokkos CUDA remains opt-in through
`SPINGNN_REQUIRE_KOKKOS=1`; **PT2 remains the recommended GPU inference path**.
Use external Kokkos for the CUDA 12.8 build.

The `kokkos-cuda` build requires `KOKKOS_ROOT` for an external build with
exactly one supported architecture. Use `HOPPER90` for H100 and see the
[Kokkos installation table](../INSTALL.md#46-optional-kokkos-cuda) for the
other architectures and versions. The [CI guide](../tools/ci/README.md) has
the opt-in verification commands.

## Tests and known boundaries

Run checkout commands through `tools/ci/source_python.py` so both packages
come from the checkout. See the [source checkout guide](SOURCE_CHECKOUT.md)
for checking import origins or using an editable installation.

```bash
cd "$SPINGNN_WORKSPACE/SpinGNN"
"$SPINGNN_ENV/bin/python" tools/ci/source_python.py nequip-train --help
"$SPINGNN_ENV/bin/python" tools/ci/source_python.py example/validate_configs.py
"$SPINGNN_ENV/bin/python" tools/ci/source_python.py -m pytest \
  src/core/tests/unit src/models/tests -q \
  -m "not slow and not gpu and not lammps and not mpi"
```

`tools/ci/cpu_unit_tests.sh` runs both CPU suites serially and checks
warnings. See the [CI guide](../tools/ci/README.md) for the Actions matrix
and CUDA test commands.

- PyTorch 2.11 emits deprecation warnings for `torch.jit.script`,
  `torch.jit.trace_method`, and `torch.jit.load`. TorchScript remains the
  deployment format used by non-spin NequIP and an alternative to
  [PT2](PT2_MIGRATION.md) for other pair styles.
- Check the expected device and run an actual CUDA model forward to verify
  model execution on a GPU.
