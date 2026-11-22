# Install and Run SpinGNN

This tutorial installs the validated CUDA 12.8 profile, builds the bundled
LAMMPS fork, deploys a trained model, and runs the first minimal simulations.
Run commands from the repository root unless a step says otherwise.

SpinGNN does not distribute training datasets, trained production models, or
system-specific LAMMPS data files. Replace every `/shared/path/...`, dataset,
model, element, and LAMMPS data path below with values for your system.
See the [documentation index](docs/README.md) for individual workflows.

## 1. Prerequisites

### 1.1 Host and network

Use a Linux x86-64 system with:

- Git 2.25 or newer and network access to clone SpinGNN and T-e3nn;
- `uv` with access to Python and PyTorch package indexes;
- a C++ compiler compatible with the selected LibTorch build;
- CMake and a build tool;
- Open MPI or another supported MPI implementation for the MPI build.

For an offline host, stage the repositories, uv-managed CPython and wheels
before installation. When using a GPU container with an environment built
outside the container, mount both the virtual environment and its base
interpreter.

Check the basic tools:

```bash
git --version
uv --version
cmake --version
c++ --version
mpicxx --version
```

### 1.2 Validated GPU and CUDA profile

| Profile | Intended hardware | Python | PyTorch / LibTorch | CUDA | C++ |
| --- | --- | --- | --- | --- | --- |
| CUDA 12.8 | NVIDIA H100 and compatible GPUs | 3.10.20 | 2.11.0+cu128 | 12.8 runtime | C++17 |

The NVIDIA host driver must support the CUDA 12.8 runtime included in the
PyTorch wheel. The full CUDA development toolkit is needed to build LAMMPS
against that wheel.

For H100, A100, RTX 4090, and RTX 5090, the pinned stack can be installed in a
repository-local environment. The single entry point is
`tools/ci/install.sh` (`--gpu auto|4090|5090|a100|h100|cpu`), which
auto-detects the GPU with `nvidia-smi`, delegates to
`tools/ci/install_gpu_env.sh` / `install_cpu_env.sh`, and then verifies
imports from this checkout and, on a GPU host, the device name and compute
capability. The pinned CUDA wheel requires an architecture newer than V100.
See
[`docs/INSTALL_CUDA128.md`](docs/INSTALL_CUDA128.md) for the environment
contract.

### 1.3 Docker alternative

For a packaged CUDA 12.8 Python and LAMMPS runtime, follow
[`docker/README.md`](docker/README.md). The host still needs a compatible
NVIDIA driver and NVIDIA Container Toolkit; the image does not bundle or
replace the host driver. Its architecture contract is A100, RTX 4090, and H100
(`sm_80;sm_89;sm_90`); RTX 5090 is not supported by the image.

## 2. Create the Python environment with uv

For the pinned stack, `tools/ci/install.sh` performs steps 2.2-2.5 in one run
(GPU auto-detection, repository-local environment, source-exact import and
device assertions). The uv steps below document the equivalent installation
for a workspace-level environment or a custom layout.

Select the intended GPU explicitly when installing on a host without a GPU:

```bash
tools/ci/install.sh --gpu 4090  # or 5090, a100, h100
export SPINGNN_ENV="$PWD/validation/env/spingnn-cu128"
source "$SPINGNN_ENV/bin/activate"
```

The default `--gpu auto` selects CPU when no GPU is detected. The CPU installer
uses an existing virtual environment. A detected GPU whose CUDA runtime cannot
initialize causes the installer to exit with a diagnostic. Prepare the
environment on a networked host and run CUDA checks on the target GPU node.

### 2.1 Clone the sources on shared storage

```bash
export SPINGNN_WORKSPACE=/shared/path/spingnn
mkdir -p "$SPINGNN_WORKSPACE"
cd "$SPINGNN_WORKSPACE"

git clone --branch v0.1.0 https://github.com/Hyyu-Lab/SpinGNN.git SpinGNN
git clone https://github.com/Hongyu-yu/T-e3nn.git validation/vendor/T-e3nn
git -C validation/vendor/T-e3nn checkout --detach 75334027fffc38f2268db8ab1eed281a18ffdbf2  # v1.0
```

The expected layout is:

```text
/shared/path/spingnn/
|-- SpinGNN/
`-- validation/
    |-- env/
    |-- python/
    `-- vendor/T-e3nn/
```

### 2.2 Install shared CPython and create the environment

```bash
cd "$SPINGNN_WORKSPACE"

uv python install 3.10.20 --install-dir validation/python --no-bin
uv venv --clear \
  --python validation/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10 \
  validation/env/spingnn-cu128

export SPINGNN_ENV="$SPINGNN_WORKSPACE/validation/env/spingnn-cu128"
export SPINGNN_PYTHON="$SPINGNN_ENV/bin/python"
readlink -f "$SPINGNN_PYTHON"
```

For shared environments, verify that the resolved interpreter is on a path
available to every host or container that will use the virtual environment.

### 2.3 Install PyTorch and the validated direct dependencies

```bash
uv pip install --python "$SPINGNN_PYTHON" \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.11.0+cu128'

uv pip install --python "$SPINGNN_PYTHON" \
  'numpy==1.26.4' 'ase==3.29.0' 'pandas==2.3.3' 'PyYAML==6.0.3' \
  'tqdm==4.70.0' 'torch-runstats==0.2.0' 'torch-ema==0.3' \
  'opt-einsum==3.4.0' 'opt-einsum-fx==0.1.4' 'packaging==26.3' \
  'pytest==9.1.1' 'scikit-learn==1.7.2' 'matplotlib==3.10.9' \
  'SciencePlots==2.2.2' 'scipy==1.15.3' \
  'pip==26.2.1' 'setuptools==78.1.0'
```

### 2.4 Install the spin forks in the required order

The order and flags in this step are a hard dependency contract:

```bash
cd "$SPINGNN_WORKSPACE"

"$SPINGNN_PYTHON" -m pip install --no-build-isolation \
  ./validation/vendor/T-e3nn

"$SPINGNN_PYTHON" -m pip install --no-deps \
  ./SpinGNN/src/core ./SpinGNN/src/models
```

T-e3nn needs `--no-build-isolation` so its build sees the already installed
PyTorch and setuptools. The repository packages need `--no-deps` because
their public version ranges can otherwise cause a resolver to replace
`e3nn==0.5.0+spin` or the Spin-NequIP fork with official distributions that
do not implement the required time-reversal and spin APIs.

To test changes to the dependency in a separate environment, follow
[Install and verify T-e3nn](docs/TE3NN_INTEGRATION.md). The procedure
verifies the installed dependency's source contents and explains when deployment
artifacts must be rebuilt.

### 2.5 Assert versions and imports

These checks validate the installed package snapshot. Checkout training,
deployment, and tests use the source launcher shown below.

```bash
"$SPINGNN_PYTHON" - <<'PY'
import pathlib
import sys

import allegro
import e3nn
import nequip
import torch

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert e3nn.__version__ == "0.5.0+spin", e3nn.__version__
assert nequip.__version__ == "0.5.6+spin", nequip.__version__
assert allegro.__version__ == "0.2.0+spin", allegro.__version__

print("IMPORT_OK")
print("python", pathlib.Path(sys.executable).resolve())
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("e3nn", e3nn.__version__)
print("nequip", nequip.__version__, nequip.__file__)
print("allegro", allegro.__version__, allegro.__file__)
PY
```

For this checkout, use `tools/ci/source_python.py` with the intended interpreter:

```bash
cd "$SPINGNN_WORKSPACE/SpinGNN"
"$SPINGNN_PYTHON" tools/ci/source_python.py nequip-train --help >/dev/null
"$SPINGNN_PYTHON" tools/ci/source_python.py nequip-deploy --help >/dev/null
"$SPINGNN_PYTHON" tools/ci/source_python.py -c \
  'import nequip, allegro; print(nequip.__file__, allegro.__file__)'
```

`_spingnn_source.py` puts both absolute package roots first in `sys.path` and
inherited `PYTHONPATH`, verifies their origins, and rejects already loaded
foreign modules. Source CLI mains call `_source_guard.guard_source_entry()`;
root `conftest.py` applies the same bootstrap before pytest fixtures load.
Restart a rejected process through the launcher and verify its import paths.
[Source checkout guidance](docs/SOURCE_CHECKOUT.md) also covers editable
installations in shared environments.

## 3. Run the first checks

### 3.1 CPU unit tests from this checkout

The wrapper changes into each package directory, prints the actual source
origins, and uses the launcher for the NequIP unit and Allegro CPU suites:

```bash
cd "$SPINGNN_WORKSPACE/SpinGNN"
SPINGNN_ENV="$SPINGNN_ENV" tools/ci/cpu_unit_tests.sh
```

This runs the CPU unit suites serially. To parse and run a forward pass for
each of the seven templates with synthetic inputs, use
`"$SPINGNN_PYTHON" tools/ci/source_python.py example/validate_configs.py`.
See the [CI guide](tools/ci/README.md) for the test matrix and exit codes.

### 3.2 Check the CUDA runtime on the target device

Run this from the checkout on a GPU node using the installed CUDA 12.8
environment. Set `SPINGNN_GPU_EXPECT` to `h100`, `a100`, `4090`, or `5090`
for the allocated GPU.

```bash
SPINGNN_ENV="$SPINGNN_ENV" SPINGNN_GPU_EXPECT=h100 tools/ci/gpu_full_tests.sh
```

The wrapper checks device identity and compute capability before running
the GPU parity tests. See [the CI guide](tools/ci/README.md) for runtime options.

## 4. Build the LAMMPS fork

For CUDA inference, build against the PyTorch installation and toolkit you
will use at runtime. The CUDA toolkit must be available when CMake configures
LibTorch.

Use `tools/lammps/build_lammps.sh` for automated builds. It checks Python and
torch and, for CUDA torch or `kokkos-cuda`, probes the selected CUDA compiler
with `--version`, honoring `CUDACXX`. Prerequisite failures point to a separate
CPU virtualenv prepared with `tools/ci/install_cpu_env.sh`, or a complete CUDA
12.8 development container or GPU node. With CUDA torch, both `serial` and
`mpi` need the toolkit headers and libraries.
CMake configuration failures retain their original diagnostic and exit code.

### 4.1 Set the LibTorch paths

```bash
cd "$SPINGNN_WORKSPACE/SpinGNN"
export SPINGNN_PYTHON="$SPINGNN_WORKSPACE/validation/env/spingnn-cu128/bin/python"
export SPINGNN_TORCH_CMAKE_PREFIX="$($SPINGNN_PYTHON -c \
  'import torch; print(torch.utils.cmake_prefix_path.splitlines()[0])')"
export SPINGNN_TORCH_ROOT="$($SPINGNN_PYTHON -c \
  'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)')"

"$SPINGNN_PYTHON" -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.utils.cmake_prefix_path)'
```

Always obtain `CMAKE_PREFIX_PATH` from the same PyTorch environment that
deploys the model. Do not download an unrelated LibTorch archive.

### 4.2 Serial build

```bash
cmake --fresh -S lammps/cmake -B build/lammps-serial \
  -D PKG_SPIN=yes \
  -D PKG_ASPHERE=yes \
  -D PKG_BROWNIAN=yes \
  -D BUILD_MPI=no \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=17 \
  -D CMAKE_CXX_STANDARD_REQUIRED=yes \
  -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -D "CMAKE_PREFIX_PATH=$SPINGNN_TORCH_CMAKE_PREFIX"

cmake --build build/lammps-serial --parallel 4
```

For repository automation with the preflight and post-build feature checks:

```bash
PYTHON="$SPINGNN_PYTHON" tools/lammps/build_lammps.sh serial build/lammps-serial
```

### 4.3 MPI build

Load the intended MPI/compiler modules first and confirm `mpicxx` and
`mpirun` come from the same MPI installation.

```bash
mpicxx --showme:version
mpirun --version

cmake --fresh -S lammps/cmake -B build/lammps-mpi \
  -D PKG_SPIN=yes \
  -D PKG_ASPHERE=yes \
  -D PKG_BROWNIAN=yes \
  -D BUILD_MPI=yes \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=17 \
  -D CMAKE_CXX_STANDARD_REQUIRED=yes \
  -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -D "CMAKE_PREFIX_PATH=$SPINGNN_TORCH_CMAKE_PREFIX"

cmake --build build/lammps-mpi --parallel 4
```

The corresponding MPI automation is:

```bash
PYTHON="$SPINGNN_PYTHON" tools/lammps/build_lammps.sh mpi build/lammps-mpi
```

The build enables the same command set in serial and MPI. The documented
GPU parity tests cover serial and two ranks on a single H100 host. Use the
[LAMMPS test guide](tests_lammps/README.md) to check your runtime.

### 4.4 What the CMake options do

| Option | Purpose |
| --- | --- |
| `PKG_SPIN=yes` | Builds spin atom styles, spin pair styles, and `fix spin/sib` / `spin/sib_npt`. |
| `PKG_ASPHERE=yes` | Supplies the ellipsoid atom-vector code required by this 2021 LAMMPS baseline during compilation. |
| `PKG_BROWNIAN=yes` | Builds the Brownian dynamics fixes. |
| `BUILD_MPI=no/yes` | Selects LAMMPS serial MPI stubs or the system MPI implementation. |
| `CMAKE_CXX_STANDARD=17` | Matches the modern PyTorch 2.11 / CUDA 12.8 LibTorch profile. |
| `CMAKE_CXX_STANDARD_REQUIRED=yes` | Prevents CMake from silently falling back to an older language standard. |
| `CMAKE_PREFIX_PATH=...` | Points `find_package(Torch)` at the active PyTorch installation. |

### 4.5 Verify the binary and runtime libraries

```bash
export LD_LIBRARY_PATH="$SPINGNN_TORCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

build/lammps-serial/lmp -h > build/lammps-serial/help.txt
grep -Eq '(^|[[:space:]])allegro([[:space:]]|$)' build/lammps-serial/help.txt
grep -q 'spin/allegro_spin' build/lammps-serial/help.txt
grep -Eq '(^|[[:space:]])bec([[:space:]]|$)' build/lammps-serial/help.txt
grep -Eq '(^|[[:space:]])brownian([[:space:]]|$)' build/lammps-serial/help.txt

ldd build/lammps-mpi/lmp | grep -E 'libmpi|libtorch'
```

The [LAMMPS fork guide](lammps/README.md) covers Allegro and NequIP pair
styles, including the spin variants and NequIP's MPI metadata requirements.

### 4.6 Optional Kokkos CUDA

External Kokkos **3.7.02**, CUDA **12.8**, and **HOPPER90/sm_90** passed all
seven host/Kokkos parity tests on one H100, including the full 21-frame SIB
trace, with `SPINGNN_DETERMINISTIC=1` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Kokkos CUDA remains opt-in through
`SPINGNN_REQUIRE_KOKKOS=1`; PT2 remains the recommended GPU inference path.
Bundled Kokkos 3.4 is incompatible with CUDA 12.8.

The `kokkos-cuda` build requires `KOKKOS_ROOT` with exactly one supported
architecture and its pinned release:

| GPU | `SPINGNN_KOKKOS_ARCH` | Kokkos | CUDA target |
| --- | --- | --- | --- |
| A100 | `AMPERE80` | 3.7.02 | sm_80 |
| H100 | `HOPPER90` | 3.7.02 | sm_90 |
| RTX 4090 | `ADA89` | 4.7.04 | sm_89 |
| RTX 5090 | `BLACKWELL120` | 4.7.04 | sm_120 |

The installer default is HOPPER90. Set the architecture explicitly for an RTX
allocation. The opt-in test wrapper installs the matching external Kokkos
and checks the selected prefix before running the parity suite:

```bash
SPINGNN_REQUIRE_KOKKOS=1 SPINGNN_KOKKOS_ARCH=HOPPER90 \
  SPINGNN_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  tools/ci/gpu_lammps_tests.sh
```

See [tools/ci/README.md](tools/ci/README.md) for the external-prefix overrides.

## 5. Deploy a trained model

Training writes `config.yaml`, `best_model.pth`, and `trainer.pth` under the
directory formed by `root/run_name` in the selected YAML file.

PT2 is the recommended GPU inference format for the PT2-capable pair styles;
see the [PT2 deployment guide](docs/PT2_MIGRATION.md) for export and
validation. The steps below build a TorchScript artifact.

### 5.1 Find and inspect the training session

```bash
cd "$SPINGNN_WORKSPACE/SpinGNN"
export SPINGNN_ENV="$SPINGNN_WORKSPACE/validation/env/spingnn-cu128"
export SPINGNN_PYTHON="$SPINGNN_ENV/bin/python"

export SPINGNN_TRAIN_DIR=results/spin_nosoc/example
test -s "$SPINGNN_TRAIN_DIR/config.yaml"
test -s "$SPINGNN_TRAIN_DIR/best_model.pth"
```

Use `trainer.pth` to resume training. `best_model.pth` is the state dictionary
used to build the deployed model.

### 5.2 Build the TorchScript artifact

```bash
mkdir -p deployed
"$SPINGNN_PYTHON" tools/ci/source_python.py nequip-deploy build \
  --train-dir "$SPINGNN_TRAIN_DIR" \
  deployed/spin-nosoc.pth
test -s deployed/spin-nosoc.pth
```

### 5.3 Verify metadata and TorchScript loading

```bash
"$SPINGNN_PYTHON" tools/ci/source_python.py nequip-deploy info deployed/spin-nosoc.pth

"$SPINGNN_PYTHON" - <<'PY'
from _spingnn_source import activate_source_tree

activate_source_tree()

import torch
from nequip.scripts.deploy import load_deployed_model

path = "deployed/spin-nosoc.pth"
model, metadata = load_deployed_model(
    path, device="cpu", freeze=False, set_global_options=True
)
assert isinstance(model, torch.jit.ScriptModule)
assert float(metadata["r_max"]) > 0.0
assert metadata["type_names"].strip()
print("DEPLOY_LOAD_OK", metadata["r_max"], metadata["type_names"])
PY
```

Before LAMMPS use, verify that `type_names` and their order match the element
names supplied to `pair_coeff`. A load check proves serialization and metadata;
it does not replace a finite GPU forward on representative data.

## 6. Run the first minimal simulations

The examples below are templates. Replace data/model paths and element names.
All Allegro paths require `units metal`, `newton on`, atom IDs, and a mapping
whose LAMMPS type order matches model metadata.

### 6.1 Structure-only Allegro: `run 0`

Create `in.structure-run0`:

```lammps
units       metal
boundary    p p p
atom_style  atomic
atom_modify map array
newton      on
read_data   /shared/path/system.data

pair_style  allegro
pair_coeff  * * /shared/path/structure-model.pth Fe O

thermo_style custom step pe press pxx pyy pzz fmax
run          0
```

Run it on a GPU node:

```bash
build/lammps-serial/lmp -in in.structure-run0
```

### 6.2 Magnetic NoSOC or SOC: short `spin/sib` MD

Both magnetic builders deploy to the same LAMMPS runtime interface. Create
`in.spin-md`:

```lammps
units       metal
boundary    p p p
atom_style  spin
atom_modify map array
newton      on
read_data   /shared/path/system-spin.data

pair_style  spin/allegro_spin
pair_coeff  * * /shared/path/spin-model.pth Fe O

velocity     all create 300.0 19081 mom yes rot no dist gaussian
fix          int all spin/sib 300.0 0.1 91081 lattice moving
timestep     0.0001
thermo       1
thermo_style custom step temp pe ke etotal fmax
run          10
```

`lattice moving` integrates lattice and spins; use `lattice frozen` for a
spin-only run. Do not add a second `nve`, `nvt`, `npt`, or `spin/sib` fix to
the same atoms. For coupled spin-lattice NPT syntax, see the
[`spin/sib_npt` example](lammps/examples/SPINGNN/README.md#coupled-spin-lattice-npt-with-sib-integration).

### 6.3 BEC model, `read_bec`, field coupling, and `bec_dipole`

Create `in.bec-run0`:

```lammps
units       metal
boundary    p p p
atom_style  bec
atom_modify map array
newton      on
read_data   /shared/path/system-bec.data
read_bec    /shared/path/system.bec

pair_style  allegro
pair_coeff  * * /shared/path/bec-model.pth Li Nb O

fix          field all efield 0.0 0.0 0.01 coupling bec piezo no
compute      dip all bec_dipole
compute      becprops all property/atom becxx becyy beczz &
             becdipolex becdipoley becdipolez
thermo       1
thermo_style custom step pe fmax c_dip c_dip[1] c_dip[2] c_dip[3] c_dip[4]
run          0
```

`read_bec` requires one row per atom with the atom ID followed by nine flattened
tensor components. It initializes the BEC state and resets the accumulated
dipole reference. After the `run 0` check, add an appropriate integrator and a
small positive step count for the intended physical experiment. The dipole is
the path increment `integral Z(R) dR` from the current reference; use
`bec_dipole_reset` after equilibration to choose a new reference.

The deployed BEC model outputs energy, forces, stress, per-atom
`born_effective_charges` with shape `(N,9)`, and system-level
`total_born_effective_charges`. It does not output piezoelectric tensors.
`piezoelectric` may still exist as an extxyz data field, but `fix efield ...
piezo yes` is intentionally rejected.

## 7. Troubleshooting

| Symptom | Cause | Resolution |
| --- | --- | --- |
| `.../bin/python: No such file or directory` inside a worker, while the venv directory exists | The venv symlink points to a login-node-only interpreter. | Install uv-managed CPython on shared storage, recreate the venv from that exact interpreter, and verify it with `readlink -f`. |
| Spin imports are missing or `e3nn` is not `0.5.0+spin` | Dependency resolution replaced a spin fork with an official distribution. | Reinstall T-e3nn with `--no-build-isolation`, then reinstall both repository packages together with `--no-deps`; rerun all version assertions. |
| `UnpicklingError` mentions the `weights_only` default while loading old trainer or dataset files | PyTorch 2.6 changed `torch.load()` to default to `weights_only=True`. | Use the current Spin-NequIP source, which sets the mode explicitly. Load pickle-based artifacts only from trusted sources. |
| Evaluation plotting fails with `ModuleNotFoundError: scienceplots` | `SciencePlots` is an optional plotting dependency. | Install the validated `SciencePlots==2.2.2`, or skip the optional plotting stage when the command supports it. |
| MPI fails at `MPI_Init` with `ucp_ep_create ... Address not valid` | UCX endpoint/device mapping is unavailable or transient on that node. | For the validated single-node two-rank correctness path, retry with `OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp UCX_TLS=self,tcp UCX_POSIX_USE_PROC_LINK=n`. This is not a multi-node or RDMA-performance validation. |
| `Pair style Allegro requires metal units because the model assumes eV and Angstrom` | The input selected an incompatible LAMMPS unit system. | Put `units metal` before `pair_style`; do not convert only part of the input and bypass this gate. |
| `Fix efield piezo yes is unavailable: model no longer provides piezoelectric` | The LAMMPS electric-field interface supports BEC coupling. | Use `coupling bec piezo no` and inspect the BEC and `bec_dipole` outputs. The separate piezoelectric response head supports Python training and evaluation. |
| `torch.cuda.is_available()` is false | PyTorch has no usable CUDA device on that host or in the container. | Check device visibility, the host driver and the CUDA wheel, then run a model forward on the target GPU. |
| Tests or CLI commands import `site-packages` instead of this checkout | A non-editable snapshot or an already loaded foreign module was selected. | Use `tools/ci/cpu_unit_tests.sh` or `python tools/ci/source_python.py ...`; restart mixed processes and follow [docs/SOURCE_CHECKOUT.md](docs/SOURCE_CHECKOUT.md) for shared-environment repair. |

For the validated single-node two-rank launch shape:

```bash
env OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp \
    UCX_TLS=self,tcp UCX_POSIX_USE_PROC_LINK=n \
  mpirun --allow-run-as-root --oversubscribe -np 2 \
  build/lammps-mpi/lmp -in in.spin-md
```

## 8. Deeper documentation

- [CUDA 12.8 shared-filesystem installation contract](docs/INSTALL_CUDA128.md)
- [CI wrappers, prerequisites, and exit codes](tools/ci/README.md)
- [Training configurations and extxyz fields](example/README.md)
- [LAMMPS build and command reference](lammps/README.md)
- [LAMMPS SpinGNN templates](lammps/examples/SPINGNN/README.md)
- [Full documentation index](docs/README.md)
