# SpinGNN2 Installation Guide

SpinGNN2 separates **training & checkpoint export** (PyTorch 2.11 + CUDA 12.8)
from **native production inference & modern LAMMPS MD** (`libspingnn2` +
`lmp_spingnn2`, zero LibTorch / zero PyTorch on execution hosts).

## 1. Python Training Environment

SpinGNN2 training targets **Linux x86-64**, **Python 3.10**, **PyTorch
2.11.0+cu128**, **CUDA 12.8**, and [T-e3nn](https://github.com/Hongyu-yu/T-e3nn)
`v1.0` (`e3nn==0.5.0+spin`, required for v1 `src/core` + `src/models`; v2
`spingnn2.symmetry` and `spingnn2.v2` have zero `e3nn` runtime dependency).

### Automated installer

```bash
# Auto-detect GPU (h100 | a100 | 4090 | 5090) or specify explicitly:
tools/ci/install.sh --gpu auto
# On a CPU-only login node targeting a GPU compute partition:
tools/ci/install.sh --gpu 5090
# Or CPU-only training/testing inside an active venv:
uv venv --seed --python 3.10 .venv && source .venv/bin/activate
tools/ci/install.sh --gpu cpu
```

Activate the installed GPU environment:

```bash
source validation/env/spingnn-cu128/bin/activate
```

Install the SpinGNN2 root package into your active environment (editable or
standard):

```bash
pip install --no-deps --no-build-isolation -e .
# Or on an inference-only host without PyTorch:
pip install .            # NumPy + SciPy + ReferenceModel + NativeModel
pip install '.[ase]'     # adds SpinGNN2Calculator
```

See [docs/INSTALL_CUDA128.md](docs/INSTALL_CUDA128.md) for manual shared
filesystem setup, [docs/SOURCE_CHECKOUT.md](docs/SOURCE_CHECKOUT.md) for
running directly via `tools/ci/source_python.py`, and
[docs/TE3NN_INTEGRATION.md](docs/TE3NN_INTEGRATION.md) for T-e3nn pin details.

## 2. Native C++17 / Kokkos Library (`libspingnn2.so`)

### Prerequisites

- CMake >= 3.24, Ninja (recommended), C++17 compiler (e.g. GCC 10.5+)
- OpenSSL development headers (`libssl-dev`)
- CPU builds: OpenBLAS (`libopenblas-dev`) or Intel oneMKL
- CUDA builds: CUDA Toolkit 12.8+ supporting `sm80` (A100/A800), `sm86`
  (RTX 3090/A40), `sm89` (RTX 4090/L40S), `sm90` (H100/H800), `sm120`
  (RTX 5090), or multi-architecture fat binaries (`--arch all-gpu` or
  comma-separated lists such as `--arch sm80,sm89,sm90,sm120`)

### Fetch pinned dependencies & build

`tools/native/build_native.py` downloads and SHA-256-verifies **Kokkos 4.7.02**
and **nlohmann/json 3.11.3** into `validation/native/deps/` (offline builds can
pass `--kokkos-source` and `--json-source`):

```bash
# 1. Populate SHA-256-verified dependency source cache (internet host)
python tools/native/build_native.py --prepare-dependencies

# 2. Build CPU/OpenMP backend (use Slurm srun/sbatch on cluster login nodes)
python tools/native/build_native.py --backend cpu

# 3. Build multi-GPU fat binary (sm_80, sm_86, sm_89, sm_90, sm_120 + compute_120 PTX)
python tools/native/build_native.py \
  --backend cuda --arch all-gpu --cuda-root /data/app/cuda/12.8

# 4. Or build a single-architecture CUDA backend (sm80|sm86|sm89|sm90|sm120)
python tools/native/build_native.py \
  --backend cuda --arch sm120 --cuda-root /data/app/cuda/12.8
```

Default build outputs live under `validation/native/build-{cpu,cuda-all-gpu,cuda-sm*}/`:
- `libspingnn2.so` — versioned C ABI inference library (`sp2_*` symbols)
- `libspingnn2_jit_prepare.so` — CUDA NVRTC offline compilation library (CUDA builds)

When `SPINGNN2_LIBRARY` and `SPINGNN2_ARCH` are unset, Python automatically
prefers `build-cuda-all-gpu/libspingnn2.so` or detects the active CUDA device
compute capability via `libcuda.so.1`. Override with
`SPINGNN2_LIBRARY=/path/to/libspingnn2.so` or `SPINGNN2_ARCH=sm89`. See
[libspingnn2/README.md](libspingnn2/README.md) for all CMake options.

## 3. Modern LAMMPS (`stable_30Sep2026_update1`, `lmp_spingnn2`)

`lammps_spingnn2/` provides a reproducible external overlay against upstream
LAMMPS `stable_30Sep2026_update1` (commit
`098ffdefa525cabc82272925de02e9884f58c657`), with zero LibTorch linkage.

### One-command fingerprinted build (Slurm-aware)

From a cluster login node, `tools/spingnn2_build.py` submits a CPU cross-compile
Slurm job (`--partition xiang` by default) to build `libspingnn2.so`, the
`spingnn2` Python wheel (including the C99 OpenMP GPST kernel `_edge_kernel.so`),
and `lmp_spingnn2` into a content-hashed prefix under
`validation/lammps/builds/<target>-<hash>/install/`:

```bash
python tools/spingnn2_build.py --target cpu
python tools/spingnn2_build.py --target all-gpu          # sm80,sm86,sm89,sm90,sm120 fat binary
python tools/spingnn2_build.py --target sm120            # or sm80|sm86|sm89|sm90|sm80,sm89,sm120
```

Inside an existing Slurm allocation (`SLURM_JOB_ID` set), the same command
executes directly in place.

### Manual overlay & CMake build

```bash
# 1. Clone pinned upstream LAMMPS
gh repo clone lammps/lammps validation/lammps/upstream \
  -- --depth 1 --branch stable_30Sep2026_update1

# 2. Materialize & verify the SpinGNN2 overlay tree
python lammps_spingnn2/apply.py \
  --upstream validation/lammps/upstream \
  --destination validation/lammps/source
python lammps_spingnn2/apply.py --check --destination validation/lammps/source

# 3. Install libspingnn2 into $PREFIX, then configure & build LAMMPS:
cmake -S validation/lammps/source/cmake -B build-lammps -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=ON -DBUILD_OMP=ON \
  -DPKG_SPIN=ON -DPKG_KOKKOS=ON -DKokkos_ENABLE_OPENMP=ON \
  -DCMAKE_CXX_STANDARD=20 -DLAMMPS_MACHINE=spingnn2 \
  -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build build-lammps --target lmp -j 8
cmake --install build-lammps --component spingnn2_runtime
```

For CUDA Kokkos builds, add `-DKokkos_ENABLE_CUDA=ON`,
`-DKokkos_ARCH_BLACKWELL120=ON` (or `ADA89`),
`-DCMAKE_CXX_COMPILER=validation/lammps/source/lib/kokkos/bin/nvcc_wrapper`,
and `-DCMAKE_USER_MAKE_RULES_OVERRIDE_CXX=$PWD/lammps_spingnn2/cmake/cuda-cxx20.cmake`.
See [lammps_spingnn2/README.md](lammps_spingnn2/README.md) for full details.

### Verify relocation & zero-LibTorch linkage

Keep `install/bin/lmp_spingnn2` and `install/lib*/libspingnn2.so` together
under the same prefix:

```bash
env -i PATH=/usr/bin:/bin ldd "$PREFIX/bin/lmp_spingnn2"
env -i PATH=/usr/bin:/bin "$PREFIX/bin/lmp_spingnn2" -h | grep -E 'spingnn2|spin/sib|spin/cfi|spin/gpst|bec'
```

## 4. Verification & Container Builds

Run the control-plane and contract oracle tests:

```bash
python tools/ci/test_inventory.py
python tools/ci/source_python.py -m pytest tests_ci tests_native/test_contract_oracle.py -q
```

For containerized CUDA 12.8 setups, see [docker/README.md](docker/README.md).
