---
name: spingnn-install
description: Complete installation guide for SpinGNN environment setup on H100/RTX 4090/RTX 5090 systems with CUDA 12.8, covering Python packages, LAMMPS build, and validation.
---

# SpinGNN Installation Guide

Complete installation workflow for SpinGNN on H100, RTX 4090, or RTX 5090 systems with CUDA 12.8. Covers Python environment setup, package installation, LAMMPS compilation, and validation.

## When to Use

Use this skill when you need to:
- Set up a fresh SpinGNN environment on a new system
- Install SpinGNN on H100, RTX 4090, or RTX 5090 with CUDA 12.8
- Build LAMMPS with SpinGNN pair styles
- Validate the complete installation
- Troubleshoot installation issues

**Trigger phrases:** "install SpinGNN", "setup environment", "安装SpinGNN", "环境配置", "build LAMMPS"

**Prerequisites:** Linux system with NVIDIA GPU (H100, RTX 4090, or RTX 5090), CUDA 12.8 driver installed

## Supported Configuration

| Component | Version | Required |
|-----------|---------|----------|
| **GPU** | H100, RTX 4090, or RTX 5090 | ✓ |
| **CUDA** | 12.8 | ✓ |
| **Python** | 3.10.20 | ✓ |
| **PyTorch** | 2.11.0+cu128 | ✓ |
| **e3nn** | 0.5.0+spin (T-e3nn fork) | ✓ |
| **NequIP** | 0.5.6+spin | ✓ |
| **Allegro** | 0.2.0+spin | ✓ |
| **LAMMPS** | SpinGNN fork | For MD simulations |

## Installation Overview

| Stage | Time | Critical Steps |
|-------|------|----------------|
| 1. System check | 5 min | Verify GPU, CUDA, driver |
| 2. Python environment | 10-30 min | `tools/ci/install.sh` (CPython, PyTorch, dependencies) |
| 3. SpinGNN packages | 5-10 min | e3nn+spin, nequip+spin, allegro+spin (same script) |
| 4. LAMMPS build | 30-60 min | CMAKE, MPI, LibTorch |
| 5. Validation | 10-20 min | Unit tests, LAMMPS parity |

**Total time:** 1-2 hours for complete installation

## Step 1: System Prerequisites Check

### Verify GPU and CUDA

```bash
# Check GPU
nvidia-smi

# Expected output should show:
# - GPU name: "H100", "NVIDIA GeForce RTX 4090", or "NVIDIA GeForce RTX 5090"
# - CUDA Version: 12.8 or higher
# - Driver Version: 550+ for CUDA 12.8; RTX 5090 (Blackwell) needs a
#   Blackwell-capable driver

# Check CUDA compiler (optional, for LAMMPS build)
nvcc --version
# Should show: cuda_12.8
```

### Verify system packages

```bash
# Required for building LAMMPS and Python extensions
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  wget \
  libopenmpi-dev \
  openmpi-bin

# Verify MPI
mpirun --version
```

**Checkpoint:** `nvidia-smi` shows your GPU and CUDA 12.8, `cmake --version` >= 3.16

## Step 2: Python Environment (one script)

Run the single entry point from the repository root. It auto-detects the
first GPU with `nvidia-smi` and selects the matching tier; pass `--gpu` to
override (`a100` is accepted, matching the Docker contract). V100 and
unrecognized cards fail closed with a diagnosis instead of installing an
unrunnable stack.

```bash
cd ~/spingnn-workspace/SpinGNN

# On a GPU-less login node, select the compute-node target explicitly.
# Use --gpu 5090, a100, or h100 for those targets.
tools/ci/install.sh --gpu 4090

export SPINGNN_ENV=$PWD/validation/env/spingnn-cu128
source "$SPINGNN_ENV/bin/activate"
```

To replace an existing repository-local environment, run
`SPINGNN_ENV_REBUILD=1 tools/ci/install.sh --gpu 4090` instead of the first
installation command, using the target GPU tier.

Use `--gpu auto` only on a host that exposes the target GPU. On a GPU-less
host it selects the CPU tier, which requires an existing virtual environment.
Setting `SPINGNN_ENV` alone does not select the shell's Python; activation
above makes the verification and LibTorch discovery below use the installation.

GPU tiers build the pinned CUDA 12.8 stack under `SpinGNN/validation/`
(CPython 3.10.20, `env/spingnn-cu128`, T-e3nn v1.0 at commit
`75334027fffc38f2268db8ab1eed281a18ffdbf2`) by
delegating to `tools/ci/install_gpu_env.sh`, which verifies the CPython
tarball SHA-256 and the T-e3nn commit, installs the spin packages with
`--no-deps`, and ends with the `IMPORT_OK` assertions. The cpu tier delegates
to `tools/ci/install_cpu_env.sh` and needs an activated virtualenv (or
`PYTHON=<path>`). After the install, `install.sh` re-checks the imports
through `tools/ci/source_python.py` (source-exact) and, on a host with a
visible GPU, asserts the device name and compute capability for the tier
(4090 = sm_89, 5090 = sm_120, a100 = sm_80, h100 = sm_90). On a GPU-less
host the device assertion reports `SKIP`.

Export `SPINGNN_ENV` as above before running the validation scripts.

### What the script does internally (manual equivalent)

The script is the supported path; the commands below document it for recovery
or non-standard layouts. The order and flags are a hard contract: T-e3nn
needs `--no-build-isolation` because its build imports the already installed
PyTorch, and the spin packages need `--no-deps` so the resolver cannot
replace the forks with official distributions.

```bash
# 1. CPython 3.10.20 (python-build-standalone tag 20260807, SHA-256 verified)
#    extracted to validation/python/, then a venv at validation/env/spingnn-cu128
#    (install_gpu_env.sh steps 1-3)

# 2. PyTorch and the pinned dependencies (install_gpu_env.sh step 4)
python -m pip install pip==26.2.1 setuptools==78.1.0 wheel
python -m pip install torch==2.11.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  numpy==1.26.4 ase==3.29.0 pandas==2.3.3 PyYAML==6.0.3 \
  tqdm==4.70.0 torch-runstats==0.2.0 torch-ema==0.3 \
  opt-einsum==3.4.0 opt-einsum-fx==0.1.4 packaging==26.3 \
  pytest==9.1.1 pytest-xdist==3.8.0 scikit-learn==1.7.2 \
  matplotlib==3.10.9 SciencePlots==2.2.2 scipy==1.15.3

# 3. T-e3nn fork at the pinned commit, then the spin packages
#    (install_gpu_env.sh step 5); both flags are mandatory
git clone https://github.com/Hongyu-yu/T-e3nn.git validation/vendor/T-e3nn
git -C validation/vendor/T-e3nn checkout --detach 75334027fffc38f2268db8ab1eed281a18ffdbf2  # v1.0
python -m pip install --no-build-isolation validation/vendor/T-e3nn
python -m pip install --no-deps src/core src/models

# 4. Version/import assertions (install_gpu_env.sh step 6); install.sh then
#    adds the source-exact import check and the GPU device assertion
```

The cpu tier (`tools/ci/install_cpu_env.sh`) uses the same order with the CPU
torch wheel (`torch==2.11.0`, pytorch.org/whl/cpu), the `constraints-ci.txt`
pin set, and editable `--no-deps` installs of the two source packages.
Development lint tools are pinned separately: `black==22.3.0`,
`flake8==7.1.1`.

**Checkpoint:** `tools/ci/install.sh` ends with `IMPORT_OK` (plus `DEVICE_OK`
on a GPU node) and `INSTALL_OK tier=...`

## Step 3: Verify the Python Installation

`install.sh` already installed the spin packages and ran the version
assertions (Step 2 lists the exact commands it executes, including the
mandatory `--no-build-isolation` for T-e3nn and `--no-deps` for the spin
packages). Use the block below to re-check an existing environment or a
manual install.

```bash
python - <<'PY'
import sys
import torch
import e3nn
import nequip
import allegro
import numpy
import ase

print("=== SpinGNN Python Environment ===")
print(f"Python:  {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.version.cuda}")
print(f"e3nn:    {e3nn.__version__}")
print(f"NequIP:  {nequip.__version__}")
print(f"Allegro: {allegro.__version__}")
print(f"NumPy:   {numpy.__version__}")
print(f"ASE:     {ase.__version__}")

# Check GPU
assert torch.cuda.is_available(), "CUDA not available"
assert torch.cuda.device_count() >= 1, "No CUDA devices found"
device_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"GPU:     {device_name}")
print(f"Compute: sm_{capability[0]}{capability[1]}")

# Verify spin forks
assert e3nn.__version__ == "0.5.0+spin", "Wrong e3nn version"
assert nequip.__version__ == "0.5.6+spin", "Wrong NequIP version"
assert allegro.__version__ == "0.2.0+spin", "Wrong Allegro version"

print("\n✓ All checks passed!")
PY
```

**Checkpoint:** Above script prints all versions and "✓ All checks passed!"

## Step 4: Build LAMMPS with SpinGNN Support

For a CUDA build, set `CUDA_HOME` to your CUDA 12.8 toolkit and follow
[the LAMMPS build instructions](../../INSTALL.md#4-build-the-lammps-fork).
Run `tools/ci/gpu_lammps_tests.sh` on a GPU host to check CUDA TorchScript
and PT2 parity.

### Option A: Serial LAMMPS (for development)

```bash
cd ~/spingnn-workspace/SpinGNN

# Set build directory
mkdir -p ../validation/build/lammps-serial
cd ../validation/build/lammps-serial

# Get LibTorch path from Python
LIBTORCH_PATH=$(python -c "import torch; import pathlib; print(pathlib.Path(torch.__file__).parent)")

# Configure with CMake
cmake ~/spingnn-workspace/SpinGNN/lammps/cmake \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
  -DPKG_SPIN=ON \
  -DPKG_ASPHERE=ON \
  -DPKG_KSPACE=ON \
  -DPKG_MOLECULE=ON \
  -DPKG_RIGID=ON \
  -DBUILD_MPI=OFF \
  -DBUILD_OMP=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PWD/install

# Build (use -j$(nproc) for parallel build)
cmake --build . --target install -- -j$(nproc)

# Verify build
./install/bin/lmp -h | head -20
./install/bin/lmp -h | grep -E "(pair_style.*allegro|pair_style.*nequip)"
```

**Expected:** `lmp -h` shows "pair_style allegro" and "pair_style spin/allegro_spin"

### Option B: MPI LAMMPS (for production)

```bash
cd ~/spingnn-workspace/SpinGNN
mkdir -p ../validation/build/lammps-mpi
cd ../validation/build/lammps-mpi

LIBTORCH_PATH=$(python -c "import torch; import pathlib; print(pathlib.Path(torch.__file__).parent)")

# Configure with MPI
cmake ~/spingnn-workspace/SpinGNN/lammps/cmake \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
  -DPKG_SPIN=ON \
  -DPKG_ASPHERE=ON \
  -DPKG_KSPACE=ON \
  -DPKG_MOLECULE=ON \
  -DPKG_RIGID=ON \
  -DBUILD_MPI=ON \
  -DBUILD_OMP=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PWD/install

cmake --build . --target install -- -j$(nproc)

# Verify MPI build
./install/bin/lmp -h | head -20
mpirun -np 2 ./install/bin/lmp -h > /dev/null && echo "MPI LAMMPS OK"
```

### Set environment variables

Add to your `~/.bashrc` or session:

```bash
# SpinGNN environment
export SPINGNN_ROOT=~/spingnn-workspace/SpinGNN
# install.sh (Step 2) builds the repository-local environment:
export SPINGNN_ENV=$SPINGNN_ROOT/validation/env/spingnn-cu128
export SPINGNN_LAMMPS_BIN=~/spingnn-workspace/validation/build/lammps-serial/install/bin/lmp
export SPINGNN_LAMMPS_MPI_BIN=~/spingnn-workspace/validation/build/lammps-mpi/install/bin/lmp

# Activate environment
alias spingnn-activate='source $SPINGNN_ENV/bin/activate && export PYTHONPATH=$SPINGNN_ROOT/src/core:$SPINGNN_ROOT/src/models'
```

Then:
```bash
source ~/.bashrc
spingnn-activate
```

**Checkpoint:** `$SPINGNN_LAMMPS_BIN -h` runs and lists SpinGNN pair styles

## Step 5: Validation

### Run Python unit tests

```bash
cd ~/spingnn-workspace/SpinGNN

# CPU unit tests (both suites)
tools/ci/cpu_unit_tests.sh

# Expected: Both src/core and src/models tests pass
# Warnings should be within budget (see warnings-budget.json)
```

### Run LAMMPS parity tests

```bash
cd ~/spingnn-workspace/SpinGNN

# Serial LAMMPS tests
export SPINGNN_LAMMPS_BIN=~/spingnn-workspace/validation/build/lammps-serial/install/bin/lmp
python -m pytest tests_lammps -m "lammps and not mpi and not slow" -v

# Expected: 21+ tests pass
# Parity tolerances: energy < 2e-4 eV, force < 1e-3 eV/Å
```

### Test GPU compilation (if on GPU node)

```bash
cd ~/spingnn-workspace/SpinGNN

# Set GPU expectation based on your hardware
export SPINGNN_GPU_EXPECT=h100  # or 4090, 5090

# Run GPU full tests
tools/ci/gpu_full_tests.sh

# Expected:
# - All 6 lanes pass (gpu-nequip, gpu-allegro, compile-×2, bf16-×2 informational)
# - TF32 precision gates pass
# - Compilation speedup gates pass (H100: 4.4×/2.0×, 4090: 4.3×/1.8×;
#   5090 has no speed floor and reports FULL_FX_CALIBRATION)
```

### Test end-to-end workflow (optional, slow)

```bash
cd ~/spingnn-workspace/SpinGNN

# Run one e2e test (nosoc case)
python -m pytest tests_lammps/test_e2e_train_lammps.py -k nosoc -v

# This runs: dataset creation → training (4 epochs) → deployment → LAMMPS parity
# Time: ~5-10 minutes
```

**Checkpoint:** All validation tests pass without errors

## Step 6: Quick Start Tutorial

### Train a minimal model

```bash
cd ~/spingnn-workspace/SpinGNN

# Copy example config
cp example/02_spin_nosoc.yaml tutorial_config.yaml

# Edit tutorial_config.yaml:
# - Set dataset_file_name to a small test dataset
# - Set max_epochs: 10
# - Set batch_size: 2

# Train (will create results/spin_nosoc/tutorial_run/)
nequip-train tutorial_config.yaml

# Deploy
nequip-deploy build --train-dir results/spin_nosoc/tutorial_run deployed_model.pth

# Check model
nequip-deploy info deployed_model.pth
```

### Run LAMMPS with your model

```bash
# Copy LAMMPS example
cp lammps/examples/spingnn/in.spin_lattice_md test_md.in

# Edit test_md.in:
# - Set model path: pair_coeff * * /path/to/deployed_model.pth Mn Ge
# - Set data path: read_data /path/to/your/structure.data

# Run
$SPINGNN_LAMMPS_BIN -in test_md.in

# Check output
tail log.lammps
```

## Common Installation Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| `torch.cuda.is_available()` False | CUDA driver mismatch | Check `nvidia-smi` CUDA version ≥ 12.8 |
| `ModuleNotFoundError: e3nn` | T-e3nn not installed | Install from Hongyu-yu/T-e3nn fork |
| `nequip-train: command not found` | Environment not activated | Run `source $SPINGNN_ENV/bin/activate` |
| LAMMPS build fails "torch/torch.h not found" | Wrong LibTorch path | Verify `CMAKE_PREFIX_PATH` points to torch package |
| LAMMPS pair style not found | Package not enabled | Add `-DPKG_SPIN=ON` to cmake |
| Tests fail with "model not found" | Environment variables | Set `SPINGNN_ROOT`, `SPINGNN_LAMMPS_BIN` |
| Import error "wrong e3nn version" | Upstream e3nn installed | Uninstall upstream e3nn, reinstall T-e3nn |
| GPU tests fail "expected H100" | Wrong GPU environment | Set `SPINGNN_GPU_EXPECT=4090` for RTX 4090 or `5090` for RTX 5090 |

### Fix wrong e3nn version

```bash
# Symptom: "AssertionError: e3nn version is 0.5.0, expected 0.5.0+spin"
pip uninstall e3nn -y
cd ~/spingnn-workspace/SpinGNN
python -m pip install --no-build-isolation validation/vendor/T-e3nn
python -m pip install --no-deps src/core src/models
python -c "import e3nn; print(e3nn.__version__)"  # Should be 0.5.0+spin
```

### Rebuild LAMMPS with different options

```bash
# Clean build directory
rm -rf ~/spingnn-workspace/validation/build/lammps-serial
mkdir -p ~/spingnn-workspace/validation/build/lammps-serial
cd ~/spingnn-workspace/validation/build/lammps-serial

# Reconfigure with different options (add -DPKG_KOKKOS=ON, etc.)
cmake ~/spingnn-workspace/SpinGNN/lammps/cmake ...
cmake --build . --target install -- -j$(nproc)
```

## Installation Variants

### Minimal (Python only, no LAMMPS)

For training and deployment without running LAMMPS simulations:

```bash
# Steps 1-3 only
# Skip LAMMPS build
# Set SPINGNN_ALLOW_LAMMPS_SKIP=1 to skip LAMMPS tests
export SPINGNN_ALLOW_LAMMPS_SKIP=1
tools/ci/cpu_unit_tests.sh
```

### Development (editable installs)

Already using editable installs (`pip install -e .`). To modify source:

```bash
cd ~/spingnn-workspace/SpinGNN/src/core
# Edit source files
# Changes take effect immediately (no reinstall needed)
python -m pytest ../../tests_lammps -k test_name
```

### Docker (containerized)

See [`docker/README.md`](../../docker/README.md) for containerized installation.
The image covers A100, RTX 4090, and H100 (`sm_80;sm_89;sm_90`); it does not
support RTX 5090, which uses the repository-local installer from Step 2.

```bash
cd ~/spingnn-workspace/SpinGNN
docker/build.sh spingnn:cuda128
docker run --rm --gpus all spingnn:cuda128 spingnn-check
```

## Uninstallation

### Remove Python environment

```bash
rm -rf ~/spingnn-workspace/SpinGNN/validation/env/spingnn-cu128
```

### Remove LAMMPS builds

```bash
rm -rf ~/spingnn-workspace/validation/build/lammps-*
```

### Remove repository

```bash
rm -rf ~/spingnn-workspace/SpinGNN
```

## Upgrading

### Update SpinGNN code

```bash
cd ~/spingnn-workspace/SpinGNN
git pull origin main

# Reinstall packages if changed (keep --no-deps so the resolver cannot
# replace the spin forks with official distributions)
uv pip install --no-deps -e ./src/core -e ./src/models

# Rebuild LAMMPS if lammps/ changed
cd ../validation/build/lammps-serial
cmake --build . --target install -- -j$(nproc)
```

### Update PyTorch (not recommended unless release notes specify)

```bash
# Only if SpinGNN releases support a new PyTorch version
uv pip install torch==NEW_VERSION --index-url https://download.pytorch.org/whl/cu128
# Rebuild LAMMPS with new LibTorch
```

## Installation Checklist

Use this checklist to verify complete installation:

- [ ] `tools/ci/install.sh` finished with `IMPORT_OK` (`DEVICE_OK` on a GPU node) and `INSTALL_OK tier=...`
- [ ] `nvidia-smi` shows H100, RTX 4090, or RTX 5090 with CUDA 12.8
- [ ] `python --version` shows 3.10.20
- [ ] `python -c "import torch; assert torch.__version__ == '2.11.0+cu128'"`
- [ ] `python -c "import e3nn; assert e3nn.__version__ == '0.5.0+spin'"`
- [ ] `python -c "import nequip; assert nequip.__version__ == '0.5.6+spin'"`
- [ ] `python -c "import allegro; assert allegro.__version__ == '0.2.0+spin'"`
- [ ] `nequip-train --help` runs without error
- [ ] `$SPINGNN_LAMMPS_BIN -h | grep allegro` shows pair styles
- [ ] `tools/ci/cpu_unit_tests.sh` passes both test suites
- [ ] `python -m pytest tests_lammps -m "lammps and not mpi and not slow"` passes 21+ tests
- [ ] (GPU) `tools/ci/gpu_full_tests.sh` passes all lanes
- [ ] Can train a model and deploy to `.pth`
- [ ] Can run LAMMPS with deployed model

If all checks pass: **✅ Installation complete!**

## See Also

- [`docs/INSTALL_CUDA128.md`](../../docs/INSTALL_CUDA128.md) - Detailed installation documentation
- [`docker/README.md`](../../docker/README.md) - Docker container installation
- [`tools/ci/README.md`](../../tools/ci/README.md) - CI testing scripts
- [`example/README.md`](../../example/README.md) - Training configuration guide
- [`skills/structure-only-e2e/SKILL.md`](../structure-only-e2e/SKILL.md) - First training workflow
- Official documentation: https://github.com/Hyyu-Lab/SpinGNN

## Support

For installation issues:
1. Check this skill's "Common Installation Issues" section
2. Verify environment with validation scripts
3. Check GitHub Issues: https://github.com/Hyyu-Lab/SpinGNN/issues
4. Review `docs/INSTALL_CUDA128.md` for detailed troubleshooting
