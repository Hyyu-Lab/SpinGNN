---
name: spingnn-install
description: Complete installation guide for SpinGNN2 on A100/RTX 3090/RTX 4090/H100/RTX 5090 systems with CUDA 12.8+, covering Python packages, T-e3nn, multi-GPU all-gpu fat-binary libspingnn2, GPST C99 OpenMP wheel, modern LAMMPS (stable_30Sep2026_update1) overlay, offline bundle_abi=1 JIT preparation, and validation.
---

# SpinGNN2 Installation Guide

Complete installation workflow for **SpinGNN2**: Python training environment
(PyTorch 2.11+cu128, pinned T-e3nn v1.0, `spingnn2.v2` + `PhysMuon`),
zero-LibTorch native C++17 / Kokkos 4.7.02 library (`libspingnn2.so`),
standalone Python/ASE wheel (bundling the C99 OpenMP GPST kernel
`_edge_kernel.so`), offline multi-GPU `bundle_abi = 1` JIT artifacts, and
modern LAMMPS (`stable_30Sep2026_update1`) external overlay (`lmp_spingnn2`).

## When to Use

Use this skill when you need to:
- Set up a fresh SpinGNN2 training environment on A100, RTX 4090, H100, or RTX 5090 with CUDA 12.8+
- Build the SpinGNN2 native Kokkos library (`libspingnn2.so`) for CPU, single-arch CUDA (`sm80`–`sm120`), or a unified multi-GPU `all-gpu` fat binary
- Build modern LAMMPS (`stable_30Sep2026_update1`) via `tools/spingnn2_build.py` or `lammps_spingnn2/apply.py`
- Pre-compile offline single-arch or `bundle_abi = 1` multi-arch JIT artifacts (`spingnn2-prepare-jit-artifact`)
- Validate the complete installation (`tests_ci`, `tests_v2`, `tests_native`)

**Trigger phrases:** "install SpinGNN", "install SpinGNN2", "setup environment", "安装SpinGNN", "环境配置", "build LAMMPS", "build libspingnn2"

**Prerequisites:** Linux x86-64; CMake >= 3.24, Ninja, GCC 10.5+, OpenSSL dev headers; CUDA Toolkit 12.8+ for GPU builds (supports `sm80`, `sm86`, `sm89`, `sm90`, `sm120`); MPI (Intel MPI or OpenMPI) for multi-rank LAMMPS.

## Supported Configuration

| Component | Validated Version / Target | Required |
| --- | --- | --- |
| **GPUs** | A100/A800 (`sm80`), RTX 3090/A40 (`sm86`), RTX 4090/L40S (`sm89`), H100/H800 (`sm90`), RTX 5090 (`sm120`) | For CUDA builds |
| **CUDA Toolkit** | 12.8+ (NVRTC runtime loads `libnvrtc.so.12` / `.13` offline only) | For CUDA builds |
| **Python** | 3.10.20 | Training & export |
| **PyTorch** | 2.11.0+cu128 | Training & export (not needed on inference hosts) |
| **T-e3nn** | `0.5.0+spin`, v1.0 (`75334027fffc38f2268db8ab1eed281a18ffdbf2`) | v1 training only (`v2` & native have zero `e3nn` dep) |
| **SpinGNN2** | `2.0.0` (`src/spingnn2`, `src/core`, `src/models`) | ✓ |
| **Kokkos / JSON** | Kokkos 4.7.02, nlohmann/json 3.11.3 (SHA-256 verified) | Native & LAMMPS |
| **Modern LAMMPS** | `stable_30Sep2026_update1` (`098ffdef...`) via `lammps_spingnn2/` | For MD simulations |

## Step 1: Python Training Environment

Run the automated installer from the repository root (`tools/ci/install.sh`):

```bash
cd /path/to/SpinGNN2

# Auto-detect GPU (h100 | a100 | 4090 | 5090) or specify explicitly on a login node:
tools/ci/install.sh --gpu 5090    # or: --gpu 4090 | a100 | h100 | auto | cpu

export SPINGNN_ENV="$PWD/validation/env/spingnn-cu128"
source "$SPINGNN_ENV/bin/activate"

# Install root SpinGNN2 package (exposes spingnn2-export, spingnn2-prepare-jit-artifact,
# spingnn2-gpst-moments, spingnn2-gpst-infer)
pip install --no-deps --no-build-isolation -e .
```

To replace an existing environment, pass `SPINGNN_ENV_REBUILD=1`. See
[`docs/INSTALL_CUDA128.md`](../../docs/INSTALL_CUDA128.md) and
[`docs/TE3NN_INTEGRATION.md`](../../docs/TE3NN_INTEGRATION.md) for manual steps.

### Verify Python Imports

```bash
python - <<'PY'
import allegro, e3nn, nequip, spingnn2, torch
assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert e3nn.__version__ == "0.5.0+spin", e3nn.__version__
assert nequip.__version__ == "0.5.6+spin", nequip.__version__
assert allegro.__version__ == "0.2.0+spin", allegro.__version__
print("IMPORT_OK", torch.__version__, spingnn2.__version__)
PY
```

### Inference-Only Host (Zero PyTorch / Zero LibTorch)

```bash
pip install .            # NumPy + SciPy + ReferenceModel + NativeModel + GPST CLI
pip install '.[ase]'     # adds SpinGNN2Calculator
```

## Step 2: Native C++17 / Kokkos Library (`libspingnn2.so`)

```bash
# 1. Populate SHA-256-verified Kokkos 4.7.02 & nlohmann/json 3.11.3 cache (internet host)
python tools/native/build_native.py --prepare-dependencies

# 2. Build CPU/OpenMP backend (use Slurm allocation on HPC clusters)
python tools/native/build_native.py --backend cpu

# 3. Build unified multi-GPU fat binary (sm_80, sm_86, sm_89, sm_90, sm_120 + compute_120 PTX)
python tools/native/build_native.py \
  --backend cuda --arch all-gpu --cuda-root /data/app/cuda/12.8

# 4. Or build a single-architecture CUDA backend (sm80|sm86|sm89|sm90|sm120)
python tools/native/build_native.py \
  --backend cuda --arch sm120 --cuda-root /data/app/cuda/12.8
```

Outputs are placed under `validation/native/build-{cpu,cuda-all-gpu,cuda-sm*}/`
(`libspingnn2.so` and `libspingnn2_jit_prepare.so`). When `SPINGNN2_LIBRARY` and
`SPINGNN2_ARCH` are unset, `spingnn2.native` automatically prefers
`build-cuda-all-gpu/libspingnn2.so` or detects the active GPU via `libcuda.so.1`.

## Step 3: Modern LAMMPS (`stable_30Sep2026_update1`, `lmp_spingnn2`)

`tools/spingnn2_build.py` performs a reproducible content-hashed build of
`libspingnn2.so`, the `spingnn2` Python wheel (including the C99 OpenMP GPST
kernel `_edge_kernel.so`), and `lmp_spingnn2` under
`validation/lammps/builds/<target>-<hash>/install/`:

```bash
# On a cluster login node, submits a Slurm build job (default --partition xiang);
# inside a Slurm allocation (SLURM_JOB_ID set), executes directly in place.
python tools/spingnn2_build.py --target cpu
python tools/spingnn2_build.py --target all-gpu   # sm80,sm86,sm89,sm90,sm120 fat binary
python tools/spingnn2_build.py --target sm120     # or sm80|sm86|sm89|sm90|sm80,sm89,sm120
```

Manual overlay materialization & integrity check:

```bash
gh repo clone lammps/lammps validation/lammps/upstream -- --depth 1 --branch stable_30Sep2026_update1
python lammps_spingnn2/apply.py --upstream validation/lammps/upstream --destination validation/lammps/source
python lammps_spingnn2/apply.py --check --destination validation/lammps/source
```

Verify zero-LibTorch linkage and registered styles in a stripped environment
(on a GPU node for CUDA builds so `libcuda.so.1` resolves):

```bash
LMP=validation/lammps/builds/all-gpu-<hash>/install/bin/lmp_spingnn2
env -i PATH=/usr/bin:/bin ldd "$LMP" | grep -E "spingnn2|cublas|torch|nvrtc|not found"
env -i PATH=/usr/bin:/bin "$LMP" -h | grep -E "spingnn2|spin/sib|spin/cfi|spin/gpst|bec"
```

## Step 4: Export Checkpoint & Pre-Compile Multi-Arch JIT Bundle

```bash
python tools/ci/source_python.py spingnn2-export \
  results/spin_nosoc/run -o deployed/spin_nosoc

# Single-arch manifest (.json) or multi-arch bundle_abi=1 manifest (.bundle.json)
python tools/ci/source_python.py spingnn2-prepare-jit-artifact \
  --model deployed/spin_nosoc.sp2.json \
  --backend cuda --arch all-gpu --precision float32 \
  --library validation/lammps/builds/all-gpu-<hash>/install/lib64/libspingnn2.so
```

## Step 5: Validation

```bash
# 1. Control-plane & CI unit tests
python tools/ci/source_python.py -m pytest tests_ci -q
python tools/ci/test_inventory.py
tools/ci/cpu_unit_tests.sh

# 2. Pure-PyTorch v2 symmetry, models & PhysMuon tests
python tools/ci/source_python.py -m pytest tests_v2 -q

# 3. Native, GPST, multi-GPU & LAMMPS suites
export SPINGNN2_LAMMPS_BIN=/path/to/install/bin/lmp_spingnn2
export SPINGNN2_LIBRARY=/path/to/install/lib64/libspingnn2.so
python tools/ci/source_python.py -m pytest tests_native -q
```

## Common Installation Issues

| Symptom | Cause | Solution |
| --- | --- | --- |
| `torch.cuda.is_available()` is `False` | CUDA driver mismatch | Check `nvidia-smi` driver supports CUDA 12.8+ |
| `e3nn version is 0.5.0, expected 0.5.0+spin` | Upstream `e3nn` installed | `pip uninstall -y e3nn`, reinstall pinned `T-e3nn` with `--no-build-isolation` and SpinGNN2 packages with `--no-deps` |
| `libcuda.so.1 => not found` on `ldd` | Running `ldd` on a GPU-less login/CPU build node | Run CUDA `ldd` and `lmp_spingnn2 -h` verification on a GPU compute node |
| `libnvrtc.so` not found during offline JIT prep | CUDA Toolkit lib64 not on `LD_LIBRARY_PATH` | Export `LD_LIBRARY_PATH=/path/to/cuda/lib64:$LD_LIBRARY_PATH` during offline preparation (runtime never loads `libnvrtc`) |
| Intel MPI 2021.13 CUDA IPC peer error on RTX 4090 | P2P IPC unavailable across PCIe | Launch with `I_MPI_OFFLOAD=1 I_MPI_OFFLOAD_IPC=0` |

## See Also

- [`INSTALL.md`](../../INSTALL.md) · [`docs/INSTALL_CUDA128.md`](../../docs/INSTALL_CUDA128.md) · [`docs/SPINGNN2_GUIDE.md`](../../docs/SPINGNN2_GUIDE.md)
- [`skills/spingnn2/SKILL.md`](../spingnn2/SKILL.md) · [`lammps_spingnn2/README.md`](../../lammps_spingnn2/README.md) · [`docker/README.md`](../../docker/README.md)
