# Install SpinGNN2 for CUDA 12.8

This guide installs the CUDA 12.8 training environment for SpinGNN2. The
installer recognizes NVIDIA H100, A100, RTX 4090, and RTX 5090. See
[INSTALL.md](../INSTALL.md) for host requirements and
[the CI guide](../tools/ci/README.md) for the CPU environment.

## Validated Versions

| Component | Validated version |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| [T-e3nn](https://github.com/Hongyu-yu/T-e3nn) | 0.5.0+spin, v1.0 (`75334027fffc38f2268db8ab1eed281a18ffdbf2`) |
| Spin-NequIP (`src/core`) | 0.5.6+spin |
| Spin-Allegro (`src/models`) | 0.2.0+spin |
| SpinGNN2 (`src/spingnn2`) | 2.0.0 |
| NumPy | 1.26.4 |
| ASE | 3.29.0 |
| pandas | 2.3.3 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |

The host NVIDIA driver must support the CUDA 12.8 runtime shipped with the
PyTorch wheel.

## Shared-Filesystem Environment

`tools/ci/install_gpu_env.sh` builds the pinned stack under
`validation/env/spingnn-cu128` and verifies the CPython tarball SHA-256 and
T-e3nn commit. Set `SPINGNN_ENV` to select an existing environment path.

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

## Install PyTorch and Direct Dependencies

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

## Install T-e3nn and SpinGNN2 Packages in Required Order

```bash
cd "$SPINGNN_WORKSPACE"
git clone https://github.com/Hongyu-yu/T-e3nn.git validation/vendor/T-e3nn
git -C validation/vendor/T-e3nn checkout --detach 75334027fffc38f2268db8ab1eed281a18ffdbf2  # v1.0

"$SPINGNN_ENV/bin/python" -m pip install \
  --no-build-isolation ./validation/vendor/T-e3nn

"$SPINGNN_ENV/bin/python" -m pip install --no-deps --no-build-isolation \
  ./SpinGNN2/src/core ./SpinGNN2/src/models ./SpinGNN2
```

Installing T-e3nn first establishes `e3nn==0.5.0+spin`, and `--no-deps`
prevents pip from replacing that fork with upstream e3nn or NequIP. Note that
SpinGNN2's v2 symmetry and model modules (`spingnn2.symmetry`, `spingnn2.v2`)
and native inference engine (`libspingnn2`) have zero `e3nn` runtime
dependency.

## Verify the Installation

```bash
"$SPINGNN_ENV/bin/python" - <<'PY'
import allegro, e3nn, nequip, spingnn2, torch
assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert e3nn.__version__ == "0.5.0+spin", e3nn.__version__
print("IMPORT_OK", torch.__version__, spingnn2.__version__)
PY
```

## Native Engine and Modern LAMMPS

The CUDA 12.8 Python environment is used for training and exporting checkpoints
(`spingnn2-export`). Execution hosts running `libspingnn2.so` or
`lmp_spingnn2` do **not** require PyTorch or LibTorch. Follow
[INSTALL.md](../INSTALL.md) and [docs/SPINGNN2_GUIDE.md](SPINGNN2_GUIDE.md)
for native and modern LAMMPS (`stable_30Sep2026_update1`) builds.

## Run Checkout Verification

```bash
cd "$SPINGNN_WORKSPACE/SpinGNN2"
"$SPINGNN_ENV/bin/python" tools/ci/source_python.py example/validate_configs.py
"$SPINGNN_ENV/bin/python" tools/ci/source_python.py -m pytest \
  tests_ci tests_native/test_contract_oracle.py -q
```
