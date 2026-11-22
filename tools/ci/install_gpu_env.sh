#!/usr/bin/env bash
# Repository-local CUDA 12.8 environment installer.
#
# Builds the pinned SpinGNN environment entirely under this checkout:
#   validation/bootstrap/  CPython tarball (+ pip install report)
#   validation/python/     extracted python-build-standalone CPython 3.10.20
#   validation/env/spingnn-cu128  the virtual environment
#   validation/vendor/T-e3nn    pinned e3nn fork checkout
#
# Usage:
#   tools/ci/install_gpu_env.sh          build the environment
#   SPINGNN_ENV_REBUILD=1 tools/ci/install_gpu_env.sh   delete + rebuild the venv
#
# Idempotency: an existing tarball, extracted interpreter, or vendor checkout is
# verified (SHA-256 / git HEAD) and reused. An existing venv is a hard error
# unless SPINGNN_ENV_REBUILD=1, so a half-built environment is never silently
# topped up. PIP_INDEX_URL and friends are honored by pip; nothing here
# hardcodes a mirror or a home directory — every path derives from the script
# location.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

BOOTSTRAP_DIR="$REPO_ROOT/validation/bootstrap"
PYTHON_DIR="$REPO_ROOT/validation/python/cpython-3.10.20"
ENV_DIR="$REPO_ROOT/validation/env/spingnn-cu128"
VENDOR_DIR="$REPO_ROOT/validation/vendor/T-e3nn"

CPYTHON_VERSION=3.10.20
CPYTHON_TAG=20260807
CPYTHON_TARBALL="$BOOTSTRAP_DIR/cpython-${CPYTHON_VERSION}-${CPYTHON_TAG}.tar.gz"
CPYTHON_SHA256=36d480cf1f8ca45308c6cfedaad54a127e973e0525a97ca19a0229dbd5ce5f13
CPYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${CPYTHON_TAG}/cpython-${CPYTHON_VERSION}%2B${CPYTHON_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz"

# T-e3nn v1.0.
TE3NN_COMMIT=75334027fffc38f2268db8ab1eed281a18ffdbf2
TE3NN_URL=https://github.com/Hongyu-yu/T-e3nn.git

TORCH_SPEC="torch==2.11.0+cu128"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

DEPS=(
  numpy==1.26.4 ase==3.29.0 pandas==2.3.3 PyYAML==6.0.3
  tqdm==4.70.0 torch-runstats==0.2.0 torch-ema==0.3
  opt-einsum==3.4.0 opt-einsum-fx==0.1.4 packaging==26.3
  pytest==9.1.1 pytest-xdist==3.8.0 scikit-learn==1.7.2
  matplotlib==3.10.9 SciencePlots==2.2.2 scipy==1.15.3
)

mkdir -p "$BOOTSTRAP_DIR"

# --- 1. CPython 3.10.20 tarball: download if missing, always verify ----------
if [[ ! -f "$CPYTHON_TARBALL" ]]; then
  echo "INSTALL_STEP download $CPYTHON_URL"
  curl -fSL --retry 3 -o "$CPYTHON_TARBALL" "$CPYTHON_URL"
fi
echo "$CPYTHON_SHA256  $CPYTHON_TARBALL" | sha256sum --check --strict

# --- 2. Extract the interpreter ----------------------------------------------
if [[ ! -x "$PYTHON_DIR/bin/python3.10" ]]; then
  mkdir -p "$PYTHON_DIR"
  tar -xzf "$CPYTHON_TARBALL" -C "$PYTHON_DIR" --strip-components=1
fi
"$PYTHON_DIR/bin/python3.10" -c 'import sys; assert sys.version_info[:3] == (3, 10, 20), sys.version'

# --- 3. Create the virtual environment ---------------------------------------
if [[ -e "$ENV_DIR" ]]; then
  if [[ "${SPINGNN_ENV_REBUILD:-0}" == "1" ]]; then
    echo "INSTALL_STEP rebuild removing existing $ENV_DIR"
    rm -rf "$ENV_DIR"
  else
    echo "ERROR: $ENV_DIR already exists; set SPINGNN_ENV_REBUILD=1 to rebuild" >&2
    exit 2
  fi
fi
"$PYTHON_DIR/bin/python3.10" -m venv "$ENV_DIR"
PYTHON="$ENV_DIR/bin/python"

"$PYTHON" - <<PY
import pathlib
import sys

assert sys.version_info[:3] == (3, 10, 20), sys.version
resolved = pathlib.Path(sys.executable).resolve()
root = pathlib.Path("$REPO_ROOT").resolve()
assert root in resolved.parents, (
    f"interpreter {resolved} escaped the repository {root}; "
    "the virtual environment needs its bundled base interpreter"
)
PY

# --- 4. Pinned packages, in the mandatory order ------------------------------
"$PYTHON" -m pip install pip==26.2.1 setuptools==78.1.0 wheel
"$PYTHON" -m pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"
"$PYTHON" -m pip install "${DEPS[@]}" \
  --report "$BOOTSTRAP_DIR/install-report-deps.json"

# --- 5. Pinned T-e3nn fork, then the source packages -------------------------
if [[ ! -d "$VENDOR_DIR/.git" ]]; then
  mkdir -p "$(dirname "$VENDOR_DIR")"
  git clone "$TE3NN_URL" "$VENDOR_DIR"
  git -C "$VENDOR_DIR" checkout "$TE3NN_COMMIT"
fi
te3nn_head=$(git -C "$VENDOR_DIR" rev-parse HEAD)
if [[ "$te3nn_head" != "$TE3NN_COMMIT" ]]; then
  echo "ERROR: T-e3nn HEAD $te3nn_head != pinned $TE3NN_COMMIT" >&2
  exit 1
fi
# T-e3nn's build imports torch, so build isolation must stay off; the source
# packages install --no-deps so the resolver cannot replace the spin forks.
"$PYTHON" -m pip install --no-build-isolation "$VENDOR_DIR"
"$PYTHON" -m pip install --no-deps \
  "$REPO_ROOT/src/core" "$REPO_ROOT/src/models"

# --- 6. Import assertions (INSTALL.md §2.5) -----------------------------------
"$PYTHON" - <<'PY'
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

echo "INSTALL_ENV_OK env=$ENV_DIR"
