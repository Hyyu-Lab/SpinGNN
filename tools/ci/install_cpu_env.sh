#!/usr/bin/env bash
# Shared CPU environment installer for CI and local development.
#
# Installs, in dependency order:
#   1. torch (CPU wheel by default; override TORCH_SPEC/TORCH_INDEX for CUDA)
#   2. T-e3nn (e3nn 0.5.0+spin) from the pinned v1.0 commit
#   3. pinned third-party dependencies (constraints-ci.txt)
#   4. src/core + src/models as editable installs with --no-deps
#      (prevents pip from resolving the official e3nn/nequip over the forks)
#
# Contract: run inside an activated virtualenv (or set PYTHON). Exits non-zero
# if the resulting e3nn is not the +spin fork.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

PYTHON=${PYTHON:-python}
# Keep in sync with docker/Dockerfile ARG TE3NN_SOURCE_COMMIT.
TE3NN_COMMIT=${TE3NN_COMMIT:-75334027fffc38f2268db8ab1eed281a18ffdbf2}
TE3NN_URL=${TE3NN_URL:-https://github.com/Hongyu-yu/T-e3nn.git}
TORCH_SPEC=${TORCH_SPEC:-torch==2.11.0}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}

echo "INSTALL_ENV python=$($PYTHON -c 'import sys; print(sys.version.split()[0])') torch_spec=$TORCH_SPEC te3nn=$TE3NN_COMMIT"

"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"
# T-e3nn's build imports torch, so build isolation must stay off.
"$PYTHON" -m pip install --no-build-isolation "git+${TE3NN_URL}@${TE3NN_COMMIT}"
"$PYTHON" -m pip install -r "$SCRIPT_DIR/constraints-ci.txt"
"$PYTHON" -m pip install --no-deps --no-build-isolation \
  -e "$REPO_ROOT/src/core" \
  -e "$REPO_ROOT/src/models"

"$PYTHON" - <<'EOF'
import allegro
import e3nn
import nequip
import torch

version = e3nn.__version__
assert "spin" in version, (
    f"expected the T-e3nn fork (e3nn *+spin), got {version}; "
    "the official e3nn wheel must not shadow the fork"
)
print(f"INSTALL_OK torch={torch.__version__} e3nn={version} "
      f"nequip={nequip.__version__} allegro={allegro.__version__}")
EOF
