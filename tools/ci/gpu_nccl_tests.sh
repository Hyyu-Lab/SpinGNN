#!/usr/bin/env bash
# Run inside a scheduler allocation with at least two visible CUDA devices.
# Keep this separate from the single-device precision and compiler lanes.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
SPINGNN_ENV=${SPINGNN_ENV:-$REPO_ROOT/validation/env/spingnn-cu128}
if [[ "$SPINGNN_ENV" != /* ]]; then
  SPINGNN_ENV="$REPO_ROOT/$SPINGNN_ENV"
fi
PYTHON="$SPINGNN_ENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python is not executable: $PYTHON" >&2
  exit 2
fi
cd "$REPO_ROOT"
source "$SCRIPT_DIR/compile_cache.sh"
# This single-node gate uses CUDA devices on the same host. Preserve an
# explicitly configured InfiniBand transport.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
SPINGNN_CI_EVIDENCE_DIR=${SPINGNN_CI_EVIDENCE_DIR:-$REPO_ROOT/validation/results/nccl-${SLURM_JOB_ID:-$$}}
mkdir -p "$SPINGNN_CI_EVIDENCE_DIR"
uv run --no-project --no-python-downloads --python "$PYTHON" \
  "$SCRIPT_DIR/source_python.py" -c '
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, (
    "NCCL gate requires at least two allocated CUDA devices"
)
assert torch.distributed.is_nccl_available(), (
    "NCCL gate requires an NCCL-enabled PyTorch"
)
'
uv run --no-project --no-python-downloads --python "$PYTHON" \
  "$SCRIPT_DIR/source_python.py" -m pytest \
  src/core/tests/unit/trainer/test_compiled_optional_ddp.py \
  -m "gpu and slow" -q -s -ra --tb=short \
  -p no:cacheprovider -p pytest_evidence \
  --evidence-json="$SPINGNN_CI_EVIDENCE_DIR/nccl.json" \
  --junitxml="$SPINGNN_CI_EVIDENCE_DIR/nccl.xml" --require-test-passes=2
