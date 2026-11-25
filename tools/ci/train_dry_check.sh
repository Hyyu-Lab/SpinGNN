#!/usr/bin/env bash
# Parse and evaluate the public example configurations without training data.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
SPINGNN_ENV=${SPINGNN_ENV:-${VIRTUAL_ENV:-$REPO_ROOT/validation/env/spingnn-cu128}}
if [[ "$SPINGNN_ENV" != /* ]]; then
  SPINGNN_ENV="$REPO_ROOT/$SPINGNN_ENV"
fi
PYTHON=${PYTHON:-$SPINGNN_ENV/bin/python}
if ! command -v -- "$PYTHON" >/dev/null; then
  echo "ERROR: Python is not executable: $PYTHON" >&2
  exit 2
fi
exec "$PYTHON" "$SCRIPT_DIR/source_python.py" "$REPO_ROOT/example/validate_configs.py" "$@"
