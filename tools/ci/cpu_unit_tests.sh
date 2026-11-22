#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
SPINGNN_ENV=${SPINGNN_ENV:-$REPO_ROOT/validation/env/spingnn-cu128}
if [[ "$SPINGNN_ENV" != /* ]]; then
  SPINGNN_ENV="$REPO_ROOT/$SPINGNN_ENV"
fi
PYTHON="$SPINGNN_ENV/bin/python"
WARNING_BUDGET="$SCRIPT_DIR/warnings-budget.json"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python is not executable: $PYTHON" >&2
  exit 2
fi

"$PYTHON" "$SCRIPT_DIR/source_python.py" -c \
  'import nequip, allegro; print("SOURCE_EXACT nequip=" + nequip.__file__ + " allegro=" + allegro.__file__)' || exit 2

run_suite() {
  local name=$1
  local tests=$2
  local package_dir="$REPO_ROOT/$name"
  local log_file
  local status
  local warning_status

  log_file=$(mktemp "/tmp/spingnn-${name##*/}-warnings.XXXXXX.log") || return 2

  echo "CI_SUITE_START name=$name cwd=$package_dir python=$PYTHON"
  (
    cd "$package_dir" || exit 2
    "$PYTHON" "$SCRIPT_DIR/source_python.py" -m pytest "$tests" -q \
      -m "not gpu and not lammps and not mpi and not slow" \
      -p no:cacheprovider
  ) 2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}

  "$PYTHON" "$SCRIPT_DIR/warning_budget.py" \
    --log "$log_file" --suite "$name" --budget "$WARNING_BUDGET"
  warning_status=$?
  rm -f -- "$log_file"

  if ((status == 0 && warning_status == 0)); then
    echo "CI_SUITE_RESULT name=$name result=PASS exit_code=0"
    return 0
  else
    if ((status != 0)); then
      echo "CI_SUITE_RESULT name=$name result=FAIL exit_code=$status"
      return "$status"
    fi
    echo "CI_SUITE_RESULT name=$name result=FAIL exit_code=$warning_status"
    return "$warning_status"
  fi
}

passed=0
failed=0
nequip_result=FAIL
allegro_result=FAIL

if run_suite src/core tests/unit; then
  nequip_result=PASS
  ((passed += 1))
else
  ((failed += 1))
fi

if run_suite src/models tests; then
  allegro_result=PASS
  ((passed += 1))
else
  ((failed += 1))
fi

echo "CI_SUMMARY suites_passed=$passed suites_failed=$failed src/core=$nequip_result src/models=$allegro_result"
((failed == 0))
