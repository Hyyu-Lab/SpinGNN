#!/usr/bin/env bash
# Container-side GPU verification body (1x H100, RTX 4090, or RTX 5090).
#
# Runs the marker lanes that CPU CI cannot cover:
#   1. gpu-marked python tests (precision ladder TF32/bf16 tiers, CUDA paths)
#   2. compile parity gates (SPINGNN_TEST_COMPILE=1)
#   3. slow numerical stress contracts, including actual CUDA parameter cases
#   4. GPU LAMMPS lanes when SPINGNN_LAMMPS_BIN points at a CUDA build
#   5. optional TorchScript inference benchmark when SPINGNN_GPU_BENCH_CONFIG is set
#
# Contract mirrors tools/ci/cpu_unit_tests.sh: SPINGNN_ENV selects the
# interpreter (default: validation/env/spingnn-cu128 in this checkout).
# Summary lines use CI_SUITE_RESULT / CI_SUMMARY. Source paths take precedence
# over installed package snapshots. Run with one visible GPU.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
source "$SCRIPT_DIR/compile_cache.sh"
SPINGNN_CI_EVIDENCE_DIR=${SPINGNN_CI_EVIDENCE_DIR:-$REPO_ROOT/validation/results/ci-${SLURM_JOB_ID:-$$}}
mkdir -p "$SPINGNN_CI_EVIDENCE_DIR"
SPINGNN_ENV=${SPINGNN_ENV:-$REPO_ROOT/validation/env/spingnn-cu128}
if [[ "$SPINGNN_ENV" != /* ]]; then
  SPINGNN_ENV="$REPO_ROOT/$SPINGNN_ENV"
fi
PYTHON="$SPINGNN_ENV/bin/python"

SPINGNN_GPU_EXPECT=${SPINGNN_GPU_EXPECT:-h100}
case "$SPINGNN_GPU_EXPECT" in
  h100 | a100 | 4090 | 5090) ;;
  v100)
    # Measured against this exact stack, not assumed: the torch 2.11.0+cu128
    # wheel ships SASS for sm_75/80/86/90/100/120 and PTX no lower than
    # compute_75 (extracted from libtorch_cuda.so, 2026-09-01). PTX only JITs
    # FORWARD, so sm_70 cannot run any kernel in this build. Supporting V100
    # requires a different torch/CUDA stack and is outside this repository's
    # pinned environment contract.
    echo "ERROR: SPINGNN_GPU_EXPECT=v100 is not runnable on this stack:" >&2
    echo "  torch 2.11.0+cu128 ships no sm_70 SASS and its minimum PTX is" >&2
    echo "  compute_75, which cannot JIT down to Volta. A V100 lane needs a" >&2
    echo "  separately pinned torch/CUDA environment." >&2
    exit 2
    ;;
  *)
    echo "ERROR: SPINGNN_GPU_EXPECT must be exactly one of: h100, a100, 4090, 5090 (v100 is rejected with a stack diagnosis); got $SPINGNN_GPU_EXPECT" >&2
    exit 2
    ;;
esac
export SPINGNN_GPU_EXPECT

# Fixture-scale full-FX speed floors, per measured architecture (~30% under
# two consistent runs). H100 measured 6.3x/2.6x and RTX 4090 measured
# 6.10x-6.18x / 2.61x-2.67x. A100 and RTX 5090 record timing for calibration.
case "$SPINGNN_GPU_EXPECT" in
  4090)
    SPINGNN_NEQUIP_SPEED_FLOOR=4.3
    SPINGNN_ALLEGRO_SPEED_FLOOR=1.8
    ;;
  *)
    SPINGNN_NEQUIP_SPEED_FLOOR=4.4
    SPINGNN_ALLEGRO_SPEED_FLOOR=2.0
    ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python is not executable: $PYTHON" >&2
  exit 2
fi
export PYTHONPATH="$REPO_ROOT/src/core:$REPO_ROOT/src/models:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

SPINGNN_REPO_ROOT="$REPO_ROOT" "$PYTHON" - <<'EOF'
import os
from pathlib import Path

import allegro
import e3nn
import nequip
import torch

repo_root = Path(os.environ["SPINGNN_REPO_ROOT"]).resolve()
source_roots = {
    "nequip": repo_root / "src/core",
    "allegro": repo_root / "src/models",
}
for name, module in (("nequip", nequip), ("allegro", allegro)):
    module_path = Path(module.__file__).resolve()
    assert module_path.is_relative_to(source_roots[name]), (
        f"{name} import is not source-exact: {module_path}"
    )
assert torch.cuda.is_available(), "gpu_full_tests requires a CUDA device"
assert torch.cuda.device_count() == 1, (
    f"gpu_full_tests requires exactly one visible GPU; got {torch.cuda.device_count()}"
)
gpu_expect = os.environ["SPINGNN_GPU_EXPECT"]
expectations = {
    "h100": ("H100", (9, 0)),
    "a100": ("A100", (8, 0)),
    "4090": ("4090", (8, 9)),
    "5090": ("5090", (12, 0)),
}
expected_name, expected_capability = expectations[gpu_expect]
device_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
assert expected_name in device_name, (
    f"gpu_full_tests expected {gpu_expect} ({expected_name}); got {device_name}"
)
assert capability == expected_capability, (
    f"gpu_full_tests expected {gpu_expect} capability {expected_capability}; "
    f"got {capability}"
)
assert torch.__version__.startswith("2.11.0+cu128"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert e3nn.__version__ == "0.5.0+spin", e3nn.__version__
print(
    "GPU_ENV "
    f"expect={gpu_expect} device={device_name} "
    f"capability={capability[0]}.{capability[1]} "
    f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
    f"e3nn={e3nn.__version__} nequip={Path(nequip.__file__).resolve()} "
    f"allegro={Path(allegro.__file__).resolve()}"
)
EOF

passed=0
failed=0
declare -A results

record_result() {
  local name=$1
  local status=$2
  if ((status == 0)); then
    results[$name]=PASS
    ((passed += 1))
  else
    results[$name]=FAIL
    ((failed += 1))
  fi
  echo "CI_SUITE_RESULT name=$name result=${results[$name]} exit_code=$status"
}

require_full_fx_benchmark() {
  local suite_log=$1
  local minimum_speedup=$2
  local maximum_loss_relative=1e-5
  "$PYTHON" - "$suite_log" "$minimum_speedup" "$maximum_loss_relative" <<'PY'
import math
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
calibration_only = sys.argv[2] == "calibration"
minimum_speedup = None if calibration_only else float(sys.argv[2])
maximum_loss_relative = float(sys.argv[3])
gpu_expect = os.environ.get("SPINGNN_GPU_EXPECT", "h100")
# H100 and 4090 have calibrated floors for the original fixture workloads.
# Checkpoint speed is a separate workload and remains a measurement only.
if gpu_expect not in {"h100", "a100", "4090", "5090"}:
    raise SystemExit(
        "SPINGNN_GPU_EXPECT must be exactly one of: h100, a100, 4090, 5090; "
        f"got {gpu_expect}"
    )
# Pytest progress markers share stdout and have no trailing newline. The
# producer starts its record with a newline so the parser can require a
# line-start marker without accepting markers in unrelated echoed text.
records = [
    line[len("FULL_FX_WALLTIME ") :].strip()
    for line in path.read_text().splitlines()
    if line.startswith("FULL_FX_WALLTIME ")
]
if len(records) != 1:
    raise SystemExit(
        f"required exactly one FULL_FX_WALLTIME record; found {len(records)}"
    )
fields = dict(part.split("=", 1) for part in records[0].split())
required = {
    "eager_seconds",
    "full_fx_seconds",
    "max_loss_relative",
    "cache_entries",
    "measurement",
    "steps",
}
missing = sorted(required - fields.keys())
if missing:
    raise SystemExit("FULL_FX_WALLTIME missing fields: " + ", ".join(missing))

eager_seconds = float(fields["eager_seconds"])
full_fx_seconds = float(fields["full_fx_seconds"])
loss_relative = float(fields["max_loss_relative"])
cache_entries = int(fields["cache_entries"])
speedup = eager_seconds / full_fx_seconds
if not all(
    math.isfinite(value) and value > 0.0
    for value in (eager_seconds, full_fx_seconds, speedup)
):
    raise SystemExit("FULL_FX_WALLTIME contains invalid timing values")
# `nan > threshold` is False in IEEE comparison, so a NaN loss would slip
# through the threshold check below. Reject malformed numeric records.
if not (math.isfinite(loss_relative) and loss_relative >= 0.0):
    raise SystemExit(
        f"FULL_FX_WALLTIME loss relative is not a finite non-negative "
        f"number: {fields['max_loss_relative']}"
    )
if fields["measurement"] != "steady_state_median" or fields["steps"] != "19":
    raise SystemExit("FULL_FX_WALLTIME is not the required 19-step steady state")
if cache_entries != 1:
    raise SystemExit(
        f"constant-shape benchmark used {cache_entries} cache entries; expected 1"
    )
if loss_relative > maximum_loss_relative:
    raise SystemExit(
        f"full-FX loss relative {loss_relative:.3e} exceeds "
        f"{maximum_loss_relative:.3e}"
    )
# Calibrated arches gate hard on the floor passed by the call site; only
# arches WITHOUT two consistent measured runs stay in calibration mode.
# RTX 4090 measured 6.098x/6.183x for NequIP and 2.611x/2.667x for Allegro.
# The fixture floors are about 30% below those measurements.
if calibration_only or gpu_expect in {"a100", "5090"}:
    print(
        f"FULL_FX_CALIBRATION gpu={gpu_expect} "
        f"speedup={speedup:.6f} eager_seconds={eager_seconds:.9f} "
        f"full_fx_seconds={full_fx_seconds:.9f} "
        f"max_loss_relative={loss_relative:.9e} "
        f"cache_entries={cache_entries} measurement=steady_state_median steps=19"
    )
    calibration_target = "this workload" if calibration_only else gpu_expect
    print(
        f"WARNING: no speed floor is calibrated for {calibration_target}; "
        "floors must be set from measured values after two consistent runs."
    )
else:
    if speedup < minimum_speedup:
        raise SystemExit(
            f"full-FX speedup {speedup:.3f}x is below {minimum_speedup:.3f}x"
        )
    print(
        f"FULL_FX_ACCEPT gpu={gpu_expect} "
        f"speedup={speedup:.6f} minimum={minimum_speedup:.6f} "
        f"max_loss_relative={loss_relative:.9e} "
        f"cache_entries={cache_entries} measurement=steady_state_median steps=19"
    )
PY
}

run_suite() {
  local name=$1
  local package_dir=$2
  local minimum_passed=$3
  shift 3
  local junit_xml
  local suite_log
  local status
  local evidence_json
  evidence_json=$(mktemp "${SPINGNN_CI_EVIDENCE_DIR:-${TMPDIR:-/tmp}}/spingnn-${name}.XXXXXX.json")
  junit_xml=$(mktemp "${TMPDIR:-/tmp}/spingnn-${name}.XXXXXX.xml")
  suite_log=$(mktemp "${TMPDIR:-/tmp}/spingnn-${name}.XXXXXX.log")

  echo "CI_SUITE_START name=$name cwd=$package_dir python=$PYTHON"
  if (cd "$package_dir" && "$PYTHON" -m pytest "$@" -q -ra \
      -p no:cacheprovider -p pytest_evidence --evidence-json="$evidence_json" \
      --require-test-passes="$minimum_passed" --junitxml="$junit_xml") 2>&1 | tee "$suite_log"; then
    status=0
  else
    status=$?
  fi
  if ((status == 0)); then
    if ! "$PYTHON" - "$junit_xml" "$minimum_passed" <<'PY'
import pathlib
import sys
import xml.etree.ElementTree as ET

path = pathlib.Path(sys.argv[1])
minimum = int(sys.argv[2])
root = ET.parse(path).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
passed = tests - skipped
print(f"CI_SUITE_COUNTS tests={tests} skipped={skipped} passed={passed}")
if passed < minimum:
    raise SystemExit(
        f"required lane passed only {passed} test(s); minimum is {minimum}"
    )
PY
    then
      status=1
    fi
  fi
  if ((status == 0)) && [[ "$name" == slow-nequip || "$name" == slow-allegro ]]; then
    if ! "$PYTHON" - "$evidence_json" "$minimum_passed" <<'PY'
import json
from pathlib import Path
import sys

from pytest_evidence import require_cuda_passes

evidence = json.loads(Path(sys.argv[1]).read_text())
minimum = int(sys.argv[2])
passed = require_cuda_passes(evidence, minimum)
print(f"SLOW_CUDA_COUNTS passed={passed} minimum={minimum}")
PY
    then
      status=1
    fi
  fi
  if ((status == 0)); then
    case "$name" in
      compile-nequip|compile-allegro|compile-checkpoint)
        if ! "$PYTHON" - "$evidence_json" "$name" <<'PY'
import json
from pathlib import Path
import sys

from pytest_evidence import require_compile_curve

evidence = json.loads(Path(sys.argv[1]).read_text())
nodeid = require_compile_curve(evidence, sys.argv[2])
print(f"FULL_FX_WORKLOAD suite={sys.argv[2]} nodeid={nodeid}")
PY
        then
          status=1
        fi
        ;;
    esac
  fi
  if ((status == 0)); then
    case "$name" in
      # The fixture-specific floors are about 30% below the measured timings
      # for the H100 and RTX 4090 workloads. Small graphs are sensitive to
      # fixed launch overhead, so these floors apply only to the CI fixtures.
      # A100 and RTX 5090 record speed for calibration while still enforcing
      # the record schema, finite timings, loss parity, and cache behavior.
      compile-nequip)
        require_full_fx_benchmark "$suite_log" "$SPINGNN_NEQUIP_SPEED_FLOOR" || status=$?
        ;;
      compile-allegro)
        require_full_fx_benchmark "$suite_log" "$SPINGNN_ALLEGRO_SPEED_FLOOR" || status=$?
        ;;
      compile-checkpoint)
        require_full_fx_benchmark "$suite_log" calibration || status=$?
        ;;
    esac
  fi
  rm -f -- "$junit_xml" "$suite_log"
  record_result "$name" "$status"
  return 0
}

run_informational_suite() {
  # Runs a lane for its MEASUREMENTS: a numerical outcome (pytest exit 0/1)
  # never contributes to suites_failed. Used for precisions that are
  # deliberately not supported (bf16, which is not reliable across
  # configurations: measured per-config force errors span 2.8e-3..4.9e-2
  # against the 1e-2 bound, so no per-config assertion exists in either
  # direction). The lane surfaces BF16_MEASUREMENT records with a
  # meets_production_bound field; the recording sanity bounds still apply
  # inside the tests.
  #
  # Treat pytest infrastructure errors as failures. Exit 4
  # (usage/configuration error) or 5 (nothing collected) means NO evidence was
  # collected -- a marker typo or collection regression would silently delete
  # the advertised measurements while the aggregate stayed green. Those exits,
  # plus 2 (interrupted) and 3 (internal error), therefore FAIL the aggregate.
  local name=$1
  local package_dir=$2
  shift 2
  local status
  local evidence_json
  evidence_json=$(mktemp "${SPINGNN_CI_EVIDENCE_DIR:-${TMPDIR:-/tmp}}/spingnn-${name}.XXXXXX.json")
  echo "CI_SUITE_START name=$name cwd=$package_dir python=$PYTHON informational=1"
  if (cd "$package_dir" && "$PYTHON" -m pytest "$@" -q -ra -s \
      -p no:cacheprovider -p pytest_evidence --evidence-json="$evidence_json" \
      --require-test-executed=1); then
    status=0
  else
    status=$?
  fi
  if ((status >= 2)); then
    echo "CI_SUITE_RESULT name=$name result=FAIL exit_code=$status" \
      "reason=informational_lane_collected_no_evidence"
    record_result "$name" "$status"
    return 0
  fi
  echo "CI_SUITE_RESULT name=$name result=INFORMATIONAL exit_code=$status not_a_gate=1"
  return 0
}

# Lane 1: gpu-marked tests (precision ladders live in the model test files).
export SPINGNN_TEST_COMPILE=1
# BF16 force errors exceed the 1e-2 bound for some configurations. Report
# BF16 separately. The required suites expect three NequIP and six Allegro
# passes, excluding the intentionally skipped float64 duplicates.
run_suite gpu-nequip "$REPO_ROOT/src/core" 3 tests/unit -m "gpu and not slow and not bf16"
run_suite gpu-allegro "$REPO_ROOT/src/models" 6 tests -m "gpu and not slow and not bf16"

# Informational: bf16 measurements, deliberately not a gate. `-s` lets the
# BF16_MEASUREMENT records reach the log.
run_informational_suite bf16-nequip "$REPO_ROOT/src/core" tests/unit -m "gpu and not slow and bf16"
run_informational_suite bf16-allegro "$REPO_ROOT/src/models" tests -m "gpu and not slow and bf16"

# Lane 2: compile parity gates (including second-order gradients).
run_suite compile-nequip "$REPO_ROOT/src/core" 19 tests/unit/model -m "compile and not gpu" -s
run_suite compile-allegro "$REPO_ROOT/src/models" 19 tests/model \
  -m "compile and not gpu" -k "not TestCheckpointCompileParity" -s
# A separate pytest process preserves the original NoSOC curve identity and
# resets its one-curve guard for the mandatory checkpoint correctness gate.
run_suite compile-checkpoint "$REPO_ROOT/src/models" 3 \
  tests/model/test_allegro_activation_checkpoint.py::TestCheckpointCompileParity \
  -m "compile and not gpu" -s
unset SPINGNN_TEST_COMPILE

# Lane 3: CPU-only scheduled runs cannot exercise CUDA device parameters.
# Keep the strict float64 tests unchanged and require their CUDA cases to pass.
run_suite slow-nequip "$REPO_ROOT/src/core" 7 tests/unit \
  -m "slow and not gpu and not lammps and not mpi"
run_suite slow-allegro "$REPO_ROOT/src/models" 26 tests \
  -m "slow and not gpu and not lammps and not mpi"

# Lane 4: GPU LAMMPS parity (requires a CUDA LAMMPS build).
if [[ -n "${SPINGNN_LAMMPS_BIN:-}" ]]; then
  export SPINGNN_REQUIRE_LAMMPS=1
  run_suite gpu-lammps "$REPO_ROOT" 1 tests_lammps -m "lammps and not mpi and not slow"
  unset SPINGNN_REQUIRE_LAMMPS
else
  echo "CI_SUITE_RESULT name=gpu-lammps result=SKIPPED exit_code=0 (SPINGNN_LAMMPS_BIN unset)"
fi

# Lane 5: optional TorchScript inference benchmark. Full-FX training speed and
# loss parity are measured and gated by the compile lanes above.
if [[ -n "${SPINGNN_GPU_BENCH_CONFIG:-}" ]]; then
  echo "CI_SUITE_START name=inference-benchmark config=$SPINGNN_GPU_BENCH_CONFIG"
  if (cd "$REPO_ROOT/src/core" && "$PYTHON" -m nequip.scripts.benchmark \
      "$SPINGNN_GPU_BENCH_CONFIG" ${SPINGNN_GPU_BENCH_ARGS:-}); then
    benchmark_status=0
  else
    benchmark_status=$?
  fi
  record_result inference-benchmark "$benchmark_status"
fi

echo "CI_SUMMARY suites_passed=$passed suites_failed=$failed $(for k in "${!results[@]}"; do printf '%s=%s ' "$k" "${results[$k]}"; done)"
((failed == 0))
