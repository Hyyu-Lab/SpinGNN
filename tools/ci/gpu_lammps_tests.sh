#!/usr/bin/env bash
# Container-side GPU LAMMPS verification (1x H100, CUDA 12.8 image).
#
# CPU CI already covers the TorchScript/PT2/Kokkos-SERIAL lanes. This script
# covers what only a GPU box can: CUDA LAMMPS builds, CUDA-targeted .pt2
# packages, and the Kokkos CUDA backend.
#
# Stages are independent so a failure in one still yields diagnostics for the
# rest:
#   1. CUDA serial LAMMPS build + TorchScript/PT2 parity
#   2. Explicit MPI LAMMPS parity when SPINGNN_LAMMPS_MPI_BIN is supplied
#   3. Opt-in Kokkos CUDA lane (3.7.02 on H100; 4.7.04 on Ada/Blackwell)
#
# Contract matches tools/ci/gpu_full_tests.sh (SPINGNN_ENV, CI_SUITE_RESULT /
# CI_SUMMARY lines, source-first PYTHONPATH).
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
source "$SCRIPT_DIR/compile_cache.sh" || exit 2
SPINGNN_ENV=${SPINGNN_ENV:-$REPO_ROOT/validation/env/spingnn-cu128}
if [[ "$SPINGNN_ENV" != /* ]]; then
  SPINGNN_ENV="$REPO_ROOT/$SPINGNN_ENV"
fi
PYTHON="$SPINGNN_ENV/bin/python"
BUILD_ROOT=${SPINGNN_GPU_BUILD_ROOT:-$REPO_ROOT/validation/build/gpu}

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python is not executable: $PYTHON" >&2
  exit 2
fi
export PYTHONPATH="$REPO_ROOT/src/core:$REPO_ROOT/src/models${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$SPINGNN_ENV/bin:$PATH"
if [[ -n "${GCC_HOME:-}" ]]; then
  export PATH="$GCC_HOME/bin:$PATH"
  export CC="${CC:-$GCC_HOME/bin/gcc}"
  export CXX="${CXX:-$GCC_HOME/bin/g++}"
fi
if [[ -n "${CC:-}" ]]; then
  export CMAKE_CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER:-$CC}"
fi

# Respect explicit toolkit settings and let the build find nvcc on PATH.
CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-${CUDAToolkit_ROOT:-}}}
if [[ -z "$CUDA_HOME" && -x /usr/local/cuda/bin/nvcc ]]; then
  CUDA_HOME=/usr/local/cuda
fi
if [[ -n "$CUDA_HOME" ]]; then
  export CUDA_HOME
  export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
  export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-$CUDA_HOME}"
  export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# Use a stable reduction order in the numerical parity lanes.
export SPINGNN_DETERMINISTIC="${SPINGNN_DETERMINISTIC:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

"$PYTHON" - <<'EOF' || exit 2
import torch

assert torch.cuda.is_available(), "gpu_lammps_tests requires a CUDA device"
print(f"GPU_ENV device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
EOF

# Keep CUDA stdout and JUnit next to the build so acceptance remains reviewable.
mkdir -p "$BUILD_ROOT"
SPINGNN_GPU_EVIDENCE_DIR=${SPINGNN_GPU_EVIDENCE_DIR:-$(mktemp -d "$BUILD_ROOT/evidence.XXXXXX")}
mkdir -p "$SPINGNN_GPU_EVIDENCE_DIR"
export SPINGNN_GPU_EVIDENCE_DIR
torch_library_path=$("$PYTHON" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')
export LD_LIBRARY_PATH="$torch_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

passed=0
failed=0
declare -A results

record() {
  local name=$1 status=$2
  if ((status == 0)); then
    results[$name]=PASS
    ((passed += 1))
  else
    results[$name]=FAIL
    ((failed += 1))
  fi
  echo "CI_SUITE_RESULT name=$name result=${results[$name]} exit_code=$status"
}

require_junit_passes() {
  local junit_xml=$1 minimum_passed=$2
  "$PYTHON" - "$junit_xml" "$minimum_passed" <<'PY'
import pathlib
import sys
import xml.etree.ElementTree as ET

path = pathlib.Path(sys.argv[1])
minimum = int(sys.argv[2])
root = ET.parse(path).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
failures = sum(int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0)) for suite in suites)
passed = tests - skipped - failures
print(f"CI_SUITE_COUNTS tests={tests} skipped={skipped} passed={passed}")
if passed < minimum:
    raise SystemExit(
        f"required lane passed only {passed} test(s); minimum is {minimum}"
    )
PY
}

# --- Stage 1: CUDA serial build + TorchScript/PT2 parity -------------------
echo "CI_SUITE_START name=build-serial-cuda"
SERIAL_BIN=${SPINGNN_LAMMPS_BIN:-$BUILD_ROOT/gpu-serial/lmp}
if [[ "${SPINGNN_SKIP_LAMMPS_BUILD:-0}" == 1 ]]; then
  if [[ ! -x "$SERIAL_BIN" ]]; then
    echo "ERROR: SPINGNN_SKIP_LAMMPS_BUILD=1 but serial binary is unavailable: $SERIAL_BIN" >&2
    serial_build_status=2
  else
    serial_build_status=0
    echo "LAMMPS_BUILD_REUSED mode=serial binary=$SERIAL_BIN"
  fi
else
  PYTHON="$PYTHON" JOBS="${JOBS:-8}" \
    "$REPO_ROOT/tools/lammps/build_lammps.sh" serial "$BUILD_ROOT/gpu-serial"
  serial_build_status=$?
  SERIAL_BIN="$BUILD_ROOT/gpu-serial/lmp"
fi
if ((serial_build_status == 0)); then
  "$PYTHON" "$SCRIPT_DIR/gpu_runtime_gate.py" --binary "$SERIAL_BIN" || serial_build_status=$?
fi
record build-serial-cuda "$serial_build_status"

if [[ "${results[build-serial-cuda]}" == PASS ]]; then
  echo "CI_SUITE_START name=gpu-lammps-parity"
  gpu_lammps_junit="$SPINGNN_GPU_EVIDENCE_DIR/serial.xml"
  (
    cd "$REPO_ROOT" && \
    SPINGNN_REQUIRE_LAMMPS=1 \
    SPINGNN_REQUIRE_CUDA_LAMMPS=1 \
    SPINGNN_LAMMPS_BIN="$SERIAL_BIN" \
    "$PYTHON" -m pytest tests_lammps -q -ra -p no:cacheprovider \
      -m "lammps and not mpi and not slow" --junitxml="$gpu_lammps_junit" \
      --basetemp="$SPINGNN_GPU_EVIDENCE_DIR/serial-pytest"
  )
  gpu_lammps_status=$?
  if ((gpu_lammps_status == 0)); then
    require_junit_passes "$gpu_lammps_junit" 40 || gpu_lammps_status=$?
    "$PYTHON" "$SCRIPT_DIR/gpu_runtime_gate.py" --junit "$gpu_lammps_junit" || gpu_lammps_status=$?
  fi
  record gpu-lammps-parity "$gpu_lammps_status"
else
  echo "CI_SUITE_RESULT name=gpu-lammps-parity result=SKIPPED exit_code=0 (build failed)"
fi

# MPI parity is a separate contract: serial success does not prove rank-local
# ownership, ghost exchange, or device assignment. Run it only when the
# caller supplied a valid MPI binary, and fail closed when a required binary
# was requested but is unavailable.
MPI_BIN=${SPINGNN_LAMMPS_MPI_BIN:-}
if [[ -n "$MPI_BIN" && -x "$MPI_BIN" ]]; then
  echo "CI_SUITE_START name=gpu-lammps-mpi-parity"
  gpu_mpi_junit="$SPINGNN_GPU_EVIDENCE_DIR/mpi.xml"
  (
    cd "$REPO_ROOT" && \
    SPINGNN_REQUIRE_LAMMPS=1 \
    SPINGNN_REQUIRE_MPI=1 \
    SPINGNN_LAMMPS_BIN="$SERIAL_BIN" \
    SPINGNN_LAMMPS_MPI_BIN="$MPI_BIN" \
    "$PYTHON" -m pytest tests_lammps -q -ra -p no:cacheprovider \
      -m "mpi and not slow" --junitxml="$gpu_mpi_junit"
  )
  gpu_mpi_status=$?
  if ((gpu_mpi_status == 0)); then
    require_junit_passes "$gpu_mpi_junit" 1 || gpu_mpi_status=$?
  fi
  record gpu-lammps-mpi-parity "$gpu_mpi_status"
else
  if [[ "${SPINGNN_REQUIRE_MPI:-0}" == 1 ]]; then
    results[gpu-lammps-mpi-parity]=FAIL
    ((failed += 1))
    echo "CI_SUITE_RESULT name=gpu-lammps-mpi-parity result=FAIL exit_code=2 reason=missing_mpi_binary"
  else
    results[gpu-lammps-mpi-parity]=SKIPPED
    echo "CI_SUITE_RESULT name=gpu-lammps-mpi-parity result=SKIPPED exit_code=0 reason=mpi_binary_not_supplied"
  fi
fi

# --- Stage 2: opt-in Kokkos CUDA verification ------------------------------
# The architecture selects the pinned Kokkos release with deterministic ordering.
# PT2 remains the default supported GPU path. An unrequested lane is not a pass.
if [[ "${SPINGNN_REQUIRE_KOKKOS:-0}" != 1 ]]; then
  results[build-kokkos-cuda]=SKIPPED
  results[gpu-kokkos-parity]=SKIPPED
  echo "CI_SUITE_RESULT name=build-kokkos-cuda result=SKIPPED exit_code=0 reason=opt_in_not_requested hint=use_PT2_or_set_SPINGNN_REQUIRE_KOKKOS=1"
  echo "CI_SUITE_RESULT name=gpu-kokkos-parity result=SKIPPED exit_code=0 reason=opt_in_not_requested"
else
  kokkos_prepare_status=0
  if [[ -z "${KOKKOS_ROOT:-}" ]]; then
    kokkos_arch=${SPINGNN_KOKKOS_ARCH:-HOPPER90}
    kokkos_release=3.7.02
    case "$kokkos_arch" in
      ADA89|BLACKWELL120) kokkos_release=4.7.04 ;;
    esac
    KOKKOS_ROOT=${SPINGNN_KOKKOS_PREFIX:-$REPO_ROOT/validation/build/kokkos-${kokkos_release}-cuda-${kokkos_arch,,}}
    if [[ "$KOKKOS_ROOT" != /* ]]; then
      KOKKOS_ROOT=$REPO_ROOT/$KOKKOS_ROOT
    fi
    export KOKKOS_ROOT
    if [[ "${SPINGNN_SKIP_KOKKOS_INSTALL:-0}" == 1 ]]; then
      echo "CI_STEP_RESULT name=install-kokkos-cuda result=SKIPPED prefix=$KOKKOS_ROOT"
    elif [[ -x "$REPO_ROOT/tools/ci/install_kokkos_cuda.sh" ]]; then
      echo "CI_STEP_START name=install-kokkos-cuda prefix=$KOKKOS_ROOT"
      SPINGNN_KOKKOS_PREFIX="$KOKKOS_ROOT" JOBS="${JOBS:-8}" \
        "$REPO_ROOT/tools/ci/install_kokkos_cuda.sh"
      kokkos_prepare_status=$?
      if ((kokkos_prepare_status == 0)); then
        echo "CI_STEP_RESULT name=install-kokkos-cuda result=PASS exit_code=0"
      else
        echo "CI_STEP_RESULT name=install-kokkos-cuda result=FAIL exit_code=$kokkos_prepare_status"
      fi
    else
      echo "ERROR: KOKKOS_ROOT is unset and the CUDA Kokkos installer is unavailable." >&2
      kokkos_prepare_status=2
    fi
  fi

  echo "CI_SUITE_START name=build-kokkos-cuda"
  if ((kokkos_prepare_status == 0)); then
    PYTHON="$PYTHON" JOBS="${JOBS:-8}" \
      "$REPO_ROOT/tools/lammps/build_lammps.sh" kokkos-cuda "$BUILD_ROOT/gpu-kokkos"
    kokkos_build_status=$?
  else
    echo "ERROR: build-kokkos-cuda cannot start because Kokkos preparation failed." >&2
    kokkos_build_status=$kokkos_prepare_status
  fi
  record build-kokkos-cuda "$kokkos_build_status"

  if [[ "${results[build-kokkos-cuda]}" == PASS ]]; then
    echo "CI_SUITE_START name=gpu-kokkos-parity"
    gpu_kokkos_junit="$SPINGNN_GPU_EVIDENCE_DIR/kokkos.xml"
    (
      cd "$REPO_ROOT" && \
      SPINGNN_REQUIRE_LAMMPS=1 \
      SPINGNN_DETERMINISTIC=1 \
      CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
      SPINGNN_LAMMPS_BIN="$BUILD_ROOT/gpu-kokkos/lmp" \
      "$PYTHON" -m pytest tests_lammps/test_kokkos_parity.py \
        tests_lammps/test_kokkos_domain.py -q -ra \
        -p no:cacheprovider -m "lammps and not slow" \
        --junitxml="$gpu_kokkos_junit"
    )
    gpu_kokkos_status=$?
    if ((gpu_kokkos_status == 0)); then
      require_junit_passes "$gpu_kokkos_junit" 9 || gpu_kokkos_status=$?
    fi
    record gpu-kokkos-parity "$gpu_kokkos_status"
  else
    results[gpu-kokkos-parity]=SKIPPED
    echo "CI_SUITE_RESULT name=gpu-kokkos-parity result=SKIPPED exit_code=0 reason=build_failed"
  fi
fi

echo "CI_SUMMARY suites_passed=$passed suites_failed=$failed $(for k in "${!results[@]}"; do printf '%s=%s ' "$k" "${results[$k]}"; done)"
((failed == 0))
