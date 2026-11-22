#!/usr/bin/env bash
# Build the bundled LAMMPS fork against the active torch.
#
# Usage: tools/lammps/build_lammps.sh [serial|mpi|kokkos-serial|kokkos-cuda] [BUILD_DIR]
#   PYTHON     python with torch installed (default: python)
#   JOBS       parallel build jobs (default: nproc, capped at 8)
#   USE_CCACHE set to 0 to disable ccache even when available (default: auto)
#   CUDACXX    CUDA compiler command, including optional arguments (default: nvcc)
#   KOKKOS_ROOT installed CUDA-enabled Kokkos (required for kokkos-cuda):
#               3.7.02 for AMPERE80/HOPPER90, 4.7.04 for ADA89/BLACKWELL120
#
# serial/mpi select MPI support, not the torch CPU/CUDA linkage. A CUDA torch
# requires a complete CUDA toolkit even for serial or kokkos-serial builds.
# CPU: create a separate virtualenv, run tools/ci/install_cpu_env.sh there,
#      then PYTHON=/path/to/cpu-env/bin/python tools/lammps/build_lammps.sh serial.
# GPU: run tools/ci/gpu_lammps_tests.sh on a GPU host or in a CUDA 12.8
#      development container with the matching CUDA torch environment.
#      An nvcc executable alone does not establish a complete toolkit.
# Deterministic H100 runs use external 3.7.02/HOPPER90 and SPINGNN_DETERMINISTIC=1
# with CUBLAS_WORKSPACE_CONFIG=:4096:8. PT2 remains the recommended GPU path;
# bundled Kokkos 3.4 is not supported with CUDA 12.8.
#
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

MODE=${1:-serial}
KOKKOS_ARGS=()
case "$MODE" in
  serial) BUILD_MPI=no ;;
  mpi)    BUILD_MPI=yes ;;
  kokkos-serial)
    BUILD_MPI=no
    KOKKOS_ARGS=(
      -D PKG_KOKKOS=yes
      -D Kokkos_ENABLE_SERIAL=on
      -D Kokkos_ENABLE_OPENMP=off
      -D Kokkos_ENABLE_CUDA=off
    )
    ;;
  kokkos-cuda)
    BUILD_MPI=no

    if [[ -z "${KOKKOS_ROOT:-}" ]]; then
      echo "ERROR: kokkos-cuda requires KOKKOS_ROOT to point to an installed" >&2
      echo "       CUDA-enabled Kokkos >= 3.7.0 (bundled Kokkos 3.4 is not CUDA 12 compatible)." >&2
      exit 2
    fi
    KOKKOS_CONFIG="$KOKKOS_ROOT/include/KokkosCore_config.h"
    KOKKOS_NVCC_WRAPPER="$KOKKOS_ROOT/bin/nvcc_wrapper"
    if [[ ! -f "$KOKKOS_CONFIG" ]]; then
      echo "ERROR: KOKKOS_ROOT has no include/KokkosCore_config.h: $KOKKOS_ROOT" >&2
      exit 2
    fi
    if [[ ! -x "$KOKKOS_NVCC_WRAPPER" ]]; then
      echo "ERROR: KOKKOS_ROOT has no executable bin/nvcc_wrapper: $KOKKOS_ROOT" >&2
      exit 2
    fi
    KOKKOS_VERSION=$(awk '$1 == "#define" && $2 == "KOKKOS_VERSION" { print $3; exit }' "$KOKKOS_CONFIG")
    if [[ ! "$KOKKOS_VERSION" =~ ^[0-9]+$ ]] || (( KOKKOS_VERSION < 30700 )); then
      echo "ERROR: kokkos-cuda requires Kokkos >= 3.7.0; found '${KOKKOS_VERSION:-unknown}'" >&2
      echo "       in $KOKKOS_CONFIG" >&2
      exit 2
    fi
    for feature in KOKKOS_ENABLE_CUDA KOKKOS_ENABLE_CUDA_LAMBDA; do
      if ! grep -Eq "^#define[[:space:]]+${feature}([[:space:]]|$)" "$KOKKOS_CONFIG"; then
        echo "ERROR: external Kokkos $KOKKOS_VERSION was not built with $feature" >&2
        echo "       ($KOKKOS_CONFIG)" >&2
        exit 2
      fi
    done
    KOKKOS_ARCHS=()
    for architecture in AMPERE80 HOPPER90 ADA89 BLACKWELL120; do
      if grep -Eq "^#define[[:space:]]+KOKKOS_ARCH_${architecture}([[:space:]]|$)" "$KOKKOS_CONFIG"; then
        KOKKOS_ARCHS+=("$architecture")
      fi
    done
    if ((${#KOKKOS_ARCHS[@]} != 1)); then
      echo "ERROR: external Kokkos must declare exactly one supported Kokkos CUDA architecture: AMPERE80, HOPPER90, ADA89 or BLACKWELL120" >&2
      echo "       ($KOKKOS_CONFIG)" >&2
      exit 2
    fi
    KOKKOS_ARCH=${KOKKOS_ARCHS[0]}
    case "$KOKKOS_ARCH" in
      AMPERE80) KOKKOS_CUDA_ARCH=sm_80; KOKKOS_EXPECTED_VERSION=30702; KOKKOS_RELEASE=3.7.02 ;;
      HOPPER90) KOKKOS_CUDA_ARCH=sm_90; KOKKOS_EXPECTED_VERSION=30702; KOKKOS_RELEASE=3.7.02 ;;
      ADA89) KOKKOS_CUDA_ARCH=sm_89; KOKKOS_EXPECTED_VERSION=40704; KOKKOS_RELEASE=4.7.04 ;;
      BLACKWELL120) KOKKOS_CUDA_ARCH=sm_120; KOKKOS_EXPECTED_VERSION=40704; KOKKOS_RELEASE=4.7.04 ;;
    esac
    if [[ "$KOKKOS_VERSION" != "$KOKKOS_EXPECTED_VERSION" ]]; then
      echo "ERROR: $KOKKOS_ARCH requires Kokkos $KOKKOS_RELEASE; found '$KOKKOS_VERSION'" >&2
      exit 2
    fi
    KOKKOS_ARGS=(
      -D PKG_KOKKOS=yes
      -D EXTERNAL_KOKKOS=yes
      -D Kokkos_ENABLE_CUDA=on
      -D "Kokkos_ROOT=$KOKKOS_ROOT"
      -D "CMAKE_CXX_FLAGS=-arch=$KOKKOS_CUDA_ARCH"
      -D "CMAKE_CXX_COMPILER=$KOKKOS_NVCC_WRAPPER"
    )
    ;;
  *)
    echo "ERROR: mode must be serial, mpi, kokkos-serial, or kokkos-cuda; got '$MODE'" >&2
    exit 2
    ;;
esac
BUILD_DIR=${2:-$REPO_ROOT/build/lammps-$MODE}

PYTHON=${PYTHON:-python}
JOBS=${JOBS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}

# CUDA containers on older host images can expose a system libstdc++ too old
# for C++17 binaries. Add the active compiler runtime before feature-gating
# the built executable; this is safe when the system runtime is already new.
if command -v gcc >/dev/null 2>&1; then
  GCC_RUNTIME=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
  if [[ -f "$GCC_RUNTIME/libstdc++.so.6" ]]; then
    export LD_LIBRARY_PATH="$GCC_RUNTIME${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

build_environment_error() {
  echo "ERROR: $1" >&2
  echo "       CPU: use a separate virtualenv prepared by tools/ci/install_cpu_env.sh" >&2
  echo "            and set PYTHON=/path/to/cpu-env/bin/python. Do not reinstall a shared env." >&2
  echo "       GPU: use a CUDA 12.8 development container or GPU host with the full toolkit" >&2
  echo "            and run tools/ci/gpu_lammps_tests.sh with the matching CUDA torch env." >&2
  echo "       serial/mpi select MPI support; CUDA torch still requires the CUDA toolkit." >&2
  exit "${2:-2}"
}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  build_environment_error "Python is not executable: PYTHON=$PYTHON"
fi
if ! TORCH_PREFIX=$("$PYTHON" -c 'import torch; print(torch.utils.cmake_prefix_path.splitlines()[0])'); then
  build_environment_error "Cannot import torch / read TorchConfig location using PYTHON=$PYTHON"
fi
TORCH_ROOT=$("$PYTHON" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)')

# A CUDA-build torch makes TorchConfig.cmake require the CUDA toolkit at
# configure time. Probe the selected compiler, including an explicit CUDACXX
# override; CMake still owns toolkit headers, libraries, and compatibility.
TORCH_CUDA=$("$PYTHON" -c 'import torch; print(torch.version.cuda or "")')
if [[ -n "$TORCH_CUDA" || "$MODE" == kokkos-cuda ]]; then
  if ! "$PYTHON" - "$TORCH_CUDA" <<'PY'
import os
import shlex
import shutil
import subprocess
import sys

selection = os.environ.get("CUDACXX") or "nvcc"
try:
    command = shlex.split(selection)
    if not command or shutil.which(command[0]) is None:
        raise ValueError(f"CUDA compiler is not executable: {selection}")
    result = subprocess.run(
        [*command, "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
        timeout=15,
    )
except (ValueError, OSError, subprocess.SubprocessError) as exc:
    print(f"ERROR: CUDA compiler precheck failed (CUDACXX={selection}): {exc}", file=sys.stderr)
    if isinstance(exc, subprocess.CalledProcessError) and exc.stdout:
        print(exc.stdout, file=sys.stderr)
    raise SystemExit(2) from exc
print(f"LAMMPS_BUILD_CUDA_COMPILER command={selection} torch_cuda={sys.argv[1] or 'none'}")
print(result.stdout.strip())
PY
  then
    build_environment_error "CUDA compiler is unavailable or unusable (PYTHON=$PYTHON, mode=$MODE)."
  fi
fi

CCACHE_ARGS=()
if [[ "$MODE" != kokkos-cuda && "${USE_CCACHE:-auto}" != 0 ]] && command -v ccache >/dev/null 2>&1; then
  CCACHE_ARGS=(
    -D CMAKE_C_COMPILER_LAUNCHER=ccache
    -D CMAKE_CXX_COMPILER_LAUNCHER=ccache
  )
fi

COMPILER_ARGS=()
if [[ -n "${CC:-}" ]]; then
  COMPILER_ARGS+=( -D "CMAKE_C_COMPILER=$CC" )
fi
if [[ -n "${CXX:-}" ]]; then
  COMPILER_ARGS+=( -D "CMAKE_CXX_COMPILER=$CXX" )
fi
if [[ -n "${CMAKE_CUDA_HOST_COMPILER:-}" ]]; then
  COMPILER_ARGS+=( -D "CMAKE_CUDA_HOST_COMPILER=$CMAKE_CUDA_HOST_COMPILER" )
fi

# pip torch wheels ship a TorchConfig.cmake whose imported target references
# MKL headers; when they are absent CMake aborts on the literal
# "MKL_INCLUDE_DIR-NOTFOUND" path. Our targets never include MKL headers
# (torch bundles MKL statically), so any existing directory satisfies the
# interface. Prefer real mkl-include headers when installed (pip mkl-include).
MKL_INCLUDE_DIR=$("$PYTHON" - <<'PY'
import glob
import os
import sysconfig

prefix_inc = os.path.join(sysconfig.get_config_var("prefix") or "", "include")
if prefix_inc and glob.glob(os.path.join(prefix_inc, "mkl*.h")):
    print(prefix_inc)
else:
    print("/usr/include")
PY
)

echo "LAMMPS_BUILD_START mode=$MODE build_dir=$BUILD_DIR jobs=$JOBS"
echo "torch=$("$PYTHON" -c 'import torch; print(torch.__version__)') prefix=$TORCH_PREFIX"
if [[ "$MODE" == kokkos-cuda ]]; then
  echo "external_kokkos=$KOKKOS_ROOT version=$KOKKOS_VERSION arch=$KOKKOS_ARCH cuda_arch=$KOKKOS_CUDA_ARCH"
fi

cmake --fresh -S "$REPO_ROOT/lammps/cmake" -B "$BUILD_DIR" \
  -D PKG_SPIN=yes \
  -D PKG_ASPHERE=yes \
  -D PKG_BROWNIAN=yes \
  -D BUILD_MPI="$BUILD_MPI" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=17 \
  -D CMAKE_CXX_STANDARD_REQUIRED=yes \
  -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -D "CMAKE_PREFIX_PATH=$TORCH_PREFIX" \
  -D "MKL_INCLUDE_DIR=$MKL_INCLUDE_DIR" \
  "${COMPILER_ARGS[@]}" \
  "${KOKKOS_ARGS[@]}" \
  "${CCACHE_ARGS[@]}" || build_environment_error "CMake configure failed; see the diagnostic above (PYTHON=$PYTHON)." "$?"

cmake --build "$BUILD_DIR" --parallel "$JOBS"
test -x "$BUILD_DIR/lmp"

# Feature gate: the fork's ML pair styles and spin/BEC machinery must be in.
export LD_LIBRARY_PATH="$TORCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HELP=$("$BUILD_DIR/lmp" -h)
for feature in 'allegro' 'nequip' 'spin/allegro_spin' 'spin/nequip_spin' 'bec' 'brownian'; do
  if ! grep -Eq "(^|[[:space:]])${feature}([[:space:]]|$)" <<<"$HELP"; then
    echo "ERROR: built lmp is missing feature '$feature'" >&2
    exit 1
  fi
done
if [[ "$MODE" == kokkos-* ]]; then
  for feature in 'allegro_spin/kk' 'spin/allegro_spin/kk' 'spin/kk'; do
    if ! grep -Eq "(^|[[:space:]])${feature}([[:space:]]|$)" <<<"$HELP"; then
      echo "ERROR: built lmp is missing Kokkos feature '$feature'" >&2
      exit 1
    fi
  done
fi

echo "LAMMPS_BUILD_OK mode=$MODE binary=$BUILD_DIR/lmp"
