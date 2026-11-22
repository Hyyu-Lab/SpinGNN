#!/usr/bin/env bash
# Install the CUDA-enabled Kokkos required by the LAMMPS kokkos-cuda build.
#
# Usage: tools/ci/install_kokkos_cuda.sh [PREFIX]
#   SPINGNN_KOKKOS_ARCH    AMPERE80/HOPPER90 (3.7.02) or ADA89/BLACKWELL120
#                         (4.7.04); default HOPPER90 preserves the H100 profile
#   SPINGNN_KOKKOS_PREFIX  default install prefix when PREFIX is omitted
#   SPINGNN_KOKKOS_TARBALL existing matching official tarball (offline install)
#   SPINGNN_KOKKOS_URL     download URL override
#   JOBS                   parallel build jobs (default: nproc, capped at 8)
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

KOKKOS_ARCH=${SPINGNN_KOKKOS_ARCH:-HOPPER90}
case "$KOKKOS_ARCH" in
  AMPERE80) KOKKOS_CUDA_ARCH=sm_80 ;;
  HOPPER90) KOKKOS_CUDA_ARCH=sm_90 ;;
  ADA89) KOKKOS_CUDA_ARCH=sm_89 ;;
  BLACKWELL120) KOKKOS_CUDA_ARCH=sm_120 ;;
  *)
    echo "ERROR: SPINGNN_KOKKOS_ARCH must be AMPERE80, HOPPER90, ADA89 or BLACKWELL120; got '$KOKKOS_ARCH'" >&2
    exit 2
    ;;
esac
case "$KOKKOS_ARCH" in
  AMPERE80|HOPPER90)
    KOKKOS_RELEASE=3.7.02
    KOKKOS_VERSION_CODE=30702
    KOKKOS_EXPECTED_SHA256=5024979f06bc8da2fb696252a66297f3e0e67098595a0cc7345312b3b4aa0f54
    KOKKOS_DEFAULT_URL=https://github.com/kokkos/kokkos/archive/refs/tags/${KOKKOS_RELEASE}.tar.gz
    ;;
  ADA89|BLACKWELL120)
    KOKKOS_RELEASE=4.7.04
    KOKKOS_VERSION_CODE=40704
    # Pin the release commit as well as archive bytes; do not trust a moved tag.
    KOKKOS_EXPECTED_SHA256=b65b7d2f03c5eee97e574566fd009daacc6ebea3a1fe69e14024e7ae94e997cc
    KOKKOS_DEFAULT_URL=https://api.github.com/repos/kokkos/kokkos/tarball/82799e4577568f9666bde36265ac15d78da3e6c8
    ;;
esac
KOKKOS_URL=${SPINGNN_KOKKOS_URL:-$KOKKOS_DEFAULT_URL}
DEFAULT_PREFIX=$REPO_ROOT/validation/build/kokkos-${KOKKOS_RELEASE}-cuda-${KOKKOS_ARCH,,}

if (($# > 1)); then
  echo "ERROR: usage: tools/ci/install_kokkos_cuda.sh [PREFIX]" >&2
  exit 2
fi
PREFIX=${1:-${SPINGNN_KOKKOS_PREFIX:-$DEFAULT_PREFIX}}
if [[ "$PREFIX" != /* ]]; then
  PREFIX=$PWD/$PREFIX
fi
if [[ "$PREFIX" == / ]]; then
  echo "ERROR: refusing to install Kokkos into filesystem root" >&2
  exit 2
fi

JOBS=${JOBS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: JOBS must be a positive integer; got '$JOBS'" >&2
  exit 2
fi

KOKKOS_INSTALLED_VERSION=
validate_install() {
  local prefix=$1
  local verbose=${2:-0}
  local config=$prefix/include/KokkosCore_config.h
  local cmake_config=$prefix/lib/cmake/Kokkos/KokkosConfig.cmake
  local wrapper=$prefix/bin/nvcc_wrapper
  local version feature architecture library library_path probe_dir probe_status

  if [[ ! -f "$cmake_config" ]]; then
    if ((verbose)); then
      echo "ERROR: installed KokkosConfig.cmake is missing: $cmake_config" >&2
    fi
    return 1
  fi
  if [[ ! -f "$config" ]]; then
    if ((verbose)); then
      echo "ERROR: installed KokkosCore_config.h is missing: $config" >&2
    fi
    return 1
  fi
  if [[ ! -x "$wrapper" ]]; then
    if ((verbose)); then
      echo "ERROR: installed nvcc_wrapper is missing or not executable: $wrapper" >&2
    fi
    return 1
  fi

  version=$(awk '$1 == "#define" && $2 == "KOKKOS_VERSION" { print $3; exit }' "$config")
  if [[ "$version" != "$KOKKOS_VERSION_CODE" ]]; then
    if ((verbose)); then
      echo "ERROR: $KOKKOS_ARCH requires Kokkos $KOKKOS_RELEASE; found '${version:-unknown}'" >&2
    fi
    return 1
  fi
  for feature in \
    KOKKOS_ENABLE_SERIAL \
    KOKKOS_ENABLE_CUDA \
    KOKKOS_ENABLE_CUDA_LAMBDA \
    "KOKKOS_ARCH_$KOKKOS_ARCH"; do
    if ! grep -Eq "^#define[[:space:]]+${feature}([[:space:]]|$)" "$config"; then
      if ((verbose)); then
        echo "ERROR: installed Kokkos $version is missing $feature ($config)" >&2
      fi
      return 1
    fi
  done
  for architecture in AMPERE80 HOPPER90 ADA89 BLACKWELL120; do
    if [[ "$architecture" != "$KOKKOS_ARCH" ]] &&
      grep -Eq "^#define[[:space:]]+KOKKOS_ARCH_${architecture}([[:space:]]|$)" "$config"; then
      if ((verbose)); then
        echo "ERROR: installed Kokkos declares multiple supported CUDA architectures ($config)" >&2
      fi
      return 1
    fi
  done

  for library in kokkoscore kokkoscontainers kokkossimd; do
    library_path=
    for candidate in \
      "$prefix/lib/lib${library}.a" \
      "$prefix/lib/lib${library}.so" \
      "$prefix/lib64/lib${library}.a" \
      "$prefix/lib64/lib${library}.so"; do
      if [[ -f "$candidate" ]]; then
        library_path=$candidate
        break
      fi
    done
    if [[ -z "$library_path" ]]; then
      if ((verbose)); then
        echo "ERROR: installed Kokkos library is missing: lib${library}.{a,so}" >&2
      fi
      return 1
    fi
  done

  if ! command -v cmake >/dev/null 2>&1; then
    if ((verbose)); then
      echo "ERROR: cmake is required for the installed Kokkos link probe" >&2
    fi
    return 1
  fi
  probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/spingnn-kokkos-probe.XXXXXX")
  cat >"$probe_dir/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(spingnn_kokkos_probe LANGUAGES CXX)
# Kokkos 3.7 does not export all transitive CUDA dependency targets.
find_package(CUDAToolkit REQUIRED)
find_package(Kokkos CONFIG REQUIRED)
add_executable(kokkos_probe probe.cpp)
target_link_libraries(kokkos_probe PRIVATE Kokkos::kokkos)
EOF
  cat >"$probe_dir/probe.cpp" <<'EOF'
#include <Kokkos_Core.hpp>
int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  Kokkos::finalize();
  return 0;
}
EOF
  probe_status=0
  cmake -S "$probe_dir" -B "$probe_dir/build" \
    -D "CMAKE_PREFIX_PATH=$prefix" \
    -D "CMAKE_CXX_COMPILER=$wrapper" >/dev/null 2>&1 || probe_status=$?
  if ((probe_status == 0)); then
    cmake --build "$probe_dir/build" --parallel 1 >/dev/null 2>&1 \
      || probe_status=$?
  fi
  rm -rf -- "$probe_dir"
  if ((probe_status != 0)); then
    if ((verbose)); then
      echo "ERROR: installed Kokkos failed a find_package compile/link probe" >&2
    fi
    return 1
  fi
  KOKKOS_INSTALLED_VERSION=$version
}

if [[ -e "$PREFIX" ]]; then
  if [[ ! -d "$PREFIX" ]]; then
    echo "ERROR: Kokkos install prefix exists but is not a directory: $PREFIX" >&2
    exit 2
  fi
  if validate_install "$PREFIX" 0; then
    echo "KOKKOS_CUDA_INSTALL_OK prefix=$PREFIX version=$KOKKOS_INSTALLED_VERSION reused=1 arch=$KOKKOS_ARCH cuda_arch=$KOKKOS_CUDA_ARCH"
    exit 0
  fi
  if [[ -n "$(find "$PREFIX" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    validate_install "$PREFIX" 1 || true
    echo "ERROR: refusing to overwrite non-empty invalid Kokkos prefix: $PREFIX" >&2
    echo "       Choose a new prefix or remove the stale prefix explicitly." >&2
    exit 2
  fi
fi

for command_name in cmake sha256sum tar; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "ERROR: required command is not available: $command_name" >&2
    exit 2
  fi
done

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/spingnn-kokkos-cuda.XXXXXX")
trap 'rm -rf -- "$WORK_DIR"' EXIT
TARBALL=$WORK_DIR/kokkos-${KOKKOS_RELEASE}.tar.gz

if [[ -n "${SPINGNN_KOKKOS_TARBALL:-}" ]]; then
  TARBALL=$SPINGNN_KOKKOS_TARBALL
  if [[ "$TARBALL" != /* ]]; then
    TARBALL=$PWD/$TARBALL
  fi
  if [[ ! -f "$TARBALL" ]]; then
    echo "ERROR: SPINGNN_KOKKOS_TARBALL does not exist: $TARBALL" >&2
    exit 2
  fi
  echo "KOKKOS_TARBALL_SOURCE mode=local path=$TARBALL"
else
  echo "KOKKOS_TARBALL_SOURCE mode=download url=$KOKKOS_URL"
  download_status=0
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --retry-delay 2 \
      --connect-timeout 20 --max-time 300 \
      "$KOKKOS_URL" --output "$TARBALL" || download_status=$?
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --timeout=20 "$KOKKOS_URL" --output-document="$TARBALL" \
      || download_status=$?
  else
    download_status=127
  fi
  if ((download_status != 0)); then
    echo "ERROR: failed to download Kokkos $KOKKOS_RELEASE (exit $download_status)." >&2
    echo "       Network access may be restricted in this container." >&2
    echo "       Stage the official tarball and set:" >&2
    echo "       SPINGNN_KOKKOS_TARBALL=/path/to/kokkos-${KOKKOS_RELEASE}.tar.gz" >&2
    echo "       Expected SHA-256: $KOKKOS_EXPECTED_SHA256" >&2
    exit 1
  fi
fi

actual_sha256=$(sha256sum "$TARBALL" | awk '{print $1}')
if [[ "$actual_sha256" != "$KOKKOS_EXPECTED_SHA256" ]]; then
  echo "ERROR: Kokkos tarball SHA-256 mismatch: $TARBALL" >&2
  echo "       expected=$KOKKOS_EXPECTED_SHA256" >&2
  echo "       actual=$actual_sha256" >&2
  exit 1
fi
echo "KOKKOS_TARBALL_OK version=$KOKKOS_RELEASE sha256=$actual_sha256"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc is required to build CUDA Kokkos but is not on PATH." >&2
  echo "       Run this installer inside the CUDA 12.8 GPU build container." >&2
  exit 2
fi

SOURCE_DIR=$WORK_DIR/kokkos-${KOKKOS_RELEASE}
mkdir -p "$SOURCE_DIR"
# GitHub release and commit archives use different top-level directory names.
tar -C "$SOURCE_DIR" --strip-components=1 -xzf "$TARBALL"
BUILD_DIR=$WORK_DIR/build
if [[ ! -x "$SOURCE_DIR/bin/nvcc_wrapper" ]]; then
  echo "ERROR: verified archive has no executable bin/nvcc_wrapper" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$PREFIX")"
echo "KOKKOS_CUDA_INSTALL_START version=$KOKKOS_RELEASE prefix=$PREFIX jobs=$JOBS arch=$KOKKOS_ARCH cuda_arch=$KOKKOS_CUDA_ARCH"
cmake --fresh -S "$SOURCE_DIR" -B "$BUILD_DIR" \
  -D "CMAKE_INSTALL_PREFIX=$PREFIX" \
  -D "CMAKE_INSTALL_LIBDIR=lib" \
  -D "CMAKE_CXX_COMPILER=$SOURCE_DIR/bin/nvcc_wrapper" \
  -D "CMAKE_CXX_FLAGS=-arch=$KOKKOS_CUDA_ARCH" \
  -D CMAKE_CXX_STANDARD=17 \
  -D CMAKE_BUILD_TYPE=Release \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
  -D "Kokkos_ARCH_$KOKKOS_ARCH=ON" \
  -D Kokkos_ENABLE_TESTS=OFF \
  -D BUILD_SHARED_LIBS=OFF
cmake --build "$BUILD_DIR" --parallel "$JOBS" --target install

validate_install "$PREFIX" 1
echo "KOKKOS_CUDA_INSTALL_OK prefix=$PREFIX version=$KOKKOS_INSTALLED_VERSION reused=0 arch=$KOKKOS_ARCH cuda_arch=$KOKKOS_CUDA_ARCH"
