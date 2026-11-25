#!/usr/bin/env bash
# Single-entry SpinGNN environment installer.
#
# Usage:
#   tools/ci/install.sh [--gpu auto|4090|5090|a100|h100|cpu|v100]
#
# --gpu auto (default) detects the first card via nvidia-smi; no nvidia-smi or
# no visible GPU selects the cpu tier. GPU tiers delegate to
# install_gpu_env.sh (the repository-local, SHA-verified CUDA 12.8 stack under
# validation/); the cpu tier delegates to install_cpu_env.sh and therefore
# needs an activated virtualenv or PYTHON=<path>. v100 fails closed with the
# measured stack diagnosis shared with gpu_full_tests.sh; unknown names fail
# closed listing the supported set.
#
# After the delegated installer finishes, this script verifies source-exact
# imports through source_python.py and, for GPU tiers with a GPU visible at
# runtime, asserts the device name and compute capability match the tier. On
# a GPU-less install host the device assertion prints a SKIP note.
#
# Exit codes (tools/ci convention): 0 success, 2 usage/environment error.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

SUPPORTED="auto, 4090, 5090, a100, h100, cpu (v100 is rejected with a stack diagnosis)"

usage() {
  echo "Usage: $0 [--gpu auto|4090|5090|a100|h100|cpu|v100]"
}

v100_reject() {
  # Measured against this exact stack, not assumed (body lines shared with
  # gpu_full_tests.sh): the torch 2.11.0+cu128 wheel ships SASS for
  # sm_75/80/86/90/100/120 and PTX no lower than compute_75 (extracted from
  # libtorch_cuda.so, 2026-09-01); PTX only JITs forward, so sm_70 cannot
  # run any kernel in this build.
  echo "ERROR: --gpu v100 is not runnable on this stack:" >&2
  echo "  torch 2.11.0+cu128 ships no sm_70 SASS and its minimum PTX is" >&2
  echo "  compute_75, which cannot JIT down to Volta. A V100 lane needs a" >&2
  echo "  separately pinned torch/CUDA environment." >&2
  exit 2
}

# Map an nvidia-smi card name or CLI word to a tier; unknown input returns 1.
classify_gpu() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    *4090*) echo 4090 ;;
    *5090*) echo 5090 ;;
    *a100*) echo a100 ;;
    *h100*) echo h100 ;;
    *v100*) echo v100 ;;
    *) return 1 ;;
  esac
}

GPU=auto
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      GPU=$2
      shift 2
      ;;
    --gpu=*) GPU=${1#*=}; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
GPU=$(printf '%s' "$GPU" | tr '[:upper:]' '[:lower:]')

# A query failure is not evidence of a CPU-only machine. Probe explicit GPU
# tiers too, so runtime initialization failures cannot masquerade as GPU-less
# installs. CPU and rejected selectors do not depend on NVIDIA tooling.
NAME=""
case "$GPU" in
  auto | 4090 | 5090 | a100 | h100)
    if command -v nvidia-smi >/dev/null 2>&1; then
      if ! NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p'); then
        echo "ERROR: nvidia-smi query failed; cannot determine GPU availability. Use --gpu cpu only for an intentional CPU install." >&2
        exit 2
      fi
    fi
    ;;
esac

case "$GPU" in
  auto)
    if [[ -z "${NAME//[[:space:]]/}" ]]; then
      TIER=cpu
      echo "INSTALL_STEP detect gpu=auto: no nvidia-smi/GPU visible -> tier=cpu"
    elif TIER=$(classify_gpu "$NAME"); then
      echo "INSTALL_STEP detect gpu=auto first_card='$NAME' -> tier=$TIER"
    else
      echo "ERROR: auto-detected GPU '$NAME' is not in the supported set: 4090, 5090, a100, h100" >&2
      echo "  (v100 is rejected with a stack diagnosis). Use --gpu cpu for a" >&2
      echo "  CPU-only install or --gpu <tier> to select one explicitly." >&2
      exit 2
    fi
    ;;
  cpu) TIER=cpu ;;
  4090 | 5090 | a100 | h100) TIER=$GPU ;;
  v100) v100_reject ;;
  *)
    echo "ERROR: --gpu must be exactly one of: $SUPPORTED; got $GPU" >&2
    exit 2
    ;;
esac
if [[ "$TIER" == v100 ]]; then
  # Auto-detection lands here on a V100 host.
  v100_reject
fi

# Expected device-name substring and compute capability per GPU tier, used by
# the post-install runtime assertion.
case "$TIER" in
  4090) EXPECT_NAME=4090; EXPECT_CAP="8,9" ;;
  5090) EXPECT_NAME=5090; EXPECT_CAP="12,0" ;;
  a100) EXPECT_NAME=A100; EXPECT_CAP="8,0" ;;
  h100) EXPECT_NAME=H100; EXPECT_CAP="9,0" ;;
  cpu) EXPECT_NAME=""; EXPECT_CAP="" ;;
esac

if [[ "$TIER" == cpu ]]; then
  if [[ -z "${PYTHON:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: the cpu tier installs into the active interpreter (install_cpu_env.sh" >&2
    echo "  contract): activate a virtualenv or set PYTHON=<venv-python> first." >&2
    echo "  Refusing to pip-install into the system python." >&2
    exit 2
  fi
  PYTHON_BIN=${PYTHON:-$VIRTUAL_ENV/bin/python}
  PYTHON_BIN=$(command -v -- "$PYTHON_BIN") || {
    echo "ERROR: CPU interpreter not found: ${PYTHON:-$VIRTUAL_ENV/bin/python}" >&2
    exit 2
  }
  [[ -x "$PYTHON_BIN" ]] || { echo "ERROR: CPU interpreter is not executable: $PYTHON_BIN" >&2; exit 2; }
  echo "INSTALL_STEP delegate install_cpu_env.sh python=$PYTHON_BIN"
  PYTHON="$PYTHON_BIN" "$SCRIPT_DIR/install_cpu_env.sh"
else
  echo "INSTALL_STEP delegate install_gpu_env.sh tier=$TIER"
  "$SCRIPT_DIR/install_gpu_env.sh"
  PYTHON_BIN="$REPO_ROOT/validation/env/spingnn-cu128/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: post-install python is not executable: $PYTHON_BIN" >&2
  exit 2
fi

echo "INSTALL_STEP verify source-exact imports and device tier=$TIER python=$PYTHON_BIN"
SPINGNN_REPO_ROOT="$REPO_ROOT" \
SPINGNN_INSTALL_TIER="$TIER" \
SPINGNN_INSTALL_EXPECT_NAME="$EXPECT_NAME" \
SPINGNN_INSTALL_EXPECT_CAP="$EXPECT_CAP" \
SPINGNN_INSTALL_DETECTED_GPU="$NAME" \
"$PYTHON_BIN" "$SCRIPT_DIR/source_python.py" -c '
import os
import pathlib

import torch

import allegro
import e3nn
import nequip

root = pathlib.Path(os.environ["SPINGNN_REPO_ROOT"]).resolve()
for module in (nequip, allegro):
    origin = pathlib.Path(module.__file__).resolve()
    assert root in origin.parents, (
        module.__name__ + " must resolve inside the checkout " + str(root)
        + ", got " + str(origin)
    )
assert e3nn.__version__ == "0.5.0+spin", e3nn.__version__
assert nequip.__version__ == "0.5.6+spin", nequip.__version__
assert allegro.__version__ == "0.2.0+spin", allegro.__version__
print("IMPORT_OK")
print("nequip", nequip.__version__, nequip.__file__)
print("allegro", allegro.__version__, allegro.__file__)
print("e3nn", e3nn.__version__, e3nn.__file__)

tier = os.environ["SPINGNN_INSTALL_TIER"]
if tier == "cpu":
    print("DEVICE_SKIP tier=cpu (no device assertion)")
elif not torch.cuda.is_available():
    detected = os.environ["SPINGNN_INSTALL_DETECTED_GPU"].strip()
    if detected:
        raise RuntimeError(
            "nvidia-smi detected " + detected + " but PyTorch cannot initialize CUDA; "
            "check the driver/runtime and CUDA_VISIBLE_DEVICES before accepting this install"
        )
    print("DEVICE_SKIP torch.cuda.is_available() is False; the install host has"
          " no visible GPU, so the device-name/capability assertion is skipped")
else:
    expect_name = os.environ["SPINGNN_INSTALL_EXPECT_NAME"]
    expect_cap = tuple(
        int(part) for part in os.environ["SPINGNN_INSTALL_EXPECT_CAP"].split(",")
    )
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    assert expect_name in name, (
        "expected a " + expect_name + " device for tier " + tier
        + ", got " + name
    )
    assert capability == expect_cap, (
        "expected compute capability " + str(expect_cap) + " for tier " + tier
        + ", got " + str(capability) + " (" + name + ")"
    )
    print("DEVICE_OK", name, "capability", capability)
'

echo "INSTALL_OK tier=$TIER python=$PYTHON_BIN"
