#!/usr/bin/env bash
# Source after selecting the checkout, before importing torch or Triton.
# Defaults are private to a job/process. Explicit paths support controlled
# cold/warm comparisons without reusing an unrelated user's global cache.
if [[ -z "${SPINGNN_COMPILE_CACHE_ROOT:-}" ]]; then
  SPINGNN_COMPILE_CACHE_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/spingnn-compile-${SLURM_JOB_ID:-local}.XXXXXX") || return 2
fi
export SPINGNN_COMPILE_CACHE_ROOT
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SPINGNN_COMPILE_CACHE_ROOT/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SPINGNN_COMPILE_CACHE_ROOT/triton}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" || return 2
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
printf 'COMPILE_CACHE inductor=%s triton=%s\n' "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
