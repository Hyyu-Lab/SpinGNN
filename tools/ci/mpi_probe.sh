#!/usr/bin/env bash
# Probe a two-rank MPI launch, falling back to transports that avoid UCX.
set -uo pipefail

MPIRUN=${MPI_PROBE_MPIRUN:-${SPINGNN_MPIRUN:-mpirun}}
RANKS=${MPI_PROBE_RANKS:-2}
TIMEOUT_SECONDS=${MPI_PROBE_TIMEOUT_SECONDS:-20}

if ! [[ "$RANKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MPI_PROBE_RESULT status=ERROR reason=invalid_ranks value=$RANKS" >&2
  exit 2
fi
if ! [[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MPI_PROBE_RESULT status=ERROR reason=invalid_timeout value=$TIMEOUT_SECONDS" >&2
  exit 2
fi
if ! command -v "$MPIRUN" >/dev/null 2>&1; then
  echo "MPI_PROBE_RESULT status=UNAVAILABLE reason=mpirun_not_found launcher=$MPIRUN" >&2
  exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "MPI_PROBE_RESULT status=UNAVAILABLE reason=timeout_not_found" >&2
  exit 1
fi

MPIRUN=$(command -v "$MPIRUN")
VERSION=$("$MPIRUN" --version 2>&1 | head -n 1)
BASE_ARGS=()
BASE_ENV=()
if [[ "$VERSION" == *"Open MPI"* || "$VERSION" == *"OpenRTE"* ]]; then
  BASE_ARGS+=(--oversubscribe)
  if (( $(id -u) == 0 )); then
    BASE_ENV+=(OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1)
  fi
fi

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/spingnn-mpi-probe.XXXXXX")
trap 'rm -rf -- "$TMP_DIR"' EXIT

# The probe payload MUST be a real MPI program.
#
# It used to be `hostname`, which never calls MPI_Init: the probe then only
# proved that the launcher could spawn N processes and said nothing about
# whether any transport works. Every combination therefore "passed" on the first
# (default) attempt, the UCX-avoiding fallbacks below became dead code, and the
# exported environment could then fail in MPI_Init for real workloads.
#
# MPI_Allreduce forces endpoint creation between ranks, which is exactly the
# step that fails under a broken UCX setup. Each rank still prints its hostname
# so the existing "one non-empty line per rank" check is unchanged.
MPICC=${MPI_PROBE_MPICC:-mpicc}
PAYLOAD_SRC="$TMP_DIR/probe.c"
PAYLOAD_BIN="$TMP_DIR/probe"
cat >"$PAYLOAD_SRC" <<'CSRC'
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {
    int rank = 0, size = 0, sum = 0, expected = 0;
    char host[256];
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        return 3;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    /* Forces inter-rank endpoint setup; this is what fails under broken UCX. */
    if (MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD)
        != MPI_SUCCESS) {
        MPI_Finalize();
        return 4;
    }
    if (gethostname(host, sizeof host) != 0) {
        MPI_Finalize();
        return 5;
    }
    printf("%s\n", host);
    fflush(stdout);
    MPI_Finalize();
    expected = size * (size - 1) / 2;
    return sum == expected ? 0 : 6;
}
CSRC

if ! command -v "$MPICC" >/dev/null 2>&1; then
  # Fail closed. Falling back to a non-MPI payload is what made this probe
  # useless in the first place.
  echo "MPI_PROBE_RESULT status=UNAVAILABLE reason=mpicc_not_found compiler=$MPICC" >&2
  exit 1
fi

build_payload() {
  "$MPICC" -O0 -o "$PAYLOAD_BIN" "$PAYLOAD_SRC" "$@" >"$TMP_DIR/mpicc.log" 2>&1
}

# Some environments put a toolchain on PATH whose linker does not search the
# system library directories (e.g. a conda gcc in front of a distro Open MPI).
# The MPI libraries are then found but their transitive dependencies are not,
# producing `undefined reference to opal_*`. Retry with explicit -rpath-link
# entries before declaring the probe unavailable, so a fixable link problem is
# not reported as "no working MPI".
if ! build_payload; then
  LINK_DIRS=()
  while IFS= read -r dir; do
    [ -n "$dir" ] && LINK_DIRS+=("$dir")
  done < <("$MPICC" -showme:libdirs 2>/dev/null | tr ' ' '\n')
  for extra in /usr/lib/"$(uname -m)"-linux-gnu /usr/lib64 /usr/lib; do
    [ -d "$extra" ] && LINK_DIRS+=("$extra")
  done
  RPATH_ARGS=()
  for dir in "${LINK_DIRS[@]}"; do
    RPATH_ARGS+=("-Wl,-rpath-link,$dir" "-Wl,-rpath,$dir")
  done
  if ((${#RPATH_ARGS[@]} == 0)) || ! build_payload "${RPATH_ARGS[@]}"; then
    echo "MPI_PROBE_RESULT status=UNAVAILABLE reason=mpi_payload_build_failed" \
      "compiler=$MPICC log=$(tr '\n' ' ' <"$TMP_DIR/mpicc.log" | cut -c1-300)" >&2
    exit 1
  fi
  echo "MPI_PROBE_NOTE payload_built_with_explicit_rpath_link dirs=${#LINK_DIRS[@]}"
fi

selected_name=
selected_args=()
selected_env=()
selected_hosts=
attempts=0

try_probe() {
  local name=$1
  shift
  local -a transport_args=("$@")
  local -a transport_env=()
  local log="$TMP_DIR/$name.log"
  local display_args display_env output
  local status nonempty_lines

  if [[ "$name" == "ob1_tcp_no_ucx" ]]; then
    transport_env+=(UCX_TLS=self,tcp UCX_POSIX_USE_PROC_LINK=n)
  fi

  # Mirror every `--mca key value` launcher argument into the equivalent
  # `OMPI_MCA_key=value` environment variable.
  #
  # Launcher arguments only reach processes started through mpirun. An
  # MPI-linked binary invoked DIRECTLY still calls MPI_Init and still needs a
  # working transport -- modern LAMMPS builds run `lmp_spingnn2 -h` as their
  # post-build feature gate. Exporting the selection as environment makes it
  # apply to direct runs and to any consumer that does not use mpirun.
  local index
  for ((index = 0; index < ${#transport_args[@]}; index++)); do
    if [[ "${transport_args[index]}" == "--mca" ]]; then
      transport_env+=("OMPI_MCA_${transport_args[index + 1]}=${transport_args[index + 2]}")
      ((index += 2))
    fi
  done

  display_args=
  display_env=
  if ((${#transport_args[@]})); then
    printf -v display_args '%q ' "${transport_args[@]}"
  fi
  if ((${#transport_env[@]})); then
    printf -v display_env '%q ' "${transport_env[@]}"
  fi
  ((attempts += 1))
  printf 'MPI_PROBE_ATTEMPT index=%d combination=%q transport_args=%q env=%q\n' \
    "$attempts" "$name" "${display_args% }" "${display_env% }"

  env "${BASE_ENV[@]}" "${transport_env[@]}" \
    timeout "${TIMEOUT_SECONDS}s" \
    "$MPIRUN" "${BASE_ARGS[@]}" "${transport_args[@]}" \
    -n "$RANKS" "$PAYLOAD_BIN" >"$log" 2>&1
  status=$?
  nonempty_lines=$(awk 'NF {count++} END {print count + 0}' "$log")
  output=$(tr '\n' ' ' <"$log")
  output=${output:0:500}

  if ((status == 0 && nonempty_lines == RANKS)); then
    selected_name=$name
    selected_args=("${BASE_ARGS[@]}" "${transport_args[@]}")
    selected_env=("${BASE_ENV[@]}" "${transport_env[@]}")
    selected_hosts=$(awk 'NF' "$log" | paste -sd, -)
    printf 'MPI_PROBE_ATTEMPT_RESULT index=%d status=PASS ranks=%d hosts=%q\n' \
      "$attempts" "$nonempty_lines" "$selected_hosts"
    return 0
  fi

  printf 'MPI_PROBE_ATTEMPT_RESULT index=%d status=FAIL exit_code=%d ranks=%d output=%q\n' \
    "$attempts" "$status" "$nonempty_lines" "$output"
  return 1
}

printf 'MPI_PROBE_START launcher=%q version=%q ranks=%d timeout_seconds=%d\n' \
  "$MPIRUN" "$VERSION" "$RANKS" "$TIMEOUT_SECONDS"

if try_probe default; then
  :
elif try_probe ob1 --mca pml ob1; then
  :
elif try_probe ob1_tcp_vader --mca pml ob1 --mca btl tcp,self,vader; then
  :
elif try_probe ob1_tcp_no_ucx --mca pml ob1 --mca btl tcp,self; then
  :
else
  printf 'MPI_PROBE_RESULT status=FAIL attempts=%d launcher=%q\n' "$attempts" "$MPIRUN" >&2
  exit 1
fi

SELECTED_ARGS=
SELECTED_ENV=
if ((${#selected_args[@]})); then
  printf -v SELECTED_ARGS '%q ' "${selected_args[@]}"
fi
if ((${#selected_env[@]})); then
  printf -v SELECTED_ENV '%q ' "${selected_env[@]}"
fi
printf 'MPI_PROBE_RESULT status=PASS combination=%q attempts=%d launcher_args=%q env=%q hosts=%q\n' \
  "$selected_name" "$attempts" "${SELECTED_ARGS% }" "${SELECTED_ENV% }" "$selected_hosts"
printf 'MPI_PROBE_EXPORT SPINGNN_MPIRUN=%q SPINGNN_MPIRUN_EXTRA_ARGS=%q SPINGNN_MPIRUN_EXTRA_ENV=%q\n' \
  "$MPIRUN" "${SELECTED_ARGS% }" "${SELECTED_ENV% }"
