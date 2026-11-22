# Standalone CUDA spin synchronization regression

Run compilation and execution inside the project's Slurm allocations.
The spin synchronization target is opt-in and uses a previously built
serial Kokkos LAMMPS library with matching headers and compiler settings.

## Spin field synchronization

`spin_sync.cpp` creates a real LAMMPS instance with two `spin/kk` atoms. Run
each case in a separate process because the broken masks can trigger a
Kokkos abort:

```bash
spin-sync-probe x-sp
spin-sync-probe v-fm
spin-sync-probe f-fml
spin-sync-probe async-fm
```

The first three cases leave the mechanical host field dirty while changing
the magnetic device field. They verify independent dirty flags, unchanged
mechanical data, and correct subsequent host/device synchronization. The
last case leaves spin synchronized but magnetic force host-dirty, then calls
the real `sync_overlapping_device(Device, FM_MASK)` and checks device values.
The program requires distinct host/CUDA allocations and prints the actual
GPU name/capability and the mask constants compiled into the test.

Configure with the **same** compiler, Kokkos installation, LAMMPS headers,
and library build used by the intended before/after experiment:

```bash
cmake -S lammps/unittest/spingnn -B "$TEST_BUILD" \
  -DCMAKE_CXX_COMPILER="$KOKKOS_ROOT/bin/nvcc_wrapper" \
  -DCMAKE_PREFIX_PATH="$KOKKOS_ROOT;$TORCH_CMAKE_PREFIX" \
  -DSPINGNN_BUILD_SPIN_SYNC_PROBE=ON \
  -DSPINGNN_LAMMPS_SOURCE_DIR="$FROZEN_SOURCE/lammps/src" \
  -DSPINGNN_LAMMPS_BUILD_DIR="$LAMMPS_BUILD" \
  -DSPINGNN_LAMMPS_LIBRARY="$LAMMPS_BUILD/liblammps.a"
cmake --build "$TEST_BUILD" --target spin-sync-probe --parallel 2
```

The target reads the original LAMMPS compile flags, rejects mismatched source
and library paths or a different Kokkos installation, and requires the
matching MPI-stub archive. It does not rebuild or modify LAMMPS. Old masks
must be tested against both the old library **and its old headers**; changing
only the test's constants would not reproduce the production defect.

A successful build alone is not CUDA evidence. Retain the binary and input
hashes, the device line, and all four runtime outcomes. Baseline nonzero
exits must occur after the two-atom setup and show the intended dirty-state
or stale-value failure; loader, architecture, and initialization errors are
infrastructure failures rather than successful defect reproductions.

## Audit build evidence

CPU Slurm job 2345517 built the identical regression source against the frozen
v2 headers/libraries for both architectures; job 2345538 built it against
matching v3 headers/libraries. Each build records input and binary SHA256
values under `validation/results/gpu-audit/20261023/spin-sync-{baseline,fixed}-{4090,5090}/`.
The source SHA256 is
`59089dad3a3bc55714337239fe1fc979c30079b3f03b3fba4f39cfaf97766287`.

The optional CMake target also configured, compiled, and linked on both
architectures in CPU job 2345539. An earlier attempt (2345537) exposed a
missing explicit OpenMP link dependency; `OpenMP::OpenMP_CXX` now supplies it
for the static LAMMPS archive. These are build checks, not GPU runtime passes.
The final standalone CMake configuration was revalidated on both architectures
in CPU Slurm job 2345574 after removing the separate rejected performance
experiment's build entry.

The subsequent runtime jobs used actual CUDA allocations and separate
processes for all four cases:

| Job | Device | Frozen build | Runtime outcome |
| --- | --- | --- | --- |
| 2345542 | RTX 4090, capability 8.9 | v2 baseline | All four defects reproduced |
| 2345543 | RTX 5090, capability 12.0 | v2 baseline | All four defects reproduced |
| 2345544 | RTX 4090, capability 8.9 | v3 fixed | All four cases passed |
| 2345545 | RTX 5090, capability 12.0 | v3 fixed | All four cases passed |

On each baseline device, `x-sp`, `v-fm`, and `f-fml` terminated with SIGABRT
and `Kokkos::DualView::modify ERROR: Concurrent modification of host and
device views`; `async-fm` exited 1 after observing stale magnetic-force
device values. Each failure occurred after the real two-atom setup completed.
The runner records `BASELINE_FAILURE_REPRODUCED`, not a baseline pass.
On each fixed device all four child processes exited zero. All four jobs
verified that their binary hashes were unchanged and matched their build
records. Per-case logs and summaries are retained under
`validation/results/gpu-audit/20261023/spin-sync-{baseline,fixed}-<job>/`.
