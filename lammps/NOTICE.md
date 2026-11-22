# Notices and modification statement

This is a modified distribution of the LAMMPS molecular dynamics code. It is
not an official release of LAMMPS.

## Upstream source

- Project: LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator)
- Upstream version: 29 Sep 2021 Update 2 (`stable_29Sep2021_update2`)
- Upstream commit: `7586adbb6a61254125992709ef2fda9134cfca6c`
- Upstream project: <https://www.lammps.org/>
- License: GNU General Public License version 2, reproduced in `LICENSE`

Copyright and authorship statements in the original files remain in effect.
In particular, this distribution preserves the Sandia Corporation notices and
the contributor credits already present in the affected LAMMPS and NequIP /
Allegro pair-style sources.

## Modifications

The modified source was prepared for public distribution on 21 August 2026.
Compared with the upstream commit above, this fork adds or changes code for:

- TorchScript NequIP and Allegro pair styles, plus PT2/AOTInductor support
  for Allegro and the spin pair styles;
- spin-aware NequIP and Allegro pair styles;
- semi-implicit spin integration, including an NPT variant;
- Born effective charge storage, input, dipole accumulation, output, and
  electric-field coupling; the former piezoelectric mode is explicitly
  rejected;
- the CMake/LibTorch linkage and the server-specific MPI make configuration
  needed by the original deployment.

The 22 October 2026 correctness update additionally records NequIP model
interaction depth in deployment metadata, sizes MPI communication for that
depth, preserves global energy offsets and additive hybrid forces, and fixes
owned-atom virial accounting. Periodic ghost representatives retain complete
model neighborhoods, and empty ranks participate in collective error checks.
Periodic-shift packing uses the native cell inverse to avoid per-edge Torch
allocations.

The 23 October 2026 GPU correctness update negotiates CUDA discovery across
MPI ranks and terminates rank-local deployment failures without entering a
collective error barrier. It adds native RTX 4090/5090 profiles using external
Kokkos 4.7.04, adapts the initialization and reducer APIs, separates spin
synchronization masks from position/velocity/force masks, and corrects spin
force clearing and asynchronous synchronization.

The complete set of changes is the difference between this source tree and
the upstream commit identified above. This notice is intended to make the
modified status and modification date explicit while preserving all upstream
license and attribution notices.
