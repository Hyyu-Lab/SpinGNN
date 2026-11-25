# SpinGNN2 LAMMPS overlay

The external checkout is pinned to the annotated upstream tag
`stable_30Sep2026_update1`. The apply tool checks the tag object and peeled
commit, applies the core patches to a separate Git index, and installs the
new sources under their LAMMPS package paths. Keep the upstream clone and
build products outside the tracked tree.

```sh
gh repo clone lammps/lammps validation/lammps/upstream -- --depth 1 --branch stable_30Sep2026_update1
python lammps_spingnn2/apply.py --upstream validation/lammps/upstream --destination validation/lammps/source
python lammps_spingnn2/apply.py --check --destination validation/lammps/source
python tools/spingnn2_build.py --target cpu
python tools/spingnn2_build.py --target all-gpu   # sm70,sm80,sm86,sm89,sm90,sm120 fat binary (or single sm*)
```

The build frontend submits to Slurm when called from the login node. Within
an allocation it performs the build directly. Use the cluster's SpinGNN
Python environment for wheel construction. Each build manifest records the
upstream and overlay identities, source hashes, compiler versions and product
hashes. Each install directory contains the executable, native library, and
Python wheel. The native and LAMMPS Kokkos runtimes communicate through the
plain C ABI. The native library hides its private C++ symbols.

Keep `bin/` and its sibling `lib64/` (or `lib/` on platforms using that
layout) together when moving an install. The Linux executable resolves the
native library relative to its own directory. The ELF search path also
records the cluster GCC, MPI and, on CUDA builds, CUDA runtime directories.
Use `env -i PATH=/usr/bin:/bin /path/to/install/bin/lmp_spingnn2 -h`
on a matching execution host to check the installation.

## Pair commands

```lammps
units metal
atom_style spin
# Create atoms and define their masses before setting the coefficients.
pair_style spingnn2 precision float32 math ieee
pair_coeff * * model.sp2.json prepared-artifact.json Fe O
```

Each LAMMPS type requires one exported species name or zero-based model
species index. The pair checks the prepared artifact at load time. Use
`spingnn2-prepare-jit-artifact` before launching LAMMPS. Each MPI process
needs read access to the contract, binary weights, prepared artifact and
its manifest. Use a CUDA artifact compiled for the assigned device.
Use the UUID-binding MPI launcher (`tools/lammps/spingnn2_gpu_rank.py`) with
`-k on g 1 -sf kk -pk kokkos gpu/aware on neigh half` for multiple GPUs on
one node with one MPI rank per device. For host Kokkos use a CPU build
with `-k on t T -sf kk`.

The default is strict FP32. Use `precision float64` with a matching artifact
or `math tf32` to opt into CUDA FP32 tensor-float math. `tile N` sets the
native tile width. `cache no` disables lattice caching during SIB runs.
`cuda_graph yes` enables CUDA Graph capture in `spingnn2/kk` after a
warm evaluation with stable inputs. Neighbor rebuilds, changed device
addresses and changed inference options require a fresh capture. The
default is `cuda_graph no`. Synchronized stage profiling uses ordinary
launches.
The CUDA pair compacts the LAMMPS skin list to physical neighbors before
neural-network evaluation. Lattice motion can change this set between list
rebuilds. Python direct inference uses precomputed physical edges. On one
RTX 5090, 4,096-atom NoSOC NVE with 300 K initial velocities and a 1 fs
timestep ran at 43.43 steps/s (3.753 ns/day) with 0.4 Å skin and 43.77
steps/s (3.781 ns/day) with 1.5 Å skin. The timed loops averaged 323 K
and rebuilt the neighbor list eight and two times, respectively.
[Thermal MD measurements](../docs/PERFORMANCE.md#2-thermal-lammps-md-with-gpu-physical-edge-compaction)
DREAM uses `atom_style bec` and writes the per-atom Born tensor. The
evidential families expose `f_sigma`, `f_Uepi`, `f_nu`, `f_kappa` or
their `sf_` equivalents, according to the exported heads. Standard LAMMPS
`fix pair` reads these fields on demand.

```lammps
pair_style spingnn2 precision float64 profile yes
pair_coeff * * model.sp2.json prepared-artifact.json 0 1
compute timers all spingnn2/timing
thermo_style custom step pe c_timers[*]
```

Timing columns are accumulated since pair initialization. They report pair
wall time excluding its measured halo communication, pair halo and SIB
midpoint communication time, Stage-1 and Stage-2 seconds, cache hits, and
evaluations. The global compute returns the maximum across ranks for each
column. Enable `profile yes` for stage timing. Timings exclude LAMMPS's
outer Verlet communication and include the overhead of profiling when enabled.

`compute ID all spingnn2/timing graph` reports successful captures, graph
replays, and fallback evaluations, in that order. Each entry is the
maximum rank-local count.

## Spin integration

```lammps
fix integrate all spin/sib 0.0 0.0 91081 lattice frozen
# For coupled spin and lattice motion, use lattice moving.
# For thermostatted, barostatted dynamics, select this fix instead:
fix integrate all spin/sib/npt temp 20 20 0.02 iso 1 1 0.05 alpha 0.05 seed 91081
```

The SIB arguments are spin-bath temperature, Gilbert damping, and a
positive random seed. `spin/sib/npt` uses the Nose–Hoover target ending
temperature for the spin bath, as in the legacy style. The equivalent Kokkos
fixes are `spin/sib/kk` and `spin/sib/npt/kk`. They are selected by `-sf kk`
and require `spingnn2/kk` on the same execution space. The `spin/sib_npt`
and `spin/sib_npt/kk` spellings also remain available. For the native pair,
the spin noise is keyed by atom ID and timestep for consistent stochastic
trajectories across MPI decomposition and device backends.

An isolated `spin/sib` fix can reuse its preceding lattice evaluation at the
midpoint. Global `lattice frozen` runs can also reuse lattice work across
timesteps while their ghost layout stays valid. If another fix can move atoms
or remap the box (for example, `recenter/kk`), the pair refreshes the geometry
even when the skin list remains valid. `spin/sib/npt` evaluates after the
first Nose–Hoover box remap, using that step's updated geometry.

For 1-evaluation-per-step spin–lattice dynamics, `fix spin/cfi` (and
`spin/cfi/kk`, aliases `spin/lpsib` / `spin/lpsib/kk`) implements the
Cached-Field Integrator with Kokkos CPU/CUDA and multi-MPI exchange support.
For unbiased spin temperature estimation on nonlinear or anisotropic ML
potentials, `compute spin/gpst` (and `spin/gpst/kk`) evaluates 7 sphere-IBP
Geometric-Probe Spin Temperature `(N_c, D_c)` pairs over magnetic atoms
(`sp[i][3] > 0`).

## DREAM fields

`atom_style bec` carries a row-major 3-by-3 tensor and an integrated
three-component dipole per atom across migration and restart. The
dipole increment is the tensor times the unwrapped displacement.
`fix efield Ex Ey Ez coupling bec` contributes the transposed tensor
times the applied field to the atomic force. `compute ID group bec`
returns the three dipole components and the acoustic sum rule residual.
`read_bec` imports reference tensors by atom ID, and `bec_dipole_reset`
resets the dipole reference.
