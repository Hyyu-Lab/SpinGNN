# SpinGNN LAMMPS fork

This directory is an unofficial LAMMPS fork for running NequIP/Allegro
machine-learning potentials with lattice, spin, and Born-effective-charge
(BEC) degrees of freedom.

The fork is based on LAMMPS `stable_29Sep2021_update2` (29 Sep 2021 Update 2),
upstream commit `7586adbb6a61254125992709ef2fda9134cfca6c`. It is not an official
LAMMPS release and is not maintained by the LAMMPS project.

## Added functionality

- `pair_style allegro` and `pair_style nequip` for structure-only models;
- `pair_style spin/allegro_spin` and `pair_style spin/nequip_spin` for models
  that return both lattice and spin forces;
- stochastic semi-implicit spin integrators `fix spin/sib` and
  `fix spin/sib_npt`;
- `atom_style bec`, `read_bec`, `bec_dipole_reset`, and
  `compute bec_dipole`;
- BEC coupling in `fix efield` through `coupling bec piezo no`;
- BEC tensor and accumulated-dipole fields in `compute property/atom` and
  `dump custom`.

## NequIP deployment and MPI

Both `nequip` and `spin/nequip_spin` require `newton off`. Models exported
from the current `EnergyModel` builder carry `nequip_num_interactions`, the
actual number of message-passing layers. MPI runs require this metadata and
`comm_modify mode single`. The pair style automatically expands ghost
communication to at least twice the model's receptive radius plus the
neighbor skin, while retaining `r_max` as the physical edge cutoff.

The wider halo is necessary for correct owned-atom forces in deep models and
can increase communication and neighbor-list memory. An old artifact without
the interaction count remains usable on one MPI rank or in a serial build;
multi-rank execution fails with a re-export diagnostic. Re-export from the
training configuration and checkpoint with the current `nequip-deploy` instead
of guessing a layer count in the artifact metadata.

Global total-energy offsets are counted once across all ranks, including
when rank zero owns no atoms. Non-spin NequIP adds its forces to other
`hybrid/overlay` substyles. Spin NequIP PT2 reduces its existing atom-virial
output by ownership for MPI pressure; the seven-input periodic PT2 ABI is
unchanged. The non-spin `nequip` style remains TorchScript-only.

## Supported build environments

The validated GPU build profile is:

- For CUDA 12.x and NVIDIA H100 systems, use Python 3.10.20,
  PyTorch/LibTorch 2.11.0+cu128, CUDA 12.8, and a C++17 build. Follow
  [`../docs/INSTALL_CUDA128.md`](../docs/INSTALL_CUDA128.md).

The CMake build requires:

- a C++17 compiler compatible with the supported LibTorch profile;
- MPI for MPI builds; serial builds use MPI stubs;
- LibTorch supplied by the same PyTorch installation used to deploy the model;
- the LAMMPS `SPIN` and `ASPHERE` packages.

Query the active PyTorch installation for its CMake prefix
instead of downloading a separate LibTorch copy:

```bash
python -c "import torch; print(torch.__version__); print(torch.utils.cmake_prefix_path)"
```

A typical CUDA 12.8/H100 CMake build is:

```bash
SPINGNN_TORCH_CMAKE_PREFIX="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S cmake -B build \
  -D CMAKE_CXX_STANDARD=17 \
  -D PKG_SPIN=yes \
  -D PKG_ASPHERE=yes \
  -D CMAKE_PREFIX_PATH="$SPINGNN_TORCH_CMAKE_PREFIX"
cmake --build build -j
```

Use the compiler and MPI installation available on your system. Install
PyTorch, CUDA or MPI as needed and provide model files for your system.

The maintained helper is `tools/lammps/build_lammps.sh`, invoked from the
SpinGNN repository root. For CUDA torch or `kokkos-cuda`, it probes the selected
CUDA compiler with `--version`, honoring `CUDACXX`. Missing or unusable
prerequisites fail with guidance to a dedicated CPU virtualenv prepared by
`tools/ci/install_cpu_env.sh`
or a complete CUDA 12.8 development environment. CUDA PyTorch
requires toolkit headers and libraries even in `serial` mode; finding `nvcc`
alone is insufficient. CMake configuration errors retain their original
diagnostic and exit status.

### Opt-in Kokkos CUDA

Kokkos builds provide `allegro_spin/kk` and `spin/allegro_spin/kk`, including
`/host` and `/device` variants. External Kokkos 3.7.02 built for `HOPPER90`
(`sm_90`) with CUDA 12.8 is validated on one H100: all seven host/Kokkos
parity tests cover the full 21-frame SIB trace. The tests use
`SPINGNN_DETERMINISTIC=1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

From the SpinGNN repository root inside the matching H100 build environment:

```bash
SPINGNN_REQUIRE_KOKKOS=1 SPINGNN_DETERMINISTIC=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 tools/ci/gpu_lammps_tests.sh
```

The optional tests install pinned external Kokkos when `KOKKOS_ROOT` is
unset. An existing external build must declare exactly one supported
architecture from the [installation table](../INSTALL.md#46-optional-kokkos-cuda).
H100 requires `HOPPER90`. Use external Kokkos for CUDA 12.8. The recommended
GPU inference path uses PT2. See the [CI guide](../tools/ci/README.md) for
hardware-specific verification commands.

## Pair-style syntax

The bundled `allegro`, `spin/allegro_spin`, `nequip`, and
`spin/nequip_spin` pair styles load deployed NequIP-compatible TorchScript
models. The `nequip` paths include the corrected contiguous local/ghost graph
mapping and spin-force sign, and pass serial dense and sparse-ID parity,
spin-force finite differences, a 10-step `spin/sib` run, and single-node
2-rank parity. Multi-node parity is not yet validated.

The `allegro`, `spin/allegro_spin`, and `spin/nequip_spin` styles also accept
PT2 `.pt2` packages with their adjacent `.pt2.meta` sidecars. Non-spin
`nequip` uses TorchScript. See the [PT2 deployment guide](../docs/PT2_MIGRATION.md)
for device and artifact rules.

The element names after the model path map LAMMPS atom types, in order, to the
model metadata:

```lammps
pair_style  allegro
pair_coeff  * * /path/to/model.pth Fe O
```

For a spin-aware Allegro model:

```lammps
units       metal
atom_style  spin
atom_modify map array
newton      on

read_data   system-spin.data
pair_style  spin/allegro_spin
pair_coeff  * * /path/to/spin-model.pth Fe O

# temperature, damping, random seed, and lattice motion
fix          int all spin/sib 300.0 0.1 4928459 lattice moving
timestep     0.0001
run          1000
```

`spin/allegro_spin` requires `newton on`; `spin/nequip_spin` requires
`newton off`. Spin simulations also require `units metal`, atom IDs, an atom
map, and spin attributes. The deployed model must expose the output keys used
by its selected pair style, including lattice forces and, for spin styles,
spin forces.

### Deploy metadata schema

Models built with the current `nequip-deploy` use metadata schema version 1.
For checkout deployment, run `python tools/ci/source_python.py nequip-deploy ...`
from the SpinGNN root. The `_spingnn_source.py` bootstrap selects both source
packages; CLI `_source_guard.guard_source_entry()` and root `conftest.py`
protect CLI work and pytest fixtures from mixed installed snapshots. The
shared environment still requires coordinated repair for bare imports; see
[source checkout guidance](../docs/SOURCE_CHECKOUT.md).
The capability fields are ASCII strings stored in the TorchScript extra-files
table:

| Field | Meaning |
| --- | --- |
| `schema_version` | Schema revision, currently `1` |
| `required_inputs` | Space-separated model input keys |
| `provided_outputs` | Space-separated model output keys |
| `default_dtype` | Model dtype, such as `float32` |
| `energy_units` / `length_units` | `eV` / `Angstrom` |
| `has_spin` / `has_bec` | `1` when the corresponding output capability is present |

All four neural pair styles inspect these fields during `pair_coeff`. The
current C++ implementation accepts only `float32` models. It also rejects a
model that omits an output required by the selected style, including
`spin_forces` for a spin style and `born_effective_charges` when a pair style
uses model-predicted BECs. The error names the missing key before the first
force step. Each pair style also rejects a schema-1 model whose
`required_inputs` contains a key that the style cannot provide.

`pair_style nequip` and `pair_style spin/nequip_spin` provide `cell` together
with a `(1, 3)` boolean `pbc` tensor derived from the LAMMPS x/y/z boundary
flags. This satisfies the explicit PBC contract of stress-capable models. A
non-periodic domain supplies false flags; a model that requires fully periodic
stress then rejects that input explicitly.

Models deployed before schema version 1 remain loadable. If
`provided_outputs` is absent, LAMMPS emits a backward-compatibility warning,
skips the new capability checks, and preserves the previous runtime behavior.
`nequip-deploy info` and `torch.jit.load` likewise continue to read such
models; the new fields are reported as empty strings.

For repeatable CUDA inference, opt in to PyTorch's deterministic scatter and
index algorithms before starting LAMMPS:

```bash
export SPINGNN_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
lmp -in in.system
```

The four neural pair styles apply the setting before their first model
forward. `SPINGNN_DETERMINISTIC=1` fails during `pair_coeff` unless
`CUBLAS_WORKSPACE_CONFIG` is `:4096:8` or `:16:8`; this prevents a run from
silently using nondeterministic CUDA kernels. The mode can be slower and is
therefore opt-in rather than the production default.

Cross-binary CUDA regression tests with tolerances below ordinary scatter and
index reduction drift must enable this strict mode on both binaries. Ordinary
mode is not an exact-equivalence oracle, even when both runs use the same
mode. A legacy baseline that predates this switch must be rebuilt in an
isolated validation tree with only the same deterministic setup backported;
the backport and binary hashes must be recorded with the result.

### Coupled spin-lattice NPT example

Use `fix spin/sib_npt` to integrate the spins and lattice together while a
Nose-Hoover thermostat and barostat control the lattice temperature and
pressure. The following example runs at 300 K and zero isotropic pressure:

```lammps
units       metal
boundary    p p p
atom_style  spin
atom_modify map array
newton      on

read_data   system-spin.data
mass         1 55.845
mass         2 15.999

pair_style  spin/allegro_spin
pair_coeff  * * /path/to/spin-model.pth Fe O

velocity     all create 300.0 12345 dist gaussian
timestep     0.0005

# temp: lattice Tstart, Tstop, and thermostat damping time (ps)
# alpha: transverse spin damping; seed: stochastic-spin random seed
# iso: pressure Pstart, Pstop (bar), and barostat damping time (ps)
fix          int all spin/sib_npt &
             temp 300.0 300.0 0.05 &
             alpha 0.4 seed 12345 &
             iso 0.0 0.0 0.5

thermo       100
thermo_style custom step time temp press vol pe ke etotal
run          10000
```

In `units metal`, time is measured in ps and pressure in bar. `Tstop` is also
the temperature of the stochastic spin bath. The pressure keyword can use the
usual NPT forms such as `iso`, `aniso`, or the individual cell components.
This fix performs both lattice and spin time integration, so do not apply a
second `nve`, `nvt`, `npt`, or `spin/sib` integration fix to the same atoms.

A longer BiFeO3 example with anisotropic zero-pressure control, stepwise
cooling, magnetic order parameters, and heat-capacity collection is provided
in [`examples/SPINGNN/bfo/in.bfo-sld`](examples/SPINGNN/bfo/in.bfo-sld). Its
required inputs and run command are documented in
[`examples/SPINGNN/bfo/README.md`](examples/SPINGNN/bfo/README.md).

## BEC and dipole syntax

Use `atom_style bec` for BEC-only simulations, or combine it with spin
attributes when both are required:

```lammps
units       metal
atom_style  hybrid spin bec
atom_modify map array
read_data   system.data

read_bec    system.bec
fix          field all efield 0.0 0.0 0.01 coupling bec piezo no
compute      pd all bec_dipole
thermo_style custom step pe c_pd c_pd[1] c_pd[2] c_pd[3] c_pd[4]
```

`read_bec` expects exactly one non-comment row for every atom:

```text
# id Zxx Zxy Zxz Zyx Zyy Zyz Zzx Zzy Zzz
1  1.0 0.0 0.0  0.0 1.0 0.0  0.0 0.0 1.0
2 -1.0 0.0 0.0  0.0 -1.0 0.0  0.0 0.0 -1.0
```

Atom IDs must be positive, unique, and match the simulation. The command logs
the acoustic-sum residual and resets the accumulated BEC dipole. Use
`bec_dipole_reset` to choose the current coordinates as a new reference.

The accumulated BEC dipole is the path polarization increment
`P = integral Z(R) dR` from that reference. Integrators update it with each
atom's actual unwrapped displacement; NPT updates include affine displacement
from box remapping. For a constant field, `fix efield` reports the coupling
energy as `-E.P` and applies `F = Z(R)^T E`. When a model predicts BECs that
change with configuration, this convention is path-dependent and is not a
claim that `P` is a globally conservative state function. Resetting the dipole
changes the zero of field energy but not the applied force.

Available `dump custom` and `compute property/atom` fields are
`becxx`, `becxy`, `becxz`, `becyx`, `becyy`, `becyz`, `beczx`, `beczy`,
`beczz`, `becdipolex`, `becdipoley`, and `becdipolez`.

Additional command notes and short templates are in
[`examples/SPINGNN/README.md`](examples/SPINGNN/README.md).

## Licensing and attribution

LAMMPS and this modified distribution are provided under GNU GPL version 2;
see [`LICENSE`](LICENSE). Upstream and contributor copyright notices remain in
the source files. See [`NOTICE.md`](NOTICE.md) for the upstream revision and a
summary of modifications. If you redistribute binaries, provide the
corresponding source and license as required by GPL-2.0.

## Citation

If this fork contributes to published work, cite the two SpinGNN papers listed
in the repository-root [`CITATION.bib`](../CITATION.bib):

- H. Yu et al., *Physical Review B* **110**, 104427 (2024),
  [doi:10.1103/PhysRevB.110.104427](https://doi.org/10.1103/PhysRevB.110.104427);
- H. Yu et al., *Physical Review B* **109**, 144426 (2024),
  [doi:10.1103/PhysRevB.109.144426](https://doi.org/10.1103/PhysRevB.109.144426).

Also cite upstream LAMMPS and the machine-learning packages used in the
workflow as appropriate.
