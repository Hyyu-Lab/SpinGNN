# From a trained SpinGNN model to LAMMPS MD

Run these commands from the SpinGNN repository root unless a step changes
directory. For CUDA, use the same CUDA/PyTorch environment used to build the
LAMMPS executable.
Use `tools/ci/source_python.py` for checkout commands so both Python packages
and their subprocesses use this source tree; see
[source checkout guidance](../../../docs/SOURCE_CHECKOUT.md).

## 1. Select and inspect the training session

The deployment source is the directory containing `config.yaml` and
`best_model.pth`:

```bash
export SPINGNN_TRAIN_DIR=/path/to/results/run_name
test -s "$SPINGNN_TRAIN_DIR/config.yaml"
test -s "$SPINGNN_TRAIN_DIR/best_model.pth"
mkdir -p deployed
```

Confirm the model family and species mapping in `config.yaml`. In
`pair_coeff`, list the deployed model's type names in LAMMPS atom-type
order.

## 2. Deploy the recommended PT2 package

For the PT2-capable `allegro`, `spin/allegro_spin`, and `spin/nequip_spin`
pair styles:

```bash
python tools/ci/source_python.py nequip-deploy build-pt2 --device auto \
  --train-dir "$SPINGNN_TRAIN_DIR" \
  deployed/model.pt2
test -s deployed/model.pt2
test -s deployed/model.pt2.meta
```

`auto` is the default. It selects `cuda` only when
`torch.cuda.is_available()` is true; otherwise it selects `cpu` and prints a
warning that CUDA LAMMPS needs a separately exported CUDA package. A successful
command prints the package path, sidecar path, `target_device`, learned-scaling
status, and a LAMMPS `pair_coeff` hint. For a normal rescaled training session,
require `PT2 learned scaling: included (N layer(s))` with `N >= 1`.

For a production GPU package, run this **GPU-only** command on the target CUDA
worker:

```bash
python tools/ci/source_python.py nequip-deploy build-pt2 --device cuda \
  --train-dir "$SPINGNN_TRAIN_DIR" \
  deployed/model-cuda.pt2
```

Do not move or rename a package without its adjacent `.pt2.meta` file. The
sidecar contains the package SHA-256. LAMMPS rejects a mismatched package and sidecar
before constructing the AOTI loader.

A `.pt2` package is tied to the checkpoint it was exported from. Weights and
learned `scale_by`/`shift_by` tensors are serialized into the package. Export a
new package after changing checkpoints; never reuse an old `.pt2` package with
another checkpoint. CUDA packages are also tied to the recorded compute
capability and Torch CUDA ABI, so export in the target LAMMPS environment.

PT2 graph expansion supports one or more outer `RescaleEnergyEtc` /
`RescaleOutput` layers. Learned scale and shift operations are embedded after
energy-derived force, spin-force, and virial construction and replayed from the
inner wrapper outwards. Stress follows the same scaling through
`stress = -virial / volume` where the input ABI supplies a cell. The sidecar
records `rescale_layers`; do not edit that value or remove the wrapper before
export. A different unsupported output wrapper fails explicitly instead of
being silently discarded.

### TorchScript export

TorchScript works with every PT2-capable style and is the format required by
non-spin `nequip`:

```bash
python tools/ci/source_python.py nequip-deploy build --train-dir "$SPINGNN_TRAIN_DIR" deployed/model.pth
python tools/ci/source_python.py nequip-deploy info deployed/model.pth
```

| Artifact | Use it when | Device behavior |
| --- | --- | --- |
| PT2 `.pt2` + `.pt2.meta` | Spin pairs, or optional non-spin Allegro/BEC inference | AOT-compiled for exactly `cpu` or `cuda`; mismatch fails before loading |
| TorchScript `.pth` | Non-spin NequIP or a TorchScript simulation | Loaded by LibTorch on the pair's runtime device |

For the spin pair styles, PT2 avoids TorchScript JIT execution and is the
recommended GPU path. Verify the chosen pair style and export on the target
GPU. Model size, neighbor counts, system size, and integration method affect
the total step time, so measure the complete LAMMPS step.

## 3. Build LAMMPS against the target PyTorch

For a CPU build, activate the environment from the
[CPU installation guide](../../../tools/ci/README.md#install-and-run-the-cpu-checks),
then run:

```bash
PYTHON="$VIRTUAL_ENV/bin/python" tools/lammps/build_lammps.sh serial
```

For a GPU build, use a [CUDA 12.8 development environment](../../../docs/INSTALL_CUDA128.md)
with the target PyTorch installation. The build helper probes the selected
compiler with `--version` and honors `CUDACXX`.

Kokkos CUDA is validated on one H100 with external Kokkos 3.7.02 built for
`HOPPER90`/`sm_90` and CUDA 12.8. The host/Kokkos parity tests cover the
21-frame SIB trace with `SPINGNN_DETERMINISTIC=1` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Enable these tests with
`SPINGNN_REQUIRE_KOKKOS=1` and follow the
[Kokkos build instructions](../../README.md#opt-in-kokkos-cuda).

## 4. Run one MD workflow

Run the following from the repository root using the spin PT2 package
exported in step 2. The example input expects a model with Mn and Ge types.

```bash
export LMP="$PWD/build/lammps-serial/lmp"
cd lammps/examples/spingnn
"$LMP" -in in.spin_lattice_md \
  -var model "$PWD/../../../deployed/model.pt2" \
  -var data "$PWD/data.spin8" \
  -var output spin_lattice
```

The structure and BEC commands are listed in [README.md](README.md). All inputs
use `units metal`, explicit neighbor updates, sorted trajectories, periodic
restarts, and `-var` model/data paths.

## 5. Spin data and dynamics contract

`atom_style spin` adds a magnetic direction and moment magnitude. The `Atoms`
rows used by this fork are:

```text
id type x y z sx sy sz magnitude
```

`sx sy sz` is the spin direction. A zero `magnitude` marks a nonmagnetic site;
its direction is retained as a finite placeholder but does not receive a
magnetic force.

For stochastic spin-lattice dynamics, define magnetic fixes in this order:

```lammps
fix lattice_bath all langevin 300.0 300.0 0.1 314159 zero yes
fix spin_bath all langevin/spin 300.0 0.01 481516
fix integrator all nve/spin lattice moving
```

`fix langevin/spin` must be the last magnetic fix before `fix nve/spin`.
`lattice moving` integrates atoms and spins; `lattice frozen` integrates spins
on fixed positions. Do not add another time-integration fix to the same atoms.
The standard `fix langevin` adds a lattice thermostat but is not an integrator.

The deployed model returns `spin_forces = -dE/ds` in eV. The pair writes the
LAMMPS magnetic force as

```text
fm = spin_forces / hbar
hbar = hplanck / (2*pi)
```

For `units metal`, `hplanck = 4.135667403e-3 eV ps`, so `fm` is in inverse
picoseconds. Dump `spx spy spz sp fmx fmy fmz` and check finite values, stable
spin magnitudes, and physically sensible energy drift or thermostat response.

The alternative `fix spin/sib` scheme evaluates the model twice per MD step.
It improves the semi-implicit spin update but approximately doubles pair
forward cost; measure it separately from `fix nve/spin`.

## 6. Performance guidance

Each neural pair step includes neighbor-list construction/update, graph edge
packing, tensor transfers, the model forward, and force/virial writeback.
Practical controls are:

- keep `neigh_modify every 1 delay 0 check yes` while validating moving atoms;
- choose skin distance from measured rebuild frequency, not only pair timing;
- benchmark representative atom and edge counts because tiny cells underfill a
  GPU and exaggerate launch/transfer overhead;
- avoid dumping every step in production;
- compare `fix nve/spin` with `fix spin/sib` knowing SIB performs two forwards;
- benchmark the same precision, PT2 target device, cutoff, MPI ranks, and
  neighbor settings used in production.

For the host `allegro` and `spin/allegro_spin` pair styles, set
`SPINGNN_PERF=1` to report per-call timings, transfer sizes, and memory
peaks. The default is off. CUDA host staging, enabled independently with
`SPINGNN_HOST_STAGING=1`, reuses pinned host and device buffers for transfers.

`SPINGNN_SIB_GEOM_CACHE=1` allows `spin/allegro_spin` to reuse graph geometry
between the two forwards in an eligible single-rank, frozen-lattice
`fix spin/sib` timestep. The cache checks geometry, atom ordering, and
neighbor-list generation before reuse. Each forward uses the current spins;
forces and other model outputs are recomputed. Set either optimization to
`0` or leave it unset to use the default path.

## 7. Correctness self-check

After rebuilding LAMMPS, run the serial correctness tests from the repository
root:

```bash
export SPINGNN_LAMMPS_BIN="$PWD/build/lammps-serial/lmp"
export SPINGNN_REQUIRE_LAMMPS=1
python tools/ci/source_python.py -m pytest tests_lammps -q -ra \
  -p no:cacheprovider -m "lammps and not mpi and not gpu and not slow"
```

For a new production model, reproduce the same method on a representative
structure: compare Python and LAMMPS total energy, sorted atomic forces,
spin forces or BEC tensors, and a short finite trajectory. Record the source,
model, sidecar, and LAMMPS binary hashes with the result.

## 8. Troubleshooting

| Error or symptom | Cause | Action |
| --- | --- | --- |
| `PT2 package targets cpu but this pair runs on cuda` | The package was exported on/for CPU | Export again in the target CUDA environment with `--device cuda` |
| `PT2 package targets cuda but this pair runs on cpu` | A CUDA package was passed to CPU LAMMPS | Export with `--device cpu`, or use the matching CUDA binary |
| `PT2 deploy metadata sidecar is missing` | `.pt2.meta` was omitted or renamed | Restore the adjacent sidecar generated with that exact package |
| `currently supports only float32 deployed models` | Training/deployment metadata says another dtype | Deploy a float32 model; do not edit the sidecar to disguise weights |
| `PT2 export does not support output wrapper ...` | The checkpoint has an outer transform other than supported `RescaleOutput` layers | Preserve the wrapper and use TorchScript until that exact transform has a tested PT2 representation |
| `PT2 learned scaling: not present (0 layer(s))` for a checkpoint expected to be rescaled | The selected config/checkpoint does not contain an outer runtime rescale layer | Stop and inspect the training session; do not deploy until the artifact's scaling contract is understood |
| `requires metal units` | Pair styles consume eV and Angstrom | Put `units metal` before the pair style |
| `requires newton pair on/off` | Pair-specific neighbor contract is wrong | Use `newton on` for Allegro and `newton off` for NequIP |
| Missing spin attributes or atom map | Magnetic pair used with the wrong atom layout | Use `atom_style spin`, `atom_modify map array`, and valid unique IDs |
| `Fix efield piezo yes is unavailable` | The unsupported piezo head was removed | Use `coupling bec piezo no` and inspect BEC/dipole outputs |
| Lost atoms or non-finite thermo values | Unstable starting structure, time step, or model extrapolation | Stop; reduce the time step and validate the structure/model before continuing |

TorchScript is a supported fallback, but it does not bypass units, dtype,
species, spin/BEC capability, or neighbor-style contracts.
