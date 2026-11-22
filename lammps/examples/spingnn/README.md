# SpinGNN production LAMMPS examples

This directory contains self-contained eight-atom inputs for checking a newly
deployed model before scaling to a production system. The structures are smoke
fixtures, not physically converged starting configurations.

## Model support

| Input | Pair style | Required artifact | Atom types |
| --- | --- | --- | --- |
| `in.spin_lattice_md` | `spin/allegro_spin` | PT2 `.pt2` plus `.pt2.meta` recommended; TorchScript `.pth` fallback | `Mn Ge` |
| `in.structure_md` | `nequip` | TorchScript `.pth` | `Mn Ge` |
| `in.bec_efield` | `allegro` | PT2 `.pt2` plus `.pt2.meta` optional; TorchScript `.pth` fallback | `Li Nb O` |

PT2 loading is implemented by `allegro` and the two spin pair styles. Non-spin
Allegro packages use `pos edge_index atom_types`; spin packages retain their
spin input ABIs. The non-spin `nequip` pair remains TorchScript-only. A PT2
package is device-specific: export a CUDA package in the CUDA environment used
to build and run the GPU LAMMPS binary, and a CPU package for a CPU binary.

## Deploy models

PT2 is the recommended GPU path for a trained spin model. `--device auto` is
the default and selects CUDA only when PyTorch reports a usable CUDA device:

```bash
mkdir -p deployed
nequip-deploy build-pt2 --device auto \
  --train-dir /path/to/spin-training-session \
  deployed/spin.pt2
test -s deployed/spin.pt2
test -s deployed/spin.pt2.meta
```

PT2 supports normal training sessions wrapped by `RescaleEnergyEtc`; learned
scale and shift tensors are serialized into the package. A `.pt2` package is
tied to the checkpoint it was exported from, and its sidecar binds the package
SHA-256 plus device contract. Export again for a different checkpoint or CUDA
architecture. The included PT2 smoke can be generated from the current test
config:

```bash
nequip-deploy build-pt2 --device auto \
  --model ../../../tests_lammps/fixtures/configs/nosoc.yaml \
  deployed/spin.pt2
```

Build a BEC PT2 package and the TorchScript rollback artifacts:

```bash
nequip-deploy build-pt2 --device auto \
  --train-dir /path/to/bec-training-session \
  deployed/bec.pt2
nequip-deploy build --train-dir /path/to/structure-training-session \
  deployed/structure.pth
nequip-deploy build --train-dir /path/to/bec-training-session \
  deployed/bec.pth
```

Keep every `.pt2.meta` sidecar beside its package. Element names after the
model path in `pair_coeff` must match the deployed `type_names` in LAMMPS atom
type order.

## Run the examples

Set `LMP` to a freshly built executable linked against the same PyTorch family
as the deployed artifacts, then run from this directory.

### Coupled spin-lattice MD

```bash
"$LMP" -in in.spin_lattice_md \
  -var model "$PWD/deployed/spin.pt2" \
  -var data "$PWD/data.spin8" \
  -var output spin_lattice
```

This input initializes lattice velocities, applies separate lattice and spin
Langevin thermostats, and uses `fix nve/spin lattice moving` as the only
integrator. Override the run controls with, for example,
`-var temperature 600 -var steps 10000`.

### Structure-only NVT MD

```bash
"$LMP" -in in.structure_md \
  -var model "$PWD/deployed/structure.pth" \
  -var data "$PWD/data.structure8" \
  -var output structure_md
```

The NequIP pair requires `newton off`. The input uses a 0.5 fs time step and a
Nosé-Hoover thermostat. Re-equilibrate and validate the time step for the
target chemistry before extending the run.

### BEC response to an electric field

```bash
"$LMP" -in in.bec_efield \
  -var model "$PWD/deployed/bec.pth" \
  -var data "$PWD/data.bec8" \
  -var bec "$PWD/bec8.dat" \
  -var output bec_efield
```

The initial BEC file is zero so its acoustic-sum residual is exactly zero.
After the first model evaluation, the pair supplies predicted BEC tensors.
Override the field with `-var field_z 0.002`; in `units metal` it is measured
in volts per Angstrom. Piezoelectric coupling is intentionally unavailable.

## Expected outputs

Each input writes:

- `<output>.lammpstrj`, sorted by atom ID;
- thermodynamic energy, temperature, pressure, and maximum force on stdout;
- `<output>.restart.final`, plus interval restarts for sufficiently long runs.

The spin dump includes direction, magnitude, and magnetic force. The BEC dump
includes all nine tensor components and the accumulated dipole. Require finite
thermo values, no lost atoms, and the expected number of dump frames before
promoting a model or time step.

See [TUTORIAL_LAMMPS.md](TUTORIAL_LAMMPS.md) for deployment, device contracts,
spin units, performance guidance, parity validation, and troubleshooting.
