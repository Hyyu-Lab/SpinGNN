---
name: spingnn-usage
description: Prepare magnetic or Born-effective-charge data, train and deploy SpinGNN structure, NoSOC, SOC, or BEC potentials, and run their LAMMPS molecular or spin-lattice dynamics workflows. Use when a researcher or AI agent operates SpinGNN data, training, deployment, parity validation, or LAMMPS MD.
---

# SpinGNN Usage

Operate from the SpinGNN repository root. Read [INSTALL.md](../../INSTALL.md)
before creating an environment or building LAMMPS. Replace all example data,
model, species, split, and output paths: datasets and production models are not
distributed with this repository.

## Preserve the workflow contracts

- Use `units metal` for `pair_style allegro` and
  `spin/allegro_spin`; these paths interpret model values as eV, Angstrom, and
  eV/Angstrom. Use `newton on`.
- Treat Allegro and `spin/allegro_spin` as the primary LAMMPS paths. Mention
  `pair_nequip*` only as a separately revalidated path and consult
  [the current LAMMPS status](../../lammps/README.md) before use.
- Keep per-atom BEC data flattened as `born_effective_charges` with shape
  `(N,9)` throughout ingestion, training, deployment, and parity checks.
- Do not restore a piezo output contract. A BEC model outputs energy, forces,
  stress, BEC, and total BEC. An extxyz `piezoelectric` field is data only;
  `fix efield ... piezo yes` must fail explicitly.
- Do not describe single-node results as multi-node validation. Serial and two
  ranks on one H100 node are validated; multi-node operation is not.
- Confirm `nequip.__file__` and `allegro.__file__` point to the intended
  worktree when source-exact behavior matters.

## 1. Prepare data

### Magnetic OUTCAR and OSZICAR data

Use `nequip-data -m` to search a VASP result tree and write ASE-readable
extxyz. The parser pairs each `OUTCAR` with the corresponding filename formed
by replacing `OUTCAR` with `OSZICAR`.

```bash
nequip-data -m \
  --path /shared/path/to/vasp-results \
  --out-path data/magnetic.extxyz \
  --nomagz 8,83 \
  --min-mag-norm 0.5 \
  --cpu 8 \
  --vasp6
```

Use `--vasp6` only for VASP 6 and later. For VASP 5, omit it because the
parser preserves the historical factor-of-two convention for
`lambda*MW_perp`. Use `--nomagz` to set known nonmagnetic atomic numbers to
zero; otherwise the magnitude threshold determines which sites are zeroed.

Required magnetic fields are:

- `initial_magmoms`: per-atom `(N,3)` model input. The data layer retains the
  magnitude separately and round-trips the full magnetic moment.
- `spin_forces_vert`: per-atom `(N,3)` supervision produced from the OSZICAR
  constrained-moment blocks.

The OSZICAR `lambda*MW_perp` quantity contains only the component perpendicular
to the spin direction. It is therefore the target `spin_forces_vert`, not a
complete `spin_forces = -dE/ds` vector. Never copy or rename this label to
`spin_forces`. If the OSZICAR lacks complete `lambda*MW_perp` and `MW_int`
blocks, the current parser warns and keeps the frame without a spin-force
label; remove that key from both losses and metrics unless every supervised
frame provides it.

The implementation and CLI contract are in
[`src/core/nequip/scripts/vasp_data.py`](../../src/core/nequip/scripts/vasp_data.py)
and
[`src/core/nequip/utils/data_outcar_spin.py`](../../src/core/nequip/utils/data_outcar_spin.py).

### Born-effective-charge data

When an existing extxyz dataset and indexed VASP outputs are aligned, add BEC
labels with:

```bash
nequip-data-mlff-bec \
  --dataset-path data/structure.extxyz \
  --folder /shared/path/to/indexed-vasp-results \
  --out-path data/bec.extxyz \
  --criteria 1.0e-2 \
  --cpu 8
```

The folder must contain `OUTCAR.<index>` and `log.out.<index>` matching the
frame order in the input dataset. The command rejects mismatched species,
cell, PBC, or positions, applying minimum-image alignment for periodic axes.
It stores each atomic tensor as nine flattened
`born_effective_charges` values. The parser may preserve a VASP
`piezoelectric` tensor as an extxyz/ASE field, but the model does not learn or
emit it.

See the implementation at
[`src/core/nequip/scripts/vasp_mlff_bec.py`](../../src/core/nequip/scripts/vasp_mlff_bec.py)
and the public invocation in
[`example/README.md`](../../example/README.md#extxyz-fields).

### Inspect schema before training

Check frame count, finite values, and shapes before editing a training config:

```bash
python - <<'PY'
import numpy as np
from ase.io import read

path = "data/magnetic.extxyz"  # replace with data/bec.extxyz as needed
frames = read(path, index=":")
assert frames, "empty dataset"

for index in (0, len(frames) - 1):
    atoms = frames[index]
    assert np.isfinite(atoms.positions).all()
    if "initial_magmoms" in atoms.arrays:
        assert atoms.arrays["initial_magmoms"].shape == (len(atoms), 3)
    if "spin_forces_vert" in atoms.arrays:
        assert atoms.arrays["spin_forces_vert"].shape == (len(atoms), 3)
    if "born_effective_charges" in atoms.arrays:
        assert atoms.arrays["born_effective_charges"].shape == (len(atoms), 9)
        assert np.isfinite(atoms.arrays["born_effective_charges"]).all()

print("DATA_SCHEMA_OK", path, "frames", len(frames))
PY
```

Validate the bundled example configs from the checkout:

```bash
python tools/ci/source_python.py example/validate_configs.py
```

Use the schema check above for each new dataset before adapting a config and
running a small training smoke test.

## 2. Train a potential

Copy one public config to a new file. Never edit a production run in place.
At minimum set `dataset_file_name`, `chemical_symbol_to_type`, `n_train`,
`n_val`, `r_max`, losses, metrics, `root`, and `run_name`.

The modern PyTorch 2.11 profile is accelerated by default. NoSOC and SOC
configs use `spingnn_compile.enabled: true`, `scope: full_fx`, dynamic shapes,
`float32_matmul_precision: high`, and `allow_tf32: true`; bf16 autocast remains
disabled. H100 reference results were 3.36x for NequIP-spin and 23.4x for
Allegro, with synchronized loss-curve relative errors below 1e-6. Read the
[`training tutorial`](../../example/TUTORIAL_TRAINING.md) before changing these
controls.

### Structure-only potential

Start from
[`example/01_structure_only.yaml`](../../example/01_structure_only.yaml).
It uses `allegro.model.Allegro` with energy, force, and stress supervision.

```bash
cp example/01_structure_only.yaml run/structure-smoke.yaml
nequip-train run/structure-smoke.yaml
```

Remove a loss and its metric when that label is absent. Do not manufacture
force or stress labels.

### Magnetic potential without SOC

Start from
[`example/02_spin_nosoc.yaml`](../../example/02_spin_nosoc.yaml).
It must use the dedicated `allegro.model.Allegro_nosoc` builder.

```bash
cp example/02_spin_nosoc.yaml run/nosoc-smoke.yaml
nequip-train run/nosoc-smoke.yaml
```

Do not substitute `Allegro_soc` with `lsoc_layer: false`. The SOC builder can
still contain a full exchange matrix and magnetic-anisotropy terms, whereas
`Allegro_nosoc` enforces spin-rotation-invariant interactions. Require
`initial_magmoms`; supervise `spin_forces_vert` only when that label is present.

### Magnetic potential with SOC

Start from
[`example/03_spin_soc.yaml`](../../example/03_spin_soc.yaml). It uses
`allegro.model.Allegro_soc` and exposes the SOC, exchange-matrix, anisotropy,
and biquadratic controls shown in that file.

```bash
cp example/03_spin_soc.yaml run/soc-smoke.yaml
nequip-train run/soc-smoke.yaml
```

Keep the `initial_magmoms` input and the same `spin_forces_vert` supervision
contract as the parser. Do not infer full spin-force supervision from the
OSZICAR perpendicular component.

For a formal SOC run, start from
[`example/05_spin_soc_production.yaml`](../../example/05_spin_soc_production.yaml)
and replace every data/species/split placeholder before submission. Its batch
size, loss weights, patience, and wall limit are starting values, not universal
hyperparameters.

### Born-effective-charge potential

Start from
[`example/04_born_effective_charge.yaml`](../../example/04_born_effective_charge.yaml).
It uses `allegro.model.Allegro_born` and the energy/force/stress/BEC/total-BEC
output chain.

```bash
cp example/04_born_effective_charge.yaml run/bec-smoke.yaml
nequip-train run/bec-smoke.yaml
```

For BEC training:

- keep `l_max >= 2`, which is required for the tensor outputs;
- keep each target at `(N,9)` rather than reshaping it to `(N,3,3)`;
- set `born_scale_list` to exactly one value per chemical type, ordered by the
  integer type index in `chemical_symbol_to_type`;
- include `born_effective_charges` and
  `total_born_effective_charges` losses; the latter enforces the acoustic sum
  rule through an internally generated zero target;
- do not add a `piezoelectric` loss or output expectation.

### Important tuning controls

- `spingnn_compile` defaults to `enabled: true`, `scope: full_fx`, and dynamic
  shapes for `ParaStressForceSpinOutput`. `full_fx` compiles the complete
  derivative graph without the eager gradient bridge. Structure-only and BEC
  use `ParaStressForceOutput`, so their examples explicitly disable compilation
  and make no compile-speed claim.
- `float32_matmul_precision: high` plus `allow_tf32: true` is the calibrated H100
  default. Measured force relative error was 2.93e-3 to 3.92e-3 (8e-3 gate),
  and energy error was 2.227e-4 to 2.264e-4 eV/atom (5e-4 gate). For strict
  publication parity, use `highest` plus `false` and rerun the numeric gates.
- `train_autocast.enabled` must remain false for force, stress, or spin-force
  supervision. H100 bf16 force relative error was 7.755e-2, which fails the
  1e-2 production gate.
- Verify every public config without user data using
  `python example/validate_configs.py`. For compile correctness, run both
  source-exact model suites with `SPINGNN_TEST_COMPILE=1`; a valid full-FX
  runtime reports one target, `full_fx` scope, populated train/eval caches, and
  zero eager gradient-bridge calls through `get_compile_stats(model)`.
- To recover an eager fp32 baseline, set `spingnn_compile.enabled: false`,
  `float32_matmul_precision: highest`, and `allow_tf32: false`. Keep autocast
  disabled in both accelerated and eager comparisons.
- `r_max` controls the neighbor cutoff and must match the deployed metadata
  consumed by LAMMPS. Select it from convergence and cost studies for the
  target chemistry.
- `edge_prune` accepts only `nonzero` (the default historical dynamic pruning)
  or `dense` (full edge shapes with cutoff-pruned contributions multiplied to
  exact zero). Explicit dense mode can amplify computation when input edges
  exceed a layer cutoff; it does not change the physical cutoff.
- `loss_coeffs` and `metrics_components` must use only labels present in the
  selected dataset. Keep `spin_forces_vert` distinct from `spin_forces`.
- `l_max` controls angular resolution and compute cost; BEC requires at least
  2.
- `chemical_symbol_to_type` defines the species order that must remain
  consistent with `born_scale_list`, deployment metadata, and LAMMPS
  `pair_coeff` element names.

### Promote smoke to formal training

Before increasing the workload, require a smoke run that builds the model,
completes several epochs, writes a nonempty `best_model.pth`, and can be
deployed. Then review this checklist:

- freeze a new config and output directory; do not append to the smoke run;
- increase `n_train` and `n_val` only within the verified dataset size and
  reserve an untouched held-out set;
- increase `max_epochs` and tune early-stopping patience deliberately;
- choose `batch_size` and `dataloader_num_workers` from measured GPU memory
  and I/O behavior;
- confirm every supervised key is finite across the selected split;
- record seeds, source commit, environment versions, dataset identity, and
  exact config;
- monitor train/validation components, not only aggregate loss;
- evaluate the held-out set and perform deployment/GPU-forward/parity gates
  before calling the model production-ready.

## 3. Deploy and inspect the model

Resolve the training directory from the config's `root/run_name`, then build
the recommended PT2 artifact for the current machine:

```bash
export SPINGNN_TRAIN_DIR=results/spin_nosoc/example
test -s "$SPINGNN_TRAIN_DIR/config.yaml"
test -s "$SPINGNN_TRAIN_DIR/best_model.pth"

mkdir -p deployed
nequip-deploy build-pt2 --device auto \
  --train-dir "$SPINGNN_TRAIN_DIR" \
  deployed/model.pt2
test -s deployed/model.pt2
test -s deployed/model.pt2.meta
```

`auto` selects CUDA only when `torch.cuda.is_available()` is true and otherwise
selects CPU. Read the printed `target_device`; a CPU package is rejected by a
CUDA pair and vice versa. Build production CUDA packages in the same CUDA and
PyTorch environment used by LAMMPS, preferably with explicit `--device cuda`.
Never separate a `.pt2` package from its `.pt2.meta` sidecar.

Deployment metadata records `edge_prune`, and loaders reject a value that does
not match the embedded config. Deployment automatically chooses dense only
when the model uses the strict `spin/allegro_spin` input ABI, every Allegro
core has one identical cutoff across all layers and cores, and the config did
not explicitly select a mode. Non-spin/BEC models, decreasing per-layer
cutoffs, missing strict-prefilter evidence, and explicit `edge_prune: nonzero`
do not take the automatic fast path. Never edit this metadata field by hand.

A `.pt2` package is tied to the checkpoint it was exported from: weights and
learned rescale tensors are serialized into the package. Export a new package
for a different checkpoint and never pair an old package with new checkpoint
metadata. LAMMPS verifies the sidecar's package SHA-256 before loading. CUDA
deployment additionally requires the package's compute capability and Torch
CUDA ABI to match the runtime environment.

Normal `RescaleEnergyEtc` checkpoints are supported. For such a session,
require the CLI to print `PT2 learned scaling: included (N layer(s))` and verify
the sidecar's `rescale_layers` value. The exporter replays runtime scale/shift
after derivative outputs, from inner to outer wrapper; construction-time
`related_*_keys` effects already reside in the loaded checkpoint. Never remove
a learned rescale wrapper to force export. A different unsupported wrapper must
fail explicitly and should use TorchScript until its transform is represented.

PT2 loading is implemented by `allegro`, `spin/allegro_spin`, and
`spin/nequip_spin`. Non-spin Allegro/BEC uses the ordered
`pos edge_index atom_types` ABI. Non-spin `nequip` remains TorchScript-only;
for it, and as the required rollback baseline for every PT2 style, build
TorchScript:

```bash
nequip-deploy build \
  --train-dir "$SPINGNN_TRAIN_DIR" \
  deployed/model.pth
nequip-deploy info deployed/model.pth
```

Load TorchScript through the canonical deployment API so metadata global
options, including TF32 and fusion settings, are applied consistently:

```bash
python - <<'PY'
import torch
from nequip.scripts.deploy import load_deployed_model

model, metadata = load_deployed_model(
    "deployed/model.pth", device="cpu", freeze=False, set_global_options=True
)
assert isinstance(model, torch.jit.ScriptModule)
assert float(metadata["r_max"]) > 0
assert metadata["type_names"].strip()
print("DEPLOY_OK", metadata["r_max"], metadata["type_names"])
PY
```

Run a GPU forward on a held-out frame and check that the expected output keys
are present and all their values are finite before deploying the model.

## 4. Run LAMMPS

Build and verify LAMMPS as documented in [INSTALL.md](../../INSTALL.md#4-build-the-lammps-fork).
Every element after the model path in `pair_coeff` maps one LAMMPS atom type,
in order, to the deployed `type_names` metadata.

Start from the runnable production examples and tutorial rather than writing an
input from memory:

- [`lammps/examples/spingnn/README.md`](../../lammps/examples/spingnn/README.md)
- [`lammps/examples/spingnn/TUTORIAL_LAMMPS.md`](../../lammps/examples/spingnn/TUTORIAL_LAMMPS.md)

PT2 is the recommended GPU path for spin pair styles and an optional backend
for non-spin Allegro/BEC. The included BEC example accepts PT2 or TorchScript;
the structure-only non-spin NequIP example remains TorchScript. Do not reuse
spin-pair H100 evidence as non-spin PT2 hardware acceptance.

### Observation-only inference records

For a dedicated benchmark run, set `SPINGNN_PERF=1` on LAMMPS. The default
unset/`0` path does not construct timers or print records. Each pair call writes
one producer-owned stderr line beginning with:

```text
SPINGNN_PERF schema=1 rank=0 step=1 call=1 pair=spin/allegro_spin ...
```

The counters are observational: they must not add a model forward, alter model
inputs/outputs, or add a CUDA synchronization point beyond the selected copy
path's output-consumption boundary. `allegro` records the BEC payload when
`atom_style bec` is active; both `allegro` and `spin/allegro_spin` record their
actual TorchScript or PT2 backend. Values other than `SPINGNN_PERF=0` or `1`
fail closed during `pair_coeff`.

For CUDA host pair styles, `SPINGNN_HOST_STAGING=1` enables persistent pinned
host/device capacity, nonblocking copies on the current LibTorch stream, and a
single consolidated output D2H plus stream wait. An unset
`SPINGNN_HOST_STAGING` or a value of `0` leaves staging disabled. Invalid
values fail during `pair_coeff`. CPU devices and Kokkos pair styles leave
staging disabled.
Staging PERF `cache_hit/cache_miss` records only whether backing storage and
the public LAMMPS generation key were reusable. Every compute still repacks and
uploads positions, types, edges, and spin; cross-compute frozen-SIB geometry
reuse is not part of this switch.

For `pair_style spin/allegro_spin`, `SPINGNN_SIB_GEOM_CACHE=1` separately
enables geometry reuse between the midpoint and regular Verlet pair calls of
`fix spin/sib ... lattice frozen` (including the `lattice no` alias). The pair
accepts the fix hint only as permission to try: it independently checks the
timestep, neighbor generation, atom counts and order, exact coordinates, box,
edge count, cutoff, and type mapping. Moving SIB, `spin/sib_npt`, MPI, hybrid
pair styles, and rebuild/reorder mismatches have zero hits. Reuse never skips
the second model forward; it uploads the new spin and runs the model again,
without caching any output.

Use `SPINGNN_SIB_GEOM_CACHE=0` or leave it unset for full graph construction
and geometry upload on every call. Invalid values fail during `pair_coeff`.
The geometry and host-staging switches are independent and support every
`0/1` combination. PERF fields `sib_geom_cache_hit`, `sib_geom_cache_miss`,
and `sib_geom_h2d_saved` report geometry reuse. On a geometry hit,
`sib_geom_h2d_saved` must equal `20*N_all + 16*E`; otherwise it is zero.

### Structure-only Allegro

```lammps
units       metal
boundary    p p p
atom_style  atomic
atom_modify map array
newton      on
read_data   /shared/path/system.data

pair_style  allegro
pair_coeff  * * /shared/path/structure-model.pth Fe O

thermo_style custom step pe press fmax
run          0
```

### NoSOC or SOC magnetic model

The two magnetic builders share the deployed `spin/allegro_spin` interface:

```lammps
units       metal
boundary    p p p
atom_style  spin
atom_modify map array
newton      on
read_data   /shared/path/system-spin.data

pair_style  spin/allegro_spin
pair_coeff  * * /shared/path/spin-model.pt2 Fe O

velocity     all create 300.0 19081 mom yes rot no dist gaussian
fix          int all spin/sib 300.0 0.1 91081 lattice moving
timestep     0.0001
thermo       1
thermo_style custom step temp pe ke etotal fmax
run          10
```

Use `lattice frozen` for spin-only integration. `fix spin/sib_npt` integrates
both lattice and spins while controlling temperature and pressure; follow the
validated syntax in
[`lammps/examples/SPINGNN/README.md`](../../lammps/examples/SPINGNN/README.md#coupled-spin-lattice-npt-with-sib-integration).
Do not attach a second time-integration fix to the same atoms.

### BEC model and electric field

```lammps
units       metal
boundary    p p p
atom_style  bec
atom_modify map array
newton      on
read_data   /shared/path/system-bec.data
read_bec    /shared/path/system.bec

pair_style  allegro
pair_coeff  * * /shared/path/bec-model.pth Li Nb O

fix          field all efield 0.0 0.0 0.01 coupling bec piezo no
compute      dip all bec_dipole
compute      becprops all property/atom becxx becxy becxz becyx becyy becyz &
             beczx beczy beczz becdipolex becdipoley becdipolez
thermo_style custom step pe fmax c_dip c_dip[1] c_dip[2] c_dip[3] c_dip[4]
run          0
```

The BEC file has exactly one non-comment row per atom:

```text
# id Zxx Zxy Zxz Zyx Zyy Zyz Zzx Zzy Zzz
1  1.0 0.0 0.0  0.0 1.0 0.0  0.0 0.0 1.0
```

Atom IDs must be positive, unique, and match the simulation. `read_bec` resets
the accumulated reference; `bec_dipole_reset` can set a new reference after
equilibration. The accumulated dipole is the path-dependent increment
`integral Z(R) dR`. With a field, LAMMPS applies `F = Z(R)^T E` and reports
coupling energy `-E.P`.

### LAMMPS error map

| Error or symptom | Contract and action |
| --- | --- |
| `PT2 package targets cpu but this pair runs on cuda` (or the reverse) | Re-export in the target environment with `nequip-deploy build-pt2 --device cuda` or `--device cpu`; never edit the sidecar. |
| `PT2 deploy metadata sidecar is missing` | Restore the exact adjacent `.pt2.meta` produced with the package. |
| `currently supports only float32 deployed models` | Deploy float32 weights; changing only metadata does not convert a model. |
| `PT2 export does not support output wrapper ...` | Preserve the unsupported transform and use TorchScript until that exact wrapper has a tested PT2 representation. |
| Expected rescaled checkpoint prints `PT2 learned scaling: not present (0 layer(s))` | Stop and inspect the selected training session; do not bypass or invent scaling metadata. |
| `Pair style Allegro requires metal units...` | Put `units metal` before `pair_style`; do not bypass the eV/Angstrom gate. |
| Allegro reports that Newton pair must be on | Use `newton on` for `allegro` and `spin/allegro_spin`. |
| A spin pair reports missing spin attributes, map, IDs, or integrator | Use `atom_style spin`, `atom_modify map array`, valid IDs, and one `spin/sib` or `spin/sib_npt` fix. |
| Model species mapping fails | Make `pair_coeff` names match all LAMMPS types and deployed `type_names`, in order. |
| `Fix efield piezo yes is unavailable...` | The piezo head is removed. Use `coupling bec piezo no` and BEC/dipole outputs. |
| `read_bec` rejects rows or IDs | Supply exactly one finite nine-component row for every unique positive atom ID. |
| MPI starts with UCX endpoint errors | For the validated single-node path, use `ob1` plus `self,tcp`; do not claim this validates multi-node or RDMA performance. |

## 5. Validate parity and troubleshoot

### Python-to-LAMMPS parity

For the same structure, deployed model, element order, dtype/global options,
and boundary conditions:

1. Load the deployed model with
   `load_deployed_model(..., set_global_options=True)` and save total energy,
   forces sorted by atom ID, and model-specific outputs.
2. Generate a LAMMPS data file from the identical cell and coordinates. Rotate
   BEC tensors consistently if converting to a restricted triclinic cell.
3. Run `run 0`, dump atom IDs and forces, sort by ID, and compare total energy
   and maximum absolute component errors.
4. For magnetic models, compare spin-force convention and complete a short
   finite `spin/sib` trajectory. For BEC, compare all nine tensor components
   and verify that `piezoelectric` is absent from model output.
5. Record model hash, source commit, binary hash, ranks, node count, and fixed
   tolerances with the result.

The repository tests implement this comparison for deterministic fixture
models:

- [`tests_lammps/test_pair_parity_run0.py`](../../tests_lammps/test_pair_parity_run0.py):
  single-frame Python versus LAMMPS `run 0`;
- [`tests_lammps/test_trajectory_parity.py`](../../tests_lammps/test_trajectory_parity.py):
  per-frame Python TorchScript/PT2 versus serial and MPI LAMMPS trajectories;
- [`tests_lammps/test_mpi_parity.py`](../../tests_lammps/test_mpi_parity.py):
  serial versus one-, two-, and four-rank LAMMPS.

They take the binaries from `SPINGNN_LAMMPS_BIN` and, for the `mpi` cases,
`SPINGNN_LAMMPS_MPI_BIN`.

### CI commands

```bash
# Complete source-exact CPU suites.
tools/ci/cpu_unit_tests.sh

# Validate the bundled example configs.
python tools/ci/source_python.py example/validate_configs.py

# Run the GPU suites on the target machine.
SPINGNN_GPU_EXPECT=h100 tools/ci/gpu_full_tests.sh
tools/ci/gpu_lammps_tests.sh
```

Read [`tools/ci/README.md`](../../tools/ci/README.md) for prerequisites, path
overrides, and exit codes. Run CUDA validation with access to the target GPU.

For a single-node two-rank run, the validated correctness transport fallback
is:

```bash
env OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp \
    UCX_TLS=self,tcp UCX_POSIX_USE_PROC_LINK=n \
  mpirun --allow-run-as-root --oversubscribe -np 2 \
  /shared/path/lmp -in /shared/path/in.spin-md
```

Do not extrapolate this result to multiple nodes. Consult the detailed
[LAMMPS runtime documentation](../../lammps/README.md) and
[extension templates](../../lammps/examples/SPINGNN/README.md) when a command
surface is not covered above.
