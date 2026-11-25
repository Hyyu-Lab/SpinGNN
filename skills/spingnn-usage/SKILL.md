---
name: spingnn-usage
description: Prepare magnetic or Born-effective-charge data, train SpinGNN++ v1 (with PhysMuon/Muon/Adam) or pure-PyTorch spingnn2.v2 potentials, export .sp2.json/.sp2.bin contracts and bundle_abi=1 multi-GPU JIT artifacts, analyze GPST spin temperatures, and run modern LAMMPS (pair_style spingnn2/kk, fix spin/sib/kk, fix spin/cfi/kk box iso, compute spin/gpst/kk, DREAM BEC, e2SLP).
---

# SpinGNN2 Usage

Operate from the repository root. Read [INSTALL.md](../../INSTALL.md) and
[docs/SPINGNN2_GUIDE.md](../../docs/SPINGNN2_GUIDE.md) before creating an
environment or building LAMMPS. Replace all example data, model, species,
split, and output paths: datasets and production models are not distributed
with this repository.

## Preserve the workflow contracts

- Use `units metal` for `pair_style spingnn2` and `spingnn2/kk`; these paths
  interpret model values as eV, Angstrom, and eV/Angstrom.
- Treat `pair_style spingnn2(/kk)` via the modern LAMMPS overlay
  ([`lammps_spingnn2/README.md`](../../lammps_spingnn2/README.md)) as the
  primary LAMMPS path for all SpinGNN++ / DREAM / e2SLP models.
- Keep per-atom BEC data flattened as `born_effective_charges` with shape
  `(N,9)` throughout ingestion, training, deployment, and parity checks.
- Do not restore a piezo output contract. A BEC model outputs energy, forces,
  stress, BEC, and total BEC. An extxyz `piezoelectric` field is data only;
  `fix efield ... piezo yes` must fail explicitly.
- Keep single-node and multi-node claims grounded in the measured protocols in
  [`docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md) and
  [`docs/VALIDATION.md`](../../docs/VALIDATION.md) (4-GPU single-node and
  2-node × 2-GPU RTX 5090 thermal MD, plus RTX 4090 / CPU qualification).
- Confirm `nequip.__file__`, `allegro.__file__`, and `spingnn2.__file__` point
  to the intended worktree when source-exact behavior matters.

## 1. Prepare data

### Magnetic OUTCAR and OSZICAR data (VASP 5 & VASP 6)

Use `spingnn-data -m` (or `python tools/ci/source_python.py spingnn-data -m`) to
process a VASP result tree in parallel across multiple CPU cores (or convert a
single `--outcar` / `--oszicar` pair) and write ASE-readable extxyz. In batch
mode, the parser pairs each `OUTCAR` with the sibling `OSZICAR` formed by
replacing `OUTCAR` with `OSZICAR` (and optional sibling `INCAR` for VASP 5 runs
without an `OUTCAR` `INCAR:` echo).

```bash
spingnn-data -m \
  --path /shared/path/to/vasp-results \
  --out-path data/magnetic.extxyz \
  --nomagz 8,83 \
  --cpu 8

# Or single-frame conversion:
spingnn-data -m \
  --outcar run/OUTCAR --oszicar run/OSZICAR \
  --out-path data/single.extxyz
```

By default (`--vasp-version auto`), `spingnn-data` automatically distinguishes
VASP 5 (`1 * lambda*MW_perp`, modes 1/2 ratio $R = \sum|\mathbf{P}_I|^2/(\lambda E_p) \approx 1 \Rightarrow \times 2$)
from VASP 6 (`2 * lambda*MW_perp`, modes 1/2 $R \approx 4 \Rightarrow \times 1$;
mode 4 linear penalty weighted ratio $R_4 = \sum|\mathbf{M}_{W,I}||\mathbf{P}_I|^2/(\lambda E_p) \approx 2 \Rightarrow \times 1$),
cross-checks against the `OUTCAR` version and `M_CONSTR`, subtracts the penalty
energy $E_p$ (`F - E_p`, or `--energy-source zero` for `E0 - E_p`), applies the
`SAXIS` Euler rotation, handles sparse nonmagnetic rows, and projects the
constraint field tangentially onto the conjugate `MW_int` direction (default;
pass `--spin-source m_int` or `outcar` if desired). Pass
`--vasp-version 5` or `--vasp-version 6` (`--vasp6`) only when you want to
assert a specific VASP convention explicitly. Use `--nomagz` to force known
nonmagnetic atomic numbers to zero; otherwise constrained sites and any
unconstrained sites with $|\mathbf{M}| \ge \text{--min-mag-norm}$ keep their
magnetic moments (with `spin_forces_vert = 0` on unconstrained sites).

Required magnetic fields are:

- `initial_magmoms`: per-atom `(N,3)` unit spin directions (zero on nonmagnetic
  sites; pass `--no-normalize-spin` for raw vectors).
- `spin_length`: per-atom `(N,)` magnetic moment magnitudes in $\mu_B$.
- `spin_forces_vert`: per-atom `(N,3)` tangential spin torque supervision
  $|\mathbf{M}_{W,I}|\mathbf{C}_{I,\perp}$ in eV from the `OSZICAR`
  constrained-moment blocks (strictly orthogonal to `initial_magmoms` and zero
  on nonmagnetic sites).

The `spin_forces_vert` target stores only the tangential torque component
perpendicular to the spin direction ($|\mathbf{M}_{W,I}|(\mathbf{C}_I -
(\mathbf{C}_I\cdot\hat{\mathbf{s}}_I)\hat{\mathbf{s}}_I)$), not a complete
`spin_forces = -dE/ds` vector. Never copy or rename this label to
`spin_forces`. For unconstrained magnetic runs without `OSZICAR` constraint
blocks, `spingnn-data -m` writes `initial_magmoms` and `spin_length` without
`spin_forces_vert`; remove `spin_forces_vert` from both losses and metrics
unless every supervised frame provides it.

The implementation and CLI contract are in
[`src/spingnn2/data/vasp_spin.py`](../../src/spingnn2/data/vasp_spin.py)
(with legacy Python helpers in
[`src/core/nequip/utils/data_outcar_spin.py`](../../src/core/nequip/utils/data_outcar_spin.py)).

### Born-effective-charge data

When an existing extxyz dataset and indexed VASP outputs are aligned, add BEC
labels with:

```bash
spingnn-data-mlff-bec \
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
disabled. H100 reference results were 23.4x for
Allegro, with synchronized loss-curve relative errors below 1e-6. Read the
[`training tutorial`](../../example/TUTORIAL_TRAINING.md) before changing these
controls.

### Structure-only potential

Start from
[`example/01_structure_only.yaml`](../../example/01_structure_only.yaml).
It uses `allegro.model.Allegro` with energy, force, and stress supervision.

```bash
cp example/01_structure_only.yaml run/structure-smoke.yaml
spingnn-train run/structure-smoke.yaml
```

Remove a loss and its metric when that label is absent. Do not manufacture
force or stress labels.

### Magnetic potential without SOC

Start from
[`example/02_spin_nosoc.yaml`](../../example/02_spin_nosoc.yaml).
It must use the dedicated `allegro.model.Allegro_nosoc` builder.

```bash
cp example/02_spin_nosoc.yaml run/nosoc-smoke.yaml
spingnn-train run/nosoc-smoke.yaml
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
spingnn-train run/soc-smoke.yaml
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
spingnn-train run/bec-smoke.yaml
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
- **First-party `PhysMuon` / `Muon` optimizers**: Both v1 YAML configs
  (`optimizer_name: PhysMuon` or `Muon`) and pure-PyTorch `spingnn2.v2`
  (`build_hybrid_muon(..., phys=True)`) support whole-step CUDA-Graphed
  Newton–Schulz orthogonalized updates with `iso`, `layer_balance`,
  `spectral_clip`, `block_norm`, and `spike_guard`.
- **Pure-PyTorch `spingnn2.v2` training**: Use `spingnn2.v2` (`v2_nosoc`,
  `v2_soc`, `v2_born`, `v2_e2slp`) with `compile_fullgraph(dynamic=True)` or
  capped-bucket `compile_bucketed(max_buckets=4)` for `3.3×–11.5×` speedups and
  `71%–91%` VRAM reduction with zero `e3nn` runtime dependency.
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

### Option A: SpinGNN2 Native Engine & Modern LAMMPS (Recommended for SpinGNN++ models)

Supported families: `Allegro_nosoc`, `Allegro_soc`, `Allegro_born` (DREAM),
`Allegro_e2IP`, and `Allegro_soc_e2IP`. (`nequip` / `non-local message-passing` are out of
scope for SpinGNN2 and use Option B below.)

1. **Export compact contract v1 (`.sp2.json` + `.sp2.bin`)** in the training
   environment:

```bash
export SPINGNN_TRAIN_DIR=results/spin_nosoc/example
mkdir -p deployed
python tools/ci/source_python.py spingnn2-export \
  "$SPINGNN_TRAIN_DIR" -o deployed/model
```

2. **Pre-compile offline JIT artifact or multi-GPU `bundle_abi = 1` bundle**
   once per target backend, architecture, and precision (modern LAMMPS never
   invokes NVRTC or a C++ compiler at runtime):

```bash
ARTIFACT=$(python tools/ci/source_python.py spingnn2-prepare-jit-artifact \
  --model deployed/model.sp2.json \
  --backend cuda --arch all-gpu --precision float32 \
  --library validation/native/build-cuda-all-gpu/libspingnn2.so)
echo "Prepared manifest: $ARTIFACT"
```

3. **Run in Python / ASE (zero PyTorch or LibTorch required on inference host)**:

```python
from spingnn2.native import NativeModel
from spingnn2.calculator import SpinGNN2Calculator

with NativeModel.load(
    "deployed/model.sp2.json",
    backend="cuda",
    precision="float32",
    execution_mode="direct",
    artifact=artifact_manifest,
    execution_profile="speed",
    radial_mode="spline",
) as model:
    out = model.evaluate(graph)

atoms.calc = SpinGNN2Calculator(
    "deployed/model.sp2.json",
    backend="cuda",
    precision="float32",
    execution_mode="direct",
    artifact=artifact_manifest,
    radial_mode="spline",
)
```

4. **Run in modern LAMMPS (`stable_30Sep2026_update1`, `pair_style spingnn2(/kk)`)**:

```lammps
units       metal
boundary    p p p
atom_style  spin
atom_modify map array
read_data   /shared/path/system-spin.data

pair_style  spingnn2 precision float32 math ieee profile yes
pair_coeff  * * deployed/model.sp2.json prepared-artifact.bundle.json Fe O

# 2-eval/step SIB, or 1-eval/step CFI with optional canonical-volume box iso barostat:
fix         int all spin/sib 300.0 0.1 91081 lattice moving
# fix       int all spin/cfi 300.0 0.1 91081 lattice moving ou 300.0 0.05 91082 box iso 0.0 100.0 box_ou 300.0 0.1 91083
compute     gpst all spin/gpst cutoff 5.0
compute     timers all spingnn2/timing
thermo_style custom step temp pe etotal c_timers[*]
timestep    0.0001
run         100
```

Offline GPST moment extraction & independent-chain CF-GMM / Fieller inference:

```bash
spingnn2-gpst-moments accepted.dump moments.npz \
  --hbar 0.0006582119569 --cutoff 5.0 --accept-every 10 --force-contract-confirmed
spingnn2-gpst-infer --records chain1.npz chain2.npz --seed-ids s1 s2 \
  --kb 8.617333262e-5 --dt 0.01 --output gpst.json
```

Multi-GPU launch (1 MPI rank per physical GPU UUID via
`tools/lammps/spingnn2_gpu_rank.py`; add `I_MPI_OFFLOAD=1 I_MPI_OFFLOAD_IPC=0`
on Intel MPI + RTX 4090 when CUDA IPC peer reads fail):

```bash
mpirun -n 4 tools/lammps/spingnn2_gpu_rank.py \
  /path/to/install/bin/lmp_spingnn2 \
  -k on g 1 -sf kk -pk kokkos gpu/aware on neigh half \
  -in in.spin_md
```

See [`docs/SPINGNN2_GUIDE.md`](../../docs/SPINGNN2_GUIDE.md) and
[`skills/spingnn2/SKILL.md`](../spingnn2/SKILL.md) for DREAM BEC, e2SLP
`fix pair`, `fix spin/sib/npt(/kk)`, `fix spin/cfi(/kk)`, `math tf32` accuracy
caveats, and CPU Kokkos launches.

### Option B: Legacy TorchScript / PT2 Deployment (`spingnn-deploy`)

> **Note:** Modern SpinGNN2 simulations use Option A (`spingnn2-export` + `spingnn2-prepare-jit-artifact` + `lmp_spingnn2` with `pair_style spingnn2(/kk)`). Use `spingnn-deploy` only for Python TorchScript evaluation (`tools/e2slp/evaluate.py`, `load_deployed_model`) or external legacy LibTorch/PT2 runners.

Resolve the training directory from the config's `root/run_name`, then build
a TorchScript (`.pth`) or PT2 (`.pt2` + `.pt2.meta`) artifact:

```bash
export SPINGNN_TRAIN_DIR=results/spin_nosoc/example
test -s "$SPINGNN_TRAIN_DIR/config.yaml"
test -s "$SPINGNN_TRAIN_DIR/best_model.pth"

mkdir -p deployed
# TorchScript (.pth) — used by tools/e2slp/evaluate.py and load_deployed_model
spingnn-deploy build --train-dir "$SPINGNN_TRAIN_DIR" deployed/model.pth
spingnn-deploy info deployed/model.pth

# Optional PT2 (.pt2 + .pt2.meta) for external legacy LibTorch runners
spingnn-deploy build-pt2 --device auto --train-dir "$SPINGNN_TRAIN_DIR" deployed/model.pt2
```

Load TorchScript through the canonical deployment API so metadata global
options, including TF32 and fusion settings, are applied consistently (note: do
**not** wrap forward calls in `torch.no_grad()` when evaluating forces or spin
forces):

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

## 4. Run Modern LAMMPS (`lmp_spingnn2`)

Build and verify modern LAMMPS (`lmp_spingnn2`, `stable_30Sep2026_update1`) as
documented in [INSTALL.md](../../INSTALL.md#3-modern-lammps-stable_30sep2026_update1-lmp_spingnn2).
Every element after the artifact manifest path in `pair_coeff` maps one LAMMPS
atom type, in order, to the exported `type_names` metadata.

Start from the runnable modern LAMMPS guides rather than writing inputs from
memory:

- [`lammps_spingnn2/README.md`](../../lammps_spingnn2/README.md)
- [`docs/SPINGNN2_GUIDE.md`](../../docs/SPINGNN2_GUIDE.md)
- [`skills/spingnn2/SKILL.md`](../spingnn2/SKILL.md)

### Common Modern LAMMPS Contracts & Error Map

| Error or symptom | Contract and action |
| --- | --- |
| `units metal` mismatch | Always place `units metal` before `pair_style spingnn2`; energy/force/field units are eV, Å, eV/Å. |
| Artifact SHA-256 / ABI / arch mismatch | Re-run `spingnn2-prepare-jit-artifact --arch all-gpu` for the exact `.sp2.json` and precision; keep `<key>.json` / `.bundle.json` and `.cubin`/`.so` in a non-group-writable directory. |
| Missing spin attributes, map, or integrator | Use `atom_style spin`, `atom_modify map array`, and one of `fix spin/sib(/npt)` or `fix spin/cfi`. |
| `box iso` rejected on triclinic box | `fix spin/cfi ... box iso` requires an orthogonal periodic 3D box. |
| Model species mapping fails | Match `pair_coeff` chemical symbols to the exported `type_names`, in exact type-index order. |
| `Fix efield piezo yes is unavailable` | Piezo output is not part of the DREAM contract; use `atom_style bec`, `fix efield ... coupling bec`, and `compute dip all bec`. |
| Non-GPU-aware MPI crash with Kokkos CUDA | Launch via `tools/lammps/spingnn2_gpu_rank.py` with `-pk kokkos gpu/aware off neigh half` (or `I_MPI_OFFLOAD=1 I_MPI_OFFLOAD_IPC=0` on Intel MPI when P2P IPC fails). |

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

- [`tests_native/test_contract_oracle.py`](../../tests_native/test_contract_oracle.py):
  PyTorch eager versus SpinGNN2 contract & NumPy reference oracle;
- [`tests_native/test_native_engine.py`](../../tests_native/test_native_engine.py):
  native `libspingnn2.so` CPU/CUDA parity & ASE calculator checks;
- [`tests_native/test_lammps_pair.py`](../../tests_native/test_lammps_pair.py):
  single-rank and multi-rank `lmp_spingnn2` (`pair_style spingnn2(/kk)`) parity.

They take the modern LAMMPS binary from `SPINGNN2_LAMMPS_BIN`.

### CI commands

```bash
# Complete source-exact CPU suites.
tools/ci/cpu_unit_tests.sh

# Validate the bundled example configs.
python tools/ci/source_python.py example/validate_configs.py

# Run the GPU suites on the target machine.
SPINGNN_GPU_EXPECT=5090 tools/ci/gpu_full_tests.sh
```

Read [`tools/ci/README.md`](../../tools/ci/README.md) for prerequisites, path
overrides, and exit codes. Run CUDA validation with access to the target GPU.

Consult the detailed
[LAMMPS runtime documentation](../../lammps_spingnn2/README.md) and
[SpinGNN2 User Guide](../../docs/SPINGNN2_GUIDE.md) when a command
surface is not covered above.
