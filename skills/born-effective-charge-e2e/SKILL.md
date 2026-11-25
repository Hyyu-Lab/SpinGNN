---
name: born-effective-charge-e2e
description: End-to-end workflow for DREAM Born effective charge (BEC) models (Allegro_born / v2_born) — from VASP DFPT BEC data extraction (spingnn-data / spingnn-data-mlff-bec) through training with acoustic sum rule, .sp2.json/.sp2.bin export, bundle_abi=1 JIT compilation, and modern LAMMPS electric-field MD (atom_style bec, pair_style spingnn2/kk, fix efield coupling bec, compute bec).
---

# DREAM Born Effective Charge (BEC): End-to-End Workflow

Complete pipeline for training equivariant Born effective charge (BEC) models
(`Allegro_born` in v1 YAML or `v2_born` in pure-PyTorch `spingnn2.v2`) and
running electric-field molecular dynamics for ferroelectric and polar
materials in modern LAMMPS (`stable_30Sep2026_update1`).

## When to Use

Use this skill when you need to:
- Predict per-atom $3 \times 3$ Born effective charge tensors $Z^*_i$ alongside energy, forces, and stress (ferroelectrics such as $\text{BaTiO}_3$, $\text{HfO}_2$, $\text{PbTiO}_3$)
- Enforce the acoustic sum rule ($\sum_i Z^*_{i,\alpha\beta} = 0$) via `total_born_effective_charges` and/or hard ASR in native/LAMMPS inference
- Run zero-LibTorch electric-field MD (`atom_style bec`, `fix efield ... coupling bec`, `pair_style spingnn2(/kk)`)

**Trigger phrases:** "Born effective charge", "BEC training", "DREAM", "ferroelectric", "电场耦合", "Z* tensor"

**NOT for:** Pure magnetic models without BEC (use [`skills/spin-nosoc-e2e/SKILL.md`](../spin-nosoc-e2e/SKILL.md) or [`skills/spin-soc-e2e/SKILL.md`](../spin-soc-e2e/SKILL.md)).

> **Contract note:** SpinGNN2 BEC models output energy, forces, stress/virial, `born_effective_charges` (flattened `(N, 9)` row-major), and `total_born_effective_charges`. There is **no** piezoelectric output head; any `piezoelectric` field in `extxyz` is stored as dataset metadata only, and `fix efield ... piezo yes` is explicitly rejected.

## Step 1: Prepare BEC Dataset

SpinGNN2 provides two parallel CPU data converters for VASP DFPT (`LEPSILON = .TRUE.` / `IBRION = 8`) outputs:

```bash
cd /path/to/SpinGNN2

# Option A: Extract energy, forces, stress, and BEC directly from a VASP tree
python tools/ci/source_python.py spingnn-data -b \
  --path /path/to/vasp-dfpt-runs \
  --out-path data/born_effective_charge.extxyz \
  --cpu 8

# Option B: Attach BEC from indexed OUTCAR.<idx> + log.out.<idx> onto an existing extxyz
python tools/ci/source_python.py spingnn-data-mlff-bec \
  --dataset-path data/structure.extxyz \
  --folder /path/to/indexed-vasp-results \
  --out-path data/born_effective_charge.extxyz \
  --criteria 1.0e-2 \
  --cpu 8
```

### Validate BEC dataset schema

```bash
python - <<'PY'
import numpy as np
from ase.io import read

frames = read("data/born_effective_charge.extxyz", index=":")
assert frames, "empty dataset"
print(f"Total BEC frames: {len(frames)}")

for idx in (0, len(frames) - 1):
    atoms = frames[idx]
    assert np.isfinite(atoms.positions).all()
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()

    bec = atoms.arrays["born_effective_charges"]
    assert bec.shape == (len(atoms), 9), f"expected (N, 9), got {bec.shape}"
    assert np.isfinite(bec).all()
    sum_err = np.abs(bec.reshape(-1, 3, 3).sum(axis=0)).max()
    print(f"Frame {idx}: {len(atoms)} atoms, ASR max |Σ Z*| = {sum_err:.3e} e")

print("BEC_DATASET_OK")
PY
```

## Step 2: Configure BEC Training

Copy the validated DREAM BEC template:

```bash
cp example/04_born_effective_charge.yaml config/my_bec.yaml
```

### Key BEC settings (matching `example/04_born_effective_charge.yaml`)

```yaml
root: results/born_effective_charge
run_name: my_bec_run
dataset_file_name: data/born_effective_charge.extxyz
n_train: 800
n_val: 100
chemical_symbol_to_type:
  O: 0
  Hf: 1
r_max: 5.0

model_builders:
  - allegro.model.Allegro_born
  - PerSpeciesRescale
  - ParaStressForceOutput
  - RescaleEnergyEtc

# l_max >= 2 is required for 3x3 rank-2 BEC tensors
l_max: 2
parity: o3_full

# One scale per type in chemical_symbol_to_type order (e.g. O: -2, Hf: +4)
born_scale_list: [-2.0, 4.0]
born_scale_trainable: false

loss_coeffs:
  total_energy: [1.0, PerAtomMSELoss]
  forces: 1.0
  stress: 1.0
  born_effective_charges: 1.0
  total_born_effective_charges: 1.0  # enforces acoustic sum rule (target = 0)

metrics_components:
  - [forces, mae]
  - [born_effective_charges, mae]
```

> **Note:** `full_fx` compilation applies to `ParaStressForceSpinOutput`; `example/04_born_effective_charge.yaml` (`ParaStressForceOutput`) keeps `spingnn_compile.enabled: false` with TF32 enabled. For `torch.compile` + `PhysMuon` BEC training (`3.8×–10.3×` speedup, `−88%` VRAM), use `spingnn2.v2` `build_v2_born` or enable `optimizer_name: PhysMuon` in YAML.

## Step 3: Train the BEC Model

```bash
cd /path/to/SpinGNN2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/ci/source_python.py spingnn-train config/my_bec.yaml
```

## Step 4: Export & Pre-Compile SpinGNN2 DREAM Artifact (Recommended)

Export your trained `Allegro_born` checkpoint to `.sp2.json` + `.sp2.bin` and
pre-compile an offline single-arch or multi-GPU `bundle_abi = 1` JIT artifact
(**3.94 µs/atom at 32k atoms**, **2,000,000 atoms in 2.01 GB VRAM on one RTX
5090** in strict FP32; [`docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md)):

```bash
TRAIN_DIR=results/born_effective_charge/my_bec_run
mkdir -p deployed

python tools/ci/source_python.py spingnn2-export \
  "$TRAIN_DIR" -o deployed/bec_model

ARTIFACT=$(python tools/ci/source_python.py spingnn2-prepare-jit-artifact \
  --model deployed/bec_model.sp2.json \
  --backend cuda --arch all-gpu --precision float32 \
  --library validation/native/build-cuda-all-gpu/libspingnn2.so)
echo "Prepared manifest: $ARTIFACT"
```

### Validate zero-LibTorch Python / ASE inference

```python
import numpy as np
from ase.io import read
from spingnn2.calculator import SpinGNN2Calculator

atoms = read("data/born_effective_charge.extxyz", index=0)
atoms.calc = SpinGNN2Calculator(
    "deployed/bec_model.sp2.json",
    backend="cuda",
    precision="float32",
    execution_mode="direct",
    artifact=artifact_manifest,
    radial_mode="spline",
)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
bec = atoms.calc.get_property("born_effective_charges", atoms)  # (N, 3, 3)
print(f"E={energy:.6f} eV  BEC shape={bec.shape}  ASR={np.abs(bec.sum(0)).max():.3e} e")
```

> If evaluating a TorchScript checkpoint directly (`load_deployed_model`), **do not** wrap the call in `torch.no_grad()` because forces require autograd.

## Step 5: Run Modern LAMMPS Electric-Field MD (`atom_style bec`, `pair_style spingnn2(/kk)`)

Modern LAMMPS (`stable_30Sep2026_update1`, built with
`tools/spingnn2_build.py --target all-gpu|sm120|sm89|cpu`) supports DREAM BEC
on both CPU and Kokkos CUDA (`-sf kk`):

```lammps
units       metal
boundary    p p p
atom_style  bec
atom_modify map array
read_data   system_bec.data
# Optional: load reference BEC tensors by atom ID
# read_bec  system.bec

pair_style  spingnn2 precision float32 math ieee
pair_coeff  * * deployed/bec_model.sp2.json prepared-artifact.bundle.json O Hf

# Uniform electric field in eV/(Angstrom*e); applies F = Z*^T E
fix         field all efield 0.0 0.0 0.001 coupling bec
velocity    all create 300.0 12345 mom yes rot no dist gaussian
fix         nvt all nvt temp 300.0 300.0 0.1
timestep    0.0005

compute     dip all bec
compute     becprops all property/atom becxx becxy becxz becyx becyy becyz &
            beczx beczy beczz becdipolex becdipoley becdipolez
thermo      100
thermo_style custom step temp pe ke etotal c_dip c_dip[1] c_dip[2] c_dip[3] c_dip[4]
dump        traj all custom 100 bec_traj.lammpstrj &
            id type x y z fx fy fz c_becprops[*]
dump_modify traj sort id
run         1000
```

Use `bec_dipole_reset` after equilibration to reset the integrated dipole
reference, and launch multi-GPU CUDA runs via
`tools/lammps/spingnn2_gpu_rank.py` with
`-k on g 1 -sf kk -pk kokkos gpu/aware on neigh half` (or `gpu/aware off` on
non-GPU-aware MPI stacks).

## Step 6: Troubleshooting & Testing

| Symptom | Cause | Solution |
| --- | --- | --- |
| Shape error on BEC targets | Dataset stores `(N, 3, 3)` instead of `(N, 9)` | Flatten row-major to `(N, 9)` (done automatically by `spingnn-data -b` and `spingnn-data-mlff-bec`) |
| `l_max` / tensor irreps error | `l_max < 2` | Set `l_max: 2` (or higher) |
| `born_scale_list` mismatch | Length or species order differs from `chemical_symbol_to_type` | Provide exactly one scale per type in integer index order |
| `Fix efield piezo yes is unavailable` | Piezo output head is not part of the DREAM contract | Use `fix efield ... coupling bec` without `piezo yes` |
| Unknown `becxx` / `atom_style bec` | Upstream unpatched LAMMPS binary | Build `lmp_spingnn2` via `tools/spingnn2_build.py` |

```bash
cd /path/to/SpinGNN2
tools/ci/cpu_unit_tests.sh
python tools/ci/source_python.py -m pytest tests_v2 -k born -v

export SPINGNN2_LAMMPS_BIN=/path/to/install/bin/lmp_spingnn2
python tools/ci/source_python.py -m pytest \
  tests_native/test_lammps_bec.py tests_native/test_spin_cfi.py -k bec -v
```

## See Also

- [`example/04_born_effective_charge.yaml`](../../example/04_born_effective_charge.yaml)
- [`docs/SPINGNN2_GUIDE.md`](../../docs/SPINGNN2_GUIDE.md) · [`lammps_spingnn2/README.md`](../../lammps_spingnn2/README.md)
- [`skills/spingnn2/SKILL.md`](../spingnn2/SKILL.md) · [`tests_native/test_lammps_bec.py`](../../tests_native/test_lammps_bec.py)
