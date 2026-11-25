---
name: spin-nosoc-e2e
description: End-to-end workflow for magnetic SpinGNN++ without SOC (Allegro_nosoc / v2_nosoc) — from parallel VASP 5 & 6 dataset preparation (spingnn-data) through training (Adam/PhysMuon), .sp2.json/.sp2.bin export, bundle_abi=1 JIT compilation, and modern LAMMPS spin-lattice MD (pair_style spingnn2/kk, fix spin/sib/kk, fix spin/cfi/kk box iso, compute spin/gpst/kk).
---

# Spin Allegro (No SOC): End-to-End Workflow

Complete pipeline for training and deploying a magnetic potential without
spin–orbit coupling (`Allegro_nosoc` in v1 YAML or `v2_nosoc` in pure-PyTorch
`spingnn2.v2`) for Heisenberg-like magnets where interactions obey decoupled
$O(3)_{\text{lat}} \times O(3)_{\text{spin}}$ symmetry.

## When to Use

Use this skill when you need to:
- Train a magnetic potential for 3d transition-metal / Heisenberg-like systems where SOC is negligible
- Supervise energy, forces, stress, and tangential spin torques (`spin_forces_vert`) from VASP 5 or VASP 6 constrained-M DFT
- Deploy with zero LibTorch into Python/ASE (`NativeModel`, `SpinGNN2Calculator`) or modern LAMMPS (`pair_style spingnn2(/kk)`)
- Run coupled spin–lattice dynamics (`fix spin/sib(/npt)/kk`, `fix spin/cfi(/kk)` with `box iso`) and GPST spin thermometry (`compute spin/gpst(/kk)`)

**Trigger phrases:** "spin nosoc", "magnetic without SOC", "Heisenberg", "非SOC自旋模型", "spin directions"

**NOT for:** Systems with strong SOC / anisotropy / DMI (use [`skills/spin-soc-e2e/SKILL.md`](../spin-soc-e2e/SKILL.md)), structure-only (use [`skills/structure-only-e2e/SKILL.md`](../structure-only-e2e/SKILL.md)).

## Prerequisites

- SpinGNN2 repository at `/path/to/SpinGNN2` with validated Python 3.10 / PyTorch 2.11+cu128 / T-e3nn environment (see [`INSTALL.md`](../../INSTALL.md))
- Modern LAMMPS binary (`lmp_spingnn2`) built via `python tools/spingnn2_build.py --target all-gpu|sm120|sm89|cpu`
- Converged VASP 5.x or VASP 6.x `OUTCAR` (+ `OSZICAR`) runs, or an ASE `extxyz` dataset with `energy`, `forces`, `initial_magmoms`, `spin_length`, and `spin_forces_vert`

## Key Differences from Structure-Only

| Feature | Structure-only | Spin (No SOC) |
| --- | --- | --- |
| Model builder | `allegro.model.Allegro` / `Allegro_e2IP` | `allegro.model.Allegro_nosoc` / `v2_nosoc` |
| Input | positions only | positions + 3D `initial_magmoms` (+ `spin_length`) |
| Supervision | energy, forces, stress | + `spin_forces_vert` (tangential torque) |
| Symmetry | $E(3)$ equivariant | Decoupled $O(3)_{\text{lat}} \times O(3)_{\text{spin}}$ ($T$-even) |
| Modern LAMMPS pair style | `spingnn2(/kk)` (for `Allegro_e2IP`) | `spingnn2(/kk)` |
| MD integrators | standard `nve`/`nvt` | `fix spin/sib(/npt)/kk`, `fix spin/cfi(/kk)` |

## Step 1: Prepare Magnetic Dataset (`spingnn-data`)

Convert converged VASP 5.x or VASP 6.x constrained-M `OUTCAR`/`OSZICAR` trees
in parallel across CPU cores with `spingnn-data -m`:

```bash
cd /path/to/SpinGNN2

python tools/ci/source_python.py spingnn-data -m \
  --path /path/to/vasp-runs \
  --out-path data/spin_nosoc.extxyz \
  --nomagz 8,83 \
  --cpu 8

# Or single-frame conversion:
python tools/ci/source_python.py spingnn-data -m \
  --outcar run/OUTCAR --oszicar run/OSZICAR \
  --out-path data/single.extxyz
```

`spingnn-data` collects only converged final steps (skipping pre-convergence
SCF blocks), automatically detects VASP 5 (`1 * lambda*MW_perp` → `×2`) vs
VASP 6 (`2 * lambda*MW_perp` → `×1`) and Mode 4 linear penalty (`×1`),
subtracts penalty energy $E_p$, applies `SAXIS` rotation, and writes:
- `initial_magmoms`: `(N, 3)` unit spin directions (zero on nonmagnetic sites)
- `spin_length`: `(N,)` magnetic moment magnitudes in $\mu_B$
- `spin_forces_vert`: `(N, 3)` tangential spin torque $|\mathbf{M}_{W,I}|\mathbf{C}_{I,\perp}$ in eV (strictly perpendicular to `initial_magmoms`)

> **Important:** Never copy or rename `spin_forces_vert` to `spin_forces`. For unconstrained runs without `OSZICAR` constraint blocks, remove `spin_forces_vert` from both `loss_coeffs` and `metrics_components`.

### Validate dataset schema

```bash
python - <<'PY'
import numpy as np
from ase.io import read

frames = read("data/spin_nosoc.extxyz", index=":")
assert frames, "empty dataset"
print(f"Total frames: {len(frames)}")

for idx in (0, len(frames) - 1):
    atoms = frames[idx]
    assert np.isfinite(atoms.positions).all()
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()

    magmoms = atoms.arrays["initial_magmoms"]
    assert magmoms.shape == (len(atoms), 3) and np.isfinite(magmoms).all()
    if "spin_forces_vert" in atoms.arrays:
        sfv = atoms.arrays["spin_forces_vert"]
        assert sfv.shape == (len(atoms), 3) and np.isfinite(sfv).all()
        dot = np.abs(np.sum(magmoms * sfv, axis=1)).max()
        print(f"Frame {idx}: {len(atoms)} atoms, max |s·f_spin_vert| = {dot:.3e}")

print("SPIN_DATASET_OK")
PY
```

## Step 2: Configure Training

Copy the validated NoSOC template:

```bash
cp example/02_spin_nosoc.yaml config/my_spin_nosoc.yaml
```

### Key NoSOC settings (matching `example/02_spin_nosoc.yaml`)

```yaml
root: results/spin_nosoc
run_name: my_spin_nosoc_run
dataset_file_name: data/spin_nosoc.extxyz
n_train: 800
n_val: 100
chemical_symbol_to_type:
  Mn: 0
  Ge: 1
r_max: 5.0

# Dedicated NoSOC builder (do NOT use Allegro_soc with lsoc_layer: false)
model_builders:
  - allegro.model.Allegro_nosoc
  - PerSpeciesRescale
  - ParaStressForceSpinOutput
  - RescaleEnergyEtc

lhb: true
lbiquadratic: false
lnosoc_layer: true
l_max: 2
parity: o3_full

loss_coeffs:
  total_energy: [1.0, PerAtomMSELoss]
  forces: 1.0
  spin_forces_vert: 1.0   # tangential spin torque supervision

metrics_components:
  - [forces, mae]
  - [spin_forces_vert, mae]
```

### Acceleration & `PhysMuon`

`example/02_spin_nosoc.yaml` enables PyTorch 2.11 `full_fx` dynamic compilation
and calibrated TF32 by default. You can also switch to SpinGNN2's first-party
`PhysMuon` optimizer (or train pure-PyTorch `spingnn2.v2` `build_v2_nosoc` with
`compile_bucketed` for `3.3×–7.9×` speedup and `−91%` VRAM):

```yaml
optimizer_name: PhysMuon
learning_rate: 0.01

spingnn_compile:
  enabled: true
  scope: full_fx
  dynamic: true

float32_matmul_precision: high
allow_tf32: true
```

## Step 3: Train the Model

```bash
cd /path/to/SpinGNN2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/ci/source_python.py spingnn-train config/my_spin_nosoc.yaml
```

### Monitor convergence

```bash
python - <<'PY'
import csv, pathlib

metrics = pathlib.Path("results/spin_nosoc/my_spin_nosoc_run/metrics_epoch.csv")
with metrics.open() as f:
    rows = list(csv.DictReader(f))

print(f"Total epochs: {len(rows)}")
f_col = next((k for k in rows[-1] if "validation" in k and "f_mae" in k or k == "validation_forces_mae"), None)
print("Last row:", {k.strip(): v.strip() for k, v in rows[-1].items() if "validation" in k})
PY
```

## Step 4: Export & Pre-Compile SpinGNN2 Native Artifact (Recommended)

Export your trained `Allegro_nosoc` checkpoint to `.sp2.json` + `.sp2.bin` and
pre-compile an offline single-arch or multi-GPU `bundle_abi = 1` JIT artifact
(**2,000,000 atoms per RTX 5090 in 1.58 GB VRAM**, 99.19% 4-GPU weak efficiency;
see [`docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md)):

```bash
TRAIN_DIR=results/spin_nosoc/my_spin_nosoc_run
mkdir -p deployed

# 1. Export compact contract (.sp2.json + .sp2.bin)
python tools/ci/source_python.py spingnn2-export \
  "$TRAIN_DIR" -o deployed/spin_nosoc

# 2. Pre-compile offline JIT artifact (all-gpu -> .bundle.json, or single sm80..sm120 / cpu)
ARTIFACT=$(python tools/ci/source_python.py spingnn2-prepare-jit-artifact \
  --model deployed/spin_nosoc.sp2.json \
  --backend cuda --arch all-gpu --precision float32 \
  --library validation/native/build-cuda-all-gpu/libspingnn2.so)
echo "Prepared manifest: $ARTIFACT"
```

### Validate zero-LibTorch Python / ASE inference

```python
from ase.io import read
from spingnn2.calculator import SpinGNN2Calculator

atoms = read("data/spin_nosoc.extxyz", index=0)
atoms.calc = SpinGNN2Calculator(
    "deployed/spin_nosoc.sp2.json",
    backend="cuda",
    precision="float32",
    execution_mode="direct",
    artifact=artifact_manifest,
    radial_mode="spline",
)
print("E =", atoms.get_potential_energy())
print("F max =", abs(atoms.get_forces()).max())
print("F_spin max =", abs(atoms.calc.get_property("spin_forces", atoms)).max())
```

> If you instead load a legacy TorchScript checkpoint (`spingnn-deploy build`), **do not** wrap inference in `torch.no_grad()` — forces and spin forces are computed via autograd even in `eval()` mode.

## Step 5: Run Modern LAMMPS Spin–Lattice MD (`pair_style spingnn2(/kk)`)

Create `in.spin_nosoc_md` for modern LAMMPS (`stable_30Sep2026_update1`, built
via `tools/spingnn2_build.py`):

```lammps
units       metal
boundary    p p p
atom_style  spin
atom_modify map array
read_data   spin_system.data

pair_style  spingnn2 precision float32 math ieee cache yes profile yes
pair_coeff  * * deployed/spin_nosoc.sp2.json prepared-artifact.bundle.json Mn Ge

# Coupled moving-lattice SIB (use `lattice frozen` for spin-only MD):
fix         integrate all spin/sib 300.0 0.1 91081 lattice moving
# Or 1-eval/step Cached-Field Integrator (alias spin/lpsib) with optional box iso NPT:
# fix integrate all spin/cfi 300.0 0.1 91081 lattice moving ou 300.0 0.05 91082 box iso 0.0 100.0 box_ou 300.0 0.1 91083
# Or spin-lattice NPT:
# fix integrate all spin/sib/npt temp 300 300 0.02 iso 1 1 0.05 alpha 0.1 seed 91081

compute     gpst all spin/gpst cutoff 5.0
compute     timers all spingnn2/timing
thermo      100
thermo_style custom step temp pe ke etotal c_timers[*]
timestep    0.0005
run         10000
```

Launch on 4 GPUs with Kokkos CUDA (one MPI rank per physical GPU UUID; use
`-pk kokkos gpu/aware off neigh half` on non-GPU-aware MPI stacks):

```bash
mpirun -n 4 tools/lammps/spingnn2_gpu_rank.py \
  /path/to/install/bin/lmp_spingnn2 \
  -k on g 1 -sf kk -pk kokkos gpu/aware on neigh half \
  -in in.spin_nosoc_md
```

**Stage-1 lattice caching**: With `cache yes` (default), `spingnn2(/kk)` reuses
Stage-1 lattice intermediates at `fix spin/sib` midpoints and across timesteps
under `lattice frozen` (**1.474× frozen** / **1.307× moving** speedup on RTX
5090). For offline GPST spin-temperature analysis from accepted-state dumps,
use `spingnn2-gpst-moments` and `spingnn2-gpst-infer`.

## Step 6: Troubleshooting & Testing

| Symptom | Cause | Solution |
| --- | --- | --- |
| `atom_style spin` errors in LAMMPS | Missing spin attributes / atom map | Set `atom_style spin` and `atom_modify map array` before `read_data` |
| `spin_forces_vert` missing during training | Unconstrained VASP data | Remove `spin_forces_vert` from `loss_coeffs` and `metrics_components` |
| Species mapping error in `pair_coeff` | Element order mismatch | Match `pair_coeff` symbols to `chemical_symbol_to_type` order |
| Slow training or OOM | Large batch / eager mode | Keep `spingnn_compile.enabled: true` or train with `spingnn2.v2` `v2_nosoc` + `PhysMuon` |

```bash
cd /path/to/SpinGNN2
tools/ci/cpu_unit_tests.sh
python tools/ci/source_python.py -m pytest tests_v2 tests_native/test_vasp_spin_io.py -q

# With lmp_spingnn2 built:
export SPINGNN2_LAMMPS_BIN=/path/to/install/bin/lmp_spingnn2
python tools/ci/source_python.py -m pytest \
  tests_native/test_lammps_spin_sib.py tests_native/test_spin_cfi.py tests_native/test_spin_gpst.py -v
```

## See Also

- [`example/02_spin_nosoc.yaml`](../../example/02_spin_nosoc.yaml) · [`example/TUTORIAL_TRAINING.md`](../../example/TUTORIAL_TRAINING.md)
- [`docs/SPINGNN2_GUIDE.md`](../../docs/SPINGNN2_GUIDE.md) · [`lammps_spingnn2/README.md`](../../lammps_spingnn2/README.md)
- [`skills/spingnn2/SKILL.md`](../spingnn2/SKILL.md) · [`skills/spin-soc-e2e/SKILL.md`](../spin-soc-e2e/SKILL.md)
