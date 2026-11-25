---
name: structure-only-e2e
description: End-to-end workflow for structure-only Allegro potential - from data preparation to LAMMPS MD simulation. Covers energy/force/stress training, deployment, and validation.
---

# Structure-Only Allegro: End-to-End Workflow

Complete pipeline for training and deploying a structure-only Allegro potential without magnetic degrees of freedom.

## When to Use

Use this skill when you need to:
- Train a structure-only (non-magnetic) Allegro potential
- Work with energy, force, and stress supervision only
- Deploy and validate a structure-only model in LAMMPS
- Troubleshoot structure-only training or deployment

**Trigger phrases:** "structure only", "non-magnetic Allegro", "纯结构模型", "without spins"

## Prerequisites

- SpinGNN2 repository at: `/path/to/SpinGNN2`
- Python environment with spin-fork packages installed (see [`docs/INSTALL_CUDA128.md`](../../docs/INSTALL_CUDA128.md) and [`INSTALL.md`](../../INSTALL.md))
- For native zero-LibTorch deployment in modern LAMMPS (`lmp_spingnn2`, `pair_style spingnn2(/kk)`), use `Allegro_e2IP` ([`example/06_allegro_e2ip.yaml`](../../example/06_allegro_e2ip.yaml))
- Dataset with `energy`, `forces`, and optionally `stress` labels

## Quick Reference

| Stage | Time | Command |
| --- | --- | --- |
| Data prep | ~1h per 1k frames | `spingnn-data` or manual extxyz |
| Training | 1-24h | `spingnn-train example/01_structure_only.yaml` |
| Deployment | ~1min | `spingnn-deploy build --train-dir <dir> model.pth` |
| LAMMPS run | seconds-hours | `lmp -in in.structure_md` |

## Step 1: Prepare Structure Data

### From VASP OUTCAR files

For non-magnetic VASP outputs:

```bash
cd /path/to/SpinGNN2

# Collect energy/force/stress from VASP tree in parallel across 8 CPU cores
spingnn-data \
  --path /path/to/vasp-results \
  --out-path data/structure.extxyz \
  --cpu 8
```

### Validate dataset schema

```bash
python - <<'PY'
import numpy as np
from ase.io import read

frames = read("data/structure.extxyz", index=":")
print(f"Total frames: {len(frames)}")

# Check first and last frames
for idx in [0, -1]:
    atoms = frames[idx]
    assert np.isfinite(atoms.positions).all(), f"Non-finite positions in frame {idx}"
    assert np.isfinite(atoms.get_potential_energy()), f"Non-finite energy in frame {idx}"
    assert np.isfinite(atoms.get_forces()).all(), f"Non-finite forces in frame {idx}"
    
    # Stress is optional
    try:
        stress = atoms.get_stress(voigt=False)
        assert np.isfinite(stress).all(), f"Non-finite stress in frame {idx}"
        print(f"Frame {idx}: {len(atoms)} atoms, has stress")
    except:
        print(f"Frame {idx}: {len(atoms)} atoms, no stress")

print("DATA_SCHEMA_OK")
PY
```

Expected fields in extxyz:
- `energy` (scalar per frame)
- `forces` (N×3 array)
- `stress` (3×3 or 6-component, optional)

## Step 2: Configure Training

Copy the structure-only template:

```bash
cp example/01_structure_only.yaml config/my_structure.yaml
```

### Minimum required edits

```yaml
# Data
dataset_file_name: data/structure.extxyz
n_train: 800  # adjust to your dataset size
n_val: 200
chemical_symbol_to_type:
  Fe: 0
  O: 1

# Model geometry
r_max: 6.0  # cutoff in Angstrom, tune for your system
l_max: 2    # angular resolution

# Loss (remove stress if not in dataset)
loss_coeffs:
  total_energy:
    - 1.0
    - PerAtomMSELoss
  forces: 1.0
  stress: 1.0  # <-- remove this line if no stress in data

# Output
root: results/structure
run_name: my_structure_run
```

### Key config parameters

| Parameter | Purpose | Typical values |
| --- | --- | --- |
| `r_max` | Neighbor cutoff | 5.0-7.0 Å |
| `l_max` | Angular channels | 1 (fast) to 3 (accurate) |
| `num_layers` | Network depth | 3-4 |
| `batch_size` | Training batch | 2-16 (GPU memory) |
| `max_epochs` | Training limit | 100-1000 |
| `learning_rate` | Optimizer step | 1e-3 to 5e-3 |

### PyTorch 2.11 acceleration & `PhysMuon` (optional)

The example disables compilation for structure-only models. To enable
`PhysMuon` and/or compilation:

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

**Note:** Magnetic models benefit more from compilation. Structure-only speedup is modest.

## Step 3: Train the Model

```bash
cd /path/to/SpinGNN2

# Train via source launcher (ensures checkout packages are used)
OMP_NUM_THREADS=1 python tools/ci/source_python.py spingnn-train config/my_structure.yaml
```

### Monitor training

Check the output directory:

```bash
TRAIN_DIR=results/structure/my_structure_run
ls -lh "$TRAIN_DIR"
```

Expected files:
- `config.yaml` - resolved config
- `metrics_epoch.csv` - per-epoch train/val losses
- `best_model.pth` - checkpoint with lowest validation loss
- `last_model.pth` - most recent checkpoint

### Verify convergence

```bash
python - <<'PY'
import csv
import pathlib

metrics = pathlib.Path("results/structure/my_structure_run/metrics_epoch.csv")
with metrics.open() as f:
    rows = list(csv.DictReader(f))

print(f"Total epochs: {len(rows)}")
print(f"Final training loss: {rows[-1]['training_loss']}")
print(f"Final validation loss: {rows[-1]['validation_loss']}")

# Check force MAE
force_mae = [float(r['validation_forces_mae']) for r in rows if 'validation_forces_mae' in r]
print(f"Best validation force MAE: {min(force_mae):.6f} eV/Å")
PY
```

Typical good values:
- Force MAE < 0.05 eV/Å (excellent)
- Force MAE < 0.1 eV/Å (good)
- Force MAE > 0.2 eV/Å (check data quality or increase epochs)

## Step 4: Deploy the Model

### TorchScript deployment (required for structure-only)

Structure-only models use the `nequip` or `allegro` pair styles, which require TorchScript `.pth` format:

```bash
TRAIN_DIR=results/structure/my_structure_run
mkdir -p deployed

spingnn-deploy build \
  --train-dir "$TRAIN_DIR" \
  deployed/structure.pth

# Verify deployment
spingnn-deploy info deployed/structure.pth
```

Expected output:
```
r_max: 6.0
type_names: Fe O
model_type: Allegro
```

### Validate deployment in Python

```bash
python - <<'PY'
import torch
from nequip.scripts.deploy import load_deployed_model

model, metadata = load_deployed_model(
    "deployed/structure.pth",
    device="cpu",
    freeze=True,
    set_global_options=True
)

print("DEPLOY_OK")
print(f"  r_max: {metadata['r_max']}")
print(f"  type_names: {metadata['type_names']}")
print(f"  Model type: {type(model)}")
assert isinstance(model, torch.jit.ScriptModule)
PY
```

## Step 5: Run LAMMPS Simulation

### Prepare LAMMPS input

Create `in.structure_nvt`:

```lammps
# Structure-only Allegro NVT MD
units       metal
boundary    p p p
atom_style  atomic
atom_modify map array
newton      off  # Required for nequip/allegro

read_data   system.data

# Allegro pair style
pair_style  allegro
pair_coeff  * * deployed/structure.pth Fe O

# Velocities and thermostat
velocity    all create 300.0 12345 mom yes rot no dist gaussian
fix         nvt all nvt temp 300.0 300.0 0.1
timestep    0.0005  # 0.5 fs

# Output
thermo      100
thermo_style custom step temp pe ke etotal press fmax
dump        traj all custom 100 output.lammpstrj id type x y z fx fy fz
dump_modify traj sort id

run         1000
```

### Run simulation

```bash
# Set LAMMPS binary
LMP=/path/to/lmp

# Run from repo root or LAMMPS example dir
cd /path/to/SpinGNN

$LMP -in in.structure_nvt
```

### Check output

```bash
# Verify completion
grep "Total wall time" log.lammps
grep "Loop time" log.lammps

# Check final energy (should be finite)
tail -5 log.lammps | grep -E "^[0-9]"
```

Expected: finite energies, no "Lost atoms", stable trajectory.

## Step 6: Validation and Troubleshooting

### Python-LAMMPS parity check

Compare LAMMPS output with Python inference on the same structure:

```bash
python - <<'PY'
import numpy as np
from ase.io import read
from nequip.scripts.deploy import load_deployed_model
from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper

# Load structure from LAMMPS dump (first frame)
atoms = read("output.lammpstrj", index=0, format="lammps-dump-text")

# Load model
model, metadata = load_deployed_model(
    "deployed/structure.pth",
    device="cpu",
    freeze=True,
    set_global_options=True
)

# Prepare input
type_map = {sym: idx for sym, idx in eval(metadata["chemical_symbol_to_type"])}
mapper = TypeMapper(chemical_symbol_to_type=type_map)
data = AtomicData.from_ase(atoms, r_max=float(metadata["r_max"]))
batch = mapper(data).to_dict()

# Inference (autograd is required for forces even in eval mode; do NOT use torch.no_grad())
output = model(batch)

python_energy = float(output[AtomicDataDict.TOTAL_ENERGY_KEY].detach())
python_forces = output[AtomicDataDict.FORCE_KEY].detach().numpy()

# Compare with LAMMPS (extract from log.lammps)
# This is a manual check - automate with your workflow

print(f"Python energy: {python_energy:.6f} eV")
print(f"Python force max: {np.abs(python_forces).max():.6f} eV/Å")
print("Compare with LAMMPS log output")
PY
```

Typical tolerances:
- Energy: < 1e-4 eV (CPU) or < 1e-3 eV (GPU float32)
- Force: < 1e-3 eV/Å (CPU) or < 5e-3 eV/Å (GPU)

### Common issues

| Symptom | Cause | Solution |
| --- | --- | --- |
| `Pair style allegro requires metal units` | Wrong unit system | Add `units metal` before `pair_style` |
| `Newton pair must be off` | Newton flag wrong | Use `newton off` for Allegro/NequIP |
| Model species mismatch | Wrong element order | Match `pair_coeff` names to deployed `type_names` |
| Non-finite forces | Bad structure | Check for overlapping atoms, increase cutoff |
| Very slow training | Too many neighbors | Reduce `r_max` or increase `batch_size` |
| High force MAE | Underfitting | Increase `l_max`, `num_layers`, or `max_epochs` |

## Testing and CI

Run structure-only tests:

```bash
cd /path/to/SpinGNN2

# CPU unit tests (includes structure-only model building)
tools/ci/cpu_unit_tests.sh

# Validate example config parsing
python tools/ci/source_python.py example/validate_configs.py
```

## SpinGNN2 Scope Note

SpinGNN2 (`spingnn2-export`, `libspingnn2`, and modern LAMMPS
`pair_style spingnn2(/kk)`) currently exports the local SpinGNN++ families:
`Allegro_nosoc`, `Allegro_soc`, `Allegro_born` (DREAM), `Allegro_e2IP`, and
`Allegro_soc_e2IP`. Plain structure-only `Allegro`
(`example/01_structure_only.yaml`, without DREAM BEC or e2SLP heads) and
Multi-layer non-local models remain outside the 1-hop native engine scope
above. If you add an e2SLP force-uncertainty head (`Allegro_e2IP`,
`example/06_allegro_e2ip.yaml`) or a DREAM BEC head (`Allegro_born`,
`example/04_born_effective_charge.yaml`), you can deploy directly with
SpinGNN2—see [`docs/SPINGNN2_GUIDE.md`](../../docs/SPINGNN2_GUIDE.md) and
[`skills/spingnn2/SKILL.md`](../spingnn2/SKILL.md).

## See Also

- `example/01_structure_only.yaml` - Full config template
- `example/06_allegro_e2ip.yaml` - Native-exportable `Allegro_e2IP` structure+uncertainty template
- `example/TUTORIAL_TRAINING.md` - Training optimization guide
- `docs/SPINGNN2_GUIDE.md` - Native engine & modern LAMMPS guide
- `lammps_spingnn2/README.md` - Modern LAMMPS overlay documentation
- `skills/spingnn-usage/SKILL.md` - Complete reference guide
- `tools/ci/README.md` - CI testing contracts
