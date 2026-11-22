---
name: spin-nosoc-e2e
description: End-to-end workflow for spin Allegro without SOC - from magnetic dataset preparation to spin-lattice MD. Covers energy/force/spin_force training and LAMMPS spin integrators.
---

# Spin Allegro (No SOC): End-to-End Workflow

Complete pipeline for training and deploying a magnetic Allegro potential without spin-orbit coupling. For non-collinear magnets where spin directions matter but SOC is negligible.

## When to Use

Use this skill when you need to:
- Train a magnetic potential for Heisenberg-like systems
- Work with vector spin degrees of freedom (spin directions, `spin_forces`)
- Model systems where SOC can be neglected (e.g., 3d transition metals far from heavy elements)
- Run spin-lattice dynamics with LAMMPS spin integrators

**Trigger phrases:** "spin nosoc", "magnetic without SOC", "Heisenberg", "非SOC自旋模型", "spin directions"

**NOT for:** Systems with strong SOC (use `skills/spin-soc-e2e` instead), structure-only (use `skills/structure-only-e2e`).

## Prerequisites

- SpinGNN repository with validated H100/4090 environment
- Dataset with `energy`, `forces`, `spin_forces`, and vector `initial_magmoms`
- LAMMPS built with `SPIN` package and `pair_style spin/allegro_spin`

## Key Differences from Structure-Only

| Feature | Structure-only | Spin (no SOC) |
| --- | --- | --- |
| Input | positions only | positions + spin directions |
| Supervision | energy, forces, stress | + spin_forces |
| Symmetry | E(3) equivariant | T-even time-reversal |
| Training time | baseline | ~1.5× (more channels) |
| LAMMPS pair style | `allegro` | `spin/allegro_spin` |
| MD integrators | standard NVE/NVT | `fix nve/spin`, `fix langevin/spin` |

## Quick Reference

| Stage | Time | Critical parameters |
| --- | --- | --- |
| Data prep | ~2h per 1k frames | Vector magmoms, spin_forces |
| Training | 2-48h | `spin: true`, `soc: false` |
| Deployment | ~1min | Same as structure-only |
| LAMMPS spin MD | minutes-days | Spin integrator + pair style |

## Step 1: Prepare Magnetic Dataset

### Dataset schema requirements

Each frame in the extxyz file must have:

```python
# Structure (same as structure-only)
atoms.positions        # (N, 3) array
atoms.info["energy"]   # scalar, total energy in eV

# Forces (same)
atoms.arrays["forces"] # (N, 3) array, eV/Å

# Magnetic additions
atoms.set_initial_magnetic_moments(spins)  # (N, 3) VECTOR magmoms
atoms.arrays["spin_forces"]                 # (N, 3) spin torques, eV
```

**Critical:** `initial_magmoms` must be 3D vectors (shape N×3), not scalar magnitudes. ASE stores this as `atoms.get_initial_magnetic_moments()`.

### From VASP with ISPIN=2 and non-collinear

```bash
# Extract magnetic moments and spin forces from VASP OUTCAR/vasprun.xml
# (workflow-specific; adjust for your DFT package)

python - <<'PY'
from ase.io import read, write
import numpy as np

# Read VASP output
frames = read("vasprun.xml", index=":")

for atoms in frames:
    # VASP stores scalar |m| by default; convert to vector if needed
    # For truly non-collinear VASP, moments are already vectors
    magmoms = atoms.get_magnetic_moments()
    
    if magmoms.ndim == 1:
        # Scalar moments: promote to z-aligned vectors
        magmoms = np.column_stack([
            np.zeros(len(atoms)),
            np.zeros(len(atoms)),
            magmoms
        ])
    
    atoms.set_initial_magnetic_moments(magmoms)
    
    # Extract spin forces from VASP (if available)
    # Format depends on VASP version and LORBIT setting
    # This is a placeholder - adapt to your parser
    spin_forces = ... # shape (N, 3)
    atoms.set_array("spin_forces", spin_forces)

write("data/spin_nosoc.extxyz", frames, format="extxyz")
PY
```

### Validate spin dataset

```bash
cd /path/to/SpinGNN

python - <<'PY'
import numpy as np
from ase.io import read

frames = read("data/spin_nosoc.extxyz", index=":")
print(f"Total frames: {len(frames)}")

for idx in [0, -1]:
    atoms = frames[idx]
    
    # Check structure fields
    assert np.isfinite(atoms.positions).all()
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()
    
    # Check spin fields
    magmoms = atoms.get_initial_magnetic_moments()
    assert magmoms.ndim == 2 and magmoms.shape == (len(atoms), 3), \
        f"Frame {idx}: magmoms must be (N, 3) array"
    assert np.isfinite(magmoms).all()
    
    spin_forces = atoms.get_array("spin_forces")
    assert spin_forces.shape == (len(atoms), 3), \
        f"Frame {idx}: spin_forces must be (N, 3) array"
    assert np.isfinite(spin_forces).all()
    
    # Check spin magnitude consistency (optional)
    spin_norms = np.linalg.norm(magmoms, axis=1)
    print(f"Frame {idx}: {len(atoms)} atoms, spin magnitudes: "
          f"min={spin_norms.min():.3f}, max={spin_norms.max():.3f}")

print("SPIN_DATASET_OK")
PY
```

## Step 2: Configure Training

Copy the spin-nosoc template:

```bash
cp example/02_spin_nosoc.yaml config/my_spin_nosoc.yaml
```

### Required spin-specific settings

```yaml
# Enable spin mode WITHOUT SOC
spin: true
soc: false

# Data fields
dataset_file_name: data/spin_nosoc.extxyz
chemical_symbol_to_type:
  Mn: 0
  Ge: 1

# Model geometry
r_max: 5.0
l_max: 2

# Loss: add spin_forces
loss_coeffs:
  total_energy:
    - 1.0
    - PerAtomMSELoss
  forces: 1.0
  spin_forces: 1.0  # <-- spin torque supervision

# Validation metrics
metrics_components:
  - [forces, mae]
  - [spin_forces, mae]  # <-- monitor spin torque MAE

# Output
root: results/spin_nosoc
run_name: my_spin_nosoc_run
```

### Key differences from structure-only config

1. **`spin: true`** — enables spin channels in the model
2. **`soc: false`** — uses T-even (time-reversal symmetric) equivariance
3. **`loss_coeffs.spin_forces`** — supervision on spin torques
4. **No `allow_tf32: true`** — spin models use default `float32_matmul_precision: highest` for numerical stability

### Training acceleration (optional)

Spin models benefit significantly from PyTorch 2.11 compilation:

```yaml
spingnn_compile:
  enabled: true
  scope: full_fx     # or energy_core for conservative start
  dynamic: true

# TF32 for H100 (calibrated gates passed)
float32_matmul_precision: high
allow_tf32: true
```

Expected speedup on H100 fixture-scale:
- NequIP-spin: ~6× (compile) + ~1.2× (TF32)
- Allegro-spin-nosoc: ~2.6× (compile + TF32)

## Step 3: Train the Model

```bash
cd /path/to/SpinGNN

export PYTHONPATH="$PWD/src/core:$PWD/src/models"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

nequip-train config/my_spin_nosoc.yaml
```

### Monitor convergence

Check both force and spin_force MAE:

```bash
TRAIN_DIR=results/spin_nosoc/my_spin_nosoc_run

python - <<'PY'
import csv
import pathlib

metrics = pathlib.Path("results/spin_nosoc/my_spin_nosoc_run/metrics_epoch.csv")
with metrics.open() as f:
    rows = list(csv.DictReader(f))

print(f"Total epochs: {len(rows)}")
print(f"Final validation force MAE: {rows[-1].get('validation_forces_mae', 'N/A')} eV/Å")
print(f"Final validation spin_force MAE: {rows[-1].get('validation_spin_forces_mae', 'N/A')} eV")

# Best values
force_mae = [float(r['validation_forces_mae']) for r in rows if 'validation_forces_mae' in r]
spin_mae = [float(r['validation_spin_forces_mae']) for r in rows if 'validation_spin_forces_mae' in r]

print(f"Best force MAE: {min(force_mae):.6f} eV/Å")
print(f"Best spin_force MAE: {min(spin_mae):.6f} eV")
PY
```

Typical good values for spin-nosoc:
- Force MAE < 0.1 eV/Å
- Spin force MAE < 0.05 eV (system-dependent; collinear systems converge faster)

## Step 4: Deploy the Model

Deployment is identical to structure-only (TorchScript `.pth`):

```bash
TRAIN_DIR=results/spin_nosoc/my_spin_nosoc_run
mkdir -p deployed

nequip-deploy build \
  --train-dir "$TRAIN_DIR" \
  deployed/spin_nosoc.pth

nequip-deploy info deployed/spin_nosoc.pth
```

Expected metadata:
```
spin: true
soc: false
r_max: 5.0
type_names: Mn Ge
```

### Validate in Python

```bash
python - <<'PY'
import torch
import numpy as np
from nequip.scripts.deploy import load_deployed_model
from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper
from ase.io import read

# Load model
model, metadata = load_deployed_model(
    "deployed/spin_nosoc.pth",
    device="cpu",
    freeze=True,
    set_global_options=True
)

# Check spin mode
assert metadata["spin"] == "true", "Model must have spin=true"
assert metadata["soc"] == "false", "Model must have soc=false for nosoc"

# Load a test structure with spins
atoms = read("data/spin_nosoc.extxyz", index=0)
type_map = {"Mn": 0, "Ge": 1}  # adjust to your system
mapper = TypeMapper(chemical_symbol_to_type=type_map)

data = AtomicData.from_ase(atoms, r_max=float(metadata["r_max"]))
batch = mapper(data).to_dict()

# Inference
with torch.no_grad():
    output = model(batch)

# Check outputs
assert AtomicDataDict.FORCE_KEY in output
assert AtomicDataDict.SPIN_FORCE_KEY in output

forces = output[AtomicDataDict.FORCE_KEY].numpy()
spin_forces = output[AtomicDataDict.SPIN_FORCE_KEY].numpy()

print(f"DEPLOY_OK spin_nosoc")
print(f"  Forces shape: {forces.shape}")
print(f"  Spin forces shape: {spin_forces.shape}")
print(f"  Max force: {np.abs(forces).max():.6f} eV/Å")
print(f"  Max spin force: {np.abs(spin_forces).max():.6f} eV")
PY
```

## Step 5: LAMMPS Spin-Lattice MD

### Prepare LAMMPS input with spin integrator

Create `in.spin_nosoc_nvt`:

```lammps
# Spin-lattice NVT with Allegro-spin (no SOC)
units       metal
boundary    p p p
atom_style  spin
atom_modify map array
newton      off

read_data   spin_system.data

# Allegro spin pair style
pair_style  spin/allegro_spin
pair_coeff  * * deployed/spin_nosoc.pth Mn Ge

# Spin integrator (lattice + spin damped dynamics)
fix         spin_damp all nve/spin lattice yes
fix_modify  spin_damp alpha 0.1  # spin damping parameter

# Thermostat for lattice
fix         thermo all langevin 300.0 300.0 0.1 48279

timestep    0.0005  # 0.5 fs

# Output
compute     spinprops all property/atom spx spy spz sp fmx fmy fmz
thermo      100
thermo_style custom step temp pe ke etotal press fmax
dump        traj all custom 100 spin_traj.lammpstrj &
            id type x y z fx fy fz &
            c_spinprops[1] c_spinprops[2] c_spinprops[3] c_spinprops[4] &
            c_spinprops[5] c_spinprops[6] c_spinprops[7]
dump_modify traj sort id

run         10000
```

### LAMMPS data file format for spins

`spin_system.data` must use `atom_style spin`. Example header:

```
Spin system for Allegro-spin nosoc

20 atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 54.938  # Mn
2 72.64   # Ge

Atoms # spin

# id type magnitude x y z spx spy spz
1 1 2.0 0.0 0.0 0.0  0.0 0.0 1.0
2 2 0.0 2.5 0.0 0.0  0.0 0.0 0.0
...
```

**Note:** `magnitude` column is for LAMMPS internal use. The actual spin direction is `(spx, spy, spz)`.

### Run simulation

```bash
LMP=/path/to/lmp
cd /path/to/SpinGNN

$LMP -in in.spin_nosoc_nvt
```

### Validate spin dynamics

Check that spin magnitudes are conserved (for Heisenberg-like models):

```bash
python - <<'PY'
import numpy as np
from ase.io import read

traj = read("spin_traj.lammpstrj", index=":", format="lammps-dump-text")

# Extract spin components (assumes dump includes c_spinprops[1:3])
# Adjust column indices if your dump format differs
for idx in [0, len(traj)//2, -1]:
    atoms = traj[idx]
    # This requires custom LAMMPS dump parsing for spin components
    # Placeholder logic:
    print(f"Frame {idx}: check spin magnitudes manually from dump")

# Expected: |S| remains ~constant for each magnetic atom
# Non-magnetic atoms (e.g., Ge) should have |S| ~ 0
PY
```

## Step 6: Validation and Troubleshooting

### Python-LAMMPS parity for spins

```bash
python - <<'PY'
import numpy as np
from ase.io import read
from nequip.scripts.deploy import load_deployed_model
from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper

# Load structure from LAMMPS dump (first frame with spins)
atoms = read("spin_traj.lammpstrj", index=0, format="lammps-dump-text")

# Load model
model, metadata = load_deployed_model(
    "deployed/spin_nosoc.pth",
    device="cpu",
    freeze=True,
    set_global_options=True
)

type_map = {"Mn": 0, "Ge": 1}
mapper = TypeMapper(chemical_symbol_to_type=type_map)
data = AtomicData.from_ase(atoms, r_max=float(metadata["r_max"]))
batch = mapper(data).to_dict()

with torch.no_grad():
    output = model(batch)

python_energy = float(output[AtomicDataDict.TOTAL_ENERGY_KEY])
python_forces = output[AtomicDataDict.FORCE_KEY].numpy()
python_spin_forces = output[AtomicDataDict.SPIN_FORCE_KEY].numpy()

print(f"Python energy: {python_energy:.6f} eV")
print(f"Python force max: {np.abs(python_forces).max():.6f} eV/Å")
print(f"Python spin_force max: {np.abs(python_spin_forces).max():.6f} eV")
print("Compare with LAMMPS thermo output")
PY
```

Parity tolerances (CPU serial):
- Energy: < 2e-4 eV
- Force: < 1e-3 eV/Å
- Spin force: < 1e-3 eV

### Common issues

| Symptom | Cause | Solution |
| --- | --- | --- |
| `atom_style spin required` | Wrong atom style | Use `atom_style spin` in LAMMPS input |
| Spin magnitudes drift | Wrong integrator | Use `fix nve/spin` or `fix langevin/spin`, not plain NVE |
| High spin_force MAE | Insufficient training | Increase `max_epochs` or check spin data quality |
| Model outputs zero spin_forces | Config error | Verify `spin: true` and `loss_coeffs.spin_forces: 1.0` |
| LAMMPS crashes with "spin force undefined" | Pair style mismatch | Use `spin/allegro_spin`, not plain `allegro` |

## Testing

Run spin-nosoc tests in CI:

```bash
cd /path/to/SpinGNN

# CPU unit tests (includes spin model building)
tools/ci/cpu_unit_tests.sh

# LAMMPS spin parity tests (requires serial binary)
export SPINGNN_LAMMPS_BIN=/path/to/lmp
python -m pytest tests_lammps -m "lammps and not mpi" -k spin

# End-to-end spin training -> LAMMPS (slow, marked)
python -m pytest tests_lammps/test_e2e_train_lammps.py -k nosoc
```

## See Also

- `example/02_spin_nosoc.yaml` - Full config template
- `example/TUTORIAL_TRAINING.md` - Training optimization
- `lammps/examples/spingnn/in.spin_lattice_md` - LAMMPS spin MD example
- `skills/spin-soc-e2e/SKILL.md` - For systems with SOC
- `tests_lammps/test_pair_parity_run0.py` - Parity regression tests
- `tools/ci/gpu_full_tests.sh` - GPU compile+precision gates
