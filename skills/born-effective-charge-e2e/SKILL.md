---
name: born-effective-charge-e2e
description: End-to-end workflow for training models with Born effective charges - from BEC dataset preparation to electric field coupling in LAMMPS. Covers ferroelectric and piezoelectric materials.
---

# Born Effective Charge (BEC) Training: End-to-End Workflow

Complete pipeline for training Allegro models with Born effective charge (BEC) tensors and deploying them for ferroelectric and piezoelectric simulations in LAMMPS.

## When to Use

Use this skill when you need to:
- Train models for ferroelectric materials (BaTiO₃, PbTiO₃, etc.)
- Predict Born effective charge tensors Z* from structure
- Model polarization, piezoelectric response, or electric field coupling
- Run LAMMPS simulations with electric fields acting on BEC-equipped atoms

**Trigger phrases:** "Born effective charge", "BEC training", "ferroelectric", "电场耦合", "piezoelectric", "Z* tensor"

**NOT for:** Pure magnetic systems (use spin skills), standard structure-only without BEC.

## Prerequisites

- SpinGNN repository with validated environment
- Dataset with `energy`, `forces`, and **`born_effective_charges`** (N×3×3 tensors)
- BEC data from DFPT or finite-difference DFT
- LAMMPS built with `SPIN` package and BEC-aware pair styles

## Physics Background

### What are Born effective charges?

Born effective charges Z*ᵢ,αβ relate atomic displacements to induced polarization:

```
ΔPα = (1/Ω) Σᵢ Z*ᵢ,αβ Δuᵢ,β
```

where:
- `α, β` are Cartesian directions (x, y, z)
- `Ω` is cell volume
- `Δuᵢ,β` is displacement of atom i in direction β
- `ΔPα` is change in polarization in direction α

**Key properties:**
- Z* is a 3×3 tensor per atom
- Acoustic sum rule: Σᵢ Z*ᵢ,αβ = 0 for each (α,β)
- In centrosymmetric crystals, Z* ≈ Z (nominal charge)
- In ferroelectrics, Z* can be much larger than Z (e.g., Z*_Ti ≈ +7 in BaTiO₃)

### SpinGNN BEC implementation

- **Output:** Per-atom 3×3 tensors (stored flattened as N×9 in datasets)
- **Loss:** MSE on all 9 components per atom
- **LAMMPS coupling:** Electric field E induces forces F = Z* · E on each atom
- **Compatibility:** BEC can be combined with structure-only OR spin models (with/without SOC)

## Quick Reference

| Stage | Time | Critical parameters |
| --- | --- | --- |
| BEC DFPT data | hours-days per config | LEPSILON, BORN_NWRITE |
| Training | 1-48h | `born_charges: true` |
| Deployment | ~1min | Same TorchScript `.pth` |
| LAMMPS E-field MD | minutes-hours | `fix efield`, BEC pair style |

## Step 1: Prepare BEC Dataset

### Dataset schema requirements

Each frame must have:

```python
# Structure (standard)
atoms.positions                  # (N, 3)
atoms.info["energy"]             # scalar, eV
atoms.arrays["forces"]           # (N, 3), eV/Å

# BEC tensors (NEW)
atoms.arrays["born_effective_charges"]  # (N, 9), flattened 3×3 tensors
```

**Storage format:** Each atom's 3×3 Z* tensor is flattened row-major:
```
Z* = [Z_xx, Z_xy, Z_xz,
      Z_yx, Z_yy, Z_yz,
      Z_zx, Z_zy, Z_zz]
```

So `born_effective_charges[i, :]` = `[Z_xx, Z_xy, Z_xz, Z_yx, Z_yy, Z_yz, Z_zx, Z_zy, Z_zz]` for atom i.

### From VASP DFPT

```bash
# VASP INCAR for BEC calculation:
# LEPSILON = .TRUE.
# IBRION = 8 or 7 (DFPT)
# (standard convergence flags: EDIFF, NSW, etc.)

python - <<'PY'
from ase.io import read, write
import numpy as np

# Parse VASP output with BEC (OUTCAR contains Born charges)
def parse_bec_from_outcar(outcar_path):
    """Extract BEC tensors from VASP OUTCAR.
    
    Returns (N, 3, 3) array of Born effective charges.
    """
    with open(outcar_path) as f:
        lines = f.readlines()
    
    bec_tensors = []
    in_bec_block = False
    
    for line in lines:
        if "BORN EFFECTIVE CHARGES" in line:
            in_bec_block = True
            bec_tensors = []
            continue
        
        if in_bec_block:
            if line.strip().startswith("ion"):
                # Next 3 lines contain the 3×3 tensor
                tensor = []
                continue
            
            parts = line.split()
            if len(parts) == 3:
                try:
                    row = [float(x) for x in parts]
                    tensor.append(row)
                    if len(tensor) == 3:
                        bec_tensors.append(tensor)
                        tensor = []
                except ValueError:
                    pass
            
            if "BORN EFFECTIVE CHARGES" in line and len(bec_tensors) > 0:
                # End of block
                break
    
    return np.array(bec_tensors)  # shape (N, 3, 3)

# Load structure
atoms = read("POSCAR")
atoms.info["energy"] = ...  # Extract from OSZICAR or vasprun.xml
atoms.set_array("forces", ...)  # From OUTCAR

# Extract BEC
bec = parse_bec_from_outcar("OUTCAR")  # (N, 3, 3)
assert bec.shape == (len(atoms), 3, 3)

# Flatten to (N, 9) for SpinGNN
bec_flat = bec.reshape(len(atoms), 9)
atoms.set_array("born_effective_charges", bec_flat)

write("data/bec_frame.extxyz", atoms, format="extxyz")
PY
```

**Note:** The above parser is a simplified example. Production code should use `pymatgen` or `vise` for robust VASP BEC parsing.

### Validate BEC dataset

```bash
cd /path/to/SpinGNN

python - <<'PY'
import numpy as np
from ase.io import read

frames = read("data/bec_dataset.extxyz", index=":")
print(f"Total BEC frames: {len(frames)}")

for idx in [0, -1]:
    atoms = frames[idx]
    
    # Standard fields
    assert np.isfinite(atoms.positions).all()
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()
    
    # BEC field
    bec = atoms.get_array("born_effective_charges")
    assert bec.shape == (len(atoms), 9), \
        f"Frame {idx}: born_effective_charges must be (N, 9)"
    assert np.isfinite(bec).all(), f"Frame {idx}: non-finite BEC"
    
    # Reshape to (N, 3, 3) for checks
    bec_tensors = bec.reshape(len(atoms), 3, 3)
    
    # Acoustic sum rule check (should be ~0 for each component)
    bec_sum = bec_tensors.sum(axis=0)  # (3, 3)
    sum_violation = np.abs(bec_sum).max()
    print(f"Frame {idx}: {len(atoms)} atoms, BEC sum rule violation: {sum_violation:.6f}")
    
    if sum_violation > 0.1:
        print(f"  WARNING: Large acoustic sum rule violation (should be << 0.1)")
    
    # Check magnitudes (sanity: |Z*| typically < 10 for most materials)
    bec_norms = np.linalg.norm(bec_tensors, axis=(1, 2))
    print(f"  BEC tensor norms: min={bec_norms.min():.3f}, max={bec_norms.max():.3f}")

print("BEC_DATASET_OK")
PY
```

**Expected:**
- Acoustic sum rule satisfied to ~1e-3 (VASP DFPT is usually well-converged)
- BEC magnitudes: 0.5 to 10 (typical range; can be higher in strongly ferroelectric materials)

## Step 2: Configure BEC Training

Copy the BEC template:

```bash
cp example/04_born_effective_charge.yaml config/my_bec.yaml
```

### Required BEC settings

```yaml
# Enable BEC output
born_charges: true

# Data
dataset_file_name: data/bec_dataset.extxyz
chemical_symbol_to_type:
  Ba: 0
  Ti: 1
  O: 2

# Model geometry
r_max: 6.0
l_max: 2

# Loss: add born_effective_charges
loss_coeffs:
  total_energy:
    - 1.0
    - PerAtomMSELoss
  forces: 1.0
  born_effective_charges: 1.0  # <-- BEC loss

# Validation metrics
metrics_components:
  - [forces, mae]
  - [born_effective_charges, mae]  # <-- Monitor BEC MAE

# Output
root: results/bec
run_name: my_bec_run
```

### Combining BEC with spin

BEC is compatible with spin models. To train BEC + spin (e.g., for magnetic ferroelectrics like BiFeO₃):

```yaml
# Enable both spin and BEC
spin: true
soc: false  # or true, depending on system
born_charges: true

# Loss includes all fields
loss_coeffs:
  total_energy: [1.0, PerAtomMSELoss]
  forces: 1.0
  spin_forces: 1.0
  born_effective_charges: 1.0

# Dataset must have: forces, spin_forces, born_effective_charges
```

### Training acceleration

```yaml
spingnn_compile:
  enabled: true
  scope: full_fx
  dynamic: true

float32_matmul_precision: high
allow_tf32: true
```

BEC models compile successfully on the same terms as spin models.

## Step 3: Train the BEC Model

```bash
cd /path/to/SpinGNN

export PYTHONPATH="$PWD/src/core:$PWD/src/models"
export OMP_NUM_THREADS=1

nequip-train config/my_bec.yaml
```

### Monitor BEC convergence

```bash
TRAIN_DIR=results/bec/my_bec_run

python - <<'PY'
import csv
import pathlib

metrics = pathlib.Path("results/bec/my_bec_run/metrics_epoch.csv")
with metrics.open() as f:
    rows = list(csv.DictReader(f))

print(f"Total epochs: {len(rows)}")
final = rows[-1]
print(f"Final validation force MAE: {final.get('validation_forces_mae', 'N/A')} eV/Å")
print(f"Final validation BEC MAE: {final.get('validation_born_effective_charges_mae', 'N/A')} e")

force_mae = [float(r['validation_forces_mae']) for r in rows if 'validation_forces_mae' in r]
bec_mae = [float(r['validation_born_effective_charges_mae']) for r in rows 
           if 'validation_born_effective_charges_mae' in r]

print(f"Best force MAE: {min(force_mae):.6f} eV/Å")
print(f"Best BEC MAE: {min(bec_mae):.6f} e")
PY
```

Typical convergence for BEC:
- Force MAE < 0.1 eV/Å (same as other models)
- BEC MAE < 0.1 e per component (good); < 0.05 e (excellent)

**Note:** BEC MAE is expressed as a multiple of the proton charge e.

## Step 4: Deploy the BEC Model

Deployment is identical to other model types:

```bash
TRAIN_DIR=results/bec/my_bec_run
mkdir -p deployed

nequip-deploy build \
  --train-dir "$TRAIN_DIR" \
  deployed/bec_model.pth

nequip-deploy info deployed/bec_model.pth
```

Expected metadata:
```
born_charges: true
r_max: 6.0
type_names: Ba Ti O
```

### Validate BEC output in Python

```bash
python - <<'PY'
import torch
import numpy as np
from nequip.scripts.deploy import load_deployed_model
from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper
from ase.io import read

model, metadata = load_deployed_model(
    "deployed/bec_model.pth",
    device="cpu",
    freeze=True,
    set_global_options=True
)

assert metadata["born_charges"] == "true", "Model must have born_charges=true"

# Load test structure
atoms = read("data/bec_dataset.extxyz", index=0)
type_map = {"Ba": 0, "Ti": 1, "O": 2}
mapper = TypeMapper(chemical_symbol_to_type=type_map)

data = AtomicData.from_ase(atoms, r_max=float(metadata["r_max"]))
batch = mapper(data).to_dict()

# Inference
with torch.no_grad():
    output = model(batch)

# Check BEC output
assert AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY in output
bec = output[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY].numpy()
assert bec.shape == (len(atoms), 9), f"BEC shape should be (N, 9), got {bec.shape}"

# Reshape to (N, 3, 3) for readability
bec_tensors = bec.reshape(len(atoms), 3, 3)
print(f"DEPLOY_OK born_charges")
print(f"  BEC shape: {bec_tensors.shape}")
print(f"  Example BEC[0] (Ba):\n{bec_tensors[0]}")

# Check acoustic sum rule
bec_sum = bec_tensors.sum(axis=0)
print(f"  Acoustic sum rule violation: {np.abs(bec_sum).max():.6f} e")
PY
```

## Step 5: LAMMPS Simulation with Electric Fields

### LAMMPS pair style for BEC

SpinGNN LAMMPS supports BEC through the modified `pair_style allegro` (or `spin/allegro_spin` for spin+BEC):

```lammps
# Structure-only + BEC
pair_style  allegro
pair_coeff  * * deployed/bec_model.pth Ba Ti O

# Spin + BEC
pair_style  spin/allegro_spin
pair_coeff  * * deployed/bec_model.pth Ba Ti O
```

The pair style automatically:
1. Predicts BEC tensors for each atom
2. Stores them in per-atom arrays accessible via `compute`
3. Couples BEC to electric fields applied via `fix efield`

### LAMMPS input with electric field

Create `in.bec_efield`:

```lammps
# BaTiO3 with electric field coupling via BEC
units       metal
boundary    p p p
atom_style  atomic  # or spin, if combining with magnetic model
atom_modify map array
newton      off

read_data   batio3.data

# Allegro with BEC
pair_style  allegro
pair_coeff  * * deployed/bec_model.pth Ba Ti O

# Apply electric field in z direction (10 MV/m = 0.0001 eV/(Å·e))
# Conversion: 1 V/Å = 10^10 V/m
# For 10 MV/m: E_z = 0.001 V/Å = 0.001 eV/(Å·e)
fix         efield all efield 0.0 0.0 0.001

# NVT ensemble
velocity    all create 300.0 12345
fix         nvt all nvt temp 300.0 300.0 0.1
timestep    0.0005

# Output BEC and induced forces
compute     bec_out all property/atom bec_xx bec_xy bec_xz bec_yx bec_yy bec_yz bec_zx bec_zy bec_zz

thermo      100
thermo_style custom step temp pe ke etotal press
dump        traj all custom 100 bec_traj.lammpstrj &
            id type x y z fx fy fz &
            c_bec_out[1] c_bec_out[2] c_bec_out[3] &
            c_bec_out[4] c_bec_out[5] c_bec_out[6] &
            c_bec_out[7] c_bec_out[8] c_bec_out[9]
dump_modify traj sort id

run         1000
```

**Key points:**
- `fix efield` applies uniform E field: E = (Ex, Ey, Ez) in eV/(Å·e)
- Forces on atoms: F_i,α = Σ_β Z*_i,αβ E_β
- BEC tensors output via `compute property/atom bec_xx bec_xy ...`

### Run BEC+E-field simulation

```bash
LMP=/path/to/lmp
cd /path/to/SpinGNN

$LMP -in in.bec_efield
```

### Verify polarization response

```bash
python - <<'PY'
import numpy as np
from ase.io import read

# Load trajectory
traj = read("bec_traj.lammpstrj", index=":", format="lammps-dump-text")

# Calculate polarization from atomic displacements and BEC
# P_α = (1/V) Σ_i Z*_i,αβ u_i,β

ref = traj[0]
final = traj[-1]

displacements = final.positions - ref.positions
# (apply MIC if needed for periodic boundaries)

# Extract BEC from dump (requires parsing custom columns)
# Placeholder:
bec_tensors = ...  # shape (N, 3, 3)

volume = final.get_volume()
polarization = np.zeros(3)

for i in range(len(final)):
    for alpha in range(3):
        for beta in range(3):
            polarization[alpha] += bec_tensors[i, alpha, beta] * displacements[i, beta]

polarization /= volume  # Units: e/Å² = C/m² * 1.602e-19

print(f"Induced polarization (e/Å²): {polarization}")
print(f"Applied E-field was in +z direction")
print(f"Expected: P_z > 0 for normal dielectric")
PY
```

## Step 6: Validation and Troubleshooting

### Python-LAMMPS BEC parity

```bash
python - <<'PY'
import numpy as np
from ase.io import read
from nequip.scripts.deploy import load_deployed_model
from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper

# Load structure
atoms = read("batio3.data", format="lammps-data", style="atomic")

# Python inference
model, metadata = load_deployed_model("deployed/bec_model.pth", device="cpu", freeze=True)
type_map = {"Ba": 0, "Ti": 1, "O": 2}
mapper = TypeMapper(chemical_symbol_to_type=type_map)
data = AtomicData.from_ase(atoms, r_max=float(metadata["r_max"]))
batch = mapper(data).to_dict()

with torch.no_grad():
    output = model(batch)

python_bec = output[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY].numpy().reshape(-1, 3, 3)

# LAMMPS BEC (extract from dump or compute output)
lammps_bec = ...  # shape (N, 3, 3)

# Parity check
bec_diff = np.abs(python_bec - lammps_bec).max()
print(f"Max BEC difference (Python vs. LAMMPS): {bec_diff:.6e} e")

if bec_diff < 1e-4:
    print("PASS: BEC parity (CPU serial)")
elif bec_diff < 1e-3:
    print("PASS: BEC parity (GPU/float32)")
else:
    print("FAIL: BEC parity check")
PY
```

Typical tolerances:
- BEC: < 1e-4 e per component (CPU), < 1e-3 e (GPU float32)

### Common issues

| Symptom | Cause | Solution |
| --- | --- | --- |
| `born_charges not in metadata` | Config error | Verify `born_charges: true` in training config |
| LAMMPS "unknown compute bec_xx" | LAMMPS version | Use SpinGNN LAMMPS fork with BEC support |
| BEC tensors all zero | Model not trained | Check BEC loss converged, not skipped |
| Large acoustic sum rule violation | Data quality | Check VASP DFPT convergence (EDIFF, ENCUT) |
| E-field has no effect | `fix efield` not applied | Verify `fix efield` line in LAMMPS input |

### Physical validation

For ferroelectric phase transitions:

```bash
# Run MD at different temperatures to find T_c
# Monitor average polarization vs. T
# Compare with experimental Curie temperature
```

For piezoelectric response:

```bash
# Apply stress (via `fix deform` or NPT with target stress)
# Measure induced polarization ΔP
# Calculate piezoelectric coefficient: d = ΔP / Δσ
# Compare with experimental values
```

## Testing

Run BEC-specific tests:

```bash
cd /path/to/SpinGNN

# CPU unit tests (includes BEC model building)
tools/ci/cpu_unit_tests.sh

# LAMMPS BEC parity tests
export SPINGNN_LAMMPS_BIN=/path/to/lmp
python -m pytest tests_lammps -m "lammps and not mpi" -k bec

# End-to-end BEC training -> LAMMPS (slow)
python -m pytest 'tests_lammps/test_e2e_train_lammps.py::test_dataset_training_deploys_and_matches_lammps[bec]'
```

## Advanced: Dipole Moment Output

SpinGNN also supports **per-atom dipole moments** as a separate output. This is distinct from BEC:

- **BEC:** Z*_i,αβ relates displacement to polarization (response property)
- **Dipole:** μ_i,α is the instantaneous dipole moment of atom i (state property)

To enable dipoles:

```yaml
dipole: true

loss_coeffs:
  # ... (energy, forces, etc.)
  dipole: 1.0
```

Dataset must have `atoms.arrays["dipole"]` with shape (N, 3).

## See Also

- `example/04_born_effective_charge.yaml` - BEC config template
- `lammps/examples/spingnn/in.bec_efield` - LAMMPS E-field example
- `tests_lammps/test_trajectory_parity.py` - BEC trajectory parity tests
- `src/core/nequip/data/_keys.py` - `BORN_EFFECTIVE_CHARGE_KEY` definition
- `src/models/allegro/nn/_allegro.py` - BEC output head implementation
