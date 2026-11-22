---
name: spin-soc-e2e
description: End-to-end workflow for spin Allegro with SOC - from SOC-aware magnetic dataset to spin-orbit-coupled MD. Handles lattice-spin-orbit interactions and T-odd equivariance.
---

# Spin Allegro with SOC: End-to-End Workflow

Complete pipeline for training and deploying a magnetic Allegro potential with spin-orbit coupling. For systems where spin-lattice-orbit interactions are non-negligible.

## When to Use

Use this skill when you need to:
- Train a magnetic potential for systems with strong SOC (heavy elements, 4d/5d/5f materials)
- Model magnetocrystalline anisotropy from first principles
- Capture orbital-moment contributions to magnetic torques
- Run spin-orbit-coupled MD with LAMMPS

**Trigger phrases:** "spin soc", "spin-orbit coupling", "SOC磁性模型", "magnetocrystalline anisotropy", "heavy element magnets"

**NOT for:** Heisenberg-only systems (use `skills/spin-nosoc-e2e`), structure-only (use `skills/structure-only-e2e`).

## Prerequisites

- SpinGNN repository with validated H100/4090 environment
- Dataset with SOC-computed `energy`, `forces`, `spin_forces` (including orbital contributions)
- Vector `initial_magmoms` from non-collinear SOC DFT
- LAMMPS built with `SPIN` package and `pair_style spin/allegro_spin`

## SOC vs. NoSOC: Key Differences

| Feature | NoSOC (Heisenberg) | SOC |
| --- | --- | --- |
| Time-reversal symmetry | T-even (S → −S gives same E) | T-odd (S → −S gives different E) |
| Spin forces | Pure exchange torques | Exchange + SOC anisotropy torques |
| Training time | baseline | ~1.8× (T-odd channels + SOC terms) |
| Typical applications | 3d metals, collinear/simple NC | 4d/5d metals, magnetocrystalline anisotropy |
| DFT requirement | ISPIN=2 sufficient | LSORBIT=.TRUE. required |

**Physical meaning:** SOC models know that flipping all spins (S → −S) while keeping the lattice fixed changes the energy (due to orbital hybridization). This captures directional magnetic anisotropy.

## Quick Reference

| Stage | Time | Critical parameters |
| --- | --- | --- |
| SOC DFT data | 5-50× slower than PBE | LSORBIT, LNONCOLLINEAR |
| Training | 3-72h | `spin: true`, `soc: true` |
| Deployment | ~1min | Same TorchScript `.pth` |
| LAMMPS SOC MD | minutes-days | Same spin integrators as nosoc |

## Step 1: Prepare SOC Magnetic Dataset

### Dataset schema (same as nosoc)

```python
# Each frame must have:
atoms.positions                             # (N, 3)
atoms.info["energy"]                        # scalar, eV
atoms.arrays["forces"]                      # (N, 3), eV/Å
atoms.set_initial_magnetic_moments(spins)   # (N, 3) VECTOR
atoms.arrays["spin_forces"]                 # (N, 3), eV (includes SOC torques)
```

The format is identical to nosoc, but the **physics content differs**:
- `spin_forces` now include orbital-moment contributions
- Energy depends on absolute spin orientations (not just relative angles)

### From VASP with SOC

```bash
# VASP INCAR must have:
# LSORBIT = .TRUE.
# LNONCOLLINEAR = .TRUE.
# GGA_COMPAT = .FALSE.  (recommended for SOC)

python - <<'PY'
from ase.io import read, write
import numpy as np

# Read VASP SOC output
frames = read("vasprun.xml", index=":")

for atoms in frames:
    # Extract non-collinear magnetic moments (already vectors from SOC calc)
    magmoms = atoms.get_magnetic_moments()
    
    # For SOC VASP, magmoms should already be (N, 3) from VASP parser
    assert magmoms.ndim == 2 and magmoms.shape[1] == 3, \
        "SOC VASP should give vector moments"
    
    atoms.set_initial_magnetic_moments(magmoms)
    
    # Extract forces (standard)
    forces = atoms.get_forces()
    
    # Extract spin forces (SOC torques)
    # VASP outputs these via LORBIT setting; format depends on version
    # This is a placeholder - use your VASP parser
    spin_forces = ...  # shape (N, 3), includes orbital contributions
    atoms.set_array("spin_forces", spin_forces)

write("data/spin_soc.extxyz", frames, format="extxyz")
PY
```

### Validate SOC dataset

```bash
cd /path/to/SpinGNN

python - <<'PY'
import numpy as np
from ase.io import read

frames = read("data/spin_soc.extxyz", index=":")
print(f"Total SOC frames: {len(frames)}")

for idx in [0, -1]:
    atoms = frames[idx]
    
    # Same validation as nosoc
    assert np.isfinite(atoms.positions).all()
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()
    
    magmoms = atoms.get_initial_magnetic_moments()
    assert magmoms.ndim == 2 and magmoms.shape == (len(atoms), 3)
    assert np.isfinite(magmoms).all()
    
    spin_forces = atoms.get_array("spin_forces")
    assert spin_forces.shape == (len(atoms), 3)
    assert np.isfinite(spin_forces).all()
    
    # SOC-specific check: non-trivial spin orientations
    # (pure collinear data defeats the purpose of SOC training)
    spin_directions = magmoms / (np.linalg.norm(magmoms, axis=1, keepdims=True) + 1e-12)
    z_component = np.abs(spin_directions[:, 2])
    non_collinear_frac = np.mean(z_component < 0.99)
    print(f"Frame {idx}: {len(atoms)} atoms, "
          f"{non_collinear_frac*100:.1f}% non-collinear spins")

print("SOC_DATASET_OK")
PY
```

**Recommendation:** Include diverse spin configurations (not just z-aligned) to teach the model about anisotropy. Pure collinear training data → model cannot learn directional dependence.

## Step 2: Configure SOC Training

Copy the SOC template:

```bash
cp example/03_spin_soc.yaml config/my_spin_soc.yaml
```

### Required SOC settings

```yaml
# Enable spin WITH SOC
spin: true
soc: true   # <-- This is the critical flag

# Data
dataset_file_name: data/spin_soc.extxyz
chemical_symbol_to_type:
  Fe: 0  # Example: Fe with strong SOC
  Pt: 1

# Model geometry
r_max: 5.0
l_max: 2   # May need l_max=3 for heavy elements

# Loss (same as nosoc)
loss_coeffs:
  total_energy:
    - 1.0
    - PerAtomMSELoss
  forces: 1.0
  spin_forces: 1.0

# Validation
metrics_components:
  - [forces, mae]
  - [spin_forces, mae]

# Output
root: results/spin_soc
run_name: my_spin_soc_run
```

### Key difference from nosoc config

**Only one line:** `soc: true` instead of `soc: false`.

This single flag changes the model's equivariance group from T-even to T-odd, enabling it to learn that E(S) ≠ E(−S).

### Training acceleration

SOC models also benefit from compilation:

```yaml
spingnn_compile:
  enabled: true
  scope: full_fx
  dynamic: true

float32_matmul_precision: high
allow_tf32: true
```

Expected speedup on H100 (SOC fixture):
- Allegro-spin-soc: ~2.6× (compile + TF32, measured 2026-08-31)

## Step 3: Train the SOC Model

```bash
cd /path/to/SpinGNN

export PYTHONPATH="$PWD/src/core:$PWD/src/models"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

nequip-train config/my_spin_soc.yaml
```

### Monitor convergence

SOC models typically need more epochs than nosoc (stronger coupling → more complex landscape):

```bash
TRAIN_DIR=results/spin_soc/my_spin_soc_run

python - <<'PY'
import csv
import pathlib

metrics = pathlib.Path("results/spin_soc/my_spin_soc_run/metrics_epoch.csv")
with metrics.open() as f:
    rows = list(csv.DictReader(f))

print(f"Total epochs: {len(rows)}")
final = rows[-1]
print(f"Final validation force MAE: {final.get('validation_forces_mae', 'N/A')} eV/Å")
print(f"Final validation spin_force MAE: {final.get('validation_spin_forces_mae', 'N/A')} eV")

force_mae = [float(r['validation_forces_mae']) for r in rows if 'validation_forces_mae' in r]
spin_mae = [float(r['validation_spin_forces_mae']) for r in rows if 'validation_spin_forces_mae' in r]

print(f"Best force MAE: {min(force_mae):.6f} eV/Å")
print(f"Best spin_force MAE: {min(spin_mae):.6f} eV")
PY
```

Typical convergence for SOC:
- Force MAE < 0.1 eV/Å (same as nosoc)
- Spin force MAE: system-dependent; anisotropy torques are often smaller than exchange, so MAE can be < 0.01 eV

## Step 4: Deploy the SOC Model

Deployment is identical to nosoc (same TorchScript `.pth` format):

```bash
TRAIN_DIR=results/spin_soc/my_spin_soc_run
mkdir -p deployed

nequip-deploy build \
  --train-dir "$TRAIN_DIR" \
  deployed/spin_soc.pth

nequip-deploy info deployed/spin_soc.pth
```

Expected metadata:
```
spin: true
soc: true   # <-- Confirms SOC model
r_max: 5.0
type_names: Fe Pt
```

### Validate SOC equivariance in Python

```bash
python - <<'PY'
import torch
import numpy as np
from nequip.scripts.deploy import load_deployed_model
from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper
from ase.io import read

model, metadata = load_deployed_model(
    "deployed/spin_soc.pth",
    device="cpu",
    freeze=True,
    set_global_options=True
)

assert metadata["spin"] == "true"
assert metadata["soc"] == "true", "Model must have soc=true for SOC"

# Load a test structure with non-collinear spins
atoms = read("data/spin_soc.extxyz", index=0)
type_map = {"Fe": 0, "Pt": 1}
mapper = TypeMapper(chemical_symbol_to_type=type_map)

data = AtomicData.from_ase(atoms, r_max=float(metadata["r_max"]))
batch = mapper(data).to_dict()

# Forward pass
with torch.no_grad():
    output = model(batch)

energy_S = float(output[AtomicDataDict.TOTAL_ENERGY_KEY])
forces_S = output[AtomicDataDict.FORCE_KEY].numpy()
spin_forces_S = output[AtomicDataDict.SPIN_FORCE_KEY].numpy()

# Flip all spins: S → −S
atoms_flipped = atoms.copy()
magmoms = atoms.get_initial_magnetic_moments()
atoms_flipped.set_initial_magnetic_moments(-magmoms)

data_flipped = AtomicData.from_ase(atoms_flipped, r_max=float(metadata["r_max"]))
batch_flipped = mapper(data_flipped).to_dict()

with torch.no_grad():
    output_flipped = model(batch_flipped)

energy_neg_S = float(output_flipped[AtomicDataDict.TOTAL_ENERGY_KEY])

# SOC check: E(S) should differ from E(−S)
energy_diff = abs(energy_S - energy_neg_S)
print(f"SOC_EQUIVARIANCE_CHECK:")
print(f"  E(S)   = {energy_S:.6f} eV")
print(f"  E(−S)  = {energy_neg_S:.6f} eV")
print(f"  |ΔE|   = {energy_diff:.6e} eV")

if energy_diff < 1e-6:
    print("  WARNING: Energy is T-even (no SOC effect detected)")
    print("  Possible causes: soc=false in training, or purely collinear data")
else:
    print("  OK: Energy is T-odd (SOC captured)")

print(f"\nDEPLOY_OK spin_soc")
PY
```

**Expected:** `|ΔE|` should be **non-zero** for a working SOC model. If it's ~0, check:
1. Training config had `soc: true`
2. Training data includes non-collinear spin configurations
3. System actually has SOC physics (light elements may show negligible SOC)

## Step 5: LAMMPS SOC Spin-Lattice MD

### LAMMPS input (same integrators as nosoc)

Create `in.spin_soc_nvt`:

```lammps
# Spin-lattice NVT with SOC-aware Allegro
units       metal
boundary    p p p
atom_style  spin
atom_modify map array
newton      off

read_data   soc_system.data

# Allegro SOC pair style (same command as nosoc!)
pair_style  spin/allegro_spin
pair_coeff  * * deployed/spin_soc.pth Fe Pt

# Spin integrator with damping
fix         spin_damp all nve/spin lattice yes
fix_modify  spin_damp alpha 0.1

# Lattice thermostat
fix         thermo all langevin 300.0 300.0 0.1 48279

timestep    0.0005

# Output (same as nosoc)
compute     spinprops all property/atom spx spy spz sp fmx fmy fmz
thermo      100
thermo_style custom step temp pe ke etotal press fmax
dump        traj all custom 100 soc_traj.lammpstrj &
            id type x y z fx fy fz &
            c_spinprops[1] c_spinprops[2] c_spinprops[3] c_spinprops[4] &
            c_spinprops[5] c_spinprops[6] c_spinprops[7]
dump_modify traj sort id

run         10000
```

**Key point:** The LAMMPS input is **identical** to nosoc. The SOC physics is encoded in the deployed model, not in the LAMMPS script.

### LAMMPS data file (same format as nosoc)

```
SOC system with Fe-Pt

40 atoms
2 atom types

... (box dimensions)

Masses

1 55.845  # Fe
2 195.08  # Pt

Atoms # spin

# id type magnitude x y z spx spy spz
1 1 2.5 0.0 0.0 0.0  0.707 0.0 0.707   # Non-collinear spin
2 2 0.3 2.5 0.0 0.0  0.0 1.0 0.0
...
```

### Run SOC simulation

```bash
LMP=/path/to/lmp
cd /path/to/SpinGNN

$LMP -in in.spin_soc_nvt
```

### Observe magnetocrystalline anisotropy

SOC models can show directional energy barriers:

```bash
# Compare energy of same structure with spins along different axes
# (requires running multiple simulations with different initial spin orientations)

# Example: z-aligned vs. x-aligned
# If SOC is strong, E(spins || z) ≠ E(spins || x)
```

This is the **key observable** that distinguishes SOC from nosoc models.

## Step 6: Validation and Troubleshooting

### Verify T-odd behavior

```bash
python - <<'PY'
import numpy as np
from nequip.scripts.deploy import load_deployed_model
from ase.io import read

model, metadata = load_deployed_model("deployed/spin_soc.pth", device="cpu", freeze=True)

# Load two frames from SOC training data
frame0 = read("data/spin_soc.extxyz", index=0)
frame1 = read("data/spin_soc.extxyz", index=10)

from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper

type_map = {"Fe": 0, "Pt": 1}
mapper = TypeMapper(chemical_symbol_to_type=type_map)

def predict(atoms):
    data = AtomicData.from_ase(atoms, r_max=float(metadata["r_max"]))
    batch = mapper(data).to_dict()
    with torch.no_grad():
        output = model(batch)
    return float(output[AtomicDataDict.TOTAL_ENERGY_KEY])

# Original energies
E0 = predict(frame0)
E1 = predict(frame1)

# Flip spins
frame0_flip = frame0.copy()
frame0_flip.set_initial_magnetic_moments(-frame0.get_initial_magnetic_moments())
frame1_flip = frame1.copy()
frame1_flip.set_initial_magnetic_moments(-frame1.get_initial_magnetic_moments())

E0_flip = predict(frame0_flip)
E1_flip = predict(frame1_flip)

print("T-odd verification:")
print(f"Frame 0: E(S) = {E0:.6f}, E(−S) = {E0_flip:.6f}, |ΔE| = {abs(E0-E0_flip):.6e} eV")
print(f"Frame 1: E(S) = {E1:.6f}, E(−S) = {E1_flip:.6f}, |ΔE| = {abs(E1-E1_flip):.6e} eV")

if abs(E0 - E0_flip) > 1e-5 and abs(E1 - E1_flip) > 1e-5:
    print("PASS: Model is T-odd (SOC working)")
else:
    print("FAIL: Model is T-even (SOC not captured)")
PY
```

### Common issues

| Symptom | Cause | Solution |
| --- | --- | --- |
| E(S) = E(−S) after training | `soc: false` in config | Verify `soc: true`, retrain |
| E(S) = E(−S) with `soc: true` | Purely collinear data | Include non-collinear configurations |
| High spin_force MAE | SOC data quality | Check DFT convergence, LSORBIT |
| LAMMPS same as nosoc | Expected! | SOC is in the model, not the script |
| No anisotropy in MD | Weak SOC in system | Check if heavy elements are present |

### Python-LAMMPS parity (same as nosoc)

```bash
python - <<'PY'
# (Same parity check code as nosoc skill)
# Tolerances are identical:
# Energy < 2e-4 eV, Force < 1e-3 eV/Å, Spin force < 1e-3 eV
PY
```

## Step 7: Physical Validation

### Magnetocrystalline anisotropy energy (MAE)

For systems with known MAE from experiment or high-level theory:

```bash
# Run MD with spins initialized along easy axis
$LMP -in in_easy_axis.lmp
E_easy=$(grep "Step.*TotEng" log.lammps | tail -1 | awk '{print $5}')

# Run MD with spins along hard axis
$LMP -in in_hard_axis.lmp
E_hard=$(grep "Step.*TotEng" log.lammps | tail -1 | awk '{print $5}')

# MAE = E_hard - E_easy (should match literature if data is good)
echo "Predicted MAE: $(echo "$E_hard - $E_easy" | bc) eV/atom"
```

### Spin dynamics timescale

SOC affects spin precession rates. For validation:
- Compare spin relaxation time from MD with experimental data (if available)
- Check that anisotropy barriers slow down spin flips as expected

## Testing

Run SOC-specific tests:

```bash
cd /path/to/SpinGNN

# CPU unit tests (includes SOC model building)
tools/ci/cpu_unit_tests.sh

# LAMMPS SOC parity tests
export SPINGNN_LAMMPS_BIN=/path/to/lmp
python -m pytest tests_lammps -m "lammps and not mpi" -k soc

# End-to-end SOC training -> LAMMPS (slow)
python -m pytest 'tests_lammps/test_e2e_train_lammps.py::test_dataset_training_deploys_and_matches_lammps[soc]'
```

## Comparison: NoSOC vs. SOC

| Property | NoSOC | SOC |
| --- | --- | --- |
| Config difference | `soc: false` | `soc: true` |
| Time-reversal | T-even | T-odd |
| E(S) vs. E(−S) | Equal | Different |
| Training time | 1× | ~1.2× |
| Use case | 3d magnets, pure exchange | 4d/5f, anisotropy, orbital effects |
| LAMMPS input | Identical | Identical |
| Pair style | `spin/allegro_spin` | `spin/allegro_spin` |

## See Also

- `example/03_spin_soc.yaml` - Full SOC config template
- `example/05_spin_soc_production.yaml` - Production-scale SOC training
- `skills/spin-nosoc-e2e/SKILL.md` - For Heisenberg systems
- `tests_lammps/test_e2e_train_lammps.py` - SOC training and LAMMPS parity tests
- `src/models/allegro/nn/_allegro_spin.py` - SOC implementation details
- `tools/ci/gpu_full_tests.sh` - GPU SOC precision gates
