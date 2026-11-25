---
name: spin-soc-e2e
description: End-to-end workflow for spin-orbit-coupled magnetic SpinGNN++ (Allegro_soc / v2_soc) — from parallel VASP 5 & 6 SOC dataset conversion (spingnn-data) through training (Adam/PhysMuon), .sp2.json/.sp2.bin export, bundle_abi=1 JIT compilation, anisotropy symmetry checks, and modern LAMMPS spin-lattice MD (pair_style spingnn2/kk, fix spin/sib/kk, fix spin/cfi/kk).
---

# Spin Allegro with SOC: End-to-End Workflow

Complete pipeline for training and deploying a spin–orbit-coupled (SOC)
magnetic potential (`Allegro_soc` in v1 YAML or `v2_soc` in pure-PyTorch
`spingnn2.v2`) obeying coupled $O(3) \times \mathbb{Z}_2^T$ symmetry for
magnetocrystalline anisotropy, Dzyaloshinskii–Moriya interaction (DMI),
symmetric anisotropic exchange, and single-ion anisotropy (SIA).

## When to Use

Use this skill when you need to:
- Train a magnetic potential for heavy-element / 4d / 5d / 5f materials with strong SOC (`LSORBIT = .TRUE.`)
- Capture magnetocrystalline anisotropy, DMI, and Kitaev / biquadratic exchange from first principles
- Deploy with zero LibTorch into Python/ASE (`NativeModel`, `SpinGNN2Calculator`) or modern LAMMPS (`pair_style spingnn2(/kk)`)

**Trigger phrases:** "spin soc", "spin-orbit coupling", "SOC磁性模型", "magnetocrystalline anisotropy", "DMI", "heavy element magnets"

**NOT for:** Isotropic Heisenberg systems without SOC (use [`skills/spin-nosoc-e2e/SKILL.md`](../spin-nosoc-e2e/SKILL.md)), structure-only (use [`skills/structure-only-e2e/SKILL.md`](../structure-only-e2e/SKILL.md)).

## SOC vs. NoSOC: Key Differences

| Feature | NoSOC (`Allegro_nosoc` / `v2_nosoc`) | SOC (`Allegro_soc` / `v2_soc`) |
| --- | --- | --- |
| Spin–lattice symmetry | Decoupled $O(3)_{\text{lat}} \times O(3)_{\text{spin}}$ | Coupled $O(3) \times \mathbb{Z}_2^T$ |
| Zero-field time reversal | $E(-\mathbf{S}, \mathbf{r}) = E(\mathbf{S}, \mathbf{r})$ | $E(-\mathbf{S}, \mathbf{r}) = E(\mathbf{S}, \mathbf{r})$ |
| Spin-only rotation at fixed lattice | $E(R\mathbf{S}, \mathbf{r}) = E(\mathbf{S}, \mathbf{r})$ | $E(R\mathbf{S}, \mathbf{r}) \neq E(\mathbf{S}, \mathbf{r})$ (anisotropy & DMI) |
| DFT requirement | `ISPIN=2` or non-collinear without SOC | `LSORBIT = .TRUE.`, `LNONCOLLINEAR = .TRUE.` |
| Modern LAMMPS pair style | `pair_style spingnn2(/kk)` | `pair_style spingnn2(/kk)` |

## Step 1: Prepare SOC Magnetic Dataset (`spingnn-data`)

Convert converged VASP 5.x or VASP 6.x non-collinear SOC (`LSORBIT = .TRUE.`,
`LNONCOLLINEAR = .TRUE.`, `I_CONSTRAINED_M = 1|2|4`) runs in parallel across
CPU cores with `spingnn-data -m`:

```bash
cd /path/to/SpinGNN2

python tools/ci/source_python.py spingnn-data -m \
  --path /path/to/vasp-soc-runs \
  --out-path data/spin_soc.extxyz \
  --nomagz 53 \
  --cpu 8
```

`spingnn-data` collects only converged final steps, automatically detects
VASP 5 (`×2`) vs VASP 6 (`×1`) and Mode 4 (`×1`), applies `SAXIS` Euler
rotation to align spins with the Cartesian lattice frame, subtracts penalty
energy $E_p$, and writes `initial_magmoms`, `spin_length`, and tangential
`spin_forces_vert` (strictly orthogonal to `initial_magmoms`).

> **Recommendation:** Include diverse non-collinear spin orientations relative to crystal axes so the model can learn directional anisotropy and DMI.

### Validate SOC dataset

```bash
python - <<'PY'
import numpy as np
from ase.io import read

frames = read("data/spin_soc.extxyz", index=":")
assert frames, "empty dataset"
print(f"Total SOC frames: {len(frames)}")

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
    norms = np.linalg.norm(magmoms, axis=1)
    mag = norms > 1e-6
    non_z = np.mean(np.abs(magmoms[mag, 2] / norms[mag]) < 0.99) if mag.any() else 0.0
    print(f"Frame {idx}: {len(atoms)} atoms, {mag.sum()} magnetic, {non_z*100:.1f}% off-z")

print("SOC_DATASET_OK")
PY
```

## Step 2: Configure SOC Training

Copy [`example/03_spin_soc.yaml`](../../example/03_spin_soc.yaml) (or
[`example/05_spin_soc_production.yaml`](../../example/05_spin_soc_production.yaml)
for production-scale runs):

```bash
cp example/03_spin_soc.yaml config/my_spin_soc.yaml
```

### Key SOC settings (matching `example/03_spin_soc.yaml`)

```yaml
root: results/spin_soc
run_name: my_spin_soc_run
dataset_file_name: data/spin_soc.extxyz
n_train: 800
n_val: 100
chemical_symbol_to_type:
  Cr: 0
  I: 1
r_max: 7.0

model_builders:
  - allegro.model.Allegro_soc
  - PerSpeciesRescale
  - ParaStressForceSpinOutput
  - RescaleEnergyEtc

ljmatrix: true       # full anisotropic exchange matrix (incl. DMI)
lani: true           # single-ion anisotropy (SIA)
lbiquadratic: true   # biquadratic spin coupling
lsoc_layer: true     # coupled L-S equivariant layers
l_max: 2
spharm_soc_on_l_max: 1
symmetry: u2

loss_coeffs:
  total_energy: [1.0, PerAtomMSELoss]
  forces: 1.0
  spin_forces_vert: 1.0

metrics_components:
  - [forces, mae]
  - [spin_forces_vert, mae]
```

### Acceleration & `PhysMuon`

SOC configs enable PyTorch 2.11 `full_fx` dynamic compilation by default.
Add `optimizer_name: PhysMuon` in YAML, or train pure-PyTorch `spingnn2.v2`
`build_v2_soc` (`3.4×–11.5×` speedup, `−86%` VRAM):

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

## Step 3: Train the SOC Model

```bash
cd /path/to/SpinGNN2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/ci/source_python.py spingnn-train config/my_spin_soc.yaml
```

## Step 4: Export, Pre-Compile JIT & Verify SOC Symmetry

Export your trained `Allegro_soc` checkpoint to `.sp2.json` + `.sp2.bin` and
pre-compile an offline single-arch or multi-GPU `bundle_abi = 1` JIT artifact
(**2,000,000 atoms on one RTX 5090 in 2.23 GB VRAM at 9.52 µs/atom strict
FP32**; [`docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md)):

```bash
TRAIN_DIR=results/spin_soc/my_spin_soc_run
mkdir -p deployed

python tools/ci/source_python.py spingnn2-export \
  "$TRAIN_DIR" -o deployed/spin_soc

ARTIFACT=$(python tools/ci/source_python.py spingnn2-prepare-jit-artifact \
  --model deployed/spin_soc.sp2.json \
  --backend cuda --arch all-gpu --precision float32 \
  --library validation/native/build-cuda-all-gpu/libspingnn2.so)
echo "Prepared manifest: $ARTIFACT"
```

### Verify coupled $O(3)$ anisotropy & zero-field time-reversal symmetry $E(-\mathbf{S}) = E(\mathbf{S})$

```python
import numpy as np
from ase.io import read
from spingnn2.calculator import SpinGNN2Calculator

atoms = read("data/spin_soc.extxyz", index=0)
calc = SpinGNN2Calculator(
    "deployed/spin_soc.sp2.json",
    backend="cuda",
    precision="float32",
    execution_mode="direct",
    artifact=artifact_manifest,
    radial_mode="spline",
)

atoms.calc = calc
e_orig = atoms.get_potential_energy()

# 1. Rotate spins 90 deg about x at fixed lattice -> SOC breaks spin-only invariance
atoms_rot = atoms.copy()
m = atoms.arrays["initial_magmoms"]
atoms_rot.arrays["initial_magmoms"] = np.column_stack([m[:, 0], -m[:, 2], m[:, 1]])
atoms_rot.calc = calc
e_rot = atoms_rot.get_potential_energy()

# 2. Flip all spins S -> -S -> zero-field time-reversal invariance holds
atoms_tr = atoms.copy()
atoms_tr.arrays["initial_magmoms"] = -m
atoms_tr.calc = calc
e_tr = atoms_tr.get_potential_energy()

print(f"E(S)={e_orig:.6f}  |ΔE_aniso|={abs(e_orig - e_rot):.3e}  |ΔE_TR|={abs(e_orig - e_tr):.3e}")
```

> If evaluating a PyTorch/TorchScript model directly (`load_deployed_model`), **never** wrap the call in `torch.no_grad()` because forces and spin forces require autograd.

## Step 5: Run Modern LAMMPS SOC Spin–Lattice MD (`pair_style spingnn2(/kk)`)

The modern LAMMPS input is identical to NoSOC — SOC physics is encoded inside
the exported `.sp2.json`/`.sp2.bin` contract:

```lammps
units       metal
boundary    p p p
atom_style  spin
atom_modify map array
read_data   soc_system.data

pair_style  spingnn2 precision float32 math ieee cache yes profile yes
pair_coeff  * * deployed/spin_soc.sp2.json prepared-artifact.bundle.json Cr I

# Use `lattice frozen` for spin-only anisotropy/precession, or `lattice moving` for coupled MD:
fix         integrate all spin/sib 300.0 0.1 91081 lattice moving
# Or 1-eval/step CFI (alias spin/lpsib) with optional canonical-volume box iso NPT:
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

Launch multi-GPU CUDA runs via `tools/lammps/spingnn2_gpu_rank.py` with
`-k on g 1 -sf kk -pk kokkos gpu/aware on neigh half` (or `gpu/aware off` on
non-GPU-aware MPI stacks). With `cache yes` (default), verified full-width SOC
Stage-1 speedups on RTX 5090 are **1.384× (frozen)** and **1.296× (moving)**.

## Step 6: Troubleshooting & Testing

| Symptom | Cause | Solution |
| --- | --- | --- |
| `E(R*S) == E(S)` after training | Wrong builder or purely collinear dataset | Verify `allegro.model.Allegro_soc`, `lsoc_layer: true`, and diverse non-collinear DFT orientations |
| High `spin_forces_vert` MAE | Unconverged SOC DFT or missing `SAXIS` | Re-export with `spingnn-data -m` (skips unconverged steps & rotates `SAXIS`) |
| Stage-1 tape spills to host in Python `NativeModel` | SOC tapes are larger than NoSOC | Raise `cache_device_limit_bytes` when calling `evaluate_stage1_lattice` on large systems |

```bash
cd /path/to/SpinGNN2
tools/ci/cpu_unit_tests.sh
python tools/ci/source_python.py -m pytest tests_v2 tests_native/test_vasp_spin_io.py -q

export SPINGNN2_LAMMPS_BIN=/path/to/install/bin/lmp_spingnn2
python tools/ci/source_python.py -m pytest \
  tests_native/test_lammps_pair.py tests_native/test_lammps_spin_sib.py tests_native/test_spin_cfi.py -k soc -v
```

## See Also

- [`example/03_spin_soc.yaml`](../../example/03_spin_soc.yaml) · [`example/05_spin_soc_production.yaml`](../../example/05_spin_soc_production.yaml)
- [`docs/SPINGNN2_GUIDE.md`](../../docs/SPINGNN2_GUIDE.md) · [`lammps_spingnn2/README.md`](../../lammps_spingnn2/README.md)
- [`skills/spingnn2/SKILL.md`](../spingnn2/SKILL.md) · [`skills/spin-nosoc-e2e/SKILL.md`](../spin-nosoc-e2e/SKILL.md)
