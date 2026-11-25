---
name: spingnn2
description: End-to-end workflow for SpinGNN2 native C++17/Kokkos inference and modern LAMMPS (stable_30Sep2026_update1) molecular dynamics. Covers checkpoint export (.sp2.json/.sp2.bin), offline NVRTC/Host JIT preparation, zero-LibTorch Python/ASE inference, pair_style spingnn2(/kk), fix spin/sib(/npt)/kk, fix spin/cfi(/kk), compute spin/gpst(/kk), DREAM BEC, e2SLP fix pair, compute spingnn2/timing, and multi-GPU MPI.
---

# SpinGNN2: Native Engine & Modern LAMMPS Workflow

Use this skill to export a trained SpinGNN++ checkpoint (`Allegro_nosoc`,
`Allegro_soc`, `Allegro_born` DREAM, `Allegro_e2IP`, or `Allegro_soc_e2IP`)
into the SpinGNN2 compact contract (`.sp2.json` + `.sp2.bin`, format
`spingnn2.contract.v1`), pre-compile model-specialized offline JIT artifacts,
and run high-throughput, **zero-LibTorch** inference in Python/ASE or modern
LAMMPS (`stable_30Sep2026_update1`).

## 1. Build Native Engine & Modern LAMMPS

### Native C ABI library (`libspingnn2.so`) for Python/ASE

```bash
# Populate SHA-256-verified Kokkos 4.7.02 & nlohmann/json 3.11.3 cache
python tools/native/build_native.py --prepare-dependencies

# Build per backend / CUDA architecture (submit via Slurm on clusters)
python tools/native/build_native.py --backend cpu
python tools/native/build_native.py --backend cuda --arch all-gpu --cuda-root /data/app/cuda/12.8
# Or single architecture (sm80|sm86|sm89|sm90|sm120):
python tools/native/build_native.py --backend cuda --arch sm120 --cuda-root /data/app/cuda/12.8

# Install standalone zero-LibTorch Python/ASE package on inference hosts
pip install '.[ase]'
```

### Modern LAMMPS (`stable_30Sep2026_update1`) external overlay

```bash
# One-command fingerprinted build (builds libspingnn2 + wheel + GPST kernel + lmp_spingnn2)
python tools/spingnn2_build.py --target all-gpu   # or: --target cpu | sm80 | sm86 | sm89 | sm90 | sm120

# Or manually materialize & verify the pinned overlay tree
gh repo clone lammps/lammps validation/lammps/upstream -- --depth 1 --branch stable_30Sep2026_update1
python lammps_spingnn2/apply.py --upstream validation/lammps/upstream --destination validation/lammps/source
python lammps_spingnn2/apply.py --check --destination validation/lammps/source
```

Outputs and `manifest.json` are written under
`validation/lammps/builds/<target>-<hash>/install/` (`bin/lmp_spingnn2`,
`lib*/libspingnn2*.so`, `wheel/*.whl`).

## 2. Export Checkpoint & Pre-Compile JIT Artifact

Run export in the PyTorch training environment; pre-compile JIT artifacts once
before launching MD (LAMMPS never invokes NVRTC or a C++ compiler at runtime):

```bash
# 1. Export training directory (or specific .pth with --config) -> .sp2.json + .sp2.bin
python tools/ci/source_python.py spingnn2-export \
  results/spin_nosoc/example -o deployed/spin_nosoc

# 2. Pre-compile model-specialized JIT artifact or multi-arch bundle (.bundle.json)
ARTIFACT=$(python tools/ci/source_python.py spingnn2-prepare-jit-artifact \
  --model deployed/spin_nosoc.sp2.json \
  --backend cuda --arch all-gpu --precision float32 \
  --library validation/native/build-cuda-all-gpu/libspingnn2.so)
echo "$ARTIFACT"
```

Keep `<cache_key>.json` and its `kernel-<sha256>.cubin`/`.so` together in a
directory owned by the caller or root and unwritable by group/others. Override
the cache path with `--cache-dir` or `SPINGNN2_JIT_CACHE`.

## 3. Python & ASE Inference (Zero PyTorch / LibTorch)

```python
from spingnn2.native import NativeModel
from spingnn2.calculator import SpinGNN2Calculator

with NativeModel.load(
    "deployed/spin_nosoc.sp2.json",
    backend="cuda",
    precision="float32",
    execution_mode="direct",      # "generic" (default), "auto", or "direct"
    artifact=artifact_manifest,
    execution_profile="speed",    # "capacity" (tile<=256) or "speed" (512; 1024 CUDA FP32 DREAM)
    radial_mode="spline",
    allow_tf32=False,             # strict IEEE FP32 default
) as model:
    out = model.evaluate(graph)
    with model.evaluate_stage1_lattice(graph) as lattice:
        full = model.evaluate_stage2_spin_cached(lattice, new_spins)
        frozen = model.evaluate_stage2_spin_cached(lattice, new_spins, spin_only=True)

# ASE Calculator
atoms.calc = SpinGNN2Calculator(
    "deployed/spin_nosoc.sp2.json",
    backend="cuda",
    precision="float32",
    execution_mode="direct",
    artifact=artifact_manifest,
    radial_mode="spline",
)
```

Use `spingnn2.reference.ReferenceModel.load(...)` for pure-NumPy/SciPy
verification without native libraries.

## 4. Modern LAMMPS Inputs (`pair_style spingnn2(/kk)`)

### Spin–lattice MD (`Allegro_nosoc` / `Allegro_soc`)

```lammps
units       metal
boundary    p p p
atom_style  spin
atom_modify map array
read_data   system-spin.data

pair_style  spingnn2 precision float32 math ieee profile yes
pair_coeff  * * deployed/spin_nosoc.sp2.json prepared-artifact.json Mn Ge

# Frozen lattice (reuses Stage-1 tape across steps & midpoints):
# fix integrate all spin/sib 300.0 0.1 91081 lattice frozen
# Coupled moving lattice (reuses Stage-1 tape at SIB midpoint):
fix         integrate all spin/sib 300.0 0.1 91081 lattice moving
# Or 1-eval/step Cached-Field Integrator (alias: spin/lpsib) with optional box iso barostat:
# fix integrate all spin/cfi 300.0 0.1 91081 lattice moving
# fix integrate all spin/cfi 300.0 0.1 91081 lattice moving ou 300.0 0.05 91082 box iso 0.0 100.0 box_ou 300.0 0.1 91083
# Or spin-lattice NPT (spin/sib_npt alias also accepted):
# fix integrate all spin/sib/npt temp 300 300 0.02 iso 1 1 0.05 alpha 0.1 seed 91081

compute     timers all spingnn2/timing
compute     gpst all spin/gpst cutoff 5.0
thermo_style custom step temp pe etotal c_timers[*]
timestep    0.0001
run         100
```

### DREAM Electric-Field MD (`Allegro_born`)

```lammps
units       metal
boundary    p p p
atom_style  bec
atom_modify map array
read_data   system-bec.data

pair_style  spingnn2 precision float32 math ieee
pair_coeff  * * deployed/born.sp2.json prepared-artifact.json Ba Ti O

fix         field all efield 0.0 0.0 0.01 coupling bec
fix         integrate all nve
compute     dip all bec
compute     becprops all property/atom becxx becxy becxz becyx becyy becyz &
            beczx beczy beczz becdipolex becdipoley becdipolez
thermo_style custom step pe fmax c_dip c_dip[1] c_dip[2] c_dip[3] c_dip[4]
run         100
```

### e2SLP Uncertainty Readouts (`Allegro_e2IP` / `Allegro_soc_e2IP`)

```lammps
fix evidence all pair 1 spingnn2 f_sigma 0 f_Uepi 0 f_nu 0 f_kappa 0
# Or for spin-force evidence:
# fix evidence all pair 1 spingnn2 sf_sigma 0 sf_Uepi 0 sf_nu 0 sf_kappa 0
dump traj all custom 10 e2slp.dump id type x y z fx fy fz f_evidence[*]
```

## 5. Multi-GPU MPI & CPU Kokkos Launch Commands

```bash
# Multi-GPU CUDA (one MPI rank per physical GPU UUID via spingnn2_gpu_rank.py)
mpirun -n 4 tools/lammps/spingnn2_gpu_rank.py \
  /path/to/install/bin/lmp_spingnn2 \
  -k on g 1 -sf kk -pk kokkos gpu/aware on neigh half \
  -in in.md

# Intel MPI 2021.13 on RTX 4090 when CUDA IPC peer read fails:
I_MPI_OFFLOAD=1 I_MPI_OFFLOAD_IPC=0 \
  mpirun -n 2 tools/lammps/spingnn2_gpu_rank.py \
  /path/to/install/bin/lmp_spingnn2 \
  -k on g 1 -sf kk -pk kokkos gpu/aware on neigh half \
  -in in.md

# Host Kokkos / OpenMP (CPU build)
OMP_NUM_THREADS=8 mpirun -n 2 /path/to/cpu-install/bin/lmp_spingnn2 \
  -k on t 8 -sf kk -in in.md
```

## 6. Data Preparation (`spingnn-data`), Pure-PyTorch v2 Training, `PhysMuon` & Offline GPST CLI

### Parallel VASP 5 / VASP 6 Data Conversion (`spingnn-data`)

Convert VASP 5.x or VASP 6.x `OUTCAR`(+`OSZICAR`) trees in parallel across
CPU cores with automatic VASP 5 vs VASP 6 `lambda*MW_perp` factor detection
(quadratic modes 1/2: $R = \sum|\mathbf{P}_I|^2 / (\lambda E_p) \in \{1, 4\}$;
mode 4 linear penalty: weighted $R_4 = \sum|\mathbf{M}_{W,I}||\mathbf{P}_I|^2 / (\lambda E_p) \approx 2$),
$E_p$ subtraction, `SAXIS` rotation, sparse nonmagnetic handling, and tangential
`spin_forces_vert` projection onto the conjugate `MW_int` direction:

```bash
spingnn-data -m --path /path/to/vasp-runs --out-path data/magnetic.extxyz --cpu 8
# Also available: spingnn-data-mlff-bec, spingnn-data-mlff, spingnn-data-delta-mag
```

### v1 YAML & v2 `PhysMuon` / `Muon` Optimization

In v1 YAML configs (`spingnn-train`), select `optimizer_name: PhysMuon` (or
`Muon`) directly; all five physics switches (`iso`, `layer_balance`,
`spectral_clip`, `block_norm`, `spike_guard`) default cleanly and integrate
with `cuda_graph` training steps. For pure-PyTorch `spingnn2.v2` (zero `e3nn`
dependency), use `model.compile_fullgraph(dynamic=True)` or
`model.compile_bucketed(...)` with `PhysMuon`:

```python
from spingnn2.v2 import PhysMuon, WSDCosineScheduler, build_v2_soc

# Also available: build_v2_nosoc, build_v2_born, build_v2_e2slp
model = build_v2_soc(num_types=2, r_cut=5.0, c0=64).cuda()
compiled = model.compile_fullgraph(dynamic=True)  # or model.compile_bucketed(...)
optimizer = PhysMuon(
    model.parameters(),
    lr=1e-3,
    named_parameters=model.named_parameters(),
    cuda_graph=True,
)
scheduler = WSDCosineScheduler(
    optimizer, warmup_steps=50, stable_steps=750, decay_steps=200
)
```

### Offline GPST Spin-Temperature Analysis (`spingnn2.gpst`)

After running LAMMPS with `fix spin/cfi ... accept <K>` and dumping accepted
states (`id type x y z spx spy spz sp fmx fmy fmz`), extract sufficient moments
with the C99 OpenMP fused kernel and run independent-chain CF-GMM / Fieller
inference:

```bash
spingnn2-gpst-moments accepted.dump moments.npz \
  --hbar 0.0006582119569 --cutoff 5.0 --accept-every 10 --force-contract-confirmed

spingnn2-gpst-infer \
  --records chain1.npz chain2.npz chain3.npz \
  --seed-ids s1 s2 s3 --kb 8.617333262e-5 --dt 0.01 --output gpst_report.json
```

## 7. Precision & Verification Notes

- Default math is **strict IEEE FP32** (`math ieee` / `allow_tf32=False`),
  verified to `<= 9.06e-6` vs PyTorch eager on RTX 5090 (`<= 2.49e-14` in
  FP64). Opt-in `math tf32` / `allow_tf32=True` is faster on CUDA FP32 with
  bounded numerical differences (see [`docs/VALIDATION.md`](../../docs/VALIDATION.md)).
- The unified `--target all-gpu` fat binary (`sm_80`, `sm_86`, `sm_89`, `sm_90`,
  `sm_120` + `compute_120` PTX) and `bundle_abi = 1` multi-arch manifests are
  verified live across CPU/`kk-cpu`, RTX 5090 (`sm_120`, 1 & 2 GPU MPI), and
  RTX 4090 (`sm_89`) with zero LibTorch / NVRTC runtime linkage.
- `compute ID all spingnn2/timing` reports 6 rank-max counters: `[1]` pair wall
  excluding halo comm, `[2]` halo + midpoint comm, `[3]` Stage-1 s (`profile
  yes`), `[4]` Stage-2 s (`profile yes`), `[5]` cache hits, `[6]` evaluations.
- `compute ID group spin/gpst [cutoff r_cut]` evaluates 7 Geometric-Probe Spin
  Temperature `(N_c, D_c)` pairs strictly over magnetic atoms (`sp[i][3] > 0`);
  pool online averages with `spingnn2.spin_temp` or run full offline GPST
  inference (`spingnn2-gpst-moments` / `spingnn2-gpst-infer`, C99 OpenMP fused).
- **Verified limits & safety guards** (see [`docs/SPINGNN2_GUIDE.md`](../../docs/SPINGNN2_GUIDE.md#7-verified-configuration-and-hardware-limits)):
  `tile_atoms <= 65535`; per-CTA `env` smem auto-tiers `<= 48 KiB` static /
  `(48, 96] KiB` opt-in / `> 96 KiB` global fallback; `fix spin/cfi ... box iso`
  requires an orthogonal periodic 3D box; on non-GPU-aware MPI stacks use
  `-pk kokkos gpu/aware off`; `SpinGNNV2` supports `lmax in (1, 2, 3)` with
  eager out-of-bucket fallback in `compile_bucketed` and safe `NaN`/`Inf`
  gradient skipping in `PhysMuon`.
- Run the qualification suites with
  `python tools/ci/source_python.py -m pytest tests_ci tests_v2 tests_native`.

## See Also

- [`docs/SPINGNN2_GUIDE.md`](../../docs/SPINGNN2_GUIDE.md)
- [`src/spingnn2/README.md`](../../src/spingnn2/README.md) · [`libspingnn2/README.md`](../../libspingnn2/README.md) · [`lammps_spingnn2/README.md`](../../lammps_spingnn2/README.md)
- [`docs/VALIDATION.md`](../../docs/VALIDATION.md) · [`docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md)
