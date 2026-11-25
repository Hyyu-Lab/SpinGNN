# SpinGNN2 User Guide

This guide walks through the end-to-end **SpinGNN2** workflow—from exporting a
trained SpinGNN++ model checkpoint into the compact `.sp2.json`/`.sp2.bin`
contract (`spingnn2.contract.v1`), pre-compiling model-specialized offline JIT
artifacts, and running high-throughput, zero-LibTorch simulations in Python/ASE
or modern LAMMPS (`stable_30Sep2026_update1`).

For installation and build commands, see [INSTALL.md](../INSTALL.md). For
component-level reference details, see
[src/spingnn2/README.md](../src/spingnn2/README.md),
[libspingnn2/README.md](../libspingnn2/README.md), and
[lammps_spingnn2/README.md](../lammps_spingnn2/README.md).

## 1. Supported Model Families

SpinGNN2 provides a center-atom-tiled C++17 / Kokkos 4.7.02 inference engine
with model-specialized offline JIT kernels for the strictly-local SpinGNN++
families, alongside the pure-PyTorch v2 magnetic symmetry and `torch.compile`
training stack (`spingnn2.symmetry` and `spingnn2.v2`):

| Family | Config Template | Outputs |
| --- | --- | --- |
| `Allegro_nosoc` | [02_spin_nosoc.yaml](../example/02_spin_nosoc.yaml) | Energy, atomic energy, forces, `spin_forces`, `spin_forces_vert`, virial, `atom_virial`, stress |
| `Allegro_soc` | [03_spin_soc.yaml](../example/03_spin_soc.yaml), [05_spin_soc_production.yaml](../example/05_spin_soc_production.yaml) | Same as NoSOC, including anisotropic exchange $J_{ij}$, DMI, single-ion anisotropy $A_i$, and SOC interactions |
| `Allegro_born` (DREAM) | [04_born_effective_charge.yaml](../example/04_born_effective_charge.yaml) | Energy, forces, virial, stress, per-atom $3\times 3$ `born_effective_charges` (none/soft/hard ASR), `total_born_effective_charges` |
| `Allegro_e2IP` / `Allegro_soc_e2IP` (e2SLP) | [06_allegro_e2ip.yaml](../example/06_allegro_e2ip.yaml), [07_allegro_soc_e2ip.yaml](../example/07_allegro_soc_e2ip.yaml) | Physical outputs + nodewise evidential heads (`f_sigma`, `f_Uepi`, `f_nu`, `f_kappa`, `f_uncertainty`, `total_f_Uepi`, and/or `sf_*`) |

### 1.1 Prepare VASP 5 / VASP 6 Datasets (`spingnn-data`)

Convert VASP 5.x or VASP 6.x `OUTCAR` (+ `OSZICAR`) directories into SpinGNN2
`extxyz` datasets in parallel across multiple CPU cores using `spingnn-data`
(implemented in [`src/spingnn2/data/vasp_spin.py`](../src/spingnn2/data/vasp_spin.py)):

```bash
# Multi-core batch conversion of magnetic constrained VASP 5 & VASP 6 runs
python tools/ci/source_python.py spingnn-data -m \
  --path /path/to/vasp-runs \
  --out-path data/magnetic.extxyz \
  --cpu 8

# Single OUTCAR/OSZICAR pair (also supports -b for BEC, --energy-source free|zero, --keep-raw)
python tools/ci/source_python.py spingnn-data -m \
  --outcar run/OUTCAR --oszicar run/OSZICAR \
  --out-path data/frame.extxyz
```

`spingnn-data` collects only converged final steps (skipping pre-convergence
SCF blocks and unconverged/soft-stopped runs), automatically distinguishes
VASP 5 (`1 * lambda*MW_perp`,
quadratic modes 1/2 ratio $R = \sum|\mathbf{P}_I|^2/(\lambda E_p) \approx 1 \Rightarrow \times 2$)
from VASP 6 (`2 * lambda*MW_perp`, modes 1/2 $R \approx 4 \Rightarrow \times 1$;
mode 4 linear penalty weighted ratio $R_4 = \sum|\mathbf{M}_{W,I}||\mathbf{P}_I|^2/(\lambda E_p) \approx 2 \Rightarrow \times 1$)
and cross-checks against the `OUTCAR` version and `M_CONSTR`, subtracts $E_p$
(`F - E_p` by default, invariant to `-m`), applies the `SAXIS` Euler rotation,
handles sparse nonmagnetic rows, projects the constraint field tangentially onto
the conjugate `MW_int` direction (default; `--spin-source m_int` optional), and
writes `energy`, `forces`, `stress`, `initial_magmoms`, `spin_length`,
`spin_forces_vert`, and `magnetic_mask`. Additional data CLIs
are `spingnn-data-mlff`, `spingnn-data-mlff-bec`, and `spingnn-data-delta-mag`.

## 2. Export a Trained Checkpoint (`.sp2.json` + `.sp2.bin`)

Export runs in the PyTorch training environment from the repository root using
`tools/ci/source_python.py` (or the installed `spingnn2-export` command):

```bash
# Export best_model.pth + adjacent config.yaml from a training directory
python tools/ci/source_python.py spingnn2-export \
  results/spin_nosoc/example -o deployed/spin_nosoc

# Or export a specific checkpoint file with an explicit config
python tools/ci/source_python.py spingnn2-export \
  results/spin_nosoc/example/last_model.pth \
  --config results/spin_nosoc/example/config.yaml \
  -o deployed/spin_nosoc_last
```

This writes two companion files bound by SHA-256 (`format = "spingnn2.contract.v1"`,
`format_version = 1`):

- `deployed/spin_nosoc.sp2.json` — static execution graph, folded MLP weights,
  sparse Clebsch–Gordan / Wigner-3j tables, Hermite radial/spin splines,
  Cartesian basis matrices, and outer rescale layers.
- `deployed/spin_nosoc.sp2.bin` — 8-byte-aligned little-endian weight blob.

Always keep the `.sp2.json` manifest and its `.sp2.bin` binary together. Every
loader verifies the binary SHA-256 before decoding weights.

## 3. Pre-Compile Offline JIT Artifacts

Modern LAMMPS (`pair_style spingnn2(/kk)`) and Python `execution_mode="direct"`
load **pre-compiled** kernel artifacts and never invoke NVRTC or a C++ compiler
at simulation time. Prepare the artifact once for your target backend,
architecture, and precision:

```bash
# Multi-GPU bundle (sm_80, sm_86, sm_89, sm_90, sm_120 -> single .bundle.json)
python tools/ci/source_python.py spingnn2-prepare-jit-artifact \
  --model deployed/spin_nosoc.sp2.json \
  --backend cuda --arch all-gpu --precision float32 \
  --library validation/native/build-cuda-all-gpu/libspingnn2.so

# Or single-architecture CUDA (sm80 | sm86 | sm89 | sm90 | sm120)
python tools/ci/source_python.py spingnn2-prepare-jit-artifact \
  --model deployed/spin_nosoc.sp2.json \
  --backend cuda --arch sm120 --precision float32 \
  --library validation/native/build-cuda-sm120/libspingnn2.so

# CPU / OpenMP, FP64
python tools/ci/source_python.py spingnn2-prepare-jit-artifact \
  --model deployed/spin_nosoc.sp2.json \
  --backend cpu --precision float64
```

Key flags for `spingnn2-prepare-jit-artifact`:

- `--model PATH` (required): path to `.sp2.json`.
- `--backend cpu|cuda` (required).
- `--precision float32|float64`: defaults to the exported contract precision.
- `--arch`: CUDA target architecture (`sm80`, `sm86`, `sm89`, `sm90`, `sm120`,
  comma-separated list, or `all-gpu`). Multi-arch specs emit a `bundle_abi = 1`
  `.bundle.json` manifest that dispatches automatically to the active GPU's
  compute capability at runtime with zero NVRTC on execution hosts.
- `--cache-dir PATH`: overrides `SPINGNN2_JIT_CACHE` (default
  `~/.cache/spingnn2/jit`).
- `--library PATH`: locates `libspingnn2_jit_prepare.so` beside
  `libspingnn2.so` (CUDA host NVRTC compilation does not require a GPU or
  driver; set `SPINGNN2_NVRTC_LIBRARY` to select a specific `libnvrtc.so`).
- `--cxx PATH`: overrides `CXX` for CPU OpenMP shared-object compilation.

Copy both `<cache_key>.json` and its referenced `kernel-<sha256>.cubin` (or
`.so`) together when moving to an execution node. The cache directory and
artifact files must be owned by the caller or root and must not be group- or
world-writable.

## 4. Python & ASE Inference (Zero PyTorch / LibTorch)

On any inference host with `spingnn2` installed (`pip install .` or
`pip install '.[ase]'`), neither PyTorch nor `T-e3nn` is required.

### 4.1 Pure-NumPy/SciPy Reference Oracle (`ReferenceModel`)

```python
from spingnn2.reference import ReferenceModel

model = ReferenceModel.load("deployed/spin_nosoc.sp2.json")
out = model.evaluate(graph, tile_atoms=256, radial_mode="spline")

# Stage-1 lattice cache across spin-only updates
cache = model.evaluate_stage1_lattice(graph, tile_atoms=256, radial_mode="spline")
updated = model.evaluate_stage2_spin_cached(cache, new_spins)
```

### 4.2 Native C++/Kokkos Engine (`NativeModel`)

`NativeModel` resolves `libspingnn2.so` via `library=...`, `SPINGNN2_LIBRARY`,
or source-tree build paths (`validation/native/build-cpu/` or
`validation/native/build-cuda-$SPINGNN2_ARCH/`):

```python
from spingnn2.native import NativeModel

with NativeModel.load(
    "deployed/spin_nosoc.sp2.json",
    backend="cuda",
    precision="float32",
    execution_mode="direct",          # "generic" (default), "auto", or "direct"
    artifact="/path/to/artifact.json",
    execution_profile="speed",        # "capacity" (tile<=256) or "speed" (512; 1024 CUDA FP32 DREAM)
    radial_mode="spline",             # "analytic" (default) or "spline"
    allow_tf32=False,                 # strict IEEE FP32 default
    profile=False,
) as model:
    result = model.evaluate(graph)
    print(model.info, model.execution_plan, model.diagnostics)

    with model.evaluate_stage1_lattice(graph) as lattice:
        full_update = model.evaluate_stage2_spin_cached(lattice, new_spins)
    with model.evaluate_stage1_lattice(graph, spin_only=True) as lattice:
        frozen_update = model.evaluate_stage2_spin_cached(
            lattice, new_spins, spin_only=True
        )
```

### 4.3 ASE Calculator (`SpinGNN2Calculator`)

```python
from spingnn2.calculator import SpinGNN2Calculator

atoms.calc = SpinGNN2Calculator(
    "deployed/spin_nosoc.sp2.json",
    backend="cuda",
    precision="float32",
    execution_mode="direct",
    artifact="/path/to/artifact.json",
    execution_profile="speed",
    radial_mode="spline",
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress_voigt = atoms.get_stress()
spin_forces = atoms.calc.get_property("spin_forces", atoms)
```

For magnetic models, attach Cartesian spin vectors via `atoms.arrays["spin"]`
or `atoms.set_initial_magnetic_moments(...)`. Pass `type_names=["Mn", "Ge"]` if
the exported contract does not embed chemical names.

## 5. Modern LAMMPS (`stable_30Sep2026_update1`) Usage

### 5.1 `pair_style spingnn2` & `pair_style spingnn2/kk`

```lammps
units       metal
boundary    p p p
pair_style  spingnn2 precision float32 math ieee tile 512 cache yes profile no
pair_coeff  * * model.sp2.json prepared-artifact.json Fe O
```

| Option | Values | Default | Notes |
| --- | --- | --- | --- |
| `precision` | `float32`, `float64` | `float32` | Must match the prepared JIT artifact precision |
| `math` | `ieee`, `tf32` | `ieee` | `tf32` is opt-in and valid only for CUDA `float32` |
| `tile` | positive integer | planner default | Overrides native center-atom tile width |
| `cache` | `yes` / `no` | `yes` | Controls Stage-1 lattice caching for `fix spin/sib` and frozen `fix spin/cfi` |
| `profile` | `yes` / `no` | `no` | Enables synchronized Stage-1 / Stage-2 timers in `compute spingnn2/timing` |
| `cuda_graph` | `yes` / `no` | `no` | Enables CUDA Graph replay for stable device-neighbor evaluations in `spingnn2/kk` |

On each lattice update, `spingnn2/kk` compacts candidate neighbor lists on GPU
to physical edges (`r < r_max`) before Stage 1, allowing atoms to cross `r_max`
cleanly between neighbor-list rebuilds with negligible skin overhead.

### 5.2 Coupled Spin–Lattice MD (`fix spin/sib`, `fix spin/sib/npt`, `fix spin/cfi`)

```lammps
units       metal
boundary    p p p
atom_style  spin
atom_modify map array
read_data   system-spin.data

pair_style  spingnn2 precision float32 math ieee profile yes
pair_coeff  * * deployed/spin_nosoc.sp2.json prepared-artifact.json Mn Ge

# Standard SIB (2 pair evals/step, Stage-1 cached at midpoint):
fix         integrate all spin/sib 300.0 0.1 91081 lattice moving

# Or Cached-Field Integrator (1 pair eval/step, alias spin/lpsib):
# fix       integrate all spin/cfi 300.0 0.1 91081 lattice moving
# Or CFI with canonical-volume isotropic barostat (NPH/NPT, multi-MPI & Kokkos):
# fix       integrate all spin/cfi 300.0 0.1 91081 lattice moving ou 300.0 0.05 91082 box iso 0.0 100.0 box_ou 300.0 0.1 91083

# Or spin-lattice NPT (spin/sib_npt alias also accepted):
# fix       integrate all spin/sib/npt temp 300 300 0.02 iso 1 1 0.05 alpha 0.1 seed 91081

compute     timers all spingnn2/timing
compute     gpst all spin/gpst cutoff 5.0
thermo      10
thermo_style custom step temp pe etotal c_timers[*]
timestep    0.0001
run         100
```

- `fix spin/sib <T_spin> <alpha> <seed> lattice frozen|moving` and
  `fix spin/cfi` use stateless atom-ID- and timestep-keyed thermal noise so
  trajectories match across MPI ranks and CPU/CUDA backends.
- `compute ID group spin/gpst [cutoff r_cut]` evaluates 7 Geometric-Probe Spin
  Temperature `(N_c, D_c)` pairs over magnetic atoms (`sp[i][3] > 0`) using
  exact sphere-IBP identities valid for nonlinear/SIA/ML magnetic potentials.
  For offline trajectory analysis (CF-GMM, Fieller CI, delete-one seed jackknife,
  Geyer $\tau$, C99 OpenMP fused shell kernel), use `spingnn2.gpst` and CLIs
  `spingnn2-gpst-moments` / `spingnn2-gpst-infer` (plus `spingnn2.spin_temp`
  for online LAMMPS vector pooling).

### 5.3 DREAM Electric-Field MD (`atom_style bec` & `fix efield coupling bec`)

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
thermo      10
thermo_style custom step pe fmax c_dip c_dip[1] c_dip[2] c_dip[3] c_dip[4]
timestep    0.0005
run         100
```

### 5.4 e2SLP Uncertainty Readouts (`fix pair`)

```lammps
# Force-evidence model (Allegro_e2IP):
fix evidence all pair 1 spingnn2 f_sigma 0 f_Uepi 0 f_nu 0 f_kappa 0

# Spin-force-evidence model (Allegro_soc_e2IP):
# fix evidence all pair 1 spingnn2 sf_sigma 0 sf_Uepi 0 sf_nu 0 sf_kappa 0

dump traj all custom 10 e2slp.dump id type x y z fx fy fz f_evidence[*]
dump_modify traj sort id
```

### 5.5 Profiling with `compute spingnn2/timing`

`compute ID all spingnn2/timing` exposes 6 rank-maximum counters:
`[1]` exclusive pair wall time (s), `[2]` halo + midpoint comm time (s),
`[3]` Stage-1 time (s, when `profile yes`), `[4]` Stage-2 time (s, when
`profile yes`), `[5]` Stage-1 cache hits, `[6]` total evaluations.
`compute ID all spingnn2/timing graph` exposes 3 CUDA Graph counters:
`[1]` captures, `[2]` replays, `[3]` fallbacks.

### 5.6 Multi-GPU MPI & Host Kokkos Launches

```bash
# Multi-GPU CUDA (one MPI rank per physical GPU UUID)
# Note: on non-GPU-aware MPI stacks (e.g., Intel MPI / MPICH without GPU offload),
# pass `-pk kokkos gpu/aware off neigh half` instead of `gpu/aware on`.
mpirun -n 4 tools/lammps/spingnn2_gpu_rank.py \
  /path/to/install/bin/lmp_spingnn2 \
  -k on g 1 -sf kk -pk kokkos gpu/aware on neigh half \
  -in in.spin_md

# Host Kokkos / OpenMP (CPU build)
OMP_NUM_THREADS=8 mpirun -n 2 /path/to/cpu-install/bin/lmp_spingnn2 \
  -k on t 8 -sf kk -in in.spin_md
```

## 6. SpinGNN2 v2 Pure-PyTorch Symmetry, `PhysMuon` & `torch.compile` Training

The `spingnn2.symmetry` and `spingnn2.v2` modules provide a standalone,
zero-`e3nn` pure-PyTorch implementation of decoupled
$O(3)_{\text{lat}} \times O(3)_{\text{spin}}$ and coupled
$O(3) \times \mathbb{Z}_2^T$ equivariant potentials designed for full-graph
`torch.compile` training with first-party `PhysMuon` and `Muon` optimizers:

```python
from spingnn2.v2 import PhysMuon, WSDCosineScheduler, build_v2_soc

# Also available: build_v2_nosoc, build_v2_born, build_v2_e2slp
model = build_v2_soc(num_types=2, r_cut=5.0, c0=64).cuda()
compiled = model.compile_fullgraph(dynamic=True)  # or model.compile_bucketed(...)
optimizer = PhysMuon(
    model.parameters(),
    lr=1e-3,
    named_parameters=model.named_parameters(),
    cuda_graph=True,  # whole-step CUDA Graph capture for 5-step quintic NS
)
scheduler = WSDCosineScheduler(
    optimizer, warmup_steps=50, stable_steps=750, decay_steps=200
)
```

`PhysMuon` and `Muon` are also integrated directly into the v1 YAML trainer
(`optimizer_name: PhysMuon` or `Muon`, with optional `optimizer_preset`,
`grad_accumulation_steps` for fluctuation-calibrated `iso`, and automatic
pre-clip `grad_norm` routing into `spike_guard`).

Run the matched v1-vs-v2 GPU training benchmark with:

```bash
python benchmarks/bench_v2_train.py --families nosoc soc born e2slp --atoms 64 128 256 512
```

## 7. Verified Configuration and Hardware Limits

Every subsystem is stress-tested across degenerate graphs, odd/non-divisible
channel widths, shared-memory ceilings, and extreme thermodynamic regimes
(`tests_native/test_stress_limits.py`, `tests_native/test_stress_cfi_gpst.py`,
`tests_v2/test_stress_v2_and_physmuon.py`, `tests_v2/test_stress_symmetry.py`):

| Subsystem | Verified Operating Envelope & Automatic Fallbacks | Explicit Guards / Constraints |
| --- | --- | --- |
| **Native Engine & NVRTC JIT (`libspingnn2`)** | `mul = 1..64` (incl. primes `1, 3, 5, 7, 13, 17`), odd MLP widths (`33, 65, ...`), `l_max = 1..3`, `num_layers = 1..4`, `num_types = 1..12`; `E = 0`, `N = 1`, empty MPI rank `N_owned = 0`; exact 256-edge CUDA Graph buckets + `5.25×` density spike + NaN-safe padded slots + triclinic box. Per-CTA `env` smem auto-tiers: `<= 48 KiB` static, `(48, 96] KiB` dynamic opt-in (fits `99 KiB = 101376 B` RTX 4090/5090 cap), `> 96 KiB` global-memory fallback (`109.4 KiB` verified `< 1e-10`). `IrrepLinear` packed scratch caps at `16M` elements with per-group fallback. | `tile_atoms <= 65535` and edge/head tensors `< INT_MAX` elements (enforced in `ExecutionPlan::size_model()`). Multi-rank Kokkos on non-GPU-aware MPI requires `-pk kokkos gpu/aware off`. |
| **LAMMPS `fix spin/cfi(/kk)` & `compute spin/gpst(/kk)`** | `N_mag = 0`, `N_mag = 1`, `E = 0`, empty MPI ranks receiving migrating atoms across PBC + `neigh_modify delay 4` + `sort`, `T = 0.01..5000 K`, `alpha = 0..10`, 500-step NPH/NPT stability, multi-`run pre no`, triclinic moving CFI + GPST (`< 1e-11`), GPST C99 OpenMP `G = 1..16` bit-deterministic across `OMP_NUM_THREADS`. | `box iso P W [box_ou]` requires an orthogonal periodic 3D box (triclinic is cleanly rejected; nonpositive volume raises an explicit error). |
| **Pure-PyTorch `spingnn2.v2` & `PhysMuon`** | `lmax in (1, 2, 3)` across all 4 families (`v2_nosoc`, `v2_soc`, `v2_born` incl. `c2e = 0`, `v2_e2slp`), `num_types >= 1`, `E = 0` isolated, `N = 1` self-image, `N = 2048` / `60k` edges (`< 12 GiB` VRAM with `edge_amp` + `edge_checkpoint`), `compile_fullgraph(dynamic=True)` and `compile_bucketed` (`<= 4` buckets; out-of-bucket shapes fall back cleanly to eager `_forward_autograd` with zero new CUDA Graph pools). `PhysMuon` with `spike_guard > 0` / `skip_nonfinite=True` skips `NaN`/`Inf` steps cleanly without corrupting weights/momentum/`iso`, supports async `grad=None` subsets across `bm_start` under `cuda_graph=True`. | `SpinGNNV2` validates `lmax in (1, 2, 3)`, `num_types >= 1`, `c0 >= max(c1, c2, c3, c_spin)` (raises explicit `ValueError`). For `compile_bucketed`, leave `>= +2` dummy-atom headroom (`n_atoms <= ab - 2` when `n_edges < eb`). |
| **`spingnn2.symmetry`** | `wigner_3j`, `SparseCGContract` (unrolled `nnz <= 128`, factorized/fallback `nnz > 128`), and `Couple` support arbitrary `l >= 0`; verified for `batch = 0`, empty irreps `""`, `mul = 1..64`, non-contiguous inputs, `[0, 0, 0]` & `~1e-12` tiny vectors through 2nd-order autograd in `fp64`/`fp32`/`bf16`, mixed-dtype promotion preserving 53-bit FP64 `_apply()`, and `torch.compile(fullgraph=True, dynamic=True)` with `0` graph breaks. | Polynomial `SphericalHarmonics` and `BipolarSphericalHarmonics` support `l = 0..4`. |
