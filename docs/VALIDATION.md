# SpinGNN2 Validation Summary

This document summarizes the verified numerical, physical, runtime, and
packaging checks for SpinGNN2 across the compact contract oracle, native
Kokkos C ABI library (`libspingnn2`), offline JIT artifacts, modern LAMMPS
overlay (`lmp_spingnn2` on `stable_30Sep2026_update1`), and the pure-PyTorch
v2 magnetic symmetry and training stack.

## 1. Contract & Reference Oracle Parity

The `.sp2.json` + `.sp2.bin` exporter (`spingnn2.contract.v1`) and pure
NumPy/SciPy `ReferenceModel` are verified against PyTorch eager evaluation
across all five exported families (`Allegro_nosoc`, `Allegro_soc`,
`Allegro_born` with none/soft/hard ASR, `Allegro_e2IP`, `Allegro_soc_e2IP`):

- **FP64 parity**: `< 1e-11` across total/atomic energy, forces, spin forces,
  global/atomic virial, stress, Born effective charges, and evidential fields.
- **FP32 parity**: `< 1e-5` across physical and evidential outputs.
- **Isolation & integrity**: Isolated subprocess tests confirm
  `ReferenceModel`, `NativeModel`, and `SpinGNN2Calculator` import zero
  training packages (`torch`, `e3nn`, `nequip`, `allegro`), and corrupted
  binary payloads are rejected via SHA-256 verification.
- **VASP 5 & VASP 6 data pipeline**: `tests_native/test_vasp_spin_io.py`
  verifies multi-core (`--cpu N`, `spawn`-safe) converged-only extraction,
  automatic VASP 5 (`1×` → `×2`) vs VASP 6 (`2×` → `×1`) and Mode 4 linear
  penalty detection with `F10.5` printing-rounding bounds, `SAXIS` rotation,
  $E_p$ subtraction, sparse nonmagnetic handling, merged Fortran columns,
  VASP 5.3/5.4 without `INCAR:` echo, BEC/piezo tensors, and `vasprun.xml`.
- Suite: `tests_native/test_contract_oracle.py`, `tests_native/test_scalar_mlp.py`,
  `tests_native/test_vasp_spin_io.py`.

## 2. Native C++17 / Kokkos Engine, Multi-GPU Fatbin & Offline JIT Parity

`libspingnn2.so` is validated on CPU/OpenMP, single-arch CUDA builds, and
unified **multi-architecture fat binaries** (`--arch all-gpu`: `sm_80`, `sm_86`,
`sm_89`, `sm_90`, `sm_120` + `compute_120` PTX verified via `cuobjdump`) with
runtime compute-capability `bundle_abi = 1` cubin dispatch:

- **Strict IEEE FP32 (`CUBLAS_COMPUTE_32F`) vs PyTorch eager**: max error
  `<= 9.06e-6` (and `< 5e-5` across all P1–P5 optimized paths including FP32
  Hermite splines and binary-squaring polynomial envelopes).
- **FP64 vs PyTorch eager (CPU & CUDA)**: max error `<= 2.49e-14`.
- **Bucketed CUDA Graph replay & density spikes**: `tests_native/test_perf_bucket_graph.py`
  (9/9 PASS on RTX 5090 & RTX 4090) verifies 100% post-warmup CUDA Graph replay
  across both growing and shrinking `E_tile` within 256-edge buckets, zero
  leakage from padded edge slots, `2.28×` density-spike dynamic reallocation,
  and `tile_atoms <= 65535` guards.
- **Stage-1 lattice caching**: Moving-lattice and frozen-lattice cached
  evaluations match full uncached evaluations within machine precision, with
  bounded device cache budgets scaled proportionally to GPU VRAM.
- **JIT artifact & bundle security**: Prepared `.cubin` / `.so` plugins and
  `bundle_abi = 1` multi-arch bundle manifests enforce ABI version, contract
  SHA-256, precision, architecture, basename-only non-escaping member paths,
  ELF header, and strict directory/file ownership & permission checks before
  `dlopen` / `cuModuleLoadData` (`tests_native/test_multigpu_arch.py`).
- **Zero runtime compiler / LibTorch**: `strace` execution audits confirm
  zero compiler invocations (`g++`, `nvcc`, `nvrtc`) and zero loads of
  `libtorch`, `libc10`, or `libnvrtc` during inference.
- Suites: `tests_native/test_native_engine.py`, `tests_native/test_jit_artifacts.py`,
  `tests_native/test_jit_runtime.py`, `tests_native/test_perf_bucket_graph.py`,
  `tests_native/test_multigpu_arch.py`.

## 3. Modern LAMMPS (`lmp_spingnn2`) Qualification

The external overlay targets LAMMPS `stable_30Sep2026_update1` (tag object
`17ea1b9effad496ec60d0f079e8b0aaa4c5b6d89`, commit
`098ffdefa525cabc82272925de02e9884f58c657`), verified with `lammps_spingnn2/apply.py --check`.

### Physical cutoff crossing & CUDA Graph replay

- GPU-side `r < r_max` compaction filters LAMMPS skin candidates on device
  before Stage 1.
- Cutoff-crossing regression moves a neighbor across `r_max` under a 1.5 Å
  skin with zero neighbor rebuilds while keeping an isolated atom present;
  SOC and DREAM recorded `max_abs = 0` between ordinary and `cuda_graph yes`
  execution in both FP32 and FP64 on RTX 5090 and RTX 4090.
- Allocation audits confirm 0 B host copies on cached SIB midpoints and zero
  steady-state CUDA allocations/frees after shape warmup.

### Frozen-lattice SIB geometry guard & MPI parity

- When an active fix (e.g. `fix recenter/kk`) modifies coordinates or box
  geometry during a `lattice frozen` run, `pair_style spingnn2(/kk)`
  automatically invalidates Stage-1 reuse so cached and uncached trajectories
  match identically across all steps (both `cuda_graph no` and `yes`).
- Multi-rank MPI domain-decomposition parity (1 vs 2 and 1 vs 4 ranks,
  `newton on` and `newton off`): `<= 4.44e-16` (FP64) and `<= 2.09e-7` (FP32);
  stateless stochastic SIB matches to `<= 5.55e-16` across rank counts.
- DREAM hard-ASR under `-pk kokkos gpu/aware off` (1 vs 2 MPI ranks): max
  error `4.44e-16` (CPU), `1.44e-15` (sm89), `1.13e-15` (sm120).

### Spin integrators (`fix spin/sib`, `fix spin/cfi`) & Spin temperature (`compute spin/gpst`)

- `fix spin/cfi(/kk)` (alias `spin/lpsib(/kk)`) performs 1 pair evaluation per
  step vs 2 for `fix spin/sib(/kk)`, fuses non-`box_iso` Kokkos stepping into
  2 kernels with 0 per-step blocking D2H syncs, preserves `|s_i| = 1` to
  `< 1e-14`, achieves 2nd-order NVE convergence, supports multi-MPI canonical-volume
  `box iso P W [box_ou T damp seed]` barostatting (preserving `image[i]` across
  periodic boundary crossings under `neigh_modify delay > 0` with zero `verlet.cpp`
  patches), tracks DREAM BEC dipoles across box rescalings, and reuses Stage-1
  lattice caches in frozen mode.
- `compute spin/gpst(/kk)` (register-only Legendre recurrence, 0 B/thread shared
  memory) computes 7 sphere-IBP `(N_c, D_c)` Geometric-Probe Spin Temperature
  pairs strictly over magnetic atoms (`sp[i][3] > 0`) using `f_i = hbar * fm_i`,
  matching the exact Python oracle to `< 1e-12` (FP64). The companion first-party
  `spingnn2.gpst` package (43/43 tests in `tests_native/test_gpst_full.py`)
  provides CF-GMM, Fieller CI, delete-one seed jackknife, Geyer $\tau$, scale-aware
  zero-signal gating, and a bit-deterministic OpenMP C99 fused shell kernel
  (`gpst_edge_moments_fused`, `< 1e-13` parity vs pure NumPy).

### Trained-checkpoint parity & evidential MD

- Exported `Allegro_e2IP` checkpoints trained on 64-atom Si AIMD frames
  matched eager PyTorch on held-out frames to `5.68e-14` (physical) /
  `2.22e-15` (evidence) in FP64 and `3.05e-5` (physical) / `5.96e-7`
  (normalized evidence) in FP32.
- 4,096-atom e2SLP trajectories extracted `f_sigma`, `f_Uepi`, `f_nu`,
  `f_kappa` (and `sf_*`) via `fix pair` with normalized error `<= 8.30e-6`
  against fresh native evaluations after neighbor rebuilds.

### Relocation & Zero-LibTorch packaging

- All CPU, `sm89`, `sm120`, and unified `all-gpu` fat-binary install trees pass
  stripped-environment relocation checks (`env -i PATH=/usr/bin:/bin ldd` and
  `lmp_spingnn2 -h`) after copying `bin/lmp_spingnn2` and `lib*/libspingnn2.so`
  to a new prefix, with zero `libtorch`, `libc10`, or `libnvrtc` dependencies.
- **Unified `all-gpu` fat-binary cross-architecture qualification**: The exact
  same `all-gpu` build (`libspingnn2.so`: 23 cubins × 5 architectures + 22
  `compute_120` PTX; `lmp_spingnn2`: 562 cubins × 5 architectures + 561 PTX;
  0 NVCC `#20011-D` overlay warnings; `spingnn2/gpst/_edge_kernel.so` bundled in
  wheel) plus the exact same 10 `bundle_abi = 1` multi-arch manifests (50 unique
  cubins) passed live end-to-end qualification across:
  - **CPU + Kokkos-OpenMP (`kk-cpu`, 1 & 2 MPI)**: `49/49` unit/GPST/multi-GPU,
    `54/54` CPU host, `55/55` `kk-cpu`.
  - **NVIDIA RTX 5090 (`sm_120`, 1 & 2 GPU MPI)**: `35/35` unit/JIT/bucket-graph/neighbors,
    `58/58` LAMMPS (`0` failed, `0` skipped, incl. CFI `box iso`, DREAM BEC
    dipole conservation, GPST, GPU-aware MPI).
  - **NVIDIA RTX 4090 (`sm_89`, same `all-gpu` binary & `bundle_abi = 1` manifests)**:
    `32/32` unit/bucket-graph/neighbors, `50/50` LAMMPS (`0` failed).
- Suites: `tests_native/test_lammps_*.py`, `tests_native/test_spin_cfi.py`,
  `tests_native/test_spin_gpst.py`, `tests_native/test_gpst_full.py`,
  `tests_native/test_multigpu_arch.py`, `tests_native/test_perf_bucket_graph.py`,
  `tests_native/test_audit_evidence.py`.

## 4. SpinGNN2 v2 Symmetry, `PhysMuon` & `torch.compile` Training Verification

- **Exact R1–R5 magnetic symmetry (`tests_v2/test_symmetry_r1_r5.py`, 15/15 PASS)**:
  recursive Wigner-D verification across decoupled
  $O(3)_{\text{lat}} \times O(3)_{\text{spin}}$ and coupled
  $O(3) \times \mathbb{Z}_2^T$ (`< 1e-13` FP64), register-fused
  `SparseCGContract` / `SparseBispectrumContract`, closed-form
  `bipolar_spin_pair_j1_j2`, structured 2D `EquivariantLinear.weight_blocks`
  equivariance after `PhysMuon(iso=True)` updates (`< 1e-13`), FP32 default
  buffers with exact 53-bit FP64 `_apply()` re-materialization, and
  `SpinScale._load_from_state_dict` checkpoint preservation.
- **Full-graph `torch.compile` models & `PhysMuon` (`tests_v2/test_v2_models.py`,
  `tests_v2/test_v2_compile_and_train.py`, `tests_v2/test_physmuon.py` — 37/37 +
  20/20 post-fix PASS on GPU; 0 dynamic recompiles; 86/86 v1 trainer regression
  PASS)**: FP64 finite-difference forces/virial/magnetic torques (`< 1e-7`) across
  all variants; SOC zero-init exactness; collinear smoothness; nonmagnetic exact
  zero 1st/2nd spin derivatives; `PhysMuon` all 5 switches + `spike_guard` +
  microbatch/DDP `iso` + whole-step CUDA Graph parity (`< 1e-5` FP32 / `< 1e-11`
  FP64 vs eager); zero-leakage capped-bucket `compile_bucketed`; v1 YAML
  `PhysMuon`/`Muon` smoke; fresh-subprocess verification that importing
  `spingnn2.symmetry` and `spingnn2.v2` loads zero `e3nn`/`nequip` modules.

## 5. Adversarial Stress & Hardware-Limit Qualification

Dedicated boundary and failure-mode suites harden every layer against silent
crashes, NaNs, OOMs, alignment issues, shared-memory caps, and empty-rank
degeneracies (see [Verified Configuration and Hardware Limits](SPINGNN2_GUIDE.md#7-verified-configuration-and-hardware-limits)):

- **Native CUDA/CPU/JIT/LAMMPS (`tests_native/test_stress_limits.py`, 54/54 PASS
  on RTX 5090 `sm_120` & RTX 4090 `sm_89`, 13/13 CPU, 6/6 empty-rank 2-GPU MPI)**:
  34 adversarial shapes (`mul = 1..64` incl. primes, odd MLP widths `33/65`,
  `l_max = 1..3`, `num_layers = 1..4`, `num_types = 1..12`), 3-tier per-CTA
  `env` shared memory (`<= 48 KiB` static, `57.6 KiB` opt-in within `99 KiB`
  RTX 4090/5090 hardware cap, `109.4 KiB` global-memory fallback `< 1e-10`),
  `16M` `IrrepLinear` fallback, `E = 0` / `N = 1` / `N_owned = 0` empty MPI
  ranks, exact 256-edge CUDA Graph bucket boundaries + `5.25×` density spike +
  NaN-poisoned padded VRAM + 500-step `0.000 MB` VRAM leak, triclinic box,
  `2000 K` thermal MD, and 500-step NVE (`<= 5.2e-8 eV/atom`).
- **LAMMPS CFI & GPST (`tests_native/test_stress_cfi_gpst.py`, 24/24 PASS on
  CPU 1/2/4-MPI, `kk-cpu` 1/2/4-MPI, and RTX 5090 2-GPU MPI)**: `N_mag = 0/1`,
  `E = 0`, empty MPI ranks receiving migrating atoms across PBC + `box iso` +
  `delay 4` + `sort`, triclinic `box iso` rejection & triclinic moving GPST
  (`< 1e-11`), 500-step NPH/NPT stability, `T = 0.01..5000 K`, `alpha = 0..10`,
  and GPST C99 OpenMP `E = 0`, `T = 1`, `G = 1..16` thread determinism.
- **`spingnn2.v2` & `PhysMuon` (`tests_v2/test_stress_v2_and_physmuon.py`,
  48/48 PASS on RTX 5090, 42/42 CPU, plus 38/38 regression on RTX 4090)**:
  `lmax in (1, 2, 3)`, `num_types in (1, 3, 8)`, odd/wide channels, `E = 0`,
  `N = 1`, `N = 2048` / `60k` edges (`< 12 GiB` VRAM), out-of-bucket eager
  fallback under `compile_bucketed` (`_bucket_overflow_count == 2`, 0 extra
  CUDA Graph pools), `PhysMuon` `lr = 0`, `wd = 0.1`, `1000×` spike, safe
  `NaN`/`Inf` step skipping (`skip_nonfinite`), async `grad=None` subsets across
  `bm_start` under `cuda_graph=True`, 50-step noisy GPU stability, and v1 YAML
  `PhysMuon`/`Muon` smoke.
- **`spingnn2.symmetry` (`tests_v2/test_stress_symmetry.py`, 13/13 PASS +
  15/15 R1–R5 = 28/28 PASS)**: `l = 0..4` polynomial harmonics & arbitrary-`l`
  CG/Couple, dtype-safe 2nd-order autograd on `[0, 0, 0]` & `~1e-12` vectors in
  `fp64`/`fp32`/`bf16`, `batch = 0`, empty irreps `""`, `mul = 1..64`, mixed
  dtypes preserving 53-bit FP64 `_apply()`, and `torch.compile(fullgraph=True,
  dynamic=True)` with `0` graph breaks.
