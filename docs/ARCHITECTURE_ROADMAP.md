# SpinGNN2 Architecture & Roadmap

SpinGNN2 unifies two tightly coupled layers around strict 1-hop physical
locality, exact magnetic & time-reversal symmetries, and zero-LibTorch
production deployment:

1. **Production Native Inference & Modern LAMMPS Stack (`libspingnn2` +
   `lammps_spingnn2`)** — Center-atom-tiled C++17 / Kokkos 4.7.02 C ABI engine
   (`sp2_*`) with multi-GPU `all-gpu` fat binaries (`sm_80`–`sm_120` +
   `compute_120` PTX), offline NVRTC/Host JIT artifacts and `bundle_abi = 1`
   runtime CC dispatch, `cublasLt` heuristic caching (strict IEEE FP32
   default), FP32 Hermite radial splines, warp-per-center scatter, bucketed
   `E_pad` CUDA Graph replay, Stage-1 lattice caching, GPU physical-edge
   compaction, and modern LAMMPS (`stable_30Sep2026_update1`) integration.
2. **Next-Generation Pure-PyTorch Magnetic Symmetry, `PhysMuon` & v2 Training
   Stack (`spingnn2.symmetry` + `spingnn2.v2`)** — Zero-`e3nn`, full-graph
   `torch.compile(fullgraph=True, dynamic=True)` and capped-bucket
   (`compile_bucketed`) magnetic equivariant architectures combining register-fused
   decoupled $O(3)_{\text{lat}} \times O(3)_{\text{spin}}$ density accumulation,
   parameter-free orthogonal $L$-$S$ coupling (`Couple`), singularity-free
   polynomial bipolar spherical harmonics (explicit 2-body DMI, Kitaev, and
   SIA), bounded extrapolation (`density_floor`, `vacuum_ref`), DREAM Born
   effective charges, e2SLP uncertainty heads, and whole-step CUDA-Graphed
   `PhysMuon` / `Muon` / `HybridMuon` optimization (also integrated into the
   v1 YAML trainer).

## Architectural Principles

- **Strict 1-Hop Locality for Million-Atom Production**: Center-atom tiling
  evaluates each tile of center atoms using only its 1-hop neighborhood,
  keeping device workspace bounded regardless of total system size (**2.0M
  atoms on a single 32 GB RTX 5090 at 1.58–2.23 GB peak VRAM**) and requiring
  only standard 1-hop MPI halo exchange (**99.19% 4-GPU weak-scaling
  efficiency** at 2.05M atoms).
- **Exact Magnetic & Spatial Symmetries**:
  - Decoupled $O(3)_{\text{lat}} \times O(3)_{\text{spin}}$ representations
    $(l, p \mid \lambda, t)$ for NoSOC models (invariant under independent spin
    rotation at fixed lattice, with 1-hop bond-angle superexchange coupling and
    paramagnetic coordination `Mw`).
  - Coupled $O(3) \times \mathbb{Z}_2^T$ representations $(l, p, t)$ via
    parameter-free orthogonal `Couple` and zero-initialized thin SOC heads
    (explicit 2-body DMI $\mathbf{D}_{ij} \cdot (\tilde{\mathbf{s}}_i \times
    \tilde{\mathbf{s}}_j)$, symmetric anisotropy, and SIA with zero
    $1/\theta$ collinear singularity).
  - Even-parity $0ee \oplus 1ee \oplus 2ee$ Born effective charge readouts for
    DREAM electric-field dynamics (with soft/hard acoustic sum rule and
    optional spin-dependent magnetoelectric coupling).
- **Stage-1 Lattice Caching & 1-Eval/Step Spin Dynamics**: Lattice-only
  geometry and Stage-1 features are decoupled from spin-dependent Stage-2
  evaluations so frozen-lattice and moving-lattice spin integrators
  (`fix spin/sib(/kk)`, `fix spin/cfi(/kk)` with 2-kernel zero-sync stepping
  and canonical-volume `box iso P W [box_ou]` barostatting) reuse Stage-1 state
  without redundant lattice evaluations.
- **Zero-LibTorch, Zero-Compiler Multi-GPU Execution**: Exported
  `.sp2.json`/`.sp2.bin` contracts (`spingnn2.contract.v1`) and
  offline-compiled NVRTC `.cubin` / `bundle_abi = 1` multi-arch manifests /
  host `.so` plugins are SHA-256 verified at load time. Execution nodes require
  neither PyTorch, LibTorch, nor a C++/CUDA compiler.

## Subsystem Map

| Layer | Location | Key Capabilities |
| --- | --- | --- |
| Magnetic symmetry core | `src/spingnn2/symmetry/` | `DIrrep`/`DIrreps`, `Couple`, fast L=1/2 `SphericalHarmonics`, closed-form `bipolar_spin_pair_j1_j2`, register-fused `SparseCGContract`/`SparseBispectrumContract`, `SpinScale`, `Mw`, 2D-block `EquivariantLinear`, exact Wigner-D verifier |
| Pure-`torch.compile` v2 models & `PhysMuon` | `src/spingnn2/v2/`, `src/core/nequip/utils/optim/physmuon.py` | `v2_nosoc`, `v2_soc`, `v2_born`, `v2_e2slp`, whole-step CUDA-Graphed `PhysMuon`/`Muon`/`HybridMuon` + WSD scheduler, edge `bfloat16` AMP, selective checkpointing, `compile_bucketed` |
| Contract, oracle, JIT prep & GPST toolkit | `src/spingnn2/` | `.sp2.json`/`.sp2.bin` exporter, NumPy/SciPy `ReferenceModel`, `NativeModel`, `SpinGNN2Calculator`, multi-arch `bundle_abi = 1` JIT prep, `spingnn2.gpst` C99 OpenMP fused toolkit (`spingnn2-gpst-moments`, `spingnn2-gpst-infer`) |
| Native C++17 / Kokkos engine | `libspingnn2/` | `libspingnn2.so` (`sp2_*` C ABI, `sm_80`–`sm_120` `all-gpu` fatbin), `libspingnn2_jit_prepare.so`, `cublasLt` heuristic cache, FP32 Hermite splines, warp-per-center scatter, opt-in dynamic smem, bucketed CUDA Graph replay |
| Modern LAMMPS overlay | `lammps_spingnn2/` | `pair_style spingnn2(/kk)`, `fix spin/sib(/npt)/kk`, `fix spin/cfi(/kk)` (`spin/lpsib`) with `box iso P W [box_ou]`, register-only `compute spin/gpst(/kk)`, DREAM `atom_style bec` / `fix efield coupling bec` / `compute bec`, `compute spingnn2/timing` |

## Ongoing Evolution

- **Contract v2 & Native v2 Kernels**: Extending `.sp2.json`/`.sp2.bin` with
  `spingnn2.contract.v2` (quintic Hermite radial splines, SVD low-rank radial
  bases, node-level ACE/bispectrum contractions, Lebedev $S^2$ FFN, and
  zero-edge-tape recomputation backward kernels) while keeping
  `spingnn2.contract.v1` permanently supported.
- **2-Hop Teacher Distillation**: Optional 2-hop magnetic teacher models for
  offline pseudo-labeling into strict 1-hop `v2` production student models.
