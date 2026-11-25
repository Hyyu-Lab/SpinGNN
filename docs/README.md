# SpinGNN2 Documentation Index

Start with [README.md](../README.md) for an overview and quick start.

## SpinGNN2 User & Component Guides

- [SpinGNN2 User Guide](SPINGNN2_GUIDE.md) — End-to-end export, offline
  single-arch & `bundle_abi = 1` multi-GPU JIT, `PhysMuon`/`Muon` training,
  Python/ASE inference, GPST toolkit (`spingnn2.gpst`, `spingnn2-gpst-moments`,
  `spingnn2-gpst-infer`), and modern LAMMPS (`pair_style spingnn2(/kk)`,
  `fix spin/sib(/npt)/kk`, `fix spin/cfi(/kk)` with `box iso P W [box_ou]`,
  `compute spin/gpst(/kk)`, DREAM BEC, e2SLP, `compute spingnn2/timing`)
- [Architecture & Roadmap](ARCHITECTURE_ROADMAP.md) — Native C++17/Kokkos
  engine design, multi-GPU fatbin architecture, v2 magnetic symmetry,
  `PhysMuon`, and `torch.compile` training architecture
- [Python & ASE Runtime](../src/spingnn2/README.md) — `.sp2.json`/`.sp2.bin`
  contract (`spingnn2.contract.v1`), NumPy reference oracle, `NativeModel`,
  `SpinGNN2Calculator`, and `spingnn2.gpst`
- [Native C ABI Library](../libspingnn2/README.md) — `libspingnn2.so`
  (single-arch & `all-gpu` `sm_80`–`sm_120` fatbin), `sp2_*` C ABI, Kokkos
  workspace, and offline NVRTC/Host JIT preparation
- [Modern LAMMPS Overlay](../lammps_spingnn2/README.md) — Pinned
  `stable_30Sep2026_update1` overlay, patches, Kokkos pair/fixes, and builds

## Validation & Performance

- [Validation Summary](VALIDATION.md) — Contract oracle parity, native
  CPU/CUDA & live `all-gpu` fatbin parity (CPU/`kk-cpu`, RTX 5090 `sm_120`,
  RTX 4090 `sm_89`), JIT/bundle security, cutoff-crossing, SIB/CFI `box iso`/GPST,
  DREAM hard-ASR, relocation, `PhysMuon`, and zero-LibTorch verification
- [Performance Benchmarks](PERFORMANCE.md) — RTX 5090 & RTX 4090 direct
  inference (up to 2.0M atoms/GPU), thermal LAMMPS MD with GPU edge compaction,
  bucketed CUDA Graph replay, 2.05M-atom 4-GPU weak scaling, and v2 + `PhysMuon`
  training speedups

## Installation & Environment

- [INSTALL.md](../INSTALL.md) — Training environment, `libspingnn2`, and
  `lmp_spingnn2` build instructions
- [CUDA 12.8 Environment](INSTALL_CUDA128.md) — Pinned Python 3.10 / PyTorch
  2.11+cu128 / T-e3nn v1.0 setup
- [Source Checkout Guide](SOURCE_CHECKOUT.md) — Running directly from the
  repository with `tools/ci/source_python.py`
- [T-e3nn Integration](TE3NN_INTEGRATION.md) — T-e3nn v1.0 pin and THP notes
- [Docker Container](../docker/README.md) — Containerized CUDA 12.8 build
- [Release Notes](OPEN_SOURCE_RELEASE.md) — Packaging and license summary

## Training, AI Skills & Development

- [Example Configurations](../example/README.md) ·
  [Training Tutorial](../example/TUTORIAL_TRAINING.md) ·
  [e2SLP Guide](../example/E2IP.md)
- [AI Assistant Skills](../skills/README.md)
- [CI & Testing Scripts](../tools/ci/README.md) ·
  [Contributing](../CONTRIBUTING.md)
