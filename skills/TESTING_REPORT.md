# SpinGNN2 Skills Verification Report

All 8 AI assistant skills under `skills/` are verified for YAML frontmatter
validity, broken relative Markdown link / `#anchor` absence, zero legacy
naming remnants, and technical consistency with SpinGNN2:

- **Frontmatter**: All 8 `skills/*/SKILL.md` files declare valid `name` and
  `description` fields matching their directory names.
- **Training & Optimizers**: Consistent with PyTorch 2.11+cu128, pinned
  `T-e3nn` v1.0 (`e3nn==0.5.0+spin`), pure-PyTorch `spingnn2.symmetry` /
  `spingnn2.v2` (zero `e3nn` runtime dependency), and first-party
  `PhysMuon` / `Muon` / `HybridMuon` in both v1 YAML configs and v2.
- **Native Engine & Multi-GPU Deployment**: Consistent with `.sp2.json` /
  `.sp2.bin` (`spingnn2.contract.v1`), `--arch all-gpu` (`sm80`, `sm86`,
  `sm89`, `sm90`, `sm120` + `compute_120` PTX) fat binaries, `bundle_abi = 1`
  multi-arch JIT manifests, and `tools/spingnn2_build.py`.
- **Modern LAMMPS & Spin Thermometry**: Consistent with upstream
  `stable_30Sep2026_update1` (`lmp_spingnn2`), `pair_style spingnn2(/kk)`,
  `fix spin/sib(/npt)/kk`, `fix spin/cfi(/kk)` with canonical-volume
  `box iso P W [box_ou]`, `compute spin/gpst(/kk)`, `spingnn2.gpst`
  (`spingnn2-gpst-moments`, `spingnn2-gpst-infer`), DREAM `atom_style bec` /
  `fix efield coupling bec`, e2SLP `fix pair`, and `compute spingnn2/timing`.
- **Test Suites**: Commands reference the active repository suites
  (`tests_ci/`, `tests_v2/`, `tests_native/`, `tools/ci/test_inventory.py`)
  and verified benchmarks in [`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md)
  and [`docs/VALIDATION.md`](../docs/VALIDATION.md).
