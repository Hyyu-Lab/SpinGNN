# SpinGNN2 Skills Summary

SpinGNN2 ships 8 reusable AI assistant skills under `skills/` covering the full
lifecycle from environment setup and training (v1 SpinGNN++ YAML with
`PhysMuon`/`Muon`/`Adam` and pure-PyTorch `spingnn2.v2` `torch.compile`)
through `.sp2.json`/`.sp2.bin` contract export, multi-GPU `all-gpu` fatbin &
`bundle_abi = 1` JIT preparation, GPST spin-temperature inference
(`spingnn2.gpst`), and modern LAMMPS (`stable_30Sep2026_update1`,
`pair_style spingnn2(/kk)`) molecular dynamics.

## Skill Inventory

| Skill | File | Scope |
| --- | --- | --- |
| `spingnn2` | [`spingnn2/SKILL.md`](spingnn2/SKILL.md) | Native C++17/Kokkos engine (`all-gpu` `sm80`–`sm120` fatbin), export, `bundle_abi = 1` JIT, `spingnn2.v2` + `PhysMuon`, GPST CLI, Python/ASE `NativeModel`, modern LAMMPS |
| `spingnn-install` | [`spingnn-install/SKILL.md`](spingnn-install/SKILL.md) | Python 3.10 / CUDA 12.8 / T-e3nn, `libspingnn2` (`cpu`/`all-gpu`/`sm*`), GPST wheel, `lmp_spingnn2` overlay |
| `spingnn-usage` | [`spingnn-usage/SKILL.md`](spingnn-usage/SKILL.md) | Comprehensive SpinGNN2 data, training (`PhysMuon`/`Muon`), deployment, GPST, and LAMMPS command reference |
| `structure-only-e2e` | [`structure-only-e2e/SKILL.md`](structure-only-e2e/SKILL.md) | Non-magnetic `Allegro` / `Allegro_e2IP` energy, force, and stress workflows |
| `spin-nosoc-e2e` | [`spin-nosoc-e2e/SKILL.md`](spin-nosoc-e2e/SKILL.md) | Decoupled $O(3)_{\text{lat}} \times O(3)_{\text{spin}}$ magnetic workflows (`Allegro_nosoc`, `v2_nosoc`, `fix spin/sib`, `fix spin/cfi` + `box iso`, `compute spin/gpst`) |
| `spin-soc-e2e` | [`spin-soc-e2e/SKILL.md`](spin-soc-e2e/SKILL.md) | Coupled $O(3) \times \mathbb{Z}_2^T$ SOC magnetic workflows (`Allegro_soc`, `v2_soc`, anisotropy, DMI) |
| `born-effective-charge-e2e` | [`born-effective-charge-e2e/SKILL.md`](born-effective-charge-e2e/SKILL.md) | DREAM Born effective charges (`Allegro_born`, `v2_born`, `atom_style bec`, `fix efield coupling bec`) |
| `e2slp-evaluation` | [`e2slp-evaluation/SKILL.md`](e2slp-evaluation/SKILL.md) | Active-learning uncertainty scoring (`Allegro_e2IP`, `Allegro_soc_e2IP`, `v2_e2slp`, `fix pair`) |

See [`skills/README.md`](README.md) and [`skills/TESTING_REPORT.md`](TESTING_REPORT.md).
