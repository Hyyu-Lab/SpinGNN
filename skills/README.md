# SpinGNN2 AI Assistant Skills

These 8 reusable guides give AI coding assistants (Codex, Claude Code,
Antigravity, Cursor, etc.) repository-specific workflows from VASP 5 & VASP 6
dataset preparation (`spingnn-data`) through training (v1 SpinGNN++ YAML with
`PhysMuon` / `Muon` / `Adam` and pure-PyTorch `spingnn2.v2` `torch.compile`),
`.sp2.json` / `.sp2.bin` checkpoint export, offline single-arch & multi-GPU
`bundle_abi = 1` JIT compilation, zero-LibTorch Python/ASE inference, GPST
spin-temperature analysis (`spingnn2.gpst`), modern LAMMPS
(`stable_30Sep2026_update1`) deployment, and validation checks.

## One-Command Install (`npx skills`)

```bash
# Interactive install (auto-detects installed AI coding agents)
npx skills add Hyyu-Lab/SpinGNN

# One-shot non-interactive global install of all 8 skills
npx skills add Hyyu-Lab/SpinGNN --all -g

# Preview all available skills without installing
npx skills add Hyyu-Lab/SpinGNN --list

# Install specific skill(s) only
npx skills add Hyyu-Lab/SpinGNN --skill spingnn-install --skill spingnn2 -g -y
```

You can also open any `SKILL.md` directly and combine it with the detailed
contracts in [`docs/README.md`](../docs/README.md).

| Skill | Workflow |
| --- | --- |
| [spingnn2](spingnn2/SKILL.md) | **SpinGNN2** native C++17/Kokkos engine (`all-gpu` `sm80`–`sm120` fatbin), export, `bundle_abi = 1` JIT, `spingnn2.v2` + `PhysMuon`, GPST CLI (`spingnn2-gpst-*`), Python/ASE `NativeModel`, and modern LAMMPS (`lmp_spingnn2`) |
| [spingnn-install](spingnn-install/SKILL.md) | Set up Python 3.10 / CUDA 12.8 / T-e3nn, `libspingnn2` (`cpu`, `all-gpu`, `sm80`–`sm120`), wheel + GPST kernel, and modern LAMMPS (`lmp_spingnn2`) builds |
| [structure-only-e2e](structure-only-e2e/SKILL.md) | Train a structure-only potential (`Allegro`, `Allegro_e2IP`) for energy, force, and stress without spins |
| [spin-nosoc-e2e](spin-nosoc-e2e/SKILL.md) | Train and deploy a magnetic potential without SOC (`Allegro_nosoc`, `v2_nosoc`) with `fix spin/sib`, `fix spin/cfi` (`box iso`), and `compute spin/gpst` |
| [spin-soc-e2e](spin-soc-e2e/SKILL.md) | Train and deploy a magnetic potential with SOC (`Allegro_soc`, `v2_soc`) for anisotropy, DMI, and coupled spin-lattice MD |
| [born-effective-charge-e2e](born-effective-charge-e2e/SKILL.md) | Train a DREAM Born effective-charge model (`Allegro_born`, `v2_born`) and run `atom_style bec` / `fix efield coupling bec` MD |
| [e2slp-evaluation](e2slp-evaluation/SKILL.md) | Score trajectories for active learning (`Allegro_e2IP`, `Allegro_soc_e2IP`, `v2_e2slp`), compare uncertainty with error, and read `fix pair` outputs |
| [spingnn-usage](spingnn-usage/SKILL.md) | Comprehensive command & configuration reference for SpinGNN2 data, training (`PhysMuon`/`Muon`), deployment, GPST, and modern LAMMPS |

Start with [INSTALL.md](../INSTALL.md), the
[SpinGNN2 User Guide](../docs/SPINGNN2_GUIDE.md), and the
[training configurations](../example/README.md).

See [tools/ci/README.md](../tools/ci/README.md) for test commands and
[CONTRIBUTING.md](../CONTRIBUTING.md) for workflow updates. All skills are
covered by the [SpinGNN MIT license](../LICENSES/MIT-SpinGNN.txt).
