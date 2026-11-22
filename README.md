# SpinGNN

**SpinGNN** is an open-source machine-learning interatomic potential (MLIP) and
molecular dynamics framework for **magnetic, polar, ferroelectric, and
multiferroic materials**.

It brings together three closely related capabilities in a single repository:

- **SpinGNN & SpinGNN++** — Time-reversal-equivariant neural-network potentials
  with explicit magnetic degrees of freedom for coupled **spin–lattice
  dynamics** in LAMMPS.
- **DREAM** — Equivariant **Born effective charge (BEC)** prediction integrated
  with LAMMPS for **electric-field-driven molecular dynamics** and polarization
  response simulations.
- **e2SLP** — Tangent-plane evidential uncertainty quantification for reliable
  **active learning** of magnetic interatomic potentials.

## What You Can Do with SpinGNN

| Research Goal | Typical Materials | Workflow | Template | AI Skill |
| --- | --- | --- | --- | --- |
| Structure-only potentials (energy, forces, stress) | Metals, semiconductors, oxides | Structure-only Allegro / NequIP | [01](example/01_structure_only.yaml) | [structure-only-e2e](skills/structure-only-e2e/SKILL.md) |
| Magnetic potentials without SOC (Heisenberg exchange) | 3d transition-metal magnets (Fe, Co, Ni, Mn, Cr) | NoSOC magnetic model (`Allegro_nosoc` / spin NequIP) | [02](example/02_spin_nosoc.yaml) | [spin-nosoc-e2e](skills/spin-nosoc-e2e/SKILL.md) |
| Magnetic potentials with SOC (anisotropy, DMI, non-collinear order) | Heavy-element / 4d–5f magnets | SOC magnetic model (`Allegro_soc` / spin NequIP) | [03](example/03_spin_soc.yaml), [05](example/05_spin_soc_production.yaml) | [spin-soc-e2e](skills/spin-soc-e2e/SKILL.md) |
| **Electric-field-driven MD** & Born effective charges (**DREAM**) | Ferroelectrics, polar crystals, multiferroics | BEC + dipole + external E-field in LAMMPS (`Allegro_born`) | [04](example/04_born_effective_charge.yaml) | [born-effective-charge-e2e](skills/born-effective-charge-e2e/SKILL.md) |
| Active learning & uncertainty scoring (**e2SLP**) | Magnetic & structural MD trajectories | Evidential force / spin-force uncertainty (`Allegro_e2IP`, `Allegro_soc_e2IP`) | [06](example/06_allegro_e2ip.yaml), [07](example/07_allegro_soc_e2ip.yaml) | [e2slp-evaluation](skills/e2slp-evaluation/SKILL.md) |

## Key Highlights

- **Exact physical symmetries**: Built on time-reversal-aware
  [T-e3nn](https://github.com/Hongyu-yu/T-e3nn) (`v1.0`) to enforce E(3) and
  time-reversal symmetries across atomic positions, spins, and response tensors.
- **Spin–lattice & electric-field MD in LAMMPS**: Includes a modified LAMMPS
  distribution (`stable_29Sep2021_update2`) supporting lattice MD, coupled
  spin–lattice dynamics (`fix spin/sib`, `fix spin/sib_npt`), dipole/BEC
  outputs, and external electric-field coupling (**DREAM**).
- **Modern PyTorch 2.11 / CUDA 12.8 stack**: Supports compiled training,
  multi-GPU DDP, TorchScript (`.pth`), and AOTInductor (`.pt2`) GPU deployment,
  plus optional Kokkos acceleration.
- **Built-in AI workflow skills (`skills/`)**: Ships seven end-to-end guides
  that coding assistants can follow directly to install, train, deploy, and
  validate simulations with minimal friction.

## Quick Installation

SpinGNN targets **Linux x86-64**, **Python 3.10**, **PyTorch 2.11.0+cu128**,
**CUDA 12.8**, and [T-e3nn](https://github.com/Hongyu-yu/T-e3nn) `v1.0`
(`e3nn==0.5.0+spin`).

### 1. Clone the repository

```bash
git clone --branch v0.1.0 --depth 1 https://github.com/Hyyu-Lab/SpinGNN.git
cd SpinGNN
```

If you only need Python training and examples without the LAMMPS C++ tree, use a
sparse checkout:

```bash
git clone --branch v0.1.0 --depth 1 --filter=blob:none --sparse \
  https://github.com/Hyyu-Lab/SpinGNN.git
cd SpinGNN
git sparse-checkout set src example skills tools docs LICENSES
```

### 2. Install the Python environment (one command)

The bundled installer detects your GPU (`H100`, `A100`, `RTX 4090`, `RTX 5090`,
or `CPU`), installs PyTorch 2.11 and the pinned `T-e3nn` `v1.0` dependency, and
verifies imports:

```bash
tools/ci/install.sh --gpu auto      # or explicitly: --gpu h100|a100|4090|5090|cpu
source validation/env/spingnn-cu128/bin/activate
```

> Prefer a container or custom environment? See the
> [full installation guide](INSTALL.md), the
> [CUDA 12.8 environment guide](docs/INSTALL_CUDA128.md), or the
> [Docker guide](docker/README.md).

### 3. Build LAMMPS (optional, for MD simulations)

```bash
PYTHON="$(command -v python)" tools/lammps/build_lammps.sh serial   # or: mpi
```

*(If you used a sparse clone, run `git sparse-checkout add lammps` first.)*

## Quick Start

Run commands from the repository root through `tools/ci/source_python.py` so
Python always loads the checkout under `src/core/` and `src/models/`.

```bash
# 1. Verify your setup against all 7 public training templates (synthetic check)
python tools/ci/source_python.py example/validate_configs.py

# 2. Edit a template with your extxyz dataset & species, then train
python tools/ci/source_python.py nequip-train example/02_spin_nosoc.yaml

# 3. Deploy as TorchScript (.pth) or AOTInductor (.pt2) for LAMMPS
mkdir -p deployed
python tools/ci/source_python.py nequip-deploy build \
  --train-dir results/spin_nosoc/example deployed/spin.pth
python tools/ci/source_python.py nequip-deploy build-pt2 --device auto \
  --train-dir results/spin_nosoc/example deployed/spin.pt2

# 4. Run coupled spin–lattice MD in LAMMPS (bundled 8-atom Mn–Ge example)
build/lammps-serial/lmp -in lammps/examples/spingnn/in.spin_lattice_md \
  -var model "$PWD/deployed/spin.pt2" \
  -var data "$PWD/lammps/examples/spingnn/data.spin8" \
  -var output spin_lattice
```

> Each `.pt2` package is tied to the checkpoint it was exported from—keep its
> adjacent `.pt2.meta` sidecar (which records the package SHA-256) next to the
> model when moving or archiving it.

For electric-field MD (**DREAM**) and active learning (**e2SLP**), follow the
dedicated guides in [example/README.md](example/README.md),
[example/TUTORIAL_TRAINING.md](example/TUTORIAL_TRAINING.md), and
[lammps/examples/spingnn/TUTORIAL_LAMMPS.md](lammps/examples/spingnn/TUTORIAL_LAMMPS.md).

## Using SpinGNN with AI Skills (`skills/`)

SpinGNN ships a curated set of reusable **AI agent skills** in
[`skills/`](skills/README.md). Each `SKILL.md` encodes tested step-by-step
commands, YAML templates, dataset format checks, LAMMPS inputs, and parity
checks so coding assistants (such as **Codex**, **Claude Code**,
**Antigravity**, or **Cursor**) can walk you through an entire scientific
workflow reliably.

### How to use them

1. **Direct prompt inside this repository** — Ask your coding assistant to read
   the matching skill before acting, for example:
   - *"Read `skills/spingnn-install/SKILL.md` and install SpinGNN + LAMMPS on this machine."*
   - *"Read `skills/born-effective-charge-e2e/SKILL.md` and help me train a DREAM model and run an electric-field MD simulation on my dataset."*
   - *"Read `skills/spin-soc-e2e/SKILL.md` and set up an SOC magnetic training run and LAMMPS verification."*
2. **Install into your agent's global skills folder** — Symlink or copy any
   folder under `skills/` into your tool's skills directory (for example,
   `~/.codex/skills/`, `~/.claude/skills/`, or `~/.gemini/skills/`) so the
   assistant can trigger it automatically whenever you work on SpinGNN projects.

### Included skills

| Skill | What It Guides You Through |
| --- | --- |
| [`spingnn-install`](skills/spingnn-install/SKILL.md) | Environment setup for Python, `T-e3nn`, and serial/MPI LAMMPS |
| [`structure-only-e2e`](skills/structure-only-e2e/SKILL.md) | Non-magnetic energy, force, and stress potentials end to end |
| [`spin-nosoc-e2e`](skills/spin-nosoc-e2e/SKILL.md) | Magnetic NoSOC potentials (Heisenberg magnets) + spin–lattice MD |
| [`spin-soc-e2e`](skills/spin-soc-e2e/SKILL.md) | Magnetic SOC potentials (magnetic anisotropy / heavy elements) + spin–lattice MD |
| [`born-effective-charge-e2e`](skills/born-effective-charge-e2e/SKILL.md) | **DREAM** BEC training and electric-field-driven MD in LAMMPS |
| [`e2slp-evaluation`](skills/e2slp-evaluation/SKILL.md) | **e2SLP** trajectory uncertainty scoring and active-learning selection |
| [`spingnn-usage`](skills/spingnn-usage/SKILL.md) | Everyday reference for training, deployment, LAMMPS, and troubleshooting |

## Repository Layout

| Path | Contents | License |
| --- | --- | --- |
| [`src/core/`](src/core/) | Core data, equivariant engine, training, and deployment (`nequip` package) | MIT |
| [`src/models/`](src/models/) | Local spin, DREAM BEC, and e2SLP models (`allegro` package) | MIT |
| [`lammps/`](lammps/) | Modified LAMMPS fork with spin–lattice & electric-field MD support | GPL-2.0 |
| [`example/`](example/) | Starting YAML templates (01–07) and training tutorials | MIT |
| [`skills/`](skills/) | End-to-end AI assistant skills for setup, training, and simulation | MIT |
| [`docs/`](docs/) | Installation, deployment, T-e3nn, and developer documentation | MIT |
| [`docker/`](docker/) | CUDA 12.8 container recipe for Python and LAMMPS | MIT |
| [`tools/`](tools/) | Installers, LAMMPS build helper, e2SLP evaluator, and CI utilities | MIT |
| [`tests_ci/`](tests_ci/) & [`tests_lammps/`](tests_lammps/) | CI, deployment, and LAMMPS regression test suites | MIT |

See [LICENSES.md](LICENSES.md) for component notices and redistribution terms
(including GPL-2.0 obligations for LAMMPS binaries).

## Documentation

For detailed technical references, environment contracts, PT2 deployment,
LAMMPS pair-style syntax, and testing lanes, start from the
[Documentation Index](docs/README.md):

- **Setup**: [INSTALL.md](INSTALL.md) · [CUDA 12.8](docs/INSTALL_CUDA128.md) · [Source Checkout](docs/SOURCE_CHECKOUT.md) · [T-e3nn](docs/TE3NN_INTEGRATION.md) · [Docker](docker/README.md)
- **Training & Models**: [Examples](example/README.md) · [Training Tutorial](example/TUTORIAL_TRAINING.md) · [e2SLP Guide](example/E2IP.md)
- **Deployment & MD**: [PT2 Deployment](docs/PT2_MIGRATION.md) · [LAMMPS Reference](lammps/README.md) · [LAMMPS Tutorial](lammps/examples/spingnn/TUTORIAL_LAMMPS.md)
- **AI Skills & Development**: [Skills Index](skills/README.md) · [CI & Testing](tools/ci/README.md) · [Contributing](CONTRIBUTING.md)

## Citation

If you use SpinGNN in your research, please cite the papers relevant to your
workflow. Machine-readable records are provided in
[CITATION.bib](CITATION.bib) and [CITATION.cff](CITATION.cff).

1. **SpinGNN**: H. Yu, Y. Zhong, L. Hong, C. Xu, W. Ren, X. Gong, and H. Xiang,
   “Spin-dependent graph neural network potential for magnetic materials,”
   *Physical Review B* **109**, 144426 (2024).
   [DOI: 10.1103/PhysRevB.109.144426](https://doi.org/10.1103/PhysRevB.109.144426)
2. **SpinGNN++**: H. Yu, B. Liu, Y. Zhong, L. Hong, J. Ji, C. Xu, X. Gong, and H. Xiang,
   “Physics-informed time-reversal equivariant neural network potential for
   magnetic materials,” *Physical Review B* **110**, 104427 (2024).
   [DOI: 10.1103/PhysRevB.110.104427](https://doi.org/10.1103/PhysRevB.110.104427)
3. **DREAM**: H. Yu, S. Deng, H. Zhu, M. Xie, Y. Zhang, X. Shi, J. Zhong, C. He, and H. Xiang,
   “Prediction of Room Temperature Ferroelectricity in Subnano Silicon Thin Films
   with an Antiferroelectric Ground State,” *Physical Review Letters* **135**,
   156801 (2025).
   [DOI: 10.1103/xlkt-4wk5](https://doi.org/10.1103/xlkt-4wk5)
4. **e2SLP**: Y. Cheng, H. Yu, and H. Xiang,
   “Tangent-plane evidential uncertainty in active learning for magnetic
   interatomic potentials,” *Physical Review Materials* **10**, 103807 (2026).
   [DOI: 10.1103/PhysRevMaterials.10.103807](https://doi.org/10.1103/PhysRevMaterials.10.103807) · [arXiv:2605.12353](https://arxiv.org/abs/2605.12353)

### Code Contributors

- Hongyu Yu (Fudan University)
- Yang Cheng (Fudan University)
- Jinyu Liu (Fudan University)

### Project Leaders

- Xingao Gong (Fudan University)
- Hongjun Xiang (Fudan University)
- Hongyu Yu (Fudan University)
