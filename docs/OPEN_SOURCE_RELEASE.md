# SpinGNN2 Open-Source Release

SpinGNN2 publishes the pure-PyTorch magnetic symmetry & v2 training package
(`src/spingnn2/`), v1 SpinGNN++ training compatibility (`src/core/`,
`src/models/`), the standalone native C++17 / Kokkos C ABI library
(`libspingnn2/`), the modern LAMMPS (`stable_30Sep2026_update1`) external
overlay (`lammps_spingnn2/`), example configurations, AI skills, and
qualification suites together.

- See [INSTALL.md](../INSTALL.md) for Python 3.10 / PyTorch 2.11+cu128 setup
  and native / LAMMPS builds.
- See [docs/SPINGNN2_GUIDE.md](SPINGNN2_GUIDE.md) for checkpoint export,
  offline JIT compilation, Python/ASE inference, and modern LAMMPS MD.
- See [LICENSES.md](../LICENSES.md) and
  [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) for MIT and GPL-2.0
  scopes and redistribution notices.
- See [CITATION.cff](../CITATION.cff) and [CITATION.bib](../CITATION.bib) for
  software citation metadata and associated publications.

The release ships configuration templates and small test fixtures; supply
datasets and trained weights appropriate for your target material system.
