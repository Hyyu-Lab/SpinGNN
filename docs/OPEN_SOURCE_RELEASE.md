# SpinGNN v0.1.0

[Hyyu-Lab/SpinGNN](https://github.com/Hyyu-Lab/SpinGNN) publishes the
Spin-NequIP and Spin-Allegro Python packages, the modified LAMMPS runtime,
and example configurations together under the `v0.1.0` source tag.

- See the [installation guide](../INSTALL.md) for the Python 3.10,
  PyTorch 2.11.0+cu128, and CUDA 12.8 environment.
- The [T-e3nn guide](TE3NN_INTEGRATION.md) covers the v1.0 dependency pin and
  installed-source verification.
- Use the [training examples](../example/README.md) to choose a model and
  the [deployment guide](PT2_MIGRATION.md) to export it for LAMMPS.
- The [LAMMPS fork notice](../lammps/NOTICE.md) records the upstream
  revision and local extensions. The [license index](../LICENSES.md)
  lists the MIT and GPL-2.0 scopes.
- [CITATION.cff](../CITATION.cff) and [CITATION.bib](../CITATION.bib)
  contain the software citation metadata and the four associated papers.

The source release includes configuration templates and small test fixtures.
Supply datasets and trained weights appropriate to your system. Use the
[documentation index](README.md) to find the guide for your workflow and
the [contributing guide](../CONTRIBUTING.md) for source and test changes.
