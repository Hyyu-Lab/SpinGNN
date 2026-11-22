# SpinGNN documentation

## Install and configure

- [Installation](../INSTALL.md) covers CPU and CUDA environments and the bundled LAMMPS build.
- [CUDA 12.8 environment](INSTALL_CUDA128.md) lists the supported dependency versions, installation commands, and checks.
- [Run from a source checkout](SOURCE_CHECKOUT.md) explains how to select the current checkout in an existing environment.
- [T-e3nn](TE3NN_INTEGRATION.md) describes the v1.0 pin, source verification, and optional Linux memory policy.
- [Docker](../docker/README.md) covers building and using the CUDA 12.8 image.

## Train and evaluate

- [Training examples](../example/README.md) lists templates 01–07, dataset fields, and configuration requirements.
- [Training tutorial](../example/TUTORIAL_TRAINING.md) walks through data preparation, training, and deployment.
- [e2SLP](../example/E2IP.md) covers evidential force and spin-force training and uncertainty scoring for active learning.
- [Packaged-backbone BEC heads](../example/README.md#born-effective-charges-on-a-packaged-foundation-model) covers fine-tuning and frame-weighted supervision.
- [Packaged-backbone dielectric and piezoelectric heads](../example/README.md#dielectric-and-piezoelectric-response-heads-on-a-packaged-foundation-model) describes the additional Cartesian response targets.
- [Reusable AI workflow skills](../skills/README.md) provide installation, model-specific, and evaluation workflows.

## Deploy and simulate

- [PT2 deployment](PT2_MIGRATION.md) covers export, sidecar metadata, and runtime compatibility.
- [LAMMPS reference](../lammps/README.md) covers builds, pair styles, spin integrators, BEC, and electric fields.
- [LAMMPS tutorial](../lammps/examples/spingnn/TUTORIAL_LAMMPS.md) follows a trained model into an MD run.
- [LAMMPS examples](../lammps/examples/spingnn/README.md) lists structure-only, spin, and electric-field input files.

## Develop and cite

- [CI and testing](../tools/ci/README.md) describes test lanes and commands. [LAMMPS tests](../tests_lammps/README.md) lists runtime prerequisites.
- [Contributing](../CONTRIBUTING.md) lists development checks and pull-request expectations.
- [Licenses](../LICENSES.md) maps the MIT code and documents to their license files and explains GPL-2.0 obligations for LAMMPS.
- [v0.1.0 release](OPEN_SOURCE_RELEASE.md) summarizes the public release scope.
- [Citation](../README.md#citation) lists publications and contributors. Machine-readable references are in [BibTeX](../CITATION.bib) and [CFF](../CITATION.cff).
