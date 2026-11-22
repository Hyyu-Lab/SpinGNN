# Contributing to SpinGNN

Thank you for improving SpinGNN. Please open an issue before a large change so
the intended component, compatibility target, and validation plan are clear.

## Choose the affected component

- `src/core/`: data, training, deployment, and shared model infrastructure;
- `src/models/`: Allegro-specific model builders and tests;
- `lammps/`: C++ pair styles, spin integration, BEC/dipole support, and LAMMPS
  examples;
- `example/`: public training templates.

Preserve the license and upstream attribution of every component. In
particular, changes under `lammps/` remain subject to GPL-2.0.

## Development checks

Activate the [supported Python environment](INSTALL.md) and run from the
repository root. The source launcher loads the packages from this checkout.

```bash
python tools/ci/source_python.py -m pytest tests_ci -q
python tools/ci/source_python.py -m pytest src/core/tests/unit src/models/tests \
  -m "not slow and not gpu and not lammps and not mpi"
python tools/ci/source_python.py example/validate_configs.py
```

The [CI workflow](.github/workflows/ci.yml) defines the full lint scope. It
uses Black 22.3.0 with `black --check` and Flake8 7.1.1 with
`flake8 --config=src/core/.flake8 --count`. Run both on your changes.
See the [CI guide](tools/ci/README.md) for test lanes and markers.

For LAMMPS changes, follow the [LAMMPS build guide](lammps/README.md) and
exercise the affected pair style or command with a compatible deployed model
and data file.

Pull requests should describe:

- the component and scientific behavior being changed;
- the environment used for validation;
- the exact test or LAMMPS command run;
- whether file formats, deployed-model keys, or numerical results change.

Do not commit trained models, private datasets, large dumps/restarts, cluster
paths, credentials, or build directories. Use a GitHub Release, an archival
repository, or Git LFS for approved large public artifacts.

