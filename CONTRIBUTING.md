# Contributing to SpinGNN2

Thank you for improving SpinGNN2. Please open an issue before a large change so
the intended component, compatibility target, and validation plan are clear.

## Choose the Affected Component

- `src/spingnn2/`: magnetic symmetry primitives (`symmetry/`), pure-PyTorch v2
  training models (`v2/`), `.sp2.json`/`.sp2.bin` contract exporter, NumPy
  reference oracle, offline JIT preparer, `NativeModel`, `SpinGNN2Calculator`,
  and GPST spin-temperature helper;
- `src/core/` & `src/models/`: SpinGNN++ v1 training engine and Allegro models;
- `libspingnn2/`: C++17 / Kokkos 4.7.02 C ABI library (`libspingnn2.so`) and
  NVRTC preparation library;
- `lammps_spingnn2/`: modern LAMMPS (`stable_30Sep2026_update1`) external
  overlay, patches, Kokkos pair/fixes, and DREAM BEC support;
- `example/`: public training templates.

Preserve the license and upstream attribution of every component. In
particular, changes under `lammps_spingnn2/` remain subject to GPL-2.0.

## Development Checks

Activate the [supported Python environment](INSTALL.md) and run from the
repository root. The source launcher loads all packages from this checkout.

```bash
python tools/ci/test_inventory.py
python tools/ci/source_python.py -m pytest tests_ci tests_v2 -q
python tools/ci/source_python.py -m pytest src/core/tests/unit src/models/tests \
  -m "not slow and not gpu and not lammps and not mpi"
python tools/ci/source_python.py example/validate_configs.py
```

The [CI workflow](.github/workflows/ci.yml) defines the full lint scope. It
uses Black 22.3.0 with `black --check` and Flake8 7.1.1 with
`flake8 --config=src/core/.flake8 --count`. Run both on your changes.
See the [CI guide](tools/ci/README.md) for test lanes and markers.

For LAMMPS changes, follow [lammps_spingnn2/README.md](lammps_spingnn2/README.md)
and verify the overlay with `python lammps_spingnn2/apply.py --check`.

Pull requests should describe:

- the component and scientific behavior being changed;
- the environment used for validation;
- the exact test or LAMMPS command run;
- whether file formats, contract keys, or numerical results change.

Do not commit trained models, private datasets, large dumps/restarts, cluster
paths, credentials, or build directories.
