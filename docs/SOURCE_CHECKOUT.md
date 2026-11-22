# Source checkout and shared environments

A shared environment can contain non-editable copies of `nequip` and
`allegro` whose versions match the checkout while their source code differs.
The source launcher selects both packages from the same checkout.

## Run the complete checkout

Use the intended Python interpreter and the repository launcher. The launcher
works from any directory when given its absolute path. Script arguments such
as configuration filenames resolve from the caller's working directory.

```bash
SPINGNN_REPO=/absolute/path/to/SpinGNN
SPINGNN_PYTHON=/absolute/path/to/python-environment/bin/python
"$SPINGNN_PYTHON" "$SPINGNN_REPO/tools/ci/source_python.py" nequip-train --help
"$SPINGNN_PYTHON" "$SPINGNN_REPO/tools/ci/source_python.py" nequip-deploy --help
"$SPINGNN_PYTHON" "$SPINGNN_REPO/tools/ci/source_python.py" -m pytest \
  "$SPINGNN_REPO/src/core/tests/unit" -q
```

The launcher reads the console commands from `src/core/setup.py`. It also
accepts `-m MODULE`, `-c CODE`, and Python script paths. It puts the two source
roots first in `sys.path` and in the `PYTHONPATH` passed to child processes.
Restart Python through the launcher if it reports that a package was already
imported from another location.

For existing shell recipes using the installed console commands, prefix both
roots before starting Python:

```bash
export PYTHONPATH="$SPINGNN_REPO/src/core:$SPINGNN_REPO/src/models${PYTHONPATH:+:$PYTHONPATH}"
"$SPINGNN_PYTHON" -c 'import nequip, allegro; print(nequip.__file__, allegro.__file__)'
```

Use absolute paths for both package roots so a child process can change its
working directory without changing which packages it imports.

## Protected entry points

| Entry | Protection |
| --- | --- |
| Repository pytest suites | Root `conftest.py` runs before package fixtures. Child processes inherit both source roots. |
| CPU helper and CI Python matrix | `source_python.py` selects the checkout before pytest loads plugins. The CPU helper prints the package origins. |
| NequIP integration and training-to-LAMMPS tests | Explicit absolute source environment passed into training and deployment subprocesses. |
| Source CLI mains | Reject mixed imports before argument parsing or work. Standalone wheels use their installed packages. |
| PT2 three-way and smoke tools, example validator | Bootstrap before model-package imports. |
| NequIP Sphinx configuration and monkeypatch example | Bootstrap located through `__file__`, independent of cwd. |
| GPU tests and training dry-check wrappers | Select the checkout packages before pytest or model imports. |

Keep both source packages in the checkout when using the launcher. A
standalone package or the [Docker image](../docker/README.md) uses its own
installed code.

[`test_source_exact.py`](../tests_lammps/test_source_exact.py) tests import
origins in the launcher, standalone scripts, pytest and child processes.

## Update a shared environment

1. Coordinate with the environment's users or create a separate environment.
   Record the Python path, source revision, and installed dependency versions
   before changing a shared installation.
2. Make both packages editable while retaining PyTorch and the
   [pinned T-e3nn release](TE3NN_INTEGRATION.md):

   ```bash
   uv pip install --python "$SPINGNN_PYTHON" --no-deps --no-build-isolation \
     -e "$SPINGNN_REPO/src/core" -e "$SPINGNN_REPO/src/models"
   ```

3. From outside the checkout, clear `PYTHONPATH` and check that
   `nequip.__file__` and `allegro.__file__` resolve to the intended source
   tree. Run the relevant source and deployment tests.
4. Restart existing Python workers and notebooks so subsequent imports use
   the new installation.

