# Source Checkout and Shared Environments

A shared Python environment can contain non-editable copies of `nequip`,
`allegro`, or `spingnn2` whose version strings match the checkout while their
source code differs. The repository source launcher selects all packages from
this checkout.

## Run from the Checkout

Use the intended Python interpreter and `tools/ci/source_python.py`. The
launcher works from any directory when given its absolute path. Script
arguments such as configuration filenames resolve from the caller's working
directory.

```bash
SPINGNN2_REPO=/absolute/path/to/SpinGNN2
SPINGNN2_PYTHON=/absolute/path/to/python-environment/bin/python
"$SPINGNN2_PYTHON" "$SPINGNN2_REPO/tools/ci/source_python.py" spingnn-train --help
"$SPINGNN2_PYTHON" "$SPINGNN2_REPO/tools/ci/source_python.py" spingnn2-export --help
"$SPINGNN2_PYTHON" "$SPINGNN2_REPO/tools/ci/source_python.py" -m pytest \
  "$SPINGNN2_REPO/tests_native" -q
```

The launcher reads console commands from `src/core/setup.py` and registers
`spingnn2-export` and `spingnn2-prepare-jit-artifact` when `src/spingnn2/` is
present, placing `src/core/`, `src/models/`, and `src/` first on `sys.path`
and `PYTHONPATH`. It also accepts `-m MODULE`, `-c CODE`, and Python script
paths (for GPST CLI tools from a checkout without installing the wheel, invoke
`-c "from spingnn2.gpst.cli import main_moments; main_moments()"` or install
the package with `pip install --no-deps --no-build-isolation -e .` to expose
`spingnn2-gpst-moments` and `spingnn2-gpst-infer`).

For shell recipes calling installed console commands directly, prefix the
checkout package roots before starting Python:

```bash
export PYTHONPATH="$SPINGNN2_REPO/src/core:$SPINGNN2_REPO/src/models:$SPINGNN2_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
"$SPINGNN2_PYTHON" -c 'import nequip, allegro, spingnn2; print(nequip.__file__, allegro.__file__, spingnn2.__file__)'
```

## Protected Entry Points

| Entry | Protection |
| --- | --- |
| Repository pytest suites | Root `conftest.py` activates checkout roots before package fixtures; child processes inherit `PYTHONPATH`. |
| CPU helper and CI Python matrix | `source_python.py` selects the checkout before pytest loads plugins. |
| Source CLI mains | Reject mixed imports before argument parsing. Standalone wheels use their installed packages. |
| Example validator & GPU checks | Bootstrap checkout roots before model-package imports. |

## Update a Shared Environment

1. Coordinate with the environment's users or create a separate environment.
2. Install the checkout packages editable while retaining PyTorch and the
   [pinned T-e3nn release](TE3NN_INTEGRATION.md):

   ```bash
   uv pip install --python "$SPINGNN2_PYTHON" --no-deps --no-build-isolation \
     -e "$SPINGNN2_REPO/src/core" -e "$SPINGNN2_REPO/src/models" -e "$SPINGNN2_REPO"
   ```

3. From outside the checkout, clear `PYTHONPATH` and verify that `nequip`,
   `allegro`, and `spingnn2` resolve to the intended source tree.
