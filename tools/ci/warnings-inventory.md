# Warnings inventory

`warnings-budget.json` defines the maximum warning count for each serial CPU
suite. Run `SPINGNN_ENV=/path/to/env tools/ci/cpu_unit_tests.sh` from the
checkout and use its `CI_WARNING_REPORT` counts when updating the inventory.
The marker expression is `not gpu and not lammps and not mpi and not slow`.

| Suite | Warning budget | Common warning sources |
| --- | ---: | --- |
| `src/core` | 8489 | TorchScript deprecations, experimental stress support, ASE and NumPy deprecations |
| `src/models` | 23463 | TorchScript deprecations and instance-level type annotations, experimental stress support |

The limits are carried forward from the existing suite. They are ceilings and
may exceed the current counts. When changing a limit, record a fresh full-suite
measurement with the Python, PyTorch, and T-e3nn versions. A test failure
still fails the suite independently of the warning count.
