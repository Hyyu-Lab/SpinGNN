# Licensing

This is a multi-license repository combining MIT-licensed Python and native
library components with a GPL-2.0-licensed external LAMMPS overlay.

| Scope | License file | License |
| --- | --- | --- |
| `src/core/` | [`src/core/LICENSE`](src/core/LICENSE) | MIT, including retained Harvard attribution |
| `src/models/` | [`src/models/LICENSE`](src/models/LICENSE) | MIT, including retained Harvard attribution |
| `src/spingnn2/`, `libspingnn2/`, `tests_native/`, `tests_v2/`, and `benchmarks/` | [`LICENSES/MIT-SpinGNN.txt`](LICENSES/MIT-SpinGNN.txt) | MIT, with [native dependency notices](THIRD_PARTY_LICENSES.md) |
| `lammps_spingnn2/overlay/` and `lammps_spingnn2/patches/` | [`LICENSES/LAMMPS-GPL-2.0.txt`](LICENSES/LAMMPS-GPL-2.0.txt) | GNU GPL version 2 |
| Repository-level documentation and metadata, `docs/`, `example/`, `skills/`, `tools/`, `docker/`, and `tests_ci/` | [`LICENSES/MIT-SpinGNN.txt`](LICENSES/MIT-SpinGNN.txt) | MIT |

Redistributing a compiled `lmp_spingnn2` binary built with the LAMMPS overlay
requires compliance with GPL-2.0, including corresponding source obligations.

Existing copyright, authorship, and third-party license notices must remain
intact. This index does not replace the license text in any component.
