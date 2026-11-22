# Licensing

This is a multi-license repository. A single license cannot accurately cover
every subtree because the project combines modified MIT-licensed Python
projects with a GPL-licensed LAMMPS distribution.

| Scope | License file | License |
| --- | --- | --- |
| `src/core/` | [`src/core/LICENSE`](src/core/LICENSE) | MIT, including retained Harvard attribution |
| `src/models/` | [`src/models/LICENSE`](src/models/LICENSE) | MIT, including retained Harvard attribution |
| `lammps/` | [`lammps/LICENSE`](lammps/LICENSE) | GNU GPL version 2, subject to notices and third-party licenses inside that distribution |
| Repository-level documentation and metadata, `docs/`, `example/`, `skills/`, `tools/`, `docker/`, `tests_ci/`, and `tests_lammps/` | [`LICENSES/MIT-SpinGNN.txt`](LICENSES/MIT-SpinGNN.txt) | MIT |

The LAMMPS fork is a modified distribution. Its upstream revision and change
summary are recorded in [`lammps/NOTICE.md`](lammps/NOTICE.md). Redistributing
a LAMMPS binary requires compliance with GPL-2.0, including corresponding
source obligations.

Existing copyright, authorship, and third-party license notices must remain
intact. This index does not replace the license text in any component.
