# SpinGNN skills

These reusable guides give coding assistants repository-specific instructions
from dataset preparation through training, deployment, and LAMMPS checks.
Open the relevant `SKILL.md` in your assistant or read it directly.

| Skill | Workflow |
| --- | --- |
| [spingnn-install](spingnn-install/SKILL.md) | Set up Python, T-e3nn, and LAMMPS |
| [structure-only-e2e](structure-only-e2e/SKILL.md) | Train a potential for energy, force, and stress without spins |
| [spin-nosoc-e2e](spin-nosoc-e2e/SKILL.md) | Train and deploy a magnetic potential without spin–orbit coupling |
| [spin-soc-e2e](spin-soc-e2e/SKILL.md) | Train and deploy a magnetic potential with spin–orbit coupling |
| [born-effective-charge-e2e](born-effective-charge-e2e/SKILL.md) | Train a Born effective-charge model and run electric-field simulations |
| [e2slp-evaluation](e2slp-evaluation/SKILL.md) | Score trajectories for active learning and compare uncertainty with model error |
| [spingnn-usage](spingnn-usage/SKILL.md) | Reference for commonly used SpinGNN commands and settings |

Start with the [installation guide](../INSTALL.md) and adapt the
[training configurations](../example/README.md) to your dataset, species, and
available labels. The kind of coupling your model needs determines whether to
use a NoSOC or SOC builder. The [LAMMPS tutorial](../lammps/examples/spingnn/TUTORIAL_LAMMPS.md)
explains how to export the chosen model and validate it in the simulation.

See the [CI guide](../tools/ci/README.md) for test commands and the
[contributing guide](../CONTRIBUTING.md) for changes to the workflows.
The skills are covered by the [SpinGNN MIT license](../LICENSES/MIT-SpinGNN.txt).
