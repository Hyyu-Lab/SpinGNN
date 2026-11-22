# BFO spin-dynamics examples

This directory contains two complete input-script templates for a three-type
BiFeO3 system:

- `in.bfo-sld`: coupled spin-lattice dynamics with NPT cooling;
- `in.bfo-spin-only`: spin-only cooling with fixed atoms and cell.

Both scripts require two files that are not distributed here:

- `BFO.data`: the 80-atom primitive simulation data file expected by the group
  definitions in the inputs;
- `spin-model.pth`: a deployed spin-aware Allegro TorchScript model whose
  LAMMPS type order is `Fe Bi O`.

These local inputs and all generated logs, dumps, averages, and restart files
are ignored by the `.gitignore` in this directory.

The scripts define G-type antiferromagnetic sublattices using atom IDs 1-8 and
9-16 before applying `replicate 2 4 2`. If a different data file or type/ID
ordering is used, update the `group`, `mass`, and `pair_coeff` commands first.

Run from this directory after supplying those files:

```bash
mpirun -np 1 ../../../build/lmp -in in.bfo-spin-only
mpirun -np 1 ../../../build/lmp -in in.bfo-sld
```

`spin/allegro_spin` requires `newton on`. The `processors ... grid numa`
setting is retained from the server workflow; remove that line if the target
MPI topology does not support LAMMPS NUMA processor mapping.

These are realistic production templates rather than small regression tests.
Their default schedules run from 1001 K toward 1 K and can be expensive. For
a smoke test, reduce `t_cool`, `t_eq`, and `t_collect` near the top of each
input.
