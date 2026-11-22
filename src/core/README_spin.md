# NequIP-spin

NequIP-spin extends NequIP with time-reversal-aware, E(3)-equivariant
interatomic potentials that accept atomic spins as input.

The package provides:

- ASE/extxyz data loading for `initial_magmoms`, `spin_forces`, and related
  magnetic labels;
- force, stress, and spin-force outputs computed by automatic differentiation;
- training, evaluation, deployment, and VASP data-conversion commands;
- the base framework required by the sibling [Allegro package](../models/).

See the repository-level `README.md` for supported versions, installation,
verification, and example training commands.
