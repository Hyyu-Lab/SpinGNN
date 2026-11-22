# SpinGNN extension examples

These snippets document the extension-specific commands. They are templates,
not self-contained regression tests: replace data, BEC, and TorchScript model
paths with files for the system being simulated.

## Non-spin Allegro or NequIP

```lammps
units       metal
atom_style  atomic
atom_modify map array
newton      on
read_data   system.data

pair_style  allegro
pair_coeff  * * model.pth Fe O
run          0
```

For `pair_style nequip`, use `newton off`. The number of element names in
`pair_coeff` must equal the number of LAMMPS atom types.

## Spin-aware model with SIB integration

```lammps
units       metal
atom_style  spin
atom_modify map array
newton      on
read_data   system-spin.data

pair_style  spin/allegro_spin
pair_coeff  * * spin-model.pth Fe O

fix          int all spin/sib 300.0 0.1 4928459 lattice moving
timestep     0.0001
run          1000
```

Use `lattice frozen` for spin-only integration. The aliases `lattice yes` and
`lattice no` are accepted for compatibility. For `spin/nequip_spin`, switch
to `newton off`.

## Coupled spin-lattice NPT with SIB integration

`fix spin/sib_npt` integrates the spins and lattice together while controlling
the lattice temperature and pressure. This minimal example runs at 300 K and
zero isotropic pressure:

```lammps
units       metal
boundary    p p p
atom_style  spin
atom_modify map array
newton      on
read_data   system-spin.data

mass         1 55.845
mass         2 15.999

pair_style  spin/allegro_spin
pair_coeff  * * spin-model.pth Fe O

velocity     all create 300.0 12345 dist gaussian
timestep     0.0005

# temp: lattice Tstart, Tstop, and thermostat damping time (ps)
# alpha: transverse spin damping; seed: stochastic-spin random seed
# iso: pressure Pstart, Pstop (bar), and barostat damping time (ps)
fix          int all spin/sib_npt &
             temp 300.0 300.0 0.05 &
             alpha 0.4 seed 12345 &
             iso 0.0 0.0 0.5

thermo       100
thermo_style custom step time temp press vol pe ke etotal
run          10000
```

In `units metal`, the damping times are in ps and pressure is in bar. `Tstop`
is also used as the stochastic spin-bath temperature. The barostat accepts
the normal NPT pressure forms, including `iso`, `aniso`, `tri`, and individual
cell components. Do not apply another time-integration fix such as `nve`,
`nvt`, `npt`, or `spin/sib` to the same atoms while `spin/sib_npt` is active.

For `spin/nequip_spin`, use `newton off`. A complete anisotropic NPT cooling
workflow is provided in [`bfo/in.bfo-sld`](bfo/in.bfo-sld), with required
input files and run instructions in [`bfo/README.md`](bfo/README.md).

## BEC electric-field coupling and output

```lammps
units       metal
atom_style  bec
read_data   system.data
read_bec    system.bec

fix          field all efield 0.0 0.0 0.01 coupling bec piezo no
compute      pd all bec_dipole
compute      pa all property/atom becxx becyy beczz &
             becdipolex becdipoley becdipolez
dump         d all custom 100 bec.dump id type x y z &
             becxx becyy beczz becdipolex becdipoley becdipolez
thermo_style custom step pe c_pd c_pd[1] c_pd[2] c_pd[3] c_pd[4]

fix          int all nve
run          1000
```

The scalar `c_pd` is the magnitude of the accumulated dipole. Vector elements
1-3 are its Cartesian components; element 4 reports the normalized residual
of the summed BEC tensor. Run `bec_dipole_reset` after equilibration when a new
coordinate reference is desired.

## BFO production templates

Two longer cooling workflows supplied with this project are available in
[`bfo/`](bfo/): coupled spin-lattice NPT dynamics and fixed-lattice spin-only
dynamics. They require a compatible `BFO.data` file and deployed spin-aware
model; see [`bfo/README.md`](bfo/README.md) before running them.
