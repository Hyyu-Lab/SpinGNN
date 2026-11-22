// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "atom_vec_bec.h"

#include "atom.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecBEC::AtomVecBEC(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = Atom::ATOMIC;
  mass_type = PER_TYPE;

  atom->bec_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = (char *) "bec bec_dipole";
  fields_copy = (char *) "bec bec_dipole";
  fields_comm = (char *) "";
  fields_comm_vel = (char *) "";
  fields_reverse = (char *) "";
  fields_border = (char *) "bec";
  fields_border_vel = (char *) "bec";
  fields_exchange = (char *) "bec bec_dipole";
  fields_restart = (char *) "bec bec_dipole";
  fields_create = (char *) "bec bec_dipole";
  fields_data_atom = (char *) "id type x";
  fields_data_vel = (char *) "id v";

  setup_fields();
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecBEC::grow_pointers()
{
  bec = atom->bec;
  bec_dipole = atom->bec_dipole;
}

/* ----------------------------------------------------------------------
   data files do not carry BEC columns; initialize them before read_bec
------------------------------------------------------------------------- */

void AtomVecBEC::data_atom_post(int ilocal)
{
  memset(&bec[ilocal][0],0,9*sizeof(double));
  memset(&bec_dipole[ilocal][0],0,3*sizeof(double));
}
