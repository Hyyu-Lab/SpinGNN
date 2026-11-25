// SPDX-License-Identifier: GPL-2.0-only
// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "atom_vec_bec.h"

#include "atom.h"

#include <algorithm>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecBEC::AtomVecBEC(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = Atom::ATOMIC;
  mass_type = PER_TYPE;

  atom->bec_flag = 1;

  // BEC and its integrated dipole belong to the owning atom.
  fields_grow = {"bec", "bec_dipole"};
  fields_copy = fields_grow;
  fields_exchange = fields_grow;
  fields_restart = fields_grow;
  fields_create = fields_grow;
  fields_data_atom = {"id", "type", "x"};
  fields_data_vel = {"id", "v"};

  setup_fields();
}

/* ----------------------------------------------------------------------
   data files do not carry BEC columns; initialize them before read_bec
------------------------------------------------------------------------- */

void AtomVecBEC::data_atom_post(int ilocal)
{
  std::fill_n(atom->bec[ilocal], 9, 0.0);
  std::fill_n(atom->bec_dipole[ilocal], 3, 0.0);
}
