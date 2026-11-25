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

#include "bec_dipole_reset.h"

#include "comm.h"
#include "error.h"
#include "atom.h"
#include "atom_masks.h"
#include "utils.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BecDipoleReset::BecDipoleReset(LAMMPS *lmp) : Command(lmp) {}

/* ---------------------------------------------------------------------- */

void BecDipoleReset::command(int narg, char ** /*arg*/)
{
  if (narg != 0) error->all(FLERR,"Illegal bec_dipole_reset command");
  if (!atom->bec_flag)
    error->all(FLERR,"Bec_dipole_reset requires atom attribute bec");

  atom->sync_host_arrays(BEC_DIPOLE_MASK);
  double **bec_dipole = atom->bec_dipole;
  for (int i = 0; i < atom->nlocal; i++) {
    bec_dipole[i][0] = 0.0;
    bec_dipole[i][1] = 0.0;
    bec_dipole[i][2] = 0.0;
  }

  atom->modified_host_arrays(BEC_DIPOLE_MASK);

  if (comm->me == 0)
    utils::logmesg(lmp,"Reset the integrated BEC dipole to zero; "
                   "current atom positions are now the reference.\n");
}
