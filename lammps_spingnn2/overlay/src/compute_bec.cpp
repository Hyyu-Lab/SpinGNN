// SPDX-License-Identifier: GPL-2.0-only
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

#include "compute_bec.h"

#include "atom.h"
#include "atom_masks.h"
#include "error.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeBEC::ComputeBEC(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR, "Illegal compute {} command", style);

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 4;
  extscalar = 1;
  extvector = -1;
  extlist = new int[size_vector];
  extlist[0] = extlist[1] = extlist[2] = 1;
  extlist[3] = 0;

  vector = new double[size_vector];
  vector[0] = vector[1] = vector[2] = vector[3] = 0.0;

}

/* ---------------------------------------------------------------------- */

ComputeBEC::~ComputeBEC()
{
  delete[] vector;
  delete[] extlist;
}

/* ---------------------------------------------------------------------- */

void ComputeBEC::init()
{
  if (!atom->bec_flag)
    error->all(FLERR, "Compute {} requires atom attribute bec", style);
}

/* ---------------------------------------------------------------------- */

void ComputeBEC::compute_vector()
{
  invoked_vector = update->ntimestep;
  invoked_flag |= INVOKED_VECTOR;
  atom->sync_host_arrays(MASK_MASK | BEC_MASK | BEC_DIPOLE_MASK);

  const auto mask = atom->mask;
  const auto nlocal = atom->nlocal;

  double dipole[3] = {0.0, 0.0, 0.0};
  double dipole_mpi[3] = {0.0, 0.0, 0.0};
  double bec_all[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double bec_all_mpi[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  bigint ngroup_local = 0;
  bigint ngroup = 0;

  const auto bec_dipole = atom->bec_dipole;
  const auto bec = atom->bec;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      dipole[0] += bec_dipole[i][0];
      dipole[1] += bec_dipole[i][1];
      dipole[2] += bec_dipole[i][2];

      for (int j = 0; j < 9; j++) bec_all[j] += bec[i][j];
      ngroup_local++;
    }
  }

  MPI_Allreduce(dipole, dipole_mpi, 3, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(bec_all, bec_all_mpi, 9, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&ngroup_local, &ngroup, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  if (ngroup == 0)
    error->all(FLERR, "Compute {} group contains no atoms", style);

  vector[0] = dipole_mpi[0];
  vector[1] = dipole_mpi[1];
  vector[2] = dipole_mpi[2];
  vector[3] = 0.0;
  for (int i = 0; i < 9; i++) {
    vector[3] += bec_all_mpi[i] * bec_all_mpi[i];
  }
  vector[3] = sqrt(vector[3]/9.0) / ngroup;
}

/* ---------------------------------------------------------------------- */

double ComputeBEC::compute_scalar()
{
  if (invoked_vector != update->ntimestep || !(invoked_flag & INVOKED_VECTOR))
    compute_vector();

  invoked_scalar = update->ntimestep;
  scalar = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
  return scalar;
}
