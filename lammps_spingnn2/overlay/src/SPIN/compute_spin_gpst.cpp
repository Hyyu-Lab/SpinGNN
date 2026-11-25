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
// SPDX-License-Identifier: GPL-2.0-only

/* Online Geometric-Probe Spin Temperature (GPST) sufficient moments.
   Uses the sphere integration-by-parts identity <N_Phi> = k_B T <D_Phi>
   on unit spins (sp[i][3] > 0) with conservative spin force f_i = hbar * fm_i
   (unscaled by moment magnitude) projected onto the tangent plane of S^2.
   Output vector layout (19 entries):
     [0..13]  : (N, D) pairs for pair_l1, pair_l2, pair_l4, pair_l8,
                geometry_Q2, geometry_zonal4, geometry_chirality
     [14]     : N_mag (number of magnetic atoms in group)
     [15..18] : Mx, My, Mz, |M| (mean unit magnetization over magnetic atoms)
   Output scalar: instantaneous pair_l1 diagnostic temperature N_l1 / (k_B D_l1). */

#include "compute_spin_gpst.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

ComputeSpinGPST::ComputeSpinGPST(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg)
{
  if (narg != 3 && narg != 5)
    error->all(FLERR, "Expected compute ID group spin/gpst [cutoff Rc]");

  if (narg == 5) {
    if (std::strcmp(arg[3], "cutoff") != 0)
      error->all(FLERR, "Expected compute ID group spin/gpst cutoff Rc");
    cutoff = utils::numeric(FLERR, arg[4], false, lmp);
    if (!std::isfinite(cutoff) || cutoff <= 0.0)
      error->all(FLERR, "compute spin/gpst cutoff must be positive");
  }

  scalar_flag = 1;
  extscalar = 0;
  vector_flag = 1;
  size_vector = 19;
  extvector = 0;
  comm_forward = 4;

  memory->create(vector, size_vector, "compute/spin/gpst:vector");
  for (int k = 0; k < size_vector; ++k) vector[k] = 0.0;
}

ComputeSpinGPST::~ComputeSpinGPST()
{
  if (copymode) return;
  memory->destroy(vector);
}

void ComputeSpinGPST::init()
{
  if (!atom->sp_flag) error->all(FLERR, "compute spin/gpst requires atom_style spin");
  if (!force->pair && cutoff == 0.0)
    error->all(FLERR, "compute spin/gpst requires a cutoff or an active pair style");

  const double skin = neighbor->skin;
  if (cutoff != 0.0) {
    const double cutghost =
        force->pair ? MAX(force->pair->cutforce + skin, comm->cutghostuser) : comm->cutghostuser;
    if (cutoff > cutghost)
      error->all(FLERR,
                 "compute spin/gpst cutoff exceeds ghost atom range - use comm_modify cutoff");
  }

  int cutflag = 1;
  if (force->pair) {
    if (cutoff == 0.0) cutoff = force->pair->cutforce;
    if (cutoff <= force->pair->cutforce + skin) cutflag = 0;
  }
  if (!std::isfinite(cutoff) || cutoff <= 0.0)
    error->all(FLERR, "compute spin/gpst requires a positive cutoff");

  cutsq = cutoff * cutoff;
  hbar = force->hplanck / MathConst::MY_2PI;
  kb = force->boltz;

  if (neighbor->style == Neighbor::MULTI)
    error->all(FLERR, "compute spin/gpst requires neighbor style bin or nsq");

  auto *req = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
  if (cutflag) req->set_cutoff_fixed(cutoff);
}

void ComputeSpinGPST::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

double ComputeSpinGPST::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  if (invoked_vector != update->ntimestep) compute_vector();
  scalar = (std::fabs(vector[1]) > 1.0e-30 && kb > 0.0) ? (vector[0] / (kb * vector[1])) : 0.0;
  return scalar;
}

void ComputeSpinGPST::finalize_vector(const double local_sums[18])
{
  double global_sums[18] = {};
  MPI_Allreduce(local_sums, global_sums, 18, MPI_DOUBLE, MPI_SUM, world);

  for (int c = 0; c < 7; ++c) {
    vector[2 * c] = global_sums[2 * c];
    vector[2 * c + 1] = global_sums[2 * c + 1];
  }
  const double n_mag = global_sums[14];
  vector[14] = n_mag;
  if (n_mag > 0.0) {
    const double inv_n = 1.0 / n_mag;
    const double mx = global_sums[15] * inv_n;
    const double my = global_sums[16] * inv_n;
    const double mz = global_sums[17] * inv_n;
    vector[15] = mx;
    vector[16] = my;
    vector[17] = mz;
    vector[18] = std::sqrt(mx * mx + my * my + mz * mz);
  } else {
    vector[15] = vector[16] = vector[17] = vector[18] = 0.0;
  }
}

void ComputeSpinGPST::compute_vector()
{
  invoked_vector = update->ntimestep;

  // Synchronize ghost unit spins so GPST is exact even outside pair->compute.
  comm->forward_comm(this);
  neighbor->build_one(list);

  const int inum = list->inum;
  const int *ilist = list->ilist;
  const int *numneigh = list->numneigh;
  int *const *firstneigh = list->firstneigh;

  double **x = atom->x;
  double **sp = atom->sp;
  double **fm = atom->fm;
  const int *mask = atom->mask;

  const double inv_cutsq = 1.0 / cutsq;

  double sums[18] = {};

  for (int ii = 0; ii < inum; ++ii) {
    const int i = ilist[ii];
    if (!(mask[i] & groupbit) || sp[i][3] <= 0.0) continue;

    const double si[3] = {sp[i][0], sp[i][1], sp[i][2]};
    const double fi[3] = {hbar * fm[i][0], hbar * fm[i][1], hbar * fm[i][2]};
    const double sdotf = si[0] * fi[0] + si[1] * fi[1] + si[2] * fi[2];
    const double fti[3] = {fi[0] - sdotf * si[0], fi[1] - sdotf * si[1], fi[2] - sdotf * si[2]};

    sums[14] += 1.0;
    sums[15] += si[0];
    sums[16] += si[1];
    sums[17] += si[2];

    const double xi = x[i][0], yi = x[i][1], zi = x[i][2];
    const int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    double Q[3][3] = {};
    double trQ = 0.0;

    for (int jj = 0; jj < jnum; ++jj) {
      const int j = jlist[jj] & NEIGHMASK;
      if (!(mask[j] & groupbit) || sp[j][3] <= 0.0) continue;

      const double dx = x[j][0] - xi;
      const double dy = x[j][1] - yi;
      const double dz = x[j][2] - zi;
      const double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq <= 1.0e-24 || rsq >= cutsq) continue;

      const double inv_r = 1.0 / std::sqrt(rsq);
      const double u[3] = {dx * inv_r, dy * inv_r, dz * inv_r};
      const double t = 1.0 - rsq * inv_cutsq;
      const double w = t * t * t;

      const double sj[3] = {sp[j][0], sp[j][1], sp[j][2]};
      const double xdot = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
      const double sj_dot_fti = sj[0] * fti[0] + sj[1] * fti[1] + sj[2] * fti[2];

      double p1, dp1, p2, dp2, p4, dp4, p8, dp8;
      SpinGPSTMath::legendre_pair_1248(xdot, p1, dp1, p2, dp2, p4, dp4, p8, dp8);
      sums[0] += w * dp1 * sj_dot_fti;
      sums[1] += 2.0 * w * p1;
      sums[2] += w * dp2 * sj_dot_fti;
      sums[3] += 6.0 * w * p2;
      sums[4] += w * dp4 * sj_dot_fti;
      sums[5] += 20.0 * w * p4;
      sums[6] += w * dp8 * sj_dot_fti;
      sums[7] += 72.0 * w * p8;

      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) Q[a][b] += w * u[a] * u[b];
      trQ += w;

      const double zdot = si[0] * u[0] + si[1] * u[1] + si[2] * u[2];
      const double fti_dot_u = fti[0] * u[0] + fti[1] * u[1] + fti[2] * u[2];
      double pz4, dpz4;
      SpinGPSTMath::legendre_4(zdot, pz4, dpz4);
      sums[10] += w * dpz4 * fti_dot_u;
      sums[11] += 20.0 * w * pz4;

      const double sxj[3] = {si[1] * sj[2] - si[2] * sj[1], si[2] * sj[0] - si[0] * sj[2],
                             si[0] * sj[1] - si[1] * sj[0]};
      const double sju[3] = {sj[1] * u[2] - sj[2] * u[1], sj[2] * u[0] - sj[0] * u[2],
                             sj[0] * u[1] - sj[1] * u[0]};
      sums[12] += w * (fti[0] * sju[0] + fti[1] * sju[1] + fti[2] * sju[2]);
      sums[13] += 2.0 * w * (u[0] * sxj[0] + u[1] * sxj[1] + u[2] * sxj[2]);
    }

    double qs[3];
    for (int a = 0; a < 3; ++a) qs[a] = Q[a][0] * si[0] + Q[a][1] * si[1] + Q[a][2] * si[2];
    sums[8] += qs[0] * fti[0] + qs[1] * fti[1] + qs[2] * fti[2];
    sums[9] += 3.0 * (qs[0] * si[0] + qs[1] * si[1] + qs[2] * si[2]) - trQ;
  }

  finalize_vector(sums);
}

int ComputeSpinGPST::pack_forward_comm(int n, int *sendlist, double *buf, int /*pbc_flag*/,
                                       int * /*pbc*/)
{
  double **sp = atom->sp;
  int m = 0;
  for (int i = 0; i < n; ++i) {
    const int idx = sendlist[i];
    buf[m++] = sp[idx][0];
    buf[m++] = sp[idx][1];
    buf[m++] = sp[idx][2];
    buf[m++] = sp[idx][3];
  }
  return m;
}

void ComputeSpinGPST::unpack_forward_comm(int n, int first, double *buf)
{
  double **sp = atom->sp;
  int m = 0;
  for (int i = first; i < first + n; ++i) {
    sp[i][0] = buf[m++];
    sp[i][1] = buf[m++];
    sp[i][2] = buf[m++];
    sp[i][3] = buf[m++];
  }
}
