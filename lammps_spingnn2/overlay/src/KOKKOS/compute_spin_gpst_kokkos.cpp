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

#include "compute_spin_gpst_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor_kokkos.h"
#include "update.h"

#include <cmath>
#include <type_traits>

using namespace LAMMPS_NS;

template <class DeviceType>
ComputeSpinGPSTKokkos<DeviceType>::ComputeSpinGPSTKokkos(LAMMPS *lmp, int narg, char **arg) :
    ComputeSpinGPST(lmp, narg, arg), d_sums("compute/spin/gpst:d_sums"),
    h_sums("compute/spin/gpst:h_sums")
{
  kokkosable = 1;
  forward_comm_device = 1;
  atomKK = static_cast<AtomKokkos *>(atom);
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
}

template <class DeviceType> void ComputeSpinGPSTKokkos<DeviceType>::init()
{
  ComputeSpinGPST::init();
  auto *request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> &&
                           !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
}

template <class DeviceType>
void ComputeSpinGPSTKokkos<DeviceType>::accumulate_device(double local_sums[18])
{
  auto *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  const int inum = k_list->inum;
  const auto d_ilist = k_list->d_ilist;
  const auto d_numneigh = k_list->d_numneigh;
  const auto d_neighbors = k_list->d_neighbors;

  atomKK->sync(execution_space, X_MASK | SP_MASK | FM_MASK | MASK_MASK);
  const auto x = atomKK->k_x.template view<DeviceType>();
  const auto sp = atomKK->k_sp.template view<DeviceType>();
  const auto fm = atomKK->k_fm.template view<DeviceType>();
  const auto mask = atomKK->k_mask.template view<DeviceType>();

  const int bit = groupbit;
  const double cutsq_v = cutsq;
  const double inv_cutsq = 1.0 / cutsq;
  const double hbar_v = hbar;

  Kokkos::deep_copy(d_sums, 0.0);
  auto sums_v = d_sums;

  // Register-only per-thread accumulation with 0 B/thread shared memory.
  Kokkos::parallel_for(
      "compute/spin/gpst", Kokkos::RangePolicy<DeviceType>(0, inum),
      KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist(ii);
        if (!(mask(i) & bit) || sp(i, 3) <= 0.0) return;

        const double si[3] = {static_cast<double>(sp(i, 0)), static_cast<double>(sp(i, 1)),
                              static_cast<double>(sp(i, 2))};
        const double fi[3] = {hbar_v * static_cast<double>(fm(i, 0)),
                              hbar_v * static_cast<double>(fm(i, 1)),
                              hbar_v * static_cast<double>(fm(i, 2))};
        const double sdotf = si[0] * fi[0] + si[1] * fi[1] + si[2] * fi[2];
        const double fti[3] = {fi[0] - sdotf * si[0], fi[1] - sdotf * si[1],
                               fi[2] - sdotf * si[2]};

        double acc[18] = {};
        acc[14] = 1.0;
        acc[15] = si[0];
        acc[16] = si[1];
        acc[17] = si[2];

        const double xi = static_cast<double>(x(i, 0));
        const double yi = static_cast<double>(x(i, 1));
        const double zi = static_cast<double>(x(i, 2));
        const int jnum = d_numneigh(i);

        double Q00 = 0.0, Q01 = 0.0, Q02 = 0.0;
        double Q11 = 0.0, Q12 = 0.0, Q22 = 0.0;
        double trQ = 0.0;

        for (int jj = 0; jj < jnum; ++jj) {
          const int j = d_neighbors(i, jj) & NEIGHMASK;
          if (!(mask(j) & bit) || sp(j, 3) <= 0.0) continue;

          const double dx = static_cast<double>(x(j, 0)) - xi;
          const double dy = static_cast<double>(x(j, 1)) - yi;
          const double dz = static_cast<double>(x(j, 2)) - zi;
          const double rsq = dx * dx + dy * dy + dz * dz;
          if (rsq <= 1.0e-24 || rsq >= cutsq_v) continue;

          const double inv_r = 1.0 / ::sqrt(rsq);
          const double ux = dx * inv_r;
          const double uy = dy * inv_r;
          const double uz = dz * inv_r;
          const double t = 1.0 - rsq * inv_cutsq;
          const double w = t * t * t;

          const double sjx = static_cast<double>(sp(j, 0));
          const double sjy = static_cast<double>(sp(j, 1));
          const double sjz = static_cast<double>(sp(j, 2));
          const double xdot = si[0] * sjx + si[1] * sjy + si[2] * sjz;
          const double sj_dot_fti = sjx * fti[0] + sjy * fti[1] + sjz * fti[2];

          double p1, dp1, p2, dp2, p4, dp4, p8, dp8;
          SpinGPSTMath::legendre_pair_1248(xdot, p1, dp1, p2, dp2, p4, dp4, p8, dp8);
          acc[0] += w * dp1 * sj_dot_fti;
          acc[1] += 2.0 * w * p1;
          acc[2] += w * dp2 * sj_dot_fti;
          acc[3] += 6.0 * w * p2;
          acc[4] += w * dp4 * sj_dot_fti;
          acc[5] += 20.0 * w * p4;
          acc[6] += w * dp8 * sj_dot_fti;
          acc[7] += 72.0 * w * p8;

          Q00 += w * ux * ux;
          Q01 += w * ux * uy;
          Q02 += w * ux * uz;
          Q11 += w * uy * uy;
          Q12 += w * uy * uz;
          Q22 += w * uz * uz;
          trQ += w;

          const double zdot = si[0] * ux + si[1] * uy + si[2] * uz;
          const double fti_dot_u = fti[0] * ux + fti[1] * uy + fti[2] * uz;
          double pz4, dpz4;
          SpinGPSTMath::legendre_4(zdot, pz4, dpz4);
          acc[10] += w * dpz4 * fti_dot_u;
          acc[11] += 20.0 * w * pz4;

          const double sxj0 = si[1] * sjz - si[2] * sjy;
          const double sxj1 = si[2] * sjx - si[0] * sjz;
          const double sxj2 = si[0] * sjy - si[1] * sjx;
          const double sju0 = sjy * uz - sjz * uy;
          const double sju1 = sjz * ux - sjx * uz;
          const double sju2 = sjx * uy - sjy * ux;
          acc[12] += w * (fti[0] * sju0 + fti[1] * sju1 + fti[2] * sju2);
          acc[13] += 2.0 * w * (ux * sxj0 + uy * sxj1 + uz * sxj2);
        }

        const double qs0 = Q00 * si[0] + Q01 * si[1] + Q02 * si[2];
        const double qs1 = Q01 * si[0] + Q11 * si[1] + Q12 * si[2];
        const double qs2 = Q02 * si[0] + Q12 * si[1] + Q22 * si[2];
        acc[8] = qs0 * fti[0] + qs1 * fti[1] + qs2 * fti[2];
        acc[9] = 3.0 * (qs0 * si[0] + qs1 * si[1] + qs2 * si[2]) - trQ;

        for (int k = 0; k < 18; ++k) {
          if (acc[k] != 0.0) Kokkos::atomic_add(&sums_v(k), acc[k]);
        }
      });

  Kokkos::deep_copy(h_sums, d_sums);
  for (int k = 0; k < 18; ++k) local_sums[k] = h_sums(k);
}

template <class DeviceType> void ComputeSpinGPSTKokkos<DeviceType>::compute_vector()
{
  const int prev_auto_sync = lmp->kokkos->auto_sync;
  lmp->kokkos->auto_sync = 0;
  invoked_vector = update->ntimestep;

  comm->forward_comm(this);
  neighbor->build_one(list);

  double local_sums[18] = {};
  accumulate_device(local_sums);
  finalize_vector(local_sums);

  lmp->kokkos->auto_sync = prev_auto_sync;
}

template <class DeviceType>
int ComputeSpinGPSTKokkos<DeviceType>::pack_forward_comm_kokkos(int n,
                                                                DAT::tdual_int_1d k_sendlist,
                                                                DAT::tdual_double_1d &k_buf,
                                                                int /*pbc_flag*/, int * /*pbc*/)
{
  auto d_sendlist = k_sendlist.template view<DeviceType>();
  auto d_buf = k_buf.template view<DeviceType>();
  atomKK->sync(execution_space, SP_MASK);
  const auto sp = atomKK->k_sp.template view<DeviceType>();

  Kokkos::parallel_for(
      "compute/spin/gpst:pack_forward", Kokkos::RangePolicy<DeviceType>(0, n),
      KOKKOS_LAMBDA(const int i) {
        const int idx = d_sendlist(i);
        d_buf(4 * i + 0) = static_cast<double>(sp(idx, 0));
        d_buf(4 * i + 1) = static_cast<double>(sp(idx, 1));
        d_buf(4 * i + 2) = static_cast<double>(sp(idx, 2));
        d_buf(4 * i + 3) = static_cast<double>(sp(idx, 3));
      });
  return 4 * n;
}

template <class DeviceType>
void ComputeSpinGPSTKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first,
                                                                   DAT::tdual_double_1d &k_buf)
{
  auto d_buf = k_buf.template view<DeviceType>();
  atomKK->sync(execution_space, SP_MASK);
  auto sp = atomKK->k_sp.template view<DeviceType>();

  Kokkos::parallel_for(
      "compute/spin/gpst:unpack_forward", Kokkos::RangePolicy<DeviceType>(0, n),
      KOKKOS_LAMBDA(const int i) {
        sp(first + i, 0) = static_cast<KK_FLOAT>(d_buf(4 * i + 0));
        sp(first + i, 1) = static_cast<KK_FLOAT>(d_buf(4 * i + 1));
        sp(first + i, 2) = static_cast<KK_FLOAT>(d_buf(4 * i + 2));
        sp(first + i, 3) = static_cast<KK_FLOAT>(d_buf(4 * i + 3));
      });
  atomKK->modified(execution_space, SP_MASK);
}

template <class DeviceType>
int ComputeSpinGPSTKokkos<DeviceType>::pack_forward_comm(int n, int *sendlist, double *buf,
                                                         int pbc_flag, int *pbc)
{
  atomKK->sync(Host, SP_MASK);
  return ComputeSpinGPST::pack_forward_comm(n, sendlist, buf, pbc_flag, pbc);
}

template <class DeviceType>
void ComputeSpinGPSTKokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf)
{
  atomKK->sync(Host, SP_MASK);
  ComputeSpinGPST::unpack_forward_comm(n, first, buf);
  atomKK->modified(Host, SP_MASK);
}

namespace LAMMPS_NS {
template class ComputeSpinGPSTKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputeSpinGPSTKokkos<LMPHostType>;
#endif
}  // namespace LAMMPS_NS
