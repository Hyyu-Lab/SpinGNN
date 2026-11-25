/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef FIX_CLASS
// clang-format off
FixStyle(spin/cfi/kk,FixSpinCFIKokkos<LMPDeviceType>);
FixStyle(spin/cfi/kk/device,FixSpinCFIKokkos<LMPDeviceType>);
FixStyle(spin/cfi/kk/host,FixSpinCFIKokkos<LMPHostType>);
FixStyle(spin/lpsib/kk,FixSpinCFIKokkos<LMPDeviceType>);
FixStyle(spin/lpsib/kk/device,FixSpinCFIKokkos<LMPDeviceType>);
FixStyle(spin/lpsib/kk/host,FixSpinCFIKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_SPIN_CFI_KOKKOS_H
#define LMP_FIX_SPIN_CFI_KOKKOS_H

#include "kokkos_type.h"
#include "kokkos_base.h"
#include "fix_spin_cfi.h"

namespace LAMMPS_NS {

template <class DeviceType> class FixSpinCFIKokkos : public FixSpinCFI, public KokkosBase {
 public:
  using device_type = DeviceType;
  using AT = ArrayTypes<DeviceType>;

  FixSpinCFIKokkos(class LAMMPS *, int, char **);
  ~FixSpinCFIKokkos() override;

  void init() override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void end_of_step() override;
  void post_run() override;
  double compute_vector(int) override;

  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  void set_arrays(int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  double memory_usage() override;

  int pack_exchange_kokkos(const int &nsend, DAT::tdual_double_2d_lr &buf,
                           DAT::tdual_int_1d k_sendlist, DAT::tdual_int_1d k_copylist,
                           ExecutionSpace space) override;
  void unpack_exchange_kokkos(DAT::tdual_double_2d_lr &k_buf, DAT::tdual_int_1d &indices, int nrecv,
                              int nrecv1, int nextrarecv1, ExecutionSpace space) override;
  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter) override;
  void sync_host_for_sort() override;
  void modified_host_for_sort() override;

  // NVCC requires public access for methods enclosing KOKKOS_LAMBDA kernels.
  void initial_integrate_fused(int nlocal);
  void final_integrate_fused(int nlocal);
  void predict_device(int nlocal, bool use_live_fm, bool do_seed);
  void correct_device(int nlocal);
  void box_bath_half_device(int nlocal, int half);
  void box_T_device(int nlocal, double a);
  void box_D_device(int nlocal, double a);
  void box_sync_device(int nlocal);
  void box_kick_device(int nlocal);
  void evaluate_pair_fresh_device(bool check_reneighbor);
  void flush_device_status();

 private:
  DAT::tdual_double_1d_6_lr k_store;
  typename AT::t_double_1d_6_lr d_store;
  DAT::tdual_double_1d_6_lr k_box_store;
  typename AT::t_double_1d_6_lr d_box_store;

  Kokkos::View<int, DeviceType> d_bad_count;
  Kokkos::View<int, Kokkos::HostSpace> h_bad_count;
  Kokkos::View<double, DeviceType> d_maxdev;
  Kokkos::View<double, Kokkos::HostSpace> h_maxdev;
};

}  // namespace LAMMPS_NS

#endif
#endif
