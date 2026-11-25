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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(spin/gpst/kk,ComputeSpinGPSTKokkos<LMPDeviceType>);
ComputeStyle(spin/gpst/kk/device,ComputeSpinGPSTKokkos<LMPDeviceType>);
ComputeStyle(spin/gpst/kk/host,ComputeSpinGPSTKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_COMPUTE_SPIN_GPST_KOKKOS_H
#define LMP_COMPUTE_SPIN_GPST_KOKKOS_H

#include "kokkos_type.h"
#include "kokkos_base.h"
#include "compute_spin_gpst.h"

namespace LAMMPS_NS {

template <class DeviceType>
class ComputeSpinGPSTKokkos : public ComputeSpinGPST, public KokkosBase {
 public:
  using device_type = DeviceType;
  using AT = ArrayTypes<DeviceType>;

  ComputeSpinGPSTKokkos(class LAMMPS *, int, char **);
  ~ComputeSpinGPSTKokkos() override = default;

  void init() override;
  void compute_vector() override;

  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d &, int,
                               int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d &) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  // NVCC requires public access for methods enclosing KOKKOS_LAMBDA kernels.
  void accumulate_device(double local_sums[18]);

 private:
  Kokkos::View<double[18], DeviceType> d_sums;
  Kokkos::View<double[18], Kokkos::HostSpace> h_sums;
};

}  // namespace LAMMPS_NS

#endif
#endif
