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

#ifdef FIX_CLASS
// clang-format off
FixStyle(spin/sib/kk,FixSpinSIBKokkos<LMPDeviceType>);
FixStyle(spin/sib/kk/device,FixSpinSIBKokkos<LMPDeviceType>);
FixStyle(spin/sib/kk/host,FixSpinSIBKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_SPIN_SIB_KOKKOS_H
#define LMP_FIX_SPIN_SIB_KOKKOS_H

#include "fix_spin_sib.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template <class DeviceType> class SpinSIBDevice : protected Pointers {
 public:
  SpinSIBDevice(LAMMPS *, SpinSIB &);
  void advance(int nlocal, int groupbit, bool reuse_lattice);
  double memory_usage() const;

  // NVCC requires public access for a member enclosing an extended lambda.
  void advance_spin(int nlocal, int groupbit, bool midpoint);

 private:
  SpinSIB &sib;
  Kokkos::View<double *[6], DeviceType> workspace;
};

template <class DeviceType> class FixSpinSIBKokkos : public FixNVESpinSIB {
 public:
  FixSpinSIBKokkos(LAMMPS *, int, char **);
  void init() override;
  void initial_integrate(int) override;
  void final_integrate() override;
  double memory_usage() override;

  // NVCC requires public access for a member enclosing an extended lambda.
  void integrate_lattice(int nlocal, bool drift);

 private:
  SpinSIBDevice<DeviceType> spins;
};

}  // namespace LAMMPS_NS

#endif
#endif
