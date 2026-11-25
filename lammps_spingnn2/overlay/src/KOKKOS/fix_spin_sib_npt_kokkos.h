// SPDX-License-Identifier: GPL-2.0-only

#ifdef FIX_CLASS
// clang-format off
FixStyle(spin/sib_npt/kk,FixSpinSIBNPTKokkos<LMPDeviceType>);
FixStyle(spin/sib_npt/kk/device,FixSpinSIBNPTKokkos<LMPDeviceType>);
FixStyle(spin/sib_npt/kk/host,FixSpinSIBNPTKokkos<LMPHostType>);
FixStyle(spin/sib/npt/kk,FixSpinSIBNPTKokkos<LMPDeviceType>);
FixStyle(spin/sib/npt/kk/device,FixSpinSIBNPTKokkos<LMPDeviceType>);
FixStyle(spin/sib/npt/kk/host,FixSpinSIBNPTKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_SPIN_SIB_NPT_KOKKOS_H
#define LMP_FIX_SPIN_SIB_NPT_KOKKOS_H

#include "fix_npt_kokkos.h"
#include "fix_spin_sib_kokkos.h"
#include "fix_spin_sib_npt.h"

namespace LAMMPS_NS {

template <class DeviceType> class FixSpinSIBNPTKokkos : public FixNPTKokkos<DeviceType> {
 public:
  FixSpinSIBNPTKokkos(LAMMPS *, int, char **);
  void init() override;
  void initial_integrate(int) override;
  void reset_dt() override;
  double memory_usage() override;

  // NVCC requires public access for a member enclosing an extended lambda.
  void track_bec_displacement(bool begin);

 protected:
  void nve_x() override;

 private:
  FixSpinSIBNPTKokkos(LAMMPS *, SpinSIBNPTArguments);

  SpinSIB sib;
  SpinSIBDevice<DeviceType> spins;
  Kokkos::View<double *[3], DeviceType> reference;
};

}  // namespace LAMMPS_NS

#endif
#endif
