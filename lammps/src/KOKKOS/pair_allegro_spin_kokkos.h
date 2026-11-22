/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(allegro_spin/kk,PairAllegroSpinKokkos<LMPDeviceType>);
PairStyle(allegro_spin/kk/device,PairAllegroSpinKokkos<LMPDeviceType>);
PairStyle(allegro_spin/kk/host,PairAllegroSpinKokkos<LMPHostType>);
PairStyle(spin/allegro_spin/kk,PairAllegroSpinKokkos<LMPDeviceType>);
PairStyle(spin/allegro_spin/kk/device,PairAllegroSpinKokkos<LMPDeviceType>);
PairStyle(spin/allegro_spin/kk/host,PairAllegroSpinKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_ALLEGRO_SPIN_KOKKOS_H
#define LMP_PAIR_ALLEGRO_SPIN_KOKKOS_H

#include "pair_allegro_spin.h"

#include "kokkos_type.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairAllegroSpinKokkos : public PairAllegro_spin {
 public:
  enum {EnabledNeighFlags=FULL};
  enum {COUL_FLAG=0};
  using device_type = DeviceType;
  using AT = ArrayTypes<DeviceType>;

  PairAllegroSpinKokkos(class LAMMPS *);
  ~PairAllegroSpinKokkos() override;

  void coeff(int, char **) override;
  void compute(int, int) override;
  void init_style() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

 private:
  Kokkos::View<int *, DeviceType> d_type_mapper;
};

} // namespace LAMMPS_NS

#endif
#endif
