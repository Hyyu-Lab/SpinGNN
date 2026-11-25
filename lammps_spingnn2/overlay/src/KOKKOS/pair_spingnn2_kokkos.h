// SPDX-License-Identifier: GPL-2.0-only
#ifdef PAIR_CLASS
PairStyle(spingnn2/kk,PairSpinGNN2Kokkos);
#else
#ifndef LMP_PAIR_SPINGNN2_KOKKOS_H
#define LMP_PAIR_SPINGNN2_KOKKOS_H
#include "pair_spingnn2.h"
#include "kokkos_base.h"

namespace LAMMPS_NS {
class PairSpinGNN2Kokkos : public PairSpinGNN2, public KokkosBase {
 public:
  explicit PairSpinGNN2Kokkos(LAMMPS *);
  void init_style() override;
  // CUDA extended lambdas require a publicly accessible enclosing function.
  template<class Real> void accumulate(const sp2_device_result &,bool);
  int pack_reverse_comm_kokkos(int,int,DAT::tdual_double_1d &) override;
  void unpack_reverse_comm_kokkos(int,DAT::tdual_int_1d,DAT::tdual_double_1d &) override;
 protected:
  sp2_device_result last_device{};
  void read_field(const std::string &, double *, size_t) override;
  int load_model(const char *) override;
  void collect_edges() override;
  void evaluate(bool,bool) override;
 private:
  using Device = LMPDeviceType;
  Kokkos::View<int*,Device> type_map;
  Kokkos::View<double*[6],Device> endpoints;
  Kokkos::View<double[9],Device> bec_sum;
  Kokkos::View<double[7],Device> globals;
#ifdef KOKKOS_ENABLE_CUDA
  Kokkos::View<double[7],Kokkos::CudaHostPinnedSpace> h_globals;
  Kokkos::View<double[9],Kokkos::CudaHostPinnedSpace> h_bec_sum;
#else
  Kokkos::View<double[7],Kokkos::HostSpace> h_globals;
  Kokkos::View<double[9],Kokkos::HostSpace> h_bec_sum;
#endif
  Kokkos::View<double*[10],Kokkos::LayoutRight,Device> atom_tallies;
  bool midpoint_active = false;
  uint64_t topology_epoch = 0, lattice_epoch = 1;
  bigint last_build = -1;
};
}
#endif
#endif
