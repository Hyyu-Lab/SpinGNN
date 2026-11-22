/* Standalone spin-field synchronization regression. Distributed under the LAMMPS GPL. */

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "atom_vec_spin_kokkos.h"
#include "input.h"
#include "kokkos.h"
#include "lammps.h"

#include <cuda_runtime.h>
#include <mpi.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace LAMMPS_NS;

static_assert(std::is_same<LMPDeviceType, Kokkos::Cuda>::value,
              "This regression requires actual Kokkos CUDA storage");

static void check(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}

template<class DualView>
static void initialize_field(DualView &field)
{
  check(field.extent(0) >= 2, "atom field has fewer than two rows");
  check(static_cast<const void *>(field.h_view.data()) !=
            static_cast<const void *>(field.d_view.data()),
        "host and CUDA views must be distinct allocations");
  field.template sync<LMPHostType>();
  Kokkos::deep_copy(field.h_view, 0.0);
  field.template modify<LMPHostType>();
  field.template sync<LMPDeviceType>();
  check(!field.template need_sync<LMPHostType>() &&
            !field.template need_sync<LMPDeviceType>(),
        "initial field synchronization failed");
}

template<class Mechanical, class Magnetic>
static void check_independent_fields(AtomKokkos &atom, Mechanical &mechanical,
                                     Magnetic &magnetic, unsigned int magnetic_mask)
{
  mechanical.h_view(0,0) = 17.0;
  mechanical.h_view(1,1) = 23.0;
  // Set only this DualView dirty: using the aliased production mask here
  // would contaminate the setup before the operation under test.
  mechanical.template modify<LMPHostType>();
  const auto device = magnetic.d_view;
  Kokkos::parallel_for("spin_sync_update", Kokkos::RangePolicy<LMPDeviceType>(0, 2),
      KOKKOS_LAMBDA(int i) {
        device(i,0) = 41.0 + i;
        device(i,1) = 51.0 + i;
        device(i,2) = 61.0 + i;
      });
  Kokkos::fence();
  atom.modified(Device, magnetic_mask);

  check(mechanical.template need_sync<LMPDeviceType>(),
        "magnetic modification erased an unrelated host-dirty field");
  check(!mechanical.template need_sync<LMPHostType>(),
        "magnetic modification also marked the mechanical device field");
  check(magnetic.template need_sync<LMPHostType>(),
        "magnetic device modification was not recorded");
  auto mechanical_device = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), mechanical.d_view);
  check(mechanical_device(0,0) == 0.0 && mechanical_device(1,1) == 0.0,
        "magnetic update unexpectedly copied mechanical host data");

  atom.sync(Host, magnetic_mask);
  check(magnetic.h_view(0,0) == 41.0 && magnetic.h_view(1,1) == 52.0,
        "magnetic device values did not reach their own host view");
  check(mechanical.h_view(0,0) == 17.0 && mechanical.h_view(1,1) == 23.0,
        "magnetic sync overwrote unrelated mechanical host values");
  check(mechanical.template need_sync<LMPDeviceType>(),
        "magnetic sync consumed an unrelated dirty flag");
  mechanical.template sync<LMPDeviceType>();
  mechanical_device = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), mechanical.d_view);
  check(mechanical_device(0,0) == 17.0 && mechanical_device(1,1) == 23.0,
        "independent mechanical synchronization lost its values");
}

static void check_overlapping_fm(AtomKokkos &atom)
{
  atom.k_fm.h_view(0,0) = 71.0;
  atom.k_fm.h_view(1,2) = 83.0;
  atom.k_fm.modify<LMPHostType>();
  check(!atom.k_sp.need_sync<LMPDeviceType>(), "spin must already be synchronized");
  check(atom.k_fm.need_sync<LMPDeviceType>(), "magnetic force must be host-dirty");
  atom.sync_overlapping_device(Device, FM_MASK);
  Kokkos::fence();
  const auto actual = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), atom.k_fm.d_view);
  check(actual(0,0) == 71.0 && actual(1,2) == 83.0,
        "overlapping FM sync used spin dirty state and left stale device values");
  check(!atom.k_fm.need_sync<LMPDeviceType>(), "overlapping FM sync left its dirty flag");
}

int main(int argc, char **argv)
{
  const std::string testcase = argc == 2 ? argv[1] : "";
  if (testcase != "x-sp" && testcase != "v-fm" && testcase != "f-fml" &&
      testcase != "async-fm") {
    std::cerr << "usage: spin-sync {x-sp|v-fm|f-fml|async-fm}\n";
    return 2;
  }
  MPI_Init(&argc, &argv);
  int status = 0;
  try {
    std::vector<std::string> arguments = {
      "spin-sync", "-k", "on", "g", "1", "-sf", "kk",
      "-log", "none", "-screen", "none", "-nocite"
    };
    std::vector<char *> pointers;
    for (auto &argument : arguments) pointers.push_back(argument.data());
    std::unique_ptr<LAMMPS> lmp(new LAMMPS(pointers.size(), pointers.data(), MPI_COMM_WORLD));
    for (const char *command : {
        "units metal", "atom_style spin/kk", "boundary p p p",
        "region box block 0 5 0 5 0 5", "create_box 1 box",
        "create_atoms 1 single 1 1 1 units box",
        "create_atoms 1 single 2 2 2 units box", "mass 1 1.0"})
      lmp->input->one(command);
    auto *atom = dynamic_cast<AtomKokkos *>(lmp->atom);
    check(atom && dynamic_cast<AtomVecSpinKokkos *>(atom->avec),
          "test did not create an actual spin/kk atom vector");
    check(atom->nlocal == 2, "test requires exactly two local atoms");
    lmp->kokkos->auto_sync = 0;
    initialize_field(atom->k_x);
    initialize_field(atom->k_v);
    initialize_field(atom->k_f);
    initialize_field(atom->k_sp);
    initialize_field(atom->k_fm);
    initialize_field(atom->k_fm_long);
    cudaDeviceProp properties;
    int device = -1;
    check(cudaGetDevice(&device) == cudaSuccess, "CUDA device query failed");
    check(cudaGetDeviceProperties(&properties, device) == cudaSuccess,
          "CUDA properties query failed");
    std::cout << "SPIN_SYNC_DEVICE name=" << properties.name
              << " capability=" << properties.major << '.' << properties.minor << '\n';
    std::cout << "SPIN_SYNC_BEGIN case=" << testcase << " nlocal=" << atom->nlocal
              << " masks=" << SP_MASK << ',' << FM_MASK << ',' << FML_MASK << std::endl;
    if (testcase == "x-sp") check_independent_fields(*atom, atom->k_x, atom->k_sp, SP_MASK);
    if (testcase == "v-fm") check_independent_fields(*atom, atom->k_v, atom->k_fm, FM_MASK);
    if (testcase == "f-fml") check_independent_fields(*atom, atom->k_f, atom->k_fm_long, FML_MASK);
    if (testcase == "async-fm") check_overlapping_fm(*atom);
    std::cout << "SPIN_SYNC_RESULT case=" << testcase << " result=PASS" << std::endl;
  } catch (const std::exception &error) {
    std::cerr << "SPIN_SYNC_CHECK_FAILED case=" << testcase << " reason=" << error.what() << '\n';
    status = 1;
  }
  // LAMMPS and all View handles are gone before the Kokkos runtime is finalized.
  KokkosLMP::finalize();
  MPI_Finalize();
  return status;
}
