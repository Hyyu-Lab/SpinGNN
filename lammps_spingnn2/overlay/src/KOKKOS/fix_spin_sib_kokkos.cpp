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

#include "fix_spin_sib_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "spin_sib_math.h"
#include "update.h"

using namespace LAMMPS_NS;

template <class DeviceType>
FixSpinSIBKokkos<DeviceType>::FixSpinSIBKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixNVESpinSIB(lmp, narg, arg), spins(lmp, sib)
{
  kokkosable = 1;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  // The integration methods synchronize their precise field set so a
  // frozen spin step needs no lattice or BEC views.
  datamask_read = datamask_modify = EMPTY_MASK;
}

template <class DeviceType> void FixSpinSIBKokkos<DeviceType>::init()
{
  FixNVESpinSIB::init();
  if (lattice_moving) {
    atomKK->k_mass.modify_host();
    atomKK->k_mass.template sync<DeviceType>();
  }
}

template <class DeviceType>
void FixSpinSIBKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
  const int nlocal = integrated_atoms();
  spins.advance(nlocal, groupbit, true);
  if (lattice_moving) integrate_lattice(nlocal, true);
}

template <class DeviceType>
SpinSIBDevice<DeviceType>::SpinSIBDevice(LAMMPS *lmp, SpinSIB &state) : Pointers(lmp), sib(state)
{
}

template <class DeviceType>
void SpinSIBDevice<DeviceType>::advance(int nlocal, int groupbit, bool reuse_lattice)
{
  if (workspace.extent(0) < static_cast<size_t>(nlocal))
    workspace = decltype(workspace)(Kokkos::view_alloc(Kokkos::WithoutInitializing, "spin/sib"),
                                    atom->nmax);
  advance_spin(nlocal, groupbit, true);
  // Comms can read the same device view immediately after the update.
  DeviceType().fence();
  sib.midpoint(reuse_lattice);
  advance_spin(nlocal, groupbit, false);
}

template <class DeviceType>
void SpinSIBDevice<DeviceType>::advance_spin(int nlocal, int groupbit, bool midpoint)
{
  constexpr auto execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  const auto tag_field = midpoint && sib.thermal != 0.0 ? TAG_MASK : EMPTY_MASK;
  atomKK->sync(execution_space, SP_MASK | FM_MASK | MASK_MASK | tag_field);
  const auto sp = atomKK->k_sp.template view<DeviceType>();
  const auto fm = atomKK->k_fm.template view<DeviceType>();
  const auto mask = atomKK->k_mask.template view<DeviceType>();
  const auto tag = atomKK->k_tag.template view<DeviceType>();
  const auto work = workspace;
  const int bit = groupbit;
  const int seed = sib.seed;
  const auto step = update->ntimestep;
  const double hbar = sib.hbar, alpha = sib.alpha, thermal = sib.thermal, dts = sib.dts;

  Kokkos::parallel_for(
      "spin/sib:rotation", Kokkos::RangePolicy<DeviceType>(0, nlocal),
      KOKKOS_LAMBDA(int i) {
        if (!(mask(i) & bit) || sp(i, 3) <= 0.0) return;
        double original[3], direction[3], torque[3], noise[3] = {}, result[3];
        for (int k = 0; k < 3; ++k) {
          direction[k] = sp(i, k);
          torque[k] = fm(i, k);
        }
        if (midpoint) {
          if (thermal != 0.0) SpinSIBMath::gaussian(seed, tag(i), step, noise);
          for (int k = 0; k < 3; ++k) {
            original[k] = work(i, k) = direction[k];
            work(i, k + 3) = noise[k];
          }
        } else {
          for (int k = 0; k < 3; ++k) {
            original[k] = work(i, k);
            noise[k] = work(i, k + 3);
          }
        }
        SpinSIBMath::advance(original, direction, torque, noise, hbar, alpha, thermal,
                             dts / sp(i, 3), midpoint, result);
        for (int k = 0; k < 3; ++k) sp(i, k) = result[k];
      });
  atomKK->modified(execution_space, SP_MASK);
}

template <class DeviceType> void FixSpinSIBKokkos<DeviceType>::final_integrate()
{
  if (lattice_moving) integrate_lattice(integrated_atoms(), false);
}

template <class DeviceType>
void FixSpinSIBKokkos<DeviceType>::integrate_lattice(int nlocal, bool drift)
{
  const bool dipoles = drift && atom->bec_flag;
  const auto bec_fields = dipoles ? BEC_MASK | BEC_DIPOLE_MASK : EMPTY_MASK;
  atomKK->sync(execution_space,
               V_MASK | F_MASK | MASK_MASK | TYPE_MASK | RMASS_MASK |
                   (drift ? X_MASK : EMPTY_MASK) | bec_fields);
  atomKK->k_mass.template sync<DeviceType>();

  const auto x = atomKK->k_x.template view<DeviceType>();
  const auto v = atomKK->k_v.template view<DeviceType>();
  const auto f = atomKK->k_f.template view<DeviceType>();
  const auto type = atomKK->k_type.template view<DeviceType>();
  const auto mask = atomKK->k_mask.template view<DeviceType>();
  const auto mass = atomKK->k_mass.template view<DeviceType>();
  const auto rmass = atomKK->k_rmass.template view<DeviceType>();
  const auto bec = atomKK->k_bec.template view<DeviceType>();
  const auto dipole = atomKK->k_bec_dipole.template view<DeviceType>();
  const double dtv = this->dtv;
  const double dtf = this->dtf;
  const int bit = groupbit;
  const bool per_atom_mass = rmass.data() != nullptr;

  Kokkos::parallel_for(
      "spin/sib:lattice", Kokkos::RangePolicy<DeviceType>(0, nlocal),
      KOKKOS_LAMBDA(int i) {
        if (!(mask(i) & bit)) return;
        const double kick = dtf / (per_atom_mass ? rmass(i) : mass(type(i)));
        for (int k = 0; k < 3; ++k) v(i, k) += kick * f(i, k);
        if (!drift) return;
        double dx[3];
        for (int k = 0; k < 3; ++k) {
          dx[k] = dtv * v(i, k);
          x(i, k) += dx[k];
        }
        if (dipoles)
          for (int k = 0; k < 3; ++k)
            dipole(i, k) += bec(i, 3 * k) * dx[0] + bec(i, 3 * k + 1) * dx[1] +
                bec(i, 3 * k + 2) * dx[2];
      });
  atomKK->modified(execution_space, V_MASK | (drift ? X_MASK : EMPTY_MASK) |
                         (dipoles ? BEC_DIPOLE_MASK : EMPTY_MASK));
}

template <class DeviceType> double FixSpinSIBKokkos<DeviceType>::memory_usage()
{
  return spins.memory_usage();
}

template <class DeviceType> double SpinSIBDevice<DeviceType>::memory_usage() const
{
  return workspace.span() * sizeof(double);
}

namespace LAMMPS_NS {
template class FixSpinSIBKokkos<LMPDeviceType>;
template class SpinSIBDevice<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixSpinSIBKokkos<LMPHostType>;
template class SpinSIBDevice<LMPHostType>;
#endif
}  // namespace LAMMPS_NS
