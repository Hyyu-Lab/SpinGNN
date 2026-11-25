// SPDX-License-Identifier: GPL-2.0-only

#include "fix_spin_sib_npt_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain.h"

using namespace LAMMPS_NS;

template <class DeviceType>
FixSpinSIBNPTKokkos<DeviceType>::FixSpinSIBNPTKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixSpinSIBNPTKokkos(lmp, parse_spin_sib_npt(lmp, narg, arg))
{
}

template <class DeviceType>
FixSpinSIBNPTKokkos<DeviceType>::FixSpinSIBNPTKokkos(LAMMPS *lmp, SpinSIBNPTArguments args) :
    FixNPTKokkos<DeviceType>(lmp, static_cast<int>(args.nh.size()), args.nh.data()),
    sib(lmp), spins(lmp, sib)
{
  sib.configure(this->t_stop, args.alpha, args.seed);
}

template <class DeviceType> void FixSpinSIBNPTKokkos<DeviceType>::init()
{
  FixNPTKokkos<DeviceType>::init();
  sib.init(true, false, this->execution_space);
}

template <class DeviceType>
void FixSpinSIBNPTKokkos<DeviceType>::initial_integrate(int vflag)
{
  const bool bec = this->atom->bec_flag;
  if (bec) track_bec_displacement(true);
  FixNPTKokkos<DeviceType>::initial_integrate(vflag);
  if (bec) track_bec_displacement(false);
}

template <class DeviceType> void FixSpinSIBNPTKokkos<DeviceType>::nve_x()
{
  const auto *atom = this->atom;
  const int nlocal = this->igroup == atom->firstgroup ? atom->nfirst : atom->nlocal;
  spins.advance(nlocal, this->groupbit, false);
  FixNPTKokkos<DeviceType>::nve_x();
}

template <class DeviceType> void FixSpinSIBNPTKokkos<DeviceType>::reset_dt()
{
  FixNPTKokkos<DeviceType>::reset_dt();
  sib.reset_dt();
}

template <class DeviceType>
void FixSpinSIBNPTKokkos<DeviceType>::track_bec_displacement(bool begin)
{
  auto *atom = this->atomKK;
  const int nlocal = atom->nlocal;
  if (begin && reference.extent(0) < static_cast<size_t>(nlocal))
    reference = decltype(reference)(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "spin/sib_npt:BEC displacement"), atom->nmax);

  const auto fields = X_MASK | IMAGE_MASK | (begin ? EMPTY_MASK : BEC_MASK | BEC_DIPOLE_MASK);
  atom->sync(this->execution_space, fields);
  const auto x = atom->k_x.template view<DeviceType>();
  const auto image = atom->k_image.template view<DeviceType>();
  const auto z = atom->k_bec.template view<DeviceType>();
  const auto dipole = atom->k_bec_dipole.template view<DeviceType>();
  const auto old = reference;
  const auto *domain = this->domain;
  const double hxx = domain->xprd, hyy = domain->yprd, hzz = domain->zprd;
  const double hyz = domain->triclinic ? domain->yz : 0.0;
  const double hxz = domain->triclinic ? domain->xz : 0.0;
  const double hxy = domain->triclinic ? domain->xy : 0.0;

  // Unwrapping with the instantaneous cell accounts for both NH remaps,
  // including atoms whose image was already nonzero at the start of the step.
  Kokkos::parallel_for(
      "spin/sib_npt:BEC displacement", Kokkos::RangePolicy<DeviceType>(0, nlocal),
      KOKKOS_LAMBDA(int i) {
        const auto ix = (image(i) & IMGMASK) - IMGMAX;
        const auto iy = ((image(i) >> IMGBITS) & IMGMASK) - IMGMAX;
        const auto iz = (image(i) >> IMG2BITS) - IMGMAX;
        const double unwrap[] = {
            x(i, 0) + hxx * ix + hxy * iy + hxz * iz,
            x(i, 1) + hyy * iy + hyz * iz,
            x(i, 2) + hzz * iz,
        };
        if (begin) {
          for (int k = 0; k < 3; ++k) old(i, k) = unwrap[k];
        } else {
          for (int k = 0; k < 3; ++k) {
            const double delta = z(i, 3 * k) * (unwrap[0] - old(i, 0)) +
                z(i, 3 * k + 1) * (unwrap[1] - old(i, 1)) +
                z(i, 3 * k + 2) * (unwrap[2] - old(i, 2));
            dipole(i, k) += delta;
          }
        }
      });
  if (!begin) atom->modified(this->execution_space, BEC_DIPOLE_MASK);
}

template <class DeviceType> double FixSpinSIBNPTKokkos<DeviceType>::memory_usage()
{
  return FixNPTKokkos<DeviceType>::memory_usage() + spins.memory_usage() +
      reference.span() * sizeof(double);
}

namespace LAMMPS_NS {
template class FixSpinSIBNPTKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixSpinSIBNPTKokkos<LMPHostType>;
#endif
}  // namespace LAMMPS_NS
