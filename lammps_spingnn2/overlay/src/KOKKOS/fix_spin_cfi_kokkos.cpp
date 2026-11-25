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

#include "fix_spin_cfi_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kokkos_few.h"
#include "memory.h"
#include "memory_kokkos.h"
#include "neighbor.h"
#include "pair.h"
#include "spin_cfi_math.h"
#include "update.h"

#include <cmath>
#include <limits>

using namespace LAMMPS_NS;

template <class DeviceType>
FixSpinCFIKokkos<DeviceType>::FixSpinCFIKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixSpinCFI(lmp, narg, arg), d_bad_count("spin/cfi:d_bad_count"),
    h_bad_count("spin/cfi:h_bad_count"), d_maxdev("spin/cfi:d_maxdev"),
    h_maxdev("spin/cfi:h_maxdev")
{
  kokkosable = 1;
  exchange_comm_device = 1;
  sort_device = 1;
  atomKK = static_cast<AtomKokkos *>(atom);
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = datamask_modify = EMPTY_MASK;

  // Replace base host-only store (and optional box_store) with Kokkos DualViews.
  auto *old_store = store;
  store = nullptr;
  auto *old_box_store = box_store;
  box_store = nullptr;
  grow_arrays(atom->nmax);

  auto h_store = k_store.view_host();
  for (int i = 0; i < atom->nlocal; ++i)
    for (int k = 0; k < 6; ++k) h_store(i, k) = old_store[i][k];
  k_store.modify_host();
  memory->destroy(old_store);

  if (box_iso) {
    auto h_box = k_box_store.view_host();
    for (int i = 0; i < atom->nlocal; ++i)
      for (int k = 0; k < 6; ++k) h_box(i, k) = old_box_store[i][k];
    k_box_store.modify_host();
    memory->destroy(old_box_store);
  }

  Kokkos::deep_copy(d_bad_count, 0);
  Kokkos::deep_copy(d_maxdev, 0.0);
}

template <class DeviceType> FixSpinCFIKokkos<DeviceType>::~FixSpinCFIKokkos()
{
  if (copymode) return;
  memoryKK->destroy_kokkos(k_store, store);
  store = nullptr;
  if (box_iso) {
    memoryKK->destroy_kokkos(k_box_store, box_store);
    box_store = nullptr;
  }
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::init()
{
  const bool did_box_init = box_iso && !box_initialized;
  if (did_box_init) {
    atomKK->sync(Host, X_MASK | V_MASK | IMAGE_MASK | TYPE_MASK | RMASS_MASK);
    k_box_store.sync_host();
  }
  FixSpinCFI::init();
  if (lattice_moving) {
    atomKK->k_mass.modify_host();
    atomKK->k_mass.template sync<DeviceType>();
  }
  if (did_box_init) {
    k_box_store.modify_host();
    k_box_store.template sync<DeviceType>();
  }
  Kokkos::deep_copy(d_bad_count, 0);
  Kokkos::deep_copy(d_maxdev, 0.0);
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::grow_arrays(int nmax)
{
  memoryKK->grow_kokkos(k_store, store, nmax, "spin/cfi:store");
  d_store = k_store.template view<DeviceType>();
  if (box_iso) {
    memoryKK->grow_kokkos(k_box_store, box_store, nmax, "spin/cfi:box_store");
    d_box_store = k_box_store.template view<DeviceType>();
  }
}

template <class DeviceType>
void FixSpinCFIKokkos<DeviceType>::copy_arrays(int i, int j, int delflag)
{
  k_store.sync_host();
  if (box_iso) k_box_store.sync_host();
  FixSpinCFI::copy_arrays(i, j, delflag);
  k_store.modify_host();
  if (box_iso) k_box_store.modify_host();
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::set_arrays(int i)
{
  k_store.sync_host();
  if (box_iso) k_box_store.sync_host();
  FixSpinCFI::set_arrays(i);
  k_store.modify_host();
  if (box_iso) k_box_store.modify_host();
}

template <class DeviceType> int FixSpinCFIKokkos<DeviceType>::pack_exchange(int i, double *buf)
{
  k_store.sync_host();
  if (box_iso) k_box_store.sync_host();
  return FixSpinCFI::pack_exchange(i, buf);
}

template <class DeviceType>
int FixSpinCFIKokkos<DeviceType>::unpack_exchange(int nlocal, double *buf)
{
  k_store.sync_host();
  if (box_iso) k_box_store.sync_host();
  const int m = FixSpinCFI::unpack_exchange(nlocal, buf);
  k_store.modify_host();
  if (box_iso) k_box_store.modify_host();
  return m;
}

template <class DeviceType> double FixSpinCFIKokkos<DeviceType>::memory_usage()
{
  double bytes = static_cast<double>(k_store.span()) * sizeof(double);
  if (box_iso) bytes += static_cast<double>(k_box_store.span()) * sizeof(double);
  return bytes;
}

template <class DeviceType>
int FixSpinCFIKokkos<DeviceType>::pack_exchange_kokkos(const int &nsend,
                                                       DAT::tdual_double_2d_lr &k_buf,
                                                       DAT::tdual_int_1d k_sendlist,
                                                       DAT::tdual_int_1d k_copylist,
                                                       ExecutionSpace space)
{
  if (nsend == 0) return 0;
  const bool has_box = box_iso;
  const int stride = has_box ? 12 : 6;

  k_buf.template sync<DeviceType>();
  k_sendlist.template sync<DeviceType>();
  k_copylist.template sync<DeviceType>();
  k_store.template sync<DeviceType>();
  if (has_box) k_box_store.template sync<DeviceType>();

  auto d_buf = typename AT::t_double_1d_um(k_buf.template view<DeviceType>().data(),
                                           k_buf.extent(0) * k_buf.extent(1));
  auto sendlist = k_sendlist.template view<DeviceType>();
  auto copylist = k_copylist.template view<DeviceType>();
  auto st = k_store.template view<DeviceType>();
  auto bst = k_box_store.template view<DeviceType>();

  Kokkos::parallel_for(
      "spin/cfi:pack_exchange", Kokkos::RangePolicy<DeviceType>(0, nsend),
      KOKKOS_LAMBDA(const int mysend) {
        const int i = sendlist(mysend);
        const int m = nsend + stride * mysend;
        d_buf(mysend) = static_cast<double>(m);
        for (int k = 0; k < 6; ++k) d_buf(m + k) = st(i, k);
        if (has_box) {
          for (int k = 0; k < 6; ++k) d_buf(m + 6 + k) = bst(i, k);
        }
        const int j = copylist(mysend);
        if (j > -1) {
          for (int k = 0; k < 6; ++k) st(i, k) = st(j, k);
          if (has_box) {
            for (int k = 0; k < 6; ++k) bst(i, k) = bst(j, k);
          }
        }
      });

  k_buf.template modify<DeviceType>();
  if (space == Host || space == HostKK)
    k_buf.sync_host();
  else
    k_buf.sync_device();
  k_store.template modify<DeviceType>();
  if (has_box) k_box_store.template modify<DeviceType>();
  return (stride + 1) * nsend;
}

template <class DeviceType>
void FixSpinCFIKokkos<DeviceType>::unpack_exchange_kokkos(DAT::tdual_double_2d_lr &k_buf,
                                                          DAT::tdual_int_1d &k_indices, int nrecv,
                                                          int nrecv1, int nextrarecv1,
                                                          ExecutionSpace /*space*/)
{
  if (nrecv == 0) return;
  const bool has_box = box_iso;

  k_buf.template sync<DeviceType>();
  k_indices.template sync<DeviceType>();
  k_store.template sync<DeviceType>();
  if (has_box) k_box_store.template sync<DeviceType>();

  auto d_buf = typename AT::t_double_1d_um(k_buf.template view<DeviceType>().data(),
                                           k_buf.extent(0) * k_buf.extent(1));
  auto indices = k_indices.template view<DeviceType>();
  auto st = k_store.template view<DeviceType>();
  auto bst = k_box_store.template view<DeviceType>();

  Kokkos::parallel_for(
      "spin/cfi:unpack_exchange", Kokkos::RangePolicy<DeviceType>(0, nrecv),
      KOKKOS_LAMBDA(const int i) {
        const int index = indices(i);
        if (index > -1) {
          int m = static_cast<int>(d_buf(i));
          if (i >= nrecv1) m = nextrarecv1 + static_cast<int>(d_buf(nextrarecv1 + i - nrecv1));
          for (int k = 0; k < 6; ++k) st(index, k) = d_buf(m + k);
          if (has_box) {
            for (int k = 0; k < 6; ++k) bst(index, k) = d_buf(m + 6 + k);
          }
        }
      });

  k_store.template modify<DeviceType>();
  if (has_box) k_box_store.template modify<DeviceType>();
}

template <class DeviceType>
void FixSpinCFIKokkos<DeviceType>::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  k_store.sync_device();
  Sorter.sort(LMPDeviceType(), k_store.view_device());
  k_store.modify_device();
  if (box_iso) {
    k_box_store.sync_device();
    Sorter.sort(LMPDeviceType(), k_box_store.view_device());
    k_box_store.modify_device();
  }
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::sync_host_for_sort()
{
  k_store.sync_host();
  if (box_iso) k_box_store.sync_host();
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::modified_host_for_sort()
{
  k_store.modify_host();
  if (box_iso) k_box_store.modify_host();
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::flush_device_status()
{
  Kokkos::deep_copy(h_bad_count, d_bad_count);
  Kokkos::deep_copy(h_maxdev, d_maxdev);
  if (h_maxdev() > maxnormdev) maxnormdev = h_maxdev();
  if (h_bad_count() > 0) {
    Kokkos::deep_copy(d_bad_count, 0);
    error->one(FLERR, "fix {} spin normalization/midpoint collapse on device", style);
  }
}

template <class DeviceType>
void FixSpinCFIKokkos<DeviceType>::initial_integrate_fused(int nlocal)
{
  const bool do_seed = !cache_valid;
  const bool do_moving = lattice_moving;
  const bool do_ou = do_moving && ou_enabled;
  const bool dipoles = do_moving && (atom->bec_flag != 0);
  const bool save_only = predictor_fresh;

  uint64_t sync_mask = SP_MASK | MASK_MASK;
  if (do_seed) sync_mask |= FM_MASK;
  if (thermal != 0.0 || (do_ou && ou_temp > 0.0)) sync_mask |= TAG_MASK;
  if (do_moving) sync_mask |= X_MASK | V_MASK;
  if (do_ou) sync_mask |= TYPE_MASK | RMASS_MASK;
  if (dipoles) sync_mask |= BEC_MASK | BEC_DIPOLE_MASK;

  atomKK->sync(execution_space, sync_mask);
  if (do_ou) atomKK->k_mass.template sync<DeviceType>();
  k_store.template sync<DeviceType>();

  auto sp = atomKK->k_sp.template view<DeviceType>();
  const auto fm = atomKK->k_fm.template view<DeviceType>();
  const auto mask = atomKK->k_mask.template view<DeviceType>();
  const auto tag = atomKK->k_tag.template view<DeviceType>();
  auto x = atomKK->k_x.template view<DeviceType>();
  auto v = atomKK->k_v.template view<DeviceType>();
  const auto type = atomKK->k_type.template view<DeviceType>();
  const auto mass = atomKK->k_mass.template view<DeviceType>();
  const auto rmass = atomKK->k_rmass.template view<DeviceType>();
  const auto bec = atomKK->k_bec.template view<DeviceType>();
  auto dipole = atomKK->k_bec_dipole.template view<DeviceType>();
  auto st = k_store.template view<DeviceType>();
  auto bad_v = d_bad_count;

  const bool per_atom_mass = rmass.data() != nullptr;
  const int bit = groupbit;
  const int sseed = seed;
  const int oseed = ou_seed;
  const uint64_t step = static_cast<uint64_t>(update->ntimestep);
  const double dt = update->dt;
  const double half_dt = 0.5 * dt;
  const double ou_decay = do_ou ? std::exp(-0.5 * dt / ou_damp) : 1.0;
  const double ou_var =
      do_ou ? (-std::expm1(-dt / ou_damp) * force->boltz * ou_temp * force->ftm2v) : 0.0;
  const double hbar_v = hbar, alpha_v = alpha, thermal_v = thermal, dts_v = dts;
  const double nan_val = std::numeric_limits<double>::quiet_NaN();

  // Single fused kernel for initial_integrate: seed_cache + OU_1 + half-drift + BEC + spin predict.
  // Zero blocking D2H reductions per step.
  Kokkos::parallel_for(
      "spin/cfi:initial_fused", Kokkos::RangePolicy<DeviceType>(0, nlocal),
      KOKKOS_LAMBDA(const int i) {
        if (!(mask(i) & bit)) return;

        if (do_moving) {
          double vel[3] = {static_cast<double>(v(i, 0)), static_cast<double>(v(i, 1)),
                           static_cast<double>(v(i, 2))};
          if (do_ou) {
            const double m = per_atom_mass ? static_cast<double>(rmass(i))
                                           : static_cast<double>(mass(type(i)));
            const uint64_t tid = ou_var > 0.0 ? static_cast<uint64_t>(tag(i)) : UINT64_C(0);
            SpinCFIMath::ou_velocity_half(vel, m, ou_decay, ou_var, oseed, tid, step, 1);
            for (int k = 0; k < 3; ++k) v(i, k) = vel[k];
          }
          double dx[3];
          for (int k = 0; k < 3; ++k) {
            dx[k] = half_dt * vel[k];
            x(i, k) += dx[k];
          }
          if (dipoles) {
            for (int k = 0; k < 3; ++k)
              dipole(i, k) += bec(i, 3 * k) * dx[0] + bec(i, 3 * k + 1) * dx[1] +
                  bec(i, 3 * k + 2) * dx[2];
          }
        }

        if (sp(i, 3) <= 0.0) return;

        double torque[3];
        if (do_seed) {
          for (int k = 0; k < 3; ++k) torque[k] = st(i, k) = static_cast<double>(fm(i, k));
        } else {
          for (int k = 0; k < 3; ++k) torque[k] = st(i, k);
        }

        double orig[3];
        for (int k = 0; k < 3; ++k) orig[k] = st(i, 3 + k) = static_cast<double>(sp(i, k));
        if (save_only) return;

        double noise[3] = {0.0, 0.0, 0.0};
        if (thermal_v != 0.0)
          SpinCFIMath::gaussian_stream(sseed, static_cast<uint64_t>(tag(i)), step, 0, noise);

        double mid[3];
        const bool ok = SpinCFIMath::cayley_step(orig, orig, torque, noise, hbar_v, alpha_v,
                                                 thermal_v, dts_v / static_cast<double>(sp(i, 3)),
                                                 true, mid);
        if (!ok) {
          for (int k = 0; k < 3; ++k) sp(i, k) = nan_val;
          Kokkos::atomic_inc(&bad_v());
          return;
        }
        for (int k = 0; k < 3; ++k) sp(i, k) = mid[k];
      });

  k_store.template modify<DeviceType>();
  uint64_t mod_mask = EMPTY_MASK;
  if (!save_only) mod_mask |= SP_MASK;
  if (do_moving) mod_mask |= X_MASK;
  if (do_ou) mod_mask |= V_MASK;
  if (dipoles) mod_mask |= BEC_DIPOLE_MASK;
  if (mod_mask != EMPTY_MASK) atomKK->modified(execution_space, mod_mask);
  if (do_seed) cache_valid = 1;
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::final_integrate_fused(int nlocal)
{
  const bool do_moving = lattice_moving;
  const bool do_ou = do_moving && ou_enabled;
  const bool dipoles = do_moving && (atom->bec_flag != 0);

  uint64_t sync_mask = SP_MASK | FM_MASK | MASK_MASK;
  if (thermal != 0.0 || (do_ou && ou_temp > 0.0)) sync_mask |= TAG_MASK;
  if (do_moving) sync_mask |= X_MASK | V_MASK | F_MASK | TYPE_MASK | RMASS_MASK;
  if (dipoles) sync_mask |= BEC_MASK | BEC_DIPOLE_MASK;

  atomKK->sync(execution_space, sync_mask);
  if (do_moving) atomKK->k_mass.template sync<DeviceType>();
  k_store.template sync<DeviceType>();

  auto sp = atomKK->k_sp.template view<DeviceType>();
  const auto fm = atomKK->k_fm.template view<DeviceType>();
  const auto mask = atomKK->k_mask.template view<DeviceType>();
  const auto tag = atomKK->k_tag.template view<DeviceType>();
  auto x = atomKK->k_x.template view<DeviceType>();
  auto v = atomKK->k_v.template view<DeviceType>();
  const auto f = atomKK->k_f.template view<DeviceType>();
  const auto type = atomKK->k_type.template view<DeviceType>();
  const auto mass = atomKK->k_mass.template view<DeviceType>();
  const auto rmass = atomKK->k_rmass.template view<DeviceType>();
  const auto bec = atomKK->k_bec.template view<DeviceType>();
  auto dipole = atomKK->k_bec_dipole.template view<DeviceType>();
  auto st = k_store.template view<DeviceType>();
  auto bad_v = d_bad_count;
  auto maxdev_v = d_maxdev;

  const bool per_atom_mass = rmass.data() != nullptr;
  const int bit = groupbit;
  const int sseed = seed;
  const int oseed = ou_seed;
  const uint64_t step = static_cast<uint64_t>(update->ntimestep);
  const double dt = update->dt;
  const double half_dt = 0.5 * dt;
  const double dt_ftm2v = dt * force->ftm2v;
  const double ou_decay = do_ou ? std::exp(-0.5 * dt / ou_damp) : 1.0;
  const double ou_var =
      do_ou ? (-std::expm1(-dt / ou_damp) * force->boltz * ou_temp * force->ftm2v) : 0.0;
  const double hbar_v = hbar, alpha_v = alpha, thermal_v = thermal, dts_v = dts;
  const double nan_val = std::numeric_limits<double>::quiet_NaN();

  // Single fused kernel for final_integrate: spin correct + velocity kick + half-drift + BEC + OU_2.
  // Zero blocking D2H reductions per step.
  Kokkos::parallel_for(
      "spin/cfi:final_fused", Kokkos::RangePolicy<DeviceType>(0, nlocal),
      KOKKOS_LAMBDA(const int i) {
        if (!(mask(i) & bit)) return;

        if (sp(i, 3) > 0.0) {
          double orig[3], mid[3], torque[3];
          for (int k = 0; k < 3; ++k) {
            st(i, k) = torque[k] = static_cast<double>(fm(i, k));
            orig[k] = st(i, 3 + k);
            mid[k] = static_cast<double>(sp(i, k));
          }
          double noise[3] = {0.0, 0.0, 0.0};
          if (thermal_v != 0.0)
            SpinCFIMath::gaussian_stream(sseed, static_cast<uint64_t>(tag(i)), step, 0, noise);

          double next[3];
          double dev = 0.0;
          const bool ok = SpinCFIMath::cayley_step(orig, mid, torque, noise, hbar_v, alpha_v,
                                                   thermal_v, dts_v / static_cast<double>(sp(i, 3)),
                                                   false, next, &dev);
          if (!ok) {
            for (int k = 0; k < 3; ++k) sp(i, k) = nan_val;
            Kokkos::atomic_inc(&bad_v());
          } else {
            for (int k = 0; k < 3; ++k) sp(i, k) = next[k];
            if (dev > 1.0e-14) Kokkos::atomic_max(&maxdev_v(), dev);
          }
        }

        if (do_moving) {
          const double m = per_atom_mass ? static_cast<double>(rmass(i))
                                         : static_cast<double>(mass(type(i)));
          const double kick = dt_ftm2v / m;
          double vel[3], dx[3];
          for (int k = 0; k < 3; ++k) {
            vel[k] = static_cast<double>(v(i, k)) + kick * static_cast<double>(f(i, k));
            dx[k] = half_dt * vel[k];
            x(i, k) += dx[k];
          }
          if (dipoles) {
            for (int k = 0; k < 3; ++k)
              dipole(i, k) += bec(i, 3 * k) * dx[0] + bec(i, 3 * k + 1) * dx[1] +
                  bec(i, 3 * k + 2) * dx[2];
          }
          if (do_ou) {
            const uint64_t tid = ou_var > 0.0 ? static_cast<uint64_t>(tag(i)) : UINT64_C(0);
            SpinCFIMath::ou_velocity_half(vel, m, ou_decay, ou_var, oseed, tid, step, 2);
          }
          for (int k = 0; k < 3; ++k) v(i, k) = vel[k];
        }
      });

  k_store.template modify<DeviceType>();
  uint64_t mod_mask = SP_MASK;
  if (do_moving) mod_mask |= X_MASK | V_MASK | (dipoles ? BEC_DIPOLE_MASK : EMPTY_MASK);
  atomKK->modified(execution_space, mod_mask);
}

template <class DeviceType>
void FixSpinCFIKokkos<DeviceType>::predict_device(int nlocal, bool use_live_fm, bool do_seed)
{
  uint64_t sync_mask = SP_MASK | MASK_MASK;
  if (thermal != 0.0) sync_mask |= TAG_MASK;
  if (use_live_fm || do_seed) sync_mask |= FM_MASK;
  atomKK->sync(execution_space, sync_mask);
  k_store.template sync<DeviceType>();

  auto sp = atomKK->k_sp.template view<DeviceType>();
  const auto fm = atomKK->k_fm.template view<DeviceType>();
  const auto mask = atomKK->k_mask.template view<DeviceType>();
  const auto tag = atomKK->k_tag.template view<DeviceType>();
  auto st = k_store.template view<DeviceType>();
  auto bad_v = d_bad_count;

  const int bit = groupbit;
  const int sseed = seed;
  const uint64_t step = static_cast<uint64_t>(update->ntimestep);
  const double hbar_v = hbar, alpha_v = alpha, thermal_v = thermal, dts_v = dts;
  const bool save_only = predictor_fresh && !use_live_fm;
  const double nan_val = std::numeric_limits<double>::quiet_NaN();

  Kokkos::parallel_for(
      "spin/cfi:predict", Kokkos::RangePolicy<DeviceType>(0, nlocal),
      KOKKOS_LAMBDA(const int i) {
        if (!(mask(i) & bit) || sp(i, 3) <= 0.0) return;

        double torque[3];
        if (do_seed && !use_live_fm) {
          for (int k = 0; k < 3; ++k) torque[k] = st(i, k) = static_cast<double>(fm(i, k));
        } else {
          for (int k = 0; k < 3; ++k)
            torque[k] = use_live_fm ? static_cast<double>(fm(i, k)) : st(i, k);
        }

        double orig[3];
        if (!use_live_fm) {
          for (int k = 0; k < 3; ++k) orig[k] = st(i, 3 + k) = static_cast<double>(sp(i, k));
        } else {
          for (int k = 0; k < 3; ++k) orig[k] = st(i, 3 + k);
        }
        if (save_only) return;

        double noise[3] = {0.0, 0.0, 0.0};
        if (thermal_v != 0.0)
          SpinCFIMath::gaussian_stream(sseed, static_cast<uint64_t>(tag(i)), step, 0, noise);

        double mid[3];
        const bool ok = SpinCFIMath::cayley_step(orig, orig, torque, noise, hbar_v, alpha_v,
                                                 thermal_v, dts_v / static_cast<double>(sp(i, 3)),
                                                 true, mid);
        if (!ok) {
          for (int k = 0; k < 3; ++k) sp(i, k) = nan_val;
          Kokkos::atomic_inc(&bad_v());
          return;
        }
        for (int k = 0; k < 3; ++k) sp(i, k) = mid[k];
      });

  if (!use_live_fm) k_store.template modify<DeviceType>();
  if (!save_only) atomKK->modified(execution_space, SP_MASK);
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::correct_device(int nlocal)
{
  const auto tag_field = thermal != 0.0 ? TAG_MASK : EMPTY_MASK;
  atomKK->sync(execution_space, SP_MASK | FM_MASK | MASK_MASK | tag_field);
  k_store.template sync<DeviceType>();

  auto sp = atomKK->k_sp.template view<DeviceType>();
  const auto fm = atomKK->k_fm.template view<DeviceType>();
  const auto mask = atomKK->k_mask.template view<DeviceType>();
  const auto tag = atomKK->k_tag.template view<DeviceType>();
  auto st = k_store.template view<DeviceType>();
  auto bad_v = d_bad_count;
  auto maxdev_v = d_maxdev;

  const int bit = groupbit;
  const int sseed = seed;
  const uint64_t step = static_cast<uint64_t>(update->ntimestep);
  const double hbar_v = hbar, alpha_v = alpha, thermal_v = thermal, dts_v = dts;
  const double nan_val = std::numeric_limits<double>::quiet_NaN();

  Kokkos::parallel_for(
      "spin/cfi:correct", Kokkos::RangePolicy<DeviceType>(0, nlocal),
      KOKKOS_LAMBDA(const int i) {
        if (!(mask(i) & bit) || sp(i, 3) <= 0.0) return;
        double orig[3], mid[3], torque[3];
        for (int k = 0; k < 3; ++k) {
          st(i, k) = torque[k] = static_cast<double>(fm(i, k));
          orig[k] = st(i, 3 + k);
          mid[k] = static_cast<double>(sp(i, k));
        }
        double noise[3] = {0.0, 0.0, 0.0};
        if (thermal_v != 0.0)
          SpinCFIMath::gaussian_stream(sseed, static_cast<uint64_t>(tag(i)), step, 0, noise);

        double next[3];
        double dev = 0.0;
        const bool ok = SpinCFIMath::cayley_step(orig, mid, torque, noise, hbar_v, alpha_v,
                                                 thermal_v, dts_v / static_cast<double>(sp(i, 3)),
                                                 false, next, &dev);
        if (!ok) {
          for (int k = 0; k < 3; ++k) sp(i, k) = nan_val;
          Kokkos::atomic_inc(&bad_v());
          return;
        }
        for (int k = 0; k < 3; ++k) sp(i, k) = next[k];
        if (dev > 1.0e-14) Kokkos::atomic_max(&maxdev_v(), dev);
      });

  k_store.template modify<DeviceType>();
  atomKK->modified(execution_space, SP_MASK);
}

template <class DeviceType>
void FixSpinCFIKokkos<DeviceType>::box_bath_half_device(int nlocal, int half)
{
  const double dt = update->dt;
  const uint64_t step = static_cast<uint64_t>(update->ntimestep);
  const uint64_t stream = static_cast<uint64_t>(half + 1);
  const double L = std::cbrt(box_V);

  if (ou_enabled) {
    const auto tag_field = ou_temp > 0.0 ? TAG_MASK : EMPTY_MASK;
    atomKK->sync(execution_space, V_MASK | TYPE_MASK | RMASS_MASK | tag_field);
    atomKK->k_mass.template sync<DeviceType>();
    k_box_store.template sync<DeviceType>();

    auto v = atomKK->k_v.template view<DeviceType>();
    const auto type = atomKK->k_type.template view<DeviceType>();
    const auto mass = atomKK->k_mass.template view<DeviceType>();
    const auto rmass = atomKK->k_rmass.template view<DeviceType>();
    const auto tag = atomKK->k_tag.template view<DeviceType>();
    auto bst = k_box_store.template view<DeviceType>();

    const bool per_atom_mass = rmass.data() != nullptr;
    const double mvv2e = force->mvv2e;
    const double decay = std::exp(-0.5 * dt / ou_damp);
    const double variance = -std::expm1(-dt / ou_damp) * force->boltz * ou_temp;
    const int oseed = ou_seed;

    Kokkos::parallel_for(
        "spin/cfi:box_bath_half", Kokkos::RangePolicy<DeviceType>(0, nlocal),
        KOKKOS_LAMBDA(const int i) {
          const double m = (per_atom_mass ? static_cast<double>(rmass(i))
                                          : static_cast<double>(mass(type(i)))) *
              mvv2e;
          double noise[3] = {0.0, 0.0, 0.0};
          if (variance > 0.0)
            SpinCFIMath::gaussian_stream(oseed, static_cast<uint64_t>(tag(i)), step, stream, noise);
          const double sigma = variance > 0.0 ? ::sqrt(variance * m) * L : 0.0;
          const double inv_mL = 1.0 / (m * L);
          for (int k = 0; k < 3; ++k) {
            const double p_new = decay * bst(i, 3 + k) + sigma * noise[k];
            bst(i, 3 + k) = p_new;
            v(i, k) = p_new * inv_mL;
          }
        });
    k_box_store.template modify<DeviceType>();
    atomKK->modified(execution_space, V_MASK);
  }

  if (box_ou_enabled) {
    const double decay = std::exp(-0.5 * dt / box_ou_damp);
    const double variance = -std::expm1(-dt / box_ou_damp) * box_W * force->boltz * box_ou_temp;
    const double noise =
        variance > 0.0 ? SpinCFIMath::gaussian_scalar(box_ou_seed, step, stream) : 0.0;
    const double sigma = variance > 0.0 ? std::sqrt(variance) : 0.0;
    box_pV = decay * box_pV + sigma * noise;
  }
}

template <class DeviceType>
void FixSpinCFIKokkos<DeviceType>::box_T_device(int nlocal, double a)
{
  atomKK->sync(execution_space, TYPE_MASK | RMASS_MASK);
  atomKK->k_mass.template sync<DeviceType>();
  k_box_store.template sync<DeviceType>();

  const auto type = atomKK->k_type.template view<DeviceType>();
  const auto mass = atomKK->k_mass.template view<DeviceType>();
  const auto rmass = atomKK->k_rmass.template view<DeviceType>();
  auto bst = k_box_store.template view<DeviceType>();
  const bool per_atom_mass = rmass.data() != nullptr;
  const double mvv2e = force->mvv2e;
  const double L = std::cbrt(box_V);
  const double L2 = L * L;

  double local_twiceK = 0.0;
  Kokkos::parallel_reduce(
      "spin/cfi:box_T", Kokkos::RangePolicy<DeviceType>(0, nlocal),
      KOKKOS_LAMBDA(const int i, double &sum_k) {
        const double m = (per_atom_mass ? static_cast<double>(rmass(i))
                                        : static_cast<double>(mass(type(i)))) *
            mvv2e;
        const double inv_denom = 1.0 / (m * L2);
        for (int k = 0; k < 3; ++k) {
          const double p = bst(i, 3 + k);
          sum_k += p * p * inv_denom;
          bst(i, k) += a * p * inv_denom;
        }
      },
      local_twiceK);

  k_box_store.template modify<DeviceType>();

  double global_twiceK = 0.0;
  MPI_Allreduce(&local_twiceK, &global_twiceK, 1, MPI_DOUBLE, MPI_SUM, world);
  box_pV += a * global_twiceK / (3.0 * box_V);
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::box_sync_device(int nlocal)
{
  const double L = std::cbrt(box_V);
  Few<double, 3> old_prd;
  old_prd[0] = domain->xprd;
  old_prd[1] = domain->yprd;
  old_prd[2] = domain->zprd;

  for (int k = 0; k < 3; ++k) domain->boxhi[k] = box_origin[k] + box_shape[k] * L;
  domain->set_global_box();
  domain->set_local_box();

  Few<double, 3> new_prd, origin;
  for (int k = 0; k < 3; ++k) {
    new_prd[k] = domain->prd[k];
    origin[k] = box_origin[k];
  }

  const bool dipoles = atom->bec_flag != 0;
  uint64_t sync_mask = X_MASK | V_MASK | IMAGE_MASK | TYPE_MASK | RMASS_MASK;
  if (dipoles) sync_mask |= BEC_MASK | BEC_DIPOLE_MASK;
  atomKK->sync(execution_space, sync_mask);
  atomKK->k_mass.template sync<DeviceType>();
  k_box_store.template sync<DeviceType>();

  auto x = atomKK->k_x.template view<DeviceType>();
  auto v = atomKK->k_v.template view<DeviceType>();
  const auto image = atomKK->k_image.template view<DeviceType>();
  const auto type = atomKK->k_type.template view<DeviceType>();
  const auto mass = atomKK->k_mass.template view<DeviceType>();
  const auto rmass = atomKK->k_rmass.template view<DeviceType>();
  const auto bec = atomKK->k_bec.template view<DeviceType>();
  auto dipole = atomKK->k_bec_dipole.template view<DeviceType>();
  const auto bst = k_box_store.template view<DeviceType>();
  const bool per_atom_mass = rmass.data() != nullptr;
  const double mvv2e = force->mvv2e;

  // Preserve existing image flags so atoms crossing box boundaries never
  // teleport between reneighbor steps; Verlet domain->pbc() wraps on reneigh.
  Kokkos::parallel_for(
      "spin/cfi:box_sync", Kokkos::RangePolicy<DeviceType>(0, nlocal),
      KOKKOS_LAMBDA(const int i) {
        const imageint img = image(i);
        const int nbox[3] = {static_cast<int>(img & IMGMASK) - IMGMAX,
                             static_cast<int>((img >> IMGBITS) & IMGMASK) - IMGMAX,
                             static_cast<int>(img >> IMG2BITS) - IMGMAX};
        double xu_old[3] = {0.0, 0.0, 0.0};
        if (dipoles) {
          for (int k = 0; k < 3; ++k)
            xu_old[k] = static_cast<double>(x(i, k)) + nbox[k] * old_prd[k];
        }

        const double m = (per_atom_mass ? static_cast<double>(rmass(i))
                                        : static_cast<double>(mass(type(i)))) *
            mvv2e;
        const double inv_mL = 1.0 / (m * L);
        double xu_new[3];
        for (int k = 0; k < 3; ++k) {
          xu_new[k] = origin[k] + L * bst(i, k);
          x(i, k) = xu_new[k] - nbox[k] * new_prd[k];
          v(i, k) = bst(i, 3 + k) * inv_mL;
        }
        if (dipoles) {
          const double dx0 = xu_new[0] - xu_old[0];
          const double dx1 = xu_new[1] - xu_old[1];
          const double dx2 = xu_new[2] - xu_old[2];
          for (int k = 0; k < 3; ++k)
            dipole(i, k) += bec(i, 3 * k) * dx0 + bec(i, 3 * k + 1) * dx1 +
                bec(i, 3 * k + 2) * dx2;
        }
      });

  atomKK->modified(execution_space,
                   X_MASK | V_MASK | (dipoles ? BEC_DIPOLE_MASK : EMPTY_MASK));
}

template <class DeviceType>
void FixSpinCFIKokkos<DeviceType>::box_D_device(int nlocal, double a)
{
  box_T_device(nlocal, 0.5 * a);
  box_V += a * box_pV / box_W;
  if (!std::isfinite(box_V) || box_V <= 0.0)
    error->all(FLERR, "fix {} box iso encountered nonpositive volume", style);
  box_T_device(nlocal, 0.5 * a);
  box_sync_device(nlocal);
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::box_kick_device(int nlocal)
{
  atomKK->sync(execution_space, F_MASK);
  k_box_store.template sync<DeviceType>();
  const auto f = atomKK->k_f.template view<DeviceType>();
  auto bst = k_box_store.template view<DeviceType>();
  const double dtL = update->dt * std::cbrt(box_V);

  Kokkos::parallel_for(
      "spin/cfi:box_kick", Kokkos::RangePolicy<DeviceType>(0, nlocal),
      KOKKOS_LAMBDA(const int i) {
        for (int k = 0; k < 3; ++k) bst(i, 3 + k) += dtL * static_cast<double>(f(i, k));
      });
  k_box_store.template modify<DeviceType>();
}

template <class DeviceType>
void FixSpinCFIKokkos<DeviceType>::evaluate_pair_fresh_device(bool check_reneighbor)
{
  DeviceType().fence();
  if (check_reneighbor && (lattice_moving || box_iso) && neighbor->check_distance()) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    neighbor->build(1);
  } else {
    comm->forward_comm();
  }

  const int nall = atom->nlocal + (force->newton ? atom->nghost : 0);
  atomKK->sync(execution_space, F_MASK | FM_MASK);
  auto f = atomKK->k_f.template view<DeviceType>();
  auto fm = atomKK->k_fm.template view<DeviceType>();
  Kokkos::parallel_for(
      "spin/cfi:force_clear", Kokkos::RangePolicy<DeviceType>(0, nall), KOKKOS_LAMBDA(const int i) {
        for (int k = 0; k < 3; ++k) {
          f(i, k) = 0.0;
          fm(i, k) = 0.0;
        }
      });
  atomKK->modified(execution_space, F_MASK | FM_MASK);
  if (force->pair) force->pair->compute(1, box_iso ? 1 : 0);
  if (!spingnn2 && force->newton) comm->reverse_comm();
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
  const int nlocal = integrated_atoms();
  if (!box_iso) {
    initial_integrate_fused(nlocal);
    return;
  }

  const bool do_seed = !cache_valid;
  if (ou_enabled || box_ou_enabled) box_bath_half_device(nlocal, 0);
  box_D_device(nlocal, 0.5 * update->dt);
  predict_device(nlocal, false, do_seed);
  if (do_seed) cache_valid = 1;
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::final_integrate()
{
  const int nlocal = integrated_atoms();
  if (predictor_fresh) {
    predict_device(nlocal, true, false);
    flush_device_status();
    evaluate_pair_fresh_device(false);
  }

  if (!box_iso) {
    final_integrate_fused(nlocal);
    ++nsteps;
    return;
  }

  correct_device(nlocal);

  if (!force->pair || update->vflag_global != update->ntimestep)
    error->all(FLERR, "fix {} box iso requires pair global virial", style);
  double local_virial[6];
  for (int k = 0; k < 6; ++k) local_virial[k] = force->pair->virial[k];
  MPI_Allreduce(local_virial, box_virial, 6, MPI_DOUBLE, MPI_SUM, world);

  box_kick_device(nlocal);
  box_pV += update->dt * ((box_virial[0] + box_virial[1] + box_virial[2]) / (3.0 * box_V) - box_P);
  box_D_device(nlocal, 0.5 * update->dt);
  if (ou_enabled || box_ou_enabled) box_bath_half_device(nlocal, 1);
  if (pressure) pressure->addstep(update->ntimestep + 1);
  ++nsteps;
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::end_of_step()
{
  flush_device_status();
  evaluate_pair_fresh_device(true);
  ++naccept;
}

template <class DeviceType> void FixSpinCFIKokkos<DeviceType>::post_run()
{
  flush_device_status();
}

template <class DeviceType> double FixSpinCFIKokkos<DeviceType>::compute_vector(int n)
{
  flush_device_status();
  return FixSpinCFI::compute_vector(n);
}

namespace LAMMPS_NS {
template class FixSpinCFIKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixSpinCFIKokkos<LMPHostType>;
#endif
}  // namespace LAMMPS_NS
