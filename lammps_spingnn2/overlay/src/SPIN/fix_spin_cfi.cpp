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

/* Cached-Field Integrator (CFI) for coupled spin-lattice dynamics.
   Uses one core potential evaluation per timestep at the joint centered
   midpoint (x_{n+1/2}, s_{n+1/2}), reusing the previous midpoint spin
   torque as the predictor field and keeping an optional read-only
   accepted-state refresh for output/thermometry.
   Supports optional isotropic canonical-volume NPH/NPT barostatting
   (box iso P_bar W [box_ou T_box damp seed_box]) without patching Verlet. */

#include "fix_spin_cfi.h"

#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "dream_bec.h"
#include "error.h"
#include "fix_precession_spin.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "pair_spin.h"
#include "pair_spingnn2.h"
#include "spin_cfi_math.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

FixSpinCFI::FixSpinCFI(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  // Syntax:
  // fix ID group spin/cfi T alpha seed lattice frozen|moving
  //   [accept N] [predictor lagged|fresh] [ou T_lat damp seed_lat]
  //   [box iso P_bar W] [box_ou T_box damp seed_box]
  if (narg < 8 || std::strcmp(arg[6], "lattice") != 0)
    error->all(FLERR,
               "Expected fix {} T alpha seed lattice frozen|moving "
               "[accept N] [predictor lagged|fresh] [ou T damp seed] "
               "[box iso P W] [box_ou T damp seed]",
               style);

  temperature = utils::numeric(FLERR, arg[3], false, lmp);
  alpha = utils::numeric(FLERR, arg[4], false, lmp);
  seed = utils::inumeric(FLERR, arg[5], false, lmp);

  if (std::strcmp(arg[7], "moving") == 0 || std::strcmp(arg[7], "yes") == 0)
    lattice_moving = true;
  else if (std::strcmp(arg[7], "frozen") == 0 || std::strcmp(arg[7], "no") == 0)
    lattice_moving = false;
  else
    error->all(FLERR, "Unknown fix {} lattice mode {}", style, arg[7]);
  lattice_moving_flag = lattice_moving ? 1 : 0;

  int iarg = 8;
  while (iarg < narg) {
    if (std::strcmp(arg[iarg], "accept") == 0 && iarg + 1 < narg) {
      accept = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (std::strcmp(arg[iarg], "predictor") == 0 && iarg + 1 < narg) {
      if (std::strcmp(arg[iarg + 1], "lagged") == 0)
        predictor_fresh = false;
      else if (std::strcmp(arg[iarg + 1], "fresh") == 0)
        predictor_fresh = true;
      else
        error->all(FLERR, "fix {} predictor must be lagged or fresh", style);
      iarg += 2;
    } else if (std::strcmp(arg[iarg], "ou") == 0 && iarg + 3 < narg) {
      if (ou_enabled) error->all(FLERR, "Duplicate ou keyword in fix {}", style);
      ou_temp = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      ou_damp = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      ou_seed = utils::inumeric(FLERR, arg[iarg + 3], false, lmp);
      if (!lattice_moving || !std::isfinite(ou_temp + ou_damp) || ou_temp < 0.0 ||
          ou_damp <= 0.0 || ou_seed <= 0 || ou_seed == seed)
        error->all(FLERR, "Invalid fix {} atomic OU bath parameters", style);
      ou_enabled = true;
      iarg += 4;
    } else if (std::strcmp(arg[iarg], "box") == 0 && iarg + 3 < narg) {
      if (box_iso || std::strcmp(arg[iarg + 1], "iso") != 0)
        error->all(FLERR, "Expected fix {} box iso P_bar W", style);
      box_Pbar = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      box_W = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      if (!std::isfinite(box_Pbar) || !std::isfinite(box_W) || box_W <= 0.0)
        error->all(FLERR, "Invalid fix {} box iso parameters", style);
      box_iso = true;
      box_iso_flag = 1;
      iarg += 4;
    } else if (std::strcmp(arg[iarg], "box_ou") == 0 && iarg + 3 < narg) {
      if (box_ou_enabled) error->all(FLERR, "Duplicate box_ou keyword in fix {}", style);
      box_ou_temp = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      box_ou_damp = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      box_ou_seed = utils::inumeric(FLERR, arg[iarg + 3], false, lmp);
      if (!std::isfinite(box_ou_temp + box_ou_damp) || box_ou_temp < 0.0 ||
          box_ou_damp <= 0.0 || box_ou_seed <= 0)
        error->all(FLERR, "Invalid fix {} box_ou parameters", style);
      box_ou_enabled = true;
      iarg += 4;
    } else {
      error->all(FLERR, "Illegal keyword {} in fix {}", arg[iarg], style);
    }
  }

  if (box_iso && !lattice_moving)
    error->all(FLERR, "fix {} box iso requires lattice moving", style);
  if (box_ou_enabled &&
      (!box_iso || box_ou_seed == seed || (ou_enabled && box_ou_seed == ou_seed)))
    error->all(FLERR, "fix {} box_ou requires box iso and an independent random seed", style);

  if (!std::isfinite(temperature) || temperature < 0.0 || !std::isfinite(alpha) || alpha < 0.0)
    error->all(FLERR, "fix {} requires nonnegative spin temperature and damping", style);
  if (seed <= 0) error->all(FLERR, "fix {} requires a positive random seed", style);
  if (accept < 0) error->all(FLERR, "fix {} accept interval must be nonnegative", style);
  if (!atom->sp_flag) error->all(FLERR, "fix {} requires atom_style spin", style);
  if (std::strcmp(update->unit_style, "metal") != 0)
    error->all(FLERR, "fix {} requires metal units", style);

  time_integrate = 1;
  nevery = accept > 0 ? accept : 1;
  vector_flag = 1;
  size_vector = box_iso ? 14 : 3;
  global_freq = 1;
  extvector = 0;
  maxexchange = box_iso ? 13 : 7;

  if (box_iso) {
    box_change = BOX_CHANGE_SIZE;
    // Own a virial-only compute solely to schedule vflag_global via Integrate::ev_set.
    // We never invoke pressure->compute_scalar() so Kokkos stays device-resident.
    id_press = std::string(id) + "_press";
    modify->add_compute(fmt::format("{} all pressure NULL virial", id_press));
    pcomputeflag = 1;
  }

  FixSpinCFI::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  for (int i = 0; i < atom->nmax; ++i) FixSpinCFI::set_arrays(i);
}

FixSpinCFI::~FixSpinCFI()
{
  if (copymode) return;
  if (pcomputeflag && modify->get_compute_by_id(id_press)) modify->delete_compute(id_press);
  atom->delete_callback(id, Atom::GROW);
  memory->destroy(store);
  memory->destroy(box_store);
}

int FixSpinCFI::setmask()
{
  int mask = INITIAL_INTEGRATE | FINAL_INTEGRATE | POST_RUN;
  if (accept > 0) mask |= END_OF_STEP;
  return mask;
}

void FixSpinCFI::grow_arrays(int nmax)
{
  memory->grow(store, nmax, 6, "spin/cfi:store");
  if (box_iso) memory->grow(box_store, nmax, 6, "spin/cfi:box_store");
}

void FixSpinCFI::copy_arrays(int i, int j, int /*delflag*/)
{
  for (int k = 0; k < 6; ++k) store[j][k] = store[i][k];
  if (box_iso) {
    for (int k = 0; k < 6; ++k) box_store[j][k] = box_store[i][k];
  }
}

void FixSpinCFI::set_arrays(int i)
{
  for (int k = 0; k < 6; ++k) store[i][k] = 0.0;
  if (box_iso) {
    for (int k = 0; k < 6; ++k) box_store[i][k] = 0.0;
  }
}

int FixSpinCFI::pack_exchange(int i, double *buf)
{
  int m = 0;
  for (int k = 0; k < 6; ++k) buf[m++] = store[i][k];
  if (box_iso) {
    for (int k = 0; k < 6; ++k) buf[m++] = box_store[i][k];
  }
  return m;
}

int FixSpinCFI::unpack_exchange(int nlocal, double *buf)
{
  int m = 0;
  for (int k = 0; k < 6; ++k) store[nlocal][k] = buf[m++];
  if (box_iso) {
    for (int k = 0; k < 6; ++k) box_store[nlocal][k] = buf[m++];
  }
  return m;
}

double FixSpinCFI::memory_usage()
{
  const double cols = box_iso ? 12.0 : 6.0;
  return static_cast<double>(atom->nmax) * cols * sizeof(double);
}

int FixSpinCFI::modify_param(int narg, char **arg)
{
  if (std::strcmp(arg[0], "box_momentum") == 0) {
    if (narg < 2 || !box_iso || box_initialized)
      error->all(FLERR, "fix {} box_momentum must be set before the first run with box iso",
                 style);
    box_pV = utils::numeric(FLERR, arg[1], false, lmp);
    if (!std::isfinite(box_pV)) error->all(FLERR, "fix {} box_momentum must be finite", style);
    return 2;
  }
  return 0;
}

void FixSpinCFI::init()
{
  if (!utils::strmatch(update->integrate_style, "^verlet"))
    error->all(FLERR, "fix {} requires the Verlet run style", style);

  if (box_iso) {
    if (igroup != 0) error->all(FLERR, "fix {} box iso requires group all", style);
    if (domain->triclinic || domain->dimension != 3 || !domain->xperiodic || !domain->yperiodic ||
        !domain->zperiodic)
      error->all(FLERR, "fix {} box iso requires an orthogonal periodic 3D simulation box", style);
    if (!force->pair || !force->pair->compute_flag || force->pair->tail_flag)
      error->all(FLERR, "fix {} box iso requires an active pair style without tail corrections",
                 style);
    pressure = modify->get_compute_by_id(id_press);
    if (!pressure || !pressure->pressflag)
      error->all(FLERR, "fix {} internal virial pressure compute {} is missing", style, id_press);
  }

  spin_pairs.clear();
  precession.clear();
  spingnn2 = dynamic_cast<PairSpinGNN2 *>(force->pair);
  if (!spingnn2 && force->pair_match("^spingnn2", 0, 1))
    error->all(FLERR, "fix {} requires spingnn2 as the active pair style", style);

  if (spingnn2) {
    if (static_cast<bool>(kokkosable) != static_cast<bool>(spingnn2->kokkosable) ||
        (kokkosable && execution_space != spingnn2->execution_space))
      error->all(FLERR, "fix {} and pair_style spingnn2 require matching execution spaces", style);
  } else {
    if (kokkosable && force->pair_match("^spin/", 0, 1))
      error->all(FLERR, "fix {} requires pair_style spingnn2/kk for device spin torques", style);
    if (!kokkosable) {
      if (force->pair_match("^hybrid", 0)) {
        for (int i = 1; Pair *pair = force->pair_match("^spin/", 0, i); ++i) {
          auto *spin = dynamic_cast<PairSpin *>(pair);
          if (!spin) error->all(FLERR, "Invalid spin pair sub-style for fix {}", style);
          spin_pairs.push_back(spin);
        }
      } else if (auto *spin = dynamic_cast<PairSpin *>(force->pair)) {
        spin_pairs.push_back(spin);
      }
    }
  }

  for (auto *other : modify->get_fix_list()) {
    if (other == this) continue;
    if (box_iso &&
        (other->box_change || utils::strmatch(other->style, "^npt") ||
         utils::strmatch(other->style, "^nph") ||
         utils::strmatch(other->style, "^press/berendsen") ||
         utils::strmatch(other->style, "^box/relax") || utils::strmatch(other->style, "^deform")))
      error->all(FLERR, "fix {} box iso conflicts with fix {}", style, other->style);
    if (other->time_integrate)
      error->all(FLERR, "fix {} cannot overlap with another time integrator ({})", style,
                 other->style);
    if (auto *prec = dynamic_cast<FixPrecessionSpin *>(other)) {
      if (prec->stt_flag)
        error->all(FLERR, "fix {} requires conservative precession/spin (STT unsupported)", style);
      precession.push_back(prec);
      continue;
    }
    if (utils::strmatch(other->style, "^langevin/spin") ||
        utils::strmatch(other->style, "^setforce/spin"))
      error->all(FLERR, "fix {} does not support {}", style, other->style);
    const bool overlap = lattice_moving && ((other->groupbit & groupbit) != 0);
    if (overlap &&
        (utils::strmatch(other->style, "^langevin") || utils::strmatch(other->style, "^temp/") ||
         utils::strmatch(other->style, "^gle") || utils::strmatch(other->style, "^gld") ||
         utils::strmatch(other->style, "^brownian")))
      error->all(FLERR, "fix {} moving mode forbids external thermostat {}; use built-in ou", style,
                 other->style);
  }

  if (kokkosable && !precession.empty())
    error->all(FLERR, "fix {} requires device spin torques from spingnn2/kk", style);

  if (spingnn2 && !lattice_moving && igroup == 0) spingnn2->set_sib_lattice(this, true);

  hbar = force->hplanck / MathConst::MY_2PI;
  thermal = std::sqrt(2.0 * alpha * force->boltz * temperature / (1.0 + alpha * alpha));
  const bool need_tags = (thermal != 0.0) || (ou_enabled && ou_temp > 0.0);
  if (need_tags && !atom->tag_enable)
    error->all(FLERR, "Stochastic fix {} requires atom IDs (atom_modify id yes)", style);

  reset_dt();
  cache_valid = 0;
  if (box_iso) box_init();
}

void FixSpinCFI::setup(int /*vflag*/)
{
  if (box_iso && pressure) pressure->addstep(update->ntimestep + 1);
}

void FixSpinCFI::reset_dt()
{
  dts = 2.0 * update->dt / ((1.0 + alpha * alpha) * hbar);
}

int FixSpinCFI::integrated_atoms() const
{
  return igroup == atom->firstgroup ? atom->nfirst : atom->nlocal;
}

void FixSpinCFI::refresh_analytic_torques(int nlocal)
{
  for (int i = 0; i < nlocal; ++i) {
    if (!(atom->mask[i] & groupbit)) continue;
    auto *spin = atom->sp[i];
    auto *torque = atom->fm[i];
    if (!spingnn2) {
      torque[0] = torque[1] = torque[2] = 0.0;
      for (auto *pair : spin_pairs) pair->compute_single_pair(i, torque);
    }
    for (auto *fix : precession) fix->compute_single_precession(i, spin, torque);
  }
}

void FixSpinCFI::evaluate_pair_fresh(bool check_reneighbor)
{
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

  size_t nbytes = sizeof(double) * static_cast<size_t>(atom->nlocal);
  if (force->newton) nbytes += sizeof(double) * static_cast<size_t>(atom->nghost);
  if (nbytes > 0) {
    std::memset(&atom->f[0][0], 0, 3 * nbytes);
    atom->avec->force_clear(0, nbytes);
  }
  if (force->pair) force->pair->compute(1, box_iso ? 1 : 0);
  if (!spingnn2 && force->newton) comm->reverse_comm();
  for (auto *fix : precession) fix->post_force(0);
}

void FixSpinCFI::box_init()
{
  box_P = box_Pbar / force->nktv2p;
  if (!box_initialized) {
    box_V = domain->xprd * domain->yprd * domain->zprd;
    const double L = std::cbrt(box_V);
    for (int k = 0; k < 3; ++k) {
      box_origin[k] = domain->boxlo[k];
      box_shape[k] = domain->prd[k] / L;
    }
    const int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; ++i) {
      double xu[3];
      domain->unmap(atom->x[i], atom->image[i], xu);
      const double m = (atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]]) * force->mvv2e;
      if (!std::isfinite(m) || m <= 0.0) error->one(FLERR, "fix {} box iso requires positive mass", style);
      for (int k = 0; k < 3; ++k) {
        box_store[i][k] = (xu[k] - box_origin[k]) / L;
        box_store[i][3 + k] = m * L * atom->v[i][k];
      }
    }
    box_initialized = 1;
  }
}

void FixSpinCFI::box_bath_half(int half)
{
  const double dt = update->dt;
  const uint64_t step = static_cast<uint64_t>(update->ntimestep);
  const uint64_t stream = static_cast<uint64_t>(half + 1);
  const double L = std::cbrt(box_V);

  if (ou_enabled) {
    const double decay = std::exp(-0.5 * dt / ou_damp);
    const double variance = -std::expm1(-dt / ou_damp) * force->boltz * ou_temp;
    const int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; ++i) {
      const double m = (atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]]) * force->mvv2e;
      double noise[3] = {0.0, 0.0, 0.0};
      if (variance > 0.0)
        SpinCFIMath::gaussian_stream(ou_seed, static_cast<uint64_t>(atom->tag[i]), step, stream,
                                     noise);
      const double sigma = variance > 0.0 ? std::sqrt(variance * m) * L : 0.0;
      const double inv_mL = 1.0 / (m * L);
      for (int k = 0; k < 3; ++k) {
        box_store[i][3 + k] = decay * box_store[i][3 + k] + sigma * noise[k];
        atom->v[i][k] = box_store[i][3 + k] * inv_mL;
      }
    }
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

void FixSpinCFI::box_T(double a)
{
  const double L = std::cbrt(box_V);
  const double L2 = L * L;
  double local_twiceK = 0.0;
  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; ++i) {
    const double m = (atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]]) * force->mvv2e;
    const double inv_denom = 1.0 / (m * L2);
    for (int k = 0; k < 3; ++k) {
      const double p = box_store[i][3 + k];
      local_twiceK += p * p * inv_denom;
      box_store[i][k] += a * p * inv_denom;
    }
  }
  double global_twiceK = 0.0;
  MPI_Allreduce(&local_twiceK, &global_twiceK, 1, MPI_DOUBLE, MPI_SUM, world);
  box_pV += a * global_twiceK / (3.0 * box_V);
}

void FixSpinCFI::box_sync()
{
  const double L = std::cbrt(box_V);
  const double old_prd[3] = {domain->xprd, domain->yprd, domain->zprd};
  for (int k = 0; k < 3; ++k) domain->boxhi[k] = box_origin[k] + box_shape[k] * L;
  domain->set_global_box();
  domain->set_local_box();
  const double new_prd[3] = {domain->xprd, domain->yprd, domain->zprd};

  const int nlocal = atom->nlocal;
  const bool dipoles = atom->bec_flag != 0;
  for (int i = 0; i < nlocal; ++i) {
    // Preserve existing image flags so atoms crossing box boundaries never
    // teleport between reneighbor steps; Verlet domain->pbc() wraps on reneigh.
    const imageint img = atom->image[i];
    const int nbox[3] = {static_cast<int>(img & IMGMASK) - IMGMAX,
                         static_cast<int>((img >> IMGBITS) & IMGMASK) - IMGMAX,
                         static_cast<int>(img >> IMG2BITS) - IMGMAX};
    double xu_old[3] = {0.0, 0.0, 0.0};
    if (dipoles) {
      for (int k = 0; k < 3; ++k) xu_old[k] = atom->x[i][k] + nbox[k] * old_prd[k];
    }
    const double m = (atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]]) * force->mvv2e;
    const double inv_mL = 1.0 / (m * L);
    double xu_new[3];
    for (int k = 0; k < 3; ++k) {
      xu_new[k] = box_origin[k] + L * box_store[i][k];
      atom->x[i][k] = xu_new[k] - nbox[k] * new_prd[k];
      atom->v[i][k] = box_store[i][3 + k] * inv_mL;
    }
    if (dipoles) {
      DREAM::add_displacement(atom, i, xu_new[0] - xu_old[0], xu_new[1] - xu_old[1],
                              xu_new[2] - xu_old[2]);
    }
  }
}

void FixSpinCFI::box_D(double a)
{
  box_T(0.5 * a);
  box_V += a * box_pV / box_W;
  if (!std::isfinite(box_V) || box_V <= 0.0)
    error->all(FLERR, "fix {} box iso encountered nonpositive volume", style);
  box_T(0.5 * a);
  box_sync();
}

void FixSpinCFI::initial_integrate(int /*vflag*/)
{
  const int nlocal = integrated_atoms();
  double **sp = atom->sp;
  double **fm = atom->fm;
  int *mask = atom->mask;

  if (!cache_valid) {
    if (!spingnn2) refresh_analytic_torques(nlocal);
    for (int i = 0; i < nlocal; ++i) {
      if ((mask[i] & groupbit) && sp[i][3] > 0.0) {
        for (int k = 0; k < 3; ++k) store[i][k] = fm[i][k];
      }
    }
    cache_valid = 1;
  }

  const double dt = update->dt;
  const uint64_t step = static_cast<uint64_t>(update->ntimestep);

  if (box_iso) {
    if (ou_enabled || box_ou_enabled) box_bath_half(0);
    box_D(0.5 * dt);
  } else if (lattice_moving) {
    if (ou_enabled) {
      const double decay = std::exp(-0.5 * dt / ou_damp);
      const double variance = -std::expm1(-dt / ou_damp) * force->boltz * ou_temp * force->ftm2v;
      for (int i = 0; i < nlocal; ++i) {
        if (!(mask[i] & groupbit)) continue;
        const double mass = atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
        SpinCFIMath::ou_velocity_half(atom->v[i], mass, decay, variance, ou_seed,
                                      static_cast<uint64_t>(atom->tag[i]), step, 1);
      }
    }
    const double half_dt = 0.5 * dt;
    for (int i = 0; i < nlocal; ++i) {
      if (!(mask[i] & groupbit)) continue;
      double dx[3];
      for (int k = 0; k < 3; ++k) {
        dx[k] = half_dt * atom->v[i][k];
        atom->x[i][k] += dx[k];
      }
      if (atom->bec_flag) DREAM::add_displacement(atom, i, dx[0], dx[1], dx[2]);
    }
  }

  for (int i = 0; i < nlocal; ++i) {
    if (!(mask[i] & groupbit) || sp[i][3] <= 0.0) continue;
    for (int k = 0; k < 3; ++k) store[i][3 + k] = sp[i][k];
    if (predictor_fresh) continue;

    double noise[3] = {0.0, 0.0, 0.0};
    if (thermal != 0.0)
      SpinCFIMath::gaussian_stream(seed, static_cast<uint64_t>(atom->tag[i]), step, 0, noise);

    double mid[3];
    const bool ok = SpinCFIMath::cayley_step(&store[i][3], &store[i][3], &store[i][0], noise, hbar,
                                             alpha, thermal, dts / sp[i][3], true, mid);
    if (!ok) error->one(FLERR, "fix {} midpoint spin denominator collapse", style);
    for (int k = 0; k < 3; ++k) sp[i][k] = mid[k];
  }
}

void FixSpinCFI::final_integrate()
{
  const int nlocal = integrated_atoms();
  double **sp = atom->sp;
  double **fm = atom->fm;
  int *mask = atom->mask;
  const double dt = update->dt;
  const uint64_t step = static_cast<uint64_t>(update->ntimestep);

  if (predictor_fresh) {
    if (!spingnn2) refresh_analytic_torques(nlocal);
    for (int i = 0; i < nlocal; ++i) {
      if (!(mask[i] & groupbit) || sp[i][3] <= 0.0) continue;
      double noise[3] = {0.0, 0.0, 0.0};
      if (thermal != 0.0)
        SpinCFIMath::gaussian_stream(seed, static_cast<uint64_t>(atom->tag[i]), step, 0, noise);
      double mid[3];
      const bool ok = SpinCFIMath::cayley_step(&store[i][3], &store[i][3], fm[i], noise, hbar,
                                               alpha, thermal, dts / sp[i][3], true, mid);
      if (!ok) error->one(FLERR, "fix {} fresh midpoint spin denominator collapse", style);
      for (int k = 0; k < 3; ++k) sp[i][k] = mid[k];
    }
    evaluate_pair_fresh(false);
  }

  if (!spingnn2) refresh_analytic_torques(nlocal);

  for (int i = 0; i < nlocal; ++i) {
    if (!(mask[i] & groupbit) || sp[i][3] <= 0.0) continue;
    for (int k = 0; k < 3; ++k) store[i][k] = fm[i][k];

    double noise[3] = {0.0, 0.0, 0.0};
    if (thermal != 0.0)
      SpinCFIMath::gaussian_stream(seed, static_cast<uint64_t>(atom->tag[i]), step, 0, noise);

    double next[3];
    double dev = 0.0;
    const bool ok = SpinCFIMath::cayley_step(&store[i][3], sp[i], fm[i], noise, hbar, alpha,
                                             thermal, dts / sp[i][3], false, next, &dev);
    if (!ok) error->one(FLERR, "fix {} corrector spin normalization failure", style);
    for (int k = 0; k < 3; ++k) sp[i][k] = next[k];
    if (dev > maxnormdev) maxnormdev = dev;
  }

  if (box_iso) {
    if (!force->pair || update->vflag_global != update->ntimestep)
      error->all(FLERR, "fix {} box iso requires pair global virial", style);
    double local_virial[6];
    for (int k = 0; k < 6; ++k) local_virial[k] = force->pair->virial[k];
    MPI_Allreduce(local_virial, box_virial, 6, MPI_DOUBLE, MPI_SUM, world);

    const double L = std::cbrt(box_V);
    const double dtL = dt * L;
    for (int i = 0; i < atom->nlocal; ++i) {
      for (int k = 0; k < 3; ++k) box_store[i][3 + k] += dtL * atom->f[i][k];
    }
    box_pV += dt * ((box_virial[0] + box_virial[1] + box_virial[2]) / (3.0 * box_V) - box_P);
    box_D(0.5 * dt);
    if (ou_enabled || box_ou_enabled) box_bath_half(1);
    if (pressure) pressure->addstep(update->ntimestep + 1);
  } else if (lattice_moving) {
    const double half_dt = 0.5 * dt;
    const double dt_ftm2v = dt * force->ftm2v;
    for (int i = 0; i < nlocal; ++i) {
      if (!(mask[i] & groupbit)) continue;
      const double mass = atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
      const double kick = dt_ftm2v / mass;
      double dx[3];
      for (int k = 0; k < 3; ++k) {
        atom->v[i][k] += kick * atom->f[i][k];
        dx[k] = half_dt * atom->v[i][k];
        atom->x[i][k] += dx[k];
      }
      if (atom->bec_flag) DREAM::add_displacement(atom, i, dx[0], dx[1], dx[2]);
    }
    if (ou_enabled) {
      const double decay = std::exp(-0.5 * dt / ou_damp);
      const double variance = -std::expm1(-dt / ou_damp) * force->boltz * ou_temp * force->ftm2v;
      for (int i = 0; i < nlocal; ++i) {
        if (!(mask[i] & groupbit)) continue;
        const double mass = atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
        SpinCFIMath::ou_velocity_half(atom->v[i], mass, decay, variance, ou_seed,
                                      static_cast<uint64_t>(atom->tag[i]), step, 2);
      }
    }
  }

  ++nsteps;
}

void FixSpinCFI::end_of_step()
{
  // Accepted-state evaluation for consistent output/GPST; does not alter cached_fm.
  evaluate_pair_fresh(true);
  ++naccept;
}

void FixSpinCFI::post_run() {}

double FixSpinCFI::compute_vector(int n)
{
  if (n == 0) return static_cast<double>(nsteps);
  if (n == 1) return static_cast<double>(naccept);
  if (n == 2) {
    double global_dev = maxnormdev;
    MPI_Allreduce(MPI_IN_PLACE, &global_dev, 1, MPI_DOUBLE, MPI_MAX, world);
    return global_dev;
  }
  if (!box_iso) return 0.0;
  if (n == 3) return box_V;
  if (n == 4) return box_pV;
  if (n == 5) return (box_W > 0.0 && box_V > 0.0) ? (box_pV / (box_W * box_V)) : 0.0;
  if (n == 6) return box_P;
  if (n == 7) return force->nktv2p;
  if (n >= 8 && n < 14) return box_virial[n - 8];
  return 0.0;
}

void *FixSpinCFI::extract(const char *name, int &dim)
{
  dim = 0;
  if (std::strcmp(name, "spin_lattice_moving") == 0) return &lattice_moving_flag;
  if (std::strcmp(name, "cfi_box_iso") == 0 || std::strcmp(name, "lpsib_box_iso") == 0)
    return &box_iso_flag;
  return nullptr;
}
