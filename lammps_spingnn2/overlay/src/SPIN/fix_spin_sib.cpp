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

/* Adapted from the SpinGNN SIB extension and the LAMMPS spin integrator
   contributed by Julien Tranchida and Aidan Thompson (SNL). */

#include "fix_spin_sib.h"

#include "atom.h"
#include "comm.h"
#include "dream_bec.h"
#include "error.h"
#include "fix_precession_spin.h"
#include "fix_setforce_spin.h"
#include "force.h"
#include "math_const.h"
#include "modify.h"
#include "pair_spin.h"
#include "pair_spingnn2.h"
#include "random_mars.h"
#include "spin_sib_math.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

SpinSIB::SpinSIB(LAMMPS *lmp) : Pointers(lmp) {}

SpinSIB::~SpinSIB() = default;

void SpinSIB::configure(double target, double damping, int random_seed)
{
  if (!std::isfinite(target) || target < 0.0 || !std::isfinite(damping) || damping < 0.0)
    error->all(FLERR, "SIB requires a nonnegative spin temperature and damping");
  if (random_seed <= 0) error->all(FLERR, "SIB requires a positive random seed");
  if (!atom->sp_flag) error->all(FLERR, "SIB requires atom attribute spin");
  temperature = target;
  alpha = damping;
  seed = random_seed;
}

void SpinSIB::init(bool kokkos, bool frozen, ExecutionSpace space, Fix *cache_owner)
{
  if (!utils::strmatch(update->integrate_style, "^verlet"))
    error->all(FLERR, "SIB requires the Verlet run style");

  spin_pairs.clear();
  precession.clear();
  setforce = nullptr;
  spingnn2 = dynamic_cast<PairSpinGNN2 *>(force->pair);
  if (!spingnn2 && force->pair_match("^spingnn2", 0, 1))
    error->all(FLERR, "SIB requires spingnn2 to be the active pair style");

  if (spingnn2) {
    if (kokkos != static_cast<bool>(spingnn2->kokkosable) ||
        (kokkos && space != spingnn2->execution_space))
      error->all(FLERR, "SIB and spingnn2 require matching execution spaces");
  } else {
    if (kokkos) error->all(FLERR, "fix spin/sib/kk requires pair style spingnn2/kk");
    if (force->pair_match("^hybrid", 0)) {
      for (int i = 1; Pair *pair = force->pair_match("^spin/", 0, i); ++i) {
        auto *spin = dynamic_cast<PairSpin *>(pair);
        if (!spin) error->all(FLERR, "Invalid SIB spin pair sub-style");
        spin_pairs.push_back(spin);
      }
    } else if (auto *spin = dynamic_cast<PairSpin *>(force->pair)) {
      spin_pairs.push_back(spin);
    }
    if (!random) random = std::make_unique<RanMars>(lmp, seed + comm->me);
  }

  for (auto *fix : modify->get_fix_list()) {
    if (auto *precess = dynamic_cast<FixPrecessionSpin *>(fix)) precession.push_back(precess);
    if (auto *set = dynamic_cast<FixSetForceSpin *>(fix)) setforce = set;
    if (spingnn2 && utils::strmatch(fix->style, "^langevin/spin"))
      error->all(FLERR, "SIB with spingnn2 already includes spin Langevin integration");
  }
  if (kokkos && (!precession.empty() || setforce))
    error->all(FLERR, "fix spin/sib/kk requires device spin torques from spingnn2/kk");
  if (spingnn2 && cache_owner) spingnn2->set_sib_lattice(cache_owner, frozen);

  hbar = force->hplanck / MathConst::MY_2PI;
  thermal = std::sqrt(2.0 * alpha * force->boltz * temperature / (1.0 + alpha * alpha));
  if (spingnn2 && thermal != 0.0 && !atom->tag_enable)
    error->all(FLERR, "Stochastic SIB with spingnn2 requires atom IDs");
  reset_dt();
}

void SpinSIB::reset_dt()
{
  dts = 2.0 * update->dt / ((1.0 + alpha * alpha) * hbar);
}

void SpinSIB::refresh_torques(int nlocal, int groupbit, bool use_spingnn2)
{
  for (int i = 0; i < nlocal; ++i) {
    if (!(atom->mask[i] & groupbit)) continue;
    auto *spin = atom->sp[i];
    auto *torque = atom->fm[i];
    if (!use_spingnn2) {
      torque[0] = torque[1] = torque[2] = 0.0;
      for (auto *pair : spin_pairs) pair->compute_single_pair(i, torque);
    }
    for (auto *fix : precession) fix->compute_single_precession(i, spin, torque);
    if (setforce) setforce->single_setforce_spin(i, torque);
  }
}

void SpinSIB::midpoint(bool reuse_lattice)
{
  // The neighbor topology still belongs to the last regular force call.
  // Its ghost spin values must be current before the midpoint evaluation.
  const double start = spingnn2 ? MPI_Wtime() : 0.0;
  comm->forward_comm();
  if (spingnn2) {
    spingnn2->timing[1] += MPI_Wtime() - start;
    spingnn2->compute_sib_midpoint(reuse_lattice);
  }
}

void SpinSIB::advance(int nlocal, int groupbit, bool reuse_lattice)
{
  if (workspace.size() < static_cast<size_t>(nlocal)) workspace.resize(nlocal);
  // The native pair's previous force already includes the spin post-force
  // fixes. Conventional SPIN styles expose their per-spin torques directly.
  if (!spingnn2) refresh_torques(nlocal, groupbit, false);

  for (int i = 0; i < nlocal; ++i) {
    auto *spin = atom->sp[i];
    if (!(atom->mask[i] & groupbit) || spin[3] <= 0.0) continue;
    auto &work = workspace[i];
    for (int k = 0; k < 3; ++k) work.original[k] = spin[k];
    work.noise.fill(0.0);
    if (!spingnn2) {
      for (double &draw : work.noise) draw = random->gaussian();
    } else if (thermal != 0.0) {
      SpinSIBMath::gaussian(seed, atom->tag[i], update->ntimestep, work.noise.data());
    }
    SpinSIBMath::advance(work.original.data(), work.original.data(), atom->fm[i], work.noise.data(),
                         hbar, alpha, thermal, dts / spin[3], true, spin);
  }

  midpoint(reuse_lattice);
  refresh_torques(nlocal, groupbit, spingnn2 != nullptr);

  for (int i = 0; i < nlocal; ++i) {
    auto *spin = atom->sp[i];
    if (!(atom->mask[i] & groupbit) || spin[3] <= 0.0) continue;
    const auto &work = workspace[i];
    SpinSIBMath::advance(work.original.data(), spin, atom->fm[i], work.noise.data(),
                         hbar, alpha, thermal, dts / spin[3], false, spin);
  }
}

FixNVESpinSIB::FixNVESpinSIB(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg), sib(lmp)
{
  if (narg != 8 || strcmp(arg[6], "lattice") != 0)
    error->all(FLERR, "Expected fix {} temperature damping seed lattice moving|frozen", style);
  if (strcmp(arg[7], "moving") == 0 || strcmp(arg[7], "yes") == 0)
    lattice_moving = true;
  else if (strcmp(arg[7], "frozen") == 0 || strcmp(arg[7], "no") == 0)
    lattice_moving = false;
  else
    error->all(FLERR, "Unknown fix {} lattice mode {}", style, arg[7]);
  sib.configure(utils::numeric(FLERR, arg[3], false, lmp),
                utils::numeric(FLERR, arg[4], false, lmp),
                utils::inumeric(FLERR, arg[5], false, lmp));
  time_integrate = 1;
}

int FixNVESpinSIB::setmask()
{
  return INITIAL_INTEGRATE | FINAL_INTEGRATE;
}

void FixNVESpinSIB::init()
{
  sib.init(kokkosable, !lattice_moving && igroup == 0, execution_space, this);
  reset_dt();
}

int FixNVESpinSIB::integrated_atoms() const
{
  return igroup == atom->firstgroup ? atom->nfirst : atom->nlocal;
}

void FixNVESpinSIB::initial_integrate(int /*vflag*/)
{
  const int nlocal = integrated_atoms();
  if (lattice_moving) final_integrate();

  sib.advance(nlocal, groupbit, true);
  if (lattice_moving) {
    for (int i = 0; i < nlocal; ++i) {
      if (!(atom->mask[i] & groupbit)) continue;
      double dx[3];
      for (int k = 0; k < 3; ++k) {
        dx[k] = dtv * atom->v[i][k];
        atom->x[i][k] += dx[k];
      }
      if (atom->bec_flag) DREAM::add_displacement(atom, i, dx[0], dx[1], dx[2]);
    }
  }
}

void FixNVESpinSIB::final_integrate()
{
  if (!lattice_moving) return;
  const int nlocal = integrated_atoms();
  for (int i = 0; i < nlocal; ++i) {
    if (!(atom->mask[i] & groupbit)) continue;
    const double kick = dtf / (atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]]);
    for (int k = 0; k < 3; ++k) atom->v[i][k] += kick * atom->f[i][k];
  }
}

void FixNVESpinSIB::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * dtv * force->ftm2v;
  sib.reset_dt();
}
