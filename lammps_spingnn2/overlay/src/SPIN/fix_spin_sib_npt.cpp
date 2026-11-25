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

/* Adapted from the SpinGNN SIB/NPT extension. */

#include "fix_spin_sib_npt.h"

#include "atom.h"
#include "error.h"
#include "lammps.h"

#include <cstring>

using namespace LAMMPS_NS;

SpinSIBNPTArguments LAMMPS_NS::parse_spin_sib_npt(LAMMPS *lmp, int narg, char **arg)
{
  if (narg < 3) lmp->error->all(FLERR, "Illegal fix spin/sib_npt command");
  SpinSIBNPTArguments result;
  result.nh.assign(arg, arg + 3);
  for (int i = 3; i < narg; ++i) {
    const bool damping = strcmp(arg[i], "alpha") == 0;
    const bool seed = strcmp(arg[i], "seed") == 0;
    if (damping || seed) {
      if (i + 1 == narg)
        lmp->error->all(FLERR, "Missing fix spin/sib_npt {} value", arg[i]);
      ++i;
      if (damping)
        result.alpha = utils::numeric(FLERR, arg[i], false, lmp);
      else
        result.seed = utils::inumeric(FLERR, arg[i], false, lmp);
    } else {
      result.nh.push_back(arg[i]);
      // The dilate group name can also be a valid SIB keyword.
      if (strcmp(arg[i], "dilate") == 0 && i + 1 < narg) result.nh.push_back(arg[++i]);
    }
  }
  return result;
}

FixNVESpinSIBNPT::FixNVESpinSIBNPT(LAMMPS *lmp, int narg, char **arg) :
    FixNVESpinSIBNPT(lmp, parse_spin_sib_npt(lmp, narg, arg))
{
}

FixNVESpinSIBNPT::FixNVESpinSIBNPT(LAMMPS *lmp, SpinSIBNPTArguments args) :
    FixNPT(lmp, static_cast<int>(args.nh.size()), args.nh.data()), sib(lmp)
{
  sib.configure(t_stop, args.alpha, args.seed);
}

void FixNVESpinSIBNPT::init()
{
  FixNPT::init();
  sib.init(false, false, execution_space);
}

void FixNVESpinSIBNPT::reset_dt()
{
  FixNPT::reset_dt();
  sib.reset_dt();
}

void FixNVESpinSIBNPT::nve_x()
{
  // Nose–Hoover remaps the box before the lattice drift. The midpoint
  // evaluation therefore needs the new geometry after forward communication.
  sib.advance(igroup == atom->firstgroup ? atom->nfirst : atom->nlocal, groupbit, false);
  FixNPT::nve_x();
}
