/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(spin/sib_npt,FixNVESpinSIBNPT);
FixStyle(spin/sib/npt,FixNVESpinSIBNPT);
// clang-format on
#else

#ifndef LMP_FIX_SPIN_SIB_NPT_H
#define LMP_FIX_SPIN_SIB_NPT_H

#include "fix_npt.h"
#include "fix_spin_sib.h"

namespace LAMMPS_NS {

struct SpinSIBNPTArguments {
  std::vector<char *> nh;
  double alpha = -1.0;
  int seed = -1;
};

SpinSIBNPTArguments parse_spin_sib_npt(LAMMPS *, int, char **);

class FixNVESpinSIBNPT : public FixNPT {
 public:
  FixNVESpinSIBNPT(LAMMPS *, int, char **);
  void init() override;
  void reset_dt() override;

 protected:
  void nve_x() override;

 private:
  FixNVESpinSIBNPT(LAMMPS *, SpinSIBNPTArguments);
  SpinSIB sib;
};

}  // namespace LAMMPS_NS

#endif
#endif
