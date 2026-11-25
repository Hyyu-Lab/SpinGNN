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
FixStyle(spin/sib,FixNVESpinSIB);
// clang-format on
#else

#ifndef LMP_FIX_SPIN_SIB_H
#define LMP_FIX_SPIN_SIB_H

#include "fix.h"

#include <array>
#include <memory>
#include <vector>

namespace LAMMPS_NS {

class PairSpin;
class PairSpinGNN2;
class FixPrecessionSpin;
class FixSetForceSpin;
class RanMars;

class SpinSIB : protected Pointers {
 public:
  explicit SpinSIB(LAMMPS *);
  ~SpinSIB();

  void configure(double temperature, double damping, int seed);
  void init(bool kokkos, bool frozen, ExecutionSpace, Fix *cache_owner = nullptr);
  void reset_dt();
  void advance(int nlocal, int groupbit, bool reuse_lattice);
  void midpoint(bool reuse_lattice);

  double alpha = 0.0, thermal = 0.0, hbar = 0.0, dts = 0.0;
  int seed = 0;

 private:
  struct Workspace {
    std::array<double, 3> original, noise;
  };

  void refresh_torques(int nlocal, int groupbit, bool use_spingnn2);

  double temperature = 0.0;
  PairSpinGNN2 *spingnn2 = nullptr;
  std::unique_ptr<RanMars> random;
  std::vector<PairSpin *> spin_pairs;
  std::vector<FixPrecessionSpin *> precession;
  FixSetForceSpin *setforce = nullptr;
  std::vector<Workspace> workspace;
};

class FixNVESpinSIB : public Fix {
 public:
  FixNVESpinSIB(LAMMPS *, int, char **);
  int setmask() override;
  void init() override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void reset_dt() override;

 protected:
  int integrated_atoms() const;
  bool lattice_moving;
  double dtv = 0.0, dtf = 0.0;
  SpinSIB sib;
};

}  // namespace LAMMPS_NS

#endif
#endif
