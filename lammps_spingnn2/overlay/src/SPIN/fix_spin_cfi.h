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
// SPDX-License-Identifier: GPL-2.0-only

#ifdef FIX_CLASS
// clang-format off
FixStyle(spin/cfi,FixSpinCFI);
FixStyle(spin/lpsib,FixSpinCFI);
// clang-format on
#else

#ifndef LMP_FIX_SPIN_CFI_H
#define LMP_FIX_SPIN_CFI_H

#include "fix.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

class Compute;
class PairSpin;
class PairSpinGNN2;
class FixPrecessionSpin;

class FixSpinCFI : public Fix {
 public:
  FixSpinCFI(class LAMMPS *, int, char **);
  ~FixSpinCFI() override;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void reset_dt() override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void end_of_step() override;
  void post_run() override;
  int modify_param(int, char **) override;

  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  void set_arrays(int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  double memory_usage() override;
  double compute_vector(int) override;
  void *extract(const char *, int &) override;

 protected:
  int integrated_atoms() const;
  void refresh_analytic_torques(int nlocal);
  void evaluate_pair_fresh(bool check_reneighbor);

  // Isotropic canonical-volume barostat helpers (active only when box_iso).
  void box_init();
  void box_bath_half(int half);
  void box_T(double a);
  void box_D(double a);
  void box_sync();

  // Per-atom persistent CFI state: [0..2] = cached midpoint fm, [3..5] = s_n.
  double **store = nullptr;
  // Optional per-atom barostat state when box_iso: [0..2] = scaled q, [3..5] = canonical p.
  double **box_store = nullptr;

  double temperature = 0.0;
  double alpha = 0.0;
  double thermal = 0.0;
  double hbar = 0.0;
  double dts = 0.0;
  int seed = 0;

  bool lattice_moving = false;
  bool predictor_fresh = false;
  int accept = 0;
  int cache_valid = 0;
  int lattice_moving_flag = 0;

  bool ou_enabled = false;
  double ou_temp = 0.0;
  double ou_damp = 0.0;
  int ou_seed = 0;

  bool box_iso = false;
  int box_iso_flag = 0;
  int box_initialized = 0;
  bool box_ou_enabled = false;
  double box_ou_temp = 0.0;
  double box_ou_damp = 0.0;
  int box_ou_seed = 0;
  double box_Pbar = 0.0;
  double box_P = 0.0;
  double box_W = 0.0;
  double box_V = 0.0;
  double box_pV = 0.0;
  double box_origin[3] = {0.0, 0.0, 0.0};
  double box_shape[3] = {1.0, 1.0, 1.0};
  double box_virial[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  std::string id_press;
  int pcomputeflag = 0;
  Compute *pressure = nullptr;

  bigint nsteps = 0;
  bigint naccept = 0;
  double maxnormdev = 0.0;

  PairSpinGNN2 *spingnn2 = nullptr;
  std::vector<PairSpin *> spin_pairs;
  std::vector<FixPrecessionSpin *> precession;
};

}  // namespace LAMMPS_NS

#endif
#endif
