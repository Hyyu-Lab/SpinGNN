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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(spin/gpst,ComputeSpinGPST);
// clang-format on
#else

#ifndef LMP_COMPUTE_SPIN_GPST_H
#define LMP_COMPUTE_SPIN_GPST_H

#include "compute.h"

#include <cmath>

namespace LAMMPS_NS {

class NeighList;

#if defined(KOKKOS_INLINE_FUNCTION)
#define GPST_INLINE KOKKOS_INLINE_FUNCTION
#elif defined(__CUDACC__) || defined(__HIPCC__)
#define GPST_INLINE __host__ __device__ inline
#else
#define GPST_INLINE inline
#endif

namespace SpinGPSTMath {

// Register-only 4-scalar sliding-window Legendre recurrence for l in {1, 2, 4, 8}.
// Eliminates local p[9]/dp[9] stack arrays inside GPU kernels.
GPST_INLINE static void legendre_pair_1248(double x, double &p1, double &dp1, double &p2,
                                           double &dp2, double &p4, double &dp4, double &p8,
                                           double &dp8)
{
  if (x > 1.0) x = 1.0;
  if (x < -1.0) x = -1.0;
  double p_prev = 1.0;
  double p_curr = x;
  double dp_prev = 0.0;
  double dp_curr = 1.0;
  p1 = p_curr;
  dp1 = dp_curr;
  p2 = dp2 = p4 = dp4 = p8 = dp8 = 0.0;
  for (int k = 1; k < 8; ++k) {
    const double a = static_cast<double>(2 * k + 1);
    const double b = static_cast<double>(k);
    const double inv = 1.0 / static_cast<double>(k + 1);
    const double p_next = (a * x * p_curr - b * p_prev) * inv;
    const double dp_next = (a * (p_curr + x * dp_curr) - b * dp_prev) * inv;
    p_prev = p_curr;
    p_curr = p_next;
    dp_prev = dp_curr;
    dp_curr = dp_next;
    if (k == 1) {
      p2 = p_curr;
      dp2 = dp_curr;
    } else if (k == 3) {
      p4 = p_curr;
      dp4 = dp_curr;
    } else if (k == 7) {
      p8 = p_curr;
      dp8 = dp_curr;
    }
  }
}

// Explicit Horner evaluation of P_4(z) and P_4'(z) with exact binary coefficients.
GPST_INLINE static void legendre_4(double z, double &p4, double &dp4)
{
  if (z > 1.0) z = 1.0;
  if (z < -1.0) z = -1.0;
  const double z2 = z * z;
  p4 = (4.375 * z2 - 3.75) * z2 + 0.375;
  dp4 = (17.5 * z2 - 7.5) * z;
}

}  // namespace SpinGPSTMath

#undef GPST_INLINE

class ComputeSpinGPST : public Compute {
 public:
  ComputeSpinGPST(class LAMMPS *, int, char **);
  ~ComputeSpinGPST() override;

  void init() override;
  void init_list(int, NeighList *) override;
  double compute_scalar() override;
  void compute_vector() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

 protected:
  void finalize_vector(const double local_sums[18]);

  double cutoff = 0.0;
  double cutsq = 0.0;
  double hbar = 0.0;
  double kb = 0.0;
  NeighList *list = nullptr;
};

}  // namespace LAMMPS_NS

#endif
#endif
