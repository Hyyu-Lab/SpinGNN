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

#ifndef LMP_SPIN_CFI_MATH_H
#define LMP_SPIN_CFI_MATH_H

#include "spin_sib_math.h"

#include <cmath>
#include <cstdint>

#if defined(KOKKOS_INLINE_FUNCTION)
#define CFI_INLINE KOKKOS_INLINE_FUNCTION
#elif defined(__CUDACC__) || defined(__HIPCC__)
#define CFI_INLINE __host__ __device__ inline
#else
#define CFI_INLINE inline
#endif

namespace LAMMPS_NS::SpinCFIMath {

// Deterministic counter-based 3D standard normal keyed by (seed, tag, step, stream).
// stream == 0 matches SpinSIBMath::gaussian(seed, tag, step) identically.
CFI_INLINE static void gaussian_stream(int seed, uint64_t tag, uint64_t step, uint64_t stream,
                                       double out[3])
{
  const uint64_t stream_mix = stream == 0
      ? UINT64_C(0)
      : SpinSIBMath::mix(stream ^ UINT64_C(0x517cc1b727220a95));
  const uint64_t key = SpinSIBMath::mix(SpinSIBMath::mix(tag) ^
                                        (step + UINT64_C(0x243f6a8885a308d3))) ^
      SpinSIBMath::mix(static_cast<uint64_t>(seed)) ^ stream_mix;
  const double first_angle = 6.2831853071795864769 * SpinSIBMath::uniform(key + 1);
  const double first_radius = ::sqrt(-2.0 * ::log(SpinSIBMath::uniform(key)));
  out[0] = first_radius * ::cos(first_angle);
  out[1] = first_radius * ::sin(first_angle);
  const double second_angle = 6.2831853071795864769 * SpinSIBMath::uniform(key + 3);
  out[2] = ::sqrt(-2.0 * ::log(SpinSIBMath::uniform(key + 2))) * ::cos(second_angle);
}

// Normalized predicted midpoint or full Cayley corrector update.
// Returns false if the unnormalized midpoint vector collapses (2*|m| < 1e-6) or is non-finite.
CFI_INLINE static bool cayley_step(const double original[3], const double direction[3],
                                   const double fm[3], const double noise[3], double hbar,
                                   double alpha, double thermal, double dt, bool midpoint,
                                   double out[3], double *norm_dev = nullptr)
{
  double field[3], field_damping[3], noise_damping[3], w[3];
  for (int k = 0; k < 3; ++k) field[k] = hbar * fm[k];
  SpinSIBMath::cross(direction, field, field_damping);
  SpinSIBMath::cross(direction, noise, noise_damping);
  const double thermal_step = 0.5 * ::sqrt(dt) * thermal;
  for (int k = 0; k < 3; ++k)
    w[k] = -0.5 * dt * (field[k] + alpha * field_damping[k]) -
        thermal_step * (noise[k] + alpha * noise_damping[k]);

  double first[3], second[3];
  SpinSIBMath::cross(original, w, first);
  SpinSIBMath::cross(first, w, second);
  const double denom = 1.0 + w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  const double scale = (midpoint ? 1.0 : 2.0) / denom;
  for (int k = 0; k < 3; ++k) out[k] = original[k] + scale * (first[k] + second[k]);
  const double norm = ::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
  if (!(norm > 0.0) || (midpoint && 2.0 * norm < 1.0e-6)) return false;
  const double inv_norm = 1.0 / norm;
  for (int k = 0; k < 3; ++k) out[k] *= inv_norm;
  if (norm_dev) {
    const double post_norm = ::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
    *norm_dev = ::fabs(post_norm - 1.0);
  }
  return true;
}

// Exact Ornstein-Uhlenbeck half-step on atomic velocities.
CFI_INLINE static void ou_velocity_half(double v[3], double mass, double decay, double variance,
                                        int ou_seed, uint64_t tag, uint64_t step,
                                        uint64_t half_stream)
{
  double noise[3] = {0.0, 0.0, 0.0};
  if (variance > 0.0) gaussian_stream(ou_seed, tag, step, half_stream, noise);
  const double sigma = variance > 0.0 ? ::sqrt(variance / mass) : 0.0;
  for (int k = 0; k < 3; ++k) v[k] = decay * v[k] + sigma * noise[k];
}

// Deterministic counter-based scalar standard normal for barostat OU noise.
// Identical on all MPI ranks and host/device without per-rank RNG divergence.
CFI_INLINE static double gaussian_scalar(int seed, uint64_t step, uint64_t stream)
{
  const uint64_t stream_mix =
      SpinSIBMath::mix(stream ^ UINT64_C(0x9e3779b97f4a7c15));
  const uint64_t key = SpinSIBMath::mix(step + UINT64_C(0x243f6a8885a308d3)) ^
      SpinSIBMath::mix(static_cast<uint64_t>(seed)) ^ stream_mix;
  const double angle = 6.2831853071795864769 * SpinSIBMath::uniform(key + 1);
  const double radius = ::sqrt(-2.0 * ::log(SpinSIBMath::uniform(key)));
  return radius * ::cos(angle);
}

}  // namespace LAMMPS_NS::SpinCFIMath

#undef CFI_INLINE
#endif
