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

#ifndef LMP_SPIN_SIB_MATH_H
#define LMP_SPIN_SIB_MATH_H

#include <cmath>
#include <cstdint>

#if defined(KOKKOS_INLINE_FUNCTION)
#define SIB_INLINE KOKKOS_INLINE_FUNCTION
#elif defined(__CUDACC__) || defined(__HIPCC__)
#define SIB_INLINE __host__ __device__ inline
#else
#define SIB_INLINE inline
#endif

namespace LAMMPS_NS::SpinSIBMath {

SIB_INLINE static void cross(const double a[3], const double b[3], double out[3])
{
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

// The SIB update is a Cayley transform of a skew-symmetric matrix.
// Its inverse has determinant 1 + |w|^2, so three cross products suffice.
SIB_INLINE static void advance(const double original[3], const double direction[3],
                               const double fm[3], const double noise[3], double hbar,
                               double alpha, double thermal, double dt, bool midpoint,
                               double out[3])
{
  double field[3], field_damping[3], noise_damping[3], w[3];
  for (int k = 0; k < 3; ++k) field[k] = hbar * fm[k];
  cross(direction, field, field_damping);
  cross(direction, noise, noise_damping);
  const double thermal_step = 0.5 * ::sqrt(dt) * thermal;
  for (int k = 0; k < 3; ++k)
    w[k] = -0.5 * dt * (field[k] + alpha * field_damping[k]) -
        thermal_step * (noise[k] + alpha * noise_damping[k]);

  double first[3], second[3];
  cross(original, w, first);
  cross(first, w, second);
  const double scale = (midpoint ? 1.0 : 2.0) / (1.0 + w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  for (int k = 0; k < 3; ++k) out[k] = original[k] + scale * (first[k] + second[k]);
  const double inverse_norm = 1.0 / ::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
  for (int k = 0; k < 3; ++k) out[k] *= inverse_norm;
}

// Key the Gaussian noise by atom identity and step so sorting, exchange,
// and the CPU/device execution order cannot change the draws.
SIB_INLINE static uint64_t mix(uint64_t value)
{
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

SIB_INLINE static double uniform(uint64_t key)
{
  return (static_cast<double>(mix(key) >> 12) + 0.5) * 0x1.0p-52;
}

SIB_INLINE static void gaussian(int seed, uint64_t tag, uint64_t step, double out[3])
{
  const uint64_t key = mix(mix(tag) ^ (step + UINT64_C(0x243f6a8885a308d3))) ^
      mix(static_cast<uint64_t>(seed));
  const double first_angle = 6.2831853071795864769 * uniform(key + 1);
  const double first_radius = ::sqrt(-2.0 * ::log(uniform(key)));
  out[0] = first_radius * ::cos(first_angle);
  out[1] = first_radius * ::sin(first_angle);
  const double second_angle = 6.2831853071795864769 * uniform(key + 3);
  out[2] = ::sqrt(-2.0 * ::log(uniform(key + 2))) * ::cos(second_angle);
}

}  // namespace LAMMPS_NS::SpinSIBMath

#undef SIB_INLINE
#endif
