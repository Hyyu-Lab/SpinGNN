// SPDX-License-Identifier: GPL-2.0-only
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(bec,ComputeBEC);
ComputeStyle(bec_dipole,ComputeBEC);
// clang-format on
#else

#ifndef LMP_COMPUTE_BEC_H
#define LMP_COMPUTE_BEC_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeBEC : public Compute {
 public:
  ComputeBEC(class LAMMPS *, int, char **);
  ~ComputeBEC() override;
  void init() override;
  void compute_vector() override;
  double compute_scalar() override;
};

}    // namespace LAMMPS_NS

#endif
#endif
