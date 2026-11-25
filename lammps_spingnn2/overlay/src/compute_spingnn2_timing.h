// SPDX-License-Identifier: GPL-2.0-only
#ifdef COMPUTE_CLASS
ComputeStyle(spingnn2/timing,ComputeSpinGNN2Timing);
#else
#ifndef LMP_COMPUTE_SPINGNN2_TIMING_H
#define LMP_COMPUTE_SPINGNN2_TIMING_H
#include "compute.h"
namespace LAMMPS_NS {
class ComputeSpinGNN2Timing : public Compute {
 public:
  ComputeSpinGNN2Timing(LAMMPS *, int, char **);
  void init() override;
  void compute_vector() override;
 private:
  class PairSpinGNN2 *pair;
  double counters[6]{};
  bool graph;
};
}
#endif
#endif
