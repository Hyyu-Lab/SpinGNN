// SPDX-License-Identifier: GPL-2.0-only
#include "compute_spingnn2_timing.h"
#include "error.h"
#include "force.h"
#include "pair_spingnn2.h"
#include "update.h"
#include <cstring>
using namespace LAMMPS_NS;

ComputeSpinGNN2Timing::ComputeSpinGNN2Timing(LAMMPS *lmp,int narg,char **arg) : Compute(lmp,narg,arg)
{
  graph = narg == 4 && std::strcmp(arg[3],"graph") == 0;
  if (narg != 3 && !graph) error->all(FLERR,"Expected compute ID all spingnn2/timing [graph]");
  vector_flag = 1;
  size_vector = graph ? 3 : 6;
  extvector = 0;
  vector = counters;
}
void ComputeSpinGNN2Timing::init()
{
  pair = dynamic_cast<PairSpinGNN2 *>(force->pair);
  if (!pair) error->all(FLERR,"compute spingnn2/timing requires pair_style spingnn2");
}
void ComputeSpinGNN2Timing::compute_vector()
{
  invoked_vector = update->ntimestep;
  MPI_Allreduce(graph ? pair->graph_timing : pair->timing,counters,size_vector,MPI_DOUBLE,MPI_MAX,world);
}
