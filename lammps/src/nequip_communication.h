/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   Copyright (2003) Sandia Corporation. Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software. This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_NEQUIP_COMMUNICATION_H
#define LMP_NEQUIP_COMMUNICATION_H

#include "comm.h"
#include "deploy_metadata.h"
#include "domain.h"
#include "lmptype.h"
#include "neighbor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace LAMMPS_NS {

inline int read_nequip_num_interactions(
    Error *error, const DeployMetadata &metadata, const std::string &pair_style)
{
  const auto entry = metadata.find("nequip_num_interactions");
  if (entry == metadata.end() || entry->second.empty()) return 0;
  const std::string &value = entry->second;
  long count = 0;
  try {
    size_t consumed = 0;
    count = std::stol(value, &consumed);
    if (consumed != value.size() || value.find_first_not_of("0123456789") !=
            std::string::npos || count <= 0 || count > std::numeric_limits<int>::max())
      count = 0;
  } catch (const std::exception &) {
    count = 0;
  }
  if (count == 0)
    error->all(FLERR, "Pair style " + pair_style +
                         " requires nequip_num_interactions to be a positive integer");
  return static_cast<int>(count);
}

inline void configure_nequip_communication(
    Error *error, Comm *comm, Neighbor *neighbor, int num_interactions,
    double cutoff, const std::string &pair_style)
{
  if (comm->nprocs == 1) return;
  if (num_interactions == 0)
    error->all(
        FLERR, "Pair style " + pair_style +
                   " requires deployed metadata nequip_num_interactions for MPI; "
                   "re-export the model with the current nequip-deploy. "
                   "Legacy artifacts remain supported on one rank");
  if (comm->mode != Comm::SINGLE)
    error->all(FLERR, "Pair style " + pair_style +
                         " requires comm_modify mode single for its MPI halo");

  // With Newton off each rank differentiates all energies that can depend on
  // an owned atom. Their centers can be L cutoffs away, and their own inputs
  // reach another L cutoffs. Enlarge communication, not the physical pair
  // cutoff, so the graph still contains only actual model-neighbor edges.
  const double required = 2.0 * num_interactions * cutoff + neighbor->skin;
  if (!std::isfinite(required) || required <= 0.0)
    error->all(FLERR, "Pair style " + pair_style + " has an invalid MPI halo");
  comm->cutghostuser = std::max(comm->cutghostuser, required);
}

inline void nequip_periodic_shift(
    const Domain *domain, double **x, int i, int j,
    const double *canonical_i, const double *canonical_j, float *shift)
{
  // Canonical graph positions and ghost images describe the same physical
  // atoms. Use native double coordinates and the existing inverse cell to
  // recover integer image shifts without allocating tiny tensors per edge.
  const double dx = (x[j][0] - canonical_j[0]) - (x[i][0] - canonical_i[0]);
  const double dy = (x[j][1] - canonical_j[1]) - (x[i][1] - canonical_i[1]);
  const double dz = (x[j][2] - canonical_j[2]) - (x[i][2] - canonical_i[2]);
  // Off-diagonal h_inv entries are only maintained for triclinic boxes.
  shift[0] = std::round(domain->h_inv[0] * dx + (domain->triclinic ?
      domain->h_inv[5] * dy + domain->h_inv[4] * dz : 0.0));
  shift[1] = std::round(domain->h_inv[1] * dy + (domain->triclinic ?
      domain->h_inv[3] * dz : 0.0));
  shift[2] = std::round(domain->h_inv[2] * dz);
}

inline bool nequip_prefer_center(
    int current, bool current_has_center, int candidate, int nlocal,
    double **x, const int *numneigh, int **firstneigh, double cutoff_sq)
{
  if (!current_has_center) return true;
  // Ownership determines energy and virial attribution, so a ghost image
  // must never replace an owned representative of the same physical atom.
  if (current < nlocal) return false;
  if (candidate < nlocal) return true;

  // A periodic image at the halo boundary can have a truncated neighbor
  // list even when another image of the same atom is complete. The 2L halo
  // contains a complete image for every center affecting an owned force.
  // Compare physical model edges, since skin and hybrid cutoffs can add
  // unrelated neighbors that do not measure graph completeness.
  auto physical_neighbors = [&](int i) {
    int count = 0;
    for (int jj = 0; jj < numneigh[i]; jj++) {
      const int j = firstneigh[i][jj] & NEIGHMASK;
      const double dx = x[i][0] - x[j][0];
      const double dy = x[i][1] - x[j][1];
      const double dz = x[i][2] - x[j][2];
      if (dx * dx + dy * dy + dz * dz < cutoff_sq) count++;
    }
    return count;
  };
  return physical_neighbors(candidate) > physical_neighbors(current);
}

} // namespace LAMMPS_NS

#endif
