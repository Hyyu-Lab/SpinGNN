// SPDX-License-Identifier: GPL-2.0-only
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

#include "read_bec.h"

#include "atom.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "tokenizer.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ReadBEC::ReadBEC(LAMMPS *lmp) : Command(lmp) {}

/* ----------------------------------------------------------------------
   read fixed Born effective charges

   File format, one atom per non-comment line:
   id Zxx Zxy Zxz Zyx Zyy Zyz Zzx Zzy Zzz
------------------------------------------------------------------------- */

void ReadBEC::command(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal read_bec command");
  if (domain->box_exist == 0)
    error->all(FLERR,"Read_bec command before simulation box is defined");
  if (!atom->tag_enable)
    error->all(FLERR,"Read_bec command requires atom IDs");
  if (!atom->bec_flag)
    error->all(FLERR,"Read_bec command requires atom attribute bec");

  std::vector<tagint> ids;
  std::vector<double> values;

  if (comm->me == 0) {
    std::ifstream file(arg[0]);
    if (!file.is_open())
      error->one(FLERR,"Cannot open BEC file {}",arg[0]);

    std::unordered_set<tagint> seen;
    std::string line;
    int lineno = 0;

    while (std::getline(file,line)) {
      lineno++;
      const std::string text = utils::trim(utils::trim_comment(line));
      if (text.empty()) continue;

      try {
        ValueTokenizer tokens(text);
        if (tokens.count() != 10)
          error->one(FLERR,"Incorrect format in BEC file {} at line {}: "
                     "expected atom ID and 9 tensor components",arg[0],lineno);

        const tagint id = tokens.next_tagint();
        if (id <= 0)
          error->one(FLERR,"Invalid atom ID in BEC file {} at line {}",arg[0],lineno);
        if (!seen.insert(id).second)
          error->one(FLERR,"Duplicate atom ID in BEC file {} at line {}",arg[0],lineno);

        ids.push_back(id);
        for (int j = 0; j < 9; j++) {
          const double component = tokens.next_double();
          if (!std::isfinite(component))
            error->one(FLERR, "Non-finite value in BEC file {} at line {}", arg[0], lineno);
          values.push_back(component);
        }
      } catch (const TokenizerException &e) {
        error->one(FLERR,"Invalid value in BEC file {} at line {}: {}",arg[0],lineno,e.what());
      }
    }

    if (ids.empty()) error->one(FLERR,"BEC file {} contains no atom data",arg[0]);
    if (ids.size() > static_cast<size_t>(MAXSMALLINT / 9))
      error->one(FLERR,"Too many entries in BEC file {}",arg[0]);
  }

  int nrecords = static_cast<int>(ids.size());
  MPI_Bcast(&nrecords,1,MPI_INT,0,world);
  if (comm->me != 0) {
    ids.resize(nrecords);
    values.resize(9*nrecords);
  }
  MPI_Bcast(ids.data(),nrecords,MPI_LMP_TAGINT,0,world);
  MPI_Bcast(values.data(),9*nrecords,MPI_DOUBLE,0,world);

  if (static_cast<bigint>(nrecords) != atom->natoms)
    error->all(FLERR,"BEC file {} contains {} atoms, but the simulation has {} atoms",
               arg[0],nrecords,atom->natoms);

  std::unordered_map<tagint,int> index;
  index.reserve(nrecords);
  for (int i = 0; i < nrecords; i++) index.emplace(ids[i],i);

  atom->sync_host_arrays(TAG_MASK);
  bigint nmissing_local = 0;
  for (int i = 0; i < atom->nlocal; ++i)
    if (index.find(atom->tag[i]) == index.end()) ++nmissing_local;
  bigint nmissing;
  MPI_Allreduce(&nmissing_local, &nmissing, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  if (nmissing)
    error->all(FLERR,"BEC file atom IDs do not match simulation atom IDs");

  // Complete the collective validation before replacing the previous BECs.
  atom->sync_host_arrays(BEC_MASK | BEC_DIPOLE_MASK);
  double **bec = atom->bec;
  double **bec_dipole = atom->bec_dipole;
  double sum_local[9] = {};
  for (int i = 0; i < atom->nlocal; ++i) {
    const double *components = values.data() + 9 * index.at(atom->tag[i]);
    std::copy_n(components, 9, bec[i]);
    std::fill_n(bec_dipole[i], 3, 0.0);
    for (int c = 0; c < 9; ++c) sum_local[c] += components[c];
  }
  atom->modified_host_arrays(BEC_MASK | BEC_DIPOLE_MASK);

  double sum[9];
  MPI_Allreduce(sum_local,sum,9,MPI_DOUBLE,MPI_SUM,world);

  double maxsum = 0.0;
  for (double component : sum) maxsum = std::max(maxsum,std::fabs(component));
  if (comm->me == 0)
    utils::logmesg(lmp,"Read fixed BEC tensors for {} atoms from {} "
                   "(max acoustic-sum residual = {:.8g} e)\n",
                   nrecords,arg[0],maxsum);
}

