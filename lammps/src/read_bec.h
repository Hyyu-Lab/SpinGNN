/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
------------------------------------------------------------------------- */

#ifdef COMMAND_CLASS
// clang-format off
CommandStyle(read_bec,ReadBEC);
// clang-format on
#else

#ifndef LMP_READ_BEC_H
#define LMP_READ_BEC_H

#include "command.h"

namespace LAMMPS_NS {

class ReadBEC : public Command {
 public:
  ReadBEC(class LAMMPS *);
  void command(int, char **);
};

}    // namespace LAMMPS_NS

#endif
#endif

