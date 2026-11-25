// SPDX-License-Identifier: GPL-2.0-only
#ifndef LMP_DREAM_BEC_H
#define LMP_DREAM_BEC_H

#include "atom.h"

#include <algorithm>
#include <iterator>
#include <string_view>

namespace LAMMPS_NS::DREAM {

inline constexpr std::string_view names[] = {
    "becdipolex", "becdipoley", "becdipolez", "becxx", "becxy", "becxz",
    "becyx", "becyy", "becyz", "beczx", "beczy", "beczz"};

inline int field_index(std::string_view name)
{
  const auto begin = std::begin(names);
  const auto end = std::end(names);
  const auto entry = std::find(begin, end, name);
  return entry == end ? -1 : static_cast<int>(entry - begin);
}

inline double value(const Atom *atom, int i, int field)
{
  return field < 3 ? atom->bec_dipole[i][field] : atom->bec[i][field - 3];
}

inline void add_displacement(Atom *atom, int i, double dx, double dy, double dz)
{
  const double *z = atom->bec[i];
  double *dipole = atom->bec_dipole[i];
  for (int c = 0; c < 3; ++c)
    dipole[c] += z[3 * c] * dx + z[3 * c + 1] * dy + z[3 * c + 2] * dz;
}

} // namespace LAMMPS_NS::DREAM

#endif
