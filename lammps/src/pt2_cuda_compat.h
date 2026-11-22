#ifndef LMP_PT2_CUDA_COMPAT_H
#define LMP_PT2_CUDA_COMPAT_H

#include <limits>
#include <string>
#include <utility>

namespace LAMMPS_NS {

inline bool parse_pt2_compute_capability(
    const std::string &text, std::pair<int, int> &capability)
{
  const size_t separator = text.find('.');
  if (separator == std::string::npos || separator == 0 ||
      separator + 1 == text.size() || text.find('.', separator + 1) != std::string::npos)
    return false;
  int values[2] = {0, 0};
  const std::string fields[2] = {
      text.substr(0, separator), text.substr(separator + 1)};
  for (int index = 0; index < 2; ++index)
  {
    for (char character : fields[index])
    {
      if (character < '0' || character > '9') return false;
      const int digit = character - '0';
      if (values[index] > (std::numeric_limits<int>::max() - digit) / 10)
        return false;
      values[index] = values[index] * 10 + digit;
    }
  }
  if (values[0] <= 0) return false;
  capability = {values[0], values[1]};
  return true;
}

inline bool pt2_cubin_capability_compatible(
    const std::string &package_capability,
    const std::string &runtime_capability)
{
  std::pair<int, int> package;
  std::pair<int, int> runtime;
  if (!parse_pt2_compute_capability(package_capability, package) ||
      !parse_pt2_compute_capability(runtime_capability, runtime))
    return false;
  return package.first == runtime.first && runtime.second >= package.second;
}

} // namespace LAMMPS_NS

#endif
