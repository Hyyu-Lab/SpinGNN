#ifndef LMP_TORCH_DETERMINISM_H
#define LMP_TORCH_DETERMINISM_H

#include "error.h"

#include <ATen/Context.h>

#include <cstdlib>
#include <cstring>

namespace LAMMPS_NS {

inline bool configure_torch_determinism(Error *error)
{
  const char *requested = std::getenv("SPINGNN_DETERMINISTIC");
  if (requested == nullptr || std::strcmp(requested, "0") == 0)
    return false;
  if (std::strcmp(requested, "1") != 0)
    error->all(FLERR, "SPINGNN_DETERMINISTIC must be 0 or 1");

  const char *workspace = std::getenv("CUBLAS_WORKSPACE_CONFIG");
  if (workspace == nullptr ||
      (std::strcmp(workspace, ":4096:8") != 0 && std::strcmp(workspace, ":16:8") != 0))
    error->all(
        FLERR,
        "SPINGNN_DETERMINISTIC=1 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8");

  at::globalContext().setDeterministicAlgorithms(true, false);
  at::globalContext().setDeterministicCuDNN(true);
  at::globalContext().setDeterministicFillUninitializedMemory(true);
  return true;
}

} // namespace LAMMPS_NS

#endif
