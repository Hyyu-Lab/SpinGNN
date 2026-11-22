#ifndef LMP_TORCH_DEVICE_H
#define LMP_TORCH_DEVICE_H

#include "error.h"

#include <mpi.h>
#include <torch/torch.h>

#include <exception>
#include <string>

namespace LAMMPS_NS {

inline torch::Device spingnn_torch_device(MPI_Comm world, Error *error)
{
  int process_count = 1;
  MPI_Comm_size(world, &process_count);
  int device_count = 0;
  int local_status[2] = {0, 0};
  std::string discovery_error;
  try
  {
    if (torch::cuda::is_available()) device_count = torch::cuda::device_count();
    local_status[0] = device_count > 0 ? 1 : 0;
  }
  catch (const std::exception &exception)
  {
    local_status[1] = 1;
    discovery_error = exception.what();
  }
  catch (...)
  {
    local_status[1] = 1;
    discovery_error = "unknown exception";
  }

  // Every rank must participate before CPU fallback or the shared-world
  // split. Even a local CUDA discovery exception must reach this agreement.
  int global_status[2] = {0, 0};
  MPI_Allreduce(local_status, global_status, 2, MPI_INT, MPI_SUM, world);
  if (global_status[1] != 0)
  {
    if (local_status[1] != 0)
      error->one(FLERR, "CUDA device discovery failed: " + discovery_error);
    error->all(FLERR, "CUDA device discovery failed on another MPI rank");
  }
  if (global_status[0] != 0 && global_status[0] != process_count)
    error->all(FLERR, "Inconsistent CUDA availability across MPI ranks; "
                      "make CUDA visible to every rank or disable it for every rank");
  if (global_status[0] == 0) return torch::Device(torch::kCPU);

  int device_index = -1;
  if (process_count > 1)
  {
    MPI_Comm shared_world;
    MPI_Comm_split_type(
        world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared_world);
    MPI_Comm_rank(shared_world, &device_index);
    MPI_Comm_free(&shared_world);
    device_index %= device_count;
  }
  return torch::Device(torch::kCUDA, device_index);
}

} // namespace LAMMPS_NS

#endif
