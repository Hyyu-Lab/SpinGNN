/* -*- c++ -*- ----------------------------------------------------------
   Persistent pinned host staging for SpinGNN CUDA pair styles.
------------------------------------------------------------------------- */

#ifndef LMP_SPINGNN_HOST_STAGING_H
#define LMP_SPINGNN_HOST_STAGING_H

#include "error.h"

#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef SPINGNN_TORCH_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace LAMMPS_NS {

struct SpingnnStagingGeneration {
  long long neighbor_ncalls = -1;
  long long neighbor_lastcall = -1;
  int nlocal = -1;
  int nghost = -1;
  int nmax = -1;

  bool operator==(const SpingnnStagingGeneration &other) const
  {
    return neighbor_ncalls == other.neighbor_ncalls &&
        neighbor_lastcall == other.neighbor_lastcall && nlocal == other.nlocal &&
        nghost == other.nghost && nmax == other.nmax;
  }
};

struct SpingnnStagingUse {
  bool generation_reused = false;
  bool allocation_grew = false;

  void note_growth(bool grew) { allocation_grew = allocation_grew || grew; }
  int cache_hit() const { return generation_reused && !allocation_grew ? 1 : 0; }
  int cache_miss() const { return cache_hit() ? 0 : 1; }
};

struct SpingnnStagedInputs {
  torch::Tensor pos;
  torch::Tensor spin;
  torch::Tensor edges;
  torch::Tensor types;
};

class SpingnnHostStaging {
 public:
  SpingnnHostStaging()
  {
    const char *value = std::getenv("SPINGNN_HOST_STAGING");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0)
      enabled_ = false;
    else if (std::strcmp(value, "1") == 0)
      enabled_ = true;
    else
      invalid_value_ = value;
  }

  void validate(Error *error) const
  {
    if (!invalid_value_.empty())
      error->all(
          FLERR, "SPINGNN_HOST_STAGING must be exactly 0 or 1; got '" +
                     invalid_value_ + "'");
  }

  bool active(const torch::Device &device) const
  {
#ifdef SPINGNN_TORCH_CUDA
    return enabled_ && device.is_cuda();
#else
    (void) device;
    return false;
#endif
  }

  void reset_generation() { generation_valid_ = false; }

  bool prepare_graph_scratch(int atom_nmax, int nlocal)
  {
    const size_t required = static_cast<size_t>(std::max(atom_nmax, nlocal));
    const bool grew = neighbor_counts_.size() < required;
    if (grew)
    {
      neighbor_counts_.resize(required);
      neighbor_offsets_.resize(required);
    }
    std::fill(neighbor_counts_.begin(), neighbor_counts_.begin() + nlocal, 0);
    std::fill(neighbor_offsets_.begin(), neighbor_offsets_.begin() + nlocal, 0);
    return grew;
  }

  int *neighbor_counts() { return neighbor_counts_.data(); }
  int *neighbor_offsets() { return neighbor_offsets_.data(); }

  SpingnnStagingUse prepare_inputs(
      const torch::Device &device, int atom_nmax, int n_atoms, int n_edges,
      bool with_spin, const SpingnnStagingGeneration &generation)
  {
    // This generation only classifies backing-storage reuse. Every compute
    // still repacks and uploads all input values; cross-compute geometry reuse
    // belongs to the separate frozen-SIB cache contract.
    SpingnnStagingUse use;
    use.generation_reused = generation_valid_ && generation == generation_;

    // Device-side allocator state must match the non-staged `.to(device)`
    // path at model entry: on sm_120+cu128 the persistent over-capacity
    // device buffers used to live here shifted the caching-allocator layout
    // and changed the autograd backward reduction schedule (energy bitwise
    // equal, forces off by a few ULPs). Only pinned host buffers persist
    // now; device tensors are allocated fresh per upload.
    device_ = device;
    const int64_t required_atoms = std::max<int64_t>(atom_nmax, n_atoms);
    if (!host_pos_.defined() || atom_capacity_ < required_atoms || with_spin_ != with_spin)
    {
      atom_capacity_ = required_atoms;
      with_spin_ = with_spin;
      const auto host_float = torch::TensorOptions()
                                  .dtype(torch::kFloat32)
                                  .device(torch::kCPU)
                                  .pinned_memory(true);
      const auto host_long = torch::TensorOptions()
                                 .dtype(torch::kInt64)
                                 .device(torch::kCPU)
                                 .pinned_memory(true);
      host_pos_ = torch::empty({atom_capacity_, 3}, host_float);
      host_types_ = torch::empty({atom_capacity_}, host_long);
      if (with_spin_)
      {
        host_spin_ = torch::empty({atom_capacity_, 3}, host_float);
      }
      else
      {
        host_spin_ = torch::Tensor();
      }
      use.note_growth(true);
    }

    const int64_t required_edges = std::max<int64_t>(n_edges, 1);
    if (!host_edges_.defined() || edge_capacity_ < required_edges)
    {
      edge_capacity_ = required_edges;
      const auto host_long = torch::TensorOptions()
                                 .dtype(torch::kInt64)
                                 .device(torch::kCPU)
                                 .pinned_memory(true);
      host_edges_ = torch::empty({2 * edge_capacity_}, host_long);
      use.note_growth(true);
    }

    generation_ = generation;
    generation_valid_ = true;
    return use;
  }

  torch::Tensor host_pos(int64_t n_atoms) const
  {
    return host_pos_.narrow(0, 0, n_atoms);
  }

  torch::Tensor host_spin(int64_t n_atoms) const
  {
    return host_spin_.narrow(0, 0, n_atoms);
  }

  torch::Tensor host_edges(int64_t n_edges) const
  {
    return host_edges_.narrow(0, 0, 2 * n_edges).view({2, n_edges});
  }

  torch::Tensor host_types(int64_t n_atoms) const
  {
    return host_types_.narrow(0, 0, n_atoms);
  }

  SpingnnStagedInputs upload_inputs(int64_t n_atoms, int64_t n_edges, bool with_spin)
  {
    constexpr bool non_blocking = true;
    at::NoGradGuard no_grad;
    // Fresh exact-size device tensors per call, allocated and filled in the
    // same order (pos, spin, edges, types) as the non-staged `.to(device)`
    // path, so the caching-allocator state at model entry is identical in
    // both paths. Pinned host buffers + non_blocking copies keep the async
    // H2D benefit; caching-allocator allocations are ~free.
    const auto device_float =
        torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    const auto device_long =
        torch::TensorOptions().dtype(torch::kInt64).device(device_);
    SpingnnStagedInputs inputs;
    inputs.pos = torch::empty({n_atoms, 3}, device_float);
    inputs.pos.copy_(host_pos(n_atoms), non_blocking);
    if (with_spin)
    {
      inputs.spin = torch::empty({n_atoms, 3}, device_float);
      inputs.spin.copy_(host_spin(n_atoms), non_blocking);
    }
    inputs.edges = torch::empty({2, n_edges}, device_long);
    inputs.edges.copy_(host_edges(n_edges), non_blocking);
    inputs.types = torch::empty({n_atoms}, device_long);
    inputs.types.copy_(host_types(n_atoms), non_blocking);
    return inputs;
  }

  torch::Tensor upload_spin(int64_t n_atoms)
  {
    constexpr bool non_blocking = true;
    at::NoGradGuard no_grad;
    torch::Tensor spin = torch::empty(
        {n_atoms, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    spin.copy_(host_spin(n_atoms), non_blocking);
    return spin;
  }

  bool prepare_output(const torch::Device &device, int64_t values)
  {
    if (host_output_.defined() && output_capacity_ >= values) return false;
    output_capacity_ = std::max<int64_t>(values, 1);
    host_output_ = torch::empty(
        {output_capacity_},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU)
            .pinned_memory(true));
    device_output_ = torch::empty(
        {output_capacity_}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    return true;
  }

  torch::Tensor pack_output(
      Error *error, const std::string &pair_style, const std::string &name,
      const torch::Tensor &source, int64_t offset)
  {
    if (source.scalar_type() != at::kFloat)
      error->all(
          FLERR, "Pair style " + pair_style + " output '" + name +
                     "' must have dtype float32 for host staging");
    if (source.device() != device_output_.device())
      error->all(
          FLERR, "Pair style " + pair_style + " output '" + name +
                     "' is on the wrong device for host staging");
    if (offset < 0 || offset + source.numel() > output_capacity_)
      error->all(
          FLERR, "Pair style " + pair_style + " output staging overflow for '" + name +
                     "'");
    at::NoGradGuard no_grad;
    torch::Tensor device_view =
        device_output_.narrow(0, offset, source.numel()).view(source.sizes());
    device_view.copy_(source.detach());
    return host_output_.narrow(0, offset, source.numel()).view(source.sizes());
  }

  void download_output(const torch::Device &device, int64_t values)
  {
    constexpr bool non_blocking = true;
    at::NoGradGuard no_grad;
    host_output_.narrow(0, 0, values).copy_(
        device_output_.narrow(0, 0, values), non_blocking);
#ifdef SPINGNN_TORCH_CUDA
    const c10::DeviceIndex index = device.has_index() ? device.index() : -1;
    c10::cuda::getCurrentCUDAStream(index).synchronize();
#else
    (void) device;
#endif
  }

 private:
  bool enabled_ = false;
  std::string invalid_value_;
  bool generation_valid_ = false;
  SpingnnStagingGeneration generation_;
  bool with_spin_ = false;
  int64_t atom_capacity_ = 0;
  int64_t edge_capacity_ = 0;
  int64_t output_capacity_ = 0;
  torch::Tensor host_pos_;
  torch::Tensor host_spin_;
  torch::Tensor host_edges_;
  torch::Tensor host_types_;
  torch::Device device_{torch::kCPU};
  torch::Tensor host_output_;
  torch::Tensor device_output_;
  std::vector<int> neighbor_counts_;
  std::vector<int> neighbor_offsets_;
};

inline long long spingnn_tensor_bytes(const torch::Tensor &tensor)
{
  return tensor.defined()
      ? static_cast<long long>(tensor.numel()) * static_cast<long long>(tensor.element_size())
      : 0LL;
}

} // namespace LAMMPS_NS

#endif
