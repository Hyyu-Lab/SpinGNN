#ifndef LMP_PT2_LOADER_H
#define LMP_PT2_LOADER_H

#include "deploy_metadata.h"

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef SPINGNN_TORCH_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

namespace LAMMPS_NS {

struct LoadedPT2Model
{
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> model;
  DeployMetadata manifest;
  std::vector<std::string> input_signature;
  std::unordered_map<std::string, size_t> output_indices;
};

inline bool is_pt2_package_path(const std::string &path)
{
  static const std::string suffix = ".pt2";
  return path.size() >= suffix.size() &&
      path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::string pt2_signature_text(const std::vector<std::string> &signature)
{
  std::string text;
  for (const auto &name : signature)
  {
    if (!text.empty()) text += " ";
    text += name;
  }
  return text;
}

inline void validate_pt2_runtime_device(
    Error *error, const DeployMetadata &metadata,
    const std::string &pair_style, const torch::Device &device)
{
  if (!device.is_cuda()) return;
#ifdef SPINGNN_TORCH_CUDA
  const c10::DeviceIndex index = device.has_index() ? device.index() : -1;
  const cudaDeviceProp *properties = at::cuda::getDeviceProperties(index);
  const std::string capability = std::to_string(properties->major) + "." +
      std::to_string(properties->minor);
  const std::string cuda_version = std::to_string(CUDART_VERSION / 1000) + "." +
      std::to_string((CUDART_VERSION % 1000) / 10);
  validate_pt2_cuda_environment(
      error, metadata, pair_style, "cuda", capability, cuda_version);
#else
  error->one(
      FLERR, "Pair style " + pair_style +
                 " selected CUDA but was built without the CUDA metadata validator");
#endif
}

inline void validate_pt2_input_signature(
    Error *error, const DeployMetadata &manifest,
    const std::string &pair_style,
    const std::vector<std::string> &expected_signature)
{
  const std::vector<std::string> signature =
      deploy_metadata_words(manifest, "input_signature");
  const auto has_spin_entry = manifest.find("has_spin");
  if (has_spin_entry == manifest.end() ||
      (has_spin_entry->second != "0" && has_spin_entry->second != "1"))
    error->one(
        FLERR, "Pair style " + pair_style +
                   " requires PT2 metadata has_spin=0 or has_spin=1");

  const bool metadata_has_spin = has_spin_entry->second == "1";
  const bool signature_has_spin =
      std::find(signature.begin(), signature.end(), "spin") != signature.end();
  if (metadata_has_spin != signature_has_spin)
    error->one(
        FLERR, "Pair style " + pair_style + " PT2 has_spin=" +
                   has_spin_entry->second + " does not match input_signature '" +
                   pt2_signature_text(signature) + "'");

  const bool expected_has_spin =
      std::find(expected_signature.begin(), expected_signature.end(), "spin") !=
      expected_signature.end();
  if (metadata_has_spin != expected_has_spin || signature != expected_signature)
    error->one(
        FLERR, "Pair style " + pair_style + " PT2 has_spin=" +
                   has_spin_entry->second + " requires input_signature: " +
                   pt2_signature_text(expected_signature));
}

inline LoadedPT2Model load_verified_pt2_model(
    Error *error, const std::string &model_path,
    const std::string &pair_style, const torch::Device &device,
    const std::vector<std::string> &expected_input_signature)
{
  const DeployMetadata sidecar =
      load_deploy_metadata_sidecar(error, model_path);
  auto private_copy = create_verified_pt2_private_copy(
      error, sidecar, pair_style, model_path);
  DeployMetadata manifest =
      load_pt2_package_manifest(error, private_copy.loader_path());
  validate_pt2_manifest_matches_sidecar(
      error, sidecar, manifest, pair_style);
  validate_pt2_target_device(
      error, manifest, pair_style, device.is_cuda() ? "cuda" : "cpu");
  validate_pt2_runtime_device(error, manifest, pair_style, device);
  const std::vector<std::string> output_signature =
      validate_pt2_output_signature(error, manifest, pair_style);
  validate_pt2_input_signature(
      error, manifest, pair_style, expected_input_signature);

  LoadedPT2Model loaded;
  const c10::DeviceIndex device_index = device.has_index() ? device.index() : -1;
  try
  {
    loaded.model =
        std::make_unique<torch::inductor::AOTIModelPackageLoader>(
            private_copy.loader_path(), "model", true, 1, device_index);
  }
  catch (const c10::Error &exception)
  {
    // Another rank may already be evaluating its model. Do not enter the
    // collective Error::all barrier for this local loader failure.
    const std::string cleanup_failure = private_copy.cleanup();
    std::string message = "Pair style " + pair_style +
        " PT2 loader construction failed: " + exception.what();
    if (!cleanup_failure.empty())
      message += "; private copy cleanup failed: " + cleanup_failure;
    error->one(FLERR, message);
  }
  catch (const std::exception &exception)
  {
    const std::string cleanup_failure = private_copy.cleanup();
    std::string message = "Pair style " + pair_style +
        " PT2 loader construction failed: " + exception.what();
    if (!cleanup_failure.empty())
      message += "; private copy cleanup failed: " + cleanup_failure;
    error->one(FLERR, message);
  }

  const std::string cleanup_failure = private_copy.cleanup();
  if (!cleanup_failure.empty())
  {
    loaded.model.reset();
    error->one(
        FLERR, "Pair style " + pair_style +
                   " cannot release private PT2 package copy: " + cleanup_failure);
  }
  loaded.manifest = std::move(manifest);
  loaded.input_signature =
      deploy_metadata_words(loaded.manifest, "input_signature");
  for (size_t index = 0; index < output_signature.size(); ++index)
    loaded.output_indices[output_signature[index]] = index;
  return loaded;
}

inline torch::Tensor checked_pt2_output(
    Error *error, const std::string &pair_style, const std::string &name,
    const std::vector<at::Tensor> &outputs,
    const std::unordered_map<std::string, size_t> &indices,
    const std::vector<int64_t> &expected_shape)
{
  const auto entry = indices.find(name);
  if (entry == indices.end() || entry->second >= outputs.size())
    error->one(
        FLERR, "Pair style " + pair_style +
                   " PT2 output_signature has no tensor for '" + name + "'");
  const torch::Tensor &tensor = outputs[entry->second];
  if (tensor.scalar_type() != at::kFloat)
    error->one(
        FLERR, "Pair style " + pair_style + " PT2 output '" + name +
                   "' must have dtype float32");
  if (tensor.dim() != static_cast<int64_t>(expected_shape.size()))
    error->one(
        FLERR, "Pair style " + pair_style + " PT2 output '" + name +
                   "' has rank " + std::to_string(tensor.dim()) + ", expected " +
                   std::to_string(expected_shape.size()));
  for (size_t axis = 0; axis < expected_shape.size(); ++axis)
    if (expected_shape[axis] >= 0 && tensor.size(axis) != expected_shape[axis])
      error->one(
          FLERR, "Pair style " + pair_style + " PT2 output '" + name +
                     "' has invalid size " + std::to_string(tensor.size(axis)) +
                     " on axis " + std::to_string(axis) + ", expected " +
                     std::to_string(expected_shape[axis]));
  return tensor;
}

} // namespace LAMMPS_NS

#endif
