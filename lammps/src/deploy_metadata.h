#ifndef LMP_DEPLOY_METADATA_H
#define LMP_DEPLOY_METADATA_H

#include "error.h"
#include "pt2_cuda_compat.h"

#include <caffe2/serialize/inline_container.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace LAMMPS_NS {

using DeployMetadata = std::map<std::string, std::string>;

// Artifacts and CUDA properties are rank-local even when their path names
// match. These helpers must abort through Error::one: Error::all waits for
// ranks that may have successfully loaded their own package or device.

class DeploySHA256 {
 public:
  DeploySHA256() : data_length_(0), bit_length_(0), state_({
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U}) {}

  void update(const unsigned char *data, size_t length)
  {
    for (size_t index = 0; index < length; ++index)
    {
      data_[data_length_++] = data[index];
      if (data_length_ == data_.size())
      {
        transform();
        bit_length_ += 512;
        data_length_ = 0;
      }
    }
  }

  std::string final()
  {
    const uint64_t total_bits = bit_length_ + data_length_ * 8;
    size_t index = data_length_;
    data_[index++] = 0x80;
    if (index > 56)
    {
      while (index < 64) data_[index++] = 0;
      transform();
      index = 0;
    }
    while (index < 56) data_[index++] = 0;
    for (size_t byte = 0; byte < 8; ++byte)
      data_[63 - byte] = static_cast<unsigned char>(total_bits >> (byte * 8));
    transform();

    std::ostringstream digest;
    digest << std::hex << std::setfill('0');
    for (uint32_t word : state_) digest << std::setw(8) << word;
    return digest.str();
  }

 private:
  static uint32_t rotate_right(uint32_t value, uint32_t count)
  {
    return (value >> count) | (value << (32 - count));
  }

  void transform()
  {
    static constexpr std::array<uint32_t, 64> constants = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<uint32_t, 64> words{};
    for (size_t index = 0; index < 16; ++index)
    {
      const size_t offset = index * 4;
      words[index] =
          (static_cast<uint32_t>(data_[offset]) << 24) |
          (static_cast<uint32_t>(data_[offset + 1]) << 16) |
          (static_cast<uint32_t>(data_[offset + 2]) << 8) |
          static_cast<uint32_t>(data_[offset + 3]);
    }
    for (size_t index = 16; index < 64; ++index)
    {
      const uint32_t s0 = rotate_right(words[index - 15], 7) ^
          rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3);
      const uint32_t s1 = rotate_right(words[index - 2], 17) ^
          rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];
    for (size_t index = 0; index < 64; ++index)
    {
      const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
          rotate_right(e, 25);
      const uint32_t choice = (e & f) ^ (~e & g);
      const uint32_t temporary1 = h + sum1 + choice + constants[index] + words[index];
      const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
          rotate_right(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<unsigned char, 64> data_{};
  size_t data_length_;
  uint64_t bit_length_;
  std::array<uint32_t, 8> state_;
};

inline std::vector<std::string> deploy_metadata_words(
    const DeployMetadata &metadata, const std::string &key)
{
  const auto entry = metadata.find(key);
  if (entry == metadata.end()) return {};
  std::stringstream stream(entry->second);
  std::vector<std::string> words;
  std::string word;
  while (stream >> word) words.push_back(word);
  return words;
}

inline std::string deploy_file_sha256(
    Error *error, const std::string &model_path)
{
  std::ifstream stream(model_path, std::ios::binary);
  if (!stream.is_open())
    error->one(FLERR, "Cannot open PT2 package for SHA-256 validation: " + model_path);
  DeploySHA256 digest;
  std::array<char, 64 * 1024> buffer{};
  while (stream)
  {
    stream.read(buffer.data(), buffer.size());
    const std::streamsize count = stream.gcount();
    if (count > 0)
      digest.update(
          reinterpret_cast<const unsigned char *>(buffer.data()),
          static_cast<size_t>(count));
  }
  if (!stream.eof())
    error->one(FLERR, "Failed while reading PT2 package for SHA-256 validation: " + model_path);
  return digest.final();
}

inline void validate_pt2_package_sha256(
    Error *error, const DeployMetadata &metadata,
    const std::string &pair_style, const std::string &model_path)
{
  const auto entry = metadata.find("package_sha256");
  if (entry == metadata.end() || entry->second.size() != 64 ||
      entry->second.find_first_not_of("0123456789abcdef") != std::string::npos)
    error->one(
        FLERR, "Pair style " + pair_style +
                   " requires package_sha256 as 64 lowercase hex digits");
  const std::string actual = deploy_file_sha256(error, model_path);
  if (actual != entry->second)
    error->one(
        FLERR, "Pair style " + pair_style +
                   " PT2 package SHA-256 does not match its sidecar: expected " +
                   entry->second + ", actual " + actual);
}

inline std::vector<std::string> validate_pt2_output_signature(
    Error *error, const DeployMetadata &metadata, const std::string &pair_style)
{
  const auto signature = deploy_metadata_words(metadata, "output_signature");
  const auto provided = deploy_metadata_words(metadata, "provided_outputs");
  if (signature.empty())
    error->one(FLERR, "Pair style " + pair_style + " requires a nonempty PT2 output_signature");
  if (provided != signature)
    error->one(
        FLERR, "Pair style " + pair_style +
                   " requires provided_outputs to exactly match output_signature");
  std::unordered_set<std::string> unique;
  for (const auto &name : signature)
    if (!unique.insert(name).second)
      error->one(
          FLERR, "Pair style " + pair_style +
                     " PT2 output_signature contains duplicate output name '" + name + "'");
  const std::vector<std::string> canonical_order = {
      "total_energy", "forces", "spin_forces", "atomic_energy", "virial",
      "atom_virial", "born_effective_charges", "total_born_effective_charges"};
  size_t canonical_index = 0;
  for (const auto &name : signature)
  {
    while (canonical_index < canonical_order.size() &&
           canonical_order[canonical_index] != name)
      ++canonical_index;
    if (canonical_index == canonical_order.size())
      error->one(
          FLERR, "Pair style " + pair_style +
                     " PT2 output_signature must follow the canonical ABI order");
    ++canonical_index;
  }
  return signature;
}

class VerifiedPT2PrivateCopy
{
 public:
  VerifiedPT2PrivateCopy(
      std::string named_path, std::string loader_path, int descriptor) :
      named_path_(std::move(named_path)), loader_path_(std::move(loader_path)),
      descriptor_(descriptor)
  {}

  VerifiedPT2PrivateCopy(const VerifiedPT2PrivateCopy &) = delete;
  VerifiedPT2PrivateCopy &operator=(const VerifiedPT2PrivateCopy &) = delete;

  ~VerifiedPT2PrivateCopy() { cleanup(); }

  const std::string &loader_path() const { return loader_path_; }

  std::string cleanup()
  {
    std::string failure;
    if (!named_path_.empty() && std::remove(named_path_.c_str()) != 0 && errno != ENOENT)
      failure = std::strerror(errno);
    named_path_.clear();
    loader_path_.clear();
    if (descriptor_ >= 0 && close(descriptor_) != 0 && failure.empty())
      failure = std::strerror(errno);
    descriptor_ = -1;
    return failure;
  }

 private:
  std::string named_path_;
  std::string loader_path_;
  int descriptor_;
};

inline VerifiedPT2PrivateCopy create_verified_pt2_private_copy(
    Error *error, const DeployMetadata &metadata,
    const std::string &pair_style, const std::string &model_path)
{
  const auto expected = metadata.find("package_sha256");
  if (expected == metadata.end() || expected->second.size() != 64 ||
      expected->second.find_first_not_of("0123456789abcdef") != std::string::npos)
    error->one(
        FLERR, "Pair style " + pair_style +
                   " requires package_sha256 as 64 lowercase hex digits");

  std::ifstream source(model_path, std::ios::binary);
  if (!source.is_open())
    error->one(FLERR, "Cannot open PT2 package for verified copy: " + model_path);
  const char *temporary_environment = std::getenv("TMPDIR");
  std::string temporary_directory =
      temporary_environment && temporary_environment[0] ? temporary_environment : "/tmp";
  if (temporary_directory.back() != '/') temporary_directory.push_back('/');
  int descriptor = -1;
  std::string private_path;
  bool use_proc_path = false;
#ifdef O_TMPFILE
  // An unnamed O_TMPFILE descriptor never appears in TMPDIR, so a rank
  // killed mid-load by another rank's abort cannot leak a spingnn-pt2-*
  // file; the kernel reclaims the inode when the descriptor dies.
  descriptor = open(temporary_directory.c_str(), O_TMPFILE | O_RDWR, 0600);
  if (descriptor >= 0)
  {
    const std::string proc_candidate =
        "/proc/self/fd/" + std::to_string(descriptor);
    if (access(proc_candidate.c_str(), R_OK) == 0)
      use_proc_path = true;
    else
    {
      close(descriptor);
      descriptor = -1;
    }
  }
#endif
  if (descriptor < 0)
  {
    std::string path_template = temporary_directory + "spingnn-pt2-XXXXXX";
    std::vector<char> writable_template(path_template.begin(), path_template.end());
    writable_template.push_back('\0');
    descriptor = mkstemp(writable_template.data());
    if (descriptor < 0)
      error->one(
          FLERR, "Cannot create private PT2 package copy: " +
                     std::string(std::strerror(errno)));
    private_path = std::string(writable_template.data());
    const std::string proc_candidate =
        "/proc/self/fd/" + std::to_string(descriptor);
    use_proc_path = access(proc_candidate.c_str(), R_OK) == 0;
    if (use_proc_path && std::remove(private_path.c_str()) != 0)
    {
      const std::string failure = std::strerror(errno);
      close(descriptor);
      error->one(
          FLERR, "Cannot unlink private PT2 package copy: " + failure);
    }
  }
  const std::string proc_path = "/proc/self/fd/" + std::to_string(descriptor);

  DeploySHA256 digest;
  std::array<char, 64 * 1024> buffer{};
  std::string failure;
  while (source && failure.empty())
  {
    source.read(buffer.data(), buffer.size());
    const std::streamsize count = source.gcount();
    if (count <= 0) continue;
    digest.update(
        reinterpret_cast<const unsigned char *>(buffer.data()),
        static_cast<size_t>(count));
    size_t offset = 0;
    while (offset < static_cast<size_t>(count))
    {
      const ssize_t written =
          write(descriptor, buffer.data() + offset, static_cast<size_t>(count) - offset);
      if (written <= 0)
      {
        failure = written == 0 ? "write returned zero bytes" : std::strerror(errno);
        break;
      }
      offset += static_cast<size_t>(written);
    }
  }
  if (!source.eof() && failure.empty()) failure = "source read failed";
  if (lseek(descriptor, 0, SEEK_SET) < 0 && failure.empty())
    failure = std::strerror(errno);
  if (!failure.empty())
  {
    close(descriptor);
    std::remove(private_path.c_str());
    error->one(
        FLERR, "Failed to create private PT2 package copy: " + failure);
  }
  const std::string actual = digest.final();
  if (actual != expected->second)
  {
    close(descriptor);
    std::remove(private_path.c_str());
    error->one(
        FLERR, "Pair style " + pair_style +
                   " PT2 package SHA-256 does not match its sidecar: expected " +
                   expected->second + ", actual " + actual);
  }
  if (use_proc_path)
    return VerifiedPT2PrivateCopy("", proc_path, descriptor);
  return VerifiedPT2PrivateCopy(private_path, private_path, descriptor);
}

inline std::string unescape_deploy_metadata_value(
    Error *error, const std::string &value, const std::string &sidecar_path,
    size_t line_number)
{
  std::string decoded;
  decoded.reserve(value.size());
  for (size_t index = 0; index < value.size(); ++index)
  {
    if (value[index] != '\\')
    {
      decoded.push_back(value[index]);
      continue;
    }
    if (++index == value.size())
      error->one(FLERR, "Malformed PT2 metadata sidecar " + sidecar_path +
                            " at line " + std::to_string(line_number) +
                            ": incomplete escape");
    if (value[index] == 'n')
      decoded.push_back('\n');
    else if (value[index] == 't')
      decoded.push_back('\t');
    else
      decoded.push_back(value[index]);
  }
  return decoded;
}

inline DeployMetadata load_deploy_metadata_stream(
    Error *error, std::istream &stream, const std::string &description)
{
  DeployMetadata metadata;
  std::string line;
  size_t line_number = 0;
  while (std::getline(stream, line))
  {
    ++line_number;
    const size_t separator = line.find('\t');
    if (separator == std::string::npos || separator == 0)
      error->one(FLERR, "Malformed " + description +
                            " at line " + std::to_string(line_number) +
                            ": expected key<TAB>value");
    const std::string key = line.substr(0, separator);
    if (metadata.count(key) != 0)
      error->one(FLERR, "Malformed " + description +
                            " at line " + std::to_string(line_number) +
                            ": duplicate key '" + key + "'");
    metadata[key] = unescape_deploy_metadata_value(
        error, line.substr(separator + 1), description, line_number);
  }
  return metadata;
}

inline DeployMetadata load_deploy_metadata_sidecar(
    Error *error, const std::string &model_path)
{
  const std::string sidecar_path = model_path + ".meta";
  std::ifstream stream(sidecar_path);
  if (!stream.is_open())
    error->one(FLERR, "PT2 deploy metadata sidecar is missing: " + sidecar_path);
  return load_deploy_metadata_stream(
      error, stream, "PT2 metadata sidecar " + sidecar_path);
}

inline DeployMetadata load_pt2_package_manifest(
    Error *error, const std::string &package_path)
{
  try
  {
    caffe2::serialize::PyTorchStreamReader reader(package_path);
    const std::string record = "spingnn_deploy.meta";
    if (!reader.hasRecord(record))
      error->one(
          FLERR, "PT2 package has no embedded SpinGNN ABI manifest: " + package_path);
    auto record_data = reader.getRecord(record);
    const auto &data = std::get<0>(record_data);
    const size_t size = std::get<1>(record_data);
    std::string encoded(static_cast<const char *>(data.get()), size);
    std::istringstream stream(encoded);
    return load_deploy_metadata_stream(
        error, stream, "embedded PT2 ABI manifest");
  }
  catch (const std::exception &exception)
  {
    error->one(
        FLERR, "Cannot read embedded PT2 ABI manifest from " + package_path +
                   ": " + exception.what());
  }
  return {};
}

inline void validate_pt2_manifest_matches_sidecar(
    Error *error, const DeployMetadata &sidecar, const DeployMetadata &manifest,
    const std::string &pair_style)
{
  std::vector<std::string> protected_keys = {
      "schema_version", "input_signature", "output_signature", "provided_outputs",
      "required_inputs", "default_dtype", "target_device", "rescale_layers",
      "r_max", "n_species", "type_names", "allow_tf32", "energy_units",
      "length_units", "has_spin", "has_bec", "edge_prune",
      "package_payload_sha256"};
  // Older packages may omit the depth, but a newly declared MPI contract
  // must be bound to the same immutable package as its other ABI metadata.
  if (sidecar.count("nequip_num_interactions") || manifest.count("nequip_num_interactions"))
    protected_keys.push_back("nequip_num_interactions");
  const auto target = manifest.find("target_device");
  if (target != manifest.end() && target->second == "cuda")
  {
    protected_keys.push_back("cuda_compute_capability");
    protected_keys.push_back("cuda_code_kind");
    protected_keys.push_back("torch_cuda_version");
  }
  for (const auto &key : protected_keys)
  {
    const auto sidecar_entry = sidecar.find(key);
    const auto manifest_entry = manifest.find(key);
    if (sidecar_entry == sidecar.end() || manifest_entry == manifest.end() ||
        sidecar_entry->second != manifest_entry->second)
      error->one(
          FLERR, "Pair style " + pair_style +
                     " PT2 sidecar does not match embedded ABI manifest key '" + key + "'");
  }
}

inline void validate_pt2_target_device(
    Error *error, const DeployMetadata &metadata,
    const std::string &pair_style, const std::string &running_device)
{
  const auto target_entry = metadata.find("target_device");
  if (target_entry == metadata.end() || target_entry->second.empty())
    error->one(
        FLERR, "Pair style " + pair_style +
                   " requires PT2 metadata key target_device");

  const std::string &target_device = target_entry->second;
  if (target_device != "cpu" && target_device != "cuda")
    error->one(
        FLERR, "Pair style " + pair_style +
                   " received unsupported PT2 target_device '" +
                   target_device + "'; expected cpu or cuda");

  if (target_device != running_device)
    error->one(
        FLERR, "Pair style " + pair_style + " PT2 package targets " +
                   target_device + " but this pair runs on " + running_device +
                   "; rebuild with `nequip-deploy build-pt2 --device " +
                   running_device + "`");
}

inline void validate_pt2_cuda_environment(
    Error *error, const DeployMetadata &metadata,
    const std::string &pair_style, const std::string &running_device,
    const std::string &running_compute_capability,
    const std::string &running_cuda_version)
{
  if (running_device != "cuda") return;
  const auto capability = metadata.find("cuda_compute_capability");
  const auto code_kind = metadata.find("cuda_code_kind");
  const auto cuda_version = metadata.find("torch_cuda_version");
  if (capability == metadata.end() || capability->second.empty())
    error->one(
        FLERR, "Pair style " + pair_style +
                   " requires CUDA PT2 metadata key cuda_compute_capability");
  if (cuda_version == metadata.end() || cuda_version->second.empty())
    error->one(
        FLERR, "Pair style " + pair_style +
                   " requires CUDA PT2 metadata key torch_cuda_version");
  if (code_kind == metadata.end() || code_kind->second != "cubin")
    error->one(
        FLERR, "Pair style " + pair_style +
                   " requires CUDA PT2 metadata cuda_code_kind=cubin");
  if (!pt2_cubin_capability_compatible(
          capability->second, running_compute_capability))
    error->one(
        FLERR, "Pair style " + pair_style + " PT2 package targets CUDA compute capability " +
                   capability->second + " but this device is " +
                   running_compute_capability + "; export a new package on the target architecture");
  if (cuda_version->second != running_cuda_version)
    error->one(
        FLERR, "Pair style " + pair_style + " PT2 package was built with CUDA " +
                   cuda_version->second + " but this LAMMPS LibTorch uses CUDA " +
                   running_cuda_version + "; export with the same LibTorch CUDA ABI");
}

inline void validate_deploy_metadata(
    Error *error, const DeployMetadata &metadata,
    const std::string &pair_style, const std::vector<std::string> &required_outputs,
    const std::vector<std::string> &available_inputs)
{
  const auto outputs_entry = metadata.find("provided_outputs");
  if (outputs_entry == metadata.end() || outputs_entry->second.empty())
  {
    error->warning(
        FLERR, "Deployed model metadata has no provided_outputs; pair style " + pair_style +
                   " is skipping capability validation for backward compatibility");
    return;
  }

  const auto dtype_entry = metadata.find("default_dtype");
  const std::string dtype =
      dtype_entry == metadata.end() ? std::string() : dtype_entry->second;
  // float64 support is deferred for both TorchScript and PT2 loaders.
  if (dtype != "float32")
  {
    error->one(FLERR, "Pair style " + pair_style +
                          " currently supports only float32 deployed models; model "
                          "default_dtype is '" +
                          (dtype.empty() ? std::string("<missing>") : dtype) + "'");
  }

  const auto edge_prune_entry = metadata.find("edge_prune");
  const bool allegro_pair = pair_style.find("allegro") != std::string::npos;
  if (allegro_pair &&
      (edge_prune_entry == metadata.end() || edge_prune_entry->second.empty()))
    error->one(
        FLERR, "Pair style " + pair_style +
                   " requires deployed model metadata edge_prune=nonzero or dense");
  if (edge_prune_entry != metadata.end() && !edge_prune_entry->second.empty() &&
      edge_prune_entry->second != "nonzero" && edge_prune_entry->second != "dense")
    error->one(
        FLERR, "Pair style " + pair_style +
                   " received invalid deployed model metadata edge_prune='" +
                   edge_prune_entry->second + "'");
  if (edge_prune_entry != metadata.end() && edge_prune_entry->second == "dense" &&
      pair_style == "allegro")
    error->warning(
        FLERR, "edge_prune=dense computes every supplied edge; the allegro pair "
               "does not provide strict LAMMPS prefiltering, so dense is computation amplification");

  std::unordered_set<std::string> provided_outputs;
  std::stringstream output_stream(outputs_entry->second);
  std::string output;
  while (output_stream >> output)
    provided_outputs.insert(output);

  std::vector<std::string> missing_outputs;
  for (const auto &required : required_outputs)
    if (provided_outputs.count(required) == 0)
      missing_outputs.push_back(required);

  if (!missing_outputs.empty())
  {
    std::ostringstream message;
    message << "Pair style " << pair_style << " requires deployed model output key";
    if (missing_outputs.size() > 1)
      message << "s";
    message << ": ";
    for (size_t index = 0; index < missing_outputs.size(); ++index)
    {
      if (index > 0)
        message << ", ";
      message << missing_outputs[index];
    }
    error->one(FLERR, message.str());
  }

  const auto inputs_entry = metadata.find("required_inputs");
  if (inputs_entry == metadata.end() || inputs_entry->second.empty())
  {
    error->warning(
        FLERR, "Deployed model metadata has no required_inputs; pair style " + pair_style +
                   " is skipping input capability validation for backward compatibility");
    return;
  }

  std::unordered_set<std::string> available_input_set(
      available_inputs.begin(), available_inputs.end());
  std::stringstream input_stream(inputs_entry->second);
  std::vector<std::string> missing_inputs;
  std::string input;
  while (input_stream >> input)
    if (available_input_set.count(input) == 0)
      missing_inputs.push_back(input);

  if (!missing_inputs.empty())
  {
    std::ostringstream message;
    message << "Pair style " << pair_style << " cannot provide deployed model input key";
    if (missing_inputs.size() > 1)
      message << "s";
    message << ": ";
    for (size_t index = 0; index < missing_inputs.size(); ++index)
    {
      if (index > 0)
        message << ", ";
      message << missing_inputs[index];
    }
    error->one(FLERR, message.str());
  }
}

inline void validate_deploy_metadata(
    Error *error, const std::unordered_map<std::string, std::string> &metadata,
    const std::string &pair_style, const std::vector<std::string> &required_outputs,
    const std::vector<std::string> &available_inputs)
{
  validate_deploy_metadata(
      error, DeployMetadata(metadata.begin(), metadata.end()), pair_style,
      required_outputs, available_inputs);
}

} // namespace LAMMPS_NS

#endif
