#include "jit_runtime.hpp"

#include <algorithm>
#include <cerrno>
#include <elf.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <unordered_set>
#include <utility>

#include "spingnn2_c_api.h"

namespace sp2 {
std::string current_cuda_arch();

namespace {
constexpr int artifact_abi = 1;
constexpr int bundle_abi = 1;

void require(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}

bool is_supported_cuda_arch(std::string_view arch) {
    return arch == "sm_80" || arch == "sm_86" || arch == "sm_89" || arch == "sm_90" ||
           arch == "sm_120";
}

bool build_supports_cuda_arch(std::string_view arch) {
#ifdef SP2_JIT_TARGET_ARCH
    const std::string_view allowed(SP2_JIT_TARGET_ARCH);
    std::size_t start = 0;
    while (start <= allowed.size()) {
        const auto sep = allowed.find_first_of(",;", start);
        const auto token =
            allowed.substr(start, sep == std::string_view::npos ? allowed.size() - start : sep - start);
        if (token == arch) return true;
        if (sep == std::string_view::npos) break;
        start = sep + 1;
    }
    return false;
#else
    (void)arch;
    return true;
#endif
}

bool safe_member_basename(std::string_view name) {
    if (name.size() <= 5 || name.size() > 255) return false;
    if (name == "." || name == "..") return false;
    if (name.find('/') != std::string_view::npos || name.find('\\') != std::string_view::npos ||
        name.find('\0') != std::string_view::npos || name.find("..") != std::string_view::npos)
        return false;
    return name.compare(name.size() - 5, 5, ".json") == 0;
}

class ReadDescriptor {
public:
    ReadDescriptor(int dir, const char* name, int type) : fd_(
        ::openat(dir, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK |
                           (type == S_IFDIR ? O_DIRECTORY : 0))) {
        if (fd_ < 0)
            throw std::system_error(errno, std::generic_category(), std::string("Open JIT path ") + name);
        try {
            if (::fstat(fd_, &info_) != 0)
                throw std::system_error(errno, std::generic_category(), "Stat JIT path");
            require((info_.st_mode & S_IFMT) == type, "JIT path has an unexpected file type");
            require((info_.st_mode & (S_IWGRP | S_IWOTH)) == 0,
                    "Unsafe JIT permissions: writable by group or others");
            require(info_.st_uid == ::geteuid() || info_.st_uid == 0,
                    "Unsafe JIT permissions: file owner is untrusted");
        } catch (...) {
            ::close(fd_);
            throw;
        }
    }
    ~ReadDescriptor() { ::close(fd_); }
    ReadDescriptor(const ReadDescriptor&) = delete;
    ReadDescriptor& operator=(const ReadDescriptor&) = delete;
    int id() const noexcept { return fd_; }

    std::vector<unsigned char> read(std::size_t limit) const {
        require(info_.st_size > 0 && static_cast<std::uintmax_t>(info_.st_size) <= limit,
                "Invalid JIT file size");
        std::vector<unsigned char> bytes(static_cast<std::size_t>(info_.st_size));
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const auto n = ::pread(fd_, bytes.data() + offset, bytes.size() - offset, offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("Could not read complete JIT file");
            offset += std::size_t(n);
        }
        return bytes;
    }

private:
    int fd_;
    struct stat info_ {};
};

bool sha_hex(std::string_view value) {
    return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == value.npos;
}

bool kernel_name(const std::string& name) {
    return name.size() > 4 && name.size() <= 128 && name.compare(0, 4, "sp2_") == 0 &&
           name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") == name.npos;
}

bool elf_image(const std::vector<unsigned char>& bytes, bool cuda) {
    if (bytes.size() < sizeof(Elf64_Ehdr)) return false;
    Elf64_Ehdr header{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (std::memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 ||
        header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_ident[EI_DATA] != ELFDATA2LSB ||
        header.e_ident[EI_VERSION] != EV_CURRENT) return false;
    // NVIDIA assigns vendor-specific e_version values to cubins. The
    // standard encoding version is e_ident[EI_VERSION] for both ELF kinds.
    if (cuda) return header.e_type == ET_EXEC && header.e_machine == EM_CUDA;
    if (header.e_type != ET_DYN || header.e_version != EV_CURRENT) return false;
#if defined(__x86_64__)
    return header.e_machine == EM_X86_64;
#elif defined(__aarch64__)
    return header.e_machine == EM_AARCH64;
#else
    return false;
#endif
}
} // namespace

JitPlugin::JitPlugin(std::unique_ptr<JitImage> image, std::string digest,
                     std::string architecture, std::vector<std::string> symbols,
                     const std::string& identity)
    : image_(std::move(image)), digest_(std::move(digest)), arch_(std::move(architecture)),
      names_(std::move(symbols)) {
    image_->resolve(identity);
    for (const auto& name : names_) kernels_.emplace(name, image_->resolve(name));
}

void* JitPlugin::kernel(const std::string& name) const {
    const auto it = kernels_.find(name);
    if (it == kernels_.end()) throw std::runtime_error("JIT artifact missing kernel " + name);
    return it->second;
}

void JitPlugin::require_symbols(const std::vector<std::string>& symbols) const {
    for (const auto& name : symbols) kernel(name);
}

std::shared_ptr<const JitPlugin> load_single_manifest(const ReadDescriptor& directory,
                                                      const Json& desc,
                                                      std::string_view contract_digest,
                                                      int precision) {
    static constexpr const char* keys[] = {"abi_version", "artifact_abi", "artifact",
        "artifact_sha256", "contract_sha256", "precision", "backend", "gpu_arch", "symbols",
        "source_sha256", "cache_key"};
    require(desc.is_object() && desc.size() == sizeof(keys) / sizeof(keys[0]),
            "Invalid JIT manifest schema");
    for (const auto* key : keys) require(desc.contains(key), "Missing JIT manifest field");
    require(desc.at("abi_version").is_number_integer() && desc.at("abi_version") == SP2_ABI_VERSION &&
            desc.at("artifact_abi").is_number_integer() && desc.at("artifact_abi") == artifact_abi,
            "JIT artifact ABI version mismatch");
    require(desc.at("contract_sha256").get<std::string>() == contract_digest,
            "JIT artifact contract SHA-256 mismatch");
    require((precision == 32 || precision == 64) &&
                desc.at("precision") == (precision == 32 ? "float32" : "float64"),
            "JIT artifact precision mismatch");
#ifdef SP2_ENABLE_CUDA
    constexpr bool cuda = true;
#else
    constexpr bool cuda = false;
#endif
    require(desc.at("backend") == (cuda ? "cuda" : "cpu"), "JIT artifact backend mismatch");
    const auto arch = desc.at("gpu_arch").get<std::string>();
    require(cuda ? is_supported_cuda_arch(arch) : arch.empty(),
            "Invalid JIT artifact GPU architecture");
    if (cuda) {
        require(build_supports_cuda_arch(arch),
                "JIT artifact GPU architecture mismatch with native build");
    }

    const auto name = desc.at("artifact").get<std::string>();
    const auto digest = desc.at("artifact_sha256").get<std::string>();
    const auto source_hash = desc.at("source_sha256").get<std::string>();
    const auto cache_key = desc.at("cache_key").get<std::string>();
    require(sha_hex(cache_key) && sha_hex(source_hash) && sha_hex(digest),
            "Invalid JIT manifest digest");
    Json key_fields = Json::object();
    for (const auto* key : keys) {
        const std::string_view field(key);
        if (field == "artifact" || field == "artifact_sha256" || field == "cache_key") continue;
        key_fields[key] = desc.at(key);
    }
    const auto key_text = key_fields.dump();
    const std::vector<unsigned char> canonical(key_text.begin(), key_text.end());
    require(sha256_bytes(canonical) == cache_key, "Invalid JIT manifest cache identity");
    require(name == "kernel-" + digest + (cuda ? ".cubin" : ".so"), "Invalid JIT artifact filename");
    const ReadDescriptor binary(directory.id(), name.c_str(), S_IFREG);
    ValidatedImage image{binary.read(512ull * 1024 * 1024)};
    require(sha256_bytes(image.bytes) == digest, "JIT artifact SHA-256 mismatch");
    require(elf_image(image.bytes, cuda), "JIT artifact is not a matching native ELF image");

    const auto symbols = desc.at("symbols").get<std::vector<std::string>>();
    require(!symbols.empty() && symbols.size() <= 16384, "Invalid JIT artifact symbol count");
    std::unordered_set<std::string> seen;
    for (const auto& symbol : symbols)
        require(kernel_name(symbol) && seen.insert(symbol).second, "Invalid JIT artifact symbol list");
    auto plugin = cuda ? open_cuda_plugin(image, arch) : open_host_plugin(image);
    return std::make_shared<JitPlugin>(std::move(plugin), digest, arch, symbols,
                                       "sp2_identity_" + cache_key);
}

std::shared_ptr<const JitPlugin> load_jit_artifact(const std::filesystem::path& manifest,
                                                   std::string_view contract_digest,
                                                   int precision) {
    const auto directory_path = manifest.has_parent_path() ? manifest.parent_path() : ".";
    const ReadDescriptor directory(AT_FDCWD, directory_path.c_str(), S_IFDIR);
    const ReadDescriptor metadata(directory.id(), manifest.filename().c_str(), S_IFREG);
    const auto text = metadata.read(1024 * 1024);
    const auto desc = Json::parse(text);
    if (desc.is_object() && desc.contains("bundle_abi")) {
        static constexpr const char* bkeys[] = {
            "abi_version", "bundle_abi", "backend", "contract_sha256", "precision", "artifacts"};
        require(desc.size() == sizeof(bkeys) / sizeof(bkeys[0]), "Invalid JIT bundle schema");
        for (const auto* key : bkeys) require(desc.contains(key), "Missing JIT bundle field");
        require(desc.at("abi_version").is_number_integer() &&
                    desc.at("abi_version") == SP2_ABI_VERSION &&
                    desc.at("bundle_abi").is_number_integer() &&
                    desc.at("bundle_abi") == bundle_abi,
                "JIT bundle ABI version mismatch");
#ifdef SP2_ENABLE_CUDA
        constexpr bool cuda = true;
#else
        constexpr bool cuda = false;
#endif
        require(cuda && desc.at("backend") == "cuda",
                "Multi-architecture JIT bundle requires the CUDA variant of libspingnn2");
        require(desc.at("contract_sha256").get<std::string>() == contract_digest,
                "JIT bundle contract SHA-256 mismatch");
        require((precision == 32 || precision == 64) &&
                    desc.at("precision") == (precision == 32 ? "float32" : "float64"),
                "JIT bundle precision mismatch");
        const auto& artifacts = desc.at("artifacts");
        require(artifacts.is_object() && !artifacts.empty() && artifacts.size() <= 16,
                "Invalid JIT bundle artifacts map");
        for (auto it = artifacts.begin(); it != artifacts.end(); ++it) {
            require(is_supported_cuda_arch(it.key()), "Invalid GPU architecture in JIT bundle");
            require(it.value().is_string() &&
                        safe_member_basename(it.value().get<std::string>()),
                    "Unsafe member manifest path in JIT bundle");
        }
        const auto active = current_cuda_arch();
        if (!artifacts.contains(active))
            throw std::runtime_error("JIT bundle does not contain an artifact for active GPU " +
                                     active);
        const auto member_name = artifacts.at(active).get<std::string>();
        const ReadDescriptor member(directory.id(), member_name.c_str(), S_IFREG);
        const auto member_text = member.read(1024 * 1024);
        const auto member_desc = Json::parse(member_text);
        return load_single_manifest(directory, member_desc, contract_digest, precision);
    }
    return load_single_manifest(directory, desc, contract_digest, precision);
}
} // namespace sp2
