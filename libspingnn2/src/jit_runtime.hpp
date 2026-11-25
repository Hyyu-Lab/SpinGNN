#pragma once

#include "contract.hpp"

#include <array>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sp2 {

// Generated host kernels and CUDA kernels use the same argument storage. The
// host ABI adds the launch dimensions ahead of the argument vector.
class JitImage {
public:
    virtual ~JitImage() = default;
    virtual void* resolve(const std::string& name) = 0;
    virtual void launch(void* kernel, void* stream, int grid, int block,
                        unsigned shared_bytes, void** args) const = 0;
};

struct ValidatedImage {
    std::vector<unsigned char> bytes;
};

std::unique_ptr<JitImage> open_host_plugin(const ValidatedImage& source);
std::unique_ptr<JitImage> open_cuda_plugin(const ValidatedImage& source,
                                          const std::string& architecture);

class JitPlugin {
public:
    JitPlugin(std::unique_ptr<JitImage>, std::string digest, std::string architecture,
              std::vector<std::string> symbols, const std::string& identity);

    const std::string& digest() const noexcept { return digest_; }
    const std::string& arch() const noexcept { return arch_; }
    const std::vector<std::string>& symbols() const noexcept { return names_; }
    void require_symbols(const std::vector<std::string>& symbols) const;

    template<class ExecutionSpace, class... Args>
    void launch(const ExecutionSpace& exec, const std::string& name, int count,
                int block, Args... args) const {
        if (count < 0 || block < 1) {
            throw std::invalid_argument("Invalid SpinGNN2 kernel launch dimensions");
        }
        if (count != 0)
            launch_grid(exec, name, (count - 1) / block + 1, block, 0, count, args...);
    }

    template<class ExecutionSpace, class... Args>
    void launch_grid(const ExecutionSpace& exec, const std::string& name, int grid,
                     int block, unsigned shared_bytes, Args... args) const {
        if (grid < 0 || block < 1)
            throw std::invalid_argument("Invalid SpinGNN2 kernel launch dimensions");
        if (grid == 0) return;
        std::array<void*, sizeof...(Args)> packed{&args...};
#ifdef SP2_ENABLE_CUDA
        void* stream = exec.cuda_stream();
#else
        (void)exec;
        void* stream = nullptr;
#endif
        image_->launch(kernel(name), stream, grid, block, shared_bytes, packed.data());
    }

private:
    void* kernel(const std::string& name) const;
    std::unique_ptr<JitImage> image_;
    std::string digest_, arch_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, void*> kernels_;
};

std::shared_ptr<const JitPlugin> load_jit_artifact(const std::filesystem::path& manifest,
                                                   std::string_view contract_digest,
                                                   int precision);

inline auto load_jit_artifact(const std::filesystem::path& manifest, const Contract& contract,
                              int precision) {
    return load_jit_artifact(manifest, contract.digest(), precision);
}

} // namespace sp2
