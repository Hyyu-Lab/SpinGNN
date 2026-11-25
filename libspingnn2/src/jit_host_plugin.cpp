#include "jit_runtime.hpp"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <system_error>
#include <unistd.h>

namespace sp2 {
namespace {

// dlopen a sealed private copy of the verified bytes. A stable descriptor
// avoids looking up a path again after validation. Hold it through dlclose so
// two loaded artifacts cannot alias via reuse of a /proc/self/fd number.
class HostImage final : public JitImage {
public:
    explicit HostImage(const ValidatedImage& image) {
        fd_ = ::memfd_create("spingnn2-kernel", MFD_CLOEXEC | MFD_ALLOW_SEALING);
        if (fd_ < 0) fail("Create immutable host JIT image");
        try {
            std::size_t done = 0;
            while (done < image.bytes.size()) {
                const auto n = ::write(fd_, image.bytes.data() + done, image.bytes.size() - done);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) fail("Write verified host JIT image");
                done += std::size_t(n);
            }
            if (::fcntl(fd_, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE) < 0)
                fail("Seal verified host JIT image");
            const auto path = "/proc/self/fd/" + std::to_string(fd_);
            handle_ = ::dlopen(path.c_str(), RTLD_LOCAL | RTLD_NOW);
            if (!handle_) throw std::runtime_error(std::string("Load host JIT artifact: ") + ::dlerror());
        } catch (...) {
            ::close(fd_);
            throw;
        }
    }

    ~HostImage() override {
        ::dlclose(handle_);
        ::close(fd_);
    }

    void* resolve(const std::string& name) override {
        ::dlerror();
        auto* value = ::dlsym(handle_, name.c_str());
        if (!value || ::dlerror()) throw std::runtime_error("Host JIT artifact missing kernel " + name);
        return value;
    }

    void launch(void* kernel, void*, int grid, int block, unsigned, void** args) const override {
        reinterpret_cast<void (*)(int, int, void**)>(kernel)(grid, block, args);
    }

private:
    [[noreturn]] static void fail(const char* message) {
        throw std::system_error(errno, std::generic_category(), message);
    }
    int fd_;
    void* handle_ = nullptr;
};

} // namespace

std::unique_ptr<JitImage> open_host_plugin(const ValidatedImage& image) {
    return std::make_unique<HostImage>(image);
}
} // namespace sp2
