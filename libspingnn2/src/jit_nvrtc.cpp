#include "spingnn2_jit_prepare.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#ifdef SP2_ENABLE_CUDA
#include <dlfcn.h>
#include <nvrtc.h>
#include <fstream>
#include <vector>
#endif

namespace {
thread_local std::string last_error;

#ifdef SP2_ENABLE_CUDA
class Nvrtc {
public:
    Nvrtc() {
        const auto* override = std::getenv("SPINGNN2_NVRTC_LIBRARY");
        if (override && *override) {
            handle_ = ::dlopen(override, RTLD_NOW | RTLD_LOCAL);
        } else {
            for (const char* candidate : {"libnvrtc.so.12", "libnvrtc.so.13", "libnvrtc.so"}) {
                handle_ = ::dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
                if (handle_) break;
            }
        }
        if (!handle_)
            throw std::runtime_error(std::string("Cannot open NVRTC library: ") + ::dlerror());
        try {
            version = bind<decltype(version)>("nvrtcVersion");
            error = bind<decltype(error)>("nvrtcGetErrorString");
            create = bind<decltype(create)>("nvrtcCreateProgram");
            destroy = bind<decltype(destroy)>("nvrtcDestroyProgram");
            compile = bind<decltype(compile)>("nvrtcCompileProgram");
            log_size = bind<decltype(log_size)>("nvrtcGetProgramLogSize");
            log = bind<decltype(log)>("nvrtcGetProgramLog");
            cubin_size = bind<decltype(cubin_size)>("nvrtcGetCUBINSize");
            cubin = bind<decltype(cubin)>("nvrtcGetCUBIN");
        } catch (...) {
            ::dlclose(handle_);
            throw;
        }
    }
    ~Nvrtc() { ::dlclose(handle_); }

    void check(nvrtcResult code, const char* operation) const {
        if (code != NVRTC_SUCCESS)
            throw std::runtime_error(std::string(operation) + ": " + error(code));
    }

    decltype(&nvrtcVersion) version = nullptr;
    decltype(&nvrtcGetErrorString) error = nullptr;
    decltype(&nvrtcCreateProgram) create = nullptr;
    decltype(&nvrtcDestroyProgram) destroy = nullptr;
    decltype(&nvrtcCompileProgram) compile = nullptr;
    decltype(&nvrtcGetProgramLogSize) log_size = nullptr;
    decltype(&nvrtcGetProgramLog) log = nullptr;
    decltype(&nvrtcGetCUBINSize) cubin_size = nullptr;
    decltype(&nvrtcGetCUBIN) cubin = nullptr;

private:
    template<class Fn> Fn bind(const char* name) const {
        const auto symbol = ::dlsym(handle_, name);
        if (!symbol) throw std::runtime_error(std::string("Missing NVRTC function ") + name);
        return reinterpret_cast<Fn>(symbol);
    }
    void* handle_ = nullptr;
};

const Nvrtc& nvrtc() {
    // This function is reached only from the separate preparation entrypoints.
    static const Nvrtc api;
    return api;
}

struct Program {
    const Nvrtc& api;
    nvrtcProgram value = nullptr;
    ~Program() {
        if (value) api.destroy(&value);
    }
};
#endif

template<class F> int prepare(F&& fn) noexcept {
    try {
        last_error.clear();
        fn();
        return 0;
    } catch (const std::exception& err) {
        last_error = err.what();
    } catch (...) {
        last_error = "Unknown SpinGNN2 JIT preparation error";
    }
    return -1;
}
} // namespace

extern "C" const char* sp2_jit_target_arch() {
#ifdef SP2_JIT_TARGET_ARCH
    return SP2_JIT_TARGET_ARCH;
#else
    return "";
#endif
}

extern "C" const char* sp2_jit_prepare_error() { return last_error.c_str(); }

extern "C" const char* sp2_jit_cuda_toolchain() {
    thread_local std::string tag;
    if (prepare([&] {
#ifdef SP2_ENABLE_CUDA
        const auto& api = nvrtc();
        int major, minor;
        api.check(api.version(&major, &minor), "Read NVRTC version");
        tag = "NVRTC " + std::to_string(major) + "." + std::to_string(minor);
#else
        throw std::runtime_error("NVRTC preparation requires the CUDA variant of libspingnn2");
#endif
    }) != 0) return nullptr;
    return tag.c_str();
}

extern "C" int sp2_jit_compile_cuda(const char* source, const char* architecture,
                                    const char* output) {
    return prepare([&] {
#ifdef SP2_ENABLE_CUDA
        if (!source || !architecture || !output)
            throw std::invalid_argument("Missing NVRTC preparation input");
        const std::string arch = architecture;
        if (arch != "sm_70" && arch != "sm_80" && arch != "sm_86" && arch != "sm_89" &&
            arch != "sm_90" && arch != "sm_120")
            throw std::invalid_argument(
                "NVRTC preparation requires sm_70, sm_80, sm_86, sm_89, sm_90, or sm_120");
        const auto& api = nvrtc();
        Program program{api};
        api.check(api.create(&program.value, source, "sp2_generated.cu", 0, nullptr, nullptr),
                  "Create NVRTC program");
        const std::string target = "--gpu-architecture=" + arch;
        const char* options[] = {target.c_str(), "--std=c++17", "--fmad=true",
                                 "--prec-div=true", "--prec-sqrt=true", "--ftz=false"};
        const auto status = api.compile(program.value, sizeof(options) / sizeof(options[0]), options);
        if (status != NVRTC_SUCCESS) {
            std::size_t size = 0;
            api.log_size(program.value, &size);
            std::string log(size, '\0');
            if (size) api.log(program.value, log.data());
            if (!log.empty() && log.back() == '\0') log.pop_back();
            throw std::runtime_error(std::string("NVRTC compilation failed: ") + api.error(status) +
                                     "\n" + log);
        }
        std::size_t size = 0;
        api.check(api.cubin_size(program.value, &size), "Size NVRTC cubin");
        std::vector<char> bytes(size);
        api.check(api.cubin(program.value, bytes.data()), "Read NVRTC cubin");
        std::ofstream stream(output, std::ios::binary);
        if (!stream.write(bytes.data(), size).flush())
            throw std::runtime_error("Cannot write prepared CUDA cubin");
#else
        (void)source;
        (void)architecture;
        (void)output;
        throw std::runtime_error("NVRTC preparation requires the CUDA variant of libspingnn2");
#endif
    });
}
