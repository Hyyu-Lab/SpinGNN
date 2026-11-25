#include "jit_runtime.hpp"
#include "spingnn2_jit_prepare.h"

#ifdef SP2_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <mutex>
#include <unordered_map>
#endif

namespace sp2 {
#ifdef SP2_ENABLE_CUDA
namespace {

class Driver {
public:
    Driver() : handle_(::dlopen("libcuda.so.1", RTLD_LOCAL | RTLD_NOW)) {
        if (!handle_) throw std::runtime_error("Cannot open libcuda.so.1");
        try {
            get_error_string = bind<decltype(get_error_string)>("cuGetErrorString");
            init = bind<decltype(init)>("cuInit");
            current_context = bind<decltype(current_context)>("cuCtxGetCurrent");
            push_context = bind<decltype(push_context)>("cuCtxPushCurrent_v2");
            pop_context = bind<decltype(pop_context)>("cuCtxPopCurrent_v2");
            current_device = bind<decltype(current_device)>("cuCtxGetDevice");
            device_attribute = bind<decltype(device_attribute)>("cuDeviceGetAttribute");
            load = bind<decltype(load)>("cuModuleLoadData");
            lookup = bind<decltype(lookup)>("cuModuleGetFunction");
            unload = bind<decltype(unload)>("cuModuleUnload");
            launch = bind<decltype(launch)>("cuLaunchKernel");
            func_set_attribute =
                reinterpret_cast<decltype(func_set_attribute)>(::dlsym(handle_, "cuFuncSetAttribute"));
            check(init(0), "Initialize CUDA driver");
        } catch (...) {
            ::dlclose(handle_);
            throw;
        }
    }
    ~Driver() { ::dlclose(handle_); }

    void check(CUresult code, const char* operation) const {
        if (code != CUDA_SUCCESS) {
            const char* details = nullptr;
            get_error_string(code, &details);
            throw std::runtime_error(std::string(operation) + ": " +
                                     (details ? details : "Unknown CUDA driver error"));
        }
    }

    decltype(&cuGetErrorString) get_error_string = nullptr;
    decltype(&cuInit) init = nullptr;
    decltype(&cuCtxGetCurrent) current_context = nullptr;
    decltype(&cuCtxPushCurrent) push_context = nullptr;
    decltype(&cuCtxPopCurrent) pop_context = nullptr;
    decltype(&cuCtxGetDevice) current_device = nullptr;
    decltype(&cuDeviceGetAttribute) device_attribute = nullptr;
    decltype(&cuModuleLoadData) load = nullptr;
    decltype(&cuModuleGetFunction) lookup = nullptr;
    decltype(&cuModuleUnload) unload = nullptr;
    decltype(&cuLaunchKernel) launch = nullptr;
    decltype(&cuFuncSetAttribute) func_set_attribute = nullptr;

private:
    template<class Fn> Fn bind(const char* name) const {
        const auto value = ::dlsym(handle_, name);
        if (!value) throw std::runtime_error(std::string("Missing CUDA driver function ") + name);
        return reinterpret_cast<Fn>(value);
    }
    void* handle_;
};

class ContextGuard {
public:
    ContextGuard(const Driver& api, CUcontext desired) : api_(api) {
        CUcontext current = nullptr;
        api_.check(api_.current_context(&current), "Read CUDA context");
        if (current != desired) {
            api_.check(api_.push_context(desired), "Make JIT module context current");
            pushed_ = true;
        }
    }

    ~ContextGuard() {
        if (pushed_) {
            CUcontext context = nullptr;
            api_.pop_context(&context);
        }
    }

private:
    const Driver& api_;
    bool pushed_ = false;
};

class CudaImage final : public JitImage {
public:
    CudaImage(const ValidatedImage& image, const std::string& architecture) {
        // Kokkos has already initialized the runtime for this model. Trigger
        // its primary context on this thread before using the driver API.
        const auto status = cudaFree(nullptr);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("Initialize CUDA context for JIT: ") +
                                     cudaGetErrorString(status));
        api_.check(api_.current_context(&context_), "Read CUDA context");
        if (!context_) throw std::runtime_error("No current CUDA context for JIT artifact");
        CUdevice device;
        int major = 0, minor = 0;
        api_.check(api_.current_device(&device), "Read CUDA device");
        api_.check(api_.device_attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
                   "Read CUDA compute capability");
        api_.check(api_.device_attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
                   "Read CUDA compute capability");
        const auto active = "sm_" + std::to_string(10 * major + minor);
        if (architecture != active)
            throw std::runtime_error("JIT artifact GPU architecture mismatch: expected " + active +
                                     ", received " + architecture);
        int optin_smem = 48 * 1024;
        if (api_.device_attribute(&optin_smem,
                                  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                                  device) == CUDA_SUCCESS &&
            optin_smem > 0) {
            max_optin_shared_ = static_cast<unsigned>(optin_smem);
        }
        api_.check(api_.load(&module_, image.bytes.data()), "Load prepared CUDA cubin");
    }

    ~CudaImage() override {
        try {
            // Module release must follow all asynchronous invocations using it.
            // Evaluations fence at tile transfer before releasing their handles.
            const ContextGuard guard(api_, context_);
            api_.unload(module_);
        } catch (...) {
            // The CUDA runtime may have torn down its context at process exit.
        }
    }

    void* resolve(const std::string& name) override {
        CUfunction fn = nullptr;
        api_.check(api_.lookup(&fn, module_, name.c_str()), ("Resolve CUDA JIT kernel " + name).c_str());
        return fn;
    }

    void launch(void* kernel, void* stream, int grid, int block,
                unsigned shared_bytes, void** args) const override {
        auto fn = static_cast<CUfunction>(kernel);
        if (shared_bytes > 48u * 1024u) {
            if (shared_bytes > max_optin_shared_)
                throw std::runtime_error(
                    "Requested dynamic shared memory exceeds device opt-in limit");
            std::lock_guard<std::mutex> lock(smem_mutex_);
            auto& configured = configured_smem_[fn];
            if (shared_bytes > configured) {
                if (!api_.func_set_attribute)
                    throw std::runtime_error(
                        "CUDA driver lacks cuFuncSetAttribute for extended dynamic shared memory");
                api_.check(api_.func_set_attribute(
                               fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               static_cast<int>(shared_bytes)),
                           "Configure dynamic shared memory for CUDA JIT kernel");
                configured = shared_bytes;
            }
        }
        api_.check(api_.launch(fn, grid, 1, 1, block, 1, 1,
                               shared_bytes, static_cast<CUstream>(stream), args, nullptr),
                   "Launch CUDA JIT kernel");
    }

private:
    Driver api_;
    CUcontext context_ = nullptr;
    CUmodule module_ = nullptr;
    unsigned max_optin_shared_ = 48u * 1024u;
    mutable std::mutex smem_mutex_;
    mutable std::unordered_map<CUfunction, unsigned> configured_smem_;
};

} // namespace
#endif

std::string current_cuda_arch() {
#ifdef SP2_ENABLE_CUDA
    const auto status = cudaFree(nullptr);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Initialize CUDA context for architecture query: ") +
                                 cudaGetErrorString(status));
    int device = 0;
    const auto dev_status = cudaGetDevice(&device);
    if (dev_status != cudaSuccess)
        throw std::runtime_error(std::string("Read active CUDA device: ") +
                                 cudaGetErrorString(dev_status));
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
        throw std::runtime_error("Read active CUDA device compute capability");
    return "sm_" + std::to_string(10 * major + minor);
#else
    throw std::runtime_error("CUDA architecture query requires the CUDA variant of libspingnn2");
#endif
}

std::unique_ptr<JitImage> open_cuda_plugin(const ValidatedImage& image,
                                          const std::string& architecture) {
#ifdef SP2_ENABLE_CUDA
    return std::make_unique<CudaImage>(image, architecture);
#else
    (void)image;
    (void)architecture;
    throw std::runtime_error("CUDA JIT artifact requires the CUDA variant of libspingnn2");
#endif
}
} // namespace sp2

extern "C" const char* sp2_jit_target_arch() {
#ifdef SP2_JIT_TARGET_ARCH
    return SP2_JIT_TARGET_ARCH;
#else
    return "";
#endif
}
