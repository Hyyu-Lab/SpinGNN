#ifndef SPINGNN2_TENSOR_HPP
#define SPINGNN2_TENSOR_HPP

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef SP2_ENABLE_CUDA
#include <cuda_runtime_api.h>
#include <cstring>
#endif

namespace sp2 {
using Exec = Kokkos::DefaultExecutionSpace;

#ifdef SP2_ENABLE_CUDA
inline void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#endif

template<class T>
using DeviceArray = Kokkos::View<T*, Kokkos::LayoutRight, Exec::memory_space>;

template<class T>
DeviceArray<T> upload(const Exec& exec, const std::vector<T>& xs) {
    DeviceArray<T> out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "model constants"), xs.size());
    if (!xs.empty()) {
        using ConstHost = Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
        Kokkos::deep_copy(exec, out, ConstHost(xs.data(), xs.size()));
        exec.fence("upload model constants");
    }
    return out;
}

template<class Functor>
void for_each(const Exec& exec, const char* label, int n, const Functor& work) {
    if (!n) return;
    const Kokkos::RangePolicy<Exec> range(exec, 0, n);
#ifdef SP2_ENABLE_CUDA
    // CUDA Graph parameters must be captured by value instead of copied from
    // Kokkos's shared host staging buffer during each replay.
    auto policy = Kokkos::Experimental::require(
        range, Kokkos::Experimental::WorkItemProperty::HintLightWeight);
#else
    const auto &policy = range;
#endif
    Kokkos::parallel_for(label, policy, work);
}

template<class T>
struct Tensor {
    T* ptr = nullptr;
    int rows = 0;
    int cols = 0;
    KOKKOS_INLINE_FUNCTION T& operator()(int row, int col) const {
        return ptr[static_cast<std::size_t>(row) * cols + col];
    }
    KOKKOS_INLINE_FUNCTION int size() const { return rows * cols; }
    KOKKOS_INLINE_FUNCTION explicit operator bool() const { return ptr != nullptr; }
};

enum class Profile : int { geometry, mlp, tp_linear, reverse };
class JitPlugin;

template<class T>
class Workspace {
public:
    using Snapshot = std::vector<std::vector<T>>;

    explicit Workspace(Exec instance = Exec()) : exec(std::move(instance)) {}

    // Slots preserve the addresses of tensors already handed to an adjoint tape.
    // Rewinding makes released slots reusable on the same execution stream.
    Tensor<T> make(int rows, int cols) {
        if (rows < 0 || cols < 0 || (cols && rows > std::numeric_limits<int>::max() / cols)) {
            throw std::length_error("Tile tensor exceeds the supported element count");
        }
        const std::size_t size = static_cast<std::size_t>(rows) * cols;
        if (cursor == slots.size()) {
#ifdef SP2_ENABLE_CUDA
            if (capturing) throw std::runtime_error("CUDA graph requires warmed workspace slots");
#endif
            slots.emplace_back();
            used.push_back(0);
        }
        auto& slot = slots[cursor];
        if (slot.extent(0) < std::max(size, std::size_t(1))) {
            if (cursor < frozen) throw std::logic_error("Lattice cache workspace was not warmed up");
#ifdef SP2_ENABLE_CUDA
            if (capturing) throw std::runtime_error("CUDA graph requires stable workspace addresses");
            // Slot growth releases the old device address even when allocation fails.
            ++allocation_epoch;
            const std::size_t headroom = size + (size >> 3);
            slot.allocate(exec.cuda_stream(), std::max<std::size_t>(headroom, 1));
#else
            if (slot.data()) exec.fence("grow workspace slot");
            slot = DeviceArray<T>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "tile scratch"),
                                  std::max(size, std::size_t(1)));
#endif
        }
        used[cursor++] = size;
        return {slot.data(), rows, cols};
    }

    std::size_t mark() const { return cursor; }
    void rewind(std::size_t mark) {
        if (mark > slots.size()) throw std::logic_error("Invalid workspace mark");
        cursor = mark;
    }
    void freeze() { frozen = std::max(frozen, cursor); }

    void seal() {
        exec.fence("release temporary tape slots");
        slots.resize(cursor);
        used.resize(cursor);
        freeze();
    }

#ifdef SP2_ENABLE_CUDA
    uint64_t allocation_generation() const { return allocation_epoch; }

    struct CaptureScope {
        explicit CaptureScope(Workspace &w) : workspace(w) { workspace.capturing = true; }
        ~CaptureScope() { workspace.capturing = false; }
        CaptureScope(const CaptureScope &) = delete;
        CaptureScope &operator=(const CaptureScope &) = delete;
    private:
        Workspace &workspace;
    };

    void reset_tape() {
        // The previous pullback and the new forward share one CUDA stream.
        // Keep both tape and forward scratch allocations for the new geometry.
        cursor = frozen = 0;
        seconds = {};
    }

    void freeze_tape() {
        // Forward can rewind temporary slots. Their stream-ordered destruction
        // leaves the cached tape at stable addresses without a device fence.
        slots.resize(cursor);
        used.resize(cursor);
        freeze();
    }

    std::size_t tape_size() const {
        std::size_t size = 0;
        for (std::size_t i = 0; i < cursor; ++i) size += used[i];
        return size;
    }

    std::vector<std::size_t> tape_shape() const {
        return {used.begin(), used.begin() + cursor};
    }

    using CudaTape = std::vector<T>;

    CudaTape make_spill() const { return CudaTape(tape_size()); }

    void spill(CudaTape& target, T* pinned, std::size_t capacity) const {
        transfer_tape(pinned, capacity, target, false);
    }

    void restore(CudaTape& source, const std::vector<std::size_t>& shape,
                 T* pinned, std::size_t capacity) {
        if (shape.size() > frozen) throw std::logic_error("Lattice cache is larger than the workspace");
        cursor = shape.size();
        for (std::size_t i = 0; i < cursor; ++i) {
            if (shape[i] > slots[i].extent(0)) throw std::logic_error("Incompatible lattice tile size");
            used[i] = shape[i];
        }
        transfer_tape(pinned, capacity, source, true);
    }

#endif

    Snapshot snapshot() const {
        Snapshot output(cursor);
        for (std::size_t slot = 0; slot < cursor; ++slot) {
            output[slot].resize(used[slot]);
            if (used[slot]) {
                Host to(output[slot].data(), used[slot]);
                const Device from(slots[slot].data(), used[slot]);
                Kokkos::deep_copy(exec, to, from);
            }
        }
        exec.fence("save lattice state to host");
        return output;
    }

    void restore(const Snapshot& input) {
        if (input.size() > frozen) throw std::logic_error("Lattice cache is larger than the workspace");
        cursor = input.size();
        for (std::size_t slot = 0; slot < cursor; ++slot) {
            const std::size_t size = input[slot].size();
            if (size > slots[slot].extent(0)) throw std::logic_error("Incompatible lattice tile size");
            used[slot] = size;
            if (size) {
                const Host from(const_cast<T*>(input[slot].data()), size);
                Device to(slots[slot].data(), size);
                Kokkos::deep_copy(exec, to, from);
            }
        }
    }

    std::size_t bytes() const {
        std::size_t total = 0;
        for (const auto& slot : slots) total += slot.extent(0) * sizeof(T);
        return total;
    }

    Exec exec;
    const JitPlugin* jit = nullptr;
    bool profile = false;
    std::array<double, 4> seconds{};

private:
    using Host = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
    using Device = Kokkos::View<T*, Exec::memory_space, Kokkos::MemoryUnmanaged>;
#ifdef SP2_ENABLE_CUDA
    bool capturing = false;
    uint64_t allocation_epoch = 0;

    // Kokkos CudaSpace allocation can fence even with an execution instance.
    // The ordered allocator keeps earlier addresses valid up to their last use
    // and reclaims growth on the same stream, with no device-wide allocator fence.
    class Slot {
        T* ptr_ = nullptr;
        std::size_t size_ = 0;
        cudaStream_t stream_ = nullptr;
    public:
        Slot() = default;
        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;
        Slot(Slot&& slot) noexcept
          : ptr_(std::exchange(slot.ptr_, nullptr)), size_(slot.size_), stream_(slot.stream_) {}
        ~Slot() { if (ptr_) cudaFreeAsync(ptr_, stream_); }
        T* data() const { return ptr_; }
        std::size_t extent(int) const { return size_; }
        void allocate(cudaStream_t stream, std::size_t size) {
            if (ptr_) {
                cuda_check(cudaFreeAsync(ptr_, stream_), "Release tile scratch after last use");
                ptr_ = nullptr;
                size_ = 0;
            }
            stream_ = stream;
            cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&ptr_), size * sizeof(T), stream),
                       "Allocate ordered CUDA scratch");
            size_ = size;
        }
    };
    std::vector<Slot> slots;

    struct HostCopy {
        T* destination;
        const T* source;
        std::size_t bytes;
    };

    static void CUDART_CB host_copy(void* data) {
        const std::unique_ptr<HostCopy> copy(static_cast<HostCopy*>(data));
        std::memcpy(copy->destination, copy->source, copy->bytes);
    }

    // Keep pageable cached tapes outside the pinned-memory budget. Stream
    // callbacks order reuse of one bounded slab with the asynchronous copies.
    void transfer_tape(T* pinned, std::size_t capacity, CudaTape& host, bool restore) const {
        const auto stream = exec.cuda_stream();
        if (host.size() != tape_size() || !capacity)
            throw std::logic_error("Invalid lattice snapshot transfer size");
        std::size_t slot = 0, begin = 0;
        for (std::size_t offset = 0; offset < host.size();) {
            const auto count = std::min(capacity, host.size() - offset);
            const auto bytes = count * sizeof(T);
            auto enqueue = [&](T* dst, const T* src) {
                auto copy = std::make_unique<HostCopy>(HostCopy{dst, src, bytes});
                cuda_check(cudaLaunchHostFunc(stream, host_copy, copy.get()), "Queue lattice host transfer");
                copy.release();
            };
            if (restore) enqueue(pinned, host.data() + offset);
            for (std::size_t full = 0; full < count;) {
                if (begin == used[slot]) { ++slot; begin = 0; continue; }
                const auto n = std::min(count - full, used[slot] - begin);
                T* const device = slots[slot].data() + begin;
                cuda_check(cudaMemcpyAsync(restore ? device : pinned + full,
                                           restore ? pinned + full : device,
                                           n * sizeof(T), restore ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,
                                           stream), "Queue lattice device transfer");
                full += n;
                begin += n;
            }
            if (!restore) enqueue(host.data() + offset, pinned);
            offset += count;
        }
    }
#else
    std::vector<DeviceArray<T>> slots;
#endif
    std::vector<std::size_t> used;
    std::size_t cursor = 0;
    std::size_t frozen = 0;
};

template<class T>
class ProfileScope {
public:
    ProfileScope(Workspace<T>& work, Profile timer) : w(work), stage(static_cast<int>(timer)) {
        if (w.profile) {
            w.exec.fence("profile stage start");
            start = std::chrono::steady_clock::now();
        }
    }
    ~ProfileScope() {
        if (w.profile) {
            w.exec.fence("profile stage end");
            w.seconds[stage] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    }
private:
    Workspace<T>& w;
    int stage;
    std::chrono::steady_clock::time_point start;
};

template<class T>
void zero(Workspace<T>& w, Tensor<T> out) {
    if (!out.size()) return;
#ifdef SP2_ENABLE_CUDA
    cuda_check(cudaMemsetAsync(out.ptr, 0, sizeof(T) * static_cast<std::size_t>(out.size()),
                               w.exec.cuda_stream()), "Zero tile tensor");
#else
    for_each(w.exec, "zero tile tensor", out.size(), KOKKOS_LAMBDA(int i) { out.ptr[i] = T(0); });
#endif
}

template<class T>
void add_inplace(Workspace<T>& w, Tensor<T> out, Tensor<T> in) {
    if (out.rows != in.rows || out.cols != in.cols) throw std::logic_error("Cannot add different tile shapes");
    const int n = out.size();
#ifdef SP2_ENABLE_CUDA
    if constexpr (std::is_same_v<T, float>) {
        if ((n & 3) == 0 &&
            (reinterpret_cast<std::uintptr_t>(out.ptr) & 15u) == 0u &&
            (reinterpret_cast<std::uintptr_t>(in.ptr) & 15u) == 0u) {
            auto* dst4 = reinterpret_cast<float4*>(out.ptr);
            const auto* src4 = reinterpret_cast<const float4*>(in.ptr);
            for_each(w.exec, "accumulate tile tensor vec4", n / 4, KOKKOS_LAMBDA(int i) {
                float4 a = dst4[i];
                const float4 b = src4[i];
                a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
                dst4[i] = a;
            });
            return;
        }
    }
#endif
    for_each(w.exec, "accumulate tile tensor", n, KOKKOS_LAMBDA(int i) { out.ptr[i] += in.ptr[i]; });
}

template<class T>
Tensor<T> add_tensors(Workspace<T>& w, Tensor<T> a, Tensor<T> b) {
    if (a.rows != b.rows || a.cols != b.cols) throw std::logic_error("Cannot add different tile shapes");
    auto out = w.make(a.rows, a.cols);
    const int n = out.size();
#ifdef SP2_ENABLE_CUDA
    if constexpr (std::is_same_v<T, float>) {
        if ((n & 3) == 0 &&
            (reinterpret_cast<std::uintptr_t>(out.ptr) & 15u) == 0u &&
            (reinterpret_cast<std::uintptr_t>(a.ptr) & 15u) == 0u &&
            (reinterpret_cast<std::uintptr_t>(b.ptr) & 15u) == 0u) {
            auto* dst4 = reinterpret_cast<float4*>(out.ptr);
            const auto* a4 = reinterpret_cast<const float4*>(a.ptr);
            const auto* b4 = reinterpret_cast<const float4*>(b.ptr);
            for_each(w.exec, "sum adjoints vec4", n / 4, KOKKOS_LAMBDA(int i) {
                const float4 x = a4[i], y = b4[i];
                dst4[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
            });
            return out;
        }
    }
#endif
    for_each(w.exec, "sum adjoints", n, KOKKOS_LAMBDA(int i) { out.ptr[i] = a.ptr[i] + b.ptr[i]; });
    return out;
}

template<class T>
Tensor<T> slice_columns(Workspace<T>& w, Tensor<T> in, int begin, int width) {
    if (begin < 0 || width < 0 || begin > in.cols - width) throw std::logic_error("Invalid tile slice");
    auto out = w.make(in.rows, width);
    if (!out.size()) return out;
    const int in_cols = in.cols;
#ifdef SP2_ENABLE_CUDA
    if constexpr (std::is_same_v<T, float>) {
        if ((width & 3) == 0 && (begin & 3) == 0 && (in_cols & 3) == 0 &&
            (reinterpret_cast<std::uintptr_t>(out.ptr) & 15u) == 0u &&
            (reinterpret_cast<std::uintptr_t>(in.ptr) & 15u) == 0u) {
            const int w4 = width / 4, b4 = begin / 4, ic4 = in_cols / 4;
            auto* dst4 = reinterpret_cast<float4*>(out.ptr);
            const auto* src4 = reinterpret_cast<const float4*>(in.ptr);
            for_each(w.exec, "slice feature columns vec4", in.rows * w4, KOKKOS_LAMBDA(int i) {
                const int row = i / w4;
                const int col = i - row * w4;
                dst4[i] = src4[static_cast<std::size_t>(row) * ic4 + b4 + col];
            });
            return out;
        }
    }
#endif
    for_each(w.exec, "slice feature columns", out.size(), KOKKOS_LAMBDA(int i) {
        const int row = i / width;
        out.ptr[i] = in.ptr[static_cast<std::size_t>(row) * in_cols + begin + (i - row * width)];
    });
    return out;
}

template<class T>
Tensor<T> concat_columns(Workspace<T>& w, Tensor<T> a, Tensor<T> b, Tensor<T> c = {}) {
    if (a.rows != b.rows || (c && a.rows != c.rows)) throw std::logic_error("Cannot concatenate different tile counts");
    const int ac = a.cols, bc = b.cols, cc = c.cols, oc = ac + bc + cc;
    auto out = w.make(a.rows, oc);
    if (!out.size()) return out;
    for_each(w.exec, "concatenate features", out.size(), KOKKOS_LAMBDA(int i) {
        const int row = i / oc;
        const int col = i - row * oc;
        out.ptr[i] = col < ac ? a.ptr[static_cast<std::size_t>(row) * ac + col] :
                     col < ac + bc ? b.ptr[static_cast<std::size_t>(row) * bc + (col - ac)] :
                                     c.ptr[static_cast<std::size_t>(row) * cc + (col - ac - bc)];
    });
    return out;
}

template<class T>
Tensor<T> copy_tensor(Workspace<T>& w, Tensor<T> in) {
    auto out = w.make(in.rows, in.cols);
    for_each(w.exec, "copy tile tensor", in.size(), KOKKOS_LAMBDA(int i) { out.ptr[i] = in.ptr[i]; });
    return out;
}
}  // namespace sp2

#endif
