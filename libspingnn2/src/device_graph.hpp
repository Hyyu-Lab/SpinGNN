#pragma once

#ifdef SP2_ENABLE_CUDA
#include "graph.hpp"
#include "tile.hpp"

#include <array>
#include <cstring>
#include <type_traits>

namespace sp2 {

// Application pointers are pageable. Stage a bounded range at a time, so
// asynchronous copies never depend on the caller's storage being page locked.
template <class T> class PinnedArray {
    T *ptr_ = nullptr;
    std::size_t size_ = 0;

  public:
    PinnedArray() = default;
    PinnedArray(const PinnedArray &) = delete;
    PinnedArray &operator=(const PinnedArray &) = delete;
    ~PinnedArray() {
        if (ptr_)
            cudaFreeHost(ptr_);
    }
    void reset() {
        if (ptr_) {
            cuda_check(cudaFreeHost(ptr_), "Free pinned host storage");
            ptr_ = nullptr;
            size_ = 0;
        }
    }
    void reserve(std::size_t size) {
        if (size <= size_)
            return;
        reset();
        cuda_check(cudaHostAlloc(reinterpret_cast<void **>(&ptr_), size * sizeof(T),
                                  cudaHostAllocDefault),
                   "Allocate pinned host storage");
        size_ = size;
    }
    T *data() const { return ptr_; }
    std::size_t size() const { return size_; }
};

class UploadRing {
    struct Slot {
        PinnedArray<unsigned char> bytes;
        cudaEvent_t copied = nullptr;
        ~Slot() {
            if (copied)
                cudaEventDestroy(copied);
        }
        void wait() const {
            if (copied)
                cuda_check(cudaEventSynchronize(copied), "Reuse pinned upload buffer");
        }
        void event() {
            if (!copied)
                cuda_check(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming),
                           "Create pinned upload event");
        }
    };

    std::array<Slot, 4> slots_;
    std::size_t next_ = 0;
    cudaStream_t stream_ = nullptr;

  public:
    template <class T>
    void copy(const Exec &exec, T *target, const T *source, std::size_t count) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (count && !stream_)
            cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "Create graph upload stream");
        constexpr std::size_t max_values = (16u * 1024u * 1024u) / sizeof(T);
        while (count) {
            const auto values = std::min(count, max_values);
            const auto bytes = values * sizeof(T);
            auto &slot = slots_[next_++ % slots_.size()];
            slot.wait();
            slot.bytes.reserve(bytes);
            slot.event();
            std::memcpy(slot.bytes.data(), source, bytes);
            cuda_check(cudaMemcpyAsync(target, slot.bytes.data(), bytes, cudaMemcpyHostToDevice,
                                       stream_),
                       "Enqueue pinned graph input transfer");
            cuda_check(cudaEventRecord(slot.copied, stream_), "Record graph upload event");
            cuda_check(cudaStreamWaitEvent(exec.cuda_stream(), slot.copied, 0), "Await graph upload");
            target += values;
            source += values;
            count -= values;
        }
    }

    // The edge slot cannot be overwritten until its last consumer completes.
    void wait_for_edge(cudaEvent_t ready) {
        if (!stream_)
            cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "Create graph upload stream");
        if (ready)
            cuda_check(cudaStreamWaitEvent(stream_, ready, 0), "Await reusable edge inputs");
    }

    UploadRing() = default;
    UploadRing(const UploadRing &) = delete;
    UploadRing &operator=(const UploadRing &) = delete;
    ~UploadRing() {
        if (stream_) cudaStreamDestroy(stream_);
        for (auto &slot : slots_)
            if (slot.copied) cudaEventSynchronize(slot.copied);
    }
};

template <class T>
DeviceArray<T> cuda_array(const Exec &exec, std::size_t values, const char *name) {
    return DeviceArray<T>(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, std::string(name)),
                          std::max<std::size_t>(1, values));
}

// Only atom/frame data and CSR row pointers are graph-sized on the GPU.
// The stable neighbor and shift arrays retain their borrowed-ABI staging
// representation on the host, and stream a bounded CSR slice at a time.
template <class T> struct DeviceGraph {
    DeviceArray<T> pos, cell;
    DeviceArray<int> types, batch, offsets;
    DeviceArray<int64_t> counts;
    bool periodic{}, shared_cell{};

    void upload(const Exec &exec, UploadRing &ring, const Graph<T> &g) {
        auto copy = [&](auto &to, const auto &from, const char *label) {
            using V = typename std::decay_t<decltype(from)>::value_type;
            to = cuda_array<V>(exec, from.size(), label);
            ring.copy(exec, to.data(), from.data(), from.size());
        };
        copy(pos, g.pos, "graph positions");
        copy(types, g.types, "graph element types");
        copy(batch, g.batch, "graph atom frames");
        copy(offsets, g.offsets, "graph CSR offsets");
        copy(counts, g.counts, "graph frame sizes");
        periodic = !g.cell.empty();
        shared_cell = g.cell.size() == 9;
        if (periodic)
            copy(cell, g.cell, "graph frame cells");
    }
    std::size_t bytes() const {
        return sizeof(T) * (pos.extent(0) + cell.extent(0)) +
               sizeof(int) * (types.extent(0) + batch.extent(0) + offsets.extent(0)) +
               sizeof(int64_t) * counts.extent(0);
    }
};

template <class T> struct DeviceEdges {
    DeviceArray<int> target, center, offsets, species_j;
    DeviceArray<T> shifts;
    int capacity{}, rows{};
    std::shared_ptr<std::remove_pointer_t<cudaEvent_t>> reusable;

    void consumed(const Exec &exec) {
        if (!reusable) {
            cudaEvent_t event = nullptr;
            cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "Create edge reuse event");
            reusable.reset(event, cudaEventDestroy);
        }
        cuda_check(cudaEventRecord(reusable.get(), exec.cuda_stream()), "Record edge input lifetime");
    }

    void reserve(const Exec &exec, int peak, int tile_atoms, bool periodic) {
        if (peak > capacity || !center.data()) {
            capacity = std::max(peak, 1);
            target = cuda_array<int>(exec, capacity, "edge neighbors");
            center = cuda_array<int>(exec, capacity, "edge centers");
            species_j = cuda_array<int>(exec, capacity, "edge neighbor types");
        }
        if (tile_atoms > rows || !offsets.data()) {
            rows = tile_atoms;
            offsets = cuda_array<int>(exec, std::size_t(rows) + 1, "tile CSR offsets");
        }
        if (periodic && shifts.extent(0) < 3ull * capacity)
            shifts = cuda_array<T>(exec, 3ull * capacity, "edge periodic shifts");
    }

    void retain_topology() {
        // Periodic shifts are needed only while forming the Stage-1 distance
        // tape. Keeping these rows would duplicate the cached displacements.
        shifts = {};
    }

    TileInput<T> prepare(Workspace<T> &w, UploadRing &ring, const DeviceGraph<T> &device,
                         const Graph<T> &g, int start, int stop, bool spline, bool geometry) {
        const int n = stop - start, first = g.offsets[start], e = g.offsets[stop] - first;
        if (e) {
            ring.wait_for_edge(reusable.get());
            ring.copy(w.exec, target.data(), g.neighbors() + first, e);
            if (geometry && device.periodic)
                ring.copy(w.exec, shifts.data(), g.images() + 3ull * first, 3ull * e);
        }

        TileInput<T> input{};
        input.center = Kokkos::subview(center, std::make_pair(0, e));
        input.offsets = Kokkos::subview(offsets, std::make_pair(0, n + 1));
        input.species = Kokkos::subview(device.types, std::make_pair(start, stop));
        input.species_j = Kokkos::subview(species_j, std::make_pair(0, e));
        input.ncenters = n;
        input.spline = spline;
        input.targets = target;
        input.frames = device.batch;
        input.positions = device.pos.data();
        input.graph_cells = device.periodic ? device.cell.data() : nullptr;
        input.start = start;
        input.shared_cell = device.shared_cell;
        // For a resident tape the dynamic device topology is reconstructed
        // into its own stable views before its Stage-1 records are used.
        const auto csr = input.offsets, centers = input.center, neighbor_species = input.species_j;
        const auto targets = target, types = device.types, graph_offsets = device.offsets;
        for_each(w.exec, "sp2::gather_tile_csr", n + 1, KOKKOS_LAMBDA(int i) {
            const int begin = graph_offsets(start + i) - first;
            csr(i) = begin;
            if (i < n) {
                for (int k = begin; k < graph_offsets(start + i + 1) - first; ++k) {
                    centers(k) = i;
                    neighbor_species(k) = types(targets(k));
                }
            }
        });

        if (!geometry)
            return input;
        input.r = w.make(e, 3);
        const auto pos = device.pos, global_cells = device.cell;
        const auto frame = device.batch;
        const bool shared = device.shared_cell, periodic = device.periodic;
        const auto images = shifts;
        if (w.jit) {
            if (periodic)
                input.shifts = {images.data(), e, 3};
        } else {
            const auto displacement = input.r;
            for_each(w.exec, "sp2::gather_edge_geometry", e, KOKKOS_LAMBDA(int k) {
                const int i = start + centers(k), j = targets(k);
                const auto image = 3ull * k;
                const auto cell_start = 9ull * (shared ? 0 : frame(i));
                for (int c = 0; c < 3; ++c) {
                    T r = pos(3ull * j + c) - pos(3ull * i + c);
                    if (periodic) {
                        const T x = cuda_rounded_product(images(image), global_cells(cell_start + c));
                        const T y = cuda_rounded_product(images(image + 1), global_cells(cell_start + 3 + c));
                        const T z = cuda_rounded_product(images(image + 2), global_cells(cell_start + 6 + c));
                        r += (x + y) + z;
                    }
                    displacement(k, c) = r;
                }
            });
        }
        return input;
    }

    void gather_spin(Workspace<T> &w, TileInput<T> &input, const DeviceGraph<T> &device,
                     const DeviceArray<T> &spin, int start, T constant, bool mean) const {
        const auto n = input.ncenters;
        const int e = input.center.extent_int(0);
        const auto frame = device.batch;
        const auto counts = device.counts;
        input.atomic_seed = w.make(n, 1);
        const auto seed = input.atomic_seed;
        for_each(w.exec, "sp2::gather_energy_seed", n, KOKKOS_LAMBDA(int i) {
            seed(i, 0) = mean ? constant / T(counts(frame(start + i))) : constant;
        });
        if (!spin.data()) return;
        input.graph_spins = spin.data();
        input.spins_i = w.make(e, 3);
        input.spins_j = w.make(e, 3);
        input.center_spins = {spin.data() + 3ull * start, n, 3};
        const auto si = input.spins_i, sj = input.spins_j;
        const auto centers = input.center, targets = target;
        for_each(w.exec, "sp2::gather_spins", e, KOKKOS_LAMBDA(int k) {
            const auto i = 3ull * (start + centers(k)), j = 3ull * targets(k);
            put3(si, k, {spin(i), spin(i + 1), spin(i + 2)});
            put3(sj, k, {spin(j), spin(j + 1), spin(j + 2)});
        });
    }

    std::size_t bytes() const {
        return sizeof(int) * (target.extent(0) + center.extent(0) + species_j.extent(0) +
                              offsets.extent(0)) +
               sizeof(T) * shifts.extent(0);
    }

    static KOKKOS_INLINE_FUNCTION T cuda_rounded_product(T x, T y) {
#ifdef __CUDA_ARCH__
        if constexpr (std::is_same_v<T, float>)
            return __fmul_rn(x, y);
        else
            return __dmul_rn(x, y);
#else
        return x * y;
#endif
    }
};

template <class T>
void device_scatter(Workspace<T> &w, int start, const TileInput<T> &input, const TileResult<T> &tile,
                    const DeviceArray<int> &targets, T *force, T *spin, T *virial) {
    ProfileScope<T> profile(w, Profile::reverse);
    const auto offsets = input.offsets;
    const auto r = tile.r, length = tile.length, dr0 = tile.dr0, dl = tile.dl, dr = tile.dr;
    const auto di = tile.ds_i, dj = tile.ds_j;
    if (w.jit) {
        const int blocks = input.ncenters ? (input.ncenters - 1) / 4 + 1 : 0;
        w.jit->launch_grid(w.exec, "sp2_geom_scatter", blocks, 128, 0, input.ncenters, start,
                           targets.data(), offsets.data(), r.ptr, length.ptr, dr0.ptr, dl.ptr,
                           di.ptr, dj.ptr, force, spin, virial);
        return;
    }
    for_each(w.exec, "sp2::atomic_center_adjoint", input.ncenters, KOKKOS_LAMBDA(int row) {
        const int i = start + row;
        T df[3]{}, ds[3]{}, va[9]{};
        for (int e = offsets(row); e < offsets(row + 1); ++e) {
            const int j = targets(e);
            if (spin) {
                for (int c = 0; c < 3; ++c) {
                    if (di) ds[c] -= di(e, c);
                    if (!dj) continue;
                    if (i == j) ds[c] -= dj(e, c);
                    else Kokkos::atomic_add(spin + 3ull * j + c, -dj(e, c));
                }
            }
            if (force) {
                for (int c = 0; c < 3; ++c) {
                    const T value = dr(e, c);
                    if (i != j) {
                        df[c] += value;
                        Kokkos::atomic_add(force + 3ull * j + c, -value);
                    }
                    for (int k = 0; k < 3; ++k)
                        va[3 * c + k] -= DeviceEdges<T>::cuda_rounded_product(value, r(e, k));
                }
            }
        }
        for (int c = 0; c < 3; ++c) {
            if (force) Kokkos::atomic_add(force + 3ull * i + c, df[c]);
            if (spin) Kokkos::atomic_add(spin + 3ull * i + c, ds[c]);
        }
        if (virial) for (int c = 0; c < 9; ++c) virial[9ull * i + c] = va[c];
    });
}
} // namespace sp2
#endif
