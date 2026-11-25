#pragma once

#ifdef SP2_ENABLE_CUDA
#include "device_graph_replay.hpp"
namespace sp2 {

// The skin topology survives LAMMPS neighbor rebuild intervals. Model inputs
// use compact active-edge storage (sized to ~1.125x E_active, not E_skin) with
// optional 256-edge bucket padding so CUDA Graphs replay across cutoff crossings.
class NeighborTopology {
    static constexpr int bucket_granularity = 256;
    static constexpr int bucket_headroom = 128;

    DeviceArray<int> counts, skin_targets, active_rows, active_boundaries;
    DeviceArray<int> targets, offsets, types;
    DeviceArray<int> tile_centers, tile_neighbor_types;
    DeviceArray<int> slot_begins, bucket_caps, overflow_flag;
    DeviceArray<int64_t> rows, stats;
    Kokkos::View<int *, Kokkos::CudaHostPinnedSpace> copied_boundaries;
    Kokkos::View<int *, Kokkos::CudaHostPinnedSpace> copied_overflow;
    std::vector<int> host_boundaries;
    std::vector<int> host_slot_begins, host_buckets;
    int max_degree = 0;
    int requested_width = -1, width = 0, profile = 0, tiles = 0;
    int last_peak_actual_edges = 0;
    size_t planner_budget = 0;
    bool valid = false;
    bool buckets_valid = false;
    bool bucketed_mode = false;
    size_t last_active_meta_bytes = 0;
    int atoms = 0, owners = 0;
    uint64_t epoch = 0;
    std::array<const int *, 5> source_arrays{};
    std::array<int64_t, 7> source_layout{};

    template<class V>
    static void reserve(const Exec &exec, DeviceArray<V> &buffer, size_t size, const char *label) {
        if (!buffer.data() || buffer.extent(0) < size)
            buffer = cuda_array<V>(exec, size, label);
    }

    template<class V>
    static bool reserve_growth(const Exec &exec, DeviceArray<V> &buffer, size_t size, const char *label) {
        if (!buffer.data() || buffer.extent(0) < size) {
            const size_t alloc = std::max<size_t>(1, size + (size >> 3));
            buffer = cuda_array<V>(exec, alloc, label);
            return true;
        }
        return false;
    }

    template<class V>
    static void metadata(const Exec &exec, V *host, const DeviceArray<V> &source, size_t size) {
        cuda_check(cudaMemcpyAsync(host, source.data(), size * sizeof(V),
                                   cudaMemcpyDeviceToHost, exec.cuda_stream()), "Read neighbor metadata");
        cuda_check(cudaStreamSynchronize(exec.cuda_stream()), "Complete neighbor metadata");
    }

    static int align_bucket(int actual) {
        if (actual <= 0) return 0;
        const int target = actual + bucket_headroom;
        return ((target + bucket_granularity - 1) / bucket_granularity) * bucket_granularity;
    }

  public:
    bool prepare(const Exec &exec, const sp2_neighbors &source, int model_types,
                 ExecutionPlan &plan, sp2_options &o) {
        const auto in = source;
        const std::array<const int *, 5> arrays{in.types, in.type_map, in.ilist,
                                               in.numneigh, in.firstneigh};
        const std::array<int64_t, 7> layout{in.type_map_size, in.neighbor_mask,
                                           in.neighbor_slot_capacity, in.ilist_stride,
                                           in.numneigh_stride, in.neighbor_atom_stride,
                                           in.neighbor_slot_stride};
        if (valid && atoms == in.num_atoms && owners == in.num_owned && epoch == in.topology_epoch &&
            requested_width == o.tile_atoms && profile == o.execution_profile &&
            source_arrays == arrays && source_layout == layout) {
            o.tile_atoms = plan.width = width;
            plan.direct = true;
            plan.profile = profile;
            plan.budget = planner_budget;
            return false;
        }
        valid = false;
        buckets_valid = false;
        last_peak_actual_edges = 0;
        last_active_meta_bytes = 0;
        requested_width = o.tile_atoms;
        profile = o.execution_profile;
        atoms = in.num_atoms;
        owners = in.num_owned;
        epoch = in.topology_epoch;
        source_arrays = arrays;
        source_layout = layout;
        reserve(exec, counts, owners, "neighbor row lengths");
        reserve(exec, types, atoms, "mapped atom types");
        reserve(exec, rows, size_t(owners) + 1, "neighbor prefix sum");
        reserve(exec, stats, 3, "neighbor metadata");
        reserve(exec, overflow_flag, 2, "bucket overflow flag");
        if (copied_overflow.extent(0) < 2)
            copied_overflow = decltype(copied_overflow)(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, "host bucket overflow"), 2);

        const auto degree = counts;
        const auto row = rows, device_stats = stats;
        const auto species = types;
        Kokkos::deep_copy(exec, degree, -1);
        Kokkos::deep_copy(exec, device_stats, int64_t(0));
        for_each(exec, "sp2::neighbor_types", atoms, KOKKOS_LAMBDA(int i) {
            const int type = in.types[i];
            if (type < 0 || type >= in.type_map_size ||
                in.type_map[type] < 0 || in.type_map[type] >= model_types) {
                Kokkos::atomic_max(device_stats.data(), int64_t(1));
                species(i) = 0;
            } else {
                species(i) = in.type_map[type];
            }
        });
        for_each(exec, "sp2::neighbor_rows", owners, KOKKOS_LAMBDA(int ii) {
            const int i = in.ilist[ii * in.ilist_stride];
            if (i < 0 || i >= in.num_owned) {
                Kokkos::atomic_max(device_stats.data(), int64_t(1));
                return;
            }
            const int n = in.numneigh[i * in.numneigh_stride];
            if (n < 0 || n > in.neighbor_slot_capacity ||
                Kokkos::atomic_exchange(degree.data() + i, n) != -1) {
                Kokkos::atomic_max(device_stats.data(), int64_t(1));
                return;
            }
            Kokkos::atomic_max(device_stats.data() + 1, int64_t(n));
        });
        using ScanRange = Kokkos::RangePolicy<Exec, Kokkos::IndexType<int64_t>>;
        Kokkos::parallel_scan("sp2::neighbor_prefix_sum", ScanRange(exec, 0, int64_t(owners) + 1),
                              KOKKOS_LAMBDA(int64_t i, int64_t &n, bool last) {
            if (last) row(i) = n;
            if (i < in.num_owned) {
                const int value = degree(i);
                if (value > 0) n += value;
            } else if (last) {
                device_stats(2) = n;
            }
        });
        std::array<int64_t, 3> host_stats{};
        metadata(exec, host_stats.data(), stats, host_stats.size());
        require(!host_stats[0], "Invalid CUDA neighbor rows or atom types");
        require(host_stats[2] <= std::numeric_limits<int>::max(), "Too many neighbors on one device");
        max_degree = host_stats[1];
        plan.select(o, owners, max_degree);
        planner_budget = plan.budget;
        width = o.tile_atoms;
        tiles = owners ? (owners - 1) / width + 1 : 0;
        const int edges = host_stats[2];
        reserve(exec, skin_targets, edges, "skin neighbors");
        reserve(exec, active_rows, size_t(owners) + 1, "active neighbor prefix sum");
        reserve(exec, active_boundaries, size_t(tiles) + 1, "active tile boundaries");
        reserve(exec, slot_begins, size_t(tiles) + 1, "bucket tile begins");
        reserve(exec, bucket_caps, std::max(1, tiles), "bucket tile capacities");
        reserve(exec, offsets, size_t(owners) + tiles, "active tile CSR offsets");
        if (copied_boundaries.extent(0) < size_t(tiles) + 1)
            copied_boundaries = decltype(copied_boundaries)(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, "host active tile boundaries"),
                size_t(tiles) + 1);
        host_boundaries.assign(size_t(tiles) + 1, 0);
        host_slot_begins.assign(size_t(tiles) + 1, 0);
        host_buckets.assign(size_t(tiles), 0);
        const auto target = skin_targets;
        for_each(exec, "sp2::neighbor_pack", owners, KOKKOS_LAMBDA(int i) {
            const int begin = row(i), end = row(i + 1);
            for (int k = begin; k < end; ++k) {
                const int j = in.firstneigh[i * in.neighbor_atom_stride +
                                            (k - begin) * in.neighbor_slot_stride] & in.neighbor_mask;
                if (j < 0 || j >= in.num_atoms) {
                    Kokkos::atomic_max(device_stats.data(), int64_t(1));
                    continue;
                }
                target(k) = j;
            }
        });
        metadata(exec, host_stats.data(), stats, 1);
        require(!host_stats[0], "Invalid CUDA neighbor endpoint");
        valid = true;
        return true;
    }

    int size() const { return tiles; }
    uint64_t peak_actual_edges() const { return static_cast<uint64_t>(last_peak_actual_edges); }
    std::size_t metadata_bytes(bool rebuilt, bool refreshed) const {
        const auto skin_bytes = rebuilt ? 4 * sizeof(int64_t) : 0;
        const auto active_bytes = refreshed && owners ? last_active_meta_bytes : 0;
        return skin_bytes + active_bytes;
    }

    template<class T>
    bool compact(const Exec &exec, const T *positions, T cutoff, bool use_buckets) {
        if (!owners) return false;
        const auto row = rows;
        const auto skin = skin_targets, degree = counts;
        const T cutoff_sq = cutoff * cutoff;
        for_each(exec, "sp2::mark_physical_neighbors", owners, KOKKOS_LAMBDA(int i) {
            const auto a = positions + 3ull * i;
            const Vec3<T> xi{a[0], a[1], a[2]};
            int n = 0;
            for (int k = row(i); k < row(i + 1); ++k) {
                const auto b = positions + 3ull * skin(k);
                const Vec3<T> d{b[0] - xi.x, b[1] - xi.y, b[2] - xi.z};
                n += (d.dot(d) < cutoff_sq);
            }
            degree(i) = n;
        });
        const auto compact_row = active_rows;
        const auto tile_bounds = active_boundaries;
        const int num_owners = owners, tile_width = width, num_tiles = tiles;
        Kokkos::parallel_scan("sp2::physical_neighbor_prefix_sum",
                              Kokkos::RangePolicy<Exec>(exec, 0, num_owners + 1),
                              KOKKOS_LAMBDA(int i, int &n, bool last) {
            if (last) {
                compact_row(i) = n;
                if (i == num_owners) tile_bounds(num_tiles) = n;
                else if (i % tile_width == 0) tile_bounds(i / tile_width) = n;
            }
            if (i < num_owners) n += degree(i);
        });

        bool layout_changed = false;
        if (use_buckets && buckets_valid && bucketed_mode) {
            // Steady-bucket fast path: check 8-byte {overflow, peak_actual} device summary.
            cuda_check(cudaMemsetAsync(overflow_flag.data(), 0, 2 * sizeof(int), exec.cuda_stream()),
                       "Clear bucket overflow flag");
            const auto caps = bucket_caps;
            const auto ovf = overflow_flag;
            for_each(exec, "sp2::check_tile_buckets", num_tiles, KOKKOS_LAMBDA(int t) {
                const int actual = tile_bounds(t + 1) - tile_bounds(t);
                const int cap = caps(t);
                Kokkos::atomic_max(ovf.data() + 1, actual);
                if (actual > cap || (actual == 0 && cap > 0) ||
                    (cap > bucket_granularity && cap - actual > 3 * bucket_granularity))
                    ovf(0) = 1;
            });
            metadata(exec, copied_overflow.data(), overflow_flag, 2);
            last_active_meta_bytes = 2 * sizeof(int);
            last_peak_actual_edges = copied_overflow(1);
            if (copied_overflow(0) != 0) {
                buckets_valid = false;
            }
        }

        if (!use_buckets || !buckets_valid || bucketed_mode != use_buckets) {
            metadata(exec, copied_boundaries.data(), active_boundaries, host_boundaries.size());
            last_active_meta_bytes = host_boundaries.size() * sizeof(int);
            const auto next = copied_boundaries.data();
            const bool exact_changed = !std::equal(host_boundaries.begin(), host_boundaries.end(), next);
            std::copy_n(next, host_boundaries.size(), host_boundaries.begin());

            std::vector<int> new_buckets(size_t(num_tiles), 0);
            std::vector<int> new_begins(size_t(num_tiles) + 1, 0);
            int peak_bucket = 0;
            int peak_actual = 0;
            for (int t = 0; t < num_tiles; ++t) {
                const int actual = host_boundaries[t + 1] - host_boundaries[t];
                const int cap = use_buckets ? align_bucket(actual) : actual;
                new_buckets[t] = cap;
                new_begins[t + 1] = new_begins[t] + cap;
                peak_bucket = std::max(peak_bucket, cap);
                peak_actual = std::max(peak_actual, actual);
            }
            last_peak_actual_edges = peak_actual;
            const bool buckets_changed = !buckets_valid || bucketed_mode != use_buckets ||
                                         new_buckets != host_buckets;
            layout_changed = use_buckets ? buckets_changed : exact_changed;
            host_buckets = std::move(new_buckets);
            host_slot_begins = std::move(new_begins);
            bucketed_mode = use_buckets;
            buckets_valid = true;

            const bool realloc_t = reserve_growth(exec, targets, size_t(host_slot_begins[num_tiles]),
                                                  "active neighbors");
            const bool realloc_c = reserve_growth(exec, tile_centers, size_t(peak_bucket),
                                                  "tile edge centers");
            const bool realloc_s = reserve_growth(exec, tile_neighbor_types, size_t(peak_bucket),
                                                  "tile neighbor types");
            if (realloc_t || realloc_c || realloc_s) layout_changed = true;

            cuda_check(cudaMemcpyAsync(slot_begins.data(), host_slot_begins.data(),
                                       host_slot_begins.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, exec.cuda_stream()),
                       "Upload bucket slot begins");
            if (num_tiles)
                cuda_check(cudaMemcpyAsync(bucket_caps.data(), host_buckets.data(),
                                           host_buckets.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, exec.cuda_stream()),
                           "Upload bucket capacities");
        }

        const auto csr = offsets, target = targets, sb = slot_begins;
        for_each(exec, "sp2::pack_physical_neighbors", owners, KOKKOS_LAMBDA(int i) {
            const int tile = i / tile_width, start = tile * tile_width;
            const int first = compact_row(start);
            const int local = compact_row(i) - first;
            int dst = sb(tile) + local;
            const auto offset = size_t(i) + tile;
            csr(offset) = local;
            const auto a = positions + 3ull * i;
            const Vec3<T> xi{a[0], a[1], a[2]};
            for (int k = row(i); k < row(i + 1); ++k) {
                const int j = skin(k);
                const auto b = positions + 3ull * j;
                const Vec3<T> d{b[0] - xi.x, b[1] - xi.y, b[2] - xi.z};
                if (d.dot(d) < cutoff_sq) {
                    target(dst) = j;
                    ++dst;
                }
            }
            if (i + 1 == num_owners || (i + 1) % tile_width == 0)
                csr(offset + 1) = dst - sb(tile);
        });
        return layout_changed;
    }

    template<class T>
    TileInput<T> tile(const Exec &exec, int index, const T *positions, T cutoff, bool spline) const {
        const auto begin = host_slot_begins[index], end = host_slot_begins[index + 1];
        const int cap = end - begin;
        const int start = index * width, n = std::min(width, owners - start);
        const auto offset = size_t(start) + index;

        TileInput<T> input{};
        input.center = Kokkos::subview(tile_centers, std::make_pair(0, cap));
        input.species_j = Kokkos::subview(tile_neighbor_types, std::make_pair(0, cap));
        input.targets = Kokkos::subview(targets, std::make_pair(begin, end));
        input.species = Kokkos::subview(types, std::make_pair(start, start + n));
        input.offsets = Kokkos::subview(offsets, std::make_pair(offset, offset + n + 1));
        input.start = start;
        input.ncenters = n;
        input.spline = spline;
        input.positions = positions;
        input.cutoff = cutoff;

        // Populate tile-local center indices and neighbor species inside the
        // compute stream (graph-capture safe), initializing padded sentinel slots.
        const auto csr = input.offsets, c_out = input.center, sj_out = input.species_j;
        const auto t_view = input.targets, species = types;
        for_each(exec, "sp2::fill_tile_meta", n + 1, KOKKOS_LAMBDA(int row) {
            if (row < n) {
                for (int k = csr(row); k < csr(row + 1); ++k) {
                    c_out(k) = row;
                    sj_out(k) = species(t_view(k));
                }
            } else {
                for (int k = csr(n); k < cap; ++k) {
                    c_out(k) = 0;
                    sj_out(k) = 0;
                    t_view(k) = start;
                }
            }
        });
        return input;
    }

    size_t bytes() const {
        return sizeof(int) * (counts.extent(0) + skin_targets.extent(0) + active_rows.extent(0) +
                              active_boundaries.extent(0) + targets.extent(0) +
                              tile_centers.extent(0) + tile_neighbor_types.extent(0) +
                              slot_begins.extent(0) + bucket_caps.extent(0) +
                              overflow_flag.extent(0) + offsets.extent(0) + types.extent(0)) +
               sizeof(int64_t) * (rows.extent(0) + stats.extent(0));
    }
};

template<class T>
struct NeighborWorkspace {
    struct Saved {
        std::unique_ptr<Workspace<T>> workspace;
        TileState<T> state;
        bool forward_only{};
    };

    struct StreamDeleter {
        void operator()(cudaStream_t stream) const { cudaStreamDestroy(stream); }
    };
    using Stream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, StreamDeleter>;
    Stream compute_stream;
    std::unique_ptr<Workspace<T>> compute;
    GraphBlas graph_blas;
    std::array<NeighborGraph, 2> graphs;
    NeighborTopology topology;
    DeviceArray<T> position_storage, spin_storage, results;
    std::vector<Saved> saved;
    std::shared_ptr<const JitPlugin> artifact;
    uint64_t lattice_epoch = 0;
    uint64_t cache_limit = 0;
    uint32_t math = SP2_MATH_IEEE;
    int radial = SP2_ANALYTIC;
    bool cache_forward_only{};
    size_t cache_bytes = 0, cache_budget = 0;
    cudaEvent_t handoff = nullptr;

    ~NeighborWorkspace() {
        if (compute) cudaStreamSynchronize(compute->exec.cuda_stream());
        reset_graphs();
        if (handoff) cudaEventDestroy(handoff);
    }

    void reset_graphs() {
        for (auto &graph : graphs) graph.reset();
    }

    Workspace<T> &workspace() {
        if (!compute) {
            // A model-local stream prevents another model's launches from
            // joining a capture running concurrently on the Kokkos default.
            cudaStream_t stream = nullptr;
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "Create device neighbor stream");
            Stream owner(stream);
            compute = std::make_unique<Workspace<T>>(Exec(stream));
            compute_stream = std::move(owner);
        }
        return *compute;
    }

    void order(cudaStream_t from, cudaStream_t to) {
        if (from == to) return;
        if (!handoff)
            cuda_check(cudaEventCreateWithFlags(&handoff, cudaEventDisableTiming),
                       "Create inference stream handoff");
        cuda_check(cudaEventRecord(handoff, from), "Record inference stream handoff");
        cuda_check(cudaStreamWaitEvent(to, handoff, 0), "Wait for inference stream handoff");
    }

    template<class Source>
    const T *positions(const Exec &exec, sp2_neighbors in) {
        const auto xyz = static_cast<const Source *>(in.positions);
        if constexpr (std::is_same_v<T, Source>) {
            if (in.position_atom_stride == 3 && in.position_component_stride == 1)
                return xyz;
        }
        if (position_storage.extent(0) < 3ull * in.num_atoms)
            position_storage = cuda_array<T>(exec, 3ull * in.num_atoms, "LAMMPS positions");
        const auto storage = position_storage;
        for_each(exec, "sp2::convert_positions", in.num_atoms, KOKKOS_LAMBDA(int i) {
            for (int k = 0; k < 3; ++k)
                storage(3ull * i + k) = T(xyz[i * in.position_atom_stride + k * in.position_component_stride]);
        });
        return storage.data();
    }

    template<class Source>
    const T *spins(const Exec &exec, sp2_neighbors in) {
        if (!in.spins) return nullptr;
        const auto src = static_cast<const Source *>(in.spins);
        if (spin_storage.extent(0) < 3ull * in.num_atoms)
            spin_storage = cuda_array<T>(exec, 3ull * in.num_atoms, "LAMMPS spins");
        const auto dst = spin_storage;
        for_each(exec, "sp2::convert_spins", in.num_atoms, KOKKOS_LAMBDA(int i) {
            const auto start = i * in.spin_atom_stride;
            const bool active = src[start + 3 * in.spin_component_stride] > Source(0);
            for (int k = 0; k < 3; ++k)
                dst(3ull * i + k) = active ? T(src[start + k * in.spin_component_stride]) : T(0);
        });
        return dst.data();
    }
};

template<class T>
void gather_neighbors_spin(Workspace<T> &w, TileInput<T> &input, const T *spin, T seed_value) {
    const int n = input.ncenters, e = input.center.extent_int(0), start = input.start;
    input.atomic_seed = w.make(n, 1);
    const auto seed = input.atomic_seed;
    for_each(w.exec, "sp2::partition_energy_seed", n, KOKKOS_LAMBDA(int i) {
        seed(i, 0) = seed_value;
    });
    if (!spin) return;
    input.graph_spins = spin;
    input.spins_i = w.make(e, 3);
    input.spins_j = w.make(e, 3);
    input.center_spins = {const_cast<T *>(spin) + 3ull * start, n, 3};
    const auto si = input.spins_i, sj = input.spins_j;
    const auto centers = input.center, targets = input.targets, offsets = input.offsets;
    for_each(w.exec, "sp2::neighbor_spins", e, KOKKOS_LAMBDA(int k) {
        if (k >= offsets(n)) {
            put3(si, k, {});
            put3(sj, k, {});
            return;
        }
        const auto i = 3ull * (start + centers(k)), j = 3ull * targets(k);
        put3(si, k, {spin[i], spin[i + 1], spin[i + 2]});
        put3(sj, k, {spin[j], spin[j + 1], spin[j + 2]});
    });
}

} // namespace sp2
#endif
