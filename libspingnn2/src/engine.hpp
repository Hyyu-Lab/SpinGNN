#pragma once
#include "contract.hpp"
#include "graph.hpp"
#include "tile.hpp"
#include "execution_plan.hpp"
#include "endpoint_reduction.hpp"
#include "device_graph.hpp"
#include "device_neighbors.hpp"
#include <map>
#include <memory>
#include <mutex>
#include <variant>
#ifdef SP2_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace sp2 {
using Numbers = std::variant<std::vector<float>, std::vector<double>>;
struct Field {
    std::vector<int64_t> shape;
    Numbers data;
};
struct Result {
    std::map<std::string, Field> fields;
    sp2_diagnostics diagnostics{};
    template <class T> std::vector<T> &field(const std::string &name, std::vector<int64_t> shape) {
        auto it = fields.find(name);
        if (it == fields.end()) {
            size_t size = 1;
            for (auto dim : shape)
                size *= dim;
            it = fields.emplace(name, Field{shape, std::vector<T>(size)}).first;
        }
        if (it->second.shape != shape)
            throw std::logic_error("Inconsistent output shape: " + name);
        return std::get<std::vector<T>>(it->second.data);
    }
    template <class T> std::vector<T> &data(const std::string &key) {
        return std::get<std::vector<T>>(fields.at(key).data);
    }
};

struct CacheBase {
    virtual ~CacheBase() = default;
};
struct Engine {
    virtual ~Engine() = default;
    virtual std::unique_ptr<Result> evaluate(const sp2_graph &, const sp2_options &) = 0;
#ifdef SP2_ENABLE_CUDA
    virtual sp2_device_result evaluate_neighbors(const sp2_neighbors &, const sp2_options &) = 0;
#endif
    virtual std::unique_ptr<CacheBase> stage1(const sp2_graph &, const sp2_options &) = 0;
    virtual std::unique_ptr<Result> cached(CacheBase &, const void *, const uint32_t *) = 0;
    virtual void install(const std::filesystem::path &path) = 0;
    ExecutionPlan plan;
    std::mutex mutex;
    std::string info;
};

template <class F>
auto time_stage(Exec &exec, bool enabled, double &seconds, F &&action) {
    if (!enabled) return action();
    exec.fence("start inference stage");
    const auto start = std::chrono::steady_clock::now();
    auto result = action();
    exec.fence("finish inference stage");
    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

template <class T> using HostSpan = Kokkos::View<T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using DeviceSpan = Kokkos::View<T *, Exec::memory_space, Kokkos::MemoryUnmanaged>;

template <class T> struct Buffers {
    Workspace<T> w;
    DeviceArray<int> center, offsets, species, species_j;
    std::vector<int> ci, csr, sj;
    std::vector<T> pos_j, centers_cell;
    EndpointReduction<T> endpoints;
    std::vector<T> r, spins_i, spins_j, center_spins, seed, dr, ds_i, ds_j;
    int edge_capacity{}, atom_capacity{};

    void reserve(int edges, int atoms) {
        if (edges > edge_capacity || !center.data()) {
            edge_capacity = std::max(edges, 1);
            center = DeviceArray<int>("local center", edge_capacity);
            species_j = DeviceArray<int>("neighbor type", edge_capacity);
        }
        if (atoms > atom_capacity) {
            atom_capacity = atoms;
            offsets = DeviceArray<int>("local CSR", atoms + 1);
            species = DeviceArray<int>("center type", atoms);
        }
    }
    template <class V> void send(V *device, const V *host, size_t n) {
        if (n)
            Kokkos::deep_copy(w.exec, DeviceSpan<V>(device, n), HostSpan<const V>(host, n));
    }
    void receive(T *host, Tensor<T> tensor) {
        if (tensor.size())
            Kokkos::deep_copy(w.exec, HostSpan<T>(host, tensor.size()),
                              DeviceSpan<const T>(tensor.ptr, tensor.size()));
    }
    Tensor<T> tensor(const std::vector<T> &values, int cols) {
        auto view = w.make(values.size() / cols, cols);
        send(view.ptr, values.data(), values.size());
        return view;
    }
    // Persistent device index buffers let every cached pullback use the same
    // addresses. Restore their contents for this tile before invoking its tape.
    TileInput<T> prepare(const Graph<T> &g, int start, int stop, bool spline, bool make_r) {
        const int n = stop - start, first = g.offsets[start], e = g.offsets[stop] - first;
        reserve(e, n);
        ci.resize(e);
        sj.resize(e);
        csr.resize(n + 1);
        if (make_r) {
            if (w.jit)
                pos_j.resize(3ull * e);
            else
                r.resize(3ull * e);
        }
        for (int i = start; i < stop; ++i) {
            csr[i - start] = g.offsets[i] - first;
            for (int k = g.offsets[i]; k < g.offsets[i + 1]; ++k) {
                int p = k - first;
                ci[p] = i - start;
                sj[p] = g.types[g.target[k]];
                if (make_r) {
                    if (w.jit)
                        std::copy_n(g.pos.data() + 3ull * g.target[k], 3, pos_j.data() + 3ull * p);
                    else
                        g.displacement(i, k, r.data() + 3ull * p);
                }
            }
        }
        csr[n] = e;
        send(center.data(), ci.data(), e);
        send(offsets.data(), csr.data(), n + 1);
        send(species.data(), g.types.data() + start, n);
        send(species_j.data(), sj.data(), e);
        TileInput<T> input{};
        input.center = Kokkos::subview(center, std::make_pair(0, e));
        input.offsets = Kokkos::subview(offsets, std::make_pair(0, n + 1));
        input.species = Kokkos::subview(species, std::make_pair(0, n));
        input.species_j = Kokkos::subview(species_j, std::make_pair(0, e));
        input.ncenters = n;
        input.spline = spline;
        if (make_r && w.jit) {
            input.pos_i = w.make(n, 3);
            send(input.pos_i.ptr, g.pos.data() + 3ull * start, 3ull * n);
            input.pos_j = tensor(pos_j, 3);
            if (!g.cell.empty()) {
                centers_cell.resize(9ull * n);
                for (int p = 0; p < n; ++p)
                    std::copy_n(g.cell.data() +
                                    9ull * (g.cell.size() == 9 ? 0 : g.batch[start + p]),
                                9, centers_cell.data() + 9ull * p);
                input.cells = tensor(centers_cell, 9);
                input.shifts = w.make(e, 3);
                if (e)
                    send(input.shifts.ptr, g.shifts.data() + 3ull * first, 3ull * e);
            }
            input.r = w.make(e, 3);
        } else if (make_r)
            input.r = tensor(r, 3);
        return input;
    }
    void spins(TileInput<T> &input, const Graph<T> &g, int start, const T *spin, T constant,
               bool mean) {
        int n = input.ncenters, e = input.center.extent_int(0);
        seed.resize(n);
        for (int i = 0; i < n; ++i)
            seed[i] = mean ? constant / T(g.counts[g.batch[start + i]]) : constant;
        input.atomic_seed = tensor(seed, 1);
        if (!spin)
            return;
        spins_i.resize(3ull * e);
        spins_j.resize(3ull * e);
        center_spins.resize(3ull * n);
        std::copy_n(spin + 3ull * start, 3ull * n, center_spins.data());
        const int first = g.offsets[start];
        for (int i = start; i < start + n; ++i) {
            for (int k = g.offsets[i]; k < g.offsets[i + 1]; ++k) {
                const auto dst = 3ull * (k - first);
                std::copy_n(spin + 3ull * i, 3, spins_i.data() + dst);
                std::copy_n(spin + 3ull * g.target[k], 3, spins_j.data() + dst);
            }
        }
        input.spins_i = tensor(spins_i, 3);
        input.spins_j = tensor(spins_j, 3);
        input.center_spins = tensor(center_spins, 3);
    }
    size_t bytes() const {
        return w.bytes() + endpoints.bytes() +
               sizeof(int) * (2ull * edge_capacity + 2ull * atom_capacity + 1);
    }
};

template <class T> struct Lattice : CacheBase {
    struct Saved {
        // Stage 2 borrows the retained fields or tape. Its temporary storage
        // comes from the shared workspace so replay cannot overwrite the cache.
        std::unique_ptr<Workspace<T>> resident;
        TileState<T> state;
        typename Workspace<T>::Snapshot spill;
        TileInput<T> input;
        bool forward_only{};
#ifdef SP2_ENABLE_CUDA
        std::unique_ptr<typename Workspace<T>::CudaTape> cuda_spill;
        std::vector<std::size_t> cuda_shape;
        DeviceEdges<T> saved_indices;
#endif
    };
    Graph<T> graph;
    sp2_options options;
    Buffers<T> buf;
    std::vector<Saved> saved;
    size_t host_bytes{}, device_bytes{}, spill_slots{};
    std::shared_ptr<const JitPlugin> plugin;
#ifdef SP2_ENABLE_CUDA
    DeviceGraph<T> device;
    DeviceEdges<T> spill_indices;
    PinnedArray<T> pinned_tape;
#endif
    Lattice(const sp2_graph &g, int types, const sp2_options &o) : graph(g, types), options(o) {}
};

template <class T> class SpinGNN2Kokkos final : public Engine {
  public:
    struct Rescale {
        T scale{1}, shift{};
    };

    explicit SpinGNN2Kokkos(const Contract &contract)
        : program(contract, contract.model, work.w.exec), model(contract.model) {
        plan.precision = sizeof(T) * 8;
        plan.contract_digest = contract.digest();
        plan.size_model(contract);
        const std::string family = model.at("family");
        magnetic =
            family == "Allegro_nosoc" || family == "Allegro_soc" || family == "Allegro_soc_e2IP";
        Json meta = {{"family", family},
                     {"precision", sizeof(T) == 4 ? "float32" : "float64"},
                     {"magnetic", magnetic}};
        for (const auto &l : model.at("layers")) {
            std::string op = l.at("op");
            if (op.find("OneHotAtomEncoding") == 0)
                types = l.at("num_types");
            if (l.contains("basis")) {
                const auto &b = l.at("basis");
                if (b.contains("cutoff"))
                    meta["r_max"] = std::max(meta.value("r_max", 0.0),
                                             1. / b.at("cutoff").at("factor").get<double>());
            }
            if (l.contains("backbone")) {
                const auto radii = contract.array<double>(l.at("backbone").at("cutoffs"));
                for (const double radius : radii)
                    meta["r_max"] = std::max(meta.value("r_max", 0.0), radius);
            }
            if (l.contains("type_names"))
                meta["type_names"] = l.at("type_names");
            if (op == "AtomwiseReduce") {
                constant = l.at("constant");
                mean = l.at("reduce") == "mean";
            }
            if (op == "CartesianBECReduce")
                bec = l;
            if (op == "e2IP_nodewise")
                prefixes = l.at("prefixes").get<std::vector<std::string>>();
        }
        require(types > 0, "Missing one-hot type encoding");
        meta["num_types"] = types;
        meta["prefixes"] = prefixes;
        if (!bec.is_null()) {
            meta["bec"] = bec.at("out_field");
            meta["bec_asr"] = bec.at("asr");
            bec_field = bec.at("out_field").get<std::string>();
            meta["bec_asr_shift"] = rescale(bec_field).shift;
        }
        cutoff = meta.value("r_max", 0.0);
        info = meta.dump();
    }
    void install(const std::filesystem::path &path) override {
        auto plugin = load_jit_artifact(path, plan.contract_digest, plan.precision);
        plugin->require_symbols(backbone_jit_symbols(model));
        plugin->require_symbols(geometry_jit_symbols<T>(model));
        plan.artifact = std::move(plugin);
        plan.direct = true;
    }
    std::unique_ptr<Result> evaluate(const sp2_graph &in, const sp2_options &options) override {
#ifdef SP2_ENABLE_CUDA
        Graph<T> g(in, types, true);
#else
        Graph<T> g(in, types);
#endif
        auto o = options;
        plan.select(o, g.offsets);
        work.w.jit = plan.direct ? plan.artifact.get() : nullptr;
#ifdef SP2_ENABLE_CUDA
        return run_device(g, o, static_cast<const T *>(in.spins), nullptr);
#else
        return run(g, o, work, static_cast<const T *>(in.spins), nullptr);
#endif
    }
    std::unique_ptr<CacheBase> stage1(const sp2_graph &in, const sp2_options &options) override {
        auto o = options;
        require(!o.eval_flags || magnetic, "Frozen spin evaluation requires a magnetic model");
        auto c = std::make_unique<Lattice<T>>(in, types, o);
        plan.select(o, c->graph.offsets);
        c->options = o;
        c->plugin = plan.direct ? plan.artifact : nullptr;
        if (!c->graph.owned)
            return c;
#ifdef SP2_ENABLE_CUDA
        populate_cuda_cache(*c);
        return c;
#else
        auto &b = c->buf;
        b.w.jit = c->plugin.get();
        auto &g = c->graph;
        const int width = o.tile_atoms;
        // Warm the reusable spill buffer to the maximal shape before recording
        // pointer-bearing pullbacks. All tiles except the last share the same
        // center count, so at most two actual shapes cover both maxima.
        int peak = 0;
        int warm_tile = 0;
        for (int start = 0; start < g.owned;) {
            int stop = start + std::min(width, g.owned - start);
            const int edges = g.offsets[stop] - g.offsets[start];
            if (edges > peak) {
                peak = edges;
                warm_tile = start;
            }
            start = stop;
        }
        b.reserve(peak, width);
        const bool spline = o.radial_mode == SP2_SPLINE;
        if (o.eval_flags & SP2_SKIP_LATTICE_FORCES) {
            for (int start = 0; start < g.owned;) {
                const int stop = start + std::min(width, g.owned - start);
                b.w.rewind(0);
                const auto input = b.prepare(g, start, stop, spline, true);
                const auto lattice = program.stage1(b.w, input);
                auto& saved = c->saved.emplace_back();
                saved.resident = std::make_unique<Workspace<T>>(b.w.exec);
                saved.forward_only = true;
                saved.state = program.cache_forward(*saved.resident, lattice);
                c->host_bytes += saved.resident->bytes();
                start = stop;
            }
            c->buf = Buffers<T>();
            return c;
        }
        auto warm = [&](int start) {
            const int stop = start + std::min(width, g.owned - start);
            b.w.rewind(0);
            const auto input = b.prepare(g, start, stop, spline, true);
            program.stage1(b.w, input);
            b.w.exec.fence("warm reusable lattice storage");
        };
        warm(warm_tile);
        if (warm_tile != 0 && g.owned - warm_tile < width)
            warm(0);
        b.w.freeze();
        c->spill_slots = b.w.mark();
        const auto budget = resident_budget(o);
        const auto max_tile_bytes = b.bytes();
        for (int start = 0; start < g.owned;) {
            const int stop = start + std::min(width, g.owned - start);
            typename Lattice<T>::Saved saved;
            // Reserve the warmed tile size, including scratch during creation.
            // The earlier tiles retain only the stable tape storage.
            if (max_tile_bytes <= budget - c->device_bytes) {
                Buffers<T> tile;
                tile.w.jit = c->plugin.get();
                saved.input = tile.prepare(g, start, stop, spline, true);
                saved.state = program.stage1(tile.w, saved.input);
                tile.w.seal();
                c->device_bytes += tile.bytes();
                saved.resident = std::make_unique<Workspace<T>>(std::move(tile.w));
            } else {
                b.w.rewind(0);
                saved.input = b.prepare(g, start, stop, spline, true);
                saved.state = program.stage1(b.w, saved.input);
                saved.spill = b.w.snapshot();
                for (const auto &slot : saved.spill)
                    c->host_bytes += slot.size() * sizeof(T);
            }
            c->saved.push_back(std::move(saved));
            start = stop;
        }
        if (!c->host_bytes)
            c->buf = Buffers<T>();
        return c;
#endif
    }
    std::unique_ptr<Result> cached(CacheBase &base, const void *spin,
                                   const uint32_t *flags) override {
        auto &c = dynamic_cast<Lattice<T> &>(base);
#ifndef SP2_ENABLE_CUDA
        auto &buffers = c.host_bytes ? c.buf : work;
#endif
        auto o = c.options;
        if (flags)
            o.eval_flags = *flags;
        require(!(o.eval_flags & ~uint32_t(SP2_SKIP_LATTICE_FORCES)),
                "Unsupported evaluation flag");
        require(!o.eval_flags || magnetic, "Frozen spin evaluation requires a magnetic model");
        require(!c.plugin || c.plugin == plan.artifact, "Cache artifact has been replaced");
        plan.direct = bool(c.plugin);
        plan.width = o.tile_atoms;
        plan.profile = o.execution_profile;
#ifdef SP2_ENABLE_CUDA
        return run_device(c.graph, o, static_cast<const T *>(spin), &c);
#else
        buffers.w.jit = c.plugin.get();
        return run(c.graph, o, buffers, static_cast<const T *>(spin), &c);
#endif
    }
#ifdef SP2_ENABLE_CUDA
    sp2_device_result evaluate_neighbors(const sp2_neighbors &, const sp2_options &) override;
#endif

  private:
    Buffers<T> work;
    TileProgram<T> program;
    Json model, bec;
    std::string bec_field;
    std::vector<std::string> prefixes;
    int types = 0;
    bool magnetic = false, mean = false;
    T constant = 1;
    T cutoff{};

#ifdef SP2_ENABLE_CUDA
    UploadRing upload_ring;
    std::array<DeviceEdges<T>, 2> staged_edges;
    DeviceArray<T> device_spin, device_result;
    PinnedArray<T> host_result;
    NeighborWorkspace<T> neighbor_work;

    void populate_cuda_cache(Lattice<T> &c);
    std::unique_ptr<Result> run_device(const Graph<T> &g, const sp2_options &o,
                                        const T *spin, Lattice<T> *cache);
#endif

    static std::size_t resident_budget(const sp2_options &o) {
#ifdef SP2_ENABLE_CUDA
        const auto asked = o.cache_device_limit_bytes;
        if (asked == std::numeric_limits<uint64_t>::max())
            return 0;
        std::size_t available = 0, total = 0;
        if (cudaMemGetInfo(&available, &total) != cudaSuccess)
            return 0;
        const auto &caps = device_caps();
        const auto default_cap = std::max<std::uint64_t>(4ull * 1024 * 1024 * 1024,
                                                         caps.total_mem / 4);
        const auto cap = asked ? asked : default_cap;
        return std::min<std::uint64_t>(cap, available / 4);
#else
        (void)o;
        return 0;
#endif
    }

    std::unique_ptr<Result> run(const Graph<T> &g, const sp2_options &o, Buffers<T> &b,
                                const T *spin, const Lattice<T> *cache) {
        require(!magnetic || spin || !g.n, "Magnetic contract requires spins");
        auto result = std::make_unique<Result>();
        auto &diag = result->diagnostics;
        diag.struct_size = sizeof(diag);
        diag.cache_host_bytes = cache ? cache->host_bytes : 0;
        diag.cache_device_bytes = cache ? cache->device_bytes : 0;
        const bool frozen = (o.eval_flags & SP2_SKIP_LATTICE_FORCES) != 0;
        require(!frozen || (cache && magnetic),
                "Frozen lattice derivatives require a magnetic stage-1 cache");
        auto *forces = frozen ? nullptr : &result->field<T>("forces", {g.n, 3});
        auto *va = frozen ? nullptr : &result->field<T>("atom_virial", {g.owned, 3, 3});
        auto *sf = magnetic ? &result->field<T>("spin_forces", {g.n, 3}) : nullptr;
        b.w.profile = o.profile != 0;
        b.w.seconds = {};
        for (int start = 0, index = 0; start < g.owned; ++index) {
            int n = std::min(o.tile_atoms, g.owned - start), first = g.offsets[start],
                e = g.offsets[start + n] - first;
            diag.peak_tile_edges = std::max<uint64_t>(diag.peak_tile_edges, e);
            b.w.rewind(0);
            TileInput<T> input{};
            TileState<T> state;
            const TileState<T> *current;
            const auto* saved_tile = cache ? &cache->saved.at(index) : nullptr;
            if (saved_tile && saved_tile->forward_only && frozen) {
                input = b.prepare(g, start, start + n, o.radial_mode == SP2_SPLINE, false);
                current = &saved_tile->state;
                input.r = current->fields.at("r");
                ++diag.stage1_cache_hits;
            } else if (saved_tile && !saved_tile->forward_only) {
                const auto &saved = *saved_tile;
                if (saved.resident) {
                    if (cache->host_bytes)
                        b.w.rewind(cache->spill_slots);
                    input = saved.input;
                } else {
                    input = b.prepare(g, start, start + n, o.radial_mode == SP2_SPLINE, false);
                    b.w.restore(saved.spill);
                    input.r = saved.input.r;
                }
                current = &saved.state;
                ++diag.stage1_cache_hits;
            } else {
                input = b.prepare(g, start, start + n, o.radial_mode == SP2_SPLINE, true);
                state = time_stage(b.w.exec, o.profile, diag.stage1_seconds,
                                   [&] { return program.stage1(b.w, input); });
                current = &state;
            }
            input.frozen = frozen;
            b.spins(input, g, start, spin, constant, mean);
            const auto tile = time_stage(b.w.exec, o.profile, diag.stage2_seconds,
                                         [&] { return program.stage2(b.w, input, *current); });
            if (!b.w.jit && !frozen) {
                b.dr.resize(3ull * e);
                b.receive(b.dr.data(), tile.dr);
            }
            if (!b.w.jit && sf) {
                b.ds_i.assign(3ull * e, T(0));
                b.ds_j.assign(3ull * e, T(0));
                if (tile.ds_i)
                    b.receive(b.ds_i.data(), tile.ds_i);
                if (tile.ds_j)
                    b.receive(b.ds_j.data(), tile.ds_j);
            }
            for (const auto &[key, v] : tile.atom) {
                bool matrix =
                    key == "f_sigma" || key == "sf_sigma" || key == "f_Uepi" || key == "sf_Uepi";
                auto shape =
                    matrix ? std::vector<int64_t>{g.owned, 3, 3}
                           : std::vector<int64_t>{g.owned, v.cols};
                require(v.rows == n, "Unexpected center output shape: " + key);
                auto &a = result->field<T>(key, std::move(shape));
                b.receive(a.data() + size_t(start) * v.cols, v);
            }
            if (b.w.jit) {
                b.endpoints.apply(b, g, start, input, tile, forces, sf, va);
                start += n;
                continue;
            }
            b.w.exec.fence("read center tile");
            for (int i = start; i < start + n; ++i) {
                for (int edge = g.offsets[i]; edge < g.offsets[i + 1]; ++edge) {
                    const int p = edge - first, j = g.target[edge];
                    T cached_r[3];
                    if (cache && !frozen)
                        g.displacement(i, edge, cached_r);
                    const T *r = cache ? cached_r : b.r.data() + 3ull * p;
                    for (int k = 0; k < 3; ++k) {

                        if (sf) {
                            (*sf)[3ull * i + k] -= b.ds_i[3ull * p + k];
                            (*sf)[3ull * j + k] -= b.ds_j[3ull * p + k];
                        }
                        if (!frozen) {
                            const T d = b.dr[3ull * p + k];
                            (*forces)[3ull * i + k] += d;
                            (*forces)[3ull * j + k] -= d;
                            for (int l = 0; l < 3; ++l)
                                (*va)[9ull * i + 3 * k + l] -= d * r[l];
                        }
                    }
                }
            }
            start += n;
        }
        diag.workspace_bytes = b.bytes();
        diag.geometry_seconds = b.w.seconds[0];
        diag.mlp_seconds = b.w.seconds[1];
        diag.tp_linear_seconds = b.w.seconds[2];
        diag.reverse_seconds = b.w.seconds[3];
        finish(*result, g, spin);
        return result;
    }
    Rescale rescale(const std::string &key) const {
        Rescale result;
        for (const auto &layer : model.at("rescales")) {
            const T scale = layer.at("scale"), shift = layer.at("shift");
            auto has = [&](const char *kind) {
                const auto &keys = layer.at(kind);
                return std::find(keys.begin(), keys.end(), key) != keys.end();
            };
            T factor = has("scale_keys") ? scale : T(1);
            if (has("square_keys")) factor *= scale * scale;
            result.scale *= factor;
            result.shift *= factor;
            if (has("shift_keys")) result.shift += shift;
        }
        return result;
    }
    void finish(Result &result, const Graph<T> &g, const T *spin);
};
} // namespace sp2

#include "device_engine.hpp"
