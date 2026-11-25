#pragma once

#ifdef SP2_ENABLE_CUDA
namespace sp2 {

// Borrowed host buffers and cached tapes may still be in use on exceptional
// exits. The normal path synchronizes once, after the packed result transfer.
class DeviceCompletion {
    const Exec &exec_;
    bool complete_ = false;
public:
    explicit DeviceCompletion(const Exec &exec) : exec_(exec) {}
    ~DeviceCompletion() {
        if (!complete_)
            cudaStreamSynchronize(exec_.cuda_stream());
    }
    void wait(const char *label) {
        cuda_check(cudaStreamSynchronize(exec_.cuda_stream()), label);
        complete_ = true;
    }
    void queued() { complete_ = true; }
};

template <class T> void SpinGNN2Kokkos<T>::populate_cuda_cache(Lattice<T> &c) {
    auto &g = c.graph;
    auto &sk = c.buf.w;
    auto &edges = c.spill_indices;
    sk.jit = c.plugin.get();
    c.saved.reserve((g.owned - 1) / c.options.tile_atoms + 1);
    DeviceCompletion complete(sk.exec);
    c.device.upload(sk.exec, upload_ring, g);
    int peak = 0, widest = 0;
    const int width = c.options.tile_atoms;
    for (int start = 0; start < g.owned;) {
        const int stop = start + std::min(width, g.owned - start);
        const int neighbors = g.offsets[stop] - g.offsets[start];
        if (neighbors > peak) {
            peak = neighbors;
            widest = start;
        }
        start = stop;
    }
    const bool spline = c.options.radial_mode == SP2_SPLINE;
    const bool frozen = (c.options.eval_flags & SP2_SKIP_LATTICE_FORCES) != 0;
    const bool host_only = c.options.cache_device_limit_bytes == std::numeric_limits<uint64_t>::max();
    const auto budget = resident_budget(c.options);
    edges.reserve(sk.exec, peak, width, c.device.periodic);
    auto spill_state = [&](int start, int stop) {
        sk.rewind(0);
        auto &forward = frozen ? work.w : sk;
        forward.jit = sk.jit;
        forward.profile = false;
        forward.rewind(0);
        const auto input = edges.prepare(forward, upload_ring, c.device, g, start, stop, spline, true);
        auto state = program.stage1(forward, input);
        edges.consumed(sk.exec);
        return frozen ? program.cache_forward(sk, state) : std::move(state);
    };
    if (!frozen || host_only) {
        spill_state(widest, widest + std::min(width, g.owned - widest));
        if (widest && g.owned - widest < width) spill_state(0, width);
        if (host_only) sk.freeze();
    }
    const auto max_tile_bytes = sk.bytes() + edges.bytes();
    if (host_only)
        c.pinned_tape.reserve((64u * 1024u * 1024u) / sizeof(T));
    for (int start = 0; start < g.owned;) {
        const int stop = start + std::min(width, g.owned - start);
        auto &saved = c.saved.emplace_back();
        if (host_only) {
            saved.forward_only = frozen;
            saved.state = spill_state(start, stop);
            saved.cuda_spill = std::make_unique<typename Workspace<T>::CudaTape>(sk.make_spill());
            saved.cuda_shape = sk.tape_shape();
            sk.spill(*saved.cuda_spill, c.pinned_tape.data(), c.pinned_tape.size());
            c.host_bytes += saved.cuda_spill->size() * sizeof(T);
        } else if (budget > c.device_bytes) {
            const auto remaining = budget - c.device_bytes;
            // An overflow tile can still serve spin-only calls without staging
            // its tape through host RAM. Full-force calls rematerialize it.
            saved.forward_only = frozen || max_tile_bytes > remaining;
            saved.resident = std::make_unique<Workspace<T>>(sk.exec);
            auto &tile = *saved.resident;
            auto &indices = saved.saved_indices;
            auto &forward = saved.forward_only ? sk : tile;
            tile.jit = c.plugin.get();
            forward.rewind(0);
            indices.reserve(tile.exec, g.offsets[stop] - g.offsets[start], stop - start,
                            c.device.periodic);
            saved.input = indices.prepare(forward, upload_ring, c.device, g, start, stop, spline, true);
            auto state = program.stage1(forward, saved.input);
            saved.state = saved.forward_only ? program.cache_forward(tile, state) : std::move(state);
            saved.input.r = saved.state.fields.at("r");
            indices.consumed(tile.exec);
            tile.freeze_tape();
            const auto bytes = tile.bytes() + indices.bytes();
            if (bytes <= remaining) c.device_bytes += bytes;
            else saved = {};
        }
        start = stop;
    }
    complete.wait("Complete lattice cache creation");
    for (auto &saved : c.saved) {
        if (!saved.resident) continue;
        saved.input.shifts = {};
        const auto old = saved.saved_indices.bytes();
        saved.saved_indices.retain_topology();
        c.device_bytes -= old - saved.saved_indices.bytes();
    }
    if (!c.host_bytes) {
        c.buf = Buffers<T>();
        c.spill_indices = {};
        c.pinned_tape.reset();
    }
}

template <class T>
std::unique_ptr<Result> SpinGNN2Kokkos<T>::run_device(
    const Graph<T> &g, const sp2_options &o, const T *spin, Lattice<T> *cache) {
    require(!magnetic || spin || !g.n, "Magnetic contract requires spins");
    const bool frozen = (o.eval_flags & SP2_SKIP_LATTICE_FORCES) != 0;
    require(!frozen || (cache && magnetic), "Frozen lattice derivatives require a magnetic stage-1 cache");
    auto result = std::make_unique<Result>();
    DeviceGraph<T> local;
    const bool restore_host = cache && cache->host_bytes &&
                             (frozen || !(cache->options.eval_flags & SP2_SKIP_LATTICE_FORCES));
    auto &w = restore_host ? cache->buf.w : work.w;
    auto &diag = result->diagnostics;
    diag.struct_size = sizeof(diag);
    diag.cache_host_bytes = cache ? cache->host_bytes : 0;
    diag.cache_device_bytes = cache ? cache->device_bytes : 0;
    w.jit = (cache ? cache->plugin : (plan.direct ? plan.artifact : nullptr)).get();
    w.profile = o.profile != 0;
    w.seconds = {};
    DeviceCompletion complete(w.exec);
    auto &device = cache ? cache->device : local;
    if (!cache)
        device.upload(w.exec, upload_ring, g);
    if (spin && device_spin.extent(0) < 3ull * g.n)
        device_spin = cuda_array<T>(w.exec, 3ull * g.n, "graph spins");
    DeviceArray<T> spins;
    if (spin) {
        spins = device_spin;
        upload_ring.copy(w.exec, spins.data(), spin, 3ull * g.n);
    }

    struct Output {
        std::vector<T> *host;
        std::size_t offset;
        int cols;
    };
    std::map<std::string, Output> outputs;
    std::size_t scalars = 0;
    auto field = [&](const std::string &key, std::vector<int64_t> shape, int cols) {
        auto &host = result->field<T>(key, std::move(shape));
        const auto offset = scalars;
        outputs.emplace(key, Output{&host, offset, cols});
        scalars += host.size();
        return offset;
    };
    const auto force_offset = frozen ? 0 : field("forces", {g.n, 3}, 3);
    const auto virial_offset = frozen ? 0 : field("atom_virial", {g.owned, 3, 3}, 9);
    const auto spin_offset = magnetic ? field("spin_forces", {g.n, 3}, 3) : 0;

    int peak = 0;
    const int width = o.tile_atoms;
    for (int start = 0; start < g.owned;) {
        const int stop = start + std::min(width, g.owned - start);
        peak = std::max(peak, g.offsets[stop] - g.offsets[start]);
        start = stop;
    }
    const auto usable = [frozen](const auto& saved) {
        return (saved.resident || saved.cuda_spill) && (frozen || !saved.forward_only);
    };
    if (!cache || !std::all_of(cache->saved.begin(), cache->saved.end(), usable))
        for (auto &edges : staged_edges) edges.reserve(w.exec, peak, width, device.periodic);
    diag.peak_tile_edges = peak;
    const bool spline = o.radial_mode == SP2_SPLINE;
    if (!g.owned) {
        finish(*result, g, spin);
        return result;
    }
    for (int start = 0, index = 0; start < g.owned; ++index) {
        const int n = std::min(width, g.owned - start);
        w.rewind(0);
        TileInput<T> input{};
        TileState<T> state;
        const TileState<T> *current = &state;
        DeviceEdges<T> *indices = &staged_edges[index % staged_edges.size()];
        if (cache && usable(cache->saved.at(index))) {
            auto &saved = cache->saved.at(index);
            if (saved.resident) {
                input = saved.input;
                indices = &saved.saved_indices;
            } else {
                indices = &cache->spill_indices;
                input = cache->spill_indices.prepare(w, upload_ring, device, g, start, start + n,
                                                     spline, false);
                w.restore(*saved.cuda_spill, saved.cuda_shape, cache->pinned_tape.data(),
                          cache->pinned_tape.size());
                input.r = saved.state.fields.at("r");
            }
            current = &saved.state;
            ++diag.stage1_cache_hits;
        } else {
            input = indices->prepare(w, upload_ring, device, g, start, start + n, spline, true);
            state = time_stage(w.exec, o.profile, diag.stage1_seconds,
                               [&] { return program.stage1(w, input); });
        }
        input.frozen = frozen;
        indices->gather_spin(w, input, device, spins, start, constant, mean);
        const auto tile = time_stage(w.exec, o.profile, diag.stage2_seconds,
                                    [&] { return program.stage2(w, input, *current); });
        if (start == 0) {
            for (const auto &[key, value] : tile.atom) {
                const bool matrix = key == "f_sigma" || key == "sf_sigma" ||
                                    key == "f_Uepi" || key == "sf_Uepi";
                field(key, matrix ? std::vector<int64_t>{g.owned, 3, 3} :
                                    std::vector<int64_t>{g.owned, value.cols}, value.cols);
            }
            if (device_result.extent(0) < scalars)
                device_result = cuda_array<T>(w.exec, scalars, "packed graph results");
            // Only accumulated fields need initializing. Readouts overwrite
            // every center, including isolated atoms.
            const auto initial = (magnetic ? 3ull * g.n : 0) +
                                 (frozen ? 0 : 3ull * g.n + 9ull * g.owned);
            if (initial)
                cuda_check(cudaMemsetAsync(device_result.data(), 0, initial * sizeof(T),
                                          w.exec.cuda_stream()), "Initialize graph adjoints");
        }
        T *const result_data = device_result.data();
        device_scatter(w, start, input, tile, indices->target,
                       frozen ? nullptr : result_data + force_offset,
                       magnetic ? result_data + spin_offset : nullptr,
                       frozen ? nullptr : result_data + virial_offset);
        indices->consumed(w.exec);
        for (const auto &[key, value] : tile.atom) {
            const auto &to = outputs.at(key);
            require(value.rows == n && value.cols == to.cols, "Unexpected center field shape");
            if (value.size())
                cuda_check(cudaMemcpyAsync(result_data + to.offset + std::size_t(start) * to.cols,
                                          value.ptr, value.size() * sizeof(T), cudaMemcpyDeviceToDevice,
                                          w.exec.cuda_stream()), "Collect center readout on the device");
        }
        start += n;
    }
    host_result.reserve(scalars);
    cuda_check(cudaMemcpyAsync(host_result.data(), device_result.data(), scalars * sizeof(T),
                               cudaMemcpyDeviceToHost, w.exec.cuda_stream()), "Copy packed graph result");
    complete.wait("Complete tiled CUDA evaluation");
    diag.workspace_bytes = w.bytes() + device.bytes() +
                           sizeof(T) * (device_result.extent(0) + device_spin.extent(0));
    for (const auto &edges : staged_edges) diag.workspace_bytes += edges.bytes();
    if (cache && cache->host_bytes)
        diag.workspace_bytes += cache->spill_indices.bytes();
    for (const auto &[key, out] : outputs)
        std::copy_n(host_result.data() + out.offset, out.host->size(), out.host->data());
    diag.geometry_seconds = w.seconds[0];
    diag.mlp_seconds = w.seconds[1];
    diag.tp_linear_seconds = w.seconds[2];
    diag.reverse_seconds = w.seconds[3];
    finish(*result, g, spin);
    return result;
}

template <class T>
sp2_device_result SpinGNN2Kokkos<T>::evaluate_neighbors(const sp2_neighbors &in,
                                                        const sp2_options &options) {
    require(in.num_owned >= 0 && in.num_owned < std::numeric_limits<int>::max() &&
                in.num_atoms >= in.num_owned && in.num_global > 0 && in.num_global >= in.num_owned,
            "Invalid local or global atom count");
    require(in.position_precision == SP2_FLOAT32 || in.position_precision == SP2_FLOAT64,
            "Invalid device coordinate precision");
    require(!in.num_atoms || (in.positions && in.types && in.type_map && in.type_map_size > 0 &&
                                 in.position_atom_stride > 0 && in.position_component_stride > 0),
            "Missing device coordinates or atom types");
    require(!in.num_atoms || !magnetic || in.spins, "Magnetic contract requires device spins");
    require(!in.num_atoms || !in.spins ||
                (in.spin_atom_stride > 0 && in.spin_component_stride > 0),
            "Invalid device spin strides");
    require(!in.num_owned || (in.ilist && in.numneigh &&
                                  (in.firstneigh || !in.neighbor_slot_capacity) &&
                                  in.neighbor_slot_capacity >= 0 && in.neighbor_mask > 0 &&
                                  in.ilist_stride > 0 && in.numneigh_stride > 0 &&
                                  (!in.neighbor_slot_capacity ||
                                   (in.neighbor_atom_stride > 0 && in.neighbor_slot_stride > 0))),
            "Invalid CUDA neighbor list view");
    require((options.execution_mode == SP2_AUTO || options.execution_mode == SP2_DIRECT) && plan.artifact,
            "Device neighbors require a matching prepared JIT artifact");
    require(options.execution_profile == SP2_CAPACITY || options.execution_profile == SP2_SPEED,
            "Invalid execution profile");
    require(cutoff > T(0), "Invalid model cutoff");
    const bool frozen = (options.eval_flags & SP2_SKIP_LATTICE_FORCES) != 0;
    const bool cache_forward_only = (options.eval_flags & SP2_CACHE_STAGE1_FORWARD_ONLY) != 0;
    require(!frozen || magnetic, "Frozen spin evaluation requires a magnetic model");

    auto o = options;
    o.execution_mode = SP2_DIRECT;
    auto &storage = neighbor_work;
    auto &w = storage.workspace();
    const auto caller = static_cast<cudaStream_t>(in.stream), stream = w.exec.cuda_stream();
    DeviceCompletion complete(w.exec);
    storage.order(caller, stream);
    w.jit = plan.artifact.get();
    w.profile = options.profile != 0;
    w.seconds = {};
    const bool rebuilt = storage.topology.prepare(w.exec, in, types, plan, o);
    const bool new_layout = rebuilt || storage.radial != o.radial_mode ||
                            storage.cache_limit != o.cache_device_limit_bytes ||
                            storage.cache_forward_only != cache_forward_only ||
                            storage.artifact != plan.artifact;
    const bool same_lattice = in.lattice_epoch && !new_layout &&
                              storage.lattice_epoch == in.lattice_epoch && storage.math == o.math_mode;
    // Explicit opt-in leaves the default capacity path free of graph allocations.
    const bool graph_enabled = (in.flags & SP2_NEIGHBOR_CUDA_GRAPH) && !w.profile &&
                               storage.topology.size() > 0;
    if (!graph_enabled || new_layout || storage.math != o.math_mode ||
        storage.lattice_epoch != in.lattice_epoch)
        storage.reset_graphs();
    if (!in.lattice_epoch || new_layout) {
        storage.saved.clear();
        storage.cache_bytes = 0;
        storage.cache_budget = 0;
        storage.radial = o.radial_mode;
        storage.cache_limit = o.cache_device_limit_bytes;
        storage.cache_forward_only = cache_forward_only;
        storage.artifact = plan.artifact;
    }
    if (in.lattice_epoch && (new_layout || !storage.lattice_epoch) && storage.saved.empty()) {
        storage.cache_budget = resident_budget(o);
        if (storage.cache_budget) storage.saved.resize(storage.topology.size());
    }
    // Publish the new epoch only after every tile and stream handoff succeeds.
    // A failed call leaves any partially updated tapes invalid for the retry.
    storage.lattice_epoch = 0;
    storage.math = o.math_mode;

    const auto positions = in.position_precision == SP2_FLOAT32 ?
                               storage.template positions<float>(w.exec, in) :
                               storage.template positions<double>(w.exec, in);
    // Refresh outside graph capture: bucketed padding keeps per-tile GEMM shapes
    // constant across thermal cutoff crossings so captured graphs replay without thrashing.
    if (!same_lattice && storage.topology.compact(w.exec, positions, cutoff, graph_enabled))
        storage.reset_graphs();
    const auto spin = in.position_precision == SP2_FLOAT32 ?
                          storage.template spins<float>(w.exec, in) :
                          storage.template spins<double>(w.exec, in);
    const auto atoms = size_t(in.num_atoms), owners = size_t(in.num_owned);
    const bool evidence_f = std::find(prefixes.begin(), prefixes.end(), "f") != prefixes.end();
    const bool evidence_sf = std::find(prefixes.begin(), prefixes.end(), "sf") != prefixes.end();
    const size_t force_values = frozen ? 0 : 3 * atoms;
    const size_t spin_values = magnetic ? 3 * atoms : 0;
    const size_t result_values = force_values + spin_values +
                                 owners * (1 + (frozen ? 0 : 9) + (bec_field.empty() ? 0 : 9) +
                                           20 * (int(evidence_f) + int(evidence_sf)));
    if (!storage.results.data() || storage.results.extent(0) < result_values)
        storage.results = cuda_array<T>(w.exec, result_values, "LAMMPS device results");
    T *result_data = storage.results.data();
    const auto take = [&](bool present, size_t size) -> T * {
        if (!present) return nullptr;
        T *next = result_data;
        result_data += size;
        return next;
    };
    auto *forces = take(!frozen, 3 * atoms);
    auto *spin_forces = take(magnetic, 3 * atoms);
    auto *energy = take(true, owners);
    auto *virial = take(!frozen, 9 * owners);
    auto *born = take(!bec_field.empty(), 9 * owners);
    struct Evidence {
        T *sigma, *uepi, *nu, *kappa;
        sp2_device_evidence view() const { return {sigma, uepi, nu, kappa}; }
    };
    const auto evidence = [&](bool present) -> Evidence {
        return {take(present, 9 * owners), take(present, 9 * owners),
                take(present, owners), take(present, owners)};
    };
    const auto ef = evidence(evidence_f), esf = evidence(evidence_sf);
    const auto accumulated = force_values + spin_values;
    sp2_device_result result{};
    result.struct_size = sizeof(result);
    result.forces = forces;
    result.spin_forces = spin_forces;
    result.owned_energy = energy;
    result.atom_virial = virial;
    result.bec = born;
    result.force_evidence = ef.view();
    result.spin_evidence = esf.view();
    auto &diag = result.diagnostics;
    diag.struct_size = sizeof(diag);
    diag.topology_metadata_bytes = storage.topology.metadata_bytes(rebuilt, !same_lattice);
    auto enqueue = [&] {
        if (accumulated)
            cuda_check(cudaMemsetAsync(storage.results.data(), 0, accumulated * sizeof(T), stream),
                       "Clear device endpoint adjoints");
        const auto physical_energy = rescale("total_energy");
        const auto physical_force = rescale("forces");
        const auto physical_spin = rescale("spin_forces");
        const auto physical_virial = rescale("atom_virial");
        const auto physical_born = bec_field.empty() ? Rescale{} : rescale(bec_field);
        const T seed = mean ? constant / T(in.num_global) : constant;
        const Rescale energy_adjust{seed * physical_energy.scale, physical_energy.shift / T(in.num_global)};
        struct Readout {
            std::string name;
            T *out;
            Rescale transform;
        };
        std::vector<Readout> readouts{{"atomic_energy", energy, energy_adjust}};
        if (born) readouts.push_back({bec_field, born, physical_born});
        auto append_evidence = [&](const std::string &prefix, const Evidence &fields) {
            if (!fields.sigma) return;
            for (const auto &entry : {std::make_pair("_sigma", fields.sigma),
                                      std::make_pair("_Uepi", fields.uepi),
                                      std::make_pair("_nu", fields.nu),
                                      std::make_pair("_kappa", fields.kappa)}) {
                const auto name = prefix + entry.first;
                readouts.push_back({name, entry.second, rescale(name)});
            }
        };
        append_evidence("f", ef);
        append_evidence("sf", esf);
        auto collect = [&](const TileResult<T> &tile, const Readout &readout, int start) {
            const auto field = tile.atom.at(readout.name);
            auto *out = readout.out + size_t(start) * field.cols;
            const auto transform = readout.transform;
            for_each(w.exec, "sp2::collect_device_center_output", field.size(), KOKKOS_LAMBDA(int k) {
                out[k] = field.ptr[k] * transform.scale + transform.shift;
            });
        };

        const bool spline = o.radial_mode == SP2_SPLINE;
        const auto budget = storage.cache_budget;
        bool room = !same_lattice && !storage.saved.empty();
        for (int index = 0; index < storage.topology.size(); ++index) {
            auto input = storage.topology.tile(w.exec, index, positions, cutoff, spline);
            const size_t edges = input.center.extent(0);
            w.rewind(0);
            TileState<T> state;
            const TileState<T> *lattice = &state;
            auto *saved = storage.saved.empty() ? nullptr : &storage.saved[index];
            std::unique_ptr<Workspace<T>> resident;
            bool forward_only = false;
            if (same_lattice && saved && saved->workspace && (frozen || !saved->forward_only)) {
                ++diag.stage1_cache_hits;
                lattice = &saved->state;
                input.r = lattice->fields.at("r");
            } else {
                const auto held = !same_lattice && saved && saved->workspace ? saved->workspace->bytes() : 0;
                if (held) {
                    forward_only = saved->forward_only;
                    resident = std::move(saved->workspace);
                    storage.cache_bytes -= held;
                    saved->state = {};
                    resident->reset_tape();
                } else if (room) {
                    const auto estimate = plan.bytes_per_edge * std::max(edges, size_t(input.ncenters));
                    forward_only = cache_forward_only || frozen || estimate > budget - storage.cache_bytes;
                    resident = std::make_unique<Workspace<T>>(w.exec);
                }
                auto &forward = resident && !forward_only ? *resident : w;
                forward.jit = w.jit;
                forward.profile = w.profile;
                input.r = forward.make(input.center.extent_int(0), 3);
                state = time_stage(forward.exec, w.profile, diag.stage1_seconds,
                                   [&] { return program.stage1(forward, input); });
                if (resident && !forward_only) {
                    resident->freeze();
                    for (size_t k = 0; k < w.seconds.size(); ++k) w.seconds[k] += resident->seconds[k];
                }
            }
            input.frozen = frozen;
            gather_neighbors_spin(w, input, spin, seed);
            const auto tile = time_stage(w.exec, w.profile, diag.stage2_seconds,
                                         [&] { return program.stage2(w, input, *lattice); });
            device_scatter(w, input.start, input, tile, input.targets, forces, spin_forces, virial);
            for (const auto &readout : readouts) collect(tile, readout, input.start);
            if (resident && forward_only) {
                state = program.cache_forward(*resident, state);
                resident->freeze();
            }
            if (resident && resident->bytes() <= budget - storage.cache_bytes) {
                storage.cache_bytes += resident->bytes();
                saved->workspace = std::move(resident);
                saved->state = std::move(state);
                saved->forward_only = forward_only;
            } else if (resident) {
                room = false;
            }
        }
        for_each(w.exec, "sp2::scale_device_endpoint_outputs", in.num_atoms, KOKKOS_LAMBDA(int i) {
            const auto offset = 3ull * i;
            if (forces)
                for (int c = 0; c < 3; ++c)
                    forces[offset + c] = forces[offset + c] * physical_force.scale +
                                         (i < in.num_owned ? physical_force.shift : T(0));
            if (spin_forces) {
                const Vec3<T> s{spin[offset], spin[offset + 1], spin[offset + 2]};
                const T weight = s.dot(s) * physical_spin.scale;
                for (int c = 0; c < 3; ++c)
                    spin_forces[offset + c] = spin_forces[offset + c] * weight +
                                              (i < in.num_owned ? physical_spin.shift : T(0));
            }
        });
        if (virial && (physical_virial.scale != T(1) || physical_virial.shift != T(0)))
            for_each(w.exec, "sp2::scale_device_virial", in.num_owned, KOKKOS_LAMBDA(int i) {
                for (int c = 0; c < 9; ++c)
                    virial[9ull * i + c] = virial[9ull * i + c] * physical_virial.scale + physical_virial.shift;
            });
    };

    const bool graph_ready = graph_enabled && (!in.lattice_epoch || same_lattice);
    auto &graph = storage.graphs[frozen];
    auto graph_key = [&] {
        return NeighborGraphKey{
            in.positions, in.spins, storage.results.data(),
            {in.num_global, in.position_precision, in.position_atom_stride,
             in.position_component_stride, in.spin_atom_stride, in.spin_component_stride},
            in.lattice_epoch, w.allocation_generation(), o.eval_flags};
    };
    if (graph_ready && graph.match(graph_key())) {
        result.cuda_graph_status = SP2_GRAPH_FALLBACK;
        if (graph.available()) {
            try {
                const UseCaptureBlas use(storage.graph_blas.get(w.exec));
                cublas(w.exec);
                result.cuda_graph_status = graph.run(stream, [&] {
                    const typename Workspace<T>::CaptureScope stable(w);
                    enqueue();
                });
            } catch (...) {
                graph.disable();
            }
        }
    }
    if (result.cuda_graph_status == SP2_GRAPH_REPLAYED) {
        diag.peak_tile_edges = storage.topology.peak_actual_edges();
        diag.stage1_cache_hits = graph.cache_hits;
    } else if (result.cuda_graph_status == SP2_GRAPH_CAPTURED) {
        diag.peak_tile_edges = storage.topology.peak_actual_edges();
        graph.peak_tile_edges = diag.peak_tile_edges;
        graph.cache_hits = diag.stage1_cache_hits;
    } else {
        diag.peak_tile_edges = storage.topology.peak_actual_edges();
        diag.stage1_cache_hits = 0;
        if (graph_ready && graph.available()) {
            // Warm cuBLASLt heuristics on the graph BLAS handle outside capture.
            const UseCaptureBlas use(storage.graph_blas.get(w.exec));
            cublas(w.exec);
            enqueue();
        } else {
            enqueue();
        }
        if (graph_ready && !graph.match(graph_key()))
            graph.warm(graph_key());
    }

    diag.geometry_seconds = w.seconds[0];
    diag.mlp_seconds = w.seconds[1];
    diag.tp_linear_seconds = w.seconds[2];
    diag.reverse_seconds = w.seconds[3];
    diag.cache_device_bytes = storage.cache_bytes;
    diag.workspace_bytes = w.bytes() + storage.topology.bytes() + storage.graph_blas.bytes() +
                           sizeof(T) * (storage.position_storage.extent(0) +
                                        storage.spin_storage.extent(0) + storage.results.extent(0));
    storage.order(stream, caller);
    storage.lattice_epoch = in.lattice_epoch;
    complete.queued();
    return result;
}

} // namespace sp2
#endif
