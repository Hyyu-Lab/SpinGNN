#include "engine.hpp"
#include <iterator>

namespace sp2 {
template <class T> void SpinGNN2Kokkos<T>::finish(Result &r, const Graph<T> &g, const T *spin) {
    auto &energy = r.field<T>("total_energy", {g.frames, 1});
    const auto &atomic = r.field<T>("atomic_energy", {g.owned, 1});
    for (int i = 0; i < g.owned; ++i)
        energy[g.batch[i]] += atomic[i];
    for (int f = 0; f < g.frames; ++f)
        energy[f] = energy[f] * constant / (mean ? T(g.counts[f]) : T(1));
    if (g.global) {
        auto &owned = r.field<T>("owned_energy", {g.owned, 1});
        const auto transform = rescale("total_energy");
        const T weight = mean ? constant / T(g.global) : constant;
        const T shift = transform.shift / T(g.global);
        for (int i = 0; i < g.owned; ++i)
            owned[i] = (atomic[i] * weight) * transform.scale + shift;
    }
    if (r.fields.count("atom_virial")) {
        auto &virial = r.field<T>("virial", {g.frames, 3, 3});
        const auto &va = r.data<T>("atom_virial");
        for (int i = 0; i < g.owned; ++i)
            for (int k = 0; k < 9; ++k)
                virial[9ull * g.batch[i] + k] += va[9ull * i + k];
        auto &stress =
            r.field<T>("stress", g.volume.empty() ? std::vector<int64_t>{0}
                                                  : std::vector<int64_t>{g.frames, 3, 3});
        for (int f = 0; f < g.frames; ++f)
            for (int k = 0; k < 3; ++k)
                for (int l = k; l < 3; ++l) {
                    size_t a = 9ull * f + 3 * k + l, b = 9ull * f + 3 * l + k;
                    virial[a] = virial[b] = (virial[a] + virial[b]) * T(.5);
                    if (!stress.empty())
                        stress[a] = stress[b] = -virial[a] / g.volume[f];
                }
    }
    if (magnetic) {
        auto &sf = r.data<T>("spin_forces");
        auto &vertical = r.field<T>("spin_forces_vert", {g.n, 3});
        T coeff = model.at("spin_fv_coeff");
        for (int i = 0; i < g.n; ++i) {
            const size_t idx = 3ull * i;
            T norm2 = 0, dot = 0;
            for (int k = 0; k < 3; ++k)
                norm2 += spin[idx + k] * spin[idx + k];
            for (int k = 0; k < 3; ++k) {
                sf[idx + k] *= norm2;
                dot += sf[idx + k] * spin[idx + k];
            }
            for (int k = 0; k < 3; ++k)
                vertical[idx + k] = coeff * (sf[idx + k] - dot * spin[idx + k]);
        }
    }
    if (!bec.is_null()) {
        auto &at = r.field<T>(bec_field, {g.owned, 9});
        if (!bec.at("pre_asr").is_null())
            r.field<T>(bec.at("pre_asr"), {g.owned, 9}) = at;
        auto averages = [&]() {
            std::vector<T> out(9ull * g.frames);
            for (int i = 0; i < g.owned; ++i)
                for (int k = 0; k < 9; ++k)
                    out[9ull * g.batch[i] + k] += at[9ull * i + k];
            for (int f = 0; f < g.frames; ++f)
                for (int k = 0; k < 9; ++k)
                    out[9ull * f + k] /= T(g.counts[f]);
            return out;
        };
        if (!g.global && bec.at("asr") == "hard") {
            const auto means = averages();
            for (int i = 0; i < g.owned; ++i)
                for (int k = 0; k < 9; ++k)
                    at[9ull * i + k] -= means[9ull * g.batch[i] + k];
        }
        if (!bec.at("total").is_null())
            r.field<T>(bec.at("total"), {g.frames, 9}) = averages();
    }
    for (const auto &prefix : prefixes) {
        if (!g.owned) {
            r.field<T>(prefix + "_sigma", {0, 3, 3});
            r.field<T>(prefix + "_nu", {0, 1});
            r.field<T>(prefix + "_kappa", {0, 1});
        }
        const auto &u = r.field<T>(prefix + "_Uepi", {g.owned, 3, 3});
        auto &total = r.field<T>("total_" + prefix + "_Uepi", {g.frames, 3, 3});
        auto &unc = r.field<T>(prefix + "_uncertainty", {g.frames, 1});
        std::vector<int> counts(g.frames);
        for (int i = 0; i < g.owned; ++i) {
            int f = g.batch[i];
            for (int k = 0; k < 9; ++k)
                total[9ull * f + k] += u[9ull * i + k];
            unc[f] += u[9ull * i] + u[9ull * i + 4] + u[9ull * i + 8];
            if (prefix == "sf") {
                const auto *s = spin + 3ull * i;
                counts[f] += Vec3<T>{s[0], s[1], s[2]}.norm() > T(1e-6);
            }
        }
        for (int f = 0; f < g.frames; ++f)
            unc[f] /= T(prefix == "sf" ? std::max(1, counts[f]) : g.counts[f]);
    }
    for (const auto &layer : model.at("rescales")) {
        const T scale = layer.at("scale"), shift = layer.at("shift");
        for (const auto *kind : {"scale_keys", "square_keys", "shift_keys"}) {
            for (const auto &key : layer.at(kind)) {
                auto it = r.fields.find(key.template get<std::string>());
                if (it == r.fields.end())
                    continue;
                auto &values = std::get<std::vector<T>>(it->second.data);
                if (kind == std::string("shift_keys")) {
                    auto count = values.size();
                    if (g.global && (key == "forces" || key == "spin_forces" ||
                                     key == "spin_forces_vert"))
                        count = 3ull * g.owned;
                    const T local_shift = shift * (g.global && key == "total_energy" ?
                                                       T(g.owned) / T(g.global) : T(1));
                    for (size_t i = 0; i < count; ++i)
                        values[i] += local_shift;
                } else {
                    const T factor = kind == std::string("square_keys") ? scale * scale : scale;
                    for (auto &v : values)
                        v *= factor;
                }
            }
        }
    }
}
} // namespace sp2

struct sp2_model {
    std::shared_ptr<sp2::Engine> engine;
};
struct sp2_cache {
    std::shared_ptr<sp2::Engine> engine;
    std::unique_ptr<sp2::CacheBase> cache;
    uint32_t math_mode;
};
struct sp2_result {
    std::unique_ptr<sp2::Result> value;
};

namespace {
thread_local std::string error;
template <class Fn> int guarded(Fn &&fn) noexcept {
    try {
        fn();
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
    } catch (...) {
        error = "Unknown native inference error";
    }
    return -1;
}
void initialize(int device) {
    // A process may already own Kokkos (e.g. LAMMPS). Only the initializer
    // that acquired the runtime is responsible for finalizing it at exit.
    struct Runtime {
        bool owner;
        explicit Runtime(int ordinal) : owner(!Kokkos::is_initialized()) {
            if (owner) {
                Kokkos::InitializationSettings settings;
                if (ordinal >= 0) settings.set_device_id(ordinal);
                Kokkos::initialize(settings);
            }
        }
        ~Runtime() {
            if (owner && Kokkos::is_initialized() && !Kokkos::is_finalized())
                Kokkos::finalize();
        }
    };
    static Runtime runtime(device);
#ifdef SP2_ENABLE_CUDA
    sp2::require(device < 0 || sp2::Exec().cuda_device() == device,
                 "Inference runtime is already bound to a different CUDA device");
#else
    sp2::require(device < 0, "A device ordinal requires the CUDA backend");
#endif
}
sp2_options read_options(const sp2_options *opt, uint32_t flags = SP2_SKIP_LATTICE_FORCES) {
    sp2::require(opt && opt->struct_size >= offsetof(sp2_options, cache_device_limit_bytes) &&
                     opt->tile_atoms >= 0 &&
                     (opt->radial_mode == SP2_ANALYTIC || opt->radial_mode == SP2_SPLINE),
                 "Invalid evaluation options");
    sp2_options value{};
    std::memcpy(&value, opt, std::min<size_t>(opt->struct_size, sizeof(value)));
    if (opt->struct_size < offsetof(sp2_options, math_mode) + sizeof(value.math_mode))
        value.math_mode = SP2_MATH_IEEE;
    sp2::require(!(value.eval_flags & ~flags), "Invalid eval_flags");
    return value;
}

sp2_options options(const sp2_graph *graph, const sp2_options *opt) {
    sp2::require(graph && graph->struct_size >= offsetof(sp2_graph, num_owned),
                 "Invalid graph ABI size");
    return read_options(opt);
}

void select_math(sp2::Engine &engine, uint32_t mode) {
    sp2::require(mode == SP2_MATH_IEEE || mode == SP2_MATH_TF32, "Unknown BLAS math mode");
#ifdef SP2_ENABLE_CUDA
    if (mode == SP2_MATH_TF32) {
        sp2::require(engine.plan.precision == 32, "TF32 requires a CUDA FP32 model");
        sp2::require(sp2::device_caps().major >= 8,
                     "TF32 requires an Ampere or newer (sm_80+) CUDA GPU");
    }
    sp2::use_tf32 = mode == SP2_MATH_TF32;
#else
    sp2::require(mode == SP2_MATH_IEEE, "TF32 requires a CUDA FP32 model");
#endif
    engine.plan.math_mode = mode;
}
} // namespace

extern "C" {
uint32_t sp2_abi_version() { return SP2_ABI_VERSION; }
const char *sp2_last_error() { return error.c_str(); }
const char *sp2_backend() {
#ifdef SP2_ENABLE_CUDA
    return "cuda";
#else
    return "cpu";
#endif
}
int sp2_model_load(const char *manifest, int32_t precision, sp2_model **out) {
    return sp2_model_load_device(manifest, precision, -1, out);
}
int sp2_model_load_device(const char *manifest, int32_t precision, int32_t device, sp2_model **out) {
    return guarded([&] {
        sp2::require(manifest && out, "Missing manifest or output pointer");
        *out = nullptr;
        sp2::Contract contract(manifest);
        sp2::require(device >= -1, "Invalid device ordinal");
        initialize(device);
        auto handle = std::make_unique<sp2_model>();
        if (precision == 0) {
            const std::string dtype = contract.model.at("dtype");
            sp2::require(dtype == "float64" || dtype == "float32", "Unknown model precision");
            precision = dtype == "float64" ? 64 : 32;
        }
        if (precision == 64)
            handle->engine = std::make_shared<sp2::SpinGNN2Kokkos<double>>(contract);
        else if (precision == 32)
            handle->engine = std::make_shared<sp2::SpinGNN2Kokkos<float>>(contract);
        else
            throw std::invalid_argument("Unsupported inference precision");
#ifdef SP2_ENABLE_CUDA
        auto info = sp2::Json::parse(handle->engine->info);
        info["supports_tf32"] = precision == 32 && sp2::device_caps().major >= 8;
        handle->engine->info = info.dump();
#endif
        *out = handle.release();
    });
}
void sp2_model_free(sp2_model *model) { delete model; }
const char *sp2_model_info(const sp2_model *model) {
    return model ? model->engine->info.c_str() : nullptr;
}
int sp2_model_load_jit_artifact(sp2_model *model, const char *path) {
    return guarded([&] {
        sp2::require(model && path, "Missing model or JIT artifact manifest");
        std::lock_guard<std::mutex> lock(model->engine->mutex);
        model->engine->install(path);
    });
}
const char *sp2_model_execution_plan(sp2_model *model) {
    thread_local std::string plan;
    return guarded([&] {
               sp2::require(model, "Missing model");
               std::lock_guard<std::mutex> lock(model->engine->mutex);
               plan = model->engine->plan.describe().dump();
           }) == 0
               ? plan.c_str()
               : nullptr;
}
int sp2_evaluate(sp2_model *model, const sp2_graph *g, const sp2_options *o, sp2_result **out) {
    return guarded([&] {
        sp2::require(model && out, "Missing model or output pointer");
        *out = nullptr;
        auto opts = options(g, o);
        std::lock_guard<std::mutex> lock(model->engine->mutex);
        select_math(*model->engine, opts.math_mode);
        auto result = std::make_unique<sp2_result>();
        result->value = model->engine->evaluate(*g, opts);
        *out = result.release();
    });
}
int sp2_evaluate_neighbors(sp2_model *model, const sp2_neighbors *neighbors, const sp2_options *o,
                           sp2_device_result *out) {
    return guarded([&] {
        constexpr auto input_v1_size = offsetof(sp2_neighbors, flags);
        constexpr auto output_v1_size = offsetof(sp2_device_result, cuda_graph_status);
        sp2::require(model && neighbors && neighbors->struct_size >= input_v1_size &&
                         out && out->struct_size >= output_v1_size,
                     "Invalid device neighbor ABI");
#ifdef SP2_ENABLE_CUDA
        sp2_neighbors input{};
        std::memcpy(&input, neighbors, neighbors->struct_size >= sizeof(input) ?
                                      sizeof(input) : input_v1_size);
        sp2::require(!(input.flags & ~uint32_t(SP2_NEIGHBOR_CUDA_GRAPH)), "Invalid device neighbor flags");
        const auto opts = read_options(o, SP2_SKIP_LATTICE_FORCES | SP2_CACHE_STAGE1_FORWARD_ONLY);
        std::lock_guard<std::mutex> lock(model->engine->mutex);
        select_math(*model->engine, opts.math_mode);
        const auto result = model->engine->evaluate_neighbors(input, opts);
        const auto bytes = out->struct_size >= sizeof(result) ? sizeof(result) : output_v1_size;
        std::memcpy(out, &result, bytes);
        out->struct_size = bytes;
#else
        (void)o;
        throw std::invalid_argument("Device neighbor evaluation requires the CUDA backend");
#endif
    });
}
int sp2_stage1_cache_create(sp2_model *model, const sp2_graph *g, const sp2_options *o,
                            sp2_cache **out) {
    return guarded([&] {
        sp2::require(model && out, "Missing model or cache output");
        *out = nullptr;
        auto opts = options(g, o);
        std::lock_guard<std::mutex> lock(model->engine->mutex);
        select_math(*model->engine, opts.math_mode);
        auto cache = std::make_unique<sp2_cache>();
        cache->math_mode = opts.math_mode;
        cache->engine = model->engine;
        cache->cache = cache->engine->stage1(*g, opts);
        *out = cache.release();
    });
}
static int evaluate_cache(sp2_model *model, const sp2_cache *cache, const void *spins,
                          const uint32_t *flags, sp2_result **out) {
    return guarded([&] {
        sp2::require(out, "Missing result output");
        *out = nullptr;
        sp2::require(model && cache && cache->engine == model->engine,
                     "Cache belongs to a different model");
        std::lock_guard<std::mutex> lock(model->engine->mutex);
        select_math(*model->engine, cache->math_mode);
        auto result = std::make_unique<sp2_result>();
        result->value = model->engine->cached(*cache->cache, spins, flags);
        *out = result.release();
    });
}
int sp2_evaluate_stage2_cached(sp2_model *model, const sp2_cache *cache, const void *spins,
                               sp2_result **out) {
    return evaluate_cache(model, cache, spins, nullptr, out);
}
int sp2_evaluate_stage2_cached_flags(sp2_model *model, const sp2_cache *cache, const void *spins,
                                     uint32_t flags, sp2_result **out) {
    return evaluate_cache(model, cache, spins, &flags, out);
}
void sp2_cache_free(sp2_cache *c) { delete c; }
int32_t sp2_result_count(const sp2_result *r) { return r ? int32_t(r->value->fields.size()) : 0; }
int sp2_result_field(const sp2_result *r, int32_t i, const char **name, const void **data,
                     int32_t *rank, const int64_t **shape) {
    return guarded([&] {
        sp2::require(r && name && data && rank && shape && i >= 0 && i < sp2_result_count(r),
                     "Invalid field query");
        auto it = std::next(r->value->fields.begin(), i);
        *name = it->first.c_str();
        *rank = it->second.shape.size();
        *shape = it->second.shape.data();
        *data = std::visit([](const auto &v) -> const void * { return v.data(); }, it->second.data);
    });
}
int sp2_result_diagnostics(const sp2_result *r, sp2_diagnostics *out) {
    return guarded([&] {
        sp2::require(r && out && out->struct_size >= offsetof(sp2_diagnostics, cache_device_bytes),
                     "Invalid diagnostics ABI size");
        std::memcpy(out, &r->value->diagnostics, std::min<size_t>(sizeof(*out), out->struct_size));
    });
}
void sp2_result_free(sp2_result *result) { delete result; }
}
