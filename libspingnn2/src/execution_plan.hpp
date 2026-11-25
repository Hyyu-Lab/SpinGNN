#pragma once

#include "contract.hpp"
#include "jit_runtime.hpp"
#include "spingnn2_c_api.h"
#ifdef SP2_ENABLE_CUDA
#include <cuda_runtime_api.h>
#include <mutex>
#include <unordered_map>
#endif

namespace sp2 {

#ifdef SP2_ENABLE_CUDA
struct DeviceCaps {
    int device = -1;
    int major = 0;
    int minor = 0;
    int sm_count = 0;
    size_t total_mem = 0;
    size_t shared_mem_optin = 49152;
};

inline const DeviceCaps &device_caps(int ordinal = -1) {
    if (ordinal < 0) {
        if (cudaGetDevice(&ordinal) != cudaSuccess) ordinal = 0;
    }
    static std::mutex mu;
    static std::unordered_map<int, DeviceCaps> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(ordinal);
    if (it != cache.end()) return it->second;
    DeviceCaps caps{};
    caps.device = ordinal;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, ordinal) == cudaSuccess) {
        caps.major = prop.major;
        caps.minor = prop.minor;
        caps.sm_count = prop.multiProcessorCount;
        caps.total_mem = prop.totalGlobalMem;
        caps.shared_mem_optin = prop.sharedMemPerBlock;
    }
    int optin = 0;
    if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, ordinal) ==
            cudaSuccess &&
        optin > 0) {
        caps.shared_mem_optin = static_cast<size_t>(optin);
    }
    return cache.emplace(ordinal, caps).first->second;
}
#endif

// Plan selection is explicit at the ABI boundary. An auto call without an
// artifact uses the generic program. Failure inside a selected kernel propagates.
struct ExecutionPlan {
    std::string contract_digest;
    int precision{};
    uint32_t math_mode = SP2_MATH_IEEE;
    std::shared_ptr<const JitPlugin> artifact;
    bool direct{};
    int width = 256, profile = SP2_CAPACITY, speed_width = 512;
    size_t bytes_per_edge{}, max_edge_cols = 1, budget{};
    void size_model(const Contract &c) {
        // The strict FP32 CUDA sweep favors a wider tile for the single-trunk
        // Cartesian readout. The two spin trunks saturate at 512 centers.
#ifdef SP2_ENABLE_CUDA
        if (precision == 32 && c.model.at("family") == "Allegro_born")
            speed_width = 1024;
#endif
        // Reserve space for the forward value, its slope, and two adjoint/scratch
        // arrays per hidden channel. This is a planning estimate, not allocator
        // accounting: evaluation reports the measured high-water mark separately.
        size_t channels = 96;
        max_edge_cols = 1;
        auto count_mlp = [&](const Json &mlp) {
            for (const auto &l : mlp.at("layers")) {
                const size_t w = c.shape(l.at("weight")).at(1);
                channels += 4 * w;
                max_edge_cols = std::max(max_edge_cols, w);
            }
        };
        auto count_linear = [&](const Json &lin) {
            if (lin.is_null()) return;
            const size_t in_mul = lin.at("mul_in").get<size_t>();
            const size_t out_mul = lin.at("mul_out").get<size_t>();
            size_t fwd_cols = 0, rev_cols = 0;
            for (const auto &block : lin.at("blocks")) {
                const size_t comp = block.at("components").get<size_t>();
                for (const auto &g : block.at("groups")) {
                    const size_t cnt = (g.at("stop").get<size_t>() - g.at("start").get<size_t>()) / comp;
                    fwd_cols += comp * in_mul * cnt;
                    rev_cols += comp * out_mul;
                }
            }
            max_edge_cols = std::max(max_edge_cols, std::max(fwd_cols, rev_cols));
        };
        for (const auto &l : c.model.at("layers")) {
            if (l.contains("mlp"))
                count_mlp(l.at("mlp"));
            if (!l.contains("backbone"))
                continue;
            const auto &b = l.at("backbone");
            for (const auto &layer : b.at("layers")) {
                count_mlp(layer.at("latent"));
                count_mlp(layer.at("env_embed_mlp"));
                const auto &tp = layer.at("tensor_product");
                const size_t tp_cols = tp.at("mul_out").get<size_t>() * tp.at("dim_out").get<size_t>();
                channels += 4 * tp_cols;
                max_edge_cols = std::max(max_edge_cols, tp_cols);
                count_linear(layer.at("linear"));
            }
            count_mlp(b.at("final_latent"));
            if (b.contains("heads")) {
                for (const auto &kv : b.at("heads").items()) {
                    const auto &head = kv.value();
                    auto count_head_linear = [&](const Json &lin) {
                        if (lin.is_null()) return;
                        const size_t in_mul = lin.at("mul_in").get<size_t>();
                        const size_t out_mul = lin.at("mul_out").get<size_t>();
                        const size_t out_dim = lin.at("dim_out").get<size_t>();
                        channels += 2 * out_mul * out_dim;
                        max_edge_cols = std::max(max_edge_cols, out_mul * out_dim);
                        for (const auto &block : lin.at("blocks")) {
                            const size_t comp = block.at("components").get<size_t>();
                            for (const auto &g : block.at("groups")) {
                                const size_t cnt =
                                    (g.at("stop").get<size_t>() - g.at("start").get<size_t>()) / comp;
                                max_edge_cols = std::max(
                                    max_edge_cols, comp * std::max(in_mul * cnt, out_mul));
                            }
                        }
                    };
                    count_head_linear(head.at("linear_in"));
                    count_head_linear(head.at("linear_out"));
                }
            }
        }
        bytes_per_edge = channels * (precision / 8);
    }
    void select(sp2_options &o, const std::vector<int> &offsets) {
        const int n = static_cast<int>(offsets.size() - 1);
        int degree = 1;
        if (!o.tile_atoms)
            for (int i = 0; i < n; ++i)
                degree = std::max(degree, offsets[i + 1] - offsets[i]);
        select(o, n, degree);
    }
    void select(sp2_options &o, int n, int degree) {
        require(o.execution_mode >= SP2_AUTO && o.execution_mode <= SP2_DIRECT,
                "Invalid execution mode");
        require(o.execution_profile == SP2_SPEED || o.execution_profile == SP2_CAPACITY,
                "Invalid execution profile");
        require(o.execution_mode != SP2_DIRECT || artifact,
                "Direct execution requires a matching JIT artifact");
        require(o.tile_atoms >= 0 && o.tile_atoms <= 65535,
                "Invalid tile_atoms (must be in [0, 65535])");
        direct = o.execution_mode != SP2_GENERIC && bool(artifact);
        profile = o.execution_profile;
        // The capacity profile bounds activation storage at 256 centers by
        // default. The speed profile amortizes launches over a larger tile.
        width = std::min(n, o.tile_atoms ? o.tile_atoms : (profile == SP2_SPEED ? speed_width : 256));
        width = std::min(width, 65535);
        budget = 0;
        if (!o.tile_atoms) {
            size_t available = 8ull << 30;
            size_t cap = (profile == SP2_SPEED ? 8ull : 2ull) << 30;
#ifdef SP2_ENABLE_CUDA
            size_t total = 0;
            if (cudaMemGetInfo(&available, &total) != cudaSuccess)
                throw std::runtime_error(
                    "Unable to determine available memory for automatic tile sizing");
            const auto &caps = device_caps();
            if (profile == SP2_SPEED && caps.total_mem > (16ull << 30))
                cap = std::max(cap, caps.total_mem / 2);
#endif
            budget = std::min<size_t>(available / 2, cap);
            const size_t int_max_edges =
                static_cast<size_t>(std::numeric_limits<int>::max()) / std::max<size_t>(max_edge_cols, 1);
            const size_t admitted =
                std::min(budget / bytes_per_edge, int_max_edges) / std::max(degree, 1);
            width = static_cast<int>(std::min<size_t>(width, std::max<size_t>(admitted, 1)));
        }
        o.tile_atoms = width;
    }
    Json describe() const {
        const char *selected = direct ? "direct" : "generic";
        return {{"mode", selected},
                {"selected_mode", selected},
                {"execution_profile", profile == SP2_SPEED ? "speed" : "capacity"},
                {"precision", precision},
                {"math_mode", math_mode == SP2_MATH_TF32 ? "tf32" : (precision == 32 ? "strict_fp32" : "fp64")},
                {"tile_atoms", width},
                {"estimated_bytes_per_edge", bytes_per_edge},
                {"planner_budget_bytes", budget},
                {"contract_sha256", contract_digest},
                {"artifact_sha256", direct ? artifact->digest() : std::string{}},
                {"fallback_count", 0}};
    }
};
} // namespace sp2
