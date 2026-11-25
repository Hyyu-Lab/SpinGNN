#ifndef SPINGNN2_BACKBONE_HPP
#define SPINGNN2_BACKBONE_HPP

#include "blas.hpp"
#include "contract.hpp"
#include "jit_runtime.hpp"

#include <map>
#include <tuple>
#include <utility>

namespace sp2 {
inline std::vector<std::string> backbone_jit_symbols(const Json& model) {
    std::vector<std::string> names;
    for (const auto& entry : model.at("layers")) {
        if (!entry.contains("backbone")) continue;
        const auto prefix = "sp2_trunk_blas_" + entry.at("name").get<std::string>();
        names.push_back(prefix + "_epilogue");
        names.push_back(prefix + "_epilogue_reverse");
        names.push_back(prefix + "_weight_initial");
        names.push_back(prefix + "_env_fwd");
        names.push_back(prefix + "_rev_env");
        const auto& specs = entry.at("backbone").at("layers");
        for (std::size_t i = 0; i < specs.size(); ++i) {
            const auto layer = prefix + "_tp" + std::to_string(i);
            for (const char* suffix : {"_packed", "_scalar", "_adjoint", "_adjoint_scalar",
                                       "_environment", "_environment_scalar"})
                names.push_back(layer + suffix);
            if (!specs.at(i).at("linear").is_null()) {
                names.push_back(layer + "_linear_seed");
                names.push_back(layer + "_linear_output");
            }
        }
    }
    for (const char* name : {"sp2_trunk_hidden_silu", "sp2_trunk_hidden_ssp",
                             "sp2_trunk_reverse_mul", "sp2_trunk_scalar_adjoint"})
        names.push_back(name);
    return names;
}

// The generated CUDA epilogues and scalar contractions assign one edge
// per warp. A host plugin interprets the original count argument instead.
inline int jit_edge_warp_grid(int count) {
    return count != 0 ? (count - 1) / 4 + 1 : 0;
}

template<class T>
inline int jit_vec4_grid(int n, const T* p1, const T* p2 = nullptr, const T* p3 = nullptr) {
#ifdef SP2_ENABLE_CUDA
    if constexpr (std::is_same_v<T, float>) {
        if ((n & 3) == 0 &&
            (reinterpret_cast<std::uintptr_t>(p1) & 15u) == 0u &&
            (!p2 || (reinterpret_cast<std::uintptr_t>(p2) & 15u) == 0u) &&
            (!p3 || (reinterpret_cast<std::uintptr_t>(p3) & 15u) == 0u)) {
            const int n4 = n >> 2;
            return n4 ? (n4 - 1) / 128 + 1 : 0;
        }
    }
#else
    (void)p1; (void)p2; (void)p3;
#endif
    return n ? (n - 1) / 128 + 1 : 0;
}

template<class T>
struct Matrix {
    Matrix() = default;
    Matrix(const Contract& contract, const Json& ref, int nrows, int ncols, const Exec& exec = Exec()) {
        auto vals = contract.array<T>(ref);
        if (nrows <= 0 || ncols <= 0 || vals.size() != static_cast<std::size_t>(nrows) * ncols) {
            throw std::invalid_argument("Model matrix has an invalid shape");
        }
        rows = nrows;
        cols = ncols;
        data = upload(exec, vals);
    }
    Tensor<T> tensor() const { return {data.data(), rows, cols}; }
    DeviceArray<T> data;
    int rows = 0, cols = 0;
};

inline int dim(const Contract& contract, const Json& ref, std::size_t axis) {
    auto n = contract.shape(ref).at(axis);
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Model dimension exceeds supported integer limits");
    }
    return static_cast<int>(n);
}

template<class T>
struct MlpTape {
    std::vector<Tensor<T>> slopes;
};

template<class T>
class MLP {
public:
    MLP() = default;
    MLP(const Contract& contract, const Json& spec) {
        for (const auto& layer : spec.at("layers")) {
            const auto& ref = layer.at("weight");
            auto activation = layer.at("activation");
            int kind = 0;
            if (activation == "silu") kind = 1;
            else if (activation == "ssp") kind = 2;
            else if (!activation.is_null()) throw std::invalid_argument("Unsupported MLP activation");
            const int width = dim(contract, ref, 0);
            if (!layers.empty() && layers.back().weight.cols != width)
                throw std::invalid_argument("Incompatible MLP layer widths");
            layers.push_back({Matrix<T>(contract, ref, width, dim(contract, ref, 1)), kind});
#ifdef SP2_ENABLE_CUDA
            if constexpr (std::is_same_v<T,float>) {
                // Pre-transpose all reverse MLP weights once at model load so
                // every reverse projection uses NN cuBLASLt layout (no CUBLAS_OP_T).
                auto& layer = layers.back();
                const int cols = layer.weight.cols;
                if (width >= 2 && cols > 1) {
                    auto values = contract.array<T>(ref);
                    std::vector<T> packed(values.size());
                    for (int i = 0; i < width; ++i)
                        for (int j = 0; j < cols; ++j)
                            packed[static_cast<size_t>(j)*width+i] = values[static_cast<size_t>(i)*cols+j];
                    layer.reverse_weight = upload(Exec(), packed);
                }
            }
#endif
        }
        if (layers.empty()) throw std::invalid_argument("MLP must contain at least one layer");
    }

    int input_width() const { return layers.front().weight.rows; }
    int output_width() const { return layers.back().weight.cols; }

    Tensor<T> forward(Workspace<T>& w, Tensor<T> input, MlpTape<T>& tape, bool save = true) const {
        ProfileScope<T> timer(w, Profile::mlp);
        // Keep only the output and activation slopes between calls. Intermediate
        // GEMM outputs share scratch slots with other MLPs and convolutions.
        auto result = w.make(input.rows, output_width());
        tape.slopes.assign(layers.size(), {});
        if (save) {
            for (std::size_t i = 0; i < layers.size(); ++i) {
                if (layers[i].kind) tape.slopes[i] = w.make(input.rows, layers[i].weight.cols);
            }
        }
        auto mark = w.mark();
        int max_width = 0;
        for (std::size_t i = 0; i + 1 < layers.size(); ++i)
            max_width = std::max(max_width, layers[i].weight.cols);
        auto scratch1 = w.make(input.rows, max_width);
        auto scratch2 = w.make(input.rows, max_width);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            auto output = i + 1 == layers.size() ? result : (i % 2 ? scratch2 : scratch1);
            output.cols = layer.weight.cols;
            gemm(w.exec, output, input, layer.weight.tensor());
            auto slopes = tape.slopes[i];
            const int kind = layer.kind;
            if (kind && w.jit) {
                const char* kernel = kind == 1 ? "sp2_trunk_hidden_silu" : "sp2_trunk_hidden_ssp";
                const int n = output.size();
                w.jit->launch_grid(w.exec, kernel, jit_vec4_grid(n, output.ptr, slopes.ptr),
                                   128, 0, n, output.ptr, slopes.ptr);
            } else if (kind) {
                for_each(w.exec, "activate scalar MLP", output.size(), KOKKOS_LAMBDA (int j) {
                    const T x = output.ptr[j];
                    if (kind == 1) {
                        const T den = T(1) + Kokkos::exp(-x);
                        output.ptr[j] = x / den;
                        if (slopes) {
                            const T sig = x < T(0) ? Kokkos::exp(x)/(T(1) + Kokkos::exp(x)) : T(1)/den;
                            slopes.ptr[j] = sig + output.ptr[j] * (T(1) - sig);
                        }
                    } else {
                        output.ptr[j] = x > T(20) ? x : Kokkos::log1p(Kokkos::exp(x));
                        output.ptr[j] -= T(0.693147180559945309417232121458);
                        if (slopes) slopes.ptr[j] = x > T(20) ? T(1) : T(1)/(T(1)+Kokkos::exp(-x));
                    }
                });
            }
            input = output;
        }
        w.rewind(mark);
        return result;
    }

    Tensor<T> reverse(Workspace<T>& w, Tensor<T> seed, const MlpTape<T>& tape) const {
        auto result = w.make(seed.rows, input_width());
        auto mark = w.mark();
        int max_width = 0;
        for (const auto& layer : layers) max_width = std::max(max_width, layer.weight.cols);
        auto scratch1 = w.make(seed.rows, max_width);
        auto scratch2 = w.make(seed.rows, max_width);
        std::size_t depth = layers.size();
        Tensor<T> fused_seed;
        bool own_seed = false;
        if (w.jit && depth >= 2 && output_width() == 1 && layers.back().kind == 0) {
            const auto slope = tape.slopes.at(depth - 2);
            if (slope) {
                fused_seed = ((depth - 2) % 2) ? scratch1 : scratch2;
                fused_seed.cols = slope.cols;
                w.jit->launch(w.exec, "sp2_trunk_scalar_adjoint", fused_seed.size(), 128,
                              fused_seed.cols, seed.ptr, layers.back().weight.data.data(),
                              slope.ptr, fused_seed.ptr);
                seed = fused_seed;
                own_seed = true;
                --depth;
            }
        }
        for (std::size_t i = depth; i-- > 0;) {
            const auto& layer = layers[i];
            auto slope = tape.slopes.at(i);
            if (fused_seed) {
                // The projection adjoint above has already applied this slope.
                slope = {};
                fused_seed = {};
            }
            if (slope) {
                auto target = own_seed ? seed : (i % 2 ? scratch1 : scratch2);
                target.cols = seed.cols;
                if (w.jit) {
                    const int n = seed.size();
                    w.jit->launch_grid(w.exec, "sp2_trunk_reverse_mul",
                                       jit_vec4_grid(n, seed.ptr, slope.ptr, target.ptr),
                                       128, 0, n, seed.ptr, slope.ptr, target.ptr);
                } else {
                    auto input = seed;
                    for_each(w.exec, "reverse scalar activation", seed.size(), KOKKOS_LAMBDA (int j) {
                        target.ptr[j] = input.ptr[j] * slope.ptr[j];
                    });
                }
                seed = target;
                own_seed = true;
            }
            auto out = i == 0 ? result : (i % 2 ? scratch2 : scratch1);
            out.cols = layer.weight.rows;
#ifdef SP2_ENABLE_CUDA
            if (layer.reverse_weight.data()) {
                Tensor<T> packed{layer.reverse_weight.data(), layer.weight.cols, layer.weight.rows};
                gemm(w.exec, out, seed, packed);
            } else
#endif
                gemm(w.exec, out, seed, layer.weight.tensor(), false, true);
            seed = out;
            own_seed = true;
        }
        w.rewind(mark);
        return result;
    }
private:
    struct Layer {
        Matrix<T> weight;
        int kind;
#ifdef SP2_ENABLE_CUDA
        DeviceArray<T> reverse_weight{};
#endif
    };
    std::vector<Layer> layers;
};

template<class T>
class IrrepLinear {
public:
    IrrepLinear() = default;
    IrrepLinear(const Contract& contract, const Json& spec) {
        if (spec.is_null()) return;
        active = true;
        in_mul = spec.at("mul_in");
        out_mul = spec.at("mul_out");
        in_dim = spec.at("dim_in");
        out_dim = spec.at("dim_out");
        Exec exec;
        for (const auto& block : spec.at("blocks")) {
            const int components = block.at("components");
            const int out = block.at("out");
            const T norm = block.at("normalization");
            if (components <= 0 || out < 0 || out > out_dim - components)
                throw std::invalid_argument("Invalid output range in equivariant Linear");
            for (const auto& group : block.at("groups")) {
                const int start = group.at("start");
                const int stop = group.at("stop");
                if (start < 0 || stop <= start || stop > in_dim || (stop - start) % components)
                    throw std::invalid_argument("Invalid input range in equivariant Linear");
                const auto& ref = group.at("weight");
                const int n = (stop - start) / components;
                Matrix<T> wt(contract, ref, out_mul, in_mul * n, exec);
                DeviceArray<T> wt_t;
#ifdef SP2_ENABLE_CUDA
                {
                    auto values = contract.array<T>(ref);
                    const int r = out_mul, c = in_mul * n;
                    std::vector<T> trans(values.size());
                    for (int i = 0; i < r; ++i)
                        for (int j = 0; j < c; ++j)
                            trans[static_cast<std::size_t>(j) * r + i] = values[static_cast<std::size_t>(i) * c + j];
                    wt_t = upload(exec, trans);
                }
#endif
                groups.push_back({std::move(wt), std::move(wt_t), start, n, out, components, norm});
            }
        }
        build_pack_tables(exec);
    }

    Tensor<T> forward(Workspace<T>& w, Tensor<T> in) const { return apply(w, in, false); }
    Tensor<T> reverse(Workspace<T>& w, Tensor<T> in) const { return apply(w, in, true); }
    int output_width(int input_width) const { return active ? out_mul * out_dim : input_width; }
    int packed_width(int width, bool reverse) const {
        if (!active) return width;
        int total = 0;
        for (const auto& g : groups)
            total += g.component * (reverse ? out_mul : in_mul * g.count);
        return total;
    }

    Tensor<T> project_packed(Workspace<T>& w, Tensor<T> buffer, int normal_width, bool reverse) const {
        if (!active) return buffer;
        auto result = w.make(buffer.rows, packed_width(normal_width, !reverse));
        int offset_in = 0, offset_out = 0;
        for (const auto& g : groups) {
            Tensor<T> in{buffer.ptr + static_cast<std::size_t>(offset_in) * buffer.rows,
                         buffer.rows * g.component, reverse ? out_mul : in_mul * g.count};
            Tensor<T> out{result.ptr + static_cast<std::size_t>(offset_out) * buffer.rows,
                          result.rows * g.component, reverse ? in_mul * g.count : out_mul};
#ifdef SP2_ENABLE_CUDA
            if (!reverse && g.weight_t.data()) {
                Tensor<T> w_nn{g.weight_t.data(), in_mul * g.count, out_mul};
                gemm(w.exec, out, in, w_nn, false, false, g.norm);
            } else
#endif
                gemm(w.exec, out, in, g.weight.tensor(), false, !reverse, g.norm);
            offset_in += g.component * in.cols;
            offset_out += g.component * out.cols;
        }
        return result;
    }

    bool present() const { return active; }

public:
    struct PackSlot { int group_off, comp, g_cols, m, ch, col; };

    // Single-pass packed equivariant Linear when scratch fits within 16M elements (~64 MiB FP32);
    // otherwise falls back to bounded per-group scratch (reusing NN weight_t) to avoid workspace bloat
    // or INT_MAX element overflow on large l_max=2 heads (e.g. soc-e2ip on 48-80 GB GPUs).
    Tensor<T> apply(Workspace<T>& w, Tensor<T> src, bool reverse) const {
        if (!active) return src;
        const int width = reverse ? in_mul * in_dim : out_mul * out_dim;
        if (src.cols != (reverse ? out_mul * out_dim : in_mul * in_dim))
            throw std::logic_error("Equivariant Linear tile has an invalid width");
        auto dst = w.make(src.rows, width);
        if (groups.empty()) {
            zero(w, dst);
            return dst;
        }
        const int rows = src.rows;
        const int p_in = reverse ? packed_rev_in : packed_fwd_in;
        const int p_out = reverse ? packed_fwd_in : packed_rev_in;
        const std::size_t peak_elems =
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(std::max(p_in, p_out));
        if (peak_elems > (16ull << 20)) {
            zero(w, dst);
            for (const auto& g : groups) {
                const auto mark = w.mark();
                const int comp = g.component, cnt = g.count;
                const int gi = in_mul * cnt, go = out_mul;
                const int src_d = reverse ? out_dim : in_dim;
                const int dst_d = reverse ? in_dim : out_dim;
                const int in_cols = reverse ? go : gi;
                const int out_cols = reverse ? gi : go;
                auto in_buf = w.make(rows * comp, in_cols);
                auto out_buf = w.make(rows * comp, out_cols);
                const int g_start = g.start, g_out = g.out;
                for_each(w.exec, "pack equivariant Linear group", in_buf.size(), KOKKOS_LAMBDA(int idx) {
                    const int rc = idx / in_cols, ch = idx - rc * in_cols;
                    const int e = rc / comp, m = rc - e * comp;
                    const int col = reverse ? ch * src_d + g_out + m
                                            : (ch / cnt) * src_d + g_start + (ch % cnt) * comp + m;
                    in_buf.ptr[idx] = src(e, col);
                });
#ifdef SP2_ENABLE_CUDA
                if (!reverse && g.weight_t.data()) {
                    Tensor<T> w_nn{g.weight_t.data(), gi, go};
                    gemm(w.exec, out_buf, in_buf, w_nn, false, false, g.norm);
                } else
#endif
                    gemm(w.exec, out_buf, in_buf, g.weight.tensor(), false, !reverse, g.norm);
                for_each(w.exec, "unpack equivariant Linear group", out_buf.size(), KOKKOS_LAMBDA(int idx) {
                    const int rc = idx / out_cols, ch = idx - rc * out_cols;
                    const int e = rc / comp, m = rc - e * comp;
                    const int col = reverse ? (ch / cnt) * dst_d + g_start + (ch % cnt) * comp + m
                                            : ch * dst_d + g_out + m;
                    dst(e, col) += out_buf.ptr[idx];
                });
                w.rewind(mark);
            }
            return dst;
        }
        const auto mark = w.mark();
        auto packed = w.make(rows, p_in);
        const auto pmap = reverse ? rev_pack_slots : fwd_pack_slots;
        for_each(w.exec, "pack equivariant Linear", rows * p_in, KOKKOS_LAMBDA(int idx) {
            const int e = idx / p_in;
            const int q = idx - e * p_in;
            const auto s = pmap(q);
            const std::size_t dst_i = static_cast<std::size_t>(s.group_off) * rows +
                (static_cast<std::size_t>(e) * s.comp + s.m) * s.g_cols + s.ch;
            packed.ptr[dst_i] = src(e, s.col);
        });
        auto result = project_packed(w, packed, 0, reverse);
        const auto uoff = reverse ? rev_unpack_offsets : fwd_unpack_offsets;
        const auto uslot = reverse ? rev_unpack_slots : fwd_unpack_slots;
        for_each(w.exec, "unpack equivariant Linear", rows * width, KOKKOS_LAMBDA(int idx) {
            const int e = idx / width;
            const int col = idx - e * width;
            T sum = T(0);
            for (int k = uoff(col); k < uoff(col + 1); ++k) {
                const auto s = uslot(k);
                const std::size_t src_i = static_cast<std::size_t>(s.group_off) * rows +
                    (static_cast<std::size_t>(e) * s.comp + s.m) * s.g_cols + s.ch;
                sum += result.ptr[src_i];
            }
            dst.ptr[idx] = sum;
        });
        w.rewind(mark);
        return dst;
    }
private:
    void build_pack_tables(const Exec& exec) {
        std::vector<PackSlot> fp, rp;
        std::vector<std::vector<PackSlot>> fu(out_mul * out_dim), ru(in_mul * in_dim);
        int off_in = 0, off_out = 0;
        for (const auto& g : groups) {
            const int comp = g.component, cnt = g.count;
            const int gi = in_mul * cnt, go = out_mul;
            for (int m = 0; m < comp; ++m) {
                for (int ch = 0; ch < gi; ++ch) {
                    const int src_col = (ch / cnt) * in_dim + g.start + (ch % cnt) * comp + m;
                    fp.push_back({off_in, comp, gi, m, ch, src_col});
                    ru[src_col].push_back({off_in, comp, gi, m, ch, src_col});
                }
                for (int u = 0; u < go; ++u) {
                    const int dst_col = u * out_dim + g.out + m;
                    rp.push_back({off_out, comp, go, m, u, dst_col});
                    fu[dst_col].push_back({off_out, comp, go, m, u, dst_col});
                }
            }
            off_in += comp * gi;
            off_out += comp * go;
        }
        packed_fwd_in = off_in;
        packed_rev_in = off_out;
        fwd_pack_slots = upload(exec, fp);
        rev_pack_slots = upload(exec, rp);
        auto pack_csr = [&](const std::vector<std::vector<PackSlot>>& buckets,
                            DeviceArray<int>& offs, DeviceArray<PackSlot>& slots) {
            std::vector<int> o{0};
            std::vector<PackSlot> flat;
            for (const auto& b : buckets) {
                flat.insert(flat.end(), b.begin(), b.end());
                o.push_back(static_cast<int>(flat.size()));
            }
            offs = upload(exec, o);
            slots = upload(exec, flat);
        };
        pack_csr(fu, fwd_unpack_offsets, fwd_unpack_slots);
        pack_csr(ru, rev_unpack_offsets, rev_unpack_slots);
    }

    struct Group { Matrix<T> weight; DeviceArray<T> weight_t; int start, count, out, component; T norm; };
    std::vector<Group> groups;
    DeviceArray<PackSlot> fwd_pack_slots, rev_pack_slots, fwd_unpack_slots, rev_unpack_slots;
    DeviceArray<int> fwd_unpack_offsets, rev_unpack_offsets;
    int packed_fwd_in = 0, packed_rev_in = 0;
    bool active = false;
    int in_mul = 0, out_mul = 0, in_dim = 0, out_dim = 0;
};

template<class T>
class GatherProduct {
public:
    GatherProduct(const Contract& contract, const Json& spec, int scalars, bool last) {
        d1 = spec.at("dim_in1");
        d2 = spec.at("dim_in2");
        d3 = spec.at("dim_out");
        m1 = spec.at("mul_in1");
        m = spec.at("mul_out");
        if (d1 <= 0 || d2 <= 0 || d3 <= 0 || m <= 0 || (m1 != 1 && m1 != m)
            || scalars < 0 || scalars > d3)
            throw std::invalid_argument("Unsupported Wigner-3j dimensions");
        scalar_width = scalars;
        const auto lhs = contract.array<int>(spec.at("offsets_in1"));
        const auto rhs = contract.array<int>(spec.at("offsets_in2"));
        const auto out = contract.array<int>(spec.at("offsets_out"));
        const auto paths = contract.array<int>(spec.at("w3j_indices"));
        const auto weights = contract.array<T>(spec.at("scaled_w3j"));
        if (paths.size() != weights.size() * 6) throw std::invalid_argument("Invalid Wigner-3j table");
        std::vector<std::vector<Path>> by_left(d1), by_right(d2), by_out(d3);
        std::vector<std::vector<Path>> scalar_by_left(last ? d1 : 0);
        std::vector<std::vector<Path>> scalar_by_right(last ? d2 : 0);
        auto component = [&](const auto& offsets, std::size_t path, int ip, int im, int extent) {
            int i = paths[path*6+ip];
            int m_local = paths[path*6+im];
            if (i < 0 || static_cast<std::size_t>(i)+1 >= offsets.size() || offsets[i] < 0
                || offsets[i] > extent || m_local < 0 || m_local >= offsets[i+1] - offsets[i]
                || m_local >= extent - offsets[i]) throw std::invalid_argument("Invalid Wigner-3j entry");
            return offsets[i] + m_local;
        };
        for (std::size_t p = 0; p < weights.size(); ++p) {
            int i = component(lhs, p, 0, 3, d1);
            int j = component(rhs, p, 1, 4, d2);
            int k = component(out, p, 2, 5, d3);
            const Path path{i, j, k, weights[p]};
            by_left[i].push_back(path);
            by_right[j].push_back(path);
            by_out[k].push_back(path);
            if (last && k < scalar_width) {
                scalar_by_left[i].push_back(path);
                scalar_by_right[j].push_back(path);
            }
        }
        left = pack(by_left);
        right = pack(by_right);
        output = pack(by_out);
        if (last) {
            scalar_left = pack(scalar_by_left);
            scalar_right = pack(scalar_by_right);
        }
    }

    Tensor<T> forward(Workspace<T>& w, Tensor<T> features, Tensor<T> env,
                      DeviceArray<int> center, bool scalars_only) const {
        if (features.cols != m1 * d1 || env.cols != m * d2)
            throw std::logic_error("Allegro tensor product tile has an invalid width");
        const int n3 = scalars_only ? scalar_width : d3;
        auto out = w.make(features.rows, m * n3);
        const int n1 = d1, n2 = d2, nm1 = m1;
        const auto p = output.values;
        const auto rowptr = output.offset;
        for_each(w.exec, "gather sparse Wigner forward", out.size(), KOKKOS_LAMBDA(int idx) {
            const int c = idx / out.cols, u = idx % out.cols / n3, k = idx % n3;
            const int i0 = (nm1 == 1 ? 0 : u) * n1, j0 = u * n2;
            const int a = center(c);
            T sum = 0;
            for (int q = rowptr(k); q < rowptr(k+1); ++q) {
                const auto path = p(q);
                sum += (features(c, i0 + path.i) * env(a, j0 + path.j)) * path.w;
            }
            out.ptr[idx] = sum;
        });
        return out;
    }

    std::array<Tensor<T>,2> reverse(Workspace<T>& w, Tensor<T> adjoint, Tensor<T> features,
                                    Tensor<T> env, DeviceArray<int> center, DeviceArray<int> offsets,
                                    bool scalars_only, int scalar_start = 0) const {
        auto dfeat = w.make(features.rows, features.cols);
        auto denv = w.make(env.rows, env.cols);
        const int n1 = d1, n2 = d2, n3 = scalars_only ? scalar_width : d3, nm1 = m1, nm = m;
        const auto& lpaths = scalars_only ? scalar_left : left;
        const auto& rpaths = scalars_only ? scalar_right : right;
        const auto pl = lpaths.values, pr = rpaths.values;
        const auto lp = lpaths.offset, rp = rpaths.offset;
        for_each(w.exec, "gather sparse Wigner edge adjoint", dfeat.size(), KOKKOS_LAMBDA(int idx) {
            const int c = idx / dfeat.cols, u = idx % dfeat.cols / n1, i = idx % n1;
            const int a = center(c);
            T sum = 0;
            for (int channel = u; channel < (nm1 == 1 ? nm : u+1); ++channel) {
                for (int q = lp(i); q < lp(i+1); ++q) {
                    const auto path = pl(q);
                    sum += (adjoint(c, scalar_start + channel*n3 + path.k) * path.w) *
                        env(a, channel*n2 + path.j);
                }
            }
            dfeat.ptr[idx] = sum;
        });
        // One work item owns each center's adjoint component and reduces its
        // incident edges in input order. No scatter atomics or per-edge env copy.
        for_each(w.exec, "reduce sparse Wigner environment adjoint", denv.size(), KOKKOS_LAMBDA(int idx) {
            const int a = idx / denv.cols, u = idx % denv.cols / n2, j = idx % n2;
            T sum = 0;
            for (int c = offsets(a); c < offsets(a+1); ++c) {
                for (int q = rp(j); q < rp(j+1); ++q) {
                    const auto path = pr(q);
                    sum += (adjoint(c, scalar_start + u*n3 + path.k) * path.w) *
                        features(c, (nm1 == 1 ? 0 : u)*n1 + path.i);
                }
            }
            denv.ptr[idx] = sum;
        });
        return {dfeat, denv};
    }

    int d1 = 0, d2 = 0, d3 = 0, m1 = 0, m = 0, scalar_width = 0;
    struct Path { int i, j, k; T w; };
private:
    struct Table { DeviceArray<Path> values; DeviceArray<int> offset; };
    Table left, right, output, scalar_left, scalar_right;

    static Table pack(const std::vector<std::vector<Path>>& entries) {
        std::vector<int> offsets = {0};
        std::vector<Path> values;
        for (const auto& row : entries) {
            values.insert(values.end(), row.begin(), row.end());
            offsets.push_back(static_cast<int>(values.size()));
        }
        Exec exec;
        return {upload(exec, values), upload(exec, offsets)};
    }
};

template<class T>
struct HeadTape {
    Tensor<T> preactivation;
    Tensor<T> value;
};

template<class T>
class TensorHead {
public:
    TensorHead(const Contract& contract, const Json& spec)
        : in(contract, spec.at("linear_in")), out(contract, spec.at("linear_out")),
          epsilon(spec.at("epsilon")), flat_dim(spec.at("flat_dim")) {
        const auto widths = spec.at("norm_components").get<std::vector<int>>();
        if (widths.empty() || flat_dim <= 0 || !(epsilon > T(0)))
            throw std::invalid_argument("Invalid normalization in tensor head");
        std::vector<int> offsets = {0};
        for (const int width : widths) {
            if (width <= 0 || width > std::numeric_limits<int>::max() - offsets.back())
                throw std::invalid_argument("Invalid norm component size in tensor head");
            offsets.push_back(offsets.back() + width);
        }
        norm_components = upload(Exec(), offsets);
        const auto& key = spec.at("cartesian_basis");
        if (!key.is_null()) {
            const auto& bref = spec.at(key.get<std::string>());
            basis = Matrix<T>(contract, bref, flat_dim, 9);
#ifdef SP2_ENABLE_CUDA
            auto vals = contract.array<T>(bref);
            std::vector<T> trans(vals.size());
            for (int i = 0; i < flat_dim; ++i)
                for (int j = 0; j < 9; ++j)
                    trans[static_cast<std::size_t>(j) * flat_dim + i] = vals[static_cast<std::size_t>(i) * 9 + j];
            basis_t = upload(Exec(), trans);
#endif
        }
        width = basis.data.data() ? 9 : flat_dim;
    }

    Tensor<T> forward(Workspace<T>& w, Tensor<T> features, HeadTape<T>& tape) const {
        tape.preactivation = in.forward(w, features);
        auto activated = normalize(w, tape.preactivation, {});
        auto value = out.forward(w, activated);
        if (value.cols != flat_dim) throw std::logic_error("Tensor head output width mismatch");
        if (basis.data.data()) {
            tape.value = w.make(value.rows, width);
            gemm(w.exec, tape.value, value, basis.tensor());
        } else {
            tape.value = value;
        }
        return tape.value;
    }

    Tensor<T> reverse(Workspace<T>& w, Tensor<T> seed, const HeadTape<T>& tape) const {
        if (basis.data.data()) {
            auto grad = w.make(seed.rows, flat_dim);
#ifdef SP2_ENABLE_CUDA
            if (basis_t.data()) {
                Tensor<T> bt{basis_t.data(), 9, flat_dim};
                gemm(w.exec, grad, seed, bt);
            } else
#endif
                gemm(w.exec, grad, seed, basis.tensor(), false, true);
            seed = grad;
        }
        auto adjoint = out.reverse(w, seed);
        return in.reverse(w, normalize(w, tape.preactivation, adjoint));
    }

public:
    Tensor<T> normalize(Workspace<T>& w, Tensor<T> x, Tensor<T> adjoint) const {
        auto result = w.make(x.rows, x.cols);
        const auto offsets = norm_components;
        const int blocks = static_cast<int>(offsets.extent(0))-1;
        const T floor = epsilon * epsilon;
        for_each(w.exec, "tensor head norm activation", x.rows * blocks, KOKKOS_LAMBDA(int index) {
            const int i = index / blocks, n = index % blocks;
            const int start = offsets(n), stop = offsets(n+1);
            T sq = 0, inner = 0;
            for (int c = start; c < stop; ++c) {
                const T a = x(i,c);
                sq += a*a;
                if (adjoint) inner += adjoint(i,c)*a;
            }
            const T norm = Kokkos::sqrt(sq < floor ? floor : sq);
            const T scale = T(1)/(T(1) + Kokkos::exp(-norm));
            const T ds = sq < floor ? T(0) : scale*(T(1)-scale)/norm;
            for (int c = start; c < stop; ++c) {
                result(i,c) = adjoint ? scale*adjoint(i,c) + x(i,c)*ds*inner : scale*x(i,c);
            }
        });
        return result;
    }

private:
    IrrepLinear<T> in, out;
    T epsilon;
    int flat_dim, width;
    Matrix<T> basis;
    DeviceArray<T> basis_t;
    DeviceArray<int> norm_components;
};

template<class T>
struct LatentTape {
    Tensor<T> cutoff, cutoff_derivative, update;
    MlpTape<T> mlp;
    T old = 0, next = 1;
    bool residual = false;
};

template<class T>
struct BackboneTape {
    struct Layer {
        LatentTape<T> latent;
        MlpTape<T> weights_mlp;
        Tensor<T> features;
        Tensor<T> env;
        Tensor<T> weights;
        bool initial = false;
    };
    std::vector<Layer> layers;
    LatentTape<T> final_latent;
    std::map<std::string, HeadTape<T>> heads;
    Tensor<T> attrs;
    DeviceArray<int> center, offsets, species;
    int attr_dim = 0, ni = 0, nj = 0;
    int centers = 0;
};

template<class T>
class Backbone {
public:
    Backbone(const Contract& contract, const Json& spec, const std::string& name)
        : final_mlp(contract, spec.at("final_latent")), latent_field(spec.at("latent_out_field")),
          cos_ratio(spec.at("r_start_cos_ratio")), poly_p(spec.at("polynomial_cutoff_p")),
          in_pad(spec.at("input_pad")), embed(spec.at("embed_initial_edge")),
          has_residual(spec.at("latent_resnet")),
          nonzero(spec.value("edge_prune", std::string("nonzero")) == "nonzero"),
          epilogue("sp2_trunk_blas_" + name + "_epilogue"), epilogue_reverse(epilogue + "_reverse"),
          weight_initial_sym("sp2_trunk_blas_" + name + "_weight_initial"),
          env_fwd_sym("sp2_trunk_blas_" + name + "_env_fwd"),
          rev_env_sym("sp2_trunk_blas_" + name + "_rev_env") {
        cosine = spec.at("cutoff_type") == "cosine";
        if (!cosine && spec.at("cutoff_type") != "polynomial")
            throw std::invalid_argument("Unsupported trunk cutoff");
        if (poly_p < 2 || in_pad < 0) throw std::invalid_argument("Invalid backbone geometry");
        const auto& layer_specs = spec.at("layers");
        for (std::size_t i = 0; i < layer_specs.size(); ++i)
            layers.emplace_back(contract, layer_specs[i], i + 1 == layer_specs.size(),
                                "sp2_trunk_blas_" + name + "_tp" + std::to_string(i));
        if (layers.empty()) throw std::invalid_argument("Backbone has no interaction layers");
        const auto& channels = spec.at("weighted_channels");
        weight_per_edge = channels.at("weight_numel");
        mul = channels.at("multiplicity");
        irreps = channels.at("irreps");
        const auto wi = contract.array<int>(channels.at("w_index"));
        attr_cols = static_cast<int>(wi.size());
        if (mul < 1 || irreps < 1 || weight_per_edge != mul * irreps || wi.empty())
            throw std::invalid_argument("Invalid weighted channels layout");
        std::vector<std::vector<int>> components(irreps);
        for (int p = 0; p < attr_cols; ++p) {
            if (wi[p] < 0 || wi[p] >= irreps) throw std::invalid_argument("Invalid w_index entry");
            components[wi[p]].push_back(p);
        }
        std::vector<int> packed, offsets = {0};
        for (const auto& block : components) {
            packed.insert(packed.end(), block.begin(), block.end());
            offsets.push_back(static_cast<int>(packed.size()));
        }
        Exec exec;
        windex = upload(exec, wi);
        channels_by_irrep = upload(exec, packed);
        offsets_by_irrep = upload(exec, offsets);

        const auto& norm_ref = spec.at("env_sum_normalizations");
        const auto norm_shape = contract.shape(norm_ref);
        if (norm_shape.empty() || norm_shape.size() > 2 || norm_shape[0] != layers.size())
            throw std::invalid_argument("Invalid environment norm shape");
        n_species = norm_shape.size() == 2 ? dim(contract, norm_ref, 1) : 0;
        normalization = upload(exec, contract.array<T>(norm_ref));
        cutoffs = contract.array<T>(spec.at("cutoffs"));
        if (cutoffs.size() != layers.size()+1 ||
            std::any_of(cutoffs.begin(), cutoffs.end(), [](T x) { return !(x > T(0)); }))
            throw std::invalid_argument("Invalid Allegro per-layer cutoffs");
        residual = contract.array<T>(spec.at("residuals"));
        if (has_residual && residual.size() != 2*layers.size())
            throw std::invalid_argument("Invalid Allegro residual coefficients");
        for (const auto& kv : spec.at("heads").items())
            heads.emplace(kv.key(), TensorHead<T>(contract, kv.value()));
    }

    std::map<std::string,Tensor<T>> forward(Workspace<T>& w, Tensor<T> attrs,
            Tensor<T> invariants, Tensor<T> node_i, Tensor<T> node_j, Tensor<T> length,
            DeviceArray<int> center, DeviceArray<int> offsets, DeviceArray<int> species,
            BackboneTape<T>& tape) const {
        tape.layers.assign(layers.size(), {});
        tape.heads.clear();
        tape.center = center;
        tape.offsets = offsets;
        tape.species = species;
        tape.centers = static_cast<int>(offsets.extent(0)) - 1;
        tape.ni = node_i.cols;
        tape.nj = node_j.cols;
        tape.attr_dim = attrs.cols;
        if (attrs.cols + in_pad != attr_cols) throw std::logic_error("Allegro harmonic width mismatch");
        if (in_pad) {
            const int real_width = attrs.cols;
            auto padded = w.make(attrs.rows, attr_cols);
            for_each(w.exec, "pad edge harmonics", padded.size(), KOKKOS_LAMBDA(int index) {
                const int col = index % padded.cols;
                padded.ptr[index] = col < real_width ? attrs(index / padded.cols, col) : T(0);
            });
            attrs = padded;
        }
        tape.attrs = attrs;
        auto inputs = concat_columns(w, node_i, node_j, invariants);
        Tensor<T> features = attrs;
        Tensor<T> latents;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            auto& lt = tape.layers[i];
            latents = update_latents(w, i, layer.mlp, inputs, latents, length, lt.latent);
            lt.weights = layer.weights.forward(w, latents, lt.weights_mlp);
            lt.initial = embed && i == 0;
            {
                ProfileScope<T> timer(w, Profile::tp_linear);
                if (lt.initial) features = weight_initial(w, attrs, lt.weights, lt.latent.cutoff);
                else features = mask(w, features, lt.latent.cutoff);
                lt.features = features;
                const int skip = lt.initial ? weight_per_edge : 0;
                auto env = environment(w, i, attrs, lt.weights, lt.latent.cutoff,
                                       offsets, species, skip);
                lt.env = layer.env_linear.forward(w, env);
                const bool scalars_only = i + 1 == layers.size() && heads.empty();
                if (w.jit) {
                    inputs = w.make(latents.rows, latents.cols + layer.tp.m * layer.scalars);
                    const int rows = features.rows;
                    const int tp_width = layer.tp.m * layer.tp.d3;
                    if (scalars_only) {
                        w.jit->launch_grid(w.exec, layer.scalar, jit_edge_warp_grid(rows), 128, 0,
                                          rows, features.ptr, lt.env.ptr, center.data(), offsets.data(),
                                          latents.ptr, latents.cols, inputs.ptr);
                    } else {
                        const bool projected = layer.linear.present();
                        auto next = w.make(rows, layer.linear.output_width(tp_width));
                        const auto mark = w.mark();
                        auto packed = projected ? w.make(rows, layer.linear.packed_width(tp_width, false)) : next;
#ifdef SP2_ENABLE_CUDA
                        const int grid = lt.env.rows * 2;
                        const std::size_t env_bytes =
                            static_cast<std::size_t>(lt.env.cols) * sizeof(T);
                        // <=48 KiB uses static smem; (48, 96] KiB uses portable dynamic opt-in smem
                        // (fits all sm_80/86/89/90/120); >96 KiB reads env directly from global memory.
                        const unsigned smem_bytes =
                            (env_bytes > 48u * 1024u && env_bytes <= 96u * 1024u)
                                ? static_cast<unsigned>(env_bytes)
                                : 0u;
#else
                        const int grid = rows;
                        const unsigned smem_bytes = 0u;
#endif
                        w.jit->launch_grid(w.exec, layer.forward, grid, 128, smem_bytes, rows,
                                          features.ptr, lt.env.ptr, center.data(), offsets.data(),
                                          latents.ptr, latents.cols, inputs.ptr, packed.ptr);
                        if (projected) {
                            auto result = layer.linear.project_packed(w, packed, tp_width, false);
                            w.jit->launch(w.exec, layer.output, next.size(), 128,
                                          rows, result.ptr, next.ptr);
                        }
                        features = next;
                        w.rewind(mark);
                    }
                } else {
                    auto contracted = layer.tp.forward(w, features, lt.env, center, scalars_only);
                    if (!scalars_only) features = layer.linear.forward(w, contracted);
                    inputs = latent_inputs(w, latents, contracted, layer.scalars,
                                           scalars_only ? layer.scalars : layer.tp.d3, layer.tp.m);
                }
            }
        }
        latents = update_latents(w, layers.size(), final_mlp, inputs, latents, length, tape.final_latent);
        std::map<std::string, Tensor<T>> result{{latent_field, latents}};
        for (const auto& entry : heads) {
            ProfileScope<T> timer(w, Profile::tp_linear);
            auto& ht = tape.heads[entry.first];
            auto raw = entry.second.forward(w, features, ht);
            result.emplace(entry.first, scale_rows(w, raw, tape.final_latent.cutoff));
        }
        return result;
    }

    std::array<Tensor<T>,5> reverse(Workspace<T>& w, const std::map<std::string,Tensor<T>>& seeds,
                                  const BackboneTape<T>& tape) const {
        if (seeds.empty()) return {};
        const auto E = tape.attrs.rows;
        const auto dim_latent = final_mlp.output_width();
        auto final_seed = seeds.find(latent_field);
        auto d_latents = final_seed != seeds.end() ? final_seed->second : w.make(E, dim_latent);
        if (final_seed == seeds.end()) zero(w, d_latents);
        Tensor<T> d_features;
        auto d_lengths = w.make(E, 1);
        zero(w, d_lengths);
        const auto final_cutoff = tape.final_latent.cutoff;
        Tensor<T> d_cutoff;
        for (const auto& seed : seeds) {
            if (seed.first == latent_field) continue;
            const auto head = heads.find(seed.first);
            if (head == heads.end()) throw std::logic_error("Unknown tensor head seed");
            if (!d_cutoff) {
                d_cutoff = w.make(E, 1);
                zero(w, d_cutoff);
            }
            const auto& ht = tape.heads.at(seed.first);
            const auto dy = scale_rows(w, seed.second, final_cutoff);
            auto delta = head->second.reverse(w, dy, ht);
            if (d_features) add_inplace(w, d_features, delta);
            else d_features = delta;
            const auto raw = ht.value, seed_tensor = seed.second;
            for_each(w.exec, "tensor head cutoff adjoint", E, KOKKOS_LAMBDA(int e) {
                T sum = 0;
                for (int c = 0; c < raw.cols; ++c) sum += seed_tensor(e,c)*raw(e,c);
                d_cutoff(e,0) += sum;
            });
        }
        if (d_cutoff) {
            const auto gd = tape.final_latent.cutoff_derivative;
            for_each(w.exec, "tensor head radius adjoint", E, KOKKOS_LAMBDA(int e) {
                d_lengths(e,0) = d_cutoff(e,0)*gd(e,0);
            });
        }
        Tensor<T> d_inputs;
        std::tie(d_inputs, d_latents) = reverse_latents(w, final_mlp, d_latents, d_lengths, tape.final_latent);
        auto d_attrs = w.make(E, tape.attrs.cols);
        zero(w, d_attrs);
        for (std::size_t i = layers.size(); i-- > 0;) {
            const auto& layer = layers[i];
            const auto& lt = tape.layers[i];
            const int expected = layer.mlp.output_width();
            auto z_latent = d_latents;
            auto z_input = d_inputs;
            const bool scalars_only = !d_features;
            std::array<Tensor<T>,2> rev;
            if (w.jit) {
                rev = {w.make(E, lt.features.cols), w.make(lt.env.rows, lt.env.cols)};
                const int tp_width = layer.tp.m * layer.tp.d3;
                const auto mark = w.mark();
                Tensor<T> dproduct = scalars_only ? Tensor<T>{} : d_features;
                if (!scalars_only && layer.linear.present()) {
                    auto packed = w.make(E, layer.linear.packed_width(tp_width, true));
                    w.jit->launch(w.exec, layer.seed, packed.size(), 128, E, d_features.ptr, packed.ptr);
                    dproduct = layer.linear.project_packed(w, packed, tp_width, true);
                }
                w.jit->launch_grid(w.exec, scalars_only ? layer.reverse_scalar : layer.reverse,
                                  jit_edge_warp_grid(E), 128, 0, E, lt.features.ptr, lt.env.ptr, tape.center.data(),
                                  z_input.ptr, expected, z_latent.ptr, dproduct.ptr, rev[0].ptr);
                const auto& env_kernel = scalars_only ? layer.env_scalar : layer.env;
#ifdef SP2_ENABLE_CUDA
                if constexpr (std::is_same_v<T,float>) {
                    const int groups = (lt.env.cols - 1) / 32 + 1;
                    w.jit->launch_grid(w.exec, env_kernel, lt.env.rows * groups, 128, 0,
                                       rev[1].size(), E, z_input.ptr, dproduct.ptr, lt.features.ptr,
                                       tape.offsets.data(), expected, rev[1].ptr);
                } else
#endif
                w.jit->launch(w.exec, env_kernel, rev[1].size(), 128, E, z_input.ptr,
                              dproduct.ptr, lt.features.ptr, tape.offsets.data(), expected, rev[1].ptr);
                w.rewind(mark);
            } else {
                for_each(w.exec, "accumulate latent input adjoint", z_latent.size(), KOKKOS_LAMBDA(int idx) {
                    z_latent.ptr[idx] += z_input(idx / expected, idx % expected);
                });
                auto d_contracted = z_input;
                if (!scalars_only) {
                    d_contracted = layer.linear.reverse(w, d_features);
                    const int scalar = layer.scalars, td = layer.tp.d3;
                    for_each(w.exec, "accumulate tensor scalar adjoint", E * layer.tp.m * scalar,
                             KOKKOS_LAMBDA(int index) {
                        const int width = d_contracted.cols / td * scalar;
                        const int e = index / width, offset = index % width;
                        d_contracted(e, offset/scalar*td + offset%scalar) += z_input(e, expected + offset);
                    });
                }
                // BEC and evidence heads are forward-only. The last contraction's
                // energy adjoint then touches only its scalar outputs.
                rev = layer.tp.reverse(w, d_contracted, lt.features, lt.env, tape.center, tape.offsets,
                                            scalars_only, scalars_only ? expected : 0);
            }
            d_features = rev[0];
            const auto d_env = layer.env_linear.reverse(w, rev[1]);
            auto d_weights = reverse_environment(w, i, d_env, d_features, lt, tape, d_attrs);
            const auto delta = layer.weights.reverse(w, d_weights, lt.weights_mlp);
            std::tie(d_inputs, d_latents) =
                reverse_latents(w, layer.mlp, d_latents, d_lengths, lt.latent, delta);
        }
        if (tape.attr_dim < d_attrs.cols) d_attrs = slice_columns(w, d_attrs, 0, tape.attr_dim);
        const int a = tape.ni, b = tape.nj;
        return {d_attrs, slice_columns(w, d_inputs, a+b, d_inputs.cols-a-b),
                slice_columns(w, d_inputs, 0, a), slice_columns(w, d_inputs, a, b), d_lengths};
    }

private:
    struct Layer {
        Layer(const Contract& c, const Json& spec, bool last, const std::string& name)
            : mlp(c, spec.at("latent")), weights(c, spec.at("env_embed_mlp")),
              env_linear(c, spec.at("env_linear")), linear(c, spec.at("linear")),
              tp(c, spec.at("tensor_product"), spec.at("n_scalar_outs"), last),
              scalars(spec.at("n_scalar_outs")), forward(name + "_packed"), scalar(name + "_scalar"),
              reverse(name + "_adjoint"), reverse_scalar(name + "_adjoint_scalar"),
              env(name + "_environment"), env_scalar(name + "_environment_scalar"),
              seed(name + "_linear_seed"), output(name + "_linear_output") {
            if (scalars < 0 || scalars > tp.d3) throw std::invalid_argument("Invalid scalar tensor channels");
        }
        MLP<T> mlp, weights;
        IrrepLinear<T> env_linear, linear;
        GatherProduct<T> tp;
        int scalars;
        std::string forward, scalar, reverse, reverse_scalar, env, env_scalar, seed, output;
    };

public:
    std::pair<Tensor<T>,Tensor<T>> cut_off(Workspace<T>& w, int i, Tensor<T> length) const {
        auto value = w.make(length.rows, 1), deriv = w.make(length.rows, 1);
        const T radius = cutoffs.at(i);
        const int p = poly_p;
        const bool use_cos = cosine, prune = nonzero;
        const T start = radius * cos_ratio;
        for_each(w.exec, "evaluate backbone cutoff", length.rows, KOKKOS_LAMBDA(int e) {
            const T r = length(e,0);
            T f = 0, df = 0;
            if (use_cos) {
                const T frequency = T(3.1415926535897932384626433832795)/(radius-start);
                const T phase = frequency*(Kokkos::max(start, Kokkos::min(r,radius))-start);
                f = T(0.5) * (Kokkos::cos(phase) + T(1));
                if (r >= start && r <= radius && (!prune || f > T(0)))
                    df = -T(0.5)*frequency*Kokkos::sin(phase);
            } else if (r < radius) {
                const T u = r/radius, tail = T(1)-u;
                T series = T(p)*T(p+1)/T(2);
                for (int k = p-2; k >= 0; --k) series = series*u + T(k+1)*T(k+2)/T(2);
                f = tail*tail*tail*series;
                df = -(T(p)*T(p+1)*T(p+2)/T(2))*Kokkos::pow(u,T(p-1))*tail*tail/radius;
            }
            value(e,0) = f;
            deriv(e,0) = df;
        });
        return {value, deriv};
    }

    Tensor<T> update_latents(Workspace<T>& w, int index, const MLP<T>& mlp,
            Tensor<T> input, Tensor<T> previous, Tensor<T> length, LatentTape<T>& tape) const {
        if (w.jit) {
            tape.cutoff = w.make(length.rows, 1);
            tape.cutoff_derivative = w.make(length.rows, 1);
        } else {
            std::tie(tape.cutoff, tape.cutoff_derivative) = cut_off(w, index, length);
        }
        tape.update = mlp.forward(w, input, tape.mlp);
        tape.residual = index > 0 && has_residual;
        if (tape.residual) {
            tape.old = residual.at(2*(index-1));
            tape.next = residual.at(2*(index-1)+1);
        }
        auto update = tape.update;
        auto out = w.make(update.rows, update.cols);
        auto cutoff = tape.cutoff;
        const bool res = tape.residual;
        const T old = tape.old, next = tape.next;
        if (previous && previous.cols != out.cols) throw std::logic_error("Latent residual width mismatch");
        if (w.jit) {
            w.jit->launch_grid(w.exec, epilogue, jit_edge_warp_grid(length.rows), 128, 0,
                              length.rows, length.ptr, update.ptr, previous.ptr, cutoff.ptr,
                              tape.cutoff_derivative.ptr, out.ptr, update.cols,
                              cutoffs.at(index), int(res), old, next);
            return out;
        }
        for_each(w.exec, "update cutoff-weighted latents", out.size(), KOKKOS_LAMBDA(int idx) {
            const int e = idx / out.cols;
            const T f = cutoff(e,0);
            const T v = f > T(0) ? update.ptr[idx]*f : T(0);
            const T prior = previous ? previous.ptr[idx] : T(0);
            out.ptr[idx] = res ? old*prior + next*v : (f > T(0) ? v : prior);
        });
        return out;
    }

    std::pair<Tensor<T>,Tensor<T>> reverse_latents(Workspace<T>& w, const MLP<T>& mlp,
            Tensor<T> seed, Tensor<T> d_lengths, const LatentTape<T>& tape,
            Tensor<T> additional = {}) const {
        auto d_update = w.make(seed.rows, seed.cols), previous = w.make(seed.rows, seed.cols);
        const auto cutoff = tape.cutoff, df = tape.cutoff_derivative, y = tape.update;
        const bool res = tape.residual;
        const T old = tape.old, next = tape.next;
        if (w.jit) {
            w.jit->launch_grid(w.exec, epilogue_reverse, jit_edge_warp_grid(seed.rows), 128, 0,
                              seed.rows, seed.ptr, additional.ptr, y.ptr, cutoff.ptr, df.ptr, d_lengths.ptr,
                              d_update.ptr, previous.ptr, seed.cols, int(res), old, next);
            return {mlp.reverse(w, d_update, tape.mlp), previous};
        }
        if (additional) add_inplace(w, seed, additional);
        for_each(w.exec, "latent update adjoint", seed.rows, KOKKOS_LAMBDA(int e) {
            const T f = cutoff(e,0);
            const T a = res ? old : (f > T(0) ? T(0) : T(1));
            const T b = res ? next : (f > T(0) ? T(1) : T(0));
            T dc = 0;
            for (int c = 0; c < seed.cols; ++c) {
                const T g = b*seed(e,c);
                dc += g*y(e,c);
                d_update(e,c) = f*g;
                previous(e,c) = a*seed(e,c);
            }
            d_lengths(e,0) += dc*df(e,0);
        });
        return {mlp.reverse(w, d_update, tape.mlp), previous};
    }

    static Tensor<T> mask(Workspace<T>& w, Tensor<T> input, Tensor<T> cutoff) {
        auto out = w.make(input.rows, input.cols);
        for_each(w.exec, "mask edges outside cutoff", out.size(), KOKKOS_LAMBDA(int index) {
            out.ptr[index] = cutoff(index / out.cols, 0) > T(0) ? input.ptr[index] : T(0);
        });
        return out;
    }

    static Tensor<T> scale_rows(Workspace<T>& w, Tensor<T> input, Tensor<T> cutoff) {
        auto out = w.make(input.rows, input.cols);
        for_each(w.exec, "scale tensor head by cutoff", input.size(), KOKKOS_LAMBDA(int index) {
            out.ptr[index] = cutoff(index / out.cols, 0)*input.ptr[index];
        });
        return out;
    }

    static Tensor<T> latent_inputs(Workspace<T>& w, Tensor<T> latents, Tensor<T> contracted,
                                   int scalars, int dim3, int mul3) {
        auto out = w.make(latents.rows, latents.cols + mul3 * scalars);
        for_each(w.exec, "concatenate latent scalars", out.size(), KOKKOS_LAMBDA(int i) {
            const int edge = i / out.cols, col = i % out.cols;
            if (col < latents.cols) out.ptr[i] = latents(edge,col);
            else {
                const int tail = col - latents.cols;
                out.ptr[i] = contracted(edge,tail / scalars * dim3 + tail % scalars);
            }
        });
        return out;
    }

    Tensor<T> weight_initial(Workspace<T>& w, Tensor<T> attrs, Tensor<T> weights, Tensor<T> cutoff) const {
        auto out = w.make(attrs.rows, mul * attr_cols);
        if (w.jit) {
            w.jit->launch_grid(w.exec, weight_initial_sym, jit_edge_warp_grid(attrs.rows), 128, 0,
                               attrs.rows, attrs.ptr, weights.ptr, cutoff.ptr, out.ptr);
            return out;
        }
        const auto wi = windex;
        const int dim = attr_cols, groups = irreps;
        for_each(w.exec, "weighted initial edge channels", out.size(), KOKKOS_LAMBDA(int i) {
            const int edge = i / out.cols, channel = i % out.cols / dim, component = i % dim;
            out.ptr[i] = cutoff(edge,0) > T(0) ?
                attrs(edge,component) * weights(edge,channel*groups + wi(component)) : T(0);
        });
        return out;
    }

    Tensor<T> environment(Workspace<T>& w, int layer, Tensor<T> attrs, Tensor<T> weights,
                          Tensor<T> cutoff, DeviceArray<int> offsets, DeviceArray<int> species, int skip) const {
        const int ncenters = static_cast<int>(offsets.extent(0)) - 1;
        auto out = w.make(ncenters, mul * attr_cols);
        if (weights.cols != mul*irreps+skip) throw std::logic_error("Edge environment MLP width mismatch");
        if (w.jit) {
#ifdef SP2_ENABLE_CUDA
            if constexpr (std::is_same_v<T, float>) {
                const int groups_env = (out.cols - 1) / 32 + 1;
                w.jit->launch_grid(w.exec, env_fwd_sym, ncenters * groups_env, 128, 0,
                                   out.size(), attrs.ptr, weights.ptr, cutoff.ptr,
                                   offsets.data(), species.data(), normalization.data(),
                                   layer, skip, out.ptr);
                return out;
            }
#endif
            w.jit->launch(w.exec, env_fwd_sym, out.size(), 128,
                          attrs.ptr, weights.ptr, cutoff.ptr,
                          offsets.data(), species.data(), normalization.data(),
                          layer, skip, out.ptr);
            return out;
        }
        const auto wi = windex;
        const auto norm = normalization;
        const int dim = attr_cols, groups = irreps, nt = n_species;
        for_each(w.exec, "accumulate weighted environment", out.size(), KOKKOS_LAMBDA(int i) {
            const int atom = i / out.cols, channel = i % out.cols / dim, component = i % dim;
            T sum = 0;
            const int col = skip + channel*groups + wi(component);
            for (int e = offsets(atom); e < offsets(atom+1); ++e)
                if (cutoff(e,0) > T(0)) sum += attrs(e,component)*weights(e,col);
            out.ptr[i] = sum * norm(nt ? layer*nt + species(atom) : layer);
        });
        return out;
    }

#ifdef SP2_ENABLE_CUDA
    // One warp consumes both adjoints of an edge while its weights,
    // normalization and cutoff are live. D and G are harmonic and irrep widths.
    template<int D, int G>
    void reverse_environment_warp(Workspace<T>& w, int layer, Tensor<T> d_env, Tensor<T> d_features,
            const typename BackboneTape<T>::Layer& cache, const BackboneTape<T>& tape,
            Tensor<T> d_attrs, Tensor<T> d_weights) const {
        const auto cutoff = cache.latent.cutoff, weights = cache.weights, attrs = tape.attrs;
        const auto is_initial = cache.initial;
        const auto wi = windex, members = channels_by_irrep, starts = offsets_by_irrep;
        const auto center = tape.center, species = tape.species;
        const auto norm = normalization;
        const int nt = n_species, skip = is_initial ? 16 * G : 0;
        for_each(w.exec, "reverse weighted environment", weights.rows * 32, KOKKOS_LAMBDA(int index) {
            const int e = index / 32, lane = index % 32;
            if (cutoff(e,0) <= T(0)) {
                for (int c = lane; c < d_weights.cols; c += 32) d_weights(e,c) = T(0);
                if (layer != 0)
                    for (int c = lane; c < d_features.cols; c += 32) d_features(e,c) = T(0);
                return;
            }
            const int a = center(e);
            const T scale = norm(nt ? layer*nt + species(a) : layer);
            if (lane < D) {
                const int group = wi(lane);
                T sum = 0;
                for (int u = 0; u < 16; ++u) {
                    sum += (d_env(a,u*D+lane)*scale)*weights(e,skip+u*G+group);
                    if (is_initial) sum += d_features(e,u*D+lane)*weights(e,u*G+group);
                }
                if (layer == 0 && !is_initial) sum += d_features(e,lane);
                d_attrs(e,lane) += sum;
            }
            for (int c = lane; c < 16*G; c += 32) {
                const int group = c % G, offset = (c / G) * D;
                T initial_sum = 0, env_sum = 0;
                for (int k = starts(group); k < starts(group+1); ++k) {
                    const int p = members(k);
                    env_sum += (d_env(a,offset+p)*scale)*attrs(e,p);
                    if (is_initial) initial_sum += d_features(e,offset+p)*attrs(e,p);
                }
                if (is_initial) d_weights(e,c) = initial_sum;
                d_weights(e,skip+c) = env_sum;
            }
        });
    }
#endif

    Tensor<T> reverse_environment(Workspace<T>& w, int layer, Tensor<T> d_env, Tensor<T> d_features,
                const typename BackboneTape<T>::Layer& cache, const BackboneTape<T>& tape, Tensor<T> d_attrs) const {
        const auto cutoff = cache.latent.cutoff, weights = cache.weights, attrs = tape.attrs;
        const auto is_initial = cache.initial;
        const auto wi = windex, members = channels_by_irrep, starts = offsets_by_irrep;
        const auto center = tape.center, species = tape.species;
        const auto norm = normalization;
        const int dim = attr_cols, groups = irreps, channels = mul, nt = n_species;
        const int skip = is_initial ? weight_per_edge : 0;
        auto d_weights = w.make(weights.rows, weights.cols);
        if (w.jit) {
            w.jit->launch_grid(w.exec, rev_env_sym, jit_edge_warp_grid(weights.rows), 128, 0,
                               weights.rows, d_env.ptr, d_features.ptr, cutoff.ptr,
                               weights.ptr, attrs.ptr, center.data(), species.data(),
                               norm.data(), layer, int(is_initial), d_features.cols,
                               d_attrs.ptr, d_weights.ptr);
            return d_weights;
        }
#ifdef SP2_ENABLE_CUDA
        if (channels == 16 && std::is_same_v<T,float> &&
            ((dim == 9 && groups == 3) || (dim == 12 && groups == 6) || (dim == 18 && groups == 6))) {
            if (dim == 9)
                reverse_environment_warp<9,3>(w, layer, d_env, d_features, cache, tape, d_attrs, d_weights);
            else if (dim == 12)
                reverse_environment_warp<12,6>(w, layer, d_env, d_features, cache, tape, d_attrs, d_weights);
            else
                reverse_environment_warp<18,6>(w, layer, d_env, d_features, cache, tape, d_attrs, d_weights);
            return d_weights;
        }
#endif
        for_each(w.exec, "weighted environment edge adjoint", d_attrs.size(), KOKKOS_LAMBDA(int index) {
            const int e = index / dim, component = index % dim;
            if (cutoff(e,0) <= T(0)) return;
            const int a = center(e), group = wi(component);
            const T scale = norm(nt ? layer*nt + species(a) : layer);
            T sum = 0;
            for (int u = 0; u < channels; ++u) {
                const int c = u*groups+group, at = u*dim + component;
                sum += (d_env(a,at)*scale)*weights(e,skip+c);
                if (is_initial) sum += d_features(e,at)*weights(e,c);
            }
            if (layer == 0 && !is_initial) sum += d_features(e,component);
            d_attrs.ptr[index] += sum;
        });
        for_each(w.exec, "weighted environment MLP adjoint", d_weights.size(), KOKKOS_LAMBDA(int index) {
            const int e = index / d_weights.cols, c = index % d_weights.cols;
            T sum = 0;
            if (cutoff(e,0) > T(0)) {
                const bool initial = c < skip;
                const int local = initial ? c : c-skip;
                const int offset = (local / groups) * dim, group = local % groups;
                const int a = center(e);
                const T scale = norm(nt ? layer*nt + species(a) : layer);
                for (int k = starts(group); k < starts(group+1); ++k) {
                    const int component = members(k);
                    const T g = initial ? d_features(e,offset+component) : d_env(a,offset+component)*scale;
                    sum += g*attrs(e,component);
                }
            }
            d_weights.ptr[index] = sum;
        });
        // The depth mask applies to the adjoint passed to the previous layer.
        for_each(w.exec, "mask tensor-product adjoint outside cutoff", d_features.size(),
                 KOKKOS_LAMBDA(int i) {
            if (cutoff(i / d_features.cols,0) <= T(0)) d_features.ptr[i] = T(0);
        });
        return d_weights;
    }

private:
    std::vector<Layer> layers;
    MLP<T> final_mlp;
    std::map<std::string, TensorHead<T>> heads;
    std::string latent_field;
    std::vector<T> cutoffs, residual;
    T cos_ratio;
    int poly_p, in_pad;
    bool embed, has_residual, nonzero, cosine;
    int mul = 0, irreps = 0, attr_cols = 0, n_species = 0, weight_per_edge = 0;
    DeviceArray<int> windex, channels_by_irrep, offsets_by_irrep;
    DeviceArray<T> normalization;
    std::string epilogue, epilogue_reverse;
    std::string weight_initial_sym, env_fwd_sym, rev_env_sym;
};
}  // namespace sp2
#endif
