#pragma once

#include "backbone.hpp"
#include "contract.hpp"
#include "geometry.hpp"
#include "evidence.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <memory>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace sp2 {

template<class T> using Fields = std::map<std::string, Tensor<T>>;

template<class T> struct TileInput {
    Tensor<T> r, spins_i, spins_j, center_spins, atomic_seed;
    Tensor<T> pos_i, pos_j, shifts, cells;
    DeviceArray<int> center, offsets, species, species_j;
#ifdef SP2_ENABLE_CUDA
    DeviceArray<int> targets, frames;
    const T *positions{}, *graph_cells{}, *graph_spins{};
    int start{};
    bool shared_cell{};
#endif
    int ncenters{};
    T cutoff{};
    bool spline{};
    bool frozen{};
};

template<class T> using TilePullback = std::function<std::vector<Tensor<T>>(
    Workspace<T>&, const std::vector<Tensor<T>>&)>;

template<class T> struct TileStep {
    std::vector<std::string> inputs, outputs;
    TilePullback<T> reverse;
};

template<class T> struct TileState {
    Fields<T> fields;
    std::vector<TileStep<T>> steps;
};

template<class T> struct TileResult {
    Tensor<T> dr, ds_i, ds_j;
    Fields<T> atom;
    Tensor<T> r, length, dr0, dl;
};

template<class T> KOKKOS_INLINE_FUNCTION Vec3<T> read3(Tensor<T> v, int i) {
    return {v(i, 0), v(i, 1), v(i, 2)};
}

template<class T> KOKKOS_INLINE_FUNCTION void put3(Tensor<T> v, int i, Vec3<T> x) {
    v(i, 0) = x.x;
    v(i, 1) = x.y;
    v(i, 2) = x.z;
}

template<class T, class F> Tensor<T> matrix_kernel(
    Workspace<T>& w, const char* label, int rows, int cols, F f) {
    auto result = w.make(rows, cols);
    if (cols != 0) for_each(w.exec, label, rows * cols, KOKKOS_LAMBDA(int q) {
        const int row = q / cols;
        result(row, q % cols) = f(row, q % cols);
    });
    return result;
}

template<class T> Tensor<T> select_columns(
    Workspace<T>& w, Tensor<T> a, int offset, int width) {
    return matrix_kernel<T>(w, "sp2::select_columns", a.rows, width,
                           KOKKOS_LAMBDA(int row, int col) { return a(row, col + offset); });
}

template<class T> Tensor<T> cat_columns(Workspace<T>& w, Tensor<T> a, Tensor<T> b) {
    if (!a) return b;
    if (a.rows != b.rows) throw std::invalid_argument("Cannot concatenate different row counts");
    return matrix_kernel<T>(w, "sp2::cat_columns", a.rows, a.cols + b.cols,
                           KOKKOS_LAMBDA(int row, int col) {
        return col < a.cols ? a(row, col) : b(row, col - a.cols);
    });
}

inline int tile_dim(std::size_t size) {
    if (size > std::size_t(INT_MAX)) throw std::invalid_argument("Tile dimension exceeds INT_MAX");
    return int(size);
}

template<class T> class Basis {
public:
    Basis(const Contract& contract, const Json& j, const Exec& exec) {
        auto weight_host = contract.array<T>(j.at("weights"));
        params_.width = tile_dim(weight_host.size());
        if (!params_.width) throw std::invalid_argument("Empty basis weight array");
        weights_ = upload(exec, weight_host);
        params_.weights = weights_.data();
        params_.prefactor = j.at("prefactor").get<T>();
        params_.p = j.at("cutoff").at("p").get<int>();
        params_.cutoff_factor = j.at("cutoff").at("factor").get<T>();
        if (params_.p < 2 || params_.p > 65535 || !(params_.cutoff_factor > 0))
            throw std::invalid_argument("Invalid polynomial envelope");
        if (j.at("kind") == "bessel") {
            params_.kind = BasisDevice<T>::bessel;
            params_.radius = j.at("radius").get<T>();
            if (!(params_.radius > 0)) throw std::invalid_argument("Invalid Bessel radius");
            if (j.contains("mean")) {
                auto mean = contract.array<T>(j.at("mean"));
                auto std = contract.array<T>(j.at("inv_std"));
                params_.mean_width = tile_dim(mean.size());
                params_.std_width = tile_dim(std.size());
                if ((params_.mean_width != 1 && params_.mean_width != params_.width) ||
                    (params_.std_width != 1 && params_.std_width != params_.width))
                    throw std::invalid_argument("Invalid Bessel normalization arrays");
                mean_ = upload(exec, mean);
                inv_std_ = upload(exec, std);
                params_.mean = mean_.data();
                params_.inv_std = inv_std_.data();
            }
        } else if (j.at("kind") == "fourier") {
            params_.kind = BasisDevice<T>::fourier;
            params_.minimum = j.at("minimum").get<T>();
            params_.maximum = j.at("maximum").get<T>();
            if (!(params_.minimum < params_.maximum)) throw std::invalid_argument("Invalid Fourier interval");
        } else throw std::invalid_argument("Unsupported basis kind");
        const auto& spline = j.at("spline");
        const auto shape = contract.shape(spline.at("values"));
        if (shape.size() != 2 || shape[0] < 2 || shape[1] != weight_host.size() ||
            contract.shape(spline.at("derivatives")) != shape)
            throw std::invalid_argument("Invalid Hermite table shape");
        params_.knots = tile_dim(shape[0]);
        params_.lower = spline.at("lower").get<double>();
        params_.upper = spline.at("upper").get<double>();
        params_.step = spline.at("step").get<double>();
        if (!(params_.step > 0) || !(params_.upper > params_.lower))
            throw std::invalid_argument("Invalid Hermite table domain");
        const auto val_d = contract.array<double>(spline.at("values"));
        const auto der_d = contract.array<double>(spline.at("derivatives"));
        values_ = upload(exec, val_d);
        derivatives_ = upload(exec, der_d);
        params_.values = values_.data();
        params_.derivatives = derivatives_.data();
        params_.lower_f = static_cast<float>(params_.lower);
        params_.upper_f = static_cast<float>(params_.upper);
        params_.step_f = static_cast<float>(params_.step);
        params_.inv_step_f = static_cast<float>(1.0 / params_.step);
        if constexpr (std::is_same_v<T, float>) {
            std::vector<float> vf(val_d.size()), df(der_d.size());
            for (std::size_t i = 0; i < val_d.size(); ++i) {
                vf[i] = static_cast<float>(val_d[i]);
                df[i] = static_cast<float>(der_d[i]);
            }
            values_f_ = upload(exec, vf);
            derivatives_f_ = upload(exec, df);
            params_.values_f = values_f_.data();
            params_.derivatives_f = derivatives_f_.data();
        }
    }

    BasisDevice<T> device() const { return params_; }

private:
    BasisDevice<T> params_;
    DeviceArray<T> weights_, mean_, inv_std_;
    DeviceArray<double> values_, derivatives_;
    DeviceArray<float> values_f_, derivatives_f_;
};

template<class T> DeviceArray<HarmonicComponent<T>> load_harmonics(
    const Exec& exec, const Json& spec) {
    std::vector<HarmonicComponent<T>> input;
    for (const auto& block : spec.at("blocks")) {
        const auto source = block.at("source").get<std::string>();
        if (source != "r" && source != "i" && source != "j")
            throw std::invalid_argument("Invalid spherical harmonic source");
        const int source_id = source == "r" ? 0 : (source == "i" ? 1 : 2);
        const auto norm = block.at("normalization").get<std::string>();
        if (norm != "component" && norm != "integral" && norm != "norm")
            throw std::invalid_argument("Invalid spherical harmonic normalization");
        const auto degrees = block.at("degrees").get<std::vector<int>>();
        if (degrees.empty()) throw std::invalid_argument("No degrees in spherical harmonic block");
        const auto maximum = *std::max_element(degrees.begin(), degrees.end());
        const bool low = maximum <= 2;
        const bool normalize = block.at("normalize").get<bool>();
        for (int l : degrees) {
            if (l < 0 || l > 11) throw std::invalid_argument("Spherical harmonic degrees must be in [0, 11]");
            for (int m = -l; m <= l; ++m) {
                const int a = std::abs(m);
                double scale = 1.0;
                if (low) {
                    if (norm == "integral") scale = 1.0 / std::sqrt(4.0 * 3.14159265358979323846);
                    if (norm == "norm") scale = 1.0 / std::sqrt(2 * l + 1);
                } else {
                    for (int n = l - a + 1; n <= l + a; ++n) scale /= n;
                    if (a != 0) scale *= 2;
                    if (norm != "norm") scale *= 2 * l + 1;
                    if (norm == "integral") scale /= 4.0 * 3.14159265358979323846;
                    scale = std::sqrt(scale);
                }
                input.push_back({source_id, l, m, normalize, low, T(scale)});
            }
        }
    }
    const auto order = spec.at("sort").get<std::vector<int>>();
    if (order.empty() || order.size() != input.size())
        throw std::invalid_argument("Invalid spherical harmonic permutation");
    std::vector<HarmonicComponent<T>> sorted;
    std::set<int> seen;
    for (int i : order) {
        if (i < 0 || i >= tile_dim(input.size()) || !seen.insert(i).second)
            throw std::invalid_argument("Invalid spherical harmonic component index");
        sorted.push_back(input.at(i));
    }
    return upload(exec, sorted);
}

template<class T> void record(
    TileState<T>& state, std::vector<std::string> inputs, std::vector<std::string> outputs,
    const std::vector<Tensor<T>>& values, TilePullback<T> pullback) {
    for (std::size_t i = 0; i < outputs.size(); ++i) state.fields[outputs[i]] = values[i];
    state.steps.push_back({std::move(inputs), std::move(outputs), std::move(pullback)});
}

template<class T, class Pullback> void record_one(
    TileState<T>& state, std::vector<std::string> inputs, std::string output,
    Tensor<T> value, Pullback pullback) {
    state.fields[output] = value;
    state.steps.push_back({std::move(inputs), {std::move(output)},
        [pullback = std::move(pullback)](Workspace<T>& w, const auto& gradients) {
            return pullback(w, gradients[0]);
        }});
}

template<class T> Fields<T> tile_reverse(
    Workspace<T>& w, const TileState<T>& state, Tensor<T> atomic_seed) {
    Fields<T> grads{{"atomic_energy", atomic_seed}};
    for (auto step = state.steps.rbegin(); step != state.steps.rend(); ++step) {
        std::vector<Tensor<T>> dy;
        bool has_grad = false;
        for (const auto& field : step->outputs) {
            auto g = grads.find(field);
            if (g == grads.end()) dy.emplace_back();
            else {
                dy.push_back(g->second);
                grads.erase(g);
                has_grad = true;
            }
        }
        if (!has_grad) continue;
        const auto deltas = step->reverse(w, dy);
        for (std::size_t i = 0; i < deltas.size(); ++i) {
            if (!deltas[i]) continue;
            // Pullbacks for additive branches can alias their incoming seed. Always
            // issue a fresh sum when two paths to the same input are encountered.
            auto [g, inserted] = grads.emplace(step->inputs[i], deltas[i]);
            if (!inserted) {
                g->second = add_tensors(w, g->second, deltas[i]);
            }
        }
    }
    return grads;
}


template<class T> Tensor<T> project(
    Workspace<T>& w, Tensor<T> seed, Tensor<T> jacobian,
    const DeviceArray<HarmonicComponent<T>>& components, int source) {
    return matrix_kernel<T>(w, "sp2::geometry_adjoint", seed.rows, 3,
        KOKKOS_LAMBDA(int row, int k) {
            T sum{};
            for (int j = 0; j < seed.cols; ++j)
                if (components(j).source == source)
                    sum += seed(row, j) * jacobian(row, j * 3 + k);
            return sum;
        });
}

template<class T> void harmonic_layer(
    Workspace<T>& w, const TileInput<T>& input, TileState<T>& state,
    const std::string& out, const DeviceArray<HarmonicComponent<T>>& spec) {
    ProfileScope<T> profile(w, Profile::geometry);
    const int width = tile_dim(spec.extent(0));
    const auto r = input.r, i = input.spins_i, j = input.spins_j;
    const auto value = w.make(r.rows, width), jacobian = w.make(r.rows, width * 3);
    for_each(w.exec, "sp2::spherical_harmonics", value.size(), KOKKOS_LAMBDA(int q) {
        const int e = q / width, col = q % width;
        const auto c = spec(col);
        const auto vector = c.source == 0 ? read3(r, e)
            : (c.source == 1 ? read3(i, e) : read3(j, e));
        const auto result = harmonic(vector, c);
        value(e, col) = result.value;
        for (int k = 0; k < 3; ++k)
            jacobian(e, col * 3 + k) = result.derivative[k];
    });
    const bool magnetic = bool(i);
    record_one(state, {"r", "i", "j"}, out, value,
        [jacobian, magnetic, spec](Workspace<T>& work, Tensor<T> dy) {
            return std::vector<Tensor<T>>{project(work, dy, jacobian, spec, 0),
                magnetic ? project(work, dy, jacobian, spec, 1) : Tensor<T>{},
                magnetic ? project(work, dy, jacobian, spec, 2) : Tensor<T>{}};
        });
}

template<class T> void radial_layer(
    Workspace<T>& w, const TileInput<T>& input, TileState<T>& state,
    const std::string& out, BasisDevice<T> basis) {
    ProfileScope<T> profile(w, Profile::geometry);
    const auto length = state.fields.at("length");
    const auto v = w.make(length.rows, basis.width);
    const auto d = w.make(length.rows, basis.width);
    const bool spline = input.spline;
    for_each(w.exec, "sp2::radial_basis", v.size(), KOKKOS_LAMBDA(int q) {
        const int e = q / basis.width, c = q % basis.width;
        const auto result = basis.at(length(e, 0), c, spline);
        v(e, c) = result.value;
        d(e, c) = result.slope;
    });
    record_one(state, {"length"}, out, v, [d](Workspace<T>& work, Tensor<T> dy) {
        return std::vector<Tensor<T>>{matrix_kernel<T>(work, "sp2::radial_adjoint",
            dy.rows, 1, KOKKOS_LAMBDA(int e, int) {
                T sum{};
                for (int k = 0; k < dy.cols; ++k) sum += dy(e, k) * d(e, k);
                return sum;
            })};
    });
}

template<class T> void spin_basis_layer(
    Workspace<T>& w, const TileInput<T>& input, TileState<T>& state,
    const std::string& field, const std::string& out, BasisDevice<T> basis) {
    ProfileScope<T> profile(w, Profile::geometry);
    const auto length = state.fields.at("length"), prev = state.fields.at(field);
    const auto i = input.spins_i, j = input.spins_j;
    const bool spline = input.spline;
    const int old_width = prev.cols, wout = old_width + basis.width;
    const auto v = w.make(length.rows, wout);
    const auto local = w.make(length.rows, 5);
    // Retain only basis values and derivatives with respect to the dot product.
    // Recover Cartesian adjoints from their common vector factors on reverse.
    const auto values = w.make(length.rows, basis.width);
    const auto slopes = w.make(length.rows, basis.width);
    for_each(w.exec, "sp2::spin_invariants", length.rows, KOKKOS_LAMBDA(int e) {
        const auto a = read3(i, e), b = read3(j, e);
        const T dot = a.dot(b), aa = a.dot(a), bb = b.dot(b);
        const auto cutoff = polynomial_envelope(length(e, 0), basis.cutoff_factor, basis.p);
        local(e, 0) = aa;
        local(e, 1) = bb;
        local(e, 2) = cutoff.value;
        local(e, 3) = cutoff.slope;
        local(e, 4) = dot;
    });
    for_each(w.exec, "sp2::spin_basis", v.size(), KOKKOS_LAMBDA(int q) {
        const int e = q / wout, feature = q % wout;
        if (feature < old_width) {
            v(e, feature) = prev(e, feature);
            return;
        }
        const int channel = feature - old_width;
        const auto result = basis.at(local(e, 4), channel, spline);
        values(e, channel) = result.value;
        slopes(e, channel) = result.slope;
        v(e, feature) = ((result.value * local(e, 1)) * local(e, 0)) * local(e, 2);
    });
    record_one(state, {field, "length", "i", "j"}, out, v,
        [=](Workspace<T>& work, Tensor<T> dy) {
            const auto trim = select_columns(work, dy, 0, old_width);
            const auto dr = work.make(dy.rows, 1), da = work.make(dy.rows, 3), db = work.make(dy.rows, 3);
            for_each(work.exec, "sp2::spin_basis_adjoint", dy.rows, KOKKOS_LAMBDA(int e) {
                T sum_v{}, sum_s{};
                for (int c = 0; c < basis.width; ++c) {
                    const T seed = dy(e, old_width + c);
                    sum_v += seed * values(e, c);
                    sum_s += seed * slopes(e, c);
                }
                const T aa = local(e, 0), bb = local(e, 1), cv = local(e, 2);
                const T cs = local(e, 3);
                const T dot_derivative = (sum_s * aa * bb) * cv;
                const auto a = read3(i, e), b = read3(j, e);
                dr(e, 0) = sum_v * aa * bb * cs;
                put3(da, e, a * (T(2) * cv * sum_v * bb) + b * dot_derivative);
                put3(db, e, b * (T(2) * cv * sum_v * aa) + a * dot_derivative);
            });
            return std::vector<Tensor<T>>{trim, dr, da, db};
        });
}

template<class T> void one_hot_layer(
    Workspace<T>& w, const TileInput<T>& input, TileState<T>& state, int ntypes, bool mag) {
    const auto center = input.center, species = input.species, neighbors = input.species_j;
    const auto offsets = input.offsets;
    const int ncenters = input.ncenters;
    for (const bool is_center : {true, false}) {
        const auto spins = is_center ? input.spins_i : input.spins_j;
        const auto value = matrix_kernel<T>(w, "sp2::node_one_hot", input.r.rows, ntypes + int(mag),
            KOKKOS_LAMBDA(int e, int c) {
                if (e >= offsets(ncenters)) return T(0);
                if (c < ntypes) return T(c == (is_center ? species(center(e)) : neighbors(e)));
                const auto s = read3(spins, e);
                const auto norm = s.norm();
                const auto u = s / (norm >= T(1e-12) ? norm : T(1e-12));
                return u.dot(u);
            });
        const std::string key = is_center ? "node_i" : "node_j";
        if (!mag) {
            state.fields[key] = value;
            continue;
        }
        record_one(state, {is_center ? "i" : "j"}, key, value,
            [spins, ntypes](Workspace<T>& work, Tensor<T> dy) {
                return std::vector<Tensor<T>>{matrix_kernel<T>(work, "sp2::one_hot_adjoint", dy.rows, 3,
                    KOKKOS_LAMBDA(int e, int c) {
                        const auto s = read3(spins, e);
                        return s.norm() < T(1e-12) ? (T(2) * s[c] / T(1e-24)) * dy(e, ntypes) : T(0);
                    })};
            });
    }
}

template<class T> void scalar_layer(
    Workspace<T>& w, TileState<T>& s, const MLP<T>& mlp,
    const std::string& field, const std::string& out, T coeff, bool collect, bool cat, bool normalize) {
    // Evidence-only heads are never on the energy path. Their pullback may be discarded.
    const bool energy_path = !(out.find("_sigma_s_raw") != std::string::npos ||
                               out.find("_nu") != std::string::npos ||
                               out.find("_kappa") != std::string::npos);
    MlpTape<T> tape;
    const auto raw = mlp.forward(w, s.fields.at(field), tape, energy_path);
    auto value = raw;
    if (coeff != T(1)) value = matrix_kernel<T>(w, "sp2::scalar_scale", raw.rows, raw.cols,
        KOKKOS_LAMBDA(int e, int c) { return coeff * raw(e, c); });
    Tensor<T> norm;
    if (normalize && !collect) {
        const auto prev = value;
        norm = matrix_kernel<T>(w, "sp2::scalar_norm", prev.rows, 1,
            KOKKOS_LAMBDA(int e, int) {
                T r2{};
                for (int c = 0; c < prev.cols; ++c) r2 += prev(e, c) * prev(e, c);
                return Kokkos::sqrt(r2);
            });
        value = matrix_kernel<T>(w, "sp2::scalar_normalize", prev.rows, prev.cols,
            KOKKOS_LAMBDA(int e, int c) {
                return prev(e, c) / (norm(e, 0) >= T(1e-12) ? norm(e, 0) : T(1e-12));
            });
    }
    const auto activated = value;
    std::vector<std::string> inputs{field};
    int leading = 0;
    if (collect || cat) {
        const auto previous = s.fields.at(out);
        inputs.push_back(out);
        if (collect) value = matrix_kernel<T>(w, "sp2::add_energy", previous.rows, previous.cols,
            KOKKOS_LAMBDA(int e, int c) { return activated(e, c) + previous(e, c); });
        else {
            leading = previous.cols;
            value = cat_columns(w, previous, value);
        }
    }
    if (!energy_path) {
        s.fields[out] = value;
        return;
    }
    const auto* network = &mlp;
    record_one(s, std::move(inputs), out, value,
        [=, tape = std::move(tape)](Workspace<T>& work, Tensor<T> dy) {
            const auto dx = matrix_kernel<T>(work, "sp2::scalar_adjoint", dy.rows, raw.cols,
                KOKKOS_LAMBDA(int e, int c) {
                    T g = dy(e, c + leading);
                    if (norm) {
                        const T length = norm(e, 0);
                        if (length >= T(1e-12)) {
                            T projection{};
                            for (int k = 0; k < raw.cols; ++k)
                                projection += dy(e, leading + k) * activated(e, k);
                            g = (g - projection * activated(e, c)) / length;
                        } else g /= T(1e-12);
                    }
                    return coeff * g;
                });
            std::vector<Tensor<T>> grads{network->reverse(work, dx, tape)};
            if (collect) grads.push_back(dy);
            else if (cat) grads.push_back(select_columns(work, dy, 0, leading));
            return grads;
        });
}

enum class SpinTerm { heisenberg, biquadratic, exchange, anisotropy };

template<class T> void spin_energy_layer(
    Workspace<T>& w, const TileInput<T>& input, TileState<T>& s,
    const std::string& field, const std::string& out, SpinTerm term, T coeff, bool collect) {
    const auto i = input.spins_i;
    const bool anisotropy = term == SpinTerm::anisotropy;
    const auto j = anisotropy ? i : input.spins_j;
    const auto coupling = s.fields.at(field);
    const auto old = collect ? s.fields.at(out) : Tensor<T>{};
    const bool tensor = term == SpinTerm::exchange || anisotropy;
    const bool biquad = term == SpinTerm::biquadratic;
    if (coupling.rows != i.rows || coupling.cols != (tensor ? 9 : 1))
        throw std::invalid_argument("Invalid spin coupling shape");
    const auto v = matrix_kernel<T>(w, "sp2::spin_energy", i.rows, 1,
        KOKKOS_LAMBDA(int e, int) {
            const auto a = read3(i, e), b = read3(j, e);
            T value{};
            if (tensor) {
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l) value += a[k] * (coupling(e, 3 * k + l) * coeff) * b[l];
            } else {
                const T dot = a.dot(b);
                value = (coupling(e, 0) * coeff) * dot;
                if (biquad) value *= dot;
            }
            return collect ? value + old(e, 0) : value;
        });
    std::vector<std::string> inputs{field, "i", anisotropy ? "i" : "j"};
    if (collect) inputs.push_back(out);
    record_one(s, std::move(inputs), out, v, [=](Workspace<T>& work, Tensor<T> dy) {
        const auto dc = work.make(dy.rows, coupling.cols), da = work.make(dy.rows, 3), db = work.make(dy.rows, 3);
        for_each(work.exec, "sp2::spin_energy_adjoint", dy.rows, KOKKOS_LAMBDA(int e) {
            const auto a = read3(i, e), b = read3(j, e);
            const T seed = dy(e, 0) * coeff;
            if (tensor) {
                for (int k = 0; k < 3; ++k) {
                    T ga{}, gb{};
                    for (int l = 0; l < 3; ++l) {
                        dc(e, 3*k + l) = seed * a[k] * b[l];
                        ga += coupling(e, 3*k + l) * b[l];
                        gb += a[l] * coupling(e, 3*l + k);
                    }
                    da(e, k) = ga * seed;
                    db(e, k) = gb * seed;
                }
            } else {
                const T dot = a.dot(b);
                dc(e, 0) = seed * (biquad ? dot * dot : dot);
                const T weight = (seed * coupling(e, 0)) * (biquad ? T(2) * dot : T(1));
                put3(da, e, b * weight);
                put3(db, e, a * weight);
            }
        });
        std::vector<Tensor<T>> result{dc, da, db};
        if (collect) result.push_back(dy);
        return result;
    });
}

template<class T> Tensor<T> reduce_edge_values(
    Workspace<T>& w, const TileInput<T>& input, Tensor<T> value, Tensor<T> length,
    T factor, bool mean) {
    const auto offsets = input.offsets;
    const auto cutoff = input.cutoff;
    return matrix_kernel<T>(w, "sp2::sum_center_edges", input.ncenters, value.cols,
        KOKKOS_LAMBDA(int i, int c) {
            T sum{};
            const int begin = offsets(i), end = offsets(i + 1);
            int count = 0;
            for (int e = begin; e < end; ++e) {
                if (length && length(e, 0) >= cutoff) continue;
                sum += value(e, c);
                ++count;
            }
            return (sum * factor) / (mean && count ? T(count) : T(1));
        });
}

template<class T> void edge_reduce(
    Workspace<T>& w, const TileInput<T>& input, TileState<T>& s,
    const std::string& field, const std::string& out, T factor, bool mean, bool reverse) {
    const auto x = s.fields.at(field);
    const auto cutoff = input.cutoff;
    const auto length = cutoff > T(0) ? s.fields.at("length") : Tensor<T>{};
    auto v = reduce_edge_values(w, input, x, length, factor, mean);
    if (!reverse) {
        s.fields[out] = v;
        return;
    }
    const auto center = input.center, offsets = input.offsets;
    const int ncenters = input.ncenters;
    record_one(s, {field}, out, v, [=](Workspace<T>& work, Tensor<T> dy) {
        return std::vector<Tensor<T>>{matrix_kernel<T>(work, "sp2::broadcast_edge_seed", x.rows, x.cols,
            KOKKOS_LAMBDA(int e, int c) {
                if (e >= offsets(ncenters)) return T(0);
                if (length && length(e, 0) >= cutoff) return T(0);
                const int i = center(e);
                int count = 1;
                if (mean) {
                    const int begin = offsets(i), end = offsets(i + 1);
                    count = end - begin;
                    if (length)
                        for (int k = begin; k < end; ++k)
                            count -= length(k, 0) >= cutoff;
                }
                return (dy(i, c) * factor) / T(count);
            })};
    });
}

template<class T> Tensor<T> species_affine(
    Workspace<T>& w, const TileInput<T>& input, Tensor<T> x,
    DeviceArray<T> scales, DeviceArray<T> shifts) {
    const auto types = input.species;
    return matrix_kernel<T>(w, "sp2::species_affine", x.rows, x.cols,
        KOKKOS_LAMBDA(int i, int c) {
            const int t = types(i);
            T value = x(i, c);
            if (scales.data()) value *= scales(t);
            if (shifts.data()) value += shifts(t);
            return value;
        });
}

template<class T> void species_energy_layer(
    Workspace<T>& w, const TileInput<T>& input, TileState<T>& state,
    DeviceArray<T> scales, DeviceArray<T> shifts) {
    const auto value = species_affine(w, input, state.fields.at("atomic_energy"), scales, shifts);
    const auto species = input.species;
    record_one(state, {"atomic_energy"}, "atomic_energy", value,
        [scales, species](Workspace<T>& work, Tensor<T> dy) {
            if (!scales.data()) return std::vector<Tensor<T>>{dy};
            return std::vector<Tensor<T>>{matrix_kernel<T>(work, "sp2::species_scale_adjoint",
                dy.rows, dy.cols, KOKKOS_LAMBDA(int i, int c) {
                    return dy(i, c) * scales(species(i));
                })};
        });
}

template<class T> void edge_energy_layer(
    Workspace<T>& w, const TileInput<T>& input, TileState<T>& s,
    T factor, DeviceArray<T> scales, int ntypes, T cutoff_factor, int p) {
    const auto edges = s.fields.at("edge_energy");
    if (edges.rows != input.r.rows || edges.cols != 1)
        throw std::invalid_argument("Invalid edge energy shape");
    if (scales.data() || p) {
        const auto length = s.fields.at("length");
        const auto center = input.center;
        const auto species = input.species, neighbors = input.species_j;
        const auto v = w.make(edges.rows, 1), derivative = w.make(edges.rows, 2);
        for_each(w.exec, "sp2::edge_energy_scale", edges.rows, KOKKOS_LAMBDA(int e) {
            const T pair = scales.data() ? scales(species(center(e)) * ntypes + neighbors(e)) : T(1);
            const auto cutoff = p ? polynomial_envelope(length(e, 0), cutoff_factor, p) : ValueSlope<T>{T(1), T(0)};
            v(e, 0) = (edges(e, 0) * pair) * cutoff.value;
            derivative(e, 0) = pair * cutoff.value;
            derivative(e, 1) = pair * cutoff.slope * edges(e, 0);
        });
        record_one(s, {"edge_energy", "length"}, "edge_energy", v,
            [derivative](Workspace<T>& work, Tensor<T> dy) {
                const auto de = matrix_kernel<T>(work, "sp2::edge_energy_seed", dy.rows, 1,
                    KOKKOS_LAMBDA(int e, int) { return derivative(e, 0) * dy(e, 0); });
                const auto dr = matrix_kernel<T>(work, "sp2::edge_energy_radial_adjoint", dy.rows, 1,
                    KOKKOS_LAMBDA(int e, int) { return derivative(e, 1) * dy(e, 0); });
                return std::vector<Tensor<T>>{de, dr};
            });
    }
    edge_reduce(w, input, s, "edge_energy", "atomic_energy", factor, false, true);
}


}  // namespace sp2

#include "jit_geometry.hpp"

namespace sp2 {

template<class T> class TileProgram {
    using RunLayer = std::function<void(Workspace<T>&, const TileInput<T>&, TileState<T>&)>;

    // Only a first read before a Stage-2 write can consume a lattice field.
    struct SpinFields {
        std::set<std::string> inputs{"r"}, outputs{"i", "j"};

        void read(const std::string& field) {
            if (!field.empty() && !outputs.count(field)) inputs.insert(field);
        }
        void write(const std::string& field) {
            if (!field.empty()) outputs.insert(field);
        }
    };

public:
    TileProgram(const Contract& contract, const Json& model, const Exec& exec) {
        bool spin = false;
        SpinFields fields;
        std::set<std::string> evidence_inputs;
        const auto& layers = model.at("layers");
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& j = layers[i];
            const auto op = j.at("op").get<std::string>();
            const auto name = j.at("name").get<std::string>();
            if (op == "AtomwiseReduce") continue;
            if (op == "e2IP_nodewise") {
                for (const auto& prefix : j.at("prefixes")) {
                    evidence_.push_back(evidence_head(contract, exec, j, prefix.get<std::string>()));
                    const auto name = prefix.get<std::string>();
                    for (auto suffix : {"_sigma_s_raw", "_sigma_v_raw", "_nu", "_kappa"}) {
                        evidence_inputs.insert(name + suffix);
                        if (name == "f") evidence_inputs.insert(std::string("socf") + suffix);
                    }
                }
                continue;
            }
            if (op == "OneHotAtomEncoding_spin" || op == "HB_Energy" || op == "Biquad_Energy"
                || op == "Spin_JMatrix_Energy" || op == "Spin_ani_Energy") spin = true;
            const bool lattice_coeff = name == "edge_hb_coeff" || name == "edge_biquad_coeff";
            const bool spin_stage = spin && !lattice_coeff;
            auto& stage = spin_stage ? spin_ : lattice_;
            auto* spin_fields = spin_stage ? &fields : nullptr;
            if (op == "OneHotAtomEncoding" || op == "OneHotAtomEncoding_spin")
                ntypes_ = j.at("num_types").get<int>();
            if (jit_basis_op(op)) {
                const bool paired = i + 1 < layers.size() && jit_harmonic_op(layers[i + 1].at("op"));
                stage.push_back(load_geometry(contract, exec, j, paired ? &layers[i + 1] : nullptr,
                                              spin_fields));
                if (paired) ++i;
                if (op == "RadialBasisEdgeEncoding" && !spin) fused_displacement_ = true;
            } else {
                stage.push_back(load_layer(contract, exec, j, op, spin_fields));
            }
            if (op == "CartesianBECReduce") bec_ = j.at("out_field").get<std::string>();
        }
        fields.read("atomic_energy");
        fields.read(bec_);
        for (const auto& field : evidence_inputs) fields.read(field);
        forward_fields_ = std::move(fields.inputs);
    }

    TileState<T> stage1(Workspace<T>& w, const TileInput<T>& input) const {
        TileState<T> state;
        const auto r = input.r;
        state.fields["r"] = r;
        {
            ProfileScope<T> profile(w, Profile::geometry);
            if (w.jit) {
                const auto length = w.make(r.rows, 1);
                state.fields["length"] = length;
                if (!fused_displacement_) jit_displacement(w, input, length);
            } else {
                state.fields["length"] = matrix_kernel<T>(w, "sp2::edge_length", r.rows, 1,
                    KOKKOS_LAMBDA(int e, int) { return read3(r, e).norm(); });
            }
        }
        for (const auto& layer : lattice_) layer(w, input, state);
        return state;
    }

    TileState<T> cache_forward(Workspace<T>& storage, const TileState<T>& lattice) const {
        TileState<T> saved;
        for (const auto& name : forward_fields_) {
            const auto from = find(lattice.fields, name);
            if (!from) continue;
            const auto to = storage.make(from.rows, from.cols);
            if (from.size()) {
#ifdef SP2_ENABLE_CUDA
                cuda_check(cudaMemcpyAsync(to.ptr, from.ptr, sizeof(T) * from.size(),
                                           cudaMemcpyDeviceToDevice, storage.exec.cuda_stream()),
                           "Retain stage-1 forward tensor");
#else
                std::copy_n(from.ptr, from.size(), to.ptr);
#endif
            }
            saved.fields.emplace(name, to);
        }
        return saved;
    }

    TileResult<T> stage2(Workspace<T>& w, const TileInput<T>& input, const TileState<T>& lattice) const {
        TileState<T> state;
        state.fields = lattice.fields;
        if (!input.frozen) state.steps = lattice.steps;
        if (input.spins_i) {
            state.fields["i"] = input.spins_i;
            state.fields["j"] = input.spins_j;
        }
        for (const auto& layer : spin_) layer(w, input, state);
        Fields<T> atom{{"atomic_energy", state.fields.at("atomic_energy")}};
        if (!bec_.empty()) atom.emplace(bec_, state.fields.at(bec_));
        for (const auto& head : evidence_) head(w, input, state, atom);

        ProfileScope<T> profile(w, Profile::reverse);
        const auto grads = tile_reverse(w, state, input.atomic_seed);
        if (input.frozen)
            return {{}, find(grads, "i"), find(grads, "j"), std::move(atom)};
        const auto dr0 = find(grads, "r"), dl = find(grads, "length");
        const auto r = state.fields.at("r"), length = state.fields.at("length");
        if (w.jit)
            return {{}, find(grads, "i"), find(grads, "j"), std::move(atom), r, length, dr0, dl};
        const auto dr = matrix_kernel<T>(w, "sp2::merge_geometry_adjoint", r.rows, 3,
            KOKKOS_LAMBDA(int e, int c) {
                T result = dr0.ptr ? dr0(e, c) : T(0);
                const T l = length(e, 0);
                if (dl.ptr && l != T(0)) result += (r(e, c) / l) * dl(e, 0);
                return result;
            });
        return {dr, find(grads, "i"), find(grads, "j"), std::move(atom), r, length, dr0, dl};
    }

private:
    using RunEvidence = std::function<void(
        Workspace<T>&, const TileInput<T>&, const TileState<T>&, Fields<T>&)>;

    int ntypes_{};
    bool fused_displacement_{};
    std::string bec_;
    std::vector<RunLayer> lattice_, spin_;
    std::vector<RunEvidence> evidence_;
    std::set<std::string> forward_fields_;

    static Tensor<T> find(const Fields<T>& f, const std::string& key) {
        const auto entry = f.find(key);
        return entry == f.end() ? Tensor<T>{} : entry->second;
    }

    static T factor(const Json& j) {
        return j.at("factor").is_null() ? T(1) : j.at("factor").get<T>();
    }

    static DeviceArray<T> load_optional(
        const Contract& contract, const Exec& exec, const Json& value, size_t count) {
        if (value.is_null()) return {};
        const auto host = contract.array<T>(value);
        if (host.size() != count) throw std::invalid_argument("Invalid species scale shape");
        return upload(exec, host);
    }

    static RunEvidence evidence_head(
        const Contract& contract, const Exec& exec, const Json& layer, const std::string& prefix) {
        const auto& head = layer.at("heads").at(prefix);
        const auto& gate = head.at("gate");
        const auto& scalar = head.at("scalar");
        const auto& matrix = head.at("matrix_to_irreps");
        const auto& q = layer.at("Q_e2ip");
        const auto shape = contract.shape(gate.at("weight"));
        const auto mshape = contract.shape(matrix);
        if (shape.size() != 2 || mshape.size() != 2 || mshape[0] != 9 || mshape[1] > 9 ||
            mshape[1] == 0 || shape[0] == 0 || shape[0] > shape[1])
            throw std::invalid_argument("Invalid evidential weight shape");
        EvidenceWeights<T> params;
        params.channels = tile_dim(shape[0]);
        params.invariants = tile_dim(shape[1]);
        params.irreps = tile_dim(mshape[1]);
        params.gate = load_optional(contract, exec, gate.at("weight"), shape[0] * shape[1]);
        params.gate_bias = load_optional(contract, exec, gate.at("bias"), shape[0]);
        params.scalar = load_optional(contract, exec, scalar.at("weight"), shape[1]);
        params.scalar_bias = load_optional(contract, exec, scalar.at("bias"), 1);
        params.matrix_to_irreps = load_optional(contract, exec, matrix, 9 * mshape[1]);
        params.q = load_optional(contract, exec, q, 9 * mshape[1]);
        const bool tangent = prefix == "sf";
        params.spin_scale = layer.at("spin_force_scale").get<T>();
        params.soc_coeff = layer.at("soc_f_coeff").get<T>();
        if ((prefix != "f" && !tangent) || !(params.spin_scale > 0))
            throw std::invalid_argument("Invalid evidential output configuration");
        return [prefix, params, tangent](Workspace<T>& w, const TileInput<T>& input,
                                         const TileState<T>& state, Fields<T>& atom) {
            const auto& f = state.fields;
            const std::array<Tensor<T>, 4> src{f.at(prefix + "_sigma_s_raw"),
                f.at(prefix + "_sigma_v_raw"), f.at(prefix + "_nu"), f.at(prefix + "_kappa")};
            std::array<Tensor<T>, 4> soc{};
            if (!tangent && f.count("socf_sigma_s_raw"))
                soc = {f.at("socf_sigma_s_raw"), f.at("socf_sigma_v_raw"),
                       f.at("socf_nu"), f.at("socf_kappa")};
            const auto out = evaluate_evidence(w, params, tangent, input.center_spins, src, soc);
            atom[prefix + "_nu"] = out[0];
            atom[prefix + "_kappa"] = out[1];
            atom[prefix + "_sigma"] = out[2];
            atom[prefix + "_Uepi"] = out[3];
        };
    }

    RunLayer load_geometry(const Contract& contract, const Exec& exec,
                           const Json& basis, const Json* harmonics, SpinFields* spin_fields) {
        const auto& desc = basis.at("basis");
        const bool combined = desc.contains("radial");
        const bool magnetic = desc.at("kind") == "fourier";
        const auto field = basis.contains("field") && !basis["field"].is_null()
            ? basis["field"].get<std::string>() : std::string{};
        const auto out = basis.at("out_field").get<std::string>();
        const auto sh_out = harmonics ? harmonics->at("out_field").get<std::string>() : std::string{};
        if (spin_fields) {
            spin_fields->read("length");
            spin_fields->read(field);
            spin_fields->write(out);
            spin_fields->write(sh_out);
        }
        const auto h = harmonics ? load_harmonics<T>(exec, harmonics->at("harmonics"))
                                 : DeviceArray<HarmonicComponent<T>>{};
        const int sh_width = tile_dim(h.extent(0));
        const std::string symbol = "sp2_geom_" + basis.at("name").get<std::string>();
        const Basis<T> b(contract, combined ? desc.at("radial") : desc, exec);
        const auto spin = combined ? std::optional<Basis<T>>(std::in_place, contract, desc, exec)
                                   : std::optional<Basis<T>>{};
        return [=](auto& w, const auto& t, auto& s) {
            if (w.jit) {
                jit_geometry_layer(w, t, s, symbol, field, out, sh_out, b.device(),
                    spin ? spin->device() : BasisDevice<T>{}, sh_width, magnetic);
            } else {
                if (spin) {
                    radial_layer(w, t, s, "_radial_part", b.device());
                    spin_basis_layer(w, t, s, "_radial_part", out, spin->device());
                } else if (magnetic) {
                    spin_basis_layer(w, t, s, field, out, b.device());
                } else {
                    radial_layer(w, t, s, out, b.device());
                }
                if (sh_width) harmonic_layer(w, t, s, sh_out, h);
            }
        };
    }

    RunLayer load_layer(const Contract& contract, const Exec& exec, const Json& j,
                        const std::string& op, SpinFields* spin_fields) {
        if (op == "OneHotAtomEncoding" || op == "OneHotAtomEncoding_spin") {
            const int n = ntypes_;
            if (n <= 0) throw std::invalid_argument("Invalid number of species");
            const bool mag = op == "OneHotAtomEncoding_spin";
            if (spin_fields) {
                spin_fields->write("node_i");
                spin_fields->write("node_j");
            }
            return [n, mag](auto& w, const auto& t, auto& s) { one_hot_layer(w, t, s, n, mag); };
        }
        const auto field = j.contains("field") && !j["field"].is_null()
            ? j["field"].get<std::string>() : std::string{};
        const auto out = j.value("out_field", std::string{});
        if (spin_fields) spin_fields->read(field);
        if (op == "SphericalHarmonicEdgeAttrs" || op == "SphericalHarmonicEdgeAttrs_spin_nosoc"
            || op == "SphericalHarmonicEdgeAttrs_spin_soc_simple") {
            const auto h = load_harmonics<T>(exec, j.at("harmonics"));
            if (spin_fields) spin_fields->write(out);
            return [h, out](auto& w, const auto& t, auto& s) { harmonic_layer(w, t, s, out, h); };
        }
        if (op == "ScalarMLP") {
            const auto mlp = std::make_shared<MLP<T>>(contract, j.at("mlp"));
            const T coeff = j.at("coeff").get<T>();
            const bool collect = j.at("add_collect").get<bool>();
            const bool cat = j.at("cat_collect").get<bool>();
            const bool normalize = j.at("normalize").get<bool>();
            if (spin_fields) {
                if (collect || cat) spin_fields->read(out);
                spin_fields->write(out);
            }
            return [=](auto& w, const auto&, auto& s) {
                scalar_layer(w, s, *mlp, field, out, coeff, collect, cat, normalize);
            };
        }
        if (op == "Allegro_og_Module" || op == "Allegro_born_Module" || op == "Allegro_spin_Module") {
            const auto net = std::make_shared<Backbone<T>>(contract, j.at("backbone"),
                                                          j.at("name").get<std::string>());
            const auto invariant = j.at("edge_invariant_field").get<std::string>();
            if (spin_fields) {
                for (auto input : {"node_i", "node_j", "length"}) spin_fields->read(input);
                spin_fields->read(invariant);
                const auto& spec = j.at("backbone");
                spin_fields->write(spec.at("latent_out_field").get<std::string>());
                for (const auto& head : spec.at("heads").items()) spin_fields->write(head.key());
            }
            return [net, field, invariant](auto& w, const auto& t, auto& s) {
                BackboneTape<T> tape;
                const auto& f = s.fields;
                const auto values = net->forward(w, f.at(field), f.at(invariant), f.at("node_i"),
                    f.at("node_j"), f.at("length"), t.center, t.offsets, t.species, tape);
                std::vector<std::string> names;
                std::vector<Tensor<T>> tensors;
                for (const auto& [name, value] : values) {
                    names.push_back(name);
                    tensors.push_back(value);
                }
                auto reverse = [net, tape = std::move(tape), names](Workspace<T>& work, const auto& seeds) {
                    Fields<T> grads;
                    for (std::size_t i = 0; i < seeds.size(); ++i)
                        if (seeds[i]) grads.emplace(names[i], seeds[i]);
                    const auto result = net->reverse(work, grads, tape);
                    return std::vector<Tensor<T>>(result.begin(), result.end());
                };
                record(s, {field, invariant, "node_i", "node_j", "length"}, names, tensors,
                       TilePullback<T>(std::move(reverse)));
            };
        }
        if (op == "HB_Energy" || op == "Biquad_Energy" || op == "Spin_JMatrix_Energy" || op == "Spin_ani_Energy") {
            const auto term = op == "HB_Energy" ? SpinTerm::heisenberg
                : op == "Biquad_Energy" ? SpinTerm::biquadratic
                : op == "Spin_JMatrix_Energy" ? SpinTerm::exchange : SpinTerm::anisotropy;
            const bool collect = j.at("add_collect").get<bool>();
            const T coeff = j.at("coeff").get<T>();
            if (spin_fields) {
                if (collect) spin_fields->read(out);
                spin_fields->write(out);
            }
            return [=](auto& w, const auto& t, auto& s) { spin_energy_layer(w,t,s,field,out,term,coeff,collect); };
        }
        if (op == "EdgewiseEnergySum") {
            const int n = ntypes_;
            const auto scales = load_optional(contract, exec, j.at("scales"), size_t(n) * n);
            const auto& cutoff = j.at("cutoff");
            const int p = cutoff.is_null() ? 0 : cutoff[1].get<int>();
            const T f = cutoff.is_null() ? T(0) : cutoff[0].get<T>();
            if (!cutoff.is_null() && (p < 2 || p > 65535 || !(f > 0)))
                throw std::invalid_argument("Invalid edge energy cutoff");
            const T scale = factor(j);
            if (spin_fields) {
                spin_fields->read("edge_energy");
                spin_fields->read("length");
                spin_fields->write("atomic_energy");
            }
            return [=](auto& w, const auto& t, auto& s) { edge_energy_layer(w,t,s,scale,scales,n,f,p); };
        }
        if (op == "EdgewiseReduce" || op == "CartesianBECReduce") {
            const bool bec = op == "CartesianBECReduce";
            const T scale = factor(j);
            const bool mean = j.at("reduce") == "mean";
            const auto scales = bec ? load_optional(contract,exec,j.at("scales"),ntypes_) : DeviceArray<T>{};
            const auto shifts = bec ? load_optional(contract,exec,j.at("shifts"),ntypes_) : DeviceArray<T>{};
            if (spin_fields) {
                spin_fields->read("length");
                spin_fields->write(out);
            }
            return [=](auto& w, const auto& t, auto& s) {
                edge_reduce(w, t, s, field, out, scale, mean, false);
                if (bec) s.fields[out] = species_affine(w,t,s.fields.at(out),scales,shifts);
            };
        }
        if (op == "PerSpeciesScaleShift") {
            const auto scales = load_optional(contract, exec, j.at("scales"), ntypes_);
            const auto shifts = load_optional(contract, exec, j.at("shifts"), ntypes_);
            if (spin_fields) {
                spin_fields->read("atomic_energy");
                spin_fields->write("atomic_energy");
            }
            return [scales, shifts](auto& w, const auto& t, auto& s) {
                species_energy_layer(w, t, s, scales, shifts);
            };
        }
        throw std::invalid_argument("Unsupported tile layer " + op);
    }
};

}  // namespace sp2
