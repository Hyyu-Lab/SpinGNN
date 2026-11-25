#pragma once

#include "geometry.hpp"
#include "tensor.hpp"

#include <array>
#include <stdexcept>
#include <string>

namespace sp2 {

template<class T> struct EvidenceWeights {
    DeviceArray<T> gate, gate_bias, scalar, scalar_bias, matrix_to_irreps, q;
    int channels{}, invariants{}, irreps{};
    T soc_coeff{}, spin_scale{};
};

template<class T> KOKKOS_INLINE_FUNCTION T damp(T x, T transition, T ceiling) {
    const T delta = ceiling - transition;
    return x < transition ? x : delta * Kokkos::tanh((x - transition) / delta) + transition;
}

template<class T> KOKKOS_INLINE_FUNCTION T softplus(T x) {
    return x > T(20) ? x : Kokkos::log1p(Kokkos::exp(x));
}

template<class T> KOKKOS_INLINE_FUNCTION void multiply3(const T* a, const T* b, T* out) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            out[3*i+j] = (a[3*i] * b[j] + a[3*i+1] * b[3+j]) + a[3*i+2] * b[6+j];
    }
}

template<class T> KOKKOS_INLINE_FUNCTION void expm3(const T* matrix, T* out) {
    // The log covariance has bounded spectrum. Scale to infinity norm <= 1/2,
    // evaluate the Taylor polynomial, then undo the scaling by squaring.
    double a[9], product[9], term[9], cumulative[9];
    double norm = 0;
    for (int i = 0; i < 3; ++i) {
        double row = 0;
        for (int j = 0; j < 3; ++j) row += Kokkos::abs(double(matrix[3*i+j]));
        if (row > norm) norm = row;
    }
    int n = 0;
    double scale = 1;
    while (norm > 0.5 && n < 16) { norm *= 0.5; scale *= 0.5; ++n; }
    for (int i = 0; i < 9; ++i) {
        a[i] = scale * double(matrix[i]);
        cumulative[i] = term[i] = (i % 4 == 0 ? 1.0 : 0.0);
    }
    for (int order = 1; order <= 16; ++order) {
        multiply3(term, a, product);
        for (int i = 0; i < 9; ++i) {
            term[i] = product[i] / order;
            cumulative[i] += term[i];
        }
    }
    for (int pass = 0; pass < n; ++pass) {
        multiply3(cumulative, cumulative, product);
        for (int i = 0; i < 9; ++i) cumulative[i] = product[i];
    }
    for (int i = 0; i < 9; ++i) out[i] = T(cumulative[i]);
}

template<class T> std::array<Tensor<T>, 4> evaluate_evidence(
    Workspace<T>& w, const EvidenceWeights<T>& weights, bool tangent, Tensor<T> spin,
    const std::array<Tensor<T>, 4>& inputs, const std::array<Tensor<T>, 4>& soc_inputs) {
    const auto scalar_features = inputs[0], vector_features = inputs[1];
    const auto raw_nu = inputs[2], raw_kappa = inputs[3];
    const auto soc_scalar = soc_inputs[0], soc_vector = soc_inputs[1];
    const auto soc_nu = soc_inputs[2], soc_kappa = soc_inputs[3];
    const bool soc = bool(soc_scalar);
    const auto ncenters = raw_nu.rows;
    const int ns = scalar_features.cols;
    const int channels = weights.channels, invariants = weights.invariants, irreps = weights.irreps;
    if (ns + channels != invariants || vector_features.cols != 3 * channels ||
        raw_nu.cols != 1 || raw_kappa.cols != 1)
        throw std::invalid_argument("Invalid evidential feature widths");
    for (const auto tensor : inputs) {
        if (tensor.rows != ncenters) throw std::invalid_argument("Invalid evidential feature rows");
    }
    if (soc) {
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (soc_inputs[i].rows != inputs[i].rows || soc_inputs[i].cols != inputs[i].cols)
                throw std::invalid_argument("Incompatible SOC evidential features");
        }
    }
    if (tangent && (spin.rows != ncenters || spin.cols != 3))
        throw std::invalid_argument("Spin covariance requires one Cartesian spin per center");
    const T coeff = weights.soc_coeff;
    const T spin_scale_sq = weights.spin_scale * weights.spin_scale;
    const auto nu = w.make(ncenters, 1), kappa = w.make(ncenters, 1);
    const auto sigma = w.make(ncenters, 9), uepi = w.make(ncenters, 9);
    const auto features = w.make(ncenters, invariants), vectors = w.make(ncenters, 3 * channels);
    for_each(w.exec, "sp2::evidence_features", features.size(), KOKKOS_LAMBDA(int index) {
        const int c = index / invariants, k = index % invariants;
        if (k < ns) {
            features(c, k) = scalar_features(c, k) + (soc ? coeff * soc_scalar(c, k) : T(0));
        } else {
            T norm_sq = 0;
            const int begin = 3 * (k - ns);
            for (int d = 0; d < 3; ++d) {
                const T v = vector_features(c, begin+d) +
                    (soc ? coeff * soc_vector(c, begin+d) : T(0));
                vectors(c, begin+d) = v;
                norm_sq += v * v;
            }
            features(c, k) = norm_sq;
        }
    });
    const auto gate_weight = weights.gate, gate_bias = weights.gate_bias;
    const auto scalar_weight = weights.scalar, scalar_bias = weights.scalar_bias;
    const auto matrix_to_irreps = weights.matrix_to_irreps, q = weights.q;
    const auto gates = w.make(ncenters, channels);
    for_each(w.exec, "sp2::evidence_gate", gates.size(), KOKKOS_LAMBDA(int index) {
        const int c = index / channels, k = index % channels;
        T g{};
        for (int f = 0; f < invariants; ++f) g += gate_weight(k*invariants+f) * features(c, f);
        gates(c, k) = (g + gate_bias(k)) / T(channels);
    });
    for_each(w.exec, "sp2::evidence_covariance", ncenters, KOKKOS_LAMBDA(int c) {
        T matrix[9]{};
        for (int h = 0; h < channels; ++h) {
            const auto v = &vectors(c, 3*h);
            const T gate = gates(c, h);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) matrix[3*i+j] += (gate * v[i]) * v[j];
        }
        T s{};
        for (int f = 0; f < invariants; ++f) s += scalar_weight(f) * features(c, f);
        s = (s + scalar_bias(0)) - (matrix[0]+matrix[4]+matrix[8])/T(3);
        for (int i = 0; i < 3; ++i) matrix[4*i] += s;
        T raw[9]{};
        T tensor_norm_sq{};
        for (int j = 0; j < irreps; ++j) {
            for (int i = 0; i < 9; ++i) raw[j] += matrix[i] * matrix_to_irreps(i*irreps+j);
            if (j != 0) tensor_norm_sq += raw[j] * raw[j];
        }
        raw[0] = -damp(-damp(raw[0], T(5), T(8)), T(20), T(25));
        const T norm = Kokkos::sqrt(tensor_norm_sq);
        const T ratio = norm >= T(1e-8) ? (damp(norm, T(5), T(8)) / norm) : T(1);
        for (int j = 1; j < irreps; ++j) raw[j] *= ratio;
        for (int j = 0; j < 9; ++j) {
            matrix[j] = T(0);
            for (int i = 0; i < irreps; ++i) matrix[j] += raw[i] * q(i*9+j);
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = i+1; j < 3; ++j) {
                const T x = (matrix[3*i+j] + matrix[3*j+i]) * T(0.5);
                matrix[3*i+j] = matrix[3*j+i] = x;
            }
        }
        T covariance[9];
        expm3(matrix, covariance);
        if (tangent) {
            const Vec3<T> vector{spin(c, 0), spin(c, 1), spin(c, 2)};
            const T length = vector.norm();
            if (length <= T(1e-6)) {
                for (int i = 0; i < 9; ++i) covariance[i] = T(0);
            } else {
                const auto u = vector / length;
                T projector[9], temporary[9];
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) projector[3*i+j] = T(i == j) - u[i] * u[j];
                multiply3(projector, covariance, temporary);
                multiply3(temporary, projector, covariance);
                for (int i = 0; i < 9; ++i) covariance[i] /= spin_scale_sq;
            }
        }
        const int dimension = tangent ? 2 : 3;
        const T n = T(dimension + 2) + softplus(raw_nu(c,0) + (soc ? coeff * soc_nu(c,0) : T(0)));
        const T k = softplus(raw_kappa(c,0) + (soc ? coeff * soc_kappa(c,0) : T(0))) + T(1e-6);
        nu(c, 0) = n;
        kappa(c, 0) = k;
        const T denominator = k * ((n - T(dimension)) - T(1));
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const T value = (covariance[3*i+j] + covariance[3*j+i]) * T(0.5);
                sigma(c,3*i+j) = value;
                uepi(c,3*i+j) = (n * value) / denominator;
            }
        }
    });
    return {nu, kappa, sigma, uepi};
}

}  // namespace sp2
