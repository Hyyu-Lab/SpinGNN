// Shared device math for the Kokkos engine and generated kernels.
namespace sp2 {

template<class T> struct Vec3 {
    T x{}, y{}, z{};

    SP2_GEOMETRY_INLINE T operator[](int i) const {
        return i == 0 ? x : (i == 1 ? y : z);
    }
    SP2_GEOMETRY_INLINE Vec3 operator+(Vec3 v) const {
        return {x + v.x, y + v.y, z + v.z};
    }
    SP2_GEOMETRY_INLINE Vec3 operator-(Vec3 v) const {
        return {x - v.x, y - v.y, z - v.z};
    }
    SP2_GEOMETRY_INLINE Vec3 operator*(T scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }
    SP2_GEOMETRY_INLINE Vec3 operator/(T scalar) const {
        return {x / scalar, y / scalar, z / scalar};
    }
    SP2_GEOMETRY_INLINE T dot(Vec3 v) const {
        return (x * v.x + y * v.y) + z * v.z;
    }
    SP2_GEOMETRY_INLINE T norm() const {
        return SP2_GEOMETRY_MATH::sqrt(SP2_GEOMETRY_MATH::fma(z, z, SP2_GEOMETRY_MATH::fma(y, y, x * x)));
    }
};

template<class T> struct ValueSlope {
    T value{};
    T slope{};
};

template<class T> SP2_GEOMETRY_INLINE T ipow_pos(T base, int exp) {
    T res = T(1);
    while (exp > 0) {
        if (exp & 1) res *= base;
        base *= base;
        exp >>= 1;
    }
    return res;
}

template<class T> SP2_GEOMETRY_INLINE ValueSlope<T> polynomial_envelope(T r, T factor, int p) {
    const T u = r * factor;
    if (u >= T(1)) return {};
    const T tail = T(1) - u;
    T quotient = T(static_cast<long long>(p) * (p + 1) / 2);
    for (int k = p - 2; k >= 0; --k)
        quotient = quotient * u + T(static_cast<long long>(k + 1) * (k + 2) / 2);
    const T derivative = -factor * (T(p) * T(p + 1) * T(p + 2) / T(2))
                        * ipow_pos(u, p - 1) * (tail * tail);
    return {tail * tail * tail * quotient, derivative};
}

template<class T> struct BasisDevice {
    enum Kind { bessel, fourier };

    Kind kind{};
    int width{}, knots{}, p{};
    T radius{}, minimum{}, maximum{}, prefactor{}, cutoff_factor{};
    double lower{}, upper{}, step{};
    float lower_f{}, upper_f{}, step_f{}, inv_step_f{};
    const T* weights{};
    const T* mean{};
    const T* inv_std{};
    int mean_width{}, std_width{};
    const double* values{};
    const double* derivatives{};
    const float* values_f{};
    const float* derivatives_f{};

    SP2_GEOMETRY_INLINE ValueSlope<T> at(T input, int channel, bool spline) const {
        if constexpr (sizeof(T) == sizeof(float)) {
            const float x = static_cast<float>(input);
            if (spline && values_f && x >= lower_f && x <= upper_f) {
                const int raw = int((x - lower_f) * inv_step_f);
                const int index = raw < 0 ? 0 : (raw < knots - 1 ? raw : knots - 2);
                const float t = SP2_GEOMETRY_MATH::fma(-step_f, float(index), x - lower_f) * inv_step_f;
                const float left = values_f[index * width + channel];
                const float right = values_f[(index + 1) * width + channel];
                const float dleft = derivatives_f[index * width + channel];
                const float dright = derivatives_f[(index + 1) * width + channel];
                const float secant = (right - left) * inv_step_f;
                const float quadratic = 3.0f * secant - 2.0f * dleft - dright;
                const float cubic = dleft + dright - 2.0f * secant;
                const float val = SP2_GEOMETRY_MATH::fma(
                    step_f * t, SP2_GEOMETRY_MATH::fma(t, SP2_GEOMETRY_MATH::fma(t, cubic, quadratic), dleft), left);
                const float slp = SP2_GEOMETRY_MATH::fma(
                    t, SP2_GEOMETRY_MATH::fma(3.0f * t, cubic, 2.0f * quadratic), dleft);
                return {static_cast<T>(val), static_cast<T>(slp)};
            }
        }
        const double x = input;
        if (spline && x >= lower && x <= upper) {
            const int raw = int((x - lower) / step);
            const int index = raw < knots - 1 ? raw : knots - 2;
            const double t = (x - (lower + step * index)) / step;
            const double left = values[index * width + channel];
            const double right = values[(index + 1) * width + channel];
            const double dleft = derivatives[index * width + channel];
            const double dright = derivatives[(index + 1) * width + channel];
            const double secant = (right - left) / step;
            const double quadratic = 3 * secant - 2 * dleft - dright;
            const double cubic = dleft + dright - 2 * secant;
            return {T(left + step * t * (dleft + t * (quadratic + t * cubic))),
                    T(dleft + t * (2 * quadratic + 3 * t * cubic))};
        }

        const T weight = weights[channel];
        if (kind == fourier) {
            const T span = maximum - minimum;
            const T phase = weight * ((input - minimum) / span);
            return {prefactor * SP2_GEOMETRY_MATH::cos(phase),
                    (-prefactor / span) * weight * SP2_GEOMETRY_MATH::sin(phase)};
        }

        const T frequency = weight / radius;
        const T phase = weight * input / radius;
        const T sine = SP2_GEOMETRY_MATH::sin(phase);
        T value = input == T(0) ? frequency : sine / input;
        T slope;
        if (SP2_GEOMETRY_MATH::abs(phase) < T(0.125)) {
            const T p2 = phase * phase;
            const T series = T(-1.0 / 3) + p2 * (T(1.0 / 30) + p2 *
                (T(-1.0 / 840) + p2 * (T(1.0 / 45360) - p2 * T(1.0 / 3991680))));
            slope = frequency * frequency * phase * series;
        } else {
            slope = (phase * SP2_GEOMETRY_MATH::cos(phase) - sine) / (input * input);
        }
        value *= prefactor;
        slope *= prefactor;
        if (mean) {
            const T inverse_std = inv_std[std_width == 1 ? 0 : channel];
            value = (value - mean[mean_width == 1 ? 0 : channel]) * inverse_std;
            slope *= inverse_std;
        }
        const auto cutoff = polynomial_envelope(input, cutoff_factor, p);
        return {value * cutoff.value, slope * cutoff.value + value * cutoff.slope};
    }
};

template<class T> struct HarmonicComponent {
    int source{}, degree{}, order{};
    bool normalize{}, low_degree{};
    T factor{};
};

template<class T> struct HarmonicValue {
    T value{};
    Vec3<T> derivative{};
};

template<class T> SP2_GEOMETRY_INLINE HarmonicValue<T>
quadratic_harmonic(Vec3<T> u, int l, int m) {
    const T x = u.x, y = u.y, z = u.z;
    const T a = T(3.8729833462074168852), b = T(2.2360679774997896964);
    const T c = T(1.7320508075688772935);
    if (l == 0) return {T(1), {}};
    if (l == 1) {
        if (m < 0) return {c * x, {c, 0, 0}};
        if (m == 0) return {c * y, {0, c, 0}};
        return {c * z, {0, 0, c}};
    }
    switch (m) {
        case -2: return {a * x * z, {a * z, 0, a * x}};
        case -1: return {a * x * y, {a * y, a * x, 0}};
        case 0: return {b * (y * y - T(0.5) * (x * x + z * z)), {-b * x, T(2) * b * y, -b * z}};
        case 1: return {a * y * z, {0, a * z, a * y}};
        default: return {(a / T(2)) * (z * z - x * x), {-a * x, 0, a * z}};
    }
}

template<class T> SP2_GEOMETRY_INLINE HarmonicValue<T>
solid_harmonic(Vec3<T> u, int degree, int order) {
    const int m = order < 0 ? -order : order;
    const T r2 = u.dot(u);
    T re = T(1), im = T(0), sectoral = T(1);
    T dre = T(0), dim = T(0);
    for (int n = 1; n <= m; ++n) {
        dre = T(n) * re;
        dim = T(n) * im;
        const T next_re = re * u.z - im * u.x;
        im = re * u.x + im * u.z;
        re = next_re;
        sectoral *= T(2 * n - 1);
    }
    T previous{}, polar = sectoral;
    Vec3<T> d_previous{}, d_polar{};
    const Vec3<T> y_axis{0, 1, 0};
    for (int l = m; l < degree; ++l) {
        const T a = T(2 * l + 1) / T(l + 1 - m);
        const T b = T(l + m) / T(l + 1 - m);
        const T next = a * u.y * polar - b * r2 * previous;
        const auto d_next = (y_axis * polar + d_polar * u.y) * a
                          - (u * (T(2) * previous) + d_previous * r2) * b;
        previous = polar;
        polar = next;
        d_previous = d_polar;
        d_polar = d_next;
    }
    if (order < 0) return {polar * im, d_polar * im + Vec3<T>{dre, 0, dim} * polar};
    return {polar * re, d_polar * re + Vec3<T>{-dim, 0, dre} * polar};
}

template<class T> SP2_GEOMETRY_INLINE HarmonicValue<T>
harmonic_with_norm(Vec3<T> input, HarmonicComponent<T> component, T norm) {
    T scale = 1;
    if (component.normalize) {
        scale = norm > T(1e-12) ? norm : T(1e-12);
        input = input / scale;
    }
    auto v = component.low_degree
        ? quadratic_harmonic(input, component.degree, component.order)
        : solid_harmonic(input, component.degree, component.order);
    v.value *= component.factor;
    v.derivative = v.derivative * component.factor;
    if (component.normalize) {
        if (norm == T(0)) v.derivative = {};
        else {
            if (norm >= T(1e-12)) v.derivative = v.derivative - input * v.derivative.dot(input);
            v.derivative = v.derivative / scale;
        }
    }
    return v;
}

template<class T> SP2_GEOMETRY_INLINE HarmonicValue<T>
harmonic(Vec3<T> input, HarmonicComponent<T> component) {
    return harmonic_with_norm(input, component, component.normalize ? input.norm() : T(0));
}

}  // namespace sp2
