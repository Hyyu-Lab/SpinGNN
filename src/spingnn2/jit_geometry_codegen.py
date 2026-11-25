"""Emit checkpoint-specialized edge geometry and its Cartesian pullback."""

from math import factorial, isfinite, pi, sqrt
from pathlib import Path
import re

_BASIS_OPS = {
    "RadialBasisEdgeEncoding",
    "RadialBasisEdgeEncoding_spin",
    "EdgeScalarAddSpin",
}
_HARMONIC_OPS = {
    "SphericalHarmonicEdgeAttrs",
    "SphericalHarmonicEdgeAttrs_spin_nosoc",
    "SphericalHarmonicEdgeAttrs_spin_soc_simple",
}
_GEOMETRY_LANES = 8


def _real(value, precision="float64"):
    lit = float(value).hex()
    return f"Real({lit}f)" if precision == "float32" else f"Real({lit})"


def _components(spec):
    if spec is None:
        return []
    slots = []
    for block in spec["blocks"]:
        source = {"r": 0, "i": 1, "j": 2}[block["source"]]
        low = max(block["degrees"]) <= 2
        norm = block["normalization"]
        normalize = int(block["normalize"])
        if source != 0 and normalize:
            raise ValueError(
                "The JIT spin encoder requires unnormalized spin harmonics"
            )
        for degree in block["degrees"]:
            if type(degree) is not int or not 0 <= degree <= 11:
                raise ValueError("JIT spherical harmonic degrees must be in [0, 11]")
            for order in range(-degree, degree + 1):
                if low:
                    factor = {
                        "component": 1.0,
                        "integral": 1.0 / sqrt(4 * pi),
                        "norm": 1.0 / sqrt(2 * degree + 1),
                    }[norm]
                else:
                    m = abs(order)
                    factor = factorial(degree - m) / factorial(degree + m)
                    if m:
                        factor *= 2
                    if norm != "norm":
                        factor *= 2 * degree + 1
                    if norm == "integral":
                        factor /= 4 * pi
                    factor = sqrt(factor)
                slots.append((source, degree, order, normalize, int(low), factor))
    return [slots[i] for i in spec["sort"]]


def _basis_setup(spec, name, precision):
    kind = spec["kind"]
    spline = spec["spline"]
    width = spec["weights"].size
    knots = spline["values"].shape[0]
    p = spec["cutoff"]["p"]
    if type(p) is not int or not 2 <= p <= 65535:
        raise ValueError("Invalid JIT polynomial envelope degree")
    if not all(isfinite(float(spline[k])) for k in ("lower", "upper", "step")):
        raise ValueError("Invalid JIT Hermite interval")
    step = float(spline["step"])
    fields = {
        "kind": f"sp2::BasisDevice<Real>::{kind}",
        "width": width,
        "knots": knots,
        "p": p,
        "prefactor": _real(spec["prefactor"], precision),
        "cutoff_factor": _real(spec["cutoff"]["factor"], precision),
        "lower": float(spline["lower"]).hex(),
        "upper": float(spline["upper"]).hex(),
        "step": step.hex(),
        "lower_f": f"{float(spline['lower']).hex()}f",
        "upper_f": f"{float(spline['upper']).hex()}f",
        "step_f": f"{step.hex()}f",
        "inv_step_f": f"{(1.0 / step).hex()}f",
        "weights": f"{name}_weights",
        "values": "nullptr" if precision == "float32" else f"{name}_table",
        "derivatives": "nullptr" if precision == "float32" else f"{name}_derivatives",
        "values_f": f"{name}_table" if precision == "float32" else "nullptr",
        "derivatives_f": f"{name}_derivatives" if precision == "float32" else "nullptr",
    }
    if kind == "bessel":
        fields["radius"] = _real(spec["radius"], precision)
        if "mean" in spec:
            for key, width_name in (("mean", "mean_width"), ("inv_std", "std_width")):
                shape = spec[key].shape
                fields[width_name] = shape[0] if shape else 1
                fields[key] = f"{name}_{key}"
    elif kind == "fourier":
        fields["minimum"] = _real(spec["minimum"], precision)
        fields["maximum"] = _real(spec["maximum"], precision)
    else:
        raise ValueError(f"Unsupported basis kind {kind}")
    setup = "\n".join(f"{name}.{field} = {value};" for field, value in fields.items())
    return width, f"sp2::BasisDevice<Real> {name};\n{setup}"


def _emit_inline_basis_helper(fn_name, spec):
    """Emit a zero-stack FP32 compile-time basis evaluator (0 bytes local spill)."""
    kind = spec["kind"]
    spline = spec["spline"]
    width = int(spec["weights"].size)
    knots = int(spline["values"].shape[0])
    p = int(spec["cutoff"]["p"])
    if not 2 <= p <= 65535:
        raise ValueError("Invalid JIT polynomial envelope degree")
    lower = float(spline["lower"])
    upper = float(spline["upper"])
    step = float(spline["step"])
    if not all(isfinite(v) for v in (lower, upper, step)):
        raise ValueError("Invalid JIT Hermite interval")
    inv_step = 1.0 / step
    pref = float(spec["prefactor"])
    cf = float(spec["cutoff"]["factor"])
    if kind == "fourier":
        minimum = float(spec["minimum"])
        maximum = float(spec["maximum"])
        analytic = f"""
    const float weight = weights[channel];
    const float span = {(maximum - minimum).hex()}f;
    const float phase = weight * ((input - {minimum.hex()}f) / span);
    return {{{pref.hex()}f * cosf(phase), ({(-pref / (maximum - minimum)).hex()}f) * weight * sinf(phase)}};"""
    elif kind == "bessel":
        radius = float(spec["radius"])
        norm_code = ""
        if "mean" in spec:
            mw = spec["mean"].shape[0] if spec["mean"].shape else 1
            sw = spec["inv_std"].shape[0] if spec["inv_std"].shape else 1
            m_idx = "0" if mw == 1 else "channel"
            s_idx = "0" if sw == 1 else "channel"
            norm_code = f"""
    const float inverse_std = inv_std[{s_idx}];
    value = (value - mean[{m_idx}]) * inverse_std;
    slope *= inverse_std;"""
        analytic = f"""
    const float weight = weights[channel];
    const float radius = {radius.hex()}f;
    const float frequency = weight / radius;
    const float phase = weight * input / radius;
    const float sine = sinf(phase);
    float value = input == 0.0f ? frequency : sine / input;
    float slope;
    if (fabsf(phase) < 0.125f) {{
        const float p2 = phase * phase;
        const float series = -0.33333334f + p2 * (0.033333335f + p2 *
            (-0.0011904762f + p2 * (2.2045855e-05f - p2 * 2.5052108e-07f)));
        slope = frequency * frequency * phase * series;
    }} else {{
        slope = (phase * cosf(phase) - sine) / (input * input);
    }}
    value *= {pref.hex()}f;
    slope *= {pref.hex()}f;{norm_code}
    const auto cutoff = sp2::polynomial_envelope<float>(input, {cf.hex()}f, {p});
    return {{value * cutoff.value, slope * cutoff.value + value * cutoff.slope}};"""
    else:
        raise ValueError(f"Unsupported basis kind {kind}")

    return f"""SP2_GEOMETRY_INLINE sp2::ValueSlope<float> {fn_name}(
    float input, int channel, int spline,
    const float* __restrict__ weights,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ values,
    const float* __restrict__ derivatives) {{
    if (spline && values && input >= {lower.hex()}f && input <= {upper.hex()}f) {{
        const int raw = int((input - {lower.hex()}f) * {inv_step.hex()}f);
        const int index = raw < 0 ? 0 : (raw < {knots - 1} ? raw : {knots - 2});
        const float t = fmaf(-{step.hex()}f, float(index), input - {lower.hex()}f) * {inv_step.hex()}f;
        const int off = index * {width} + channel;
        const float left = values[off];
        const float right = values[off + {width}];
        const float dleft = derivatives[off];
        const float dright = derivatives[off + {width}];
        const float secant = (right - left) * {inv_step.hex()}f;
        const float quadratic = 3.0f * secant - 2.0f * dleft - dright;
        const float cubic = dleft + dright - 2.0f * secant;
        const float val = fmaf({step.hex()}f * t, fmaf(t, fmaf(t, cubic, quadratic), dleft), left);
        const float slp = fmaf(t, fmaf(3.0f * t, cubic, 2.0f * quadratic), dleft);
        return {{val, slp}};
    }}{analytic}
}}"""


def _host_wrapper(symbol, arguments):
    args = ", ".join(
        f"*static_cast<{type_}*>(args[{i}])" for i, (type_, _) in enumerate(arguments)
    )
    return f"""extern "C" void {symbol}(int, int, void** args) {{
    sp2_geometry::{symbol}({args});
}}"""


def _cuda_param(type_, name):
    if type_.endswith("*"):
        return f"{type_} __restrict__ {name}"
    return f"{type_} {name}"


def _kernel(backend, symbol, arguments, body, *, launch_bounds=None):
    if backend == "cuda":
        params = ", ".join(_cuda_param(t, n) for t, n in arguments)
        lb = f"__launch_bounds__({launch_bounds}) " if launch_bounds else ""
        if symbol.endswith("_warp8_fwd") or symbol.endswith("_warp8_rev"):
            return f"""extern "C" __global__ {lb}void {symbol}({params}) {{
    constexpr int lanes = {_GEOMETRY_LANES};
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int e = idx / lanes, lane = idx % lanes;
    const unsigned live = __ballot_sync(0xffffffffu, e < count);
    if (e >= count) return;
    {body}
}}"""
        if symbol == "sp2_geom_scatter":
            return f"""extern "C" __global__ {lb}void {symbol}({params}) {{
    const int e = int(blockIdx.x * 4 + threadIdx.x / 32);
    if (e >= count) return;
    const int lane = int(threadIdx.x) & 31;
    {body}
}}"""
        return f"""extern "C" __global__ {lb}void {symbol}({params}) {{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= count) return;
    {body}
}}"""
    params = ", ".join(f"{type_} {name}" for type_, name in arguments)
    return f"""void {symbol}({params}) {{
    #pragma omp parallel for schedule(static) if(count >= 64)
    for (int e = 0; e < count; ++e) {{
        {body}
    }}
}}"""


def _arguments(scalar, real):
    return scalar + [
        ("Real*" if writable else "const Real*", name) for name, writable in real
    ]


def _geometry_fwd(basis, harmonics, initialize, precision, cuda=False):
    attrs = _components(harmonics)
    magnetic = basis["kind"] == "fourier"
    combined = magnetic and "radial" in basis
    base = basis["radial"] if combined else basis
    k_basis, params = _basis_setup(base, "b", precision)
    if combined:
        k_fourier, extra = _basis_setup(basis, "spin", precision)
        params += "\n" + extra
    else:
        k_fourier = k_basis if magnetic else 0
    width = len(attrs)
    matrix_range = f"old_width + {k_basis}"
    if combined:
        matrix_range = f"{k_basis + k_fourier}"

    body = f"""
    const int live_e = offsets ? offsets[ncenters] : count;
    const long long el = e;
    const int embedding_width = {matrix_range};
    if (e >= live_e) {{
        if ({int(initialize)}) {{ put3(r, e, {{}}); length[e] = Real(1e30); }}
        for (int c = 0; c < embedding_width; ++c) value[el * embedding_width + c] = Real(0);
        for (int c = 0; c < {k_basis}; ++c) slopes[el * {k_basis} + c] = Real(0);
"""
    if magnetic:
        body += f"""        for (int c = 0; c < 4; ++c) spin_local[el * 4 + c] = Real(0);
        for (int c = 0; c < {k_fourier}; ++c) spin_values[el * {k_fourier} + c] = Real(0);
"""
        if combined:
            body += f"        for (int c = 0; c < {k_fourier}; ++c) spin_slopes[el * {k_fourier} + c] = Real(0);\n"
    if width:
        body += f"""        for (int c = 0; c < {width}; ++c) {{
            features[el * {width} + c] = Real(0);
            put3(jacobian, el * {width} + c, {{}});
        }}
"""
    body += f"        {'return;' if cuda else 'continue;'}\n    }}\n"

    if initialize:
        displacement = (
            "resident_displacement(e, center, targets, frames, start, shared_cell, positions, shifts, cells)"
            if cuda
            else "displacement(e, center, pos_i, pos_j, shifts, cells)"
        )
        body += f"""
    const Vec rvec = {displacement};
    put3(r, e, rvec);
    const Real radius = rvec.norm();
    length[e] = radius;
"""
    else:
        body += """
    const Vec rvec = read3(r, e);
    const Real radius = length[e];
"""
    if magnetic:
        spins = (
            "read3(spins, start + center[e]), bvec = read3(spins, targets[e])"
            if cuda
            else "read3(si, e), bvec = read3(sj, e)"
        )
        body += f"""
    const Vec a = {spins};
    const Real aa = a.dot(a), bb = bvec.dot(bvec), dot = a.dot(bvec);
"""
        source = "spin" if combined else "b"
        body += f"""
    const auto envelope = sp2::polynomial_envelope(radius, {source}.cutoff_factor, {source}.p);
    spin_local[el * 4 + 0] = aa;
    spin_local[el * 4 + 1] = bb;
    spin_local[el * 4 + 2] = envelope.value;
    spin_local[el * 4 + 3] = envelope.slope;
"""
        if not combined:
            body += """
    for (int c = 0; c < old_width; ++c)
        value[el * (old_width + b.width) + c] = previous[el * old_width + c];
"""
    for k in range(k_basis):
        inp = "dot" if magnetic and not combined else "radius"
        col = str(k) if combined or not magnetic else f"old_width + {k}"
        v = (
            "((result.value * bb) * aa) * envelope.value"
            if magnetic and not combined
            else "result.value"
        )
        body += f"""
    {{
        const auto result = b.at({inp}, {k}, spline);
        value[el * embedding_width + ({col})] = {v};
        slopes[el * {k_basis} + {k}] = result.slope;
"""
        if magnetic and not combined:
            body += f"        spin_values[el * {k_basis} + {k}] = result.value;\n"
        body += "    }\n"
    if combined:
        for k in range(k_fourier):
            body += f"""
    {{
        const auto result = spin.at(dot, {k}, spline);
        value[el * embedding_width + {k_basis + k}] = ((result.value * bb) * aa) * envelope.value;
        spin_values[el * {k_fourier} + {k}] = result.value;
        spin_slopes[el * {k_fourier} + {k}] = result.slope;
    }}
"""
    for k, (source, degree, order, normalize, low, factor) in enumerate(attrs):
        vector = ["rvec", "a", "bvec"][source]
        body += f"""
    {{
        const auto h = sp2::harmonic({vector}, {{{source}, {degree}, {order}, {normalize}, {low}, {_real(factor, precision)}}});
        features[el * {width} + {k}] = h.value;
        put3(jacobian, el * {width} + {k}, h.derivative);
    }}
"""
    return params + body


def _cooperative_geometry_fwd(basis, harmonics, initialize, layout, base_fn, spin_fn):
    attrs = _components(harmonics)
    magnetic = basis["kind"] == "fourier"
    combined = magnetic and "radial" in basis
    base_spec = basis["radial"] if combined else basis
    k_basis = int(base_spec["weights"].size)
    if combined:
        k_spin = int(basis["weights"].size)
        env_cf = float(basis["cutoff"]["factor"])
        env_p = int(basis["cutoff"]["p"])
    elif magnetic:
        k_spin = k_basis
        env_cf = float(base_spec["cutoff"]["factor"])
        env_p = int(base_spec["cutoff"]["p"])
    else:
        k_spin = 0
        env_cf = 0.0
        env_p = 2

    emb_expr = (
        f"{k_basis + (k_spin if combined else 0)}"
        if (not magnetic or combined)
        else f"(old_width + {k_spin})"
    )
    body = f"""
    const int live_e = offsets ? offsets[ncenters] : count;
    const long long el = e;
    const unsigned active = __ballot_sync(live, e < live_e);
    const int embedding_width = {emb_expr};
    if (e >= live_e) {{
        if ({int(initialize)} && lane == 0) {{ put3(r, e, {{}}); length[e] = 1e30f; }}
        for (int c = lane; c < embedding_width; c += lanes) value[el * embedding_width + c] = 0.0f;
        for (int c = lane; c < {k_basis}; c += lanes) slopes[el * {k_basis} + c] = 0.0f;
"""
    if magnetic:
        body += f"""        if (lane < 4) spin_local[el * 4 + lane] = 0.0f;
        for (int c = lane; c < {k_spin}; c += lanes) spin_values[el * {k_spin} + c] = 0.0f;
"""
        if combined:
            body += f"        for (int c = lane; c < {k_spin}; c += lanes) spin_slopes[el * {k_spin} + c] = 0.0f;\n"
    if attrs:
        body += f"""        for (int c = lane; c < {len(attrs)}; c += lanes) {{
            features[el * {len(attrs)} + c] = 0.0f;
            put3(jacobian, el * {len(attrs)} + c, {{}});
        }}
"""
    body += "        return;\n    }\n"

    if initialize:
        body += """
    Vec rvec{};
    Real radius{};
    if (lane == 0) {
        rvec = resident_displacement(e, center, targets, frames, start, shared_cell,
                                     positions, shifts, cells);
        radius = rvec.norm();
        put3(r, e, rvec);
        length[e] = radius;
    }
    radius = __shfl_sync(active, radius, 0, lanes);
    rvec = {__shfl_sync(active, rvec.x, 0, lanes), __shfl_sync(active, rvec.y, 0, lanes),
            __shfl_sync(active, rvec.z, 0, lanes)};
"""
    else:
        body += """
    const Vec rvec = read3(r, e);
    const Real radius = length[e];
"""
    if magnetic:
        body += f"""
    const Vec a = read3(spins, start + center[e]), bvec = read3(spins, targets[e]);
    Real aa{{}}, bb{{}}, dot{{}}, envelope_value{{}};
    if (lane == 0) {{
        aa = a.dot(a); bb = bvec.dot(bvec); dot = a.dot(bvec);
        const auto envelope = sp2::polynomial_envelope<float>(radius, {env_cf.hex()}f, {env_p});
        envelope_value = envelope.value;
        spin_local[e * 4 + 0] = aa;
        spin_local[e * 4 + 1] = bb;
        spin_local[e * 4 + 2] = envelope_value;
        spin_local[e * 4 + 3] = envelope.slope;
    }}
    aa = __shfl_sync(active, aa, 0, lanes);
    bb = __shfl_sync(active, bb, 0, lanes);
    dot = __shfl_sync(active, dot, 0, lanes);
    envelope_value = __shfl_sync(active, envelope_value, 0, lanes);
"""
    if not magnetic or combined:
        body += f"""
    for (int c = lane; c < {k_basis}; c += lanes) {{
        const auto result = {base_fn}(radius, c, spline, b_weights, b_mean, b_inv_std, b_table, b_derivatives);
        value[el * embedding_width + c] = result.value;
        slopes[el * {k_basis} + c] = result.slope;
    }}
"""
    else:
        body += """
    for (int c = lane; c < old_width; c += lanes)
        value[el * embedding_width + c] = previous[el * old_width + c];
"""
    if magnetic:
        offset = str(k_basis) if combined else "old_width"
        spin_slopes = "spin_slopes" if combined else "slopes"
        w_arg = "spin_weights" if combined else "b_weights"
        t_arg = "spin_table" if combined else "b_table"
        d_arg = "spin_derivatives" if combined else "b_derivatives"
        eval_fn = spin_fn if combined else base_fn
        body += f"""
    for (int c = lane; c < {k_spin}; c += lanes) {{
        const auto result = {eval_fn}(dot, c, spline, {w_arg}, nullptr, nullptr, {t_arg}, {d_arg});
        value[el * embedding_width + {offset} + c] = ((result.value * bb) * aa) * envelope_value;
        spin_values[el * {k_spin} + c] = result.value;
        {spin_slopes}[el * {k_spin} + c] = result.slope;
    }}
"""
    if attrs:
        vector = (
            "h.source == 0 ? rvec : (h.source == 1 ? a : bvec)" if magnetic else "rvec"
        )
        body += f"""
    for (int c = lane; c < {len(attrs)}; c += lanes) {{
        const auto h = {layout}[c];
        const auto result = sp2::harmonic_with_norm({vector}, h, radius);
        features[el * {len(attrs)} + c] = result.value;
        put3(jacobian, el * {len(attrs)} + c, result.derivative);
    }}
"""
    return body


def _geometry_reverse(basis, harmonics, cuda=False):
    attrs = _components(harmonics)
    magnetic = basis["kind"] == "fourier"
    combined = magnetic and "radial" in basis
    k = basis["weights"].size
    k_r = basis["radial"]["weights"].size if combined else (0 if magnetic else k)
    width = len(attrs)
    emb_width = f"(old_width + {k})" if not combined else str(k + k_r)
    exit_stmt = "return;" if cuda else "continue;"
    body = f"""
    const int live_e = offsets ? offsets[ncenters] : count;
    const long long el = e;
    if (e >= live_e) {{
        if (dprevious) for (int c = 0; c < old_width; ++c) dprevious[el * old_width + c] = Real(0);
        if (dradial) dradial[e] = Real(0);
        if (dr) put3(dr, e, {{}});
        if (dsi) put3(dsi, e, {{}});
        if (dsj) put3(dsj, e, {{}});
        {exit_stmt}
    }}
    Real radial{{}}; Vec dx{{}}, di{{}}, dj{{}};
    if (dvalue) {{
"""
    if not magnetic or combined:
        for channel in range(k_r):
            body += f"        radial += dvalue[el * {emb_width} + {channel}] * slopes[el * {k_r} + {channel}];\n"
    if magnetic:
        body += "        Real sum_v{}, sum_s{};\n"
        for channel in range(k):
            offset = f"old_width + {channel}" if not combined else str(k_r + channel)
            slope = "spin_slopes" if combined else "slopes"
            body += f"""        sum_v += dvalue[el * {emb_width} + ({offset})] * spin_values[el * {k} + {channel}];
        sum_s += dvalue[el * {emb_width} + ({offset})] * {slope}[el * {k} + {channel}];
"""
        if not combined:
            body += f"""
        if (dprevious) for (int c = 0; c < old_width; ++c)
            dprevious[el * old_width + c] = dvalue[el * {emb_width} + c];
"""
        body += """
        const Real aa = spin_local[el * 4 + 0], bb = spin_local[el * 4 + 1];
        const Real cv = spin_local[el * 4 + 2], cs = spin_local[el * 4 + 3];
        const auto a = read3(si, e), b = read3(sj, e);
        const Real dot_derivative = (sum_s * aa * bb) * cv;
        radial += sum_v * aa * bb * cs;
        di = a * (Real(2) * cv * sum_v * bb) + b * dot_derivative;
        dj = b * (Real(2) * cv * sum_v * aa) + a * dot_derivative;
"""
    body += "    }\n    if (dfeatures) {\n"
    for c, (source, *_) in enumerate(attrs):
        target = ["dx", "di", "dj"][source]
        body += f"        {target} = {target} + read3(jacobian, el * {width} + {c}) * dfeatures[el * {width} + {c}];\n"
    body += """    }
    if (dradial) dradial[e] = radial;
    if (dr) put3(dr, e, dx);
    if (dsi) put3(dsi, e, di);
    if (dsj) put3(dsj, e, dj);
"""
    return body


def _cooperative_geometry_rev(basis, harmonics, layout):
    """8-lane warp-cooperative coalesced reverse geometry kernel for CUDA FP32."""
    attrs = _components(harmonics)
    magnetic = basis["kind"] == "fourier"
    combined = magnetic and "radial" in basis
    k = int(basis["weights"].size)
    k_r = int(basis["radial"]["weights"].size) if combined else (0 if magnetic else k)
    width = len(attrs)
    emb_width = f"(old_width + {k})" if not combined else str(k + k_r)
    body = """
    const int live_e = offsets ? offsets[ncenters] : count;
    const long long el = e;
    const unsigned active = __ballot_sync(live, e < live_e);
    if (e >= live_e) {
        if (dprevious) for (int c = lane; c < old_width; c += lanes) dprevious[el * old_width + c] = 0.0f;
        if (lane == 0) {
            if (dradial) dradial[e] = 0.0f;
            if (dr) put3(dr, e, {});
            if (dsi) put3(dsi, e, {});
            if (dsj) put3(dsj, e, {});
        }
        return;
    }
    auto reduce8 = [&](Real v) {
        v += __shfl_down_sync(active, v, 4, lanes);
        v += __shfl_down_sync(active, v, 2, lanes);
        v += __shfl_down_sync(active, v, 1, lanes);
        return v;
    };
    Real radial = 0.0f;
    Vec dx{}, di{}, dj{};
    if (dvalue) {
"""
    if not magnetic or combined:
        body += f"""        for (int c = lane; c < {k_r}; c += lanes)
            radial += dvalue[el * {emb_width} + c] * slopes[el * {k_r} + c];
        radial = reduce8(radial);
"""
    if magnetic:
        offset = "old_width + c" if not combined else f"{k_r} + c"
        slope = "spin_slopes" if combined else "slopes"
        body += f"""        Real sum_v = 0.0f, sum_s = 0.0f;
        for (int c = lane; c < {k}; c += lanes) {{
            const Real dv = dvalue[el * {emb_width} + ({offset})];
            sum_v += dv * spin_values[el * {k} + c];
            sum_s += dv * {slope}[el * {k} + c];
        }}
        sum_v = reduce8(sum_v);
        sum_s = reduce8(sum_s);
"""
        if not combined:
            body += f"""        if (dprevious) {{
            for (int c = lane; c < old_width; c += lanes)
                dprevious[el * old_width + c] = dvalue[el * {emb_width} + c];
        }}
"""
        body += """        if (lane == 0) {
            const Real aa = spin_local[el * 4 + 0], bb = spin_local[el * 4 + 1];
            const Real cv = spin_local[el * 4 + 2], cs = spin_local[el * 4 + 3];
            const auto a = read3(si, e), b = read3(sj, e);
            const Real dot_derivative = (sum_s * aa * bb) * cv;
            radial += sum_v * aa * bb * cs;
            di = a * (Real(2) * cv * sum_v * bb) + b * dot_derivative;
            dj = b * (Real(2) * cv * sum_v * aa) + a * dot_derivative;
        }
"""
    body += "    }\n"
    if width:
        body += f"""    if (dfeatures) {{
        Vec ldx{{}}, ldi{{}}, ldj{{}};
        for (int c = lane; c < {width}; c += lanes) {{
            const Real df = dfeatures[el * {width} + c];
            const Vec jv = read3(jacobian, el * {width} + c) * df;
            const int src = {layout}[c].source;
            if (src == 0) ldx = ldx + jv;
            else if (src == 1) ldi = ldi + jv;
            else ldj = ldj + jv;
        }}
        ldx.x = reduce8(ldx.x); ldx.y = reduce8(ldx.y); ldx.z = reduce8(ldx.z);
        if (lane == 0) dx = dx + ldx;
"""
        if magnetic:
            body += """        ldi.x = reduce8(ldi.x); ldi.y = reduce8(ldi.y); ldi.z = reduce8(ldi.z);
        ldj.x = reduce8(ldj.x); ldj.y = reduce8(ldj.y); ldj.z = reduce8(ldj.z);
        if (lane == 0) { di = di + ldi; dj = dj + ldj; }
"""
        body += "    }\n"
    body += """    if (lane == 0) {
        if (dradial) dradial[e] = radial;
        if (dr) put3(dr, e, dx);
        if (dsi) put3(dsi, e, di);
        if (dsj) put3(dsj, e, dj);
    }
"""
    return body


_FORWARD = (
    _arguments(
        [
            ("int", "count"),
            ("int", "spline"),
            ("int", "old_width"),
            ("const int*", "center"),
        ],
        [
            (name, False)
            for name in (
                "pos_i",
                "pos_j",
                "shifts",
                "cells",
                "si",
                "sj",
                "previous",
                "b_weights",
                "b_mean",
                "b_inv_std",
            )
        ]
        + [
            (name, True)
            for name in (
                "r",
                "length",
                "value",
                "slopes",
                "spin_local",
                "spin_values",
                "spin_slopes",
                "features",
                "jacobian",
            )
        ],
    )
    + [
        ("const Real*", name)
        for name in (
            "b_table",
            "b_derivatives",
            "spin_table",
            "spin_derivatives",
            "spin_weights",
        )
    ]
    + [("const int*", "offsets"), ("int", "ncenters")]
)

_RESIDENT_INDEX = [
    ("const int*", "targets"),
    ("const int*", "frames"),
    ("int", "start"),
    ("int", "shared_cell"),
]
_RESIDENT_COORDINATES = [
    ("const Real*", name) for name in ("positions", "shifts", "cells")
]
_RESIDENT_FORWARD = (
    _FORWARD[:4]
    + _RESIDENT_INDEX
    + _RESIDENT_COORDINATES
    + [("const Real*", "spins")]
    + _FORWARD[10:]
)

_REVERSE = _arguments(
    [("int", "count"), ("int", "old_width")],
    [
        (name, False)
        for name in (
            "si",
            "sj",
            "dvalue",
            "dfeatures",
            "slopes",
            "spin_local",
            "spin_values",
            "spin_slopes",
            "jacobian",
        )
    ]
    + [(name, True) for name in ("dprevious", "dradial", "dr", "dsi", "dsj")],
) + [("const int*", "offsets"), ("int", "ncenters")]


_HELPERS = r"""
using Vec = sp2::Vec3<Real>;

SP2_GEOMETRY_INLINE Vec read3(const Real* data, long long row) {
    const auto at = 3ull * row;
    return {data[at], data[at + 1], data[at + 2]};
}

SP2_GEOMETRY_INLINE void put3(Real* data, long long row, Vec vec) {
    const auto at = 3ull * row;
    data[at] = vec.x;
    data[at + 1] = vec.y;
    data[at + 2] = vec.z;
}

SP2_GEOMETRY_INLINE Real rounded_product(Real x, Real y) {
#if defined(__CUDACC_RTC__)
    if constexpr (sizeof(Real) == sizeof(float)) return __fmul_rn(x, y);
    else return __dmul_rn(x, y);
#else
    return x * y;
#endif
}

SP2_GEOMETRY_INLINE Vec displacement(int edge, const int* center,
    const Real* pos_i, const Real* pos_j, const Real* shifts, const Real* cells) {
    const int i = center[edge];
    const Vec delta = read3(pos_j, edge) - read3(pos_i, i);
    if (!cells) return delta;
    const Real* cell = cells + 9ull * i;
    const auto s = read3(shifts, edge);
    const auto shift = [&] (int c) {
        return rounded_product(s.x, cell[c]) + rounded_product(s.y, cell[c + 3])
               + rounded_product(s.z, cell[c + 6]);
    };
    return delta + Vec{shift(0), shift(1), shift(2)};
}

SP2_GEOMETRY_INLINE Vec resident_displacement(int edge, const int* center,
    const int* targets, const int* frames, int start, int shared_cell,
    const Real* positions, const Real* shifts, const Real* cells) {
    const int i = start + center[edge], j = targets[edge];
    const Vec delta = read3(positions, j) - read3(positions, i);
    if (!cells) return delta;
    const Real* cell = cells + 9ull * (shared_cell ? 0 : frames[i]);
    const auto s = read3(shifts, edge);
    const auto shift = [&] (int c) {
        return rounded_product(s.x, cell[c]) + rounded_product(s.y, cell[c + 3])
               + rounded_product(s.z, cell[c + 6]);
    };
    return delta + Vec{shift(0), shift(1), shift(2)};
}

SP2_GEOMETRY_INLINE Vec merged_gradient(int edge, const Real* r, const Real* length,
    const Real* dr, const Real* dl) {
    Vec out = dr ? read3(dr, edge) : Vec{};
    const Real l = length[edge];
    if (dl && l != Real(0)) out = out + (read3(r, edge) / l) * dl[edge];
    return out;
}
"""

_SCATTER = r"""
    if (e < touched) {
        Vec df{}, ds{};
        if (force) df = read3(force, e);
        if (spin_force) ds = read3(spin_force, e);
        for (int k = endpoints[e]; k < endpoints[e + 1]; ++k) {
            const int index = signed_edges[k];
            const int edge = index > 0 ? index - 1 : -index - 1;
            if (force) {
                const Vec d = merged_gradient(edge, r, length, dr, dl);
                df = index > 0 ? df + d : df - d;
            }
            if (spin_force) {
                const Real* src = index > 0 ? dsi : dsj;
                if (src) ds = ds - read3(src, edge);
            }
        }
        if (force) put3(force, e, df);
        if (spin_force) put3(spin_force, e, ds);
    } else {
        const int i = e - touched;
        Real virial[9]{};
        for (int k = centers[i]; k < centers[i + 1]; ++k) {
            const Vec d = merged_gradient(k, r, length, dr, dl), delta = read3(r, k);
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    virial[3 * a + b] -= rounded_product(d[a], delta[b]);
        }
        for (int c = 0; c < 9; ++c) atom_virial[9ull * i + c] = virial[c];
    }
"""

_DEVICE_SCATTER = r"""
    const int i = start + e;
    Vec df{}, ds{};
    Real va[9]{};
    const int begin = centers[e], end = centers[e + 1];
    for (int k = begin + lane; k < end; k += 32) {
        const int j = targets[k];
        if (spin_force) {
            if (dsi) ds = ds - read3(dsi, k);
            if (dsj) {
                const Vec d = read3(dsj, k);
                if (i == j) ds = ds - d;
                else for (int c = 0; c < 3; ++c) atomicAdd(spin_force + 3ull * j + c, -d[c]);
            }
        }
        if (force) {
            const Vec d = merged_gradient(k, r, length, dr, dl), delta = read3(r, k);
            if (i != j) {
                df = df + d;
                for (int c = 0; c < 3; ++c) atomicAdd(force + 3ull * j + c, -d[c]);
            }
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    va[3 * a + b] -= rounded_product(d[a], delta[b]);
        }
    }
    auto wsum = [](Real v) {
        for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
        return v;
    };
    if (force) {
        df.x = wsum(df.x); df.y = wsum(df.y); df.z = wsum(df.z);
        if (atom_virial) for (int c = 0; c < 9; ++c) va[c] = wsum(va[c]);
        if (lane == 0) {
            for (int c = 0; c < 3; ++c) atomicAdd(force + 3ull * i + c, df[c]);
            if (atom_virial) for (int c = 0; c < 9; ++c) atom_virial[9ull * i + c] = va[c];
        }
    }
    if (spin_force) {
        ds.x = wsum(ds.x); ds.y = wsum(ds.y); ds.z = wsum(ds.z);
        if (lane == 0)
            for (int c = 0; c < 3; ++c) atomicAdd(spin_force + 3ull * i + c, ds[c]);
    }
"""


def generate_geometry(model, backend, precision, gpu_arch=None):
    """Generate kernels from a decoded compact model contract."""
    scalar = {"float32": "float", "float64": "double"}[precision]
    text = (Path(__file__).parent / "geometry_kernels.hpp").read_text(encoding="utf-8")
    cuda = backend == "cuda"
    if cuda:
        preamble = """#define SP2_GEOMETRY_INLINE __device__ __forceinline__
#define SP2_GEOMETRY_MATH
"""
    else:
        preamble = """#include <cmath>
#define SP2_GEOMETRY_INLINE inline
#define SP2_GEOMETRY_MATH std
"""
    preamble += text + "\n#undef SP2_GEOMETRY_MATH\n"
    preamble += f"\nnamespace sp2_geometry {{\nusing Real = {scalar};\n" + _HELPERS
    disp_body = (
        """const int live_e = offsets ? offsets[ncenters] : count;
    if (e >= live_e) { put3(r, e, {}); length[e] = Real(1e30); return; }
    const Vec v = resident_displacement(e, center, targets, frames, start, shared_cell, positions, shifts, cells);
    put3(r, e, v);
    length[e] = v.norm();"""
        if cuda
        else """const Vec v = displacement(e, center, pos_i, pos_j, shifts, cells);
    put3(r, e, v);
    length[e] = v.norm();"""
    )
    kernels = [
        (
            "sp2_geom_resident_displacement" if cuda else "sp2_geom_displacement",
            (
                [("int", "count"), ("const int*", "center")]
                + _RESIDENT_INDEX
                + _RESIDENT_COORDINATES
                + [
                    ("const int*", "offsets"),
                    ("int", "ncenters"),
                    ("Real*", "r"),
                    ("Real*", "length"),
                ]
                if cuda
                else _arguments(
                    [("int", "count"), ("const int*", "center")],
                    [
                        (name, name in ("r", "length"))
                        for name in ("pos_i", "pos_j", "shifts", "cells", "r", "length")
                    ],
                )
            ),
            disp_body,
        ),
        (
            "sp2_geom_scatter",
            _arguments(
                [
                    ("int", "count"),
                    *(
                        ([("int", "start"), ("const int*", "targets")])
                        if cuda
                        else [
                            ("int", "touched"),
                            ("const int*", "endpoints"),
                            ("const int*", "signed_edges"),
                        ]
                    ),
                    ("const int*", "centers"),
                ],
                [
                    (name, name in ("force", "spin_force", "atom_virial"))
                    for name in (
                        "r",
                        "length",
                        "dr",
                        "dl",
                        "dsi",
                        "dsj",
                        "force",
                        "spin_force",
                        "atom_virial",
                    )
                ],
            ),
            _DEVICE_SCATTER if cuda else _SCATTER,
        ),
    ]
    layers = model["layers"]
    paired_harmonic = -1
    layouts = []
    inline_helpers = []
    for i, layer in enumerate(layers):
        if i == paired_harmonic:
            continue
        if layer["op"] in _HARMONIC_OPS:
            raise ValueError("JIT spherical harmonics require an adjacent edge basis")
        if layer["op"] not in _BASIS_OPS:
            continue
        basis = layer["basis"]
        harmonics = None
        if i + 1 < len(layers) and layers[i + 1]["op"] in _HARMONIC_OPS:
            paired_harmonic = i + 1
            harmonics = layers[paired_harmonic]["harmonics"]
        name = layer["name"]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            raise ValueError(f"Invalid JIT geometry layer name {name!r}")
        name = "sp2_geom_" + name
        init = layer["op"] == "RadialBasisEdgeEncoding"
        forward = name + ("_resident_fwd" if cuda else "_fwd")
        reverse = name + "_rev"
        if cuda and precision == "float32":
            layout = name + "_layout"
            slots = _components(harmonics)
            if slots:
                values = ",\n".join(
                    f"    {{{source}, {degree}, {order}, {normalize}, {low}, {_real(factor, precision)}}}"
                    for source, degree, order, normalize, low, factor in slots
                )
                layouts.append(
                    f"__device__ const sp2::HarmonicComponent<Real> {layout}[] = {{\n{values}\n}};"
                )
            combined = basis["kind"] == "fourier" and "radial" in basis
            base_fn = name + "_eval_base"
            spin_fn = name + "_eval_spin"
            inline_helpers.append(
                _emit_inline_basis_helper(
                    base_fn, basis["radial"] if combined else basis
                )
            )
            if combined:
                inline_helpers.append(_emit_inline_basis_helper(spin_fn, basis))
            forward = name + "_resident_warp8_fwd"
            reverse = name + "_warp8_rev"
            fwd_body = _cooperative_geometry_fwd(
                basis, harmonics, init, layout, base_fn, spin_fn
            )
            rev_body = _cooperative_geometry_rev(basis, harmonics, layout)
        else:
            fwd_body = _geometry_fwd(basis, harmonics, init, precision, cuda)
            rev_body = _geometry_reverse(basis, harmonics, cuda)
        kernels.extend(
            (
                (forward, _RESIDENT_FORWARD if cuda else _FORWARD, fwd_body),
                (reverse, _REVERSE, rev_body),
            )
        )
    lb = "128, 4" if cuda else None
    source = (
        preamble
        + "\n".join(layouts)
        + "\n"
        + "\n".join(inline_helpers)
        + "\n"
        + "\n".join(_kernel(backend, *kernel, launch_bounds=lb) for kernel in kernels)
    )
    source += "\n}  // namespace sp2_geometry\n#undef SP2_GEOMETRY_INLINE\n"
    if backend != "cuda":
        source += "using sp2_geometry::Real;\n" + "\n".join(
            _host_wrapper(name, args) for name, args, _ in kernels
        )
    return source, [name for name, _, _ in kernels]
