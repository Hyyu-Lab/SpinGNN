"""Analytical edge bases and calibrated cubic Hermite interpolation."""

import math

import numpy as np


def polynomial_cutoff(x, factor, p):
    """Evaluate the compact polynomial envelope and its distance derivative."""
    u = np.minimum(x * factor, 1.0)
    tail = 1.0 - u
    series = np.full_like(u, p * (p + 1) // 2)
    for k in range(p - 2, -1, -1):
        series = series * u + float((k + 1) * (k + 2) // 2)
    # The factored envelope avoids cancellation as u approaches one at high p.
    value = tail * tail * tail * series
    derivative = (-factor * (p * (p + 1) * (p + 2) / 2)) * u ** (p - 1) * tail**2
    return value, derivative


def _bessel(spec, x):
    radius = spec["radius"]
    weights = np.asarray(spec["weights"], dtype=x.dtype)
    frequency = weights / radius
    x = x[..., None]
    phase = weights * x / radius
    denominator = np.where(x == 0, 1.0, x)
    sine = np.sin(phase)
    value = np.where(x == 0, frequency, sine / denominator)
    # Expand the removable singularity in the derivative near the origin.
    small = np.abs(phase) < 0.125
    derivative_denominator = np.where(small, 1.0, denominator)
    derivative = (phase * np.cos(phase) - sine) / derivative_denominator**2
    z = np.where(small, phase, 0.0)
    z2 = z * z
    series = -1.0 / 3 + z2 * (
        1.0 / 30 + z2 * (-1.0 / 840 + z2 * (1.0 / 45360 - z2 * (1.0 / 3991680)))
    )
    derivative = np.where(small, frequency**2 * z * series, derivative)
    value *= spec["prefactor"]
    derivative *= spec["prefactor"]
    if "mean" in spec:
        mean = np.asarray(spec["mean"], dtype=x.dtype)
        inv_std = np.asarray(spec["inv_std"], dtype=x.dtype)
        value = (value - mean) * inv_std
        derivative *= inv_std
    cutoff, d_cutoff = polynomial_cutoff(x, **spec["cutoff"])
    return value * cutoff, derivative * cutoff + value * d_cutoff


def _fourier(spec, x):
    weights = np.asarray(spec["weights"], dtype=x.dtype)
    width = spec["maximum"] - spec["minimum"]
    phase = weights * ((x[..., None] - spec["minimum"]) / width)
    prefactor = spec["prefactor"]
    return prefactor * np.cos(phase), (-prefactor / width) * weights * np.sin(phase)


def _analytic(spec, x):
    if spec["kind"] == "bessel":
        return _bessel(spec, x)
    if spec["kind"] == "fourier":
        return _fourier(spec, x)
    raise ValueError(f"Unsupported basis {spec['kind']!r}")


def _hermite(x, lower, step, values, derivatives):
    index = np.clip(np.floor((x - lower) / step), 0, len(values) - 2).astype(np.intp)
    t = ((x - (lower + step * index)) / step)[..., None]
    left, right = values[index], values[index + 1]
    dleft, dright = derivatives[index], derivatives[index + 1]
    secant = (right - left) / step
    quadratic = 3 * secant - 2 * dleft - dright
    cubic = dleft + dright - 2 * secant
    value = left + step * t * (dleft + t * (quadratic + t * cubic))
    derivative = dleft + t * (2 * quadratic + 3 * t * cubic)
    return value, derivative


def _cutoff_derivative_bound(factor, p, order):
    # Differentiate u**(p-1) * (1-u)**2 in the factored first derivative.
    bound = 0.0
    for k in range(order):
        a = p - 1 - k
        b = 2 - (order - 1 - k)
        if min(a, b) < 0:
            continue
        u = a / (a + b) if a + b else 0
        term = u**a * (1 - u) ** b
        bound += (
            math.comb(order - 1, k)
            * math.perm(p - 1, k)
            * math.perm(2, order - 1 - k)
            * term
        )
    return (p * (p + 1) * (p + 2) / 2) * factor**order * bound


def _fourth_derivative_bound(spec):
    weights = np.abs(np.asarray(spec["weights"], dtype=np.float64))
    if spec["kind"] == "fourier":
        frequency = weights / (spec["maximum"] - spec["minimum"])
        return float(np.max(abs(spec["prefactor"]) * frequency**4))

    # sin(kr)/r = k * integral_0^1 cos(krt) dt gives global Bessel bounds.
    frequency = weights / spec["radius"]
    bessel = [abs(spec["prefactor"]) * frequency ** (n + 1) / (n + 1) for n in range(5)]
    if "mean" in spec:
        bessel[0] = bessel[0] + np.abs(spec["mean"])
        bessel = [bound * np.abs(spec["inv_std"]) for bound in bessel]
    envelope = [1.0] + [
        _cutoff_derivative_bound(order=n, **spec["cutoff"]) for n in range(1, 5)
    ]
    return float(
        np.max(sum(math.comb(4, n) * bessel[4 - n] * envelope[n] for n in range(5)))
    )


def calibrate_spline(spec, lower, upper):
    """Check spline error on a subgrid and bound its increase between samples."""
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError(f"Invalid spline domain [{lower}, {upper}]")
    derivative_four = _fourth_derivative_bound(spec)
    if not np.isfinite(derivative_four):
        raise ValueError(f"Non-finite parameters in {spec['kind']} basis")
    subdivisions = 16
    for power in range(7, 18):
        intervals = 2**power
        step = (upper - lower) / intervals
        knots = np.linspace(lower, upper, intervals + 1, dtype=np.float64)
        values, derivatives = _analytic(spec, knots)
        value_error = derivative_error = 0.0
        for fraction in np.arange(1, subdivisions) / subdivisions:
            x = knots[:-1] + step * fraction
            exact, d_exact = _analytic(spec, x)
            approx, d_approx = _hermite(x, lower, step, values, derivatives)
            value_error = max(value_error, float(np.max(np.abs(exact - approx))))
            derivative_error = max(
                derivative_error, float(np.max(np.abs(d_exact - d_approx)))
            )
        # Subtract the exact Taylor cubic about the interval midpoint. Bounding
        # the remainder at each endpoint in the Hermite formula gives
        # |f''' - H'''| <= 13/16 h max|f''''|. Apply the linear interpolation
        # error bound to f' - H' between adjacent calibration points.
        substep = step / subdivisions
        derivative_bound = (
            derivative_error + (13 / 16) * step * derivative_four * substep**2 / 8
        )
        value_bound = value_error + derivative_bound * substep / 2
        if (
            max(value_error, derivative_error) < 1.0e-8
            and max(value_bound, derivative_bound) < 1.0e-7
        ):
            return {
                "lower": float(lower),
                "upper": float(upper),
                "step": float(step),
                "values": values,
                "derivatives": derivatives,
                "max_value_error": value_error,
                "max_derivative_error": derivative_error,
                "value_error_bound": value_bound,
                "derivative_error_bound": derivative_bound,
            }
    raise ValueError(
        f"Unable to calibrate {spec['kind']} spline for value and derivative errors below 1e-7 "
        f"(measured {value_error:.3g}, {derivative_error:.3g})"
    )


def evaluate_basis(spec, x, radial_mode="analytic"):
    """Return basis values and scalar-input derivatives with shape ``x.shape + (K,)``."""
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
    if radial_mode == "analytic":
        return _analytic(spec, x)
    if radial_mode != "spline":
        raise ValueError(f"Unsupported radial mode {radial_mode!r}")
    table = spec["spline"]
    lower, upper = table["lower"], table["upper"]
    outside = (x < lower) | (x > upper)
    value, derivative = _hermite(
        np.clip(x.astype(np.float64), lower, upper),
        lower,
        table["step"],
        table["values"],
        table["derivatives"],
    )
    value = value.astype(x.dtype, copy=False)
    derivative = derivative.astype(x.dtype, copy=False)
    # Fourier bases accept dot products outside the usual unit-spin domain.
    if np.any(outside):
        value[outside], derivative[outside] = _analytic(spec, x[outside])
    return value, derivative


def spin_embedding(spec, distances, spins_j, spins_i, radial_mode="analytic"):
    """Return Fourier spin features and derivatives with respect to distance and both spins."""
    dot = np.sum(spins_i * spins_j, axis=-1)
    squares_j = np.sum(spins_j * spins_j, axis=-1, keepdims=True)
    squares_i = np.sum(spins_i * spins_i, axis=-1, keepdims=True)
    fourier, d_fourier = evaluate_basis(spec, dot, radial_mode)
    cutoff, d_cutoff = polynomial_cutoff(distances[:, None], **spec["cutoff"])
    product = fourier * squares_j * squares_i
    d_dot = (cutoff * d_fourier * squares_j * squares_i)[..., None]
    d_j = 2 * (cutoff * fourier * squares_i)[..., None] * spins_j[:, None, :]
    d_i = 2 * (cutoff * fourier * squares_j)[..., None] * spins_i[:, None, :]
    return (
        product * cutoff,
        product * d_cutoff,
        d_j + d_dot * spins_i[:, None, :],
        d_i + d_dot * spins_j[:, None, :],
    )
