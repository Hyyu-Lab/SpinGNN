"""Real solid harmonics and Cartesian derivatives in SpinGNN component order."""

import math

import numpy as np


def cartesian_norm(vectors, keepdims=False):
    """Euclidean norm with the source CPU kernel's three-component accumulation."""
    if vectors.dtype != np.float32:
        return np.linalg.norm(vectors, axis=-1, keepdims=keepdims)
    # Binary64 represents the product of two binary32 significands exactly.
    # Round once per multiply-add, in x/y/z order, as in vector_norm.
    x, y, z = np.moveaxis(vectors, -1, 0)
    xy = (y.astype(np.float64) * y + x * x).astype(np.float32)
    squared = (z.astype(np.float64) * z + xy).astype(np.float32)
    norm = np.sqrt(squared)
    return norm[..., None] if keepdims else norm


def spherical_harmonics(vectors, degrees, normalize=True, normalization="component"):
    """Return harmonics and their Jacobian with respect to the input vectors.

    The polar axis is y. Negative and positive orders use sine and cosine of
    the azimuth measured from +z toward +x, respectively. This gives the
    linear block sqrt(3) * [x, y, z] in component normalization.
    """
    vectors = np.asarray(vectors)
    if not np.issubdtype(vectors.dtype, np.floating):
        vectors = vectors.astype(np.float64)
    if vectors.shape[-1:] != (3,):
        raise ValueError("Spherical harmonics require vectors with last dimension 3")
    if normalization not in {"component", "integral", "norm"}:
        raise ValueError(
            f"Unsupported spherical harmonic normalization {normalization!r}"
        )
    degrees = tuple(degrees)
    if not degrees or any(not 0 <= degree <= 11 for degree in degrees):
        raise ValueError("Spherical harmonic degrees must be in [0, 11]")

    coordinates = vectors
    if normalize:
        radii = cartesian_norm(vectors, keepdims=True)
        denominator = np.maximum(radii, 1.0e-12)
        coordinates = vectors / denominator
    if max(degrees) <= 2:
        blocks = _quadratic_harmonics(coordinates, degrees, normalization)
    else:
        blocks = _solid_harmonics(coordinates, degrees, normalization)
    value = np.concatenate([blocks[degree][0] for degree in degrees], axis=-1)
    jacobian = np.concatenate([blocks[degree][1] for degree in degrees], axis=-2)
    if normalize:
        # Above the normalization floor, only tangent motion changes direction.
        radial = np.sum(jacobian * coordinates[..., None, :], axis=-1, keepdims=True)
        tangent = jacobian - radial * coordinates[..., None, :]
        jacobian = np.where((radii >= 1.0e-12)[..., None], tangent, jacobian)
        jacobian = np.where(
            (radii > 0)[..., None], jacobian / denominator[..., None], 0
        )
    return value, jacobian


def _quadratic_harmonics(coords, degrees, normalization):
    """Direct Cartesian polynomials preserve the FP32 ordering used by the models."""
    x, y, z = np.moveaxis(coords, -1, 0)
    blocks = {}
    for degree in set(degrees):
        if degree == 0:
            value = np.ones_like(x)[..., None]
            jac = np.zeros_like(coords)[..., None, :]
        elif degree == 1:
            value = math.sqrt(3) * coords
            jac = np.broadcast_to(
                math.sqrt(3) * np.eye(3, dtype=coords.dtype), value.shape + (3,)
            )
        else:
            a, b = math.sqrt(15), math.sqrt(5)
            zero = np.zeros_like(x)
            value = np.stack(
                (
                    a * x * z,
                    a * x * y,
                    b * (y * y - 0.5 * (x * x + z * z)),
                    a * y * z,
                    (a / 2) * (z * z - x * x),
                ),
                axis=-1,
            )
            jac = np.stack(
                (
                    a * z,
                    zero,
                    a * x,
                    a * y,
                    a * x,
                    zero,
                    -b * x,
                    2 * b * y,
                    -b * z,
                    zero,
                    a * z,
                    a * y,
                    -a * x,
                    zero,
                    a * z,
                ),
                axis=-1,
            )
            jac = jac.reshape(value.shape + (3,))
        if normalization != "component":
            divisor = math.sqrt(
                4 * math.pi if normalization == "integral" else 2 * degree + 1
            )
            value, jac = value / divisor, jac / divisor
        blocks[degree] = value, jac
    return blocks


def _solid_harmonics(coordinates, degrees, normalization):
    x, y, z = np.moveaxis(coordinates, -1, 0)
    radius2 = np.sum(coordinates * coordinates, axis=-1)
    d_radius2 = 2 * coordinates
    d_y = np.array([0.0, 1.0, 0.0], dtype=coordinates.dtype)
    blocks = {
        degree: (
            np.empty(x.shape + (2 * degree + 1,), dtype=coordinates.dtype),
            np.empty(x.shape + (2 * degree + 1, 3), dtype=coordinates.dtype),
        )
        for degree in degrees
    }

    complex_dtype = np.result_type(coordinates.dtype, np.complex64)
    azimuth = z.astype(complex_dtype) + 1j * x
    d_azimuth = np.array([1j, 0, 1], dtype=complex_dtype)
    power = np.ones_like(azimuth)
    d_power = np.zeros_like(coordinates, dtype=complex_dtype)
    sectoral = 1
    for order in range(max(degrees) + 1):
        if order:
            d_power = order * power[..., None] * d_azimuth
            power = power * azimuth
            sectoral *= 2 * order - 1

        previous = np.zeros_like(x)
        d_previous = np.zeros_like(coordinates)
        polar = np.full_like(x, sectoral)
        d_polar = np.zeros_like(coordinates)
        for degree in range(order, max(degrees) + 1):
            if degree in blocks:
                value, jacobian = blocks[degree]
                scale = _normalization(degree, order, normalization)
                mode = scale * polar * power
                d_mode = scale * (
                    d_polar * power[..., None] + polar[..., None] * d_power
                )
                value[..., degree + order] = mode.real
                jacobian[..., degree + order, :] = d_mode.real
                if order:
                    value[..., degree - order] = mode.imag
                    jacobian[..., degree - order, :] = d_mode.imag

            # This is the differentiated Legendre recurrence with powers of r
            # absorbed into the polynomials, so it stays regular at r = 0.
            a = (2 * degree + 1) / (degree + 1 - order)
            b = (degree + order) / (degree + 1 - order)
            next_polar = a * y * polar - b * radius2 * previous
            next_derivative = a * (
                polar[..., None] * d_y + y[..., None] * d_polar
            ) - b * (previous[..., None] * d_radius2 + radius2[..., None] * d_previous)
            previous, polar = polar, next_polar
            d_previous, d_polar = d_polar, next_derivative

    return blocks


def _normalization(degree, order, normalization):
    ratio = math.factorial(degree - order) / math.factorial(degree + order)
    factor = ratio * (2 if order else 1)
    if normalization != "norm":
        factor *= 2 * degree + 1
    if normalization == "integral":
        factor /= 4 * math.pi
    return math.sqrt(factor)


def evaluate_harmonics(spec, vectors, spins_j=None, spins_i=None):
    """Evaluate exported harmonics layout and the Jacobian for each vector input."""
    sources = {"r": vectors, "j": spins_j, "i": spins_i}
    values, derivatives = [], []
    for block in spec["blocks"]:
        source = block["source"]
        value, jacobian = spherical_harmonics(
            sources[source],
            block["degrees"],
            block["normalize"],
            block["normalization"],
        )
        values.append(value)
        derivatives.append((source, jacobian))

    order = spec["sort"]
    value = np.concatenate(values, axis=-1)[..., order]
    jacobians = {
        source: np.concatenate(
            [jac if key == source else np.zeros_like(jac) for key, jac in derivatives],
            axis=-2,
        )[..., order, :]
        for source, _ in derivatives
    }
    return value, jacobians
