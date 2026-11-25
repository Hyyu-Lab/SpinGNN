"""Forward-only nodewise normal-inverse-Wishart readout."""

import numpy as np
from scipy.linalg import expm

from .harmonics import cartesian_norm


def _damp(x, transition, ceiling):
    width = ceiling - transition
    return np.where(
        x < transition, x, width * np.tanh((x - transition) / width) + transition
    )


def _affine(x, head):
    return x @ head["weight"].T + head["bias"]


def _softplus(x):
    # The eager readout switches to its linear branch above 20.
    return np.where(x > 20.0, x, np.logaddexp(0.0, x))


def evidence(layer, fields, spin):
    outputs = {}
    for prefix, head in layer["heads"].items():

        def read(suffix):
            x = fields[prefix + "_" + suffix]
            if prefix == "f" and "socf_sigma_s_raw" in fields:
                x = x + layer["soc_f_coeff"] * fields["socf_" + suffix]
            return x

        dim = 2 if prefix == "sf" else 3
        nu = dim + 2.0 + _softplus(read("nu"))
        kappa = _softplus(read("kappa")) + 1e-6
        vectors = read("sigma_v_raw").reshape(len(nu), -1, 3)
        invariants = np.concatenate(
            (read("sigma_s_raw"), (vectors**2).sum(-1)), axis=-1
        )
        gates = _affine(invariants, head["gate"]) / vectors.shape[1]
        matrix = np.einsum("nc,nci,ncj->nij", gates, vectors, vectors)
        trace = np.trace(matrix, axis1=-2, axis2=-1)[:, None] / 3
        matrix += (_affine(invariants, head["scalar"]) - trace)[:, :, None] * np.eye(
            3, dtype=matrix.dtype
        )
        raw = matrix.reshape(-1, 9) @ head["matrix_to_irreps"]
        scalar = -_damp(-_damp(raw[:, :1], 5.0, 8.0), 20.0, 25.0)
        tensor = raw[:, 1:]
        norm = np.linalg.norm(tensor, axis=-1, keepdims=True)
        safe = np.maximum(norm, 1e-8)
        tensor = tensor * np.where(norm < 1e-8, 1, _damp(safe, 5.0, 8.0) / safe)
        log_sigma = (
            np.concatenate((scalar, tensor), axis=1) @ layer["Q_e2ip"]
        ).reshape(-1, 3, 3)
        sigma = np.stack(
            [expm(m) for m in (log_sigma + log_sigma.swapaxes(1, 2)) * 0.5]
        )
        if prefix == "sf":
            norm = cartesian_norm(spin, keepdims=True)
            unit = spin / np.maximum(norm, 1e-6)
            projector = (
                np.eye(3, dtype=unit.dtype) - unit[:, :, None] * unit[:, None, :]
            )
            sigma = (
                (projector @ sigma @ projector)
                * (norm > 1e-6)[:, :, None]
                / layer["spin_force_scale"] ** 2
            )
        sigma = (sigma + sigma.swapaxes(1, 2)) * 0.5
        uepi = nu[:, :, None] * sigma / (kappa * (nu - dim - 1))[:, :, None]
        for key, value in (
            ("nu", nu),
            ("kappa", kappa),
            ("sigma", sigma),
            ("Uepi", uepi),
        ):
            outputs[prefix + "_" + key] = value
    return outputs
