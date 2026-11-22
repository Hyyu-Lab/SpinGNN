"""Cartesian evidential covariances, with tangent projection for spin forces."""

import math
import torch
from e3nn import o3
from e3nn.io import CartesianTensor
from torch_runstats.scatter import scatter

from nequip.data import AtomicDataDict, register_fields
from nequip.nn import GraphModuleMixin


register_fields(
    node_fields=[
        p + "_" + k
        for p in ("f", "sf", "socf")
        for k in (
            "nu",
            "kappa",
            "sigma",
            "Uepi",
            "sigma_s_raw",
            "sigma_v_raw",
            "node_features",
        )
    ],
    edge_fields=[p + "_edge_vectors" for p in ("f", "sf", "socf")],
    graph_fields=[p + "_uncertainty" for p in ("f", "sf")]
    + ["total_" + p + "_Uepi" for p in ("f", "sf")],
)


def linear_tanh_damper(
    x: torch.Tensor, transition: float, ceiling: float
) -> torch.Tensor:
    return torch.where(
        x < transition,
        x,
        (ceiling - transition) * torch.tanh((x - transition) / (ceiling - transition))
        + transition,
    )


class DampedSymmetricOutput(torch.nn.Module):
    """Bound log-covariance eigenvalue scale without selecting a spatial axis."""

    def forward(self, x):
        scalar = linear_tanh_damper(x[:, :1], 5.0, 8.0)
        scalar = -linear_tanh_damper(-scalar, 20.0, 25.0)
        nonscalar = x[:, 1:]
        # This region is exactly linear. Keep even the inactive norm away from
        # zero, whose undefined Hessian would otherwise contaminate backward.
        small = torch.linalg.vector_norm(nonscalar, dim=-1, keepdim=True) < 1e-8
        safe = torch.where(small, torch.ones_like(nonscalar), nonscalar)
        norm = torch.linalg.vector_norm(safe, dim=-1, keepdim=True).clamp_min(1e-8)
        nonscalar = torch.where(
            small,
            nonscalar,
            nonscalar * (linear_tanh_damper(norm, 5.0, 8.0) / norm),
        )
        return torch.cat((scalar, nonscalar), dim=-1)


class ScriptFriendlySigmaHead(torch.nn.Module):
    def __init__(self, scalar_channels, vector_channels, matrix_to_irreps):
        super().__init__()
        self.vector_channels = vector_channels
        self.scalar_linear = torch.nn.Linear(scalar_channels + vector_channels, 1)
        self.gate_linear = torch.nn.Linear(
            scalar_channels + vector_channels, vector_channels
        )
        self.register_buffer("matrix_to_irreps", matrix_to_irreps)
        self.register_buffer("eye", torch.eye(3))

    def forward(self, scalars, vector_raw):
        vectors = vector_raw.reshape(-1, self.vector_channels, 3)
        invariants = torch.cat((scalars, vectors.square().sum(-1)), dim=-1)
        gates = self.gate_linear(invariants) / self.vector_channels
        symmetric = torch.einsum("nc,nci,ncj->nij", gates, vectors, vectors)
        trace = symmetric.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) / 3.0
        matrix = (
            symmetric
            + (self.scalar_linear(invariants) - trace).unsqueeze(-1) * self.eye
        )
        return matrix.reshape(-1, 9) @ self.matrix_to_irreps


class e2IP_nodewise(GraphModuleMixin, torch.nn.Module):
    """Activate NIW parameters and emit covariance and epistemic uncertainty.

    ``sigma`` and ``Uepi`` are Cartesian (N, 3, 3) tensors, in squared
    target units. Graph uncertainty is the mean trace over valid atoms.
    """

    def __init__(
        self,
        irreps_in=None,
        include_f=True,
        include_sf=False,
        soc_f_coeff=0.01,
        e2ip_spin_force_scale=40.0,
    ):
        super().__init__()
        if not math.isfinite(e2ip_spin_force_scale) or e2ip_spin_force_scale <= 0:
            raise ValueError("e2ip_spin_force_scale must be finite and positive")
        if not math.isfinite(soc_f_coeff):
            raise ValueError("soc_f_coeff must be finite")
        self.spin_force_scale = e2ip_spin_force_scale
        self.soc_f_coeff = soc_f_coeff
        self.prefixes = [
            p for p, include in (("f", include_f), ("sf", include_sf)) if include
        ]
        transform = (
            CartesianTensor("ij=ji")
            .reduced_tensor_products()
            .change_of_basis.flatten(-2)
        )
        self.register_buffer("transform", transform)
        self.damper = DampedSymmetricOutput()
        self.sigma_heads = torch.nn.ModuleDict()
        outputs = {}
        for prefix in self.prefixes:
            scalars = o3.Irreps(irreps_in[prefix + "_sigma_s_raw"])
            vectors = o3.Irreps(irreps_in[prefix + "_sigma_v_raw"])
            if any(ir.l != 0 for _, ir in scalars) or any(
                ir.l != 1 for _, ir in vectors
            ):
                raise ValueError("e2IP sigma head requires scalar and vector features")
            self.sigma_heads[prefix] = ScriptFriendlySigmaHead(
                scalars.dim, vectors.num_irreps, torch.linalg.pinv(transform)
            )
            outputs.update(
                {
                    prefix + "_" + k: o3.Irreps("0e")
                    for k in ("nu", "kappa", "uncertainty")
                }
            )
            outputs.update(
                {
                    prefix + "_sigma": None,
                    prefix + "_Uepi": None,
                    "total_" + prefix + "_Uepi": None,
                }
            )
        self._init_irreps(irreps_in=irreps_in, irreps_out=outputs)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_batch(data)
        batch = data[AtomicDataDict.BATCH_KEY]
        if "f" in self.prefixes and "socf_sigma_s_raw" in data:
            for suffix in ("nu", "kappa", "sigma_s_raw", "sigma_v_raw"):
                data["f_" + suffix] = (
                    data["f_" + suffix] + self.soc_f_coeff * data["socf_" + suffix]
                )
        for prefix, head in self.sigma_heads.items():
            dimension = 2 if prefix == "sf" else 3
            nu = dimension + 2.0 + torch.nn.functional.softplus(data[prefix + "_nu"])
            kappa = torch.nn.functional.softplus(data[prefix + "_kappa"]) + 1e-6
            raw = head(data[prefix + "_sigma_s_raw"], data[prefix + "_sigma_v_raw"])
            log_sigma = (self.damper(raw) @ self.transform).reshape(-1, 3, 3)
            sigma = torch.linalg.matrix_exp(
                0.5 * (log_sigma + log_sigma.transpose(-1, -2))
            )
            valid = torch.ones_like(nu)
            if prefix == "sf":
                spin = data[AtomicDataDict.SPIN_KEY]
                norm = torch.linalg.vector_norm(spin, dim=-1, keepdim=True)
                unit = spin / norm.clamp_min(1e-6)
                projector = torch.eye(
                    3, dtype=spin.dtype, device=spin.device
                ) - unit.unsqueeze(-1) * unit.unsqueeze(-2)
                valid = (norm > 1e-6).to(sigma.dtype)
                sigma = (
                    (projector @ sigma @ projector)
                    * valid.unsqueeze(-1)
                    / self.spin_force_scale**2
                )
            sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
            uepi = (
                nu.unsqueeze(-1) * sigma / (kappa * (nu - dimension - 1)).unsqueeze(-1)
            )
            data[prefix + "_nu"], data[prefix + "_kappa"] = nu, kappa
            data[prefix + "_sigma"], data[prefix + "_Uepi"] = sigma, uepi
            data["total_" + prefix + "_Uepi"] = scatter(
                uepi, batch, dim=0, reduce="sum"
            )
            trace = uepi.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
            data[prefix + "_uncertainty"] = scatter(
                trace, batch, dim=0, reduce="sum"
            ) / scatter(valid, batch, dim=0, reduce="sum").clamp_min(1.0)
        return data
