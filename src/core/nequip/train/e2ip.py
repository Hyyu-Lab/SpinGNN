"""Multivariate evidential Student-t loss (3D forces or 2D tangent spins)."""
import math
import torch
from nequip.data import AtomicDataDict
from .loss import Loss


def tangent_basis(spin):
    unit = spin / torch.linalg.vector_norm(spin, dim=-1, keepdim=True).clamp_min(1e-6)
    z = torch.zeros_like(unit)
    z[:, 2] = 1.0
    x = torch.zeros_like(unit)
    x[:, 0] = 1.0
    ref = torch.where((unit[:, 2:3].abs() > 0.9), x, z)
    first = torch.nn.functional.normalize(torch.linalg.cross(unit, ref), dim=-1)
    return torch.stack((first, torch.linalg.cross(unit, first)), dim=-1)


class EvidentialLoss:
    """One loss per valid atom, with an additive reduction for DDP.

    sigma is always in the same squared units as the target. spin_force_scale
    sets the legacy residual units. likelihood_scale changes arithmetic units
    with compensation to preserve the original objective and regularizer.
    likelihood_dtype can promote only this calculation to float64.
    """

    def __init__(
        self,
        reg=0.1,
        epsilon=1e-6,
        spin_force_scale=40.0,
        likelihood_scale=1.0,
        likelihood_dtype="float32",
    ):
        if not math.isfinite(reg) or reg < 0:
            raise ValueError("e2IP reg must be finite and nonnegative")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("e2IP epsilon must be finite and positive")
        if not math.isfinite(spin_force_scale) or spin_force_scale <= 0:
            raise ValueError("e2IP spin_force_scale must be finite and positive")
        self.reg = reg
        self.epsilon = epsilon
        self.spin_force_scale = spin_force_scale
        if not math.isfinite(likelihood_scale) or likelihood_scale <= 0:
            raise ValueError("e2IP likelihood_scale must be finite and positive")
        if likelihood_dtype not in ("float32", "float64"):
            raise ValueError("e2IP likelihood_dtype must be float32 or float64")
        self.likelihood_scale = float(likelihood_scale)
        self.likelihood_dtype = likelihood_dtype

    def components(self, pred, ref, key):
        is_spin = key == AtomicDataDict.SPIN_FORCE_VERT_KEY
        if key not in (AtomicDataDict.FORCE_KEY, AtomicDataDict.SPIN_FORCE_VERT_KEY):
            raise ValueError("e2IP supports only forces and spin_forces_vert")
        prefix, dimension = ("sf", 2) if is_spin else ("f", 3)
        # Promote before subtraction and tangent projection. Keep autograd through
        # the casts; the much larger energy model can remain in float32.
        dtype = torch.float64 if self.likelihood_dtype == "float64" else pred[key].dtype
        delta = pred[key].to(dtype) - ref[key].to(dtype)
        sigma = pred[prefix + "_sigma"].to(dtype)
        nu = pred[prefix + "_nu"].reshape(-1).to(dtype)
        kappa = pred[prefix + "_kappa"].reshape(-1).to(dtype)
        if is_spin:
            spin = ref[AtomicDataDict.SPIN_KEY].to(dtype)
            mask = torch.linalg.vector_norm(spin, dim=-1) > 1e-6
            # Select before Cholesky: nonmagnetic atoms have exactly zero covariance.
            basis = tangent_basis(spin[mask])
            delta = (
                torch.einsum("ni,nia->na", delta[mask], basis) * self.spin_force_scale
            )
            sigma = (
                torch.einsum("nia,nij,njb->nab", basis, sigma[mask], basis)
                * self.spin_force_scale**2
            )
            nu, kappa = nu[mask], kappa[mask]
        # This numerical change of units must not reweight the regularizer,
        # jitter, or NLL. Its log-Jacobian is removed below.
        regularizer = (nu + kappa) * torch.linalg.vector_norm(delta, dim=-1)
        delta = delta * self.likelihood_scale
        sigma = sigma * self.likelihood_scale**2
        eye = torch.eye(dimension, device=sigma.device, dtype=sigma.dtype)
        chol = torch.linalg.cholesky(
            0.5 * (sigma + sigma.transpose(-1, -2))
            + (self.epsilon * self.likelihood_scale**2) * eye
        )
        logdet = 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(-1)
        solved = torch.cholesky_solve(delta.unsqueeze(-1), chol)
        maha = (delta.unsqueeze(-2) @ solved).reshape(-1).clamp_min(0.0)
        alpha = kappa / ((1.0 + kappa) * nu)
        nll = (
            torch.lgamma((nu - dimension + 1.0) / 2.0)
            - torch.lgamma((nu + 1.0) / 2.0)
            + dimension / 2.0 * torch.log(math.pi * nu * (1.0 + kappa) / kappa)
            + 0.5 * logdet
            + (nu + 1.0) / 2.0 * torch.log1p(alpha * maha)
            - dimension * math.log(self.likelihood_scale)
        )
        return nll, regularizer

    def __call__(self, pred, ref, key, mean=True):
        nll, regularizer = self.components(pred, ref, key)
        values = nll + self.reg * regularizer
        return values.sum() / max(values.numel(), 1) if mean else values

    def sum_and_count(self, pred, ref, key):
        values = self(pred, ref, key, mean=False)
        return values.sum(), values.new_tensor(values.numel()).detach()


class e2IPLoss(Loss):
    """Keep current loss parsing; replace only explicitly selected targets.

    loss_coeffs[key] multiplies (NLL + loss_reg * regularizer), preserving
    the legacy convention. For independent paper weights a and b, use
    loss_coeffs[key]=a and loss_reg=b/a.
    """

    def __init__(
        self,
        coeffs,
        coeff_schedule="constant",
        ldetach_method=False,
        lauto_loss=False,
        evidential_keys=("forces",),
        reg=0.1,
        epsilon=1e-6,
        spin_force_scale=40.0,
        likelihood_scale=1.0,
        likelihood_dtype="float32",
    ):
        if ldetach_method or lauto_loss:
            raise ValueError("e2IP requires fixed explicit loss weights")
        super().__init__(coeffs, coeff_schedule, ldetach_method, lauto_loss)
        if not evidential_keys or any(k not in self.keys for k in evidential_keys):
            raise ValueError("evidential_key must be present in loss_coeffs")
        for key in evidential_keys:
            if key not in ("forces", "spin_forces_vert"):
                raise ValueError("unsupported evidential_key: " + key)
            self.funcs[key] = EvidentialLoss(
                reg, epsilon, spin_force_scale, likelihood_scale, likelihood_dtype
            )
