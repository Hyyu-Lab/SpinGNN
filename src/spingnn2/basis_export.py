"""Resolve basis parameters and harmonic layouts from the eager modules."""

from allegro.nn._norm_basis import NormalizedBasis
from nequip.nn.cutoffs import PolynomialCutoff
from nequip.nn.embedding import (
    EdgeScalarAddSpin,
    RadialBasisEdgeEncoding,
    RadialBasisEdgeEncoding_spin,
    SphericalHarmonicEdgeAttrs,
    SphericalHarmonicEdgeAttrs_spin_nosoc,
    SphericalHarmonicEdgeAttrs_spin_soc_simple,
)
from nequip.nn.radial_basis import BesselBasis, FourierBasis

from .splines import calibrate_spline


def _array(tensor):
    return tensor.detach().cpu().numpy().copy()


def _cutoff(cutoff):
    if type(cutoff) is not PolynomialCutoff:
        raise ValueError(f"Unsupported cutoff module {type(cutoff).__name__}")
    return {"factor": float(cutoff._factor), "p": int(cutoff.p)}


def _radial_description(basis, cutoff):
    spec = {"kind": "bessel", "cutoff": _cutoff(cutoff)}
    if type(basis) is NormalizedBasis:
        spec["mean"] = _array(basis._mean)
        spec["inv_std"] = _array(basis._inv_std)
        basis = basis.basis
    if type(basis) is not BesselBasis:
        raise ValueError(f"Unsupported radial basis module {type(basis).__name__}")
    spec.update(
        weights=_array(basis.bessel_weights),
        radius=float(basis.r_max),
        prefactor=float(basis.prefactor),
    )
    spec["spline"] = calibrate_spline(
        spec, lower=0.0, upper=1.0 / spec["cutoff"]["factor"]
    )
    return spec


def export_radial(module):
    """Capture the possibly normalized Bessel basis and its radial envelope."""
    if type(module) is not RadialBasisEdgeEncoding:
        raise ValueError(f"Unsupported radial encoding module {type(module).__name__}")
    return _radial_description(module.basis, module.cutoff)


def export_spin_basis(module):
    """Capture the Fourier basis used by the spin edge encoding."""
    if type(module) not in (EdgeScalarAddSpin, RadialBasisEdgeEncoding_spin):
        raise ValueError(f"Unsupported spin basis module {type(module).__name__}")
    basis = module.basis_sisj
    if type(basis) is not FourierBasis:
        raise ValueError(f"Unsupported spin basis module {type(basis).__name__}")
    spec = {
        "kind": "fourier",
        "minimum": float(basis.r_min),
        "maximum": float(basis.r_max),
        "prefactor": float(basis.prefactor),
        "weights": _array(basis.fourier_weights),
        "cutoff": _cutoff(module.cutoff),
    }
    spec["spline"] = calibrate_spline(spec, spec["minimum"], spec["maximum"])
    if type(module) is RadialBasisEdgeEncoding_spin:
        spec["radial"] = _radial_description(module.basis, module.cutoff)
    return spec


def _harmonic_block(module, source):
    return {
        "source": source,
        "degrees": list(module.irreps_out.ls),
        "normalize": bool(module.normalize),
        "normalization": str(module.normalization),
    }


def export_harmonics(module):
    """Capture harmonic blocks and the T-e3nn component sorting permutation."""
    if type(module) is SphericalHarmonicEdgeAttrs:
        return {
            "blocks": [_harmonic_block(module.sh, "r")],
            "sort": list(range(module.sh.irreps_out.dim)),
        }
    blocks = []
    if type(module) is SphericalHarmonicEdgeAttrs_spin_soc_simple:
        blocks.append(_harmonic_block(module.sh_rij.sh, "r"))
    elif type(module) is not SphericalHarmonicEdgeAttrs_spin_nosoc:
        raise ValueError(
            f"Unsupported spherical harmonics module {type(module).__name__}"
        )
    blocks.extend(
        (_harmonic_block(module.sh_si, "j"), _harmonic_block(module.sh_si, "i"))
    )
    return {"blocks": blocks, "sort": list(module.sort_list)}
