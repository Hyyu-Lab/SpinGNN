"""Pure-PyTorch magnetic symmetry primitives for SpinGNN2.

Zero ``e3nn`` runtime dependency, zero Triton dependency, ``torch.compile(fullgraph=True)`` ready.
Enforces magnetic symmetry rules R1-R5 across both coupled :math:`O(3) \\times \\mathbb{Z}_2^T`
and decoupled :math:`O(3)_{\\mathrm{lat}} \\times O(3)_{\\mathrm{spin}}` representations.
"""

from .contractions import (
    ContractionPath,
    SparseBispectrumContract,
    SparseCGContract,
    SparseCGSpec,
    StaticTensorProduct,
    build_sparse_cg,
    validate_contraction_paths,
)
from .couple import Couple, CoupleBlock
from .harmonics import (
    BipolarSphericalHarmonics,
    SphericalHarmonics,
    bipolar_spherical_harmonics,
    bipolar_spin_pair_j1_j2,
    bond_spin,
    spherical_harmonics,
    spin_cross_axial,
    spin_quadrupole_symmetric,
    spin_spin,
)
from .irreps import (
    DIrrep,
    DIrreps,
    DMulIr,
    Irrep,
    Irreps,
    MulIr,
    decoupled_body_key,
    decoupled_clebsch_gordan,
    o3_body_key,
    symmetry_spec_hash,
)
from .layers import (
    DecoupledGate,
    EquivariantGate,
    EquivariantLinear,
    EquivariantRMSNorm,
    probe_activation_parity,
    validate_invariant_scalar,
    validate_scalar_activation,
)
from .spin import ParamagneticCoordination, SpinScale, paramagnetic_coordination
from .verify import (
    check_coupled_equivariance,
    check_decoupled_equivariance,
    check_scalar_invariance,
)
from .wigner import (
    cartesian_tensor_basis,
    cartesian_to_irreps,
    change_basis_real_to_complex,
    irreps_to_cartesian,
    proper_wigner_D,
    proper_wigner_Ds,
    random_rotation_matrix,
    wigner_3j,
)

__all__ = [
    # Wigner & Cartesian basis
    "change_basis_real_to_complex",
    "wigner_3j",
    "proper_wigner_D",
    "proper_wigner_Ds",
    "random_rotation_matrix",
    "cartesian_tensor_basis",
    "cartesian_to_irreps",
    "irreps_to_cartesian",
    # Coupled & decoupled irrep algebras
    "Irrep",
    "MulIr",
    "Irreps",
    "DIrrep",
    "DMulIr",
    "DIrreps",
    "decoupled_clebsch_gordan",
    "decoupled_body_key",
    "o3_body_key",
    "symmetry_spec_hash",
    # Orthogonal L-S coupling
    "Couple",
    "CoupleBlock",
    # Spherical & bipolar spherical harmonics
    "spherical_harmonics",
    "SphericalHarmonics",
    "BipolarSphericalHarmonics",
    "bipolar_spherical_harmonics",
    "bond_spin",
    "spin_spin",
    "spin_cross_axial",
    "spin_quadrupole_symmetric",
    "bipolar_spin_pair_j1_j2",
    # Spin conditioning & paramagnetic coordination
    "SpinScale",
    "paramagnetic_coordination",
    "ParamagneticCoordination",
    # Static contractions (R1)
    "ContractionPath",
    "SparseCGSpec",
    "build_sparse_cg",
    "validate_contraction_paths",
    "SparseCGContract",
    "SparseBispectrumContract",
    "StaticTensorProduct",
    # Equivariant layers (R2-R3)
    "probe_activation_parity",
    "validate_scalar_activation",
    "validate_invariant_scalar",
    "EquivariantLinear",
    "EquivariantRMSNorm",
    "DecoupledGate",
    "EquivariantGate",
    # Verification helpers (R5)
    "check_decoupled_equivariance",
    "check_coupled_equivariance",
    "check_scalar_invariance",
]
