"""First-party Geometric-Probe Spin Temperature (GPST) toolkit for SpinGNN2.

Implements the exact sphere integration-by-parts identity on (S^2)^{N_mag}:

    k_B * T_GP = <N_Phi> / <D_Phi>
    N_Phi = sum_i grad_{S^2,i}(Phi) . f_i^perp
    D_Phi = -sum_i Delta_{S^2,i}(Phi)

Valid for arbitrary smooth magnetic Hamiltonians (including nonlinear SIA,
biquadratic exchange, DMI, and general many-body ML spin potentials) using only
unit spin directions |s_i| = 1 and conservative spin forces f_i = -dH/ds_i =
hbar * fm_i (unscaled by magnetic moment magnitude), excluding nonmagnetic atoms.
"""

from .api import STATUS_CODES, analyze, pooled_ratio
from .basis import evaluate_basis, learn_basis, learned_names, upper_bounds
from .core import (
    CHANNELS,
    FrameError,
    FrameInput,
    ORDERS,
    build_graph,
    compute_moments,
    from_lammps_fm,
    load_frame_npz,
    validate_frame,
)
from .dump_adapter import FIELDS as DUMP_FIELDS, read_dump
from .geometry import FIXED, legendre, minimum_image, moment_arrays
from .inference import (
    contains,
    covariance_jackknife,
    crossfit,
    fieller,
    geyer,
    pooled_pair,
    summary,
)
from .native import has_native_kernel

__version__ = "2.0.0"

__all__ = [
    "CHANNELS",
    "DUMP_FIELDS",
    "FIXED",
    "ORDERS",
    "STATUS_CODES",
    "FrameError",
    "FrameInput",
    "analyze",
    "build_graph",
    "compute_moments",
    "contains",
    "covariance_jackknife",
    "crossfit",
    "evaluate_basis",
    "fieller",
    "from_lammps_fm",
    "geyer",
    "has_native_kernel",
    "learn_basis",
    "learned_names",
    "legendre",
    "load_frame_npz",
    "minimum_image",
    "moment_arrays",
    "pooled_pair",
    "pooled_ratio",
    "read_dump",
    "summary",
    "upper_bounds",
    "validate_frame",
]
