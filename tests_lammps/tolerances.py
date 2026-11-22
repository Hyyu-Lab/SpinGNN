"""Single source of numerical thresholds for the LAMMPS test suite."""

# Production serial energy gate, in eV.
E_ATOL_EV = 2.0e-4

# Serial comparisons use an absolute error bound.
SERIAL_RTOL = 0.0

# Existing Kokkos/host gates, moved unchanged from test_kokkos_parity.py.
# SERIAL measured zero difference; CUDA must meet these same strict gates.
KOKKOS_RTOL = 1.0e-6
KOKKOS_ATOL = 1.0e-8

# Production serial force gate, in eV/Angstrom.
F_ATOL_EV_A = 1.0e-3

# Spin-force gate, compared in eV after multiplying LAMMPS fm by hbar.
SPIN_F_ATOL = 1.0e-3

# The original production BEC bound was 1e-4;
# CI showed a random-weight float32 model landing at 1.01e-4 on one of 1080
# elements (BLAS variation). 2e-4 keeps 100x discrimination against the known
# 1.7e-2 periodic-ghost BEC discrepancy seen with the earlier ghost-image handling.
BEC_ATOL_E = 2.0e-4

# Atomic virial uses the production force-scale absolute gate in eV.
VIRIAL_ATOL_EV = 1.0e-3

# Serial-vs-MPI-rank differences come from float32 reduction-order noise
# (different edge/ghost ordering per decomposition). CI measured max 1.2e-6
# absolute / 8.5e-5 relative on the 8-atom cells; gates sit ~10x above that
# noise floor and ~100x below the physics gates, so real ghost/ownership bugs
# (which show up at >=1e-2) still fail loudly.
MPI_RTOL = 1.0e-4

# Same-binary MPI energy comparisons: absolute floor in eV.
MPI_E_ATOL_EV = 1.0e-5

# Same-binary MPI force comparisons: absolute floor in eV/A.
MPI_F_ATOL_EV_A = 1.0e-5

# Same-binary MPI spin-force comparisons: absolute floor in eV.
MPI_SPIN_F_ATOL_EV = 1.0e-5

# Same-binary MPI BEC comparisons: absolute floor in electron units.
MPI_BEC_ATOL_E = 1.0e-5

# spin/sib should preserve normalized magnetic spin directions to this tolerance.
MD_SPIN_NORM_ATOL = 1.0e-6

# Spin magnitudes above this floor are treated as magnetic in MD assertions.
MAGNETIC_SPIN_MIN = 1.0e-12

# A deliberately broad random-model energy-drift gate for the short MD smoke test.
MD_ENERGY_DRIFT_ATOL_EV = 1.0e4

# The short spin/sib trajectory length.
MD_STEPS = 20

# Fixed subprocess ceiling prevents a broken pair run from hanging the suite.
LAMMPS_TIMEOUT_SECONDS = 180

# MPI startup probes should fail or succeed quickly on a local host.
MPI_PROBE_TIMEOUT_SECONDS = 30

# TorchScript versus PT2 must agree at float32 inference precision.
PT2_RTOL = 1.0e-6

# TorchScript versus PT2 near-zero absolute floor. Raised from 1e-7 after the
# H100 run measured 2.384e-7 eV on the nequip_spin run-0 energy:
# the two float32 compiled backends differ at that scale for some random-weight
# draws, so 1e-7 was inside the cross-backend envelope rather than above it.
# 5e-7 keeps ~2x headroom over that measurement while staying far below the
# production energy gate (E_ATOL_EV = 2e-4).
PT2_ATOL = 5.0e-7

# CPU AOTInductor and TorchScript use different fusion/reduction schedules. After
# factorizing the cutoff, the previous 1.793e-4 outlier fell to 5.72e-6. Across
# all four model families, the worst normalized error was 1.985e-6 (SOC spin
# force), with the absolute floor supplying additional low-scale margin. This
# CPU-only gate leaves the H100 strict gate above unchanged. The energy
# observable is evaluated separately with the scale-appropriate PT2_E_RTOL
# (see below), since a larger system's global scalar reduction carries a
# bigger cross-backend tail.
PT2_CPU_RTOL = 2.0e-6
PT2_CPU_ATOL = 2.0e-6

# LAMMPS exercises separate TorchScript and AOTI float32 compiled backends.
# The observed 7.9e-6 eV/A force envelope is backend rounding, not model drift;
# 5e-5 leaves over 6x margin while remaining 20x tighter than F_ATOL_EV_A.
PT2_LAMMPS_F_ATOL_EV_A = 5.0e-5

# Spin forces sit in the same cross-backend rounding class (local 1.14e-7 vs
# CI 1.22e-7 over the razor-thin 1e-7 floor); reuse the calibrated envelope.
PT2_LAMMPS_SPIN_F_ATOL_EV = 5.0e-5

# Python AOTInductor CUDA vs Python TorchScript, derivative observables. The
# energy keeps the strict PT2_ATOL absolute floor; the derivatives go through
# per-edge TP / scatter with device-dependent reduction order, so they show a
# larger float32 cross-backend tail. Calibration: the first GPU-acceptance
# canary (H100, 2026-09-05) measured force max_abs_error 7.748e-7 against the
# PT2_ATOL=5e-7 floor (limit 6.368e-7 at scale 0.1368), a ~1.2x overshoot and
# ~5.7e-6 relative -- backend rounding, not model drift, with spin_forces and
# energy well inside their gates. 5e-6 leaves >6x margin, far below the
# production F_ATOL_EV_A=1e-3. The 4090 canary passed even the 5e-7 floor, so
# these are device-noise bounds, not a loosened physics gate.
PT2_CUDA_F_ATOL_EV_A = 5.0e-6
PT2_CUDA_SPIN_F_ATOL_EV = 5.0e-6

# Cross-backend (AOTInductor/TorchScript) energy observable across every route:
# Python CUDA, Python CPU and LAMMPS PT2 all reduce a system's energy as a
# global scalar, and the float32 merging tail grows with system scale, so the
# strict absolute floor needs a scale-appropriate relative term. Calibration
# from the 3-line GPU-acceptance runs (2026-09-05): BEC (120 atoms / 10872
# edges, E=22.87 eV) measured energy max_abs_error 3.815e-5 (python PT2 CUDA,
# 1.67e-6 relative), 3.076e-5 (LAMMPS PT2 CUDA) and 4.959e-5 (python PT2 CPU)
# against the ~2.3e-5 strict limit -- a ~1.0-1.6x overshoot that is the same
# float32 merging class as the derivative gates, not model drift (the PT2 CUDA
# kernel reproduces the LAMMPS TorchScript energy to ~3e-15, and the LAMMPS
# physics gate passes at 3.7e-5 under E_ATOL_EV=2e-4). 5e-6 rtol leaves 2.3-3.7x
# margin at BEC scale while keeping the strict absolute floor for the small
# nosoc/soc cells, whose cross-backend energy error is ~0. It stays far below
# E_ATOL_EV=2e-4.
PT2_E_RTOL = 5.0e-6


# Order-invariance bounds for the hybrid overlay: the same model evaluated in
# two sub-style orders must agree far below any physics gate.
#
# cpu: torch inference is deterministic, so the two runs are bitwise-equal up
# to double-accumulation noise; 1e-10 holds.
# cuda: float32 scatter/reduction kernels use atomics whose ordering differs
# between separate LAMMPS processes, so even IDENTICAL orderings differ at the
# ~1e-8 level (measured on H100 2026-08-31: energy delta 1.49e-8 eV on a
# -0.3148 eV total, forces delta ~1e-8 eV/A). 5e-7 gives ~30x headroom over
# that noise while staying ~400x tighter than the calibrated production energy
# gate, so a real order-dependence bug still fails.
HYBRID_ORDER_ATOL_BY_DEVICE = {"cpu": 1.0e-10, "cuda": 5.0e-7}
