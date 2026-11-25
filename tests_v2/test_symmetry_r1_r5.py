"""Comprehensive R1-R5 positive & negative symmetry tests for ``spingnn2.symmetry``.

Covers:
- Zero ``e3nn`` runtime dependency verification;
- Exact recursive ``proper_wigner_D`` properties & FP64 precision (< 1e-13);
- ``Irrep``/``Irreps`` and ``DIrrep``/``DIrreps`` algebra & R4 canonical spec/hash;
- Parameter-free orthogonal ``Couple`` & ``CoupleBlock`` (orthogonality, SOC equivariance,
  and negative test showing intentional non-equivariance under independent spin rotation);
- Singularity-free polynomial ``SphericalHarmonics`` (l=0..4) & ``BipolarSphericalHarmonics``
  (including collinear theta -> 0 smoothness and axial T-even s_i x s_j / symmetric quadrupole);
- ``SpinScale`` (exact zero 1st & 2nd derivatives for nonmagnetic atoms) & paramagnetic ``Mw``;
- Static sparse CG / ``StaticTensorProduct`` (R1 positive & negative tests);
- ``EquivariantLinear`` / ``EquivariantRMSNorm`` / ``DecoupledGate`` / ``EquivariantGate``
  (R2-R3 positive & negative tests: feeding non-0e|0e into gate, wrong parity activation,
  mixing distinct (p, t) channels, non-invariant bias);
- Hybrid NoSOC backbone + zero-init thin SOC head (exact 0.0 diff on spin-only rotation + nonzero grad);
- ``torch.compile(fullgraph=True, dynamic=True)`` zero-graph-break verification across all modules.
"""

import math
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spingnn2.symmetry import (
    BipolarSphericalHarmonics,
    Couple,
    CoupleBlock,
    DIrrep,
    DIrreps,
    DecoupledGate,
    EquivariantGate,
    EquivariantLinear,
    EquivariantRMSNorm,
    Irrep,
    Irreps,
    ParamagneticCoordination,
    SparseBispectrumContract,
    SparseCGContract,
    SphericalHarmonics,
    SpinScale,
    StaticTensorProduct,
    bipolar_spherical_harmonics,
    bipolar_spin_pair_j1_j2,
    bond_spin,
    build_sparse_cg,
    cartesian_tensor_basis,
    cartesian_to_irreps,
    check_coupled_equivariance,
    check_decoupled_equivariance,
    check_scalar_invariance,
    decoupled_clebsch_gordan,
    irreps_to_cartesian,
    paramagnetic_coordination,
    probe_activation_parity,
    proper_wigner_D,
    proper_wigner_Ds,
    random_rotation_matrix,
    spherical_harmonics,
    spin_cross_axial,
    spin_quadrupole_symmetric,
    spin_spin,
    symmetry_spec_hash,
    validate_contraction_paths,
    validate_invariant_scalar,
    validate_scalar_activation,
    wigner_3j,
)

FP64_TOL = 1e-13


def test_zero_e3nn_runtime_dependency():
    """spingnn2.symmetry and spingnn2.v2 must never import e3nn in a clean runtime."""
    import os
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, spingnn2.symmetry, spingnn2.v2; "
                "assert 'e3nn' not in sys.modules, 'e3nn unexpectedly imported'"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_wigner_3j_and_recursive_wigner_D_fp64():
    """Verify exact orthogonality, homomorphism, and 3j invariance to < 1e-13 in FP64."""
    gen = torch.Generator(device="cpu").manual_seed(42)
    R1 = random_rotation_matrix(dtype=torch.float64, generator=gen)
    R2 = random_rotation_matrix(dtype=torch.float64, generator=gen)
    R12 = R1 @ R2

    Ds1 = proper_wigner_Ds(4, R1)
    Ds2 = proper_wigner_Ds(4, R2)
    Ds12 = proper_wigner_Ds(4, R12)
    assert torch.max(torch.abs(Ds1[1] - R1)).item() < FP64_TOL

    for l in range(5):
        D1, D2, D12 = Ds1[l], Ds2[l], Ds12[l]
        eye = torch.eye(2 * l + 1, dtype=torch.float64)
        assert torch.max(torch.abs(D1.T @ D1 - eye)).item() < FP64_TOL
        assert torch.max(torch.abs(D1 @ D2 - D12)).item() < FP64_TOL
        assert torch.max(torch.abs(proper_wigner_D(l, R1) - D1)).item() < FP64_TOL

    # Small rotation angle (where matrix_exp loses accuracy)
    eps = 1e-7
    c, s = math.cos(eps), math.sin(eps)
    R_small = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    D_small = proper_wigner_D(4, R_small)
    assert (
        torch.max(
            torch.abs(D_small.T @ D_small - torch.eye(9, dtype=torch.float64))
        ).item()
        < FP64_TOL
    )

    # Wigner 3j equivariance and Frobenius norm = 1
    for l1 in range(4):
        for l2 in range(4):
            for l3 in range(abs(l1 - l2), min(l1 + l2, 4) + 1):
                C = wigner_3j(l1, l2, l3, dtype=torch.float64)
                assert abs(torch.linalg.norm(C).item() - 1.0) < FP64_TOL
                C_rot = torch.einsum("ijk,il,jm,kn->lmn", C, Ds1[l1], Ds1[l2], Ds1[l3])
                assert torch.max(torch.abs(C_rot - C)).item() < FP64_TOL

    # Negative check: improper rotation rejected by proper_wigner_Ds
    with pytest.raises(ValueError, match="proper rotations"):
        proper_wigner_Ds(2, -R1)


def test_cartesian_tensor_basis_equivariance():
    """Verify 3x3 Cartesian <-> 0e + 1e + 2e orthogonal basis for DREAM BEC."""
    Q = cartesian_tensor_basis(dtype=torch.float64)
    assert (
        torch.max(torch.abs(Q.T @ Q - torch.eye(9, dtype=torch.float64))).item()
        < FP64_TOL
    )

    gen = torch.Generator(device="cpu").manual_seed(7)
    R = random_rotation_matrix(dtype=torch.float64, generator=gen)
    T = torch.randn((5, 3, 3), dtype=torch.float64, generator=gen)

    # Roundtrip
    ir_vec = cartesian_to_irreps(T)
    T_rec = irreps_to_cartesian(ir_vec)
    assert torch.max(torch.abs(T_rec - T)).item() < FP64_TOL

    # Equivariance: T -> R T R^T corresponds to D^(0ee+1ee+2ee)(R) on irreps
    T_rot = torch.einsum("ij,bjk,lk->bil", R, T, R)
    D = Irreps("1x0ee + 1x1ee + 1x2ee").D_from_matrix(R, k_inv=0, k_t=0)
    assert (
        torch.max(torch.abs(cartesian_to_irreps(T_rot) - ir_vec @ D.T)).item()
        < FP64_TOL
    )


def test_irreps_and_dirreps_r4_canonical_hash():
    """R4: canonical spec/hash must preserve both p and t and distinguish groups."""
    # Coupled Irreps: differing only in p or t must produce distinct hashes
    h_0ee = Irreps("4x0ee").canonical_hash()
    h_0eo = Irreps("4x0eo").canonical_hash()
    h_0oe = Irreps("4x0oe").canonical_hash()
    h_0oo = Irreps("4x0oo").canonical_hash()
    assert len({h_0ee, h_0eo, h_0oe, h_0oo}) == 4

    # Decoupled DIrreps: differing only in p or t must produce distinct hashes
    dh_ee = DIrreps("4x1e|1e").canonical_hash()
    dh_eo = DIrreps("4x1e|1o").canonical_hash()
    dh_oe = DIrreps("4x1o|1e").canonical_hash()
    dh_oo = DIrreps("4x1o|1o").canonical_hash()
    assert len({dh_ee, dh_eo, dh_oe, dh_oo}) == 4

    # Coupled vs Decoupled scalar hashes must also differ (group identity preserved)
    assert Irreps("4x0ee").canonical_hash() != DIrreps("4x0e|0e").canonical_hash()

    # R2: simplify must NEVER merge blocks with different p or t
    mixed_c = Irreps("2x0ee + 2x0eo + 2x0ee").simplify()
    assert repr(mixed_c) == "2x0ee+2x0eo+2x0ee"
    mixed_d = DIrreps("2x0e|0e + 2x0e|0o + 2x0o|0e").simplify()
    assert repr(mixed_d) == "2x0e|0e+2x0e|0o+2x0o|0e"

    # Sort order puts 0e|0e first
    s = DIrreps("1x1o|1o + 1x0o|0o + 1x0e|0o + 1x0e|0e").sort().irreps
    assert s[0].ir == DIrrep("0e|0e")

    # symmetry_spec_hash distinguishes p/t metadata
    assert symmetry_spec_hash(Irrep("1oe")) != symmetry_spec_hash(Irrep("1eo"))


def test_couple_orthogonality_and_soc_vs_nosoc_symmetry():
    """Couple is orthogonal, SOC-equivariant (< 1e-13), and NOT spin-rotation invariant."""
    irreps_in = DIrreps("3x0e|0e + 2x1o|1o + 2x2e|1o + 1x2e|2e")
    couple = Couple(irreps_in)
    assert not couple.truncated
    # Check (p, t) preservation (R2)
    for mul, ir in couple.irreps_out:
        assert ir.p in (-1, 1) and ir.t in (-1, 1)
    assert repr(Couple("2x1o|1o + 1x0e|1o").irreps_out) == "2x0oo+2x1oo+2x2oo+1x1eo"

    gen = torch.Generator(device="cpu").manual_seed(11)
    x = torch.randn((6, irreps_in.dim), dtype=torch.float64, generator=gen)
    y = couple(x)
    # Orthogonal norm preservation & exact inverse
    assert torch.max(torch.abs(x.norm(dim=-1) - y.norm(dim=-1))).item() < FP64_TOL
    assert torch.max(torch.abs(couple.inverse(y) - x)).item() < FP64_TOL

    # Positive test: exact equivariance under SOC subgroup (R_lat = R_spin = R, P, T)
    R = random_rotation_matrix(dtype=torch.float64, generator=gen)
    for k_inv, k_t in ((0, 0), (1, 0), (0, 1), (1, 1)):
        D_in = irreps_in.D_from_matrices(R, R, k_inv=k_inv, k_t=k_t)
        D_out = couple.irreps_out.D_from_matrix(R, k_inv=k_inv, k_t=k_t)
        err = torch.max(torch.abs(couple(x @ D_in.T) - y @ D_out.T)).item()
        assert err < FP64_TOL

    # Negative test: independent spin rotation (fixed lattice) MUST NOT be equivariant
    I3 = torch.eye(3, dtype=torch.float64)
    R_spin = random_rotation_matrix(dtype=torch.float64, generator=gen)
    D_in_spin = irreps_in.D_from_matrices(I3, R_spin, k_inv=0, k_t=0)
    D_out_id = couple.irreps_out.D_from_matrix(I3, k_inv=0, k_t=0)
    err_indep = torch.max(torch.abs(couple(x @ D_in_spin.T) - y @ D_out_id.T)).item()
    assert err_indep > 0.1

    # Truncated Couple(jmax=2) rejects inverse
    c_trunc = Couple("2x2e|2e", jmax=2)
    assert c_trunc.truncated
    with pytest.raises(RuntimeError, match="inverse"):
        c_trunc.inverse(torch.zeros((2, c_trunc.irreps_out.dim), dtype=torch.float64))

    # Structured CoupleBlock matches flat Couple
    cb = CoupleBlock(DIrrep("1o|1o"), jmax=2)
    blk = torch.randn((4, 3, 3, 5), dtype=torch.float64, generator=gen)
    out_dict = cb(blk)
    assert set(out_dict.keys()) == {0, 1, 2}
    # Verify equivariance of CoupleBlock under SOC (R, P, T)
    D_lat = proper_wigner_D(1, R)
    D_spin = proper_wigner_D(1, R)
    blk_rot = torch.einsum("mi,nj,bijc->bmnc", D_lat, D_spin, blk)
    out_rot = cb(blk_rot)
    for J in (0, 1, 2):
        DJ = proper_wigner_D(J, R)
        expected = torch.einsum("Mm,bmc->bMc", DJ, out_dict[J])
        assert torch.max(torch.abs(out_rot[J] - expected)).item() < FP64_TOL


def test_spherical_and_bipolar_harmonics_equivariance_and_collinear_smoothness():
    """Verify Y^l (l=0..4), B^J_{l,lam}, axial cross, quadrupole, and collinear smoothness."""
    gen = torch.Generator(device="cpu").manual_seed(19)
    v = torch.randn((8, 3), dtype=torch.float64, generator=gen)
    R = random_rotation_matrix(dtype=torch.float64, generator=gen)

    # 1. SphericalHarmonics l = 0..4 exact O(3) x Z2^T equivariance
    for parity, tr, norm_flag in ((-1, 1, True), (1, -1, False)):
        sh = SphericalHarmonics(
            4,
            normalize=norm_flag,
            normalization="component",
            parity=parity,
            time_reversal=tr,
        )
        y0 = sh(v)
        for k_inv, k_t in ((0, 0), (1, 0), (0, 1), (1, 1)):
            v_tr = (v @ R.T) * (parity**k_inv) * (tr**k_t)
            D = sh.irreps_out.D_from_matrix(R, k_inv=k_inv, k_t=k_t)
            assert torch.max(torch.abs(sh(v_tr) - y0 @ D.T)).item() < FP64_TOL

        # Functional matches module
        assert (
            torch.max(
                torch.abs(
                    spherical_harmonics(list(range(5)), v, normalize=norm_flag) - y0
                )
            ).item()
            < FP64_TOL
        )

    # 2. BipolarSphericalHarmonics: bond_spin, spin_spin, and general bipolar_spherical_harmonics
    r_vec = torch.randn((6, 3), dtype=torch.float64, generator=gen)
    si = torch.randn((6, 3), dtype=torch.float64, generator=gen)
    sj = torch.randn((6, 3), dtype=torch.float64, generator=gen)

    ir_gen, y_gen = bipolar_spherical_harmonics(
        1,
        1,
        r_vec,
        si,
        parity_a=-1,
        parity_b=1,
        time_reversal_a=1,
        time_reversal_b=-1,
        jmax=2,
    )
    assert repr(ir_gen) == "1x0ee+1x1eo+1x1oe+1x0oo+1x1oo+1x2oo"
    assert y_gen.shape == (6, ir_gen.dim)

    b_bs = BipolarSphericalHarmonics.for_bond_spin(
        lmax_r=2, lmax_s=2, jmax=2, spin_scale=0.25
    )
    y_bs = b_bs(r_vec, si)
    ir_bs_fn, y_bs_fn = bond_spin(
        r_vec, si, lmax_r=2, lmax_s=2, jmax=2, spin_scale=0.25
    )
    assert ir_bs_fn == b_bs.irreps_out
    assert torch.max(torch.abs(y_bs - y_bs_fn)).item() < FP64_TOL
    for k_inv, k_t in ((0, 0), (1, 0), (0, 1), (1, 1)):
        r_tr = (r_vec @ R.T) * ((-1) ** k_inv)
        s_tr = (si @ R.T) * ((-1) ** k_t)
        D = b_bs.irreps_out.D_from_matrix(R, k_inv=k_inv, k_t=k_t)
        assert torch.max(torch.abs(b_bs(r_tr, s_tr) - y_bs @ D.T)).item() < FP64_TOL

    b_ss = BipolarSphericalHarmonics.for_spin_spin(lmax=2, jmax=2, spin_scale=0.2)
    y_ss = b_ss(si, sj)
    ir_ss_fn, y_ss_fn = spin_spin(si, sj, lmax=2, jmax=2, spin_scale=0.2)
    assert ir_ss_fn == b_ss.irreps_out
    assert torch.max(torch.abs(y_ss - y_ss_fn)).item() < FP64_TOL
    for k_inv, k_t in ((0, 0), (1, 0), (0, 1), (1, 1)):
        si_tr = (si @ R.T) * ((-1) ** k_t)
        sj_tr = (sj @ R.T) * ((-1) ** k_t)
        D = b_ss.irreps_out.D_from_matrix(R, k_inv=k_inv, k_t=k_t)
        assert torch.max(torch.abs(b_ss(si_tr, sj_tr) - y_ss @ D.T)).item() < FP64_TOL

    # Verify B^1_{11}(si, sj) is axial T-even (1ee) proportional to si x sj,
    # and B^2_{11}(si, sj) is symmetric traceless quadrupole (2ee)
    b11 = BipolarSphericalHarmonics.for_spin_spin(lmax=1, jmax=2)
    y11 = b11(si, sj)
    sl_j1 = b11.get_block_slice(1, 1, 1)
    sl_j2 = b11.get_block_slice(1, 1, 2)
    cross_ref = spin_cross_axial(si, sj)
    quad_ref = spin_quadrupole_symmetric(si, sj)
    # Exact collinearity (proportional up to nonzero constant)
    ratio_j1 = y11[:, sl_j1] / cross_ref
    assert torch.max(torch.abs(ratio_j1 - ratio_j1[0, 0])).item() < FP64_TOL
    ratio_j2 = y11[:, sl_j2] / quad_ref
    assert torch.max(torch.abs(ratio_j2 - ratio_j2[0, 0])).item() < FP64_TOL

    # 3. Collinear theta -> 0 smoothness (NO 1/theta singularity!)
    grad_norms = []
    for theta in (1e-2, 1e-4, 1e-6, 0.0):
        s_a = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        s_b = torch.tensor(
            [[math.sin(theta), 0.0, math.cos(theta)]],
            dtype=torch.float64,
            requires_grad=True,
        )
        out = b11(s_a, s_b)[:, sl_j1]
        jac = torch.autograd.functional.jacobian(lambda sb: b11(s_a, sb)[0, sl_j1], s_b)
        gnorm = float(torch.linalg.norm(jac).item())
        assert math.isfinite(gnorm)
        grad_norms.append(gnorm)
        if theta == 0.0:
            assert torch.max(torch.abs(out)).item() == 0.0
    # Derivative stays O(1) constant across theta in [0, 1e-2], never diverging like 1/theta!
    assert max(grad_norms) / min(grad_norms) < 1.05


def test_spin_scale_zero_derivatives_and_paramagnetic_mw():
    """Verify SpinScale exact zero 1st & 2nd derivatives for nonmagnetic atoms and Mw behavior."""
    # Type 0: nonmagnetic (O), Type 1: magnetic (Fe, s_ref=4.0), Type 2: magnetic (Mn, s_ref=5.0)
    sc = SpinScale(
        num_types=3, use_spin=[False, True, True], spin_reference=[1.0, 4.0, 5.0]
    )
    atom_types = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    spins = torch.tensor(
        [[1.2, -0.5, 0.7], [3.0, 2.0, 1.0], [-2.0, 4.0, 0.5], [0.9, 0.9, -0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )

    s_hat = sc(spins, atom_types)
    assert torch.all(s_hat[0] == 0.0) and torch.all(s_hat[3] == 0.0)
    assert torch.allclose(s_hat[1], spins[1] / 4.0, atol=FP64_TOL)
    assert torch.allclose(s_hat[2], spins[2] / 5.0, atol=FP64_TOL)

    # First AND second derivatives of a nonlinear polynomial energy w.r.t. nonmagnetic spins are identically 0
    energy = (s_hat.pow(2).sum(dim=-1) + 0.5 * s_hat.pow(4).sum(dim=-1)).sum()
    (grad1,) = torch.autograd.grad(energy, spins, create_graph=True)
    assert torch.all(grad1[0] == 0.0) and torch.all(grad1[3] == 0.0)
    assert torch.linalg.norm(grad1[1]).item() > 0.1

    loss2 = grad1.pow(2).sum()
    (grad2,) = torch.autograd.grad(loss2, spins)
    assert torch.all(grad2[0] == 0.0) and torch.all(grad2[3] == 0.0)

    # Paramagnetic coordination Mw: nonzero at s=0 when magnetic neighbors exist, zero when only nonmagnetic neighbors exist
    edge_src = torch.tensor([1, 2, 0, 3], dtype=torch.long)
    edge_dst = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    edge_w = torch.tensor(
        [[0.8, 0.4], [0.6, 0.2], [0.5, 0.5], [0.9, 0.1]], dtype=torch.float64
    )
    mw_mod = ParamagneticCoordination(sc)
    mw = mw_mod(edge_w, edge_src, edge_dst, atom_types, num_nodes=4)
    # Node 0 has magnetic neighbors 1 & 2 -> Mw > 0 even if spins == 0
    assert torch.allclose(mw[0], edge_w[0] + edge_w[1], atol=FP64_TOL)
    # Node 1 has only nonmagnetic neighbors 0 & 3 -> Mw == 0
    assert torch.all(mw[1] == 0.0)
    # Functional matches module
    mw_fn = paramagnetic_coordination(
        edge_w, edge_src, edge_dst, atom_types, sc.mask_mag, num_nodes=4
    )
    assert torch.max(torch.abs(mw - mw_fn)).item() < FP64_TOL

    # Calibration updates magnetic species only, preserving nonmagnetic reference at 1.0
    cal = SpinScale(num_types=3, use_spin=[False, True, True])
    ref = cal.calibrate_reference(spins.detach(), atom_types)
    assert ref[0].item() == 1.0
    assert abs(ref[1].item() - torch.linalg.norm(spins[1]).item()) < 1e-6
    assert abs(ref[2].item() - torch.linalg.norm(spins[2]).item()) < 1e-6
    # Converting to float64 re-materializes exact 53-bit calibrated reference
    cal_f64 = cal.double()
    assert (
        abs(cal_f64.spin_reference[1].item() - torch.linalg.norm(spins[1]).item())
        < FP64_TOL
    )
    assert (
        abs(cal_f64.spin_reference[2].item() - torch.linalg.norm(spins[2]).item())
        < FP64_TOL
    )


def test_r1_static_contractions_positive_and_negative():
    """R1: StaticTensorProduct & build_sparse_cg pass exact symmetry and reject forbidden paths."""
    # Positive decoupled test (independent R_lat, R_spin, P, T)
    ir1 = DIrreps("2x0e|0e + 2x1o|1o + 1x0e|1o")
    ir2 = DIrreps("1x0e|0e + 1x1o|0e + 1x1o|1o")
    ir_out = DIrreps("2x0e|0e + 2x1o|1o + 1x2e|0e + 1x1o|0e")
    tp_dec = StaticTensorProduct(ir1, ir2, ir_out).to(dtype=torch.float64)

    gen = torch.Generator(device="cpu").manual_seed(23)
    x1 = torch.randn((5, ir1.dim), dtype=torch.float64, generator=gen)
    x2 = torch.randn((5, ir2.dim), dtype=torch.float64, generator=gen)
    y0 = tp_dec(x1, x2)
    R_lat = random_rotation_matrix(dtype=torch.float64, generator=gen)
    R_spin = random_rotation_matrix(dtype=torch.float64, generator=gen)
    I3 = torch.eye(3, dtype=torch.float64)
    for rl, rs, kp, kt in (
        (I3, R_spin, 0, 0),
        (R_lat, I3, 0, 0),
        (R_lat, R_spin, 1, 1),
    ):
        D1 = ir1.D_from_matrices(rl, rs, k_inv=kp, k_t=kt)
        D2 = ir2.D_from_matrices(rl, rs, k_inv=kp, k_t=kt)
        Do = ir_out.D_from_matrices(rl, rs, k_inv=kp, k_t=kt)
        err = torch.max(torch.abs(tp_dec(x1 @ D1.T, x2 @ D2.T) - y0 @ Do.T)).item()
        assert err < FP64_TOL

    # Positive coupled test (O(3) x Z2^T)
    c1 = Irreps("2x0ee + 2x1oe + 2x1eo")
    c2 = Irreps("1x1oe + 1x1ee")
    co = Irreps("2x0ee + 2x1ee + 2x1oo + 2x2ee")
    tp_c = StaticTensorProduct(c1, c2, co).to(dtype=torch.float64)
    cx1 = torch.randn((5, c1.dim), dtype=torch.float64, generator=gen)
    cx2 = torch.randn((5, c2.dim), dtype=torch.float64, generator=gen)
    cy0 = tp_c(cx1, cx2)
    for kp, kt in ((0, 0), (1, 0), (0, 1), (1, 1)):
        D1 = c1.D_from_matrix(R_lat, k_inv=kp, k_t=kt)
        D2 = c2.D_from_matrix(R_lat, k_inv=kp, k_t=kt)
        Do = co.D_from_matrix(R_lat, k_inv=kp, k_t=kt)
        err = torch.max(torch.abs(tp_c(cx1 @ D1.T, cx2 @ D2.T) - cy0 @ Do.T)).item()
        assert err < FP64_TOL

    # SparseCGContract matches exact dense contraction & is equivariant
    sc_mod = SparseCGContract(Irrep("1oe"), Irrep("1eo"), Irrep("1oo"))
    a = torch.randn((4, 3, 6), dtype=torch.float64, generator=gen)
    b = torch.randn((4, 3, 6), dtype=torch.float64, generator=gen)
    out_sp = sc_mod(a, b)
    C_dense = wigner_3j(1, 1, 1, dtype=torch.float64)
    out_dense = torch.einsum("ijk,bic,bjc->bkc", C_dense, a, b)
    assert torch.max(torch.abs(out_sp - out_dense)).item() < FP64_TOL

    # Verify decoupled_clebsch_gordan Frobenius norm = 1 and parity rejection
    dcg = decoupled_clebsch_gordan("1o|1o", "1o|1o", "2e|0e")
    assert dcg.shape == (9, 9, 5)
    assert abs(torch.linalg.norm(dcg).item() - 1.0) < FP64_TOL
    with pytest.raises(ValueError, match="time-reversal"):
        decoupled_clebsch_gordan("1o|1o", "0e|0e", "1o|1e")

    # NEGATIVE R1 tests: forbidden parity, time-reversal, or triangle paths MUST raise ValueError
    with pytest.raises(ValueError, match="time-reversal"):
        build_sparse_cg(Irrep("1eo"), Irrep("0ee"), Irrep("1ee"))
    with pytest.raises(ValueError, match="parity"):
        build_sparse_cg(Irrep("1oe"), Irrep("1oe"), Irrep("1oe"))
    with pytest.raises(ValueError, match="triangle"):
        build_sparse_cg(DIrrep("1o|0e"), DIrrep("0e|1o"), DIrrep("2o|1o"))
    with pytest.raises(ValueError, match="R1"):
        validate_contraction_paths(
            Irreps("1x1eo"), Irreps("1x0ee"), Irreps("1x1ee"), [(0, 0, 0, "uvw", True)]
        )
    with pytest.raises(ValueError, match="R1"):
        validate_contraction_paths(
            DIrreps("1x1o|1o"),
            DIrreps("1x0e|0e"),
            DIrreps("1x1o|1e"),
            [(0, 0, 0, "uvw", True)],
        )


def test_r2_r3_layers_positive_and_negative():
    """R2 & R3: Linear, RMSNorm, DecoupledGate, EquivariantGate positive & negative tests."""
    # --- Positive R2: EquivariantLinear & EquivariantRMSNorm (decoupled + coupled) ---
    d_in = DIrreps("4x0e|0e + 2x0e|0o + 3x1o|1o + 2x2e|1o")
    d_out = DIrreps("3x0e|0e + 2x0e|0o + 2x1o|1o + 2x2e|1o")
    lin_d = EquivariantLinear(d_in, d_out, biases=True).to(dtype=torch.float64)
    errs = check_decoupled_equivariance(lin_d, d_in, d_out)
    assert max(errs.values()) < FP64_TOL

    norm_d = EquivariantRMSNorm(d_in, center_scalars=True).to(dtype=torch.float64)
    errs_n = check_decoupled_equivariance(norm_d, d_in, d_in)
    assert max(errs_n.values()) < FP64_TOL

    c_in = Irreps("4x0ee + 2x0eo + 3x1oe + 2x1eo + 2x2ee")
    c_out = Irreps("3x0ee + 2x0eo + 2x1oe + 2x1eo + 2x2ee")
    lin_c = EquivariantLinear(c_in, c_out, biases=True).to(dtype=torch.float64)
    errs_c = check_coupled_equivariance(lin_c, c_in, c_out)
    assert max(errs_c.values()) < FP64_TOL

    norm_c = EquivariantRMSNorm(c_in, center_scalars=True).to(dtype=torch.float64)
    errs_nc = check_coupled_equivariance(norm_c, c_in, c_in)
    assert max(errs_nc.values()) < FP64_TOL

    # --- Positive R3: DecoupledGate & EquivariantGate ---
    # 0e|0e can use SiLU; 0e|0o and 0o|0e MUST use odd/even activations (tanh / square)
    gate_d = DecoupledGate(
        irreps_scalars="3x0e|0e + 2x0e|0o + 2x0o|0e",
        act_scalars=[F.silu, torch.tanh, lambda x: x * x],
        irreps_gates="4x0e|0e",
        act_gates=[torch.sigmoid],
        irreps_gated="2x1o|1o + 2x0e|1o",
    ).to(dtype=torch.float64)
    # Note: x^2 is even, so 0o|0e becomes 0e|0e!
    assert gate_d.irreps_out == DIrreps(
        "3x0e|0e + 2x0e|0o + 2x0e|0e + 2x1o|1o + 2x0e|1o"
    )
    errs_gd = check_decoupled_equivariance(gate_d, gate_d.irreps_in, gate_d.irreps_out)
    assert max(errs_gd.values()) < FP64_TOL

    gate_c = EquivariantGate(
        irreps_scalars="3x0ee + 2x0eo",
        act_scalars=[F.silu, torch.tanh],
        irreps_gates="4x0ee",
        act_gates=[torch.sigmoid],
        irreps_gated="2x1oe + 2x1eo",
    ).to(dtype=torch.float64)
    errs_gc = check_coupled_equivariance(gate_c, gate_c.irreps_in, gate_c.irreps_out)
    assert max(errs_gc.values()) < FP64_TOL

    # Probe helper checks
    assert probe_activation_parity(torch.tanh) == -1
    assert probe_activation_parity(lambda x: x * x) == 1
    assert probe_activation_parity(F.silu) == 0

    # --- NEGATIVE R2 tests: mixing distinct (p, t) channels or biasing non-invariant scalars ---
    with pytest.raises(ValueError, match="Rule R2"):
        EquivariantLinear("1x0ee + 1x0eo", "1x0ee", instructions=[(0, 0), (1, 0)])
    with pytest.raises(ValueError, match="Rule R2"):
        EquivariantLinear("1x1o|1o", "1x1o|1e", instructions=[(0, 0)])
    with pytest.raises(ValueError, match="bias"):
        EquivariantLinear("2x0e|0o", "2x0e|0o", biases=[True])
    with pytest.raises(ValueError, match="bias"):
        EquivariantLinear("2x0eo", "2x0eo", biases=[True])

    # --- NEGATIVE R3 tests: feeding non-0e|0e into gate, wrong parity activation ---
    # 1) Feeding T-odd 0e|0o or P-odd 0o|0e into gate MUST be rejected!
    with pytest.raises(ValueError, match="Rule R3"):
        DecoupledGate("2x0e|0e", [F.silu], "2x0e|0o", [torch.tanh], "2x1o|1o")
    with pytest.raises(ValueError, match="Rule R3"):
        DecoupledGate("2x0e|0e", [F.silu], "2x0o|0e", [torch.tanh], "2x1o|1o")
    with pytest.raises(ValueError, match="Rule R3"):
        EquivariantGate("2x0ee", [F.silu], "2x0eo", [torch.tanh], "2x1eo")
    with pytest.raises(ValueError, match="Rule R3"):
        validate_invariant_scalar(DIrreps("1x0e|0o"), role="attention")

    # 2) Non-even/non-odd activation (SiLU / sigmoid) on T-odd or P-odd scalar MUST be rejected!
    with pytest.raises(ValueError, match="time-reversal"):
        DecoupledGate("2x0e|0o", [F.silu], "", [], "")
    with pytest.raises(ValueError, match="spatial parity"):
        DecoupledGate("2x0o|0e", [torch.sigmoid], "", [], "")
    with pytest.raises(ValueError, match="Rule R3"):
        validate_scalar_activation(Irrep("0oo"), F.silu)
    with pytest.raises(ValueError, match="non-scalar"):
        validate_scalar_activation(DIrrep("1o|0e"), torch.tanh)


def test_hybrid_nosoc_backbone_and_zero_init_soc_head():
    """Verify decoupled superexchange |A^(1o|1o)|^2 + zero-init thin SOC head (0.0 diff + nonzero grad)."""
    gen = torch.Generator(device="cpu").manual_seed(31)
    pos = torch.randn((6, 3), dtype=torch.float64, generator=gen)
    spins = torch.randn((6, 3), dtype=torch.float64, generator=gen)

    sh_lat = SphericalHarmonics(1, normalize=True, parity=-1, time_reversal=1)
    sh_spin = SphericalHarmonics(1, normalize=False, parity=1, time_reversal=-1)
    couple = Couple("1x1o|1o", jmax=2)
    # Coupled irreps of 1o|1o: 1x0oo + 1x1oo + 1x2oo.
    # Contract with itself via StaticTensorProduct -> 0ee scalar, then zero-init linear head.
    soc_tp = StaticTensorProduct(couple.irreps_out, couple.irreps_out, "2x0ee").to(
        dtype=torch.float64
    )
    soc_head = (
        EquivariantLinear("2x0ee", "1x0ee", biases=False)
        .to(dtype=torch.float64)
        .zero_init_()
    )

    def build_features(p, s):
        # All pairs i != j
        diff = p.unsqueeze(1) - p.unsqueeze(0)  # (N, N, 3)
        dist2 = (diff * diff).sum(dim=-1)
        mask = (~torch.eye(p.shape[0], dtype=torch.bool)).to(dtype=p.dtype)
        w = torch.exp(-0.5 * dist2) * mask
        y_r = sh_lat(diff)[..., 1:4]  # (N, N, 3) = 1o
        y_s = sh_spin(s)[..., 1:4]  # (N, 3) = 1o (spin)
        # Node-level decoupled density A_i^(1o|1o)[m, n] = sum_j w_ij Y^1_m(r_ij) Y^1_n(s_j)
        A = torch.einsum("ij,ijm,jn->imn", w, y_r, y_s).reshape(p.shape[0], 9)
        return A

    def energy_fn(p, s):
        A = build_features(p, s)
        e_nosoc = (
            A * A
        ).sum()  # |A^(1o|1o)|^2 contains superexchange (r_ij.r_ik)(s_j.s_k)
        c_feat = couple(A)
        soc_inv = soc_tp(c_feat, c_feat)
        e_soc = soc_head(soc_inv).sum()
        return e_nosoc + e_soc

    # At zero-init: E_SOC == E_noSOC identically -> spin-only rotation error is literally 0.0 (to machine eps)
    errs_init = check_scalar_invariance(energy_fn, pos, spins, seed=5)
    assert errs_init["rot_spin_only"] < FP64_TOL
    assert errs_init["rot_lat_only"] < FP64_TOL
    assert errs_init["rot_joint"] < FP64_TOL
    assert errs_init["parity"] < FP64_TOL
    assert errs_init["time_reversal"] < FP64_TOL

    # Yet SOC head parameters have strictly NON-ZERO gradient!
    e_val = energy_fn(pos, spins)
    e_val.backward()
    assert soc_head.weight_blocks[0].grad is not None
    assert torch.linalg.norm(soc_head.weight_blocks[0].grad).item() > 1e-6

    # After activating SOC head weights: joint R/P/T remain < 1e-13, while spin-only rotation is O(1)
    with torch.no_grad():
        soc_head.weight_blocks[0].copy_(
            torch.tensor([[0.7, -0.4]], dtype=torch.float64)
        )
    errs_soc = check_scalar_invariance(energy_fn, pos, spins, seed=5)
    assert errs_soc["rot_joint"] < FP64_TOL
    assert errs_soc["parity"] < FP64_TOL
    assert errs_soc["time_reversal"] < FP64_TOL
    assert errs_soc["full_soc"] < FP64_TOL
    assert errs_soc["rot_spin_only"] > 1e-3


def test_fp32_default_buffers_and_exact_53bit_double_rematerialization():
    """Verify 0 FP64 buffers in default FP32 AND exact 53-bit FP64 re-materialization on .double()."""
    # Construct all buffer-holding symmetry modules under default float32
    assert torch.get_default_dtype() == torch.float32
    mods = [
        Couple("2x0e|0e + 2x1o|1o + 1x2e|1o", jmax=2),
        CoupleBlock(DIrrep("2e|1o"), jmax=2),
        BipolarSphericalHarmonics.for_bond_spin(
            lmax_r=2, lmax_s=2, jmax=2, spin_scale=0.3
        ),
        SphericalHarmonics(3, normalize=True, normalization="norm"),
        SpinScale(3, [False, True, True], [1.0, 3.7, 5.1]),
        StaticTensorProduct(
            "2x0e|0e + 2x1o|1o", "1x0e|0e + 1x1o|1o", "2x0e|0e + 2x1o|1o"
        ),
        SparseCGContract(
            DIrrep("2e|2e"), DIrrep("2e|2e"), DIrrep("2e|2e")
        ),  # nnz=625 factorized
    ]

    # 1. Strictly ZERO FP64 buffers in default FP32 (eliminates FP64->FP32 runtime casts!)
    for mod in mods:
        for bname, buf in mod.named_buffers():
            assert (
                buf.dtype != torch.float64
            ), f"{mod.__class__.__name__}.{bname} is float64 in FP32 default"

    # 2. Convert via .double() AND .to(torch.float64) -> exact 53-bit FP64 constants (NOT upcast FP32!)
    couple_f32 = Couple("2x1o|1o + 1x2e|1o")
    couple_d = couple_f32.double()
    exact_u = torch.cat(
        [
            math.sqrt(2 * J + 1)
            * wigner_3j(1, 1, J, dtype=torch.float64).reshape(9, 2 * J + 1)
            for J in (0, 1, 2)
        ],
        dim=1,
    )
    # Exact 53-bit match (upcast FP32 would have ~1e-8 error!)
    assert torch.max(torch.abs(couple_d._couple_1_1 - exact_u)).item() == 0.0

    cb_d = CoupleBlock(DIrrep("1o|1o")).to(dtype=torch.float64)
    assert torch.max(torch.abs(cb_d._U_T - exact_u.T)).item() == 0.0

    sc_d = SpinScale(3, [False, True, True], [1.0, 3.7, 5.1]).double()
    assert abs(sc_d._combined_scale[1].item() - (1.0 / 3.7)) == 0.0
    assert abs(sc_d._combined_scale[2].item() - (1.0 / 5.1)) == 0.0

    # SpinScale state_dict -> load_state_dict -> forward() + .double() roundtrip
    sc_src = SpinScale(3, [False, True, True], [1.0, 3.7, 5.1])
    sc_dst = SpinScale(3, [False, True, True])
    sc_dst.load_state_dict(sc_src.state_dict())
    s_chk = torch.tensor([[1.0, 2.0, -1.0], [3.0, -2.0, 1.5], [-4.0, 1.0, 2.5]])
    t_chk = torch.tensor([0, 1, 2], dtype=torch.long)
    assert (
        torch.max(torch.abs(sc_dst(s_chk, t_chk) - sc_src(s_chk, t_chk))).item() == 0.0
    )
    sc_dst_d = sc_dst.double()
    assert torch.allclose(
        sc_dst_d.spin_reference,
        sc_src.spin_reference.to(dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )
    assert (
        torch.max(
            torch.abs(
                sc_dst_d(s_chk.double(), t_chk)
                - s_chk.double()
                * (sc_dst_d.mask_mag / sc_dst_d.spin_reference)[t_chk].unsqueeze(-1)
            )
        ).item()
        < FP64_TOL
    )

    # Loading an FP64 SpinScale checkpoint into an FP32 module preserves 53-bit precision on .double()
    sc_f64_src = SpinScale(3, [False, True, True], [1.0, 3.7, 5.1]).double()
    sc_f32_dst = SpinScale(3, [False, True, True])
    sc_f32_dst.load_state_dict(sc_f64_src.state_dict())
    sc_f32_to_f64 = sc_f32_dst.double()
    assert abs(sc_f32_to_f64.spin_reference[1].item() - 3.7) == 0.0
    assert abs(sc_f32_to_f64._combined_scale[1].item() - (1.0 / 3.7)) == 0.0
    assert (
        torch.max(
            torch.abs(
                sc_f32_to_f64(s_chk.double(), t_chk) - sc_f64_src(s_chk.double(), t_chk)
            )
        ).item()
        == 0.0
    )

    # 3. Full FP64 R1-R5 equivariance < 1e-13 after FP32 -> .double()
    tp_d = StaticTensorProduct(
        "2x0e|0e + 2x1o|1o", "1x0e|0e + 1x1o|1o", "2x0e|0e + 2x1o|1o"
    ).double()
    gen = torch.Generator(device="cpu").manual_seed(77)
    x1 = torch.randn((4, tp_d.irreps_in1.dim), dtype=torch.float64, generator=gen)
    x2 = torch.randn((4, tp_d.irreps_in2.dim), dtype=torch.float64, generator=gen)
    R_lat = random_rotation_matrix(dtype=torch.float64, generator=gen)
    R_spin = random_rotation_matrix(dtype=torch.float64, generator=gen)
    D1 = tp_d.irreps_in1.D_from_matrices(R_lat, R_spin, k_inv=1, k_t=1)
    D2 = tp_d.irreps_in2.D_from_matrices(R_lat, R_spin, k_inv=1, k_t=1)
    Do = tp_d.irreps_out.D_from_matrices(R_lat, R_spin, k_inv=1, k_t=1)
    assert (
        torch.max(torch.abs(tp_d(x1 @ D1.T, x2 @ D2.T) - tp_d(x1, x2) @ Do.T)).item()
        < FP64_TOL
    )

    # Bipolar FP32 -> .to(torch.float64) exact SOC equivariance < 1e-13
    bip_d = BipolarSphericalHarmonics.for_bond_spin(
        lmax_r=2, lmax_s=2, jmax=2, spin_scale=0.3
    ).to(torch.float64)
    r = torch.randn((5, 3), dtype=torch.float64, generator=gen)
    s = torch.randn((5, 3), dtype=torch.float64, generator=gen)
    Db = bip_d.irreps_out.D_from_matrix(R_lat, k_inv=1, k_t=1)
    assert (
        torch.max(
            torch.abs(bip_d(-r @ R_lat.T, -s @ R_lat.T) - bip_d(r, s) @ Db.T)
        ).item()
        < FP64_TOL
    )


def test_bipolar_spin_pair_j1_j2_and_single_degree_sh():
    """Verify exact < 1e-15 parity of fused bipolar_spin_pair_j1_j2 and single-degree SH."""
    gen = torch.Generator(device="cpu").manual_seed(91)
    si = torch.randn((8, 3), dtype=torch.float64, generator=gen)
    sj = torch.randn((8, 3), dtype=torch.float64, generator=gen)
    v = torch.randn((8, 3), dtype=torch.float64, generator=gen)

    # 1. bipolar_spin_pair_j1_j2 matches BipolarSphericalHarmonics.for_spin_spin(1, 2) bit-for-bit
    scale = 0.35
    b_mod = BipolarSphericalHarmonics.for_spin_spin(
        lmax=1, jmax=2, spin_scale=scale
    ).double()
    y_mod = b_mod(si, sj)
    b_cross, b_quad = bipolar_spin_pair_j1_j2(si, sj, spin_scale=scale)
    assert (
        torch.max(torch.abs(b_cross - y_mod[:, b_mod.get_block_slice(1, 1, 1)])).item()
        < FP64_TOL
    )
    assert (
        torch.max(torch.abs(b_quad - y_mod[:, b_mod.get_block_slice(1, 1, 2)])).item()
        < FP64_TOL
    )

    # Also verify exact O(3) x Z2^T equivariance (1ee and 2ee)
    R = random_rotation_matrix(dtype=torch.float64, generator=gen)
    D1 = Irrep("1ee").D_from_matrix(R, k_inv=1, k_t=1)
    D2 = Irrep("2ee").D_from_matrix(R, k_inv=1, k_t=1)
    bc_rot, bq_rot = bipolar_spin_pair_j1_j2(-si @ R.T, -sj @ R.T, spin_scale=scale)
    assert torch.max(torch.abs(bc_rot - b_cross @ D1.T)).item() < FP64_TOL
    assert torch.max(torch.abs(bq_rot - b_quad @ D2.T)).item() < FP64_TOL

    # 2. Fast single-degree spherical_harmonics matches full l=0..4 across all normalizations
    for norm_mode in ("component", "norm", "integral"):
        sh_full = SphericalHarmonics(
            4, normalize=False, normalization=norm_mode
        ).double()
        y_full = sh_full(v)
        for deg in range(5):
            y_deg = spherical_harmonics(
                deg, v, normalize=False, normalization=norm_mode
            )
            y_mod_deg = SphericalHarmonics(
                [deg], normalize=False, normalization=norm_mode
            ).double()(v)
            ref = y_full[:, deg * deg : (deg + 1) ** 2]
            assert torch.max(torch.abs(y_deg - ref)).item() < FP64_TOL
            assert torch.max(torch.abs(y_mod_deg - ref)).item() < FP64_TOL


def test_sparse_cg_unrolled_factorized_and_bispectrum():
    """Verify unrolled (nnz<=128), factorized (nnz>128) SparseCGContract, and SparseBispectrumContract."""
    gen = torch.Generator(device="cpu").manual_seed(103)

    # 1. Unrolled O(3) CG: (2,2,1) and (2,2,2)
    for ir1, ir2, ir3 in (
        (Irrep("2ee"), Irrep("2ee"), Irrep("1ee")),
        (Irrep("2ee"), Irrep("2ee"), Irrep("2ee")),
        (Irrep("1oe"), Irrep("1oe"), Irrep("2ee")),
    ):
        mod = SparseCGContract(ir1, ir2, ir3)
        assert mod._use_unrolled and not mod._use_factorized
        x1 = torch.randn((6, ir1.dim, 5), dtype=torch.float64, generator=gen)
        x2 = torch.randn((6, ir2.dim, 5), dtype=torch.float64, generator=gen)
        C = wigner_3j(ir1.l, ir2.l, ir3.l, dtype=torch.float64)
        ref = torch.einsum("ijk,bic,bjc->bkc", C, x1, x2)
        assert torch.max(torch.abs(mod(x1, x2) - ref)).item() < FP64_TOL

    # 2. Factorized decoupled CG with nnz > 128: (2e|2e) x (2e|2e) -> (2e|2e), nnz = 25*25 = 625
    d_ir = DIrrep("2e|2e")
    mod_fac = SparseCGContract(d_ir, d_ir, d_ir).double()
    assert mod_fac._use_factorized and mod_fac.nnz == 625
    dx1 = torch.randn((4, 25, 3), dtype=torch.float64, generator=gen)
    dx2 = torch.randn((4, 25, 3), dtype=torch.float64, generator=gen)
    C_dec = decoupled_clebsch_gordan(d_ir, d_ir, d_ir, dtype=torch.float64)
    ref_dec = torch.einsum("ijk,bic,bjc->bkc", C_dec, dx1, dx2)
    assert torch.max(torch.abs(mod_fac(dx1, dx2) - ref_dec)).item() < FP64_TOL

    # 3. SparseBispectrumContract: pair mode (1,1,2) and channelwise mode (2,2,2)
    b112 = SparseBispectrumContract(1, 1, 2, mode="pair")
    p1 = torch.randn((6, 3, 4), dtype=torch.float64, generator=gen)
    p2 = torch.randn((6, 5, 3), dtype=torch.float64, generator=gen)
    ref_112 = torch.einsum(
        "ijk,nia,nja,nkb->nab", wigner_3j(1, 1, 2, dtype=torch.float64), p1, p1, p2
    )
    assert torch.max(torch.abs(b112(p1, p1, p2) - ref_112)).item() < FP64_TOL

    b222 = SparseBispectrumContract(2, 2, 2, mode="channelwise")
    ref_222 = torch.einsum(
        "ijk,nic,njc,nkc->nc", wigner_3j(2, 2, 2, dtype=torch.float64), p2, p2, p2
    )
    assert torch.max(torch.abs(b222(p2, p2, p2) - ref_222)).item() < FP64_TOL

    # R1 negative test: odd l1+l2+l3 bispectrum rejected
    with pytest.raises(ValueError, match="Parity rule"):
        SparseBispectrumContract(1, 1, 1)


def test_equivariant_linear_2d_weight_blocks_and_physmuon_equivariance():
    """Verify EquivariantLinear 2D weight_blocks preserve R2 & exact FP64 equivariance after PhysMuon(iso=True) step."""
    ir_in = DIrreps("8x0e|0e + 6x1o|1o + 4x2e|1o")
    ir_out = DIrreps("8x0e|0e + 6x1o|1o + 4x2e|1o")
    lin = EquivariantLinear(ir_in, ir_out, biases=True).double()

    # Check 2D per-irrep weight_blocks shape, weight_numel, canonical_hash, and flat_weight equivalence
    assert len(lin.weight_blocks) == 3
    assert [tuple(w.shape) for w in lin.weight_blocks] == [(8, 8), (6, 6), (4, 4)]
    assert lin.weight_numel == 64 + 36 + 16
    assert lin.flat_weight().shape == (lin.weight_numel,)

    # External flat weight must produce bit-identical output to structured weight_blocks
    gen = torch.Generator(device="cpu").manual_seed(131)
    x = torch.randn((5, ir_in.dim), dtype=torch.float64, generator=gen)
    y_struct = lin(x)
    y_flat = lin(x, weight=lin.flat_weight().detach(), bias=lin.bias.detach())
    assert torch.max(torch.abs(y_struct - y_flat)).item() < FP64_TOL

    # zero_init_() zeroes all weight_blocks and bias
    lin_z = EquivariantLinear(ir_in, ir_out, biases=True).double().zero_init_()
    assert torch.max(torch.abs(lin_z(x))).item() == 0.0

    # Simulate an exact PhysMuon(iso=True) micro-batch diagonal-metric polar step on every 2D block
    micro_grads = [[] for _ in lin.weight_blocks]
    for mb in range(3):
        lin.zero_grad(set_to_none=True)
        xm = torch.randn((4, ir_in.dim), dtype=torch.float64, generator=gen)
        loss = (lin(xm) ** 2).mean()
        loss.backward()
        for idx, wb in enumerate(lin.weight_blocks):
            assert wb.grad is not None
            micro_grads[idx].append(wb.grad.detach().clone())

    with torch.no_grad():
        for wb, gs in zip(lin.weight_blocks, micro_grads):
            g_stack = torch.stack(gs, dim=0)  # (K, rows, cols)
            gm = g_stack.mean(dim=0)
            rh = (
                g_stack.square().mean(dim=0).sum(dim=1) - gm.square().sum(dim=1)
            ) / wb.size(1)
            ch = (
                g_stack.square().mean(dim=0).sum(dim=0) - gm.square().sum(dim=0)
            ) / wb.size(0)
            rh = rh.clamp_min(0.0) + 1e-30
            ch = ch.clamp_min(0.0) + 1e-30
            bi = ((rh / rh.mean()).sqrt().clamp(0.5, 2.0)).rsqrt()[:, None]
            ai = ((ch / ch.mean()).sqrt().clamp(0.5, 2.0)).rsqrt()[None, :]
            # Exact polar factor via SVD in diagonal metric (equivalent to converged NS)
            U_s, _, Vh_s = torch.linalg.svd((gm * bi) * ai, full_matrices=False)
            P = U_s @ Vh_s
            D = (P * bi) * ai
            D = D * (math.sqrt(min(wb.size(0), wb.size(1))) / (D.norm() + 1e-12))
            wb.add_(D, alpha=-0.02)

    # Exact O(3)_lat x O(3)_spin equivariance (< 1e-13 FP64) MUST still hold after PhysMuon(iso=True)!
    errs = check_decoupled_equivariance(lin, ir_in, ir_out)
    assert max(errs.values()) < FP64_TOL


def test_torch_compile_fullgraph_dynamic_zero_breaks():
    """All core symmetry modules compile with torch.compile(fullgraph=True, dynamic=True) and zero graph breaks."""

    class CompositeSymmetryBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.spin_scale = SpinScale(3, [False, True, True], [1.0, 2.5, 4.0])
            self.mw = ParamagneticCoordination(self.spin_scale)
            self.sh_r = SphericalHarmonics(
                2, normalize=True, parity=-1, time_reversal=1
            )
            self.sh_s = SphericalHarmonics(
                2, normalize=False, parity=1, time_reversal=-1
            )
            self.bipolar = BipolarSphericalHarmonics.for_spin_spin(
                lmax=1, jmax=2, spin_scale=0.5
            )
            self.couple = Couple("2x0e|0e + 2x1o|1o", jmax=2)
            self.couple_blk = CoupleBlock(DIrrep("1o|1o"), jmax=2)
            self.lin = EquivariantLinear(
                "2x0e|0e + 2x1o|1o", "2x0e|0e + 2x1o|1o", biases=True
            )
            self.norm = EquivariantRMSNorm("2x0e|0e + 2x1o|1o", center_scalars=True)
            self.gate = DecoupledGate(
                "2x0e|0e", [F.silu], "2x0e|0e", [torch.sigmoid], "2x1o|1o"
            )
            self.tp = StaticTensorProduct(
                "2x0e|0e + 2x1o|1o", "1x0e|0e + 1x1o|1o", "2x0e|0e + 2x1o|1o"
            )
            self.sparse_cg = SparseCGContract(Irrep("1oe"), Irrep("1eo"), Irrep("1oo"))
            self.bisp_112 = SparseBispectrumContract(1, 1, 2, mode="pair")
            self.bisp_222 = SparseBispectrumContract(2, 2, 2, mode="channelwise")

        def forward(
            self,
            r,
            si,
            sj,
            atom_types,
            dec_feat,
            dec_attr,
            gate_in,
            v1,
            v2,
            edge_src,
            edge_dst,
            edge_w,
            q2,
        ):
            si_s = self.spin_scale(si, atom_types)
            sj_s = self.spin_scale(sj, atom_types)
            mw_out = self.mw(edge_w, edge_src, edge_dst, atom_types, si.shape[0])
            yr = self.sh_r(r)
            ys = self.sh_s(si_s)
            ys1 = spherical_harmonics(1, si_s, normalize=False)
            ys2 = spherical_harmonics(2, si_s, normalize=False)
            bip = self.bipolar(si_s, sj_s)
            bc_fused, bq_fused = bipolar_spin_pair_j1_j2(si_s, sj_s, spin_scale=0.5)
            xc = spin_cross_axial(si_s, sj_s)
            xq = spin_quadrupole_symmetric(si_s, sj_s)
            cart_rt = irreps_to_cartesian(cartesian_to_irreps(yr.reshape(-1, 3, 3)))
            h = self.norm(self.lin(dec_feat))
            h = self.tp(h, dec_attr)
            g = self.gate(gate_in)
            c = self.couple(h)
            cb = self.couple_blk(h[:, 2:].reshape(-1, 2, 3, 3).permute(0, 2, 3, 1))[1]
            cg = self.sparse_cg(v1, v2)
            b112 = self.bisp_112(v1, v2, q2)
            b222 = self.bisp_222(q2, q2, q2)
            return (
                yr.sum()
                + ys.sum()
                + ys1.sum()
                + ys2.sum()
                + bip.sum()
                + bc_fused.sum()
                + bq_fused.sum()
                + xc.sum()
                + xq.sum()
                + mw_out.sum()
                + cart_rt.sum()
                + g.sum()
                + c.sum()
                + cb.sum()
                + cg.sum()
                + b112.sum()
                + b222.sum()
            )

    model = CompositeSymmetryBlock().eval()
    # Explicitly verify 0 graph breaks via dynamo.explain before full inductor compile
    r0 = torch.randn((5, 3))
    si0 = torch.randn((5, 3))
    sj0 = torch.randn((5, 3))
    at0 = torch.arange(5, dtype=torch.long) % 3
    df0 = torch.randn((5, DIrreps("2x0e|0e + 2x1o|1o").dim))
    da0 = torch.randn((5, DIrreps("1x0e|0e + 1x1o|1o").dim))
    gi0 = torch.randn((5, model.gate.irreps_in.dim))
    v10 = torch.randn((5, 3, 4))
    v20 = torch.randn((5, 3, 4))
    es0 = torch.arange(5, dtype=torch.long)
    ed0 = torch.roll(es0, 1)
    ew0 = torch.randn((5, 2))
    q20 = torch.randn((5, 5, 3))
    exp = torch._dynamo.explain(model)(
        r0, si0, sj0, at0, df0, da0, gi0, v10, v20, es0, ed0, ew0, q20
    )
    assert exp.graph_break_count == 0
    assert exp.graph_count == 1

    compiled = torch.compile(model, fullgraph=True, dynamic=True)

    # Run with two different batch sizes to exercise dynamic shapes + autograd backward
    for n in (5, 9):
        r = torch.randn((n, 3), requires_grad=True)
        si = torch.randn((n, 3), requires_grad=True)
        sj = torch.randn((n, 3), requires_grad=True)
        atypes = torch.arange(n, dtype=torch.long) % 3
        dec_feat = torch.randn(
            (n, DIrreps("2x0e|0e + 2x1o|1o").dim), requires_grad=True
        )
        dec_attr = torch.randn((n, DIrreps("1x0e|0e + 1x1o|1o").dim))
        gate_in = torch.randn((n, model.gate.irreps_in.dim))
        v1 = torch.randn((n, 3, 4))
        v2 = torch.randn((n, 3, 4))
        edge_src = torch.arange(n, dtype=torch.long)
        edge_dst = torch.roll(edge_src, 1)
        edge_w = torch.randn((n, 2))
        q2 = torch.randn((n, 5, 3))

        out_eager = model(
            r,
            si,
            sj,
            atypes,
            dec_feat,
            dec_attr,
            gate_in,
            v1,
            v2,
            edge_src,
            edge_dst,
            edge_w,
            q2,
        )
        out_comp = compiled(
            r,
            si,
            sj,
            atypes,
            dec_feat,
            dec_attr,
            gate_in,
            v1,
            v2,
            edge_src,
            edge_dst,
            edge_w,
            q2,
        )
        assert torch.allclose(out_eager, out_comp, atol=1e-4, rtol=1e-4)

        grad_eager = torch.autograd.grad(out_eager, (r, si), retain_graph=True)
        grad_comp = torch.autograd.grad(out_comp, (r, si))
        assert torch.allclose(grad_eager[0], grad_comp[0], atol=1e-4, rtol=1e-4)
        assert torch.allclose(grad_eager[1], grad_comp[1], atol=1e-4, rtol=1e-4)

    # Also verify large factorized SparseCGContract (nnz > 128) compiles with 0 graph breaks
    cg_fac = SparseCGContract(DIrrep("2e|2e"), DIrrep("2e|2e"), DIrrep("2e|2e")).eval()
    d22_0 = torch.randn((5, 25, 2))
    exp_fac = torch._dynamo.explain(cg_fac)(d22_0, d22_0)
    assert exp_fac.graph_break_count == 0
    assert exp_fac.graph_count == 1
    cg_fac_comp = torch.compile(cg_fac, fullgraph=True, dynamic=True)
    for n in (5, 9):
        d_a = torch.randn((n, 25, 2), requires_grad=True)
        d_b = torch.randn((n, 25, 2), requires_grad=True)
        y_e = cg_fac(d_a, d_b).sum()
        y_c = cg_fac_comp(d_a, d_b).sum()
        assert torch.allclose(y_e, y_c, atol=1e-4, rtol=1e-4)
        g_e = torch.autograd.grad(y_e, (d_a, d_b), retain_graph=True)
        g_c = torch.autograd.grad(y_c, (d_a, d_b))
        assert torch.allclose(g_e[0], g_c[0], atol=1e-4, rtol=1e-4)
        assert torch.allclose(g_e[1], g_c[1], atol=1e-4, rtol=1e-4)
