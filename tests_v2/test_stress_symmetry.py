"""Adversarial stress & boundary-limit tests for pure-PyTorch ``spingnn2.symmetry``.

Exercises extreme and degenerate regimes to ensure zero silent crashes, NaNs,
Infs, dtype mismatches, or Dynamo graph breaks:
1. Full ``l = 0..4`` combinations across ``SphericalHarmonics``,
   ``BipolarSphericalHarmonics``, ``Couple``, ``CoupleBlock``, ``SparseCGContract``
   (unrolled, decoupled factorized, and high-l fallback), and ``SparseBispectrumContract``.
2. Zero-length ``[0, 0, 0]``, tiny sub-cutoff, and collinear vectors through
   forward + 1st-order + 2nd-order autograd in ``float64``, ``float32``, and ``bfloat16``.
3. Empty ``batch = 0`` tensors, ``E = 0`` / ``N = 0`` graphs, empty irreps ``""``,
   zero-multiplicity blocks, no-path ``EquivariantLinear`` / ``StaticTensorProduct``,
   and multi-dimensional ``edge_weights`` in ``paramagnetic_coordination``.
4. Non-divisible-by-4 and wide multiplicities ``mul in (1, 3, 5, 17, 64)``.
5. Non-contiguous, transposed, permuted, and sliced inputs.
6. Mixed ``bf16`` / ``fp32`` / ``fp64`` casts, ``bf16 -> fp64`` exact 53-bit
   ``_apply()`` re-materialization, ``torch.autocast``, and ``state_dict`` roundtrips.
7. ``torch.compile(fullgraph=True, dynamic=True)`` across ``batch = 0 -> 1 -> 5 -> 17``,
   non-contiguous inputs, zero vectors, ``bf16`` casts, and ``state_dict`` reloads.
"""

import math

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
    cartesian_to_irreps,
    check_coupled_equivariance,
    check_decoupled_equivariance,
    irreps_to_cartesian,
    paramagnetic_coordination,
    proper_wigner_D,
    proper_wigner_Ds,
    random_rotation_matrix,
    spherical_harmonics,
    spin_cross_axial,
    spin_quadrupole_symmetric,
    wigner_3j,
)

FP64_TOL = 1e-12


def test_all_l0_to_4_harmonics_bipolar_couple_cg_and_bispectrum():
    """Verify exact FP64 equivariance and parity across all l = 0..4 combinations."""
    gen = torch.Generator(device="cpu").manual_seed(101)
    R = random_rotation_matrix(dtype=torch.float64, generator=gen)
    v = torch.randn((4, 3), dtype=torch.float64, generator=gen)
    a = torch.randn((4, 3), dtype=torch.float64, generator=gen)
    b = torch.randn((4, 3), dtype=torch.float64, generator=gen)

    # 1. SphericalHarmonics l = 0..4 across all normalizations and normalize flags
    for norm in ("component", "norm", "integral"):
        for normalize in (True, False):
            for lmax in range(5):
                sh = SphericalHarmonics(
                    lmax,
                    normalize=normalize,
                    normalization=norm,
                    parity=-1,
                    time_reversal=1,
                ).double()
                y = sh(v)
                D = sh.irreps_out.D_from_matrix(R, k_inv=1, k_t=1)
                assert torch.max(torch.abs(sh(-v @ R.T) - y @ D.T)).item() < FP64_TOL
                # Single-degree functional matches the corresponding slice
                y_single = spherical_harmonics(
                    lmax, v, normalize=normalize, normalization=norm
                )
                assert (
                    torch.max(
                        torch.abs(y_single - y[:, lmax * lmax : (lmax + 1) ** 2])
                    ).item()
                    < FP64_TOL
                )

    # 2. BipolarSphericalHarmonics for all 5x5 (lmax_a, lmax_b) in 0..4 x 0..4
    for la in range(5):
        for lb in range(5):
            for norm in ("component", "norm", "integral"):
                for jmax in (0, 2, None):
                    bip = BipolarSphericalHarmonics(
                        la,
                        lb,
                        parity_a=-1,
                        parity_b=1,
                        time_reversal_a=1,
                        time_reversal_b=-1,
                        normalize_a=True,
                        normalize_b=False,
                        normalization=norm,
                        jmax=jmax,
                        scale_b=0.7,
                    ).double()
                    y = bip(a, b)
                    D = bip.irreps_out.D_from_matrix(R, k_inv=1, k_t=1)
                    assert (
                        torch.max(torch.abs(bip(-a @ R.T, -b @ R.T) - y @ D.T)).item()
                        < FP64_TOL
                    )

    # 3. Couple (+ inverse) and CoupleBlock for all 5x5 (l, lam) in 0..4 x 0..4
    for l in range(5):
        for lam in range(5):
            ir = DIrrep(l, (-1) ** l, lam, (-1) ** lam)
            c = Couple(DIrreps([(3, ir)])).double()
            x = torch.randn((2, c.irreps_in.dim), dtype=torch.float64, generator=gen)
            y = c(x)
            assert torch.max(torch.abs(c.inverse(y) - x)).item() < FP64_TOL
            D_in = c.irreps_in.D_from_matrices(R, R, k_inv=1, k_t=1)
            D_out = c.irreps_out.D_from_matrix(R, k_inv=1, k_t=1)
            assert torch.max(torch.abs(c(x @ D_in.T) - y @ D_out.T)).item() < FP64_TOL

            cb = CoupleBlock(ir).double()
            blk = torch.randn(
                (2, 2 * l + 1, 2 * lam + 1, 5), dtype=torch.float64, generator=gen
            )
            out_d = cb(blk)
            Dl = proper_wigner_D(l, R)
            Dlam = proper_wigner_D(lam, R)
            out_r = cb(torch.einsum("mi,nj,bijc->bmnc", Dl, Dlam, blk))
            for J, val in out_d.items():
                DJ = proper_wigner_D(J, R)
                exp = torch.einsum("Mm,bmc->bMc", DJ, val)
                assert torch.max(torch.abs(out_r[J] - exp)).item() < FP64_TOL

    # 4. SparseCGContract & SparseBispectrumContract across all valid l1, l2, l3 in 0..4
    for l1 in range(5):
        for l2 in range(5):
            for l3 in range(abs(l1 - l2), min(4, l1 + l2) + 1):
                ir1 = Irrep(l1, (-1) ** l1, 1)
                ir2 = Irrep(l2, (-1) ** l2, 1)
                ir3 = Irrep(l3, (-1) ** (l1 + l2), 1)
                cg = SparseCGContract(ir1, ir2, ir3).double()
                x1 = torch.randn((3, 2 * l1 + 1, 5), dtype=torch.float64, generator=gen)
                x2 = torch.randn((3, 2 * l2 + 1, 5), dtype=torch.float64, generator=gen)
                w3 = wigner_3j(l1, l2, l3, dtype=torch.float64)
                ref = torch.einsum("ijk,nic,njc->nkc", w3, x1, x2)
                assert torch.max(torch.abs(cg(x1, x2) - ref)).item() < FP64_TOL

                if (l1 + l2 + l3) % 2 == 0:
                    x3 = torch.randn(
                        (3, 2 * l3 + 1, 5), dtype=torch.float64, generator=gen
                    )
                    x3p = torch.randn(
                        (3, 2 * l3 + 1, 7), dtype=torch.float64, generator=gen
                    )
                    bc = SparseBispectrumContract(l1, l2, l3, mode="channelwise")
                    bp = SparseBispectrumContract(l1, l2, l3, mode="pair")
                    rc = torch.einsum("ijk,nic,njc,nkc->nc", w3, x1, x2, x3)
                    rp = torch.einsum("ijk,nia,nja,nkb->nab", w3, x1, x2, x3p)
                    assert torch.max(torch.abs(bc(x1, x2, x3) - rc)).item() < FP64_TOL
                    assert torch.max(torch.abs(bp(x1, x2, x3p) - rp)).item() < FP64_TOL

    # 5. High-l O(3) fallback (nnz > 128) & decoupled factorized (nnz > 128)
    cg_hi = SparseCGContract(Irrep(5, -1, 1), Irrep(5, -1, 1), Irrep(6, 1, 1)).double()
    assert cg_hi.nnz > 128 and not cg_hi._use_unrolled and not cg_hi._use_factorized
    xh1 = torch.randn((1, 11, 3), dtype=torch.float64, generator=gen)
    xh2 = torch.randn((3, 11, 3), dtype=torch.float64, generator=gen)
    ref_hi = torch.einsum(
        "ijk,nic,njc->nkc",
        wigner_3j(5, 5, 6, dtype=torch.float64),
        xh1.expand(3, 11, 3),
        xh2,
    )
    assert torch.max(torch.abs(cg_hi(xh1, xh2) - ref_hi)).item() < FP64_TOL

    cg_fac = SparseCGContract(
        DIrrep("3o|2e"), DIrrep("3o|2e"), DIrrep("4e|4e")
    ).double()
    assert cg_fac._use_factorized and cg_fac.nnz > 128
    xf1 = torch.randn((2, 35, 3), dtype=torch.float64, generator=gen)
    xf2 = torch.randn((2, 35, 3), dtype=torch.float64, generator=gen)
    R2 = random_rotation_matrix(dtype=torch.float64, generator=gen)
    D_in = DIrrep("3o|2e").D_from_matrices(R, R2, k_inv=1, k_t=1)
    D_out = DIrrep("4e|4e").D_from_matrices(R, R2, k_inv=0, k_t=0)
    y_fac = cg_fac(xf1, xf2)
    y_fac_r = cg_fac(
        torch.einsum("ij,njc->nic", D_in, xf1),
        torch.einsum("ij,njc->nic", D_in, xf2),
    )
    assert (
        torch.max(torch.abs(y_fac_r - torch.einsum("ij,njc->nic", D_out, y_fac))).item()
        < FP64_TOL
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_zero_and_tiny_vectors_first_and_second_order_gradients(dtype):
    """Zero [0,0,0], tiny near-zero, and collinear vectors have finite 0th/1st/2nd derivatives."""
    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1e-20, 0.0, 0.0],
            [1e-12, -1e-12, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )

    # SphericalHarmonics l = 1..4 with normalize=True/False and eps=0.0/1e-6
    for normalize in (True, False):
        for eps in (0.0, 1e-6):
            for lmax in (1, 2, 3, 4):
                sh = SphericalHarmonics(lmax, normalize=normalize, eps=eps).to(
                    dtype=dtype
                )
                v = pts.clone().requires_grad_(True)
                y = sh(v)
                assert torch.isfinite(y).all()
                (g1,) = torch.autograd.grad((y * y).sum(), v, create_graph=True)
                assert torch.isfinite(g1).all()
                (g2,) = torch.autograd.grad((g1 * g1).sum(), v)
                assert torch.isfinite(g2).all()

    # BipolarSphericalHarmonics at zero, tiny, and collinear inputs
    for la, lb in ((1, 1), (2, 2), (3, 2), (4, 1)):
        bip = BipolarSphericalHarmonics.for_bond_spin(la, lb, jmax=2).to(dtype=dtype)
        r = pts.clone().requires_grad_(True)
        s = pts.clone().requires_grad_(True)
        y = bip(r, s)
        assert torch.isfinite(y).all()
        g1r, g1s = torch.autograd.grad((y * y).sum(), (r, s), create_graph=True)
        assert torch.isfinite(g1r).all() and torch.isfinite(g1s).all()
        g2r, g2s = torch.autograd.grad(g1r.pow(2).sum() + g1s.pow(2).sum(), (r, s))
        assert torch.isfinite(g2r).all() and torch.isfinite(g2s).all()

    # Fused bipolar_spin_pair_j1_j2, spin_cross_axial, spin_quadrupole_symmetric
    si = pts.clone().requires_grad_(True)
    sj = pts.clone().requires_grad_(True)
    bc, bq = bipolar_spin_pair_j1_j2(si, sj, spin_scale=0.5)
    xc = spin_cross_axial(si, sj)
    xq = spin_quadrupole_symmetric(si, sj)
    tot = bc.pow(2).sum() + bq.pow(2).sum() + xc.pow(2).sum() + xq.pow(2).sum()
    assert torch.isfinite(tot)
    g1i, g1j = torch.autograd.grad(tot, (si, sj), create_graph=True)
    assert torch.isfinite(g1i).all() and torch.isfinite(g1j).all()
    g2i, g2j = torch.autograd.grad(g1i.pow(2).sum() + g1j.pow(2).sum(), (si, sj))
    assert torch.isfinite(g2i).all() and torch.isfinite(g2j).all()

    # SpinScale at zero & nonmagnetic atoms (exact 0 1st and 2nd derivatives for nonmagnetic)
    sc = SpinScale(3, [False, True, True], [1.0, 2.5, 4.0]).to(dtype=dtype)
    atypes = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    s_in = pts.clone().requires_grad_(True)
    s_hat = sc(s_in, atypes)
    assert torch.isfinite(s_hat).all()
    assert torch.all(s_hat[0] == 0.0) and torch.all(s_hat[3] == 0.0)
    e_sc = (s_hat.pow(2).sum(dim=-1) + 0.5 * s_hat.pow(4).sum(dim=-1)).sum()
    (g1s,) = torch.autograd.grad(e_sc, s_in, create_graph=True)
    assert torch.isfinite(g1s).all()
    assert torch.all(g1s[0] == 0.0) and torch.all(g1s[3] == 0.0)
    (g2s,) = torch.autograd.grad(g1s.pow(2).sum(), s_in)
    assert torch.isfinite(g2s).all()
    assert torch.all(g2s[0] == 0.0) and torch.all(g2s[3] == 0.0)

    # EquivariantRMSNorm at exact zero input
    ir = DIrreps("3x0e|0e + 5x1o|1o + 2x2e|1o")
    rn = EquivariantRMSNorm(ir, affine=True, center_scalars=True).to(dtype=dtype)
    xz = torch.zeros((4, ir.dim), dtype=dtype, requires_grad=True)
    yz = rn(xz)
    assert torch.isfinite(yz).all()
    (g1n,) = torch.autograd.grad((yz * yz).sum(), xz, create_graph=True)
    assert torch.isfinite(g1n).all()
    (g2n,) = torch.autograd.grad((g1n * g1n).sum(), xz)
    assert torch.isfinite(g2n).all()


def test_empty_batch_zero_and_degenerate_shapes():
    """Verify batch=0, E=0/N=0, empty irreps, 0-mul blocks, and 3D edge weights."""
    v0 = torch.empty((0, 3))
    for lmax in range(5):
        assert SphericalHarmonics(lmax, normalize=True)(v0).shape == (
            0,
            (lmax + 1) ** 2,
        )
        assert spherical_harmonics(lmax, v0, normalize=True).shape == (0, 2 * lmax + 1)

    b11 = BipolarSphericalHarmonics.for_spin_spin(1, 2)
    assert b11(v0, v0).shape == (0, b11.dim_out)
    b22 = BipolarSphericalHarmonics.for_bond_spin(2, 2, jmax=2)
    assert b22(v0, v0).shape == (0, b22.dim_out)
    bc, bq = bipolar_spin_pair_j1_j2(v0, v0)
    assert bc.shape == (0, 3) and bq.shape == (0, 5)
    assert spin_cross_axial(v0, v0).shape == (0, 3)
    assert spin_quadrupole_symmetric(v0, v0).shape == (0, 5)
    assert cartesian_to_irreps(torch.empty((0, 3, 3))).shape == (0, 9)
    assert irreps_to_cartesian(torch.empty((0, 9))).shape == (0, 3, 3)

    ir = DIrreps("3x0e|0e + 5x1o|1o + 2x2e|1o")
    x0 = torch.empty((0, ir.dim))

    # Couple & CoupleBlock on batch=0, empty irreps, jmax=0 truncation, and 0-mul blocks
    c = Couple(ir)
    y0 = c(x0)
    assert y0.shape == (0, c.irreps_out.dim)
    assert c.inverse(y0).shape == (0, ir.dim)

    c_empty = Couple("")
    assert c_empty(torch.empty((4, 0))).shape == (4, 0)
    assert c_empty.inverse(torch.empty((4, 0))).shape == (4, 0)
    assert c_empty(torch.empty((0, 0))).shape == (0, 0)

    c_j0_empty = Couple("2x2e|1o", jmax=0)
    assert c_j0_empty.truncated and c_j0_empty.irreps_out.dim == 0
    assert c_j0_empty(torch.randn((2, 3, 30))).shape == (2, 3, 0)
    assert CoupleBlock(DIrrep("2e|1o"), jmax=0)(torch.randn((2, 5, 3, 4))) == {}

    c_zmul = Couple("0x1o|1o + 2x0e|0e")
    assert not c_zmul.truncated
    xz = torch.randn((3, 2))
    assert torch.allclose(c_zmul.inverse(c_zmul(xz)), xz)

    cb = CoupleBlock(DIrrep("2e|1o"))
    out_cb = cb(torch.empty((0, 5, 3, 7)))
    assert all(val.shape == (0, 2 * J + 1, 7) for J, val in out_cb.items())

    # EquivariantLinear: batch=0, empty irreps, 0-mul, no matching paths, non-shared weights
    for sw in (True, False):
        lin = EquivariantLinear(ir, ir, biases=True, structured_weights=sw)
        assert lin(x0).shape == (0, ir.dim)
        lin_nomatch = EquivariantLinear("2x1o|0e", "2x1e|0e", structured_weights=sw)
        assert lin_nomatch.weight_numel == 0
        lin_nomatch.zero_init_()
        assert lin_nomatch.flat_weight().numel() == 0
        ynm = lin_nomatch(torch.randn((4, 6)))
        assert ynm.shape == (4, 6) and torch.all(ynm == 0.0)
    assert EquivariantLinear("", "")(torch.empty((4, 0))).shape == (4, 0)
    assert EquivariantLinear("", "")(torch.empty((0, 0))).shape == (0, 0)
    assert EquivariantLinear("0x0e|0e + 3x1o|1o", "3x1o|1o")(
        torch.randn((4, 27))
    ).shape == (4, 27)
    lin_ns = EquivariantLinear(ir, ir, shared_weights=False, internal_weights=False)
    assert lin_ns(
        torch.empty((0, 3, ir.dim)),
        torch.empty((0, 3, lin_ns.weight_numel)),
    ).shape == (0, 3, ir.dim)

    # EquivariantRMSNorm & Gates on batch=0 and empty scalar/gate slices
    assert EquivariantRMSNorm(ir, affine=True, center_scalars=True)(x0).shape == (
        0,
        ir.dim,
    )
    assert EquivariantRMSNorm("")(torch.empty((4, 0))).shape == (4, 0)
    g_empty_sc = DecoupledGate("", [], "3x0e|0e", [torch.sigmoid], "3x1o|1o")
    assert g_empty_sc(torch.empty((0, g_empty_sc.irreps_in.dim))).shape == (
        0,
        g_empty_sc.irreps_out.dim,
    )
    g_empty_gt = DecoupledGate("2x0e|0e + 1x0o|0e", [F.silu, torch.tanh], "", [], "")
    assert g_empty_gt(torch.empty((0, g_empty_gt.irreps_in.dim))).shape == (
        0,
        g_empty_gt.irreps_out.dim,
    )
    g_odd = EquivariantGate(
        "1x0ee", [F.silu], "2x0oe", [torch.tanh], "2x1oe", strict_invariant_gates=False
    ).double()
    assert (
        max(
            check_coupled_equivariance(
                g_odd, g_odd.irreps_in, g_odd.irreps_out
            ).values()
        )
        < FP64_TOL
    )

    # StaticTensorProduct: all 5 connection modes, empty instructions, empty irreps, batch=0
    tp_ir1 = DIrreps("3x0e|0e + 3x1o|1o")
    tp_ir2 = DIrreps("3x0e|0e + 3x1o|1o")
    tp_iro = DIrreps("3x0e|0e + 3x1o|1o + 1x0e|0e")
    mode_cases = (
        (0, 0, 0, "uvw", True),
        (0, 0, 0, "uvu", True),
        (0, 0, 0, "uvu", False),
        (0, 1, 1, "uvv", True),
        (0, 1, 1, "uvv", False),
        (0, 0, 0, "uuw", True),
        (0, 0, 2, "uuw", False),
        (0, 0, 0, "uuu", True),
        (0, 0, 0, "uuu", False),
    )
    for ins in mode_cases:
        has_w = ins[4]
        for sw in (True, False):
            tp = StaticTensorProduct(
                tp_ir1,
                tp_ir2,
                tp_iro,
                [ins],
                shared_weights=sw,
                internal_weights=(sw and has_w),
            )
            w = (
                None
                if (sw and has_w)
                else (
                    torch.randn((2, 3, tp.weight_numel)) if (not sw and has_w) else None
                )
            )
            y = tp(torch.randn((2, 1, tp_ir1.dim)), torch.randn((1, 3, tp_ir2.dim)), w)
            assert y.shape == (2, 3, tp_iro.dim)
            wz = (
                None
                if (sw and has_w)
                else (
                    torch.empty((0, 3, tp.weight_numel)) if (not sw and has_w) else None
                )
            )
            assert tp(
                torch.empty((0, 3, tp_ir1.dim)), torch.empty((0, 3, tp_ir2.dim)), wz
            ).shape == (0, 3, tp_iro.dim)
    tp_nopath = StaticTensorProduct("1x1o|0e", "1x1o|0e", "1x1o|0e")
    assert len(tp_nopath.instructions) == 0
    assert torch.all(tp_nopath(torch.randn((4, 3)), torch.randn((4, 3))) == 0.0)
    tp_empty = StaticTensorProduct("", "", "")
    assert tp_empty(torch.empty((4, 0)), torch.empty((4, 0))).shape == (4, 0)

    # SparseCGContract & SparseBispectrumContract on batch=0
    assert SparseCGContract(Irrep("1oe"), Irrep("1eo"), Irrep("1oo"))(
        torch.empty((0, 3, 5)), torch.empty((0, 3, 5))
    ).shape == (0, 3, 5)
    assert SparseCGContract(DIrrep("2e|2e"), DIrrep("2e|2e"), DIrrep("2e|2e"))(
        torch.empty((0, 25, 5)), torch.empty((0, 25, 5))
    ).shape == (0, 25, 5)
    assert SparseCGContract(Irrep(5, -1, 1), Irrep(5, -1, 1), Irrep(6, 1, 1))(
        torch.empty((0, 11, 5)), torch.empty((0, 11, 5))
    ).shape == (0, 13, 5)
    assert SparseBispectrumContract(2, 2, 2, mode="channelwise")(
        torch.empty((0, 5, 5)), torch.empty((0, 5, 5)), torch.empty((0, 5, 5))
    ).shape == (0, 5)
    assert SparseBispectrumContract(1, 1, 2, mode="pair")(
        torch.empty((0, 3, 5)), torch.empty((0, 3, 5)), torch.empty((0, 5, 7))
    ).shape == (0, 5, 7)

    # SpinScale & paramagnetic_coordination on batch=0, E=0/N=0, E=0/N=3, and 1D/2D/3D edge_weights
    sc = SpinScale(2, [False, True], [1.0, 3.5], include_padding_type=True)
    assert sc(v0, torch.empty((0,), dtype=torch.long)).shape == (0, 3)
    sc.calibrate_reference(v0, torch.empty((0,), dtype=torch.long))
    assert paramagnetic_coordination(
        torch.empty((0, 4)),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        sc.mask_mag,
        num_nodes=0,
    ).shape == (0, 4)
    mw_e0 = paramagnetic_coordination(
        torch.empty((0, 4)),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.tensor([0, 1, 2]),
        sc.mask_mag,
        num_nodes=3,
    )
    assert mw_e0.shape == (3, 4) and torch.all(mw_e0 == 0.0)
    es = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    ed = torch.tensor([1, 0, 1, 2], dtype=torch.long)
    at = torch.tensor([0, 1, 2], dtype=torch.long)
    norm1 = torch.tensor([0.5, 1.0, 2.0])
    assert paramagnetic_coordination(
        torch.randn((4,)), es, ed, at, sc.mask_mag, 3, norm1
    ).shape == (3,)
    assert paramagnetic_coordination(
        torch.randn((4, 2, 3)), es, ed, at, sc.mask_mag, 3, norm1
    ).shape == (3, 2, 3)

    # verify.py helpers on batch_size=0
    lin_v = EquivariantLinear("2x1oe", "2x1oe").double()
    assert all(
        err == 0.0
        for err in check_coupled_equivariance(
            lin_v, "2x1oe", "2x1oe", batch_size=0
        ).values()
    )
    lin_d = EquivariantLinear("2x1o|1o", "2x1o|1o").double()
    assert all(
        err == 0.0
        for err in check_decoupled_equivariance(
            lin_d, "2x1o|1o", "2x1o|1o", batch_size=0
        ).values()
    )


@pytest.mark.parametrize("mul", [1, 3, 5, 17, 64])
def test_odd_and_wide_multiplicities_1_3_5_17_64(mul):
    """Exercise odd, prime, and wide multiplicities across all equivariant layers."""
    ir = DIrreps(f"{mul}x0e|0e + {mul}x1o|1o + {mul}x2e|1o")
    x = torch.randn((4, ir.dim))

    for sw in (True, False):
        lin = EquivariantLinear(ir, ir, biases=True, structured_weights=sw)
        y = lin(x)
        assert y.shape == (4, ir.dim) and torch.isfinite(y).all()

    rn = EquivariantRMSNorm(ir, affine=True, center_scalars=True)
    assert rn(x).shape == (4, ir.dim)

    c = Couple(ir, jmax=2)
    assert c(x).shape == (4, c.irreps_out.dim)

    cb = CoupleBlock(DIrrep("2e|1o"), jmax=2)
    out_cb = cb(torch.randn((4, 5, 3, mul)))
    assert all(val.shape == (4, 2 * J + 1, mul) for J, val in out_cb.items())

    gate = DecoupledGate(
        f"{mul}x0e|0e",
        [F.silu],
        f"{2 * mul}x0e|0e",
        [torch.sigmoid],
        f"{mul}x1o|1o + {mul}x2e|1o",
    )
    gx = torch.randn((4, gate.irreps_in.dim))
    assert gate(gx).shape == (4, gate.irreps_out.dim)

    cg = SparseCGContract(Irrep("1oe"), Irrep("1eo"), Irrep("1oo"))
    assert cg(torch.randn((4, 3, mul)), torch.randn((4, 3, mul))).shape == (4, 3, mul)
    bisp = SparseBispectrumContract(2, 2, 2, mode="channelwise")
    assert bisp(
        torch.randn((4, 5, mul)), torch.randn((4, 5, mul)), torch.randn((4, 5, mul))
    ).shape == (4, mul)


def test_non_contiguous_transposed_and_sliced_inputs():
    """Non-contiguous transposed/sliced/permuted inputs match contiguous results exactly."""
    v = torch.randn((3, 14)).T[::2, :]  # (7, 3) non-contiguous
    assert not v.is_contiguous()

    for lmax in range(5):
        sh = SphericalHarmonics(lmax, normalize=True, normalization="norm")
        assert torch.allclose(sh(v), sh(v.contiguous()), atol=1e-6, rtol=1e-6)
        assert torch.allclose(
            spherical_harmonics(lmax, v, normalize=True),
            spherical_harmonics(lmax, v.contiguous(), normalize=True),
            atol=1e-6,
            rtol=1e-6,
        )

    b11 = BipolarSphericalHarmonics.for_spin_spin(1, 2)
    assert torch.allclose(b11(v, v), b11(v.contiguous(), v.contiguous()), atol=1e-6)
    b22 = BipolarSphericalHarmonics.for_bond_spin(2, 2, jmax=2)
    assert torch.allclose(b22(v, v), b22(v.contiguous(), v.contiguous()), atol=1e-6)
    bc, bq = bipolar_spin_pair_j1_j2(v, v, spin_scale=0.6)
    bc_c, bq_c = bipolar_spin_pair_j1_j2(v.contiguous(), v.contiguous(), spin_scale=0.6)
    assert torch.allclose(bc, bc_c) and torch.allclose(bq, bq_c)

    ir = DIrreps("3x0e|0e + 5x1o|1o + 2x2e|1o")
    x = torch.randn((ir.dim, 14)).T[::2, :]
    assert not x.is_contiguous()
    c = Couple(ir)
    yc = c(x)
    assert torch.allclose(yc, c(x.contiguous()), atol=1e-6)
    yc_nc = (
        yc.T.T
        if not yc.T.T.is_contiguous()
        else yc.unsqueeze(0).transpose(0, 1).squeeze(1)
    )
    assert torch.allclose(c.inverse(yc_nc), c.inverse(yc.contiguous()), atol=1e-6)

    lin = EquivariantLinear(ir, ir, biases=True)
    assert torch.allclose(lin(x), lin(x.contiguous()), atol=1e-6)
    rn = EquivariantRMSNorm(ir, affine=True, center_scalars=True)
    assert torch.allclose(rn(x), rn(x.contiguous()), atol=1e-6)

    blk = torch.randn((7, 5, 3, 6)).permute(3, 1, 2, 0)
    assert not blk.is_contiguous()
    cb = CoupleBlock(DIrrep("2e|1o"))
    o1, o2 = cb(blk), cb(blk.contiguous())
    for k in o1:
        assert torch.allclose(o1[k], o2[k], atol=1e-6)

    at = torch.arange(14, dtype=torch.long)[::2] % 3
    sc = SpinScale(3, [False, True, True], [1.0, 2.0, 3.0])
    assert torch.allclose(sc(v, at), sc(v.contiguous(), at.contiguous()))

    cg_f = SparseCGContract(DIrrep("2e|2e"), DIrrep("2e|2e"), DIrrep("2e|2e"))
    d22 = torch.randn((4, 25, 6)).permute(2, 1, 0)
    assert not d22.is_contiguous()
    assert torch.allclose(
        cg_f(d22, d22), cg_f(d22.contiguous(), d22.contiguous()), atol=1e-5
    )

    bisp = SparseBispectrumContract(1, 1, 2, mode="pair")
    v1_nc = torch.randn((4, 3, 6)).permute(2, 1, 0)
    q2_nc = torch.randn((5, 5, 6)).permute(2, 1, 0)
    assert torch.allclose(
        bisp(v1_nc, v1_nc, q2_nc),
        bisp(v1_nc.contiguous(), v1_nc.contiguous(), q2_nc.contiguous()),
        atol=1e-5,
    )


def test_mixed_bf16_fp32_fp64_casts_and_state_dict_roundtrips():
    """Verify bf16->fp64 exact re-materialization, mixed dtypes, autocast, and state_dict."""
    # 1. bf16 -> fp64 exact 53-bit re-materialization
    c = Couple("2x1o|1o + 1x2e|1o").bfloat16().double()
    exact_u11 = torch.cat(
        [
            math.sqrt(2 * J + 1)
            * wigner_3j(1, 1, J, dtype=torch.float64).reshape(9, 2 * J + 1)
            for J in (0, 1, 2)
        ],
        dim=1,
    )
    assert torch.max(torch.abs(c._couple_1_1 - exact_u11)).item() == 0.0

    cb = CoupleBlock(DIrrep("2e|1o")).bfloat16().double()
    exact_cb = torch.cat(
        [
            math.sqrt(2 * J + 1)
            * wigner_3j(2, 1, J, dtype=torch.float64).reshape(15, 2 * J + 1)
            for J in (1, 2, 3)
        ],
        dim=1,
    ).T
    assert torch.max(torch.abs(cb._U_T - exact_cb)).item() == 0.0

    sh_n = SphericalHarmonics(4, normalization="norm").bfloat16().double()
    assert abs(sh_n._norm_scale[-1].item() - (1.0 / 3.0)) == 0.0

    sc = SpinScale(2, [False, True], [1.0, 3.7], include_padding_type=True)
    sc.set_spin_reference([1.0, 4.2])
    sc_rt = sc.bfloat16().double()
    assert abs(sc_rt.spin_reference[1].item() - 4.2) == 0.0
    assert abs(sc_rt._combined_scale[1].item() - (1.0 / 4.2)) == 0.0

    cg_hi = (
        SparseCGContract(Irrep(5, -1, 1), Irrep(5, -1, 1), Irrep(6, 1, 1))
        .bfloat16()
        .double()
    )
    xh = torch.randn((2, 11, 3), dtype=torch.float64)
    ref_hi = torch.einsum(
        "ijk,nic,njc->nkc", wigner_3j(5, 5, 6, dtype=torch.float64), xh, xh
    )
    assert torch.max(torch.abs(cg_hi(xh, xh) - ref_hi)).item() < FP64_TOL

    # 2. Mixed-dtype inputs into Bipolar, SparseCG, SparseBispectrum, StaticTensorProduct
    a_bf16 = torch.randn((4, 3), dtype=torch.bfloat16)
    a_f32 = torch.randn((4, 3), dtype=torch.float32)
    b_f64 = torch.randn((4, 3), dtype=torch.float64)
    assert (
        BipolarSphericalHarmonics.for_bond_spin(2, 2, jmax=2)(a_f32, b_f64).dtype
        == torch.float64
    )
    assert (
        BipolarSphericalHarmonics.for_spin_spin(1, 2)(a_bf16, a_f32).dtype
        == torch.float32
    )
    assert bond_spin(a_f32, b_f64, 2, 2, jmax=2)[1].dtype == torch.float64
    assert (
        bipolar_spherical_harmonics(
            2,
            2,
            a_bf16,
            a_f32,
            parity_a=-1,
            parity_b=1,
            time_reversal_a=1,
            time_reversal_b=-1,
            jmax=2,
        )[1].dtype
        == torch.float32
    )

    cg_f = SparseCGContract(DIrrep("2e|2e"), DIrrep("2e|2e"), DIrrep("2e|2e"))
    assert (
        cg_f(
            torch.randn((2, 25, 3), dtype=torch.float32),
            torch.randn((2, 25, 3), dtype=torch.float64),
        ).dtype
        == torch.float64
    )
    assert (
        cg_hi(
            torch.randn((2, 11, 3), dtype=torch.float32),
            torch.randn((2, 11, 3), dtype=torch.float64),
        ).dtype
        == torch.float64
    )
    bisp_p = SparseBispectrumContract(1, 1, 2, mode="pair")
    assert (
        bisp_p(
            torch.randn((2, 3, 4), dtype=torch.bfloat16),
            torch.randn((2, 3, 4), dtype=torch.bfloat16),
            torch.randn((2, 5, 3), dtype=torch.float32),
        ).dtype
        == torch.float32
    )
    tp = StaticTensorProduct(
        "2x0e|0e + 1x1o|1o", "1x0e|0e + 1x1o|1o", "2x0e|0e + 1x1o|1o"
    )
    assert (
        tp(
            torch.randn((2, tp.irreps_in1.dim), dtype=torch.bfloat16),
            torch.randn((2, tp.irreps_in2.dim), dtype=torch.float32),
        ).dtype
        == torch.float32
    )

    # 3. Wigner-D & random_rotation_matrix in bfloat16
    R_bf16 = random_rotation_matrix(
        dtype=torch.bfloat16, generator=torch.Generator().manual_seed(7)
    )
    assert R_bf16.dtype == torch.bfloat16
    Ds_bf16 = proper_wigner_Ds(4, R_bf16)
    assert all(D.dtype == torch.bfloat16 and torch.isfinite(D).all() for D in Ds_bf16)
    assert Irrep("2ee").D_from_matrix(R_bf16).dtype == torch.bfloat16
    assert Irreps("1x1oe + 1x2ee").D_from_matrix(R_bf16).dtype == torch.bfloat16

    # 4. torch.autocast("cpu", dtype=torch.bfloat16) across all layers
    ir = DIrreps("3x0e|0e + 5x1o|1o + 2x2e|1o")
    lin = EquivariantLinear(ir, ir, biases=True)
    rn = EquivariantRMSNorm(ir, affine=True, center_scalars=True)
    c_ac = Couple(ir)
    cb_ac = CoupleBlock(DIrrep("2e|1o"))
    bip22 = BipolarSphericalHarmonics.for_bond_spin(2, 2, jmax=2)
    sc_ac = SpinScale(3, [False, True, True], [1.0, 2.0, 3.0])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        v = torch.randn((4, 3))
        sv = sc_ac(v, torch.tensor([0, 1, 2, 1]))
        assert torch.isfinite(bip22(v, sv)).all()
        assert torch.isfinite(c_ac(rn(lin(torch.randn((4, ir.dim)))))).all()
        assert torch.isfinite(cb_ac(torch.randn((4, 5, 3, 7)))[1]).all()
        assert torch.isfinite(
            tp(torch.randn((4, tp.irreps_in1.dim)), torch.randn((4, tp.irreps_in2.dim)))
        ).all()

    # 5. Full state_dict -> load_state_dict roundtrips across stateful modules
    x_sd = torch.randn((4, ir.dim))
    for sw in (True, False):
        l1 = EquivariantLinear(ir, ir, biases=True, structured_weights=sw)
        l2 = EquivariantLinear(ir, ir, biases=True, structured_weights=sw)
        l2.load_state_dict(l1.state_dict())
        assert torch.max(torch.abs(l1(x_sd) - l2(x_sd))).item() == 0.0

    rn1 = EquivariantRMSNorm(ir, affine=True, center_scalars=True)
    with torch.no_grad():
        for p in rn1.parameters():
            p.uniform_(0.5, 1.5)
    rn2 = EquivariantRMSNorm(ir, affine=True, center_scalars=True)
    rn2.load_state_dict(rn1.state_dict())
    assert torch.max(torch.abs(rn1(x_sd) - rn2(x_sd))).item() == 0.0

    tp1 = StaticTensorProduct(
        "2x0e|0e + 1x1o|1o", "1x0e|0e + 1x1o|1o", "2x0e|0e + 1x1o|1o"
    )
    tp2 = StaticTensorProduct(
        "2x0e|0e + 1x1o|1o", "1x0e|0e + 1x1o|1o", "2x0e|0e + 1x1o|1o"
    )
    tp2.load_state_dict(tp1.state_dict())
    t1_in = torch.randn((3, tp1.irreps_in1.dim))
    t2_in = torch.randn((3, tp1.irreps_in2.dim))
    assert torch.max(torch.abs(tp1(t1_in, t2_in) - tp2(t1_in, t2_in))).item() == 0.0


def test_compile_fullgraph_dynamic_stress_batch0_noncontig_mixed_and_state_dict():
    """Stress torch.compile(fullgraph=True, dynamic=True) on batch=0->1->5->17, non-contig, zero, bf16, state_dict."""

    class StressBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.sc = SpinScale(3, [False, True, True], [1.0, 2.5, 4.0])
            self.mw = ParamagneticCoordination(self.sc)
            self.sh = SphericalHarmonics(3, normalize=True, normalization="norm")
            self.bip = BipolarSphericalHarmonics.for_bond_spin(2, 2, jmax=2)
            self.lin = EquivariantLinear(
                "3x0e|0e + 5x1o|1o", "3x0e|0e + 5x1o|1o", biases=True
            )
            self.rn = EquivariantRMSNorm(
                "3x0e|0e + 5x1o|1o", affine=True, center_scalars=True
            )
            self.c = Couple("3x0e|0e + 5x1o|1o", jmax=2)
            self.cb = CoupleBlock(DIrrep("2e|1o"), jmax=2)
            self.bisp = SparseBispectrumContract(1, 1, 2, mode="pair")

        def forward(self, r, s, at, es, ed, ew, x, blk, v1, q2):
            s_s = self.sc(s, at)
            mw_out = self.mw(ew, es, ed, at, r.shape[0])
            y = self.sh(r)
            b = self.bip(r, s_s)
            bc, bq = bipolar_spin_pair_j1_j2(s_s, s_s, spin_scale=0.5)
            h = self.c(self.rn(self.lin(x)))
            cb_out = self.cb(blk)[1]
            bp = self.bisp(v1, v1, q2)
            return (
                mw_out.sum()
                + y.sum()
                + b.sum()
                + bc.sum()
                + bq.sum()
                + h.sum()
                + cb_out.sum()
                + bp.sum()
            )

    def _make_inputs(
        n: int, dtype: torch.dtype, non_contig: bool = False, zeros: bool = False
    ):
        if zeros:
            r = torch.zeros((n, 3), dtype=dtype, requires_grad=True)
            s = torch.zeros((n, 3), dtype=dtype, requires_grad=True)
        elif non_contig and n > 0:
            r = (
                torch.randn((3, 2 * n), dtype=dtype)
                .T[::2, :]
                .detach()
                .requires_grad_(True)
            )
            s = (
                torch.randn((3, 2 * n), dtype=dtype)
                .T[::2, :]
                .detach()
                .requires_grad_(True)
            )
            assert not r.is_contiguous() and not s.is_contiguous()
        else:
            r = torch.randn((n, 3), dtype=dtype, requires_grad=True)
            s = torch.randn((n, 3), dtype=dtype, requires_grad=True)
        at = torch.arange(n, dtype=torch.long) % 3
        es = torch.arange(n, dtype=torch.long)
        ed = torch.roll(es, 1) if n > 0 else es
        ew = torch.randn((n, 2), dtype=dtype)
        x = (
            torch.randn((48, 2 * n), dtype=dtype).T[::2, :]
            if (non_contig and n > 0)
            else torch.randn((n, 48), dtype=dtype)
        )
        blk = torch.randn((4, 5, 3, n), dtype=dtype).permute(3, 1, 2, 0)
        v1 = torch.randn((n, 3, 4), dtype=dtype)
        q2 = torch.randn((n, 5, 3), dtype=dtype)
        return r, s, at, es, ed, ew, x, blk, v1, q2

    # Build, save state_dict, reload into fresh instance with different SpinScale init
    src = StressBlock().eval()
    dst = StressBlock().eval()
    dst.sc.set_spin_reference([1.0, 9.9, 9.9])
    dst.load_state_dict(src.state_dict())

    # 1. Verify 0 graph breaks via dynamo.explain (both FP32 and BF16)
    exp_f32 = torch._dynamo.explain(dst)(*_make_inputs(5, torch.float32))
    assert exp_f32.graph_break_count == 0 and exp_f32.graph_count == 1

    dst_bf16 = StressBlock().eval()
    dst_bf16.load_state_dict(src.state_dict())
    dst_bf16 = dst_bf16.bfloat16()
    exp_bf16 = torch._dynamo.explain(dst_bf16)(*_make_inputs(5, torch.bfloat16))
    assert exp_bf16.graph_break_count == 0 and exp_bf16.graph_count == 1

    # 2. Full torch.compile(fullgraph=True, dynamic=True) across batch=0, 1, 5, 17, non-contig, zero vecs
    comp_f32 = torch.compile(dst, fullgraph=True, dynamic=True)
    for n, nc, z in (
        (0, False, False),
        (1, False, False),
        (5, True, False),
        (17, True, False),
        (4, False, True),
    ):
        args = _make_inputs(n, torch.float32, non_contig=nc, zeros=z)
        out_e = dst(*args)
        out_c = comp_f32(*args)
        assert torch.isfinite(out_e) and torch.isfinite(out_c)
        assert torch.allclose(out_e, out_c, atol=1e-4, rtol=1e-4)
        # Verify reloaded model matches source model identically
        assert torch.allclose(out_e, src(*args), atol=1e-5, rtol=1e-5)
        ge = torch.autograd.grad(out_e, (args[0], args[1]), retain_graph=True)
        gc = torch.autograd.grad(out_c, (args[0], args[1]))
        assert torch.isfinite(ge[0]).all() and torch.isfinite(ge[1]).all()
        assert torch.isfinite(gc[0]).all() and torch.isfinite(gc[1]).all()
        if n > 0:
            assert torch.allclose(ge[0], gc[0], atol=1e-4, rtol=1e-4)
            assert torch.allclose(ge[1], gc[1], atol=1e-4, rtol=1e-4)

    # 3. Also verify bf16 compiled fullgraph dynamic across batch=0 -> 5 -> 9
    comp_bf16 = torch.compile(dst_bf16, fullgraph=True, dynamic=True)
    for n in (0, 5, 9):
        args_b = _make_inputs(n, torch.bfloat16, non_contig=(n > 0))
        out_eb = dst_bf16(*args_b)
        out_cb = comp_bf16(*args_b)
        assert torch.isfinite(out_eb) and torch.isfinite(out_cb)
        assert torch.allclose(out_eb, out_cb, atol=5e-2, rtol=5e-2)
