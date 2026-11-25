"""Tests for VASP 5 & VASP 6 magnetic OUTCAR/OSZICAR -> SpinGNN2 extxyz conversion."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import ase
from ase.io import read as ase_read
import numpy as np
import pytest

from spingnn2.data import (
    VaspSpinParseError,
    build_extxyz_frame,
    convert_directory_to_extxyz,
    convert_pair_to_extxyz,
    detect_field_scale,
    parse_outcar,
    saxis_rotation_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_vasp_pair(
    directory: Path,
    *,
    version: str = "6.4.0",
    saxis=(0.0, 0.0, 1.0),
    lam: float = 10.0,
    i_constrained_m: int = 1,
    include_incar_echo: bool = True,
    write_sibling_incar: bool = False,
    incar_name: str = "INCAR",
    converged: bool = True,
    include_born: bool = False,
    include_ionic_piezo: bool = False,
    zero_torque: bool = False,
    corrupt_ep_factor: float = 1.0,
    flip_field_sign: bool = False,
    outcar_name: str = "OUTCAR",
    oszicar_name: str = "OSZICAR",
    scf_tag: str = "DAV",
    unconstrained_mag_site2: bool = False,
):
    """Synthesize physically self-consistent VASP 5 or VASP 6 OUTCAR + OSZICAR."""
    directory.mkdir(parents=True, exist_ok=True)
    major = int(version.split(".", 1)[0])
    rot = saxis_rotation_matrix(saxis)

    # 3 atoms: Fe (mag), Fe (mag), O (nonmag unless unconstrained_mag_site2=True)
    if i_constrained_m == 2:
        m_constr_cart = np.array([[0.0, 0.0, 3.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i_constrained_m == 0:
        m_constr_cart = np.zeros((3, 3), dtype=np.float64)
    else:
        m_constr_cart = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    site2_m = np.array([2.4, 0.0, 0.0]) if unconstrained_mag_site2 else np.zeros(3)
    if zero_torque:
        mw_cart = np.array([[0.0, 0.0, 3.2], [3.1, 0.0, 0.0], site2_m])
        m_int_cart = np.array([[0.0, 0.0, 3.5], [3.4, 0.0, 0.0], site2_m])
    else:
        mw_cart = np.array([[0.04, -0.02, 3.20], [3.10, 0.05, -0.03], site2_m])
        m_int_cart = np.array([[0.05, -0.02, 3.50], [3.40, 0.06, -0.03], site2_m])

    # VASP internal quantities live in the SAXIS frame (v_saxis = v_cart @ R).
    m_constr_s = m_constr_cart @ rot
    mw_s = mw_cart @ rot
    m_int_s = m_int_cart @ rot

    # Compute exact constraint field C_i = dE_p/dM_{W,i} in SAXIS frame.
    c_s = np.zeros((3, 3), dtype=np.float64)
    ep = 0.0
    if i_constrained_m in (1, 2, 4):
        for i in range(2):
            if i_constrained_m == 1:
                u = m_constr_s[i] / np.linalg.norm(m_constr_s[i])
                mw_perp = mw_s[i] - np.dot(mw_s[i], u) * u
                c_s[i] = 2.0 * lam * mw_perp
                ep += lam * float(np.dot(mw_perp, mw_perp))
            elif i_constrained_m == 2:
                diff = mw_s[i] - m_constr_s[i]
                c_s[i] = 2.0 * lam * diff
                ep += lam * float(np.dot(diff, diff))
            elif i_constrained_m == 4:
                # VASP 6.4+ linear penalty: E_p = lam*(|MW| - u.MW), C = lam*(MW_hat - u)
                u = m_constr_s[i] / np.linalg.norm(m_constr_s[i])
                mw_norm = float(np.linalg.norm(mw_s[i]))
                mw_hat = mw_s[i] / mw_norm
                c_s[i] = lam * (mw_hat - u)
                ep += lam * (mw_norm - float(np.dot(u, mw_s[i])))

    if flip_field_sign:
        c_s = -c_s

    # VASP 5 prints 1*lambda*MW_perp (= C/2) for quadratic modes; VASP 6 prints C directly.
    printed_s = (0.5 * c_s) if (major == 5 and i_constrained_m != 4) else c_s
    ep_printed = ep * corrupt_ep_factor

    f_raw = -24.50000000 + ep
    e0_raw = -24.51000000 + ep

    m_constr_tokens = " ".join(f"{x:.8f}" for x in m_constr_s.reshape(-1))
    saxis_tokens = " ".join(f"{x:.8f}" for x in saxis)
    incar_lines = [
        "   LNONCOLLINEAR = .TRUE.",
        "   LSORBIT = .TRUE.",
        f"   I_CONSTRAINED_M = {i_constrained_m}",
        f"   SAXIS = {saxis_tokens}",
        "   NSW = 0",
        "   NELM = 60",
    ]
    if i_constrained_m in (1, 2, 4):
        incar_lines.extend(
            [
                f"   LAMBDA = {lam:.4f}",
                "   RWIGS = 1.30 0.82",
                f"   M_CONSTR = {m_constr_tokens}",
            ]
        )
    incar_block = "\n".join(incar_lines) + "\n"

    if write_sibling_incar:
        (directory / incar_name).write_text(incar_block, encoding="utf-8")

    incar_echo_section = (
        f" INCAR:\n{incar_block}\n POTCAR: PAW_PBE Fe\n"
        if include_incar_echo
        else " POTCAR: PAW_PBE Fe\n"
    )
    saxis_print = "\n".join(
        f" {rot[r, 0]:12.8f} m_x {rot[r, 1]:12.8f} m_y {rot[r, 2]:12.8f} m_z"
        for r in range(3)
    )
    born_section = ""
    if include_born:
        ionic_piezo_block = ""
        if include_ionic_piezo:
            ionic_piezo_block = (
                " PIEZOELECTRIC TENSOR IONIC CONTR  for field in x, y, z        (e  Angst)\n"
                "          XX          YY          ZZ          XY          YZ          ZX\n"
                " -------------------------------------------------------------------------\n"
                " x     9.10000     9.20000     9.30000     9.01000     9.02000     9.03000\n"
                " y     9.04000     9.05000     9.06000     9.07000     9.08000     9.09000\n"
                " z     9.11000     9.12000     9.13000     9.14000     9.15000     9.16000\n"
                " PIEZOELECTRIC TENSOR IONIC CONTR  for field in x, y, z        (C/m^2)\n"
            )
        born_section = (
            "\n BORN EFFECTIVE CHARGES (including local field effects) (in |e|, cumulative output)\n"
            " -------------------------------------------------\n"
            " ion    1\n"
            "    1     2.10000     0.05000    -0.02000\n"
            "    2     0.01000     2.20000     0.03000\n"
            "    3    -0.04000     0.02000     2.30000\n"
            " ion    2\n"
            "    1     2.10000     0.00000     0.00000\n"
            "    2     0.00000     2.20000     0.00000\n"
            "    3     0.00000     0.00000     2.30000\n"
            " ion    3\n"
            "    1    -4.20000    -0.05000     0.02000\n"
            "    2    -0.01000    -4.40000    -0.03000\n"
            "    3     0.04000    -0.02000    -4.60000\n"
            " CHARGE: end\n"
            " PIEZOELECTRIC TENSOR  for field in x, y, z        (e  Angst)\n"
            "          XX          YY          ZZ          XY          YZ          ZX\n"
            " -------------------------------------------------------------------------\n"
            " x     0.10000     0.20000     0.30000     0.01000     0.02000     0.03000\n"
            " y     0.04000     0.05000     0.06000     0.07000     0.08000     0.09000\n"
            " z     0.11000     0.12000     0.13000     0.14000     0.15000     0.16000\n"
            " PIEZOELECTRIC TENSOR  for field in x, y, z        (C/m^2)\n"
            + ionic_piezo_block
        )

    runtime_constr = (
        f"   I_CONSTRAINED_M = {i_constrained_m}\n   LAMBDA = {lam:.4f}\n   RWIGS = 1.30 0.82\n"
        if i_constrained_m in (1, 2, 4)
        else "   I_CONSTRAINED_M = 0\n"
    )
    outcar_text = f""" vasp.{version} 01Jan25 (build) complex
{incar_echo_section}
   VRHFIN =Fe: d7 s1
   VRHFIN =O: s2p4
   ions per type =               2   1
 Dimension of arrays:
   NIONS =      3
   NSW   =      0
   NELM  =     60
   LNONCOLLINEAR = T
   LSORBIT = T
{runtime_constr}   SAXIS = {saxis_tokens}
 transformation matrix from SAXIS to cartesian coordinates
 ---------------------------------------------------------
{saxis_print}

--------------------------------------- Iteration      1(   5) ---------------------------------------
 direct lattice vectors                 reciprocal lattice vectors
     5.000000000  0.000000000  0.000000000     0.200000000  0.000000000  0.000000000
     0.000000000  5.000000000  0.000000000     0.000000000  0.200000000  0.000000000
     0.000000000  0.000000000  5.000000000     0.000000000  0.000000000  0.200000000

 in kB    10.00000   20.00000   30.00000    1.00000    2.00000    3.00000
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000      0.00000      0.00000         0.120000     -0.050000      0.030000
      2.50000      2.50000      0.00000        -0.100000      0.040000     -0.010000
      1.25000      1.25000      2.50000        -0.020000      0.010000     -0.020000
 -----------------------------------------------------------------------------------
 magnetization (x)
 # of ion       s       p       d       tot
 ------------------------------------------
    1        0.000   0.000   {m_int_s[0, 0]:7.4f}   {m_int_s[0, 0]:7.4f}
    2        0.000   0.000   {m_int_s[1, 0]:7.4f}   {m_int_s[1, 0]:7.4f}
    3        0.000   0.000   {m_int_s[2, 0]:7.4f}   {m_int_s[2, 0]:7.4f}
 magnetization (y)
 # of ion       s       p       d       tot
 ------------------------------------------
    1        0.000   0.000   {m_int_s[0, 1]:7.4f}   {m_int_s[0, 1]:7.4f}
    2        0.000   0.000   {m_int_s[1, 1]:7.4f}   {m_int_s[1, 1]:7.4f}
    3        0.000   0.000   {m_int_s[2, 1]:7.4f}   {m_int_s[2, 1]:7.4f}
 magnetization (z)
 # of ion       s       p       d       tot
 ------------------------------------------
    1        0.000   0.000   {m_int_s[0, 2]:7.4f}   {m_int_s[0, 2]:7.4f}
    2        0.000   0.000   {m_int_s[1, 2]:7.4f}   {m_int_s[1, 2]:7.4f}
    3        0.000   0.000   {m_int_s[2, 2]:7.4f}   {m_int_s[2, 2]:7.4f}
{born_section}
  free  energy   TOTEN  =       {f_raw:16.8f} eV
  energy  without entropy=      {e0_raw:16.8f}  energy(sigma->0) =      {e0_raw:16.8f}
{"------------------------ aborting loop because EDIFF is reached ----------------------------------------" if converged else ""}
{"General timing and accounting informations for this job:" if converged else ""}
"""

    mw_rows = "\n".join(
        f"  {i + 1:3d}  {mw_s[i, 0]:11.6f} {mw_s[i, 1]:11.6f} {mw_s[i, 2]:11.6f}"
        f"  {m_int_s[i, 0]:11.6f} {m_int_s[i, 1]:11.6f} {m_int_s[i, 2]:11.6f}"
        for i in range(3)
    )
    # Sparse lambda*MW_perp table: ONLY constrained ions 1 and 2 (ion 3 is unconstrained!)
    field_rows = "\n".join(
        f"  {i + 1:3d}  {printed_s[i, 0]:12.7f} {printed_s[i, 1]:12.7f} {printed_s[i, 2]:12.7f}"
        for i in range(2)
    )
    if i_constrained_m in (1, 2, 4):
        oszicar_text = f""" E_p = {ep_printed:15.8E}  lambda = {lam:10.4E}
  ion        MW_int                             M_int
{mw_rows}
{scf_tag}:   5    {f_raw:16.8E}   -0.10000E-06   -0.10000E-06   100   0.100E-04   0.100E-05
   1 F= {f_raw:16.8E} E0= {e0_raw:16.8E}  d E = {f_raw:16.8E}  mag=     3.5000     0.0000     3.5000
 E_p = {ep_printed:15.8E}  lambda = {lam:10.4E}
  ion    lambda*MW_perp
{field_rows}
"""
    else:
        oszicar_text = f"""{scf_tag}:   5    {f_raw:16.8E}   -0.10000E-06   -0.10000E-06   100   0.100E-04   0.100E-05
   1 F= {f_raw:16.8E} E0= {e0_raw:16.8E}  d E = {f_raw:16.8E}  mag=     3.5000     0.0000     3.5000
"""
    out_p = directory / outcar_name
    osz_p = directory / oszicar_name
    out_p.write_text(outcar_text, encoding="utf-8")
    osz_p.write_text(oszicar_text, encoding="utf-8")
    return (
        out_p,
        osz_p,
        {"ep": ep, "c_cart": c_s @ rot.T, "mw_cart": mw_cart, "m_int_cart": m_int_cart},
    )


def test_vasp5_and_vasp6_produce_identical_physics_with_saxis_and_nonmag(tmp_path):
    """VASP 5.4.4 (no INCAR echo, 1x field) & VASP 6.4.0/6.5.1 (2x field) match identically."""
    saxis = (1.0, 1.0, 1.0)
    out5, osz5, ref = _make_vasp_pair(
        tmp_path / "vasp5",
        version="5.4.4",
        saxis=saxis,
        include_incar_echo=False,
        write_sibling_incar=True,
        scf_tag="CGA",  # ALGO=All electronic step tag
    )
    out64, osz64, _ = _make_vasp_pair(
        tmp_path / "vasp64", version="6.4.0", saxis=saxis, include_incar_echo=True
    )
    out65, osz65, _ = _make_vasp_pair(
        tmp_path / "vasp65", version="6.5.1", saxis=saxis, include_incar_echo=True
    )

    f5 = build_extxyz_frame(out5, osz5, keep_raw=True)
    f64 = build_extxyz_frame(out64, osz64, keep_raw=True)
    f65 = build_extxyz_frame(out65, osz65, keep_raw=True)
    assert f5 is not None and f64 is not None and f65 is not None

    assert f5.info["vasp_field_scale"] == 2.0
    assert f5.info["constraint_consistency_ratio"] == pytest.approx(1.0, abs=1e-4)
    assert f64.info["vasp_field_scale"] == 1.0
    assert f64.info["constraint_consistency_ratio"] == pytest.approx(4.0, abs=1e-4)
    assert f65.info["vasp_field_scale"] == 1.0

    for fr in (f5, f64, f65):
        assert fr.get_potential_energy() == pytest.approx(-24.5, abs=1e-6)
        assert fr.info["energy_policy"] == "F_minus_Ep"

    for fr in (f64, f65):
        np.testing.assert_allclose(
            f5.get_initial_magnetic_moments(),
            fr.get_initial_magnetic_moments(),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            f5.arrays["spin_length"], fr.arrays["spin_length"], atol=1e-6
        )
        np.testing.assert_allclose(
            f5.arrays["spin_forces_vert"], fr.arrays["spin_forces_vert"], atol=1e-5
        )
        np.testing.assert_allclose(f5.get_forces(), fr.get_forces(), atol=1e-8)
        np.testing.assert_allclose(f5.get_stress(), fr.get_stress(), atol=1e-10)

    # Exact tangential projection onto default conjugate mw_int & orthogonality to spin directions
    mw = ref["mw_cart"]
    c_cart = ref["c_cart"]
    s_hat = np.zeros_like(mw)
    s_hat[:2] = mw[:2] / np.linalg.norm(mw[:2], axis=1, keepdims=True)
    expected_sfv = np.zeros_like(c_cart)
    for i in range(2):
        expected_sfv[i] = np.linalg.norm(mw[i]) * (
            c_cart[i] - np.dot(c_cart[i], s_hat[i]) * s_hat[i]
        )
    np.testing.assert_allclose(f5.arrays["spin_forces_vert"], expected_sfv, atol=1e-5)
    assert f5.info["spin_source"] == "mw_int"
    # Energy is invariant to -m / read_magnetic on a constrained run
    f5_nomag = build_extxyz_frame(out5, osz5, read_magnetic=False)
    assert f5_nomag is not None
    assert f5_nomag.get_potential_energy() == pytest.approx(
        f5.get_potential_energy(), abs=1e-12
    )
    assert np.allclose(
        np.sum(
            f5.arrays["spin_forces_vert"] * f5.get_initial_magnetic_moments(), axis=1
        ),
        0.0,
        atol=1e-12,
    )
    assert list(f5.arrays["magnetic_mask"]) == [True, True, False]
    np.testing.assert_allclose(f5.get_initial_magnetic_moments()[2], 0.0, atol=1e-12)
    np.testing.assert_allclose(f5.arrays["spin_forces_vert"][2], 0.0, atol=1e-12)


def test_vasp5_without_incar_and_indexed_incar_pairing(tmp_path):
    """VASP 5 with indexed OUTCAR.1/OSZICAR.1/INCAR.1 and without INCAR both work."""
    outcar, _, _ = _make_vasp_pair(
        tmp_path / "no_incar",
        version="5.4.4",
        saxis=(0.0, 1.0, 0.0),
        include_incar_echo=False,
        write_sibling_incar=False,
        outcar_name="OUTCAR.1",
        oszicar_name="OSZICAR.1",
    )
    frame = build_extxyz_frame(outcar, read_magnetic=True)
    assert frame is not None
    assert frame.info["vasp_field_scale"] == 2.0
    assert list(frame.arrays["magnetic_mask"]) == [True, True, False]

    # Indexed INCAR.1 should be preferred over generic INCAR
    out_idx, _, _ = _make_vasp_pair(
        tmp_path / "idx_incar",
        version="5.4.4",
        include_incar_echo=False,
        write_sibling_incar=True,
        incar_name="INCAR.1",
        outcar_name="OUTCAR.1",
        oszicar_name="OSZICAR.1",
    )
    (tmp_path / "idx_incar" / "INCAR").write_text("LAMBDA = 999.0\n", encoding="utf-8")
    fr_idx = build_extxyz_frame(out_idx, read_magnetic=True)
    assert fr_idx is not None and fr_idx.info["lambda"] == pytest.approx(10.0)

    # Explicit nonexistent --incar must raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="INCAR not found"):
        build_extxyz_frame(out_idx, incar_path=tmp_path / "missing_INCAR")


def test_mode2_mode4_and_quantization_aware_scale_detection(tmp_path):
    """Exact mode 2 (target vector + sign flip), exact mode 4 linear penalty, and F10.5 rounding."""
    out_m2_5, osz_m2_5, _ = _make_vasp_pair(
        tmp_path / "m2_5",
        version="5.4.4",
        i_constrained_m=2,
        include_incar_echo=False,
        write_sibling_incar=True,
    )
    out_m2_6, osz_m2_6, _ = _make_vasp_pair(
        tmp_path / "m2_6", version="6.4.0", i_constrained_m=2
    )
    f5 = build_extxyz_frame(out_m2_5, osz_m2_5)
    f6 = build_extxyz_frame(out_m2_6, osz_m2_6)
    assert f5 is not None and f6 is not None
    assert f5.info["vasp_field_scale"] == 2.0 and f6.info["vasp_field_scale"] == 1.0
    np.testing.assert_allclose(
        f5.arrays["spin_forces_vert"], f6.arrays["spin_forces_vert"], atol=1e-5
    )

    # Mode 2 flipped sign passes R check alone, but MUST be rejected by M_CONSTR vector check!
    out_m2_bad, osz_m2_bad, _ = _make_vasp_pair(
        tmp_path / "m2_flip", version="6.4.0", i_constrained_m=2, flip_field_sign=True
    )
    with pytest.raises(VaspSpinParseError, match="Mode-2 field mismatch"):
        build_extxyz_frame(out_m2_bad, osz_m2_bad)

    # Mode 4: exact VASP 6.4+ linear functional E_p = lam*(|MW| - u.MW), C = lam*(MW_hat - u), R_4 = 2.0
    out_m4, osz_m4, _ = _make_vasp_pair(
        tmp_path / "m4", version="6.4.0", i_constrained_m=4
    )
    fm4 = build_extxyz_frame(out_m4, osz_m4)
    assert fm4 is not None
    assert fm4.info["vasp_field_scale"] == 1.0
    assert fm4.info["constraint_consistency_ratio"] == pytest.approx(2.0, abs=1e-3)

    # Zero-torque fallback + F10.5 near-collinear printing quantization
    out_z5, osz_z5, _ = _make_vasp_pair(
        tmp_path / "z5", version="5.4.4", zero_torque=True, include_incar_echo=False
    )
    fz5 = build_extxyz_frame(out_z5, osz_z5)
    assert fz5 is not None and fz5.info["vasp_field_scale"] == 2.0
    np.testing.assert_allclose(fz5.arrays["spin_forces_vert"], 0.0, atol=1e-12)

    # Requesting VASP 6 on VASP 5 zero-torque run must still reject header contradiction!
    with pytest.raises(VaspSpinParseError, match="contradicts OUTCAR VASP 5.x header"):
        build_extxyz_frame(out_z5, osz_z5, vasp_version_mode="6")

    # Near-collinear F10.5 quantized VASP 5 frame (printed 0.00010 vs true 9.51e-5 -> R=1.1057)
    scale_q, _ = detect_field_scale(
        np.array([[0.00010, 0.0, 0.0]]),
        penalty_energy=9.044e-10,
        lam=10.0,
        vasp_major=5,
    )
    assert scale_q == 2.0

    # Contradictory E_p vs P must still raise
    with pytest.raises(VaspSpinParseError, match="Inconsistent OSZICAR penalty"):
        detect_field_scale(np.zeros((2, 3)), penalty_energy=0.5, lam=10.0, vasp_major=6)
    with pytest.raises(VaspSpinParseError, match="Inconsistent OSZICAR penalty"):
        detect_field_scale(
            np.array([[1.0, 0.0, 0.0]]), penalty_energy=0.0, lam=10.0, vasp_major=6
        )
    with pytest.raises(VaspSpinParseError, match="Negative OSZICAR penalty"):
        detect_field_scale(
            np.zeros((1, 3)), penalty_energy=-0.1, lam=10.0, vasp_major=6
        )


def test_silent_ep_contamination_and_corruption_blocked(tmp_path):
    """Missing/truncated OSZICAR on constrained OUTCAR, cross-step leaks, and version conflicts fail loudly."""
    out_c, osz_c, _ = _make_vasp_pair(
        tmp_path / "missing_osz", version="6.4.0", i_constrained_m=1
    )
    osz_c.unlink()
    with pytest.raises(VaspSpinParseError, match="refusing to emit E_p-contaminated"):
        build_extxyz_frame(out_c, read_magnetic=True)

    # Truncated OSZICAR missing final lambda*MW_perp block
    out_tr, osz_tr, _ = _make_vasp_pair(
        tmp_path / "trunc_osz", version="6.4.0", i_constrained_m=1
    )
    osz_lines = osz_tr.read_text(encoding="utf-8").splitlines()
    osz_tr.write_text("\n".join(osz_lines[:-4]) + "\n", encoding="utf-8")
    with pytest.raises(
        VaspSpinParseError, match="incomplete/truncated constrained OSZICAR"
    ):
        build_extxyz_frame(out_tr, osz_tr, read_magnetic=True)

    # Cross-ionic-step leak: step 2 summary lacks MW_int -> must NOT reuse step 1 MW_int!
    out_leak, osz_leak, _ = _make_vasp_pair(tmp_path / "step_leak", version="6.4.0")
    osz_txt = osz_leak.read_text(encoding="utf-8")
    # Insert a step-2 summary right before the final E_p / lambda*MW_perp block
    marker = "\n E_p ="
    parts = osz_txt.rsplit(marker, 1)
    leaked = (
        parts[0]
        + "\n   2 F= -24.50000000E+00 E0= -24.51000000E+00  d E = 0.0  mag= 3.5 0.0 3.5"
        + marker
        + parts[1]
    )
    osz_leak.write_text(leaked, encoding="utf-8")
    with pytest.raises(
        VaspSpinParseError,
        match="missing converged SCF step|no matching ionic-step MW_int",
    ):
        build_extxyz_frame(out_leak, osz_leak)

    # Wrong --vasp-version flag vs penalty ratio
    out5, osz5, _ = _make_vasp_pair(
        tmp_path / "v5_flag",
        version="5.4.4",
        include_incar_echo=False,
        write_sibling_incar=True,
    )
    with pytest.raises(VaspSpinParseError, match="contradicts OSZICAR penalty ratio"):
        build_extxyz_frame(out5, osz5, vasp_version_mode="6")

    # Corrupted E_p ratio
    out_bad, osz_bad, _ = _make_vasp_pair(
        tmp_path / "bad_ep", version="6.4.0", corrupt_ep_factor=2.5
    )
    with pytest.raises(VaspSpinParseError, match="Inconsistent OSZICAR penalty"):
        build_extxyz_frame(out_bad, osz_bad)


def test_merged_fortran_columns_and_isif3_cell_and_relax_convergence(tmp_path):
    """Merged negative Fortran columns parse cleanly; ISIF=3 uses step-k cell; unconverged relax skipped."""
    outcar, oszicar, _ = _make_vasp_pair(tmp_path / "merged", version="6.4.0", lam=10.0)
    # Inject merged negative stress columns and merged negative force columns
    txt = outcar.read_text(encoding="utf-8").replace(
        " in kB    10.00000   20.00000   30.00000    1.00000    2.00000    3.00000",
        " in kB -1023.45678-1024.50000-1025.25000    1.00000    2.00000    3.00000",
    )
    txt = txt.replace(
        "      0.00000      0.00000      0.00000         0.120000     -0.050000      0.030000",
        "      0.00000      0.00000      0.00000      -10.120000-100.050000-200.030000",
    )
    outcar.write_text(txt, encoding="utf-8")
    fr = build_extxyz_frame(outcar, oszicar)
    assert fr is not None
    assert fr.get_stress() is not None
    np.testing.assert_allclose(
        fr.get_forces()[0], [-10.12, -100.05, -200.03], atol=1e-6
    )

    # Fortran overflow '**********' in lambda*MW_perp must raise rather than silently zero
    osz_ovf = oszicar.read_text(encoding="utf-8")
    osz_lines = osz_ovf.splitlines()
    osz_lines[-2] = "    1  **********   0.00000   0.00000"
    oszicar.write_text("\n".join(osz_lines) + "\n", encoding="utf-8")
    with pytest.raises(VaspSpinParseError, match="Fortran overflow"):
        build_extxyz_frame(outcar, oszicar)

    # ISIF=3: cell printed AFTER TOTAL-FORCE (for step k+1) must NOT overwrite step k's cell;
    # and NSW=2 IBRION=2 finishing at step 2 without "reached required accuracy" is unconverged!
    out_isif, _, _ = _make_vasp_pair(
        tmp_path / "isif3", version="6.4.0", i_constrained_m=0
    )
    isif_txt = out_isif.read_text(encoding="utf-8")
    isif_txt = isif_txt.replace("NSW   =      0", "NSW   =      2\n   IBRION =      2")
    isif_txt = isif_txt.replace("NSW = 0", "NSW = 2\n   IBRION = 2")
    isif_txt = isif_txt.replace("Iteration      1(   5)", "Iteration      2(   5)")
    # Append next-step lattice a=9.9 AFTER TOTEN (VASP ISIF=3 order)
    isif_txt = isif_txt.replace(
        "General timing and accounting informations for this job:",
        " direct lattice vectors                 reciprocal lattice vectors\n"
        "     9.900000000  0.000000000  0.000000000     0.101010101  0.000000000  0.000000000\n"
        "     0.000000000  9.900000000  0.000000000     0.000000000  0.101010101  0.000000000\n"
        "     0.000000000  0.000000000  9.900000000     0.000000000  0.000000000  0.101010101\n"
        "General timing and accounting informations for this job:",
    )
    out_isif.write_text(isif_txt, encoding="utf-8")
    parsed_unconv = parse_outcar(out_isif, read_magnetic=False)
    assert parsed_unconv.converged is False
    np.testing.assert_allclose(np.diag(parsed_unconv.cell), [5.0, 5.0, 5.0], atol=1e-8)

    # With "reached required accuracy" added, it IS converged and still keeps step-2 cell (5.0)!
    out_isif.write_text(
        isif_txt.replace(
            "General timing",
            "reached required accuracy - stopping structural energy minimisation\nGeneral timing",
        ),
        encoding="utf-8",
    )
    parsed_conv = parse_outcar(out_isif, read_magnetic=False)
    assert parsed_conv.converged is True
    np.testing.assert_allclose(np.diag(parsed_conv.cell), [5.0, 5.0, 5.0], atol=1e-8)


def test_unconstrained_magnetic_site_and_real_ase_vasp53_fixture(tmp_path):
    """Unconstrained magnetic atoms keep spin_length; nomagz=(0,) falls through; ASE VASP 5.3.3 works."""
    outcar, oszicar, _ = _make_vasp_pair(
        tmp_path / "unconstr_site", version="6.4.0", unconstrained_mag_site2=True
    )
    fr = build_extxyz_frame(outcar, oszicar, nomagz=(0,))
    assert fr is not None
    # Site 2 is unconstrained (M_CONSTR=0) but has real moment 2.4 mu_B -> kept magnetic, zero torque!
    assert list(fr.arrays["magnetic_mask"]) == [True, True, True]
    assert fr.arrays["spin_length"][2] == pytest.approx(2.4, abs=1e-4)
    np.testing.assert_allclose(fr.arrays["spin_forces_vert"][2], 0.0, atol=1e-12)

    # Real ASE VASP 5.3.3 fixture (POTCAR-only species without VRHFIN, empty INCAR: echo)
    ase_outcar = next(Path(ase.__file__).parent.rglob("OUTCAR_example_1"))
    fr_real = build_extxyz_frame(ase_outcar, read_magnetic=False)
    assert fr_real is not None
    assert len(fr_real) == 18
    assert set(fr_real.get_chemical_symbols()) == {"Ni"}
    assert fr_real.get_potential_energy() == pytest.approx(-68.22868532, abs=1e-6)


def test_parallel_cpu_batch_and_bec_extxyz_roundtrip(tmp_path):
    """Multi-core --cpu 4 matches --cpu 1 deterministically across mixed VASP 5/6 + BEC + piezo."""
    root = tmp_path / "runs"
    for idx in range(6):
        ver = "5.4.4" if idx % 2 == 0 else "6.5.1"
        _make_vasp_pair(
            root / f"frame_{idx:02d}",
            version=ver,
            saxis=(1.0, 0.0, 1.0),
            include_incar_echo=(ver.startswith("6")),
            write_sibling_incar=(ver.startswith("5")),
            converged=(idx != 3),  # frame 3 unconverged -> skipped
            include_born=True,
            include_ionic_piezo=True,  # must NOT overwrite clamped-ion tensor with 9.x!
        )

    out_seq = tmp_path / "seq.extxyz"
    out_par = tmp_path / "par.extxyz"
    s1 = convert_directory_to_extxyz(
        root, out_seq, cpu=1, read_magnetic=True, read_born=True, energy_source="zero"
    )
    s4 = convert_directory_to_extxyz(
        root, out_par, cpu=4, read_magnetic=True, read_born=True, energy_source="zero"
    )
    assert s1["discovered"] == s4["discovered"] == 6
    assert s1["written"] == s4["written"] == 5
    assert s1["unconverged"] == s4["unconverged"] == 1
    assert s1["errors"] == s4["errors"] == []

    frames_1 = ase_read(str(out_seq), index=":", format="extxyz")
    frames_4 = ase_read(str(out_par), index=":", format="extxyz")
    assert len(frames_1) == len(frames_4) == 5
    for a, b in zip(frames_1, frames_4):
        assert a.get_potential_energy() == pytest.approx(-24.51, abs=1e-6)
        assert a.get_potential_energy() == pytest.approx(
            b.get_potential_energy(), abs=1e-12
        )
        np.testing.assert_allclose(a.get_forces(), b.get_forces(), atol=1e-12)
        np.testing.assert_allclose(
            a.arrays["spin_forces_vert"], b.arrays["spin_forces_vert"], atol=1e-12
        )
        bec_a = (
            a.arrays["born_effective_charges"]
            if "born_effective_charges" in a.arrays
            else a.calc.results["born_effective_charges"]
        )
        bec_b = (
            b.arrays["born_effective_charges"]
            if "born_effective_charges" in b.arrays
            else b.calc.results["born_effective_charges"]
        )
        np.testing.assert_allclose(bec_a, bec_b, atol=1e-12)
        assert bec_a.shape == (3, 9)
        assert a.info["piezoelectric"].shape == (3, 3, 3)
        # Verify clamped-ion piezoelectric tensor (0.10, NOT 9.10 ionic contr!)
        assert a.info["piezoelectric"][0, 0, 0] == pytest.approx(0.10, abs=1e-6)

    # Verify SpinGNN2 AtomicData.from_ase consumes the extxyz frame cleanly.
    from nequip.data import AtomicData, AtomicDataDict

    graph = AtomicData.from_ase(frames_1[0], r_max=4.0)
    assert float(graph[AtomicDataDict.TOTAL_ENERGY_KEY].item()) == pytest.approx(
        -24.51, abs=1e-5
    )
    assert tuple(graph[AtomicDataDict.FORCE_KEY].shape) == (3, 3)
    assert tuple(graph[AtomicDataDict.STRESS_KEY].shape) == (3, 3)
    assert tuple(graph[AtomicDataDict.SPIN_KEY].shape) == (3, 3)
    assert tuple(graph[AtomicDataDict.SPIN_LENGTH_KEY].shape) == (3, 1)
    assert tuple(graph[AtomicDataDict.SPIN_FORCE_VERT_KEY].shape) == (3, 3)
    assert tuple(graph[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY].shape) == (3, 9)
    assert tuple(graph[AtomicDataDict.PIEZOELECTRIC_KEY].shape) == (3, 3, 3)


def test_structure_only_and_unconstrained_magnetic_modes(tmp_path):
    """Genuine unconstrained (I_CONSTRAINED_M=0) OUTCAR works in both structural and magnetic modes."""
    outcar, oszicar, ref = _make_vasp_pair(
        tmp_path / "unconstrained",
        version="5.4.4",
        i_constrained_m=0,
        include_incar_echo=False,
    )
    oszicar.unlink()

    struct_frame = build_extxyz_frame(outcar, read_magnetic=False)
    assert struct_frame is not None
    assert "spin_forces_vert" not in struct_frame.arrays
    assert "spin_length" not in struct_frame.arrays
    assert struct_frame.get_potential_energy() == pytest.approx(-24.5, abs=1e-6)

    mag_frame = build_extxyz_frame(outcar, read_magnetic=True)
    assert mag_frame is not None
    assert "spin_forces_vert" not in mag_frame.arrays
    assert mag_frame.arrays["spin_length"].shape == (3,)
    assert mag_frame.info["energy_policy"] == "F_raw"
    np.testing.assert_allclose(
        mag_frame.arrays["spin_length"],
        np.linalg.norm(ref["m_int_cart"], axis=1),
        atol=1e-3,
    )


def test_spingnn_data_cli_guards_and_backward_compat(tmp_path):
    """CLI `spingnn-data`, safety guards, backward-compat wrapper, and zero legacy `nequip-*` CLIs."""
    root = tmp_path / "cli_runs"
    out5, osz5, _ = _make_vasp_pair(
        root / "a", version="5.4.4", include_incar_echo=False, write_sibling_incar=True
    )
    _make_vasp_pair(root / "b", version="6.4.0", include_incar_echo=True)

    help_proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools/ci/source_python.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "spingnn-train" in help_proc.stdout
    assert "spingnn-evaluate" in help_proc.stdout
    assert "spingnn-benchmark" in help_proc.stdout
    assert "spingnn-deploy" in help_proc.stdout
    assert "spingnn-data" in help_proc.stdout
    assert "spingnn-data-mlff-bec" in help_proc.stdout
    assert "spingnn-data-delta-mag" in help_proc.stdout
    assert "nequip-" not in help_proc.stdout

    # Core-only PYTHONPATH backward compatibility of nequip.scripts.vasp_data
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src/core'}:{REPO_ROOT / 'src/models'}"
    compat_proc = subprocess.run(
        [sys.executable, "-m", "nequip.scripts.vasp_data", "--help"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "spingnn-data" in compat_proc.stdout

    # CLI guards: --oszicar without --outcar, and --vasp6 --vasp-version 5
    bad_osz = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/ci/source_python.py"),
            "spingnn-data",
            "--oszicar",
            str(osz5),
        ],
        capture_output=True,
        text=True,
    )
    assert bad_osz.returncode == 2
    bad_ver = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/ci/source_python.py"),
            "spingnn-data",
            "--vasp-version",
            "5",
            "--vasp6",
        ],
        capture_output=True,
        text=True,
    )
    assert bad_ver.returncode == 2

    # Refuses to overwrite or delete source OUTCAR / auto-paired OSZICAR in single & batch modes
    with pytest.raises(
        ValueError, match="Refusing to overwrite or remove input VASP file"
    ):
        convert_pair_to_extxyz(out5, None, osz5, read_magnetic=True)
    with pytest.raises(
        ValueError, match="Refusing to overwrite or remove input VASP file"
    ):
        convert_directory_to_extxyz(out5, out5, cpu=1, read_magnetic=True)
    assert out5.exists() and osz5.exists()

    # Batch mode via CLI with --cpu 2
    batch_out = tmp_path / "batch.extxyz"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/ci/source_python.py"),
            "spingnn-data",
            "-m",
            "-p",
            str(root),
            "-o",
            str(batch_out),
            "-c",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = ase_read(str(batch_out), index=":", format="extxyz")
    assert len(loaded) == 2

    # Single-pair mode
    single_out = tmp_path / "single.extxyz"
    convert_pair_to_extxyz(out5, osz5, single_out, read_magnetic=True)
    single_loaded = ase_read(str(single_out), format="extxyz")
    assert single_loaded.arrays["spin_forces_vert"].shape == (3, 3)


def test_codex_adversarial_edge_cases(tmp_path):
    """Regression suite for trailing unfinished steps, stray site fields, overflow, soft-stop, and mode-4 rounding."""
    # 1. Trailing unfinished next ionic step with flipped field must NOT overwrite previous step!
    out_t, osz_t, _ = _make_vasp_pair(
        tmp_path / "trailing",
        version="5.4.4",
        include_incar_echo=False,
        write_sibling_incar=True,
    )
    osz_orig = osz_t.read_text(encoding="utf-8")
    tail = osz_orig.replace("DAV:   5", "DAV:   1").splitlines()
    # Append step-2 MW_int + DAV:1 + field WITHOUT step-2 summary
    osz_t.write_text(
        osz_orig + "\n".join(tail[:5]) + "\n" + "\n".join(tail[-4:]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(VaspSpinParseError, match="unexpected trailing"):
        build_extxyz_frame(out_t, osz_t)

    # 2. Stray constraint field on inactive site 3 rejected by all-site M_CONSTR check
    out_s, osz_s, _ = _make_vasp_pair(
        tmp_path / "stray", version="6.4.0", zero_torque=True
    )
    osz_s.write_text(
        osz_s.read_text(encoding="utf-8")
        + "    3    0.5000000    0.0000000    0.0000000\n",
        encoding="utf-8",
    )
    with pytest.raises(VaspSpinParseError):
        build_extxyz_frame(out_s, osz_s)

    # 3. I_CONSTRAINED_M=0 OUTCAR paired with constrained OSZICAR rejected
    out_u, osz_u, _ = _make_vasp_pair(
        tmp_path / "u_out", version="6.4.0", i_constrained_m=0
    )
    _, osz_c, _ = _make_vasp_pair(
        tmp_path / "c_osz", version="6.4.0", i_constrained_m=1
    )
    with pytest.raises(VaspSpinParseError, match="I_CONSTRAINED_M=0"):
        build_extxyz_frame(out_u, osz_c, read_magnetic=False)

    # 4. Lattice Fortran overflow '***********' must NOT pull reciprocal cell into direct cell
    out_ov, _, _ = _make_vasp_pair(
        tmp_path / "lat_ov", version="6.4.0", i_constrained_m=0
    )
    out_ov.write_text(
        out_ov.read_text(encoding="utf-8").replace(
            "5.000000000  0.000000000  0.000000000",
            "5.000000000  ***********  0.000000000",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(VaspSpinParseError, match="Fortran overflow in direct lattice"):
        parse_outcar(out_ov, read_magnetic=False)

    # 5. Soft-stop relaxation (last_ionic < NSW without 'reached required accuracy') is unconverged
    out_ss, _, _ = _make_vasp_pair(
        tmp_path / "softstop", version="6.4.0", i_constrained_m=0
    )
    out_ss.write_text(
        out_ss.read_text(encoding="utf-8")
        .replace("NSW   =      0", "NSW   =      5\n   IBRION =      2")
        .replace("NSW = 0", "NSW = 5\n   IBRION = 2"),
        encoding="utf-8",
    )
    assert parse_outcar(out_ss, read_magnetic=False).converged is False

    # 6. Mode 4 printing uncertainty on tiny |MW_int| (4.6e-5 printed as 5e-5 -> R4=2.1739) accepted
    scale_m4, _ = detect_field_scale(
        np.array([[10.0, 0.0, 0.0]]),
        penalty_energy=2.3e-4,
        lam=10.0,
        vasp_major=6,
        mw_int=np.array([[0.0, 0.0, 5e-5]]),
        i_constrained_m=4,
    )
    assert scale_m4 == 1.0

    # 7. Duplicate piezoelectric axis row rejected
    out_pz, osz_pz, _ = _make_vasp_pair(
        tmp_path / "bad_piezo", version="6.4.0", include_born=True
    )
    out_pz.write_text(
        out_pz.read_text(encoding="utf-8").replace(
            " y     0.04000     0.05000", " x     0.04000     0.05000", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(VaspSpinParseError, match="duplicate piezoelectric axis row"):
        build_extxyz_frame(out_pz, osz_pz, read_born=True)

    # 8. Negative E_p rejected even in structure-only mode (read_magnetic=False)
    out_neg, osz_neg, _ = _make_vasp_pair(tmp_path / "neg_ep", version="6.4.0")
    osz_neg.write_text(
        osz_neg.read_text(encoding="utf-8").replace("E_p =  ", "E_p = -"),
        encoding="utf-8",
    )
    with pytest.raises(VaspSpinParseError, match="Negative OSZICAR penalty"):
        build_extxyz_frame(out_neg, osz_neg, read_magnetic=False)

    # 9. All-zero M_CONSTR with tiny nonzero field (0.005 > F10.5 half-ULP) rejected
    with pytest.raises(VaspSpinParseError, match="Mode-1 field mismatch"):
        detect_field_scale(
            np.array([[0.005, 0.0, 0.0]]),
            penalty_energy=6.25e-7,
            lam=10.0,
            vasp_major=6,
            mw_int=np.array([[0.0, 0.0, 3.0]]),
            m_constr=np.zeros((1, 3)),
            i_constrained_m=1,
        )

    # 9b. Mode 4 box-interval check accepts valid tiny-moment rounding AND rejects flipped sign!
    scale_ok, _ = detect_field_scale(
        np.array([[9.944, 1.059, -10.0]]),
        penalty_energy=4.626e-4,
        lam=10.0,
        vasp_major=6,
        mw_int=np.array([[5e-5, 0.0, 0.0]]),
        m_constr=np.array([[0.0, 0.0, 1.0]]),
        i_constrained_m=4,
    )
    assert scale_ok == 1.0
    # Valid tiny y-component (0, +0.001, 0) accepted; flipped (0, -0.001, 0) rejected!
    scale_ok2, _ = detect_field_scale(
        np.array([[0.0, 0.001, 0.0]]),
        penalty_energy=5e-7,
        lam=1.0,
        vasp_major=6,
        mw_int=np.array([[1.0, 0.001, 0.0]]),
        m_constr=np.array([[1.0, 0.0, 0.0]]),
        i_constrained_m=4,
    )
    assert scale_ok2 == 1.0
    with pytest.raises(VaspSpinParseError, match="Mode-4 field mismatch"):
        detect_field_scale(
            np.array([[0.0, -0.001, 0.0], [2.0, 0.0, 0.0]]),
            penalty_energy=2.0000005,
            lam=1.0,
            vasp_major=6,
            mw_int=np.array([[1.0, 0.001, 0.0], [1.0, 0.0, 0.0]]),
            m_constr=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            i_constrained_m=4,
        )
    with pytest.raises(VaspSpinParseError, match="Mode-4 field mismatch"):
        detect_field_scale(
            np.array([[-0.5, 0.86603, -1.0]]),
            penalty_energy=1e-5,
            lam=1.0,
            vasp_major=6,
            mw_int=np.array([[1e-5, 0.0, 0.0]]),
            m_constr=np.array([[0.0, 0.0, 1.0]]),
            i_constrained_m=4,
        )

    # 9c. Pre-convergence SCF steps + VASP 6.4 `<lVp>` / `DBL` diagnostic lines handled cleanly;
    # soft-stop unconverged static run skipped; truncated trailing summary rejected.
    out_pre, osz_pre, _ = _make_vasp_pair(tmp_path / "pre_scf", version="6.4.0")
    pre_noise = (
        " E_p = 9.99999E+02  lambda = 1.0000E+01\n"
        "  ion        MW_int                             M_int\n"
        "    1   **********   0.000000   0.000000   99.000000   0.000000   0.000000\n"
        "DAV:   1    -10.00000000E+00   0.0   0.0   100   0.1E+00   0.1E+00\n"
        "RMM:   2    -15.00000000E+00   0.0   0.0   100   0.1E-01   0.1E-01\n"
    )
    osz_body = osz_pre.read_text(encoding="utf-8").replace(
        "  ion        MW_int",
        " <lVp> = 0.01000\n DBL   = 0.00100\n  ion        MW_int",
        1,
    )
    osz_pre.write_text(pre_noise + osz_body, encoding="utf-8")
    fr_pre = build_extxyz_frame(out_pre, osz_pre)
    assert fr_pre is not None and fr_pre.get_potential_energy() == pytest.approx(
        -24.5, abs=1e-6
    )

    # Soft-stop static OUTCAR without EDIFF reached is unconverged -> returns None
    out_es, osz_es, _ = _make_vasp_pair(tmp_path / "elec_softstop", version="6.4.0")
    out_es.write_text(
        out_es.read_text(encoding="utf-8").replace(
            "------------------------ aborting loop because EDIFF is reached ----------------------------------------",
            "soft stop encountered!  aborting job",
        ),
        encoding="utf-8",
    )
    assert build_extxyz_frame(out_es, osz_es) is None

    # Truncated trailing ionic summary `2 F= -24.5` (missing E0) rejected
    out_ts, osz_ts, _ = _make_vasp_pair(tmp_path / "trunc_sum", version="6.4.0")
    osz_ts.write_text(
        osz_ts.read_text(encoding="utf-8") + "   2 F= -24.50000000E+00\n",
        encoding="utf-8",
    )
    with pytest.raises(
        VaspSpinParseError, match="malformed/truncated final ionic summary"
    ):
        build_extxyz_frame(out_ts, osz_ts)

    # 10. Adjacent SCF step without its own MW_int (MW_int -> DAV:4 -> DAV:5 -> summary) rejected
    out_adj, osz_adj, _ = _make_vasp_pair(tmp_path / "adj_scf", version="6.4.0")
    osz_adj.write_text(
        osz_adj.read_text(encoding="utf-8").replace(
            "DAV:   5",
            "DAV:   4    -24.50E+00   0.0   0.0   100   0.1E-04   0.1E-05\nDAV:   5",
        ),
        encoding="utf-8",
    )
    with pytest.raises(VaspSpinParseError, match="no matching ionic-step MW_int"):
        build_extxyz_frame(out_adj, osz_adj)

    # 10b. Trailing DAV: 1 after complete OSZICAR rejected
    out_scf, osz_scf, _ = _make_vasp_pair(tmp_path / "trail_scf", version="6.4.0")
    osz_scf.write_text(
        osz_scf.read_text(encoding="utf-8")
        + "DAV:   1    -24.50E+00   0.0   0.0   100   0.1E-04   0.1E-05\n",
        encoding="utf-8",
    )
    with pytest.raises(
        VaspSpinParseError, match="incomplete/truncated constrained OSZICAR"
    ):
        build_extxyz_frame(out_scf, osz_scf)

    # 11. Batch explicit oszicar_path / hardlink protected from overwrite or unlink
    hardlink_out = tmp_path / "hardlink_osz"
    os.link(osz_scf, hardlink_out)
    with pytest.raises(
        ValueError, match="Refusing to overwrite or remove input VASP file"
    ):
        convert_directory_to_extxyz(
            tmp_path / "trail_scf", hardlink_out, oszicar_path=osz_scf, cpu=1
        )
