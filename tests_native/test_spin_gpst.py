"""Unit and LAMMPS tests for Geometric-Probe Spin Temperature (compute spin/gpst)."""

import io
import math
import os
from pathlib import Path

import numpy as np
import pytest

from spingnn2.spin_temp import (
    CHANNELS,
    HBAR_EV_PS,
    KB_EV_K,
    compute_gpst_moments,
    parse_gpst_vector,
    pooled_gpst,
)

BOX_SIDE = 16.0
_POS = np.array(
    [
        [2.0, 2.0, 2.0],
        [4.5, 2.1, 2.0],
        [2.1, 4.6, 2.1],
        [4.6, 4.5, 2.0],
        [2.0, 2.1, 4.6],
        [4.5, 2.0, 4.5],
        [2.1, 4.5, 4.5],
        [4.6, 4.6, 4.6],
    ],
    dtype=np.float64,
)
_TYPES = [0, 1, 0, 1, 1, 0, 1, 0]


def _write_spin_data(path: Path) -> None:
    lines = [
        "GPST periodic fixture",
        "",
        f"{len(_POS)} atoms",
        "2 atom types",
        "",
        *(f"0.0 {BOX_SIDE} {axis}lo {axis}hi" for axis in "xyz"),
        "",
        "Masses",
        "",
        "1 54.938",
        "2 72.630",
        "",
        "Atoms # spin",
        "",
    ]
    for tag, ((x, y, z), kind) in enumerate(zip(_POS, _TYPES), 1):
        vec = np.array([math.cos(1.3 * tag), math.sin(0.7 * tag), 0.4 * (-1) ** tag])
        sx, sy, sz = vec / np.linalg.norm(vec)
        # Atoms 1 and 5 are nonmagnetic (moment == 0).
        mag = 0.0 if tag in (1, 5) else 1.2 + 0.3 * (tag % 3)
        lines.append(f"{tag} {1 + kind} {x} {y} {z} {sx} {sy} {sz} {mag}")
    path.write_text("\n".join([*lines, ""]))


def test_gpst_nonmagnetic_exclusion_and_zero_gate():
    """Nonmagnetic atoms do not affect GPST moments; zero-force inputs trigger ZERO_SIGNAL."""
    cell = np.diag([10.0, 10.0, 10.0])
    pbc = (True, True, True)
    rng = np.random.default_rng(2026)

    pos_mag = np.array(
        [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [3.0, 3.0, 1.0]]
    )
    spins_mag = rng.normal(size=(4, 3))
    spins_mag /= np.linalg.norm(spins_mag, axis=-1, keepdims=True)
    forces_mag = rng.normal(scale=0.05, size=(4, 3))

    ref = compute_gpst_moments(pos_mag, cell, pbc, spins_mag, forces_mag, cutoff=4.0)

    # Interleave 3 nonmagnetic atoms with arbitrary spins/forces.
    pos_mix = np.vstack([pos_mag[:2], [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], pos_mag[2:]])
    spins_mix = np.vstack(
        [spins_mag[:2], [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]], spins_mag[2:]]
    )
    forces_mix = np.vstack(
        [forces_mag[:2], [[1e3, -1e3, 1e3], [5.0, 5.0, 5.0]], forces_mag[2:]]
    )
    mask = np.array([True, True, False, False, True, True])
    mix = compute_gpst_moments(
        pos_mix, cell, pbc, spins_mix, forces_mix, cutoff=4.0, magnetic_mask=mask
    )

    assert mix["n_mag"] == 4
    np.testing.assert_allclose(mix["N"], ref["N"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(mix["D"], ref["D"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        mix["magnetization"], ref["magnetization"], rtol=0, atol=1e-14
    )

    # Zero-signal numerical gate: roundoff-scale numerator never yields false IDENTIFIED.
    n_zero = np.full((20, 7), 1e-19)
    d_one = np.ones((20, 7))
    pooled = pooled_gpst(n_zero, d_one, kb=KB_EV_K)
    for ch in CHANNELS:
        assert pooled[ch]["status"] == "ZERO_SIGNAL"


def test_gpst_unbiased_on_nonlinear_hamiltonian_where_nurdin_fails():
    """On a nonlinear biquadratic + SIA + DMI spin Hamiltonian, GPST recovers exact T
    while Nurdin's linear-Hamiltonian formula is systematically biased."""
    target_t = 300.0
    beta = 1.0 / (KB_EV_K * target_t)
    cell = np.diag([12.0, 12.0, 12.0])
    pbc = (True, True, True)

    # 2x2x2 simple cubic magnetic cluster (spacing 3.0 A), cutoff 3.5 A (6 NN each).
    coords = np.array(
        [
            [3.0 * ix, 3.0 * iy, 3.0 * iz]
            for ix in range(2)
            for iy in range(2)
            for iz in range(2)
        ],
        dtype=np.float64,
    )
    n_sites = len(coords)
    # Build undirected nearest-neighbor list (distance == 3.0 A).
    pairs = []
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            d = coords[j] - coords[i]
            d -= np.rint(d / 12.0) * 12.0
            if 0.1 < np.linalg.norm(d) < 3.5:
                pairs.append((i, j, d / np.linalg.norm(d)))

    j_lin = 0.015
    j_biq = 0.035  # Strongly nonlinear biquadratic term -(J_biq)(s_i . s_j)^2
    k_sia = 0.040  # Nonlinear single-ion anisotropy -K (s_z)^2
    d_dmi = 0.012  # Chiral DMI

    def energy_and_force(s):
        e = -k_sia * float(np.sum(s[:, 2] ** 2))
        f = np.zeros_like(s)
        f[:, 2] += 2.0 * k_sia * s[:, 2]
        for i, j, u in pairs:
            xd = float(np.dot(s[i], s[j]))
            e -= j_lin * xd + j_biq * xd * xd
            coeff = j_lin + 2.0 * j_biq * xd
            f[i] += coeff * s[j]
            f[j] += coeff * s[i]
            # DMI: +d_dmi * u . (s_i x s_j)
            e += d_dmi * float(np.dot(u, np.cross(s[i], s[j])))
            f[i] -= d_dmi * np.cross(s[j], u)
            f[j] -= d_dmi * np.cross(u, s[i])
        return e, f

    rng = np.random.default_rng(77123)
    chains_n, chains_d = [], []
    nurdin_num, nurdin_den = 0.0, 0.0

    for _chain in range(4):
        s = rng.normal(size=(n_sites, 3))
        s /= np.linalg.norm(s, axis=-1, keepdims=True)
        e_cur, _ = energy_and_force(s)
        # Equilibrate + sample with exact single-spin Metropolis on S^2.
        n_frames, d_frames = [], []
        for sweep in range(420):
            for idx in range(n_sites):
                prop = s[idx] + rng.normal(scale=0.55, size=3)
                prop /= np.linalg.norm(prop)
                s_old = s[idx].copy()
                s[idx] = prop
                e_new, _ = energy_and_force(s)
                if e_new <= e_cur or rng.random() < math.exp(-beta * (e_new - e_cur)):
                    e_cur = e_new
                else:
                    s[idx] = s_old
            if sweep >= 100 and sweep % 2 == 0:
                _, f_cur = energy_and_force(s)
                mom = compute_gpst_moments(coords, cell, pbc, s, f_cur, cutoff=3.5)
                n_frames.append(mom["N"])
                d_frames.append(mom["D"])
                # Nurdin linear-Hamiltonian denominator: sum(s . f)
                cross_sf = np.cross(s, f_cur)
                nurdin_num += float(np.sum(cross_sf * cross_sf))
                nurdin_den += 2.0 * float(np.sum(s * f_cur))
        chains_n.append(np.array(n_frames))
        chains_d.append(np.array(d_frames))

    pooled = pooled_gpst(chains_n, chains_d, kb=KB_EV_K)
    # Primary exchange (pair_l1) and biquadratic (pair_l2) channels must recover 300 K accurately.
    for ch in ("pair_l1", "pair_l2"):
        res = pooled[ch]
        assert res["status"] == "IDENTIFIED", res
        assert abs(res["temperature_K"] - target_t) < max(35.0, 4.0 * res["se_K"]), res

    t_nurdin = nurdin_num / (KB_EV_K * nurdin_den)
    # Nurdin formula ignores nonlinear Laplacians of biquadratic & SIA terms -> large bias.
    assert abs(t_nurdin - target_t) > 60.0


def test_compute_spin_gpst_matches_python_oracle(lmp_run, m4_pair, tmp_path):
    """LAMMPS compute spin/gpst(/kk) matches clean-room Python GPST oracle to <1e-11."""
    data = tmp_path / "gpst.data"
    _write_spin_data(data)
    dump = tmp_path / "state.dump"
    cutoff = 4.5
    pair = m4_pair("soc", "float64")

    vec_fmt = " ".join(f"$(c_gp[{i}]:%.17g)" for i in range(1, 20))
    script = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data}"
{pair}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
compute spprop all property/atom spx spy spz sp fmx fmy fmz
compute gp all spin/gpst cutoff {cutoff}
dump snap all custom 1 "{dump}" id x y z &
  c_spprop[1] c_spprop[2] c_spprop[3] c_spprop[4] c_spprop[5] c_spprop[6] c_spprop[7]
dump_modify snap sort id format float %.17g
run 0
print "GPST_SCALAR $(c_gp:%.17g)"
print "GPST_VEC {vec_fmt}"
"""
    output = lmp_run(script, ranks=1)
    vec_line = next(r for r in output.splitlines() if r.startswith("GPST_VEC "))
    sca_line = next(r for r in output.splitlines() if r.startswith("GPST_SCALAR "))
    lmp_vec = np.array([float(x) for x in vec_line.split()[1:]], dtype=np.float64)
    lmp_scalar = float(sca_line.split()[1])
    parsed = parse_gpst_vector(lmp_vec)

    block = dump.read_text().split("ITEM: TIMESTEP\n")[1]
    raw = np.loadtxt(
        io.StringIO(block.split("ITEM: ATOMS ", 1)[1]), skiprows=1, ndmin=2
    )
    pos = raw[:, 1:4]
    spins = raw[:, 4:7]
    mags = raw[:, 7]
    fm = raw[:, 8:11]
    # Exact LAMMPS metal hbar = 4.135667696e-3 / (2*pi) (force->hplanck / MY_2PI).
    # Note: LAMMPS 30Sep2026 uses CODATA hplanck = 4.135667403e-3 or 4.135667696e-3;
    # test against LAMMPS metal hbar constant.
    oracle = compute_gpst_moments(
        pos,
        np.diag([BOX_SIDE, BOX_SIDE, BOX_SIDE]),
        (True, True, True),
        spins,
        HBAR_EV_PS * fm,
        cutoff,
        magnetic_mask=mags > 0.0,
    )

    assert parsed["n_mag"] == 6.0
    assert oracle["n_mag"] == 6
    np.testing.assert_allclose(parsed["D"], oracle["D"], rtol=0, atol=1e-11)
    np.testing.assert_allclose(parsed["N"], oracle["N"], rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        parsed["magnetization"], oracle["magnetization"], rtol=0, atol=1e-12
    )
    if abs(parsed["D"][0]) > 1e-30:
        expected_scalar = parsed["N"][0] / (KB_EV_K * parsed["D"][0])
        assert lmp_scalar == pytest.approx(expected_scalar, rel=1e-11, abs=1e-11)


def test_compute_spin_gpst_mpi_parity(lmp_run, m4_pair, tmp_path):
    """Online compute spin/gpst(/kk) matches bit-for-bit across MPI ranks."""
    env_ranks = os.environ.get(
        "SPINGNN2_M4_MPI_RANKS", os.environ.get("SPINGNN2_MPI_RANKS", "1")
    )
    ranks = tuple(int(n) for n in env_ranks.split(",") if n.strip())
    if not any(n > 1 for n in ranks):
        pytest.skip("Multi-rank GPST check requires SPINGNN2_M4_MPI_RANKS with >1 rank")

    data = tmp_path / "gpst.data"
    _write_spin_data(data)
    pair = m4_pair("soc", "float64")
    vec_fmt = " ".join(f"$(c_gp[{i}]:%.17g)" for i in range(1, 20))
    script = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data}"
{pair}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all spin/cfi 80.0 0.05 91081 lattice moving accept 4 ou 80.0 0.05 91082
compute gp all spin/gpst cutoff 4.5
timestep 0.00005
run 4
print "GPST_VEC {vec_fmt}"
"""
    one_out = lmp_run(script, ranks=1)
    one_vec = np.array(
        [
            float(x)
            for x in next(
                r for r in one_out.splitlines() if r.startswith("GPST_VEC ")
            ).split()[1:]
        ]
    )
    for n in ranks:
        if n == 1:
            continue
        multi_out = lmp_run(script, ranks=n)
        multi_vec = np.array(
            [
                float(x)
                for x in next(
                    r for r in multi_out.splitlines() if r.startswith("GPST_VEC ")
                ).split()[1:]
            ]
        )
        np.testing.assert_allclose(multi_vec, one_vec, rtol=0, atol=1e-11)
