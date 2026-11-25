"""Unit and LAMMPS qualification tests for Cached-Field Integrator (fix spin/cfi)."""

import io
import math
import os
from pathlib import Path

import numpy as np
import pytest

HBAR_METAL = 4.135667696e-3 / (2.0 * math.pi)
BOX_SIDE = 16.0

# Self-contained 8-atom magnetic/nonmagnetic test crystal.
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


def _write_spin_data(
    path: Path, *, box_side: float = BOX_SIDE, v_scale: float = 0.03
) -> None:
    lines = [
        "CFI periodic spin-lattice fixture",
        "",
        f"{len(_POS)} atoms",
        "2 atom types",
        "",
        *(f"0.0 {box_side} {axis}lo {axis}hi" for axis in "xyz"),
        "",
        "Masses",
        "",
        "1 54.938",
        "2 72.630",
        "",
        "Atoms # spin",
        "",
    ]
    velocities = ["", "Velocities", ""]
    for tag, ((x, y, z), kind) in enumerate(zip(_POS, _TYPES), 1):
        vec = np.array([math.cos(tag), math.sin(tag), 0.35], dtype=np.float64)
        sx, sy, sz = vec / np.linalg.norm(vec)
        # Atom 1 is nonmagnetic (moment == 0) to verify strict exclusion.
        mag = 0.0 if tag == 1 else 0.8 + 0.1 * (tag % 4)
        lines.append(f"{tag} {1 + kind} {x} {y} {z} {sx} {sy} {sz} {mag}")
        vx, vy, vz = v_scale * np.array(
            [math.sin(tag), math.cos(tag), math.sin(2 * tag)]
        )
        velocities.append(f"{tag} {vx} {vy} {vz}")
    path.write_text("\n".join([*lines, *velocities, ""]))


def _read_dump_frames(path: Path) -> np.ndarray:
    frames = []
    for block in path.read_text().split("ITEM: TIMESTEP\n")[1:]:
        _, body = block.split("ITEM: ATOMS ", maxsplit=1)
        frames.append(np.loadtxt(io.StringIO(body), skiprows=1, ndmin=2))
    return np.stack(frames)


def _cayley_py(orig, direction, torque, dt_spin, *, midpoint):
    w = -0.5 * dt_spin * torque
    first = np.cross(orig, w)
    second = np.cross(first, w)
    scale = (1.0 if midpoint else 2.0) / (1.0 + float(np.dot(w, w)))
    out = orig + scale * (first + second)
    return out / np.linalg.norm(out)


def test_cfi_python_second_order_nve_model():
    """Verify deterministic CFI achieves O(dt^2) energy conservation on a coupled dimer."""
    j0 = 0.05
    k_spring = 0.25
    r0 = 2.5
    mass = 50.0

    def energy_and_forces(r, s1, s2):
        # J(r) = j0 * exp(-(r - r0))
        j_r = j0 * math.exp(-(r - r0))
        sdot = float(np.dot(s1, s2))
        e = 0.5 * k_spring * (r - r0) ** 2 - j_r * sdot
        f_r = -k_spring * (r - r0) - j_r * sdot
        f_s1 = j_r * s2
        f_s2 = j_r * s1
        return e, f_r, f_s1, f_s2

    drifts = []
    total_time = 2.0
    for dt in (0.04, 0.02, 0.01):
        steps = int(round(total_time / dt))
        r = r0 + 0.15
        v = 0.02
        s1 = np.array([1.0, 0.0, 0.0])
        s2 = np.array([0.3, 0.9, 0.316227766])
        s2 /= np.linalg.norm(s2)

        e0, _, f1_cache, f2_cache = energy_and_forces(r, s1, s2)
        e0 += 0.5 * mass * v * v
        max_drift = 0.0

        for _ in range(steps):
            r_half = r + 0.5 * dt * v
            s1_orig, s2_orig = s1.copy(), s2.copy()
            s1_mid = _cayley_py(s1_orig, s1_orig, f1_cache, dt, midpoint=True)
            s2_mid = _cayley_py(s2_orig, s2_orig, f2_cache, dt, midpoint=True)
            _, f_r, f1_mid, f2_mid = energy_and_forces(r_half, s1_mid, s2_mid)
            f1_cache, f2_cache = f1_mid, f2_mid
            s1 = _cayley_py(s1_orig, s1_mid, f1_mid, dt, midpoint=False)
            s2 = _cayley_py(s2_orig, s2_mid, f2_mid, dt, midpoint=False)
            v = v + (dt / mass) * f_r
            r = r_half + 0.5 * dt * v
            e_cur = energy_and_forces(r, s1, s2)[0] + 0.5 * mass * v * v
            max_drift = max(max_drift, abs(e_cur - e0))

        assert abs(np.linalg.norm(s1) - 1.0) < 1e-14
        assert abs(np.linalg.norm(s2) - 1.0) < 1e-14
        drifts.append(max_drift)

    # Halving dt should reduce error by ~4x (allow >= 3.2x for pre-asymptotic regime).
    assert drifts[0] / drifts[1] > 3.2
    assert drifts[1] / drifts[2] > 3.2


def _run_cfi(
    lmp_run,
    pair: str,
    data: Path,
    dump: Path,
    integrator: str,
    *,
    steps: int = 16,
    dt: float = 0.00005,
    ranks: int = 1,
    sort_cmd: str = "atom_modify map array",
):
    script = f"""units metal
boundary p p p
atom_style spin
{sort_cmd}
newton on
read_data "{data}"
{pair}
neighbor 0.4 bin
neigh_modify every 1 delay 0 check yes
fix integrate all {integrator}
compute spin all property/atom spx spy spz sp fmx fmy fmz
compute timers all spingnn2/timing
timestep {dt}
dump traj all custom {steps} "{dump}" id x y z vx vy vz fx fy fz &
  c_spin[1] c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7]
dump_modify traj sort id format float %.17g
thermo {steps}
thermo_style custom step pe ke etotal c_timers[5] c_timers[6] &
  f_integrate[1] f_integrate[2] f_integrate[3]
thermo_modify format float %.17g
run {steps}
variable etot equal etotal
variable hits equal c_timers[5]
variable evals equal c_timers[6]
variable cfi_steps equal f_integrate[1]
variable cfi_acc equal f_integrate[2]
variable cfi_norm equal f_integrate[3]
print "CFI_RESULT ${{etot}} ${{hits}} ${{evals}} ${{cfi_steps}} ${{cfi_acc}} ${{cfi_norm}}"
"""
    output = lmp_run(script, ranks=ranks)
    line = next(row for row in output.splitlines() if row.startswith("CFI_RESULT "))
    vals = [float(x) for x in line.split()[1:]]
    stats = dict(
        zip(("energy", "hits", "calls", "steps", "accepts", "maxnormdev"), vals)
    )
    first_thermo = next(
        fields
        for row in output.splitlines()
        if len(fields := row.split()) == 9 and fields[0] == "0"
    )
    stats["e0"] = float(first_thermo[3])
    traj = _read_dump_frames(dump)
    assert traj.shape[0] == 2
    assert np.isfinite(traj).all()
    assert np.max(np.abs(np.linalg.norm(traj[:, :, 10:13], axis=-1) - 1.0)) < 1e-12
    # Nonmagnetic atom 1 must never evolve its spin direction or moment.
    np.testing.assert_array_equal(traj[0, 0, 10:14], traj[1, 0, 10:14])
    return traj, stats


@pytest.mark.parametrize("lattice", ("frozen", "moving"))
def test_cfi_eval_count_and_accept_invariance(lmp_run, m4_pair, tmp_path, lattice):
    """CFI uses 1 core eval/step (vs 2 for fresh control) and accept N is dynamics-invariant."""
    data = tmp_path / "spin.data"
    _write_spin_data(data)
    pair = m4_pair("soc", "float64")
    steps = 16

    traj_a0, stats_a0 = _run_cfi(
        lmp_run,
        pair,
        data,
        tmp_path / f"cfi-{lattice}-a0.dump",
        f"spin/cfi 0.0 0.0 91081 lattice {lattice} accept 0",
        steps=steps,
    )
    traj_a4, stats_a4 = _run_cfi(
        lmp_run,
        pair,
        data,
        tmp_path / f"cfi-{lattice}-a4.dump",
        f"spin/cfi 0.0 0.0 91081 lattice {lattice} accept 4",
        steps=steps,
    )
    _, stats_fresh = _run_cfi(
        lmp_run,
        pair,
        data,
        tmp_path / f"cfi-{lattice}-fresh.dump",
        f"spin/lpsib 0.0 0.0 91081 lattice {lattice} accept 0 predictor fresh",
        steps=steps,
    )

    # 1 setup + 1 core/step for CFI vs 1 setup + 2/step for predictor fresh.
    assert stats_a0["calls"] == 1 + steps
    assert stats_a4["calls"] == 1 + steps + (steps // 4)
    assert stats_fresh["calls"] == 1 + 2 * steps
    assert stats_a0["steps"] == steps
    assert stats_a0["accepts"] == 0
    assert stats_a4["accepts"] == steps // 4
    assert stats_a0["maxnormdev"] < 1e-13

    if lattice == "frozen":
        # Frozen CFI reuses PairSpinGNN2 Stage-1 cache on every step after setup.
        assert stats_a0["hits"] == steps
        np.testing.assert_array_equal(traj_a0[1, :, 1:7], traj_a0[0, :, 1:7])
    else:
        assert np.max(np.abs(traj_a0[1, :, 1:4] - traj_a0[0, :, 1:4])) > 1e-7

    # Accepted-state refresh must NOT perturb dynamics (positions, velocities, spins).
    np.testing.assert_allclose(
        traj_a0[:, :, 1:7], traj_a4[:, :, 1:7], rtol=0, atol=1e-13
    )
    np.testing.assert_allclose(
        traj_a0[:, :, 10:14], traj_a4[:, :, 10:14], rtol=0, atol=1e-13
    )


def test_cfi_moving_nve_convergence(lmp_run, m4_pair, tmp_path):
    """Undamped moving CFI shows second-order energy drift reduction as dt halves."""
    data = tmp_path / "spin.data"
    _write_spin_data(data)
    pair = m4_pair("soc", "float64")

    drifts = []
    # Match total physical time T = 0.00032 ps across dt = 0.00004, 0.00002, 0.00001 ps.
    for dt, steps in ((0.00004, 8), (0.00002, 16), (0.00001, 32)):
        _, stats = _run_cfi(
            lmp_run,
            pair,
            data,
            tmp_path / f"nve-{steps}.dump",
            "spin/cfi 0.0 0.0 91081 lattice moving accept 1",
            steps=steps,
            dt=dt,
        )
        drifts.append(abs(stats["energy"] - stats["e0"]) / len(_POS))

    assert drifts[0] / drifts[1] > 3.0
    assert drifts[1] / drifts[2] > 3.0
    assert drifts[2] < 5e-5


def test_cfi_mpi_stochastic_ou_migration(lmp_run, m4_pair, tmp_path):
    """Stochastic moving CFI with built-in OU bath is invariant across MPI ranks & sorting."""
    env_ranks = os.environ.get(
        "SPINGNN2_M4_MPI_RANKS", os.environ.get("SPINGNN2_MPI_RANKS", "1")
    )
    ranks = tuple(int(n) for n in env_ranks.split(",") if n.strip())
    if not any(n > 1 for n in ranks):
        pytest.skip("Multi-rank CFI check requires SPINGNN2_M4_MPI_RANKS with >1 rank")

    data = tmp_path / "spin.data"
    _write_spin_data(data, v_scale=0.15)
    pair = m4_pair("nosoc", "float64")
    integrator = "spin/cfi 100.0 0.1 91081 lattice moving accept 4 ou 100.0 0.05 91082"

    one, _ = _run_cfi(
        lmp_run,
        pair,
        data,
        tmp_path / "one.dump",
        integrator,
        steps=12,
        ranks=1,
        sort_cmd="atom_modify map array sort 2 0.0",
    )
    for n in ranks:
        if n == 1:
            continue
        multi, _ = _run_cfi(
            lmp_run,
            pair,
            data,
            tmp_path / f"rank-{n}.dump",
            integrator,
            steps=12,
            ranks=n,
            sort_cmd="atom_modify map array sort 2 0.0",
        )
        np.testing.assert_allclose(multi[:, :, 1:7], one[:, :, 1:7], rtol=0, atol=1e-11)
        np.testing.assert_allclose(
            multi[:, :, 10:14], one[:, :, 10:14], rtol=0, atol=1e-11
        )


def test_cfi_dream_bec_dipole_conservation(lmp_run, tmp_path):
    """Moving CFI tracks DREAM BEC dipole accurately across both half-drifts."""
    tensor = np.array([[1.2, 0.1, -0.3], [0.4, 0.9, 0.2], [-0.1, 0.5, 1.4]])
    bec_file = tmp_path / "fixed.bec"
    np.savetxt(
        bec_file,
        np.column_stack([[1, 2], np.stack([tensor, -tensor]).reshape(2, 9)]),
        fmt="%g",
    )
    dump = tmp_path / "bec_cfi.dump"
    script = f"""units metal
boundary p p p
atom_style hybrid spin bec
atom_modify map array
newton on
region box block 0 12 0 12 0 12
create_box 1 box
create_atoms 1 single 3.0 3.0 3.0 units box
create_atoms 1 single 5.5 3.0 3.0 units box
mass 1 55.845
set atom 1 spin/atom 2.0 1.0 0.0 0.0
set atom 2 spin/atom 2.0 0.0 1.0 0.0
read_bec "{bec_file}"
velocity all set 0.5 -0.25 0.15 units box
pair_style lj/cut 3.0
pair_coeff * * 0.01 2.2
fix integrate all spin/cfi 0.0 0.0 91081 lattice moving accept 0
timestep 0.001
dump tr all custom 5 "{dump}" id xu yu zu becdipolex becdipoley becdipolez
dump_modify tr sort id format float %.17g
run 5
"""
    lmp_run(script, ranks=1)
    frames = _read_dump_frames(dump)
    assert frames.shape == (2, 2, 7)
    dx = frames[1, :, 1:4] - frames[0, :, 1:4]
    expected_p = np.stack([tensor @ dx[0], -tensor @ dx[1]])
    np.testing.assert_allclose(frames[1, :, 4:7], expected_p, rtol=0, atol=1e-12)


def test_cfi_box_iso_nph_enthalpy_conservation(lmp_run, m4_pair, tmp_path):
    """Deterministic box iso NPH conserves H = pe + ke + P*V + pV^2/(2*W) to O(dt^2)."""
    data = tmp_path / "spin_nph.data"
    _write_spin_data(data, box_side=12.0, v_scale=0.04)
    pair = m4_pair("soc", "float64")
    p_bar = 5000.0
    box_w = 500.0
    pv0 = 0.25

    drifts = []
    # Match total time T = 0.0004 ps across dt = 0.00005, 0.000025, 0.0000125 ps.
    for dt, steps in ((0.00005, 8), (0.000025, 16), (0.0000125, 32)):
        script = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data}"
{pair}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all spin/cfi 0.0 0.0 91081 lattice moving accept 1 box iso {p_bar} {box_w}
fix_modify integrate box_momentum {pv0}
timestep {dt}
thermo {steps}
thermo_style custom step pe ke f_integrate[4] f_integrate[5] f_integrate[7] f_integrate[8]
thermo_modify format float %.17g
run {steps}
print "NPH_FINAL $(pe:%.17g) $(ke:%.17g) $(f_integrate[4]:%.17g) $(f_integrate[5]:%.17g) $(f_integrate[6]:%.17g) $(f_integrate[7]:%.17g) $(f_integrate[8]:%.17g) $(f_integrate[9]:%.17g) $(f_integrate[10]:%.17g) $(f_integrate[11]:%.17g)"
"""
        output = lmp_run(script, ranks=1)
        row0 = next(
            fields
            for line in output.splitlines()
            if len(fields := line.split()) == 7 and fields[0] == "0"
        )
        pe0, ke0, v0, pv_init, p_int0 = (float(row0[i]) for i in (1, 2, 3, 4, 5))
        h0 = pe0 + ke0 + p_int0 * v0 + 0.5 * (pv_init**2) / box_w

        fin = [
            float(x)
            for x in next(
                line for line in output.splitlines() if line.startswith("NPH_FINAL ")
            ).split()[1:]
        ]
        pe1, ke1, v1, pv1, vdot_over_v, p_int1, nktv2p, vxx, vyy, vzz = fin
        h1 = pe1 + ke1 + p_int1 * v1 + 0.5 * (pv1**2) / box_w

        assert v1 > 0.0 and abs(v1 - v0) > 1e-8
        assert math.isclose(
            vdot_over_v, pv1 / (box_w * v1), rel_tol=1e-12, abs_tol=1e-15
        )
        assert math.isclose(p_int1, p_bar / nktv2p, rel_tol=1e-12, abs_tol=1e-15)
        assert np.isfinite([vxx, vyy, vzz]).all()
        drifts.append(abs(h1 - h0) / len(_POS))

    assert drifts[0] / drifts[1] > 3.0
    assert drifts[1] / drifts[2] > 3.0
    assert drifts[2] < 5e-5


def test_cfi_box_iso_npt_accept_invariance_and_mpi(lmp_run, m4_pair, tmp_path):
    """Stochastic NPT (box iso + ou + box_ou) is accept-invariant and MPI-invariant."""
    data = tmp_path / "spin_npt.data"
    _write_spin_data(data, box_side=12.0, v_scale=0.05)
    pair = m4_pair("nosoc", "float64")

    def run_npt(accept_n: int, dump: Path, ranks: int = 1):
        script = f"""units metal
boundary p p p
atom_style spin
atom_modify map array sort 2 0.0
newton on
read_data "{data}"
{pair}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all spin/cfi 100.0 0.08 91081 lattice moving accept {accept_n} &
  ou 100.0 0.05 91082 box iso 0.0 200.0 box_ou 100.0 0.1 91083
compute spin all property/atom spx spy spz sp fmx fmy fmz
timestep 0.00005
dump traj all custom 8 "{dump}" id x y z vx vy vz c_spin[1] c_spin[2] c_spin[3] c_spin[4]
dump_modify traj sort id format float %.17g
run 8
print "NPT_STATE $(f_integrate[4]:%.17g) $(f_integrate[5]:%.17g) $(f_integrate[3]:%.17g)"
"""
        output = lmp_run(script, ranks=ranks)
        vals = [
            float(x)
            for x in next(
                line for line in output.splitlines() if line.startswith("NPT_STATE ")
            ).split()[1:]
        ]
        return _read_dump_frames(dump), vals

    tr0, st0 = run_npt(0, tmp_path / "npt-a0.dump", ranks=1)
    tr4, st4 = run_npt(4, tmp_path / "npt-a4.dump", ranks=1)
    assert st0[2] < 1e-13
    np.testing.assert_allclose(st0[:2], st4[:2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(tr0, tr4, rtol=0, atol=1e-12)

    env_ranks = os.environ.get(
        "SPINGNN2_M4_MPI_RANKS", os.environ.get("SPINGNN2_MPI_RANKS", "1")
    )
    ranks = tuple(int(n) for n in env_ranks.split(",") if n.strip())
    for n in ranks:
        if n <= 1:
            continue
        tr_m, st_m = run_npt(4, tmp_path / f"npt-r{n}.dump", ranks=n)
        np.testing.assert_allclose(st_m[:2], st4[:2], rtol=0, atol=1e-10)
        np.testing.assert_allclose(tr_m, tr4, rtol=0, atol=1e-10)


def test_cfi_box_iso_dream_bec_dipole(lmp_run, tmp_path):
    """DREAM BEC dipole remains exact under box iso volume dilation + particle drift."""
    tensor = np.array([[1.2, 0.1, -0.3], [0.4, 0.9, 0.2], [-0.1, 0.5, 1.4]])
    bec_file = tmp_path / "fixed_nph.bec"
    np.savetxt(
        bec_file,
        np.column_stack([[1, 2], np.stack([tensor, -tensor]).reshape(2, 9)]),
        fmt="%g",
    )
    dump = tmp_path / "bec_cfi_nph.dump"
    script = f"""units metal
boundary p p p
atom_style hybrid spin bec
atom_modify map array
newton on
region box block 0 12 0 12 0 12
create_box 1 box
create_atoms 1 single 3.0 3.0 3.0 units box
create_atoms 1 single 5.5 3.0 3.0 units box
mass 1 55.845
set atom 1 spin/atom 2.0 1.0 0.0 0.0
set atom 2 spin/atom 2.0 0.0 1.0 0.0
read_bec "{bec_file}"
velocity all set 0.5 -0.25 0.15 units box
pair_style lj/cut 3.0
pair_coeff * * 0.01 2.2
fix integrate all spin/cfi 0.0 0.0 91081 lattice moving accept 0 box iso 1000.0 50.0
fix_modify integrate box_momentum 2.0
timestep 0.001
dump tr all custom 5 "{dump}" id xu yu zu becdipolex becdipoley becdipolez
dump_modify tr sort id format float %.17g
run 5
"""
    lmp_run(script, ranks=1)
    frames = _read_dump_frames(dump)
    assert frames.shape == (2, 2, 7)
    dx = frames[1, :, 1:4] - frames[0, :, 1:4]
    expected_p = np.stack([tensor @ dx[0], -tensor @ dx[1]])
    np.testing.assert_allclose(frames[1, :, 4:7], expected_p, rtol=0, atol=1e-12)


def test_cfi_box_iso_pbc_crossing_with_neighbor_delay(lmp_run, tmp_path):
    """Atoms crossing periodic boundaries under box iso + neigh_modify delay > 0
    must never teleport between reneighbor steps (continuous pair distance & H_NPH)."""
    dump = tmp_path / "pbc_cross.dump"
    box_w = 500.0
    p_bar = 0.0
    script = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
region box block 0 12 0 12 0 12
create_box 1 box
create_atoms 1 single 11.98 6.0 6.0 units box
create_atoms 1 single 1.48 6.0 6.0 units box
mass 1 55.845
set atom 1 spin/atom 2.0 1.0 0.0 0.0
set atom 2 spin/atom 2.0 0.0 1.0 0.0
velocity all set 15.0 0.0 0.0 units box
pair_style lj/cut 3.0
pair_coeff * * 0.05 1.3
neighbor 0.8 bin
neigh_modify every 1 delay 4 check yes
fix integrate all spin/cfi 0.0 0.0 91081 lattice moving accept 0 box iso {p_bar} {box_w}
fix_modify integrate box_momentum 0.2
timestep 0.001
thermo 1
thermo_style custom step pe ke f_integrate[4] f_integrate[5] f_integrate[7]
thermo_modify format float %.17g
dump tr all custom 1 "{dump}" id x xu ix
dump_modify tr sort id format float %.17g
run 8
"""
    output = lmp_run(script, ranks=1)
    rows = [
        [float(x) for x in fields]
        for line in output.splitlines()
        if len(fields := line.split()) == 6 and fields[0].isdigit()
    ]
    assert len(rows) == 9
    enthalpies = [r[1] + r[2] + r[5] * r[3] + 0.5 * (r[4] ** 2) / box_w for r in rows]
    assert max(abs(h - enthalpies[0]) for h in enthalpies) < 1e-4

    frames = _read_dump_frames(dump)
    # Atom 1 starts at x=11.98 with vx=+15 A/ps, crosses x=12.0 on step 2 while delay=4
    # prevents reneighboring until step >= 4, and must remain continuously unwrapped.
    xu1 = frames[:, 0, 2]
    assert xu1[0] < 12.0 < xu1[-1]
    assert np.all(np.diff(xu1) > 0.0)
    # Pair distance in unwrapped space stays smooth (~1.5 A), never jumping by ~12 A.
    pair_dx = np.abs((frames[:, 1, 2] - frames[:, 0, 2]) + 12.0)
    assert np.all(np.abs(np.diff(pair_dx)) < 0.05)
