"""Adversarial stress and limit tests for GPST (Python/C99) and LAMMPS CFI/GPST."""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest

from spingnn2 import gpst
from spingnn2.gpst.build_ext import build_edge_kernel
from spingnn2.gpst.geometry import moment_arrays
from spingnn2.gpst.native import has_native_kernel, reload_native_kernel
from spingnn2.spin_temp import (
    HBAR_EV_PS,
    LAMMPS_METAL_KB,
    compute_gpst_moments,
    parse_gpst_vector,
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


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _read_dump_frames(path: Path) -> np.ndarray:
    frames = []
    for block in path.read_text().split("ITEM: TIMESTEP\n")[1:]:
        _, body = block.split("ITEM: ATOMS ", maxsplit=1)
        frames.append(np.loadtxt(io.StringIO(body), skiprows=1, ndmin=2))
    return np.stack(frames)


def _parse_thermo(output: str, ncols: int) -> np.ndarray:
    """Extract numeric thermo rows strictly between 'Step ' and 'Loop time'."""
    rows = []
    in_table = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Step "):
            in_table = True
            continue
        if in_table:
            if stripped.startswith("Loop time"):
                in_table = False
                continue
            fields = stripped.split()
            if len(fields) == ncols and fields[0].isdigit():
                rows.append([float(x) for x in fields])
    return np.array(rows, dtype=np.float64)


def _write_spin_data(
    path: Path,
    *,
    positions: np.ndarray = _POS,
    types: list[int] = _TYPES,
    box_side: float = BOX_SIDE,
    mags: list[float] | None = None,
    vel_fn=None,
) -> None:
    lines = [
        "CFI/GPST stress fixture",
        "",
        f"{len(positions)} atoms",
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
    for idx, ((x, y, z), kind) in enumerate(zip(positions, types)):
        tag = idx + 1
        vec = np.array([math.cos(1.1 * tag), math.sin(0.9 * tag), 0.35], dtype=float)
        sx, sy, sz = vec / np.linalg.norm(vec)
        mag = (
            mags[idx]
            if mags is not None
            else (0.0 if tag == 1 else 0.8 + 0.1 * (tag % 4))
        )
        lines.append(f"{tag} {1 + kind} {x} {y} {z} {sx} {sy} {sz} {mag}")
        if vel_fn is None:
            vx, vy, vz = 0.03 * np.array(
                [math.sin(tag), math.cos(tag), math.sin(2 * tag)]
            )
        else:
            vx, vy, vz = vel_fn(tag)
        velocities.append(f"{tag} {vx} {vy} {vz}")
    path.write_text("\n".join([*lines, *velocities, ""]))


def _lmp_run_raw(m4_binary: Path, m4_backend: str, tmp_path: Path, script: str):
    """Run LAMMPS returning CompletedProcess without asserting exit code == 0."""
    source = tmp_path / "in.err"
    source.write_text(script + "\n")
    cmd = [str(m4_binary), "-log", "none", "-in", str(source)]
    if m4_backend == "cuda":
        cmd += ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "neigh", "half"]
        launcher = (
            Path(__file__).resolve().parents[1] / "tools/lammps/spingnn2_gpu_rank.py"
        )
        cmd = [str(launcher), *cmd]
    elif m4_backend == "kk-cpu":
        cmd += ["-k", "on", "t", os.getenv("OMP_NUM_THREADS", "1"), "-sf", "kk"]
    return subprocess.run(
        cmd,
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Section A: GPST Python + C99 OpenMP Stress & Boundary Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_groups", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("n_frames", [1, 7])
def test_gpst_c99_vs_numpy_edge_zero_single_frame_and_multi_shells(
    monkeypatch: pytest.MonkeyPatch, n_groups: int, n_frames: int
):
    """C99 OpenMP fused kernel matches pure NumPy on E=0, T=1, G=1..16, and strided views."""
    build_edge_kernel()
    monkeypatch.delenv("SPINGNN2_GPST_NUMPY_ONLY", raising=False)
    monkeypatch.delenv("TSPIN_NUMPY_ONLY", raising=False)
    assert reload_native_kernel() is True
    assert has_native_kernel() is True

    rng = np.random.default_rng(4000 + 31 * n_groups + n_frames)
    # 1. Zero-edge graph (isolated atoms far apart, plus single-site N_mag=1)
    for pos_iso in (
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
    ):
        box = np.eye(3) * 30.0
        cuts = np.linspace(0.5, 2.5, n_groups - 1) if n_groups > 1 else np.empty(0)
        g = gpst.build_graph(pos_iso, box, np.ones(3, bool), 3.0, shell_cuts=cuts)
        assert len(g["i"]) == 0
        s = _unit(rng.normal(size=(n_frames, len(pos_iso), 3)))
        f = rng.normal(size=s.shape)

        nat = moment_arrays(s, f, g)
        monkeypatch.setenv("SPINGNN2_GPST_NUMPY_ONLY", "1")
        assert reload_native_kernel() is False
        npy = moment_arrays(s, f, g)
        monkeypatch.delenv("SPINGNN2_GPST_NUMPY_ONLY", raising=False)
        assert reload_native_kernel() is True

        for key in ("N", "D", "shell_N", "shell_D", "shell_SS", "shell_AN"):
            assert nat[key].shape == npy[key].shape
            assert np.isfinite(nat[key]).all()
            assert_allclose(nat[key], npy[key], rtol=0, atol=1e-14)
            assert np.all(nat[key] == 0.0)

    # 2. Multi-shell dense graph with G=1..16 and non-contiguous strided views
    pos = rng.uniform(0.0, 8.0, size=(20, 3))
    box = np.eye(3) * 8.0
    cuts = np.linspace(0.8, 3.2, n_groups - 1) if n_groups > 1 else np.empty(0)
    g = gpst.build_graph(pos, box, np.ones(3, bool), 3.5, shell_cuts=cuts)
    assert len(g["i"]) > 0
    s_raw = _unit(rng.normal(size=(2 * n_frames, 40, 3)))
    f_raw = rng.normal(size=s_raw.shape)
    # Non-contiguous strided slices (both reversed and stepped)
    s_view = s_raw[::-2, ::2, :]
    f_view = f_raw[::-2, ::2, :]
    assert not s_view.flags["C_CONTIGUOUS"]

    nat = moment_arrays(s_view, f_view, g)
    monkeypatch.setenv("SPINGNN2_GPST_NUMPY_ONLY", "1")
    assert reload_native_kernel() is False
    npy = moment_arrays(s_view, f_view, g)
    monkeypatch.delenv("SPINGNN2_GPST_NUMPY_ONLY", raising=False)
    assert reload_native_kernel() is True

    for key in ("N", "D", "shell_N", "shell_D", "SS", "FS", "shell_SS", "shell_AN"):
        assert_allclose(nat[key], npy[key], rtol=1e-13, atol=1e-13)


def test_gpst_c99_openmp_determinism_empty_and_16_shells(tmp_path: Path):
    """C99 OpenMP is bit-deterministic across OMP_NUM_THREADS=1,4 for E=0, T=1, and G=16."""
    build_edge_kernel()
    rng = np.random.default_rng(7701)
    pos = rng.uniform(0.0, 9.0, size=(24, 3))
    box = np.eye(3) * 9.0
    s = _unit(rng.normal(size=(8, 24, 3)))
    f = rng.normal(size=s.shape)
    np.savez(tmp_path / "data.npz", pos=pos, box=box, s=s, f=f)

    script = (
        "import sys, numpy as np; from spingnn2 import gpst; "
        "from spingnn2.gpst.geometry import moment_arrays; "
        "from spingnn2.gpst.native import has_native_kernel; "
        "assert has_native_kernel(); "
        f"d = np.load({str(tmp_path / 'data.npz')!r}); "
        "cuts16 = np.linspace(0.6, 3.6, 15); "
        "g16 = gpst.build_graph(d['pos'], d['box'], np.ones(3, bool), 3.8, shell_cuts=cuts16); "
        "r16 = moment_arrays(d['s'], d['f'], g16); "
        "r1 = moment_arrays(d['s'][:1], d['f'][:1], g16); "
        "g0 = gpst.build_graph(np.array([[0.,0.,0.]]), np.eye(3)*20., np.ones(3, bool), 2.0); "
        "r0 = moment_arrays(d['s'][:1, :1], d['f'][:1, :1], g0); "
        "np.savez(sys.argv[1], N16=r16['N'], D16=r16['D'], sN16=r16['shell_N'], "
        "N1=r1['N'], N0=r0['N'])"
    )
    out1 = tmp_path / "t1.npz"
    out4 = tmp_path / "t4.npz"
    for nt, dst in (("1", out1), ("4", out4)):
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = nt
        env.pop("SPINGNN2_GPST_NUMPY_ONLY", None)
        env.pop("TSPIN_NUMPY_ONLY", None)
        subprocess.run([sys.executable, "-c", script, str(dst)], env=env, check=True)
    a, b = np.load(out1), np.load(out4)
    for k in a.files:
        assert np.array_equal(a[k], b[k]), f"OMP mismatch on {k}"


def test_gpst_variable_cell_mixed_empty_frames_and_extreme_temperatures():
    """Variable-cell mixed E>0/E=0 frames, N_mag=0 rejection, and extreme T=0.01K & 5000K."""
    rng = np.random.default_rng(9911)
    # 1. Mixed frames: frame 0 has pairs within rc=3.0, frame 1 expands so E=0.
    pos0 = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]])
    pos1 = pos0 * 4.0
    frame = gpst.validate_frame(
        atom_ids=np.array([1, 2, 3]),
        magnetic_mask=np.ones(3, bool),
        positions=np.stack([pos0, pos1]),
        cell=np.broadcast_to(np.eye(3) * 30.0, (2, 3, 3)).copy(),
        pbc=np.ones(3, bool),
        spins=_unit(rng.normal(size=(2, 3, 3))),
        spin_forces=rng.normal(size=(2, 3, 3)),
        frame_index=np.array([0, 1]),
        force_contract_confirmed=True,
    )
    mom = gpst.compute_moments(frame, 3.0)
    assert np.isfinite(mom["N"]).all() and np.isfinite(mom["D"]).all()
    assert np.any(mom["D"][0] != 0.0)
    assert np.all(mom["N"][1] == 0.0) and np.all(mom["D"][1] == 0.0)

    # 2. All-nonmagnetic N_mag=0: validate_frame rejects cleanly; spin_temp returns n_mag=0.
    with pytest.raises(ValueError, match="ID_MISMATCH"):
        gpst.validate_frame(
            atom_ids=np.array([1, 2]),
            magnetic_mask=np.zeros(2, bool),
            positions=np.zeros((2, 2, 3)),
            cell=np.broadcast_to(np.eye(3) * 10.0, (2, 3, 3)).copy(),
            pbc=np.ones(3, bool),
            spins=np.zeros((2, 0, 3)),
            spin_forces=np.zeros((2, 0, 3)),
            frame_index=np.array([0, 1]),
            force_contract_confirmed=True,
        )
    zero_mag = compute_gpst_moments(
        pos0,
        np.eye(3) * 10.0,
        (True, True, True),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        cutoff=3.0,
        magnetic_mask=np.zeros(3, bool),
    )
    assert zero_mag["n_mag"] == 0
    assert np.all(zero_mag["N"] == 0.0) and np.all(zero_mag["D"] == 0.0)

    # 3. Analyze on E=0 records (nonzero forces): must never crash or produce NaN.
    g_empty = gpst.build_graph(
        np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]]),
        np.eye(3) * 30.0,
        np.ones(3, bool),
        3.0,
    )
    empty_records = [
        moment_arrays(
            _unit(rng.normal(size=(8, 2, 3))),
            rng.normal(size=(8, 2, 3)),
            g_empty,
        )
        for _ in range(6)
    ]
    res_empty = gpst.analyze(empty_records, list(range(6)), kb=LAMMPS_METAL_KB)
    assert res_empty["channels"]["CF_GMM"]["status"] in {
        "RANK_DEFICIENT",
        "ZERO_SIGNAL",
        "WEAK_DENOMINATOR",
        "UNBOUNDED_CI",
        "EMPTY_CI",
        "MOMENT_MODEL_MISFIT",
        "JACKKNIFE_FAILED",
    }

    # 4. Extreme temperatures T = 0.01 K and T = 5000 K exact Boltzmann recovery.
    g4 = gpst.build_graph(_POS[:4], np.eye(3) * 20.0, np.zeros(3, bool), 4.0)
    for target_t in (0.01, 5000.0):
        kappa = 2.5
        field_z = LAMMPS_METAL_KB * target_t * kappa
        recs = []
        for chain in range(6):
            crng = np.random.default_rng(5000 + int(target_t) + chain)
            u = crng.random((2000, 4))
            cos_t = (
                np.log(np.exp(-kappa) + u * (np.exp(kappa) - np.exp(-kappa))) / kappa
            )
            sin_t = np.sqrt(np.clip(1.0 - cos_t**2, 0.0, 1.0))
            phi = crng.uniform(0.0, 2.0 * np.pi, size=(2000, 4))
            s = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=-1)
            f = np.zeros_like(s)
            f[:, :, 2] = field_z
            recs.append(moment_arrays(s, f, g4))
        out = gpst.analyze(recs, list(range(6)), kb=LAMMPS_METAL_KB)
        for ch_name in ("pair_l1", "CF_GMM"):
            ch = out["channels"][ch_name]
            assert ch["status"] == "IDENTIFIED", f"T={target_t} {ch_name}: {ch}"
            assert abs(ch["temperature_K"] - target_t) / target_t < 0.15


# ---------------------------------------------------------------------------
# Section B: LAMMPS fix spin/cfi(/kk) & compute spin/gpst(/kk) Stress Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [
        "lattice frozen accept 2",
        "lattice moving accept 2 ou 100.0 0.05 91082",
        "lattice moving accept 2 ou 100.0 0.05 91082 box iso 0.0 200.0 box_ou 100.0 0.05 91083",
    ],
)
def test_cfi_gpst_all_nonmagnetic_nmag_zero(lmp_run, tmp_path, mode):
    """N_mag = 0 (all atoms nonmagnetic) runs cleanly in frozen/moving/box-iso CFI + GPST."""
    data = tmp_path / "nonmag.data"
    _write_spin_data(data, box_side=12.0, mags=[0.0] * len(_POS))
    vec_fmt = " ".join(f"$(c_gp[{i}]:%.17g)" for i in range(1, 20))
    script = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data}"
pair_style lj/cut 4.0
pair_coeff * * 0.02 2.4
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all spin/cfi 100.0 0.1 91081 {mode}
compute gp all spin/gpst cutoff 4.0
timestep 0.0005
run 6
print "NMAG0_RES $(pe:%.17g) $(ke:%.17g) $(c_gp:%.17g) $(f_integrate[3]:%.17g)"
print "NMAG0_VEC {vec_fmt}"
"""
    out = lmp_run(script, ranks=1)
    res = [
        float(x)
        for x in next(
            r for r in out.splitlines() if r.startswith("NMAG0_RES ")
        ).split()[1:]
    ]
    vec = np.array(
        [
            float(x)
            for x in next(
                r for r in out.splitlines() if r.startswith("NMAG0_VEC ")
            ).split()[1:]
        ]
    )
    assert np.isfinite(res).all()
    assert res[2] == 0.0 and res[3] == 0.0
    assert np.all(vec == 0.0)


def test_cfi_gpst_single_magnetic_and_isolated_zero_edges(lmp_run, m4_pair, tmp_path):
    """N_mag = 1 (single magnetic site) and E = 0 (isolated atoms > cutoff) stay exact & finite."""
    # Case 1: N_mag = 1 among 4 atoms (moving + box iso + lj/cut)
    data1 = tmp_path / "nmag1.data"
    _write_spin_data(
        data1,
        positions=_POS[:4],
        types=_TYPES[:4],
        box_side=12.0,
        mags=[1.5, 0.0, 0.0, 0.0],
    )
    vec_fmt = " ".join(f"$(c_gp[{i}]:%.17g)" for i in range(1, 20))
    script1 = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data1}"
pair_style lj/cut 4.0
pair_coeff * * 0.02 2.4
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all spin/cfi 80.0 0.05 91081 lattice moving accept 2 ou 80.0 0.05 91082 box iso 0.0 200.0
compute gp all spin/gpst cutoff 4.0
timestep 0.0002
run 6
print "NMAG1_VEC {vec_fmt}"
print "NMAG1_NORM $(f_integrate[3]:%.17g)"
"""
    out1 = lmp_run(script1, ranks=1)
    v1 = parse_gpst_vector(
        np.array(
            [
                float(x)
                for x in next(
                    r for r in out1.splitlines() if r.startswith("NMAG1_VEC ")
                ).split()[1:]
            ]
        )
    )
    norm1 = float(
        next(r for r in out1.splitlines() if r.startswith("NMAG1_NORM ")).split()[1]
    )
    assert v1["n_mag"] == 1.0
    assert math.isclose(v1["magnetization"][3], 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert np.all(v1["N"] == 0.0) and np.all(v1["D"] == 0.0)
    assert norm1 < 1e-12

    # Case 1b: N_mag = 0 with spingnn2 soc pair style + CFI + GPST
    data_nm_soc = tmp_path / "nmag0_soc.data"
    _write_spin_data(data_nm_soc, box_side=12.0, mags=[0.0] * len(_POS))
    pair = m4_pair("soc", "float64")
    script_nm_soc = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data_nm_soc}"
{pair}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all spin/cfi 100.0 0.05 91081 lattice moving accept 2 ou 100.0 0.05 91082 box iso 0.0 300.0
compute gp all spin/gpst cutoff 4.5
timestep 0.00005
run 6
print "NMSOC_VEC {vec_fmt}"
"""
    out_nm_soc = lmp_run(script_nm_soc, ranks=1)
    v_nm_soc = np.array(
        [
            float(x)
            for x in next(
                r for r in out_nm_soc.splitlines() if r.startswith("NMSOC_VEC ")
            ).split()[1:]
        ]
    )
    assert np.all(v_nm_soc == 0.0)

    # Case 2: 4 magnetic atoms separated by >10 A (E = 0 edges for cutoff 3.0 & spingnn2)
    iso_pos = np.array(
        [[2.0, 2.0, 2.0], [16.0, 2.0, 2.0], [2.0, 16.0, 2.0], [16.0, 16.0, 16.0]]
    )
    data0 = tmp_path / "iso.data"
    _write_spin_data(
        data0,
        positions=iso_pos,
        types=[0, 1, 0, 1],
        box_side=32.0,
        mags=[1.2, 1.4, 1.3, 1.1],
    )
    script0 = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data0}"
{pair}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all spin/cfi 100.0 0.1 91081 lattice moving accept 2 ou 100.0 0.05 91082 box iso 0.0 400.0
compute gp all spin/gpst cutoff 3.0
timestep 0.0001
run 6
print "ISO_VEC {vec_fmt}"
print "ISO_NORM $(f_integrate[3]:%.17g)"
"""
    out0 = lmp_run(script0, ranks=1)
    v0 = parse_gpst_vector(
        np.array(
            [
                float(x)
                for x in next(
                    r for r in out0.splitlines() if r.startswith("ISO_VEC ")
                ).split()[1:]
            ]
        )
    )
    norm0 = float(
        next(r for r in out0.splitlines() if r.startswith("ISO_NORM ")).split()[1]
    )
    assert v0["n_mag"] == 4.0
    assert np.isfinite(v0["magnetization"]).all()
    assert np.all(v0["N"] == 0.0) and np.all(v0["D"] == 0.0)
    assert norm0 < 1e-12


def test_cfi_gpst_empty_mpi_rank_migration_and_box_iso(lmp_run, m4_backend, tmp_path):
    """Empty MPI ranks (nlocal=0) receiving migrating atoms under box iso + delay 4 + sort."""
    env_ranks = os.environ.get(
        "SPINGNN2_M4_MPI_RANKS", os.environ.get("SPINGNN2_MPI_RANKS", "1")
    )
    ranks = [int(n) for n in env_ranks.split(",") if n.strip() and int(n) > 1]
    if not ranks:
        pytest.skip("Empty-rank MPI stress test requires SPINGNN2_M4_MPI_RANKS with >1")

    # Place all 6 atoms strictly inside [5.2, 7.8]^3 of a [0, 20]^3 box so rank >= 1
    # starts with nlocal == 0 under ANY 1x1x2 / 1x2x2 / 2x2x1 MPI split. Give
    # diagonal velocity (+45, +45, +45) A/ps with slow OU damp (2.0 ps) so atoms
    # travel >10 A along x, y, and z during 32 steps, migrating into previously-empty
    # ranks AND crossing PBC in all 3 dimensions under box iso + delay 4 + sort!
    cluster_pos = np.array(
        [
            [5.2, 5.2, 5.2],
            [7.6, 5.3, 5.2],
            [5.3, 7.6, 5.3],
            [7.7, 7.5, 5.2],
            [5.2, 5.3, 7.7],
            [7.6, 5.2, 7.6],
        ],
        dtype=np.float64,
    )
    data = tmp_path / "empty_rank.data"
    _write_spin_data(
        data,
        positions=cluster_pos,
        types=[0, 1, 0, 1, 1, 0],
        box_side=20.0,
        mags=[0.0, 1.2, 1.5, 1.1, 1.4, 1.3],
        vel_fn=lambda tag: (
            45.0 + 0.5 * tag,
            45.0 + 0.3 * math.sin(tag),
            45.0 - 0.2 * math.cos(tag),
        ),
    )
    vec_fmt = " ".join(f"$(c_gp[{i}]:%.17g)" for i in range(1, 20))

    def run_case(n_ranks: int, tag: str):
        dump = tmp_path / f"mig-r{n_ranks}-{tag}.dump"
        script = f"""units metal
boundary p p p
atom_style spin
atom_modify map array sort 2 0.0
newton on
read_data "{data}"
pair_style lj/cut 4.0
pair_coeff * * 0.02 2.4
neighbor 0.8 bin
neigh_modify every 1 delay 4 check yes
fix integrate all spin/cfi 120.0 0.08 91081 lattice moving accept 4 &
  ou 120.0 2.0 91082 box iso 0.0 400.0 box_ou 120.0 0.5 91083
compute spin all property/atom spx spy spz sp
compute gp all spin/gpst cutoff 4.0
timestep 0.01
dump traj all custom 16 "{dump}" id xu yu zu vx vy vz c_spin[1] c_spin[2] c_spin[3]
dump_modify traj sort id format float %.17g
run 32
print "MIG_BOX $(f_integrate[4]:%.17g) $(f_integrate[5]:%.17g) $(f_integrate[3]:%.17g)"
print "MIG_GP {vec_fmt}"
"""
        # Use gpu_aware=False so multi-rank Kokkos works on both non-GPU-aware
        # MPICH/Intel MPI and GPU-aware OpenMPI stacks.
        out = lmp_run(script, ranks=n_ranks, gpu_aware=False)
        box_vals = np.array(
            [
                float(x)
                for x in next(
                    r for r in out.splitlines() if r.startswith("MIG_BOX ")
                ).split()[1:]
            ]
        )
        gp_vals = np.array(
            [
                float(x)
                for x in next(
                    r for r in out.splitlines() if r.startswith("MIG_GP ")
                ).split()[1:]
            ]
        )
        return _read_dump_frames(dump), box_vals, gp_vals

    ref_tr, ref_box, ref_gp = run_case(1, "ref")
    # Confirm atoms actually traveled >10 A across x/y/z=10 subdomain splits & =20 PBC.
    for axis in (1, 2, 3):
        assert np.min(ref_tr[-1, :, axis] - ref_tr[0, :, axis]) > 10.0
        assert np.max(ref_tr[-1, :, axis]) > 20.0
    assert ref_box[2] < 1e-12

    test_ranks = list(ranks)
    if m4_backend in ("cpu", "kk-cpu") and 4 not in test_ranks:
        test_ranks.append(4)
    for n_ranks in test_ranks:
        tr, bx, gp = run_case(n_ranks, f"p{n_ranks}")
        assert bx[2] < 1e-12
        assert_allclose(bx[:2], ref_box[:2], rtol=0, atol=1e-9)
        assert_allclose(tr, ref_tr, rtol=0, atol=1e-9)
        assert_allclose(gp, ref_gp, rtol=0, atol=1e-9)


def test_cfi_triclinic_box_iso_rejected_and_triclinic_moving_gpst_oracle(
    lmp_run, m4_binary, m4_backend, m4_pair, tmp_path
):
    """Triclinic + box iso gives a clean error; triclinic moving CFI + GPST matches oracle."""
    data = tmp_path / "tri.data"
    _write_spin_data(data, box_side=16.0)
    pair = m4_pair("soc", "float64")

    # 1. Triclinic + box iso must reject cleanly with explicit message (no segfault).
    bad_script = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data}"
change_box all triclinic xy delta 2.0
pair_style lj/cut 3.5
pair_coeff * * 0.02 2.4
fix integrate all spin/cfi 0.0 0.0 91081 lattice moving accept 0 box iso 0.0 200.0
timestep 0.0001
run 1
"""
    proc = _lmp_run_raw(m4_binary, m4_backend, tmp_path, bad_script)
    assert proc.returncode != 0
    assert "orthogonal periodic 3D simulation box" in proc.stdout

    # 1b. Nonpositive box volume under extreme inward barostat momentum fails cleanly.
    collapse_script = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data}"
pair_style lj/cut 3.5
pair_coeff * * 0.02 2.4
fix integrate all spin/cfi 0.0 0.0 91081 lattice moving accept 0 box iso 0.0 1.0
fix_modify integrate box_momentum -1.0e8
timestep 0.001
run 1
"""
    proc_vol = _lmp_run_raw(m4_binary, m4_backend, tmp_path, collapse_script)
    assert proc_vol.returncode != 0
    assert "nonpositive volume" in proc_vol.stdout

    # 2. Triclinic moving CFI (no box iso) + compute spin/gpst matches Python oracle.
    dump = tmp_path / "tri_state.dump"
    vec_fmt = " ".join(f"$(c_gp[{i}]:%.17g)" for i in range(1, 20))
    ok_script = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data}"
change_box all triclinic xy delta 2.0 xz delta 1.0 yz delta 0.5
{pair}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all spin/cfi 80.0 0.05 91081 lattice moving accept 2 ou 80.0 0.05 91082
compute spprop all property/atom spx spy spz sp fmx fmy fmz
compute gp all spin/gpst cutoff 4.5
timestep 0.00005
run 4
dump snap all custom 1 "{dump}" id x y z &
  c_spprop[1] c_spprop[2] c_spprop[3] c_spprop[4] c_spprop[5] c_spprop[6] c_spprop[7]
dump_modify snap sort id format float %.17g
run 0
print "TRI_GP {vec_fmt}"
"""
    out = lmp_run(ok_script, ranks=1)
    lmp_vec = np.array(
        [
            float(x)
            for x in next(
                r for r in out.splitlines() if r.startswith("TRI_GP ")
            ).split()[1:]
        ]
    )
    parsed = parse_gpst_vector(lmp_vec)

    raw = _read_dump_frames(dump)[0]
    cell = np.array([[16.0, 0.0, 0.0], [2.0, 16.0, 0.0], [1.0, 0.5, 16.0]])
    oracle = compute_gpst_moments(
        raw[:, 1:4],
        cell,
        (True, True, True),
        raw[:, 4:7],
        HBAR_EV_PS * raw[:, 8:11],
        4.5,
        magnetic_mask=raw[:, 7] > 0.0,
    )
    assert parsed["n_mag"] == oracle["n_mag"]
    assert_allclose(parsed["D"], oracle["D"], rtol=0, atol=1e-11)
    assert_allclose(parsed["N"], oracle["N"], rtol=0, atol=1e-11)


def test_cfi_box_iso_500_step_nph_npt_and_predictor_fresh(lmp_run, m4_pair, tmp_path):
    """500-step NPH & NPT box iso stability (zero NaN, bounded drift) + predictor fresh."""
    data = tmp_path / "stab500.data"
    _write_spin_data(data, box_side=12.0)
    pair = m4_pair("soc", "float64")
    box_w = 400.0
    p_bar = 1000.0

    # 1. 500-step deterministic NPH (dt = 0.000025 ps -> drift < 5e-4 eV/atom)
    script_nph = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data}"
{pair}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all spin/cfi 0.0 0.0 91081 lattice moving accept 50 box iso {p_bar} {box_w}
fix_modify integrate box_momentum 0.15
timestep 0.000025
thermo 50
thermo_style custom step pe ke f_integrate[4] f_integrate[5] f_integrate[7] f_integrate[3]
thermo_modify format float %.17g
run 500
"""
    out_nph = lmp_run(script_nph, ranks=1)
    arr = _parse_thermo(out_nph, 7)
    assert len(arr) == 11
    assert np.isfinite(arr).all()
    assert np.all(arr[:, 3] > 0.0)
    assert np.max(arr[:, 6]) < 1e-12
    h = arr[:, 1] + arr[:, 2] + arr[:, 5] * arr[:, 3] + 0.5 * (arr[:, 4] ** 2) / box_w
    assert np.max(np.abs(h - h[0])) / len(_POS) < 5e-4

    # 2. 500-step stochastic NPT (spin + atomic OU + box OU + GPST)
    script_npt = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data}"
{pair}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all spin/cfi 150.0 0.05 91081 lattice moving accept 50 &
  ou 150.0 0.05 91082 box iso 0.0 300.0 box_ou 150.0 0.05 91083
compute gp all spin/gpst cutoff 4.5
timestep 0.00005
thermo 50
thermo_style custom step pe ke f_integrate[4] f_integrate[5] f_integrate[3] c_gp[1] c_gp[2]
thermo_modify format float %.17g
run 500
"""
    out_npt = lmp_run(script_npt, ranks=1)
    arr_npt = _parse_thermo(out_npt, 8)
    assert len(arr_npt) == 11
    assert np.isfinite(arr_npt).all()
    assert np.all(arr_npt[:, 3] > 0.0)
    assert np.max(arr_npt[:, 5]) < 1e-12

    # 3. Multi-run continuation with box iso: run 8 + run 8 pre no == run 16
    def run_cont(run_block: str, name: str):
        dump = tmp_path / f"cont-{name}.dump"
        sc = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data}"
{pair}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all spin/cfi 100.0 0.05 91081 lattice moving accept 0 &
  ou 100.0 0.05 91082 box iso 0.0 300.0 box_ou 100.0 0.05 91083
compute spin all property/atom spx spy spz
timestep 0.00005
{run_block}
dump tr all custom 1 "{dump}" id x y z vx vy vz c_spin[1] c_spin[2] c_spin[3]
dump_modify tr sort id format float %.17g
run 0
print "CONT_STATE $(f_integrate[4]:%.17g) $(f_integrate[5]:%.17g)"
"""
        out = lmp_run(sc, ranks=1)
        st = np.array(
            [
                float(x)
                for x in next(
                    r for r in out.splitlines() if r.startswith("CONT_STATE ")
                ).split()[1:]
            ]
        )
        return _read_dump_frames(dump)[0], st

    tr_single, st_single = run_cont("run 16", "single")
    tr_split, st_split = run_cont("run 8\nrun 8 pre no", "split")
    assert_allclose(st_split, st_single, rtol=0, atol=1e-13)
    assert_allclose(tr_split, tr_single, rtol=0, atol=1e-13)

    # 4. predictor fresh + box iso (lj/cut + spingnn2)
    script_fresh = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data}"
{pair}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all spin/cfi 0.0 0.0 91081 lattice moving accept 1 predictor fresh box iso {p_bar} {box_w}
fix_modify integrate box_momentum 0.15
timestep 0.00005
thermo 8
thermo_style custom step pe ke f_integrate[4] f_integrate[5] f_integrate[7] f_integrate[3]
thermo_modify format float %.17g
run 16
"""
    out_fresh = lmp_run(script_fresh, ranks=1)
    rows_f = _parse_thermo(out_fresh, 7)
    assert len(rows_f) == 3 and np.isfinite(rows_f).all()
    assert np.max(rows_f[:, 6]) < 1e-12


@pytest.mark.parametrize(
    "temp_k,alpha,dt",
    [
        (0.01, 1e-4, 0.00005),
        (0.01, 10.0, 0.00005),
        (5000.0, 0.0, 0.00005),
        (5000.0, 1e-4, 0.00005),
        (5000.0, 10.0, 0.00002),
    ],
)
def test_cfi_extreme_temperature_and_damping(
    lmp_run, m4_pair, tmp_path, temp_k, alpha, dt
):
    """Extreme T (0.01K..5000K) and alpha (0, 1e-4, 10.0) stay finite with unit spin norm."""
    data = tmp_path / f"ext-{temp_k}-{alpha}.data"
    _write_spin_data(data, box_side=16.0)
    pair = m4_pair("soc", "float64")
    dump = tmp_path / f"ext-{temp_k}-{alpha}.dump"
    script = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data}"
{pair}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all spin/cfi {temp_k} {alpha} 91081 lattice moving accept 10 ou {temp_k} 0.1 91082
compute spin all property/atom spx spy spz sp
compute gp all spin/gpst cutoff 4.5
timestep {dt}
dump tr all custom 40 "{dump}" id x y z vx vy vz c_spin[1] c_spin[2] c_spin[3] c_spin[4]
dump_modify tr sort id format float %.17g
run 40
print "EXT_RES $(pe:%.17g) $(ke:%.17g) $(f_integrate[3]:%.17g) $(c_gp[1]:%.17g) $(c_gp[2]:%.17g)"
"""
    out = lmp_run(script, ranks=1)
    vals = [
        float(x)
        for x in next(r for r in out.splitlines() if r.startswith("EXT_RES ")).split()[
            1:
        ]
    ]
    assert np.isfinite(vals).all()
    assert vals[2] < 1e-12
    frames = _read_dump_frames(dump)
    assert frames.shape[0] == 2 and np.isfinite(frames).all()
    norms = np.linalg.norm(frames[:, :, 7:10], axis=-1)
    assert np.max(np.abs(norms - 1.0)) < 1e-12
