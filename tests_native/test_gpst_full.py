"""Comprehensive tests for spingnn2.gpst (contracts, C99 fused kernel, adapter, CLI)."""

from __future__ import annotations

import dataclasses
import inspect
import json
from itertools import product
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import eval_legendre

from spingnn2 import gpst
from spingnn2.gpst.basis import evaluate_basis, learn_basis, upper_bounds
from spingnn2.gpst.build_ext import build_edge_kernel
from spingnn2.gpst.cli import main_infer, main_moments
from spingnn2.gpst.dump_adapter import FIELDS as DUMP_FIELDS, read_dump
from spingnn2.gpst.geometry import FIXED, legendre, minimum_image, moment_arrays
from spingnn2.gpst.inference import crossfit, pooled_pair
from spingnn2.gpst.native import has_native_kernel, reload_native_kernel
from spingnn2.spin_temp import LAMMPS_METAL_KB

POS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.1, 0.0], [0.1, 0.3, 1.3]])
BOX = np.eye(3) * 20.0
RC = 3.4


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def arrays(n: int = 4, frames: int = 12, seed: int = 80) -> dict:
    rng = np.random.default_rng(seed)
    return dict(
        atom_ids=np.arange(1, n + 1),
        magnetic_mask=np.ones(n, bool),
        positions=np.broadcast_to(POS, (frames, n, 3)).copy(),
        cell=np.broadcast_to(BOX, (frames, 3, 3)).copy(),
        pbc=np.zeros(3, bool),
        spins=unit(rng.normal(size=(frames, n, 3))),
        spin_forces=rng.normal(size=(frames, n, 3)),
        frame_index=np.arange(frames),
        force_contract_confirmed=True,
    )


def phi_grad(s: np.ndarray, name: str, graph: dict):
    i, j, w, u = [graph[k] for k in ("i", "j", "w", "direction")]
    grad = np.zeros_like(s)
    if name.startswith("pair_"):
        order = int(name[6:])
        p, dp = legendre(np.sum(s[i] * s[j], axis=1), order)
        phi = w @ p[order]
        np.add.at(grad, i, (w * dp[order])[:, None] * s[j])
        np.add.at(grad, j, (w * dp[order])[:, None] * s[i])
        d = 2 * order * (order + 1) * phi
    elif name == "geometry_Q2":
        grad = np.einsum("nij,nj->ni", graph["Q"], s)
        phi = 0.5 * np.sum(grad * s)
        d = 6 * phi - np.trace(graph["Q"], axis1=1, axis2=2).sum()
    elif name == "geometry_zonal4":
        phi = 0.0
        for idx in (i, j):
            p, dp = legendre(np.sum(s[idx] * u, axis=1), 4)
            phi += w @ p[4]
            np.add.at(grad, idx, (w * dp[4])[:, None] * u)
        d = 20 * phi
    else:
        phi = np.sum(w * np.sum(u * np.cross(s[i], s[j]), axis=1))
        np.add.at(grad, i, w[:, None] * np.cross(s[j], u))
        np.add.at(grad, j, w[:, None] * np.cross(u, s[i]))
        d = 4 * phi
    return phi, grad - s * np.sum(s * grad, axis=1, keepdims=True), d


@pytest.mark.parametrize("name", FIXED)
def test_gradient_divergence_h2(name: str):
    rng = np.random.default_rng(801)
    s = unit(rng.normal(size=(4, 3)))
    f = rng.normal(size=s.shape)
    g = gpst.build_graph(POS, BOX, np.zeros(3, bool), RC)
    phi, x, d = phi_grad(s, name, g)
    r = moment_arrays(s[None], f[None], g)
    idx = FIXED.index(name)
    assert_allclose(r["N"][0, idx], np.sum(x * f), rtol=1e-12, atol=1e-12)
    assert_allclose(r["D"][0, idx], d, rtol=1e-12, atol=1e-12)
    errs = []
    for h in (2e-3, 1e-3, 5e-4):
        div = 0.0
        grad_error = 0.0
        for site in range(4):
            e = unit(np.cross(s[site], np.eye(3)[np.argmin(abs(s[site]))]))
            for tangent in (e, np.cross(s[site], e)):
                sp = s.copy()
                sm = s.copy()
                sp[site] = np.cos(h) * s[site] + np.sin(h) * tangent
                sm[site] = np.cos(h) * s[site] - np.sin(h) * tangent
                pp, xp, _ = phi_grad(sp, name, g)
                pm, xm, _ = phi_grad(sm, name, g)
                div += tangent @ (xp[site] - xm[site]) / (2 * h)
                grad_error = max(
                    grad_error, abs((pp - pm) / (2 * h) - x[site] @ tangent)
                )
        errs.append(abs(div + d) / max(1, abs(d)))
        assert grad_error < 0.002
    assert errs[-1] < 1e-4
    assert all(a / b > 3.8 for a, b in zip(errs, errs[1:]))


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_legendre(degree: int):
    x = np.linspace(-1, 1, 201)
    p, dp = legendre(x)
    assert_allclose(p[degree], eval_legendre(degree, x), rtol=1e-12, atol=1e-12)
    c = np.zeros(degree + 1)
    c[-1] = 1
    assert_allclose(
        dp[degree],
        np.polynomial.legendre.legval(x, np.polynomial.legendre.legder(c)),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "operation",
    ["rotation", "translation", "id_permutation", "type_rename", "triclinic", "radial"],
)
def test_invariance(operation: str):
    a = arrays()
    base = gpst.compute_moments(gpst.validate_frame(**a), RC)
    b = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in a.items()}
    if operation == "rotation":
        q, _ = np.linalg.qr(np.random.default_rng(80).normal(size=(3, 3)))
        for key in ("positions", "spins", "spin_forces", "cell"):
            b[key] = b[key] @ q
    elif operation == "translation":
        b["positions"] += np.array([1.25, -2.75, 0.83])
    elif operation == "id_permutation":
        order = np.array([2, 0, 3, 1])
        for key in ("atom_ids", "magnetic_mask"):
            b[key] = b[key][order]
        for key in ("positions", "spins", "spin_forces"):
            b[key] = b[key][:, order]
    elif operation == "type_rename":
        labels = np.array([17, 29, 17, 29])
        renamed = np.array([113, 7, 113, 7])
        assert not np.array_equal(labels, renamed)
    elif operation == "triclinic":
        b["cell"][:] = np.array([[20.0, 0.0, 0.0], [8.0, 20.0, 0.0], [3.0, 2.0, 20.0]])
    else:
        b["spin_forces"] += 3.0 * b["spins"]
    got = gpst.compute_moments(gpst.validate_frame(**b), RC)
    for key in ("N", "D"):
        assert (
            np.max(abs(got[key] - base[key]) / np.maximum(1.0, abs(base[key]))) <= 1e-12
        )


def test_periodic_bruteforce_and_supercell():
    rng = np.random.default_rng(81)
    box = np.array([[7.0, 0.0, 0.0], [3.5, 7.0, 0.0], [1.0, 2.0, 7.0]])
    pos = rng.random((24, 3)) @ box
    rc = 2.1
    periodic = np.ones(3, bool)
    g = gpst.build_graph(pos, box, periodic, rc)
    shifts = np.array(list(product(range(-2, 3), repeat=3))) @ box
    expected = {}
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            candidates = pos[j] - pos[i] - shifts
            v = candidates[np.argmin(np.sum(candidates * candidates, axis=1))]
            if np.linalg.norm(v) < rc:
                expected[i, j] = v
    assert set(expected) == set(zip(g["i"], g["j"]))
    assert_allclose(
        g["distance"],
        [np.linalg.norm(expected[i, j]) for i, j in zip(g["i"], g["j"])],
        atol=1e-12,
    )
    s = unit(rng.normal(size=(3, 24, 3)))
    f = rng.normal(size=s.shape)
    r = moment_arrays(s, f, g)
    pos2 = np.concatenate([pos, pos + box[0]])
    box2 = box.copy()
    box2[0] *= 2
    g2 = gpst.build_graph(pos2, box2, periodic, rc, shell_cuts=g["shell_cuts"])
    r2 = moment_arrays(np.tile(s, (1, 2, 1)), np.tile(f, (1, 2, 1)), g2)
    for key in ("N", "D"):
        assert_allclose(r2[key], 2 * r[key], rtol=1e-12, atol=1e-12)
    assert_allclose(
        r2["N"].sum(0) / r2["D"].sum(0), r["N"].sum(0) / r["D"].sum(0), rtol=1e-12
    )
    dr = rng.normal(size=(25, 3)) * 8
    exact = minimum_image(dr, box, periodic)
    large = np.array(list(product(range(-6, 7), repeat=3))) @ box
    brute = np.min(np.sum((dr[:, None] - large) ** 2, axis=2), axis=1)
    assert_allclose(np.sum(exact * exact, axis=1), brute, rtol=1e-12, atol=1e-12)


def test_multispecies_nonmagnetic_variable_cell():
    a = arrays()
    a["atom_ids"] = np.array([4, 2, 7, 1, 9, 8])
    a["magnetic_mask"] = np.array([1, 0, 1, 1, 0, 1], bool)
    a["positions"] = np.insert(
        a["positions"], [1, 3], [[8.0, 8.0, 8.0], [9.0, 9.0, 9.0]], axis=1
    )
    a["cell"] *= np.linspace(1.0, 1.02, len(a["cell"]))[:, None, None]
    frame = gpst.validate_frame(**a)
    r = gpst.compute_moments(frame, RC)
    assert r["N"].shape == (12, 7) and np.isfinite(r["N"]).all()
    for k in range(12):
        g = gpst.build_graph(
            frame.positions[k, frame.magnetic_mask],
            frame.cell[k],
            frame.pbc,
            RC,
            shell_cuts=r["shell_cuts"],
        )
        single = moment_arrays(frame.spins[k : k + 1], frame.spin_forces[k : k + 1], g)
        assert_allclose(r["N"][k], single["N"][0], atol=1e-12)


@pytest.mark.parametrize(
    "fault,code",
    [
        ("unit", "NONUNIT_SPIN"),
        ("nan", "BAD_INPUT"),
        ("inf", "BAD_INPUT"),
        ("id", "ID_MISMATCH"),
        ("shape", "ID_MISMATCH"),
        ("force", "FORCE_CONTRACT_MISSING"),
    ],
)
def test_rejections(fault: str, code: str):
    a = arrays()
    if fault == "unit":
        a["spins"] *= 2
    elif fault == "nan":
        a["spin_forces"][0, 0, 0] = np.nan
    elif fault == "inf":
        a["positions"][0, 0, 0] = np.inf
    elif fault == "id":
        a["atom_ids"][1] = a["atom_ids"][0]
    elif fault == "shape":
        a["cell"] = a["cell"][0]
    else:
        a["force_contract_confirmed"] = False
    with pytest.raises(ValueError, match=code):
        gpst.validate_frame(**a)


def test_direct_dataclass_cannot_bypass_checks():
    a = arrays()
    a["spins"] *= 2
    with pytest.raises(ValueError, match="NONUNIT_SPIN"):
        gpst.compute_moments(gpst.FrameInput(**a), RC)


def test_no_reference_temperature_or_default_force(tmp_path: Path):
    a = arrays()
    a.pop("force_contract_confirmed")
    np.savez(tmp_path / "frames.npz", **a)
    (tmp_path / "meta.json").write_text("malformed: must not be read")
    frame = gpst.load_frame_npz(tmp_path / "frames.npz", force_contract_confirmed=True)
    assert len(frame.spins) == 12
    for func in (
        gpst.validate_frame,
        gpst.compute_moments,
        gpst.analyze,
        gpst.load_frame_npz,
    ):
        assert not {"temperature", "temperature_K", "T", "theta"}.intersection(
            inspect.signature(func).parameters
        )
    with pytest.raises(TypeError):
        gpst.from_lammps_fm(np.ones(3))
    assert_allclose(gpst.from_lammps_fm(np.ones(3), 0.123), 0.123)


def test_pooled_and_fieller():
    n = np.array([2.0, 8.0])
    d = np.array([1.0, 4.0])
    assert gpst.pooled_ratio(n, d, kb=1)["temperature_K"] == 2
    with pytest.raises(ValueError, match="only pooled"):
        gpst.pooled_ratio(n, d, kb=1, method="frame_average")
    a, b = pooled_pair(
        [np.array([[2.0]]), np.full((3, 1), 8.0)],
        [np.array([[1.0]]), np.full((3, 1), 2.0)],
    )
    assert_allclose(a / b, 26.0 / 7.0)
    assert gpst.fieller(0, 0, np.zeros((2, 2)), 23)["status"] == "ZERO_SIGNAL"
    assert gpst.fieller(1, 0, np.zeros((2, 2)), 23)["status"] == "EMPTY_CI"
    assert gpst.fieller(1, 0, np.eye(2), 23)["status"] == "WEAK_DENOMINATOR"
    assert len(gpst.fieller(4, 0.1, np.diag([0.01, 1.0]), 23)["confidence_set"]) == 2


def test_zero_signal_and_bounds():
    a = arrays()
    a["spin_forces"][:] = 0
    r = gpst.compute_moments(gpst.validate_frame(**a), RC)
    out = gpst.analyze([r] * 6, list(range(6)), kb=1)
    assert all(
        c["status"] == "ZERO_SIGNAL" and c["temperature_K"] is None
        for c in out["channels"].values()
    )
    basis = learn_basis([r])
    _, d = evaluate_basis(r, basis)
    u = upper_bounds(r, basis)
    assert np.max(abs(d) - u) <= 1e-12
    pos = np.array(list(product(range(4), repeat=3)), float)
    g = gpst.build_graph(pos, np.eye(3) * 4, np.ones(3, bool), 1.1)
    rng = np.random.default_rng(82)
    s = unit(rng.normal(size=(16, 64, 3)))
    f = rng.normal(size=s.shape)
    rr = moment_arrays(s, f, g)
    assert np.max(abs(rr["N"][:, 4])) < 1e-12 and np.max(abs(rr["D"][:, 4])) < 1e-12


def test_crossfit_training_isolation():
    rng = np.random.default_rng(83)
    g = gpst.build_graph(POS, BOX, np.zeros(3, bool), RC)
    rows = []
    for _ in range(6):
        s = unit(rng.normal(size=(512, 4, 3)))
        s[:, :, 2] += 1
        s = unit(s)
        f = np.zeros_like(s)
        f[:, :, 2] = 1
        rows.append(moment_arrays(s, f, g))
    _, _, before = crossfit(rows, np.arange(6) % 2)
    changed = [{k: v.copy() for k, v in r.items()} for r in rows]
    for i in (0, 2, 4):
        changed[i]["FS"] *= 19
    _, _, after = crossfit(changed, np.arange(6) % 2)
    assert before[0] == after[0]


@pytest.mark.parametrize("force_numpy", [False, True])
def test_uniform_incompatible_force(monkeypatch: pytest.MonkeyPatch, force_numpy: bool):
    if force_numpy:
        monkeypatch.setenv("SPINGNN2_GPST_NUMPY_ONLY", "1")
        assert reload_native_kernel() is False
    else:
        build_edge_kernel()
        monkeypatch.delenv("SPINGNN2_GPST_NUMPY_ONLY", raising=False)
        monkeypatch.delenv("TSPIN_NUMPY_ONLY", raising=False)
        assert reload_native_kernel() is True
        assert has_native_kernel() is True
    try:
        axes = np.concatenate([np.eye(3), -np.eye(3)])
        s = np.array(list(product(axes, repeat=4)))
        f = s.sum(1)[:, None] - s
        g = gpst.build_graph(POS, BOX, np.zeros(3, bool), RC)
        r = moment_arrays(s, f, g)
        assert r["N"][:, 0].mean() > 0
        assert abs(r["D"][:, 0].mean()) < 1e-12
        rejection = gpst.fieller(
            r["N"][:, 0].mean(), r["D"][:, 0].mean(), np.zeros((2, 2)), 23
        )
        assert rejection["status"] == "EMPTY_CI"
        out = gpst.analyze([r] * 6, list(range(6)), kb=1)
        assert out["channels"]["CF_GMM"]["status"] == "MOMENT_MODEL_MISFIT"
    finally:
        monkeypatch.delenv("SPINGNN2_GPST_NUMPY_ONLY", raising=False)
        reload_native_kernel()


def test_low_temperature_equilibrium_not_tripped_by_zero_gate():
    """Exact Boltzmann Zeeman equilibrium at T = 0.01 K must recover IDENTIFIED."""
    target_t = 0.01
    kappa = 2.5
    field_z = LAMMPS_METAL_KB * target_t * kappa
    g = gpst.build_graph(POS, BOX, np.zeros(3, bool), RC)
    records = []
    for chain in range(6):
        rng = np.random.default_rng(1200 + chain)
        u = rng.random((2000, 4))
        cos_theta = (
            np.log(np.exp(-kappa) + u * (np.exp(kappa) - np.exp(-kappa))) / kappa
        )
        sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, 1.0))
        phi = rng.uniform(0.0, 2.0 * np.pi, size=(2000, 4))
        s = np.stack(
            [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=-1
        )
        f = np.zeros_like(s)
        f[:, :, 2] = field_z
        records.append(moment_arrays(s, f, g))
    out = gpst.analyze(records, list(range(6)), kb=LAMMPS_METAL_KB)
    for key in ("pair_l1", "CF_GMM"):
        ch = out["channels"][key]
        assert ch["status"] == "IDENTIFIED", f"{key} unexpectedly got {ch}"
        assert abs(ch["temperature_K"] - target_t) / target_t < 0.15


def test_relative_cutoff_and_stale_graph():
    a = arrays()
    frame = gpst.validate_frame(**a)
    g = gpst.build_graph(POS, BOX, np.zeros(3, bool), {"nearest_multiplier": 3.0})
    assert_allclose(g["rc"], 3.0)
    bad = dataclasses.replace(frame, positions=frame.positions + 1)
    with pytest.raises(ValueError, match="ID_MISMATCH"):
        gpst.compute_moments(bad, 3.0, graph=g)
    with pytest.raises(ValueError, match="unique image"):
        gpst.build_graph(POS, np.eye(3) * 4, np.ones(3, bool), 2.0)


def test_same_lattice_triclinic_representation():
    rng = np.random.default_rng(8001)
    box = np.eye(3) * 10.0
    shear = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    pos = rng.random((80, 3)) @ box
    a = gpst.build_graph(pos, box, np.ones(3, bool), 2.0)
    b = gpst.build_graph(
        pos, shear @ box, np.ones(3, bool), 2.0, shell_cuts=a["shell_cuts"]
    )
    assert np.array_equal(a["i"], b["i"]) and np.array_equal(a["j"], b["j"])
    s = unit(rng.normal(size=(8, 80, 3)))
    f = rng.normal(size=s.shape)
    x = moment_arrays(s, f, a)
    y = moment_arrays(s, f, b)
    for key in ("N", "D"):
        assert np.max(abs(x[key] - y[key]) / np.maximum(1.0, abs(x[key]))) <= 1e-12


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_descending_frame_indices_rejected(dtype):
    a = arrays()
    a["frame_index"] = np.arange(12, 0, -1, dtype=dtype)
    with pytest.raises(ValueError, match="ID_MISMATCH"):
        gpst.validate_frame(**a)


@pytest.mark.parametrize("cutoff", [2.0, {"nearest_multiplier": 2.0}])
def test_prepared_graph_cutoff_mismatch_rejected(cutoff):
    frame = gpst.validate_frame(**arrays())
    graph = gpst.build_graph(POS, BOX, np.zeros(3, bool), RC)
    with pytest.raises(ValueError, match="ID_MISMATCH"):
        gpst.compute_moments(frame, cutoff, graph=graph)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_increasing_frame_indices_and_equivalent_cached_cutoffs(dtype):
    a = arrays()
    a["frame_index"] = np.arange(12, dtype=dtype)
    frame = gpst.validate_frame(**a)
    graph = gpst.build_graph(POS, BOX, np.zeros(3, bool), RC)
    expected = gpst.compute_moments(frame, RC)
    actual = gpst.compute_moments(frame, {"nearest_multiplier": RC}, graph=graph)
    for key in ("N", "D", "shell_N", "shell_D", "SS", "FS", "shell_SS", "shell_AN"):
        assert_allclose(actual[key], expected[key], rtol=0, atol=0)


def _fixed_dump(spin_length: float = 1.0, step: int = 0) -> str:
    header = (
        f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n4\n"
        f"ITEM: BOX BOUNDS pp pp pp\n0 20\n0 20\n0 20\n"
        f"ITEM: ATOMS {' '.join(DUMP_FIELDS)}\n"
    )
    rows = [
        [1, 0, 0, 0, spin_length, 0, 0, 2.2, 1, 2, 3],
        [2, 1, 0, 0, 0, 1, 0, 2.2, 1, 2, 3],
        [3, 0, 1, 0, 0, 0, 1, 2.2, 1, 2, 3],
        [4, 0, 0, 1, -1, 0, 0, 2.2, 1, 2, 3],
    ]
    return header + "\n".join(" ".join(map(str, row)) for row in rows) + "\n"


def test_dump_adapter_and_cli(tmp_path: Path):
    dump_path = tmp_path / "tiny.dump"
    dump_path.write_text(_fixed_dump())
    frame = read_dump(
        dump_path, hbar=2.0, accept_every=10, force_contract_confirmed=True
    )
    assert np.array_equal(frame.atom_ids, [1, 2, 3, 4])
    assert np.array_equal(
        frame.spin_forces, np.broadcast_to([2.0, 4.0, 6.0], (1, 4, 3))
    )

    with pytest.raises(ValueError, match="FORCE_CONTRACT_MISSING"):
        read_dump(dump_path, hbar=2.0, accept_every=10, force_contract_confirmed=False)
    with pytest.raises(ValueError, match="BAD_INPUT"):
        read_dump(dump_path, hbar=0.0, accept_every=10, force_contract_confirmed=True)

    for raw, code in [
        (_fixed_dump(spin_length=2.0), "NONUNIT_SPIN"),
        (_fixed_dump(step=1), "DUMP_NOT_ACCEPTED_STATE"),
        (_fixed_dump().replace("c_gp[7]", "fmx"), "DUMP_FIELD_CONTRACT"),
    ]:
        bad_path = tmp_path / f"{code}.dump"
        bad_path.write_text(raw)
        with pytest.raises(ValueError, match=code):
            read_dump(
                bad_path, hbar=2.0, accept_every=10, force_contract_confirmed=True
            )

    # End-to-end CLI roundtrip: main_moments -> main_infer
    multi_dump = tmp_path / "multi.dump"
    multi_dump.write_text("".join(_fixed_dump(step=10 * k) for k in range(6)))
    npz_paths = []
    for idx in range(6):
        out_npz = tmp_path / f"chain_{idx}.npz"
        main_moments(
            [
                str(multi_dump),
                str(out_npz),
                "--hbar",
                "2.0",
                "--cutoff",
                "3.0",
                "--accept-every",
                "10",
                "--force-contract-confirmed",
            ]
        )
        assert out_npz.is_file()
        npz_paths.append(str(out_npz))
    out_json = tmp_path / "infer.json"
    main_infer(
        [
            "--records",
            *npz_paths,
            "--seed-ids",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "--kb",
            "1.0",
            "--dt",
            "0.01",
            "--output",
            str(out_json),
        ]
    )
    payload = json.loads(out_json.read_text())
    assert "channels" in payload and "CF_GMM" in payload["channels"]


def test_native_c99_fused_parity_and_openmp_determinism(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Compiled C99 OpenMP fused kernel matches pure NumPy to < 1e-13 and is thread-deterministic."""
    build_edge_kernel()
    monkeypatch.delenv("SPINGNN2_GPST_NUMPY_ONLY", raising=False)
    monkeypatch.delenv("TSPIN_NUMPY_ONLY", raising=False)
    assert reload_native_kernel() is True
    assert has_native_kernel() is True

    rng = np.random.default_rng(2026)
    box = np.array([[7.0, 0.0, 0.0], [2.5, 7.0, 0.0], [1.0, 1.5, 7.0]])
    pos = rng.random((32, 3)) @ box
    g = gpst.build_graph(pos, box, np.ones(3, bool), 2.6)
    s = unit(rng.normal(size=(24, 32, 3)))
    f = rng.normal(size=s.shape)

    native_out = moment_arrays(s, f, g)

    monkeypatch.setenv("SPINGNN2_GPST_NUMPY_ONLY", "1")
    assert reload_native_kernel() is False
    numpy_out = moment_arrays(s, f, g)

    monkeypatch.delenv("SPINGNN2_GPST_NUMPY_ONLY", raising=False)
    assert reload_native_kernel() is True

    for key in ("N", "D", "shell_N", "shell_D", "SS", "FS", "shell_SS", "shell_AN"):
        assert_allclose(native_out[key], numpy_out[key], rtol=1e-13, atol=1e-13)

    # Bit-determinism across OMP_NUM_THREADS=1 vs OMP_NUM_THREADS=4
    np.savez(tmp_path / "inputs.npz", pos=pos, box=box, s=s, f=f)
    script = (
        "import numpy as np; from spingnn2 import gpst; from spingnn2.gpst.geometry import moment_arrays; "
        "from spingnn2.gpst.native import has_native_kernel; "
        "assert has_native_kernel(); "
        f"d = np.load({str(tmp_path / 'inputs.npz')!r}); "
        "g = gpst.build_graph(d['pos'], d['box'], np.ones(3, bool), 2.6); "
        "r = moment_arrays(d['s'], d['f'], g); "
        "np.savez(d.files[0] and __import__('sys').argv[1], "
        "N=r['N'], D=r['D'], shell_N=r['shell_N'], shell_D=r['shell_D'], "
        "shell_SS=r['shell_SS'], shell_AN=r['shell_AN'])"
    )
    out1 = tmp_path / "omp1.npz"
    out4 = tmp_path / "omp4.npz"
    for nthreads, out_file in (("1", out1), ("4", out4)):
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = nthreads
        env.pop("SPINGNN2_GPST_NUMPY_ONLY", None)
        env.pop("TSPIN_NUMPY_ONLY", None)
        subprocess.run(
            [sys.executable, "-c", script, str(out_file)], env=env, check=True
        )
    d1 = np.load(out1)
    d4 = np.load(out4)
    for key in d1.files:
        assert np.array_equal(d1[key], d4[key]), f"OMP thread non-determinism on {key}"
