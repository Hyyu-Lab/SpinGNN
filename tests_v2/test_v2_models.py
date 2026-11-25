"""Physics, symmetry (R1-R5), finite-difference, and edge-case tests for SpinGNN2 v2."""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from spingnn2.symmetry import proper_wigner_D
from spingnn2.v2 import (
    C3CutoffEnvelope,
    SpinGNNV2,
    SpinGNNV2Config,
    build_v2_born,
    build_v2_e2slp,
    build_v2_nosoc,
    build_v2_soc,
)


def _random_so3(seed: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn((3, 3), generator=gen, dtype=dtype)
    q, r = torch.linalg.qr(a)
    d = torch.sign(torch.diag(r))
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = q * d.unsqueeze(0)
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _make_periodic_cluster(
    num_atoms: int = 8,
    num_types: int = 2,
    r_cut: float = 5.0,
    seed: int = 7,
    dtype: torch.dtype = torch.float64,
    num_graphs: int = 1,
) -> dict[str, torch.Tensor]:
    """Create a periodic multi-graph cluster with non-collinear spins and PBC edges."""
    gen = torch.Generator().manual_seed(seed)
    pos_list = []
    types_list = []
    spin_list = []
    batch_list = []
    cell_list = []
    edge_dst = []
    edge_src = []
    edge_shifts = []

    offset = 0
    box = 7.5
    for g in range(num_graphs):
        # Asymmetric distorted crystal-like positions well inside r_cut
        base = torch.tensor(
            [
                [0.2, 0.3, 0.1],
                [1.9, 0.4, 0.3],
                [0.5, 2.1, 0.2],
                [2.2, 1.8, 0.6],
                [0.3, 0.6, 2.0],
                [2.0, 0.2, 2.3],
                [0.4, 2.2, 1.9],
                [2.4, 2.1, 2.2],
            ],
            dtype=dtype,
        )
        if num_atoms <= 8:
            p = base[:num_atoms].clone()
        else:
            p = torch.rand((num_atoms, 3), generator=gen, dtype=dtype) * (box * 0.7)
        p = p + 0.17 * torch.randn(p.shape, generator=gen, dtype=dtype)
        cell = torch.diag(torch.tensor([box, box + 0.3, box - 0.2], dtype=dtype))
        cell[0, 1] = 0.25
        cell[1, 2] = -0.15

        t = torch.arange(num_atoms, dtype=torch.long) % num_types
        s = torch.randn((num_atoms, 3), generator=gen, dtype=dtype)
        s = s / torch.linalg.vector_norm(s, dim=-1, keepdim=True) * 2.2

        # Build periodic neighbor list strictly inside r_cut - 0.2 to keep FD graph fixed
        for i in range(num_atoms):
            for j in range(num_atoms):
                for sx in (-1, 0, 1):
                    for sy in (-1, 0, 1):
                        for sz in (-1, 0, 1):
                            if i == j and sx == 0 and sy == 0 and sz == 0:
                                continue
                            sh = torch.tensor([sx, sy, sz], dtype=dtype)
                            vec = p[j] - p[i] + sh @ cell
                            dist = float(torch.linalg.vector_norm(vec).item())
                            if 0.4 < dist < r_cut - 0.25:
                                edge_dst.append(offset + i)
                                edge_src.append(offset + j)
                                edge_shifts.append(sh)

        pos_list.append(p)
        types_list.append(t)
        spin_list.append(s)
        batch_list.append(torch.full((num_atoms,), g, dtype=torch.long))
        cell_list.append(cell)
        offset += num_atoms

    return {
        "pos": torch.cat(pos_list, dim=0),
        "atom_types": torch.cat(types_list, dim=0),
        "spin": torch.cat(spin_list, dim=0),
        "batch": torch.cat(batch_list, dim=0),
        "cell": torch.stack(cell_list, dim=0),
        "edge_index": torch.tensor([edge_dst, edge_src], dtype=torch.long),
        "edge_cell_shift": torch.stack(edge_shifts, dim=0),
        "num_graphs": num_graphs,
    }


def _activate_all_magnetic_heads(model: SpinGNNV2, seed: int = 123) -> None:
    """Set non-zero weights on zero-initialized spin_gate / SOC / mag-BEC heads."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if model.has_spin:
            model.set_spin_gate(0.65)
        if model.has_soc:
            for mod in (
                model.soc_dmi_out,
                model.soc_kit_out,
                model.soc_sia_out,
                model.soc_couple_out,
            ):
                mod.weight.copy_(
                    0.35
                    * torch.randn(
                        mod.weight.shape, generator=gen, dtype=mod.weight.dtype
                    )
                )
        if model.bec_head is not None:
            for mod in (
                model.bec_head.mag_proj_0e,
                model.bec_head.mag_proj_1e,
                model.bec_head.mag_proj_2e,
            ):
                if mod is not None:
                    mod.weight.copy_(
                        0.3
                        * torch.randn(
                            mod.weight.shape, generator=gen, dtype=mod.weight.dtype
                        )
                    )


def _compact_kwargs() -> dict:
    return dict(
        c0=16,
        c1=6,
        c2=4,
        radial_hidden=24,
        fitting_hidden=24,
        c_spin=4,
        c_spin_01=4,
        c_spin_02=2,
        c_spin_11=3,
        c_spin_21=2,
        c_spin_22=2,
        c_soc_dmi=3,
        c_soc_kit=3,
        e2slp_vec_channels=4,
        e2slp_hidden=16,
    )


def test_zero_e3nn_runtime_dependency():
    """Importing spingnn2.v2 in a fresh interpreter must never import e3nn."""
    import spingnn2.v2.model as v2_model
    import spingnn2.v2.optim as v2_optim
    import spingnn2.v2.readouts as v2_readouts
    import spingnn2.v2.stage0 as v2_stage0

    for mod in (v2_model, v2_optim, v2_readouts, v2_stage0):
        src = open(mod.__file__, "r", encoding="utf-8").read()
        assert "import e3nn" not in src
        assert "from e3nn" not in src

    src_root = str(Path(v2_model.__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{src_root}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_root
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, spingnn2.v2; assert 'e3nn' not in sys.modules",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Fresh import failed: {proc.stderr}"


@pytest.mark.parametrize(
    "family",
    ["v2_nosoc", "v2_soc", "v2_born", "v2_born_mag", "v2_e2slp"],
)
def test_finite_difference_forces_torques_virial_fp64(family: str):
    """Exact FP64 finite-difference verification of forces, magnetic torques, and virial."""
    torch.manual_seed(11)
    kw = _compact_kwargs()
    if family == "v2_nosoc":
        model = build_v2_nosoc(num_types=2, r_cut=5.0, use_spin=(True, False), **kw)
    elif family == "v2_soc":
        model = build_v2_soc(num_types=2, r_cut=5.0, use_spin=(True, True), **kw)
    elif family == "v2_born":
        model = build_v2_born(num_types=2, r_cut=5.0, asr="hard", has_spin=False, **kw)
    elif family == "v2_born_mag":
        model = build_v2_born(
            num_types=2,
            r_cut=5.0,
            asr="soft",
            has_spin=True,
            has_soc=True,
            use_spin=(True, False),
            **kw,
        )
    else:
        model = build_v2_e2slp(
            num_types=2,
            r_cut=5.0,
            has_spin=True,
            has_soc=True,
            use_spin=(True, True),
            **kw,
        )

    model = model.double().eval()
    _activate_all_magnetic_heads(model, seed=19)
    data = _make_periodic_cluster(
        num_atoms=6, num_types=2, r_cut=5.0, seed=5, dtype=torch.float64
    )

    out = model(
        pos=data["pos"],
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=data["cell"],
        batch=data["batch"],
        spin=data["spin"] if model.has_spin else None,
        num_graphs=1,
        compute_forces=True,
        compute_virial=True,
        compute_spin_forces=model.has_spin,
        create_graph=False,
    )

    eps = 1e-6
    # 1) Forces FD
    pos0 = data["pos"]
    fd_forces = torch.zeros_like(pos0)
    for a in range(pos0.shape[0]):
        for c in range(3):
            p_plus = pos0.clone()
            p_plus[a, c] += eps
            p_minus = pos0.clone()
            p_minus[a, c] -= eps
            e_plus = model.forward_core(
                p_plus,
                data["atom_types"],
                data["edge_index"],
                data["edge_cell_shift"],
                data["cell"],
                data["batch"],
                data["spin"] if model.has_spin else None,
            )["total_energy"].sum()
            e_minus = model.forward_core(
                p_minus,
                data["atom_types"],
                data["edge_index"],
                data["edge_cell_shift"],
                data["cell"],
                data["batch"],
                data["spin"] if model.has_spin else None,
            )["total_energy"].sum()
            fd_forces[a, c] = -(e_plus - e_minus) / (2.0 * eps)
    torch.testing.assert_close(out["forces"], fd_forces, atol=1e-7, rtol=1e-6)

    # 2) Virial FD (symmetric strain)
    virial = out["virial"][0]
    fd_virial = torch.zeros((3, 3), dtype=torch.float64)
    for i in range(3):
        for j in range(3):
            strain = torch.zeros((3, 3), dtype=torch.float64)
            strain[i, j] += 0.5 * eps
            strain[j, i] += 0.5 * eps
            p_plus = pos0 + pos0 @ strain
            c_plus = data["cell"] + data["cell"] @ strain
            p_minus = pos0 - pos0 @ strain
            c_minus = data["cell"] - data["cell"] @ strain
            e_plus = model.forward_core(
                p_plus,
                data["atom_types"],
                data["edge_index"],
                data["edge_cell_shift"],
                c_plus,
                data["batch"],
                data["spin"] if model.has_spin else None,
            )["total_energy"].sum()
            e_minus = model.forward_core(
                p_minus,
                data["atom_types"],
                data["edge_index"],
                data["edge_cell_shift"],
                c_minus,
                data["batch"],
                data["spin"] if model.has_spin else None,
            )["total_energy"].sum()
            fd_virial[i, j] = -(e_plus - e_minus) / (2.0 * eps)
    torch.testing.assert_close(virial, fd_virial, atol=1e-7, rtol=1e-6)

    # 3) Magnetic torques / spin forces FD
    if model.has_spin:
        s0 = data["spin"]
        fd_spin = torch.zeros_like(s0)
        for a in range(s0.shape[0]):
            for c in range(3):
                s_plus = s0.clone()
                s_plus[a, c] += eps
                s_minus = s0.clone()
                s_minus[a, c] -= eps
                e_plus = model.forward_core(
                    pos0,
                    data["atom_types"],
                    data["edge_index"],
                    data["edge_cell_shift"],
                    data["cell"],
                    data["batch"],
                    s_plus,
                )["total_energy"].sum()
                e_minus = model.forward_core(
                    pos0,
                    data["atom_types"],
                    data["edge_index"],
                    data["edge_cell_shift"],
                    data["cell"],
                    data["batch"],
                    s_minus,
                )["total_energy"].sum()
                fd_spin[a, c] = -(e_plus - e_minus) / (2.0 * eps)
        torch.testing.assert_close(out["spin_forces"], fd_spin, atol=1e-7, rtol=1e-6)
        assert float(out["spin_forces"].abs().max().item()) > 1e-4


def test_r1_r5_symmetry_nosoc_decoupled_fp64():
    """Exact O(3)_lat x O(3)_spin symmetry (< 1e-13 FP64) + superexchange & Mw checks."""
    torch.manual_seed(21)
    model = (
        build_v2_nosoc(
            num_types=2, r_cut=5.0, use_spin=(True, False), **_compact_kwargs()
        )
        .double()
        .eval()
    )
    _activate_all_magnetic_heads(model, seed=31)
    data = _make_periodic_cluster(
        num_atoms=6, num_types=2, r_cut=5.0, seed=9, dtype=torch.float64
    )

    base = model(
        pos=data["pos"],
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=data["cell"],
        batch=data["batch"],
        spin=data["spin"],
        compute_forces=True,
        compute_virial=True,
        compute_spin_forces=True,
        create_graph=False,
    )

    r_lat = _random_so3(101)
    r_spin = _random_so3(202)
    # Cross-check Wigner-D l=1 matches R_lat
    d1 = proper_wigner_D(1, r_lat)
    torch.testing.assert_close(d1, r_lat, atol=1e-13, rtol=1e-13)

    # 1) Independent spin-only rotation (fixed lattice!)
    out_srot = model(
        pos=data["pos"],
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=data["cell"],
        batch=data["batch"],
        spin=data["spin"] @ r_spin.T,
        compute_forces=True,
        compute_virial=True,
        compute_spin_forces=True,
        create_graph=False,
    )
    torch.testing.assert_close(
        out_srot["total_energy"], base["total_energy"], atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(
        out_srot["forces"], base["forces"], atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(
        out_srot["spin_forces"], base["spin_forces"] @ r_spin.T, atol=1e-13, rtol=1e-13
    )

    # 2) Independent lattice-only rotation (fixed spins!)
    out_lrot = model(
        pos=data["pos"] @ r_lat.T,
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=data["cell"] @ r_lat.T,
        batch=data["batch"],
        spin=data["spin"],
        compute_forces=True,
        compute_virial=True,
        compute_spin_forces=True,
        create_graph=False,
    )
    torch.testing.assert_close(
        out_lrot["total_energy"], base["total_energy"], atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(
        out_lrot["forces"], base["forces"] @ r_lat.T, atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(
        out_lrot["spin_forces"], base["spin_forces"], atol=1e-13, rtol=1e-13
    )

    # 3) Spatial inversion P: r -> -r, s -> +s
    out_p = model(
        pos=-data["pos"],
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=-data["cell"],
        batch=data["batch"],
        spin=data["spin"],
        compute_forces=True,
        compute_spin_forces=True,
        create_graph=False,
    )
    torch.testing.assert_close(
        out_p["total_energy"], base["total_energy"], atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(out_p["forces"], -base["forces"], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(
        out_p["spin_forces"], base["spin_forces"], atol=1e-13, rtol=1e-13
    )

    # 4) Time reversal T: r -> +r, s -> -s
    out_t = model(
        pos=data["pos"],
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=data["cell"],
        batch=data["batch"],
        spin=-data["spin"],
        compute_forces=True,
        compute_spin_forces=True,
        create_graph=False,
    )
    torch.testing.assert_close(
        out_t["total_energy"], base["total_energy"], atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(out_t["forces"], base["forces"], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(
        out_t["spin_forces"], -base["spin_forces"], atol=1e-13, rtol=1e-13
    )

    # 5) Paramagnetic coordination Mw distinguishes demagnetized vs nonmagnetic
    out_zero_spin = model.forward_core(
        pos=data["pos"],
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=data["cell"],
        batch=data["batch"],
        spin=torch.zeros_like(data["spin"]),
    )
    assert float(out_zero_spin["paramagnetic_mw"].abs().max().item()) > 1e-3

    # 6) Superexchange bond-angle sensitivity at fixed bond lengths:
    # Center nonmagnetic O at origin, two magnetic Fe neighbors at distance 2.0
    # with varying bond angle theta and antiparallel vs parallel spins.
    def _tri_energy(angle: float, parallel: bool) -> float:
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0 * math.cos(angle), 2.0 * math.sin(angle), 0.0],
            ],
            dtype=torch.float64,
        )
        # Edges only between center (0) and neighbors (1, 2) so d(1,2) doesn't enter
        eidx = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        atypes = torch.tensor([1, 0, 0], dtype=torch.long)
        s2 = [1.0, 0.0, 0.0] if parallel else [-1.0, 0.0, 0.0]
        sp = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], s2], dtype=torch.float64)
        return float(
            model.forward_core(pos, atypes, eidx, spin=sp)["per_atom_energy"][0].item()
        )

    diff_90 = _tri_energy(math.pi / 2.0, True) - _tri_energy(math.pi / 2.0, False)
    diff_180 = _tri_energy(math.pi, True) - _tri_energy(math.pi, False)
    assert (
        abs(diff_90 - diff_180) > 1e-6
    ), "NoSOC must capture superexchange bond-angle coupling!"


def test_r1_r5_symmetry_soc_born_e2slp_fp64():
    """Exact O(3) x Z_2^T symmetry (< 1e-13 FP64) for v2_soc, v2_born, and v2_e2slp."""
    torch.manual_seed(33)
    kw = _compact_kwargs()
    model = (
        SpinGNNV2(
            SpinGNNV2Config(
                num_types=2,
                r_cut=5.0,
                has_spin=True,
                has_soc=True,
                use_spin=(True, False),
                predict_bec=True,
                bec_asr="hard",
                predict_e2slp=True,
                e2slp_include_f=True,
                e2slp_include_sf=True,
                predict_gamma=True,
                **kw,
            )
        )
        .double()
        .eval()
    )
    _activate_all_magnetic_heads(model, seed=44)

    data = _make_periodic_cluster(
        num_atoms=6, num_types=2, r_cut=5.0, seed=13, dtype=torch.float64, num_graphs=2
    )
    base = model(
        pos=data["pos"],
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=data["cell"],
        batch=data["batch"],
        spin=data["spin"],
        num_graphs=2,
        compute_forces=True,
        compute_virial=True,
        compute_spin_forces=True,
        create_graph=False,
    )

    # Hard ASR: per-structure sum of BEC vanishes exactly
    for g in range(2):
        mask = data["batch"] == g
        torch.testing.assert_close(
            base["born_effective_charges"][mask].sum(dim=0),
            torch.zeros((3, 3), dtype=torch.float64),
            atol=1e-13,
            rtol=1e-13,
        )
    torch.testing.assert_close(
        base["total_born_effective_charges"],
        torch.zeros((2, 3, 3), dtype=torch.float64),
        atol=1e-13,
        rtol=1e-13,
    )

    # Verify SOC genuinely breaks independent spin-only rotation
    r_spin_only = _random_so3(303)
    out_s_only = model.forward_core(
        pos=data["pos"],
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=data["cell"],
        batch=data["batch"],
        spin=data["spin"] @ r_spin_only.T,
        num_graphs=2,
    )
    assert (
        float((out_s_only["total_energy"] - base["total_energy"]).abs().max().item())
        > 1e-5
    )

    # 1) Global SO(3) rotation (R*pos, R*spin)
    r_glob = _random_so3(404)
    out_rot = model(
        pos=data["pos"] @ r_glob.T,
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=data["cell"] @ r_glob.T,
        batch=data["batch"],
        spin=data["spin"] @ r_glob.T,
        num_graphs=2,
        compute_forces=True,
        compute_virial=True,
        compute_spin_forces=True,
        create_graph=False,
    )
    torch.testing.assert_close(
        out_rot["total_energy"], base["total_energy"], atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(
        out_rot["forces"], base["forces"] @ r_glob.T, atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(
        out_rot["spin_forces"], base["spin_forces"] @ r_glob.T, atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(
        out_rot["virial"], r_glob @ base["virial"] @ r_glob.T, atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(
        out_rot["born_effective_charges"],
        r_glob @ base["born_effective_charges"] @ r_glob.T,
        atol=1e-13,
        rtol=1e-13,
    )
    for key in ("f_sigma", "f_Uepi", "sf_sigma", "sf_Uepi"):
        torch.testing.assert_close(
            out_rot[key], r_glob @ base[key] @ r_glob.T, atol=1e-13, rtol=1e-13
        )
    for key in (
        "f_nu",
        "f_kappa",
        "f_uncertainty",
        "sf_nu",
        "sf_kappa",
        "sf_uncertainty",
        "gamma",
    ):
        torch.testing.assert_close(out_rot[key], base[key], atol=1e-13, rtol=1e-13)

    # 2) Spatial inversion P: r -> -r, s -> +s (BEC is EVEN parity: Z* -> +Z*!)
    out_p = model(
        pos=-data["pos"],
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=-data["cell"],
        batch=data["batch"],
        spin=data["spin"],
        num_graphs=2,
        compute_forces=True,
        compute_spin_forces=True,
        create_graph=False,
    )
    torch.testing.assert_close(
        out_p["total_energy"], base["total_energy"], atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(out_p["forces"], -base["forces"], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(
        out_p["spin_forces"], base["spin_forces"], atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(
        out_p["born_effective_charges"],
        base["born_effective_charges"],
        atol=1e-13,
        rtol=1e-13,
    )
    for key in ("f_sigma", "f_Uepi", "sf_sigma", "sf_Uepi", "f_nu", "sf_nu", "gamma"):
        torch.testing.assert_close(out_p[key], base[key], atol=1e-13, rtol=1e-13)

    # 3) Time reversal T: r -> +r, s -> -s (BEC is T-even: Z* -> +Z*!)
    out_t = model(
        pos=data["pos"],
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=data["cell"],
        batch=data["batch"],
        spin=-data["spin"],
        num_graphs=2,
        compute_forces=True,
        compute_spin_forces=True,
        create_graph=False,
    )
    torch.testing.assert_close(
        out_t["total_energy"], base["total_energy"], atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(out_t["forces"], base["forces"], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(
        out_t["spin_forces"], -base["spin_forces"], atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(
        out_t["born_effective_charges"],
        base["born_effective_charges"],
        atol=1e-13,
        rtol=1e-13,
    )
    for key in ("f_sigma", "f_Uepi", "sf_sigma", "sf_Uepi"):
        torch.testing.assert_close(out_t[key], base[key], atol=1e-13, rtol=1e-13)

    # 4) Evidential covariance SPD & spin tangent projection checks
    f_eigs = torch.linalg.eigvalsh(base["f_sigma"])
    assert bool((f_eigs > 0.0).all().item())
    mag_mask = data["atom_types"] == 0
    nonmag_mask = data["atom_types"] == 1
    torch.testing.assert_close(
        base["sf_sigma"][nonmag_mask],
        torch.zeros_like(base["sf_sigma"][nonmag_mask]),
        atol=1e-15,
        rtol=0.0,
    )
    # sf_sigma @ spin == 0 on tangent plane
    proj_check = torch.bmm(
        base["sf_sigma"][mag_mask], data["spin"][mag_mask].unsqueeze(-1)
    )
    torch.testing.assert_close(
        proj_check, torch.zeros_like(proj_check), atol=1e-13, rtol=1e-13
    )


def test_soc_zero_init_exact_identity_and_nonzero_grad():
    """At init: E_SOC == E_noSOC (0.0 diff on spin-only rot) AND SOC params get nonzero grads."""
    torch.manual_seed(55)
    model = build_v2_soc(
        num_types=2, r_cut=5.0, use_spin=(True, True), **_compact_kwargs()
    ).double()
    data = _make_periodic_cluster(
        num_atoms=6, num_types=2, r_cut=5.0, seed=17, dtype=torch.float64
    )

    out0 = model.forward_core(
        pos=data["pos"],
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=data["cell"],
        batch=data["batch"],
        spin=data["spin"],
    )
    # Exact 0.0 difference between E_SOC and E_noSOC at initialization
    assert (
        float((out0["per_atom_energy"] - out0["energy_nosoc"]).abs().max().item())
        == 0.0
    )
    assert float(out0["energy_soc_delta"].abs().max().item()) == 0.0

    # Spin-only rotation diff at init is strictly 0.0
    r_spin = _random_so3(505)
    out_rot = model.forward_core(
        pos=data["pos"],
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=data["cell"],
        batch=data["batch"],
        spin=data["spin"] @ r_spin.T,
    )
    assert (
        float((out_rot["total_energy"] - out0["total_energy"]).abs().max().item())
        == 0.0
    )

    # Backward: every SOC output head must receive strictly non-zero gradients!
    out0["total_energy"].sum().backward()
    for name, mod in (
        ("soc_dmi_out", model.soc_dmi_out),
        ("soc_kit_out", model.soc_kit_out),
        ("soc_sia_out", model.soc_sia_out),
        ("soc_couple_out", model.soc_couple_out),
    ):
        assert mod.weight.grad is not None, f"{name} missing gradient"
        assert (
            float(mod.weight.grad.abs().max().item()) > 1e-6
        ), f"{name} got zero gradient!"
    assert model.adam_spin_gate.grad is not None
    assert float(model.adam_spin_gate.grad.abs().max().item()) > 1e-6


def test_collinear_theta_to_zero_smoothness():
    """Verify v2_soc magnetic torques remain smooth and finite as angle theta -> 0."""
    torch.manual_seed(66)
    model = (
        build_v2_soc(num_types=2, r_cut=5.0, use_spin=(True, True), **_compact_kwargs())
        .double()
        .eval()
    )
    _activate_all_magnetic_heads(model, seed=67)

    # Asymmetric 3-atom geometry so DMI & Kitaev & SIA are all active
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [2.1, 0.2, 0.1], [0.4, 2.0, 0.7]], dtype=torch.float64
    )
    atypes = torch.tensor([0, 0, 1], dtype=torch.long)
    eidx = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long)

    torques = []
    for theta in (1e-1, 1e-3, 1e-5, 1e-7, 0.0):
        spins = torch.tensor(
            [
                [0.0, 0.0, 2.0],
                [2.0 * math.sin(theta), 0.0, 2.0 * math.cos(theta)],
                [0.0, 2.0 * math.sin(theta), 2.0 * math.cos(theta)],
            ],
            dtype=torch.float64,
        )
        out = model(
            pos=pos,
            atom_types=atypes,
            edge_index=eidx,
            spin=spins,
            compute_forces=True,
            compute_spin_forces=True,
            create_graph=False,
        )
        assert torch.isfinite(out["total_energy"]).all()
        assert torch.isfinite(out["forces"]).all()
        assert torch.isfinite(out["spin_forces"]).all()
        torques.append(out["spin_forces"].detach())

    # Torque converges smoothly to the exact theta=0 value (O(theta) difference, zero 1/theta blowup)
    diff_1e5 = float((torques[2] - torques[-1]).abs().max().item())
    diff_1e7 = float((torques[3] - torques[-1]).abs().max().item())
    assert diff_1e5 < 1e-3
    assert diff_1e7 < diff_1e5 * 0.05 + 1e-12


def test_nonmagnetic_atom_zero_torque_and_second_derivatives():
    """Nonmagnetic species must have exact zero 1st & 2nd spin derivatives."""
    torch.manual_seed(77)
    model = (
        build_v2_soc(
            num_types=2, r_cut=5.0, use_spin=(True, False), **_compact_kwargs()
        )
        .double()
        .eval()
    )
    _activate_all_magnetic_heads(model, seed=78)
    data = _make_periodic_cluster(
        num_atoms=6, num_types=2, r_cut=5.0, seed=23, dtype=torch.float64
    )

    nonmag_idx = (data["atom_types"] == 1).nonzero(as_tuple=True)[0]
    assert nonmag_idx.numel() > 0

    pos = data["pos"].clone().requires_grad_(True)
    spin = data["spin"].clone().requires_grad_(True)
    out = model(
        pos=pos,
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        edge_cell_shift=data["edge_cell_shift"],
        cell=data["cell"],
        batch=data["batch"],
        spin=spin,
        compute_forces=True,
        compute_spin_forces=True,
        create_graph=True,
    )
    fm = out["spin_forces"]
    assert float(fm[nonmag_idx].abs().max().item()) == 0.0

    # Second-order derivatives wrt nonmagnetic spins must also be strictly zero
    loss = out["forces"].square().sum() + fm.square().sum()
    (d_spin,) = torch.autograd.grad(loss, spin, allow_unused=True)
    assert d_spin is not None
    assert float(d_spin[nonmag_idx].abs().max().item()) == 0.0


def test_vacuum_ref_and_c3_cutoff_smoothness():
    """Isolated atom matches E_iso exactly; C^3 envelope & derivatives vanish at r_cut."""
    torch.manual_seed(88)
    model = (
        build_v2_soc(
            num_types=2, r_cut=4.0, use_spin=(True, False), **_compact_kwargs()
        )
        .double()
        .eval()
    )
    _activate_all_magnetic_heads(model, seed=89)
    with torch.no_grad():
        model.energy_head_lat.adam_iso_energy.copy_(
            torch.tensor([-3.25, -1.75], dtype=torch.float64)
        )

    pos = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float64)
    atypes = torch.tensor([0, 1], dtype=torch.long)
    empty_edges = torch.zeros((2, 0), dtype=torch.long)
    spin = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)

    out_vac = model.forward_core(pos, atypes, empty_edges, spin=spin)
    torch.testing.assert_close(
        out_vac["per_atom_energy"],
        torch.tensor([-3.25, -1.75], dtype=torch.float64),
        atol=1e-14,
        rtol=1e-14,
    )

    # C^3 envelope value, 1st, 2nd, 3rd derivatives vanish at r -> r_cut
    env = C3CutoffEnvelope(r_cut=4.0, exponent=5).double()
    r = torch.tensor([4.0 - 1e-4, 4.0, 4.5], dtype=torch.float64, requires_grad=True)
    y = env(r)
    (dy,) = torch.autograd.grad(y.sum(), r, create_graph=True)
    (d2y,) = torch.autograd.grad(dy.sum(), r, create_graph=True)
    (d3y,) = torch.autograd.grad(d2y.sum(), r)
    assert float(y[1].item()) == 0.0 and float(y[2].item()) == 0.0
    assert abs(float(dy[0].item())) < 1e-9
    assert abs(float(d2y[0].item())) < 1e-5
    assert abs(float(d3y[0].item())) < 2e-2
