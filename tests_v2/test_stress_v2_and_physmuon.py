"""Adversarial stress & limit audit suite for SpinGNN2 v2, PhysMuon/Muon, and v1 YAML trainer.

Covers:
1. Extreme v2 configs: num_types in (1, 3, 8), lmax in (1, 2, 3), narrow/odd/wide channels,
   all-nonmagnetic / mixed / all-magnetic spin masks, all 4 model families, and invalid-config
   ValueError validation.
2. Degenerate & large graphs: E = 0 isolated atoms, N = 1 single atom (isolated + periodic),
   and N = 2048 large supercell (~60k edges on CUDA) with edge_amp + edge_checkpoint.
3. Varying-shape batch sequence [32/400 -> 512/15k -> 64/0 -> 1024/30k -> 2048/60k] under
   both compile_fullgraph(dynamic=True) and compile_bucketed(..., mode="reduce-overhead")
   verifying graceful out-of-bucket overflow fallback without unbounded CUDA Graph pools.
4. PhysMuon & Muon adversarial stress (CPU & CUDA cuda_graph=True): lr=0, wd=0.1, zero grads,
   1000x spike_guard freeze, NaN/Inf safe skip, frozen requires_grad=False subset, weird
   shapes (1xK, Kx1, 2x2, 3x3, 4x4, 3D, 4D), changing grad=None subsets crossing bm_start
   asynchronously, and mid-bm_start state_dict save/resume parity.
5. 50-step realistic noisy GPU training stability (E + F + SF + BEC + bf16 edge AMP +
   selective checkpointing + PhysMuon + WSD + injected spike & NaN recovery -> monotonic
   loss drop, 0 NaN/Inf, flat VRAM).
6. Real v1 YAML Allegro_nosoc, Allegro_soc, Allegro_born, and Allegro_soc_e2IP Trainer smoke
   with both PhysMuon (interatomic_potential_gated) and Muon.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from spingnn2.v2 import (
    Muon,
    PhysMuon,
    SpinGNNV2,
    SpinGNNV2Config,
    WSDCosineScheduler,
    build_v2_born,
    build_v2_e2slp,
    build_v2_nosoc,
    build_v2_soc,
    interatomic_potential_gated,
)


def _make_graph(
    n_atoms: int,
    n_edges: int,
    num_types: int = 2,
    r_cut: float = 5.5,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    num_graphs: int = 1,
) -> dict[str, torch.Tensor]:
    """Build a synthetic well-conditioned periodic-free graph with exact (N, E) sizes."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    box = max(3.5, (float(n_atoms) * 14.0) ** (1.0 / 3.0))
    pos = torch.rand((n_atoms, 3), generator=gen, dtype=dtype) * box
    atom_types = torch.randint(
        0, num_types, (n_atoms,), generator=gen, dtype=torch.long
    )

    # Non-collinear unit-ish spins
    spin = torch.randn((n_atoms, 3), generator=gen, dtype=dtype)
    spin = spin / spin.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 2.0

    if num_graphs == 1:
        batch = torch.zeros(n_atoms, dtype=torch.long)
    else:
        batch = torch.arange(n_atoms, dtype=torch.long) * num_graphs // max(n_atoms, 1)

    if n_edges == 0 or n_atoms == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_cell_shift = torch.zeros((0, 3), dtype=dtype)
        cell = torch.eye(3, dtype=dtype).unsqueeze(0).expand(num_graphs, 3, 3) * (
            box + 2.0 * r_cut
        )
    elif n_atoms == 1:
        # Self-image periodic edges along +/- axes within r_cut
        dirs = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=dtype,
        )
        reps = (n_edges + 5) // 6
        edge_cell_shift = dirs.repeat(reps, 1)[:n_edges]
        edge_index = torch.zeros((2, n_edges), dtype=torch.long)
        cell = torch.eye(3, dtype=dtype).unsqueeze(0).expand(num_graphs, 3, 3) * (
            0.55 * r_cut
        )
    else:
        # Construct deterministic pairs with distances strictly inside (0.8, 0.9*r_cut)
        dst = torch.arange(n_edges, dtype=torch.long) % n_atoms
        offset = (torch.arange(n_edges, dtype=torch.long) // n_atoms) % (
            n_atoms - 1
        ) + 1
        src = (dst + offset) % n_atoms
        # Keep edges within the same graph when num_graphs > 1
        if num_graphs > 1:
            same_graph = batch[dst] == batch[src]
            if not bool(same_graph.all()):
                # Map each atom to neighbors in its own graph segment
                dst_list = []
                src_list = []
                per_g = max(1, n_edges // num_graphs)
                for g in range(num_graphs):
                    idx_g = torch.nonzero(batch == g, as_tuple=False).squeeze(-1)
                    ng = idx_g.numel()
                    if ng < 2:
                        continue
                    cnt = per_g if g < num_graphs - 1 else (n_edges - len(dst_list))
                    d_loc = torch.arange(cnt, dtype=torch.long) % ng
                    o_loc = (torch.arange(cnt, dtype=torch.long) // ng) % (ng - 1) + 1
                    s_loc = (d_loc + o_loc) % ng
                    dst_list.append(idx_g[d_loc])
                    src_list.append(idx_g[s_loc])
                dst = torch.cat(dst_list)[:n_edges]
                src = torch.cat(src_list)[:n_edges]
        edge_index = torch.stack((dst, src), dim=0)
        # Compact coordinates so edges lie comfortably within r_cut
        pos = torch.rand((n_atoms, 3), generator=gen, dtype=dtype) * (0.45 * r_cut)
        edge_cell_shift = torch.zeros((n_edges, 3), dtype=dtype)
        cell = torch.eye(3, dtype=dtype).unsqueeze(0).expand(num_graphs, 3, 3) * (
            2.0 * r_cut
        )

    dev = torch.device(device)
    return {
        "pos": pos.to(dev),
        "atom_types": atom_types.to(dev),
        "edge_index": edge_index.to(dev),
        "edge_cell_shift": edge_cell_shift.to(dev),
        "cell": cell.to(dev),
        "batch": batch.to(dev),
        "spin": spin.to(dev),
    }


def _assert_all_finite(out: dict[str, torch.Tensor], model: nn.Module) -> None:
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"Non-finite output tensor {k!r}"
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"Non-finite parameter {name!r}"
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name!r}"


# ===========================================================================
# 1. Extreme v2 Configs & Input Validation
# ===========================================================================


@pytest.mark.parametrize("num_types", [1, 3, 8])
@pytest.mark.parametrize("lmax", [1, 2, 3])
def test_extreme_v2_num_types_and_lmax(num_types: int, lmax: int) -> None:
    """Verify all 4 families across num_types=(1,3,8) and lmax=(1,2,3)."""
    torch.manual_seed(100 + num_types * 10 + lmax)
    data = _make_graph(
        n_atoms=10, n_edges=40, num_types=num_types, seed=num_types + lmax
    )
    use_spin = tuple(i % 2 == 0 for i in range(num_types))

    common = dict(
        num_types=num_types,
        lmax=lmax,
        c0=24,
        c1=5,
        c2=4,
        c3=3,
        radial_hidden=24,
        fitting_hidden=24,
        c_spin=4,
        c_spin_01=3,
        c_spin_02=2,
        c_spin_11=3,
        c_spin_21=2,
        c_spin_22=2,
        c_soc_dmi=2,
        c_soc_kit=2,
    )
    models = [
        build_v2_nosoc(use_spin=use_spin, **common),
        build_v2_soc(use_spin=use_spin, **common),
        build_v2_born(
            asr="hard",
            has_spin=True,
            has_soc=True,
            use_spin=use_spin,
            **common,
        ),
        build_v2_e2slp(
            has_spin=True,
            has_soc=True,
            use_spin=use_spin,
            include_f=True,
            include_sf=True,
            predict_gamma=True,
            **common,
        ),
    ]

    for model in models:
        model.train()
        if model.has_spin:
            model.set_spin_gate(0.5)
        out = model(
            pos=data["pos"].clone().requires_grad_(True),
            atom_types=data["atom_types"],
            edge_index=data["edge_index"],
            edge_cell_shift=data["edge_cell_shift"],
            cell=data["cell"],
            batch=data["batch"],
            spin=data["spin"].clone().requires_grad_(True) if model.has_spin else None,
            num_graphs=1,
            compute_forces=True,
            compute_virial=True,
            compute_spin_forces=model.has_spin,
            create_graph=True,
        )
        loss = (
            out["total_energy"].sum()
            + 0.1 * out["forces"].square().mean()
            + 0.1 * out["virial"].square().mean()
        )
        if model.has_spin:
            loss = loss + 0.1 * out["spin_forces"].square().mean()
            # Nonmagnetic species must have identically zero spin_forces
            nonmag = ~torch.tensor(use_spin)[data["atom_types"]]
            if bool(nonmag.any()):
                assert torch.max(torch.abs(out["spin_forces"][nonmag])).item() == 0.0
        if "born_effective_charges" in out:
            loss = loss + 0.1 * out["born_effective_charges"].square().mean()
            # Hard ASR must satisfy exact zero sum per structure
            bec_sum = out["born_effective_charges"].sum(dim=0)
            assert torch.max(torch.abs(bec_sum)).item() < 1e-5
        if "f_uncertainty" in out:
            loss = loss + 0.1 * out["f_uncertainty"].sum()
        if "sf_uncertainty" in out:
            loss = loss + 0.1 * out["sf_uncertainty"].sum()
        if model.has_soc:
            # Zero-init SOC head must produce exact 0.0 delta at init
            assert torch.max(torch.abs(out["energy_soc_delta"])).item() == 0.0

        loss.backward()
        _assert_all_finite(out, model)
        if model.has_soc:
            # Zero-init SOC output weights must immediately receive non-zero grads
            assert model.soc_dmi_out.weight.grad is not None
            assert float(model.soc_dmi_out.weight.grad.abs().sum().item()) > 0.0


@pytest.mark.parametrize(
    "channels",
    [
        (16, 2, 2, 2),
        (33, 5, 5, 3),
        (64, 16, 8, 4),
        (128, 16, 8, 4),
    ],
)
@pytest.mark.parametrize(
    "use_spin",
    [
        (False, False, False),
        (True, False, True),
        (True, True, True),
    ],
)
def test_extreme_v2_channels_and_spin_masks(
    channels: tuple[int, int, int, int],
    use_spin: tuple[bool, bool, bool],
) -> None:
    """Test narrow, odd, and wide channel widths with all-nonmag / mixed / all-mag masks."""
    c0, c1, c2, c3 = channels
    torch.manual_seed(c0 + c1 + sum(use_spin))
    data = _make_graph(n_atoms=12, n_edges=48, num_types=3, seed=c0)
    model = build_v2_soc(
        num_types=3,
        lmax=2,
        c0=c0,
        c1=c1,
        c2=c2,
        c3=c3,
        c_spin=min(c0, 6),
        c_spin_01=4,
        c_spin_02=3,
        c_spin_11=3,
        c_spin_21=2,
        c_spin_22=2,
        c_soc_dmi=3,
        c_soc_kit=3,
        use_spin=use_spin,
        predict_bec=True,
        predict_e2slp=True,
        e2slp_include_sf=True,
    )
    model.set_spin_gate(0.5)
    out = model(
        pos=data["pos"].clone().requires_grad_(True),
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        spin=data["spin"].clone().requires_grad_(True),
        batch=data["batch"],
        num_graphs=1,
        compute_forces=True,
        compute_virial=False,
        compute_spin_forces=True,
        create_graph=True,
    )
    loss = (
        out["total_energy"].sum()
        + 0.1 * out["forces"].square().mean()
        + 0.1 * out["spin_forces"].square().mean()
        + 0.1 * out["born_effective_charges"].square().mean()
        + 0.05 * out["f_uncertainty"].sum()
        + 0.05 * out["sf_uncertainty"].sum()
    )
    loss.backward()
    _assert_all_finite(out, model)
    if not any(use_spin):
        assert torch.max(torch.abs(out["spin_forces"])).item() == 0.0


def test_v2_invalid_configs_raise_clear_value_error() -> None:
    """Invalid channel/lmax/spin configs must raise explicit ValueError at __init__."""
    with pytest.raises(ValueError, match="lmax"):
        SpinGNNV2(SpinGNNV2Config(lmax=0))
    with pytest.raises(ValueError, match="lmax"):
        SpinGNNV2(SpinGNNV2Config(lmax=4))
    with pytest.raises(ValueError, match="num_types"):
        SpinGNNV2(SpinGNNV2Config(num_types=0))
    with pytest.raises(ValueError, match="c0"):
        SpinGNNV2(SpinGNNV2Config(c0=8, c1=16, c2=4))
    with pytest.raises(ValueError, match="c0"):
        SpinGNNV2(SpinGNNV2Config(c0=8, c1=4, c2=4, has_spin=True, c_spin=12))
    with pytest.raises(ValueError, match="has_soc"):
        SpinGNNV2(SpinGNNV2Config(has_spin=False, has_soc=True))
    with pytest.raises(ValueError, match="e2slp_include_sf"):
        SpinGNNV2(
            SpinGNNV2Config(has_spin=False, predict_e2slp=True, e2slp_include_sf=True)
        )


# ===========================================================================
# 2. Degenerate Graphs (E = 0, N = 1) & Large Supercell (N = 2048)
# ===========================================================================


@pytest.mark.parametrize("n_atoms,n_edges", [(1, 0), (8, 0), (1, 6)])
@pytest.mark.parametrize("lmax", [1, 2])
def test_degenerate_graphs_e0_and_n1(n_atoms: int, n_edges: int, lmax: int) -> None:
    """E=0 isolated atoms and N=1 single atoms (isolated + periodic) across all 4 families."""
    torch.manual_seed(303 + n_atoms + n_edges + lmax)
    data = _make_graph(n_atoms=n_atoms, n_edges=n_edges, num_types=2, seed=77)

    common = dict(
        num_types=2,
        lmax=lmax,
        c0=24,
        c1=6,
        c2=4,
        radial_hidden=24,
        fitting_hidden=24,
        c_spin=4,
        c_spin_01=3,
        c_spin_02=2,
        c_spin_11=3,
        c_spin_21=2,
        c_spin_22=2,
        c_soc_dmi=2,
        c_soc_kit=2,
    )
    models = [
        build_v2_nosoc(use_spin=(True, False), **common),
        build_v2_soc(use_spin=(True, True), **common),
        build_v2_born(asr="hard", has_spin=True, has_soc=True, **common),
        build_v2_e2slp(
            has_spin=True,
            has_soc=True,
            include_f=True,
            include_sf=True,
            **common,
        ),
    ]
    for model in models:
        model.train()
        out = model(
            pos=data["pos"].clone().requires_grad_(True),
            atom_types=data["atom_types"],
            edge_index=data["edge_index"],
            edge_cell_shift=data["edge_cell_shift"],
            cell=data["cell"],
            batch=data["batch"],
            spin=data["spin"].clone().requires_grad_(True) if model.has_spin else None,
            num_graphs=1,
            compute_forces=True,
            compute_virial=True,
            compute_spin_forces=model.has_spin,
            create_graph=True,
        )
        if n_edges == 0:
            # Isolated atoms must hit exact vacuum reference energy and zero forces
            vac_expected = model.energy_head_lat.adam_iso_energy.index_select(
                0, data["atom_types"]
            )
            assert torch.allclose(out["energy_onlyr"], vac_expected, atol=1e-6)
            assert torch.max(torch.abs(out["forces"])).item() == 0.0
            assert torch.max(torch.abs(out["virial"])).item() == 0.0
        if n_atoms == 1 and "born_effective_charges" in out:
            # 1-atom hard ASR must be identically zero
            assert torch.max(torch.abs(out["born_effective_charges"])).item() < 1e-6

        loss = out["total_energy"].sum() + 0.1 * out["forces"].square().sum()
        if model.has_spin:
            loss = loss + 0.1 * out["spin_forces"].square().sum()
        if "born_effective_charges" in out:
            loss = loss + 0.1 * out["born_effective_charges"].square().sum()
        if "f_uncertainty" in out:
            loss = loss + 0.1 * out["f_uncertainty"].sum()
        loss.backward()
        _assert_all_finite(out, model)


def test_tiny_edge_and_spin_second_order_fp32() -> None:
    """Tiny ~1e-10 edges & spins must stay finite through 2nd-order autograd in FP32."""
    torch.manual_seed(991)
    data = _make_graph(n_atoms=6, n_edges=18, num_types=2, seed=99)
    # Collapse one edge to ~1e-10 and one spin to ~1e-10
    pos = data["pos"].clone()
    pos[1] = pos[0] + torch.tensor([1e-10, -1e-10, 0.0])
    spin = data["spin"].clone()
    spin[0] = torch.tensor([1e-10, 0.0, 0.0])
    spin[2] = torch.tensor([0.0, 0.0, 0.0])

    model = build_v2_soc(
        num_types=2,
        lmax=2,
        c0=24,
        c1=6,
        c2=4,
        predict_bec=True,
        predict_e2slp=True,
        e2slp_include_sf=True,
    )
    model.set_spin_gate(0.5)
    out = model(
        pos=pos.requires_grad_(True),
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        spin=spin.requires_grad_(True),
        batch=data["batch"],
        num_graphs=1,
        compute_forces=True,
        compute_virial=False,
        compute_spin_forces=True,
        create_graph=True,
    )
    loss = (
        out["total_energy"].sum()
        + out["forces"].square().mean()
        + out["spin_forces"].square().mean()
        + out["spin_forces_vert"].square().mean()
        + out["born_effective_charges"].square().mean()
        + out["f_uncertainty"].sum()
        + out["sf_uncertainty"].sum()
    )
    loss.backward()
    _assert_all_finite(out, model)


def test_large_supercell_n2048() -> None:
    """Stress test N=2048 supercell (~60k edges on CUDA, ~12k on CPU) with edge AMP & ckpt."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_edges = 60000 if device.type == "cuda" else 12000
    torch.manual_seed(2048)
    data = _make_graph(
        n_atoms=2048,
        n_edges=n_edges,
        num_types=2,
        seed=2048,
        device=device,
    )
    model = build_v2_soc(
        num_types=2,
        lmax=2,
        c0=64,
        c1=16,
        c2=8,
        radial_hidden=64,
        fitting_hidden=64,
        c_spin=8,
        c_spin_01=6,
        c_spin_02=3,
        c_spin_11=4,
        c_spin_21=3,
        c_spin_22=2,
        c_soc_dmi=4,
        c_soc_kit=4,
        predict_bec=True,
        predict_e2slp=True,
        e2slp_include_sf=True,
        edge_amp=(device.type == "cuda"),
        edge_checkpoint=True,
    ).to(device)
    model.set_spin_gate(0.5)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    out = model(
        pos=data["pos"].clone().requires_grad_(True),
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        spin=data["spin"].clone().requires_grad_(True),
        batch=data["batch"],
        num_graphs=1,
        compute_forces=True,
        compute_virial=False,
        compute_spin_forces=True,
        create_graph=True,
    )
    loss = (
        out["total_energy"].sum()
        + 0.1 * out["forces"].square().mean()
        + 0.1 * out["spin_forces"].square().mean()
        + 0.1 * out["born_effective_charges"].square().mean()
        + 0.05 * out["f_uncertainty"].sum()
        + 0.05 * out["sf_uncertainty"].sum()
    )
    loss.backward()
    _assert_all_finite(out, model)
    assert out["forces"].shape == (2048, 3)
    assert out["born_effective_charges"].shape == (2048, 3, 3)
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        # Must fit comfortably within consumer 24GB GPU even for 2048 atoms / 60k edges
        assert peak_gb < 12.0, f"Unexpectedly high peak VRAM {peak_gb:.2f} GiB"


# ===========================================================================
# 3. Varying-Shape Batch Sequence under compile_fullgraph & compile_bucketed
# ===========================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for compile stress"
)
def test_varying_shapes_compile_dynamic_and_bucketed_gpu() -> None:
    """Stress [32/400 -> 512/15k -> 64/0 -> 1024/30k -> 2048/60k] on CUDA."""
    device = torch.device("cuda")
    shapes = [
        (32, 400),
        (512, 15000),
        (64, 0),
        (1024, 30000),
        (2048, 60000),
    ]
    batches = [
        _make_graph(n_a, n_e, num_types=2, seed=500 + i, device=device)
        for i, (n_a, n_e) in enumerate(shapes)
    ]

    torch.manual_seed(909)
    base = build_v2_soc(
        num_types=2,
        lmax=2,
        c0=32,
        c1=8,
        c2=4,
        radial_hidden=32,
        fitting_hidden=32,
        c_spin=6,
        c_spin_01=4,
        c_spin_02=2,
        c_spin_11=3,
        c_spin_21=2,
        c_spin_22=2,
        c_soc_dmi=2,
        c_soc_kit=2,
        edge_amp=True,
        edge_checkpoint=True,
    ).to(device)
    base.set_spin_gate(0.5)

    # Eager reference outputs on all 5 shapes
    eager_refs: list[dict[str, torch.Tensor]] = []
    for d in batches:
        base.zero_grad(set_to_none=True)
        out_e = base(
            pos=d["pos"].clone().requires_grad_(True),
            atom_types=d["atom_types"],
            edge_index=d["edge_index"],
            spin=d["spin"].clone().requires_grad_(True),
            batch=d["batch"],
            num_graphs=1,
            compute_forces=True,
            compute_virial=False,
            compute_spin_forces=True,
            create_graph=False,
        )
        eager_refs.append({k: v.detach().clone() for k, v in out_e.items()})

    # ---- Part A: compile_fullgraph(dynamic=True) ----
    torch._dynamo.reset()
    m_dyn = copy.deepcopy(base).compile_fullgraph(dynamic=True)
    for (n_a, n_e), d, ref in zip(shapes, batches, eager_refs):
        m_dyn.zero_grad(set_to_none=True)
        out_d = m_dyn(
            pos=d["pos"].clone().requires_grad_(True),
            atom_types=d["atom_types"],
            edge_index=d["edge_index"],
            spin=d["spin"].clone().requires_grad_(True),
            batch=d["batch"],
            num_graphs=1,
            compute_forces=True,
            compute_virial=False,
            compute_spin_forces=True,
            create_graph=True,
        )
        loss = (
            out_d["total_energy"].sum()
            + 0.1 * out_d["forces"].square().mean()
            + 0.1 * out_d["spin_forces"].square().mean()
        )
        loss.backward()
        _assert_all_finite(out_d, m_dyn)
        assert out_d["forces"].shape == (n_a, 3)
        e_err_per_atom = torch.max(
            torch.abs(out_d["total_energy"] - ref["total_energy"])
        ).item() / max(n_a, 1)
        f_scale = torch.max(torch.abs(ref["forces"])).item() + 1.0
        f_diff = torch.abs(out_d["forces"] - ref["forces"])
        f_rel_max = torch.max(f_diff).item() / f_scale
        f_rel_mean = torch.mean(f_diff).item() / f_scale
        assert (
            e_err_per_atom < 2e-3
        ), f"dynamic parity energy/atom err {e_err_per_atom} on ({n_a}, {n_e})"
        # bfloat16 edge AMP has 1 ULP = 2^-7 = 0.0078125; allow ~3 ULP max, < 0.5% mean
        assert (
            f_rel_max < 2.5e-2 and f_rel_mean < 5e-3
        ), f"dynamic parity rel force err max={f_rel_max} mean={f_rel_mean} on ({n_a}, {n_e})"

    # ---- Part B: compile_bucketed(..., mode="reduce-overhead") + overflow fallback ----
    torch._dynamo.reset()
    m_bkt = copy.deepcopy(base).compile_bucketed(
        atom_buckets=(64, 128, 256, 520),
        edge_buckets=(2048, 4096, 8192, 16384),
        mode="reduce-overhead",
    )
    torch.cuda.reset_peak_memory_stats(device)
    for idx, ((n_a, n_e), d, ref) in enumerate(zip(shapes, batches, eager_refs)):
        m_bkt.zero_grad(set_to_none=True)
        out_b = m_bkt(
            pos=d["pos"].clone().requires_grad_(True),
            atom_types=d["atom_types"],
            edge_index=d["edge_index"],
            spin=d["spin"].clone().requires_grad_(True),
            batch=d["batch"],
            num_graphs=1,
            compute_forces=True,
            compute_virial=False,
            compute_spin_forces=True,
            create_graph=True,
        )
        loss = (
            out_b["total_energy"].sum()
            + 0.1 * out_b["forces"].square().mean()
            + 0.1 * out_b["spin_forces"].square().mean()
        )
        loss.backward()
        _assert_all_finite(out_b, m_bkt)
        assert out_b["forces"].shape == (n_a, 3)
        e_err_per_atom = torch.max(
            torch.abs(out_b["total_energy"] - ref["total_energy"])
        ).item() / max(n_a, 1)
        f_scale = torch.max(torch.abs(ref["forces"])).item() + 1.0
        f_diff = torch.abs(out_b["forces"] - ref["forces"])
        f_rel_max = torch.max(f_diff).item() / f_scale
        f_rel_mean = torch.mean(f_diff).item() / f_scale
        assert (
            e_err_per_atom < 2e-3
        ), f"bucketed parity energy/atom err {e_err_per_atom} on ({n_a}, {n_e})"
        assert (
            f_rel_max < 2.5e-2 and f_rel_mean < 5e-3
        ), f"bucketed parity rel force err max={f_rel_max} mean={f_rel_mean} on ({n_a}, {n_e})"
        if idx == 2:
            # First 3 shapes fit inside buckets: (32/400)->(64,2048),
            # (512/15k)->(520,16384), (64/0)->(128,4096) => 0 overflows!
            assert m_bkt._bucket_overflow_count == 0
    # Out-of-bucket shapes (1024/30k) and (2048/60k) gracefully hit _overflow_forward
    assert m_bkt._bucket_overflow_count == 2
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    assert peak_gb < 12.0, f"Unexpected bucketed+overflow VRAM {peak_gb:.2f} GiB"


def test_bucketed_overflow_fallback_cpu() -> None:
    """CPU check of compile_bucketed overflow routing & exact eager parity."""
    torch.manual_seed(111)
    model = build_v2_nosoc(num_types=2, lmax=2, c0=16, c1=4, c2=4, c_spin=4)
    model.set_spin_gate(0.5)
    d_small = _make_graph(8, 24, num_types=2, seed=1)
    d_big = _make_graph(40, 120, num_types=2, seed=2)

    ref_big = model(
        pos=d_big["pos"].clone().requires_grad_(True),
        atom_types=d_big["atom_types"],
        edge_index=d_big["edge_index"],
        spin=d_big["spin"].clone().requires_grad_(True),
        batch=d_big["batch"],
        num_graphs=1,
        create_graph=False,
    )
    # Emulate bucketed config with tiny bucket ceiling so d_big overflows
    model._compiled_forward = model._forward_autograd
    model._overflow_forward = model._forward_autograd
    model._bucketed_config = {
        "atom_buckets": (16, 24),
        "edge_buckets": (64, 96),
        "mode": "default",
    }
    model._bucket_overflow_count = 0

    out_small = model(
        pos=d_small["pos"].clone().requires_grad_(True),
        atom_types=d_small["atom_types"],
        edge_index=d_small["edge_index"],
        spin=d_small["spin"].clone().requires_grad_(True),
        batch=d_small["batch"],
        num_graphs=1,
        create_graph=False,
    )
    assert model._bucket_overflow_count == 0
    assert out_small["forces"].shape == (8, 3)

    out_big = model(
        pos=d_big["pos"].clone().requires_grad_(True),
        atom_types=d_big["atom_types"],
        edge_index=d_big["edge_index"],
        spin=d_big["spin"].clone().requires_grad_(True),
        batch=d_big["batch"],
        num_graphs=1,
        create_graph=False,
    )
    assert model._bucket_overflow_count == 1
    assert torch.allclose(out_big["total_energy"], ref_big["total_energy"], atol=1e-6)
    assert torch.allclose(out_big["forces"], ref_big["forces"], atol=1e-6)


# ===========================================================================
# 4. PhysMuon & Muon Adversarial Stress (CPU & CUDA cuda_graph=True)
# ===========================================================================


@pytest.mark.parametrize(
    "device_str,use_cg",
    [
        ("cpu", False),
        pytest.param(
            "cuda",
            True,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_physmuon_adversarial_edge_cases(device_str: str, use_cg: bool) -> None:
    """lr=0, wd=0.1, zero grads, 1000x spike, NaN/Inf skip, frozen subset, weird shapes."""
    dev = torch.device(device_str)
    torch.manual_seed(404)

    # ---- A. lr = 0.0 (even with wd = 0.1) must leave parameters bit-identical ----
    w0 = nn.Parameter(torch.randn(8, 8, device=dev))
    b0 = nn.Parameter(torch.randn(8, device=dev))
    w0_init = w0.detach().clone()
    b0_init = b0.detach().clone()
    opt_zero_lr = PhysMuon(
        [w0, b0],
        lr=0.0,
        weight_decay=0.1,
        bm_start=1,
        bm_shadow=True,
        osc_gate=True,
        cuda_graph=use_cg,
    )
    for _ in range(4):
        w0.grad = torch.randn_like(w0)
        b0.grad = torch.randn_like(b0)
        opt_zero_lr.step()
    assert torch.equal(w0, w0_init)
    assert torch.equal(b0, b0_init)

    # ---- B. wd = 0.1 with all-zero gradients shrinks weights cleanly ----
    w_wd = nn.Parameter(torch.ones(8, 8, device=dev) * 2.0)
    b_wd = nn.Parameter(torch.ones(8, device=dev) * 2.0)
    opt_wd = Muon([w_wd, b_wd], lr=0.1, weight_decay=0.1, cuda_graph=use_cg)
    for _ in range(3):
        w_wd.grad = torch.zeros_like(w_wd)
        b_wd.grad = torch.zeros_like(b_wd)
        opt_wd.step()
    assert torch.isfinite(w_wd).all() and torch.isfinite(b_wd).all()
    assert float(w_wd.norm().item()) < float((torch.ones(8, 8) * 2.0).norm().item())

    # ---- C. 1000x spike_guard freeze + NaN/Inf safe skip + frozen param subset ----
    w_train = nn.Parameter(torch.randn(8, 8, device=dev))
    w_frozen = nn.Parameter(torch.randn(8, 8, device=dev), requires_grad=False)
    w_frozen_init = w_frozen.detach().clone()
    opt_sg = PhysMuon(
        [{"params": [w_train, w_frozen], "use_muon": True}],
        lr=1e-2,
        weight_decay=1e-2,
        bm_start=2,
        bm_beta_fast=0.85,
        bm_beta_slow=0.98,
        bm_shadow=True,
        osc_gate=True,
        iso=True,
        iso_min_dim=4,
        spike_guard=10.0,
        spike_window=12,
        cuda_graph=use_cg,
    )
    # 12 normal 2-microbatch steps to fill median window
    for s in range(12):
        g1 = torch.randn_like(w_train) * 0.1
        g2 = torch.randn_like(w_train) * 0.1
        w_train.grad = g1.clone()
        opt_sg.observe_microbatch()
        w_train.grad = g1 + g2
        opt_sg.observe_microbatch()
        opt_sg.step()

    assert opt_sg.spike_frozen_steps == 0
    assert opt_sg.nonfinite_skipped_steps == 0
    mslow_before_spike = opt_sg.state[w_train]["m_slow"].clone()
    rh_before_spike = opt_sg.state[w_train]["row_heat"].clone()
    ch_before_spike = opt_sg.state[w_train]["col_heat"].clone()

    # Inject 1000x finite spike
    g_spike1 = torch.randn_like(w_train) * 100.0
    g_spike2 = torch.randn_like(w_train) * 100.0
    w_train.grad = g_spike1.clone()
    opt_sg.observe_microbatch()
    w_train.grad = g_spike1 + g_spike2
    opt_sg.observe_microbatch()
    opt_sg.step()

    assert opt_sg.spike_frozen_steps == 1
    assert opt_sg.nonfinite_skipped_steps == 0
    assert torch.isfinite(w_train).all()
    # m_slow and iso heats must stay frozen on a finite spike
    assert torch.equal(opt_sg.state[w_train]["m_slow"], mslow_before_spike)
    assert torch.equal(opt_sg.state[w_train]["row_heat"], rh_before_spike)
    assert torch.equal(opt_sg.state[w_train]["col_heat"], ch_before_spike)

    # Inject NaN gradient -> must skip cleanly without corrupting weights or states!
    w_before_nan = w_train.detach().clone()
    m_before_nan = opt_sg.state[w_train]["momentum"].clone()
    g_nan = torch.randn_like(w_train)
    g_nan[0, 0] = float("nan")
    w_train.grad = g_nan.clone()
    opt_sg.observe_microbatch()
    opt_sg.step()

    assert opt_sg.nonfinite_skipped_steps == 1
    assert torch.equal(w_train, w_before_nan)
    assert torch.equal(opt_sg.state[w_train]["momentum"], m_before_nan)
    assert opt_sg.state[w_train]["iso_count"] == 0

    # Inject +Inf gradient -> must also skip cleanly!
    g_inf = torch.randn_like(w_train)
    g_inf[1, 2] = float("inf")
    w_train.grad = g_inf.clone()
    opt_sg.observe_microbatch()
    opt_sg.step()

    assert opt_sg.nonfinite_skipped_steps == 2
    assert torch.equal(w_train, w_before_nan)
    assert torch.equal(w_frozen, w_frozen_init)
    assert w_frozen not in opt_sg.state

    # Healthy step immediately after NaN/Inf must recover cleanly
    g_ok1 = torch.randn_like(w_train) * 0.1
    g_ok2 = torch.randn_like(w_train) * 0.1
    w_train.grad = g_ok1.clone()
    opt_sg.observe_microbatch()
    w_train.grad = g_ok1 + g_ok2
    opt_sg.observe_microbatch()
    opt_sg.step()
    assert torch.isfinite(w_train).all()
    assert not torch.equal(w_train, w_before_nan)

    # ---- D. Weird parameter shapes under auto-routing ("2d" & "slice") and forced Muon ----
    shapes_map = {
        "row_1x16": (1, 16),
        "col_16x1": (16, 1),
        "sq_2x2": (2, 2),
        "sq_3x3": (3, 3),
        "sq_4x4": (4, 4),
        "t3d_2x4x8": (2, 4, 8),
        "t4d_2x2x4x4": (2, 2, 4, 4),
    }
    for mode in ("2d", "slice"):
        named = [
            (k, nn.Parameter(torch.randn(sh, device=dev)))
            for k, sh in shapes_map.items()
        ]
        opt_sh = PhysMuon(
            [p for _, p in named],
            lr=1e-2,
            weight_decay=1e-2,
            bm_start=1,
            bm_shadow=True,
            osc_gate=True,
            iso=True,
            iso_min_dim=2,
            muon_mode=mode,
            named_parameters=named,
            cuda_graph=use_cg,
        )
        summary = opt_sh.routing_summary()
        # 1x16 and 16x1 have effective shape (16,) -> must ALWAYS route to adam, never muon
        assert "row_1x16" in summary["adam"]
        assert "col_16x1" in summary["adam"]
        assert "sq_2x2" in summary["muon"]
        assert "sq_3x3" in summary["muon"]
        assert "sq_4x4" in summary["muon"]
        if mode == "2d":
            assert "t3d_2x4x8" in summary["adamw"]
            assert "t4d_2x2x4x4" in summary["adamw"]
        else:
            assert "t3d_2x4x8" in summary["muon"]
            assert "t4d_2x2x4x4" in summary["muon"]

        for _ in range(4):
            for _, p in named:
                p.grad = torch.randn_like(p) * 0.1
            opt_sh.observe_microbatch()
            for _, p in named:
                p.grad = p.grad + torch.randn_like(p) * 0.1
            opt_sh.observe_microbatch()
            opt_sh.step()
            for k, p in named:
                assert torch.isfinite(p).all(), f"Non-finite {k} in mode={mode}"


@pytest.mark.parametrize(
    "device_str,use_cg",
    [
        ("cpu", False),
        pytest.param(
            "cuda",
            True,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
@pytest.mark.parametrize("bm_shadow", [False, True])
def test_physmuon_changing_grad_none_subsets_across_bm_start(
    device_str: str,
    use_cg: bool,
    bm_shadow: bool,
) -> None:
    """Verify mixed (t < bm_start, t >= bm_start) in same param group + CUDA Graph parity."""
    dev = torch.device(device_str)
    torch.manual_seed(707)

    p1_ref = nn.Parameter(torch.randn(8, 8, device=dev))
    p2_ref = nn.Parameter(torch.randn(8, 8, device=dev))
    p3_ref = nn.Parameter(torch.randn(8, device=dev))

    p1_tst = nn.Parameter(p1_ref.detach().clone())
    p2_tst = nn.Parameter(p2_ref.detach().clone())
    p3_tst = nn.Parameter(p3_ref.detach().clone())

    cfg = dict(
        lr=1e-2,
        weight_decay=1e-2,
        bm_start=3,
        bm_beta_fast=0.85,
        bm_beta_slow=0.98,
        osc_gate=True,
        osc_wmax=0.5,
        bm_shadow=bm_shadow,
        fresh_cap=0.25,
        iso=False,
    )
    opt_ref = PhysMuon([p1_ref, p2_ref, p3_ref], cuda_graph=False, **cfg)
    opt_tst = PhysMuon([p1_tst, p2_tst, p3_tst], cuda_graph=use_cg, **cfg)

    # Pattern causes p1 to reach bm_start=3 at step 3 while p2 is still at step 1!
    active_masks = [
        (True, False, True),
        (True, False, True),
        (True, True, False),
        (True, True, True),  # p1 crosses bm_start=3, p2 is at t=1 (mixed group!)
        (True, True, True),  # p1 post-bm, p2 at t=2 (mixed group!)
        (True, True, True),  # p1 post-bm, p2 crosses bm_start=3 (mixed seed_now!)
        (True, True, True),
        (True, False, True),
        (True, True, True),
        (True, True, True),
    ]
    for step_idx, (a1, a2, a3) in enumerate(active_masks):
        gen = torch.Generator(device="cpu").manual_seed(9000 + step_idx)
        g1 = torch.randn((8, 8), generator=gen).to(dev) * 0.1
        g2 = torch.randn((8, 8), generator=gen).to(dev) * 0.1
        g3 = torch.randn((8,), generator=gen).to(dev) * 0.1
        for (p1, p2, p3, opt) in (
            (p1_ref, p2_ref, p3_ref, opt_ref),
            (p1_tst, p2_tst, p3_tst, opt_tst),
        ):
            p1.grad = g1.clone() if a1 else None
            p2.grad = g2.clone() if a2 else None
            p3.grad = g3.clone() if a3 else None
            opt.step()

    assert torch.max(torch.abs(p1_ref - p1_tst)).item() < 1e-6
    assert torch.max(torch.abs(p2_ref - p2_tst)).item() < 1e-6
    assert torch.max(torch.abs(p3_ref - p3_tst)).item() < 1e-6
    assert len(opt_tst._cuda_graphs) <= 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_physmuon_checkpoint_resume_mid_bm_start_cuda_graph() -> None:
    """Save/resume PhysMuon state_dict mid-bm_start on CUDA with cuda_graph=True."""
    dev = torch.device("cuda")
    torch.manual_seed(808)
    w_cont = nn.Parameter(torch.randn(16, 16, device=dev))
    b_cont = nn.Parameter(torch.randn(16, device=dev))
    w_res = nn.Parameter(w_cont.detach().clone())
    b_res = nn.Parameter(b_cont.detach().clone())

    cfg = dict(
        lr=1e-2,
        weight_decay=1e-2,
        bm_start=4,
        bm_beta_fast=0.85,
        bm_beta_slow=0.98,
        osc_gate=True,
        osc_wmax=0.4,
        bm_shadow=True,
        fresh_cap=0.2,
        iso=True,
        iso_min_dim=4,
        cuda_graph=True,
    )
    opt_cont = PhysMuon([w_cont, b_cont], **cfg)
    opt_res = PhysMuon([w_res, b_res], **cfg)

    for step in range(10):
        gen = torch.Generator(device="cpu").manual_seed(1200 + step)
        gw1 = torch.randn((16, 16), generator=gen).to(dev) * 0.1
        gw2 = torch.randn((16, 16), generator=gen).to(dev) * 0.1
        gb1 = torch.randn((16,), generator=gen).to(dev) * 0.1
        gb2 = torch.randn((16,), generator=gen).to(dev) * 0.1
        for w, b, opt in ((w_cont, b_cont, opt_cont), (w_res, b_res, opt_res)):
            w.grad = gw1.clone()
            b.grad = gb1.clone()
            opt.observe_microbatch()
            w.grad = gw1 + gw2
            b.grad = gb1 + gb2
            opt.observe_microbatch()
            opt.step()
        if step == 3:
            # Checkpoint right at bm_start boundary, restore onto fresh optimizer!
            sd = copy.deepcopy(opt_res.state_dict())
            opt_res = PhysMuon([w_res, b_res], **cfg)
            opt_res.load_state_dict(sd)

    assert torch.max(torch.abs(w_cont - w_res)).item() < 1e-6
    assert torch.max(torch.abs(b_cont - b_res)).item() < 1e-6


# ===========================================================================
# 5. 50-Step Realistic Noisy GPU Training Stability (E + F + SF + BEC + AMP)
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_50step_realistic_noisy_training_stability_gpu() -> None:
    """50-step noisy E+F+SF+BEC training with bf16 edge AMP, ckpt, PhysMuon, WSD."""
    dev = torch.device("cuda")
    torch.manual_seed(5050)

    model = build_v2_born(
        num_types=2,
        lmax=2,
        c0=48,
        c1=12,
        c2=6,
        radial_hidden=48,
        fitting_hidden=48,
        has_spin=True,
        has_soc=True,
        use_spin=(True, False),
        c_spin=6,
        c_spin_01=4,
        c_spin_02=2,
        c_spin_11=3,
        c_spin_21=2,
        c_spin_22=2,
        c_soc_dmi=3,
        c_soc_kit=3,
        asr="soft",
        bec_scale=(3.0, -1.5),
        edge_amp=True,
        edge_checkpoint=True,
    ).to(dev)
    model.set_spin_gate(0.3)

    # Create 2 fixed micro-batch graphs + synthetic physical targets from a teacher
    batches = [
        _make_graph(n_atoms=32, n_edges=384, num_types=2, seed=600 + mb, device=dev)
        for mb in range(2)
    ]
    with torch.no_grad():
        targets: list[dict[str, torch.Tensor]] = []
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.05)
    for d in batches:
        t_out = teacher(
            pos=d["pos"].clone().requires_grad_(True),
            atom_types=d["atom_types"],
            edge_index=d["edge_index"],
            spin=d["spin"].clone().requires_grad_(True),
            batch=d["batch"],
            num_graphs=1,
            compute_forces=True,
            compute_virial=False,
            compute_spin_forces=True,
            create_graph=False,
        )
        targets.append({k: v.detach().clone() for k, v in t_out.items()})

    preset = interatomic_potential_gated(total_steps=50)
    opt = PhysMuon(
        model.parameters(),
        lr=1.5e-2,
        weight_decay=1e-4,
        named_parameters=model.named_parameters(),
        spike_guard=10.0,
        spike_window=12,
        cuda_graph=True,
        **preset,
    )
    sched = WSDCosineScheduler(
        opt,
        total_steps=50,
        warmup_steps=5,
        decay_steps=15,
        min_lr_ratio=0.05,
    )

    losses: list[float] = []
    mem_step5 = 0
    mem_step50 = 0
    torch.cuda.synchronize(dev)

    for step in range(50):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for mb, (d, tgt) in enumerate(zip(batches, targets)):
            out = model(
                pos=d["pos"].clone().requires_grad_(True),
                atom_types=d["atom_types"],
                edge_index=d["edge_index"],
                spin=d["spin"].clone().requires_grad_(True),
                batch=d["batch"],
                num_graphs=1,
                compute_forces=True,
                compute_virial=False,
                compute_spin_forces=True,
                create_graph=True,
            )
            # Add small realistic label noise (per-atom energy MSE + F + SF + BEC)
            n_e = torch.randn_like(tgt["total_energy"]) * 1e-3
            n_f = torch.randn_like(tgt["forces"]) * 1e-3
            n_sf = torch.randn_like(tgt["spin_forces"]) * 1e-3
            n_z = torch.randn_like(tgt["born_effective_charges"]) * 1e-3
            loss = (
                ((out["total_energy"] - (tgt["total_energy"] + n_e)) / 32.0)
                .square()
                .mean()
                + (out["forces"] - (tgt["forces"] + n_f)).square().mean()
                + (out["spin_forces"] - (tgt["spin_forces"] + n_sf)).square().mean()
                + (
                    out["born_effective_charges"]
                    - (tgt["born_effective_charges"] + n_z)
                )
                .square()
                .mean()
            )
            (loss / 2.0).backward()
            opt.observe_microbatch()
            step_loss += float(loss.detach().item()) / 2.0

        # Inject a 500x finite spike at step 25 and a NaN grad at step 35
        if step == 25:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(500.0)
        elif step == 35:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.view(-1)[0] = float("nan")
                    break

        opt.step()
        sched.step()

        assert math.isfinite(step_loss), f"Non-finite loss at step {step}"
        losses.append(step_loss)
        if step == 4:
            torch.cuda.synchronize(dev)
            mem_step5 = torch.cuda.memory_allocated(dev)
        elif step == 49:
            torch.cuda.synchronize(dev)
            mem_step50 = torch.cuda.memory_allocated(dev)

    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"Non-finite param {name} after 50 steps"

    # Must have caught both the injected NaN and the injected spike
    assert opt.nonfinite_skipped_steps == 1
    assert opt.spike_frozen_steps >= 2

    # Monotonic / strong convergence across the 50 noisy steps
    init_avg = sum(losses[:5]) / 5.0
    final_avg = sum(losses[-5:]) / 5.0
    assert (
        final_avg < 0.5 * init_avg
    ), f"Expected strong loss drop: init_avg={init_avg:.4f}, final_avg={final_avg:.4f}"

    # Flat VRAM (no tensor / graph leak over 50 steps)
    assert (
        mem_step50 <= int(mem_step5 * 1.15) + 4 * 1024 * 1024
    ), f"Memory grew from {mem_step5 / 1e6:.2f} MB to {mem_step50 / 1e6:.2f} MB"


# ===========================================================================
# 6. Real v1 YAML Allegro_nosoc / Allegro_soc / Allegro_born / e2SLP Smoke
# ===========================================================================


def _build_compact_v1_model(family: str) -> nn.Module:
    from nequip.model import model_from_config

    common: dict[str, Any] = {
        "num_types": 2,
        "type_names": ["Fe", "O"],
        "r_max": 5.5,
        "avg_num_neighbors": 8.0,
        "l_max": 2,
        "parity": "o3_full",
        "num_layers": 1,
        "env_embed_multiplicity": 4,
        "two_body_latent_mlp_latent_dimensions": [16, 16],
        "latent_mlp_latent_dimensions": [16],
        "env_embed_mlp_latent_dimensions": [],
        "edge_eng_mlp_latent_dimensions": [16],
        "BesselBasis_trainable": False,
        "PolynomialCutoff_p": 6,
        "default_dtype": "float32",
    }
    if family == "nosoc":
        cfg = dict(
            common,
            model_builders=[
                "allegro.model.Allegro_nosoc",
                "ParaStressForceSpinOutput",
            ],
            allegro_onlyr_num_layers=1,
            allegro_nosoc_num_layers=1,
            lhb=True,
            lbiquadratic=True,
            lnosoc_layer=True,
            lte3nn=True,
            edge_onlyr_eng_mlp_latent_dimensions=[16],
            edge_hb_coeff_mlp_latent_dimensions=[16],
            edge_biquad_coeff_mlp_latent_dimensions=[16],
            edge_eng_nosoc_mlp_latent_dimensions=[16],
        )
    elif family == "soc":
        cfg = dict(
            common,
            model_builders=[
                "allegro.model.Allegro_soc",
                "ParaStressForceSpinOutput",
            ],
            symmetry="u2",
            allegro_onlyr_num_layers=1,
            allegro_soc_on_num_layers=1,
            spharm_soc_on_l_max=1,
            ljmatrix=True,
            lani=True,
            lbiquadratic=True,
            lsoc_layer=True,
            edge_onlyr_eng_mlp_latent_dimensions=[16],
            edge_biquad_coeff_mlp_latent_dimensions=[16],
            edge_eng_soc_on_mlp_latent_dimensions=[16],
        )
    elif family == "born":
        cfg = dict(
            common,
            model_builders=[
                "allegro.model.Allegro_born",
                "ParaStressForceOutput",
            ],
            born_scale_list=[4.0, -2.0],
            born_scale_trainable=False,
        )
    elif family == "e2slp":
        cfg = dict(
            common,
            model_builders=[
                "allegro.model.Allegro_soc_e2IP",
                "ParaStressForceSpinOutput",
            ],
            symmetry="u2",
            allegro_onlyr_num_layers=1,
            allegro_soc_on_num_layers=1,
            spharm_soc_on_l_max=1,
            ljmatrix=True,
            lani=True,
            lbiquadratic=True,
            lsoc_layer=True,
            edge_onlyr_eng_mlp_latent_dimensions=[16],
            edge_biquad_coeff_mlp_latent_dimensions=[16],
            edge_eng_soc_on_mlp_latent_dimensions=[16],
            e2ip_latent_channels=4,
            e2ip_mlp_latent_dimensions=[16],
            evidential_key="spin_forces_vert",
            spingnn_compile={"enabled": False},
        )
    else:
        raise ValueError(family)
    return model_from_config(cfg)


@pytest.mark.parametrize("family", ["nosoc", "soc", "born", "e2slp"])
@pytest.mark.parametrize("opt_name", ["PhysMuon", "Muon"])
def test_v1_yaml_allegro_families_with_physmuon_and_muon(
    family: str,
    opt_name: str,
    tmp_path: Path,
) -> None:
    """Real v1 Allegro_nosoc/soc/born/e2SLP Trainer smoke with PhysMuon & Muon."""
    from nequip.data import AtomicData, AtomicDataDict
    from nequip.train.trainer import Trainer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)
    model = _build_compact_v1_model(family)

    loss_coeffs: Any
    if family == "born":
        loss_coeffs = {
            "total_energy": 1.0,
            "forces": 1.0,
            "born_effective_charges": 1.0,
        }
    else:
        loss_coeffs = {"total_energy": 1.0, "forces": 1.0}

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        root=str(tmp_path / f"run_{family}_{opt_name}"),
        run_name=f"smoke_{family}_{opt_name}",
        device=dev,
        loss_coeffs=loss_coeffs,
        optimizer_name=opt_name,
        optimizer_kwargs={"lr": 5e-3, "cuda_graph": (dev == "cuda")},
        max_epochs=1,
    )
    if opt_name == "PhysMuon":
        trainer_kwargs["optimizer_preset"] = "interatomic_potential_gated"
        trainer_kwargs["optimizer_total_steps"] = 10
        trainer_kwargs["grad_accumulation_steps"] = 2
    else:
        trainer_kwargs["grad_accumulation_steps"] = 1

    trainer = Trainer(**trainer_kwargs)
    if opt_name == "PhysMuon":
        assert isinstance(trainer.optim, PhysMuon)
    else:
        assert isinstance(trainer.optim, Muon)
    trainer.init_metrics()
    trainer.loss_stat.to(trainer.torch_device)
    trainer.metrics.to(trainer.torch_device)

    # Verify equivariance-safe routing: no 1-row/1-col or 1D tensor on Muon
    for grp in trainer.optim.param_groups:
        for p in grp["params"]:
            route, _ = trainer.optim._resolve_route(p, grp)
            if route == "muon":
                eff = tuple(d for d in p.shape if d != 1)
                assert len(eff) == 2 and min(eff) >= 2

    # Build a real AtomicData batch matching v1 Allegro input keys
    g = _make_graph(n_atoms=6, n_edges=24, num_types=2, seed=42, device="cpu")
    data_dict: dict[str, Any] = {
        AtomicDataDict.POSITIONS_KEY: g["pos"],
        AtomicDataDict.ATOM_TYPE_KEY: g["atom_types"],
        AtomicDataDict.EDGE_INDEX_KEY: g["edge_index"],
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: g["edge_cell_shift"],
        AtomicDataDict.CELL_KEY: g["cell"],
        AtomicDataDict.PBC_KEY: torch.tensor([[True, True, True]], dtype=torch.bool),
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.tensor([[0.5]], dtype=torch.float32),
        AtomicDataDict.FORCE_KEY: torch.zeros((6, 3), dtype=torch.float32),
    }
    if family == "born":
        data_dict["born_effective_charges"] = torch.zeros((6, 9), dtype=torch.float32)
    else:
        data_dict["spin"] = g["spin"]
        data_dict["spin_length"] = g["spin"].norm(dim=-1, keepdim=True)
        data_dict["spin_forces"] = torch.zeros((6, 3), dtype=torch.float32)

    batch = AtomicData.from_dict(data_dict)
    batch.batch = g["batch"]
    batch.ptr = torch.tensor([0, 6], dtype=torch.long)

    # Snapshot initial parameters
    before = [p.detach().clone() for p in trainer.model.parameters() if p.requires_grad]

    # Run 2 steps (or 4 micro-batches for PhysMuon grad_accumulation_steps=2)
    n_calls = 4 if opt_name == "PhysMuon" else 2
    trainer.n_batches = n_calls
    for idx in range(n_calls):
        trainer.ibatch = idx
        trainer.batch_step(batch.clone(), validation=False)

    after = [p.detach().clone() for p in trainer.model.parameters() if p.requires_grad]
    for p in after:
        assert torch.isfinite(p).all(), f"Non-finite v1 param in {family}/{opt_name}"
    moved = sum(1 for a, b in zip(after, before) if not torch.equal(a, b))
    assert moved > 0, f"No parameters updated in {family}/{opt_name}"
