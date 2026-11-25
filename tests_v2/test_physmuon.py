"""Comprehensive tests for PhysMuon, Muon, HybridMuon, CUDA Graph parity,
equivariance preservation, FP64 buffer re-materialization, bucketed padding,
and v1 YAML trainer integration.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from spingnn2.symmetry import random_rotation_matrix
from spingnn2.v2 import (
    HybridMuon,
    Muon,
    PhysMuon,
    WSDCosineScheduler,
    age,
    beta_for_age,
    build_v2_born,
    build_v2_e2slp,
    build_v2_nosoc,
    build_v2_soc,
    fast_weight_for_age,
    interatomic_potential,
    interatomic_potential_gated,
    kernel_age,
    language_model,
    switch_step,
)


def _make_small_cluster(
    n_atoms: int = 6,
    r_cut: float = 5.5,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    seed: int = 123,
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = (torch.randn(n_atoms, 3, generator=g, dtype=dtype) * 1.1).to(device)
    atom_types = (torch.arange(n_atoms, device=device) % 2).to(torch.long)
    spin = torch.randn(n_atoms, 3, generator=g, dtype=dtype).to(device)
    spin = spin / torch.linalg.vector_norm(spin, dim=-1, keepdim=True) * 2.0
    dst_list, src_list = [], []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j and float(torch.linalg.vector_norm(pos[j] - pos[i])) < r_cut:
                dst_list.append(i)
                src_list.append(j)
    edge_index = torch.tensor([dst_list, src_list], dtype=torch.long, device=device)
    batch = torch.zeros(n_atoms, dtype=torch.long, device=device)
    return {
        "pos": pos,
        "atom_types": atom_types,
        "edge_index": edge_index,
        "spin": spin,
        "batch": batch,
    }


def _build_family(family: str) -> torch.nn.Module:
    if family == "nosoc":
        m = build_v2_nosoc(num_types=2, r_cut=5.5, use_spin=(True, False))
        m.set_spin_gate(0.5)
        return m
    if family == "soc":
        m = build_v2_soc(num_types=2, r_cut=5.5, use_spin=(True, True))
        m.set_spin_gate(0.5)
        return m
    if family == "born":
        return build_v2_born(
            num_types=2,
            r_cut=5.5,
            asr="soft",
            has_spin=True,
            has_soc=True,
            use_spin=(True, True),
        )
    if family == "e2slp":
        m = build_v2_e2slp(
            num_types=2,
            r_cut=5.5,
            has_spin=True,
            has_soc=True,
            use_spin=(True, True),
            include_f=True,
            include_sf=True,
            predict_gamma=True,
        )
        m.set_spin_gate(0.5)
        return m
    raise ValueError(family)


def test_physmuon_schedule_helpers_and_presets() -> None:
    assert abs(age(0.95) - 19.0) < 1e-12
    assert abs(beta_for_age(19.0) - 0.95) < 1e-12
    assert abs(kernel_age(0.85, 0.98, 0.5) - 0.5 * (age(0.85) + age(0.98))) < 1e-12
    w = fast_weight_for_age(0.85, 0.98, 30.0)
    assert abs(kernel_age(0.85, 0.98, w) - 30.0) < 1e-12
    assert switch_step(1000, 0.3077) == 308

    p_gated = interatomic_potential_gated(1000)
    assert p_gated["bm_start"] == 308
    assert p_gated["osc_gate"] is True
    assert p_gated["bm_shadow"] is True
    assert p_gated["iso"] is True

    p_staged = interatomic_potential(1000, memory=170.0)
    assert p_staged["bm_w_fast"] == 0.0
    assert abs(p_staged["bm_beta_slow"] - 170.0 / 171.0) < 1e-12

    p_lm = language_model(3250)
    assert p_lm["fresh_cap"] == 0.25
    assert p_lm["bm_start"] == 750


def test_physmuon_zero_e3nn_nequip_import_in_fresh_subprocess() -> None:
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=src_dir)
    code = (
        "import sys; import spingnn2.v2 as v2; "
        "assert 'e3nn' not in sys.modules; "
        "assert 'nequip' not in sys.modules; "
        "assert hasattr(v2, 'PhysMuon') and hasattr(v2, 'Muon')"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize("family", ["nosoc", "soc", "born", "e2slp"])
def test_equivariance_safe_routing_protects_single_row_and_gates(family: str) -> None:
    model = _build_family(family)
    opt = PhysMuon(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-2,
        named_parameters=model.named_parameters(),
        **interatomic_potential_gated(total_steps=100),
    )
    summary = opt.routing_summary()
    muon_set = set(summary["muon"])
    adam_all = set(summary["adam"]) | set(summary["adamw"])

    # Every single-row (1, K) scalar readout head & gate MUST stay off Muon!
    protected_substrings = [
        "out_proj",
        "adam_spin_gate",
        "adam_iso_energy",
        "adam_onsite_s1",
        "adam_onsite_s2",
        "adam_pair_gamma",
        "adam_pair_beta",
        "adam_born_scales",
        "type_embedding",
        "pair_mode_mix",
        "soc_dmi_out",
        "soc_kit_out",
        "soc_sia_out",
        "soc_couple_out",
        "bec_head.proj_0e",
        "bec_head.proj_1e",
        "bec_head.proj_2e",
        "trace_linear",
    ]
    for name, p in model.named_parameters():
        eff = [d for d in p.shape if d != 1]
        if len(eff) < 2 or min(eff) < 2:
            assert name not in muon_set, f"1D/single-row param {name} routed to Muon!"
            assert name in adam_all
        for token in protected_substrings:
            if token in name:
                # Note: radial_mlp.out_proj has shape (C0+modes, H) -> genuine 2D matrix;
                # only 1-row out_proj (energy/nu/kappa) must be protected.
                if token == "out_proj" and len(eff) == 2 and min(eff) >= 2:
                    continue
                assert name not in muon_set, f"Protected param {name} routed to Muon!"

    # True 2D equivariant channel matrices MUST route to Muon!
    for must_muon in (
        "mix_1o.weight",
        "mix_2e.weight",
        "axial_1o_ab.weight",
        "axial_2e_ab.weight",
        "radial_mlp.in_gate_val.weight",
    ):
        assert must_muon in muon_set, f"Expected 2D matrix {must_muon} on Muon"


def test_default_fp32_has_zero_v2_fp64_buffers_and_double_rematerializes() -> None:
    model = build_v2_soc(num_types=2, r_cut=5.5)
    # Check ALL floating-point buffers across v2 + symmetry are FP32 by default (zero FP64 casts!)
    for name, buf in model.named_buffers():
        if buf.is_floating_point():
            assert (
                buf.dtype == torch.float32
            ), f"Floating buffer {name} was {buf.dtype}, expected float32"

    # Converting FP32 -> .double() MUST re-materialize exact 53-bit FP64 constants!
    model_f64 = model.double()
    exact_centers = torch.linspace(0.0, 5.5, model.cfg.num_rbf, dtype=torch.float64)
    assert torch.max(torch.abs(model_f64.rbf.centers - exact_centers)).item() < 1e-15
    assert (
        torch.max(torch.abs(model_f64.gram_1o.triu_scale[1] - math.sqrt(2.0))).item()
        < 1e-15
    )


@pytest.mark.parametrize("family", ["nosoc", "soc", "born", "e2slp"])
def test_r1_r5_fp64_equivariance_after_physmuon_all5_iso_spike_step(
    family: str,
) -> None:
    """Build in default FP32 -> .double() -> step PhysMuon(all 5 switches + iso + spike_guard)
    -> assert exact R1-R5 symmetry to < 1e-13 in FP64!
    """
    torch.manual_seed(7)
    model = _build_family(family).double().train()
    opt = PhysMuon(
        model.parameters(),
        lr=1e-2,
        weight_decay=1e-2,
        named_parameters=model.named_parameters(),
        bm_start=2,
        bm_beta_fast=0.85,
        bm_beta_slow=0.98,
        iso=True,
        iso_min_dim=4,
        osc_gate=True,
        osc_wmax=0.4,
        bm_shadow=True,
        fresh_cap=0.25,
        spike_guard=5.0,
        spike_window=12,
        cuda_graph=False,
    )

    # Run 4 steps (crossing bm_start=2) with 2 micro-batches per step for iso=True
    for step_idx in range(4):
        opt.zero_grad(set_to_none=True)
        for mb in range(2):
            data = _make_small_cluster(seed=100 + step_idx * 10 + mb)
            out = model(
                pos=data["pos"].clone().requires_grad_(True),
                atom_types=data["atom_types"],
                edge_index=data["edge_index"],
                spin=data["spin"].clone().requires_grad_(True)
                if model.has_spin
                else None,
                batch=data["batch"],
                num_graphs=1,
                compute_forces=True,
                compute_virial=False,
                compute_spin_forces=model.has_spin,
                create_graph=True,
            )
            loss = out["total_energy"].sum() + 0.2 * out["forces"].square().mean()
            if model.has_spin:
                loss = loss + 0.2 * out["spin_forces"].square().mean()
            if "born_effective_charges" in out:
                loss = loss + 0.2 * out["born_effective_charges"].square().mean()
            if "uncertainty" in out:
                loss = loss + 0.1 * out["uncertainty"].sum()
            (loss / 2.0).backward()
            opt.observe_microbatch()
        opt.step(grad_norm=1.0)

    # Verify iso heats were populated on eligible 2D matrices
    heats_found = sum(
        1 for st in opt.state.values() if isinstance(st, dict) and "row_heat" in st
    )
    assert heats_found > 0

    # Now check exact FP64 R1-R5 equivariance (< 1e-13)!
    model.eval()
    data = _make_small_cluster(seed=999)
    pos = data["pos"]
    spin = data["spin"]
    at = data["atom_types"]
    ei = data["edge_index"]
    batch = data["batch"]

    def _eval(p_in: torch.Tensor, s_in: torch.Tensor):
        return model(
            pos=p_in.detach().clone().requires_grad_(True),
            atom_types=at,
            edge_index=ei,
            spin=s_in.detach().clone().requires_grad_(True),
            batch=batch,
            num_graphs=1,
            compute_forces=True,
            compute_virial=False,
            compute_spin_forces=model.has_spin,
            create_graph=False,
        )

    ref = _eval(pos, spin)
    R = random_rotation_matrix(dtype=torch.float64)
    # Joint SO(3) rotation (pos -> pos @ R^T, spin -> spin @ R^T)
    rot = _eval(pos @ R.T, spin @ R.T)
    assert (
        torch.max(torch.abs(rot["total_energy"] - ref["total_energy"])).item() < 1e-13
    )
    assert torch.max(torch.abs(rot["forces"] - ref["forces"] @ R.T)).item() < 1e-13
    if model.has_spin:
        assert (
            torch.max(torch.abs(rot["spin_forces"] - ref["spin_forces"] @ R.T)).item()
            < 1e-13
        )
    if "born_effective_charges" in ref:
        z_rot_exp = torch.einsum("ia,nab,jb->nij", R, ref["born_effective_charges"], R)
        assert (
            torch.max(torch.abs(rot["born_effective_charges"] - z_rot_exp)).item()
            < 1e-13
        )

    # Inversion P: pos -> -pos, spin -> +spin (axial!)
    inv = _eval(-pos, spin)
    assert (
        torch.max(torch.abs(inv["total_energy"] - ref["total_energy"])).item() < 1e-13
    )
    assert torch.max(torch.abs(inv["forces"] + ref["forces"])).item() < 1e-13
    if model.has_spin:
        assert (
            torch.max(torch.abs(inv["spin_forces"] - ref["spin_forces"])).item() < 1e-13
        )
    if "born_effective_charges" in ref:
        assert (
            torch.max(
                torch.abs(inv["born_effective_charges"] - ref["born_effective_charges"])
            ).item()
            < 1e-13
        )

    # Time reversal T: pos -> +pos, spin -> -spin
    tr = _eval(pos, -spin)
    assert torch.max(torch.abs(tr["total_energy"] - ref["total_energy"])).item() < 1e-13
    assert torch.max(torch.abs(tr["forces"] - ref["forces"])).item() < 1e-13
    if model.has_spin:
        assert (
            torch.max(torch.abs(tr["spin_forces"] + ref["spin_forces"])).item() < 1e-13
        )

    # Spin-only rotation for NoSOC (strict O(3)_lat x O(3)_spin)
    if family == "nosoc":
        Rs = random_rotation_matrix(dtype=torch.float64)
        srot = _eval(pos, spin @ Rs.T)
        assert (
            torch.max(torch.abs(srot["total_energy"] - ref["total_energy"])).item()
            < 1e-13
        )
        assert torch.max(torch.abs(srot["forces"] - ref["forces"])).item() < 1e-13
        assert (
            torch.max(torch.abs(srot["spin_forces"] - ref["spin_forces"] @ Rs.T)).item()
            < 1e-13
        )


def test_physmuon_spike_guard_freezes_slow_mode_and_iso_heat() -> None:
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.float64))
    opt = PhysMuon(
        [{"params": [w], "use_muon": True}],
        lr=1e-2,
        bm_start=0,
        bm_beta_fast=0.85,
        bm_beta_slow=0.98,
        bm_shadow=True,
        iso=True,
        iso_spike_skip=True,
        spike_guard=3.0,
        spike_window=12,
        cuda_graph=False,
    )
    # Fill spike window (12 normal steps with grad_norm=1.0)
    for _ in range(12):
        w.grad = torch.randn_like(w) * 0.1
        opt.observe_microbatch()
        w.grad = w.grad + torch.randn_like(w) * 0.1
        opt.observe_microbatch()
        opt.step(grad_norm=1.0)

    slow_before = opt.state[w]["m_slow"].clone()
    rh_before = opt.state[w]["row_heat"].clone()
    frozen_before = opt.spike_frozen_steps

    # Inject spike step (grad_norm=100.0 > 3 * median=3.0)
    w.grad = torch.ones_like(w) * 10.0
    opt.observe_microbatch()
    w.grad = w.grad + torch.ones_like(w) * 10.0
    opt.observe_microbatch()
    opt.step(grad_norm=100.0)

    assert opt.spike_frozen_steps == frozen_before + 1
    assert torch.equal(opt.state[w]["m_slow"], slow_before)
    assert torch.equal(opt.state[w]["row_heat"], rh_before)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for CUDA Graph parity"
)
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-5), (torch.float64, 1e-11)])
@pytest.mark.parametrize("bm_shadow", [True, False])
def test_physmuon_cuda_graph_exact_parity(
    dtype: torch.dtype, atol: float, bm_shadow: bool
) -> None:
    """Verify cuda_graph=True matches cuda_graph=False across WSD LR changes,
    bm_start transition, spike_guard skips, weight_decay > 0, iso=True, and grad=None subsets.
    """
    device = torch.device("cuda")
    torch.manual_seed(42)

    base_model = build_v2_soc(num_types=2, r_cut=5.5).to(device=device, dtype=dtype)
    base_model.set_spin_gate(0.5)
    m_eager = copy.deepcopy(base_model)
    m_graph = copy.deepcopy(base_model)

    common_kwargs = dict(
        lr=1e-2,
        weight_decay=1e-2,
        bm_start=4,
        bm_beta_fast=0.85,
        bm_beta_slow=0.98,
        osc_gate=True,
        osc_wmax=0.4,
        bm_shadow=bm_shadow,
        fresh_cap=0.2,
        iso=True,
        iso_min_dim=4,
        spike_guard=3.0,
        spike_window=10,
    )
    opt_eager = PhysMuon(
        m_eager.parameters(),
        named_parameters=m_eager.named_parameters(),
        cuda_graph=False,
        **common_kwargs,
    )
    opt_graph = PhysMuon(
        m_graph.parameters(),
        named_parameters=m_graph.named_parameters(),
        cuda_graph=True,
        **common_kwargs,
    )
    sched_eager = WSDCosineScheduler(opt_eager, total_steps=16, warmup_steps=3)
    sched_graph = WSDCosineScheduler(opt_graph, total_steps=16, warmup_steps=3)

    params_e = list(m_eager.parameters())
    params_g = list(m_graph.parameters())

    for step_idx in range(14):
        opt_eager.zero_grad(set_to_none=True)
        opt_graph.zero_grad(set_to_none=True)
        # Drop gradients on a subset on step 10 to test grad=None subset switching
        drop_some = step_idx == 10
        for mb in range(2):
            gen = torch.Generator(device="cpu").manual_seed(500 + step_idx * 7 + mb)
            for idx_p, (pe, pg) in enumerate(zip(params_e, params_g)):
                if drop_some and idx_p % 5 == 0:
                    continue
                g_mb = (
                    torch.randn(pe.shape, generator=gen, dtype=dtype).to(device) * 0.05
                )
                pe.grad = g_mb.clone() if pe.grad is None else pe.grad + g_mb
                pg.grad = g_mb.clone() if pg.grad is None else pg.grad + g_mb
            opt_eager.observe_microbatch()
            opt_graph.observe_microbatch()

        # Trigger spike_guard on step 12 (window already >= 10)
        gn = 50.0 if step_idx == 12 else 1.0
        opt_eager.step(grad_norm=gn)
        opt_graph.step(grad_norm=gn)
        sched_eager.step()
        sched_graph.step()

    assert len(opt_graph._cuda_graphs) >= 1, "Expected CUDA graph to be captured"
    max_diff = max(
        torch.max(torch.abs(pe - pg)).item() for pe, pg in zip(params_e, params_g)
    )
    assert max_diff < atol, f"CUDA Graph vs eager drift {max_diff} >= {atol}"


def test_compile_bucketed_zero_leakage_and_bucket_cap() -> None:
    model = build_v2_born(
        num_types=2, r_cut=5.5, asr="hard", has_spin=True, has_soc=True
    ).double()
    model.set_spin_gate(0.5)
    with pytest.raises(ValueError, match="1..4"):
        model.compile_bucketed(atom_buckets=(8, 16, 32, 64, 128))

    data = _make_small_cluster(n_atoms=6, dtype=torch.float64)
    # Eager reference (exact)
    m_ref = copy.deepcopy(model)
    out_ref = m_ref(
        pos=data["pos"].clone().requires_grad_(True),
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        spin=data["spin"].clone().requires_grad_(True),
        batch=data["batch"],
        num_graphs=1,
        compute_forces=True,
        compute_virial=True,
        compute_spin_forces=True,
        create_graph=True,
    )
    loss_ref = (
        out_ref["total_energy"].sum()
        + out_ref["forces"].square().mean()
        + out_ref["spin_forces"].square().mean()
        + out_ref["born_effective_charges"].square().mean()
    )
    loss_ref.backward()

    # Bucketed padded path (using eager backend so CPU test stays fast)
    m_pad = copy.deepcopy(model)
    m_pad.compile_bucketed(
        atom_buckets=(10, 16),
        edge_buckets=(40, 80),
        mode="default",
        backend="eager",
    )
    out_pad = m_pad(
        pos=data["pos"].clone().requires_grad_(True),
        atom_types=data["atom_types"],
        edge_index=data["edge_index"],
        spin=data["spin"].clone().requires_grad_(True),
        batch=data["batch"],
        num_graphs=1,
        compute_forces=True,
        compute_virial=True,
        compute_spin_forces=True,
        create_graph=True,
    )
    loss_pad = (
        out_pad["total_energy"].sum()
        + out_pad["forces"].square().mean()
        + out_pad["spin_forces"].square().mean()
        + out_pad["born_effective_charges"].square().mean()
    )
    loss_pad.backward()

    for key in (
        "total_energy",
        "per_atom_energy",
        "forces",
        "virial",
        "spin_forces",
        "born_effective_charges",
        "total_born_effective_charges",
    ):
        diff = torch.max(torch.abs(out_ref[key] - out_pad[key])).item()
        assert diff < 1e-12, f"Bucketed padding leaked on {key}: diff={diff}"

    for (n, p1), (_, p2) in zip(m_ref.named_parameters(), m_pad.named_parameters()):
        if p1.grad is None:
            continue
        gdiff = torch.max(torch.abs(p1.grad - p2.grad)).item()
        assert gdiff < 1e-11, f"Bucketed gradient leaked on {n}: diff={gdiff}"


def test_v1_yaml_trainer_physmuon_and_muon_smoke(tmp_path: Path) -> None:
    """Verify v1 nequip Trainer instantiates & steps PhysMuon, Muon, and param groups."""
    core_dir = str(Path(__file__).resolve().parents[1] / "src/core")
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)

    from nequip.train._param_groups import check_param_group_options
    from nequip.utils.optim import HybridMuon as V1HybridMuon
    from nequip.utils.optim import Muon as V1Muon
    from nequip.utils.optim import PhysMuon as V1PhysMuon

    assert V1PhysMuon is PhysMuon
    assert V1Muon is Muon
    assert V1HybridMuon is HybridMuon

    # Verify YAML optimizer_param_groups with `use_muon` passes check_param_group_options
    tiny = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 1))
    specs = [
        {"name": "hidden", "match": ["0.weight"], "use_muon": True, "lr": 1e-2},
        {"name": "rest", "use_muon": False, "lr": 1e-3},
    ]
    opt = V1PhysMuon(
        [
            {"params": [tiny[0].weight], "use_muon": True, "lr": 1e-2},
            {
                "params": [tiny[0].bias, tiny[1].weight, tiny[1].bias],
                "use_muon": False,
                "lr": 1e-3,
            },
        ],
        **interatomic_potential_gated(total_steps=50),
        cuda_graph=False,
    )
    check_param_group_options(opt, specs)

    # Full nequip Trainer smoke check with PhysMuon + preset + grad_accumulation_steps=2
    from nequip.data import AtomicData, AtomicDataDict
    from nequip.train.trainer import Trainer

    class _ToyV1Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = torch.nn.Linear(3, 8)
            self.hidden = torch.nn.Linear(8, 8)
            self.head = torch.nn.Linear(8, 1)

        def forward(self, data: dict) -> dict:
            out = dict(data)
            pos = data[AtomicDataDict.POSITIONS_KEY]
            h = torch.tanh(self.hidden(torch.tanh(self.in_proj(pos))))
            e_atom = self.head(h)
            out[AtomicDataDict.PER_ATOM_ENERGY_KEY] = e_atom
            out[AtomicDataDict.TOTAL_ENERGY_KEY] = e_atom.sum(dim=0, keepdim=True)
            return out

    model = _ToyV1Model()
    trainer = Trainer(
        model=model,
        root=str(tmp_path / "run_pm"),
        run_name="smoke",
        device="cpu",
        loss_coeffs="total_energy",
        optimizer_name="PhysMuon",
        optimizer_preset="interatomic_potential_gated",
        optimizer_total_steps=20,
        optimizer_kwargs={"lr": 1e-2, "spike_guard": 5.0, "cuda_graph": False},
        grad_accumulation_steps=2,
        max_gradient_norm=10.0,
        max_epochs=1,
    )
    assert isinstance(trainer.optim, PhysMuon)
    assert trainer.optim.param_groups[0]["iso"] is True  # grad_accum=2 keeps iso=True!
    trainer.init_metrics()

    # Step 2 micro-batches through batch_step
    for idx in range(2):
        trainer.ibatch = idx
        trainer.n_batches = 2
        d = AtomicData(
            pos=torch.randn(4, 3),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
            atom_types=torch.zeros(4, dtype=torch.long),
            total_energy=torch.tensor([[0.5]]),
        )
        trainer.batch_step(d, validation=False)

    # Confirm iso heat was populated via trainer's observe_microbatch() hook
    assert any(
        isinstance(st, dict) and "row_heat" in st for st in trainer.optim.state.values()
    )

    # Also test single-GPU grad_accumulation_steps=1 auto-disables iso cleanly
    trainer_muon = Trainer(
        model=_ToyV1Model(),
        root=str(tmp_path / "run_muon"),
        run_name="smoke_muon",
        device="cpu",
        loss_coeffs="total_energy",
        optimizer_name="Muon",
        optimizer_kwargs={"lr": 1e-2, "cuda_graph": False},
        max_epochs=1,
    )
    assert isinstance(trainer_muon.optim, Muon)

    trainer_single_pm = Trainer(
        model=_ToyV1Model(),
        root=str(tmp_path / "run_pm1"),
        run_name="smoke_pm1",
        device="cpu",
        loss_coeffs="total_energy",
        optimizer_name="PhysMuon",
        optimizer_preset="interatomic_potential_gated",
        optimizer_total_steps=20,
        optimizer_kwargs={"lr": 1e-2, "cuda_graph": False},
        grad_accumulation_steps=1,
        max_epochs=1,
    )
    assert trainer_single_pm.optim.param_groups[0]["iso"] is False


def test_physmuon_state_dict_resume_parity() -> None:
    """Verify PhysMuon state_dict/load_state_dict resumes identically across bm_start."""
    torch.manual_seed(11)
    w_cont = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.float64))
    w_res = torch.nn.Parameter(w_cont.detach().clone())

    cfg = dict(
        lr=1e-2,
        weight_decay=1e-2,
        bm_start=2,
        bm_beta_fast=0.85,
        bm_beta_slow=0.98,
        osc_gate=True,
        osc_wmax=0.4,
        bm_shadow=True,
        fresh_cap=0.2,
        iso=True,
        iso_min_dim=4,
        cuda_graph=False,
    )
    opt_cont = PhysMuon([{"params": [w_cont], "use_muon": True}], **cfg)
    opt_res = PhysMuon([{"params": [w_res], "use_muon": True}], **cfg)

    for step in range(5):
        gen = torch.Generator().manual_seed(700 + step)
        g1 = torch.randn_like(w_cont, generator=gen) * 0.1
        g2 = torch.randn_like(w_cont, generator=gen) * 0.1
        for w, opt in ((w_cont, opt_cont), (w_res, opt_res)):
            w.grad = g1.clone()
            opt.observe_microbatch()
            w.grad = g1 + g2
            opt.observe_microbatch()
            opt.step()
        if step == 2:
            # Checkpoint & restore opt_res from state_dict!
            sd = copy.deepcopy(opt_res.state_dict())
            opt_res = PhysMuon([{"params": [w_res], "use_muon": True}], **cfg)
            opt_res.load_state_dict(sd)

    assert torch.max(torch.abs(w_cont - w_res)).item() < 1e-13


def test_physmuon_iso_single_microbatch_tail_and_ddp_local_reset() -> None:
    """Verify k_mb==1 tail never shrinks EMA heat and _ddp_rh_local clears iso_count."""
    torch.manual_seed(17)
    w = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.float32))
    opt = PhysMuon(
        [{"params": [w], "use_muon": True}],
        lr=1e-2,
        iso=True,
        cuda_graph=False,
    )
    # Normal 2-microbatch step establishes non-trivial row_heat / col_heat
    w.grad = torch.randn_like(w) * 0.2
    opt.observe_microbatch()
    w.grad = w.grad + torch.randn_like(w) * 0.2
    opt.observe_microbatch()
    opt.step()

    rh_before = opt.state[w]["row_heat"].clone()
    ch_before = opt.state[w]["col_heat"].clone()
    assert float(rh_before.min().item()) > 1e-6

    # Single-microbatch epoch-end tail (k_mb == 1) must reset state & leave EMA untouched
    w.grad = torch.randn_like(w) * 0.2
    opt.observe_microbatch()
    opt.step()
    assert opt.state[w]["iso_count"] == 0
    assert "iso_prev" not in opt.state[w]
    assert torch.equal(opt.state[w]["row_heat"], rh_before)
    assert torch.equal(opt.state[w]["col_heat"], ch_before)

    # Main-thread DDP local-diff consumption must clear any stale iso_count/iso_prev
    w.grad = torch.randn_like(w) * 0.2
    opt.observe_microbatch()
    opt.state[w]["_ddp_rh_local"] = torch.ones(8, dtype=torch.float32) * 0.5
    opt.state[w]["_ddp_ch_local"] = torch.ones(8, dtype=torch.float32) * 0.5
    opt.step()
    assert opt.state[w]["iso_count"] == 0
    assert "iso_prev" not in opt.state[w]
    assert "_ddp_rh_local" not in opt.state[w]
    assert not torch.equal(opt.state[w]["row_heat"], rh_before)
