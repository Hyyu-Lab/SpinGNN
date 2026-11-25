"""torch.compile(fullgraph=True, dynamic=True) & HybridMuon training tests for SpinGNN2 v2."""

from __future__ import annotations

import pytest
import torch

from spingnn2.v2 import (
    HybridMuon,
    WSDCosineScheduler,
    build_v2_born,
    build_v2_e2slp,
    build_v2_nosoc,
    build_v2_soc,
    newton_schulz_orthogonalize,
)


def _make_synthetic_graph(
    num_atoms: int,
    num_types: int = 2,
    r_cut: float = 5.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    box = 7.0
    pos = torch.rand((num_atoms, 3), generator=gen, dtype=dtype) * box
    cell = torch.eye(3, dtype=dtype).unsqueeze(0) * box
    atom_types = torch.arange(num_atoms, dtype=torch.long) % num_types
    spin = torch.randn((num_atoms, 3), generator=gen, dtype=dtype)
    spin = spin / torch.linalg.vector_norm(spin, dim=-1, keepdim=True) * 2.0
    batch = torch.zeros(num_atoms, dtype=torch.long)

    dst = []
    src = []
    shifts = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    for sz in (-1, 0, 1):
                        if i == j and sx == 0 and sy == 0 and sz == 0:
                            continue
                        sh = torch.tensor([sx, sy, sz], dtype=dtype)
                        vec = pos[j] - pos[i] + sh * box
                        d = float(torch.linalg.vector_norm(vec).item())
                        if 0.3 < d < r_cut:
                            dst.append(i)
                            src.append(j)
                            shifts.append(sh)
    return {
        "pos": pos,
        "atom_types": atom_types,
        "edge_index": torch.tensor([dst, src], dtype=torch.long),
        "edge_cell_shift": torch.stack(shifts, dim=0),
        "cell": cell,
        "batch": batch,
        "spin": spin,
    }


def _build_family(family: str, edge_amp: bool = False, edge_checkpoint: bool = False):
    kw = dict(
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
        edge_amp=edge_amp,
        edge_checkpoint=edge_checkpoint,
    )
    if family == "v2_nosoc":
        return build_v2_nosoc(num_types=2, r_cut=5.0, use_spin=(True, False), **kw)
    if family == "v2_soc":
        return build_v2_soc(num_types=2, r_cut=5.0, use_spin=(True, True), **kw)
    if family == "v2_born":
        return build_v2_born(num_types=2, r_cut=5.0, asr="soft", has_spin=False, **kw)
    if family == "v2_e2slp":
        return build_v2_e2slp(
            num_types=2,
            r_cut=5.0,
            has_spin=True,
            has_soc=True,
            use_spin=(True, True),
            **kw,
        )
    raise ValueError(family)


@pytest.mark.parametrize("family", ["v2_nosoc", "v2_soc", "v2_born", "v2_e2slp"])
def test_torch_compile_fullgraph_zero_breaks_fwd_and_second_order_bwd(family: str):
    """All 4 families must compile with fullgraph=True, dynamic=True including force/torque grad.

    Uses full Inductor codegen on CUDA (fast Triton compilation) and ``aot_eager``
    on CPU so double-backward tests complete rapidly on CPU-only runners while
    still enforcing Dynamo ``fullgraph=True, dynamic=True`` + AOTAutograd joint
    forward/backward/double-backward compilation with zero graph breaks.
    """
    torch._dynamo.reset()
    torch.manual_seed(101)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "inductor" if device.type == "cuda" else "aot_eager"

    model = (
        _build_family(family, edge_amp=False, edge_checkpoint=False).to(device).train()
    )
    if model.has_spin:
        model.set_spin_gate(0.5)
    if model.has_soc:
        with torch.no_grad():
            model.soc_dmi_out.weight.fill_(0.1)
            model.soc_kit_out.weight.fill_(0.1)
            model.soc_sia_out.weight.fill_(0.1)
            model.soc_couple_out.weight.fill_(0.1)

    compiled_model = (
        _build_family(family, edge_amp=False, edge_checkpoint=False).to(device).train()
    )
    compiled_model.load_state_dict(model.state_dict())

    # Verify Dynamo explain sees 1 graph and 0 graph breaks for FULL forward + forces + virial + torques!
    g0 = {
        k: v.to(device) for k, v in _make_synthetic_graph(num_atoms=6, seed=1).items()
    }
    disp0 = torch.zeros(
        (1, 3, 3), dtype=torch.float32, device=device, requires_grad=True
    )
    explain = torch._dynamo.explain(compiled_model)(
        pos=g0["pos"].clone().requires_grad_(True),
        atom_types=g0["atom_types"],
        edge_index=g0["edge_index"],
        edge_cell_shift=g0["edge_cell_shift"],
        cell=g0["cell"],
        batch=g0["batch"],
        spin=g0["spin"].clone().requires_grad_(True) if model.has_spin else None,
        num_graphs=1,
        compute_forces=True,
        compute_virial=True,
        compute_spin_forces=model.has_spin,
        create_graph=True,
        displacement=disp0,
    )
    assert explain.graph_break_count == 0, f"Graph breaks: {explain.break_reasons}"
    assert explain.graph_count == 1

    torch._dynamo.reset()
    compiled_fn = torch.compile(
        compiled_model, fullgraph=True, dynamic=True, backend=backend
    )

    g1 = {
        k: v.to(device) for k, v in _make_synthetic_graph(num_atoms=6, seed=1).items()
    }
    g2 = {
        k: v.to(device) for k, v in _make_synthetic_graph(num_atoms=9, seed=2).items()
    }

    for step_idx, g in enumerate((g1, g2)):
        pos_e = g["pos"].clone().requires_grad_(True)
        spin_e = g["spin"].clone().requires_grad_(True) if model.has_spin else None
        disp_e = torch.zeros(
            (1, 3, 3), dtype=torch.float32, device=device, requires_grad=True
        )
        out_eager = model(
            pos=pos_e,
            atom_types=g["atom_types"],
            edge_index=g["edge_index"],
            edge_cell_shift=g["edge_cell_shift"],
            cell=g["cell"],
            batch=g["batch"],
            spin=spin_e,
            num_graphs=1,
            compute_forces=True,
            compute_virial=True,
            compute_spin_forces=model.has_spin,
            create_graph=True,
            displacement=disp_e,
        )

        pos_c = g["pos"].clone().requires_grad_(True)
        spin_c = g["spin"].clone().requires_grad_(True) if model.has_spin else None
        disp_c = torch.zeros(
            (1, 3, 3), dtype=torch.float32, device=device, requires_grad=True
        )
        out_comp = compiled_fn(
            pos=pos_c,
            atom_types=g["atom_types"],
            edge_index=g["edge_index"],
            edge_cell_shift=g["edge_cell_shift"],
            cell=g["cell"],
            batch=g["batch"],
            spin=spin_c,
            num_graphs=1,
            compute_forces=True,
            compute_virial=True,
            compute_spin_forces=model.has_spin,
            create_graph=True,
            displacement=disp_c,
        )

        torch.testing.assert_close(
            out_comp["total_energy"], out_eager["total_energy"], atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            out_comp["forces"], out_eager["forces"], atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            out_comp["virial"], out_eager["virial"], atol=1e-4, rtol=1e-4
        )
        if model.has_spin:
            torch.testing.assert_close(
                out_comp["spin_forces"], out_eager["spin_forces"], atol=1e-4, rtol=1e-4
            )
        if model.predict_bec:
            torch.testing.assert_close(
                out_comp["born_effective_charges"],
                out_eager["born_effective_charges"],
                atol=1e-4,
                rtol=1e-4,
            )
        if model.predict_e2slp:
            torch.testing.assert_close(
                out_comp["f_Uepi"], out_eager["f_Uepi"], atol=1e-4, rtol=1e-4
            )

        # Second-order backward through compiled graph (forces + virial + torques + heads)
        compiled_model.zero_grad(set_to_none=True)
        loss_comp = (
            out_comp["total_energy"].square().sum()
            + out_comp["forces"].square().mean()
            + out_comp["virial"].square().mean()
        )
        if model.has_spin:
            loss_comp = loss_comp + out_comp["spin_forces"].square().mean()
        if model.predict_bec:
            loss_comp = (
                loss_comp
                + out_comp["born_effective_charges"].square().mean()
                + out_comp["total_born_effective_charges"].square().mean()
            )
        if model.predict_e2slp:
            loss_comp = (
                loss_comp
                + out_comp["f_Uepi"].square().mean()
                + out_comp["f_nu"].square().mean()
                + out_comp["f_kappa"].square().mean()
            )
            if "sf_Uepi" in out_comp:
                loss_comp = loss_comp + out_comp["sf_Uepi"].square().mean()

        loss_comp.backward()
        grads = [p.grad for p in compiled_model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(gr).all() for gr in grads)
        assert step_idx in (0, 1)


def test_hybrid_muon_routing_orthogonalization_and_wsd():
    """Verify Newton-Schulz orthogonality, HybridMuon routing, and WSD LR schedule."""
    torch.manual_seed(7)
    mat = torch.randn((16, 32), dtype=torch.float64)
    orth = newton_schulz_orthogonalize(mat)
    gram = orth @ orth.T
    torch.testing.assert_close(
        gram, torch.eye(16, dtype=torch.float64), atol=1e-3, rtol=1e-3
    )
    orth_tight = newton_schulz_orthogonalize(mat, polish_steps=4)
    torch.testing.assert_close(
        orth_tight @ orth_tight.T,
        torch.eye(16, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )

    model = _build_family("v2_soc")
    opt = HybridMuon(
        model.parameters(),
        lr=1e-2,
        weight_decay=1e-2,
        named_parameters=model.named_parameters(),
    )
    summary = opt.routing_summary()
    assert len(summary["muon"]) > 0, "2D linear weights must route to Muon"
    assert len(summary["adam"]) > 0, "1D biases/scales must route to Adam"
    assert any(
        "type_embedding" in n for n in summary["adamw"]
    ), "Embeddings must route to AdamW"
    assert any(
        "adam_spin_gate" in n for n in summary["adam"]
    ), "spin_gate must route to Adam"

    sched = WSDCosineScheduler(
        opt,
        warmup_steps=5,
        stable_steps=5,
        decay_steps=10,
        min_lr_ratio=0.1,
        warmup_init_ratio=0.2,
    )
    lrs = []
    for _ in range(22):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert abs(lrs[0] - 0.002) < 1e-8
    assert abs(lrs[5] - 0.01) < 1e-8
    assert abs(lrs[9] - 0.01) < 1e-8
    assert lrs[15] < lrs[10]
    assert abs(lrs[20] - 0.001) < 1e-8
    assert abs(lrs[21] - 0.001) < 1e-8


def test_end_to_end_training_convergence_hybrid_muon_amp_checkpoint():
    """End-to-end convergence test with HybridMuon + WSD + edge bf16 AMP + checkpointing."""
    torch.manual_seed(2026)
    model = _build_family("v2_soc", edge_amp=True, edge_checkpoint=True).train()
    g = _make_synthetic_graph(num_atoms=8, seed=42)

    # Synthetic physical targets (energy + forces + spin torques)
    target_e = torch.tensor([-2.5], dtype=torch.float32)
    target_f = torch.zeros_like(g["pos"])
    target_fm = torch.zeros_like(g["spin"])
    target_fm[0] = torch.tensor([0.2, -0.1, 0.05])
    target_fm[1] = torch.tensor([-0.15, 0.1, 0.05])

    opt = HybridMuon(
        model.parameters(),
        lr=8e-3,
        momentum=0.9,
        weight_decay=1e-3,
        named_parameters=model.named_parameters(),
        magma_damping=True,
    )
    sched = WSDCosineScheduler(
        opt, warmup_steps=3, stable_steps=10, decay_steps=12, min_lr_ratio=0.1
    )

    losses = []
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        out = model(
            pos=g["pos"].clone().requires_grad_(True),
            atom_types=g["atom_types"],
            edge_index=g["edge_index"],
            edge_cell_shift=g["edge_cell_shift"],
            cell=g["cell"],
            batch=g["batch"],
            spin=g["spin"].clone().requires_grad_(True),
            num_graphs=1,
            compute_forces=True,
            compute_virial=False,
            compute_spin_forces=True,
            create_graph=True,
        )
        loss = (
            (out["total_energy"] - target_e).square().mean()
            + (out["forces"] - target_f).square().mean()
            + 2.0 * (out["spin_forces"] - target_fm).square().mean()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))

    assert all(torch.isfinite(torch.tensor(losses))), f"Non-finite losses: {losses}"
    assert (
        losses[-1] < losses[0] * 0.25
    ), f"Training failed to converge cleanly: start={losses[0]:.4f}, end={losses[-1]:.4f}"
    # Confirm SOC & spin_gate learned non-zero weights from the spin-torque target
    assert abs(float(model.adam_spin_gate.item())) > 1e-4
    assert float(model.soc_dmi_out.weight.abs().max().item()) > 1e-5


def test_v2_pre_ampere_volta_bf16_fallback(monkeypatch):
    """Pre-Ampere CUDA devices (e.g. V100 sm_70) fall back from bfloat16 to float32."""
    from spingnn2.v2.optim import _ns_work_dtype
    from spingnn2.v2.stage0 import edge_bf16_autocast

    class _FakeCudaTensor:
        def __init__(self, dtype):
            self.dtype = dtype
            self.is_cuda = True
            self.device = torch.device("cuda", 0)

    # Simulate Volta V100 (compute capability 7.0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev=None: (7, 0))
    assert _ns_work_dtype(_FakeCudaTensor(torch.float32)) == torch.float32
    assert _ns_work_dtype(_FakeCudaTensor(torch.float64)) == torch.float64

    entered = []
    orig_autocast = torch.amp.autocast

    def _recording_autocast(*args, **kwargs):
        entered.append((args, kwargs))
        return orig_autocast(*args, **kwargs)

    monkeypatch.setattr(torch.amp, "autocast", _recording_autocast)
    with edge_bf16_autocast(True, _FakeCudaTensor(torch.float32)):
        pass
    assert entered == [], "edge_bf16_autocast must disable bf16 autocast on sm_70"

    # Simulate Ampere+ (compute capability 8.0) -> bf16 stays enabled
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev=None: (8, 0))
    assert _ns_work_dtype(_FakeCudaTensor(torch.float32)) == torch.bfloat16
    assert _ns_work_dtype(_FakeCudaTensor(torch.float64)) == torch.float64
    with edge_bf16_autocast(True, _FakeCudaTensor(torch.float32)):
        pass
    assert len(entered) == 1
    assert entered[0][1]["dtype"] == torch.bfloat16
    assert entered[0][1]["device_type"] == "cuda"
