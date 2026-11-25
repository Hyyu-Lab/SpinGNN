"""Fair RTX 5090 A/B training benchmark: SpinGNN2 v2 vs v1 Allegro models.

Compares matched realistic periodic supercells (default 64, 128, 256, 512 atoms)
across all four model families:
- ``nosoc``: ``v2_nosoc`` vs v1 ``Allegro_nosoc``
- ``soc``: ``v2_soc`` vs v1 ``Allegro_soc``
- ``born`` (DREAM): ``v2_born`` vs v1 ``Allegro_born``
- ``e2slp``: ``v2_e2slp`` vs v1 ``Allegro_e2IP`` (and ``Allegro_soc_e2IP``)

Each timed training step includes forward + conservative autograd forces
(and virial/spin torques/BEC/e2SLP outputs per family) + loss + backward +
optimizer step, measured with synchronized CUDA events and exact peak VRAM via
``torch.cuda.max_memory_allocated()``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "src/core", ROOT / "src/models"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from spingnn2.v2 import (  # noqa: E402
    HybridMuon,
    Muon,
    PhysMuon,
    build_v2_born,
    build_v2_e2slp,
    build_v2_nosoc,
    build_v2_soc,
    interatomic_potential_gated,
)


def make_supercell_batch(
    num_atoms: int,
    r_cut: float = 5.0,
    num_types: int = 2,
    seed: int = 42,
    device: torch.device = torch.device("cuda"),
) -> dict[str, Any]:
    """Build a realistic periodic rocksalt-like supercell with ~24-36 neighbors/atom."""
    rng = np.random.default_rng(seed)
    # Choose cubic grid side n_side so 2 * n_side^3 >= num_atoms (BCC/FCC density)
    # Target number density ~0.075 atoms/A^3 -> at r_cut=5.0 A, ~39 neighbors/atom
    density = 0.065
    box = float((num_atoms / density) ** (1.0 / 3.0))
    n_grid = int(math.ceil(num_atoms ** (1.0 / 3.0)))
    coords = []
    for ix in range(n_grid):
        for iy in range(n_grid):
            for iz in range(n_grid):
                if len(coords) < num_atoms:
                    coords.append(
                        [
                            (ix + 0.5) * box / n_grid,
                            (iy + 0.5) * box / n_grid,
                            (iz + 0.5) * box / n_grid,
                        ]
                    )
    pos_np = np.asarray(coords, dtype=np.float32)
    pos_np = (
        pos_np + rng.normal(0.0, 0.12, size=pos_np.shape).astype(np.float32)
    ) % box

    # Vectorized periodic neighbor list
    diff = pos_np[None, :, :] - pos_np[:, None, :]  # (N, N, 3) = pos[j] - pos[i]
    shifts_np = -np.round(diff / box)
    vec = diff + shifts_np * box
    dist = np.linalg.norm(vec, axis=-1)
    mask = (dist > 0.1) & (dist < r_cut)
    dst_np, src_np = np.nonzero(mask)
    edge_shifts_np = shifts_np[dst_np, src_np].astype(np.float32)

    # Periodic image wrap for small boxes where box < 2*r_cut
    if box < 2.0 * r_cut:
        dst_list = []
        src_list = []
        sh_list = []
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                for sz in (-1, 0, 1):
                    sh = np.array([sx, sy, sz], dtype=np.float32)
                    v_img = diff + sh[None, None, :] * box
                    d_img = np.linalg.norm(v_img, axis=-1)
                    m_img = (d_img > 0.1) & (d_img < r_cut)
                    di, si = np.nonzero(m_img)
                    if di.size > 0:
                        dst_list.append(di)
                        src_list.append(si)
                        sh_list.append(np.broadcast_to(sh, (di.size, 3)).copy())
        dst_np = np.concatenate(dst_list)
        src_np = np.concatenate(src_list)
        edge_shifts_np = np.concatenate(sh_list, axis=0)

    atom_types_np = (np.arange(num_atoms) % num_types).astype(np.int64)
    spin_np = rng.normal(size=(num_atoms, 3)).astype(np.float32)
    spin_np /= np.linalg.norm(spin_np, axis=-1, keepdims=True)
    spin_np *= 2.0

    pos = torch.tensor(pos_np, dtype=torch.float32, device=device)
    atom_types = torch.tensor(atom_types_np, dtype=torch.long, device=device)
    edge_index = torch.tensor(
        np.stack([dst_np, src_np], axis=0), dtype=torch.long, device=device
    )
    edge_cell_shift = torch.tensor(edge_shifts_np, dtype=torch.float32, device=device)
    cell = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0) * box
    pbc = torch.ones((1, 3), dtype=torch.bool, device=device)
    batch = torch.zeros((num_atoms,), dtype=torch.long, device=device)
    spin = torch.tensor(spin_np, dtype=torch.float32, device=device)

    avg_nbr = float(edge_index.shape[1]) / float(num_atoms)
    return {
        "pos": pos,
        "atom_types": atom_types,
        "edge_index": edge_index,
        "edge_cell_shift": edge_cell_shift,
        "cell": cell,
        "pbc": pbc,
        "batch": batch,
        "spin": spin,
        "num_atoms": num_atoms,
        "num_edges": int(edge_index.shape[1]),
        "avg_num_neighbors": avg_nbr,
    }


def build_v1_model(
    family: str,
    r_cut: float,
    avg_num_neighbors: float,
    device: torch.device,
) -> torch.nn.Module:
    """Build realistic production-scale v1 Allegro baseline matching repo configs."""
    from nequip.model import model_from_config

    common: dict[str, Any] = {
        "num_types": 2,
        "type_names": ["Fe", "O"],
        "r_max": float(r_cut),
        "avg_num_neighbors": float(avg_num_neighbors),
        "l_max": 2,
        "parity": "o3_full",
        "num_layers": 2,
        "env_embed_multiplicity": 16,
        "two_body_latent_mlp_latent_dimensions": [32, 64, 128],
        "latent_mlp_latent_dimensions": [128],
        "env_embed_mlp_latent_dimensions": [],
        "edge_eng_mlp_latent_dimensions": [128, 64],
        "BesselBasis_trainable": False,
        "PolynomialCutoff_p": 48,
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
            edge_onlyr_eng_mlp_latent_dimensions=[128, 64],
            edge_hb_coeff_mlp_latent_dimensions=[128, 64],
            edge_biquad_coeff_mlp_latent_dimensions=[128, 64],
            edge_eng_nosoc_mlp_latent_dimensions=[128],
            edge_onlyr_eng_coeff=1.0,
            edge_hb_eng_coeff=0.05,
            edge_eng_nosoc_coeff=0.05,
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
            edge_onlyr_eng_mlp_latent_dimensions=[128, 64],
            edge_biquad_coeff_mlp_latent_dimensions=[128, 64],
            edge_eng_soc_on_mlp_latent_dimensions=[128, 64],
            edge_onlyr_eng_coeff=1.0,
            edge_jmatrix_eng_coeff=0.1,
            edge_ani_eng_coeff=0.01,
            edge_biquad_eng_coeff=0.1,
            edge_eng_soc_on_coeff=0.01,
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
            edge_onlyr_eng_mlp_latent_dimensions=[128, 64],
            edge_biquad_coeff_mlp_latent_dimensions=[128, 64],
            edge_eng_soc_on_mlp_latent_dimensions=[128, 64],
            e2ip_latent_channels=8,
            e2ip_mlp_latent_dimensions=[32, 32],
            evidential_key="spin_forces_vert",
            spingnn_compile={"enabled": False},
        )
    else:
        raise ValueError(f"Unknown family {family!r}")

    return model_from_config(cfg).to(device=device)


def build_v2_model(
    family: str,
    r_cut: float,
    device: torch.device,
    edge_amp: bool = True,
    edge_checkpoint: bool = True,
) -> torch.nn.Module:
    """Build production-width SpinGNN2 v2 model for fair comparison."""
    common = dict(
        num_types=2,
        r_cut=float(r_cut),
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
        edge_amp=edge_amp,
        edge_checkpoint=edge_checkpoint,
    )
    if family == "nosoc":
        model = build_v2_nosoc(use_spin=(True, False), **common)
    elif family == "soc":
        model = build_v2_soc(use_spin=(True, True), **common)
    elif family == "born":
        model = build_v2_born(
            asr="soft", has_spin=False, bec_scale=(4.0, -2.0), **common
        )
    elif family == "e2slp":
        model = build_v2_e2slp(
            has_spin=True,
            has_soc=True,
            use_spin=(True, True),
            include_f=True,
            include_sf=True,
            e2slp_vec_channels=8,
            e2slp_hidden=32,
            **common,
        )
    else:
        raise ValueError(f"Unknown family {family!r}")
    return model.to(device=device)


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_v1(
    family: str,
    batch_data: dict[str, Any],
    r_cut: float,
    warmup: int,
    steps: int,
    device: torch.device,
) -> dict[str, Any]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = build_v1_model(
        family, r_cut, batch_data["avg_num_neighbors"], device
    ).train()
    n_params = _count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def _step() -> tuple[float, float]:
        opt.zero_grad(set_to_none=True)
        inp = {
            "pos": batch_data["pos"].detach().clone(),
            "atom_types": batch_data["atom_types"],
            "edge_index": batch_data["edge_index"],
            "edge_cell_shift": batch_data["edge_cell_shift"],
            "cell": batch_data["cell"],
            "pbc": batch_data["pbc"],
            "batch": batch_data["batch"],
        }
        if family in ("nosoc", "soc", "e2slp"):
            inp["spin"] = batch_data["spin"].detach().clone()
        t0 = time.perf_counter()
        out = model(inp)
        loss = (
            out["total_energy"].square().mean()
            + out["forces"].square().mean()
            + out["virial"].square().mean()
        )
        if "spin_forces" in out:
            loss = loss + out["spin_forces"].square().mean()
        if "born_effective_charges" in out:
            loss = (
                loss
                + out["born_effective_charges"].square().mean()
                + out["total_born_effective_charges"].square().mean()
            )
        if "sf_Uepi" in out:
            loss = (
                loss
                + out["sf_Uepi"].square().mean()
                + out["sf_nu"].square().mean()
                + out["sf_kappa"].square().mean()
            )
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

    for _ in range(warmup):
        _step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    times_ms = []
    fb_ms = []
    opt_ms = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fb, om = _step()
        fb_ms.append(fb)
        opt_ms.append(om)
        times_ms.append(fb + om)

    peak_mb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    med_ms = statistics.median(times_ms)
    return {
        "params": n_params,
        "median_ms": med_ms,
        "fwd_bwd_ms": statistics.median(fb_ms),
        "opt_ms": statistics.median(opt_ms),
        "mean_ms": statistics.mean(times_ms),
        "atoms_per_sec": batch_data["num_atoms"] / (med_ms / 1000.0),
        "peak_vram_mb": peak_mb,
    }


def _build_v2_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    cuda_graph: bool,
) -> torch.optim.Optimizer:
    name = optimizer_name.lower()
    if name == "physmuon":
        preset = interatomic_potential_gated(total_steps=1000)
        preset["iso"] = False  # single-batch micro-benchmark (no micro-batching)
        return PhysMuon(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
            named_parameters=model.named_parameters(),
            cuda_graph=cuda_graph,
            **preset,
        )
    if name == "muon":
        return Muon(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
            named_parameters=model.named_parameters(),
            cuda_graph=cuda_graph,
        )
    if name == "hybridmuon":
        return HybridMuon(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
            named_parameters=model.named_parameters(),
            cuda_graph=cuda_graph,
        )
    raise ValueError(f"Unknown optimizer {optimizer_name!r}")


def benchmark_v2_for_family(
    family: str,
    batches_by_atoms: dict[int, dict[str, Any]],
    r_cut: float,
    warmup: int,
    steps: int,
    device: torch.device,
    use_compile: bool = True,
    edge_amp: bool = True,
    edge_checkpoint: bool = True,
    optimizer_name: str = "physmuon",
    opt_cuda_graph: bool = True,
    bucketed_reduce_overhead: bool = False,
) -> dict[int, dict[str, Any]]:
    """Benchmark v2 across all supercell sizes (dynamic=True or capped-bucket reduce-overhead)."""
    torch._dynamo.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = build_v2_model(
        family, r_cut, device, edge_amp=edge_amp, edge_checkpoint=edge_checkpoint
    ).train()
    if model.has_spin:
        model.set_spin_gate(0.5)
    n_params = _count_params(model)
    if use_compile:
        if bucketed_reduce_overhead:
            a_buckets = tuple(sorted(batches_by_atoms.keys()))[:4]
            e_buckets = tuple(batches_by_atoms[a]["num_edges"] for a in a_buckets)
            model.compile_bucketed(
                atom_buckets=a_buckets,
                edge_buckets=e_buckets,
                mode="reduce-overhead",
            )
        else:
            model.compile_fullgraph(dynamic=True)
    opt = _build_v2_optimizer(
        model, optimizer_name=optimizer_name, cuda_graph=opt_cuda_graph
    )

    per_size: dict[int, dict[str, Any]] = {}
    for n_atoms, batch_data in batches_by_atoms.items():
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        def _step(bdata: dict[str, Any] = batch_data) -> tuple[float, float]:
            opt.zero_grad(set_to_none=True)
            pos = bdata["pos"].detach().clone().requires_grad_(True)
            spin = (
                bdata["spin"].detach().clone().requires_grad_(True)
                if model.has_spin
                else None
            )
            t0 = time.perf_counter()
            out = model(
                pos=pos,
                atom_types=bdata["atom_types"],
                edge_index=bdata["edge_index"],
                edge_cell_shift=bdata["edge_cell_shift"],
                cell=bdata["cell"],
                batch=bdata["batch"],
                spin=spin,
                num_graphs=1,
                compute_forces=True,
                compute_virial=True,
                compute_spin_forces=model.has_spin,
                create_graph=True,
            )
            loss = (
                out["total_energy"].square().mean()
                + out["forces"].square().mean()
                + out["virial"].square().mean()
            )
            if "spin_forces" in out:
                loss = loss + out["spin_forces"].square().mean()
            if "born_effective_charges" in out:
                loss = (
                    loss
                    + out["born_effective_charges"].square().mean()
                    + out["total_born_effective_charges"].square().mean()
                )
            if "f_Uepi" in out:
                loss = (
                    loss
                    + out["f_Uepi"].square().mean()
                    + out["f_nu"].square().mean()
                    + out["f_kappa"].square().mean()
                )
            if "sf_Uepi" in out:
                loss = (
                    loss
                    + out["sf_Uepi"].square().mean()
                    + out["sf_nu"].square().mean()
                    + out["sf_kappa"].square().mean()
                )
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

        for _ in range(warmup):
            _step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        times_ms = []
        fb_ms = []
        opt_ms = []
        for _ in range(steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            fb, om = _step()
            fb_ms.append(fb)
            opt_ms.append(om)
            times_ms.append(fb + om)

        peak_mb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )
        reserved_mb = (
            float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )
        med_ms = statistics.median(times_ms)
        per_size[n_atoms] = {
            "params": n_params,
            "median_ms": med_ms,
            "fwd_bwd_ms": statistics.median(fb_ms),
            "opt_ms": statistics.median(opt_ms),
            "mean_ms": statistics.mean(times_ms),
            "atoms_per_sec": batch_data["num_atoms"] / (med_ms / 1000.0),
            "peak_vram_mb": peak_mb,
            "peak_reserved_mb": reserved_mb,
        }

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return per_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atoms",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="Supercell atom counts to benchmark.",
    )
    parser.add_argument(
        "--families",
        type=str,
        nargs="+",
        default=["nosoc", "soc", "born", "e2slp"],
        choices=["nosoc", "soc", "born", "e2slp"],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="physmuon",
        choices=["physmuon", "muon", "hybridmuon"],
    )
    parser.add_argument("--no-opt-cuda-graph", action="store_true")
    parser.add_argument(
        "--include-bucketed",
        action="store_true",
        help="Also benchmark opt-in capped-bucket (<=4) reduce-overhead mode separately.",
    )
    parser.add_argument("--r-cut", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device: {gpu_name} | PyTorch: {torch.__version__}", flush=True)
    print(
        f"v2 flags: compile={not args.no_compile}, edge_amp={not args.no_amp}, "
        f"edge_checkpoint={not args.no_checkpoint}, optimizer={args.optimizer}, "
        f"opt_cuda_graph={not args.no_opt_cuda_graph}, "
        f"include_bucketed={args.include_bucketed}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    header = (
        f"{'Family':<8} {'Atoms':>6} {'Edges':>7} | "
        f"{'v1 ms':>8} {'v2 dyn':>8} {'(fb+opt)':>13} {'Spd(dyn)':>8} | "
        f"{'v1 MB':>8} {'v2 MB':>8} {'VRAM Red':>9}"
    )
    if args.include_bucketed:
        header += f" | {'v2 bkt':>8} {'(fb+opt)':>13} {'Spd(bkt)':>8} {'BktResMB':>9}"
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)

    batches_by_atoms = {
        n_atoms: make_supercell_batch(
            num_atoms=n_atoms, r_cut=args.r_cut, device=device
        )
        for n_atoms in args.atoms
    }

    for family in args.families:
        v1_by_atoms = {
            n_atoms: benchmark_v1(
                family=family,
                batch_data=batches_by_atoms[n_atoms],
                r_cut=args.r_cut,
                warmup=args.warmup,
                steps=args.steps,
                device=device,
            )
            for n_atoms in args.atoms
        }
        v2_dyn_by_atoms = benchmark_v2_for_family(
            family=family,
            batches_by_atoms=batches_by_atoms,
            r_cut=args.r_cut,
            warmup=args.warmup,
            steps=args.steps,
            device=device,
            use_compile=not args.no_compile,
            edge_amp=not args.no_amp,
            edge_checkpoint=not args.no_checkpoint,
            optimizer_name=args.optimizer,
            opt_cuda_graph=not args.no_opt_cuda_graph,
            bucketed_reduce_overhead=False,
        )
        v2_bkt_by_atoms: dict[int, dict[str, Any]] | None = None
        if args.include_bucketed and not args.no_compile:
            v2_bkt_by_atoms = benchmark_v2_for_family(
                family=family,
                batches_by_atoms=batches_by_atoms,
                r_cut=args.r_cut,
                warmup=args.warmup,
                steps=args.steps,
                device=device,
                use_compile=True,
                edge_amp=not args.no_amp,
                edge_checkpoint=not args.no_checkpoint,
                optimizer_name=args.optimizer,
                opt_cuda_graph=not args.no_opt_cuda_graph,
                bucketed_reduce_overhead=True,
            )

        for n_atoms in args.atoms:
            batch_data = batches_by_atoms[n_atoms]
            v1_res = v1_by_atoms[n_atoms]
            v2_res = v2_dyn_by_atoms[n_atoms]
            speedup = v1_res["median_ms"] / max(v2_res["median_ms"], 1e-9)
            vram_red = (
                100.0
                * (1.0 - v2_res["peak_vram_mb"] / max(v1_res["peak_vram_mb"], 1e-9))
                if v1_res["peak_vram_mb"] > 0
                else 0.0
            )
            fb_opt_str = f"{v2_res['fwd_bwd_ms']:.1f}+{v2_res['opt_ms']:.2f}"
            row = (
                f"{family:<8} {n_atoms:>6} {batch_data['num_edges']:>7} | "
                f"{v1_res['median_ms']:>8.2f} {v2_res['median_ms']:>8.2f} {fb_opt_str:>13} {speedup:>7.2f}x | "
                f"{v1_res['peak_vram_mb']:>8.1f} {v2_res['peak_vram_mb']:>8.1f} {vram_red:>8.1f}%"
            )
            entry: dict[str, Any] = {
                "family": family,
                "num_atoms": n_atoms,
                "num_edges": batch_data["num_edges"],
                "avg_num_neighbors": batch_data["avg_num_neighbors"],
                "v1": v1_res,
                "v2_dynamic": v2_res,
                "speedup_dynamic": speedup,
                "vram_reduction_pct": vram_red,
            }
            if v2_bkt_by_atoms is not None and n_atoms in v2_bkt_by_atoms:
                bkt_res = v2_bkt_by_atoms[n_atoms]
                bkt_spd = v1_res["median_ms"] / max(bkt_res["median_ms"], 1e-9)
                bkt_fb_opt = f"{bkt_res['fwd_bwd_ms']:.1f}+{bkt_res['opt_ms']:.2f}"
                row += (
                    f" | {bkt_res['median_ms']:>8.2f} {bkt_fb_opt:>13} "
                    f"{bkt_spd:>7.2f}x {bkt_res['peak_reserved_mb']:>8.1f}"
                )
                entry["v2_bucketed"] = bkt_res
                entry["speedup_bucketed"] = bkt_spd
            print(row, flush=True)
            results.append(entry)

    payload = {
        "device": gpu_name,
        "torch_version": torch.__version__,
        "config": {
            "r_cut": args.r_cut,
            "warmup": args.warmup,
            "steps": args.steps,
            "compile": not args.no_compile,
            "edge_amp": not args.no_amp,
            "edge_checkpoint": not args.no_checkpoint,
            "optimizer": args.optimizer,
            "opt_cuda_graph": not args.no_opt_cuda_graph,
            "include_bucketed": args.include_bucketed,
        },
        "results": results,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote benchmark JSON to {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
