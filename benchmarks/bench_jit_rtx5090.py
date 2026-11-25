"""Matched M3 and M2 execution against eager using the M2 graph and weight digests."""

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np

from bench_native_rtx5090 import (
    GPU,
    MODELS,
    SHAPES,
    benchmark as eager_benchmark,
    configuration,
    graph,
    lattice_offsets,
    measure,
    verify_results,
)

GPU_IDS = {"sm89": "0x2684", "sm120": "0x2b85"}
PROFILE_STAGES = (
    "geometry_seconds",
    "mlp_seconds",
    "tp_linear_seconds",
    "reverse_seconds",
)


def native_row(args, config, record):
    from spingnn2.native import NativeModel

    data = graph(
        args.atoms,
        config["r_max"],
        len(config["chemical_symbol_to_type"]),
        args.model != "born",
        args.neighbor_radius,
    )
    gpu = GPU(args.cudart)
    if gpu.device_id != GPU_IDS[args.expected_gpu]:
        raise ValueError(
            f"Expected {args.expected_gpu}, got PCI device {gpu.device_id}"
        )
    record.update(device_id=gpu.device_id, device_pci=gpu.pci)
    mode = args.method if args.method in ("generic", "direct") else "direct"
    kwargs = dict(
        backend="cuda",
        precision="float32",
        execution_mode=mode,
        tile_atoms=args.tile_atoms,
        execution_profile=args.execution_profile,
        radial_mode="spline",
        library=args.library,
        artifact=args.artifact if mode == "direct" else None,
        allow_tf32=args.allow_tf32,
    )

    with gpu.sample_peak(record):
        with NativeModel.load(args.contract, **kwargs) as model:
            if args.method == "cache":
                if args.model == "born":
                    raise ValueError("A magnetic model is required for SIB")
                record["cache_kind"] = args.cache_kind
                start = perf_counter()
                cache = model.evaluate_stage1_lattice(
                    data, spin_only=args.cache_kind == "lean-frozen"
                )
                record["cache_create_seconds"] = perf_counter() - start
                update = dict(data, spin=np.roll(data["spin"], 1, axis=0))
                with cache:
                    ref = model.evaluate(update)
                    full = measure(
                        lambda: model.evaluate(update), args.warmups, args.repetitions
                    )
                    modes = (("cached", False), ("frozen", True))
                    if args.cache_kind == "moving":
                        modes = modes[:1]
                    if args.cache_kind in ("frozen", "lean-frozen"):
                        modes = modes[1:]
                    for name, flag in modes:

                        def evaluate():
                            return model.evaluate_stage2_spin_cached(
                                cache, update["spin"], spin_only=flag
                            )

                        samples = measure(evaluate, args.warmups, args.repetitions)
                        check = evaluate()
                        if flag and set(check) & {
                            "atom_virial",
                            "virial",
                            "stress",
                            "forces",
                        }:
                            raise ValueError(
                                "Frozen evaluation returned lattice derivatives"
                            )
                        for key in ("total_energy", "spin_forces", "spin_forces_vert"):
                            np.testing.assert_allclose(
                                check[key], ref[key], atol=5e-5, rtol=1e-6
                            )
                        record[name + "_seconds"] = samples
                        record[name + "_us_per_atom"] = (
                            median(samples) * 1e6 / args.atoms
                        )
                        record[name + "_speedup"] = median(full) / median(samples)
                        record[name + "_cache"] = model.diagnostics.copy()
                    record["full_seconds"] = full
            else:

                def evaluate():
                    return model.evaluate(data)

                if args.warmups == 0 and args.repetitions == 1:
                    start = perf_counter()
                    check = evaluate()
                    samples = [perf_counter() - start]
                else:
                    samples = measure(evaluate, args.warmups, args.repetitions)
                    check = evaluate()
                verify_results(check, args.atoms, args.model != "born")
                record["elapsed_seconds"] = samples
                record["us_per_atom"] = median(samples) * 1e6 / args.atoms
                if args.trace_hot_call:
                    gpu.check(gpu.lib.cudaProfilerStart())
                    try:
                        start = perf_counter()
                        evaluate()
                        record["trace_wall_seconds"] = perf_counter() - start
                    finally:
                        gpu.check(gpu.lib.cudaProfilerStop())
                if args.profile:
                    model.profile = True
                    start = perf_counter()
                    evaluate()
                    record["profile_wall_seconds"] = perf_counter() - start
                    record["profile"] = model.diagnostics.copy()
                    record["profile_unclassified_seconds"] = record[
                        "profile_wall_seconds"
                    ] - sum(record["profile"][stage] for stage in PROFILE_STAGES)
            record["workspace"] = model.diagnostics.copy()
            record["plan"] = model.execution_plan
            if record["plan"].get("math_mode") != record["math_mode"]:
                raise ValueError(
                    "The native execution plan reports a different math mode"
                )
            if (
                record["plan"]["selected_mode"] != mode
                or record["plan"]["fallback_count"]
            ):
                raise ValueError("Unexpected execution plan")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument(
        "--method", choices=("generic", "direct", "eager", "cache"), required=True
    )
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--artifact", type=Path)
    p.add_argument(
        "--cache-kind",
        choices=("both", "moving", "frozen", "lean-frozen"),
        default="both",
    )
    p.add_argument("--library", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--cudart", default="libcudart.so.12")
    p.add_argument("--atoms", type=int, choices=SHAPES, default=4096)
    p.add_argument("--tile-atoms", type=int, default=256)
    p.add_argument(
        "--execution-profile", choices=("speed", "capacity"), default="speed"
    )
    p.add_argument("--neighbor-radius", type=float, default=2.42)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--profile", action="store_true")
    p.add_argument(
        "--trace-hot-call",
        action="store_true",
        help="Capture one additional hot native evaluation with Nsight Systems",
    )
    p.add_argument("--expected-gpu", choices=GPU_IDS, default="sm120")
    p.add_argument(
        "--allow-tf32", action="store_true", help="Opt in to TF32 tensor-core GEMMs"
    )
    args = p.parse_args(argv)
    if args.tile_atoms < 1 or args.warmups < 0 or args.repetitions < 1:
        p.error("Invalid sample or tile count")
    if args.method in ("direct", "cache") and not args.artifact:
        p.error("Direct execution requires --artifact")
    if args.trace_hot_call and args.method not in ("generic", "direct"):
        p.error("--trace-hot-call requires generic or direct native execution")
    degree = len(lattice_offsets(args.neighbor_radius))
    cfg = configuration(args.model, degree)
    record = dict(
        method=args.method,
        model=args.model,
        atoms=args.atoms,
        edges=args.atoms * degree,
        tile_atoms=args.tile_atoms,
        precision="float32",
        r_max=cfg["r_max"],
        warmups=args.warmups,
        repetitions=args.repetitions,
        expected_gpu=args.expected_gpu,
        math_mode="tf32" if args.allow_tf32 else "strict_fp32",
    )
    try:
        metadata = json.loads(args.contract.with_suffix(".benchmark.json").read_text())
        if json.loads(args.contract.read_text())["sha256"] != metadata["binary_sha256"]:
            raise ValueError("Benchmark weights disagree with the export record")
        record.update(metadata)
        if args.method == "eager":
            args.eager_copyback = True
            if GPU(args.cudart).device_id != GPU_IDS[args.expected_gpu]:
                raise ValueError(f"Expected {args.expected_gpu} for eager comparison")
            args.allow_other_gpu = args.expected_gpu != "sm120"
            eager_benchmark(args, cfg, record)
        else:
            native_row(args, cfg, record)
        record["status"] = "passed"
    except (ValueError, RuntimeError, MemoryError, OSError) as exc:
        record.update(status="failed", error=str(exc))
    print("M3_BENCH " + json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return int(record["status"] != "passed")


if __name__ == "__main__":
    sys.exit(main())
