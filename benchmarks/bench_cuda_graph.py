"""Measure single-GPU LAMMPS CUDA Graph replay against ordinary launches."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import subprocess

import numpy as np

from bench_native_rtx5090 import GPU


def measure(args, model, enabled, trial, spec, device):
    side = args.cells
    atoms = side**3
    contract = Path(spec["contract"]).resolve()
    artifact = Path(spec["manifest"]).resolve()
    layers = json.loads(contract.read_text())["model"]["layers"]
    cutoff = max(
        1.0 / layer["basis"]["cutoff"]["factor"]
        for layer in layers
        if "cutoff" in layer.get("basis", {})
    )
    spin = (
        """set group all spin/atom 1 1 0 0
variable sx atom cos(0.163*(id%19))
variable sy atom sin(0.163*(id%19))
set group all spin/atom 1 v_sx v_sy 0
"""
        if model != "born"
        else ""
    )
    rebuild = (
        f"every {args.rebuild_every} check no"
        if args.rebuild_every
        else "every 1 check yes"
    )
    script = f"""units metal
atom_style {"bec" if model == "born" else "spin"}
boundary p p p
newton on
lattice sc {cutoff / 2.42:.17g}
region box block 0 {side} 0 {side} 0 {side}
create_box 1 box
create_atoms 1 box
mass 1 55.845
{spin}pair_style spingnn2 precision float32 math ieee cuda_graph {enabled}
pair_coeff * * "{contract}" "{artifact}" 0
neighbor {args.skin} bin
neigh_modify {rebuild} delay 0
velocity all set 0.01 -0.02 0.03
fix integrate all nve/kk
compute pair_time all spingnn2/timing
compute graph all spingnn2/timing graph
thermo {args.steps}
thermo_style custom step atoms pe c_pair_time[1] c_graph[*]
thermo_modify norm no format float %.17g
timestep 0.000005
run {args.warmup}
run {args.steps} pre no
write_dump all custom last.dump id fx fy fz modify sort id format float %.17g
"""
    run_dir = args.output / f"{model}-{trial}-{enabled}"
    run_dir.mkdir(parents=True)
    (run_dir / "in.md").write_text(script)
    launcher = Path(__file__).resolve().parents[1] / "tools/lammps/spingnn2_gpu_rank.py"
    cmd = [
        str(launcher),
        str(args.binary.resolve()),
        "-k",
        "on",
        "g",
        "1",
        "-sf",
        "kk",
        "-pk",
        "kokkos",
        "gpu/aware",
        "on",
        "neigh",
        "half",
        "-in",
        "in.md",
        "-log",
        "none",
    ]
    log = run_dir / "stdout.log"
    memory = {}
    with log.open("w") as output, device.sample_peak(memory):
        subprocess.run(
            cmd, cwd=run_dir, stdout=output, stderr=subprocess.STDOUT, check=True
        )
    if "sampling_error" in memory:
        raise RuntimeError(memory["sampling_error"])
    stdout = log.read_text()
    (gpu,) = re.findall(
        r"^SPINGNN2_GPU .* uuid=(GPU-[a-f0-9-]+)$", stdout, re.MULTILINE
    )
    loops = re.findall(
        r"Loop time of ([\d.eE+-]+) on 1 procs for (\d+) steps with (\d+) atoms",
        stdout,
    )
    if len(loops) != 2 or tuple(map(int, loops[-1][1:])) != (args.steps, atoms):
        raise RuntimeError(f"Incomplete timing in {log}")
    rebuilds = int(re.findall(r"Neighbor list builds =\s*(\d+)", stdout)[-1])
    if bool(rebuilds) != bool(args.rebuild_every):
        raise RuntimeError(f"Unexpected neighbor rebuild count in {log}: {rebuilds}")
    rows = {}
    for line in stdout.splitlines():
        v = line.split()
        if len(v) == 7 and v[0].isdigit() and v[1] == str(atoms):
            rows[int(v[0])] = list(map(float, v[2:]))
    start, end = rows[args.warmup], rows[args.warmup + args.steps]
    if not all(math.isfinite(value) for value in start + end):
        raise RuntimeError(f"Nonfinite LAMMPS output in {log}")
    captures, replays, fallback = [int(value) for value in end[2:]]
    if enabled == "yes" and (replays <= start[3] or fallback or captures <= rebuilds):
        raise RuntimeError(f"Graph replay unavailable in {log}")
    if enabled == "no" and any((captures, replays, fallback)):
        raise RuntimeError(
            f"Graph disabled but execution statistics are nonzero in {log}"
        )
    return dict(
        graph=enabled,
        model=model,
        atoms=atoms,
        gpu=gpu,
        rebuilds=rebuilds,
        captures=captures,
        replays=replays,
        fallbacks=fallback,
        wall_s=float(loops[-1][0]),
        warmup_s=float(loops[0][0]),
        pair_s=end[1] - start[1],
        energy_per_atom=end[0] / atoms,
        **memory,
        log=str(log),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model",
        choices=("nosoc", "soc", "born"),
        nargs="+",
        default=["nosoc", "soc", "born"],
    )
    p.add_argument("--cells", type=int, default=16)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--skin", type=float, default=0.4)
    p.add_argument("--rebuild-every", type=int, default=0)
    p.add_argument("--cudart", default="libcudart.so.12")
    args = p.parse_args()
    if not args.skin > 0:
        p.error("Neighbor skin must be positive")
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Run inside a single-GPU Slurm allocation")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    models = json.loads(args.models.read_text())
    device = GPU(args.cudart)
    payload = dict(
        slurm_job=os.environ["SLURM_JOB_ID"],
        lammps_sha256=sha256(args.binary.read_bytes()).hexdigest(),
        warmup_steps=args.warmup,
        measured_steps=args.steps,
        models={},
    )
    for model in args.model:
        samples = [
            measure(args, model, graph, trial, models[model], device)
            for trial, graph in enumerate(("no", "yes", "yes", "no"))
        ]
        if len({sample["gpu"] for sample in samples}) != 1:
            raise RuntimeError("A/B trials did not use the same GPU")
        energies = [sample["energy_per_atom"] for sample in samples]
        if max(energies) - min(energies) >= 5e-5:
            raise RuntimeError("A/B energy parity failed")
        force = [
            np.loadtxt(Path(sample["log"]).with_name("last.dump"), skiprows=9)
            for sample in samples
        ]
        force_error = max(float(np.max(np.abs(value - force[0]))) for value in force)
        if not force_error < 5e-5:
            raise RuntimeError(f"A/B final force parity failed: {force_error:.6g}")

        def ratio(key):
            return median(s[key] for s in samples if s["graph"] == "no") / median(
                s[key] for s in samples if s["graph"] == "yes"
            )

        payload["models"][model] = dict(
            artifact=models[model],
            wall_speedup=ratio("wall_s"),
            pair_speedup=ratio("pair_s"),
            max_force_error=force_error,
            peak_delta_bytes=median(
                s["peak_increase_bytes"] for s in samples if s["graph"] == "yes"
            )
            - median(s["peak_increase_bytes"] for s in samples if s["graph"] == "no"),
            samples=samples,
        )
        print(json.dumps(payload["models"][model]), flush=True)
        (args.output / "report.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
