"""Run matched 1/2/4/8-GPU LAMMPS weak scaling inside one Slurm allocation."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess


def measure(args, ranks):
    side = args.cells
    atoms = ranks * side**3
    script = f"""units metal
atom_style spin
boundary p p p
newton on
processors {ranks} 1 1
lattice sc {args.spacing:.17g}
region box block 0 {side*ranks} 0 {side} 0 {side}
create_box 1 box
create_atoms 1 box
mass 1 55.845
set group all spin/atom 1.0 1 0 0
variable sx atom cos(0.163*(id%19))
variable sy atom sin(0.163*(id%19))
set group all spin/atom 1.0 v_sx v_sy 0.0
pair_style spingnn2 precision float32
pair_coeff * * "{args.model.resolve()}" "{args.artifact.resolve()}" 0
neighbor 0.4 bin
neigh_modify every 1 delay 0 check yes
velocity all set 0.01 -0.02 0.03
fix integrate all nve/kk
thermo {args.steps}
thermo_style custom step atoms pe ke etotal
thermo_modify norm no format float %.17g
timestep 0.00002
run {args.warmup}
run {args.steps} pre no
"""
    run_dir = args.output / f"ranks-{ranks}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "in.scaling").write_text(script)
    cmd = [
        args.mpirun,
        "-n",
        str(ranks),
        str(Path(__file__).resolve().parents[1] / "tools/lammps/spingnn2_gpu_rank.py"),
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
        "in.scaling",
        "-log",
        "none",
    ]
    log = run_dir / "stdout.log"
    with log.open("w") as output:
        subprocess.run(
            cmd, cwd=run_dir, stdout=output, stderr=subprocess.STDOUT, check=True
        )
    ids = re.findall(
        r"^SPINGNN2_GPU .* uuid=(GPU-[a-f0-9-]+)$", log.read_text(), re.MULTILINE
    )
    if len(ids) != ranks or len(set(ids)) != ranks:
        raise RuntimeError(f"GPU ownership mismatch in {log}: {ids}")
    loops = re.findall(
        r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps with (\d+) atoms",
        log.read_text(),
    )
    if len(loops) != 2 or tuple(map(int, loops[-1][1:])) != (args.steps, atoms):
        raise RuntimeError(f"Incomplete LAMMPS benchmark: {log}")
    thermos = re.findall(
        r"^\s*\d+\s+(\d+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s*$",
        log.read_text(),
        re.MULTILINE,
    )
    energies = [float(row[-1]) for row in thermos if int(row[0]) == atoms]
    if len(energies) < 4 or not all(math.isfinite(e) for e in energies):
        raise RuntimeError(f"Missing or nonfinite thermodynamic output: {log}")
    return dict(
        ranks=ranks,
        gpu_uuids=ids,
        atoms=atoms,
        spacing_angstrom=args.spacing,
        energy_drift_per_atom=(max(energies) - min(energies)) / atoms,
        steps=args.steps,
        seconds=float(loops[-1][0]),
        log=str(log),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "artifact", "binary", "output"):
        p.add_argument(f"--{name}", required=True, type=Path)
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--cells", type=int, default=64)
    p.add_argument(
        "--spacing",
        type=float,
        default=6.0 / 2.42,
        help="Simple cubic lattice spacing in angstroms",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--mpirun", default="mpirun")
    args = p.parse_args()
    if args.ranks != sorted(set(args.ranks)) or args.ranks[0] != 1:
        p.error("ranks must be unique, increasing, and start with one")
    if min(args.steps, args.warmup, args.cells) < 1 or not args.spacing > 0:
        p.error("steps, warmup and cells must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for ranks in args.ranks:
        row = measure(args, ranks)
        results.append(row)
        row["weak_efficiency"] = results[0]["seconds"] / row["seconds"]
        print(json.dumps(row), flush=True)
        payload = dict(slurm_job=os.getenv("SLURM_JOB_ID"), results=results)
        tmp = args.output / "weak-scaling.tmp"
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(args.output / "weak-scaling.json")


if __name__ == "__main__":
    main()
