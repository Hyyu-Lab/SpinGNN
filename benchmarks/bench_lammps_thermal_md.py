"""Measure 300 K RTX MD with displacement-triggered rebuilding and rank migration."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEMPERATURE = 300
MODELS = ("nosoc", "soc", "born")


def digest(path):
    with path.open("rb") as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def md_input(
    spec,
    model,
    ranks,
    cells,
    skin,
    mode,
    equilibration,
    warmup,
    steps,
    *,
    snapshot=None,
    replicas=1,
    write_snapshot=False,
):
    contract = Path(spec["contract"]).resolve()
    artifact = Path(spec["manifest"]).resolve()
    layers = json.loads(contract.read_text())["model"]["layers"]
    cutoff = max(
        1.0 / layer["basis"]["cutoff"]["factor"]
        for layer in layers
        if "cutoff" in layer.get("basis", {})
    )
    spacing = cutoff / 2.42
    spin = (
        """set group all spin/atom 1 1 0 0
variable sx atom cos(0.163*(id%19))
variable sy atom sin(0.163*(id%19))
set group all spin/atom 1 v_sx v_sy 0
"""
        if model != "born"
        else ""
    )
    timestep = 0.001 if mode == "nve" else 0.0005
    if mode == "nve":
        integrator = "nve/kk"
    else:
        alpha = 0.05 if mode == "sib" else 0.0
        integrator = f"spin/sib/kk {TEMPERATURE} {alpha} 91081 lattice moving"

    # Replicate an equilibrated cell to give every binary, skin and rank
    # count the same starting positions, spins and thermal velocity sample.
    if snapshot is None:
        geometry = f"""lattice sc {spacing:.17g} origin 0.0045 0 0
region box block 0 {cells * ranks} 0 {cells} 0 {cells}
create_box 1 box
create_atoms 1 box
mass 1 55.845
{spin}"""
        velocity = f"velocity all create {TEMPERATURE} 92317 mom yes rot no dist gaussian loop all"
    else:
        geometry = f'read_data "{snapshot}"\nreplicate {replicas * ranks} {replicas} {replicas}\n'
        velocity = f"velocity all scale {TEMPERATURE}"
    script = f"""units metal
atom_style {"bec" if model == "born" else "spin"}
atom_modify map array
boundary p p p
newton on
processors {ranks} 1 1
{geometry}pair_style spingnn2/kk precision float32 math ieee
pair_coeff * * "{contract}" "{artifact}" 0
neighbor {skin} bin
neigh_modify every 1 delay 0 check yes
{velocity}
fix integrate all {integrator}
compute timing all spingnn2/timing
thermo 8
thermo_style custom step atoms temp pe ke etotal c_timing[1] c_timing[5] c_timing[6]
thermo_modify norm no format float %.17g
timestep {timestep}
"""
    if equilibration:
        # A separate small-cell run amortizes equilibration over the A/B matrix.
        script += f"""fix thermalize all langevin {TEMPERATURE} {TEMPERATURE} 0.01 49721 zero yes
run {equilibration}
unfix thermalize
velocity all scale {TEMPERATURE}
"""
    if write_snapshot:
        return script + "write_data equilibrated.data nocoeff\n", timestep
    script += f"""print "SPINGNN2_THERMAL_PRODUCTION"
run {warmup}
write_dump all custom owners-before.dump id proc modify sort id
run {steps} pre no
write_dump all custom owners-after.dump id proc modify sort id
"""
    return script, timestep


def read_owners(path, atoms, ranks, step):
    with path.open() as stream:
        lines = [stream.readline().strip() for _ in range(9)]
        if lines[:4] != [
            "ITEM: TIMESTEP",
            str(step),
            "ITEM: NUMBER OF ATOMS",
            str(atoms),
        ]:
            raise RuntimeError(f"Unexpected timestep or atom count in {path}")
        if lines[-1] != "ITEM: ATOMS id proc":
            raise RuntimeError(f"Unexpected ownership columns in {path}")
        owners = np.loadtxt(stream, dtype=np.int64, ndmin=2)
    if (
        owners.shape != (atoms, 2)
        or not np.array_equal(owners[:, 0], np.arange(1, atoms + 1))
        or np.min(owners[:, 1]) < 0
        or np.max(owners[:, 1]) >= ranks
    ):
        raise RuntimeError(f"Invalid atom IDs or ranks in {path}")
    return owners[:, 1]


def read_md(
    output, directory, atoms, ranks, equilibration, warmup, steps, timestep, mode
):
    _, marker, output = output.partition("\nSPINGNN2_THERMAL_PRODUCTION\n")
    if not marker:
        raise RuntimeError(f"Missing production marker in {directory}")
    loops = re.findall(
        r"Loop time of ([\d.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms",
        output,
    )
    if [tuple(map(int, row[1:])) for row in loops] != [
        (ranks, warmup, atoms),
        (ranks, steps, atoms),
    ]:
        raise RuntimeError(f"Incomplete LAMMPS loops in {directory}")
    wall = float(loops[-1][0])
    if not math.isfinite(wall) or wall <= 0:
        raise RuntimeError(f"Invalid LAMMPS timing in {directory}")
    builds = [
        int(count) for count in re.findall(r"Neighbor list builds = (\d+)", output)
    ]
    if len(builds) != 2:
        raise RuntimeError(f"Missing neighbor rebuild counters in {directory}")
    # The production run uses pre no, so the neighbor counter is cumulative.
    rebuilds = builds[1] - builds[0]
    if rebuilds <= 0:
        raise RuntimeError(f"No displacement-triggered rebuilds in {directory}")
    dangerous = [
        int(count) for count in re.findall(r"Dangerous builds = (\d+)", output)
    ]
    if len(dangerous) != 2 or any(dangerous):
        raise RuntimeError(
            f"Missing or unsafe neighbor rebuild counters in {directory}"
        )
    rows = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 9 and fields[0].isdigit() and fields[1] == str(atoms):
            rows[int(fields[0])] = list(map(float, fields[2:]))
    start = equilibration + warmup
    end = start + steps
    if (
        not {equilibration, start, end} <= rows.keys()
        or not np.isfinite(list(rows.values())).all()
    ):
        raise RuntimeError(f"Missing or nonfinite thermodynamic samples in {directory}")
    if not math.isclose(rows[equilibration][0], TEMPERATURE, rel_tol=1e-6):
        raise RuntimeError(f"Unexpected initial temperature in {directory}")
    measured = sorted(k for k in rows if k >= start)
    temperatures = [rows[k][0] for k in measured]
    energies = [rows[k][3] / atoms for k in measured]
    previous = read_owners(directory / "owners-before.dump", atoms, ranks, start)
    current = read_owners(directory / "owners-after.dump", atoms, ranks, end)
    moved = int(np.count_nonzero(previous != current))
    if ranks > 1 and not moved:
        raise RuntimeError(f"No atom changed rank in {directory}")
    cache_hits, pair_calls = [int(rows[end][i] - rows[start][i]) for i in (5, 6)]
    is_sib = mode != "nve"
    if (cache_hits, pair_calls) != (steps * is_sib, steps * (1 + is_sib)):
        raise RuntimeError(f"Unexpected pair or SIB cache counts in {directory}")
    candidates = re.findall(r"Ave neighs/atom = ([\d.eE+-]+)", output)
    if len(candidates) != 2:
        raise RuntimeError(f"Missing LAMMPS neighbor counts in {directory}")
    steps_per_second = steps / wall
    return dict(
        wall_seconds=wall,
        steps_per_second=steps_per_second,
        ns_per_day=steps_per_second * timestep * 86.4,
        wall_us_per_atom_step=wall * 1e6 / (atoms * steps),
        pair_wall_seconds=rows[end][4] - rows[start][4],
        neighbor_builds=rebuilds,
        warmup_neighbor_builds=builds[0],
        dangerous_builds=sum(dangerous),
        lammps_neighbors_per_atom=float(candidates[-1]),
        atoms_with_changed_rank=moved,
        stage1_cache_hits=cache_hits,
        pair_calls=pair_calls,
        initial_temperature_K=rows[equilibration][0],
        mean_timed_temperature_K=float(np.mean(temperatures)),
        min_timed_temperature_K=min(temperatures),
        max_timed_temperature_K=max(temperatures),
        energy_range_eV_per_atom=max(energies) - min(energies),
        energy_change_eV_per_atom=energies[-1] - energies[0],
        thermo_rows=[(k, *rows[k]) for k in measured],
    )


def launch(args, binary, work, script, ranks):
    work.mkdir(parents=True)
    (work / "in.md").write_text(script)
    log = work / "stdout.log"
    command = [
        str(ROOT / "tools/lammps/spingnn2_gpu_rank.py"),
        str(binary),
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
    if ranks > 1:
        command = [*shlex.split(args.mpirun), "-n", str(ranks), *command]
    with log.open("w") as output:
        result = subprocess.run(
            command, cwd=work, stdout=output, stderr=subprocess.STDOUT
        )
        if result.returncode:
            raise RuntimeError(f"LAMMPS exited {result.returncode}; see {log}")
    output = log.read_text()
    uuids = re.findall(r"^SPINGNN2_GPU .* uuid=(GPU-[a-f0-9-]+)$", output, re.MULTILINE)
    if len(uuids) != ranks or len(set(uuids)) != ranks:
        raise RuntimeError(f"Incorrect GPU assignment in {log}: {uuids}")
    return output, uuids, log


def prepare_snapshot(args, binary, spec, model, mode):
    work = args.output / f"{model}-{mode}-equilibration"
    cells = args.cells // 2 if args.cells >= 16 and args.cells % 2 == 0 else args.cells
    script, _ = md_input(
        spec, model, 1, cells, 0.4, mode, args.equilibrate, 0, 0, write_snapshot=True
    )
    _, _, log = launch(args, binary, work, script, 1)
    path = work / "equilibrated.data"
    return dict(
        path=str(path),
        sha256=digest(path),
        log=str(log),
        cells=cells,
        steps=args.equilibrate,
    )


def measure(args, arm, model, ranks, skin, mode, trial, binary, spec, equilibrium):
    steps = args.nve_steps if mode == "nve" else args.sib_steps
    script, timestep = md_input(
        spec,
        model,
        ranks,
        args.cells,
        skin,
        mode,
        0,
        args.warmup,
        steps,
        snapshot=equilibrium["path"],
        replicas=args.cells // equilibrium["cells"],
    )
    work = args.output / f"{model}-{mode}-skin-{skin:g}-{ranks}gpu-{arm}-{trial}"
    output, uuids, log = launch(args, binary, work, script, ranks)
    atoms = ranks * args.cells**3
    row = read_md(output, work, atoms, ranks, 0, args.warmup, steps, timestep, mode)
    return dict(
        arm=arm,
        trial=trial,
        model=model,
        mode=mode,
        ranks=ranks,
        skin=skin,
        atoms=atoms,
        atoms_per_rank=args.cells**3,
        timestep_ps=timestep,
        thermal_snapshot=equilibrium,
        warmup_steps=args.warmup,
        measured_steps=steps,
        gpu_uuids=uuids,
        model_contract=spec,
        logfile=str(log),
        **row,
    )


def load_models(path, models):
    index = json.loads(path.read_text())
    result = {}
    for model in models:
        spec = index[model]
        contract = Path(spec["contract"]).resolve()
        artifact = Path(spec["manifest"]).resolve()
        manifest = json.loads(artifact.read_text())
        actual = dict(
            contract_sha256=digest(contract),
            jit_manifest_sha256=digest(artifact),
            jit_image_sha256=digest(artifact.with_name(manifest["artifact"])),
        )
        for field, value in actual.items():
            if field in spec and spec[field] != value:
                raise RuntimeError(f"Input {field} does not match model index: {model}")
        if manifest["contract_sha256"] != actual["contract_sha256"]:
            raise RuntimeError(f"JIT artifact has the wrong model contract: {model}")
        result[model] = dict(contract=str(contract), manifest=str(artifact), **actual)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-index", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path)
    parser.add_argument("--candidate-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1])
    parser.add_argument("--skins", type=float, nargs="+", default=[0.4, 1.5])
    parser.add_argument(
        "--modes", nargs="+", choices=("nve", "sib", "sib-nve"), default=["nve", "sib"]
    )
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--equilibrate", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--nve-steps", type=int, default=192)
    parser.add_argument("--sib-steps", type=int, default=320)
    parser.add_argument("--blocks", type=int, choices=(1, 2), default=1)
    parser.add_argument("--mpirun", default=os.getenv("SPINGNN2_MPIRUN", "mpirun"))
    args = parser.parse_args(argv)
    if "SLURM_JOB_ID" not in os.environ:
        parser.error("Run inside a Slurm GPU allocation")
    if args.ranks != sorted(set(args.ranks)) or args.ranks[0] != 1:
        parser.error("Rank counts must be unique and increasing, starting at one")
    if (
        args.cells < 8
        or min(args.equilibrate, args.warmup, args.nve_steps, args.sib_steps) < 1
    ):
        parser.error(
            "Use at least eight cells in each direction and positive step counts"
        )
    if any(skin <= 0 for skin in args.skins):
        parser.error("Neighbor skins must be positive")
    if bool(args.candidate_binary) != bool(args.candidate_index):
        parser.error("A candidate binary and model index must be supplied together")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    binaries = {"control": args.binary.resolve()}
    models = {"control": load_models(args.model_index, args.models)}
    order = ["control"]
    if args.candidate_binary:
        binaries["candidate"] = args.candidate_binary.resolve()
        models["candidate"] = load_models(args.candidate_index, args.models)
        order.append("candidate")
        for model in args.models:
            left, right = [models[arm][model] for arm in order]
            if any(
                left[k] != right[k] for k in ("contract_sha256", "jit_image_sha256")
            ):
                raise RuntimeError(
                    f"Control and candidate used different inputs for {model}"
                )

    payload = dict(
        job=os.environ["SLURM_JOB_ID"],
        binaries={
            key: dict(path=str(path), sha256=digest(path))
            for key, path in binaries.items()
        },
        driver_sha256=digest(Path(__file__).resolve()),
        results=[],
    )
    for model in args.models:
        for mode in args.modes:
            if model == "born" and mode != "nve":
                continue
            preparation = order[-1]
            equilibrium = prepare_snapshot(
                args, binaries[preparation], models[preparation][model], model, mode
            )
            for skin in args.skins:
                for trial in range(args.blocks):
                    single = {}
                    owners = {}
                    for ranks in args.ranks:
                        for arm in order if trial == 0 else reversed(order):
                            row = measure(
                                args,
                                arm,
                                model,
                                ranks,
                                skin,
                                mode,
                                trial,
                                binaries[arm],
                                models[arm][model],
                                equilibrium,
                            )
                            rank_uuids = set(row["gpu_uuids"])
                            if owners.setdefault(ranks, rank_uuids) != rank_uuids:
                                raise RuntimeError("A/B arms used different GPUs")
                            if not owners.setdefault(1, rank_uuids) <= rank_uuids:
                                raise RuntimeError(
                                    "Scaled runs did not include the single GPU"
                                )
                            rate = row["steps_per_second"]
                            row["weak_efficiency"] = rate / single.setdefault(arm, rate)
                            payload["results"].append(row)
                            print(
                                "SPINGNN2_THERMAL_MD",
                                json.dumps(
                                    {k: v for k, v in row.items() if k != "thermo_rows"}
                                ),
                                flush=True,
                            )
                            pending = args.output / "summary.tmp"
                            pending.write_text(json.dumps(payload, indent=2) + "\n")
                            pending.replace(args.output / "summary.json")


if __name__ == "__main__":
    main()
