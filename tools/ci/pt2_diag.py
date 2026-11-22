#!/usr/bin/env python3
"""One-off diagnostic for the LAMMPS-PT2 vs Python-PT2 trajectory mismatch.

Evaluates one deployment pair on (a) the fixture input used by run-0 tests and
(b) the trajectory step-0 frame input, via TorchScript and Python AOTI, and
prints energies + edge/spin/pos summaries so the outlier input is identified.
Usage: pt2_diag.py <case_name> <scripted.pth> <compiled.pt2> <traj.dump>
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests_lammps"))

from helpers import (  # noqa: E402
    CASES,
    _input_from_atoms,
    atoms_from_fixture,
    frame_to_atoms,
    parse_trajectory_dump,
)


def main() -> None:
    case_name, pth, pt2, dump = sys.argv[1:5]
    case = CASES[case_name]
    device = "cuda"

    # Build the step-0 frame the same way run_trajectory_case does, without
    # rerunning LAMMPS: parse the dump + take thermo values from the LAMMPS
    # log's PotEng column (energy is unused by the reference evaluation).
    raw = parse_trajectory_dump(Path(dump))[0]
    atoms0 = raw["atoms"]
    from helpers import TrajectoryFrame  # noqa: E402
    import numpy as np  # noqa: E402

    def col(name):
        return np.asarray([row[name] for row in atoms0], dtype=float)

    spins_dir = np.stack(
        (col("c_spinprops[1]"), col("c_spinprops[2]"), col("c_spinprops[3]")), axis=1
    )
    spins_mag = col("c_spinprops[4]")
    frame = TrajectoryFrame(
        step=int(raw["step"]),
        ids=[int(row["id"]) for row in atoms0],
        types=[int(row["type"]) for row in atoms0],
        origin=np.asarray(raw["origin"], dtype=float),
        cell=np.asarray(raw["cell"], dtype=float),
        positions=np.stack((col("x"), col("y"), col("z")), axis=1),
        spin_directions=spins_dir,
        spin_magnitudes=spins_mag,
    )

    fixture_input = _input_from_atoms(case, atoms_from_fixture(case), device)
    frame_input = _input_from_atoms(case, frame_to_atoms(case, frame), device)
    print(
        f"INPUT edges fixture={fixture_input['edge_index'].shape[1]} "
        f"frame={frame_input['edge_index'].shape[1]}",
        flush=True,
    )
    if "spin" in fixture_input:
        print(
            f"INPUT spin[0] fixture={fixture_input['spin'][0].tolist()} "
            f"frame={frame_input['spin'][0].tolist()}",
            flush=True,
        )
    print(
        f"INPUT pos[0] fixture={fixture_input['pos'][0].tolist()} "
        f"frame={frame_input['pos'][0].tolist()}",
        flush=True,
    )
    print(
        f"INPUT pos[1] fixture={fixture_input['pos'][1].tolist()} "
        f"frame={frame_input['pos'][1].tolist()}",
        flush=True,
    )

    from nequip.scripts import deploy as deploy_mod  # noqa: E402
    from nequip.scripts.deploy import load_deployed_model  # noqa: E402

    model, _ = load_deployed_model(
        Path(pth), device=device, freeze=True, set_global_options=True
    )
    for label, data in (("fixture", fixture_input), ("frame0", frame_input)):
        out = model({k: v.clone() for k, v in data.items()})
        atomic = out["atomic_energy"].detach().cpu().numpy().reshape(-1)
        print(
            f"TS {label} energy={float(out['total_energy'][0, 0])} "
            f"sum_atomic={float(atomic.sum())} "
            f"forces0={out['forces'][0].detach().cpu().numpy().tolist()}",
            flush=True,
        )

    meta = deploy_mod.read_deploy_metadata_sidecar(Path(pt2))
    ins = tuple(meta[deploy_mod.INPUT_SIGNATURE_KEY].split())
    outs = tuple(meta[deploy_mod.OUTPUT_SIGNATURE_KEY].split())
    print(f"AOTI output_signature={outs}", flush=True)
    compiled = torch._inductor.aoti_load_package(pt2, run_single_threaded=True)
    from helpers import _run_pt2_with_ghost_graph  # noqa: E402

    for label, data in (("fixture", fixture_input), ("frame0", frame_input)):
        result = _run_pt2_with_ghost_graph(
            case, compiled, data, ins, outs, f"diag {label}"
        )
        atomic = result["atomic_energy"].detach().cpu().numpy().reshape(-1)
        print(
            f"AOTI-GHOST {label} energy={float(result['total_energy'][0, 0])} "
            f"sum_atomic={float(atomic.sum())} "
            f"forces0={result['forces'][0].detach().cpu().numpy().tolist()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
