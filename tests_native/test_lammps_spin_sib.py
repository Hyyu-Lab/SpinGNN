"""Trajectory, cache and domain-decomposition checks for the SIB integrators."""

import io
import json
import os
from pathlib import Path
import re

import numpy as np
import pytest

from spingnn2.reference import ReferenceModel
from test_lammps_pair import POS, TYPES, _cutoff


BOX_SIDE = 16.0


def _spin_data(path):
    data = [
        "SIB periodic spin-lattice fixture",
        "",
        f"{len(POS)} atoms",
        "2 atom types",
        "",
        *(f"0.0 {BOX_SIDE} {axis}lo {axis}hi" for axis in "xyz"),
        "",
        "Masses",
        "",
        "1 54.938",
        "2 72.630",
        "",
        "Atoms # spin",
        "",
    ]
    velocities = ["", "Velocities", ""]
    for tag, ((x, y, z), kind) in enumerate(zip(POS, TYPES), 1):
        direction = np.array([np.cos(tag), np.sin(tag), 0.3])
        sx, sy, sz = direction / np.linalg.norm(direction)
        atom_type = 1 + kind
        magnitude = 0.0 if tag == 1 else 0.7 + 0.1 * (tag % 5)
        data.append(f"{tag} {atom_type} {x} {y} {z} {sx} {sy} {sz} {magnitude}")
        vx, vy, vz = 0.03 * np.array([np.sin(tag), np.cos(tag), np.sin(2 * tag)])
        velocities.append(f"{tag} {vx} {vy} {vz}")
    path.write_text("\n".join([*data, *velocities, ""]))


def _trajectory(path):
    frames = []
    for block in path.read_text().split("ITEM: TIMESTEP\n")[1:]:
        _, data = block.split("ITEM: ATOMS ", maxsplit=1)
        frames.append(np.loadtxt(io.StringIO(data), skiprows=1))
    assert len(frames) == 2
    return np.stack(frames)


def _run_sib(
    lmp_run,
    pair,
    data,
    dump,
    integrator,
    *,
    fixes="",
    ranks=1,
    steps=24,
):
    script = f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data "{data}"
{pair}
neighbor 0.4 bin
neigh_modify every 1 delay 0 check yes
fix integrate all {integrator}
{fixes}
compute spin all property/atom spx spy spz sp fmx fmy fmz
compute timers all spingnn2/timing
timestep 0.000005
dump trajectory all custom {steps} "{dump}" id x y z vx vy vz fx fy fz &
  c_spin[1] c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7]
dump_modify trajectory sort id format float %.17g
thermo {steps}
thermo_style custom step pe ke etotal vol &
  c_timers[1] c_timers[2] c_timers[3] c_timers[4] c_timers[5] c_timers[6]
thermo_modify format float %.17g
run {steps}
variable ep equal etotal
variable vol equal vol
variable seconds equal c_timers[1]
variable stage1 equal c_timers[3]
variable hits equal c_timers[5]
variable evals equal c_timers[6]
print "SIB_RESULT ${{ep}} ${{vol}} ${{seconds}} ${{stage1}} ${{hits}} ${{evals}}"
"""
    output = lmp_run(script, ranks=ranks)
    line = next(line for line in output.splitlines() if line.startswith("SIB_RESULT "))
    stats = dict(
        zip(
            ("energy", "volume", "pair", "stage1", "hits", "calls"),
            (float(value) for value in line.split()[1:]),
        )
    )
    trajectory = _trajectory(dump)
    first_thermo = next(
        fields
        for line in output.splitlines()
        if len(fields := line.split()) == 11 and fields[0] == "0"
    )
    stats["energy_drift_per_atom"] = abs(stats["energy"] - float(first_thermo[3])) / (
        trajectory.shape[1]
    )
    assert np.isfinite(trajectory).all()
    assert np.isfinite(list(stats.values())).all()
    assert np.max(np.abs(np.linalg.norm(trajectory[:, :, 10:13], axis=-1) - 1)) < 1e-12
    np.testing.assert_array_equal(trajectory[0, :, 13], trajectory[1, :, 13])
    np.testing.assert_array_equal(trajectory[0, 0, 10:14], trajectory[1, 0, 10:14])
    return trajectory, stats


def _uncached(pair):
    style, coeff = pair.strip().splitlines()
    return f"{style} cache no\n{coeff}"


def _record_or_compare(name, value, tolerance):
    record = os.environ.get("SPINGNN2_M4_SIB_RECORD")
    compare = os.environ.get("SPINGNN2_M4_SIB_COMPARE")
    if record:
        directory = Path(record)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{name}.npy", value)
    if compare:
        reference = np.load(Path(compare) / f"{name}.npy", allow_pickle=False)
        np.testing.assert_allclose(value, reference, rtol=0, atol=tolerance)


def _physical_trajectory(trajectory):
    physical = trajectory.copy()
    physical[:, :, 14:17] *= 4.135667403e-3 / (2 * np.pi)
    return physical


@pytest.mark.parametrize("lattice", ("frozen", "moving"))
@pytest.mark.parametrize("precision,tol", (("float64", 1e-11), ("float32", 5e-5)))
def test_sib_cache_and_trajectory(lmp_run, m4_pair, tmp_path, lattice, precision, tol):
    data = tmp_path / "spin.data"
    _spin_data(data)
    pair = m4_pair("soc", precision)
    integrator = f"spin/sib 0.0 0.0 91081 lattice {lattice}"
    cached, stats = _run_sib(lmp_run, pair, data, tmp_path / "cached.dump", integrator)
    uncached, baseline = _run_sib(
        lmp_run, _uncached(pair), data, tmp_path / "uncached.dump", integrator
    )
    np.testing.assert_allclose(
        _physical_trajectory(cached), _physical_trajectory(uncached), rtol=0, atol=tol
    )
    _record_or_compare(f"sib-{lattice}-{precision}", _physical_trajectory(cached), tol)
    assert stats["hits"] >= 24
    assert baseline["hits"] == 0
    assert stats["calls"] == baseline["calls"] == 49
    assert stats["energy_drift_per_atom"] < 5e-6
    assert np.max(np.abs(cached[1, :, 10:13] - cached[0, :, 10:13])) > 1e-8
    if lattice == "frozen":
        np.testing.assert_array_equal(cached[1, :, 1:7], cached[0, :, 1:7])
    else:
        assert np.max(np.abs(cached[1, :, 1:4] - cached[0, :, 1:4])) > 1e-8
    pair_speedup = baseline["pair"] / stats["pair"]
    stage1_speedup = baseline["stage1"] / stats["stage1"]
    print(
        "M4_SIB_RESULT",
        json.dumps(
            dict(
                lattice=lattice,
                precision=precision,
                cached=stats,
                uncached=baseline,
                pair_speedup=pair_speedup,
                stage1_speedup=stage1_speedup,
            ),
            sort_keys=True,
        ),
    )
    if os.environ.get("SPINGNN2_M4_SIB_REQUIRE_SPEEDUP") == "1":
        assert pair_speedup > 1.0


@pytest.mark.parametrize("cuda_graph", ("no", "yes"))
def test_frozen_sib_recenter_cutoff(
    lmp_run, m4_backend, m4_models, m4_pair, tmp_path, cuda_graph
):
    cutoff = _cutoff(ReferenceModel.load(m4_models["soc"]["float64"]["contract"]))
    suffix = "/kk" if m4_backend in ("kk-cpu", "cuda") else ""
    steps = 6
    observations = {}
    for cache in ("yes", "no"):
        pair = m4_pair("soc", "float64").replace(
            "profile yes", f"profile no cache {cache} cuda_graph {cuda_graph}"
        )
        output = lmp_run(
            f"""units metal
boundary p p p
atom_style spin
atom_modify map array sort 0 0
newton on
region box block 0 32 0 32 0 32 units box
create_box 2 box
create_atoms 1 single 4 4 4 units box
create_atoms 2 single {4 + cutoff - 0.05:.17g} 4 4 units box
create_atoms 1 single {4 + cutoff + 1.45:.17g} 6 4 units box
mass 1 55
mass 2 81
set atom 1 spin/atom 1 1 0 0
set atom 2 spin/atom 1 0 1 0
set atom 3 spin/atom 1 0 0 1
{pair}
neighbor 1.5 bin
neigh_modify every 1 delay 0 check yes
group mobile id 2
fix integrate all spin/sib{suffix} 0.0 0.0 91081 lattice frozen
fix translation mobile recenter{suffix} {4 + cutoff + 0.05:.17g} NULL NULL units box
compute spin all property/atom spx spy spz fmx fmy fmz
compute timers all spingnn2/timing
compute graph all spingnn2/timing graph
timestep 0.000005
thermo 1
thermo_style custom step pe c_timers[5] c_graph[*]
thermo_modify norm no format float %.17g
dump trajectory all custom 1 recenter-{cache}.dump id x y z fx fy fz c_spin[*]
dump_modify trajectory sort id format float %.17g
run {steps}
"""
        )
        assert re.findall(r"Neighbor list builds\s*=\s*(\d+)", output) == ["0"]
        thermo = np.array(
            [
                [float(value) for value in row.split()]
                for row in output.splitlines()
                if re.fullmatch(r"\s*\d+(?:\s+[-+0-9.eE]+){5}\s*", row)
            ]
        )
        assert thermo.shape == (steps + 1, 6)
        np.testing.assert_array_equal(thermo[:, 0], np.arange(steps + 1))
        dump = tmp_path / f"recenter-{cache}.dump"
        blocks = dump.read_text().split("ITEM: TIMESTEP\n")[1:]
        frames = np.array([np.loadtxt(block.splitlines()[8:]) for block in blocks])
        assert frames.shape == (steps + 1, 3, 13)
        np.testing.assert_array_equal(
            frames[:, :, 0], np.tile([1, 2, 3], (steps + 1, 1))
        )
        positions = frames[:, :, 1:4]
        shift = np.zeros_like(positions)
        shift[1:, 1, 0] = 0.10
        np.testing.assert_allclose(positions - positions[0], shift, rtol=0, atol=1e-12)
        distances = np.linalg.norm(positions[:, 1:2] - positions[:, ::2], axis=2)
        assert distances[0, 0] == pytest.approx(cutoff - 0.05)
        np.testing.assert_allclose(distances[1:, 0], cutoff + 0.05, rtol=0, atol=1e-12)
        assert np.all(distances[:, 1] < cutoff)
        assert np.linalg.norm(frames[1, 1, 4:7]) > 1e-8
        frames[:, :, 10:13] *= 4.135667403e-3 / (2 * np.pi)
        observations[cache] = (frames, thermo)

    cached_frames, cached_thermo = observations["yes"]
    uncached_frames, uncached_thermo = observations["no"]
    # Check every step, including the first force after the recenter callback.
    np.testing.assert_allclose(cached_frames, uncached_frames, rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        cached_thermo[:, 1], uncached_thermo[:, 1], rtol=0, atol=1e-11
    )
    for _, thermo in observations.values():
        np.testing.assert_array_equal(thermo[:, 2], 0)
        if cuda_graph == "yes" and m4_backend == "cuda":
            assert min(thermo[-1, 3:5]) > 0
            assert thermo[-1, 5] == 0
        else:
            np.testing.assert_array_equal(thermo[:, 3:], 0)
    difference = cached_frames - uncached_frames
    max_force_torque_error = np.max(np.abs(difference[:, :, [4, 5, 6, 10, 11, 12]]))
    print(
        f"M4_SIB_RECENTER_CUTOFF backend={m4_backend} graph={cuda_graph} rebuilds=0 "
        f"max_force_torque_error={max_force_torque_error:.6g} "
        f"max_energy_error={np.max(np.abs(cached_thermo[:, 1] - uncached_thermo[:, 1])):.6g} "
        f"captures={int(cached_thermo[-1, 3])} replays={int(cached_thermo[-1, 4])}"
    )


def test_frozen_sib_keeps_cache_with_diagnostic_and_langevin(
    lmp_run, m4_pair, tmp_path
):
    data = tmp_path / "spin.data"
    _spin_data(data)
    steps = 6
    _, stats = _run_sib(
        lmp_run,
        m4_pair("soc-e2ip", "float64"),
        data,
        tmp_path / "diagnostic.dump",
        "spin/sib 0.0 0.0 91081 lattice frozen",
        fixes=(
            "fix evidence all pair 1 spingnn2 sf_sigma 0\n"
            "fix bath all langevin 10.0 10.0 1.0 137931"
        ),
        steps=steps,
    )
    assert stats["calls"] == 1 + 2 * steps
    assert stats["hits"] == 2 * steps
    print(f"M4_SIB_DIAGNOSTIC_CACHE hits={int(stats['hits'])}")


def test_sib_mpi_stochastic(lmp_run, m4_pair, tmp_path):
    ranks = tuple(
        int(n) for n in os.environ.get("SPINGNN2_M4_MPI_RANKS", "1").split(",")
    )
    if not any(n > 1 for n in ranks):
        pytest.skip("SIB multi-rank qualification requires two or four ranks")
    data = tmp_path / "spin.data"
    _spin_data(data)
    integrator = "spin/sib 50.0 0.07 91081 lattice moving"
    pair = m4_pair("nosoc", "float64")
    one, _ = _run_sib(lmp_run, pair, data, tmp_path / "one.dump", integrator)
    one = _physical_trajectory(one)
    _record_or_compare("sib-stochastic-float64", one, 1e-11)
    for n in ranks:
        if n == 1:
            continue
        multi, stats = _run_sib(
            lmp_run, pair, data, tmp_path / f"rank-{n}.dump", integrator, ranks=n
        )
        error = float(np.max(np.abs(_physical_trajectory(multi) - one)))
        print("M4_SIB_MPI", json.dumps(dict(ranks=n, max_abs=error, **stats)))
        assert error <= 1e-11


def test_spin_sib_npt(lmp_run, m4_pair, tmp_path):
    data = tmp_path / "spin.data"
    _spin_data(data)
    integrator = (
        "spin/sib/npt temp 20.0 20.0 0.02 iso 1.0 1.0 0.05 alpha 0.05 seed 91081"
    )
    trajectory, stats = _run_sib(
        lmp_run, m4_pair("nosoc", "float64"), data, tmp_path / "npt.dump", integrator
    )
    assert stats["hits"] == 0
    assert stats["calls"] == 49
    assert abs(stats["volume"] - BOX_SIDE**3) > 1e-8
    assert np.max(np.abs(trajectory[1, :, 10:13] - trajectory[0, :, 10:13])) > 1e-8
    _record_or_compare("sib-npt-float64", _physical_trajectory(trajectory), 1e-11)
    _record_or_compare("sib-npt-volume-float64", stats["volume"], 1e-11)
    print("M4_SIB_NPT", json.dumps(stats, sort_keys=True))
