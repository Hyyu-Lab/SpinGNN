"""Check the prepared pair against the independent M1 oracle and across MPI domains."""

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from spingnn2.reference import ReferenceModel


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["nosoc", "soc", "born", "e2ip", "soc-e2ip"]
# Put atoms across a processor plane and across the periodic boundary.
POS = np.array(
    [
        [0.8, 1.2, 2.1],
        [15.1, 1.6, 1.4],
        [7.1, 7.5, 7.1],
        [9.2, 7.3, 7.9],
        [6.9, 8.9, 9.1],
        [8.1, 8.6, 6.5],
    ]
)
SPIN = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
TYPES = np.arange(len(POS)) % 2


def test_overlay_pin_and_files(tmp_path):
    """Independently apply, check idempotence and detect modified core code."""
    upstream = Path(
        os.getenv("SPINGNN2_LAMMPS_UPSTREAM", ROOT / "validation/lammps/upstream")
    )
    if not (upstream / ".git").exists():
        if "SPINGNN2_LAMMPS_BACKEND" in os.environ:
            pytest.fail(f"Missing checkout {upstream}")
        pytest.skip("Set SPINGNN2_LAMMPS_UPSTREAM to the pinned Git checkout")
    spec = importlib.util.spec_from_file_location(
        "overlay", ROOT / "lammps_spingnn2/apply.py"
    )
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    repo = tmp_path / "source"
    expected = overlay.materialize(upstream, repo)
    assert overlay.materialize(upstream, repo) == expected
    core = (
        "atom.h",
        "atom.cpp",
        "compute_property_atom.cpp",
        "dump_custom.cpp",
        "fix_efield.cpp",
        "fix_nh.cpp",
        "fix_nve.cpp",
        "library.cpp",
        "min_fire.cpp",
        "min_quickmin.cpp",
        "KOKKOS/fix_efield_kokkos.cpp",
    )
    changed = subprocess.check_output(
        ["git", "-C", repo, "diff", "--name-only", "HEAD"], text=True
    ).splitlines()
    assert all("src/" + name in changed for name in core)
    with (repo / "src/atom.cpp").open("a") as f:
        f.write("// unreviewed edit\n")
    with pytest.raises(subprocess.CalledProcessError):
        overlay.verify(repo)


def _input(family, pair, newton, steps=0):
    born = family == "born"
    magnetic = family in ("nosoc", "soc", "soc-e2ip")
    style = "bec" if born else "spin" if magnetic else "atomic"
    create = []
    for i, (pos, spin) in enumerate(zip(POS, SPIN)):
        create.append(
            f"create_atoms {1+TYPES[i]} single "
            + " ".join(map(str, pos))
            + " units box"
        )
        if magnetic:
            create.append(f"set atom {i+1} spin/atom 1 " + " ".join(map(str, spin)))
    extra = "compute sf all property/atom fmx fmy fmz" if magnetic else ""
    cols = "c_sf[*]" if magnetic else ""
    if family in ("e2ip", "soc-e2ip"):
        prefix = "f" if family == "e2ip" else "sf"
        fields = [f"{prefix}_{name}" for name in ("sigma", "Uepi", "nu", "kappa")]
        extra += "\nfix evidence all pair 1 spingnn2 " + " ".join(
            f"{field} 0" for field in fields
        )
        cols += " f_evidence[*]"
    return f"""units metal
atom_style {style}
atom_modify map array
boundary p p p
newton {newton}
region box block 0 16 0 16 0 16 units box
create_box 2 box
{chr(10).join(create)}
mass 1 55
mass 2 81
{pair}
neighbor 0.7 bin
neigh_modify every 1 delay 0 check yes
fix integrate all nve
velocity all set 0.013 -0.021 0.017
compute at all stress/atom NULL pair
compute ct all centroid/stress/atom NULL pair
{extra}
compute p all pressure NULL pair
thermo_style custom step pe etotal c_p[1] c_p[2] c_p[3] c_p[4] c_p[5] c_p[6]
thermo 1
thermo_modify norm no format float %.17g
timestep 0.00001
dump snapshot all custom {max(steps, 1)} result.dump id fx fy fz c_at[*] c_ct[*] {cols}
dump_modify snapshot sort id format float %.17g
run {steps}
"""


def _thermo(output):
    return np.array(
        [
            [float(v) for v in row.split()]
            for row in output.splitlines()
            if re.fullmatch(r"\s*\d+(?:\s+[-+0-9.eE]+){8}\s*", row)
        ]
    )


def _cutoff(ref):
    radii = [
        1 / layer["basis"]["cutoff"]["factor"]
        for layer in ref.contract["layers"]
        if "cutoff" in layer.get("basis", {})
    ]
    for layer in ref.contract["layers"]:
        if "backbone" in layer:
            radii.extend(np.ravel(layer["backbone"]["cutoffs"]))
    return max(radii)


def _reference_graph(ref, positions, types, spins, box):
    delta = positions[None, :, :] - positions[:, None, :]
    shift = -np.rint(delta / box)
    distances = np.linalg.norm(delta + box * shift, axis=-1)
    i, j = np.nonzero((distances < _cutoff(ref)) & (distances > 0))
    graph = dict(
        pos=positions,
        atom_types=types,
        edge_index=np.array([i, j]),
        cell=np.eye(3) * box,
        pbc=True,
        edge_cell_shift=shift[i, j],
    )
    if ref.magnetic:
        graph["spin"] = spins
    return ref.evaluate(graph)


def _reference(model):
    return _reference_graph(ReferenceModel.load(model), POS, TYPES, SPIN, 16.0)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("precision,tol", [("float64", 1e-11), ("float32", 5e-5)])
@pytest.mark.parametrize("newton", ["on", "off"])
def test_domain_physics(
    lmp_run, m4_pair, m4_models, tmp_path, family, precision, tol, newton
):
    ref = _reference(m4_models[family][precision]["contract"])
    script = _input(family, m4_pair(family, precision), newton)
    order = [0, 4, 8, 1, 2, 5, 3, 6, 7]
    atomic = ref["atom_virial"].reshape(-1, 9)[:, order]
    expected = np.column_stack([ref["forces"], atomic[:, :6], atomic])
    spin = family in ("soc", "nosoc", "soc-e2ip")
    if spin:
        expected = np.column_stack([expected, ref["spin_forces"]])
    if family in ("e2ip", "soc-e2ip"):
        prefix = "f" if family == "e2ip" else "sf"
        expected = np.column_stack(
            [
                expected,
                *[
                    ref[f"{prefix}_{name}"].reshape(len(POS), -1)
                    for name in ("sigma", "Uepi", "nu", "kappa")
                ],
            ]
        )
    baseline = None
    for n in map(int, os.environ.get("SPINGNN2_M4_MPI_RANKS", "1").split(",")):
        thermo = _thermo(lmp_run(script, ranks=n))[-1]
        values = np.loadtxt(tmp_path / "result.dump", skiprows=9)[:, 1:]
        values[:, 3:18] /= -1.6021765e6
        if spin:
            values[:, 18:21] *= 4.135667403e-3 / (2 * np.pi)
        np.testing.assert_allclose(
            thermo[3:] * 16**3 / 1.6021765e6,
            ref["virial"].reshape(9)[order[:6]],
            rtol=0,
            atol=tol,
        )
        np.testing.assert_allclose(values, expected, rtol=0, atol=tol)
        assert abs(thermo[1] - ref["total_energy"].item()) <= tol
        if baseline is None:
            assert n == 1
            baseline = values
        delta = np.max(np.abs(values - baseline))
        assert delta <= tol if precision == "float64" else delta < tol
        print(
            "M4_DOMAIN",
            json.dumps(
                dict(
                    family=family,
                    newton=newton,
                    precision=precision,
                    ranks=n,
                    oracle=float(np.max(np.abs(values - expected))),
                    rank_error=float(delta),
                )
            ),
        )


def test_relocated_runtime_without_environment(m4_binary, tmp_path):
    """The installed executable must load its sibling ABI library after relocation."""
    prefix = m4_binary.parent.parent
    relocated = tmp_path / "bin" / m4_binary.name
    relocated.parent.mkdir()
    shutil.copy2(m4_binary, relocated)
    (library,) = prefix.glob("lib*/libspingnn2.so.1")
    sibling = tmp_path / library.parent.name / library.name
    sibling.parent.mkdir()
    shutil.copy2(library, sibling)

    env = {"PATH": "/usr/bin:/bin"}
    for target in (relocated, sibling):
        deps = subprocess.check_output(["ldd", target], env=env, text=True)
        assert not re.search(r"lib(?:torch|c10|nvrtc)|not found", deps), deps
        if target == relocated:
            match = re.search(r"libspingnn2\.so\.1 => (.+?) \(0x", deps)
            assert match and Path(match[1]).resolve() == sibling, deps
    help_text = subprocess.check_output(
        [relocated, "-h"], env=env, cwd=tmp_path, text=True, stderr=subprocess.STDOUT
    )
    assert "spingnn2" in help_text


def test_nve_and_runtime(lmp_run, m4_pair, m4_binary, tmp_path):
    output = lmp_run(_input("soc", m4_pair("soc"), "on", steps=50), trace=True)
    trajectory = _thermo(output)
    assert len(trajectory) == 51
    assert np.ptp(trajectory[:, 2]) / len(POS) < 1e-8
    deps = subprocess.check_output(["ldd", m4_binary], text=True)
    assert not re.search(r"lib(?:torch|c10|nvrtc)|not found", deps)
    _check_runtime_trace(tmp_path / "runtime.trace")


def _check_runtime_trace(path):
    trace = path.read_text()
    assert not re.search(
        r'execve(?:at)?\([^\n]*(?:/(?:g\+\+|c\+\+|gcc|clang\+\+|nvcc|ptxas)"|libnvrtc)',
        trace,
    )
    assert not re.search(r"openat\([^\n]*libnvrtc[^\n]*= [0-9]+", trace)


@pytest.mark.parametrize("precision,tolerance", [("float64", 1e-11), ("float32", 5e-5)])
def test_cuda_graph_trajectory(
    lmp_run, m4_pair, m4_backend, tmp_path, precision, tolerance
):
    if m4_backend != "cuda":
        pytest.skip("CUDA Graph replay requires the CUDA pair style")

    observations = {}
    for enabled in ("no", "yes"):
        pair = m4_pair("soc", precision).replace(
            "profile yes", f"profile no cuda_graph {enabled}"
        )
        script = (
            _input("soc", pair, "on", steps=21)
            .replace("every 1 delay 0 check yes", "every 5 delay 0 check no")
            .replace(
                "compute p all",
                "compute graph all spingnn2/timing graph\ncompute p all",
            )
            .replace("c_p[6]", "c_p[6] c_graph[*]")
        )
        output = lmp_run(script, trace=enabled == "yes")
        if enabled == "yes":
            _check_runtime_trace(tmp_path / "runtime.trace")
        thermo = np.array(
            [
                [float(value) for value in row.split()]
                for row in output.splitlines()
                if re.fullmatch(r"\s*\d+(?:\s+[-+0-9.eE]+){11}\s*", row)
            ]
        )
        assert thermo.shape == (22, 12)
        captures, replays, fallbacks = thermo[-1, -3:]
        if enabled == "yes":
            assert captures >= 5 and replays >= 10 and fallbacks == 0
        else:
            assert captures == replays == fallbacks == 0
        rows = (tmp_path / "result.dump").read_text().splitlines()[-len(POS) :]
        per_atom = np.loadtxt(rows)
        per_atom[:, 4:19] /= -1.6021765e6
        per_atom[:, 19:22] *= 4.135667403e-3 / (2 * np.pi)
        thermo[:, 3:9] *= 16**3 / 1.6021765e6
        observations[enabled] = (per_atom, thermo[:, :9])

    errors = [
        float(np.max(np.abs(eager - replay)))
        for eager, replay in zip(observations["no"], observations["yes"])
    ]
    error = max(errors)
    assert error <= tolerance if precision == "float64" else error < tolerance
    print(
        f"PHASE_A_GRAPH {precision} error={error:.6g} "
        f"captures={int(captures)} replays={int(replays)}"
    )


@pytest.mark.parametrize("family", ("soc", "born"))
@pytest.mark.parametrize("precision,tol", [("float64", 1e-11), ("float32", 5e-5)])
def test_cuda_graph_cutoff_crossing(
    lmp_run, m4_pair, m4_models, m4_backend, tmp_path, family, precision, tol
):
    ref = ReferenceModel.load(m4_models[family][precision]["contract"])
    cutoff = _cutoff(ref)
    types = np.array([0, 1, 0])
    spins = SPIN[:3]
    box = 32.0
    order = [0, 4, 8, 1, 2, 5, 3, 6, 7]
    born = family == "born"
    spin_setup = (
        ""
        if born
        else "set atom 1 spin/atom 1 1 0 0\nset atom 2 spin/atom 1 0 1 0\n"
        "set atom 3 spin/atom 1 0 0 1\ncompute spin all property/atom fmx fmy fmz"
    )
    extras = (
        "becxx becxy becxz becyx becyy becyz beczx beczy beczz" if born else "c_spin[*]"
    )
    observations = {}
    graph_modes = ("no", "yes") if m4_backend == "cuda" else ("no",)
    for enabled in graph_modes:
        pair = m4_pair(family, precision).replace(
            "profile yes", f"profile no cuda_graph {enabled}"
        )
        output = lmp_run(
            f"""units metal
atom_style {"bec" if born else "spin"}
atom_modify map array sort 0 0
boundary p p p
newton on
region box block 0 {box} 0 {box} 0 {box} units box
create_box 2 box
create_atoms 1 single 4 4 4 units box
create_atoms 2 single {4 + cutoff - 0.066:.17g} 4 4 units box
create_atoms 1 single 4 4 20 units box
mass 1 55
mass 2 81
{spin_setup}
{pair}
neighbor 1.5 bin
neigh_modify every 1 delay 0 check yes
group mobile id 2
fix translation mobile move linear 12 0 0 units box
compute graph all spingnn2/timing graph
compute at all stress/atom NULL pair
compute ct all centroid/stress/atom NULL pair
compute p all pressure NULL pair
thermo_style custom step pe c_p[*] c_graph[*]
thermo 1
thermo_modify norm no format float %.17g
timestep 0.001
dump snapshot all custom 1 cutoff.dump id x y z fx fy fz c_at[*] c_ct[*] {extras}
dump_modify snapshot sort id format float %.17g
run 14
"""
        )
        assert re.search(r"Neighbor list builds\s*=\s*0\b", output)
        thermo = np.array(
            [
                [float(value) for value in row.split()]
                for row in output.splitlines()
                if re.fullmatch(r"\s*\d+(?:\s+[-+0-9.eE]+){10}\s*", row)
            ]
        )
        assert thermo.shape == (15, 11)
        if enabled == "yes":
            assert thermo[5, 8] == 1 and thermo[5, 9] > 0
            assert thermo[-1, 8] == 2 and thermo[-1, 9] > thermo[5, 9]
            assert thermo[-1, 10] == 0
        else:
            np.testing.assert_array_equal(thermo[:, 8:], 0)
        blocks = (tmp_path / "cutoff.dump").read_text().split("ITEM: TIMESTEP\n")[1:]
        frames = np.array([np.loadtxt(block.splitlines()[8:]) for block in blocks])
        np.testing.assert_array_equal(frames.shape, (15, 3, 31 if born else 25))
        assert frames[5, 1, 1] - frames[5, 0, 1] < cutoff
        assert frames[6, 1, 1] - frames[6, 0, 1] > cutoff
        frames[:, :, 7:22] /= -1.6021765e6
        if not born:
            frames[:, :, 22:25] *= 4.135667403e-3 / (2 * np.pi)
        thermo[:, 2:8] *= box**3 / 1.6021765e6
        observations[enabled] = (thermo[:, 1:8], frames[:, :, 4:])
        for step, frame in enumerate(frames):
            expected = _reference_graph(ref, frame[:, 1:4], types, spins, box)
            virial = expected["atom_virial"].reshape(3, 9)[:, order]
            extra = (
                expected["born_effective_charges"].reshape(3, 9)
                if born
                else expected["spin_forces"]
            )
            ref_atom = np.column_stack(
                [expected["forces"], virial[:, :6], virial, extra]
            )
            ref_thermo = np.r_[
                expected["total_energy"].item(),
                expected["virial"].reshape(9)[order[:6]],
            ]
            np.testing.assert_allclose(frame[:, 4:], ref_atom, rtol=0, atol=tol)
            np.testing.assert_allclose(thermo[step, 1:8], ref_thermo, rtol=0, atol=tol)

    if "yes" not in observations:
        return
    max_error = 0.0
    for ordinary, replay in zip(observations["no"], observations["yes"]):
        max_error = max(max_error, np.max(np.abs(ordinary - replay)))
    assert max_error <= tol if precision == "float64" else max_error < tol
    print(f"M4_GRAPH_CUTOFF_CROSSING {family} {precision} max_abs={max_error:.6g}")


def test_missing_artifact_is_fatal(lmp_run, m4_pair, m4_models, tmp_path):
    spec = m4_models["soc"]["float64"]
    pair = m4_pair("soc").replace(spec["manifest"], str(tmp_path / "missing.json"))
    with pytest.raises(AssertionError, match="Open JIT path missing.json"):
        lmp_run(_input("soc", pair, "on"))


def test_tf32_opt_in(lmp_run, m4_pair, m4_backend):
    if m4_backend != "cuda":
        pytest.skip("CUDA FP32 tensor math requires a GPU")
    commands = m4_pair("soc", "float32").replace("profile yes", "profile yes math tf32")
    thermo = _thermo(lmp_run(_input("soc", commands, "on")))
    assert len(thermo) == 1 and np.isfinite(thermo).all()
