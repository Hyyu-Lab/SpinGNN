"""Exercise evidential MD and trained-checkpoint deployment in modern LAMMPS."""

from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list

from spingnn2.contract import load_contract
from spingnn2.native import NativeModel


# Conversion used by LAMMPS in metal units.
BAR_PER_EV_ANGSTROM3 = 1.6021765e6
VOIGT = [0, 4, 8, 1, 2, 5]
CENTROID = VOIGT + [3, 6, 7]
READOUTS = {"sigma": 9, "Uepi": 9, "nu": 1, "kappa": 1}


def dump_frames(path):
    lines = path.read_text().split("ITEM: TIMESTEP\n")[1:]
    return [
        np.loadtxt(io.StringIO(block.split("ITEM: ATOMS ", 1)[1]), skiprows=1)
        for block in lines
    ]


def pair_commands(spec, precision, species):
    return (
        f"pair_style spingnn2 precision {precision} math ieee\n"
        f'pair_coeff * * "{spec["contract"]}" "{spec["manifest"]}" {species}'
    )


def fix_evidence(prefix):
    return "fix evidence all pair 1 spingnn2 " + " ".join(
        f"{prefix}_{field} 0" for field in READOUTS
    )


def check_fields(
    values, expected, prefix, tolerance, *, normalized, covariance_scale=1
):
    maxima = {}
    offset = 0
    for field, width in READOUTS.items():
        key = f"{prefix}_{field}"
        target = expected[key].reshape(len(values), width)
        actual = values[:, offset : offset + width]
        assert actual.shape == target.shape
        error = np.abs(actual - target)
        if normalized:
            units = covariance_scale if width == 9 else 1.0
            error /= np.maximum(units, np.abs(target))
        maxima[key] = float(np.max(error))
        assert maxima[key] < tolerance if normalized else maxima[key] <= tolerance, (
            key,
            maxima[key],
        )
        offset += width
    assert offset == values.shape[1]
    return maxima


def cutoff_radius(contract):
    layers = load_contract(contract)["layers"]
    radii = [
        1.0 / layer["basis"]["cutoff"]["factor"]
        for layer in layers
        if "basis" in layer and "cutoff" in layer["basis"]
    ]
    for layer in layers:
        if "backbone" in layer:
            radii.extend(np.ravel(layer["backbone"]["cutoffs"]))
    return float(max(radii))


def production_rebuilds(output):
    builds = re.findall(r"Neighbor list builds =\s*(\d+)", output)
    assert len(builds) == 2, output
    # The production run uses pre no, which preserves the neighbor counter.
    rebuilds = int(builds[1]) - int(builds[0])
    assert rebuilds > 0, "No displacement-triggered rebuilds in production"
    return rebuilds


def run_thermo(output, atoms):
    headers = re.split(
        r"^[ \t]*Step\s+Atoms\s+Temp\s+PotEng\s+TotEng\s+Econserve[ \t]*$",
        output,
        flags=re.MULTILINE,
    )
    assert len(headers) == 3, output
    runs = []
    for block in headers[1:]:
        samples, boundary, _ = block.partition("\nLoop time of ")
        assert boundary, output
        rows = np.array(
            [
                fields
                for line in samples.splitlines()
                if len(fields := line.split()) == 6
                and fields[0].isdigit()
                and fields[1] == str(atoms)
            ],
            dtype=float,
        )
        assert rows.size and np.isfinite(rows).all(), samples
        runs.append(rows)
    return runs


def test_evidential_production_rebuilds():
    output = "Neighbor list builds = 9\nNeighbor list builds = 13\n"
    assert production_rebuilds(output) == 4


def test_evidential_rejects_warmup_only_rebuilds():
    output = "Neighbor list builds = 1\nNeighbor list builds = 1\n"
    with pytest.raises(AssertionError, match="No displacement-triggered rebuilds"):
        production_rebuilds(output)


def test_evidential_production_thermo_after_bath_reset():
    output = """   Step        Atoms         Temp          PotEng         TotEng       Econserve
         0        4096  300.0 -21578.245031833649 -21419.448943701147 -21419.448943701147
       200        4096  309.1550863122302 -21434.053029060364 -21270.410967951539 -21419.281806767114
Loop time of 1.09121 on 1 procs for 200 steps with 4096 atoms
Neighbor list builds = 9
Dangerous builds = 0
   Step        Atoms         Temp          PotEng         TotEng       Econserve
       200        4096  309.1550863122302 -21434.053029060364 -21270.410967951539 -21270.410967951539
       210        4096  304.70291199455511 -21434.45663690567 -21273.171202014615 -21270.24877724106
       290        4096  298.99043052628065 -21432.143057823181 -21273.881355301095 -21270.60161860671
Loop time of 0.506946 on 1 procs for 90 steps with 4096 atoms
Neighbor list builds = 13
Dangerous builds = 0
"""
    warmup, production = run_thermo(output, 4096)
    assert warmup[-1, 0] == production[0, 0] == 200
    assert warmup[-1, 5] == -21419.281806767114
    assert production[0, 5] == -21270.410967951539
    assert list(production[:, 0]) == [200, 210, 290]
    assert np.ptp(production[:, 5]) / 4096 == pytest.approx(
        0.000086142911535525, abs=1e-14
    )


@pytest.mark.parametrize(
    "family,prefix", [("e2ip", "f"), ("soc-e2ip", "sf"), ("trained-e2ip", "f")]
)
def test_evidential_md_after_rebuild(family, prefix, lmp_run, m4_backend, tmp_path):
    qualified = os.getenv("SPINGNN2_AUDIT_EVIDENCE")
    if m4_backend != "cuda" or not qualified:
        pytest.skip("Set SPINGNN2_AUDIT_EVIDENCE for the prepared 4096-atom CUDA lane")
    if family == "trained-e2ip":
        trained = os.getenv("SPINGNN2_AUDIT_TRAINED")
        if not trained:
            pytest.skip(
                "Set SPINGNN2_AUDIT_TRAINED for Si MD with real trained weights"
            )
        spec = json.loads(Path(trained).read_text())["models"]["float32"]
        cells, lattice, pitch, mass = 8, "diamond", 5.4309, 28.085
        atoms = 8 * cells**3
        # Short training verifies the deployment path. A thermal bath keeps
        # this checkpoint's lattice near 300 K during the MD readout test.
        thermostat = (
            "fix bath all langevin/kk 300.0 300.0 0.01 137931 tally yes zero yes"
        )
        warmup = 200
    else:
        spec = json.loads(Path(qualified).read_text())[family]
        cells, lattice, mass = 16, "sc", 50.0
        pitch = cutoff_radius(spec["contract"]) / 2.42
        atoms = cells**3
        thermostat = ""
        warmup = 10
    cutoff = cutoff_radius(spec["contract"])
    box_size = cells * pitch
    kinds = "spin" if prefix == "sf" else "atomic"
    spin_setup = ""
    spin_columns = ""
    if prefix == "sf":
        spin_setup = """variable sx atom cos(0.163*(id%19))
variable sy atom sin(0.163*(id%19))
set group all spin/atom 1.0 v_sx v_sy 0.0
compute s all property/atom spx spy spz sp
"""
        spin_columns = "c_s[*]"

    # The full-width SOC evidence model is much larger. A 45 fs window at
    # 1 fs per step still permits natural skin rebuilds in this workload.
    steps, timestep = (45 if prefix == "sf" else 90), 0.001
    output = lmp_run(
        f"""units metal
log md-live.log
atom_style {kinds}
atom_modify map array
boundary p p p
newton on
lattice {lattice} {pitch:.17g}
region box block 0 {cells} 0 {cells} 0 {cells}
create_box 1 box
create_atoms 1 box
mass 1 {mass}
{spin_setup}{pair_commands(spec, "float32", "0")}
neighbor 0.4 bin
neigh_modify every 1 delay 0 check yes
velocity all create 300.0 16831 mom yes rot no dist gaussian
fix integrate all nve/kk
{thermostat}
{fix_evidence(prefix)}
timestep {timestep}
thermo 10
thermo_style custom step atoms temp pe etotal econserve
thermo_modify norm no format float %.17g flush yes
dump state all custom {warmup + steps} result.dump id x y z {spin_columns} f_evidence[*]
dump_modify state sort id format float %.17g
run {warmup}
run {steps} pre no
"""
    )
    (tmp_path / "md.log").write_text(output)
    loops = re.findall(
        r"Loop time of ([\d.eE+-]+) on 1 procs for (\d+) steps with (\d+) atoms",
        output,
    )
    assert len(loops) == 2 and tuple(map(int, loops[1][1:])) == (steps, atoms)
    elapsed = float(loops[1][0])
    assert np.isfinite(elapsed) and elapsed > 0
    timing = re.split(r"^Loop time of .*", output, flags=re.MULTILINE)[2]
    phase_seconds = {
        name: float(seconds)
        for name, seconds in re.findall(
            r"^(Pair|Neigh|Comm|Output|Modify)\s*\|\s*([\d.eE+-]+)",
            timing,
            re.MULTILINE,
        )
    }
    assert len(phase_seconds) == 5
    rebuilds = production_rebuilds(output)
    warmup_thermo, measured_thermo = run_thermo(output, atoms)
    assert warmup_thermo[0, 0] == 0
    assert warmup_thermo[-1, 0] == measured_thermo[0, 0] == warmup
    assert measured_thermo[-1, 0] == warmup + steps
    initial_temperature = warmup_thermo[0, 2]
    assert initial_temperature == pytest.approx(300.0, rel=1e-10)
    temperature = measured_thermo[:, 2]
    assert 200.0 <= temperature.min() <= temperature.max() <= 450.0, measured_thermo
    first, last = dump_frames(tmp_path / "result.dump")
    np.testing.assert_array_equal(last[:, 0], np.arange(1, atoms + 1))
    assert np.isfinite(first).all() and np.isfinite(last).all()
    assert np.max(np.abs(last[:, 1:4] - first[:, 1:4])) > 0.01
    offset = 8 if prefix == "sf" else 4
    evidence = last[:, offset:]
    covariance_scale = spec.get("output_scale", 1.0) ** 2
    covariance_change = np.max(
        np.abs(evidence[:, :18] - first[:, offset : offset + 18])
    )
    assert covariance_change / covariance_scale > 1e-8
    dimension = 2 if prefix == "sf" else 3
    assert np.min(evidence[:, 18]) >= dimension + 2
    assert np.min(evidence[:, 19]) > 0
    for matrix in (evidence[:, :9], evidence[:, 9:18]):
        matrix = matrix.reshape(atoms, 3, 3)
        scale = np.maximum(1.0, np.max(np.abs(matrix), axis=(1, 2)))[:, None, None]
        normalized_matrix = matrix / scale
        np.testing.assert_allclose(
            normalized_matrix, normalized_matrix.transpose(0, 2, 1), atol=1e-6, rtol=0
        )
        assert np.min(np.linalg.eigvalsh(normalized_matrix)) >= -1e-6

    # Compare the evidence written by fix pair during the timed trajectory.
    position = last[:, 1:4]
    cell = np.eye(3) * box_size
    i, j, shifts = neighbor_list(
        "ijS",
        Atoms(numbers=[1] * atoms, positions=position, cell=cell, pbc=True),
        cutoff,
    )
    graph = dict(
        pos=position,
        atom_types=np.zeros(atoms, dtype=np.int64),
        edge_index=np.stack([i, j]),
        edge_cell_shift=shifts,
        cell=cell,
        pbc=True,
    )
    if prefix == "sf":
        graph["spin"] = np.where(last[:, 7:8] > 0, last[:, 4:7], 0)
    with NativeModel.load(
        spec["contract"],
        backend="cuda",
        precision="float32",
        artifact=spec["manifest"],
        execution_mode="direct",
        allow_tf32=False,
    ) as model:
        expected = model.evaluate(graph)
    peak_error = check_fields(evidence, expected, prefix, 1e-4, normalized=True)
    unscaled = check_fields(
        evidence,
        expected,
        prefix,
        1e-4,
        normalized=True,
        covariance_scale=covariance_scale,
    )
    energy_error = abs(measured_thermo[-1, 3] - expected["total_energy"].item()) / atoms
    assert energy_error < 5e-5
    rate = steps / elapsed
    result = dict(
        family=family,
        atoms=atoms,
        steps=steps,
        temperature_K=initial_temperature,
        temperature_range_K=[float(temperature.min()), float(temperature.max())],
        planned_timestep_fs=timestep * 1000,
        skin_angstrom=0.4,
        method="nve/kk + langevin/kk" if thermostat else "nve/kk",
        spins="fixed noncollinear" if prefix == "sf" else None,
        warmup_steps=warmup,
        rebuilds=rebuilds,
        active_edges_at_last_step=len(i),
        elapsed_seconds=elapsed,
        lammps_section_seconds=phase_seconds,
        steps_per_second=rate,
        ns_per_day=rate * timestep * 86.4,
        energy_range_eV_per_atom=np.ptp(measured_thermo[:, 4]) / atoms,
        conserved_energy_range_eV_per_atom=np.ptp(measured_thermo[:, 5]) / atoms,
        evidence_normalized_error=peak_error,
        evidence_normalized_error_before_output_scale=unscaled,
        covariance_max_change_before_output_scale=float(
            covariance_change / covariance_scale
        ),
        final_energy_error_eV_per_atom=energy_error,
        gpu_uuid=re.findall(r"^SPINGNN2_GPU .* uuid=(GPU-[a-f0-9-]+)$", output, re.M)[
            0
        ],
    )
    (tmp_path / "evidence-md.json").write_text(json.dumps(result, indent=2) + "\n")
    print("AUDIT_EVIDENCE_MD", json.dumps(result))


@pytest.mark.parametrize("precision,tolerance", [("float64", 1e-11), ("float32", 5e-5)])
def test_trained_checkpoint_lammps_eager(precision, tolerance, lmp_run, tmp_path):
    trained = os.getenv("SPINGNN2_AUDIT_TRAINED")
    if not trained:
        pytest.skip("Set SPINGNN2_AUDIT_TRAINED to the prepared trained-model index")
    # The eager control runs from PyTorch's saved checkpoint in CPU mode.
    # LAMMPS loads the separately exported contract and a prepared artifact.
    from nequip.data import AtomicData
    from nequip.data.transforms import TypeMapper
    from nequip.train import Trainer
    from nequip.utils import Config
    from nequip.utils._global_options import global_options_context

    report = json.loads(Path(trained).read_text())
    spec = report["models"][precision]
    directory = Path(spec["directory"])
    assert (
        sha256((directory / "best_model.pth").read_bytes()).hexdigest()
        == spec["checkpoint_sha256"]
    )
    held_out = Path(report["held_out"])
    assert sha256(held_out.read_bytes()).hexdigest() == report["held_out_sha256"]
    frames = read(held_out, index=":", format="extxyz")
    assert len(frames) == len(report["frames"]["parity"]) == 2
    config = Config.from_file(str(directory / "config.yaml"))
    evidence_tolerance = 1e-4 if precision == "float32" else tolerance
    maxima = []
    with global_options_context(config):
        model, _ = Trainer.load_model_from_training_session(directory, device="cpu")
        model.eval()
        for frame_id, frame in zip(report["frames"]["parity"], frames):
            data = tmp_path / f"frame-{frame_id}.data"
            # The pinned Si trajectory has a cubic cell. Wrapping into that
            # cell preserves the frame. Both arms receive the same IEEE
            # coordinates and cell before constructing their periodic images.
            frame.wrap()
            np.testing.assert_array_equal(
                frame.cell.array, np.diag(frame.cell.lengths())
            )
            frame.positions[:] = frame.positions.astype(precision)
            frame.cell[:] = frame.cell.array.astype(precision)
            write(data, frame, format="lammps-data", atom_style="atomic", masses=True)
            atoms = AtomicData.from_ase(frame, r_max=config["r_max"])
            atoms = TypeMapper(chemical_symbol_to_type={"Si": 0})(atoms)
            inputs = AtomicData.to_AtomicDataDict(atoms)
            np.testing.assert_array_equal(inputs["pos"].numpy(), frame.positions)
            np.testing.assert_array_equal(
                inputs["cell"].numpy().reshape(3, 3), frame.cell.array
            )
            expected = {
                key: value.detach().numpy()
                for key, value in model(dict(inputs)).items()
            }
            output = lmp_run(
                f"""units metal
atom_style atomic
boundary p p p
newton on
read_data "{data}"
{pair_commands(spec, precision, "Si")}
neighbor 1.5 bin
neigh_modify every 1 delay 0 check yes
{fix_evidence("f")}
compute atomic all pe/atom
compute stress all centroid/stress/atom NULL pair
compute p all pressure NULL pair
thermo 1
thermo_style custom step pe c_p[*]
thermo_modify norm no format float %.17g
dump state all custom 1 result.dump id c_atomic fx fy fz c_stress[*] f_evidence[*]
dump_modify state sort id format float %.17g
run 0
""",
                trace=True,
            )
            (tmp_path / f"frame-{frame_id}.log").write_text(output)
            trace = tmp_path / f"frame-{frame_id}.trace"
            (tmp_path / "runtime.trace").replace(trace)
            raw_trace = trace.read_text()
            assert not re.search(
                r'execve(?:at)?\([^\n]*"/[^"\n]*/(?:cc|c\+\+|gcc|g\+\+|clang(?:\+\+)?|nvcc|ptxas|nvlink)"',
                raw_trace,
            )
            assert not re.search(
                r'openat(?:2)?\([^\n]*"(?:[^"\n]*/)?lib(?:torch|c10|nvrtc)[^"]*"[^\n]*= [0-9]+',
                raw_trace,
            )
            image_name = json.loads(Path(spec["manifest"]).read_text())["artifact"]
            assert re.search(
                rf'openat\([^\n]*{re.escape(image_name)}"[^\n]*= [0-9]+', raw_trace
            )
            (values,) = dump_frames(tmp_path / "result.dump")
            np.testing.assert_array_equal(values[:, 0], np.arange(1, len(frame) + 1))
            rows = np.array(
                [
                    fields
                    for line in output.splitlines()
                    if len(fields := line.split()) == 8 and fields[0] == "0"
                ],
                dtype=float,
            )
            assert rows.shape == (1, 8), output
            pressure = rows[0, 2:] / BAR_PER_EV_ANGSTROM3
            virial = pressure * frame.get_volume()
            checks = {
                "total_energy": (rows[0, 1], expected["total_energy"]),
                "atomic_energy": (values[:, 1], expected["atomic_energy"].ravel()),
                "forces": (values[:, 2:5], expected["forces"]),
                "atom_virial": (
                    -values[:, 5:14] / BAR_PER_EV_ANGSTROM3,
                    expected["atom_virial"].reshape(-1, 9)[:, CENTROID],
                ),
                "virial": (virial, expected["virial"].reshape(9)[VOIGT]),
                "stress": (-pressure, expected["stress"].reshape(9)[VOIGT]),
            }
            errors = {}
            for key, (actual, target) in checks.items():
                error = float(np.max(np.abs(actual - target)))
                errors[key] = error
            uncertainty = check_fields(
                values[:, 14:],
                expected,
                "f",
                evidence_tolerance,
                normalized=precision == "float32",
            )
            maxima.append(dict(frame=frame_id, physical=errors, evidence=uncertainty))
    result = dict(
        precision=precision,
        shared_coordinate_dtype=precision,
        checkpoint=spec["checkpoint_sha256"],
        errors=maxima,
    )
    (tmp_path / "trained-parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print("AUDIT_TRAINED_PARITY", json.dumps(result))
    for row in maxima:
        for field, error in row["physical"].items():
            assert (
                error < tolerance if precision == "float32" else error <= tolerance
            ), (row["frame"], field, error)
