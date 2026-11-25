"""Exercise displacement-triggered neighbor rebuilding in thermal LAMMPS MD."""

import importlib.util
import json
import os
from pathlib import Path

import pytest


THERMAL_DRIVER = (
    Path(__file__).resolve().parents[1] / "benchmarks/bench_lammps_thermal_md.py"
)
spec = importlib.util.spec_from_file_location("thermal_driver", THERMAL_DRIVER)
thermal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(thermal)


def thermal_run_log(warmup_builds, total_builds):
    return f"""LAMMPS
SPINGNN2_THERMAL_PRODUCTION
   Step Atoms Temp PotEng KinEng TotEng c_timing[1] c_timing[5] c_timing[6]
      0     1  300     -2      1     -1           0           0           1
    200     1  300     -2      1     -1           1           0         201
Loop time of 2 on 1 procs for 200 steps with 1 atoms
Ave neighs/atom = 2
Neighbor list builds = {warmup_builds}
Dangerous builds = 0
   Step Atoms Temp PotEng KinEng TotEng c_timing[1] c_timing[5] c_timing[6]
    200     1  300     -2      1     -1           1           0         201
    290     1  300     -2      1     -1         1.5           0         291
Loop time of 1 on 1 procs for 90 steps with 1 atoms
Ave neighs/atom = 2
Neighbor list builds = {total_builds}
Dangerous builds = 0
"""


def test_thermal_production_rebuilds(tmp_path):
    for suffix, step in (("before", 200), ("after", 290)):
        (tmp_path / f"owners-{suffix}.dump").write_text(
            f"""ITEM: TIMESTEP
{step}
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 1
0 1
0 1
ITEM: ATOMS id proc
1 0
"""
        )
    result = thermal.read_md(
        thermal_run_log(9, 13), tmp_path, 1, 1, 0, 200, 90, 0.001, "nve"
    )
    assert result["warmup_neighbor_builds"] == 9
    assert result["neighbor_builds"] == 4


def test_thermal_rejects_warmup_only_rebuilds(tmp_path):
    with pytest.raises(RuntimeError, match="No displacement-triggered rebuilds"):
        thermal.read_md(thermal_run_log(1, 1), tmp_path, 1, 1, 0, 200, 90, 0.001, "nve")


@pytest.mark.parametrize(("mode", "skin"), (("nve", 0.4), ("sib", 1.5)))
def test_thermal_rebuilds(lmp_run, m4_backend, tmp_path, mode, skin):
    if m4_backend != "cuda":
        pytest.skip("Thermal skin rebuilding requires the CUDA pair style")
    fixtures = Path(os.environ["SPINGNN2_LAMMPS_MODELS"]).with_name(
        "performance-models.json"
    )
    if not fixtures.exists():
        pytest.skip("Prepare the full-width thermal benchmark model index")
    models = json.loads(fixtures.read_text())
    counts = os.getenv("SPINGNN2_M4_MPI_RANKS", "1").split(",")
    ranks = 2 if "2" in counts else 1
    model = "nosoc"
    cells = 8
    warmup = 8
    steps = 400
    snapshot = tmp_path / "equilibrated.data"
    seed, _ = thermal.md_input(
        models[model],
        model,
        1,
        cells,
        0.4,
        mode,
        8192,
        0,
        0,
        write_snapshot=True,
    )
    lmp_run(seed)
    script, dt = thermal.md_input(
        models[model],
        model,
        ranks,
        cells,
        skin,
        mode,
        0,
        warmup,
        steps,
        snapshot=snapshot,
    )
    output = lmp_run(script, ranks=ranks)
    result = thermal.read_md(
        output,
        tmp_path,
        ranks * cells**3,
        ranks,
        0,
        warmup,
        steps,
        dt,
        mode,
    )
    assert result["dangerous_builds"] == 0
    assert 225 < result["mean_timed_temperature_K"] < 375
    if mode == "nve":
        assert result["energy_range_eV_per_atom"] < 5e-5
