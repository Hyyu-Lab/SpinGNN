"""Exercise the Kokkos box-extents reducer through shrink-wrapped geometry."""

import numpy as np
import pytest

from helpers import parse_thermo, run_lammps
from test_kokkos_parity import _kokkos_wrapper
from tolerances import KOKKOS_ATOL, KOKKOS_RTOL


@pytest.mark.lammps
def test_kokkos_shrink_wrapped_box_matches_host(lammps_bin, lammps_env, tmp_path):
    kokkos_bin = _kokkos_wrapper(lammps_bin, lammps_env, tmp_path)
    script = tmp_path / "shrink.in"
    script.write_text(
        "units metal\n"
        "atom_style atomic\n"
        "boundary s s s\n"
        "region initial block 0 10 0 10 0 10\n"
        "create_box 1 initial\n"
        "create_atoms 1 single 1 2 3\n"
        "create_atoms 1 single 7 8 9\n"
        "create_atoms 1 single 4 4 4\n"
        "mass 1 1.0\n"
        "pair_style zero 2.0\n"
        "pair_coeff * *\n"
        "thermo_style custom step pe atoms xlo xhi ylo yhi zlo zhi\n"
        "thermo_modify format float %.16g\n"
        "run 0\n"
        "displace_atoms all move 0.25 -0.5 0.75 units box\n"
        "run 0\n"
    )
    rows = []
    for label, binary in (("host", lammps_bin), ("kokkos", kokkos_bin)):
        result = run_lammps(
            binary=binary,
            input_path=script,
            workdir=tmp_path / label,
            env=lammps_env,
            variables={},
        )
        rows.append(parse_thermo(result.log_path))
    assert len(rows[0]) == len(rows[1]) == 2
    for host, kokkos in zip(*rows):
        assert host["Atoms"] == kokkos["Atoms"] == 3
        keys = ("Xlo", "Xhi", "Ylo", "Yhi", "Zlo", "Zhi")
        np.testing.assert_allclose(
            [kokkos[key] for key in keys],
            [host[key] for key in keys],
            atol=KOKKOS_ATOL,
            rtol=KOKKOS_RTOL,
        )
    assert rows[1][-1]["Xlo"] > rows[1][0]["Xlo"]
    assert rows[1][-1]["Ylo"] < rows[1][0]["Ylo"]


@pytest.mark.lammps
def test_kokkos_langevin_reducers_match_zero_temperature_host(
    lammps_bin, lammps_env, tmp_path
):
    kokkos_bin = _kokkos_wrapper(lammps_bin, lammps_env, tmp_path)
    script = tmp_path / "langevin.in"
    # Zero temperature removes stochastic differences while still exercising
    # the zero-force and energy-tally reducers on every integration step.
    script.write_text(
        "units metal\n"
        "atom_style atomic\n"
        "boundary p p p\n"
        "region initial block 0 10 0 10 0 10\n"
        "create_box 1 initial\n"
        "create_atoms 1 single 2 2 2\n"
        "create_atoms 1 single 7 7 7\n"
        "mass 1 1.0\n"
        "velocity all set 0.1 -0.2 0.3\n"
        "pair_style zero 2.0\n"
        "pair_coeff * *\n"
        "fix integrate all nve\n"
        "fix heat all langevin 0.0 0.0 0.1 12345 tally yes zero yes\n"
        "timestep 0.001\n"
        "thermo 10\n"
        "thermo_style custom step pe ke f_heat\n"
        "thermo_modify format float %.16g\n"
        "run 20\n"
    )
    rows = []
    for label, binary in (("host", lammps_bin), ("kokkos", kokkos_bin)):
        result = run_lammps(
            binary=binary,
            input_path=script,
            workdir=tmp_path / label,
            env=lammps_env,
            variables={},
        )
        rows.append(parse_thermo(result.log_path))
    assert len(rows[0]) == len(rows[1]) == 3
    for host, kokkos in zip(*rows):
        np.testing.assert_allclose(
            [kokkos["KinEng"], kokkos["f_heat"]],
            [host["KinEng"], host["f_heat"]],
            atol=KOKKOS_ATOL,
            rtol=KOKKOS_RTOL,
        )
    assert 0 < rows[1][-1]["KinEng"] < rows[1][0]["KinEng"]
