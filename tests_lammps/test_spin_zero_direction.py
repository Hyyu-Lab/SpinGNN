"""Zero spin directions in atom_style spin input (bulk2 NaN report).

read_data and ``set ... spin`` used to normalize the direction by 1/sqrt(0).
The NaN direction of a non-magnetic atom slipped through run 0 (the ML spin
pair styles gate on the magnitude) and spin/lbfgs then spread it to every spin
through its global reductions. A zero-magnitude atom now gets the placeholder
direction (0, 0, 1); a magnetic atom with a zero direction is rejected.

Classical spin/exchange keeps this model- and GPU-free: it proves the input
contract and minimizer finiteness, not any ML-potential result.
"""

from __future__ import annotations

import numpy as np
import pytest

from helpers import columns, parse_dump, parse_thermo, run_lammps
from tolerances import MD_SPIN_NORM_ATOL, SERIAL_RTOL


# id type x y z spx spy spz sp
ZERO_ROW = "1 1 0.5 0.5 0.5 0.0 0.0 0.0 0.0"
OTHER_ROWS = (
    "2 2 2.5 0.5 0.5 1.0 0.0 0.0 1.4",
    "3 2 0.5 2.5 0.5 0.0 1.0 0.0 1.4",
    "4 1 2.5 2.5 2.5 1.0 0.0 0.0 0.0",
)
SPIN_COLUMNS = ("c_sp[1]", "c_sp[2]", "c_sp[3]", "c_sp[4]")


def _write_case(tmp_path, first_row: str, set_command: str):
    data = tmp_path / "zero.data"
    data.write_text(
        "zero spin direction regression\n\n"
        "4 atoms\n2 atom types\n\n"
        "0.0 5.0 xlo xhi\n0.0 5.0 ylo yhi\n0.0 5.0 zlo zhi\n\n"
        "Masses\n\n1 87.62\n2 101.07\n\n"
        "Atoms # spin\n\n" + "\n".join((first_row, *OTHER_ROWS)) + "\n"
    )
    script = tmp_path / "zero.in"
    script.write_text(
        f"""units metal
atom_style spin
atom_modify map array
boundary p p p
read_data ${{data}}
{set_command}
pair_style spin/exchange 3.0
pair_coeff * * exchange 3.0 0.0446928 0.003496 1.4885
neighbor 0.1 bin
compute sp all property/atom spx spy spz sp
dump d all custom 1 ${{dump}} id c_sp[1] c_sp[2] c_sp[3] c_sp[4]
dump_modify d sort id format float %.16g
thermo 1
thermo_style custom step pe etotal
thermo_modify format float %.16g
min_style spin/lbfgs
minimize 0.0 1.0e-12 20 20
"""
    )
    return data, script


def _run(lammps_bin, lammps_env, tmp_path, first_row, set_command, check):
    data, script = _write_case(tmp_path, first_row, set_command)
    dump = tmp_path / "zero.dump"
    process = run_lammps(
        binary=lammps_bin,
        input_path=script,
        workdir=tmp_path,
        env=lammps_env,
        variables={"data": data, "dump": dump},
        dump_path=dump,
        check=check,
    )
    return process, dump


@pytest.mark.lammps
def test_zero_magnitude_atoms_get_unit_direction_and_minimize_finite(
    lammps_bin, lammps_env, tmp_path
):
    process, dump = _run(
        lammps_bin,
        lammps_env,
        tmp_path,
        ZERO_ROW,
        "set atom 4 spin 0.0 0.0 0.0 0.0",
        check=True,
    )
    # Both parsers raise on any non-finite row.
    thermo = parse_thermo(process.log_path)
    frames = parse_dump(dump)

    assert max(int(row["Step"]) for row in thermo) > 0
    initial = columns(frames[0], SPIN_COLUMNS)
    np.testing.assert_array_equal(initial[[0, 3]], [[0.0, 0.0, 1.0, 0.0]] * 2)
    for frame in frames:
        directions = columns(frame, SPIN_COLUMNS[:3])
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(
            norms, np.ones_like(norms), rtol=SERIAL_RTOL, atol=MD_SPIN_NORM_ATOL
        )


@pytest.mark.lammps
@pytest.mark.parametrize(
    ("first_row", "set_command", "message"),
    (
        (
            "1 1 0.5 0.5 0.5 0.0 0.0 0.0 1.4",
            "",
            "Atom 1 in Atoms section of data file has spin magnitude 1.4 "
            "but zero spin direction",
        ),
        (
            ZERO_ROW,
            "set atom 2 spin 1.4 0.0 0.0 0.0",
            "Cannot set spin magnitude 1.4 with zero spin direction for atom 2",
        ),
    ),
    ids=("read_data", "set"),
)
def test_magnetic_atom_with_zero_direction_is_rejected(
    first_row, set_command, message, lammps_bin, lammps_env, tmp_path
):
    process, _ = _run(
        lammps_bin, lammps_env, tmp_path, first_row, set_command, check=False
    )
    assert process.returncode != 0
    assert message in process.output
