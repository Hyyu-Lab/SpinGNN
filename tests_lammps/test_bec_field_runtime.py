"""Analytic BEC electric-field checks independent of a trained model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helpers import columns, parse_dump, parse_thermo, run_lammps
from tolerances import E_ATOL_EV, F_ATOL_EV_A, SERIAL_RTOL


FIELD = np.asarray((0.2, -0.1, 0.3))
# A non-symmetric tensor distinguishes Z E from the required Z^T E.
BEC = np.asarray(((1.0, 0.2, -0.4), (0.7, 1.3, 0.5), (-0.2, 0.8, 1.7)))
CHARGES = np.asarray((0.4, -0.4))


def _field_run(binary, env, directory: Path, coupling: str, *, prefix=(), ranks=1):
    directory.mkdir(parents=True, exist_ok=True)
    tensors = np.stack((BEC, -BEC))
    bec_path = directory / "fixed.bec"
    bec_path.write_text(
        "\n".join(
            " ".join((str(index + 1), *(f"{value:.16g}" for value in tensor.flat)))
            for index, tensor in enumerate(tensors)
        )
        + "\n"
    )
    dump_path = directory / "trajectory.dump"
    reset_dump = directory / "reset.dump"
    restart_dump = directory / "restart.dump"
    restart_path = directory / "state.restart"
    properties = "compute dip all property/atom becdipolex becdipoley becdipolez"
    field_fix = f"fix field all efield 0.2 -0.1 0.3 coupling {coupling}"
    dump_fields = "id type xu yu zu fx fy fz c_dip[1] c_dip[2] c_dip[3]"
    input_path = directory / "field.in"
    input_path.write_text(
        f"""units metal
boundary p p p
atom_style hybrid charge bec
atom_modify map hash
processors {ranks} 1 1
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 9.99 2 2 units box
create_atoms 1 single 4.99 6 6 units box
mass 1 100
set atom 1 charge 0.4
set atom 2 charge -0.4
read_bec {bec_path}
pair_style zero 4.0
pair_coeff * *
{field_fix}
fix_modify field energy yes
{properties}
velocity all set 20 -10 5 units box
fix integrate all nve
neighbor 0.1 bin
neigh_modify every 1 delay 0 check no
timestep 0.001
dump trajectory all custom 1 {dump_path} {dump_fields}
dump_modify trajectory sort id format float %.16g
thermo 1
thermo_style custom step pe f_field
thermo_modify format float %.16g
run 2
write_restart {restart_path}
undump trajectory
bec_dipole_reset
dump reset all custom 1 {reset_dump} {dump_fields}
dump_modify reset sort id format float %.16g
run 0
clear
units metal
read_restart {restart_path}
{field_fix}
fix_modify field energy yes
{properties}
dump restored all custom 1 {restart_dump} {dump_fields}
dump_modify restored sort id format float %.16g
thermo 1
thermo_style custom step pe f_field
thermo_modify format float %.16g
run 0
"""
    )
    result = run_lammps(
        binary=binary,
        input_path=input_path,
        workdir=directory,
        env=env,
        variables={},
        prefix=prefix,
        dump_path=dump_path,
    )
    frames = parse_dump(dump_path)
    initial, final = frames[0], frames[-1]
    restored = parse_dump(restart_dump)[-1]
    reset = parse_dump(reset_dump)[-1]
    expected_force = np.zeros((2, 3))
    if coupling in ("bec", "both"):
        expected_force += np.einsum("nij,i->nj", tensors, FIELD)
    if coupling in ("charge", "both"):
        expected_force += CHARGES[:, None] * FIELD
    for frame in (initial, final, restored, reset):
        np.testing.assert_allclose(
            columns(frame, ("fx", "fy", "fz")),
            expected_force,
            rtol=SERIAL_RTOL,
            atol=F_ATOL_EV_A,
        )
    position_columns = ("xu", "yu", "zu")
    dipole_columns = ("c_dip[1]", "c_dip[2]", "c_dip[3]")
    displacement = columns(final, position_columns) - columns(initial, position_columns)
    expected_dipole = np.einsum("nij,nj->ni", tensors, displacement)
    np.testing.assert_allclose(
        columns(final, dipole_columns),
        expected_dipole,
        rtol=SERIAL_RTOL,
        atol=E_ATOL_EV,
    )
    np.testing.assert_allclose(
        columns(restored, (*position_columns, *dipole_columns)),
        columns(final, (*position_columns, *dipole_columns)),
        rtol=SERIAL_RTOL,
        atol=E_ATOL_EV,
    )
    np.testing.assert_array_equal(columns(reset, dipole_columns), np.zeros((2, 3)))
    if coupling == "bec":
        expected_energy = -float(np.sum(expected_dipole @ FIELD))
        assert (
            abs(parse_thermo(result.log_path)[-1]["f_field"] - expected_energy)
            <= E_ATOL_EV
        )
    return columns(final, (*position_columns, *dipole_columns, "fx", "fy", "fz"))


@pytest.mark.lammps
@pytest.mark.parametrize("coupling", ("charge", "bec", "both"))
def test_nonzero_field_uses_tensor_transpose_and_preserves_bec_restart(
    coupling, lammps_bin, lammps_env, tmp_path
):
    _field_run(lammps_bin, lammps_env, tmp_path, coupling)


@pytest.mark.lammps
@pytest.mark.mpi
def test_bec_dipole_survives_periodic_wrap_and_mpi_migration(
    lammps_bin, lammps_mpi_bin, lammps_env, mpi_prefix, tmp_path
):
    serial = _field_run(lammps_bin, lammps_env, tmp_path / "serial", "bec")
    parallel = _field_run(
        lammps_mpi_bin,
        lammps_env,
        tmp_path / "mpi",
        "bec",
        prefix=mpi_prefix(2),
        ranks=2,
    )
    np.testing.assert_allclose(parallel, serial, rtol=SERIAL_RTOL, atol=E_ATOL_EV)
