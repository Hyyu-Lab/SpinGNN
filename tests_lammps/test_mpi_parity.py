"""Serial versus one-, two-, and four-rank LAMMPS parity."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import (
    CASES,
    CASE_NAMES,
    FIXTURE_ROOT,
    columns,
    parse_dump,
    parse_thermo,
    run_case,
    run_lammps,
)
from tolerances import (
    MPI_BEC_ATOL_E,
    MPI_E_ATOL_EV,
    MPI_F_ATOL_EV_A,
    MPI_RTOL,
    MPI_SPIN_F_ATOL_EV,
    MD_STEPS,
)


@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_serial_matches_one_two_and_four_mpi_ranks(
    case_name: str,
    model_paths,
    lammps_bin,
    lammps_mpi_bin,
    mpi_prefix,
    lammps_env,
    tmp_path,
):
    case = CASES[case_name]
    serial = run_case(
        case=case,
        model_path=model_paths[case_name],
        binary=lammps_bin,
        workdir=tmp_path / "serial",
        env=lammps_env,
    )

    for ranks in (1, 2, 4):
        parallel = run_case(
            case=case,
            model_path=model_paths[case_name],
            binary=lammps_mpi_bin,
            workdir=tmp_path / f"mpi-{ranks}",
            env=lammps_env,
            prefix=mpi_prefix(ranks),
        )
        assert math.isclose(
            parallel.energy_ev,
            serial.energy_ev,
            rel_tol=MPI_RTOL,
            abs_tol=MPI_E_ATOL_EV,
        )
        np.testing.assert_allclose(
            parallel.forces_ev_a,
            serial.forces_ev_a,
            rtol=MPI_RTOL,
            atol=MPI_F_ATOL_EV_A,
        )
        if case.spin:
            assert serial.spin_forces_ev is not None
            assert parallel.spin_forces_ev is not None
            np.testing.assert_allclose(
                parallel.spin_forces_ev,
                serial.spin_forces_ev,
                rtol=MPI_RTOL,
                atol=MPI_SPIN_F_ATOL_EV,
            )
        if case.bec:
            assert serial.bec_e is not None
            assert parallel.bec_e is not None
            np.testing.assert_allclose(
                parallel.bec_e,
                serial.bec_e,
                rtol=MPI_RTOL,
                atol=MPI_BEC_ATOL_E,
            )


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("case_name", ("nosoc", "bec"))
def test_non_extensive_shift_matches_serial_on_two_and_four_ranks(
    case_name,
    shifted_artifact_factory,
    lammps_bin,
    lammps_mpi_bin,
    lammps_pt2_target_device,
    mpi_prefix,
    lammps_env,
    tmp_path,
):
    case = CASES[case_name]
    # LAMMPS rejects a PT2 package before loading if its target device differs
    # from the pair style's device.
    artifacts = shifted_artifact_factory(lammps_pt2_target_device, case_name=case_name)
    for artifact in artifacts:
        serial = run_case(
            case=case,
            model_path=artifact,
            binary=lammps_bin,
            workdir=tmp_path / f"serial-{artifact.suffix[1:]}",
            env=lammps_env,
        )
        for ranks in (2, 4):
            parallel = run_case(
                case=case,
                model_path=artifact,
                binary=lammps_mpi_bin,
                workdir=tmp_path / f"mpi-{artifact.suffix[1:]}-{ranks}",
                env=lammps_env,
                prefix=mpi_prefix(ranks),
            )
            assert math.isclose(
                parallel.energy_ev,
                serial.energy_ev,
                rel_tol=MPI_RTOL,
                abs_tol=MPI_E_ATOL_EV,
            )


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("backend_index", (0, 1), ids=("torchscript", "pt2"))
def test_bec_torchscript_and_pt2_match_one_two_and_four_mpi_ranks(
    backend_index,
    pt2_artifact_factory,
    lammps_bin,
    lammps_mpi_bin,
    lammps_pt2_target_device,
    mpi_prefix,
    lammps_env,
    tmp_path,
):
    case = CASES["bec"]
    artifacts = pt2_artifact_factory(case.name, lammps_pt2_target_device)
    artifact = artifacts[backend_index]
    backend = "torchscript" if backend_index == 0 else "pt2"
    serial = run_case(
        case=case,
        model_path=artifact,
        binary=lammps_bin,
        workdir=tmp_path / f"serial-{backend}",
        env=lammps_env,
    )

    for ranks in (1, 2, 4):
        parallel = run_case(
            case=case,
            model_path=artifact,
            binary=lammps_mpi_bin,
            workdir=tmp_path / f"mpi-{backend}-{ranks}",
            env=lammps_env,
            prefix=mpi_prefix(ranks),
        )
        assert math.isclose(
            parallel.energy_ev,
            serial.energy_ev,
            rel_tol=MPI_RTOL,
            abs_tol=MPI_E_ATOL_EV,
        )
        np.testing.assert_allclose(
            parallel.forces_ev_a,
            serial.forces_ev_a,
            rtol=MPI_RTOL,
            atol=MPI_F_ATOL_EV_A,
        )
        assert serial.bec_e is not None
        assert parallel.bec_e is not None
        np.testing.assert_allclose(
            parallel.bec_e,
            serial.bec_e,
            rtol=MPI_RTOL,
            atol=MPI_BEC_ATOL_E,
        )


@pytest.mark.lammps
@pytest.mark.mpi
def test_spin_sib_trajectory_matches_serial_on_two_and_four_ranks(
    model_paths,
    lammps_bin,
    lammps_mpi_bin,
    mpi_prefix,
    lammps_env,
    tmp_path,
):
    case = CASES["soc"]

    def run(binary, name, prefix=()):
        workdir = tmp_path / name
        dump_path = workdir / "soc-md.dump"
        process = run_lammps(
            binary=binary,
            input_path=FIXTURE_ROOT / "in/spin_sib_mpi.in",
            workdir=workdir,
            env=lammps_env,
            prefix=prefix,
            variables={
                "data": case.data_path,
                "model": model_paths[case.name],
                "dump": dump_path,
            },
            dump_path=dump_path,
        )
        return parse_dump(dump_path), parse_thermo(process.log_path)

    serial_frames, serial_thermo = run(lammps_bin, "serial")
    assert len(serial_frames) == len(serial_thermo) == MD_STEPS + 1
    trajectory_columns = (
        "x",
        "y",
        "z",
        "fx",
        "fy",
        "fz",
        *(f"c_spinprops[{index}]" for index in range(1, 8)),
    )
    for ranks in (2, 4):
        frames, thermo = run(
            lammps_mpi_bin,
            f"mpi-{ranks}",
            mpi_prefix(ranks),
        )
        assert len(frames) == len(thermo) == MD_STEPS + 1
        np.testing.assert_array_equal(
            [row["Step"] for row in thermo],
            [row["Step"] for row in serial_thermo],
        )
        for frame, reference in zip(frames, serial_frames):
            assert frame["step"] == reference["step"]
            np.testing.assert_allclose(
                columns(frame, trajectory_columns),
                columns(reference, trajectory_columns),
                rtol=MPI_RTOL,
                atol=MPI_F_ATOL_EV_A,
            )
