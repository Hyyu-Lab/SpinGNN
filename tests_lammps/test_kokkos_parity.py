"""Kokkos SERIAL/CUDA versus host spin/Allegro pair-style parity."""

from __future__ import annotations

import dataclasses
import math
import shlex
import subprocess

import numpy as np
import pytest

from helpers import (
    CASES,
    FIXTURE_ROOT,
    columns,
    parse_dump,
    parse_thermo,
    run_case,
    run_lammps,
    write_cutoff_boundary_data,
)
from tolerances import KOKKOS_ATOL, KOKKOS_RTOL, MD_STEPS


KOKKOS_CASE_NAMES = ("nosoc", "soc")


def _kokkos_spin_data(case, tmp_path):
    """Translate the current spin data row to this fork's legacy /kk order."""
    output = tmp_path / f"{case.name}-kokkos.data"
    converted = []
    in_atoms = False
    for line in case.data_path.read_text().splitlines():
        if line.startswith("Atoms # spin"):
            in_atoms = True
            converted.append(line)
            continue
        fields = line.split()
        if in_atoms and len(fields) == 9 and fields[0].isdigit():
            atom_id, atom_type, x, y, z, sx, sy, sz, magnitude = fields
            line = " ".join((atom_id, atom_type, magnitude, x, y, z, sx, sy, sz))
        converted.append(line)
    output.write_text("\n".join(converted) + "\n")
    return output


def _kokkos_wrapper(lammps_bin, lammps_env, tmp_path):
    help_result = subprocess.run(
        [str(lammps_bin), "-h"],
        env=lammps_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    if help_result.returncode != 0 or "spin/allegro_spin/kk" not in help_result.stdout:
        pytest.skip("configured LAMMPS binary has no spin/allegro_spin/kk style")
    kokkos_args = ["-k", "on"]
    for line in help_result.stdout.splitlines():
        if line.startswith("KOKKOS package API:") and "CUDA" in line.split():
            kokkos_args.extend(("g", "1"))
            break
    kokkos_args.extend(("-sf", "kk"))
    wrapper = tmp_path / "lmp-kokkos"
    wrapper.write_text(
        "#!/bin/sh\n" f'exec {shlex.join([str(lammps_bin), *kokkos_args])} "$@"\n'
    )
    wrapper.chmod(0o755)
    return wrapper


@pytest.mark.lammps
@pytest.mark.parametrize("case_name", KOKKOS_CASE_NAMES)
def test_kokkos_pair_matches_host_run0(
    case_name,
    model_paths,
    lammps_bin,
    lammps_env,
    tmp_path,
):
    case = CASES[case_name]
    kokkos_bin = _kokkos_wrapper(lammps_bin, lammps_env, tmp_path)
    kokkos_case = dataclasses.replace(
        case,
        data_name=str(_kokkos_spin_data(case, tmp_path)),
    )
    host = run_case(
        case=case,
        model_path=model_paths[case_name],
        binary=lammps_bin,
        workdir=tmp_path / "host",
        env=lammps_env,
    )
    kokkos = run_case(
        case=kokkos_case,
        model_path=model_paths[case_name],
        binary=kokkos_bin,
        workdir=tmp_path / "kokkos",
        env=lammps_env,
    )

    assert math.isclose(
        kokkos.energy_ev,
        host.energy_ev,
        rel_tol=KOKKOS_RTOL,
        abs_tol=KOKKOS_ATOL,
    )
    np.testing.assert_allclose(
        kokkos.forces_ev_a,
        host.forces_ev_a,
        rtol=KOKKOS_RTOL,
        atol=KOKKOS_ATOL,
    )
    assert host.spin_forces_ev is not None
    assert kokkos.spin_forces_ev is not None
    np.testing.assert_allclose(
        kokkos.spin_forces_ev,
        host.spin_forces_ev,
        rtol=KOKKOS_RTOL,
        atol=KOKKOS_ATOL,
    )
    assert host.atom_virial_ev is not None
    assert kokkos.atom_virial_ev is not None
    np.testing.assert_allclose(
        kokkos.atom_virial_ev,
        host.atom_virial_ev,
        rtol=KOKKOS_RTOL,
        atol=KOKKOS_ATOL,
    )


@pytest.mark.compile
@pytest.mark.lammps
def test_kokkos_preserves_non_extensive_energy_shift(
    shifted_artifact_factory,
    lammps_bin,
    lammps_pt2_target_device,
    lammps_env,
    tmp_path,
):
    case = CASES["nosoc"]
    kokkos_bin = _kokkos_wrapper(lammps_bin, lammps_env, tmp_path)
    kokkos_case = dataclasses.replace(
        case,
        data_name=str(_kokkos_spin_data(case, tmp_path)),
    )
    # Same latent mismatch as test_mpi_parity: the PT2 target must follow the
    # binary, not a hardcoded "cpu" (FIX3-D1 rejects a mismatch before loading).
    for artifact in shifted_artifact_factory(lammps_pt2_target_device):
        host = run_case(
            case=case,
            model_path=artifact,
            binary=lammps_bin,
            workdir=tmp_path / f"host-{artifact.suffix[1:]}",
            env=lammps_env,
        )
        kokkos = run_case(
            case=kokkos_case,
            model_path=artifact,
            binary=kokkos_bin,
            workdir=tmp_path / f"kokkos-{artifact.suffix[1:]}",
            env=lammps_env,
        )
        assert math.isclose(
            kokkos.energy_ev,
            host.energy_ev,
            rel_tol=KOKKOS_RTOL,
            abs_tol=KOKKOS_ATOL,
        )


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.parametrize("boundary_distance", (3.999, 4.0, 4.001))
def test_kokkos_cutoff_boundary_matches_host_for_torchscript_and_pt2(
    boundary_distance,
    pt2_artifact_factory,
    lammps_bin,
    lammps_pt2_target_device,
    lammps_env,
    tmp_path,
):
    case = CASES["nosoc"]
    kokkos_bin = _kokkos_wrapper(lammps_bin, lammps_env, tmp_path)
    host_case = dataclasses.replace(
        case,
        data_name=str(
            write_cutoff_boundary_data(
                tmp_path / "host-cutoff-boundary.data", boundary_distance
            )
        ),
    )
    kokkos_case = dataclasses.replace(
        case,
        data_name=str(
            write_cutoff_boundary_data(
                tmp_path / "kokkos-cutoff-boundary.data",
                boundary_distance,
                kokkos_order=True,
            )
        ),
    )
    # The PT2 target must follow the binary (FIX3-D1 rejects mismatches before
    # loading); a hardcoded "cpu" made the opt-in Kokkos CUDA lane fail on the
    # artifact instead of testing cutoff parity (RV4D-03).
    for artifact in pt2_artifact_factory(case.name, lammps_pt2_target_device):
        host = run_case(
            case=host_case,
            model_path=artifact,
            binary=lammps_bin,
            workdir=tmp_path / f"host-{artifact.suffix[1:]}",
            env=lammps_env,
        )
        kokkos = run_case(
            case=kokkos_case,
            model_path=artifact,
            binary=kokkos_bin,
            workdir=tmp_path / f"kokkos-{artifact.suffix[1:]}",
            env=lammps_env,
        )
        assert math.isclose(
            kokkos.energy_ev,
            host.energy_ev,
            rel_tol=KOKKOS_RTOL,
            abs_tol=KOKKOS_ATOL,
        )
        np.testing.assert_allclose(
            kokkos.forces_ev_a,
            host.forces_ev_a,
            rtol=KOKKOS_RTOL,
            atol=KOKKOS_ATOL,
        )


@pytest.mark.lammps
def test_kokkos_soc_spin_sib_trajectory_matches_host(
    model_paths,
    lammps_bin,
    lammps_env,
    tmp_path,
):
    case = CASES["soc"]
    kokkos_bin = _kokkos_wrapper(lammps_bin, lammps_env, tmp_path)
    kokkos_data = _kokkos_spin_data(case, tmp_path)

    def run(binary, name):
        workdir = tmp_path / name
        dump_path = workdir / "soc-md.dump"
        process = run_lammps(
            binary=binary,
            input_path=FIXTURE_ROOT / "in" / "spin_md.in",
            workdir=workdir,
            env=lammps_env,
            variables={
                "data": kokkos_data if name == "kokkos" else case.data_path,
                "model": model_paths[case.name],
                "dump": dump_path,
            },
            dump_path=dump_path,
        )
        return parse_dump(dump_path), parse_thermo(process.log_path)

    host_frames, host_thermo = run(lammps_bin, "host")
    kokkos_frames, kokkos_thermo = run(kokkos_bin, "kokkos")
    assert len(host_frames) == len(kokkos_frames) == MD_STEPS + 1
    assert len(host_thermo) == len(kokkos_thermo) == MD_STEPS + 1

    host_steps = np.asarray([row["Step"] for row in host_thermo])
    kokkos_steps = np.asarray([row["Step"] for row in kokkos_thermo])
    np.testing.assert_array_equal(kokkos_steps, host_steps)
    for key in ("PotEng", "TotEng", "Temp", "Fmax"):
        np.testing.assert_allclose(
            [row[key] for row in kokkos_thermo],
            [row[key] for row in host_thermo],
            rtol=KOKKOS_RTOL,
            atol=KOKKOS_ATOL,
        )

    trajectory_columns = (
        "x",
        "y",
        "z",
        "fx",
        "fy",
        "fz",
        *(f"c_spinprops[{index}]" for index in range(1, 8)),
    )
    for host_frame, kokkos_frame in zip(host_frames, kokkos_frames):
        assert host_frame["step"] == kokkos_frame["step"]
        np.testing.assert_allclose(
            columns(kokkos_frame, trajectory_columns),
            columns(host_frame, trajectory_columns),
            rtol=KOKKOS_RTOL,
            atol=KOKKOS_ATOL,
        )
