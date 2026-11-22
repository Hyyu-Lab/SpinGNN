"""Short spin/sib stability and spin-normalization test."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import CASES, FIXTURE_ROOT, columns, parse_dump, parse_thermo, run_lammps
from tolerances import (
    MAGNETIC_SPIN_MIN,
    MD_ENERGY_DRIFT_ATOL_EV,
    MD_SPIN_NORM_ATOL,
    MD_STEPS,
    SERIAL_RTOL,
)


@pytest.mark.lammps
def test_soc_spin_sib_twenty_steps_are_finite_and_bounded(
    model_paths,
    lammps_bin,
    lammps_env,
    tmp_path,
):
    case = CASES["soc"]
    dump_path = tmp_path / "soc-md.dump"
    process = run_lammps(
        binary=lammps_bin,
        input_path=FIXTURE_ROOT / "in" / "spin_md.in",
        workdir=tmp_path,
        env=lammps_env,
        variables={
            "data": case.data_path,
            "model": model_paths[case.name],
            "dump": dump_path,
        },
        dump_path=dump_path,
    )
    frames = parse_dump(dump_path)
    thermo = parse_thermo(process.log_path)

    assert max(int(row["Step"]) for row in thermo) == MD_STEPS
    assert len(frames) == MD_STEPS + 1
    energies = np.asarray([row["PotEng"] for row in thermo], dtype=float)
    assert np.isfinite(energies).all()
    assert float(np.ptp(energies)) <= MD_ENERGY_DRIFT_ATOL_EV

    for frame in frames:
        values = columns(
            frame,
            (
                "x",
                "y",
                "z",
                "fx",
                "fy",
                "fz",
                *(f"c_spinprops[{index}]" for index in range(1, 8)),
            ),
        )
        assert np.isfinite(values).all()
        directions = columns(
            frame, tuple(f"c_spinprops[{index}]" for index in range(1, 4))
        )
        magnitudes = columns(frame, ("c_spinprops[4]",)).reshape(-1)
        magnetic = magnitudes > MAGNETIC_SPIN_MIN
        assert magnetic.any()
        norms = np.linalg.norm(directions[magnetic], axis=1)
        np.testing.assert_allclose(
            norms,
            np.ones_like(norms),
            rtol=SERIAL_RTOL,
            atol=MD_SPIN_NORM_ATOL,
        )
