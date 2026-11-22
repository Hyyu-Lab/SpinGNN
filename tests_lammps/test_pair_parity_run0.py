"""Single-frame Python versus LAMMPS parity tests."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import CASES, CASE_NAMES, python_reference, run_case
from tolerances import (
    BEC_ATOL_E,
    E_ATOL_EV,
    F_ATOL_EV_A,
    SERIAL_RTOL,
    SPIN_F_ATOL,
    VIRIAL_ATOL_EV,
)


@pytest.mark.lammps
@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_pair_python_lammps_run0_parity(
    case_name: str,
    model_paths,
    lammps_bin,
    lammps_env,
    tmp_path,
):
    case = CASES[case_name]
    python = python_reference(model_paths[case_name], case)
    lammps = run_case(
        case=case,
        model_path=model_paths[case_name],
        binary=lammps_bin,
        workdir=tmp_path,
        env=lammps_env,
    )

    assert abs(python.energy_ev - lammps.energy_ev) <= E_ATOL_EV
    np.testing.assert_allclose(
        lammps.forces_ev_a,
        python.forces_ev_a,
        rtol=SERIAL_RTOL,
        atol=F_ATOL_EV_A,
    )
    if case.spin:
        assert python.spin_forces_ev is not None
        assert lammps.spin_forces_ev is not None
        np.testing.assert_allclose(
            lammps.spin_forces_ev,
            python.spin_forces_ev,
            rtol=SERIAL_RTOL,
            atol=SPIN_F_ATOL,
        )
    if python.atom_virial_ev is not None and lammps.atom_virial_ev is not None:
        np.testing.assert_allclose(
            lammps.atom_virial_ev,
            python.atom_virial_ev,
            rtol=SERIAL_RTOL,
            atol=VIRIAL_ATOL_EV,
        )
    if case.bec:
        assert python.bec_e is not None
        assert lammps.bec_e is not None
        np.testing.assert_allclose(
            lammps.bec_e,
            python.bec_e,
            rtol=SERIAL_RTOL,
            atol=BEC_ATOL_E,
        )
