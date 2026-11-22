"""Regression tests for energy offsets and hybrid spin-pair ownership."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from helpers import (
    CASES,
    atomic_input,
    cloned,
    lammps_parity_result,
    run_case,
    run_lammps,
)
from nequip.data import AtomicDataDict
from nequip.scripts import deploy
from tolerances import (
    E_ATOL_EV,
    F_ATOL_EV_A,
    HYBRID_ORDER_ATOL_BY_DEVICE,
    SERIAL_RTOL,
    SPIN_F_ATOL,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.compile
@pytest.mark.lammps
def test_non_extensive_energy_shift_is_preserved_by_torchscript_and_pt2(
    shifted_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES["nosoc"]
    scripted_path, pt2_path = shifted_artifact_factory(lammps_pt2_target_device)
    model, _ = deploy.load_deployed_model(
        scripted_path,
        device="cpu",
        freeze=True,
        set_global_options=False,
    )
    output = model(cloned(atomic_input(case)))
    total = float(output[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(-1)[0])
    atomic_sum = float(output[AtomicDataDict.PER_ATOM_ENERGY_KEY].sum())
    assert math.isclose(total - atomic_sum, 13.0, rel_tol=0.0, abs_tol=2.0e-5)

    for name, artifact in (("torchscript", scripted_path), ("pt2", pt2_path)):
        result = run_case(
            case=case,
            model_path=artifact,
            binary=lammps_bin,
            workdir=tmp_path / name,
            env=lammps_env,
        )
        assert abs(result.energy_ev - total) <= E_ATOL_EV


def _write_hybrid_input(path: Path, nequip_first: bool) -> Path:
    styles = (
        "spin/nequip_spin lj/cut 4.0" if nequip_first else "lj/cut 4.0 spin/nequip_spin"
    )
    path.write_text(
        f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton off
read_data ${{data}}

pair_style hybrid/overlay {styles}
pair_coeff * * spin/nequip_spin ${{model}} Mn Ge
pair_coeff * * lj/cut 0.01 2.0 4.0

compute spinprops all property/atom spx spy spz sp fmx fmy fmz
dump parity all custom 1 ${{dump}} id type x y z fx fy fz &
  c_spinprops[1] c_spinprops[2] c_spinprops[3] c_spinprops[4] &
  c_spinprops[5] c_spinprops[6] c_spinprops[7]
dump_modify parity sort id format float %.16g
thermo 1
thermo_style custom step pe etotal fmax
thermo_modify format float %.16g
run 0
"""
    )
    return path


@pytest.mark.lammps
def test_nequip_spin_hybrid_overlay_is_order_invariant(
    model_paths,
    lammps_bin,
    lammps_pt2_target_device,
    lammps_env,
    tmp_path,
):
    atol = HYBRID_ORDER_ATOL_BY_DEVICE[lammps_pt2_target_device]
    case = CASES["nequip_spin"]
    results = []
    for nequip_first in (True, False):
        name = "nequip-first" if nequip_first else "nequip-last"
        workdir = tmp_path / name
        dump_path = workdir / "hybrid.dump"
        process = run_lammps(
            binary=lammps_bin,
            input_path=_write_hybrid_input(tmp_path / f"{name}.in", nequip_first),
            workdir=workdir,
            env=lammps_env,
            variables={
                "data": case.data_path,
                "model": model_paths[case.name],
                "dump": dump_path,
            },
            dump_path=dump_path,
        )
        results.append(lammps_parity_result(process, case))

    first, second = results
    assert math.isclose(
        first.energy_ev,
        second.energy_ev,
        rel_tol=0.0,
        abs_tol=atol,
    )
    np.testing.assert_allclose(
        first.forces_ev_a,
        second.forces_ev_a,
        rtol=0.0,
        atol=atol,
    )
    assert first.spin_forces_ev is not None
    assert second.spin_forces_ev is not None
    np.testing.assert_allclose(
        first.spin_forces_ev,
        second.spin_forces_ev,
        rtol=0.0,
        atol=atol,
    )


def _write_duplicate_nequip_input(path: Path) -> Path:
    path.write_text(
        """units metal
boundary p p p
atom_style spin
atom_modify map array
newton off
read_data ${data}

pair_style hybrid/overlay spin/nequip_spin spin/nequip_spin
pair_coeff * * spin/nequip_spin 1 ${model} Mn Ge
pair_coeff * * spin/nequip_spin 2 ${model} Mn Ge

compute spinprops all property/atom spx spy spz sp fmx fmy fmz
dump parity all custom 1 ${dump} id type x y z fx fy fz &
  c_spinprops[1] c_spinprops[2] c_spinprops[3] c_spinprops[4] &
  c_spinprops[5] c_spinprops[6] c_spinprops[7]
dump_modify parity sort id format float %.16g
thermo 1
thermo_style custom step pe etotal fmax
thermo_modify format float %.16g
run 0
"""
    )
    return path


@pytest.mark.lammps
def test_two_nequip_spin_substyles_sum_each_pair_contribution_once(
    model_paths,
    lammps_bin,
    lammps_env,
    tmp_path,
):
    case = CASES["nequip_spin"]
    single = run_case(
        case=case,
        model_path=model_paths[case.name],
        binary=lammps_bin,
        workdir=tmp_path / "single",
        env=lammps_env,
    )
    dump_path = tmp_path / "double/double.dump"
    process = run_lammps(
        binary=lammps_bin,
        input_path=_write_duplicate_nequip_input(tmp_path / "double.in"),
        workdir=tmp_path / "double",
        env=lammps_env,
        variables={
            "data": case.data_path,
            "model": model_paths[case.name],
            "dump": dump_path,
        },
        dump_path=dump_path,
    )
    double = lammps_parity_result(process, case)
    assert math.isclose(
        double.energy_ev,
        2.0 * single.energy_ev,
        rel_tol=0.0,
        abs_tol=E_ATOL_EV,
    )
    np.testing.assert_allclose(
        double.forces_ev_a,
        2.0 * single.forces_ev_a,
        rtol=SERIAL_RTOL,
        atol=F_ATOL_EV_A,
    )
    assert single.spin_forces_ev is not None
    assert double.spin_forces_ev is not None
    np.testing.assert_allclose(
        double.spin_forces_ev,
        2.0 * single.spin_forces_ev,
        rtol=SERIAL_RTOL,
        atol=SPIN_F_ATOL,
    )


def test_nequip_spin_single_pair_uses_only_pair_owned_buffer():
    source = (REPO_ROOT / "lammps/src/SPIN/pair_nequip_spin.cpp").read_text()
    body = source.split("void PairNEQUIP_spin::compute_single_pair", 1)[1]
    body = body.split("}", 1)[0]
    assert "spin_force_contribution[ii]" in body
    assert "atom->fm" not in body
