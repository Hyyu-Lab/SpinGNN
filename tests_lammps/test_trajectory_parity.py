"""Per-frame trajectory parity: Python TS/PT2 versus serial/MPI LAMMPS.

Python references are evaluated on every frame's own state (own cell/PBC/
types/spins); serial versus MPI is a separate deterministic gate that uses
the frozen fixtures because ``velocity create`` is processor-count dependent.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from helpers import (
    CASES,
    CASE_NAMES,
    MD_STEPS,
    TrajectoryFrame,
    TrajectoryScopeError,
    assert_trajectory_parity,
    parse_trajectory_dump,
    python_trajectory_reference,
    require_observables,
    run_trajectory_case,
    validate_trajectory,
)
from tolerances import (
    BEC_ATOL_E,
    E_ATOL_EV,
    F_ATOL_EV_A,
    MPI_BEC_ATOL_E,
    MPI_E_ATOL_EV,
    MPI_F_ATOL_EV_A,
    MPI_RTOL,
    MPI_SPIN_F_ATOL_EV,
    PT2_LAMMPS_F_ATOL_EV_A,
    PT2_LAMMPS_SPIN_F_ATOL_EV,
    SERIAL_RTOL,
    SPIN_F_ATOL,
    VIRIAL_ATOL_EV,
)

SPIN_CASE_NAMES = ("nosoc", "soc", "nequip_spin")


def _reference_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.lammps
@pytest.mark.parametrize("case_name", SPIN_CASE_NAMES)
@pytest.mark.parametrize("frozen", (False, True), ids=("moving", "frozen"))
def test_python_torchscript_matches_serial_trajectory(
    case_name,
    frozen,
    model_paths,
    lammps_bin,
    lammps_env,
    tmp_path,
):
    case = CASES[case_name]
    serial = run_trajectory_case(
        case_name,
        model_paths[case_name],
        lammps_bin,
        tmp_path / "serial",
        lammps_env,
        frozen=frozen,
    )
    reference = python_trajectory_reference(
        case,
        serial.frames,
        {"torchscript": model_paths[case_name]},
        _reference_device(),
    )["torchscript"]
    assert_trajectory_parity(
        reference,
        serial.frames,
        rtol=SERIAL_RTOL,
        energy_atol=E_ATOL_EV,
        force_atol=F_ATOL_EV_A,
        spin_force_atol=SPIN_F_ATOL,
        virial_atol=VIRIAL_ATOL_EV,
        label=f"{case_name}/torchscript",
    )


@pytest.mark.lammps
def test_python_torchscript_matches_serial_bec_trajectory(
    model_paths,
    lammps_bin,
    lammps_env,
    tmp_path,
):
    case = CASES["bec"]
    serial = run_trajectory_case(
        "bec",
        model_paths["bec"],
        lammps_bin,
        tmp_path / "serial",
        lammps_env,
    )
    reference = python_trajectory_reference(
        case,
        serial.frames,
        {"torchscript": model_paths["bec"]},
        _reference_device(),
    )["torchscript"]
    assert_trajectory_parity(
        reference,
        serial.frames,
        rtol=SERIAL_RTOL,
        energy_atol=E_ATOL_EV,
        force_atol=F_ATOL_EV_A,
        virial_atol=VIRIAL_ATOL_EV,
        bec_atol=BEC_ATOL_E,
        label="bec/torchscript",
    )


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_python_pt2_matches_serial_pt2_trajectory(
    case_name,
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES[case_name]
    if not torch.cuda.is_available():
        pytest.skip(
            "CPU AOTI evaluation poisons the next numpy LAPACK call on "
            "CUDA-less hosts (known torch 2.11 AOTI-CPU/OpenBLAS issue, "
            "reproduced 2026-09-16); the CUDA lane is covered on GPU nodes"
        )
    _, pt2_path = pt2_artifact_factory(case_name, lammps_pt2_target_device)
    serial = run_trajectory_case(
        case_name,
        pt2_path,
        lammps_bin,
        tmp_path / "serial-pt2",
        lammps_env,
    )
    reference = python_trajectory_reference(
        case,
        serial.frames,
        {"pt2": pt2_path},
        lammps_pt2_target_device,
    )["pt2"]
    assert_trajectory_parity(
        reference,
        serial.frames,
        rtol=SERIAL_RTOL,
        energy_atol=E_ATOL_EV,
        force_atol=PT2_LAMMPS_F_ATOL_EV_A,
        spin_force_atol=PT2_LAMMPS_SPIN_F_ATOL_EV if case.spin else None,
        virial_atol=VIRIAL_ATOL_EV,
        bec_atol=BEC_ATOL_E if case.bec else None,
        label=f"{case_name}/pt2",
    )


@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_serial_matches_mpi_trajectory(
    case_name,
    model_paths,
    lammps_bin,
    lammps_mpi_bin,
    mpi_prefix,
    lammps_env,
    tmp_path,
):
    # frozen=True selects the deterministic start (velocity create is
    # processor-count dependent); the BEC NVE lattice still moves under forces.
    serial = run_trajectory_case(
        case_name,
        model_paths[case_name],
        lammps_bin,
        tmp_path / "serial",
        lammps_env,
        frozen=True,
    )
    for ranks in (2, 4):
        parallel = run_trajectory_case(
            case_name,
            model_paths[case_name],
            lammps_mpi_bin,
            tmp_path / f"mpi-{ranks}",
            lammps_env,
            prefix=mpi_prefix(ranks),
            frozen=True,
        )
        # tolerances.py has no named MPI virial threshold; the global virial
        # therefore uses MPI_RTOL with the production physical VIRIAL_ATOL_EV
        # floor, not an invented stricter gate.
        assert_trajectory_parity(
            serial.frames,
            parallel.frames,
            rtol=MPI_RTOL,
            energy_atol=MPI_E_ATOL_EV,
            force_atol=MPI_F_ATOL_EV_A,
            spin_force_atol=MPI_SPIN_F_ATOL_EV,
            virial_atol=VIRIAL_ATOL_EV,
            bec_atol=MPI_BEC_ATOL_E,
            label=f"{case_name}/mpi-{ranks}",
        )


# ---------------------------------------------------------------------------
# Pure-python comparator/parser negatives (no markers: these run in the CPU
# lane and must not need LAMMPS, MPI, CUDA or model artifacts).
# ---------------------------------------------------------------------------

ATOM_COUNT = 3


def _synthetic_frames(case_name="soc", steps=MD_STEPS, atom_count=ATOM_COUNT):
    """Deterministic synthetic trajectory for comparator corruption tests."""

    spin = CASES[case_name].spin
    bec = CASES[case_name].bec
    frames = []
    for step in range(steps + 1):
        offset = step * 1.0e-3
        frames.append(
            TrajectoryFrame(
                step=step,
                ids=np.arange(1, atom_count + 1),
                types=np.arange(1, atom_count + 1) % 2 + 1,
                origin=np.zeros(3),
                cell=np.eye(3) * 20.0,
                positions=np.full((atom_count, 3), 5.0) + offset,
                spin_directions=(
                    np.tile([1.0, 0.0, 0.0], (atom_count, 1)) if spin else None
                ),
                spin_magnitudes=np.full(atom_count, 2.0) if spin else None,
                energy_ev=-10.0 + offset,
                forces_ev_a=np.full((atom_count, 3), 0.1) + offset,
                spin_forces_ev=(
                    np.full((atom_count, 3), 0.01) + offset if spin else None
                ),
                global_virial_ev=np.arange(6.0) * 0.05 + offset,
                atom_virial_ev=np.full((atom_count, 3, 3), 0.02) + offset,
                bec_e=np.full((atom_count, 3, 3), 0.5) + offset if bec else None,
            )
        )
    return frames


def _compare(reference, actual, case_name="soc"):
    assert_trajectory_parity(
        reference,
        actual,
        rtol=SERIAL_RTOL,
        energy_atol=E_ATOL_EV,
        force_atol=F_ATOL_EV_A,
        spin_force_atol=SPIN_F_ATOL if CASES[case_name].spin else None,
        virial_atol=VIRIAL_ATOL_EV,
        bec_atol=BEC_ATOL_E if CASES[case_name].bec else None,
        label=f"{case_name}/negative",
    )


def test_comparator_accepts_identical_trajectories():
    frames = _synthetic_frames()
    _compare(frames, list(frames))


def test_comparator_detects_corrupted_force_in_frame_seven():
    reference = _synthetic_frames()
    actual = list(reference)
    forces = actual[7].forces_ev_a.copy()
    forces[1, 2] += 1.0
    actual[7] = dataclasses.replace(actual[7], forces_ev_a=forces)
    with pytest.raises(
        AssertionError, match=r"step 7 forces mismatch at index \(1, 2\)"
    ):
        _compare(reference, actual)


def test_comparator_detects_corrupted_final_energy():
    reference = _synthetic_frames()
    actual = list(reference)
    actual[-1] = dataclasses.replace(actual[-1], energy_ev=actual[-1].energy_ev + 1.0)
    with pytest.raises(AssertionError, match=r"step 20 energy mismatch"):
        _compare(reference, actual)


def test_comparator_rejects_missing_step():
    reference = _synthetic_frames()
    actual = list(reference[:7]) + list(reference[8:])
    with pytest.raises(AssertionError, match="step sequence mismatch"):
        _compare(reference, actual)


def test_comparator_rejects_reordered_steps():
    reference = _synthetic_frames()
    actual = list(reference)
    actual[7], actual[8] = actual[8], actual[7]
    with pytest.raises(AssertionError, match="step sequence mismatch"):
        _compare(reference, actual)


def test_comparator_rejects_duplicate_steps():
    reference = _synthetic_frames()
    actual = list(reference)
    actual[7] = dataclasses.replace(actual[7], step=6)
    with pytest.raises(AssertionError, match="step sequence mismatch"):
        _compare(reference, actual)


def test_comparator_detects_corrupted_later_global_virial_component():
    reference = _synthetic_frames()
    actual = list(reference)
    virial = actual[10].global_virial_ev.copy()
    virial[4] += 0.5
    actual[10] = dataclasses.replace(actual[10], global_virial_ev=virial)
    with pytest.raises(
        AssertionError, match=r"step 10 global_virial mismatch at index \(4,\)"
    ):
        _compare(reference, actual)


def test_comparator_detects_one_corrupted_bec_component():
    reference = _synthetic_frames(case_name="bec")
    actual = list(reference)
    bec = actual[12].bec_e.copy()
    bec[2, 1, 2] += 0.5
    actual[12] = dataclasses.replace(actual[12], bec_e=bec)
    with pytest.raises(
        AssertionError, match=r"step 12 bec mismatch at index \(2, 1, 2\)"
    ):
        _compare(reference, actual, case_name="bec")


@pytest.mark.parametrize("observable", ("atom_virial", "bec"))
def test_nequip_route_rejects_out_of_scope_observables(observable):
    with pytest.raises(TrajectoryScopeError, match="nequip_spin"):
        require_observables("nequip_spin", ("energy", "forces", observable))


def test_unknown_observable_rejected():
    with pytest.raises(TrajectoryScopeError, match="unknown observables"):
        require_observables("soc", ("energy", "stress"))


def _parsed_dump_frame(step, ids=(1, 2, 3), atom_type=1.0):
    return {
        "step": step,
        "origin": np.zeros(3),
        "cell": np.eye(3) * 20.0,
        "atoms": [{"id": float(atom_id), "type": atom_type} for atom_id in ids],
        "columns": ["id", "type"],
    }


def _thermo_rows(steps=MD_STEPS):
    return [{"Step": float(step), "PotEng": -10.0} for step in range(steps + 1)]


def test_validate_rejects_duplicate_dump_steps():
    frames = [_parsed_dump_frame(step) for step in range(MD_STEPS + 1)]
    frames[7] = _parsed_dump_frame(6)
    with pytest.raises(ValueError, match="duplicate dump steps"):
        validate_trajectory(frames, _thermo_rows(), MD_STEPS)


def test_validate_rejects_missing_thermo_step():
    frames = [_parsed_dump_frame(step) for step in range(MD_STEPS + 1)]
    rows = _thermo_rows()
    rows[7] = {"Step": float(MD_STEPS + 1), "PotEng": -10.0}
    with pytest.raises(ValueError, match="thermo steps must be exactly"):
        validate_trajectory(frames, rows, MD_STEPS)


def test_validate_rejects_truncated_thermo():
    frames = [_parsed_dump_frame(step) for step in range(MD_STEPS + 1)]
    with pytest.raises(ValueError, match="expected 21 thermo rows, got 20"):
        validate_trajectory(frames, _thermo_rows()[:-1], MD_STEPS)


def test_validate_rejects_non_integer_thermo_step():
    frames = [_parsed_dump_frame(step) for step in range(MD_STEPS + 1)]
    rows = _thermo_rows()
    rows[5] = {"Step": 5.5, "PotEng": -10.0}
    with pytest.raises(ValueError, match="non-integer thermo step"):
        validate_trajectory(frames, rows, MD_STEPS)


def test_validate_rejects_changed_atom_ids_at_later_frame():
    frames = [_parsed_dump_frame(step) for step in range(MD_STEPS + 1)]
    frames[9] = _parsed_dump_frame(9, ids=(1, 2, 4))
    with pytest.raises(ValueError, match="atom IDs/types changed at step 9"):
        validate_trajectory(frames, _thermo_rows(), MD_STEPS)


def _dump_text(
    step, ids=(1, 2), bounds_header="ITEM: BOX BOUNDS pp pp pp", bounds=None
):
    rows = "\n".join(f"{atom_id} {atom_id} {atom_id}.0 2.0 3.0" for atom_id in ids)
    if bounds is None:
        bounds_lines = "0.0 20.0\n0.0 20.0\n0.0 20.0\n"
    else:
        bounds_lines = "".join(" ".join(str(v) for v in axis) + "\n" for axis in bounds)
    return (
        "ITEM: TIMESTEP\n"
        f"{step}\n"
        "ITEM: NUMBER OF ATOMS\n"
        f"{len(ids)}\n"
        f"{bounds_header}\n"
        f"{bounds_lines}"
        "ITEM: ATOMS id type x y z\n"
        f"{rows}\n"
    )


def test_parse_trajectory_dump_reads_cell_and_origin(tmp_path):
    path = tmp_path / "frames.dump"
    path.write_text(_dump_text(0) + _dump_text(1))
    frames = parse_trajectory_dump(path)
    assert [frame["step"] for frame in frames] == [0, 1]
    np.testing.assert_array_equal(frames[0]["origin"], np.zeros(3))
    np.testing.assert_array_equal(frames[0]["cell"], np.eye(3) * 20.0)


def test_parse_trajectory_dump_rejects_triclinic(tmp_path):
    path = tmp_path / "triclinic.dump"
    path.write_text(_dump_text(0, bounds_header="ITEM: BOX BOUNDS xy xz yz pp pp pp"))
    with pytest.raises(ValueError, match="triclinic"):
        parse_trajectory_dump(path)


def test_parse_trajectory_dump_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicates.dump"
    path.write_text(_dump_text(0, ids=(1, 1)))
    with pytest.raises(ValueError, match=r"duplicate atom IDs \[1\] at step 0"):
        parse_trajectory_dump(path)


def test_deployed_torchscript_honors_owned_ghost_batch_split(tmp_path):
    """Model-side contract of the pair_nequip_spin MPI virial fix.

    The pair style feeds a ``batch`` key (0 = owned-centered nodes,
    1 = ghost-centered nodes) so that virial frame 0 is the rank's owned
    contribution. This pins the deployed TorchScript behavior that fix
    relies on: per-frame virial/total_energy split, unchanged per-node
    forces and atomic energies. CPU-only; builds the same random-weight
    deployed model as the LAMMPS lanes.
    """

    from model_factory import build_deployed_model

    model_path = build_deployed_model(
        CASES["nequip_spin"], tmp_path / "nequip_spin.pth"
    )
    model = torch.jit.load(str(model_path))
    model.eval()

    generator = torch.Generator().manual_seed(20260916)
    n_atoms = 4
    base = {
        "pos": torch.rand(n_atoms, 3, generator=generator) * 3.0,
        "spin": torch.rand(n_atoms, 3, generator=generator),
        "edge_index": torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
            dtype=torch.int64,
        ),
        "edge_cell_shift": torch.zeros(8, 3),
        "cell": torch.eye(3) * 10.0,
        "pbc": torch.ones(1, 3, dtype=torch.bool),
        "atom_types": torch.tensor([0, 1, 0, 1], dtype=torch.int64),
    }
    unbatched = model(dict(base))
    batched_input = dict(base)
    batched_input["batch"] = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    batched = model(batched_input)

    assert unbatched["virial"].shape == (1, 3, 3)
    assert batched["virial"].shape == (2, 3, 3)
    torch.testing.assert_close(
        batched["virial"].sum(dim=0, keepdim=True),
        unbatched["virial"],
        rtol=1e-5,
        atol=1e-6,
    )
    # Per-frame total_energy equals the frame's atomic-energy sum, so the
    # pair style can recover the owned-subgraph total from frame 0.
    atomic = batched["atomic_energy"].flatten()
    torch.testing.assert_close(
        batched["total_energy"].flatten(),
        torch.stack([atomic[:2].sum(), atomic[2:].sum()]),
        rtol=1e-5,
        atol=1e-6,
    )
    # Per-node outputs are unaffected by the batch split.
    torch.testing.assert_close(
        batched["forces"], unbatched["forces"], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        batched["atomic_energy"], unbatched["atomic_energy"], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        batched["spin_forces"], unbatched["spin_forces"], rtol=1e-5, atol=1e-6
    )
