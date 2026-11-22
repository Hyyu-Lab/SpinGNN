"""NequIP energy ownership, composition, and finite-receptive-field regressions."""

from __future__ import annotations

import dataclasses
import itertools
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from helpers import (
    CASES,
    LAMMPS_METAL_NKTV2P,
    LammpsRunError,
    atomic_input,
    atoms_from_fixture,
    cloned,
    lammps_parity_result,
    parse_thermo,
    python_reference,
    run_lammps,
)
from model_factory import build_deployed_model, resave_with_metadata
from nequip.data import AtomicDataDict
from nequip.model import model_from_config
from nequip.scripts import deploy
from nequip.scripts.train import default_config
from nequip.utils import Config
from nequip.utils._global_options import _set_global_options
from tolerances import (
    E_ATOL_EV,
    F_ATOL_EV_A,
    HYBRID_ORDER_ATOL_BY_DEVICE,
    MPI_E_ATOL_EV,
    MPI_F_ATOL_EV_A,
    MPI_RTOL,
    MPI_SPIN_F_ATOL_EV,
    SERIAL_RTOL,
    SPIN_F_ATOL,
    VIRIAL_ATOL_EV,
)


def _case(directory: Path, *, spin: bool, large: bool = False):
    """Keep a finite box wider than the MPI halo instead of a tiny periodic cell."""
    directory.mkdir(parents=True, exist_ok=True)
    original = CASES["nequip_spin"]
    atoms = atoms_from_fixture(original)
    repeats = (6, 2, 2) if large else (1, 1, 1)
    cell = np.diag(atoms.cell.array)
    lengths = cell * np.asarray(repeats)
    positions = []
    symbols = []
    moments = []
    for image in itertools.product(*(range(size) for size in repeats)):
        positions.extend(atoms.positions + cell * np.asarray(image))
        symbols.extend(atoms.get_chemical_symbols())
        moments.extend(atoms.get_initial_magnetic_moments())
    style = "spin" if spin else "atomic"
    lines = [
        "NequIP ownership regression",
        "",
        f"{len(positions)} atoms",
        "2 atom types",
        "",
        *(f"0 {length:.16g} {axis}lo {axis}hi" for axis, length in zip("xyz", lengths)),
        "",
        "Masses",
        "",
        "1 54.938044",
        "2 72.630000",
        "",
        f"Atoms # {style}",
        "",
    ]
    for index, (position, symbol, moment) in enumerate(
        zip(positions, symbols, moments)
    ):
        # Sparse tags catch accidental indexing by global ID after decomposition.
        values = [1 + 7 * index, original.type_names.index(symbol) + 1, *position]
        if spin:
            magnitude = np.linalg.norm(moment)
            direction = moment / magnitude if magnitude > 0 else (0.0, 0.0, 1.0)
            values.extend((*direction, magnitude))
        lines.append(" ".join(f"{value:.16g}" for value in values))
    data_path = directory / "system.data"
    data_path.write_text("\n".join(lines) + "\n")
    return dataclasses.replace(
        original,
        name="nequip_spin" if spin else "nequip",
        spin=spin,
        data_name=str(data_path),
    )


def _model(case, directory: Path, *, depth: int = 1, shifted: bool = False):
    overrides = {
        "num_layers": depth,
        "include_spin": case.spin,
        "time_reversal": case.spin,
        "soc": case.spin,
        "model_builders": [
            "SimpleIrrepsConfig",
            "EnergyModel",
            "ParaStressForceSpinOutput" if case.spin else "ParaStressForceOutput",
        ],
    }
    if shifted:
        overrides["model_builders"].append("RescaleEnergyEtc")
        overrides["global_rescale_shift"] = 13.0
        overrides["global_rescale_scale"] = None
    path = build_deployed_model(
        case, directory / "model.pth", config_overrides=overrides
    )
    if shifted:
        # initialize=False installs placeholder rescaling; set a checkpoint-like
        # learned value before saving the test artifact without freezing it.
        model, metadata = deploy.load_deployed_model(
            path, device="cpu", freeze=False, set_global_options=False
        )
        with torch.no_grad():
            model.shift_by.fill_(13.0)
        torch.jit.save(
            model,
            path,
            _extra_files={
                key: value.encode("ascii") for key, value in metadata.items()
            },
        )
    return path


def _run(
    case,
    model,
    binary,
    directory,
    env,
    *,
    prefix=(),
    ranks=1,
    overlay=None,
    neighbor_skin=2.0,
):
    directory.mkdir(parents=True, exist_ok=True)
    pair = "spin/nequip_spin" if case.spin else "nequip"
    pair_lines = f"pair_style {pair}\npair_coeff * * ${{model}} Mn Ge"
    if overlay is not None:
        styles = f"{pair} lj/cut 4.0" if overlay else f"lj/cut 4.0 {pair}"
        pair_lines = (
            f"pair_style hybrid/overlay {styles}\n"
            f"pair_coeff * * {pair} ${{model}} Mn Ge\n"
            "pair_coeff * * lj/cut 0.01 2.0 4.0"
        )
    spin_compute = (
        "compute spinprops all property/atom spx spy spz sp fmx fmy fmz"
        if case.spin
        else ""
    )
    spin_dump = (
        " ".join(f"c_spinprops[{axis}]" for axis in range(1, 8)) if case.spin else ""
    )
    input_path = directory / "run.in"
    input_path.write_text(
        f"""units metal
boundary p p p
atom_style {"spin" if case.spin else "atomic"}
atom_modify map hash
newton off
processors {ranks} 1 1
read_data ${{data}}
{pair_lines}
neighbor {neighbor_skin} bin
{spin_compute}
dump parity all custom 1 ${{dump}} id type x y z fx fy fz {spin_dump}
dump_modify parity sort id format float %.16g
thermo 1
thermo_style custom step pe pxx pyy pzz pxy pxz pyz vol
thermo_modify format float %.16g
run 0
"""
    )
    dump_path = directory / "atoms.dump"
    process = run_lammps(
        binary=binary,
        input_path=input_path,
        workdir=directory,
        env=env,
        prefix=prefix,
        variables={"data": case.data_path, "model": model, "dump": dump_path},
        dump_path=dump_path,
    )
    (directory / "stdout.log").write_text(process.stdout)
    row = parse_thermo(process.log_path)[-1]
    xx, yy, zz, xy, xz, yz = [
        row[name] for name in ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")
    ]
    virial = np.asarray(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))
    virial *= row["Volume"] / LAMMPS_METAL_NKTV2P
    return lammps_parity_result(process, case), virial


@pytest.mark.lammps
def test_python_loader_rejects_missing_deployment_version(tmp_path):
    case = _case(tmp_path, spin=False)
    original = _model(case, tmp_path)
    malformed = resave_with_metadata(
        original, tmp_path / "missing-version.pth", {deploy.NEQUIP_VERSION_KEY: ""}
    )
    with pytest.raises(ValueError, match="does not seem to be a deployed"):
        deploy.load_deployed_model(
            malformed, device="cpu", freeze=False, set_global_options=False
        )


@pytest.mark.lammps
@pytest.mark.mpi
def test_deploy_cli_keeps_constructed_depth_for_mpi(
    lammps_mpi_bin, mpi_prefix, lammps_env, tmp_path
):
    case = _case(tmp_path, spin=True)
    model = tmp_path / "cli.pth"
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "tools/ci/source_python.py"),
            "nequip-deploy",
            "build",
            "--model",
            str(case.config_path),
            str(model),
        ],
        cwd=tmp_path,
        env=lammps_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout
    _, metadata = deploy.load_deployed_model(
        model, device="cpu", freeze=False, set_global_options=False
    )
    assert metadata[deploy.NEQUIP_NUM_INTERACTIONS_KEY] == "1"
    reference = python_reference(model, case, device="cpu")
    actual, _ = _run(
        case,
        model,
        lammps_mpi_bin,
        tmp_path / "mpi",
        lammps_env,
        prefix=mpi_prefix(2),
        ranks=2,
    )
    assert math.isclose(
        actual.energy_ev, reference.energy_ev, rel_tol=MPI_RTOL, abs_tol=MPI_E_ATOL_EV
    )
    np.testing.assert_allclose(
        actual.forces_ev_a, reference.forces_ev_a, rtol=MPI_RTOL, atol=MPI_F_ATOL_EV_A
    )
    np.testing.assert_allclose(
        actual.spin_forces_ev,
        reference.spin_forces_ev,
        rtol=MPI_RTOL,
        atol=MPI_SPIN_F_ATOL_EV,
    )


@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("spin", (False, True), ids=("structure", "spin"))
def test_nequip_periodic_ghost_uses_complete_neighbor_image(
    spin, lammps_bin, lammps_mpi_bin, mpi_prefix, lammps_env, tmp_path
):
    # The first image of x=9 on rank 0 is at the halo edge; its x=-1 image
    # has the complete neighborhood needed to differentiate the owned force.
    lines = [
        "Periodic ghost neighborhood regression",
        "",
        "4 atoms",
        "2 atom types",
        "",
        "0 10 xlo xhi",
        "0 20 ylo yhi",
        "0 20 zlo zhi",
        "",
        "Masses",
        "",
        "1 54.938044",
        "2 72.63",
        "",
        f"Atoms # {'spin' if spin else 'atomic'}",
        "",
    ]
    moments = ((0.6, 0.8, 0, 1.5), (0, 0, 1, 0), (0, 0.6, 0.8, 2), (0, 0, 1, 0))
    for index, (x, moment) in enumerate(zip((0.5, 1.3, 8.0, 9.0), moments)):
        values = [index + 1, index % 2 + 1, x, 10, 10]
        if spin:
            values.extend(moment)
        lines.append(" ".join(str(value) for value in values))
    data = tmp_path / "system.data"
    data.write_text("\n".join(lines) + "\n")
    case = dataclasses.replace(
        CASES["nequip_spin"],
        name="nequip_spin" if spin else "nequip",
        data_name=str(data),
        spin=spin,
        r_max=2.0,
    )
    model = build_deployed_model(
        case,
        tmp_path / "model.pth",
        config_overrides={
            "r_max": case.r_max,
            "num_layers": 1,
            "include_spin": spin,
            "time_reversal": spin,
            "soc": spin,
            "model_builders": [
                "SimpleIrrepsConfig",
                "EnergyModel",
                "ParaStressForceSpinOutput" if spin else "ParaStressForceOutput",
            ],
        },
    )
    loaded, _ = deploy.load_deployed_model(
        model, device="cpu", freeze=True, set_global_options=False
    )
    reference = loaded(cloned(atomic_input(case)))
    for ranks in (1, 2):
        actual, virial = _run(
            case,
            model,
            lammps_bin if ranks == 1 else lammps_mpi_bin,
            tmp_path / f"ranks-{ranks}",
            lammps_env,
            prefix=() if ranks == 1 else mpi_prefix(ranks),
            ranks=ranks,
            neighbor_skin=0.1,
        )
        assert math.isclose(
            actual.energy_ev,
            float(reference[AtomicDataDict.TOTAL_ENERGY_KEY].sum().detach()),
            rel_tol=MPI_RTOL,
            abs_tol=MPI_E_ATOL_EV,
        )
        np.testing.assert_allclose(
            actual.forces_ev_a,
            reference[AtomicDataDict.FORCE_KEY].detach().cpu().numpy(),
            rtol=MPI_RTOL,
            atol=MPI_F_ATOL_EV_A,
        )
        np.testing.assert_allclose(
            virial,
            reference[AtomicDataDict.VIRIAL_KEY].detach().cpu().numpy().reshape(3, 3),
            rtol=MPI_RTOL,
            atol=VIRIAL_ATOL_EV,
        )
        if spin:
            np.testing.assert_allclose(
                actual.spin_forces_ev,
                reference[AtomicDataDict.SPIN_FORCE_KEY].detach().cpu().numpy(),
                rtol=MPI_RTOL,
                atol=MPI_SPIN_F_ATOL_EV,
            )


@pytest.mark.lammps
def test_nequip_nonspin_preserves_total_only_energy_shift(
    lammps_bin, lammps_env, tmp_path
):
    case = _case(tmp_path, spin=False)
    model = _model(case, tmp_path, shifted=True)
    reference = python_reference(model, case, device="cpu")
    actual, _ = _run(case, model, lammps_bin, tmp_path / "lammps", lammps_env)
    assert math.isclose(
        actual.energy_ev, reference.energy_ev, rel_tol=SERIAL_RTOL, abs_tol=E_ATOL_EV
    )
    np.testing.assert_allclose(
        actual.forces_ev_a, reference.forces_ev_a, rtol=SERIAL_RTOL, atol=F_ATOL_EV_A
    )


@pytest.mark.lammps
def test_nequip_nonspin_hybrid_overlay_is_order_invariant(
    lammps_bin, lammps_env, lammps_pt2_target_device, tmp_path
):
    case = _case(tmp_path, spin=False)
    model = _model(case, tmp_path)
    first, first_virial = _run(
        case, model, lammps_bin, tmp_path / "first", lammps_env, overlay=True
    )
    last, last_virial = _run(
        case, model, lammps_bin, tmp_path / "last", lammps_env, overlay=False
    )
    atol = HYBRID_ORDER_ATOL_BY_DEVICE[lammps_pt2_target_device]
    assert math.isclose(first.energy_ev, last.energy_ev, rel_tol=0.0, abs_tol=atol)
    np.testing.assert_allclose(first.forces_ev_a, last.forces_ev_a, rtol=0.0, atol=atol)
    np.testing.assert_allclose(first_virial, last_virial, rtol=0.0, atol=atol)


@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("spin", (False, True), ids=("structure", "spin"))
@pytest.mark.parametrize("depth", (1, 3))
def test_nequip_large_box_mpi_matches_complete_python_graph(
    spin, depth, lammps_bin, lammps_mpi_bin, mpi_prefix, lammps_env, tmp_path
):
    case = _case(tmp_path, spin=spin, large=True)
    model = _model(case, tmp_path, depth=depth)
    reference = python_reference(model, case, device="cpu")
    serial, serial_virial = _run(
        case, model, lammps_bin, tmp_path / "serial", lammps_env
    )
    assert math.isclose(
        serial.energy_ev, reference.energy_ev, rel_tol=SERIAL_RTOL, abs_tol=E_ATOL_EV
    )
    np.testing.assert_allclose(
        serial.forces_ev_a, reference.forces_ev_a, rtol=SERIAL_RTOL, atol=F_ATOL_EV_A
    )
    loaded, _ = deploy.load_deployed_model(
        model, device="cpu", freeze=True, set_global_options=False
    )
    output = loaded(cloned(atomic_input(case)))
    reference_virial = (
        output[AtomicDataDict.VIRIAL_KEY].detach().cpu().numpy().reshape(3, 3)
    )
    np.testing.assert_allclose(
        serial_virial, reference_virial, rtol=SERIAL_RTOL, atol=VIRIAL_ATOL_EV
    )
    if spin:
        np.testing.assert_allclose(
            serial.spin_forces_ev,
            reference.spin_forces_ev,
            rtol=SERIAL_RTOL,
            atol=SPIN_F_ATOL,
        )
    for ranks in (2, 4):
        actual, virial = _run(
            case,
            model,
            lammps_mpi_bin,
            tmp_path / f"mpi-{ranks}",
            lammps_env,
            prefix=mpi_prefix(ranks),
            ranks=ranks,
        )
        assert math.isclose(
            actual.energy_ev, serial.energy_ev, rel_tol=MPI_RTOL, abs_tol=MPI_E_ATOL_EV
        )
        np.testing.assert_allclose(
            actual.forces_ev_a, serial.forces_ev_a, rtol=MPI_RTOL, atol=MPI_F_ATOL_EV_A
        )
        np.testing.assert_allclose(
            virial, serial_virial, rtol=MPI_RTOL, atol=VIRIAL_ATOL_EV
        )
        if spin:
            np.testing.assert_allclose(
                actual.spin_forces_ev,
                serial.spin_forces_ev,
                rtol=MPI_RTOL,
                atol=MPI_SPIN_F_ATOL_EV,
            )


@pytest.mark.lammps
@pytest.mark.mpi
def test_nequip_nonspin_global_shift_is_added_once_across_ranks(
    lammps_mpi_bin, mpi_prefix, lammps_env, tmp_path
):
    case = _case(tmp_path, spin=False)
    model = _model(case, tmp_path, shifted=True)
    reference = python_reference(model, case, device="cpu")
    for ranks in (2, 4):
        actual, _ = _run(
            case,
            model,
            lammps_mpi_bin,
            tmp_path / f"mpi-{ranks}",
            lammps_env,
            prefix=mpi_prefix(ranks),
            ranks=ranks,
        )
        assert math.isclose(
            actual.energy_ev,
            reference.energy_ev,
            rel_tol=SERIAL_RTOL,
            abs_tol=E_ATOL_EV,
        )


@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("spin", (False, True), ids=("structure", "spin"))
def test_legacy_nequip_without_interaction_depth_is_serial_only(
    spin, lammps_bin, lammps_mpi_bin, mpi_prefix, lammps_env, tmp_path
):
    case = _case(tmp_path, spin=spin)
    current = _model(case, tmp_path)
    legacy = resave_with_metadata(
        current, tmp_path / "legacy.pth", {"nequip_num_interactions": ""}
    )
    reference = python_reference(legacy, case, device="cpu")
    actual, _ = _run(case, legacy, lammps_bin, tmp_path / "serial", lammps_env)
    assert math.isclose(
        actual.energy_ev,
        reference.energy_ev,
        rel_tol=SERIAL_RTOL,
        abs_tol=E_ATOL_EV,
    )
    with pytest.raises(LammpsRunError, match="nequip_num_interactions"):
        _run(
            case,
            legacy,
            lammps_mpi_bin,
            tmp_path / "mpi",
            lammps_env,
            prefix=mpi_prefix(2),
            ranks=2,
        )


@pytest.mark.lammps
def test_nequip_nonspin_triclinic_periodic_shifts_match_python(
    lammps_bin, lammps_env, tmp_path
):
    case = _case(tmp_path, spin=False)
    case.data_path.write_text(
        case.data_path.read_text().replace(
            "\nMasses\n", "\n2.2 -1.7 1.9 xy xz yz\n\nMasses\n"
        )
    )
    model = _model(case, tmp_path, depth=2)
    reference = python_reference(model, case, device="cpu")
    actual, _ = _run(case, model, lammps_bin, tmp_path / "lammps", lammps_env)
    assert math.isclose(
        actual.energy_ev,
        reference.energy_ev,
        rel_tol=SERIAL_RTOL,
        abs_tol=E_ATOL_EV,
    )
    np.testing.assert_allclose(
        actual.forces_ev_a,
        reference.forces_ev_a,
        rtol=SERIAL_RTOL,
        atol=F_ATOL_EV_A,
    )


def _empty_owner_case(directory: Path, spin: bool):
    case = _case(directory, spin=spin)
    lines = case.data_path.read_text().splitlines()
    in_atoms = False
    for index, line in enumerate(lines):
        if line.endswith("xlo xhi"):
            lines[index] = "0 40 xlo xhi"
        elif line.startswith("Atoms #"):
            in_atoms = True
        elif in_atoms and line.strip():
            fields = line.split()
            fields[2] = f"{float(fields[2]) + 25.0:.16g}"
            lines[index] = " ".join(fields)
    case.data_path.write_text("\n".join(lines) + "\n")
    return case


@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("spin", (False, True), ids=("structure", "spin"))
def test_nequip_empty_rank_rejects_per_atom_virial_collectively(
    spin, lammps_mpi_bin, mpi_prefix, lammps_env, tmp_path
):
    case = _empty_owner_case(tmp_path, spin)
    model = _model(case, tmp_path)
    pair = "spin/nequip_spin" if spin else "nequip"
    input_path = tmp_path / "unsupported.in"
    input_path.write_text(
        f"""units metal
boundary p p p
atom_style {"spin" if spin else "atomic"}
atom_modify map hash
newton off
processors 2 1 1
read_data ${{data}}
pair_style {pair}
pair_coeff * * ${{model}} Mn Ge
compute peratom all stress/atom NULL
compute summed all reduce sum c_peratom[1]
thermo 1
thermo_style custom step pe c_summed
run 0
"""
    )
    # Bound the whole MPI process group, including Hydra child processes, if
    # a regression leaves empty and nonempty ranks in different collectives.
    result = run_lammps(
        binary=lammps_mpi_bin,
        input_path=input_path,
        workdir=tmp_path / "mpi",
        env=lammps_env,
        variables={"data": case.data_path, "model": model},
        prefix=("timeout", "--kill-after=5", "30", *mpi_prefix(2)),
        check=False,
        timeout=45,
    )
    assert result.returncode not in (0, 124, 137), result.stdout
    assert "does not support per-atom virial" in result.stdout


@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("spin", (False, True), ids=("structure", "spin"))
def test_nequip_empty_rank_zero_keeps_global_shift_and_owned_outputs(
    spin, lammps_mpi_bin, mpi_prefix, lammps_env, tmp_path
):
    case = _empty_owner_case(tmp_path, spin)
    model = _model(case, tmp_path, shifted=True)
    reference = python_reference(model, case, device="cpu")
    actual, _ = _run(
        case,
        model,
        lammps_mpi_bin,
        tmp_path / "mpi",
        lammps_env,
        prefix=mpi_prefix(2),
        ranks=2,
    )
    assert math.isclose(
        actual.energy_ev,
        reference.energy_ev,
        rel_tol=SERIAL_RTOL,
        abs_tol=E_ATOL_EV,
    )
    np.testing.assert_allclose(
        actual.forces_ev_a,
        reference.forces_ev_a,
        rtol=SERIAL_RTOL,
        atol=F_ATOL_EV_A,
    )
    if spin:
        np.testing.assert_allclose(
            actual.spin_forces_ev,
            reference.spin_forces_ev,
            rtol=SERIAL_RTOL,
            atol=SPIN_F_ATOL,
        )


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("empty_owner", (False, True), ids=("split", "empty-rank-zero"))
def test_pt2_nequip_spin_mpi_owned_virial_matches_torchscript(
    empty_owner,
    pt2_artifact_factory,
    lammps_pt2_target_device,
    lammps_bin,
    lammps_mpi_bin,
    mpi_prefix,
    lammps_env,
    tmp_path,
):
    case = (
        _empty_owner_case(tmp_path, True)
        if empty_owner
        else _case(tmp_path, spin=True, large=True)
    )
    scripted, compiled = pt2_artifact_factory("nequip_spin", lammps_pt2_target_device)
    reference, reference_virial = _run(
        case, scripted, lammps_bin, tmp_path / "serial", lammps_env
    )
    actual, virial = _run(
        case,
        compiled,
        lammps_mpi_bin,
        tmp_path / "mpi",
        lammps_env,
        prefix=mpi_prefix(2),
        ranks=2,
    )
    assert math.isclose(
        actual.energy_ev,
        reference.energy_ev,
        rel_tol=SERIAL_RTOL,
        abs_tol=E_ATOL_EV,
    )
    np.testing.assert_allclose(
        actual.forces_ev_a,
        reference.forces_ev_a,
        rtol=SERIAL_RTOL,
        atol=F_ATOL_EV_A,
    )
    np.testing.assert_allclose(
        actual.spin_forces_ev,
        reference.spin_forces_ev,
        rtol=SERIAL_RTOL,
        atol=SPIN_F_ATOL,
    )
    np.testing.assert_allclose(
        virial,
        reference_virial,
        rtol=SERIAL_RTOL,
        atol=VIRIAL_ATOL_EV,
    )


@pytest.mark.lammps
def test_nequip_depth_metadata_uses_constructed_core_instead_of_later_config():
    config = Config.from_file(
        str(CASES["nequip_spin"].config_path), defaults=default_config
    )
    config["num_layers"] = 3
    _set_global_options(config)
    model = model_from_config(config, initialize=False, deploy=True)
    config["num_layers"] = 1
    metadata = deploy.assemble_deploy_metadata(
        config, model, sorted(model.irreps_out), sorted(model.irreps_in)
    )
    assert metadata[deploy.NEQUIP_NUM_INTERACTIONS_KEY] == "3"


@pytest.mark.lammps
@pytest.mark.parametrize("depth", ("0", "-1", "1.5", "nan"))
def test_nequip_rejects_invalid_deployed_interaction_depth(
    depth, lammps_bin, lammps_env, tmp_path
):
    case = _case(tmp_path, spin=False)
    model = _model(case, tmp_path)
    malformed = resave_with_metadata(
        model, tmp_path / "malformed.pth", {"nequip_num_interactions": depth}
    )
    with pytest.raises(LammpsRunError, match="nequip_num_interactions"):
        _run(case, malformed, lammps_bin, tmp_path / "lammps", lammps_env)


@pytest.mark.compile
@pytest.mark.lammps
def test_pt2_nequip_depth_is_bound_to_the_embedded_manifest(
    pt2_artifact_factory, lammps_pt2_target_device, lammps_bin, lammps_env, tmp_path
):
    case = _case(tmp_path, spin=True)
    _, package = pt2_artifact_factory("nequip_spin", lammps_pt2_target_device)
    modified = tmp_path / "modified.pt2"
    shutil.copyfile(package, modified)
    metadata = deploy.read_deploy_metadata_sidecar(package)
    metadata[deploy.NEQUIP_NUM_INTERACTIONS_KEY] = str(
        int(metadata[deploy.NEQUIP_NUM_INTERACTIONS_KEY]) + 1
    )
    deploy.write_deploy_metadata_sidecar(modified, metadata)
    with pytest.raises(
        LammpsRunError, match="embedded ABI manifest key 'nequip_num_interactions'"
    ):
        _run(case, modified, lammps_bin, tmp_path / "lammps", lammps_env)


@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("spin", (False, True), ids=("structure", "spin"))
def test_standard_stress_wrapper_nequip_mpi_matches_complete_graph(
    spin, lammps_bin, lammps_mpi_bin, mpi_prefix, lammps_env, tmp_path
):
    case = _case(tmp_path, spin=spin, large=True)
    model = build_deployed_model(
        case,
        tmp_path / "standard-stress.pth",
        config_overrides={
            "num_layers": 3,
            "include_spin": spin,
            "time_reversal": spin,
            "soc": spin,
            "model_builders": [
                "SimpleIrrepsConfig",
                "EnergyModel",
                "StressForceSpinOutput" if spin else "StressForceOutput",
            ],
        },
    )
    loaded, _ = deploy.load_deployed_model(
        model, device="cpu", freeze=True, set_global_options=False
    )
    reference = loaded(cloned(atomic_input(case)))
    expected_energy = float(reference[AtomicDataDict.TOTAL_ENERGY_KEY].sum())
    expected_forces = reference[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
    expected_virial = (
        reference[AtomicDataDict.VIRIAL_KEY].detach().cpu().numpy().reshape(3, 3)
    )
    for ranks in (1, 2, 4):
        actual, virial = _run(
            case,
            model,
            lammps_bin if ranks == 1 else lammps_mpi_bin,
            tmp_path / f"ranks-{ranks}",
            lammps_env,
            prefix=() if ranks == 1 else mpi_prefix(ranks),
            ranks=ranks,
        )
        assert math.isclose(
            actual.energy_ev, expected_energy, rel_tol=SERIAL_RTOL, abs_tol=E_ATOL_EV
        )
        np.testing.assert_allclose(
            actual.forces_ev_a, expected_forces, rtol=SERIAL_RTOL, atol=F_ATOL_EV_A
        )
        np.testing.assert_allclose(
            virial, expected_virial, rtol=SERIAL_RTOL, atol=VIRIAL_ATOL_EV
        )
        if spin:
            expected_spin = (
                reference[AtomicDataDict.SPIN_FORCE_KEY].detach().cpu().numpy()
            )
            np.testing.assert_allclose(
                actual.spin_forces_ev, expected_spin, rtol=SERIAL_RTOL, atol=SPIN_F_ATOL
            )
