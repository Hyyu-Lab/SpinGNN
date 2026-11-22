"""LAMMPS process, parser, and Python-reference helpers."""

from __future__ import annotations

import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io.lammpsdata import read_lammps_data
from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper
from nequip.scripts import deploy
from nequip.scripts.deploy import load_deployed_model

from tolerances import LAMMPS_TIMEOUT_SECONDS, MD_STEPS


TEST_ROOT = Path(__file__).resolve().parent
FIXTURE_ROOT = TEST_ROOT / "fixtures"

# LAMMPS metal-unit hbar: Planck's constant in eV ps divided by 2 pi.
H_BAR_EV_PS = 4.135667403e-3 / (2.0 * math.pi)


def absorb_aoti_lapack_poison() -> None:
    """Absorb the one-shot numpy LAPACK failure left by CPU AOTI runs.

    On CUDA-less hosts, running an AOTInductor CPU package in-process makes
    the NEXT numpy LAPACK call fail once ("SVD did not converge"), after
    which later calls succeed again (reproduced 2026-09-16; GPU/CUDA lanes
    are unaffected). Call this right after every in-process AOTI run so the
    poisoned call is a sacrificial one instead of a real assertion.
    """
    if torch.cuda.is_available():
        return
    try:
        np.linalg.svd(np.eye(3))
    except np.linalg.LinAlgError:
        pass


# update.cpp's nktv2p for metal units (bar Angstrom^3 per eV).
LAMMPS_METAL_NKTV2P = 1.6021765e6


@dataclass(frozen=True)
class CaseSpec:
    name: str
    config_name: str
    data_name: str
    input_name: str
    type_names: tuple[str, ...]
    r_max: float
    spin: bool = False
    bec: bool = False

    @property
    def config_path(self) -> Path:
        return FIXTURE_ROOT / "configs" / self.config_name

    @property
    def data_path(self) -> Path:
        return FIXTURE_ROOT / "data" / self.data_name

    @property
    def input_path(self) -> Path:
        return FIXTURE_ROOT / "in" / self.input_name

    @property
    def bec_path(self) -> Path | None:
        return FIXTURE_ROOT / "data" / "bec.bec" if self.bec else None


CASES: dict[str, CaseSpec] = {
    "nosoc": CaseSpec(
        name="nosoc",
        config_name="nosoc.yaml",
        data_name="nosoc.data",
        input_name="nosoc_run0.in",
        type_names=("Mn", "Ge"),
        r_max=4.0,
        spin=True,
    ),
    "soc": CaseSpec(
        name="soc",
        config_name="soc.yaml",
        data_name="soc.data",
        input_name="soc_run0.in",
        type_names=("Mn", "Ge"),
        r_max=4.0,
        spin=True,
    ),
    "nequip_spin": CaseSpec(
        name="nequip_spin",
        config_name="nequip_spin.yaml",
        data_name="soc.data",
        input_name="nequip_spin_run0.in",
        type_names=("Mn", "Ge"),
        r_max=4.0,
        spin=True,
    ),
    "bec": CaseSpec(
        name="bec",
        config_name="bec.yaml",
        data_name="bec.data",
        input_name="bec_run0.in",
        type_names=("Li", "Nb", "O"),
        r_max=6.2,
        bec=True,
    ),
}
CASE_NAMES = tuple(CASES)


@dataclass(frozen=True)
class ProcessResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    log_path: Path
    dump_path: Path | None

    @property
    def output(self) -> str:
        log_text = (
            self.log_path.read_text(errors="replace") if self.log_path.exists() else ""
        )
        return f"{self.stdout}\n{log_text}"


class LammpsRunError(RuntimeError):
    def __init__(self, result: ProcessResult):
        super().__init__(
            f"LAMMPS exited with {result.returncode}: {' '.join(result.command)}\n"
            f"{result.output[-8000:]}"
        )
        self.result = result


@dataclass(frozen=True)
class ParityResult:
    energy_ev: float
    forces_ev_a: np.ndarray
    spin_forces_ev: np.ndarray | None = None
    atom_virial_ev: np.ndarray | None = None
    bec_e: np.ndarray | None = None


def _lammps_arguments(
    binary: Path,
    input_path: Path,
    log_path: Path,
    variables: Mapping[str, str | os.PathLike[str]],
) -> list[str]:
    command = [str(binary), "-in", str(input_path), "-log", str(log_path)]
    for name, value in variables.items():
        command.extend(("-var", name, str(value)))
    return command


def run_lammps(
    *,
    binary: Path,
    input_path: Path,
    workdir: Path,
    env: Mapping[str, str],
    variables: Mapping[str, str | os.PathLike[str]],
    prefix: Sequence[str] = (),
    dump_path: Path | None = None,
    check: bool = True,
    timeout: int = LAMMPS_TIMEOUT_SECONDS,
) -> ProcessResult:
    """Run one isolated LAMMPS process with all paths supplied by ``-var``."""

    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "lammps.log"
    command = [
        *prefix,
        *_lammps_arguments(binary, input_path, log_path, variables),
    ]
    completed = subprocess.run(
        command,
        cwd=workdir,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    result = ProcessResult(
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        log_path=log_path,
        dump_path=dump_path,
    )
    if check and result.returncode != 0:
        raise LammpsRunError(result)
    return result


def run_case(
    *,
    case: CaseSpec,
    model_path: Path,
    binary: Path,
    workdir: Path,
    env: Mapping[str, str],
    prefix: Sequence[str] = (),
) -> ParityResult:
    dump_path = workdir / f"{case.name}.dump"
    variables: dict[str, str | os.PathLike[str]] = {
        "data": case.data_path,
        "model": model_path,
        "dump": dump_path,
    }
    if case.bec_path is not None:
        variables["bec"] = case.bec_path
    process = run_lammps(
        binary=binary,
        input_path=case.input_path,
        workdir=workdir,
        env=env,
        variables=variables,
        prefix=prefix,
        dump_path=dump_path,
    )
    if env.get("SPINGNN_REQUIRE_CUDA_LAMMPS") == "1":
        (workdir / "cuda-runtime.stdout.log").write_text(process.stdout)
        if not re.search(
            r"(?:Allegro(?:_spin)?|NEQUIP) is using device cuda\b", process.stdout
        ):
            raise RuntimeError(f"LAMMPS did not select a CUDA device: {workdir}")
    return lammps_parity_result(process, case)


def write_cutoff_boundary_data(
    path: Path,
    boundary_distance: float,
    *,
    kokkos_order: bool = False,
    atom_style: str = "spin",
    type_names: Sequence[str] = ("Mn", "Ge"),
) -> Path:
    """Write three atoms with one inner pair and one cutoff-boundary pair."""

    if atom_style not in {"atomic", "spin"}:
        raise ValueError(f"unsupported cutoff-boundary atom style: {atom_style}")
    if kokkos_order and atom_style != "spin":
        raise ValueError("kokkos_order is only meaningful for spin atom data")
    if len(type_names) < 2:
        raise ValueError("cutoff-boundary data requires at least two atom types")

    if atom_style == "spin":
        positions = (
            (1, 1, 8.0, 10.0, 10.0),
            (2, 2, 9.0, 10.0, 10.0),
            (3, 1, 9.0 + boundary_distance, 10.0, 10.0),
        )
    else:
        # Anchor the exact-cutoff pair at x=0 so subtracting its decimal cutoff
        # coordinate reproduces the binary64 value read from model metadata.
        positions = (
            (1, 1, 0.0, 10.0, 10.0),
            (2, 2, 1.0, 10.0, 10.0),
            (3, 1, boundary_distance, 10.0, 10.0),
        )
    spins = (
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
    )
    atom_lines = []
    for (atom_id, atom_type, x, y, z), (sx, sy, sz, magnitude) in zip(positions, spins):
        if atom_style == "atomic":
            values = (atom_id, atom_type, x, y, z)
        elif kokkos_order:
            values = (atom_id, atom_type, magnitude, x, y, z, sx, sy, sz)
        else:
            values = (atom_id, atom_type, x, y, z, sx, sy, sz, magnitude)
        atom_lines.append(" ".join(str(value) for value in values))
    mass_lines = tuple(
        f"{index} {atomic_masses[atomic_numbers[symbol]]} # {symbol}"
        for index, symbol in enumerate(type_names, start=1)
    )
    path.write_text(
        "\n".join(
            (
                "Three-atom cutoff boundary regression",
                "",
                "3 atoms",
                f"{len(type_names)} atom types",
                "",
                "0.0 24.0 xlo xhi",
                "0.0 20.0 ylo yhi",
                "0.0 20.0 zlo zhi",
                "",
                "Masses",
                "",
                *mass_lines,
                "",
                f"Atoms # {atom_style}",
                "",
                *atom_lines,
                "",
            )
        )
    )
    return path


def parse_dump(path: Path) -> list[dict[str, object]]:
    lines = path.read_text().splitlines()
    frames: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        if lines[index] != "ITEM: TIMESTEP":
            index += 1
            continue
        step = int(lines[index + 1])
        if lines[index + 2] != "ITEM: NUMBER OF ATOMS":
            raise ValueError(f"malformed dump near line {index + 3}: {path}")
        atom_count = int(lines[index + 3])
        index += 4
        if not lines[index].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"missing box bounds in {path}")
        index += 4
        if not lines[index].startswith("ITEM: ATOMS "):
            raise ValueError(f"missing atom header in {path}")
        columns = lines[index].split()[2:]
        index += 1
        atoms: list[dict[str, float]] = []
        for _ in range(atom_count):
            values = [float(value) for value in lines[index].split()]
            if len(values) != len(columns) or not np.isfinite(values).all():
                raise ValueError(f"invalid atom row in {path}: {lines[index]}")
            atoms.append(dict(zip(columns, values)))
            index += 1
        atoms.sort(key=lambda row: row["id"])
        frames.append({"step": step, "atoms": atoms, "columns": columns})
    if not frames:
        raise ValueError(f"no frames in {path}")
    return frames


def parse_thermo(path: Path) -> list[dict[str, float]]:
    lines = path.read_text().splitlines()
    rows: list[dict[str, float]] = []
    for index, line in enumerate(lines):
        header = line.split()
        if not header or header[0] != "Step" or "PotEng" not in header:
            continue
        for row_line in lines[index + 1 :]:
            values = row_line.split()
            if len(values) != len(header):
                break
            try:
                numbers = [float(value) for value in values]
            except ValueError:
                break
            if not np.isfinite(numbers).all():
                raise ValueError(f"nonfinite thermo row in {path}: {row_line}")
            rows.append(dict(zip(header, numbers)))
    if not rows:
        raise ValueError(f"no thermo rows in {path}")
    return rows


def columns(frame: Mapping[str, object], names: Sequence[str]) -> np.ndarray:
    available = frame["columns"]
    assert isinstance(available, list)
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(f"dump columns missing: {missing}")
    atoms = frame["atoms"]
    assert isinstance(atoms, list)
    return np.asarray([[row[name] for name in names] for row in atoms], dtype=float)


def _cstress_atom_virial(frame: Mapping[str, object]) -> np.ndarray:
    """centroid/stress/atom columns -> per-atom virial in eV (n, 3, 3).

    centroid/stress columns are xx,yy,zz,xy,xz,yz,yx,zx,zy; the compute
    returns -virial multiplied by metal nktv2p.
    """

    virial_names = tuple(f"c_cstress[{index}]" for index in range(1, 10))
    stress = columns(frame, virial_names)
    atom_virial = np.empty((len(stress), 3, 3), dtype=float)
    atom_virial[:, 0, 0] = stress[:, 0]
    atom_virial[:, 1, 1] = stress[:, 1]
    atom_virial[:, 2, 2] = stress[:, 2]
    atom_virial[:, 0, 1] = stress[:, 3]
    atom_virial[:, 0, 2] = stress[:, 4]
    atom_virial[:, 1, 2] = stress[:, 5]
    atom_virial[:, 1, 0] = stress[:, 6]
    atom_virial[:, 2, 0] = stress[:, 7]
    atom_virial[:, 2, 1] = stress[:, 8]
    return atom_virial * (-1.0 / LAMMPS_METAL_NKTV2P)


def lammps_parity_result(process: ProcessResult, case: CaseSpec) -> ParityResult:
    if process.dump_path is None:
        raise ValueError("parity result requires a dump path")
    frame = parse_dump(process.dump_path)[0]
    thermo = parse_thermo(process.log_path)[0]
    forces = columns(frame, ("fx", "fy", "fz"))
    spin_forces = None
    atom_virial = None
    bec = None
    if case.spin:
        fm_names = tuple(f"c_spinprops[{index}]" for index in range(5, 8))
        spin_forces = columns(frame, fm_names) * H_BAR_EV_PS
    virial_names = tuple(f"c_cstress[{index}]" for index in range(1, 10))
    if all(name in frame["columns"] for name in virial_names):
        atom_virial = _cstress_atom_virial(frame)
    if case.bec:
        bec_names = tuple(f"c_becprops[{index}]" for index in range(1, 10))
        bec = columns(frame, bec_names).reshape(-1, 3, 3)
    return ParityResult(
        energy_ev=float(thermo["PotEng"]),
        forces_ev_a=forces,
        spin_forces_ev=spin_forces,
        atom_virial_ev=atom_virial,
        bec_e=bec,
    )


def _spin_atoms_from_lammps(path: Path, type_names: Sequence[str]) -> Atoms:
    lines = path.read_text().splitlines()
    bounds: dict[str, tuple[float, float]] = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 4 and fields[2:] in (
            ["xlo", "xhi"],
            ["ylo", "yhi"],
            ["zlo", "zhi"],
        ):
            bounds[fields[2][0]] = (float(fields[0]), float(fields[1]))
    if set(bounds) != {"x", "y", "z"}:
        raise ValueError(f"orthogonal box bounds missing in {path}")
    header = next(
        index for index, line in enumerate(lines) if line.startswith("Atoms # spin")
    )
    rows: list[tuple[int, int, list[float]]] = []
    for line in lines[header + 1 :]:
        fields = line.split()
        if not fields:
            if rows:
                continue
            continue
        if not fields[0].isdigit():
            if rows:
                break
            continue
        rows.append(
            (int(fields[0]), int(fields[1]), [float(value) for value in fields[2:]])
        )
    rows.sort(key=lambda row: row[0])
    positions = np.asarray([row[2][:3] for row in rows], dtype=float)
    spins = np.asarray([row[2][3:6] for row in rows], dtype=float)
    magnitudes = np.asarray([row[2][6] for row in rows], dtype=float)
    spins *= magnitudes[:, None]
    symbols = [type_names[row[1] - 1] for row in rows]
    origin = np.asarray([bounds[axis][0] for axis in "xyz"], dtype=float)
    cell = np.diag([bounds[axis][1] - bounds[axis][0] for axis in "xyz"])
    atoms = Atoms(symbols=symbols, positions=positions - origin, cell=cell, pbc=True)
    atoms.set_initial_magnetic_moments(spins)
    return atoms


def atoms_from_fixture(case: CaseSpec) -> Atoms:
    if case.spin:
        return _spin_atoms_from_lammps(case.data_path, case.type_names)
    z_by_type = {
        index: atomic_numbers[symbol]
        for index, symbol in enumerate(case.type_names, start=1)
    }
    with case.data_path.open() as stream:
        atoms = read_lammps_data(
            stream,
            Z_of_type=z_by_type,
            sort_by_id=True,
            units="metal",
            atom_style="atomic",
        )
    atoms.pbc = True
    return atoms


def _input_from_atoms(
    case: CaseSpec, atoms: Atoms, device: str | torch.device
) -> dict[str, torch.Tensor]:
    data = AtomicData.from_ase(atoms, r_max=case.r_max)
    mapper = TypeMapper(
        chemical_symbol_to_type={
            symbol: index for index, symbol in enumerate(case.type_names)
        }
    )
    return mapper(data).to(device).to_dict()


def atomic_input(
    case: CaseSpec, device: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    return _input_from_atoms(case, atoms_from_fixture(case), device)


def cloned(data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in data.items()}


def python_reference(
    model_path: Path,
    case: CaseSpec,
    device: str | torch.device | None = None,
) -> ParityResult:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_deployed_model(
        model_path,
        device=device,
        freeze=True,
        set_global_options=True,
    )
    output = model(cloned(atomic_input(case, device=device)))
    energy = float(
        output[AtomicDataDict.TOTAL_ENERGY_KEY].detach().reshape(-1)[0].cpu()
    )
    forces = (
        output[AtomicDataDict.FORCE_KEY].detach().reshape(-1, 3).cpu().double().numpy()
    )
    spin_forces = None
    atom_virial = None
    bec = None
    if case.spin:
        spin_forces = (
            output[AtomicDataDict.SPIN_FORCE_KEY]
            .detach()
            .reshape(-1, 3)
            .cpu()
            .double()
            .numpy()
        )
    if AtomicDataDict.ATOM_VIRIAL_KEY in output:
        atom_virial = (
            output[AtomicDataDict.ATOM_VIRIAL_KEY]
            .detach()
            .reshape(-1, 3, 3)
            .cpu()
            .double()
            .numpy()
        )
    if case.bec:
        bec = (
            output[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY]
            .detach()
            .reshape(-1, 3, 3)
            .cpu()
            .double()
            .numpy()
        )
    arrays = [forces, spin_forces, atom_virial, bec]
    if not math.isfinite(energy) or any(
        array is not None and not np.isfinite(array).all() for array in arrays
    ):
        raise ValueError(f"nonfinite Python reference output for {case.name}")
    return ParityResult(
        energy_ev=energy,
        forces_ev_a=forces,
        spin_forces_ev=spin_forces,
        atom_virial_ev=atom_virial,
        bec_e=bec,
    )


# ---------------------------------------------------------------------------
# Per-frame trajectory parity interface.
#
# This API is additive: run_case / lammps_parity_result / python_reference
# above remain the frame-0 compatibility entry points. The trajectory
# interface joins dump and thermo by unique step, validates geometry and
# ID/type stability, and checks the observables defined for each route.
# ---------------------------------------------------------------------------

#: Every observable name the trajectory interface can carry.
ALL_OBSERVABLES = frozenset(
    ("energy", "forces", "spin_forces", "global_virial", "atom_virial", "bec")
)

#: Approved observable contract per route. NequIP-spin is global virial only:
#: pair_nequip_spin explicitly rejects per-atom virial and there is no
#: approved NequIP BEC route.
ROUTE_OBSERVABLES: dict[str, frozenset[str]] = {
    "nosoc": frozenset(
        ("energy", "forces", "spin_forces", "global_virial", "atom_virial")
    ),
    "soc": frozenset(
        ("energy", "forces", "spin_forces", "global_virial", "atom_virial")
    ),
    "nequip_spin": frozenset(("energy", "forces", "spin_forces", "global_virial")),
    "bec": frozenset(("energy", "forces", "global_virial", "atom_virial", "bec")),
}

#: Trajectory fixture input per route (tests_lammps/fixtures/in/).
TRAJECTORY_INPUTS = {
    "nosoc": "traj_spin.in",
    "soc": "traj_spin.in",
    "nequip_spin": "traj_nequip.in",
    "bec": "traj_bec.in",
}


class TrajectoryScopeError(ValueError):
    """A route was asked for an observable outside its approved contract."""


def require_observables(
    case_name: str, observables: Sequence[str] | None
) -> frozenset[str]:
    """Validate a requested observable set against the route contract."""

    if case_name not in ROUTE_OBSERVABLES:
        raise TrajectoryScopeError(f"unknown trajectory route: {case_name!r}")
    requested = (
        ROUTE_OBSERVABLES[case_name] if observables is None else frozenset(observables)
    )
    unknown = requested - ALL_OBSERVABLES
    if unknown:
        raise TrajectoryScopeError(f"unknown observables: {sorted(unknown)}")
    unsupported = requested - ROUTE_OBSERVABLES[case_name]
    if unsupported:
        raise TrajectoryScopeError(
            f"route {case_name!r} does not supply {sorted(unsupported)}; "
            f"supported observables: {sorted(ROUTE_OBSERVABLES[case_name])}"
        )
    return requested


@dataclass(frozen=True)
class TrajectoryFrame:
    """One validated trajectory state plus its observable results."""

    step: int
    ids: np.ndarray
    types: np.ndarray
    origin: np.ndarray
    cell: np.ndarray
    positions: np.ndarray
    pbc: tuple[bool, bool, bool] = (True, True, True)
    spin_directions: np.ndarray | None = None
    spin_magnitudes: np.ndarray | None = None
    energy_ev: float | None = None
    forces_ev_a: np.ndarray | None = None
    spin_forces_ev: np.ndarray | None = None
    global_virial_ev: np.ndarray | None = None
    atom_virial_ev: np.ndarray | None = None
    bec_e: np.ndarray | None = None


@dataclass(frozen=True)
class TrajectoryResult:
    case: str
    frames: tuple[TrajectoryFrame, ...]
    process: ProcessResult


def parse_trajectory_dump(path: Path) -> list[dict[str, object]]:
    """Parse a custom dump with box bounds, strict IDs and valid cells.

    Unlike parse_dump (frame-0 legacy helper), this keeps per-frame origin and
    cell (orthogonal or restricted triclinic), rejects malformed bounds, and
    rejects duplicate atom IDs.
    """

    lines = path.read_text().splitlines()
    frames: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        if lines[index] != "ITEM: TIMESTEP":
            index += 1
            continue
        step = int(lines[index + 1])
        if lines[index + 2] != "ITEM: NUMBER OF ATOMS":
            raise ValueError(f"malformed dump near line {index + 3}: {path}")
        atom_count = int(lines[index + 3])
        bounds_header = lines[index + 4]
        if not bounds_header.startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"missing box bounds at step {step} in {path}")
        # Orthogonal: "ITEM: BOX BOUNDS pp pp pp" (2 fields/axis); restricted
        # triclinic: "ITEM: BOX BOUNDS xy xz yz pp pp pp" (3 fields/axis with
        # tilt factors appended). General triclinic is rejected.
        boundary_fields = bounds_header.split()[3:]
        triclinic = any(field in ("xy", "xz", "yz") for field in boundary_fields)
        bounds: list[tuple[float, float]] = []
        tilt = [0.0, 0.0, 0.0]  # xy, xz, yz
        for axis in range(3):
            fields = lines[index + 5 + axis].split()
            expected = 3 if triclinic else 2
            if len(fields) != expected:
                raise ValueError(
                    f"malformed box bounds at step {step} in {path}: "
                    f"{lines[index + 5 + axis]}"
                )
            bounds.append((float(fields[0]), float(fields[1])))
            if triclinic:
                tilt[axis] = float(fields[2])
        if triclinic:
            xy, xz, yz = tilt
            # LAMMPS writes bound lo/hi; convert to true box lo/hi.
            (xlo_b, xhi_b), (ylo_b, yhi_b), (zlo, zhi) = bounds
            xlo = xlo_b - min(0.0, xy, xz, xy + xz)
            xhi = xhi_b - max(0.0, xy, xz, xy + xz)
            ylo = ylo_b - min(0.0, yz)
            yhi = yhi_b - max(0.0, yz)
            bounds = [(xlo, xhi), (ylo, yhi), (zlo, zhi)]
        index += 8
        if not lines[index].startswith("ITEM: ATOMS "):
            raise ValueError(f"missing atom header at step {step} in {path}")
        frame_columns = lines[index].split()[2:]
        index += 1
        atoms: list[dict[str, float]] = []
        for _ in range(atom_count):
            values = [float(value) for value in lines[index].split()]
            if len(values) != len(frame_columns) or not np.isfinite(values).all():
                raise ValueError(f"invalid atom row in {path}: {lines[index]}")
            atoms.append(dict(zip(frame_columns, values)))
            index += 1
        atoms.sort(key=lambda row: row["id"])
        ids = [int(row["id"]) for row in atoms]
        if len(set(ids)) != len(ids):
            duplicates = sorted({atom_id for atom_id in ids if ids.count(atom_id) > 1})
            raise ValueError(
                f"duplicate atom IDs {duplicates} at step {step} in {path}"
            )
        origin = np.asarray([bound[0] for bound in bounds], dtype=float)
        lengths = np.asarray([bound[1] - bound[0] for bound in bounds], dtype=float)
        if not (lengths > 0).all():
            raise ValueError(
                f"invalid cell lengths {lengths.tolist()} at step {step} in {path}"
            )
        cell = (
            np.array(
                [
                    [lengths[0], 0.0, 0.0],
                    [tilt[0], lengths[1], 0.0],
                    [tilt[1], tilt[2], lengths[2]],
                ]
            )
            if triclinic
            else np.diag(lengths)
        )
        frames.append(
            {
                "step": step,
                "origin": origin,
                "cell": cell,
                "atoms": atoms,
                "columns": frame_columns,
            }
        )
    if not frames:
        raise ValueError(f"no frames in {path}")
    return frames


def validate_trajectory(
    dump_frames: Sequence[Mapping[str, object]],
    thermo_rows: Sequence[Mapping[str, float]],
    steps: int,
) -> list[tuple[Mapping[str, object], Mapping[str, float]]]:
    """Strictly join dump frames and thermo rows by unique step.

    Requires exactly steps 0..steps on both sides, unique steps, and stable
    unique atom ID/type sets. Anything else raises instead of shortening or
    reordering the comparison.
    """

    expected = list(range(steps + 1))
    if len(dump_frames) != steps + 1:
        raise ValueError(f"expected {steps + 1} dump frames, got {len(dump_frames)}")
    dump_steps = [int(frame["step"]) for frame in dump_frames]
    if len(set(dump_steps)) != len(dump_steps):
        raise ValueError(f"duplicate dump steps: {sorted(dump_steps)}")
    if sorted(dump_steps) != expected:
        raise ValueError(
            f"dump steps must be exactly {expected[0]}..{expected[-1]}, "
            f"got {sorted(dump_steps)}"
        )
    if len(thermo_rows) != steps + 1:
        raise ValueError(f"expected {steps + 1} thermo rows, got {len(thermo_rows)}")
    thermo_steps: list[int] = []
    for row in thermo_rows:
        step_value = float(row["Step"])
        if step_value != int(step_value):
            raise ValueError(f"non-integer thermo step: {step_value}")
        thermo_steps.append(int(step_value))
    if len(set(thermo_steps)) != len(thermo_steps):
        raise ValueError(f"duplicate thermo steps: {sorted(thermo_steps)}")
    if sorted(thermo_steps) != expected:
        raise ValueError(
            f"thermo steps must be exactly {expected[0]}..{expected[-1]}, "
            f"got {sorted(thermo_steps)}"
        )
    thermo_by_step = dict(zip(thermo_steps, thermo_rows))

    reference_ids: np.ndarray | None = None
    reference_types: np.ndarray | None = None
    joined: list[tuple[Mapping[str, object], Mapping[str, float]]] = []
    for frame in sorted(dump_frames, key=lambda item: int(item["step"])):
        atoms = frame["atoms"]
        assert isinstance(atoms, list)
        ids = np.asarray([int(row["id"]) for row in atoms], dtype=int)
        types = np.asarray([int(row["type"]) for row in atoms], dtype=int)
        if len(set(ids.tolist())) != len(ids):
            raise ValueError(f"duplicate atom IDs at step {frame['step']}")
        if reference_ids is None:
            reference_ids = ids
            reference_types = types
        elif not np.array_equal(ids, reference_ids) or not np.array_equal(
            types, reference_types
        ):
            raise ValueError(
                f"atom IDs/types changed at step {frame['step']} "
                f"(expected stable {reference_ids.tolist()})"
            )
        joined.append((frame, thermo_by_step[int(frame["step"])]))
    return joined


def _assemble_trajectory_frame(
    case: CaseSpec,
    dump_frame: Mapping[str, object],
    thermo_row: Mapping[str, float],
) -> TrajectoryFrame:
    atoms = dump_frame["atoms"]
    assert isinstance(atoms, list)
    cell = np.asarray(dump_frame["cell"], dtype=float)
    volume = float(abs(np.linalg.det(cell)))
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError(f"invalid cell at step {dump_frame['step']}: {cell}")
    ids = np.asarray([int(row["id"]) for row in atoms], dtype=int)
    types = np.asarray([int(row["type"]) for row in atoms], dtype=int)
    positions = columns(dump_frame, ("x", "y", "z"))
    forces = columns(dump_frame, ("fx", "fy", "fz"))
    spin_directions = None
    spin_magnitudes = None
    spin_forces = None
    if case.spin:
        spin_directions = columns(
            dump_frame, tuple(f"c_spinprops[{index}]" for index in range(1, 4))
        )
        spin_magnitudes = columns(dump_frame, ("c_spinprops[4]",)).reshape(-1)
        if (spin_magnitudes < 0.0).any():
            raise ValueError(f"negative spin magnitude at step {dump_frame['step']}")
        spin_forces = (
            columns(dump_frame, tuple(f"c_spinprops[{index}]" for index in range(5, 8)))
            * H_BAR_EV_PS
        )
    press_names = tuple(f"c_press[{index}]" for index in range(1, 7))
    missing_press = [name for name in press_names if name not in thermo_row]
    if missing_press:
        raise ValueError(
            f"thermo missing pair-pressure columns {missing_press} "
            f"at step {dump_frame['step']}"
        )
    # W = P_pair * volume / nktv2p (positive sign; compute_pressure.cpp).
    global_virial = (
        np.asarray([float(thermo_row[name]) for name in press_names], dtype=float)
        * volume
        / LAMMPS_METAL_NKTV2P
    )
    atom_virial = None
    if "atom_virial" in ROUTE_OBSERVABLES[case.name]:
        virial_names = tuple(f"c_cstress[{index}]" for index in range(1, 10))
        missing = [name for name in virial_names if name not in dump_frame["columns"]]
        if missing:
            raise ValueError(
                f"route {case.name!r} requires per-atom virial but the dump "
                f"is missing {missing} at step {dump_frame['step']}"
            )
        atom_virial = _cstress_atom_virial(dump_frame)
    bec = None
    if "bec" in ROUTE_OBSERVABLES[case.name]:
        bec_names = tuple(f"c_becprops[{index}]" for index in range(1, 10))
        bec = columns(dump_frame, bec_names).reshape(-1, 3, 3)
    energy = float(thermo_row["PotEng"])
    observables = [forces, spin_forces, global_virial, atom_virial, bec]
    if not math.isfinite(energy) or any(
        array is not None and not np.isfinite(array).all() for array in observables
    ):
        raise ValueError(f"nonfinite observable at step {dump_frame['step']}")
    return TrajectoryFrame(
        step=int(dump_frame["step"]),
        ids=ids,
        types=types,
        origin=np.asarray(dump_frame["origin"], dtype=float),
        cell=cell,
        positions=positions,
        spin_directions=spin_directions,
        spin_magnitudes=spin_magnitudes,
        energy_ev=energy,
        forces_ev_a=forces,
        spin_forces_ev=spin_forces,
        global_virial_ev=global_virial,
        atom_virial_ev=atom_virial,
        bec_e=bec,
    )


def run_trajectory_case(
    case_name: str,
    model_path: Path,
    binary: Path,
    workdir: Path,
    env: Mapping[str, str],
    prefix: Sequence[str] = (),
    steps: int = MD_STEPS,
    frozen: bool = False,
) -> TrajectoryResult:
    """Run a trajectory fixture and return validated per-frame results."""

    case = CASES[case_name]
    dump_path = workdir / f"{case_name}.traj.dump"
    variables: dict[str, str | os.PathLike[str]] = {
        "data": case.data_path,
        "model": model_path,
        "dump": dump_path,
        "steps": str(steps),
        "frozen": "1" if frozen else "0",
    }
    if case.bec_path is not None:
        variables["bec"] = case.bec_path
    process = run_lammps(
        binary=binary,
        input_path=FIXTURE_ROOT / "in" / TRAJECTORY_INPUTS[case_name],
        workdir=workdir,
        env=env,
        variables=variables,
        prefix=prefix,
        dump_path=dump_path,
    )
    joined = validate_trajectory(
        parse_trajectory_dump(dump_path), parse_thermo(process.log_path), steps
    )
    frames = tuple(
        _assemble_trajectory_frame(case, dump_frame, thermo_row)
        for dump_frame, thermo_row in joined
    )
    return TrajectoryResult(case=case_name, frames=frames, process=process)


def frame_to_atoms(case: CaseSpec, frame: TrajectoryFrame) -> Atoms:
    """Reconstruct the ASE atoms of one frame's own state.

    Zero spins are preserved: AtomicData normalizes directions with an
    epsilon-clamped denominator (src/core/nequip/data/AtomicData.py).
    """

    symbols = [case.type_names[atom_type - 1] for atom_type in frame.types]
    atoms = Atoms(
        symbols=symbols,
        positions=frame.positions - frame.origin,
        cell=frame.cell,
        pbc=True,
    )
    if case.spin:
        assert frame.spin_directions is not None
        assert frame.spin_magnitudes is not None
        atoms.set_initial_magnetic_moments(
            frame.spin_directions * frame.spin_magnitudes[:, None]
        )
    return atoms


def _observables_from_output(
    case: CaseSpec,
    output: Mapping[str, torch.Tensor],
    requested: frozenset[str],
    context: str,
) -> dict[str, object]:
    """Extract requested observables from one model output dict."""

    def tensor(key: str) -> np.ndarray:
        if key not in output:
            raise ValueError(f"{context}: model output missing {key!r}")
        return output[key].detach().cpu().double().numpy()

    values: dict[str, object] = {}
    if "energy" in requested:
        values["energy_ev"] = float(
            tensor(AtomicDataDict.TOTAL_ENERGY_KEY).reshape(-1)[0]
        )
    if "forces" in requested:
        values["forces_ev_a"] = tensor(AtomicDataDict.FORCE_KEY).reshape(-1, 3)
    if "spin_forces" in requested:
        values["spin_forces_ev"] = tensor(AtomicDataDict.SPIN_FORCE_KEY).reshape(-1, 3)
    if "global_virial" in requested:
        virial = tensor(AtomicDataDict.VIRIAL_KEY).reshape(-1, 3, 3)[0]
        values["global_virial_ev"] = np.asarray(
            [
                virial[0, 0],
                virial[1, 1],
                virial[2, 2],
                virial[0, 1],
                virial[0, 2],
                virial[1, 2],
            ],
            dtype=float,
        )
    if "atom_virial" in requested:
        values["atom_virial_ev"] = tensor(AtomicDataDict.ATOM_VIRIAL_KEY).reshape(
            -1, 3, 3
        )
    if "bec" in requested:
        values["bec_e"] = tensor(AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY).reshape(
            -1, 3, 3
        )
    for name, value in values.items():
        if isinstance(value, float):
            finite = math.isfinite(value)
        else:
            finite = bool(np.isfinite(np.asarray(value)).all())
        if not finite:
            raise ValueError(f"{context}: nonfinite Python reference {name}")
    return values


def _reference_frame(
    frame: TrajectoryFrame, observables: Mapping[str, object]
) -> TrajectoryFrame:
    return TrajectoryFrame(
        step=frame.step,
        ids=frame.ids,
        types=frame.types,
        origin=frame.origin,
        cell=frame.cell,
        positions=frame.positions,
        pbc=frame.pbc,
        spin_directions=frame.spin_directions,
        spin_magnitudes=frame.spin_magnitudes,
        **dict(observables),
    )


def _evaluate_frames_torchscript(
    case: CaseSpec,
    frames: Sequence[TrajectoryFrame],
    model_path: Path,
    target_device: str | torch.device,
    requested: frozenset[str],
) -> tuple[TrajectoryFrame, ...]:
    model, _ = load_deployed_model(
        model_path,
        device=target_device,
        freeze=True,
        set_global_options=True,
    )
    evaluated = []
    for frame in frames:
        data = cloned(
            _input_from_atoms(case, frame_to_atoms(case, frame), target_device)
        )
        output = model(data)
        evaluated.append(
            _reference_frame(
                frame,
                _observables_from_output(
                    case, output, requested, f"torchscript step {frame.step}"
                ),
            )
        )
    return tuple(evaluated)


def _evaluate_frames_pt2(
    case: CaseSpec,
    frames: Sequence[TrajectoryFrame],
    pt2_path: Path,
    target_device: str | torch.device,
    requested: frozenset[str],
) -> tuple[TrajectoryFrame, ...]:
    metadata = deploy.read_deploy_metadata_sidecar(pt2_path)
    input_signature = tuple(metadata[deploy.INPUT_SIGNATURE_KEY].split())
    output_signature = tuple(metadata[deploy.OUTPUT_SIGNATURE_KEY].split())
    compiled = torch._inductor.aoti_load_package(
        str(pt2_path), run_single_threaded=True
    )
    evaluated = []
    for frame in frames:
        data = _input_from_atoms(case, frame_to_atoms(case, frame), target_device)
        missing = [key for key in input_signature if key not in data]
        if missing:
            raise ValueError(
                f"pt2 step {frame.step}: input signature keys {missing} "
                "not produced by AtomicData"
            )
        output = (
            _run_pt2_with_ghost_graph(
                case,
                compiled,
                data,
                input_signature,
                output_signature,
                f"pt2 step {frame.step}",
            )
            if case.name != "nequip_spin"
            else _run_pt2_plain(
                compiled,
                data,
                input_signature,
                output_signature,
                f"pt2 step {frame.step}",
            )
        )
        evaluated.append(
            _reference_frame(
                frame,
                _observables_from_output(
                    case, output, requested, f"pt2 step {frame.step}"
                ),
            )
        )
    return tuple(evaluated)


def _run_pt2_plain(
    compiled,
    data: dict[str, torch.Tensor],
    input_signature: tuple[str, ...],
    output_signature: tuple[str, ...],
    context: str,
) -> dict[str, torch.Tensor]:
    """Plain signature-ordered PT2 evaluation for tag-merged graphs.

    Used for nequip_spin: pair_nequip_spin.cpp merges periodic images into
    one graph node per physical atom (tag dedup), so its LAMMPS-side graph
    matches the unexpanded AtomicData construction; ghost expansion would
    double-count (5090 targeted run 2291000: expanded 1.1381 vs plain
    -0.0991 matching LAMMPS-PT2).
    """

    args = tuple(data[key].clone() for key in input_signature)
    raw = tuple(compiled(*args))
    absorb_aoti_lapack_poison()
    if len(raw) != len(output_signature):
        raise ValueError(
            f"{context}: expected {len(output_signature)} outputs from the "
            f"sidecar signature, got {len(raw)}"
        )
    return dict(zip(output_signature, raw))


def _run_pt2_with_ghost_graph(
    case: CaseSpec,
    compiled,
    data: dict[str, torch.Tensor],
    input_signature: tuple[str, ...],
    output_signature: tuple[str, ...],
    context: str,
) -> dict[str, torch.Tensor]:
    """Evaluate a PBC-blind PT2 package on a ghost-expanded graph.

    The exported ABI (``pos spin edge_index atom_types``) carries no cell or
    edge_cell_shift: the package computes edge vectors as raw position
    differences. LAMMPS feeds it a graph in which periodic images appear as
    ghost nodes with unwrapped positions, then reverse-communicates per-atom
    contributions onto owners (pair_allegro_spin.cpp graph construction).
    Feeding wrapped positions directly changes boundary edge vectors and
    produced wrong energies/forces for the allegro routes (5090 job 2290874,
    diagnostic job 2290961/2290969: TS 17.0248 vs AOTI 12.7049 on identical
    input). This helper mirrors the C++ construction: edges with nonzero
    cell shift are retargeted to ghost copies at ``pos + shift @ cell``;
    per-atom outputs are summed over every copy of each physical atom, and
    the energy uses the C++ offset convention
    ``sum(atomic[owners]) + (total - sum(atomic[all]))``.
    """

    pos = data["pos"]
    n_atoms = pos.size(0)
    device = pos.device
    cell = data["cell"].reshape(3, 3)
    edge_index = data["edge_index"]
    shifts = data["edge_cell_shift"].to(cell.dtype)
    shift_keys = shifts.round().to(torch.int64)

    ghost_pos: list[torch.Tensor] = []
    ghost_owner: list[int] = []
    ghost_map: dict[tuple[int, int, int, int], int] = {}
    remapped = edge_index.clone()
    for edge in range(edge_index.size(1)):
        key_shift = tuple(int(v) for v in shift_keys[edge].tolist())
        if key_shift == (0, 0, 0):
            continue
        target = int(edge_index[1, edge])
        key = (target, *key_shift)
        ghost = ghost_map.get(key)
        if ghost is None:
            ghost = n_atoms + len(ghost_pos)
            ghost_map[key] = ghost
            ghost_pos.append(pos[target] + shifts[edge] @ cell)
            ghost_owner.append(target)
        remapped[1, edge] = ghost

    if ghost_pos:
        pos_ext = torch.cat([pos, torch.stack(ghost_pos)], dim=0)
        owner = torch.cat(
            [
                torch.arange(n_atoms, device=device),
                torch.tensor(ghost_owner, device=device, dtype=torch.long),
            ]
        )
    else:
        pos_ext = pos
        owner = torch.arange(n_atoms, device=device)

    expanded: dict[str, torch.Tensor] = {"pos": pos_ext, "edge_index": remapped}
    for key in input_signature:
        if key in ("pos", "edge_index"):
            continue
        source = data[key]
        if key in ("spin",) and ghost_pos:
            source = torch.cat([source, source[owner[n_atoms:]]], dim=0)
        elif key in ("atom_types",) and ghost_pos:
            source = torch.cat([source, source[owner[n_atoms:]]], dim=0)
        expanded[key] = source

    args = tuple(expanded[key].clone() for key in input_signature)
    raw = tuple(compiled(*args))
    absorb_aoti_lapack_poison()
    if len(raw) != len(output_signature):
        raise ValueError(
            f"{context}: expected {len(output_signature)} outputs from the "
            f"sidecar signature, got {len(raw)}"
        )
    full = dict(zip(output_signature, raw))

    def reduce_owned(tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.reshape(tensor.size(0), -1).to(torch.float64)
        summed = torch.zeros(
            (n_atoms, flat.size(1)), dtype=torch.float64, device=device
        )
        summed.index_add_(0, owner, flat)
        return summed.to(tensor.dtype).reshape((n_atoms, *tensor.shape[1:]))

    output: dict[str, torch.Tensor] = {}
    atomic_energy = full.get("atomic_energy")
    if atomic_energy is not None:
        atomic_owned = reduce_owned(atomic_energy)
        output[AtomicDataDict.PER_ATOM_ENERGY_KEY] = atomic_owned
    if "total_energy" in full:
        if atomic_energy is not None:
            offset = (
                full["total_energy"].reshape(-1)[0].to(torch.float64)
                - atomic_energy.reshape(-1).to(torch.float64).sum()
            )
            total = atomic_owned.reshape(-1).to(torch.float64).sum() + offset
            output[AtomicDataDict.TOTAL_ENERGY_KEY] = total.reshape(1, 1).to(
                full["total_energy"].dtype
            )
        else:
            output[AtomicDataDict.TOTAL_ENERGY_KEY] = full["total_energy"]
    for key in ("forces", "spin_forces", "atom_virial"):
        if key in full:
            output[key] = reduce_owned(full[key])
    for key in ("virial", "born_effective_charges"):
        if key in full:
            tensor = full[key]
            if key == "born_effective_charges" and tensor.size(0) == pos_ext.size(0):
                # Intensive per-atom property: the owner's own copy.
                output[key] = tensor[:n_atoms]
            else:
                output[key] = tensor
    return output


def python_trajectory_reference(
    case: CaseSpec,
    frames: Sequence[TrajectoryFrame],
    model_paths: Mapping[str, Path],
    target_device: str | torch.device,
    observables: Sequence[str] | None = None,
) -> dict[str, tuple[TrajectoryFrame, ...]]:
    """Evaluate TorchScript/PT2 references on every frame's own state.

    ``model_paths`` maps backend names to artifacts: ``"torchscript"`` (.pth)
    and optionally ``"pt2"`` (.pt2). Each frame is reconstructed with its own
    cell/PBC/types/spins. Returns one validated frame tuple per backend, with
    geometry copied from ``frames`` and observables filled per the route
    capability contract (TrajectoryScopeError otherwise).
    """

    requested = require_observables(case.name, observables)
    results: dict[str, tuple[TrajectoryFrame, ...]] = {}
    scripted_path = model_paths.get("torchscript")
    if scripted_path is not None:
        results["torchscript"] = _evaluate_frames_torchscript(
            case, frames, scripted_path, target_device, requested
        )
    pt2_path = model_paths.get("pt2")
    if pt2_path is not None:
        results["pt2"] = _evaluate_frames_pt2(
            case, frames, pt2_path, target_device, requested
        )
    if not results:
        raise ValueError("model_paths must supply 'torchscript' and/or 'pt2'")
    return results


def _check_close(
    name: str,
    expected: object,
    actual: object,
    rtol: float,
    atol: float,
    context: str,
) -> None:
    if expected is None:
        if actual is not None:
            raise AssertionError(f"{context}{name}: unexpected value present")
        return
    if actual is None:
        raise AssertionError(f"{context}{name}: value missing")
    expected_array = np.atleast_1d(np.asarray(expected, dtype=float))
    actual_array = np.atleast_1d(np.asarray(actual, dtype=float))
    if expected_array.shape != actual_array.shape:
        raise AssertionError(
            f"{context}{name}: shape {actual_array.shape} != " f"{expected_array.shape}"
        )
    finite = np.isfinite(actual_array)
    if not finite.all():
        index = tuple(int(i) for i in np.argwhere(~finite)[0])
        raise AssertionError(f"{context}{name}: nonfinite value at index {index}")
    bad = ~np.isclose(expected_array, actual_array, rtol=rtol, atol=atol)
    if bad.any():
        index = tuple(int(i) for i in np.argwhere(bad)[0])
        raise AssertionError(
            f"{context}{name} mismatch at index {index}: "
            f"expected {expected_array[index]!r}, got {actual_array[index]!r} "
            f"(rtol={rtol}, atol={atol})"
        )


def assert_trajectory_parity(
    reference: Sequence[TrajectoryFrame],
    actual: Sequence[TrajectoryFrame],
    *,
    rtol: float,
    energy_atol: float,
    force_atol: float,
    spin_force_atol: float | None = None,
    virial_atol: float | None = None,
    bec_atol: float | None = None,
    label: str = "",
) -> None:
    """Compare two frame sequences observable-by-observable.

    Steps must match exactly in number, identity and order. Every observable
    present on the reference frames is compared; failures raise AssertionError
    naming the step, observable and first offending index (atom, component).
    """

    reference_steps = [frame.step for frame in reference]
    actual_steps = [frame.step for frame in actual]
    if actual_steps != reference_steps:
        raise AssertionError(
            f"{label}step sequence mismatch: expected {reference_steps}, "
            f"got {actual_steps}"
        )
    context_prefix = f"{label} " if label else ""
    for reference_frame, actual_frame in zip(reference, actual):
        context = f"{context_prefix}step {reference_frame.step} "
        if not np.array_equal(actual_frame.ids, reference_frame.ids):
            raise AssertionError(f"{context}atom IDs differ")
        _check_close(
            "energy",
            reference_frame.energy_ev,
            actual_frame.energy_ev,
            rtol,
            energy_atol,
            context,
        )
        _check_close(
            "forces",
            reference_frame.forces_ev_a,
            actual_frame.forces_ev_a,
            rtol,
            force_atol,
            context,
        )
        if spin_force_atol is not None:
            _check_close(
                "spin_forces",
                reference_frame.spin_forces_ev,
                actual_frame.spin_forces_ev,
                rtol,
                spin_force_atol,
                context,
            )
        if virial_atol is not None:
            _check_close(
                "global_virial",
                reference_frame.global_virial_ev,
                actual_frame.global_virial_ev,
                rtol,
                virial_atol,
                context,
            )
            _check_close(
                "atom_virial",
                reference_frame.atom_virial_ev,
                actual_frame.atom_virial_ev,
                rtol,
                virial_atol,
                context,
            )
        if bec_atol is not None:
            _check_close(
                "bec",
                reference_frame.bec_e,
                actual_frame.bec_e,
                rtol,
                bec_atol,
                context,
            )
