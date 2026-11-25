"""Fast, parallel VASP 5 & VASP 6 magnetic OUTCAR/OSZICAR -> SpinGNN2 extxyz converter.

Handles constrained noncollinear magnetic calculations (I_CONSTRAINED_M = 1, 2, 4)
across both VASP 5.x and VASP 6.x with automatic factor-of-2 detection from the
exact penalty-energy consistency relation:
  - Modes 1 & 2 (quadratic penalty E_p = lambda * sum(|dMW_i|^2)):
      R = sum(|P_i|^2) / (lambda * E_p)
      VASP 5 prints P_i = 1 * lambda * dMW_i -> R ~ 1.0 -> C_i = 2 * P_i
      VASP 6 prints P_i = 2 * lambda * dMW_i -> R ~ 4.0 -> C_i = 1 * P_i
  - Mode 4 (VASP 6.4+ linear penalty E_p = lambda * sum(|MW_i| - u_i . MW_i)):
      R_4 = sum(|MW_i| * |P_i|^2) / (lambda * E_p)
      VASP 6 prints P_i = lambda * (MW_hat_i - u_i) -> R_4 ~ 2.0 -> C_i = 1 * P_i
where C_i = +dE_p/dM_{W,i} = -dE_DFT/dM_{W,i} (eV / mu_B), and the tangential
spin torque label is:
  spin_forces_vert_i = |MW_int,i| * (C_i - (C_i . s_hat_i) * s_hat_i)   [eV]
with penalty energy E_p subtracted from the total energy (F - E_p default, consistent
with VASP free-energy forces/stress; or E0 - E_p via --energy-source zero), SAXIS
Euler rotation applied to spin vectors, and nonmagnetic sites cleanly zeroed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import math
import multiprocessing
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.data import atomic_numbers
    from ase.io import read as ase_read
    from ase.io import write as ase_write
except ImportError as exc:  # pragma: no cover - ASE is optional in minimal installs
    Atoms = None  # type: ignore[assignment]
    SinglePointCalculator = None  # type: ignore[assignment]
    atomic_numbers = None  # type: ignore[assignment]
    ase_read = None  # type: ignore[assignment]
    ase_write = None  # type: ignore[assignment]
    _ASE_IMPORT_ERROR: Optional[Exception] = exc
else:
    _ASE_IMPORT_ERROR = None


# 1 kB * A^3 -> eV: 1e8 Pa * 1e-30 m^3 / 1.602176634e-19 J/eV = 6.241509074460763e-4
KB_TO_EV_PER_A3 = 0.0006241509074460763
_NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_ITER_RE = re.compile(r"Iteration\s+(\d+)\s*\(\s*(\d+)\s*\)")
_SCF_RE = re.compile(r"^\s*(?:DAV|RMM|CG|CGA|DMP|DIA|SDA|RMM-DIIS)\s*:\s*(\d+)\s+")
_SUMMARY_RE = re.compile(r"^\s*(\d+)\s+(?:F\s*=|T\s*=)")
_PARAM_ASSIGN_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=")

_PARSED_KEYS = {
    "I_CONSTRAINED_M",
    "M_CONSTR",
    "LAMBDA",
    "RWIGS",
    "SAXIS",
    "LNONCOLLINEAR",
    "LSORBIT",
    "LSPIRAL",
    "QSPIRAL",
    "NSW",
    "NELM",
    "NIONS",
    "IBRION",
    "LDAU",
    "LDAUTYPE",
    "LDAUL",
    "LDAUU",
    "LDAUJ",
}
_VECTOR_KEYS = {"M_CONSTR", "RWIGS", "SAXIS", "QSPIRAL", "LDAUL", "LDAUU", "LDAUJ"}
_BOOL_KEYS = {"LNONCOLLINEAR", "LSORBIT", "LSPIRAL", "LDAU"}
_INT_KEYS = {"I_CONSTRAINED_M", "NSW", "NELM", "NIONS", "IBRION", "LDAUTYPE"}


class VaspSpinParseError(ValueError):
    """Raised when VASP OUTCAR/OSZICAR parsing or physics self-consistency fails."""


@dataclass
class VaspSpinFrame:
    """Parsed and validated representation of a single VASP calculation."""

    outcar_path: str
    oszicar_path: Optional[str]
    vasp_version: str
    vasp_major: int
    symbols: List[str]
    cell: np.ndarray
    positions: np.ndarray
    forces: np.ndarray
    stress: Optional[np.ndarray]
    free_energy: float
    energy_zero: float
    converged: bool
    last_ionic_step: int
    last_electronic_step: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    projected_mag: Optional[np.ndarray] = None
    born_effective_charges: Optional[np.ndarray] = None
    piezoelectric: Optional[np.ndarray] = None
    oszicar: Optional[Dict[str, Any]] = None


def _as_float(token: str) -> float:
    try:
        val = float(token.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise VaspSpinParseError(f"Invalid float token: {token!r}") from exc
    if not math.isfinite(val):
        raise VaspSpinParseError(f"Non-finite float token: {token!r}")
    return val


def _expand_tokens(tokens: Sequence[str]) -> List[float]:
    values: List[float] = []
    for token in tokens:
        if "*" in token:
            count_s, val_s = token.split("*", 1)
            if not count_s.isdigit():
                raise VaspSpinParseError(f"Invalid repeated token: {token!r}")
            count = int(count_s)
            if count < 0 or count > 1_000_000:
                raise VaspSpinParseError(f"Out-of-range repeat count: {token!r}")
            values.extend([_as_float(val_s)] * count)
        else:
            values.append(_as_float(token))
    return values


def _parse_kv_block(text: str) -> Dict[str, Any]:
    """Parse VASP INCAR-style or OUTCAR runtime parameter assignments."""
    result: Dict[str, Any] = {}
    joined = re.sub(r"\\\s*\n", " ", text)
    for line in joined.splitlines():
        line = re.split(r"[!#]", line, maxsplit=1)[0]
        matches = [
            m
            for m in _PARAM_ASSIGN_RE.finditer(line)
            if m.group(1).upper() in _PARSED_KEYS
        ]
        for idx, match in enumerate(matches):
            key = match.group(1).upper()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
            raw = line[match.end() : end].split(";", 1)[0].strip()
            tokens = raw.split()
            if not tokens:
                continue
            if key in _BOOL_KEYS:
                word = tokens[0].strip(".").upper()
                if word in ("T", "TRUE"):
                    result[key] = True
                elif word in ("F", "FALSE"):
                    result[key] = False
            elif key in _VECTOR_KEYS:
                numeric = []
                for tok in tokens:
                    if not re.fullmatch(r"(?:\d+\*)?" + _NUM_RE, tok):
                        break
                    numeric.append(tok)
                if numeric:
                    vals = _expand_tokens(numeric)
                    if key == "LDAUL":
                        result[key] = [int(round(v)) for v in vals]
                    else:
                        result[key] = vals
            else:
                if not re.fullmatch(_NUM_RE, tokens[0]):
                    continue
                val = _as_float(tokens[0])
                if key in _INT_KEYS:
                    if val != int(val):
                        continue
                    result[key] = int(val)
                else:
                    result[key] = val
    return result


def _merge_incar_params(
    runtime_params: Dict[str, Any],
    incar_params: Dict[str, Any],
    *,
    strict: bool,
    source_label: str,
) -> Dict[str, Any]:
    """Merge INCAR parameters without letting stale INCAR silently override OUTCAR."""
    merged = dict(runtime_params)
    for key, val in incar_params.items():
        if key not in merged:
            merged[key] = val
            continue
        if strict and key in ("I_CONSTRAINED_M", "LAMBDA", "SAXIS"):
            rval = merged[key]
            if key == "I_CONSTRAINED_M" and int(val) != int(rval):
                raise VaspSpinParseError(
                    f"{source_label}: I_CONSTRAINED_M={val} contradicts OUTCAR runtime value {rval}"
                )
            if key == "LAMBDA" and abs(float(val) - float(rval)) > max(
                1e-3, 1e-3 * abs(float(rval))
            ):
                raise VaspSpinParseError(
                    f"{source_label}: LAMBDA={val} contradicts OUTCAR runtime value {rval}"
                )
            if key == "SAXIS":
                v_a = np.asarray(val, dtype=np.float64)
                v_b = np.asarray(rval, dtype=np.float64)
                if v_a.shape == (3,) and v_b.shape == (3,):
                    n_a = float(np.linalg.norm(v_a))
                    n_b = float(np.linalg.norm(v_b))
                    if (
                        n_a > 0
                        and n_b > 0
                        and np.max(np.abs(v_a / n_a - v_b / n_b)) > 1e-3
                    ):
                        raise VaspSpinParseError(
                            f"{source_label}: SAXIS={val} contradicts OUTCAR runtime value {rval}"
                        )
    return merged


def saxis_rotation_matrix(saxis: Sequence[float]) -> np.ndarray:
    """Return the VASP 3x3 rotation matrix mapping SAXIS frame -> Cartesian frame."""
    axis = np.asarray(saxis, dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise VaspSpinParseError(f"SAXIS must be a finite 3-vector, got {saxis!r}")
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise VaspSpinParseError("SAXIS must be nonzero")
    alpha = math.atan2(axis[1], axis[0])
    beta = math.atan2(math.hypot(axis[0], axis[1]), axis[2])
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    return np.array(
        [
            [ca * cb, -sa, ca * sb],
            [sa * cb, ca, sa * sb],
            [-sb, 0.0, cb],
        ],
        dtype=np.float64,
    )


def _parse_printed_saxis_matrix(text: str) -> Optional[np.ndarray]:
    marker = "transformation matrix from SAXIS to cartesian coordinates"
    if marker not in text:
        return None
    rows: List[List[float]] = []
    for line in text.rsplit(marker, 1)[1].splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= {"-"}:
            continue
        tokens = stripped.split()
        if len(tokens) == 6 and tokens[1::2] == ["m_x", "m_y", "m_z"]:
            rows.append([_as_float(tok) for tok in tokens[::2]])
            if len(rows) == 3:
                return np.asarray(rows, dtype=np.float64)
        else:
            break
    return None


def _parse_species_from_outcar(text: str, n_species: int) -> List[str]:
    """Parse chemical species list from VRHFIN or POTCAR header lines (VASP 5 & 6)."""
    vrhfin = re.findall(r"(?m)^\s*VRHFIN\s*=\s*([A-Z][a-z]?)\s*:", text)
    if len(vrhfin) == n_species:
        return vrhfin

    potcar_lines = re.findall(r"(?m)^\s*POTCAR:\s*([^\n]+)", text)
    potcar_syms: List[str] = []
    for payload in potcar_lines:
        toks = payload.split()
        if len(toks) >= 2:
            raw_sym = re.split(r"[_.]", toks[1], maxsplit=1)[0]
            if re.fullmatch(r"[A-Z][a-z]?", raw_sym):
                if atomic_numbers is None or raw_sym in atomic_numbers:
                    potcar_syms.append(raw_sym)
    if len(potcar_syms) == n_species:
        return potcar_syms
    if len(potcar_syms) == 2 * n_species:
        # VASP prints POTCAR header lines once per species at top, and once per species in PAW block.
        return potcar_syms[:n_species]
    if len(vrhfin) > n_species:
        return vrhfin[:n_species]
    raise VaspSpinParseError(
        f"Could not determine {n_species} species from VRHFIN ({len(vrhfin)}) "
        f"or POTCAR ({len(potcar_syms)}) lines"
    )


def _is_oszicar_block_boundary(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _SCF_RE.match(line) or _SUMMARY_RE.match(line):
        return True
    if re.match(r"E_p\s*=", stripped) or re.match(r"ion\s+", stripped):
        return True
    if re.match(r"^(?:N|RMM|DAV|CG|CGA|DMP|DIA|SDA)\b", stripped):
        return True
    return False


def _parse_atom_table(
    lines: Sequence[str],
    start: int,
    ncols: int,
    natom: int,
    *,
    source: str,
    table_name: str,
) -> Tuple[Dict[int, List[float]], int]:
    """Parse 1-indexed atom rows `ion v1 ... vk` supporting merged Fortran columns."""
    rows: Dict[int, List[float]] = {}
    cursor = start
    while cursor < len(lines):
        line = lines[cursor]
        stripped = line.strip()
        if not stripped or set(stripped) <= {"-"}:
            if rows:
                break
            cursor += 1
            continue
        if _is_oszicar_block_boundary(line):
            break
        m = re.match(r"^\s*(\d+)\s+(.*)$", line)
        if m is None:
            if rows:
                break
            raise VaspSpinParseError(
                f"{source}: malformed row in {table_name} table: {line!r}"
            )
        atom_idx = int(m.group(1)) - 1
        rest = m.group(2)
        if "*" in rest:
            raise VaspSpinParseError(
                f"{source}: Fortran overflow ('*') in {table_name} row {atom_idx + 1}: {line!r}"
            )
        nums = re.findall(_NUM_RE, rest)
        if len(nums) != ncols:
            raise VaspSpinParseError(
                f"{source}: expected {ncols} floats in {table_name} row {atom_idx + 1}, "
                f"got {len(nums)}: {line!r}"
            )
        if atom_idx < 0 or atom_idx >= natom or atom_idx in rows:
            raise VaspSpinParseError(
                f"{source}: invalid/duplicate atom index {atom_idx + 1} in {table_name}"
            )
        rows[atom_idx] = [_as_float(tok) for tok in nums]
        cursor += 1
    return rows, cursor


def _parse_outcar_magnetization(
    final_lines: Sequence[str],
    natom: int,
    *,
    require_noncollinear: bool = False,
) -> Optional[np.ndarray]:
    headers = []
    for idx, line in enumerate(final_lines):
        m = re.match(r"^\s*magnetization\s*\(([xyz])\)\s*$", line, re.IGNORECASE)
        if m:
            headers.append((idx, m.group(1).lower()))
    x_starts = [idx for idx, axis in headers if axis == "x"]
    if not x_starts:
        return None
    selected = [(idx, axis) for idx, axis in headers if idx >= x_starts[-1]]
    axes = [axis for _, axis in selected]
    if require_noncollinear:
        if axes != ["x", "y", "z"]:
            raise VaspSpinParseError(
                f"Noncollinear/SOC OUTCAR requires magnetization (x,y,z), got {axes}"
            )
    elif axes not in (["x"], ["x", "y", "z"]):
        raise VaspSpinParseError(
            f"Incomplete final magnetization tables in OUTCAR: {axes}"
        )

    mag = np.zeros((natom, 3), dtype=np.float64)
    for pos, (start, axis) in enumerate(selected):
        end = selected[pos + 1][0] if pos + 1 < len(selected) else len(final_lines)
        block = final_lines[start + 1 : end]
        hdr_idx = next(
            (i for i, ln in enumerate(block) if ln.strip().startswith("# of ion")),
            None,
        )
        if hdr_idx is None:
            raise VaspSpinParseError(
                f"magnetization ({axis}) table missing '# of ion' header"
            )
        cols = block[hdr_idx].split()
        if "tot" not in cols:
            raise VaspSpinParseError(
                f"magnetization ({axis}) table missing 'tot' column"
            )
        # '# of ion' occupies 3 tokens, remaining columns are numeric channels ending in 'tot'
        num_cols = len(cols) - 3
        tot_col = cols.index("tot") - 3
        rows, _ = _parse_atom_table(
            block,
            hdr_idx + 1,
            num_cols,
            natom,
            source="OUTCAR",
            table_name=f"magnetization ({axis})",
        )
        if len(rows) != natom:
            raise VaspSpinParseError(
                f"magnetization ({axis}) table has {len(rows)} rows, expected {natom}"
            )
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        for atom_idx in range(natom):
            mag[atom_idx, axis_idx] = rows[atom_idx][tot_col]
    return mag


def _parse_outcar_born_and_piezo(
    text: str, natom: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    born_map: Dict[int, Dict[int, List[float]]] = {}
    current_ion: Optional[int] = None
    piezo_map: Dict[str, List[float]] = {}
    read_born = False
    read_piezo = False
    for raw_line in text.splitlines():
        clean = raw_line.strip().strip("-").strip()
        if read_born:
            if not clean:
                continue
            ion_m = re.match(r"^ion\s+(\d+)\s*$", clean)
            if ion_m:
                current_ion = int(ion_m.group(1))
                if current_ion < 1 or current_ion > natom or current_ion in born_map:
                    raise VaspSpinParseError(
                        f"OUTCAR: invalid/duplicate BEC ion index {current_ion}"
                    )
                born_map[current_ion] = {}
                continue
            if clean.startswith("CHARGE:") or clean.startswith("LOOP+:"):
                read_born = False
                current_ion = None
            else:
                row_m = re.match(r"^\s*(\d+)\s+(.*)$", clean)
                if row_m is not None and current_ion is not None:
                    comp = int(row_m.group(1))
                    nums = re.findall(_NUM_RE, row_m.group(2))
                    if comp in (1, 2, 3) and len(nums) == 3:
                        if comp in born_map[current_ion]:
                            raise VaspSpinParseError(
                                f"OUTCAR: duplicate BEC component {comp} for ion {current_ion}"
                            )
                        born_map[current_ion][comp] = [_as_float(x) for x in nums]
                        continue
                if born_map:
                    read_born = False
                    current_ion = None
        if clean.startswith("BORN EFFECTIVE CHARGES"):
            born_map = {}
            current_ion = None
            read_born = True

        if read_piezo:
            if clean.startswith("XX") or not clean:
                continue
            if "(C/m^2)" in clean or "IONIC CONTR" in clean:
                read_piezo = False
            else:
                axis_m = re.match(r"^([xyzXYZ])\s+(.*)$", clean)
                if axis_m is not None:
                    axis_key = axis_m.group(1).lower()
                    if "*" in axis_m.group(2):
                        raise VaspSpinParseError(
                            f"OUTCAR: Fortran overflow in piezoelectric row {axis_key!r}"
                        )
                    nums = re.findall(_NUM_RE, axis_m.group(2))
                    if len(nums) == 6:
                        if axis_key in piezo_map:
                            raise VaspSpinParseError(
                                f"OUTCAR: duplicate piezoelectric axis row {axis_key!r}"
                            )
                        piezo_map[axis_key] = [_as_float(x) for x in nums]
                        continue
                if piezo_map:
                    read_piezo = False
        if (
            clean.startswith("PIEZOELECTRIC TENSOR")
            and "(e  Angst)" in clean
            and "IONIC CONTR" not in clean
        ):
            piezo_map = {}
            read_piezo = True

    born: Optional[np.ndarray] = None
    if len(born_map) == natom and all(
        set(born_map.get(i + 1, {}).keys()) == {1, 2, 3} for i in range(natom)
    ):
        flat_rows = [
            born_map[i + 1][1] + born_map[i + 1][2] + born_map[i + 1][3]
            for i in range(natom)
        ]
        born = np.asarray(flat_rows, dtype=np.float64)

    piezo: Optional[np.ndarray] = None
    if set(piezo_map.keys()) == {"x", "y", "z"}:
        piezo = np.zeros((3, 3, 3), dtype=np.float64)
        for i, axis_key in enumerate(("x", "y", "z")):
            s1, s2, s3, s4, s5, s6 = piezo_map[axis_key]
            piezo[i] = np.array(
                [[s1, s4, s6], [s4, s2, s5], [s6, s5, s3]], dtype=np.float64
            )
    return born, piezo


def parse_oszicar_constrained(
    oszicar_path: Path | str, natom: int
) -> Optional[Dict[str, Any]]:
    """Parse ONLY the final converged constrained-moment state from OSZICAR (VASP 5 & 6).

    Pre-convergence SCF blocks are skipped without parsing or recording. Returns None
    ONLY if OSZICAR contains zero constrained-moment records at all; raises
    VaspSpinParseError if constrained records appear but the final converged step is
    incomplete, truncated, or followed by an unfinished step.
    """
    source = str(oszicar_path)
    text = Path(oszicar_path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    has_any_constrained = any(
        re.match(r"^\s*(?:E_p\s*=|ion\s+MW_int\s+M_int\s*$|ion\s+lambda\*\S+\s*$)", ln)
        for ln in lines
    )
    if not has_any_constrained:
        return None

    # Find the LAST ionic summary line first (so a truncated trailing summary like `2 F= ...`
    # without `E0=` is rejected rather than silently falling back to an earlier step).
    sum_idx: Optional[int] = None
    ionic_step = 0
    free_energy = 0.0
    energy_zero = 0.0
    for idx in range(len(lines) - 1, -1, -1):
        m = _SUMMARY_RE.match(lines[idx])
        if m is not None:
            f_match = re.search(r"F\s*=\s*(\S+).*?E0\s*=\s*(\S+)", lines[idx])
            if f_match is None:
                raise VaspSpinParseError(
                    f"{source}: malformed/truncated final ionic summary: {lines[idx]!r}"
                )
            sum_idx = idx
            ionic_step = int(m.group(1))
            free_energy = _as_float(f_match.group(1))
            energy_zero = _as_float(f_match.group(2))
            break
    if sum_idx is None:
        raise VaspSpinParseError(
            f"{source}: incomplete/truncated constrained OSZICAR on final ionic step"
        )

    # Locate the final converged SCF line immediately preceding sum_idx (no unconsumed MW_int between them).
    scf_idx: Optional[int] = None
    electronic_step: Optional[int] = None
    for idx in range(sum_idx - 1, -1, -1):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        m_scf = _SCF_RE.match(lines[idx])
        if m_scf is not None:
            scf_idx = idx
            electronic_step = int(m_scf.group(1))
            break
        if re.match(r"ion\s+MW_int\s+M_int\s*$", stripped):
            raise VaspSpinParseError(
                f"{source}: MW_int table has no subsequent SCF step before ionic summary"
            )
        if _SUMMARY_RE.match(lines[idx]):
            break
    if scf_idx is None:
        raise VaspSpinParseError(
            f"{source}: missing converged SCF step before final ionic summary"
        )

    # Parse ONLY the single converged E_p + MW_int/M_int table immediately preceding scf_idx.
    mw_hdr_idx: Optional[int] = None
    for idx in range(scf_idx - 1, -1, -1):
        stripped = lines[idx].strip()
        if re.match(r"ion\s+MW_int\s+M_int\s*$", stripped):
            mw_hdr_idx = idx
            break
        if _SCF_RE.match(lines[idx]) or _SUMMARY_RE.match(lines[idx]):
            break
    if mw_hdr_idx is None:
        raise VaspSpinParseError(
            f"{source}: lambda*MW_perp block has no matching ionic-step MW_int table"
        )

    short_ep: Optional[Tuple[float, float]] = None
    for idx in range(mw_hdr_idx - 1, -1, -1):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        if re.match(r"E_p\s*=", stripped):
            m_ep = re.search(r"E_p\s*=\s*(\S+).*?lambda\s*=\s*(\S+)", lines[idx])
            if m_ep is None:
                raise VaspSpinParseError(
                    f"{source}: malformed E_p line: {lines[idx]!r}"
                )
            short_ep = (_as_float(m_ep.group(1)), _as_float(m_ep.group(2)))
            break
        if (
            _SCF_RE.match(lines[idx])
            or _SUMMARY_RE.match(lines[idx])
            or re.match(r"ion\s+", stripped)
        ):
            break
        # Skip VASP 6.4+ diagnostic lines between E_p and `ion MW_int M_int` (e.g. <lVp>, DBL).
    if short_ep is None:
        raise VaspSpinParseError(f"{source}: MW_int block without preceding E_p")

    mw_rows, _ = _parse_atom_table(
        lines,
        mw_hdr_idx + 1,
        6,
        natom,
        source=source,
        table_name="MW_int/M_int",
    )
    if len(mw_rows) != natom:
        raise VaspSpinParseError(
            f"{source}: incomplete MW_int/M_int table ({len(mw_rows)}/{natom})"
        )
    mw_arr_all = np.asarray([mw_rows[i] for i in range(natom)], dtype=np.float64)

    # Parse ONLY the single converged E_p + lambda*MW_perp block after sum_idx,
    # rejecting any trailing unfinished next ionic step or duplicate field block.
    field_arr: Optional[np.ndarray] = None
    field_mask: Optional[np.ndarray] = None
    final_ep: Optional[Tuple[float, float]] = None
    pending_ep: Optional[Tuple[float, float]] = None
    cursor = sum_idx + 1
    while cursor < len(lines):
        line = lines[cursor]
        stripped = line.strip()
        if not stripped:
            cursor += 1
            continue
        if _SCF_RE.match(line) or re.match(r"ion\s+MW_int\s+M_int\s*$", stripped):
            raise VaspSpinParseError(
                f"{source}: incomplete/truncated constrained OSZICAR on final ionic step"
            )
        if re.match(r"E_p\s*=", stripped):
            if field_arr is not None:
                raise VaspSpinParseError(
                    f"{source}: unexpected trailing E_p/block after final lambda*MW_perp"
                )
            m_ep = re.search(r"E_p\s*=\s*(\S+).*?lambda\s*=\s*(\S+)", line)
            if m_ep is None:
                raise VaspSpinParseError(f"{source}: malformed E_p line: {line!r}")
            pending_ep = (_as_float(m_ep.group(1)), _as_float(m_ep.group(2)))
        elif re.match(r"ion\s+lambda\*\S+\s*$", stripped):
            if field_arr is not None:
                raise VaspSpinParseError(
                    f"{source}: unexpected trailing lambda*MW_perp block without ionic-step summary"
                )
            if pending_ep is None:
                raise VaspSpinParseError(
                    f"{source}: lambda*MW_perp block without preceding E_p"
                )
            rows, end = _parse_atom_table(
                lines,
                cursor + 1,
                3,
                natom,
                source=source,
                table_name="lambda*MW_perp",
            )
            field_arr = np.zeros((natom, 3), dtype=np.float64)
            field_mask = np.zeros(natom, dtype=bool)
            for atom_idx, vals in rows.items():
                field_arr[atom_idx] = vals
                field_mask[atom_idx] = True
            final_ep = pending_ep
            pending_ep = None
            cursor = end - 1
        cursor += 1

    if field_arr is None or field_mask is None or final_ep is None:
        raise VaspSpinParseError(
            f"{source}: incomplete/truncated constrained OSZICAR on final ionic step"
        )
    if pending_ep is not None:
        raise VaspSpinParseError(f"{source}: trailing unpaired E_p block")

    return {
        "ionic_step": ionic_step,
        "electronic_step": electronic_step,
        "free_energy": free_energy,
        "energy_zero": energy_zero,
        "mw_int": mw_arr_all[:, :3],
        "m_int": mw_arr_all[:, 3:],
        "short_penalty_energy": short_ep[0],
        "short_lambda": short_ep[1],
        "constraint_field_raw": field_arr,
        "field_mask": field_mask,
        "penalty_energy": final_ep[0],
        "lambda": final_ep[1],
    }


def detect_field_scale(
    raw_field: np.ndarray,
    penalty_energy: float,
    lam: float,
    vasp_major: int,
    *,
    vasp_version_mode: str = "auto",
    mw_int: Optional[np.ndarray] = None,
    m_constr: Optional[np.ndarray] = None,
    i_constrained_m: Optional[int] = None,
    strict: bool = True,
) -> Tuple[float, Optional[float]]:
    """Determine multiplier mapping OSZICAR `lambda*MW_perp` -> C_i = dE_p/dM_{W,i}.

    Uses exact dimensionless penalty consistency ratios with OSZICAR F10.5 printing-
    rounding uncertainty bounds:
      - Modes 1 & 2 (quadratic): R = sum(|P_i|^2) / (lambda * E_p)
          VASP 5 (1*lambda*dMW): R = 1.0 -> scale = 2.0
          VASP 6 (2*lambda*dMW): R = 4.0 -> scale = 1.0
      - Mode 4 (linear, VASP 6.4+): R_4 = sum(|MW_i| * |P_i|^2) / (lambda * E_p)
          VASP 6 (lambda*(MW_hat - u_hat)): R_4 = 2.0 -> scale = 1.0
          (Half-scaled variant):            R_4 = 0.5 -> scale = 2.0
    Cross-checks against OUTCAR VASP major version and (when available) M_CONSTR.
    """
    if vasp_version_mode not in ("auto", "5", "6"):
        raise ValueError(
            f"vasp_version_mode must be 'auto', '5', or '6', got {vasp_version_mode!r}"
        )
    if lam <= 0.0:
        raise VaspSpinParseError(f"LAMBDA must be positive, got {lam}")
    if penalty_energy < -1e-12:
        raise VaspSpinParseError(
            f"Negative OSZICAR penalty energy E_p={penalty_energy}"
        )
    ep_val = max(0.0, float(penalty_energy))

    raw_arr = np.asarray(raw_field, dtype=np.float64).reshape(-1, 3)
    delta_p = 5e-6
    if i_constrained_m == 4:
        if mw_int is None:
            raise VaspSpinParseError(
                "I_CONSTRAINED_M=4 requires mw_int for consistency check"
            )
        weights = np.linalg.norm(
            np.asarray(mw_int, dtype=np.float64).reshape(-1, 3), axis=1
        )
        delta_w = 5e-6 * math.sqrt(3.0)
        weights_min = np.maximum(0.0, weights - delta_w)
        weights_max = weights + delta_w
        r_v5, r_v6 = 0.5, 2.0
    else:
        weights = np.ones(raw_arr.shape[0], dtype=np.float64)
        weights_min = weights
        weights_max = weights
        r_v5, r_v6 = 1.0, 4.0

    # Propagate OSZICAR F10.5 (delta_p = 5e-6), MW_int norm, and E12.5 printing rounding.
    p_sq = np.sum(raw_arr * raw_arr, axis=1)
    p_sq_min = np.sum(np.maximum(0.0, np.abs(raw_arr) - delta_p) ** 2, axis=1)
    p_sq_max = np.sum((np.abs(raw_arr) + delta_p) ** 2, axis=1)
    num = float(np.sum(weights * p_sq))
    num_min = float(np.sum(weights_min * p_sq_min))
    num_max = float(np.sum(weights_max * p_sq_max))
    den = lam * ep_val
    delta_ep = max(5e-11, 1e-4 * ep_val)
    delta_den = lam * delta_ep

    den_min = max(0.0, den - delta_den)
    den_max = den + delta_den

    rel_tol = 0.05
    compat_v5 = (num_max >= (1.0 - rel_tol) * r_v5 * den_min) and (
        num_min <= (1.0 + rel_tol) * r_v5 * den_max
    )
    compat_v6 = (num_max >= (1.0 - rel_tol) * r_v6 * den_min) and (
        num_min <= (1.0 + rel_tol) * r_v6 * den_max
    )

    ratio: Optional[float] = (num / den) if den > 1e-18 else None
    inferred_scale: Optional[float] = None
    if compat_v5 and not compat_v6:
        inferred_scale = 2.0
    elif compat_v6 and not compat_v5:
        inferred_scale = 1.0
    elif not compat_v5 and not compat_v6 and strict:
        ratio_desc = f"{ratio:.6f}" if ratio is not None else "inf (E_p ~ 0)"
        raise VaspSpinParseError(
            f"Inconsistent OSZICAR penalty consistency ratio R = {ratio_desc} "
            f"(expected ~{r_v5} or ~{r_v6}; E_p={ep_val:.4e}, num={num:.4e}, lambda={lam})"
        )

    version_scale = 2.0 if vasp_major == 5 else (1.0 if vasp_major >= 6 else None)
    if vasp_version_mode == "5":
        requested_scale: Optional[float] = 2.0
    elif vasp_version_mode == "6":
        requested_scale = 1.0
    else:
        requested_scale = None

    if requested_scale is not None:
        if inferred_scale is not None and requested_scale != inferred_scale and strict:
            raise VaspSpinParseError(
                f"Requested VASP {vasp_version_mode} convention (scale={requested_scale}) "
                f"contradicts OSZICAR penalty ratio R={ratio} (implies scale={inferred_scale})"
            )
        if version_scale is not None and requested_scale != version_scale and strict:
            raise VaspSpinParseError(
                f"Requested VASP {vasp_version_mode} convention (scale={requested_scale}) "
                f"contradicts OUTCAR VASP {vasp_major}.x header (scale={version_scale})"
            )
        scale = requested_scale
    elif inferred_scale is not None:
        if version_scale is not None and version_scale != inferred_scale and strict:
            raise VaspSpinParseError(
                f"OUTCAR VASP {vasp_major}.x convention (scale={version_scale}) "
                f"contradicts OSZICAR penalty ratio R={ratio} (implies scale={inferred_scale})"
            )
        scale = inferred_scale
    elif version_scale is not None:
        scale = version_scale
    else:
        raise VaspSpinParseError(
            "Cannot determine VASP 5 vs VASP 6 convention: E_p ~ 0 and OUTCAR version unknown"
        )

    # Cross-check exact C_i vector against M_CONSTR for modes 1, 2, and 4 across ALL sites
    # (including inactive/all-zero sites so stray fields on unconstrained atoms always fail).
    if (
        strict
        and mw_int is not None
        and m_constr is not None
        and i_constrained_m in (1, 2, 4)
    ):
        mw_arr = np.asarray(mw_int, dtype=np.float64).reshape(-1, 3)
        target = np.asarray(m_constr, dtype=np.float64).reshape(-1, 3)
        norms = np.linalg.norm(target, axis=1, keepdims=True)
        active = norms[:, 0] > 0.0
        unit = np.divide(target, norms, out=np.zeros_like(target), where=norms > 0.0)
        actual_c = scale * raw_arr
        inactive_tol = scale * delta_p + 1e-12
        if i_constrained_m == 1:
            dots = np.sum(mw_arr * unit, axis=1)
            expected_c = np.where(
                active[:, None], 2.0 * lam * (mw_arr - dots[:, None] * unit), 0.0
            )
            expected_lo = expected_c
            expected_hi = expected_c
        elif i_constrained_m == 2:
            expected_c = np.where(active[:, None], 2.0 * lam * (mw_arr - target), 0.0)
            expected_lo = expected_c
            expected_hi = expected_c
        else:
            # Mode 4: E_p = lambda * sum(|MW| - u_hat . MW), C = lambda * (MW_hat - u_hat)
            mw_norms = np.linalg.norm(mw_arr, axis=1, keepdims=True)
            if np.any(mw_norms[active, 0] <= 1e-12):
                raise VaspSpinParseError(
                    "I_CONSTRAINED_M=4 has zero |MW_int| on an active constrained site"
                )
            mw_hat = np.divide(
                mw_arr, mw_norms, out=np.zeros_like(mw_arr), where=mw_norms > 1e-12
            )
            expected_c = np.where(active[:, None], lam * (mw_hat - unit), 0.0)
            # Exact component-wise interval for MW_hat over the F10.5 rounding box [m - d, m + d]
            d_box = delta_p
            hat_lo = np.zeros_like(mw_arr)
            hat_hi = np.zeros_like(mw_arr)
            for comp in range(3):
                others = [j for j in range(3) if j != comp]
                r_min_sq = np.sum(
                    np.maximum(0.0, np.abs(mw_arr[:, others]) - d_box) ** 2, axis=1
                )
                r_max_sq = np.sum((np.abs(mw_arr[:, others]) + d_box) ** 2, axis=1)
                x_lo = mw_arr[:, comp] - d_box
                x_hi = mw_arr[:, comp] + d_box
                den_lo = np.sqrt(
                    x_lo * x_lo + np.where(x_lo >= 0.0, r_max_sq, r_min_sq)
                )
                den_hi = np.sqrt(
                    x_hi * x_hi + np.where(x_hi >= 0.0, r_min_sq, r_max_sq)
                )
                hat_lo[:, comp] = np.where(
                    den_lo > 0.0, x_lo / np.maximum(den_lo, 1e-30), -1.0
                )
                hat_hi[:, comp] = np.where(
                    den_hi > 0.0, x_hi / np.maximum(den_hi, 1e-30), 1.0
                )
            expected_lo = np.where(active[:, None], lam * (hat_lo - unit), 0.0)
            expected_hi = np.where(active[:, None], lam * (hat_hi - unit), 0.0)
            expected_ep = float(
                lam
                * np.sum(
                    mw_norms[active, 0] - np.sum(mw_arr[active] * unit[active], axis=1)
                )
            )
            ep_tol = max(1e-3, 1e-3 * lam + 5e-3 * abs(expected_ep))
            if abs(ep_val - expected_ep) > ep_tol:
                raise VaspSpinParseError(
                    f"Mode-4 E_p mismatch: OSZICAR E_p={ep_val:.6e} vs "
                    f"expected {expected_ep:.6e} (tol={ep_tol:.2e})"
                )
        site_max_exp = np.max(np.abs(expected_c), axis=1)
        if i_constrained_m == 4:
            # Box bounds [expected_lo, expected_hi] already include exact MW_int F10.5
            # rounding, so per-site active_tol only needs P printing rounding + relative margin.
            active_tol_vec = scale * delta_p + 1e-3 * site_max_exp + 1e-6
        else:
            active_tol_vec = np.maximum(
                2e-3,
                5e-4 * 2.0 * lam + 1e-3 * site_max_exp,
            )
        # Distance from [expected_lo, expected_hi] (0 when inside exact rounding box)
        box_err = np.maximum(
            0.0, np.maximum(expected_lo - actual_c, actual_c - expected_hi)
        )
        site_err = np.max(box_err, axis=1)
        site_tol = np.where(active, active_tol_vec, inactive_tol)
        if np.any(site_err > site_tol):
            worst = int(np.argmax(site_err - site_tol))
            raise VaspSpinParseError(
                f"Mode-{i_constrained_m} field mismatch at site {worst + 1}: "
                f"max|C - C_expected| = {site_err[worst]:.4e} > tol {site_tol[worst]:.4e} "
                f"(scale={scale}, lambda={lam})"
            )
    return scale, ratio


def parse_outcar(
    outcar_path: Path | str,
    *,
    incar_path: Optional[Path | str] = None,
    oszicar_path: Optional[Path | str] = None,
    read_magnetic: bool = True,
    read_born: bool = False,
    force_convergence: bool = False,
    strict_consistency: bool = True,
) -> VaspSpinFrame:
    """Parse geometry, energies, forces, stress, parameters, and optional spin/BEC from OUTCAR."""
    outcar_file = Path(outcar_path).expanduser().resolve()
    text = outcar_file.read_text(encoding="utf-8", errors="replace")

    ver_match = re.search(
        r"^\s*vasp\.(\d+\.\d+\.\d+)\b", text, re.IGNORECASE | re.MULTILINE
    )
    if ver_match is None:
        raise VaspSpinParseError(f"{outcar_file}: missing VASP version header")
    vasp_version = ver_match.group(1)
    vasp_major = int(vasp_version.split(".", 1)[0])

    # Parse species and atom counts (supports VRHFIN and VASP 5.3 POTCAR-only headers)
    counts_matches = re.findall(r"(?m)^\s*ions per type\s*=\s*([^\n]+)", text)
    if not counts_matches:
        raise VaspSpinParseError(f"{outcar_file}: missing 'ions per type' metadata")
    counts = [int(tok) for tok in counts_matches[0].split()]
    species = _parse_species_from_outcar(text, len(counts))
    symbols = [sym for sym, cnt in zip(species, counts) for _ in range(cnt)]
    natom = len(symbols)

    # Parse INCAR parameters: OUTCAR runtime block is authoritative; overlay non-empty
    # INCAR echo (VASP 6) or explicit/sibling INCAR (VASP 5) with contradiction checks.
    params: Dict[str, Any] = {}
    dim_idx = text.find("Dimension of arrays:")
    first_iter_m = _ITER_RE.search(text)
    if dim_idx >= 0:
        runtime_end = first_iter_m.start() if first_iter_m else len(text)
        params.update(_parse_kv_block(text[dim_idx:runtime_end]))

    echo_m = re.search(r"(?ms)^\s*INCAR:\s*\n(.*?)^\s*POTCAR:", text)
    echo_body = echo_m.group(1).strip() if echo_m is not None else ""
    if echo_body:
        params = _merge_incar_params(
            params,
            _parse_kv_block(echo_body),
            strict=strict_consistency,
            source_label=f"{outcar_file} INCAR echo",
        )

    if incar_path is not None:
        explicit_incar = Path(incar_path).expanduser().resolve()
        if not explicit_incar.is_file():
            raise FileNotFoundError(f"INCAR not found: {explicit_incar}")
        params = _merge_incar_params(
            params,
            _parse_kv_block(
                explicit_incar.read_text(encoding="utf-8", errors="replace")
            ),
            strict=strict_consistency,
            source_label=str(explicit_incar),
        )
    elif not echo_body:
        # VASP 5 does not echo INCAR (or echoes an empty section); try paired INCAR.<idx>, then INCAR.
        candidates = []
        paired_incar = outcar_file.with_name(
            outcar_file.name.replace("OUTCAR", "INCAR")
        )
        if paired_incar != outcar_file:
            candidates.append(paired_incar)
        generic_incar = outcar_file.with_name("INCAR")
        if generic_incar not in candidates:
            candidates.append(generic_incar)
        for cand in candidates:
            if cand.is_file():
                params = _merge_incar_params(
                    params,
                    _parse_kv_block(cand.read_text(encoding="utf-8", errors="replace")),
                    strict=strict_consistency,
                    source_label=str(cand),
                )
                break

    iterations = list(_ITER_RE.finditer(text))
    if not iterations:
        raise VaspSpinParseError(f"{outcar_file}: no Iteration records found")
    last_ionic, last_elec = int(iterations[-1].group(1)), int(iterations[-1].group(2))
    final_elec_start = iterations[-1].start()
    final_elec_text = text[final_elec_start:]
    final_ionic_start = next(
        m.start() for m in iterations if int(m.group(1)) == last_ionic
    )
    final_ionic_text = text[final_ionic_start:]

    finished = (
        "General timing and accounting informations for this job:" in final_elec_text
    )
    ediff_reached = bool(
        re.search(
            r"aborting loop because EDIFF is reached", final_elec_text, re.IGNORECASE
        )
    )
    soft_stop = bool(
        re.search(r"soft\s+stop\s+encountered|STOPCAR", final_elec_text, re.IGNORECASE)
    )
    nsw = int(params.get("NSW", 0))
    ibrion = int(params.get("IBRION", -1 if nsw == 0 else 2))
    elec_ok = bool(ediff_reached and not soft_stop)
    if nsw > 0 and ibrion in (1, 2, 3):
        ionic_ok = bool(
            re.search(
                r"reached required accuracy\s*-\s*stopping structural energy minimisation",
                final_elec_text,
                re.IGNORECASE,
            )
        )
    else:
        ionic_ok = True
    converged = bool(finished and elec_ok and ionic_ok)
    if force_convergence:
        converged = True

    energies = list(
        re.finditer(
            r"free\s+energy\s+TOTEN\s*=\s*(\S+)\s+eV", final_elec_text, re.IGNORECASE
        )
    )
    zeros = list(
        re.finditer(
            r"energy\s*\(\s*sigma\s*->\s*0\s*\)\s*=\s*(\S+)",
            final_elec_text,
            re.IGNORECASE,
        )
    )
    if not energies or not zeros:
        raise VaspSpinParseError(
            f"{outcar_file}: missing final TOTEN / energy(sigma->0)"
        )
    free_energy = _as_float(energies[-1].group(1))
    energy_zero = _as_float(zeros[-1].group(1))

    # Positions & forces (supports merged negative Fortran columns)
    force_matches = list(
        re.finditer(r"POSITION[^\n]*TOTAL-FORCE\s*\(eV/Angst\)", final_elec_text)
    )
    if not force_matches:
        raise VaspSpinParseError(
            f"{outcar_file}: missing final POSITION/TOTAL-FORCE block"
        )
    pos_force_rows: List[List[float]] = []
    for ln in final_elec_text[force_matches[-1].end() :].splitlines():
        stripped = ln.strip()
        if not stripped or set(stripped) <= {"-"}:
            if pos_force_rows:
                break
            continue
        if "*" in stripped:
            raise VaspSpinParseError(
                f"{outcar_file}: Fortran overflow in TOTAL-FORCE: {ln!r}"
            )
        nums = re.findall(_NUM_RE, stripped)
        if len(nums) != 6:
            break
        pos_force_rows.append([_as_float(tok) for tok in nums])
    if len(pos_force_rows) != natom:
        raise VaspSpinParseError(
            f"{outcar_file}: force table has {len(pos_force_rows)} rows, expected {natom}"
        )
    pf_arr = np.asarray(pos_force_rows, dtype=np.float64)
    positions = pf_arr[:, :3]
    forces = pf_arr[:, 3:6]

    # Direct lattice vectors strictly BEFORE the evaluated final POSITION/TOTAL-FORCE block
    # (prevents grabbing the step k+1 updated cell in ISIF=3 variable-cell relaxations).
    pre_force_text = text[: final_elec_start + force_matches[-1].start()]
    lat_matches = list(
        re.finditer(r"direct lattice vectors[^\n]*\n((?:[^\n]+\n){3})", pre_force_text)
    )
    if not lat_matches:
        raise VaspSpinParseError(f"{outcar_file}: missing direct lattice vectors")
    cell_rows: List[List[float]] = []
    for ln in lat_matches[-1].group(1).splitlines():
        if "*" in ln:
            raise VaspSpinParseError(
                f"{outcar_file}: Fortran overflow in direct lattice vectors: {ln!r}"
            )
        nums = re.findall(_NUM_RE, ln)
        if len(nums) < 3:
            raise VaspSpinParseError(
                f"{outcar_file}: malformed lattice vector line: {ln!r}"
            )
        cell_rows.append([_as_float(tok) for tok in nums[:3]])
    cell = np.asarray(cell_rows, dtype=np.float64)
    if cell.shape != (3, 3) or abs(float(np.linalg.det(cell))) <= 0.0:
        raise VaspSpinParseError(f"{outcar_file}: singular lattice cell")

    # Optional stress (ASE tension-positive Voigt [xx, yy, zz, yz, xz, xy], eV/A^3)
    stress: Optional[np.ndarray] = None
    stress_matches = list(
        re.finditer(r"^\s*in kB\s+([^\n]+)", final_elec_text, re.MULTILINE)
    )
    if stress_matches:
        sraw = stress_matches[-1].group(1)
        if "*" in sraw:
            raise VaspSpinParseError(
                f"{outcar_file}: Fortran overflow in stress line: {sraw!r}"
            )
        stoks = re.findall(_NUM_RE, sraw)
        if len(stoks) < 6:
            if strict_consistency:
                raise VaspSpinParseError(
                    f"{outcar_file}: malformed 'in kB' stress line ({len(stoks)} values): {sraw!r}"
                )
        else:
            raw_kb = np.asarray([_as_float(tok) for tok in stoks[:6]], dtype=np.float64)
            stress = -raw_kb[[0, 1, 2, 4, 5, 3]] * KB_TO_EV_PER_A3

    # Verify printed SAXIS matrix if present
    rot = saxis_rotation_matrix(params.get("SAXIS", [0.0, 0.0, 1.0]))
    printed_rot = _parse_printed_saxis_matrix(text)
    if printed_rot is not None and np.max(np.abs(rot - printed_rot)) > 1e-4:
        raise VaspSpinParseError(
            f"{outcar_file}: SAXIS matrix disagrees with OUTCAR printout"
        )

    projected_mag: Optional[np.ndarray] = None
    oszicar_data: Optional[Dict[str, Any]] = None
    osz_resolved: Optional[str] = None
    born: Optional[np.ndarray] = None
    piezo: Optional[np.ndarray] = None
    # Skip expensive post-SCF / OSZICAR / BEC parsing for unconverged runs.
    if converged:
        if read_magnetic:
            req_nc = bool(params.get("LNONCOLLINEAR") or params.get("LSORBIT"))
            projected_mag = _parse_outcar_magnetization(
                final_elec_text.splitlines(), natom, require_noncollinear=req_nc
            )
            if projected_mag is not None:
                projected_mag = projected_mag @ rot.T
        # Always inspect explicit or auto-paired OSZICAR so constrained penalty energy E_p
        # is validated and subtracted identically regardless of whether -m is passed.
        if oszicar_path is not None:
            osz_file = Path(oszicar_path).expanduser().resolve()
            if not osz_file.is_file():
                raise FileNotFoundError(f"OSZICAR not found: {osz_file}")
            osz_resolved = str(osz_file)
            oszicar_data = parse_oszicar_constrained(osz_file, natom)
        else:
            auto_osz = outcar_file.with_name(
                outcar_file.name.replace("OUTCAR", "OSZICAR")
            )
            if auto_osz != outcar_file and auto_osz.is_file():
                osz_resolved = str(auto_osz)
                oszicar_data = parse_oszicar_constrained(auto_osz, natom)

        if read_born:
            born_scope = final_ionic_text if last_ionic > 1 else text
            born, piezo = _parse_outcar_born_and_piezo(born_scope, natom)

    return VaspSpinFrame(
        outcar_path=str(outcar_file),
        oszicar_path=osz_resolved,
        vasp_version=vasp_version,
        vasp_major=vasp_major,
        symbols=symbols,
        cell=cell,
        positions=positions,
        forces=forces,
        stress=stress,
        free_energy=free_energy,
        energy_zero=energy_zero,
        converged=converged,
        last_ionic_step=last_ionic,
        last_electronic_step=last_elec,
        parameters=params,
        projected_mag=projected_mag,
        born_effective_charges=born,
        piezoelectric=piezo,
        oszicar=oszicar_data,
    )


def build_extxyz_frame(
    outcar_path: Path | str,
    oszicar_path: Optional[Path | str] = None,
    *,
    incar_path: Optional[Path | str] = None,
    read_magnetic: bool = True,
    read_born: bool = False,
    nomagz: Sequence[int] = (),
    min_mag_norm: float = 0.5,
    normalize_spin: bool = True,
    energy_source: str = "free",
    spin_source: str = "mw_int",
    vasp_version_mode: str = "auto",
    force_convergence: bool = False,
    strict_consistency: bool = True,
    keep_raw: bool = False,
) -> Optional["Atoms"]:
    """Convert one VASP OUTCAR(+OSZICAR) pair into a SpinGNN2-ready ASE Atoms frame.

    Returns None when the calculation is not converged (unless force_convergence=True).
    """
    if _ASE_IMPORT_ERROR is not None:  # pragma: no cover
        raise ImportError(
            "ASE is required to build extxyz frames"
        ) from _ASE_IMPORT_ERROR
    if energy_source not in ("free", "zero"):
        raise ValueError("energy_source must be 'free' or 'zero'")
    if spin_source not in ("m_int", "mw_int", "outcar"):
        raise ValueError("spin_source must be 'm_int', 'mw_int', or 'outcar'")

    src_path = Path(outcar_path).expanduser().resolve()
    if src_path.suffix.lower() == ".xml":
        if read_magnetic or read_born:
            raise VaspSpinParseError(
                f"{src_path}: XML input is only supported for structure-only conversion "
                "(use OUTCAR + OSZICAR for -m / -b)"
            )
        atoms_xml = ase_read(str(src_path))
        if isinstance(atoms_xml, list):
            atoms_xml = atoms_xml[-1]
        if (
            energy_source == "free"
            and atoms_xml.calc is not None
            and "free_energy" in atoms_xml.calc.results
        ):
            res = dict(atoms_xml.calc.results)
            res["energy"] = float(res["free_energy"])
            atoms_xml.calc = SinglePointCalculator(atoms_xml, **res)
        atoms_xml.info["energy_policy"] = (
            "F_raw" if energy_source == "free" else "E0_raw"
        )
        return atoms_xml

    parsed = parse_outcar(
        outcar_path,
        incar_path=incar_path,
        oszicar_path=oszicar_path,
        read_magnetic=read_magnetic,
        read_born=read_born,
        force_convergence=force_convergence,
        strict_consistency=strict_consistency,
    )
    if not parsed.converged:
        return None

    atoms = Atoms(
        parsed.symbols, positions=parsed.positions, cell=parsed.cell, pbc=True
    )
    atoms.info["vasp_version"] = parsed.vasp_version
    raw_energy = parsed.free_energy if energy_source == "free" else parsed.energy_zero
    penalty_energy = 0.0
    natom = len(parsed.symbols)
    z_set = {int(z) for z in nomagz if int(z) > 0}

    if read_born:
        if parsed.born_effective_charges is None:
            raise VaspSpinParseError(
                f"{parsed.outcar_path}: missing valid BORN EFFECTIVE CHARGES"
            )
        atoms.set_array("born_effective_charges", parsed.born_effective_charges)
        if parsed.piezoelectric is not None:
            atoms.info["piezoelectric"] = parsed.piezoelectric

    i_constr = parsed.parameters.get("I_CONSTRAINED_M")
    osz = parsed.oszicar
    if osz is None and i_constr in (1, 2, 4):
        raise VaspSpinParseError(
            f"{parsed.outcar_path}: OUTCAR declares I_CONSTRAINED_M={i_constr} "
            "but paired OSZICAR constrained blocks are missing (refusing to emit "
            "E_p-contaminated raw energy)"
        )
    if strict_consistency and osz is not None and i_constr == 0:
        raise VaspSpinParseError(
            f"{parsed.outcar_path}: OUTCAR declares I_CONSTRAINED_M=0 "
            "but paired OSZICAR contains constrained penalty blocks"
        )

    field_scale: Optional[float] = None
    ratio: Optional[float] = None
    m_constr: Any = None
    if osz is not None:
        # Always verify OUTCAR / OSZICAR pairing, penalty physics, and subtract E_p consistently even if -m is off.
        if strict_consistency:
            if (
                osz.get("ionic_step") != parsed.last_ionic_step
                or osz.get("electronic_step") is None
                or osz.get("electronic_step") != parsed.last_electronic_step
            ):
                raise VaspSpinParseError(
                    f"{parsed.outcar_path}: OUTCAR step "
                    f"({parsed.last_ionic_step},{parsed.last_electronic_step}) != "
                    f"OSZICAR step ({osz.get('ionic_step')},{osz.get('electronic_step')})"
                )
            f_tol = max(1e-4, 1e-6 * max(abs(parsed.free_energy), 1.0))
            if (
                "free_energy" in osz
                and abs(osz["free_energy"] - parsed.free_energy) > f_tol
            ):
                raise VaspSpinParseError(
                    f"{parsed.outcar_path}: OUTCAR/OSZICAR free energy mismatch"
                )
            e0_tol = max(1e-4, 1e-6 * max(abs(parsed.energy_zero), 1.0))
            if (
                "energy_zero" in osz
                and abs(osz["energy_zero"] - parsed.energy_zero) > e0_tol
            ):
                raise VaspSpinParseError(
                    f"{parsed.outcar_path}: OUTCAR/OSZICAR E0 energy mismatch"
                )
            if "short_lambda" in osz and abs(osz["short_lambda"] - osz["lambda"]) > max(
                1e-6, 1e-4 * abs(osz["lambda"])
            ):
                raise VaspSpinParseError(
                    f"{parsed.outcar_path}: OSZICAR MW_int vs lambda*MW_perp LAMBDA mismatch"
                )
            if "short_penalty_energy" in osz and abs(
                osz["short_penalty_energy"] - osz["penalty_energy"]
            ) > max(1e-5, 1e-3 * abs(osz["penalty_energy"])):
                raise VaspSpinParseError(
                    f"{parsed.outcar_path}: OSZICAR MW_int vs lambda*MW_perp E_p mismatch"
                )
            outcar_lam = parsed.parameters.get("LAMBDA")
            if outcar_lam is not None and abs(float(outcar_lam) - osz["lambda"]) > max(
                1e-3, 1e-3 * abs(float(outcar_lam))
            ):
                raise VaspSpinParseError(
                    f"{parsed.outcar_path}: OUTCAR LAMBDA={outcar_lam} != OSZICAR lambda={osz['lambda']}"
                )
        m_constr = parsed.parameters.get("M_CONSTR")
        if m_constr is not None and len(m_constr) != 3 * natom:
            if strict_consistency:
                raise VaspSpinParseError(
                    f"{parsed.outcar_path}: M_CONSTR has {len(m_constr)} values, "
                    f"expected {3 * natom}"
                )
            m_constr = None
        penalty_energy = float(osz["penalty_energy"])
        field_scale, ratio = detect_field_scale(
            osz["constraint_field_raw"],
            penalty_energy,
            float(osz["lambda"]),
            parsed.vasp_major,
            vasp_version_mode=vasp_version_mode,
            mw_int=osz["mw_int"],
            m_constr=m_constr,
            i_constrained_m=i_constr,
            strict=strict_consistency,
        )
        atoms.info["penalty_energy"] = penalty_energy
        atoms.info[
            "energy_policy"
        ] = f"{'F' if energy_source == 'free' else 'E0'}_minus_Ep"

    if read_magnetic:
        rot = saxis_rotation_matrix(parsed.parameters.get("SAXIS", [0.0, 0.0, 1.0]))
        if osz is not None:
            assert field_scale is not None
            mw_saxis = osz["mw_int"]
            m_int_saxis = osz["m_int"]
            raw_field_saxis = osz["constraint_field_raw"]
            lam = float(osz["lambda"])

            # Rotate all spin quantities from SAXIS frame to Cartesian frame
            mw = mw_saxis @ rot.T
            m_int = m_int_saxis @ rot.T
            c_field = (field_scale * raw_field_saxis) @ rot.T

            if spin_source == "m_int":
                spin_vec = m_int.copy()
            elif spin_source == "mw_int":
                spin_vec = mw.copy()
            else:
                if parsed.projected_mag is None:
                    raise VaspSpinParseError(
                        f"{parsed.outcar_path}: spin_source='outcar' requires LORBIT magnetization"
                    )
                spin_vec = parsed.projected_mag.copy()

            spin_len_raw = np.linalg.norm(spin_vec, axis=1)

            # Separate constrained sites from general magnetic sites so unconstrained
            # magnetic atoms with |M| >= min_mag_norm are never silently wiped to zero.
            if m_constr is not None:
                mc = np.asarray(m_constr, dtype=np.float64).reshape(natom, 3)
                constrained_mask = np.linalg.norm(mc, axis=1) > 0.0
            elif np.any(osz["field_mask"]):
                constrained_mask = osz["field_mask"].copy()
            else:
                constrained_mask = spin_len_raw >= min_mag_norm

            if z_set:
                mag_mask = np.array(
                    [z not in z_set for z in atoms.get_atomic_numbers()], dtype=bool
                )
            else:
                mag_mask = constrained_mask | (spin_len_raw >= min_mag_norm)

            has_valid_dir = mag_mask & (spin_len_raw > 1e-12)
            unit_dir = np.divide(
                spin_vec,
                spin_len_raw[:, None],
                out=np.zeros_like(spin_vec),
                where=has_valid_dir[:, None],
            )
            spin_len = np.where(mag_mask, spin_len_raw, 0.0)
            spin_out = (
                unit_dir
                if normalize_spin
                else np.where(mag_mask[:, None], spin_vec, 0.0)
            )

            # Strict tangential projection onto actual spin direction, scaled by |MW_int| [eV]
            mw_len = np.linalg.norm(mw, axis=1)
            radial = np.sum(c_field * unit_dir, axis=1)[:, None] * unit_dir
            tangent_field = c_field - radial
            degenerate = constrained_mask & mag_mask & (~has_valid_dir)
            if strict_consistency and np.any(
                degenerate & (np.linalg.norm(c_field, axis=1) > 1e-6)
            ):
                raise VaspSpinParseError(
                    f"{parsed.outcar_path}: nonzero constraint field on site with zero spin norm "
                    "(undefined tangent plane)"
                )
            spin_forces_vert = np.where(
                (constrained_mask & has_valid_dir)[:, None],
                mw_len[:, None] * tangent_field,
                0.0,
            )

            atoms.set_initial_magnetic_moments(spin_out)
            atoms.set_array("spin_length", spin_len)
            atoms.set_array("spin_forces_vert", spin_forces_vert)
            atoms.set_array("magnetic_mask", mag_mask)

            atoms.info["vasp_version"] = parsed.vasp_version
            atoms.info["vasp_field_scale"] = float(field_scale)
            atoms.info["spin_source"] = spin_source
            atoms.info["penalty_energy"] = penalty_energy
            atoms.info["lambda"] = lam
            atoms.info[
                "energy_policy"
            ] = f"{'F' if energy_source == 'free' else 'E0'}_minus_Ep"
            if ratio is not None:
                atoms.info["constraint_consistency_ratio"] = float(ratio)

            if keep_raw:
                atoms.set_array("mw_raw", mw)
                atoms.set_array("m_int_raw", m_int)
                atoms.set_array("constraint_field_raw", c_field)
                atoms.set_array("constrained_mask", constrained_mask)
                atoms.set_array(
                    "constraint_moment_length", np.where(mag_mask, mw_len, 0.0)
                )
                atoms.set_array(
                    "spin_force_normal_diagnostic",
                    np.where(
                        (constrained_mask & has_valid_dir)[:, None],
                        mw_len[:, None] * radial,
                        0.0,
                    ),
                )
                if parsed.projected_mag is not None:
                    atoms.set_array("projected_moment_raw", parsed.projected_mag)
                atoms.info["free_energy_raw"] = parsed.free_energy
                atoms.info["energy_zero_raw"] = parsed.energy_zero
        else:
            # Unconstrained magnetic OUTCAR (I_CONSTRAINED_M == 0 / omitted, no OSZICAR constraint)
            if parsed.projected_mag is None:
                raise VaspSpinParseError(
                    f"{parsed.outcar_path}: no OSZICAR constrained block and no OUTCAR magnetization"
                )
            spin_vec = parsed.projected_mag.copy()
            spin_len_raw = np.linalg.norm(spin_vec, axis=1)
            if z_set:
                mag_mask = np.array(
                    [z not in z_set for z in atoms.get_atomic_numbers()], dtype=bool
                )
            else:
                mag_mask = spin_len_raw >= min_mag_norm
            has_valid_dir = mag_mask & (spin_len_raw > 1e-12)
            unit_dir = np.divide(
                spin_vec,
                spin_len_raw[:, None],
                out=np.zeros_like(spin_vec),
                where=has_valid_dir[:, None],
            )
            spin_len = np.where(mag_mask, spin_len_raw, 0.0)
            spin_out = (
                unit_dir
                if normalize_spin
                else np.where(mag_mask[:, None], spin_vec, 0.0)
            )
            atoms.set_initial_magnetic_moments(spin_out)
            atoms.set_array("spin_length", spin_len)
            atoms.set_array("magnetic_mask", mag_mask)
            atoms.info["vasp_version"] = parsed.vasp_version
            atoms.info["spin_source"] = "outcar"
            atoms.info["energy_policy"] = (
                "F_raw" if energy_source == "free" else "E0_raw"
            )

    calc_kwargs: Dict[str, Any] = {
        "energy": float(raw_energy - penalty_energy),
        "forces": parsed.forces,
    }
    if parsed.stress is not None:
        calc_kwargs["stress"] = parsed.stress
    atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
    return atoms


def _worker_convert_one(
    task: Tuple[int, str, Dict[str, Any]]
) -> Tuple[int, str, Optional["Atoms"], Optional[str]]:
    idx, outcar_str, kwargs = task
    try:
        frame = build_extxyz_frame(outcar_str, **kwargs)
        return idx, outcar_str, frame, None
    except Exception as exc:  # pragma: no cover - surfaced to caller
        return idx, outcar_str, None, f"{type(exc).__name__}: {exc}"


def discover_outcars(root_path: Path | str, fname: str = "OUTCAR") -> List[Path]:
    """Recursively discover OUTCAR files in deterministic sorted order."""
    root = Path(root_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    if root.is_file():
        return [root]
    found: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if fname in name:
                found.append(Path(dirpath) / name)
    return found


def _paired_vasp_inputs(
    outcar_path: Path,
    oszicar_path: Optional[Path | str] = None,
    incar_path: Optional[Path | str] = None,
) -> List[Path]:
    """Return resolved input paths (OUTCAR + explicit/auto-paired OSZICAR & INCAR)."""
    outcar = outcar_path.expanduser().resolve()
    paths = [outcar]
    if oszicar_path is not None:
        paths.append(Path(oszicar_path).expanduser().resolve())
    else:
        auto_osz = outcar.with_name(outcar.name.replace("OUTCAR", "OSZICAR"))
        if auto_osz != outcar and auto_osz.exists():
            paths.append(auto_osz)
    if incar_path is not None:
        paths.append(Path(incar_path).expanduser().resolve())
    else:
        for cand in (
            outcar.with_name(outcar.name.replace("OUTCAR", "INCAR")),
            outcar.with_name("INCAR"),
        ):
            if cand != outcar and cand.exists():
                paths.append(cand)
    return paths


def _guard_output_not_input(out_file: Path, protected_inputs: Sequence[Path]) -> None:
    resolved_out = out_file.expanduser().resolve()
    for src in protected_inputs:
        resolved_src = src.expanduser().resolve()
        if resolved_out == resolved_src or (
            resolved_out.exists()
            and resolved_src.exists()
            and os.path.samefile(resolved_out, resolved_src)
        ):
            raise ValueError(
                f"Refusing to overwrite or remove input VASP file: {resolved_out}"
            )


def convert_pair_to_extxyz(
    outcar_path: Path | str,
    oszicar_path: Optional[Path | str],
    output_path: Path | str,
    **kwargs: Any,
) -> "Atoms":
    """Convert a single VASP OUTCAR(+OSZICAR) pair to extxyz."""
    out_file = Path(output_path).expanduser().resolve()
    _guard_output_not_input(
        out_file,
        _paired_vasp_inputs(
            Path(outcar_path),
            oszicar_path=oszicar_path,
            incar_path=kwargs.get("incar_path"),
        ),
    )

    frame = build_extxyz_frame(outcar_path, oszicar_path, **kwargs)
    if frame is None:
        raise VaspSpinParseError(f"{outcar_path}: calculation did not converge")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(out_file), frame, format="extxyz")
    return frame


def convert_directory_to_extxyz(
    root_path: Path | str,
    output_path: Path | str,
    *,
    fname: str = "OUTCAR",
    cpu: int = 1,
    fail_fast: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convert all VASP OUTCARs under root_path in parallel across `cpu` worker processes."""
    if cpu < 1:
        raise ValueError(f"cpu must be >= 1, got {cpu}")
    out_file = Path(output_path).expanduser().resolve()
    outcars = discover_outcars(root_path, fname=fname)
    protected: List[Path] = []
    if kwargs.get("oszicar_path") is not None:
        protected.append(Path(kwargs["oszicar_path"]).expanduser().resolve())
    if kwargs.get("incar_path") is not None:
        protected.append(Path(kwargs["incar_path"]).expanduser().resolve())
    for p in outcars:
        protected.extend(
            _paired_vasp_inputs(
                p,
                oszicar_path=kwargs.get("oszicar_path"),
                incar_path=kwargs.get("incar_path"),
            )
        )
    _guard_output_not_input(out_file, protected)

    tasks = [(idx, str(p), kwargs) for idx, p in enumerate(outcars)]

    ordered_frames: Dict[int, "Atoms"] = {}
    unconverged: List[str] = []
    errors: List[Tuple[str, str]] = []

    if cpu == 1 or len(tasks) <= 1:
        for task in tasks:
            idx, path_str, frame, err = _worker_convert_one(task)
            if err is not None:
                if fail_fast:
                    raise VaspSpinParseError(f"{path_str}: {err}")
                errors.append((path_str, err))
            elif frame is None:
                unconverged.append(path_str)
            else:
                ordered_frames[idx] = frame
    else:
        n_workers = min(cpu, len(tasks))
        # Always use "spawn" so calling convert_directory_to_extxyz() from a
        # process with PyTorch/CUDA/OpenMP/threads initialized never deadlocks.
        src_root = str(Path(__file__).resolve().parents[2])
        prev_pythonpath = os.environ.get("PYTHONPATH")
        pp_parts = [p for p in (prev_pythonpath or "").split(os.pathsep) if p]
        added_src = src_root not in pp_parts
        if added_src:
            os.environ["PYTHONPATH"] = os.pathsep.join([src_root, *pp_parts])
        try:
            mp_ctx = multiprocessing.get_context("spawn")
            pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx)
            aborted = False
            try:
                if fail_fast:
                    futures = [pool.submit(_worker_convert_one, t) for t in tasks]
                    for fut in as_completed(futures):
                        idx, path_str, frame, err = fut.result()
                        if err is not None:
                            aborted = True
                            pool.shutdown(wait=False, cancel_futures=True)
                            raise VaspSpinParseError(f"{path_str}: {err}")
                        if frame is None:
                            unconverged.append(path_str)
                        else:
                            ordered_frames[idx] = frame
                else:
                    for idx, path_str, frame, err in pool.map(
                        _worker_convert_one,
                        tasks,
                        chunksize=max(1, len(tasks) // (n_workers * 4)),
                    ):
                        if err is not None:
                            errors.append((path_str, err))
                        elif frame is None:
                            unconverged.append(path_str)
                        else:
                            ordered_frames[idx] = frame
            finally:
                if not aborted:
                    pool.shutdown(wait=True)
        finally:
            if added_src:
                if prev_pythonpath is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = prev_pythonpath

    frames = [ordered_frames[i] for i in sorted(ordered_frames)]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        ase_write(str(out_file), frames, format="extxyz")
    elif out_file.exists():
        # Never leave a stale extxyz from an earlier run when 0 frames were produced.
        out_file.unlink()
    return {
        "discovered": len(outcars),
        "written": len(frames),
        "unconverged": len(unconverged),
        "errors": errors,
        "output": str(out_file),
    }


def _parse_nomagz(raw: str) -> Tuple[int, ...]:
    vals = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    return tuple(v for v in vals if v > 0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spingnn-data",
        description=(
            "Convert VASP 5 & VASP 6 OUTCAR(+OSZICAR) outputs into SpinGNN2 extxyz "
            "datasets with multi-core CPU parallelism and automatic VASP 5/6 spin-force scaling."
        ),
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "-p",
        "--path",
        type=Path,
        default=None,
        help="Root directory to recursively search for OUTCAR files (batch mode, default: ./)",
    )
    src.add_argument(
        "--outcar",
        type=Path,
        default=None,
        help="Single OUTCAR file path (single-frame mode)",
    )
    parser.add_argument(
        "--oszicar",
        type=Path,
        default=None,
        help="Optional OSZICAR path for single-frame mode (requires --outcar)",
    )
    parser.add_argument(
        "--incar",
        type=Path,
        default=None,
        help="Optional INCAR path (useful for VASP 5 OUTCAR without INCAR echo)",
    )
    parser.add_argument(
        "-o",
        "--out-path",
        "--output",
        dest="out_path",
        type=Path,
        default=Path("data.extxyz"),
        help="Output extxyz file path (default: ./data.extxyz)",
    )
    parser.add_argument(
        "-m",
        "--magnetic",
        "--lmagnet",
        dest="magnetic",
        action="store_true",
        help="Extract magnetic moments and OSZICAR spin_forces_vert",
    )
    parser.add_argument(
        "-b",
        "--born",
        "--lborn",
        dest="born",
        action="store_true",
        help="Extract Born effective charges and piezoelectric tensor",
    )
    parser.add_argument(
        "-z",
        "--nomagz",
        type=str,
        default="0",
        help="Comma-separated nonmagnetic atomic numbers (e.g. 8,83); 0 = infer from M_CONSTR / threshold",
    )
    parser.add_argument(
        "-n",
        "--min-mag-norm",
        type=float,
        default=0.5,
        help="Fallback magnetization norm threshold (mu_B) when M_CONSTR and --nomagz are absent",
    )
    parser.add_argument(
        "--vasp-version",
        choices=("auto", "5", "6"),
        default=None,
        help="VASP convention for OSZICAR lambda*MW_perp: auto (recommended), 5 (x2), or 6 (x1)",
    )
    parser.add_argument(
        "--vasp6",
        action="store_true",
        help="Alias for --vasp-version 6",
    )
    parser.add_argument(
        "--energy-source",
        choices=("free", "zero"),
        default="free",
        help="Energy baseline before subtracting E_p: free (TOTEN - E_p, default) or zero (E0 - E_p)",
    )
    parser.add_argument(
        "--spin-source",
        choices=("mw_int", "m_int", "outcar"),
        default="mw_int",
        help="Source of magnetic moment direction (default: mw_int, exact penalty conjugate; also m_int or outcar)",
    )
    parser.add_argument(
        "--no-normalize-spin",
        action="store_true",
        help="Store unnormalized moment vectors in initial_magmoms instead of unit directions",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw mw_raw, m_int_raw, constraint_field_raw, and energy diagnostics",
    )
    parser.add_argument(
        "-c",
        "--cpu",
        type=int,
        default=1,
        help="Number of CPU worker processes for batch conversion (default: 1)",
    )
    parser.add_argument(
        "--fname",
        type=str,
        default="OUTCAR",
        help="Filename substring to match in batch mode (default: OUTCAR)",
    )
    parser.add_argument(
        "-f",
        "--force-convergence",
        "--lforce_convergence",
        dest="force_convergence",
        action="store_true",
        help="Include unconverged VASP runs",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort batch conversion on the first malformed or inconsistent frame",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.oszicar is not None and args.outcar is None:
        parser.error("--oszicar requires --outcar (single-frame mode)")
    if args.vasp6 and args.vasp_version == "5":
        parser.error("--vasp6 conflicts with --vasp-version 5")
    vasp_mode = "6" if args.vasp6 else (args.vasp_version or "auto")
    read_magnetic = bool(args.magnetic or args.oszicar is not None)
    common_kwargs: Dict[str, Any] = {
        "incar_path": args.incar,
        "read_magnetic": read_magnetic,
        "read_born": bool(args.born),
        "nomagz": _parse_nomagz(args.nomagz),
        "min_mag_norm": float(args.min_mag_norm),
        "normalize_spin": not bool(args.no_normalize_spin),
        "energy_source": args.energy_source,
        "spin_source": args.spin_source,
        "vasp_version_mode": vasp_mode,
        "force_convergence": bool(args.force_convergence),
        "keep_raw": bool(args.keep_raw),
    }

    if args.outcar is not None:
        frame = convert_pair_to_extxyz(
            args.outcar,
            args.oszicar,
            args.out_path,
            **common_kwargs,
        )
        print(
            f"Converted {args.outcar} ({len(frame)} atoms, "
            f"VASP {frame.info.get('vasp_version', '?')}) -> {args.out_path}"
        )
        return 0

    root = args.path if args.path is not None else Path(".")
    summary = convert_directory_to_extxyz(
        root,
        args.out_path,
        fname=args.fname,
        cpu=args.cpu,
        fail_fast=bool(args.fail_fast),
        **common_kwargs,
    )
    print(
        f"Discovered {summary['discovered']} OUTCAR file(s) in {root} with {args.cpu} CPU(s); "
        f"wrote {summary['written']} frame(s) to {summary['output']} "
        f"(unconverged={summary['unconverged']}, errors={len(summary['errors'])})."
    )
    for bad_path, msg in summary["errors"]:
        print(f"  [ERROR] {bad_path}: {msg}")
    if summary["written"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
