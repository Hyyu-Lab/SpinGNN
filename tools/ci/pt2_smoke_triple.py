#!/usr/bin/env python3
"""Validate the real NoSOC smoke model across PT2, TorchScript, and LAMMPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from ase.io import read

REPO = Path(__file__).resolve().parents[2]
runpy.run_path(str(REPO / "_spingnn_source.py"))["activate_source_tree"]()

from nequip.data import AtomicData  # noqa: E402
from nequip.data.transforms import TypeMapper  # noqa: E402
from nequip.scripts import deploy  # noqa: E402
from nequip.utils.pt2_deploy import build_pt2  # noqa: E402


DEFAULT_RESULTS_DIR = REPO / "validation/results/pt2-smoke"
LAMMPS_INPUT = REPO / "tests_lammps/fixtures/in/nosoc_run0.in"
PREFIX = os.environ.get("SPINGNN_PT2_VALIDATION_PREFIX", "pt2-review")
RESULTS_DIR = Path(
    os.environ.get("SPINGNN_PT2_RESULTS_DIR", str(DEFAULT_RESULTS_DIR))
).resolve()
PACKAGE_DIR = Path(
    os.environ.get("SPINGNN_PT2_PACKAGE_DIR", str(RESULTS_DIR))
).resolve()
PACKAGE = PACKAGE_DIR / f"{PREFIX}-nosoc.pt2"
SUMMARY = RESULTS_DIR / f"{PREFIX}-summary.json"
PYTHON_RESULT = RESULTS_DIR / f"{PREFIX}-python.json"
LAMMPS_LOG = RESULTS_DIR / f"{PREFIX}-lammps-ts.log"
LAMMPS_DUMP = RESULTS_DIR / f"{PREFIX}-lammps-ts.dump"

ENERGY_ATOL_EV = 2.0e-4
FORCE_ATOL_EV_A = 1.0e-3
SPIN_FORCE_ATOL_EV = 1.0e-3
VALID_CUBLAS_CONFIGS = {":4096:8", ":16:8"}
SOURCE_PATHS = (
    "src/core/nequip/utils/pt2_deploy.py",
    "src/core/nequip/utils/fx_compat.py",
    "src/core/nequip/scripts/deploy.py",
    "lammps/src/deploy_metadata.h",
    "lammps/src/SPIN/pair_allegro_spin.cpp",
    "lammps/src/SPIN/pair_allegro_spin.h",
    "lammps/src/SPIN/pair_nequip_spin.cpp",
    "lammps/src/SPIN/pair_nequip_spin.h",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "--no-optional-locks",
            "-c",
            f"safe.directory={REPO}",
            "-C",
            str(REPO),
            *args,
        ],
        text=True,
    ).strip()


def clone_tensors(data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in data.items()}


def require_environment(inputs: argparse.Namespace) -> None:
    if os.environ.get("SPINGNN_DETERMINISTIC") != "1":
        raise RuntimeError("SPINGNN_DETERMINISTIC=1 is required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in VALID_CUBLAS_CONFIGS:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic validation"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the PT2 triple gate requires exactly one CUDA device")
    for path in (
        inputs.model,
        inputs.train_dir / "best_model.pth",
        inputs.reference,
        inputs.lammps_data,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"NoSOC validation input is missing: {path}")
    if not inputs.lammps_bin.is_file() or not os.access(inputs.lammps_bin, os.X_OK):
        raise FileNotFoundError(f"LAMMPS binary is unavailable: {inputs.lammps_bin}")

    import allegro
    import nequip

    nequip_path = Path(nequip.__file__).resolve()
    allegro_path = Path(allegro.__file__).resolve()
    if not nequip_path.is_relative_to((REPO / "src/core").resolve()):
        raise RuntimeError(f"non-source nequip import rejected: {nequip_path}")
    if not allegro_path.is_relative_to((REPO / "src/models").resolve()):
        raise RuntimeError(f"non-source allegro import rejected: {allegro_path}")
    if torch.__version__ != "2.11.0+cu128":
        raise RuntimeError(f"expected torch 2.11.0+cu128, found {torch.__version__}")

    torch.use_deterministic_algorithms(True)


def make_lammps_graph(
    data: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """Expand periodic target images as LAMMPS ghost atoms."""
    pos = data["pos"]
    spin = data["spin"]
    edges = data["edge_index"]
    shifts = data["edge_cell_shift"]
    cell = data["cell"].reshape(-1, 3, 3)[0]
    atom_types = data["atom_types"].reshape(-1)
    n_atoms = int(pos.shape[0])

    expanded_pos = [value for value in pos]
    expanded_spin = [value for value in spin]
    expanded_types = [value for value in atom_types]
    image_owner = list(range(n_atoms))
    image_lookup: dict[tuple[int, tuple[int, int, int]], int] = {}
    target_indices: list[int] = []

    for edge_id in range(int(edges.shape[1])):
        target = int(edges[1, edge_id])
        raw_shift = shifts[edge_id]
        rounded_shift = tuple(int(round(float(value))) for value in raw_shift)
        if not torch.allclose(
            raw_shift,
            raw_shift.new_tensor(rounded_shift),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(f"non-integral edge shift at edge {edge_id}: {raw_shift}")
        if rounded_shift == (0, 0, 0):
            target_indices.append(target)
            continue

        image_key = (target, rounded_shift)
        if image_key not in image_lookup:
            image_lookup[image_key] = len(expanded_pos)
            expanded_pos.append(pos[target] + raw_shift @ cell)
            expanded_spin.append(spin[target])
            expanded_types.append(atom_types[target])
            image_owner.append(target)
        target_indices.append(image_lookup[image_key])

    expanded = {
        "pos": torch.stack(expanded_pos).contiguous(),
        "spin": torch.stack(expanded_spin).contiguous(),
        "edge_index": torch.stack(
            (
                edges[0],
                torch.tensor(target_indices, dtype=torch.long, device=edges.device),
            )
        ).contiguous(),
        "atom_types": torch.stack(expanded_types).reshape(-1).contiguous(),
    }
    owners = torch.tensor(image_owner, dtype=torch.long)
    return expanded, owners, n_atoms


def physical_outputs(
    outputs: Mapping[str, torch.Tensor], owners: torch.Tensor, n_atoms: int
) -> dict[str, Any]:
    forces = outputs["forces"].detach().cpu().reshape(-1, 3)
    spin_forces = outputs["spin_forces"].detach().cpu().reshape(-1, 3)
    reduced_forces = torch.zeros((n_atoms, 3), dtype=forces.dtype)
    reduced_spin_forces = torch.zeros((n_atoms, 3), dtype=spin_forces.dtype)
    reduced_forces.index_add_(0, owners, forces)
    reduced_spin_forces.index_add_(0, owners, spin_forces)
    owned_energy = outputs["atomic_energy"].detach().cpu()[:n_atoms].sum()
    return {
        "energy_eV": float(owned_energy),
        "forces_eV_per_Angstrom": reduced_forces.double().numpy(),
        "spin_forces_eV": reduced_spin_forces.double().numpy(),
        "raw_total_energy_eV": float(
            outputs["total_energy"].detach().cpu().reshape(-1)[0]
        ),
    }


def canonical_outputs(outputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "energy_eV": float(outputs["total_energy"].detach().cpu().reshape(-1)[0]),
        "forces_eV_per_Angstrom": outputs["forces"]
        .detach()
        .cpu()
        .reshape(-1, 3)
        .double()
        .numpy(),
        "spin_forces_eV": outputs["spin_forces"]
        .detach()
        .cpu()
        .reshape(-1, 3)
        .double()
        .numpy(),
    }


def run_python(inputs: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    print(f"PT2_BUILD_START package={PACKAGE}", flush=True)
    build_pt2(inputs.train_dir, PACKAGE, device="cuda")
    print("PT2_BUILD_COMPLETE", flush=True)
    metadata = deploy.read_deploy_metadata_sidecar(PACKAGE)
    expected_metadata = {
        "schema_version": "1",
        "required_inputs": "pos spin edge_index atom_types",
        "default_dtype": "float32",
        "energy_units": "eV",
        "length_units": "Angstrom",
        "has_spin": "1",
        "has_bec": "0",
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    required_outputs = {
        "total_energy",
        "forces",
        "spin_forces",
        "atomic_energy",
        "virial",
        "atom_virial",
    }
    provided_outputs = set(metadata.get("provided_outputs", "").split())
    if mismatches or not required_outputs.issubset(provided_outputs):
        raise RuntimeError(
            f"PT2 metadata contract failed: mismatches={mismatches}, "
            f"missing_outputs={sorted(required_outputs - provided_outputs)}"
        )

    atoms = read(inputs.reference)
    atoms.calc = None
    atomic_data = AtomicData.from_ase(atoms, r_max=4.0)
    mapper = TypeMapper(chemical_symbol_to_type={"Mn": 0, "Ge": 1})
    canonical_input = mapper(atomic_data).to("cuda").to_dict()
    expanded_input, owners, n_atoms = make_lammps_graph(canonical_input)

    scripted, scripted_metadata = deploy.load_deployed_model(
        inputs.model,
        device="cuda",
        freeze=True,
        set_global_options=True,
    )
    scripted_canonical = canonical_outputs(scripted(clone_tensors(canonical_input)))
    scripted_expanded = physical_outputs(
        scripted(clone_tensors(expanded_input)), owners, n_atoms
    )
    print("TORCHSCRIPT_FORWARD_COMPLETE", flush=True)

    compiled = torch._inductor.aoti_load_package(PACKAGE, run_single_threaded=True)
    input_signature = tuple(metadata[deploy.INPUT_SIGNATURE_KEY].split())
    output_signature = tuple(metadata[deploy.OUTPUT_SIGNATURE_KEY].split())

    def compiled_call() -> tuple[torch.Tensor, ...]:
        result = compiled(*(expanded_input[key].clone() for key in input_signature))
        if isinstance(result, torch.Tensor):
            result = (result,)
        return tuple(value.detach().clone() for value in result)

    first = compiled_call()
    second = compiled_call()
    if len(first) != len(output_signature) or len(second) != len(first):
        raise RuntimeError("PT2 output count does not match output_signature")
    if not all(torch.equal(left, right) for left, right in zip(first, second)):
        raise RuntimeError("PT2 repeated deterministic forwards were not bitwise equal")
    pt2_expanded = physical_outputs(dict(zip(output_signature, first)), owners, n_atoms)
    print("PT2_FORWARD_COMPLETE", flush=True)

    payload = {
        "status": "completed",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "allow_tf32_metadata": scripted_metadata["allow_tf32"],
        "allow_tf32_effective": torch.backends.cuda.matmul.allow_tf32,
        "n_atoms": n_atoms,
        "n_edges": int(expanded_input["edge_index"].shape[1]),
        "n_atoms_with_images": int(expanded_input["pos"].shape[0]),
        "input_signature": list(input_signature),
        "output_signature": list(output_signature),
        "metadata": metadata,
        "pt2_repeat_bitwise_equal": True,
        "torchscript_canonical": json_ready(scripted_canonical),
        "torchscript_lammps_graph": json_ready(scripted_expanded),
        "pt2_lammps_graph": json_ready(pt2_expanded),
    }
    write_json(PYTHON_RESULT, payload)
    return payload, {
        "package_sha256": sha256(PACKAGE),
        "sidecar_sha256": sha256(Path(str(PACKAGE) + ".meta")),
    }


def parse_lammps_dump(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if line.startswith("ITEM: ATOMS ")
    )
    columns = lines[header_index].split()[2:]
    count = int(lines[3])
    rows = []
    for line in lines[header_index + 1 : header_index + 1 + count]:
        values = [float(value) for value in line.split()]
        rows.append(dict(zip(columns, values)))
    rows.sort(key=lambda row: row["id"])
    return np.asarray([[row[key] for key in ("fx", "fy", "fz")] for row in rows])


def parse_lammps_energy(text: str) -> float:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        header = line.split()
        if not header or header[0] != "Step" or "PotEng" not in header:
            continue
        values = lines[index + 1].split()
        if len(values) == len(header):
            return float(dict(zip(header, values))["PotEng"])
    raise ValueError("LAMMPS output has no parseable PotEng row")


def run_lammps(inputs: argparse.Namespace) -> dict[str, Any]:
    print(f"LAMMPS_TORCHSCRIPT_START binary={inputs.lammps_bin}", flush=True)
    torch_root = Path(torch.__file__).resolve().parent
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, (str(torch_root / "lib"), environment.get("LD_LIBRARY_PATH")))
    )
    command = [
        str(inputs.lammps_bin),
        "-in",
        str(LAMMPS_INPUT),
        "-log",
        str(LAMMPS_LOG),
        "-var",
        "data",
        str(inputs.lammps_data),
        "-var",
        "model",
        str(inputs.model),
        "-var",
        "dump",
        str(LAMMPS_DUMP),
    ]
    completed = subprocess.run(
        command,
        cwd=RESULTS_DIR,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=300,
    )
    LAMMPS_LOG.write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"LAMMPS TorchScript run failed with {completed.returncode}:\n"
            + completed.stdout[-4000:]
        )
    forces = parse_lammps_dump(LAMMPS_DUMP)
    energy = parse_lammps_energy(completed.stdout)
    if not math.isfinite(energy) or not np.isfinite(forces).all():
        raise RuntimeError("LAMMPS produced non-finite energy or forces")
    print("LAMMPS_TORCHSCRIPT_COMPLETE", flush=True)
    return {
        "energy_eV": energy,
        "forces_eV_per_Angstrom": forces.tolist(),
        "command": command,
        "log": str(LAMMPS_LOG),
        "dump": str(LAMMPS_DUMP),
    }


def json_ready(values: Mapping[str, Any]) -> dict[str, Any]:
    ready = {}
    for key, value in values.items():
        ready[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return ready


def compare(
    label: str,
    actual: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    include_spin: bool,
) -> dict[str, Any]:
    actual_forces = np.asarray(actual["forces_eV_per_Angstrom"], dtype=float)
    reference_forces = np.asarray(reference["forces_eV_per_Angstrom"], dtype=float)
    if actual_forces.shape != reference_forces.shape:
        raise ValueError(f"{label} force shape mismatch")
    energy_error = abs(float(actual["energy_eV"]) - float(reference["energy_eV"]))
    force_error = float(np.max(np.abs(actual_forces - reference_forces)))
    result: dict[str, Any] = {
        "energy_abs_error_eV": energy_error,
        "energy_atol_eV": ENERGY_ATOL_EV,
        "energy_passed": energy_error <= ENERGY_ATOL_EV,
        "force_max_abs_error_eV_per_Angstrom": force_error,
        "force_atol_eV_per_Angstrom": FORCE_ATOL_EV_A,
        "force_passed": force_error <= FORCE_ATOL_EV_A,
    }
    passed = result["energy_passed"] and result["force_passed"]
    if include_spin:
        actual_spin = np.asarray(actual["spin_forces_eV"], dtype=float)
        reference_spin = np.asarray(reference["spin_forces_eV"], dtype=float)
        spin_error = float(np.max(np.abs(actual_spin - reference_spin)))
        result.update(
            {
                "spin_force_max_abs_error_eV": spin_error,
                "spin_force_atol_eV": SPIN_FORCE_ATOL_EV,
                "spin_force_passed": spin_error <= SPIN_FORCE_ATOL_EV,
            }
        )
        passed = passed and result["spin_force_passed"]
    result["status"] = "passed" if passed else "failed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--lammps-data", type=Path, required=True)
    parser.add_argument("--lammps-bin", type=Path, required=True)
    inputs = parser.parse_args()
    for name, value in vars(inputs).items():
        setattr(inputs, name, value.resolve())
    require_environment(inputs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"PT2_TRIPLE_START device={torch.cuda.get_device_name(0)} "
        f"torch={torch.__version__} results={RESULTS_DIR}",
        flush=True,
    )
    source_hashes_before = {path: sha256(REPO / path) for path in SOURCE_PATHS}
    head_before = git_output("rev-parse", "HEAD")
    status_before = git_output("status", "--short").splitlines()

    python_result, artifact_hashes = run_python(inputs)
    lammps_result = run_lammps(inputs)
    source_hashes_after = {path: sha256(REPO / path) for path in SOURCE_PATHS}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("reviewed PT2/LAMMPS sources changed during validation")

    ts = python_result["torchscript_canonical"]
    ts_graph = python_result["torchscript_lammps_graph"]
    pt2 = python_result["pt2_lammps_graph"]
    comparisons = {
        "torchscript_graph_vs_torchscript_canonical": compare(
            "TorchScript graph/canonical", ts_graph, ts, include_spin=True
        ),
        "pt2_vs_torchscript": compare("PT2/TorchScript", pt2, ts, include_spin=True),
        "lammps_torchscript_vs_python_torchscript": compare(
            "LAMMPS/Python TorchScript", lammps_result, ts, include_spin=False
        ),
        "pt2_vs_lammps_torchscript": compare(
            "PT2/LAMMPS TorchScript", pt2, lammps_result, include_spin=False
        ),
    }
    passed = all(result["status"] == "passed" for result in comparisons.values())
    payload = {
        "status": "passed" if passed else "failed",
        "scope": "NoSOC smoke model; Python CUDA PT2, Python CUDA TorchScript, existing LAMMPS binary TorchScript",
        "head_before": head_before,
        "head_after": git_output("rev-parse", "HEAD"),
        "git_status_before": status_before,
        "git_status_after": git_output("status", "--short").splitlines(),
        "source_hashes": source_hashes_after,
        "artifacts": {
            "model": str(inputs.model),
            "model_sha256": sha256(inputs.model),
            "checkpoint": str(inputs.train_dir / "best_model.pth"),
            "checkpoint_sha256": sha256(inputs.train_dir / "best_model.pth"),
            "reference": str(inputs.reference),
            "reference_sha256": sha256(inputs.reference),
            "lammps_data": str(inputs.lammps_data),
            "lammps_data_sha256": sha256(inputs.lammps_data),
            "lammps_binary": str(inputs.lammps_bin),
            "lammps_binary_sha256": sha256(inputs.lammps_bin),
            "pt2_package": str(PACKAGE),
            **artifact_hashes,
        },
        "environment": {
            "device": python_result["device"],
            "torch_version": python_result["torch_version"],
            "torch_cuda": python_result["torch_cuda"],
            "deterministic_algorithms": python_result["deterministic_algorithms"],
            "cublas_workspace_config": python_result["cublas_workspace_config"],
            "allow_tf32_metadata": python_result["allow_tf32_metadata"],
            "allow_tf32_effective": python_result["allow_tf32_effective"],
        },
        "metadata": python_result["metadata"],
        "graph": {
            "n_atoms": python_result["n_atoms"],
            "n_edges": python_result["n_edges"],
            "n_atoms_with_images": python_result["n_atoms_with_images"],
            "energy_semantics": "sum atomic_energy for owned atoms; reduce image forces to owners",
            "pt2_repeat_bitwise_equal": python_result["pt2_repeat_bitwise_equal"],
        },
        "outputs": {
            "python_torchscript": ts,
            "python_pt2": pt2,
            "lammps_torchscript": lammps_result,
        },
        "comparisons": comparisons,
    }
    write_json(SUMMARY, payload)
    print(json.dumps({"status": payload["status"], "summary": str(SUMMARY)}))
    if not passed:
        raise SystemExit("PT2 triple comparison failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
