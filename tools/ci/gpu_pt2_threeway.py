#!/usr/bin/env python3
"""Run per-device PT2/TorchScript/LAMMPS correctness checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import runpy
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(REPO_ROOT / "_spingnn_source.py"))["activate_source_tree"]()
TEST_ROOT = REPO_ROOT / "tests_lammps"
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from helpers import (  # noqa: E402
    CASES,
    ParityResult,
    atomic_input,
    cloned,
    python_reference,
    run_case,
    run_lammps,
)
from nequip.data import AtomicDataDict  # noqa: E402
from nequip.scripts import deploy  # noqa: E402
from tolerances import (  # noqa: E402
    BEC_ATOL_E,
    E_ATOL_EV,
    F_ATOL_EV_A,
    PT2_ATOL,
    PT2_CPU_ATOL,
    PT2_CPU_RTOL,
    PT2_E_RTOL,
    PT2_CUDA_F_ATOL_EV_A,
    PT2_CUDA_SPIN_F_ATOL_EV,
    PT2_LAMMPS_F_ATOL_EV_A,
    PT2_LAMMPS_SPIN_F_ATOL_EV,
    PT2_RTOL,
    SPIN_F_ATOL,
)


LINES = ("nosoc", "soc", "bec")
REQUIRED_NEGATIVES = ("target_device_mismatch", "cuda_capability_mismatch")
ROUTES = (
    "python_torchscript",
    "python_pt2_cpu",
    "python_pt2_cuda",
    "lammps_torchscript",
    "lammps_pt2_cuda",
)


class ThreeWayContractError(ValueError):
    """Raised when a requested correctness check cannot be verified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def route_plan(*, devices: Sequence[str], has_lammps: bool) -> dict[str, str]:
    requested = set(devices)
    unknown = requested - {"cpu", "cuda"}
    if unknown:
        raise ThreeWayContractError(f"unsupported devices: {sorted(unknown)}")
    cuda = "cuda" in requested
    return {
        "python_torchscript": "RUN",
        "python_pt2_cpu": "RUN" if "cpu" in requested else "NOT_RUN",
        "python_pt2_cuda": "RUN" if cuda else "NOT_RUN",
        "lammps_torchscript": "RUN" if cuda and has_lammps else "NOT_RUN",
        "lammps_pt2_cuda": "RUN" if cuda and has_lammps else "NOT_RUN",
    }


def _observable_arrays(result: ParityResult) -> dict[str, np.ndarray | None]:
    return {
        "energy": np.asarray(result.energy_ev, dtype=float),
        "forces": np.asarray(result.forces_ev_a, dtype=float),
        "spin_forces": (
            None
            if result.spin_forces_ev is None
            else np.asarray(result.spin_forces_ev, dtype=float)
        ),
        "atom_virial": (
            None
            if result.atom_virial_ev is None
            else np.asarray(result.atom_virial_ev, dtype=float)
        ),
        "bec": None if result.bec_e is None else np.asarray(result.bec_e, dtype=float),
    }


def _limits(profile: str, observable: str) -> tuple[float, float]:
    if profile == "physics":
        return {
            "energy": (E_ATOL_EV, 0.0),
            "forces": (F_ATOL_EV_A, 0.0),
            "spin_forces": (SPIN_F_ATOL, 0.0),
            "atom_virial": (F_ATOL_EV_A, 0.0),
            "bec": (BEC_ATOL_E, 0.0),
        }[observable]
    if profile == "pt2_cpu":
        return {
            "energy": (PT2_CPU_ATOL, PT2_E_RTOL),
            "forces": (PT2_CPU_ATOL, PT2_CPU_RTOL),
            "spin_forces": (PT2_CPU_ATOL, PT2_CPU_RTOL),
            "atom_virial": (PT2_CPU_ATOL, PT2_CPU_RTOL),
            "bec": (PT2_CPU_ATOL, PT2_CPU_RTOL),
        }[observable]
    if profile == "pt2_cuda":
        return {
            "energy": (PT2_ATOL, PT2_E_RTOL),
            "forces": (PT2_CUDA_F_ATOL_EV_A, PT2_RTOL),
            "spin_forces": (PT2_CUDA_SPIN_F_ATOL_EV, PT2_RTOL),
            "atom_virial": (PT2_CUDA_F_ATOL_EV_A, PT2_RTOL),
            "bec": (PT2_CUDA_F_ATOL_EV_A, PT2_RTOL),
        }[observable]
    if profile == "lammps_pt2":
        return {
            "energy": (PT2_ATOL, PT2_E_RTOL),
            "forces": (PT2_LAMMPS_F_ATOL_EV_A, PT2_RTOL),
            "spin_forces": (PT2_LAMMPS_SPIN_F_ATOL_EV, PT2_RTOL),
            "atom_virial": (PT2_LAMMPS_F_ATOL_EV_A, PT2_RTOL),
            "bec": (BEC_ATOL_E, 0.0),
        }[observable]
    raise ThreeWayContractError(f"unknown comparison profile: {profile}")


def compare_results(
    label: str,
    actual: ParityResult,
    reference: ParityResult,
    *,
    profile: str,
) -> dict[str, object]:
    """Compare every observable present in either result with a named gate."""

    actual_values = _observable_arrays(actual)
    reference_values = _observable_arrays(reference)
    observables: dict[str, object] = {}
    passed = True
    for name in actual_values:
        left, right = actual_values[name], reference_values[name]
        if left is None and right is None:
            continue
        if left is None or right is None:
            observables[name] = {
                "passed": False,
                "error": "observable is present on only one route",
            }
            passed = False
            continue
        if left.shape != right.shape:
            observables[name] = {
                "passed": False,
                "actual_shape": list(left.shape),
                "reference_shape": list(right.shape),
                "error": "shape mismatch",
            }
            passed = False
            continue
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            observables[name] = {"passed": False, "error": "nonfinite value"}
            passed = False
            continue
        max_abs_error = float(np.max(np.abs(left - right)))
        reference_scale = float(np.max(np.abs(right)))
        atol, rtol = _limits(profile, name)
        limit = float(atol + rtol * reference_scale)
        within = max_abs_error <= limit
        observables[name] = {
            "passed": within,
            "max_abs_error": max_abs_error,
            "reference_scale": reference_scale,
            "atol": atol,
            "rtol": rtol,
            "limit": limit,
        }
        passed = passed and within
    if not observables:
        raise ThreeWayContractError(f"comparison {label} has no observables")
    return {
        "label": label,
        "profile": profile,
        "status": "PASS" if passed else "FAIL",
        "observables": observables,
    }


def validate_pt2_metadata(
    metadata: Mapping[str, str],
    *,
    expected_device: str,
    expected_capability: Sequence[int] | None = None,
) -> None:
    observed = metadata.get(deploy.TARGET_DEVICE_KEY)
    if observed != expected_device:
        raise ThreeWayContractError(
            f"PT2 target_device mismatch: expected {expected_device}, observed {observed}"
        )
    if expected_device != "cuda":
        return
    capability = ".".join(str(value) for value in expected_capability or ())
    if metadata.get(deploy.CUDA_COMPUTE_CAPABILITY_KEY) != capability:
        raise ThreeWayContractError(
            "PT2 cuda_compute_capability mismatch: "
            f"expected {capability}, observed "
            f"{metadata.get(deploy.CUDA_COMPUTE_CAPABILITY_KEY)}"
        )
    if metadata.get(deploy.CUDA_CODE_KIND_KEY) != "cubin":
        raise ThreeWayContractError(
            "PT2 CUDA package must declare cuda_code_kind=cubin"
        )
    if metadata.get(deploy.TORCH_CUDA_VERSION_KEY) != torch.version.cuda:
        raise ThreeWayContractError(
            "PT2 torch CUDA ABI mismatch: "
            f"expected {torch.version.cuda}, observed "
            f"{metadata.get(deploy.TORCH_CUDA_VERSION_KEY)}"
        )


def _make_lammps_graph(
    data: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """Expand periodic targets to the explicit ghost graph consumed by Allegro."""

    pos = data[AtomicDataDict.POSITIONS_KEY]
    edges = data[AtomicDataDict.EDGE_INDEX_KEY]
    shifts = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
    cell = data[AtomicDataDict.CELL_KEY].reshape(-1, 3, 3)[0]
    atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].reshape(-1)
    spin = data.get(AtomicDataDict.SPIN_KEY)
    n_atoms = int(pos.shape[0])

    expanded_pos = [value for value in pos]
    expanded_types = [value for value in atom_types]
    expanded_spin = [value for value in spin] if spin is not None else None
    owners = list(range(n_atoms))
    image_lookup: dict[tuple[int, tuple[int, int, int]], int] = {}
    targets: list[int] = []
    for edge_id in range(int(edges.shape[1])):
        target = int(edges[1, edge_id])
        raw_shift = shifts[edge_id]
        rounded = tuple(int(round(float(value))) for value in raw_shift)
        if not torch.equal(raw_shift, raw_shift.new_tensor(rounded)):
            raise ThreeWayContractError(
                f"non-integral periodic shift at edge {edge_id}: {raw_shift.tolist()}"
            )
        if rounded == (0, 0, 0):
            targets.append(target)
            continue
        key = (target, rounded)
        if key not in image_lookup:
            image_lookup[key] = len(expanded_pos)
            expanded_pos.append(pos[target] + raw_shift @ cell)
            expanded_types.append(atom_types[target])
            if expanded_spin is not None:
                expanded_spin.append(spin[target])
            owners.append(target)
        targets.append(image_lookup[key])

    expanded = {
        AtomicDataDict.POSITIONS_KEY: torch.stack(expanded_pos).contiguous(),
        AtomicDataDict.EDGE_INDEX_KEY: torch.stack(
            (
                edges[0],
                torch.tensor(targets, dtype=torch.long, device=edges.device),
            )
        ).contiguous(),
        AtomicDataDict.ATOM_TYPE_KEY: torch.stack(expanded_types)
        .reshape(-1)
        .contiguous(),
    }
    if expanded_spin is not None:
        expanded[AtomicDataDict.SPIN_KEY] = torch.stack(expanded_spin).contiguous()
    return expanded, torch.tensor(owners, dtype=torch.long), n_atoms


def _physical_result(
    outputs: Mapping[str, torch.Tensor],
    *,
    owners: torch.Tensor,
    n_atoms: int,
) -> ParityResult:
    total_energy = (
        outputs[AtomicDataDict.TOTAL_ENERGY_KEY].detach().cpu().reshape(-1)[0]
    )
    atomic_energy = (
        outputs[AtomicDataDict.PER_ATOM_ENERGY_KEY].detach().cpu().reshape(-1)
    )
    energy = atomic_energy[:n_atoms].sum() + (total_energy - atomic_energy.sum())

    def reduce_owned(key: str, shape: tuple[int, ...]) -> np.ndarray | None:
        value = outputs.get(key)
        if value is None:
            return None
        tensor = value.detach().cpu().reshape(-1, *shape)
        reduced = torch.zeros((n_atoms, *shape), dtype=tensor.dtype)
        reduced.index_add_(0, owners, tensor)
        return reduced.double().numpy()

    bec = outputs.get(AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY)
    return ParityResult(
        energy_ev=float(energy),
        forces_ev_a=reduce_owned(AtomicDataDict.FORCE_KEY, (3,)),
        spin_forces_ev=reduce_owned(AtomicDataDict.SPIN_FORCE_KEY, (3,)),
        atom_virial_ev=reduce_owned(AtomicDataDict.ATOM_VIRIAL_KEY, (3, 3)),
        bec_e=(
            None
            if bec is None
            else bec.detach().cpu()[:n_atoms].reshape(-1, 3, 3).double().numpy()
        ),
    )


def _result_summary(result: ParityResult) -> dict[str, object]:
    summary: dict[str, object] = {"energy_eV": result.energy_ev}
    for name, value in _observable_arrays(result).items():
        if name == "energy" or value is None:
            continue
        summary[name] = {
            "shape": list(value.shape),
            "max_abs": float(np.max(np.abs(value))),
            "sha256_float64": hashlib.sha256(
                np.ascontiguousarray(value, dtype=np.float64).tobytes()
            ).hexdigest(),
        }
    return summary


def _gpu_memory_snapshot() -> tuple[int, str, str]:
    commands = (
        (
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            "computed",
        ),
        (
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            "query-gpu",
        ),
    )
    for command, mode in commands:
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue
        values = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            try:
                if mode == "computed":
                    if len(fields) == 2 and int(fields[0]) == os.getpid():
                        values.append(int(fields[1]))
                elif len(fields) == 1:
                    values.append(int(fields[0]))
            except ValueError:
                continue
        if values:
            note = (
                "nvidia-smi compute-apps matched this process PID"
                if mode == "computed"
                else "whole-GPU memory.used; may include other processes"
            )
            return max(values) * 1024 * 1024, mode, note
    return 0, "unavailable", "no positive nvidia-smi sample"


def _measure(call, *, device: str, warmup: int, iterations: int):
    if warmup < 0 or iterations < 1:
        raise ThreeWayContractError(
            "warmup must be non-negative and iterations positive"
        )
    output = None
    for _ in range(warmup):
        output = call()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(iterations):
        output = call()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / iterations
    nvidia_peak, nvidia_match, nvidia_note = (
        _gpu_memory_snapshot() if device == "cuda" else (0, "NOT_RUN", "CPU route")
    )
    metrics = {
        "warmup_forwards": warmup,
        "measured_forwards": iterations,
        "statistic": "mean_wall_ms",
        "ms_per_forward": elapsed_ms,
        "ms_per_step": None,
        "torch_alloc_peak": (
            torch.cuda.max_memory_allocated() if device == "cuda" else 0
        ),
        "torch_reserved_peak": (
            torch.cuda.max_memory_reserved() if device == "cuda" else 0
        ),
        "nvidia_smi_peak": nvidia_peak,
        "nvidia_smi_match": nvidia_match,
        "nvidia_smi_note": nvidia_note,
        "host_rss_peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    if output is None:
        raise ThreeWayContractError("backend produced no output")
    return output, metrics


def _run_python_torchscript(
    path: Path,
    graph: Mapping[str, torch.Tensor],
    owners: torch.Tensor,
    n_atoms: int,
    *,
    device: str,
    warmup: int,
    iterations: int,
) -> tuple[ParityResult, dict[str, object]]:
    model, _ = deploy.load_deployed_model(
        path, device=device, freeze=True, set_global_options=True
    )
    device_graph = {key: value.to(device) for key, value in graph.items()}
    output, metrics = _measure(
        lambda: model(cloned(device_graph)),
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    return _physical_result(output, owners=owners, n_atoms=n_atoms), metrics


def _run_python_pt2(
    path: Path,
    graph: Mapping[str, torch.Tensor],
    owners: torch.Tensor,
    n_atoms: int,
    *,
    device: str,
    capability: Sequence[int] | None,
    warmup: int,
    iterations: int,
) -> tuple[ParityResult, dict[str, object], dict[str, str]]:
    metadata = deploy.read_deploy_metadata_sidecar(path)
    validate_pt2_metadata(
        metadata, expected_device=device, expected_capability=capability
    )
    input_signature = tuple(metadata[deploy.INPUT_SIGNATURE_KEY].split())
    output_signature = tuple(metadata[deploy.OUTPUT_SIGNATURE_KEY].split())
    device_graph = {key: value.to(device) for key, value in graph.items()}
    missing = sorted(set(input_signature) - device_graph.keys())
    if missing:
        raise ThreeWayContractError(f"PT2 graph is missing inputs: {missing}")
    compiled = torch._inductor.aoti_load_package(path, run_single_threaded=True)

    def call() -> dict[str, torch.Tensor]:
        raw = compiled(*(device_graph[key].clone() for key in input_signature))
        values = (raw,) if isinstance(raw, torch.Tensor) else tuple(raw)
        if len(values) != len(output_signature):
            raise ThreeWayContractError(
                "PT2 output count does not match output_signature"
            )
        return dict(zip(output_signature, values))

    output, metrics = _measure(
        call, device=device, warmup=warmup, iterations=iterations
    )
    return (
        _physical_result(output, owners=owners, n_atoms=n_atoms),
        metrics,
        metadata,
    )


def _lammps_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT / "src/core"), str(REPO_ROOT / "src/models"))
    )
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    inherited = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = (
        str(torch_lib)
        if not inherited
        else os.pathsep.join((str(torch_lib), inherited))
    )
    env["SPINGNN_DETERMINISTIC"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def _artifact_record(path: Path, metadata: Mapping[str, str] | None = None):
    record: dict[str, object] = {"path": str(path), "sha256": _sha256(path)}
    sidecar = Path(str(path) + ".meta")
    if sidecar.is_file():
        record["sidecar"] = str(sidecar)
        record["sidecar_sha256"] = _sha256(sidecar)
    if metadata is not None:
        record["target_device"] = metadata.get(deploy.TARGET_DEVICE_KEY)
        record["input_signature"] = metadata.get(deploy.INPUT_SIGNATURE_KEY)
        record["output_signature"] = metadata.get(deploy.OUTPUT_SIGNATURE_KEY)
        record["cuda_compute_capability"] = metadata.get(
            deploy.CUDA_COMPUTE_CAPABILITY_KEY
        )
        record["cuda_code_kind"] = metadata.get(deploy.CUDA_CODE_KIND_KEY)
        record["torch_cuda_version"] = metadata.get(deploy.TORCH_CUDA_VERSION_KEY)
    return record


def _rewrite_embedded_manifest(path: Path, updates: Mapping[str, str]) -> None:
    replacement = path.with_suffix(".rewritten.pt2")
    with zipfile.ZipFile(path, "r") as source:
        records = [
            name for name in source.namelist() if name.endswith("/spingnn_deploy.meta")
        ]
        if len(records) != 1:
            raise ThreeWayContractError(
                f"expected one embedded PT2 manifest, observed {len(records)}"
            )
        manifest_name = records[0]
        manifest = dict(
            line.split("\t", 1)
            for line in source.read(manifest_name).decode("utf-8").splitlines()
        )
        manifest.update(updates)
        with zipfile.ZipFile(replacement, "w") as destination:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename == manifest_name:
                    data = deploy.encode_deploy_metadata(manifest).encode("utf-8")
                destination.writestr(info, data)
    os.replace(replacement, path)


def _negative_lammps_run(*, case, package: Path, lammps_bin: Path, workdir: Path):
    variables: dict[str, Any] = {
        "data": case.data_path,
        "model": package,
        "dump": workdir / "negative.dump",
    }
    if case.bec_path is not None:
        variables["bec"] = case.bec_path
    return run_lammps(
        binary=lammps_bin,
        input_path=case.input_path,
        workdir=workdir,
        env=_lammps_environment(),
        variables=variables,
        check=False,
    )


def _run_negatives(
    *,
    models: Mapping[str, Mapping[str, Path]],
    line: str,
    primary_pt2_device: str,
    lammps_bin: Path,
    output_dir: Path,
) -> dict[str, object]:
    case = CASES[line]
    artifacts = models[line]
    cpu_package = (
        artifacts["pt2"] if primary_pt2_device == "cpu" else artifacts.get("pt2_cpu")
    )
    cuda_package = (
        artifacts["pt2"] if primary_pt2_device == "cuda" else artifacts.get("pt2_cuda")
    )
    results: dict[str, object] = {}
    if cpu_package is None or cuda_package is None:
        raise ThreeWayContractError("negative tests require CPU and CUDA PT2 packages")

    target_result = _negative_lammps_run(
        case=case,
        package=cpu_package,
        lammps_bin=lammps_bin,
        workdir=output_dir / "target-device-mismatch",
    )
    target_text = target_result.output
    target_match = (
        "PT2 package targets cpu but this pair runs on cuda" in target_text
        and "build-pt2 --device cuda" in target_text
    )
    results["target_device_mismatch"] = {
        "status": (
            "PASS" if target_result.returncode != 0 and target_match else "FAIL"
        ),
        "returncode": target_result.returncode,
        "diagnostic_matched": target_match,
        "output_tail": target_text[-4000:],
    }

    wrong_arch = output_dir / "cuda-capability-mismatch" / "wrong-architecture.pt2"
    wrong_arch.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(cuda_package, wrong_arch)
    metadata = deploy.read_deploy_metadata_sidecar(cuda_package)
    metadata[deploy.CUDA_COMPUTE_CAPABILITY_KEY] = "99.0"
    _rewrite_embedded_manifest(wrong_arch, {deploy.CUDA_COMPUTE_CAPABILITY_KEY: "99.0"})
    metadata[deploy.PACKAGE_SHA256_KEY] = _sha256(wrong_arch)
    deploy.write_deploy_metadata_sidecar(wrong_arch, metadata)
    arch_result = _negative_lammps_run(
        case=case,
        package=wrong_arch,
        lammps_bin=lammps_bin,
        workdir=output_dir / "cuda-capability-mismatch" / "run",
    )
    arch_text = arch_result.output
    arch_match = "targets CUDA compute capability 99.0" in arch_text
    results["cuda_capability_mismatch"] = {
        "status": "PASS" if arch_result.returncode != 0 and arch_match else "FAIL",
        "returncode": arch_result.returncode,
        "diagnostic_matched": arch_match,
        "output_tail": arch_text[-4000:],
        "artifact": _artifact_record(wrong_arch, metadata),
    }
    return results


def _pt2_path(
    artifacts: Mapping[str, Path], *, device: str, primary_pt2_device: str
) -> Path:
    key = "pt2" if device == primary_pt2_device else f"pt2_{device}"
    try:
        return artifacts[key]
    except KeyError as error:
        raise ThreeWayContractError(
            f"missing {device} PT2 artifact key {key!r}"
        ) from error


def prepare_correctness_models(
    output_dir: Path,
    *,
    lines: Sequence[str],
    devices: Sequence[str],
    primary_pt2_device: str,
) -> tuple[dict[str, dict[str, Path]], dict[str, object]]:
    """Build fixture-width artifacts with one deterministic weight identity."""

    from model_factory import (
        MODEL_SEED,
        build_deployment_model_pair,
        build_pt2_model,
    )

    if primary_pt2_device not in devices:
        raise ThreeWayContractError("primary PT2 device was not requested")
    models: dict[str, dict[str, Path]] = {}
    provenance: dict[str, object] = {}
    for line in lines:
        case = CASES[line]
        line_dir = output_dir / line
        scripted = line_dir / f"{line}.pth"
        primary_pt2 = line_dir / f"{line}-{primary_pt2_device}.pt2"
        build_deployment_model_pair(
            case,
            scripted,
            primary_pt2,
            target_device=primary_pt2_device,
        )
        artifacts = {"torchscript": scripted, "pt2": primary_pt2}
        for device in devices:
            if device == primary_pt2_device:
                continue
            artifacts[f"pt2_{device}"] = build_pt2_model(
                case,
                line_dir / f"{line}-{device}.pt2",
                target_device=device,
            )
        models[line] = artifacts
        provenance[line] = {
            "config": str(case.config_path.relative_to(REPO_ROOT)),
            "config_sha256": _sha256(case.config_path),
            "model_seed": MODEL_SEED,
            "weight_identity": "same MODEL_SEED and exact fixture config",
            "primary_pt2_device": primary_pt2_device,
            "artifacts": {
                backend: _artifact_record(path) for backend, path in artifacts.items()
            },
        }
    return models, provenance


def run_threeway(
    *,
    lines: Sequence[str],
    devices: Sequence[str],
    primary_pt2_device: str,
    models: Mapping[str, Mapping[str, Path]],
    model_provenance: Mapping[str, object],
    output_path: Path,
    source: Mapping[str, object],
    gpu_identity: Mapping[str, object] | None = None,
    lammps_bin: Path | None = None,
    warmup: int = 2,
    iterations: int = 5,
) -> dict[str, object]:
    requested_lines = tuple(lines)
    if not requested_lines or len(set(requested_lines)) != len(requested_lines):
        raise ThreeWayContractError("lines must be nonempty and unique")
    unknown_lines = set(requested_lines) - set(LINES)
    if unknown_lines:
        raise ThreeWayContractError(f"unsupported lines: {sorted(unknown_lines)}")
    plan = route_plan(devices=devices, has_lammps=lammps_bin is not None)
    capability = None if gpu_identity is None else gpu_identity.get("capability")
    evidence: dict[str, object] = {
        "schema": 1,
        "kind": "gpu_pt2_threeway",
        "status": "RUNNING",
        "passed": False,
        "cpu_smoke_passed": False,
        "requested_lines": list(requested_lines),
        "requested_devices": list(devices),
        "route_plan": plan,
        "source": dict(source),
        "gpu_identity": (
            {"status": "NOT_RUN"} if gpu_identity is None else dict(gpu_identity)
        ),
        "models": dict(model_provenance),
        "thresholds": {
            "physics": {
                "energy_atol_eV": E_ATOL_EV,
                "force_atol_eV_per_Angstrom": F_ATOL_EV_A,
                "spin_force_atol_eV": SPIN_F_ATOL,
                "bec_atol_e": BEC_ATOL_E,
            },
            "pt2_cuda": {"atol": PT2_ATOL, "rtol": PT2_RTOL},
            "pt2_cpu": {"atol": PT2_CPU_ATOL, "rtol": PT2_CPU_RTOL},
            "lammps_pt2": {
                "force_atol_eV_per_Angstrom": PT2_LAMMPS_F_ATOL_EV_A,
                "spin_force_atol_eV": PT2_LAMMPS_SPIN_F_ATOL_EV,
            },
        },
        "lines": [],
        "negative_tests": {
            name: {
                "status": "NOT_RUN",
                "reason": "requires CPU and CUDA PT2 archives and LAMMPS",
            }
            for name in REQUIRED_NEGATIVES
        },
    }
    _write_json_atomic(output_path, evidence)

    line_results = evidence["lines"]
    assert isinstance(line_results, list)
    for line in requested_lines:
        case = CASES[line]
        artifacts = models[line]
        result: dict[str, object] = {
            "line": line,
            "status": "RUNNING",
            "routes": {route: {"status": status} for route, status in plan.items()},
            "comparisons": {},
            "artifacts": {},
        }
        line_results.append(result)
        try:
            metadata_path = _pt2_path(
                artifacts,
                device=("cpu" if "cpu" in devices else "cuda"),
                primary_pt2_device=primary_pt2_device,
            )
            graph_metadata = deploy.read_deploy_metadata_sidecar(metadata_path)
            try:
                graph_r_max = float(graph_metadata[deploy.R_MAX_KEY])
            except (KeyError, ValueError) as error:
                raise ThreeWayContractError(
                    "PT2 metadata r_max must be present and numeric"
                ) from error
            graph_case = replace(case, r_max=graph_r_max)
            canonical = python_reference(
                artifacts["torchscript"],
                graph_case,
                device="cuda" if "cuda" in devices else "cpu",
            )
            graph, owners, n_atoms = _make_lammps_graph(atomic_input(graph_case, "cpu"))
            result["graph"] = {
                "r_max_Angstrom": graph_r_max,
                "n_owned_atoms": n_atoms,
                "n_atoms_with_images": int(
                    graph[AtomicDataDict.POSITIONS_KEY].shape[0]
                ),
                "n_edges": int(graph[AtomicDataDict.EDGE_INDEX_KEY].shape[1]),
            }
            ts_device = "cuda" if "cuda" in devices else "cpu"
            python_ts, ts_metrics = _run_python_torchscript(
                artifacts["torchscript"],
                graph,
                owners,
                n_atoms,
                device=ts_device,
                warmup=warmup,
                iterations=iterations,
            )
            result["routes"]["python_torchscript"] = {
                "status": "PASS",
                "device": ts_device,
                "metrics": ts_metrics,
                "output": _result_summary(python_ts),
            }
            result["artifacts"]["torchscript"] = _artifact_record(
                artifacts["torchscript"]
            )
            comparisons = result["comparisons"]
            comparisons["python_torchscript_graph_vs_canonical"] = compare_results(
                "Python TorchScript explicit-ghost/canonical",
                python_ts,
                canonical,
                profile="physics",
            )

            if plan["python_pt2_cpu"] == "RUN":
                cpu_path = _pt2_path(
                    artifacts, device="cpu", primary_pt2_device=primary_pt2_device
                )
                cpu_result, cpu_metrics, cpu_metadata = _run_python_pt2(
                    cpu_path,
                    graph,
                    owners,
                    n_atoms,
                    device="cpu",
                    capability=None,
                    warmup=warmup,
                    iterations=iterations,
                )
                result["routes"]["python_pt2_cpu"] = {
                    "status": "PASS",
                    "device": "cpu",
                    "metrics": cpu_metrics,
                    "output": _result_summary(cpu_result),
                }
                result["artifacts"]["pt2_cpu"] = _artifact_record(
                    cpu_path, cpu_metadata
                )
                comparisons["python_pt2_cpu_vs_torchscript"] = compare_results(
                    "Python PT2 CPU/TorchScript",
                    cpu_result,
                    python_ts,
                    profile="pt2_cpu",
                )

            if plan["python_pt2_cuda"] == "RUN":
                if not torch.cuda.is_available():
                    raise ThreeWayContractError(
                        "CUDA route requested but CUDA is unavailable"
                    )
                cuda_path = _pt2_path(
                    artifacts, device="cuda", primary_pt2_device=primary_pt2_device
                )
                cuda_result, cuda_metrics, cuda_metadata = _run_python_pt2(
                    cuda_path,
                    graph,
                    owners,
                    n_atoms,
                    device="cuda",
                    capability=capability,
                    warmup=warmup,
                    iterations=iterations,
                )
                result["routes"]["python_pt2_cuda"] = {
                    "status": "PASS",
                    "device": "cuda",
                    "metrics": cuda_metrics,
                    "output": _result_summary(cuda_result),
                }
                result["artifacts"]["pt2_cuda"] = _artifact_record(
                    cuda_path, cuda_metadata
                )
                comparisons["python_pt2_cuda_vs_torchscript"] = compare_results(
                    "Python PT2 CUDA/TorchScript",
                    cuda_result,
                    python_ts,
                    profile="pt2_cuda",
                )

            if plan["lammps_torchscript"] == "RUN":
                assert lammps_bin is not None
                lammps_env = _lammps_environment()
                lammps_env["SPINGNN_REQUIRE_CUDA_LAMMPS"] = "1"
                lammps_ts = run_case(
                    case=case,
                    model_path=artifacts["torchscript"],
                    binary=lammps_bin,
                    workdir=output_path.parent / "lammps" / line / "torchscript",
                    env=lammps_env,
                )
                cuda_path = _pt2_path(
                    artifacts, device="cuda", primary_pt2_device=primary_pt2_device
                )
                lammps_pt2 = run_case(
                    case=case,
                    model_path=cuda_path,
                    binary=lammps_bin,
                    workdir=output_path.parent / "lammps" / line / "pt2-cuda",
                    env=lammps_env,
                )
                result["routes"]["lammps_torchscript"] = {
                    "status": "PASS",
                    "device": "cuda",
                    "output": _result_summary(lammps_ts),
                }
                result["routes"]["lammps_pt2_cuda"] = {
                    "status": "PASS",
                    "device": "cuda",
                    "output": _result_summary(lammps_pt2),
                }
                comparisons["lammps_torchscript_vs_python"] = compare_results(
                    "LAMMPS TorchScript/Python TorchScript",
                    lammps_ts,
                    canonical,
                    profile="physics",
                )
                comparisons["lammps_pt2_cuda_vs_python"] = compare_results(
                    "LAMMPS PT2 CUDA/Python TorchScript",
                    lammps_pt2,
                    canonical,
                    profile="physics",
                )
                comparisons["lammps_pt2_cuda_vs_torchscript"] = compare_results(
                    "LAMMPS PT2 CUDA/TorchScript",
                    lammps_pt2,
                    lammps_ts,
                    profile="lammps_pt2",
                )

            failed = [
                name
                for name, comparison in comparisons.items()
                if comparison["status"] != "PASS"
            ]
            result["status"] = "PASS" if not failed else "FAIL"
            if failed:
                result["failed_comparisons"] = failed
        except Exception as error:  # preserve evidence before the outer gate fails
            result["status"] = "FAIL"
            result["error_type"] = type(error).__name__
            result["error"] = " ".join(str(error).split())[:2000]
        _write_json_atomic(output_path, evidence)

    if set(devices) == {"cpu", "cuda"} and lammps_bin is not None:
        try:
            evidence["negative_tests"] = _run_negatives(
                models=models,
                line=requested_lines[0],
                primary_pt2_device=primary_pt2_device,
                lammps_bin=lammps_bin,
                output_dir=output_path.parent / "negative-tests",
            )
        except Exception as error:
            evidence["negative_tests"] = {
                name: {
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": " ".join(str(error).split())[:2000],
                }
                for name in REQUIRED_NEGATIVES
            }

    line_pass = all(item.get("status") == "PASS" for item in line_results)
    cpu_pass = line_pass and "cpu" in devices
    full_routes = all(plan[route] == "RUN" for route in ROUTES)
    negatives_pass = all(
        evidence["negative_tests"].get(name, {}).get("status") == "PASS"
        for name in REQUIRED_NEGATIVES
    )
    evidence["cpu_smoke_passed"] = cpu_pass
    evidence["passed"] = line_pass and full_routes and negatives_pass
    negative_failure = any(
        item.get("status") == "FAIL" for item in evidence["negative_tests"].values()
    )
    if not line_pass or negative_failure:
        evidence["status"] = "FAIL"
    else:
        evidence["status"] = "PASS" if evidence["passed"] else "PARTIAL"
    _write_json_atomic(output_path, evidence)
    return evidence


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line", action="append", choices=LINES)
    parser.add_argument(
        "--devices", nargs="+", choices=("cpu", "cuda"), default=("cpu", "cuda")
    )
    parser.add_argument("--gpu", choices=("4090", "5090", "h100"))
    parser.add_argument("--lammps-bin", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    return parser


def _gpu_identity(expected: str) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ThreeWayContractError("CUDA checks require exactly one visible GPU")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    expected_capability = {"4090": (8, 9), "5090": (12, 0), "h100": (9, 0)}[expected]
    if expected not in name.lower() or capability != expected_capability:
        raise ThreeWayContractError(
            f"GPU mismatch: expected {expected} {expected_capability}, "
            f"observed {name} {capability}"
        )
    device = torch.cuda.get_device_properties(0)
    return {
        "name": name,
        "capability": list(capability),
        "count": 1,
        "uuid": str(device.uuid),
        "total_memory_bytes": device.total_memory,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _run_parser().parse_args(argv)
        lines = tuple(args.line or LINES)
        devices = tuple(dict.fromkeys(args.devices))
        if "cuda" in devices and args.gpu is None:
            raise ThreeWayContractError(
                "--gpu is required when --devices includes cuda"
            )
        if args.lammps_bin is not None:
            if "cuda" not in devices:
                raise ThreeWayContractError("--lammps-bin requires the CUDA route")
            args.lammps_bin = args.lammps_bin.resolve(strict=True)
        if len(set(lines)) != len(lines):
            raise ThreeWayContractError("--line selections must be unique")
        if args.warmup < 0 or args.iterations < 1:
            raise ThreeWayContractError(
                "warmup must be nonnegative and iterations positive"
            )
        gpu_identity = _gpu_identity(args.gpu) if "cuda" in devices else None
        output_dir = args.output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ThreeWayContractError(f"output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        primary = "cuda" if "cuda" in devices else "cpu"
        models, provenance = prepare_correctness_models(
            output_dir / "models",
            lines=lines,
            devices=devices,
            primary_pt2_device=primary,
        )
        import allegro
        import e3nn
        import nequip

        source = {
            "nequip": nequip.__file__,
            "allegro": allegro.__file__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "e3nn": e3nn.__version__,
        }
        evidence = run_threeway(
            lines=lines,
            devices=devices,
            primary_pt2_device=primary,
            models=models,
            model_provenance=provenance,
            output_path=output_dir / "threeway.json",
            source=source,
            gpu_identity=gpu_identity,
            lammps_bin=args.lammps_bin,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(
            json.dumps(
                {
                    "status": evidence["status"],
                    "passed": evidence["passed"],
                    "cpu_smoke_passed": evidence["cpu_smoke_passed"],
                    "evidence": str(output_dir / "threeway.json"),
                },
                sort_keys=True,
            )
        )
        return 0 if evidence["status"] in {"PASS", "PARTIAL"} else 1
    except (
        ThreeWayContractError,
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"GPU_PT2_THREEWAY_ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
