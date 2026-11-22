"""Contracts for the PT2, TorchScript, and LAMMPS correctness runner."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

from helpers import ParityResult
from tolerances import (
    PT2_ATOL,
    PT2_CPU_ATOL,
    PT2_E_RTOL,
    PT2_RTOL,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_CI = REPO_ROOT / "tools" / "ci"
sys.path.insert(0, str(TOOLS_CI))


def _threeway_module():
    return importlib.import_module("gpu_pt2_threeway")


def _result(*, delta: float = 0.0) -> ParityResult:
    return ParityResult(
        energy_ev=1.25 + delta,
        forces_ev_a=np.asarray([[0.1 + delta, -0.2, 0.3]], dtype=float),
        spin_forces_ev=np.asarray([[0.4, 0.5 + delta, -0.6]], dtype=float),
        bec_e=np.asarray([np.eye(3) * (0.7 + delta)], dtype=float),
    )


def test_cross_backend_comparison_covers_energy_force_spin_and_bec():
    threeway = _threeway_module()
    comparison = threeway.compare_results(
        "strict", _result(delta=1.0e-8), _result(), profile="pt2_cuda"
    )
    assert comparison["status"] == "PASS"
    assert set(comparison["observables"]) == {
        "energy",
        "forces",
        "spin_forces",
        "bec",
    }


def test_5090_is_a_first_class_target_architecture():
    threeway = _threeway_module()
    assert (
        threeway._run_parser().parse_args(["--gpu", "5090", "--output-dir", "x"]).gpu
        == "5090"
    )


def test_cross_backend_strict_gate_rejects_out_of_tolerance_force():
    threeway = _threeway_module()
    delta = 4.0 * (PT2_ATOL + PT2_RTOL)
    comparison = threeway.compare_results(
        "strict-red", _result(delta=delta), _result(), profile="pt2_cuda"
    )
    assert comparison["status"] == "FAIL"
    assert comparison["observables"]["forces"]["passed"] is False


def test_pt2_cuda_energy_gate_keeps_strict_floor_plus_scale_rtol():
    # BEC-scale cross-backend fp32 aggregation needs a relative term above the
    # strict absolute floor. The floor alone would still reject the
    # ~1e-5-scale BEC energy tail; the calibrated rtol absorbs it while
    # staying ~2x below the physics gate. Every cross-backend energy route
    # (pt2_cuda, pt2_cpu, lammps_pt2) reduces a global scalar at the same
    # scale, so they all carry the same relative term.
    threeway = _threeway_module()
    for profile in ("pt2_cuda", "pt2_cpu", "lammps_pt2"):
        atol = PT2_ATOL if profile in ("pt2_cuda", "lammps_pt2") else PT2_CPU_ATOL
        assert threeway._limits(profile, "energy") == (atol, PT2_E_RTOL)
        # At a BEC-like scale the relative term dominates and the strict floor
        # still applies at a tiny scale.
        bec = compare_results_at(threeway, energy=22.87 + 3.815e-5, ref=22.87)
        assert bec["observables"]["energy"]["passed"] is True
        assert bec["observables"]["energy"]["limit"] > 1.0e-4
    tiny = _result(delta=1.0e-8)
    assert (
        threeway.compare_results("strict", tiny, _result(), profile="pt2_cuda")[
            "status"
        ]
        == "PASS"
    )


def compare_results_at(threeway, *, energy, ref):
    """compare_results with a controlled energy scale (forces/spin/virial neutral)."""
    from helpers import ParityResult
    import numpy as np

    finite = np.asarray([[0.0, 0.0, 0.0]], dtype=float)
    actual = ParityResult(
        energy_ev=energy,
        forces_ev_a=finite.copy(),
        spin_forces_ev=finite.copy(),
        bec_e=np.asarray([np.eye(3) * 0.0], dtype=float),
    )
    reference = ParityResult(
        energy_ev=ref,
        forces_ev_a=finite.copy(),
        spin_forces_ev=finite.copy(),
        bec_e=np.asarray([np.eye(3) * 0.0], dtype=float),
    )
    return threeway.compare_results("bec", actual, reference, profile="pt2_cuda")


def test_cpu_only_plan_marks_cuda_and_lammps_routes_not_run():
    threeway = _threeway_module()
    plan = threeway.route_plan(devices=("cpu",), has_lammps=False)
    assert plan["python_torchscript"] == "RUN"
    assert plan["python_pt2_cpu"] == "RUN"
    assert plan["python_pt2_cuda"] == "NOT_RUN"
    assert plan["lammps_torchscript"] == "NOT_RUN"
    assert plan["lammps_pt2_cuda"] == "NOT_RUN"
