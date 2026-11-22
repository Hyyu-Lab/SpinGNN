"""Test evidence and all-skipped rejection using independent pytest processes."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evidence_tool", ROOT / "tools/ci/pytest_evidence.py"
)
evidence_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence_tool)


def run_fixture(tmp_path, source, *extra):
    (tmp_path / "test_example.py").write_text(source)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "tools/ci"))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test_example.py",
            "-q",
            "-p",
            "pytest_evidence",
            "-p",
            "no:cacheprovider",
            "--evidence-json",
            str(tmp_path / "evidence.json"),
            "--require-test-passes",
            "1",
            *extra,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, json.loads((tmp_path / "evidence.json").read_text())


def test_all_skipped_cannot_satisfy_required_lane(tmp_path):
    result, evidence = run_fixture(
        tmp_path,
        "import pytest\ndef test_skip(): pytest.skip('missing prerequisite')\n",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert evidence["passed"] == 0
    assert "missing prerequisite" in str(evidence["reports"])


def test_selection_and_phase_outcomes_are_preserved(tmp_path):
    result, evidence = run_fixture(
        tmp_path, "def test_pass(): pass\ndef test_other(): pass\n", "-k", "test_pass"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert evidence["passed"] == 1
    assert [row["nodeid"] for row in evidence["selected"]] == [
        "test_example.py::test_pass"
    ]
    assert [row["nodeid"] for row in evidence["deselected"]] == [
        "test_example.py::test_other"
    ]
    assert {row["phase"] for row in evidence["reports"]} == {
        "setup",
        "call",
        "teardown",
    }


def test_informational_all_skipped_is_infrastructure_failure(tmp_path):
    result, evidence = run_fixture(
        tmp_path,
        "import pytest\ndef test_skip(): pytest.skip('no GPU')\n",
        "--require-test-passes",
        "0",
        "--require-test-executed",
        "1",
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert evidence["executed"] == 0


def test_informational_numerical_failure_keeps_test_failure_code(tmp_path):
    result, evidence = run_fixture(
        tmp_path,
        "def test_value(): assert 1 == 2\n",
        "--require-test-passes",
        "0",
        "--require-test-executed",
        "1",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert evidence["executed"] == 1


def test_device_and_dtype_parameters_survive_selection(tmp_path):
    result, evidence = run_fixture(
        tmp_path,
        "import pytest\n"
        "@pytest.mark.parametrize('device,float_tolerance,config', "
        "[('cpu','float32',{'depth':3}),('cuda:0','float64',{'depth':4})])\n"
        "def test_value(device,float_tolerance,config): pass\n",
        "-k",
        "float64",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert evidence["selected"][0]["parameters"] == {
        "device": "cuda:0",
        "float_tolerance": "float64",
    }
    assert evidence["deselected"][0]["parameters"] == {
        "device": "cpu",
        "float_tolerance": "float32",
    }


@pytest.mark.parametrize("cuda_outcome", ["skipped", "passed", "xpass"])
def test_cuda_gate_requires_actual_cuda_call_pass(cuda_outcome):
    cuda_report = {
        "nodeid": "test_stress[cuda]",
        "phase": "call",
        "outcome": "passed" if cuda_outcome == "xpass" else cuda_outcome,
    }
    if cuda_outcome == "xpass":
        cuda_report["xfail"] = "known failure"
    evidence = {
        "selected": [
            {"nodeid": "test_stress[cpu]", "parameters": {"device": "cpu"}},
            {"nodeid": "test_stress[cuda]", "parameters": {"device": "cuda:0"}},
        ],
        "reports": [
            {"nodeid": "test_stress[cpu]", "phase": "call", "outcome": "passed"},
            cuda_report,
        ],
    }
    if cuda_outcome == "passed":
        assert evidence_tool.require_cuda_passes(evidence, 1) == 1
    else:
        with pytest.raises(ValueError, match="CUDA parameter cases passed 0"):
            evidence_tool.require_cuda_passes(evidence, 1)


@pytest.mark.parametrize(
    "suite", ["compile-nequip", "compile-allegro", "compile-checkpoint"]
)
def test_compile_curve_matches_its_suite(suite):
    nodeid = evidence_tool.COMPILE_CURVE_NODEIDS[suite]
    evidence = {
        "exit_code": 0,
        "selected": [{"nodeid": nodeid}],
        "reports": [{"nodeid": nodeid, "phase": "call", "outcome": "passed"}],
    }
    assert evidence_tool.require_compile_curve(evidence, suite) == nodeid


@pytest.mark.parametrize(
    "fault",
    [
        "replacement",
        "duplicate",
        "missing",
        "skipped",
        "xfail",
        "setup-only",
        "not-selected",
        "pytest-failed",
    ],
)
def test_compile_curve_rejects_replacement_or_nonpass_evidence(fault):
    expected = evidence_tool.COMPILE_CURVE_NODEIDS["compile-allegro"]
    report = {"nodeid": expected, "phase": "call", "outcome": "passed"}
    evidence = {"exit_code": 0, "selected": [{"nodeid": expected}], "reports": [report]}
    if fault == "replacement":
        report["nodeid"] = evidence_tool.COMPILE_CURVE_NODEIDS["compile-checkpoint"]
        evidence["selected"].append({"nodeid": report["nodeid"]})
    elif fault == "duplicate":
        evidence["reports"].append(dict(report))
    elif fault == "missing":
        evidence["reports"] = []
    elif fault == "skipped":
        report["outcome"] = "skipped"
    elif fault == "xfail":
        report["xfail"] = "known failure"
    elif fault == "setup-only":
        report["phase"] = "setup"
    elif fault == "not-selected":
        evidence["selected"] = []
    elif fault == "pytest-failed":
        evidence["exit_code"] = 1
    with pytest.raises(ValueError, match="requires exactly one successful curve"):
        evidence_tool.require_compile_curve(evidence, "compile-allegro")


def run_curve_gate(tmp_path, fields, minimum="calibration", gpu_expect="4090"):
    script = (ROOT / "tools/ci/gpu_full_tests.sh").read_text()
    inside = script.split("require_full_fx_benchmark() {", 1)[1]
    parser = inside.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    path = tmp_path / "curve.log"
    path.write_text(
        "FULL_FX_WALLTIME " + " ".join(f"{k}={v}" for k, v in fields.items()) + "\n"
    )
    return subprocess.run(
        [sys.executable, "-c", parser, str(path), minimum, "1e-5"],
        env=dict(os.environ, SPINGNN_GPU_EXPECT=gpu_expect),
        capture_output=True,
        text=True,
        timeout=30,
    )


def valid_curve_fields():
    return {
        "eager_seconds": "0.1",
        "full_fx_seconds": "1",
        "max_loss_relative": "1e-6",
        "cache_entries": "1",
        "measurement": "steady_state_median",
        "steps": "19",
    }


def test_checkpoint_speed_is_calibration_without_weakening_original_floor(tmp_path):
    fields = valid_curve_fields()
    measured = run_curve_gate(tmp_path, fields)
    assert measured.returncode == 0, measured.stdout + measured.stderr
    assert "FULL_FX_CALIBRATION gpu=4090 speedup=0.100000" in measured.stdout
    calibrated = run_curve_gate(tmp_path, fields, minimum="1.8")
    assert calibrated.returncode != 0
    assert "below 1.800" in calibrated.stderr


@pytest.mark.parametrize(
    "gpu_expect,minimum",
    [("h100", "4.4"), ("h100", "2.0"), ("4090", "4.3"), ("4090", "1.8")],
)
def test_original_calibrated_floors_keep_their_boundary(tmp_path, gpu_expect, minimum):
    fields = valid_curve_fields()
    fields["eager_seconds"] = minimum
    accepted = run_curve_gate(tmp_path, fields, minimum, gpu_expect)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    fields["eager_seconds"] = str(float(minimum) - 0.001)
    rejected = run_curve_gate(tmp_path, fields, minimum, gpu_expect)
    assert rejected.returncode != 0
    assert "is below" in rejected.stderr


@pytest.mark.parametrize(
    "key,value",
    [
        ("eager_seconds", "inf"),
        ("full_fx_seconds", "nan"),
        ("max_loss_relative", "nan"),
        ("max_loss_relative", "2e-5"),
        ("cache_entries", "2"),
        ("steps", "18"),
    ],
)
def test_checkpoint_calibration_keeps_numerical_and_measurement_gates(
    tmp_path, key, value
):
    fields = valid_curve_fields()
    fields[key] = value
    result = run_curve_gate(tmp_path, fields)
    assert result.returncode != 0, result.stdout


def test_checkpoint_collection_cannot_replace_calibrated_allegro_workload():
    script = (ROOT / "tools/ci/gpu_full_tests.sh").read_text().replace("\\\n", "")
    commands = {
        tokens[1]: tokens
        for line in script.splitlines()
        if line.startswith("run_suite compile-")
        for tokens in [shlex.split(line)]
    }
    allegro = commands["compile-allegro"]
    assert allegro[allegro.index("-k") + 1] == "not TestCheckpointCompileParity"
    assert allegro[3] == "19"
    checkpoint = commands["compile-checkpoint"]
    assert checkpoint[3] == "3"
    assert checkpoint[4] == (
        "tests/model/test_allegro_activation_checkpoint.py::TestCheckpointCompileParity"
    )
    assert evidence_tool.COMPILE_CURVE_NODEIDS["compile-allegro"] == (
        "src/models/tests/model/test_allegro_spin_nosoc.py::"
        "TestAllegroNoSOCParity::test_full_fx_training_curve[float32-config0-base_config0]"
    )
