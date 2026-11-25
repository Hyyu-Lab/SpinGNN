"""Optional pytest plugin recording node identities, skips and phase timings.

Load with ``-p pytest_evidence --evidence-json PATH`` through source_python.py.
Each xdist worker writes a separate report so collection metadata survives
parallel execution; the controller records the aggregated test outcomes.
"""

import json
import os
from pathlib import Path
import platform
import resource
import sys

import pytest


COMPILE_CURVE_NODEIDS = {
    "compile-allegro": (
        "src/models/tests/model/test_allegro_spin_nosoc.py::"
        "TestAllegroNoSOCParity::test_full_fx_training_curve[float32-config0-base_config0]"
    ),
    "compile-checkpoint": (
        "src/models/tests/model/test_allegro_activation_checkpoint.py::"
        "TestCheckpointCompileParity::test_full_fx_training_curve[float32]"
    ),
}


def require_compile_curve(evidence, suite):
    """Bind speed evidence to the fixture whose workload the gate calibrated."""
    expected = COMPILE_CURVE_NODEIDS[suite]
    selected = {row["nodeid"] for row in evidence["selected"]}
    curves = [
        row["nodeid"]
        for row in evidence["reports"]
        if row["phase"] == "call"
        and row["outcome"] == "passed"
        and "xfail" not in row
        and row["nodeid"].rsplit("::", 1)[-1].split("[", 1)[0]
        == "test_full_fx_training_curve"
    ]
    if evidence["exit_code"] != 0 or expected not in selected or curves != [expected]:
        raise ValueError(
            f"{suite} requires exactly one successful curve from {expected}; "
            f"observed {curves}, pytest exit {evidence['exit_code']}"
        )
    return expected


def require_cuda_passes(evidence, minimum):
    """Reject a device gate satisfied only by CPU cases or expected failures."""
    cuda_nodes = {
        row["nodeid"]
        for row in evidence["selected"]
        if row.get("parameters", {}).get("device", "").split(":", 1)[0] == "cuda"
    }
    passed_nodes = {
        row["nodeid"]
        for row in evidence["reports"]
        if row["phase"] == "call" and row["outcome"] == "passed" and "xfail" not in row
    }
    passed = len(cuda_nodes & passed_nodes)
    if passed < minimum:
        raise ValueError(f"CUDA parameter cases passed {passed}; minimum is {minimum}")
    return passed


def pytest_addoption(parser):
    group = parser.getgroup("spingnn evidence")
    group.addoption(
        "--evidence-json", help="Write test identities and outcomes as JSON"
    )
    group.addoption(
        "--require-test-passes",
        type=int,
        default=0,
        help="Fail a successful suite with fewer actual passes",
    )
    group.addoption(
        "--require-test-executed",
        type=int,
        default=0,
        help="Treat missing executed test bodies as an infrastructure failure",
    )


class Evidence:
    """Keep evidence per pytest configuration rather than shared global state."""

    def __init__(self, config):
        self.config = config
        self.selected = []
        self.deselected = []
        self.reports = []

    @staticmethod
    def describe(item):
        parameters = getattr(getattr(item, "callspec", None), "params", {})
        return {
            "nodeid": item.nodeid,
            "markers": sorted({mark.name for mark in item.iter_markers()}),
            "parameters": {
                name: str(parameters[name])
                for name in ("device", "float_tolerance")
                if name in parameters
            },
        }

    def pytest_deselected(self, items):
        self.deselected.extend(self.describe(item) for item in items)

    def pytest_collection_finish(self, session):
        self.selected = [self.describe(item) for item in session.items]

    def pytest_runtest_logreport(self, report):
        row = {
            "nodeid": report.nodeid,
            "phase": report.when,
            "outcome": report.outcome,
            "seconds": report.duration,
        }
        if report.longrepr:
            row["reason"] = str(report.longrepr)
        if hasattr(report, "wasxfail"):
            row["xfail"] = str(report.wasxfail)
        self.reports.append(row)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        passed = sum(
            row["phase"] == "call" and row["outcome"] == "passed" and "xfail" not in row
            for row in self.reports
        )
        # A subprocess launched by a worker inherits PYTEST_XDIST_WORKER even
        # when its own pytest session is serial. Only xdist's config payload
        # identifies a real worker and may bypass the controller-level gate.
        worker = getattr(self.config, "workerinput", {}).get("workerid")
        executed = sum(
            row["phase"] == "call"
            and row["outcome"] in ("passed", "failed")
            and "xfail" not in row
            for row in self.reports
        )
        if (
            not worker
            and exitstatus in (0, 1)
            and executed < self.config.getoption("--require-test-executed")
        ):
            # Informational numerical failures may keep exit 1. No executed
            # bodies must be distinguishable from that accepted experiment.
            session.exitstatus = pytest.ExitCode.INTERNAL_ERROR
        if (
            not worker
            and session.exitstatus == 0
            and passed < self.config.getoption("--require-test-passes")
        ):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            reporter = self.config.pluginmanager.getplugin("terminalreporter")
            if reporter is not None:
                reporter.write_line(
                    f"Required lane produced {passed} passes; minimum is "
                    f"{self.config.getoption('--require-test-passes')}",
                    red=True,
                )
        target = self.config.getoption("--evidence-json")
        if not target:
            return
        path = Path(target)
        if worker:
            path = path.with_name(path.stem + "." + worker + path.suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        origins = {
            name: str(getattr(sys.modules[name], "__file__", None))
            for name in ("nequip", "allegro", "torch", "e3nn")
            if name in sys.modules
        }
        usage = resource.getrusage(resource.RUSAGE_SELF)
        payload = {
            "schema_version": 1,
            "python": platform.python_version(),
            "executable": sys.executable,
            "worker": worker,
            "selected": self.selected,
            "deselected": self.deselected,
            "reports": self.reports,
            "passed": passed,
            "executed": executed,
            "exit_code": int(session.exitstatus),
            "origins": origins,
            "process_resources": {
                "peak_rss_bytes": int(
                    usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
                ),
                "user_cpu_seconds": usage.ru_utime,
                "system_cpu_seconds": usage.ru_stime,
            },
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "PYTHONHASHSEED",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "TRITON_CACHE_DIR",
                    "SPINGNN_TEST_COMPILE",
                    "SPINGNN_TEST_AUTOCAST",
                )
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)


def pytest_configure(config):
    config.pluginmanager.register(Evidence(config), "spingnn-evidence-recorder")
