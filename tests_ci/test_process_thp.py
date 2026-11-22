"""Explicit THP control must survive exec without changing its parent."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/ci/process_thp.py"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux prctl API")
PROBE = (
    "import json,os; from pathlib import Path; "
    "s=dict(line.split(':',1) for line in "
    "Path('/proc/self/status').read_text().splitlines() if ':' in line); "
    "print(json.dumps({'pid':os.getpid(),'thp':int(s['THP_enabled'])}))"
)


@pytest.mark.parametrize("policy", ["inherit", "disable"])
def test_policy_survives_exec_and_fork_without_affecting_parent(tmp_path, policy):
    before = subprocess.check_output([sys.executable, "-c", PROBE], text=True)
    inherited = json.loads(before)["thp"]
    record = tmp_path / "transition.json"
    payload = (
        "import subprocess,sys; "
        + PROBE
        + "; print(subprocess.check_output([sys.executable,'-c',"
        + repr(PROBE)
        + "],text=True).strip())"
    )
    command = [sys.executable, "-c", payload]
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--policy",
            policy,
            "--record",
            str(record),
            "--",
            *command,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 0, stderr
    actual, descendant = [json.loads(line) for line in stdout.splitlines()]
    transition = json.loads(record.read_text())
    expected = 0 if policy == "disable" else inherited
    assert actual["thp"] == descendant["thp"] == expected
    assert actual["pid"] == transition["pid"] == process.pid
    assert descendant["pid"] != process.pid
    assert transition["command"] == command
    assert transition["previous_disable_flag"] == 1 - inherited
    assert transition["effective_disable_flag"] == 1 - expected
    assert (
        transition["launcher_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    )
    after = subprocess.check_output([sys.executable, "-c", PROBE], text=True)
    assert json.loads(after)["thp"] == inherited


@pytest.mark.parametrize("command", [[], ["relative-python"]])
def test_invalid_command_does_not_publish_transition(tmp_path, command):
    record = tmp_path / "transition.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--policy",
            "disable",
            "--record",
            str(record),
            "--",
            *command,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert not record.exists()


def test_existing_record_is_never_overwritten(tmp_path):
    record = tmp_path / "transition.json"
    record.write_text("preserve original evidence\n")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--policy",
            "disable",
            "--record",
            str(record),
            "--",
            sys.executable,
            "-c",
            "raise AssertionError('must not execute')",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert record.read_text() == "preserve original evidence\n"


def test_exec_preserves_payload_failure_exit(tmp_path):
    record = tmp_path / "transition.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--policy",
            "disable",
            "--record",
            str(record),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 7
    assert json.loads(record.read_text())["effective_disable_flag"] == 1
