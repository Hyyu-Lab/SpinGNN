"""LAMMPS RSS evidence must exclude the memory of its Python launcher."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from helpers import CASES, run_lammps


def _rss_launcher(path: Path, binary: Path, reserve_bytes: int) -> Path:
    # A separate stdlib-only interpreter controls the launcher high-water mark
    # independently of model construction and allocations in the pytest process.
    path.write_text(
        f"#!{sys.executable}\n"
        "import os, re, subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"payload = bytearray({reserve_bytes})\n"
        f"binary = {str(binary)!r}\n"
        "output = Path('rss-child.out')\n"
        "peak = 0\n"
        "with output.open('w') as stream:\n"
        "    process = subprocess.Popen([binary, *sys.argv[1:]], stdout=stream, stderr=subprocess.STDOUT)\n"
        "    while process.poll() is None:\n"
        "        try:\n"
        "            if os.readlink(f'/proc/{process.pid}/exe') == str(Path(binary).resolve()):\n"
        "                status = Path(f'/proc/{process.pid}/status').read_text()\n"
        "                match = re.search(r'^VmHWM:\\s+(\\d+) kB$', status, re.MULTILINE)\n"
        "                if match:\n"
        "                    peak = max(peak, int(match.group(1)) * 1024)\n"
        "        except (FileNotFoundError, ProcessLookupError):\n"
        "            pass\n"
        "        time.sleep(0.005)\n"
        "print(output.read_text(), end='')\n"
        "print(f'SPINGNN_TEST_OBSERVED_VMHWM bytes={peak}')\n"
        "sys.exit(process.returncode)\n"
    )
    path.chmod(0o755)
    return path


@pytest.mark.lammps
@pytest.mark.inference_perf
@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux current-executable RSS contract"
)
def test_host_rss_peak_does_not_inherit_python_launcher_allocations(
    lammps_bin, lammps_env, model_paths, tmp_path
):
    case = CASES["nosoc"]
    input_path = tmp_path / "rss.in"
    input_path.write_text(
        case.input_path.read_text().replace("run 0", "run 0\nshell sleep 0.2")
    )
    env = dict(lammps_env, SPINGNN_PERF="1")

    def run(reserve_bytes, name):
        directory = tmp_path / name
        directory.mkdir()
        dump_path = directory / "atoms.dump"
        process = run_lammps(
            binary=_rss_launcher(directory / "launcher", lammps_bin, reserve_bytes),
            input_path=input_path,
            workdir=directory,
            env=env,
            variables={
                "data": case.data_path,
                "model": model_paths[case.name],
                "dump": dump_path,
            },
            dump_path=dump_path,
        )
        observed = int(
            re.search(r"SPINGNN_TEST_OBSERVED_VMHWM bytes=(\d+)", process.stdout).group(
                1
            )
        )
        assert observed > 0
        records = [
            int(re.search(r"host_rss_peak=(\d+)", line).group(1))
            for line in process.stdout.splitlines()
            if line.startswith("SPINGNN_PERF ")
        ]
        assert records and min(records) > 0
        assert "SPINGNN_PERF_RSS_FALLBACK" not in process.stdout
        return max(records), observed

    _, baseline_observed = run(0, "small-parent")
    # This padding is a controlled input, not a relaxed assertion threshold:
    # the child's reported peak must fit its independently observed VmHWM.
    reported, observed = run(baseline_observed + 64 * 1024**2, "large-parent")
    assert reported <= observed, (
        f"reported RSS {reported} exceeds current-executable VmHWM {observed}; "
        "the launcher high-water mark leaked across exec"
    )
