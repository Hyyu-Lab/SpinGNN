"""GPU MPI rank↔UUID ownership evidence (parent R5: 8 and 2x8 cards).

The MPI test launches ``mpi_uuid_probe.py`` once per rank through the shared
``mpi_prefix`` launcher fixture. Every rank pins its node-local GPU and
reports the physical device UUID; rank 0 gloo-gathers the reports and prints
``GPU_MPI_OWNERSHIP ranks=N nodes=M status=PASS uuids=…``. The pytest process
independently re-parses the per-rank lines and re-checks bijectivity, so the
verdict does not rest on the probe's own arithmetic alone. The pure verdict
logic is covered by CPU unit tests below with mocked rank data.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mpi_uuid_probe import OWNERSHIP_RE, build_ownership_line, parse_rank_uuid_lines

PROBE_SCRIPT = Path(__file__).resolve().parent / "mpi_uuid_probe.py"
PROBE_TIMEOUT_SECONDS = 600


def _mock_entries(count: int, *, hosts: tuple[str, ...] = ("gpu-node-0",)):
    return {
        rank: {
            "local_rank": str(rank),
            "host": hosts[rank % len(hosts)],
            "device": f"cuda:{rank}",
            "uuid": f"GPU-mock-{rank:04d}",
        }
        for rank in range(count)
    }


def test_parse_rank_uuid_lines_roundtrip():
    output = "\n".join(
        f"GPU_MPI_RANK_UUID rank={rank} local_rank={rank} host=gpu-node-0"
        f" device=cuda:{rank} uuid=GPU-mock-{rank:04d}"
        for rank in range(8)
    )
    entries = parse_rank_uuid_lines(output)
    assert len(entries) == 8
    assert entries[3]["uuid"] == "GPU-mock-0003"
    assert entries[7]["device"] == "cuda:7"


def test_parse_rank_uuid_lines_rejects_duplicate_rank():
    output = (
        "GPU_MPI_RANK_UUID rank=0 local_rank=0 host=a device=cuda:0 uuid=GPU-x\n"
        "GPU_MPI_RANK_UUID rank=0 local_rank=1 host=a device=cuda:1 uuid=GPU-y\n"
    )
    with pytest.raises(ValueError, match="duplicate GPU_MPI_RANK_UUID"):
        parse_rank_uuid_lines(output)


def test_build_ownership_line_passes_bijective_mapping():
    line = build_ownership_line(_mock_entries(8), 8)
    assert "ranks=8 nodes=1 status=PASS" in line
    assert "0:GPU-mock-0000" in line and "7:GPU-mock-0007" in line


def test_build_ownership_line_passes_two_node_mapping():
    entries = _mock_entries(16, hosts=("gpu-node-0", "gpu-node-1"))
    line = build_ownership_line(entries, 16, expected_nodes=2)
    assert "ranks=16 nodes=2 status=PASS" in line


def test_build_ownership_line_flags_duplicate_uuid():
    entries = _mock_entries(8)
    entries[7]["uuid"] = entries[3]["uuid"]
    line = build_ownership_line(entries, 8)
    assert "status=FAIL" in line
    assert "duplicate_uuid:GPU-mock-0003@rank3+7" in line


def test_build_ownership_line_flags_missing_rank():
    entries = _mock_entries(8)
    del entries[5]
    line = build_ownership_line(entries, 8)
    assert "status=FAIL" in line
    assert "missing_ranks:5" in line


def test_build_ownership_line_flags_node_shortfall():
    line = build_ownership_line(_mock_entries(16), 16, expected_nodes=2)
    assert "status=FAIL" in line
    assert "node_count:1!=expected:2" in line


def _free_port() -> str:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return str(sock.getsockname()[1])


def _mock_probe_output(ranks: int, *, hosts: tuple[str, ...] = ("gpu-node-0",)) -> str:
    lines = [
        f"GPU_MPI_RANK_UUID rank={rank} local_rank={rank % 8}"
        f" host={hosts[rank % len(hosts)]} device=cuda:{rank % 8}"
        f" uuid=GPU-mock-{rank:04d}"
        for rank in range(ranks)
    ]
    uuid_field = ",".join(f"{rank}:GPU-mock-{rank:04d}" for rank in range(ranks))
    lines.append(
        f"GPU_MPI_OWNERSHIP ranks={ranks} nodes={len(hosts)}"
        f" status=PASS uuids={uuid_field}"
    )
    return "\n".join(lines)


def _verify_ownership_output(output: str, ranks: int, expected_nodes: int) -> str:
    """Independently re-parse probe output and re-check the ownership rules."""
    entries = parse_rank_uuid_lines(output)
    verdict = build_ownership_line(entries, ranks, expected_nodes)
    print(verdict)
    assert "status=PASS" in verdict, verdict
    probe_verdicts = OWNERSHIP_RE.findall(output)
    assert probe_verdicts, "rank 0 did not print a GPU_MPI_OWNERSHIP verdict"
    assert all(status == "PASS" for _, _, status, _ in probe_verdicts), output[-4000:]
    return verdict


def test_verify_ownership_output_accepts_mocked_two_node_probe():
    output = _mock_probe_output(16, hosts=("gpu-node-0", "gpu-node-1"))
    verdict = _verify_ownership_output(output, 16, 2)
    assert "ranks=16 nodes=2 status=PASS" in verdict


def test_verify_ownership_output_rejects_duplicate_uuid():
    output = _mock_probe_output(8).replace("GPU-mock-0007", "GPU-mock-0003")
    with pytest.raises(AssertionError):
        _verify_ownership_output(output, 8, 1)


def test_verify_ownership_output_rejects_missing_probe_verdict():
    output = "\n".join(
        _mock_probe_output(8).splitlines()[:-1]  # drop the OWNERSHIP line
    )
    with pytest.raises(AssertionError, match="did not print"):
        _verify_ownership_output(output, 8, 1)


def _expected_nodes(ranks: int, gpus_per_node: int) -> int:
    configured = int(os.environ.get("SPINGNN_MPI_UUID_EXPECT_NODES", "0"))
    if configured > 0:
        return configured
    return max(1, -(-ranks // gpus_per_node))


def test_gpu_mpi_uuid_ownership_replay():
    """Re-verify the output from an external MPI launcher."""
    replay = os.environ.get("SPINGNN_MPI_UUID_REPLAY")
    if not replay:
        pytest.skip("Set SPINGNN_MPI_UUID_REPLAY to the captured probe output")
    ranks = int(os.environ.get("SPINGNN_MPI_UUID_RANKS", "0"))
    if ranks < 1:
        pytest.skip("SPINGNN_MPI_UUID_RANKS names the MPI world size")
    gpus_per_node = int(
        os.environ.get("SPINGNN_MPI_UUID_GPUS_PER_NODE", str(min(ranks, 8)))
    )
    output = Path(replay).read_text()
    _verify_ownership_output(output, ranks, _expected_nodes(ranks, gpus_per_node))


@pytest.mark.gpu
@pytest.mark.mpi
def test_gpu_mpi_uuid_ownership_is_bijective(mpi_prefix, lammps_env):
    if os.environ.get("SPINGNN_MPI_UUID_REPLAY"):
        pytest.skip("replay mode: probe output was captured by the job launcher")
    ranks = int(os.environ.get("SPINGNN_MPI_UUID_RANKS", "0"))
    if ranks < 1:
        pytest.skip("Set SPINGNN_MPI_UUID_RANKS to the MPI world size")
    gpus_per_node = int(
        os.environ.get("SPINGNN_MPI_UUID_GPUS_PER_NODE", str(min(ranks, 8)))
    )
    visible = torch.cuda.device_count()
    if visible < gpus_per_node:
        pytest.skip(
            f"UUID ownership needs {gpus_per_node} visible GPUs per node, "
            f"found {visible}"
        )
    expected_nodes = _expected_nodes(ranks, gpus_per_node)

    env = dict(lammps_env)
    env["MASTER_ADDR"] = (
        os.environ.get("SPINGNN_MPI_UUID_MASTER_ADDR") or socket.gethostname()
    )
    env["MASTER_PORT"] = os.environ.get("SPINGNN_MPI_UUID_MASTER_PORT") or _free_port()
    env["SPINGNN_MPI_UUID_EXPECT_NODES"] = str(expected_nodes)

    # cwd must exist on every allocated node (remote ranks chdir into it);
    # the repo root is on the shared filesystem, pytest tmp_path is not.
    command = [*mpi_prefix(ranks), sys.executable, str(PROBE_SCRIPT)]
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=PROBE_TIMEOUT_SECONDS,
        check=False,
    )
    print(completed.stdout)
    assert completed.returncode == 0, completed.stdout[-4000:]
    _verify_ownership_output(completed.stdout, ranks, expected_nodes)
