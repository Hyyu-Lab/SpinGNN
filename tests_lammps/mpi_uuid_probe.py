"""Rank↔GPU UUID ownership probe for the multi-GPU MPI acceptance lane (R5).

Runs standalone under ``mpirun`` with one process per rank: each rank pins
its node-local GPU, reports the device UUID, and rank 0 gathers every report
and prints the machine-readable verdict::

    GPU_MPI_OWNERSHIP ranks=N nodes=M status=PASS uuids=0:GPU-…,1:GPU-…,…

``test_gpu_mpi_uuid_ownership.py`` reuses the pure parsing/verdict helpers
below, so the bijection logic is unit-testable on CPU with mocked rank data.
Torch and CUDA are imported lazily inside ``main`` so this module stays
importable without a GPU.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from datetime import timedelta
from typing import Mapping

RANK_UUID_RE = re.compile(
    r"^GPU_MPI_RANK_UUID rank=(\d+) local_rank=(\d+) host=(\S+)"
    r" device=(\S+) uuid=(\S+)\s*$",
    re.MULTILINE,
)
OWNERSHIP_RE = re.compile(
    r"^GPU_MPI_OWNERSHIP ranks=(\d+) nodes=(\d+) status=(\w+) uuids=(\S+)",
    re.MULTILINE,
)

# Rank environment variables per launcher, in lookup order: OpenMPI, Intel
# MPI (hydra), MVAPICH2, Slurm/srun, torchrun.
_RANK_NAMES = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "MV2_COMM_WORLD_RANK",
    "SLURM_PROCID",
    "RANK",
)
_SIZE_NAMES = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_NTASKS",
    "WORLD_SIZE",
)
_LOCAL_RANK_NAMES = (
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "SLURM_LOCALID",
    "LOCAL_RANK",
)


def parse_rank_uuid_lines(output: str) -> dict[int, dict[str, str]]:
    """Parse ``GPU_MPI_RANK_UUID`` lines keyed by rank.

    A second line for the same rank means the launcher started duplicate
    processes; that corrupts the ownership evidence, so it is an error.
    """
    entries: dict[int, dict[str, str]] = {}
    for match in RANK_UUID_RE.finditer(output):
        rank = int(match.group(1))
        if rank in entries:
            raise ValueError(f"duplicate GPU_MPI_RANK_UUID line for rank {rank}")
        entries[rank] = {
            "local_rank": match.group(2),
            "host": match.group(3),
            "device": match.group(4),
            "uuid": match.group(5),
        }
    return entries


def build_ownership_line(
    entries: Mapping[int, Mapping[str, str]],
    expected_ranks: int,
    expected_nodes: int | None = None,
) -> str:
    """Build the rank↔UUID verdict line.

    ``status=PASS`` requires exactly the ranks ``0..expected_ranks-1``, a
    bijective rank↔UUID mapping (no two ranks on the same physical GPU), and
    — when ``expected_nodes`` is given — exactly that many distinct hosts.
    Failures are reported as ``status=FAIL`` with machine-readable reasons.
    """
    errors: list[str] = []
    ranks = sorted(entries)
    missing = sorted(set(range(expected_ranks)) - set(ranks))
    extra = sorted(set(ranks) - set(range(expected_ranks)))
    if missing:
        errors.append("missing_ranks:" + ",".join(str(rank) for rank in missing))
    if extra:
        errors.append("extra_ranks:" + ",".join(str(rank) for rank in extra))
    by_uuid: dict[str, list[int]] = {}
    for rank in ranks:
        by_uuid.setdefault(entries[rank]["uuid"], []).append(rank)
    for uuid, holders in sorted(by_uuid.items()):
        if len(holders) > 1:
            errors.append(
                f"duplicate_uuid:{uuid}@rank" + "+".join(str(r) for r in holders)
            )
    hosts = sorted({entries[rank]["host"] for rank in ranks})
    if expected_nodes is not None and len(hosts) != expected_nodes:
        errors.append(f"node_count:{len(hosts)}!=expected:{expected_nodes}")
    status = "PASS" if not errors else "FAIL"
    uuid_field = ",".join(f"{rank}:{entries[rank]['uuid']}" for rank in ranks)
    line = (
        f"GPU_MPI_OWNERSHIP ranks={expected_ranks} nodes={len(hosts)}"
        f" status={status} uuids={uuid_field}"
    )
    if errors:
        line += " errors=" + ";".join(errors)
    return line


def _env_int(names: tuple[str, ...]) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return None


def _probe_error(reason: str, **fields: object) -> int:
    extras = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"GPU_MPI_PROBE_ERROR reason={reason} {extras}".rstrip(), flush=True)
    return 2


def _device_uuid(local_rank: int) -> str:
    import torch

    uuid = getattr(torch.cuda.get_device_properties(local_rank), "uuid", None)
    if uuid is not None:
        return str(uuid)
    # Older torch without device-properties UUID: ask the driver, mapping the
    # node-local index back through CUDA_VISIBLE_DEVICES when Slurm set it.
    index: int | str = local_rank
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        index = visible.split(",")[local_rank]
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader",
            "-i",
            str(index),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=True,
    )
    return query.stdout.strip()


def _gather(rank: int, size: int, payload: tuple[int, int, str, str]):
    """All-gather ``(rank, local_rank, host, uuid)`` over the gloo backend."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_gloo_available()):
        return None
    if not os.environ.get("MASTER_ADDR") or not os.environ.get("MASTER_PORT"):
        return None
    dist.init_process_group(
        "gloo",
        init_method="env://",
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=180),
    )
    try:
        gathered: list = [None] * size
        dist.all_gather_object(gathered, payload)
        dist.barrier()
        return gathered
    finally:
        dist.destroy_process_group()


def main() -> int:
    rank = _env_int(_RANK_NAMES)
    size = _env_int(_SIZE_NAMES)
    local_rank = _env_int(_LOCAL_RANK_NAMES)
    if rank is None or size is None or local_rank is None:
        return _probe_error(
            "rank_env_missing", rank=rank, size=size, local_rank=local_rank
        )

    import torch

    if not torch.cuda.is_available():
        return _probe_error("cuda_unavailable", rank=rank)
    device_count = torch.cuda.device_count()
    if not 0 <= local_rank < device_count:
        return _probe_error(
            "local_rank_out_of_range",
            rank=rank,
            local_rank=local_rank,
            device_count=device_count,
        )

    torch.cuda.set_device(local_rank)
    uuid = _device_uuid(local_rank)
    host = socket.gethostname()
    print(
        f"GPU_MPI_RANK_UUID rank={rank} local_rank={local_rank} host={host}"
        f" device=cuda:{local_rank} uuid={uuid}",
        flush=True,
    )

    gathered = _gather(rank, size, (rank, local_rank, host, uuid))
    if rank == 0:
        if gathered is None:
            print(
                f"GPU_MPI_OWNERSHIP ranks={size} nodes=0 status=FAIL"
                " uuids=NONE errors=gather_unavailable",
                flush=True,
            )
            return 1
        entries = {
            entry[0]: {
                "local_rank": str(entry[1]),
                "host": entry[2],
                "device": f"cuda:{entry[1]}",
                "uuid": entry[3],
            }
            for entry in gathered
        }
        expected_nodes = int(os.environ.get("SPINGNN_MPI_UUID_EXPECT_NODES", "0"))
        print(
            build_ownership_line(entries, size, expected_nodes or None),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
