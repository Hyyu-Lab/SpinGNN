#!/usr/bin/env python3
"""Bind one MPI local rank to one physical CUDA device before launching LAMMPS."""

import ctypes as ct
import os
import socket
import sys
import uuid


def device_uuid(local_rank):
    driver = ct.CDLL("libcuda.so.1")
    device = ct.c_int()
    count = ct.c_int()
    ident = (ct.c_ubyte * 16)()
    for routine, args in (
        (driver.cuInit, (0,)),
        (driver.cuDeviceGetCount, (ct.byref(count),)),
    ):
        status = routine(*args)
        if status:
            raise RuntimeError(f"{routine.__name__} returned CUDA status {status}")
    # A scheduler may have masked this process to its single assigned GPU.
    ordinal = 0 if count.value == 1 else local_rank
    if driver.cuDeviceGet(ct.byref(device), ordinal) or driver.cuDeviceGetUuid(
        ct.byref(ident), device
    ):
        raise RuntimeError(
            f"Cannot select local rank {local_rank} from {count.value} visible CUDA devices"
        )
    return "GPU-" + str(uuid.UUID(bytes=bytes(ident)))


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Supply a LAMMPS command to execute")
    # Intel Hydra may propagate the launcher's SLURM_LOCALID to all children.
    # The local rank must be resolved before MPI or Kokkos creates CUDA contexts.
    local = next(
        (
            int(os.environ[key])
            for key in (
                "MPI_LOCALRANKID",
                "OMPI_COMM_WORLD_LOCAL_RANK",
                "PMI_LOCAL_RANK",
                "SLURM_LOCALID",
            )
            if key in os.environ
        ),
        0,
    )
    ident = device_uuid(local)
    rank = os.getenv(
        "PMI_RANK", os.getenv("OMPI_COMM_WORLD_RANK", os.getenv("SLURM_PROCID", "0"))
    )
    print(
        f"SPINGNN2_GPU rank={rank} local={local} host={socket.gethostname()} uuid={ident}",
        flush=True,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ident
    # Only the selected physical GPU is visible to the new process.
    os.environ["PMI_LOCAL_RANK"] = "0"
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
