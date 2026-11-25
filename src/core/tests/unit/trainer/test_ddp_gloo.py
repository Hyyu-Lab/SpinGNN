"""Real two-process gloo test for native DDP construction and one step.

Locks the torch 2.11/cu128 crash workaround end to end on CPU: the
production ``warmup_process_group_for_ddp`` helper runs before
``DistributedDataParallel`` construction, then one SGD step must keep both
ranks bitwise consistent.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from nequip.train.trainer import warmup_process_group_for_ddp

from ._multiprocessing import join_workers

_WORLD_SIZE = 2
_JOIN_TIMEOUT = 60.0


def _gloo_ddp_worker(rank, world_size, init_file):
    torch.manual_seed(1234)  # identical initial parameters on every rank
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        warmup_process_group_for_ddp()
        model = DistributedDataParallel(torch.nn.Linear(4, 3))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = model(torch.ones(2, 4)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss).all(), f"rank {rank}: non-finite loss"
        local = torch.cat([p.detach().flatten() for p in model.parameters()])
        assert torch.isfinite(local).all(), f"rank {rank}: non-finite parameters"
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        for other in gathered:
            assert torch.equal(local, other), f"rank {rank}: parameters diverged"
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch distributed gloo backend unavailable",
)
def test_native_ddp_gloo_two_process_step(tmp_path):
    init_file = tmp_path / "gloo_init_store"
    context = mp.start_processes(
        _gloo_ddp_worker,
        args=(_WORLD_SIZE, str(init_file)),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    join_workers(context, nprocs=_WORLD_SIZE, timeout=_JOIN_TIMEOUT)


class _UnusedBranchNet(torch.nn.Module):
    """One branch that never contributes to the loss (batch-varying usage)."""

    def __init__(self):
        super().__init__()
        self.used = torch.nn.Linear(4, 3)
        self.unused = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.used(x)


def _gloo_ddp_unused_worker(rank, world_size, init_file):
    torch.manual_seed(1234)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        warmup_process_group_for_ddp()
        # mirrors the trainer's default (SPINGNN_DDP_FIND_UNUSED!=0)
        model = DistributedDataParallel(_UnusedBranchNet(), find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(3):
            loss = model(torch.ones(2, 4)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        assert torch.isfinite(loss).all(), f"rank {rank}: non-finite loss"
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch distributed gloo backend unavailable",
)
def test_native_ddp_gloo_unused_parameters_two_steps(tmp_path):
    """Second-iteration reducer failure (smoke 2290758) must not recur."""
    init_file = tmp_path / "gloo_init_store_unused"
    context = mp.start_processes(
        _gloo_ddp_unused_worker,
        args=(_WORLD_SIZE, str(init_file)),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    join_workers(context, nprocs=_WORLD_SIZE, timeout=_JOIN_TIMEOUT)
