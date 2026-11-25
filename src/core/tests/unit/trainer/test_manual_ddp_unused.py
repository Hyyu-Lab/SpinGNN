"""Manual reduction must agree on parameter ownership across ranks."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nequip.train.trainer import Trainer


def _rank_dependent_gradient_worker(rank, rendezvous, dtype, empty_rank):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(2, dtype=dtype)) for _ in range(4)]
        )
        # Opposite local heads have the same shape: a mismatched collective
        # sequence can silently mix them instead of merely timing out.
        if not empty_rank or rank == 0:
            loss = model[rank].sum() + 3 * model[2].sum()
            loss.backward()
        trainer = SimpleNamespace(
            model=model,
            torch_device=torch.device("cpu"),
            _ddp_enabled=True,
            _ddp_impl="manual",
            _ddp_world_size=2,
        )
        Trainer._ddp_reduce_gradients(trainer)
        for index, parameter in enumerate(model[:2]):
            if empty_rank and index == 1:
                assert parameter.grad is None
                continue
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.5))
        torch.testing.assert_close(
            model[2].grad, torch.full_like(model[2], 1.5 if empty_rank else 3)
        )
        assert model[3].grad is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("empty_rank", [False, True])
def test_manual_ddp_reduces_rank_dependent_unused_parameters(tmp_path, empty_rank):
    mp.spawn(
        _rank_dependent_gradient_worker,
        args=(
            (tmp_path / "rendezvous").as_uri(),
            torch.get_default_dtype(),
            empty_rank,
        ),
        nprocs=2,
        join=True,
    )
