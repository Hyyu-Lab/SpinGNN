"""Independent compiled heads preserve native DDP optimizer behavior."""

import copy
from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from nequip.train.trainer import warmup_process_group_for_ddp
from nequip.utils._compile_grad_partitions import compile_gradient_partitions
from nequip.utils.test import set_irreps_debug


class _OptionalOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor(0.7))
        self.head = torch.nn.Parameter(torch.tensor(1.2))

    def forward(self, x):
        feature = self.shared * x.square()
        return feature, feature.sin() * self.head


def _compiled_optional_ddp_worker(
    rank, rendezvous, dtype, device_type="cpu", cache_root=None
):
    torch.set_num_threads(1)
    torch.set_default_dtype(dtype)
    set_irreps_debug(False)
    device = (
        torch.device(device_type, rank)
        if device_type == "cuda"
        else torch.device("cpu")
    )
    backend = "nccl" if device_type == "cuda" else "gloo"
    compiler_backend = "inductor" if device_type == "cuda" else "aot_eager"
    if device_type == "cuda":
        rank_cache = Path(cache_root) / f"rank-{rank}"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(rank_cache / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(rank_cache / "triton")
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        torch.cuda.set_device(rank)
        from torch._inductor.runtime.cache_dir_utils import cache_dir

        assert Path(cache_dir()) == rank_cache / "inductor"
    dist.init_process_group(
        backend,
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=600 if device_type == "cuda" else 120),
    )
    try:
        warmup_process_group_for_ddp()
        reference = _OptionalOutput().to(device)
        candidate = copy.deepcopy(reference)
        for expected, actual in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        compiled = compile_gradient_partitions(
            torch.fx.symbolic_trace(candidate),
            lambda module: torch.compile(
                module, backend=compiler_backend, fullgraph=True
            ),
        )
        ddp_options = (
            {"device_ids": [rank], "output_device": rank}
            if device_type == "cuda"
            else {}
        )
        models = [
            DistributedDataParallel(model, find_unused_parameters=True, **ddp_options)
            for model in (reference, compiled)
        ]
        optimizers = [
            torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
            for model in (reference, candidate)
        ]
        x = torch.tensor([0.3 + rank * 0.1, 0.8], device=device)
        phases = (
            "unused",
            "supervised",
            "rank_subset",
            "explicit_zero",
            "unused_again",
            "zero_rank_subset",
        )
        for phase in phases:
            for model, optimizer in zip(models, optimizers):
                optimizer.zero_grad(set_to_none=True)
                outputs = model(x)
                loss = outputs[0].square().sum()
                if phase == "supervised" or (phase == "rank_subset" and rank == 0):
                    loss = loss + outputs[1].square().sum()
                elif phase == "explicit_zero" or (
                    phase == "zero_rank_subset" and rank == 0
                ):
                    loss = loss + 0 * outputs[1].sum()
                loss.backward()
                optimizer.step()
            for expected, actual in zip(reference.parameters(), candidate.parameters()):
                assert (expected.grad is None) == (actual.grad is None)
                torch.testing.assert_close(actual, expected)
                if expected.grad is not None:
                    torch.testing.assert_close(actual.grad, expected.grad)
                expected_state = optimizers[0].state.get(expected, {})
                actual_state = optimizers[1].state.get(actual, {})
                assert actual_state.keys() == expected_state.keys()
                for key in expected_state:
                    torch.testing.assert_close(actual_state[key], expected_state[key])
            assert len(optimizers[0].state) == len(optimizers[1].state)
            print(
                "DDP_COMPILE_PARITY "
                + json.dumps(
                    {
                        "rank": rank,
                        "device": str(device),
                        "backend": backend,
                        "compiler": compiler_backend,
                        "dtype": str(dtype),
                        "phase": phase,
                        "head_gradient_present": candidate.head.grad is not None,
                        "head_optimizer_step": float(
                            optimizers[1].state.get(candidate.head, {}).get("step", 0)
                        ),
                        "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                        "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
                    }
                ),
                flush=True,
            )
        if device_type == "cuda":
            assert torch.cuda.current_device() == rank
    finally:
        dist.destroy_process_group()


def test_compiled_optional_head_matches_native_ddp(tmp_path):
    mp.spawn(
        _compiled_optional_ddp_worker,
        args=((tmp_path / "rendezvous").as_uri(), torch.get_default_dtype()),
        nprocs=2,
        join=True,
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_compiled_optional_head_matches_nccl_inductor(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("NCCL/Inductor parity requires two allocated CUDA devices")
    assert dist.is_nccl_available(), "the two-GPU gate requires an NCCL-enabled PyTorch"
    mp.spawn(
        _compiled_optional_ddp_worker,
        args=(
            (tmp_path / "rendezvous").as_uri(),
            torch.get_default_dtype(),
            "cuda",
            str(tmp_path / "cache"),
        ),
        nprocs=2,
        join=True,
    )
