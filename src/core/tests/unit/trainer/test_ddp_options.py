"""Two-process checks for configurable DDP reducer options."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_WORLD_SIZE = 2


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.register_buffer("offset", torch.ones(2))

    def forward(self, value):
        return self.linear(value) + self.offset


def _ddp_options_worker(rank, world_size, init_file, result_file):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        options = [
            {},
            {"broadcast_buffers": False},
            {"static_graph": True},
            {"gradient_as_bucket_view": True},
            {"find_unused_parameters": False},
            {
                "broadcast_buffers": False,
                "static_graph": True,
                "gradient_as_bucket_view": True,
                "find_unused_parameters": False,
            },
        ]
        reference = None
        for option_index, option in enumerate(options):
            torch.manual_seed(23)
            model = _Net()
            ddp = torch.nn.parallel.DistributedDataParallel(model, **option)
            optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)
            snapshots = []
            for step in range(4):
                value = torch.full((3, 3), float(rank + step + 1))
                optimizer.zero_grad(set_to_none=True)
                ddp(value).square().mean().backward()
                gradients = [p.grad.detach().clone() for p in ddp.parameters()]
                optimizer.step()
                snapshots.append(
                    (
                        [g.cpu() for g in gradients],
                        [p.detach().cpu().clone() for p in model.parameters()],
                    )
                )
            if reference is None:
                reference = snapshots
            else:
                for baseline, result in zip(reference, snapshots):
                    for expected, actual in zip(
                        baseline[0] + baseline[1], result[0] + result[1]
                    ):
                        assert torch.equal(expected, actual)
        if rank == 0:
            torch.save({"option_count": len(options)}, result_file)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="torch distributed gloo backend unavailable",
)
def test_ddp_options_preserve_gradients_and_parameters(tmp_path):
    context = mp.start_processes(
        _ddp_options_worker,
        args=(
            _WORLD_SIZE,
            str(tmp_path / "init"),
            str(tmp_path / "result.pth"),
        ),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    for process in context.processes:
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("gloo worker hung; terminated after timeout")
    assert [process.exitcode for process in context.processes] == [0, 0]
    assert torch.load(tmp_path / "result.pth", weights_only=True)["option_count"] == 6
