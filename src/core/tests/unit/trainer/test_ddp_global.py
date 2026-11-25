"""Behavioral two-process gloo tests for the global DDP training contract.

Covers the research/ddp-contract items 2, 3, 4 and 7: unequal-count loss
objective and gradients, additive metric aggregation, nonpadding validation
slicing with empty ranks, and invocation-scoped dataset markers. Workers
propagate exceptions to pytest and every test is timeout-guarded like
test_ddp_gloo.py.
"""

import math
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ._multiprocessing import join_workers

_WORLD_SIZE = 2
_JOIN_TIMEOUT = 60.0

_GLOO_AVAILABLE = dist.is_available() and dist.is_gloo_available()
_SKIP_REASON = "torch distributed gloo backend unavailable"


def _init_gloo(rank, world_size, init_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


# ---------------------------------------------------------------------------
# loss: unequal counts, gradients, optimizer update
# ---------------------------------------------------------------------------

# rank 0: 1 frame with 2 atoms; rank 1: 2 frames with 2 and 3 atoms
_RANK_DATA = [
    dict(
        x=[[1.0, 0.0, -1.0], [0.5, 0.25, 1.0]],
        frames=[0, 0],
        ref_f=[[0.3], [-0.2]],
        ref_e=[[1.0]],
    ),
    dict(
        x=[
            [0.1, 0.2, 0.3],
            [-0.4, 0.5, -0.6],
            [0.7, -0.8, 0.9],
            [1.1, 0.0, -0.1],
            [0.2, -0.3, 0.4],
        ],
        frames=[0, 0, 1, 1, 1],
        ref_f=[[0.0], [0.1], [-0.1], [0.2], [-0.3]],
        ref_e=[[-0.5], [0.8]],
    ),
]

_LOSS_COEFFS = {"f": (2.0, "MSELoss"), "e": (1.0, "L1Loss")}


def _forward(model, x, frames, n_frames):
    pred_f = model(x)
    pred_e = torch.zeros(n_frames, 1, dtype=pred_f.dtype)
    pred_e.index_add_(0, frames, pred_f)
    return {"f": pred_f, "e": pred_e}


def _reference_run():
    """Concatenated single-process reference for the unequal-count loss."""
    from nequip.train import Loss

    torch.manual_seed(7)
    model = torch.nn.Linear(3, 1)
    x = torch.cat([torch.as_tensor(d["x"]) for d in _RANK_DATA])
    frames = torch.as_tensor(
        _RANK_DATA[0]["frames"] + [f + 1 for f in _RANK_DATA[1]["frames"]]
    )
    ref = {
        "f": torch.cat([torch.as_tensor(d["ref_f"]) for d in _RANK_DATA]),
        "e": torch.cat([torch.as_tensor(d["ref_e"]) for d in _RANK_DATA]),
    }
    loss_inst = Loss(coeffs=_LOSS_COEFFS)
    pred = _forward(model, x, frames, 3)
    objective, contrib = loss_inst(pred, ref)
    objective.backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    params = [p.detach().clone() for p in model.parameters()]
    return objective.detach(), contrib, grads, params


def _loss_worker(rank, world_size, init_file):
    from torch.nn.parallel import DistributedDataParallel

    from nequip.train import Loss
    from nequip.train.loss import DistributedLossStat, distributed_objective
    from nequip.train.trainer import warmup_process_group_for_ddp

    torch.set_default_dtype(torch.float64)
    _init_gloo(rank, world_size, init_file)
    try:
        warmup_process_group_for_ddp()
        torch.manual_seed(7)  # identical initial parameters on every rank
        model = torch.nn.Linear(3, 1)

        data = _RANK_DATA[rank]
        x = torch.as_tensor(data["x"])
        frames = torch.as_tensor(data["frames"])
        n_frames = int(frames.max()) + 1
        ref = {
            "f": torch.as_tensor(data["ref_f"]),
            "e": torch.as_tensor(data["ref_e"]),
        }

        ref_objective, ref_contrib, ref_grads, ref_params = _reference_run()

        # --- native DDP path ---
        ddp = DistributedDataParallel(model)
        loss_inst = Loss(coeffs=_LOSS_COEFFS)
        pred = _forward(ddp, x, frames, n_frames)
        objective, numerators, counts = distributed_objective(
            loss_inst, pred, ref, world_size
        )
        objective.backward()

        for param, ref_grad in zip(model.parameters(), ref_grads):
            assert param.grad is not None, f"rank {rank}: missing gradient"
            torch.testing.assert_close(param.grad, ref_grad)

        # component losses and total via additive stats aggregation
        stat = DistributedLossStat(loss_inst)
        stat.accumulate(numerators, counts)
        finalized = stat.global_result()
        torch.testing.assert_close(
            torch.as_tensor(finalized["loss_f"]), ref_contrib["f"].detach()
        )
        torch.testing.assert_close(
            torch.as_tensor(finalized["loss_e"]), ref_contrib["e"].detach()
        )
        torch.testing.assert_close(torch.as_tensor(finalized["loss"]), ref_objective)

        # one optimizer update matches the reference update
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1)
        optimizer.step()
        for param, ref_param in zip(model.parameters(), ref_params):
            torch.testing.assert_close(param.detach(), ref_param)
        local = torch.cat([p.detach().flatten() for p in model.parameters()])
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        for other in gathered:
            assert torch.equal(local, other), f"rank {rank}: parameters diverged"

        # --- manual reduction path: same scaling, explicit average ---
        for param in model.parameters():
            param.grad = None
        pred = _forward(model, x, frames, n_frames)
        objective, _, _ = distributed_objective(loss_inst, pred, ref, world_size)
        objective.backward()
        # reference gradients for the *updated* parameters: recompute from a
        # fresh copy of the post-step state
        torch.manual_seed(7)
        ref_model = torch.nn.Linear(3, 1)
        with torch.no_grad():
            for p_new, p_old in zip(ref_model.parameters(), ref_params):
                p_new.copy_(p_old)
        x_full = torch.cat([torch.as_tensor(d["x"]) for d in _RANK_DATA])
        frames_full = torch.as_tensor(
            _RANK_DATA[0]["frames"] + [f + 1 for f in _RANK_DATA[1]["frames"]]
        )
        ref_full = {
            "f": torch.cat([torch.as_tensor(d["ref_f"]) for d in _RANK_DATA]),
            "e": torch.cat([torch.as_tensor(d["ref_e"]) for d in _RANK_DATA]),
        }
        ref_loss_inst = Loss(coeffs=_LOSS_COEFFS)
        ref_obj, _ = ref_loss_inst(
            _forward(ref_model, x_full, frames_full, 3), ref_full
        )
        ref_obj.backward()
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(param.grad, ref_param.grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason=_SKIP_REASON)
def test_ddp_loss_unequal_counts_match_reference(tmp_path):
    context = mp.start_processes(
        _loss_worker,
        args=(_WORLD_SIZE, str(tmp_path / "gloo_init")),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    join_workers(context, nprocs=_WORLD_SIZE, timeout=_JOIN_TIMEOUT)


# ---------------------------------------------------------------------------
# loss: zero-valid-label rank, globally-zero target, unsupported reduction
# ---------------------------------------------------------------------------


def _loss_edge_worker(rank, world_size, init_file):
    from nequip.train import Loss
    from nequip.train.loss import distributed_objective

    torch.set_default_dtype(torch.float64)
    _init_gloo(rank, world_size, init_file)
    try:
        # rank 1 has zero valid labels (all NaN); it must contribute a
        # differentiable zero and still run the same collectives
        torch.manual_seed(11)
        model = torch.nn.Linear(3, 1)
        x = torch.ones(2, 3)
        ref = torch.tensor([[0.5], [-0.5]])
        if rank == 1:
            ref[:] = float("nan")
        loss_inst = Loss(coeffs={"f": (1.0, "MSELoss", {"ignore_nan": True})})
        pred = {"f": model(x)}
        objective, numerators, counts = distributed_objective(
            loss_inst, pred, {"f": ref}, world_size
        )
        objective.backward()
        assert torch.isfinite(model.weight.grad).all()
        if rank == 0:
            expected = torch.mean((pred["f"] - ref) ** 2) * world_size
            torch.testing.assert_close(objective.detach(), expected.detach())
        else:
            assert counts["f"].item() == 0
            torch.testing.assert_close(
                objective.detach(), torch.zeros((), dtype=torch.float64)
            )
        # the gradient equals the single-process gradient on rank 0 data alone
        torch.manual_seed(11)
        ref_model = torch.nn.Linear(3, 1)
        ref_pred = ref_model(torch.ones(2, 3))
        ref_loss = torch.mean((ref_pred - torch.tensor([[0.5], [-0.5]])) ** 2)
        ref_loss.backward()
        # DDP-style averaging: sum of local grads / world size
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            grad = param.grad.clone()
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad.div_(world_size)
            torch.testing.assert_close(grad, ref_param.grad)

        # globally-zero required target: descriptive error on every rank
        bad_ref = torch.full((2, 1), float("nan"))
        with pytest.raises(RuntimeError, match="globally zero valid-element count"):
            distributed_objective(
                loss_inst, {"f": model(x)}, {"f": bad_ref}, world_size
            )

        # unsupported distributed reduction config: explicit error
        species_loss = Loss(coeffs={"f": (1.0, "PerSpeciesMSELoss")})
        with pytest.raises(NotImplementedError, match="additive"):
            species_loss.assert_distributed_compatible()
        custom_loss = Loss(
            coeffs={"f": (1.0, lambda pred, ref, key, mean: pred[key].sum())}
        )
        with pytest.raises(NotImplementedError, match="additive numerator/count"):
            custom_loss.assert_distributed_compatible()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason=_SKIP_REASON)
def test_ddp_loss_zero_valid_and_unsupported(tmp_path):
    context = mp.start_processes(
        _loss_edge_worker,
        args=(_WORLD_SIZE, str(tmp_path / "gloo_init")),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    join_workers(context, nprocs=_WORLD_SIZE, timeout=_JOIN_TIMEOUT)


# ---------------------------------------------------------------------------
# per-frame loss: frame-sum gradient semantics, masked term keeps the graph
# ---------------------------------------------------------------------------

# atoms per local frame and per-frame mask of the masked term, for each rank
_PER_FRAME_SCENARIOS = [
    # one frame per rank: coefficient = world_size
    dict(atoms=[[2], [5]], keep=[[True], [True]], coeff=2.0),
    # ragged local batches: coefficient = global frames per optimizer step
    dict(atoms=[[2], [3, 4]], keep=[[True], [False, True]], coeff=3.0),
    # rank 1 selects no frame for the masked term
    dict(atoms=[[3], [2]], keep=[[True], [False]], coeff=2.0),
]


class _TwoHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.baseline = torch.nn.Linear(3, 9)
        self.masked = torch.nn.Linear(3, 9)

    def forward(self, x):
        return {"baseline": self.baseline(x), "masked": self.masked(x)}


def _per_frame_rank_data(scenario_index, rank):
    atoms = _PER_FRAME_SCENARIOS[scenario_index]["atoms"][rank]
    generator = torch.Generator().manual_seed(100 * scenario_index + rank)
    x = torch.randn(sum(atoms), 3, generator=generator)
    target = torch.randn(sum(atoms), 9, generator=generator)
    batch = torch.repeat_interleave(torch.arange(len(atoms)), torch.as_tensor(atoms))
    return x, target, batch


def _per_frame_reference_grads(scenario_index, world_size):
    """Sum over all frames of the gradient of each frame's own mean |error|."""
    scenario = _PER_FRAME_SCENARIOS[scenario_index]
    torch.manual_seed(5)
    model = _TwoHeads()
    total = 0.0
    for rank in range(world_size):
        x, target, batch = _per_frame_rank_data(scenario_index, rank)
        out = model(x)
        for frame, keep in enumerate(scenario["keep"][rank]):
            sel = batch == frame
            total = total + (out["baseline"][sel] - target[sel]).abs().mean()
            if keep:
                total = total + (out["masked"][sel] - target[sel]).abs().mean()
    total.backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def _per_frame_loss_worker(rank, world_size, init_file):
    from torch.nn.parallel import DistributedDataParallel

    from nequip.train import Loss
    from nequip.train.loss import distributed_objective
    from nequip.train.trainer import warmup_process_group_for_ddp

    torch.set_default_dtype(torch.float64)
    _init_gloo(rank, world_size, init_file)
    try:
        warmup_process_group_for_ddp()
        for index, scenario in enumerate(_PER_FRAME_SCENARIOS):
            coeff = scenario["coeff"]
            loss_inst = Loss(
                coeffs={
                    "baseline": [coeff, "PerFrameL1Loss", {"ref_key": "target"}],
                    "masked": [
                        coeff,
                        "PerFrameL1Loss",
                        {"ref_key": "target", "mask_field": "keep"},
                    ],
                }
            )
            loss_inst.assert_distributed_compatible()
            ref_grads = _per_frame_reference_grads(index, world_size)

            torch.manual_seed(5)  # identical initial parameters on every rank
            model = _TwoHeads()
            # every parameter must receive a gradient on every rank, also when
            # the masked term selects nothing locally
            ddp = DistributedDataParallel(model, find_unused_parameters=False)
            x, target, batch = _per_frame_rank_data(index, rank)
            ref = {"target": target, "batch": batch}
            for _ in range(2):  # a second step checks DDP's reduction bookkeeping
                model.zero_grad(set_to_none=True)
                pred = ddp(x)
                pred["keep"] = torch.as_tensor(scenario["keep"][rank])
                objective, _, counts = distributed_objective(
                    loss_inst, pred, ref, world_size
                )
                assert counts["masked"].item() == len(scenario["atoms"][rank])
                objective.backward()
                for param, ref_grad in zip(model.parameters(), ref_grads):
                    assert param.grad is not None, f"rank {rank}: missing gradient"
                    torch.testing.assert_close(param.grad, ref_grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason=_SKIP_REASON)
def test_ddp_per_frame_loss_sums_frame_gradients(tmp_path):
    context = mp.start_processes(
        _per_frame_loss_worker,
        args=(_WORLD_SIZE, str(tmp_path / "gloo_init")),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    join_workers(context, nprocs=_WORLD_SIZE, timeout=_JOIN_TIMEOUT)


# ---------------------------------------------------------------------------
# metrics: additive sufficient statistics across batches and ranks
# ---------------------------------------------------------------------------


def _metrics_worker(rank, world_size, init_file):
    from nequip.train import Metrics

    torch.set_default_dtype(torch.float64)
    _init_gloo(rank, world_size, init_file)
    try:

        def run(partitioned_errors, components):
            metrics = Metrics(components=components, distributed=True)
            for errors in partitioned_errors:
                pred = {"f": torch.zeros(len(errors))}
                ref = {"f": torch.as_tensor(errors, dtype=torch.get_default_dtype())}
                metrics(pred, ref)
            return metrics.global_result()

        mae_hash = Metrics.hash_component(("f", "mae"))
        rmse_hash = Metrics.hash_component(("f", "rmse"))
        components = [("f", "mae"), ("f", "rmse")]

        # [1] vs [3,3,3]: MAE 2.5, MSE 7, RMSE sqrt(7)
        scenario_a = run([[[1.0]], [[3.0, 3.0, 3.0]]][rank], components)
        torch.testing.assert_close(
            scenario_a[("f", mae_hash)].squeeze(), torch.tensor(2.5)
        )
        torch.testing.assert_close(
            scenario_a[("f", rmse_hash)].squeeze(), torch.tensor(math.sqrt(7.0))
        )
        assert scenario_a[("f", rmse_hash)].square().item() == pytest.approx(7.0)

        # different batch repartition: identical finalized statistics
        scenario_b = run(
            [[[1.0, 3.0]], [[3.0, 3.0]]][rank],
            components,
        )
        # same data split into two batches on one rank
        scenario_c = run(
            [[[1.0], [3.0]], [[3.0], [3.0]]][rank],
            components,
        )
        for key in scenario_a:
            torch.testing.assert_close(scenario_a[key], scenario_b[key])
            torch.testing.assert_close(scenario_a[key], scenario_c[key])

        # zero-valid-label rank contributes nothing but stays in the schema
        nan_components = [("f", "mae", {"ignore_nan": True})]
        nan_hash = Metrics.hash_component(("f", "mae", {"ignore_nan": True}))
        batches = [[[1.0, 2.0]], [[float("nan"), float("nan")]]][rank]
        result = run(batches, nan_components)
        torch.testing.assert_close(result[("f", nan_hash)].squeeze(), torch.tensor(1.5))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason=_SKIP_REASON)
def test_ddp_metrics_additive_aggregation(tmp_path):
    context = mp.start_processes(
        _metrics_worker,
        args=(_WORLD_SIZE, str(tmp_path / "gloo_init")),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    join_workers(context, nprocs=_WORLD_SIZE, timeout=_JOIN_TIMEOUT)


# ---------------------------------------------------------------------------
# validation: nonpadding slices, empty rank, global statistics
# ---------------------------------------------------------------------------


def _validation_worker(rank, world_size, init_file, out_dir):
    from nequip.train import Metrics

    torch.set_default_dtype(torch.float64)
    _init_gloo(rank, world_size, init_file)
    try:
        for n_frames in (5, 1):
            indices = list(range(rank, n_frames, world_size))
            # record visited frames for the partition check
            with open(Path(out_dir) / f"visited-{n_frames}-{rank}.txt", "w") as f:
                f.write(",".join(str(i) for i in indices))

            errors = torch.arange(1, n_frames + 1, dtype=torch.get_default_dtype())
            metrics = Metrics(
                components=[("f", "mae"), ("f", "rmse")], distributed=True
            )
            # deterministic nonpadding slices, validation_batch_size=2; a rank
            # may run zero batches (n_frames=1, rank 1)
            for start in range(0, len(indices), 2):
                batch_idcs = indices[start : start + 2]
                pred = {"f": torch.zeros(len(batch_idcs))}
                ref = {"f": errors[batch_idcs]}
                metrics(pred, ref)
            result = metrics.global_result()

            # single-process reference over all frames in one batch
            reference = Metrics(components=[("f", "mae"), ("f", "rmse")])
            reference({"f": torch.zeros(n_frames)}, {"f": errors})
            expected = reference.current_result()
            assert set(result) == set(expected)
            for key in expected:
                torch.testing.assert_close(result[key], expected[key])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason=_SKIP_REASON)
def test_ddp_validation_nonpadding_slices(tmp_path):
    context = mp.start_processes(
        _validation_worker,
        args=(_WORLD_SIZE, str(tmp_path / "gloo_init"), str(tmp_path)),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    join_workers(context, nprocs=_WORLD_SIZE, timeout=_JOIN_TIMEOUT)
    for n_frames in (5, 1):
        visited = []
        for rank in range(_WORLD_SIZE):
            content = (tmp_path / f"visited-{n_frames}-{rank}.txt").read_text()
            visited.extend(int(i) for i in content.split(",") if i != "")
        assert sorted(visited) == list(
            range(n_frames)
        ), f"each of the {n_frames} frames must be visited exactly once"


# ---------------------------------------------------------------------------
# dataset markers: invocation identity, stale markers, failure propagation
# ---------------------------------------------------------------------------


def _dataset_worker(rank, world_size, init_file, marker_root, job_id, fail_rank):
    os.environ["SLURM_JOB_ID"] = job_id
    _init_gloo(rank, world_size, init_file)
    try:
        import nequip.scripts.train as train_module

        def fake_dataset_from_config(config, prefix="dataset"):
            if fail_rank == rank:
                raise RuntimeError(f"injected failure on rank {rank}")
            return f"dataset-{prefix}"

        train_module.dataset_from_config = fake_dataset_from_config
        outcome = "ok"
        try:
            train_module._distributed_dataset_from_config(
                {"dataset_root": marker_root, "dataset": "dummy"}, prefix="dataset"
            )
        except RuntimeError as exc:
            outcome = f"error: {exc}"
        (Path(marker_root) / f"outcome-{rank}.txt").write_text(outcome)
    finally:
        dist.destroy_process_group()


def _run_dataset_test(tmp_path, job_id, fail_rank=-1):
    marker_root = tmp_path / "markers"
    marker_root.mkdir(exist_ok=True)
    context = mp.start_processes(
        _dataset_worker,
        args=(
            _WORLD_SIZE,
            str(tmp_path / "gloo_init"),
            str(marker_root),
            job_id,
            fail_rank,
        ),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    join_workers(context, nprocs=_WORLD_SIZE, timeout=_JOIN_TIMEOUT)
    outcomes = [
        (marker_root / f"outcome-{rank}.txt").read_text() for rank in range(_WORLD_SIZE)
    ]
    return marker_root, outcomes


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason=_SKIP_REASON)
def test_ddp_dataset_markers_ignore_stale_identity(tmp_path):
    marker_root = tmp_path / "markers"
    marker_root.mkdir()
    # stale success AND failure markers from an old invocation identity
    stale_key = ".spingnn-ddp-dataset-2-slurm-OLDJOB"
    for rank in range(_WORLD_SIZE):
        (marker_root / f"{stale_key}-rank-{rank}.ok").write_text("")
        (marker_root / f"{stale_key}-rank-{rank}.failed").write_text("old failure")
    _, outcomes = _run_dataset_test(tmp_path, job_id="NEWJOB")
    assert outcomes == ["ok", "ok"], outcomes
    # the new invocation wrote markers in its own namespace
    new_key = ".spingnn-ddp-dataset-2-slurm-NEWJOB"
    for rank in range(_WORLD_SIZE):
        assert (marker_root / f"{new_key}-rank-{rank}.ok").exists()


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason=_SKIP_REASON)
@pytest.mark.parametrize("fail_rank", [0, 1])
def test_ddp_dataset_marker_failure_reaches_all_ranks(tmp_path, fail_rank):
    _, outcomes = _run_dataset_test(tmp_path, job_id="FAILJOB", fail_rank=fail_rank)
    for outcome in outcomes:
        assert outcome.startswith("error:"), outcomes
        assert "injected failure" in outcome or "dataset preparation failed" in outcome


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason=_SKIP_REASON)
def test_ddp_dataset_failure_checked_despite_all_ok(tmp_path):
    marker_root = tmp_path / "markers"
    marker_root.mkdir()
    # all success markers plus one failure marker in the CURRENT namespace
    key = ".spingnn-ddp-dataset-2-slurm-SEEDJOB"
    for rank in range(_WORLD_SIZE):
        (marker_root / f"{key}-rank-{rank}.ok").write_text("")
    (marker_root / f"{key}-rank-1.failed").write_text("poisoned failure")
    _, outcomes = _run_dataset_test(tmp_path, job_id="SEEDJOB")
    assert all(outcome.startswith("error:") for outcome in outcomes), outcomes
    assert all("poisoned failure" in outcome for outcome in outcomes), outcomes


# ---------------------------------------------------------------------------
# static contract checks
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parents[3]


def test_dataset_marker_namespace_uses_invocation_identity():
    source = (_ROOT / "nequip/scripts/train.py").read_text()
    assert "def _distributed_invocation_id" in source
    key_start = source.index("marker_key = ")
    key_block = source[key_start : source.index("ok_marker", key_start)]
    assert "_distributed_invocation_id()" in key_block
    # resume path goes through the same coordinated loader
    restart = source[source.index("def restart(") :]
    assert restart.count("_distributed_dataset_from_config") >= 2
    assert restart.count("dataset_from_config(config, prefix=") == restart.count(
        "_distributed_dataset_from_config(config, prefix="
    )


def test_trainer_distributed_validation_contract():
    source = (_ROOT / "nequip/train/trainer.py").read_text()
    # validation forwards go through the unwrapped module when distributed
    assert "forward_model.module" in source
    # buffers are synced once before the validation loop
    assert "_ddp_sync_buffers" in source
    # AWL, SAM and detach-mode objectives are rejected when distributed
    init_start = source.index("    def __init__(")
    init_end = source.index("    def init_objects", init_start)
    init = source[init_start:init_end]
    assert "lauto_loss" in init and "not supported in" in init
    assert "ldetach_method" in init
    assert '"sam"' in init and "SAM is not supported" in init
