"""Process supervision contracts independent of models and distributed backends."""

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import time
import types

import pytest
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessExitedException, ProcessRaisedException

from . import _multiprocessing as supervisor


def _worker(rank, modes, ready_directory):
    mode = modes[rank]
    if mode == "ignore_term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(ready_directory, str(rank)).write_text(str(os.getpid()))
    if mode == "success":
        return
    if mode == "raise":
        raise ValueError(f"intentional worker exception from rank {rank}")
    if mode == "nonzero":
        os._exit(7)
    while True:
        time.sleep(0.01)


@contextmanager
def _workers(tmp_path, modes):
    context = mp.start_processes(
        _worker,
        args=(modes, str(tmp_path)),
        nprocs=len(modes),
        join=False,
        start_method="spawn",
    )
    try:
        # The short fault deadlines test supervision, not interpreter import
        # latency. Existing DDP callers retain their original startup-inclusive 60s.
        startup_deadline = time.monotonic() + 60
        while not all((tmp_path / str(rank)).exists() for rank in range(len(modes))):
            if time.monotonic() >= startup_deadline:
                pytest.fail("Synthetic workers did not reach their ready boundary")
            time.sleep(0.01)
        yield context
    finally:
        # Assertions inspect the helper's result before this emergency teardown,
        # so a regression cannot pass because the fixture later removed a peer.
        for process in context.processes:
            if process.is_alive():
                process.kill()
        for process in context.processes:
            process.join(timeout=5)


def _assert_reaped(context):
    for process in context.processes:
        assert not process.is_alive() and process.exitcode is not None
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)


@pytest.mark.skipif(os.name != "posix", reason="POSIX worker signal contract")
def test_success_reaps_all_workers(tmp_path):
    with _workers(tmp_path, ("success", "success")) as context:
        supervisor.join_workers(context, nprocs=2, timeout=1)
        assert [process.exitcode for process in context.processes] == [0, 0]
        _assert_reaped(context)


@pytest.mark.skipif(os.name != "posix", reason="POSIX worker signal contract")
def test_worker_exception_keeps_traceback_and_reaps_peer(tmp_path):
    with _workers(tmp_path, ("ignore_term", "raise")) as context:
        with pytest.raises(
            ProcessRaisedException, match="intentional worker exception from rank 1"
        ) as failure:
            supervisor.join_workers(context, nprocs=2, timeout=1)
        assert failure.value.error_index == 1
        assert "ValueError" in str(failure.value)
        assert context.processes[0].exitcode == -signal.SIGKILL
        _assert_reaped(context)


@pytest.mark.skipif(os.name != "posix", reason="POSIX worker signal contract")
def test_nonzero_exit_keeps_exitcode_and_reaps_peer(tmp_path):
    with _workers(tmp_path, ("ignore_term", "nonzero")) as context:
        with pytest.raises(ProcessExitedException) as failure:
            supervisor.join_workers(context, nprocs=2, timeout=1)
        assert failure.value.error_index == 1 and failure.value.exit_code == 7
        assert context.processes[0].exitcode == -signal.SIGKILL
        _assert_reaped(context)


@pytest.mark.skipif(os.name != "posix", reason="POSIX worker signal contract")
def test_timeout_kills_all_sigterm_ignoring_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "_CLEANUP_TIMEOUT", 1.0)
    monkeypatch.setattr(supervisor, "_TERMINATE_GRACE", 0.1)
    with _workers(tmp_path, ("ignore_term", "ignore_term")) as context:
        with pytest.raises(TimeoutError, match="worker 0 .*0.1 second join deadline"):
            supervisor.join_workers(context, nprocs=2, timeout=0.1)
        assert [process.exitcode for process in context.processes] == [
            -signal.SIGKILL
        ] * 2
        _assert_reaped(context)


class _Process:
    def __init__(self, pid):
        self.pid = pid
        self.alive = True
        self.exitcode = None

    def is_alive(self):
        return self.alive

    def finish(self):
        self.alive = False
        self.exitcode = 0

    def join(self, timeout):
        assert timeout is not None


def test_peer_completion_does_not_extend_deadline_or_accept_late_exit(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(supervisor, "monotonic", lambda: now[0])
    processes = [_Process(1), _Process(2)]
    waits = []

    def join(timeout, grace_period):
        assert grace_period == 0
        waits.append(timeout)
        if len(waits) == 1:
            now[0] = 59.0
            processes[1].finish()
        elif len(waits) == 2:
            now[0] = 60.0
        else:
            # The original timeout was already observed. A subsequent natural
            # exit must not turn that failure into a successful test.
            processes[0].finish()
        return not any(process.is_alive() for process in processes)

    context = types.SimpleNamespace(processes=processes, join=join)
    with pytest.raises(TimeoutError, match="60 second join deadline"):
        supervisor.join_workers(context, nprocs=2, timeout=60)
    assert waits == [60.0, 1.0, 0]


def test_each_worker_keeps_its_original_sequential_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(supervisor, "monotonic", lambda: now[0])
    processes = [_Process(1), _Process(2)]
    waits = []

    def join(timeout, grace_period):
        assert grace_period == 0
        waits.append(timeout)
        if timeout:
            now[0] += 59.0
            next(process for process in processes if process.is_alive()).finish()
        return not any(process.is_alive() for process in processes)

    supervisor.join_workers(
        types.SimpleNamespace(processes=processes, join=join), nprocs=2, timeout=60
    )
    assert waits == [60.0, 0, 60.0, 0]
    assert now[0] == 118.0
