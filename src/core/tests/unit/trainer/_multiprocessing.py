"""Deadline-aware joins for spawned workers used by trainer contract tests."""

import math
from time import monotonic


_CLEANUP_TIMEOUT = 5.0
_TERMINATE_GRACE = 1.0


def _stop_workers(processes):
    """Abort the whole group; a failure must not leave peers in collectives."""
    deadline = monotonic() + _CLEANUP_TIMEOUT
    for process in processes:
        if process.is_alive():
            process.terminate()
    terminate_deadline = min(deadline, monotonic() + _TERMINATE_GRACE)
    for process in processes:
        process.join(timeout=max(0.0, terminate_deadline - monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join(timeout=max(0.0, deadline - monotonic()))
    remaining = [process.pid for process in processes if process.is_alive()]
    if remaining:
        raise RuntimeError(
            f"Unable to reap worker processes after cleanup: {remaining}"
        )


def join_workers(context, nprocs, timeout):
    """Preserve each worker's join deadline while surfacing errors from any rank.

    ProcessContext owns sentinel handling, traceback decoding, and cleanup when
    a worker fails. Normal peer exits must not restart the current worker's
    timeout. An observed timeout stays a failure even if the group finishes
    during cleanup; the cleanup budget is never extra successful work time.

    Torch's public join still performs an untimed reap after SIGKILL on failure.
    Our bounded fallback cannot impose a deadline inside that call or guarantee
    termination of an uninterruptible kernel wait; the job needs an outer limit.
    """
    try:
        if len(context.processes) != nprocs:
            raise ValueError("Unexpected number of worker processes")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Worker join timeout must be finite and positive")
        for rank, process in enumerate(context.processes):
            deadline = monotonic() + timeout
            while process.is_alive():
                remaining = deadline - monotonic()
                if remaining <= 0:
                    timed_out = process.is_alive()
                    # Preserve an already available worker traceback, but do
                    # not turn a recorded timeout into success on a late exit.
                    context.join(timeout=0, grace_period=0)
                    if timed_out:
                        raise TimeoutError(
                            f"worker {rank} (pid={process.pid}) did not exit "
                            f"within its {timeout:g} second join deadline"
                        )
                    break
                context.join(timeout=remaining, grace_period=0)
            context.join(timeout=0, grace_period=0)
        exitcodes = [process.exitcode for process in context.processes]
        assert exitcodes == [0] * nprocs, f"worker exitcodes: {exitcodes}"
    except BaseException as error:
        try:
            _stop_workers(context.processes)
        except RuntimeError as cleanup_error:
            raise error from cleanup_error
        raise
