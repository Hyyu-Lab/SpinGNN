"""A scheduler-mediated MPI abort is different from a timeout escalation."""

import os
import subprocess

import pytest

from test_mpi_runtime_failures import _verify_failure_exit


ABORT_OUTPUT = """ERROR on proc 1: injected rank-local failure
Abort(1) on node 1 (rank 1 in comm 0): application called MPI_Abort(MPI_COMM_WORLD, 1) - process 1
slurmstepd: error: *** STEP 12345.7 ON node1 CANCELLED AT 2026-10-23T13:18:23 ***
srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
srun: error: node2: task 1: Killed
srun: Terminating StepId=12345.7
srun: error: node1: task 0: Killed
"""


def test_slurm_pmi_abort_requires_the_complete_matching_exit_chain():
    assert _verify_failure_exit(137, ABORT_OUTPUT, 0.9, "12345") == {
        "kind": "slurm_pmi2_abort",
        "step": "12345.7",
        "rank": 1,
    }


@pytest.mark.parametrize(
    "code,output,elapsed,job",
    [
        (0, ABORT_OUTPUT, 0.9, "12345"),
        (124, ABORT_OUTPUT, 0.9, "12345"),
        (-9, ABORT_OUTPUT, 0.9, "12345"),
        (139, ABORT_OUTPUT, 0.9, "12345"),
        (137, ABORT_OUTPUT, 31, "12345"),
        (137, ABORT_OUTPUT, 0.9, None),
        (137, ABORT_OUTPUT, 0.9, "54321"),
        (137, "killed", 0.9, "12345"),
        (
            137,
            ABORT_OUTPUT.replace("ERROR on proc 1:", "ERROR on proc 0:"),
            0.9,
            "12345",
        ),
        (
            137,
            ABORT_OUTPUT.replace(
                "Terminating StepId=12345.7", "Terminating StepId=12345.8"
            ),
            0.9,
            "12345",
        ),
        (
            137,
            ABORT_OUTPUT.replace("application called MPI_Abort", "unrelated MPI_Abort"),
            0.9,
            "12345",
        ),
        (
            137,
            ABORT_OUTPUT + "timeout: sending signal TERM to command 'srun'\n",
            0.9,
            "12345",
        ),
        (
            1,
            ABORT_OUTPUT + "timeout: sending signal KILL to command 'srun'\n",
            0.9,
            "12345",
        ),
    ],
)
def test_incomplete_abort_or_supervisor_kill_is_rejected(code, output, elapsed, job):
    with pytest.raises(AssertionError):
        _verify_failure_exit(code, output, elapsed, job)


def test_abort_text_cannot_hide_a_real_timeout_kill():
    environment = dict(os.environ, LC_ALL="C", MPI_ABORT_TEXT=ABORT_OUTPUT)
    result = subprocess.run(
        [
            "timeout",
            "--verbose",
            "--kill-after=0.1",
            "0.1",
            "bash",
            "-c",
            'trap "" TERM; printf "%s" "$MPI_ABORT_TEXT"; while :; do sleep 1; done',
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        timeout=5,
        check=False,
    )
    assert result.returncode in (-9, 137)
    assert "timeout: sending signal" in result.stdout
    with pytest.raises(AssertionError):
        _verify_failure_exit(result.returncode, result.stdout, 0.5, "12345")
