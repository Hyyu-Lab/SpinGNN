"""Rank-local failures must terminate the complete LAMMPS MPI process group."""

from __future__ import annotations

import json
import re
import shutil
import time

import pytest
import torch

from helpers import CASES, run_lammps
from nequip.scripts import deploy
import test_pt2_parity
from test_pt2_parity import (
    _corrupt_aoti_library,
    _file_sha256,
)

malformed_bec_pt2_factory = test_pt2_parity.malformed_bec_pt2_factory


def _verify_failure_exit(returncode, output, elapsed, slurm_job_id):
    """Recognize Slurm's abort signal without accepting supervisor kills."""
    assert elapsed < 30, output
    assert not re.search(r"^timeout:.*sending signal", output, re.MULTILINE), output
    if returncode != 137:
        assert 0 < returncode < 128 and returncode != 124, output
        return {"kind": "nonzero_exit"}

    # srun returns 128 + signal for a rank killed by PMI2 MPI_Abort. The exact
    # error rank and Slurm step must agree, and GNU timeout must not have fired.
    assert slurm_job_id and re.fullmatch(r"[0-9]+", slurm_job_id), output
    abort = re.search(
        r"Abort\(1\) on node \d+ \(rank (\d+) in comm \d+\): "
        r"application called MPI_Abort\(MPI_COMM_WORLD, 1\)",
        output,
    )
    assert abort is not None, output
    assert f"ERROR on proc {abort.group(1)}:" in output, output
    cancelled = re.search(
        rf"slurmstepd: error: \*\*\* STEP ({re.escape(slurm_job_id)}\.\d+) "
        r"ON \S+ CANCELLED AT ",
        output,
    )
    assert cancelled is not None, output
    step = cancelled.group(1)
    assert re.search(
        rf"^srun: Terminating StepId={re.escape(step)}$", output, re.MULTILINE
    ), output
    return {"kind": "slurm_pmi2_abort", "step": step, "rank": int(abort.group(1))}


def _rank_wrapper(tmp_path, binary, env, *, hidden_cuda=False, failed_model=None):
    wrapper = tmp_path / "rank-local-failure.sh"
    wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-0}}}
args=("$@")
if [[ "$rank" == 1 ]]; then
  if [[ "${SPINGNN_HIDE_RANK_CUDA:-0}" == 1 ]]; then
    export CUDA_VISIBLE_DEVICES=""
  fi
  if [[ -n "${SPINGNN_FAILED_MODEL:-}" ]]; then
    for ((i=0; i+2<${#args[@]}; i++)); do
      if [[ "${args[i]}" == -var && "${args[i+1]}" == model ]]; then
        args[i+2]=$SPINGNN_FAILED_MODEL
      fi
    done
  fi
fi
exec "$SPINGNN_REAL_LAMMPS" "${args[@]}"
"""
    )
    wrapper.chmod(0o755)
    run_env = dict(env)
    run_env["SPINGNN_REAL_LAMMPS"] = str(binary)
    run_env["SPINGNN_HIDE_RANK_CUDA"] = "1" if hidden_cuda else "0"
    if failed_model is not None:
        run_env["SPINGNN_FAILED_MODEL"] = str(failed_model)
    return wrapper, run_env


def _bounded_failure(wrapper, env, input_path, variables, mpi_prefix, tmp_path):
    # GNU timeout supervises Hydra and its children as one process group. A
    # subprocess timeout alone only kills the launcher and can leave ranks alive.
    started = time.monotonic()
    result = run_lammps(
        binary=wrapper,
        input_path=input_path,
        workdir=tmp_path / "mpi",
        env=dict(env, LC_ALL="C"),
        variables=variables,
        prefix=("timeout", "--verbose", "--kill-after=5", "30", *mpi_prefix(2)),
        check=False,
        timeout=45,
    )
    elapsed = time.monotonic() - started
    (tmp_path / "mpi" / "stdout.log").write_text(result.stdout)
    evidence = {
        "returncode": result.returncode,
        "elapsed": elapsed,
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "supervisor": "GNU timeout --verbose --kill-after=5 30",
    }
    evidence_path = tmp_path / "mpi" / "exit-evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    evidence["classification"] = _verify_failure_exit(
        result.returncode, result.stdout, elapsed, env.get("SLURM_JOB_ID")
    )
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    return result.stdout


@pytest.mark.gpu
@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize(
    "pair_style", ("allegro", "nequip", "spin/allegro_spin", "spin/nequip_spin")
)
def test_mixed_cuda_visibility_fails_collectively(
    pair_style, lammps_mpi_bin, mpi_prefix, lammps_env, tmp_path
):
    assert torch.cuda.is_available(), "this regression requires a visible CUDA device"
    wrapper, env = _rank_wrapper(tmp_path, lammps_mpi_bin, lammps_env, hidden_cuda=True)
    input_path = tmp_path / "construct.in"
    atom_style = "spin" if pair_style.startswith("spin/") else "atomic"
    input_path.write_text(
        f"units metal\natom_style {atom_style}\npair_style {pair_style}\n"
    )
    output = _bounded_failure(wrapper, env, input_path, {}, mpi_prefix, tmp_path)
    assert "Inconsistent CUDA availability across MPI ranks" in output


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("case_name", ("nosoc", "bec", "nequip_spin"))
def test_rank_local_pt2_loader_failure_does_not_hang(
    case_name,
    pt2_artifact_factory,
    lammps_pt2_target_device,
    lammps_mpi_bin,
    mpi_prefix,
    lammps_env,
    tmp_path,
):
    case = CASES[case_name]
    _, good = pt2_artifact_factory(case_name, lammps_pt2_target_device)
    broken = tmp_path / "rank1-corrupt-library.pt2"
    shutil.copy2(good, broken)
    _corrupt_aoti_library(broken)
    metadata = deploy.read_deploy_metadata_sidecar(good)
    metadata[deploy.PACKAGE_SHA256_KEY] = _file_sha256(broken)
    deploy.write_deploy_metadata_sidecar(broken, metadata)
    wrapper, env = _rank_wrapper(
        tmp_path, lammps_mpi_bin, lammps_env, failed_model=broken
    )
    variables = {
        "data": case.data_path,
        "model": good,
        "dump": tmp_path / "output.dump",
    }
    if case.bec_path is not None:
        variables["bec"] = case.bec_path
    output = _bounded_failure(
        wrapper,
        env,
        case.input_path,
        variables,
        mpi_prefix,
        tmp_path,
    )
    assert "PT2 loader construction failed" in output


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("mutation", ("shape", "dtype"))
def test_rank_local_pt2_output_failure_does_not_hang(
    mutation,
    malformed_bec_pt2_factory,
    lammps_mpi_bin,
    mpi_prefix,
    lammps_env,
    tmp_path,
):
    case = CASES["bec"]
    good = malformed_bec_pt2_factory("valid")
    broken = malformed_bec_pt2_factory(mutation)
    wrapper, env = _rank_wrapper(
        tmp_path, lammps_mpi_bin, lammps_env, failed_model=broken
    )
    variables = {
        "data": case.data_path,
        "model": good,
        "dump": tmp_path / "output.dump",
        "bec": case.bec_path,
    }
    output = _bounded_failure(
        wrapper,
        env,
        case.input_path,
        variables,
        mpi_prefix,
        tmp_path,
    )
    expected = "invalid size" if mutation == "shape" else "must have dtype float32"
    assert "PT2 output 'born_effective_charges'" in output
    assert expected in output
