"""GPU-only MPI device-placement regression for deployed pair styles."""

from __future__ import annotations

import re

import pytest
import torch

from helpers import CASES, run_lammps


@pytest.mark.gpu
@pytest.mark.lammps
@pytest.mark.mpi
def test_nequip_spin_uses_distinct_visible_gpus_for_two_local_ranks(
    model_paths,
    lammps_mpi_bin,
    mpi_prefix,
    lammps_env,
    tmp_path,
):
    if torch.cuda.device_count() < 2:
        pytest.skip("two-rank CUDA assignment requires two visible GPUs")
    case = CASES["nequip_spin"]
    result = run_lammps(
        binary=lammps_mpi_bin,
        input_path=case.input_path,
        workdir=tmp_path,
        env=lammps_env,
        prefix=mpi_prefix(2),
        variables={
            "data": case.data_path,
            "model": model_paths[case.name],
            "dump": tmp_path / "nequip-spin.dump",
        },
    )
    selected = sorted(
        set(re.findall(r"NEQUIP is using device cuda:(\d+)", result.output))
    )
    assert selected == ["0", "1"], result.output
