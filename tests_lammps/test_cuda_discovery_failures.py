"""CUDA discovery exceptions must reach the MPI agreement before termination."""

from __future__ import annotations

import os
import shlex
import subprocess

import pytest

from test_mpi_runtime_failures import _bounded_failure, _rank_wrapper


@pytest.fixture(scope="module")
def cuda_discovery_failure_library(tmp_path_factory):
    # Interpose only the public discovery functions. No CUDA allocation is
    # needed to exercise the failure before device selection, including in CI.
    directory = tmp_path_factory.mktemp("cuda-discovery-fault")
    source = directory / "discovery.cpp"
    source.write_text(
        r"""#include <cstdlib>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace {
bool failed_rank() {
  const char *rank = std::getenv("OMPI_COMM_WORLD_RANK");
  if (!rank) rank = std::getenv("PMI_RANK");
  if (!rank) rank = std::getenv("SLURM_PROCID");
  return rank && std::string(rank) == "1";
}
bool count_failure() {
  const char *mode = std::getenv("SPINGNN_DISCOVERY_FAILURE");
  return mode && std::string(mode) == "count";
}
}

namespace torch { namespace cuda {
bool is_available() {
  if (failed_rank() && !count_failure())
    throw std::runtime_error("injected CUDA availability failure");
  return count_failure();
}
std::size_t device_count() {
  if (failed_rank())
    throw std::runtime_error("injected CUDA device count failure");
  return 1;
}
}}
"""
    )
    library = directory / "libcuda-discovery-fault.so"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    completed = subprocess.run(
        [*compiler, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout
    return library


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.mpi
@pytest.mark.parametrize("failure", ("availability", "count"))
def test_rank_local_cuda_discovery_exception_is_a_lammps_error(
    failure,
    cuda_discovery_failure_library,
    lammps_mpi_bin,
    mpi_prefix,
    lammps_env,
    tmp_path,
):
    wrapper, env = _rank_wrapper(tmp_path, lammps_mpi_bin, lammps_env)
    inherited = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = str(cuda_discovery_failure_library) + (
        f":{inherited}" if inherited else ""
    )
    env["SPINGNN_DISCOVERY_FAILURE"] = failure
    input_path = tmp_path / "construct.in"
    input_path.write_text("units metal\natom_style atomic\npair_style allegro\n")
    output = _bounded_failure(wrapper, env, input_path, {}, mpi_prefix, tmp_path)
    assert "CUDA device discovery failed: injected CUDA" in output
    assert "ERROR on proc 1" in output
