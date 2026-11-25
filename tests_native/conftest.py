"""Runtime fixtures shared by the modern LAMMPS qualification lanes."""

import json
import os
import re
from pathlib import Path
import shlex
import subprocess

import pytest


@pytest.fixture(scope="session")
def m4_backend():
    return os.environ.get("SPINGNN2_LAMMPS_BACKEND", "cpu")


@pytest.fixture(scope="session")
def m4_binary():
    path = os.environ.get("SPINGNN2_LAMMPS_BIN")
    if path is None:
        if "SPINGNN2_LAMMPS_BACKEND" in os.environ:
            pytest.fail("SPINGNN2_LAMMPS_BIN is required for the selected M4 lane")
        pytest.skip("Set SPINGNN2_LAMMPS_BIN to run the modern LAMMPS lane")
    assert Path(path).is_file(), path
    return Path(path).resolve()


@pytest.fixture(scope="session")
def m4_models(m4_binary):
    return json.loads(Path(os.environ["SPINGNN2_LAMMPS_MODELS"]).read_text())


@pytest.fixture
def m4_pair(m4_models):
    def commands(family, precision="float64"):
        spec = m4_models[family][precision]
        return (
            f"pair_style spingnn2 precision {precision} profile yes\n"
            f'pair_coeff * * "{spec["contract"]}" "{spec["manifest"]}" 0 1'
        )

    return commands


@pytest.fixture
def lmp_run(m4_binary, m4_backend, tmp_path):
    def run(script, ranks=1, *, trace=False, gpu_aware=True, allow_shared_gpu=False):
        source = tmp_path / "in.m4"
        source.write_text(script + "\n")
        cmd = [str(m4_binary), "-log", "none", "-in", str(source)]
        if m4_backend == "cuda":
            cmd += [
                "-k",
                "on",
                "g",
                "1",
                "-sf",
                "kk",
                "-pk",
                "kokkos",
                "gpu/aware",
                "on" if gpu_aware else "off",
                "neigh",
                "half",
            ]
        elif m4_backend == "kk-cpu":
            cmd += ["-k", "on", "t", os.getenv("OMP_NUM_THREADS", "1"), "-sf", "kk"]
            if not gpu_aware:
                cmd += ["-pk", "kokkos", "gpu/aware", "off"]
        if trace:
            cmd = [
                "strace",
                "-f",
                "-e",
                "trace=execve,execveat,openat",
                "-o",
                str(tmp_path / "runtime.trace"),
                *cmd,
            ]
        if m4_backend == "cuda":
            launcher = (
                Path(__file__).resolve().parents[1]
                / "tools/lammps/spingnn2_gpu_rank.py"
            )
            cmd = [str(launcher), *cmd]
        if ranks > 1:
            cmd = [
                *shlex.split(os.getenv("SPINGNN2_MPIRUN", "mpirun")),
                "-n",
                str(ranks),
                *cmd,
            ]
        proc = subprocess.run(
            cmd,
            cwd=tmp_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stdout
        if m4_backend == "cuda":
            ids = re.findall(
                r"^SPINGNN2_GPU .* uuid=(GPU-[a-f0-9-]+)$", proc.stdout, re.MULTILINE
            )
            assert len(ids) == ranks, proc.stdout
            if not allow_shared_gpu:
                assert len(set(ids)) == ranks, proc.stdout
        return proc.stdout

    return run
