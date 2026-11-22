"""Compiler cache isolation must survive parallel jobs and explicit overrides."""

import os
from pathlib import Path
import subprocess


HELPER = Path(__file__).resolve().parents[1] / "tools/ci/compile_cache.sh"


def run_cache(tmp_path, **updates):
    env = os.environ.copy()
    for name in (
        "SPINGNN_COMPILE_CACHE_ROOT",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    ):
        env.pop(name, None)
    env.update(TMPDIR=str(tmp_path), **updates)
    result = subprocess.run(
        ["bash", "-c", 'source "$1"', "bash", str(HELPER)],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return dict(field.split("=", 1) for field in result.stdout.strip().split()[1:])


def test_independent_invocations_get_private_caches(tmp_path):
    first = run_cache(tmp_path, SLURM_JOB_ID="123")
    second = run_cache(tmp_path, SLURM_JOB_ID="123")
    assert first != second
    assert all(Path(path).is_dir() for path in [*first.values(), *second.values()])


def test_explicit_cache_root_supports_warm_rerun(tmp_path):
    root = tmp_path / "warm"
    first = run_cache(tmp_path, SPINGNN_COMPILE_CACHE_ROOT=str(root))
    second = run_cache(tmp_path, SPINGNN_COMPILE_CACHE_ROOT=str(root))
    assert first == second
    assert first["inductor"] == str(root / "inductor")
