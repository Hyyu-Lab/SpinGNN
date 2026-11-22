"""Exercise installer routing and runtime failures without installing packages."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def install_tree(tmp_path):
    ci = tmp_path / "tools" / "ci"
    ci.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / "tools/ci/install.sh", ci / "install.sh")
    for tier in ("cpu", "gpu"):
        path = ci / f"install_{tier}_env.sh"
        path.write_text(
            '#!/bin/bash\nprintf "DELEGATE ' + tier + ' python=%s\\n" "${PYTHON:-}"\n'
        )
        path.chmod(0o755)
    # Keep the real post-install payload, replacing only external packages.
    (ci / "source_python.py").write_text(
        "import os, sys, types\n"
        "from pathlib import Path\n"
        'root = Path(os.environ["SPINGNN_REPO_ROOT"])\n'
        'for name, version in (("e3nn", "0.5.0+spin"), '
        '("nequip", "0.5.6+spin"), ("allegro", "0.2.0+spin")):\n'
        "    sys.modules[name] = types.SimpleNamespace(__name__=name, "
        "__version__=version, __file__=str(root/name/'__init__.py'))\n"
        "cuda = types.SimpleNamespace(\n"
        '    is_available=lambda: os.environ.get("TEST_CUDA") == "1",\n'
        '    get_device_name=lambda index: os.environ.get("TEST_DEVICE", "RTX 4090"),\n'
        '    get_device_capability=lambda index: tuple(map(int, os.environ.get("TEST_CAP", "8,9").split(","))))\n'
        'sys.modules["torch"] = types.SimpleNamespace(cuda=cuda)\n'
        "exec(sys.argv[2])\n"
    )
    python = tmp_path / "validation/env/spingnn-cu128/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    # Override a host nvidia-smi deterministically, including CPU-only hosts.
    smi = fakebin / "nvidia-smi"
    smi.write_text(
        '#!/bin/bash\nif [[ "${TEST_SMI_ERROR:-0}" == 1 ]]; then exit 9; fi\n'
        '[[ -z "${TEST_DEVICE:-}" ]] || printf "%s\\n" "$TEST_DEVICE"\n'
        "exit 0\n"
    )
    smi.chmod(0o755)
    env = os.environ.copy()
    for key in ("PYTHON", "VIRTUAL_ENV", "TEST_DEVICE", "TEST_CUDA"):
        env.pop(key, None)
    env["PATH"] = str(fakebin) + os.pathsep + env["PATH"]
    return ci / "install.sh", env, python


def run_install(install_tree, *args, **updates):
    script, env, _ = install_tree
    return subprocess.run(
        ["bash", str(script), *args],
        env={**env, **updates},
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.mark.parametrize("tier", ["v100", "unknown", "4090-extra"])
def test_invalid_gpu_is_rejected_before_install(install_tree, tier):
    result = run_install(install_tree, "--gpu", tier)
    assert result.returncode == 2
    assert "DELEGATE" not in result.stdout


def test_cpu_without_interpreter_is_rejected(install_tree):
    result = run_install(install_tree, "--gpu", "cpu")
    assert result.returncode == 2
    assert "DELEGATE" not in result.stdout


def test_cpu_uses_same_interpreter_for_install_and_check(install_tree):
    _, _, python = install_tree
    result = run_install(install_tree, "--gpu", "cpu", PYTHON=str(python))
    assert result.returncode == 0, result.stderr
    assert f"DELEGATE cpu python={python}" in result.stdout
    assert "INSTALL_OK tier=cpu" in result.stdout


def test_virtualenv_interpreter_is_forwarded_to_cpu_delegate(install_tree):
    _, _, python = install_tree
    result = run_install(
        install_tree, "--gpu", "cpu", VIRTUAL_ENV=str(python.parent.parent)
    )
    assert result.returncode == 0, result.stderr
    assert f"DELEGATE cpu python={python}" in result.stdout


@pytest.mark.parametrize(
    "name,cap,tier",
    [
        ("RTX 4090", "8,9", "4090"),
        ("RTX 5090", "12,0", "5090"),
        ("NVIDIA A100", "8,0", "a100"),
        ("NVIDIA H100", "9,0", "h100"),
    ],
)
def test_auto_detects_supported_gpu(install_tree, name, cap, tier):
    result = run_install(install_tree, TEST_DEVICE=name, TEST_CAP=cap, TEST_CUDA="1")
    assert result.returncode == 0, result.stderr
    assert "DEVICE_OK" in result.stdout
    assert f"INSTALL_OK tier={tier}" in result.stdout


def test_detected_gpu_with_broken_cuda_fails(install_tree):
    result = run_install(install_tree, TEST_DEVICE="RTX 4090", TEST_CUDA="0")
    assert result.returncode != 0
    assert "INSTALL_OK" not in result.stdout
    assert "CUDA" in result.stderr


def test_explicit_gpu_on_login_node_can_install(install_tree):
    result = run_install(install_tree, "--gpu", "5090", TEST_CUDA="0")
    assert result.returncode == 0, result.stderr
    assert "DEVICE_SKIP" in result.stdout
    assert "INSTALL_OK tier=5090" in result.stdout


def test_wrong_runtime_capability_fails(install_tree):
    result = run_install(
        install_tree, TEST_DEVICE="RTX 4090", TEST_CUDA="1", TEST_CAP="8,0"
    )
    assert result.returncode != 0
    assert "INSTALL_OK" not in result.stdout


def test_nvidia_query_failure_does_not_install_cpu(install_tree):
    _, _, python = install_tree
    result = run_install(install_tree, TEST_SMI_ERROR="1", PYTHON=str(python))
    assert result.returncode == 2
    assert "DELEGATE" not in result.stdout
