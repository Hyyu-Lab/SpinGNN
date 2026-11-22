"""Exercise the real Kokkos entry-point selection without compiling locally."""

import os
from pathlib import Path
import subprocess

import pytest


SOURCE = Path(__file__).resolve().parents[1]


def prefix_fixture(tmp_path, architecture, version):
    prefix = tmp_path / "kokkos"
    (prefix / "include").mkdir(parents=True)
    (prefix / "bin").mkdir()
    (prefix / "lib/cmake/Kokkos").mkdir(parents=True)
    (prefix / "include/KokkosCore_config.h").write_text(
        f"#define KOKKOS_VERSION {version}\n"
        "#define KOKKOS_ENABLE_SERIAL\n"
        "#define KOKKOS_ENABLE_CUDA\n"
        "#define KOKKOS_ENABLE_CUDA_LAMBDA\n"
        f"#define KOKKOS_ARCH_{architecture}\n"
    )
    wrapper = prefix / "bin/nvcc_wrapper"
    wrapper.write_text("#!/bin/sh\nexit 99\n")
    wrapper.chmod(0o755)
    (prefix / "lib/cmake/Kokkos/KokkosConfig.cmake").write_text("# fixture\n")
    return prefix


def run_selection(tmp_path, prefix):
    script = (SOURCE / "tools/lammps/build_lammps.sh").read_text()
    probe = tmp_path / "selection.sh"
    probe.write_text(
        script.split("BUILD_DIR=${2:-", 1)[0] + '\nprintf "%s\\n" "${KOKKOS_ARGS[@]}"\n'
    )
    env = dict(os.environ, KOKKOS_ROOT=str(prefix))
    return subprocess.run(
        ["bash", str(probe), "kokkos-cuda"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "architecture,cuda_arch", [("ADA89", "sm_89"), ("BLACKWELL120", "sm_120")]
)
def test_native_rtx_architecture_is_selected(tmp_path, architecture, cuda_arch):
    prefix = prefix_fixture(tmp_path, architecture, 40704)
    result = run_selection(tmp_path, prefix)
    assert result.returncode == 0, result.stderr
    assert f"CMAKE_CXX_FLAGS=-arch={cuda_arch}" in result.stdout
    assert "EXTERNAL_KOKKOS=yes" in result.stdout


@pytest.mark.parametrize("architecture", ["ADA89", "BLACKWELL120"])
def test_rtx_installer_validates_recognized_prefix(tmp_path, architecture):
    prefix = prefix_fixture(tmp_path, architecture, 40704)
    result = subprocess.run(
        ["bash", str(SOURCE / "tools/ci/install_kokkos_cuda.sh"), str(prefix)],
        env=dict(os.environ, SPINGNN_KOKKOS_ARCH=architecture),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "installed Kokkos library is missing" in result.stderr
    assert "refusing to overwrite non-empty invalid Kokkos prefix" in result.stderr


@pytest.mark.parametrize(
    "architecture,version",
    [
        ("ADA89", 30702),
        ("BLACKWELL120", 40703),
        ("AMPERE80", 40704),
        ("HOPPER90", 40704),
    ],
)
def test_build_rejects_unvalidated_architecture_release_pair(
    tmp_path, architecture, version
):
    prefix = prefix_fixture(tmp_path, architecture, version)
    result = run_selection(tmp_path, prefix)
    assert result.returncode == 2
    assert "requires Kokkos" in result.stderr
