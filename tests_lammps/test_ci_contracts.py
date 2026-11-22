"""Regression tests for deployment CI and build fail-closed contracts."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import conftest as lammps_conftest


REPO_ROOT = Path(__file__).resolve().parents[1]

# `mpi_probe.sh` compiles a real MPI payload (it must: a non-MPI payload cannot
# tell a working transport from a broken one). Environments without an MPI
# compiler cannot exercise it, so those cases skip instead of failing.
requires_mpicc = pytest.mark.skipif(
    shutil.which(os.environ.get("MPI_PROBE_MPICC", "mpicc")) is None,
    reason="mpi_probe.sh requires an MPI C compiler (mpicc) to build its payload",
)


@pytest.mark.lammps_contract
def test_mpi_binary_contract_probe(lammps_mpi_bin):
    assert lammps_mpi_bin.is_file()


@pytest.mark.lammps_contract
def test_serial_binary_contract_probe(lammps_bin):
    assert lammps_bin.is_file()


def test_default_missing_serial_binary_fails_pytest_session(tmp_path):
    env = os.environ.copy()
    env.pop("SPINGNN_REQUIRE_LAMMPS", None)
    env["SPINGNN_LAMMPS_BIN"] = str(tmp_path / "missing-lmp")
    env.pop("SPINGNN_ALLOW_LAMMPS_SKIP", None)
    result = subprocess.run(
        [
            os.sys.executable,
            "-m",
            "pytest",
            "tests_lammps/test_ci_contracts.py",
            "-k",
            "test_serial_binary_contract_probe",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout
    assert "serial LAMMPS environment contract is unavailable" in result.stdout


def test_required_missing_mpi_binary_fails_pytest_session(tmp_path):
    env = os.environ.copy()
    env["SPINGNN_REQUIRE_LAMMPS"] = "1"
    env["SPINGNN_LAMMPS_BIN"] = os.sys.executable
    env["SPINGNN_LAMMPS_MPI_BIN"] = str(tmp_path / "missing-lmp")
    env.pop("SPINGNN_ALLOW_LAMMPS_SKIP", None)
    result = subprocess.run(
        [
            os.sys.executable,
            "-m",
            "pytest",
            "tests_lammps/test_ci_contracts.py",
            "-k",
            "test_mpi_binary_contract_probe",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout
    assert "MPI LAMMPS environment contract is unavailable" in result.stdout


@pytest.mark.parametrize(
    ("fixture", "environment_name", "kind"),
    (
        (lammps_conftest.lammps_bin, "SPINGNN_LAMMPS_BIN", "serial"),
        (lammps_conftest.lammps_mpi_bin, "SPINGNN_LAMMPS_MPI_BIN", "MPI"),
    ),
)
def test_missing_lammps_binary_is_a_hard_error_by_default(
    monkeypatch, fixture, environment_name, kind
):
    monkeypatch.delenv(environment_name, raising=False)
    monkeypatch.delenv("SPINGNN_ALLOW_LAMMPS_SKIP", raising=False)
    with pytest.raises(pytest.UsageError, match=f"{kind} LAMMPS environment contract"):
        fixture.__wrapped__()


def test_missing_lammps_binary_has_explicit_skip_opt_out(monkeypatch):
    monkeypatch.delenv("SPINGNN_LAMMPS_BIN", raising=False)
    monkeypatch.setenv("SPINGNN_ALLOW_LAMMPS_SKIP", "1")
    with pytest.raises(pytest.skip.Exception, match="serial LAMMPS"):
        lammps_conftest.lammps_bin.__wrapped__()


def test_gpu_full_lane_rejects_pytest_exit_five(tmp_path):
    environment = tmp_path / "fake-env"
    python = environment / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  exit 0
fi
for arg in "$@"; do
  case "$arg" in
    --junitxml=*)
      report=${arg#--junitxml=}
      printf '<testsuites tests="0" failures="0" errors="0" skipped="0" />\\n' >"$report"
      ;;
  esac
done
exit 5
"""
    )
    python.chmod(0o755)
    env = os.environ.copy()
    env["SPINGNN_ENV"] = str(environment)
    env["SPINGNN_CI_EVIDENCE_DIR"] = str(tmp_path / "evidence")
    env.pop("SPINGNN_LAMMPS_BIN", None)
    env.pop("SPINGNN_GPU_BENCH_CONFIG", None)
    result = subprocess.run(
        [str(REPO_ROOT / "tools/ci/gpu_full_tests.sh")],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "name=gpu-nequip result=FAIL exit_code=5" in result.stdout
    # Infrastructure failure must reach every required Python lane, including
    # the informational precision probes; unrequested LAMMPS stays skipped.
    required_lanes = {
        "gpu-nequip",
        "gpu-allegro",
        "bf16-nequip",
        "bf16-allegro",
        "compile-nequip",
        "compile-allegro",
        "compile-checkpoint",
        "slow-nequip",
        "slow-allegro",
    }
    failed_lanes = {
        line.split("name=", 1)[1].split()[0]
        for line in result.stdout.splitlines()
        if line.startswith("CI_SUITE_RESULT ") and "result=FAIL" in line
    }
    assert failed_lanes == required_lanes
    summary = next(
        line for line in result.stdout.splitlines() if line.startswith("CI_SUMMARY ")
    )
    fields = dict(token.split("=", 1) for token in summary.split()[1:])
    assert int(fields["suites_failed"]) == len(required_lanes)
    assert int(fields["suites_passed"]) == 0


def test_gpu_full_lane_rejects_unknown_gpu_expectation(tmp_path):
    environment = tmp_path / "fake-env"
    python = environment / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  exit 0
fi
exit 5
"""
    )
    python.chmod(0o755)
    env = os.environ.copy()
    env["SPINGNN_ENV"] = str(environment)
    env["SPINGNN_CI_EVIDENCE_DIR"] = str(tmp_path / "evidence")
    env["SPINGNN_GPU_EXPECT"] = "any-cuda"
    result = subprocess.run(
        [str(REPO_ROOT / "tools/ci/gpu_full_tests.sh")],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stdout
    assert (
        "SPINGNN_GPU_EXPECT must be exactly one of: h100, a100, 4090, 5090"
        in result.stdout
    )


def test_binary_contract_probe_is_excluded_from_production_marker():
    env = os.environ.copy()
    env["SPINGNN_LAMMPS_BIN"] = os.sys.executable
    result = subprocess.run(
        [
            os.sys.executable,
            "-m",
            "pytest",
            "tests_lammps/test_ci_contracts.py",
            "-k",
            "test_serial_binary_contract_probe",
            "-m",
            "lammps and not mpi and not slow",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, result.stdout


def test_both_gpu_lammps_lanes_require_a_non_skipped_test():
    for relative_path in (
        "tools/ci/gpu_full_tests.sh",
        "tools/ci/gpu_lammps_tests.sh",
    ):
        source = (REPO_ROOT / relative_path).read_text()
        assert "--junitxml=" in source, relative_path
        assert "required lane passed only" in source, relative_path


def _run_junit_floor_gate(tests: int, skipped: int, minimum: int, tmp_path):
    """Run gpu_lammps_tests.sh's real require_junit_passes on a crafted XML."""
    script = (REPO_ROOT / "tools/ci/gpu_lammps_tests.sh").read_text()
    inside = script.split("require_junit_passes() {", 1)[1]
    parser = inside.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    junit = tmp_path / f"junit-{tests}-{skipped}-{minimum}.xml"
    junit.write_text(
        f'<testsuite tests="{tests}" skipped="{skipped}" '
        'failures="0" errors="0" />\n'
    )
    return subprocess.run(
        [os.sys.executable, "-c", parser, str(junit), str(minimum)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("backend", "gpu_args"),
    (("Serial", []), ("CUDA Serial", ["g", "1"])),
)
def test_kokkos_wrapper_requests_gpu_from_binary_backend(tmp_path, backend, gpu_args):
    from test_kokkos_parity import _kokkos_wrapper

    binary = tmp_path / "lmp with spaces"
    binary.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-h" ]; then\n'
        f"  printf '%s\\n' 'KOKKOS package API: {backend}' "
        "'spin/allegro_spin/kk'\n"
        "else\n"
        '  printf "%s\\n" "$@"\n'
        "fi\n"
    )
    binary.chmod(0o755)
    wrapper = _kokkos_wrapper(binary, os.environ.copy(), tmp_path)
    result = subprocess.run(
        [str(wrapper), "-in", "input with spaces.in"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        "-k",
        "on",
        *gpu_args,
        "-sf",
        "kk",
        "-in",
        "input with spaces.in",
    ]


def test_gpu_lammps_lanes_require_the_current_non_skipped_test_counts(tmp_path):
    """Run the embedded parser at the configured floors and check both callers."""
    # The real parser at the real floors: measured counts pass, one fewer
    # non-skipped test fails.
    assert _run_junit_floor_gate(47, 7, 40, tmp_path).returncode == 0
    assert _run_junit_floor_gate(47, 8, 40, tmp_path).returncode != 0
    assert _run_junit_floor_gate(9, 0, 9, tmp_path).returncode == 0
    assert _run_junit_floor_gate(9, 1, 9, tmp_path).returncode != 0
    # All-skipped and malformed XML fail closed.
    assert _run_junit_floor_gate(40, 40, 40, tmp_path).returncode != 0
    broken = tmp_path / "broken.xml"
    broken.write_text("<testsuite")
    script = (REPO_ROOT / "tools/ci/gpu_lammps_tests.sh").read_text()
    parser = script.split("require_junit_passes() {", 1)[1]
    parser = parser.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    malformed = subprocess.run(
        [os.sys.executable, "-c", parser, str(broken), "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert malformed.returncode != 0
    # Both floor call sites must be live code, not comments.
    import re

    for call in (
        r'require_junit_passes "\$gpu_lammps_junit" 40',
        r'require_junit_passes "\$gpu_kokkos_junit" 9',
    ):
        live = [
            line
            for line in script.splitlines()
            if re.search(call, line) and not line.lstrip().startswith("#")
        ]
        assert live, f"floor call is not live code: {call}"


@pytest.mark.parametrize("bad_case", (None, "skipped", "missing"))
def test_gpu_runtime_gate_requires_all_three_parity_results(tmp_path, bad_case):
    junit = tmp_path / "serial.xml"
    cases = [
        '<testcase classname="test_pt2_parity" '
        'name="test_pt2_lammps_matches_torchscript_run0[nosoc]" />',
        '<testcase classname="test_pt2_parity" '
        'name="test_bec_pt2_lammps_matches_torchscript_and_python_run0" />',
    ]
    if bad_case != "missing":
        child = "<skipped />" if bad_case == "skipped" else ""
        cases.append(
            '<testcase classname="test_pt2_parity" '
            f'name="test_pt2_lammps_matches_torchscript_run0[soc]">{child}</testcase>'
        )
    junit.write_text("<testsuite>" + "".join(cases) + "</testsuite>")
    result = subprocess.run(
        [
            os.sys.executable,
            str(REPO_ROOT / "tools/ci/gpu_runtime_gate.py"),
            "--junit",
            str(junit),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if bad_case is None else 1), result.stdout


@requires_mpicc
def test_mpi_probe_exports_selected_environment(tmp_path):
    launcher = tmp_path / "mpirun"
    launcher.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  echo "mpirun (Open MPI) 4.1.5"
  exit 0
fi
printf 'host-a\\nhost-b\\n'
"""
    )
    launcher.chmod(0o755)
    env = os.environ.copy()
    env["MPI_PROBE_MPIRUN"] = str(launcher)
    result = subprocess.run(
        [str(REPO_ROOT / "tools/ci/mpi_probe.sh")],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    export_line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("MPI_PROBE_EXPORT ")
    )
    assert "SPINGNN_MPIRUN_EXTRA_ENV=" in export_line
    assert " env=" not in export_line


def test_mpi_prefix_applies_probe_environment(monkeypatch, tmp_path):
    launcher = tmp_path / "mpirun"
    launcher.write_text(
        """#!/usr/bin/env bash
[[ "${UCX_TLS:-}" == "self,tcp" ]] || exit 9
printf 'host-a\\nhost-b\\n'
"""
    )
    launcher.chmod(0o755)
    monkeypatch.setenv("SPINGNN_MPIRUN", str(launcher))
    monkeypatch.setenv(
        "SPINGNN_MPIRUN_EXTRA_ENV",
        "UCX_TLS=self,tcp UCX_POSIX_USE_PROC_LINK=n",
    )
    environment: dict[str, str] = {}
    prefix = lammps_conftest.mpi_prefix.__wrapped__(environment)
    assert environment["UCX_TLS"] == "self,tcp"
    assert environment["UCX_POSIX_USE_PROC_LINK"] == "n"
    assert prefix(2)[0] == str(launcher)


def test_mpi_prefix_oversubscribe_only_for_openmpi(monkeypatch, tmp_path):
    """--oversubscribe must be added for Open MPI but not Intel MPI (hydra).

    Regression: A multi-node MPI run failed every MPI test with
    "unrecognized argument oversubscribe" from Intel's hydra launcher.
    """
    for flavor, version_line, expect_flag in (
        ("openmpi", "mpirun (Open MPI) 4.1.5", True),
        ("intel", "Intel(R) MPI Library for Linux* OS, Version 2021.13", False),
    ):
        launcher = tmp_path / f"mpirun-{flavor}"
        launcher.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "$1" == "--version" ]]; then\n'
            f"  echo '{version_line}'\n"
            "  exit 0\n"
            "fi\n"
            "printf 'host-a\\nhost-b\\n'\n"
        )
        launcher.chmod(0o755)
        monkeypatch.setenv("SPINGNN_MPIRUN", str(launcher))
        monkeypatch.delenv("SPINGNN_MPIRUN_EXTRA_ARGS", raising=False)
        monkeypatch.delenv("SPINGNN_MPIRUN_EXTRA_ENV", raising=False)
        prefix = lammps_conftest.mpi_prefix.__wrapped__({})
        assert ("--oversubscribe" in prefix(4)) is expect_flag, flavor


def test_deployed_pairs_share_local_rank_cuda_selector():
    paths = (
        "lammps/src/pair_allegro.cpp",
        "lammps/src/pair_nequip.cpp",
        "lammps/src/SPIN/pair_allegro_spin.cpp",
        "lammps/src/SPIN/pair_nequip_spin.cpp",
    )
    for relative_path in paths:
        source = (REPO_ROOT / relative_path).read_text()
        assert '#include "torch_device.h"' in source, relative_path
        assert "spingnn_torch_device(world, error)" in source, relative_path


def test_verified_copy_uses_rank_local_errors_for_rank_local_io():
    source = (REPO_ROOT / "lammps/src/deploy_metadata.h").read_text()
    copy_source = source.split("create_verified_pt2_private_copy", 1)[1].split(
        "load_deploy_metadata_stream", 1
    )[0]
    for message in (
        "Cannot create private PT2 package copy",
        "Cannot unlink private PT2 package copy",
        "Failed to create private PT2 package copy",
    ):
        before_message = copy_source.split(message, 1)[0]
        last_error_call = "error->" + before_message.rsplit("error->", 1)[-1]
        assert "error->one(" in last_error_call
    sha_branch = copy_source.split("PT2 package SHA-256 does not match its sidecar", 1)[
        0
    ]
    last_error_call = "error->" + sha_branch.rsplit("error->", 1)[-1]
    # A private copy can differ on one rank; a collective error can deadlock
    # peers that have already entered their model or integration work.
    assert "error->one(" in last_error_call


def test_cuda_cubin_capability_compatibility_is_directional(tmp_path):
    compiler = shutil.which("c++")
    assert compiler is not None
    source = tmp_path / "cuda-compat.cpp"
    executable = tmp_path / "cuda-compat"
    source.write_text(
        """#include "pt2_cuda_compat.h"
#include <cassert>
int main() {
  using LAMMPS_NS::pt2_cubin_capability_compatible;
  assert(pt2_cubin_capability_compatible("8.0", "8.0"));
  assert(pt2_cubin_capability_compatible("8.0", "8.6"));
  assert(!pt2_cubin_capability_compatible("8.6", "8.0"));
  assert(!pt2_cubin_capability_compatible("8.0", "9.0"));
  assert(!pt2_cubin_capability_compatible("bad", "8.6"));
}
"""
    )
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(REPO_ROOT / "lammps/src"),
            str(source),
            "-o",
            str(executable),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout
    subprocess.run([str(executable)], check=True)


def _incomplete_kokkos_prefix(tmp_path, architectures):
    prefix = tmp_path / "kokkos"
    (prefix / "include").mkdir(parents=True)
    (prefix / "lib/cmake/Kokkos").mkdir(parents=True)
    (prefix / "bin").mkdir(parents=True)
    (prefix / "include/KokkosCore_config.h").write_text(
        """#define KOKKOS_VERSION 30702
#define KOKKOS_ENABLE_SERIAL
#define KOKKOS_ENABLE_CUDA
#define KOKKOS_ENABLE_CUDA_LAMBDA
"""
        + "".join(f"#define KOKKOS_ARCH_{arch}\n" for arch in architectures)
    )
    (prefix / "lib/cmake/Kokkos/KokkosConfig.cmake").write_text("# incomplete\n")
    wrapper = prefix / "bin/nvcc_wrapper"
    wrapper.write_text("#!/usr/bin/env bash\nexit 99\n")
    wrapper.chmod(0o755)
    return prefix


@pytest.mark.parametrize(
    ("selected_arch", "installed_arch"),
    (("AMPERE80", "AMPERE80"), ("HOPPER90", "HOPPER90"), (None, "HOPPER90")),
)
def test_incomplete_kokkos_prefix_is_rejected(tmp_path, selected_arch, installed_arch):
    prefix = _incomplete_kokkos_prefix(tmp_path, (installed_arch,))
    env = os.environ.copy()
    env.pop("SPINGNN_KOKKOS_ARCH", None)
    if selected_arch is not None:
        env["SPINGNN_KOKKOS_ARCH"] = selected_arch

    result = subprocess.run(
        [str(REPO_ROOT / "tools/ci/install_kokkos_cuda.sh"), str(prefix)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "installed Kokkos library is missing: libkokkoscore" in result.stdout
    assert "refusing to overwrite non-empty invalid Kokkos prefix" in result.stdout


@pytest.mark.parametrize("architecture", ("VOLTA70", "hopper90", "AMPERE80;HOPPER90"))
def test_kokkos_installer_rejects_unsupported_architecture(tmp_path, architecture):
    prefix = tmp_path / "must-not-be-created"
    env = os.environ.copy()
    env["SPINGNN_KOKKOS_ARCH"] = architecture
    env["SPINGNN_KOKKOS_TARBALL"] = str(tmp_path / "must-not-be-read.tar.gz")
    result = subprocess.run(
        [str(REPO_ROOT / "tools/ci/install_kokkos_cuda.sh"), str(prefix)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert (
        "SPINGNN_KOKKOS_ARCH must be AMPERE80, HOPPER90, ADA89 or BLACKWELL120"
        in result.stderr
    )
    assert not prefix.exists()


@pytest.mark.parametrize(
    ("selected_arch", "installed_arches", "expected_error"),
    (
        ("HOPPER90", ("AMPERE80",), "missing KOKKOS_ARCH_HOPPER90"),
        ("AMPERE80", ("HOPPER90",), "missing KOKKOS_ARCH_AMPERE80"),
        (
            "HOPPER90",
            ("AMPERE80", "HOPPER90"),
            "multiple supported CUDA architectures",
        ),
    ),
)
def test_kokkos_installer_rejects_wrong_architecture_prefix(
    tmp_path, selected_arch, installed_arches, expected_error
):
    prefix = _incomplete_kokkos_prefix(tmp_path, installed_arches)
    env = os.environ.copy()
    env["SPINGNN_KOKKOS_ARCH"] = selected_arch
    result = subprocess.run(
        [str(REPO_ROOT / "tools/ci/install_kokkos_cuda.sh"), str(prefix)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert expected_error in result.stderr
    assert "refusing to overwrite non-empty invalid Kokkos prefix" in result.stderr
    assert "installed Kokkos library is missing" not in result.stderr


def _run_kokkos_build_mode(tmp_path, prefix):
    # Execute production argument selection before torch/CMake side effects.
    script = (REPO_ROOT / "tools/lammps/build_lammps.sh").read_text()
    mode_script = tmp_path / "build-mode.sh"
    mode_script.write_text(
        script.split("BUILD_DIR=${2:-", 1)[0] + '\nprintf "%s\\n" "${KOKKOS_ARGS[@]}"\n'
    )
    env = os.environ.copy()
    env["KOKKOS_ROOT"] = str(prefix)
    return subprocess.run(
        ["bash", str(mode_script), "kokkos-cuda"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("architectures", ((), ("VOLTA70",), ("AMPERE80", "HOPPER90")))
def test_kokkos_build_rejects_missing_or_ambiguous_architecture(
    tmp_path, architectures
):
    prefix = _incomplete_kokkos_prefix(tmp_path, architectures)
    result = _run_kokkos_build_mode(tmp_path, prefix)
    assert result.returncode == 2
    assert "exactly one supported Kokkos CUDA architecture" in result.stderr


@pytest.mark.parametrize(
    ("architecture", "cuda_arch"), (("AMPERE80", "sm_80"), ("HOPPER90", "sm_90"))
)
def test_kokkos_build_uses_installed_architecture(tmp_path, architecture, cuda_arch):
    prefix = _incomplete_kokkos_prefix(tmp_path, (architecture,))
    result = _run_kokkos_build_mode(tmp_path, prefix)
    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert f"CMAKE_CXX_FLAGS=-arch={cuda_arch}" in arguments
    assert f"Kokkos_ROOT={prefix}" in arguments
    assert "Kokkos_ENABLE_CUDA=on" in arguments
    assert "EXTERNAL_KOKKOS=yes" in arguments


def _lammps_build_preflight_environment(tmp_path, torch_cuda="12.8"):
    # Run the full build entry point with isolated torch metadata and stop at
    # CMake configuration. Reaching this sentinel is not a successful build.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for utility in ("dirname", "awk", "grep"):
        (bin_dir / utility).symlink_to(shutil.which(utility))
    cmake = bin_dir / "cmake"
    cmake.write_text("#!/bin/sh\necho CMAKE_CONFIGURE_REACHED\nexit 86\n")
    cmake.chmod(0o755)
    torch_dir = tmp_path / "packages" / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "__init__.py").write_text(
        "from types import SimpleNamespace\n"
        "__version__ = '2.11.0'\n"
        f"version = SimpleNamespace(cuda={torch_cuda!r})\n"
        f"utils = SimpleNamespace(cmake_prefix_path={str(torch_dir)!r})\n"
    )
    env = os.environ.copy()
    for variable in (
        "CUDACXX",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDAToolkit_ROOT",
        "KOKKOS_ROOT",
    ):
        env.pop(variable, None)
    env.update(
        PATH=str(bin_dir),
        PYTHON=os.sys.executable,
        PYTHONPATH=str(torch_dir.parent),
        JOBS="1",
        USE_CCACHE="0",
    )
    return env


def _run_lammps_build_preflight(tmp_path, env, mode="serial"):
    return subprocess.run(
        [
            "/bin/bash",
            str(REPO_ROOT / "tools/lammps/build_lammps.sh"),
            mode,
            str(tmp_path / "build"),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _assert_lammps_build_environment_guidance(result):
    assert "tools/ci/install_cpu_env.sh" in result.stderr
    assert "tools/ci/gpu_lammps_tests.sh" in result.stderr
    assert "CUDA 12.8" in result.stderr


@pytest.mark.parametrize("mode", ("serial", "mpi", "kokkos-serial", "kokkos-cuda"))
def test_lammps_build_preflight_rejects_missing_cuda_compiler(tmp_path, mode):
    env = _lammps_build_preflight_environment(tmp_path)
    if mode == "kokkos-cuda":
        env["KOKKOS_ROOT"] = str(_incomplete_kokkos_prefix(tmp_path, ("HOPPER90",)))
    result = _run_lammps_build_preflight(tmp_path, env, mode)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "CUDA" in result.stderr
    assert "CMAKE_CONFIGURE_REACHED" not in result.stdout
    _assert_lammps_build_environment_guidance(result)


@pytest.mark.parametrize("failure", ("missing", "not_executable", "broken", "quoting"))
def test_lammps_build_preflight_rejects_invalid_cudacxx(tmp_path, failure):
    env = _lammps_build_preflight_environment(tmp_path)
    # An explicit broken CUDACXX must not be rescued by a usable PATH compiler.
    nvcc = Path(env["PATH"]) / "nvcc"
    nvcc.write_text("#!/bin/sh\necho usable-path-nvcc\n")
    nvcc.chmod(0o755)
    compiler = tmp_path / "selected-nvcc"
    if failure in ("not_executable", "broken"):
        compiler.write_text("#!/bin/sh\necho broken-cuda-driver >&2\nexit 73\n")
        if failure == "broken":
            compiler.chmod(0o755)
    env["CUDACXX"] = str(compiler) if failure != "quoting" else "'unterminated"
    result = _run_lammps_build_preflight(tmp_path, env)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "CUDACXX" in result.stderr
    assert "CMAKE_CONFIGURE_REACHED" not in result.stdout
    if failure == "broken":
        assert "broken-cuda-driver" in result.stderr
    _assert_lammps_build_environment_guidance(result)


@pytest.mark.parametrize("failure", ("missing_python", "broken_torch"))
def test_lammps_build_preflight_rejects_unusable_python(tmp_path, failure):
    env = _lammps_build_preflight_environment(tmp_path)
    if failure == "missing_python":
        env["PYTHON"] = str(tmp_path / "missing-python")
    else:
        (Path(env["PYTHONPATH"]) / "torch" / "__init__.py").write_text(
            "raise ImportError('broken-torch-installation')\n"
        )
    result = _run_lammps_build_preflight(tmp_path, env)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "PYTHON=" in result.stderr
    assert "CMAKE_CONFIGURE_REACHED" not in result.stdout
    if failure == "broken_torch":
        assert "broken-torch-installation" in result.stderr
    _assert_lammps_build_environment_guidance(result)


@pytest.mark.parametrize("mode", ("serial", "mpi", "kokkos-serial"))
def test_lammps_build_preflight_cpu_torch_needs_no_cuda_toolkit(tmp_path, mode):
    env = _lammps_build_preflight_environment(tmp_path, torch_cuda=None)
    result = _run_lammps_build_preflight(tmp_path, env, mode)
    assert result.returncode == 86, result.stdout + result.stderr
    assert "CMAKE_CONFIGURE_REACHED" in result.stdout
    # Configuration failures retain their original status and gain guidance.
    _assert_lammps_build_environment_guidance(result)


@pytest.mark.parametrize("selection", ("path", "cudacxx_with_arguments"))
def test_lammps_build_preflight_accepts_usable_cuda_compiler(tmp_path, selection):
    env = _lammps_build_preflight_environment(tmp_path)
    if selection == "path":
        compiler = Path(env["PATH"]) / "nvcc"
    else:
        compiler = tmp_path / "compiler with spaces"
        env["CUDACXX"] = f"'{compiler}' --keep"
    compiler.write_text('#!/bin/sh\necho CUDA_COMPILER_PROBED "$@"\n')
    compiler.chmod(0o755)
    result = _run_lammps_build_preflight(tmp_path, env)
    assert result.returncode == 86, result.stdout + result.stderr
    assert "CMAKE_CONFIGURE_REACHED" in result.stdout
    expected_args = "--version" if selection == "path" else "--keep --version"
    assert f"CUDA_COMPILER_PROBED {expected_args}" in result.stdout
    _assert_lammps_build_environment_guidance(result)


def test_lammps_build_gate_requires_all_deployed_pair_styles():
    build_script = (REPO_ROOT / "tools/lammps/build_lammps.sh").read_text()
    feature_gate = build_script.split("# Feature gate:", 1)[1]
    for style in (
        "'allegro'",
        "'nequip'",
        "'spin/allegro_spin'",
        "'spin/nequip_spin'",
    ):
        assert style in feature_gate


def test_pt2_production_documentation_keeps_checkpoint_binding_contract():
    paths = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs/PT2_MIGRATION.md",
        REPO_ROOT / "lammps/examples/spingnn/TUTORIAL_LAMMPS.md",
        REPO_ROOT / "skills/spingnn-usage/SKILL.md",
    )
    for path in paths:
        text = path.read_text()
        assert ".pt2` package is tied to the checkpoint" in text, path
        assert "package SHA-256" in text, path
    migration = (REPO_ROOT / "docs/PT2_MIGRATION.md").read_text()
    assert "experimental candidate" not in migration
    assert "currently builds PT2 on CPU" not in migration
    assert "preserves nested outer `RescaleOutput` layers" in migration
    assert "PT2 learned scaling: included" in migration


def _mpi_probe_payload_source() -> str:
    """Extract the C payload exactly as the probe compiles it (the heredoc)."""
    script = (REPO_ROOT / "tools/ci/mpi_probe.sh").read_text()
    body = script.split("<<'CSRC'\n", 1)[1]
    return body.split("\nCSRC\n", 1)[0]


def test_mpi_probe_payload_is_a_real_mpi_program():
    """The probe must not validate a transport with a non-MPI payload.

    It used to launch `hostname`, which never calls MPI_Init. The probe then
    only proved the launcher could spawn N processes, so the `default`
    combination always passed and the UCX-avoiding fallbacks below it were
    unreachable even when LAMMPS subsequently failed in MPI_Init with
    `ucp_ep_create(proc=0) failed: Address not valid`.

    Inspect the extracted C payload because shell comments can contain the
    expected MPI calls even when the compiled payload does not perform them.
    """
    payload = _mpi_probe_payload_source()
    # The payload must initialise MPI and force inter-rank endpoint setup;
    # endpoint creation is the step that fails under a broken UCX stack.
    assert "MPI_Init" in payload
    assert "MPI_Allreduce" in payload
    # A failed init/allreduce must be a nonzero exit, not a fallthrough.
    assert "MPI_SUCCESS" in payload
    script = (REPO_ROOT / "tools/ci/mpi_probe.sh").read_text()
    # And the probe must launch exactly that compiled payload.
    assert '-n "$RANKS" hostname' not in script
    assert '-n "$RANKS" "$PAYLOAD_BIN"' in script


@requires_mpicc
def test_mpi_probe_falls_back_when_the_default_transport_fails(tmp_path):
    """The fallback chain must be reachable, not dead code.

    A launcher that only works with an explicit `--mca pml ob1` must drive the
    probe past `default` and make it select and export `ob1`.
    """
    launcher = tmp_path / "mpirun"
    launcher.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  echo "mpirun (Open MPI) 4.1.5"
  exit 0
fi
for arg in "$@"; do
  if [[ "$arg" == "ob1" ]]; then
    printf 'host-a\\nhost-b\\n'
    exit 0
  fi
done
echo "simulated transport failure" >&2
exit 1
"""
    )
    launcher.chmod(0o755)
    env = os.environ.copy()
    env["MPI_PROBE_MPIRUN"] = str(launcher)
    result = subprocess.run(
        [str(REPO_ROOT / "tools/ci/mpi_probe.sh")],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "combination=default" in result.stdout
    assert "status=FAIL" in result.stdout, "default attempt should have failed"
    result_line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("MPI_PROBE_RESULT status=PASS ")
    )
    assert "combination=ob1" in result_line
    export_line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("MPI_PROBE_EXPORT ")
    )
    assert "pml" in export_line and "ob1" in export_line
    # The selection must also travel as environment, not only as launcher args:
    # `build_lammps.sh` runs `lmp -h` directly and never sees mpirun arguments.
    assert "OMPI_MCA_pml=ob1" in export_line


def _run_full_fx_gate(record_lines, minimum="4.4", gpu_expect="h100"):
    """Run the production require_full_fx_benchmark gate on given log content."""
    import tempfile

    script = (REPO_ROOT / "tools/ci/gpu_full_tests.sh").read_text()
    # Execute the same parser that the shell gate uses.
    inside = script.split("require_full_fx_benchmark() {", 1)[1]
    parser = inside.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as log:
        log.write("\n".join(record_lines) + "\n")
        log_path = log.name
    env = os.environ.copy()
    env["SPINGNN_GPU_EXPECT"] = gpu_expect
    result = subprocess.run(
        [os.sys.executable, "-c", parser, log_path, minimum, "1e-5"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    os.unlink(log_path)
    return result


_VALID_RECORD_TAIL = (
    "eager_seconds=10 full_fx_seconds=1 max_loss_relative=1e-6 "
    "cache_entries=1 measurement=steady_state_median steps=19"
)


def test_full_fx_gate_rejects_non_line_start_records():
    """Only a line emitted by the producer may satisfy the gate.

    The producer prints the record with a LEADING newline, so it always owns
    its line; the parser therefore requires the marker at line start and must
    reject a record embedded in unrelated echoed text.
    """
    embedded = _run_full_fx_gate(
        [f"CAPTURED_ECHO FULL_FX_WALLTIME {_VALID_RECORD_TAIL}"]
    )
    assert embedded.returncode != 0, embedded.stdout
    clean = _run_full_fx_gate([f"FULL_FX_WALLTIME {_VALID_RECORD_TAIL}"])
    assert clean.returncode == 0, clean.stdout


def test_full_fx_producer_owns_its_line():
    """The producer must start the record on a new line."""
    source = (REPO_ROOT / "src/core/nequip/utils/unittests/parity_tests.py").read_text()
    assert '"\\nFULL_FX_WALLTIME ' in source, (
        "the FULL_FX_WALLTIME print must begin with a leading newline so pytest "
        "progress characters can never glue onto the record; the strict "
        "line-start parser depends on it"
    )


def test_full_fx_gate_rejects_non_finite_loss():
    """Reject non-finite and negative loss values."""
    for bad in ("nan", "inf", "-inf", "-1e-7"):
        record = (
            f"FULL_FX_WALLTIME eager_seconds=10 full_fx_seconds=1 "
            f"max_loss_relative={bad} cache_entries=1 "
            f"measurement=steady_state_median steps=19"
        )
        result = _run_full_fx_gate([record])
        assert result.returncode != 0, f"{bad}: {result.stdout}"


def test_full_fx_4090_mode_gates_on_calibrated_floors():
    """RTX 4090 floors reflect two consistent measurements
    (NequIP 6.098x/6.183x, Allegro 2.611x/2.667x) and
    are now HARD gates at ~30% under the measurements: 4.3x / 1.8x. The old
    calibration mode (slow records pass informationally) must be gone."""

    def record(speedup):
        return (
            f"FULL_FX_WALLTIME eager_seconds={speedup} full_fx_seconds=1 "
            "max_loss_relative=1e-6 cache_entries=1 "
            "measurement=steady_state_median steps=19"
        )

    ok = _run_full_fx_gate([record("4.3")], minimum="4.3", gpu_expect="4090")
    assert ok.returncode == 0, ok.stdout
    assert "FULL_FX_ACCEPT gpu=4090" in ok.stdout
    slow = _run_full_fx_gate([record("4.299")], minimum="4.3", gpu_expect="4090")
    assert slow.returncode != 0, slow.stdout
    assert (
        _run_full_fx_gate([record("1.8")], minimum="1.8", gpu_expect="4090").returncode
        == 0
    )
    assert (
        _run_full_fx_gate(
            [record("1.799")], minimum="1.8", gpu_expect="4090"
        ).returncode
        != 0
    )
    # Loss/cache gates stay hard too.
    bad_loss = (
        "FULL_FX_WALLTIME eager_seconds=10 full_fx_seconds=1 "
        "max_loss_relative=1e-4 cache_entries=1 "
        "measurement=steady_state_median steps=19"
    )
    assert _run_full_fx_gate([bad_loss], gpu_expect="4090").returncode != 0
    # And the shell wires exactly these floors for 4090.
    script = (REPO_ROOT / "tools/ci/gpu_full_tests.sh").read_text()
    assert "SPINGNN_NEQUIP_SPEED_FLOOR=4.3" in script
    assert "SPINGNN_ALLEGRO_SPEED_FLOOR=1.8" in script


def test_full_fx_h100_mode_keeps_nequip_and_allegro_speedup_floors():
    def record(speedup):
        return (
            f"FULL_FX_WALLTIME eager_seconds={speedup} full_fx_seconds=1 "
            "max_loss_relative=1e-6 cache_entries=1 "
            "measurement=steady_state_median steps=19"
        )

    assert _run_full_fx_gate([record("4.4")], minimum="4.4").returncode == 0
    assert _run_full_fx_gate([record("4.399")], minimum="4.4").returncode != 0
    assert _run_full_fx_gate([record("2.0")], minimum="2.0").returncode == 0
    assert _run_full_fx_gate([record("1.999")], minimum="2.0").returncode != 0

    script = (REPO_ROOT / "tools/ci/gpu_full_tests.sh").read_text()
    assert "SPINGNN_NEQUIP_SPEED_FLOOR=4.4" in script
    assert "SPINGNN_ALLEGRO_SPEED_FLOOR=2.0" in script
    assert (
        'require_full_fx_benchmark "$suite_log" "$SPINGNN_NEQUIP_SPEED_FLOOR"' in script
    )
    assert (
        'require_full_fx_benchmark "$suite_log" "$SPINGNN_ALLEGRO_SPEED_FLOOR"'
        in script
    )


def test_informational_lane_fails_on_pytest_infrastructure_errors(tmp_path):
    """Treat pytest usage and empty-collection exits as errors.

    A usage/collection error must make the aggregate fail; exits 0/1 (tests
    ran) stay informational and never gate.
    """
    script = (REPO_ROOT / "tools/ci/gpu_full_tests.sh").read_text()
    assert "run_informational_suite" in script
    for exit_code, must_fail in ((0, False), (1, False), (4, True), (5, True)):
        fake = tmp_path / f"python-{exit_code}"
        fake.write_text(f"#!/usr/bin/env bash\nexit {exit_code}\n")
        fake.chmod(0o755)
        probe = tmp_path / f"probe-{exit_code}.sh"
        probe.write_text(
            "#!/usr/bin/env bash\n"
            "set -u\n"
            "declare -A results\n"
            "failed=0\n"
            'record_result() { results[$1]=$2; if [ "$2" != 0 ]; then '
            "failed=$((failed+1)); fi; }\n"
            f"PYTHON={fake}\n"
            + _extract_function("run_informational_suite")
            + "\nrun_informational_suite probe-lane . tests -m x\n"
            "echo FAILED_COUNT=$failed\n"
        )
        probe.chmod(0o755)
        result = subprocess.run(
            ["bash", str(probe)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        line = next(
            l for l in result.stdout.splitlines() if l.startswith("FAILED_COUNT=")
        )
        count = int(line.split("=", 1)[1])
        if must_fail:
            assert count > 0, f"exit {exit_code} not counted: {result.stdout}"
        else:
            assert count == 0, f"exit {exit_code} wrongly gated: {result.stdout}"


def _extract_function(name: str) -> str:
    script = (REPO_ROOT / "tools/ci/gpu_full_tests.sh").read_text()
    start = script.index(f"{name}() {{")
    depth = 0
    for index in range(start, len(script)):
        if script[index] == "{":
            depth += 1
        elif script[index] == "}":
            depth -= 1
            if depth == 0:
                return script[start : index + 1]
    raise AssertionError(f"unbalanced braces extracting {name}")


def test_workflow_keeps_the_two_explicit_load_bearing_steps():
    """Run the end-to-end and binary contracts outside the parity selectors."""
    import yaml

    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["build-test-lammps"]["steps"]
    by_run = {step.get("run", ""): step for step in steps if isinstance(step, dict)}

    e2e_step = next(
        (
            step
            for run, step in by_run.items()
            if "tests_lammps/test_e2e_train_lammps.py" in run
        ),
        None,
    )
    assert e2e_step is not None, "the explicit e2e step was removed from ci.yml"
    assert e2e_step.get("if") == "matrix.mode == 'serial'"
    assert e2e_step["env"]["SPINGNN_REQUIRE_LAMMPS"] == "1"
    assert "lammps-serial/lmp" in e2e_step["env"]["SPINGNN_LAMMPS_BIN"]

    contract_step = next(
        (
            step
            for run, step in by_run.items()
            if "tests_lammps/test_ci_contracts.py" in run
        ),
        None,
    )
    assert (
        contract_step is not None
    ), "the explicit contract step was removed from ci.yml"
    assert contract_step.get("if") == "matrix.mode == 'mpi'"
    assert contract_step["env"]["SPINGNN_REQUIRE_LAMMPS"] == "1"
    assert "lammps-serial/lmp" in contract_step["env"]["SPINGNN_LAMMPS_BIN"]
    assert "lammps-mpi/lmp" in contract_step["env"]["SPINGNN_LAMMPS_MPI_BIN"]


def test_gpu_full_lane_gives_v100_a_stack_diagnosis(tmp_path):
    """v100 must be rejected with the measured stack incompatibility, not a
    generic unknown-value error: torch 2.11.0+cu128 ships no sm_70 SASS and
    its minimum PTX is compute_75 (extracted from libtorch_cuda.so), and PTX
    only JITs forward, so Volta cannot run any kernel in this build."""
    environment = tmp_path / "fake-env"
    python = environment / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/usr/bin/env bash\nexit 5\n")
    python.chmod(0o755)
    env = os.environ.copy()
    env["SPINGNN_ENV"] = str(environment)
    env["SPINGNN_CI_EVIDENCE_DIR"] = str(tmp_path / "evidence")
    env["SPINGNN_GPU_EXPECT"] = "v100"
    result = subprocess.run(
        [str(REPO_ROOT / "tools/ci/gpu_full_tests.sh")],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stdout
    assert "no sm_70 SASS" in result.stdout
    assert "compute_75" in result.stdout


def test_full_fx_a100_mode_calibrates_like_every_uncalibrated_arch():
    """a100 has no measured floors yet: valid records emit FULL_FX_CALIBRATION
    while the loss/cache/schema gates stay hard, same policy as 4090."""
    slow_record = (
        "FULL_FX_WALLTIME eager_seconds=1 full_fx_seconds=10 "
        "max_loss_relative=1e-6 cache_entries=1 "
        "measurement=steady_state_median steps=19"
    )
    calibration = _run_full_fx_gate([slow_record], gpu_expect="a100")
    assert calibration.returncode == 0, calibration.stdout
    assert "FULL_FX_CALIBRATION gpu=a100 speedup=0.100000" in calibration.stdout
    bad_loss = (
        "FULL_FX_WALLTIME eager_seconds=10 full_fx_seconds=1 "
        "max_loss_relative=1e-4 cache_entries=1 "
        "measurement=steady_state_median steps=19"
    )
    rejected = _run_full_fx_gate([bad_loss], gpu_expect="a100")
    assert rejected.returncode != 0, rejected.stdout


def test_gpu_probe_expectations_lock_name_and_capability():
    """The probe's arch table is a trust anchor; lock the exact pairs."""
    script = (REPO_ROOT / "tools/ci/gpu_full_tests.sh").read_text()
    assert '"h100": ("H100", (9, 0)),' in script
    assert '"a100": ("A100", (8, 0)),' in script
    assert '"4090": ("4090", (8, 9)),' in script
    assert '"5090": ("5090", (12, 0)),' in script
