"""Source selection must survive foreign cwd, stale installs, and subprocesses."""

import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

import pytest
import yaml


REPO = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO / "_spingnn_source.py"
LAUNCHER = REPO / "tools/ci/source_python.py"
SOURCE_ROOTS = (REPO / "src/core", REPO / "src/models")


def _foreign_env(tmp_path):
    foreign = tmp_path / "snapshot"
    for name in ("nequip", "allegro"):
        package = foreign / name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(
            'raise RuntimeError("SNAPSHOT_IMPORTED")\n'
        )
    env = os.environ.copy()
    inherited = env.get("PYTHONPATH", "").split(os.pathsep)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(foreign)]
        + [
            path
            for path in inherited
            if path and Path(path).resolve() not in (*SOURCE_ROOTS, REPO)
        ]
    )
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("SPINGNN_REQUIRE_LAMMPS", None)
    return env


def _python(tmp_path, args, env=None):
    return subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=tmp_path,
        env=env if env is not None else _foreign_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
    )


def test_bootstrap_is_lazy_idempotent_and_exports_absolute_paths(tmp_path):
    result = _python(
        tmp_path,
        [
            "-c",
            f"import runpy, sys, os, json; "
            f"activate = runpy.run_path({str(BOOTSTRAP)!r})['activate_source_tree']; "
            "first = activate(); second = activate(); "
            "assert first == second; assert 'torch' not in sys.modules; "
            "print(json.dumps({'origins': first, 'path': sys.path, "
            "'env': os.environ['PYTHONPATH'].split(os.pathsep)}))",
        ],
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    roots = list(map(str, SOURCE_ROOTS))
    assert payload["path"][:2] == roots
    assert payload["env"][:2] == roots
    for root in roots:
        assert payload["env"].count(root) == 1
        assert payload["path"].count(root) == 1
    assert payload["origins"] == {
        "nequip": str(SOURCE_ROOTS[0] / "nequip/__init__.py"),
        "allegro": str(SOURCE_ROOTS[1] / "allegro/__init__.py"),
    }


@pytest.mark.parametrize("name", ("nequip", "allegro", "nequip.data", "allegro.nn"))
def test_bootstrap_rejects_already_loaded_foreign_modules(tmp_path, name):
    result = _python(
        tmp_path,
        [
            "-c",
            "import runpy, sys, types; "
            f"module = types.ModuleType({name!r}); module.__file__ = '/old/snapshot.py'; "
            f"sys.modules[{name!r}] = module; "
            f"runpy.run_path({str(BOOTSTRAP)!r})['activate_source_tree']()",
        ],
    )
    assert result.returncode != 0
    assert f"source-exactness violation: {name} already loaded" in result.stdout
    assert "source_python.py" in result.stdout


def test_launcher_real_imports_and_child_use_checkout_from_foreign_cwd(tmp_path):
    probe = (
        "import json, nequip, allegro; "
        "print(json.dumps([nequip.__file__, allegro.__file__]))"
    )
    result = _python(
        tmp_path,
        [
            LAUNCHER,
            "-c",
            f"{probe}; import subprocess, sys; "
            f"subprocess.run([sys.executable, '-c', {probe!r}], check=True)",
        ],
    )
    assert result.returncode == 0, result.stdout
    paths = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("[")
    ]
    assert len(paths) == 2, result.stdout
    for origins in paths:
        assert Path(origins[0]).is_relative_to(SOURCE_ROOTS[0])
        assert Path(origins[1]).is_relative_to(SOURCE_ROOTS[1])


@pytest.mark.parametrize("mode", ("console", "module", "file"))
def test_launcher_deploy_help_uses_source_before_import(tmp_path, mode):
    target = {
        "console": ["nequip-deploy"],
        "module": ["-m", "nequip.scripts.deploy"],
        "file": [REPO / "src/core/nequip/scripts/deploy.py"],
    }[mode]
    result = _python(tmp_path, [LAUNCHER, *target, "--help"])
    assert result.returncode == 0, result.stdout
    assert "usage:" in result.stdout
    assert "SNAPSHOT_IMPORTED" not in result.stdout


@pytest.mark.parametrize(
    "entry",
    (
        "tools/ci/gpu_pt2_threeway.py",
        "tools/ci/pt2_smoke_triple.py",
        "example/validate_configs.py",
    ),
)
def test_standalone_tools_bootstrap_before_model_import(tmp_path, entry):
    result = _python(
        tmp_path,
        [
            "-c",
            f"import runpy; runpy.run_path({str(REPO / entry)!r}); "
            "import nequip, allegro; "
            f"assert nequip.__file__.startswith({str(SOURCE_ROOTS[0])!r}); "
            f"assert allegro.__file__.startswith({str(SOURCE_ROOTS[1])!r})",
        ],
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    "test_root", ("src/core/tests", "src/models/tests", "tests_lammps")
)
def test_root_conftest_protects_test_imports_and_subprocesses(tmp_path, test_root):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for filename in ("conftest.py", "_spingnn_source.py", "pytest.ini"):
        shutil.copyfile(REPO / filename, checkout / filename)
    for parent, name in (("src/core", "nequip"), ("src/models", "allegro")):
        package = checkout / parent / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("SOURCE_CHECKOUT = True\n")
    tests = checkout / test_root
    tests.mkdir(parents=True)
    (tests / "conftest.py").write_text(
        "import nequip, allegro\n"
        "assert nequip.SOURCE_CHECKOUT and allegro.SOURCE_CHECKOUT\n"
    )
    (tests / "test_child.py").write_text(
        "import subprocess, sys\n"
        "def test_child(tmp_path):\n"
        "    subprocess.run([sys.executable, '-c', "
        "'import nequip, allegro; assert nequip.SOURCE_CHECKOUT and allegro.SOURCE_CHECKOUT'], "
        "cwd=tmp_path, check=True)\n"
    )
    result = _python(tmp_path, ["-m", "pytest", tests, "-q", "-p", "no:cacheprovider"])
    assert result.returncode == 0, result.stdout
    assert "1 passed" in result.stdout


def test_e2e_subprocess_uses_explicit_source_environment(tmp_path, lammps_env):
    from test_e2e_train_lammps import _run

    result = _run(
        [
            sys.executable,
            "-c",
            "import nequip, allegro; "
            f"assert nequip.__file__.startswith({str(SOURCE_ROOTS[0])!r}); "
            f"assert allegro.__file__.startswith({str(SOURCE_ROOTS[1])!r})",
        ],
        cwd=tmp_path,
        env=lammps_env,
    )
    assert result.returncode == 0


@pytest.mark.parametrize("failed_suite", ("", "src/core", "src/models"))
def test_cpu_helper_keeps_source_paths_and_aggregates_suite_failures(
    tmp_path, failed_suite
):
    environment = tmp_path / "env"
    (environment / "bin").mkdir(parents=True)
    python = environment / "bin/python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    # Replace only pytest's workload. The real shell helper, source launcher,
    # package imports, warning-budget parser, and exit aggregation still run.
    pytest_package = tmp_path / "workload/pytest"
    pytest_package.mkdir(parents=True)
    (pytest_package / "__init__.py").touch()
    (pytest_package / "__main__.py").write_text(
        "import os, sys\nfrom pathlib import Path\nimport nequip, allegro\n"
        f"assert nequip.__file__.startswith({str(SOURCE_ROOTS[0])!r})\n"
        f"assert allegro.__file__.startswith({str(SOURCE_ROOTS[1])!r})\n"
        f"failed = Path.cwd().relative_to({str(REPO)!r}).as_posix() == {failed_suite!r}\n"
        "print('1 failed in 0.01s' if failed else '1 passed in 0.01s')\n"
        "sys.exit(7 if failed else 0)\n"
    )
    env = _foreign_env(tmp_path)
    env["SPINGNN_ENV"] = str(environment)
    env["PYTHONPATH"] = str(pytest_package.parent) + os.pathsep + env["PYTHONPATH"]
    result = subprocess.run(
        ["bash", str(REPO / "tools/ci/cpu_unit_tests.sh")],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
    )
    assert result.returncode == (1 if failed_suite else 0), result.stdout
    assert "SOURCE_EXACT nequip=" in result.stdout
    assert "CI_SUITE_START name=src/core" in result.stdout
    assert "CI_SUITE_START name=src/models" in result.stdout
    if failed_suite:
        assert (
            f"CI_SUITE_RESULT name={failed_suite} result=FAIL exit_code=7"
            in result.stdout
        )
        assert "CI_SUMMARY suites_passed=1 suites_failed=1" in result.stdout
    else:
        assert "CI_SUMMARY suites_passed=2 suites_failed=0" in result.stdout


def test_source_regressions_and_all_allegro_modules_are_selected_in_ci():
    workflow = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["build-test-lammps"]["steps"]
    assert any(
        "tests_lammps/test_source_exact.py" in step.get("run", "") for step in steps
    )
    assert workflow["jobs"]["test-python"]["strategy"]["matrix"] == (
        "${{ fromJSON(needs.inventory.outputs.matrix) }}"
    )
    assert any(
        "tools/ci/test_inventory.py --matrix" in step.get("run", "")
        for step in workflow["jobs"]["inventory"]["steps"]
    )
    matrix = json.loads((REPO / "tools/ci/test-lanes.json").read_text())["python_lanes"]
    selected = set()
    for shard in matrix:
        for argument in shard["tests"]:
            if not argument.startswith("src/models/"):
                continue
            path = REPO / argument
            selected.update(path.rglob("test_*.py") if path.is_dir() else [path])
    assert set((REPO / "src/models/tests").rglob("test_*.py")) <= selected
