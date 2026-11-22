"""Environment contract and session fixtures for LAMMPS tests."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest


TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TEST_ROOT.parent
NEQUIP_SOURCE_ROOT = (REPO_ROOT / "src/core").resolve()
ALLEGRO_SOURCE_ROOT = (REPO_ROOT / "src/models").resolve()
SOURCE_ROOTS = (NEQUIP_SOURCE_ROOT, ALLEGRO_SOURCE_ROOT)

# The shared environment contains stale non-editable copies. Source roots must win.
for source_root in reversed(SOURCE_ROOTS):
    source_text = str(source_root)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)

import allegro  # noqa: E402
import nequip  # noqa: E402
import torch  # noqa: E402
from nequip.utils.test import set_irreps_debug  # noqa: E402


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _assert_source_exact_imports() -> None:
    nequip_path = Path(nequip.__file__).resolve()
    allegro_path = Path(allegro.__file__).resolve()
    if not _is_below(nequip_path, NEQUIP_SOURCE_ROOT):
        raise pytest.UsageError(
            f"non-source nequip import rejected: {nequip_path}; expected below {NEQUIP_SOURCE_ROOT}"
        )
    if not _is_below(allegro_path, ALLEGRO_SOURCE_ROOT):
        raise pytest.UsageError(
            f"non-source allegro import rejected: {allegro_path}; "
            f"expected below {ALLEGRO_SOURCE_ROOT}"
        )


_assert_source_exact_imports()


def _required() -> bool:
    return os.environ.get("SPINGNN_REQUIRE_LAMMPS") == "1"


def _allow_lammps_skip() -> bool:
    return os.environ.get("SPINGNN_ALLOW_LAMMPS_SKIP") == "1"


def _unavailable_lammps_contract(kind: str, reason: str) -> None:
    message = f"{kind} LAMMPS environment contract is unavailable: {reason}"
    if _allow_lammps_skip():
        pytest.skip(message)
    raise pytest.UsageError(
        f"{message}; set SPINGNN_ALLOW_LAMMPS_SKIP=1 only for an explicit "
        "collection/developer opt-out"
    )


def _configured_binary(name: str) -> tuple[Path | None, str | None]:
    raw = os.environ.get(name)
    if not raw:
        return None, f"{name} is not set"
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        return None, f"{name} does not name a file: {path}"
    if not os.access(path, os.X_OK):
        return None, f"{name} is not executable: {path}"
    return path, None


def pytest_configure(config: pytest.Config) -> None:
    _, reason = _configured_binary("SPINGNN_LAMMPS_BIN")
    if reason and _required():
        raise pytest.UsageError(
            f"SPINGNN_REQUIRE_LAMMPS=1 makes the LAMMPS contract mandatory: {reason}"
        )


@pytest.fixture(autouse=True)
def _standalone_process_state():
    """Restore standalone-run process state around every test.

    The acceptance gate runs src/core, src/models, and this suite in one
    pytest process. src/core's unittest conftest arms nequip's global
    irreps-debug forward hooks at import time, and earlier suites can leave
    torch's default dtype at float64; neither happens when this suite runs
    alone, and both change results here (the no-cell deploy path legitimately
    emits an empty 1-D stress sentinel that the armed hooks reject, and bare
    ``torch.tensor(...)`` constructors follow the ambient default dtype).
    """
    set_irreps_debug(False)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.fixture(scope="session")
def source_import_paths() -> dict[str, Path]:
    _assert_source_exact_imports()
    return {
        "nequip": Path(nequip.__file__).resolve(),
        "allegro": Path(allegro.__file__).resolve(),
    }


@pytest.fixture(scope="session")
def lammps_bin() -> Path:
    path, reason = _configured_binary("SPINGNN_LAMMPS_BIN")
    if path is None:
        _unavailable_lammps_contract(
            "serial", reason or "SPINGNN_LAMMPS_BIN is unavailable"
        )
        raise AssertionError("unreachable")
    return path


@pytest.fixture(scope="session")
def lammps_env() -> dict[str, str]:
    env = os.environ.copy()
    source_pythonpath = os.pathsep.join(str(path) for path in SOURCE_ROOTS)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_pythonpath
        if not existing_pythonpath
        else os.pathsep.join((source_pythonpath, existing_pythonpath))
    )
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    existing_ld_path = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = (
        str(torch_lib)
        if not existing_ld_path
        else os.pathsep.join((str(torch_lib), existing_ld_path))
    )
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.geteuid() == 0:
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    return env


@pytest.fixture(scope="session")
def lammps_pt2_target_device(lammps_bin: Path, lammps_env: dict[str, str]) -> str:
    """Return the AOTI target required by the configured LAMMPS binary."""

    ldd = shutil.which("ldd")
    if ldd is None:
        _unavailable_lammps_contract(
            "serial", "cannot determine PT2 target device because ldd is unavailable"
        )
        raise AssertionError("unreachable")
    linked = subprocess.run(
        [ldd, str(lammps_bin)],
        env=lammps_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    if linked.returncode != 0:
        _unavailable_lammps_contract(
            "serial",
            "cannot determine PT2 target device: "
            f"ldd exited with {linked.returncode}: {linked.stdout[-1000:]}",
        )
        raise AssertionError("unreachable")
    cuda_libtorch = any(
        library in linked.stdout for library in ("libtorch_cuda", "libc10_cuda")
    )
    return "cuda" if cuda_libtorch and torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def model_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    from model_factory import build_all_models

    return build_all_models(tmp_path_factory.mktemp("deployed_models"))


@pytest.fixture(scope="session")
def pt2_artifact_factory(tmp_path_factory: pytest.TempPathFactory):
    from helpers import CASES
    from model_factory import build_deployment_model_pair

    output_dir = tmp_path_factory.mktemp("pt2_models")
    artifacts = {}

    def build(case_name: str, target_device: str = "cpu"):
        key = (case_name, target_device)
        if key in artifacts:
            return artifacts[key]
        case = CASES[case_name]
        if target_device == "cuda" and not torch.cuda.is_available():
            pytest.skip(
                "CUDA-target PT2 export required by LAMMPS, but "
                "torch.cuda is unavailable"
            )
        artifacts[key] = build_deployment_model_pair(
            case,
            output_dir / f"{case_name}.pth",
            output_dir / f"{case_name}-{target_device}.pt2",
            target_device=target_device,
        )
        return artifacts[key]

    return build


@pytest.fixture(scope="session")
def shifted_artifact_factory(tmp_path_factory: pytest.TempPathFactory):
    from helpers import CASES
    from model_factory import build_shifted_deployment_models

    output_dir = tmp_path_factory.mktemp("shifted_deployed_models")
    artifacts: dict[tuple[str, str], tuple[Path, Path]] = {}

    def build(
        target_device: str = "cpu", *, case_name: str = "nosoc"
    ) -> tuple[Path, Path]:
        if target_device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA-target shifted PT2 export requires torch.cuda")
        key = (case_name, target_device)
        if key not in artifacts:
            artifacts[key] = build_shifted_deployment_models(
                CASES[case_name],
                output_dir,
                target_device=target_device,
            )
        return artifacts[key]

    return build


@pytest.fixture(scope="session")
def schema_model_paths(
    model_paths: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Path]:
    from model_factory import resave_with_metadata
    from nequip.scripts import deploy

    output_dir = tmp_path_factory.mktemp("schema_models")
    return {
        "float64": resave_with_metadata(
            model_paths["nosoc"],
            output_dir / "nosoc-float64-metadata.pth",
            {deploy.DEFAULT_DTYPE_KEY: "float64"},
        ),
        "missing_spin": model_paths["bec"],
        "invalid_edge_prune": resave_with_metadata(
            model_paths["nosoc"],
            output_dir / "nosoc-invalid-edge-prune.pth",
            {deploy.EDGE_PRUNE_KEY: "invalid"},
        ),
        "missing_metadata": resave_with_metadata(
            model_paths["nosoc"],
            output_dir / "nosoc-no-metadata.pth",
            omit_all_metadata=True,
        ),
    }


@pytest.fixture(scope="session")
def lammps_mpi_bin() -> Path:
    path, reason = _configured_binary("SPINGNN_LAMMPS_MPI_BIN")
    if path is None:
        _unavailable_lammps_contract("MPI", reason or "unknown binary error")
        raise AssertionError("unreachable")
    return path


@pytest.fixture(scope="session")
def mpi_prefix(lammps_env: dict[str, str]) -> Callable[[int], tuple[str, ...]]:
    from tolerances import MPI_PROBE_TIMEOUT_SECONDS

    mpirun_value = os.environ.get("SPINGNN_MPIRUN", "mpirun")
    launcher = shlex.split(mpirun_value)
    if not launcher:
        _unavailable_lammps_contract(
            "MPI", "SPINGNN_MPIRUN resolves to an empty command"
        )
        raise AssertionError("unreachable")
    executable = launcher[0]
    if os.path.sep not in executable:
        resolved = shutil.which(executable)
    else:
        resolved = (
            str(Path(executable).expanduser()) if Path(executable).is_file() else None
        )
    if resolved is None:
        _unavailable_lammps_contract(
            "MPI", f"MPI launcher is unavailable: {executable}"
        )
        raise AssertionError("unreachable")
    launcher[0] = resolved
    extra = shlex.split(os.environ.get("SPINGNN_MPIRUN_EXTRA_ARGS", ""))
    extra_environment = shlex.split(os.environ.get("SPINGNN_MPIRUN_EXTRA_ENV", ""))
    for assignment in extra_environment:
        name, separator, value = assignment.partition("=")
        if not separator or not name:
            _unavailable_lammps_contract(
                "MPI",
                "SPINGNN_MPIRUN_EXTRA_ENV entries must use NAME=VALUE",
            )
        lammps_env[name] = value

    # --oversubscribe is an OpenMPI-only flag: Intel MPI (hydra) rejects it
    # (5090 acceptance 2290752: "unrecognized argument oversubscribe").
    # Detect the implementation once instead of hardcoding either way.
    try:
        version_probe = subprocess.run(
            [*launcher, "--version"],
            env=lammps_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        version_text = version_probe.stdout.lower()
    except Exception:
        version_text = ""
    oversubscribe = (
        ("--oversubscribe",)
        if "open mpi" in version_text or "openmpi" in version_text
        else ()
    )

    def prefix(ranks: int) -> tuple[str, ...]:
        return tuple((*launcher, *extra, *oversubscribe, "-n", str(ranks)))

    probe = subprocess.run(
        [*prefix(2), "hostname"],
        env=lammps_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=MPI_PROBE_TIMEOUT_SECONDS,
        check=False,
    )
    if probe.returncode != 0:
        _unavailable_lammps_contract(
            "MPI", f"local two-rank MPI probe failed: {probe.stdout[-2000:]}"
        )
    return prefix
