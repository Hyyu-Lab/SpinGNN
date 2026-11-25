"""Prepared CPU/CUDA artifact parity, cache reuse and loader rejection checks."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from spingnn2.jit import prepare_jit_artifact
from spingnn2.native import NativeModel
from test_contract_oracle import CASES, MAGNETIC_CASES, REPO, _observables
from test_native_engine import BACKEND, native_case as _native_case  # noqa: F401

FP64 = 1e-11
FP32 = 5e-5
EVIDENTIAL_REL = 1e-4
LATTICE_FIELDS = {"forces", "stress", "atom_virial", "virial"}


@pytest.fixture
def native_case(request):
    return request.getfixturevalue("_native_case")


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    suffix = (
        "cpu"
        if BACKEND == "cpu"
        else "cuda-" + os.environ.get("SPINGNN2_ARCH", "sm120")
    )
    default = REPO / "validation" / "m3" / ("build-" + suffix) / "libspingnn2.so"
    library = Path(os.environ.get("SPINGNN2_LIBRARY", default))
    if not library.is_file():
        if "SPINGNN2_TEST_BACKEND" in os.environ or "SPINGNN2_LIBRARY" in os.environ:
            pytest.fail(f"Requested M3 library does not exist: {library}")
        pytest.skip("Build the M3 native library to exercise generated kernels")
    shared_cache = os.environ.get("SPINGNN2_TEST_JIT_CACHE")
    cache = (
        Path(shared_cache) if shared_cache else tmp_path_factory.mktemp("jit-artifacts")
    )
    gpu_arch = os.environ.get("SPINGNN2_ARCH") if BACKEND == "cuda" else None

    def prepare(fixture):
        return (
            prepare_jit_artifact(
                fixture.manifest,
                backend=BACKEND,
                cache_dir=cache,
                gpu_arch=gpu_arch,
                library=library,
            ),
            library,
        )

    return prepare


def _assert_parity(case, kind, actual, expected, limit, fields=None):
    if fields is None:
        fields = _observables(case)
    maxima = {}
    for field in fields:
        value, target = actual[field], expected[field]
        assert value.dtype == target.dtype, field
        assert value.shape == target.shape, field
        assert np.isfinite(value).all(), field
        delta = np.abs(value - target)
        absolute = float(np.max(delta, initial=0))
        if field.startswith(("f_", "sf_", "total_f_", "total_sf_")) and limit == FP32:
            evidence = float(np.max(delta / np.maximum(1, np.abs(target)), initial=0))
            assert evidence < EVIDENTIAL_REL, (case.name, kind, field, evidence)
            maxima[field + "_normalized"] = evidence
        else:
            assert absolute < limit, (case.name, kind, field, absolute)
        maxima[field] = absolute
    print(
        "M3_RESULT",
        json.dumps(
            {"backend": BACKEND, "case": case.name, "check": kind, **maxima},
            sort_keys=True,
        ),
    )


def _open(fixture, artifact, library, mode, **options):
    return NativeModel.load(
        fixture.manifest,
        backend=BACKEND,
        library=library,
        execution_mode=mode,
        artifact=artifact,
        **options,
    )


def _assert_direct(model, artifact, profile):
    plan = model.execution_plan
    manifest = json.loads(artifact.read_text())
    assert plan["selected_mode"] == "direct"
    assert plan["fallback_count"] == 0
    assert plan["execution_profile"] == profile
    assert plan["artifact_sha256"] == manifest["artifact_sha256"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_jit_parity(native_case, artifacts, case, dtype):
    fixture = native_case(case, dtype)
    artifact, library = artifacts(fixture)
    limit = FP64 if dtype == "float64" else FP32
    with _open(
        fixture, artifact, library, "direct", execution_profile="speed", tile_atoms=3
    ) as direct, _open(fixture, None, library, "generic", tile_atoms=3) as generic:
        for mode in ("analytic", "spline"):
            actual = direct.evaluate(fixture.data, radial_mode=mode)
            expected = generic.evaluate(fixture.data, radial_mode=mode)
            _assert_parity(case, mode + "-generic-" + dtype, actual, expected, limit)
            _assert_direct(direct, artifact, "speed")
            assert generic.execution_plan["selected_mode"] == "generic"
            assert direct.execution_plan["tile_atoms"] == 3
            if mode == "analytic":
                _assert_parity(case, "eager-" + dtype, actual, fixture.eager, limit)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_jit_tiling(native_case, artifacts, case):
    fixture = native_case(case)
    artifact, library = artifacts(fixture)
    with _open(
        fixture,
        artifact,
        library,
        "direct",
        execution_profile="capacity",
        radial_mode="spline",
    ) as direct:
        whole = direct.evaluate(fixture.data, tile_atoms=None)
        for size in (1, 3):
            part = direct.evaluate(fixture.data, tile_atoms=size)
            _assert_parity(case, f"tile-{size}", part, whole, FP64)
            _assert_direct(direct, artifact, "capacity")
            assert direct.execution_plan["tile_atoms"] == size


def test_jit_periodic_self_images(native_case, artifacts):
    case = next(case for case in CASES if case.name == "soc-u2")
    fixture = native_case(case)
    artifact, library = artifacts(fixture)
    # Both endpoints refer to the same atom, while periodic shifts produce
    # nonzero edge displacement and virial. Frame-specific cells distinguish
    # identical image indices in different frames.
    data = {
        "pos": np.zeros((2, 3)),
        "spin": fixture.data["spin"][:2],
        "atom_types": fixture.data["atom_types"][:2],
        "edge_index": np.array([[1, 0, 1, 0], [1, 0, 1, 0]]),
        "edge_cell_shift": np.array([[1, 0, 0], [-1, 0, 0], [-1, 0, 0], [1, 0, 0]]),
        "cell": np.array([np.diag([3.5, 11.0, 12.0]), np.diag([4.1, 11.0, 12.0])]),
        "pbc": np.ones((2, 3), dtype=bool),
        "batch": np.array([0, 1]),
    }
    with _open(fixture, artifact, library, "direct", tile_atoms=1) as direct, _open(
        fixture, None, library, "generic", tile_atoms=2
    ) as generic:
        actual = direct.evaluate(data)
        expected = generic.evaluate(data)
        _assert_parity(case, "self-periodic-batches", actual, expected, FP64)
        np.testing.assert_array_equal(actual["forces"], 0.0)
        assert np.max(np.abs(actual["virial"])) > 0.0
        _assert_direct(direct, artifact, "capacity")


@pytest.mark.parametrize("case", MAGNETIC_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("spill", (False, True))
@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_jit_spin_cache(native_case, artifacts, case, spill, dtype):
    fixture = native_case(case, dtype)
    limit = FP64 if dtype == "float64" else FP32
    artifact, library = artifacts(fixture)
    lattice = {key: value for key, value in fixture.data.items() if key != "spin"}
    with _open(
        fixture,
        artifact,
        library,
        "direct",
        execution_profile="capacity",
        tile_atoms=3,
        radial_mode="spline",
        cache_device_limit_bytes=0 if spill else None,
    ) as direct:
        with direct.evaluate_stage1_lattice(lattice) as cache:
            spin = np.roll(fixture.data["spin"], 1, axis=0)
            expected = direct.evaluate({**lattice, "spin": spin})
            full = direct.evaluate_stage2_spin_cached(cache, spin)
            _assert_parity(case, f"full-cache-spill-{spill}", full, expected, limit)
            _assert_direct(direct, artifact, "capacity")
            frozen = direct.evaluate_stage2_spin_cached(cache, spin, spin_only=True)
            fields = [
                field for field in _observables(case) if field not in LATTICE_FIELDS
            ]
            assert not LATTICE_FIELDS.intersection(frozen)
            _assert_parity(
                case, f"spin-cache-spill-{spill}", frozen, full, limit, fields
            )
            _assert_direct(direct, artifact, "capacity")
            if BACKEND == "cuda" and not spill:
                assert direct.diagnostics["cache_device_bytes"] > 0
            else:
                assert direct.diagnostics["cache_device_bytes"] == 0
                assert direct.diagnostics["cache_host_bytes"] > 0
            full_bytes = sum(
                direct.diagnostics[key]
                for key in ("cache_device_bytes", "cache_host_bytes")
            )
        with direct.evaluate_stage1_lattice(lattice, spin_only=True) as cache:
            lean = direct.evaluate_stage2_spin_cached(cache, spin, spin_only=True)
            assert not LATTICE_FIELDS.intersection(lean)
            _assert_parity(
                case, f"spin-forward-spill-{spill}", lean, full, limit, fields
            )
            lean_bytes = sum(
                direct.diagnostics[key]
                for key in ("cache_device_bytes", "cache_host_bytes")
            )
            assert 0 < lean_bytes < full_bytes
            if BACKEND == "cuda" and not spill:
                assert direct.diagnostics["cache_host_bytes"] == 0
                assert direct.diagnostics["stage1_cache_hits"] > 0
            recomputed = direct.evaluate_stage2_spin_cached(cache, spin)
            _assert_parity(
                case, f"force-forward-spill-{spill}", recomputed, full, limit
            )


def _copy_artifact(artifact, path):
    path.mkdir()
    copied = path / artifact.name
    shutil.copyfile(artifact, copied)
    metadata = json.loads(copied.read_text())
    shutil.copyfile(
        artifact.with_name(metadata["artifact"]), path / metadata["artifact"]
    )
    return copied, metadata


def test_jit_cache_reuse(native_case, artifacts):
    fixture = native_case(CASES[0])
    artifact, _ = artifacts(fixture)
    metadata = json.loads(artifact.read_text())
    image = artifact.with_name(metadata["artifact"])
    before = (artifact.stat().st_mtime_ns, image.stat().st_mtime_ns)
    cached, _ = artifacts(fixture)
    assert cached == artifact
    assert before == (artifact.stat().st_mtime_ns, image.stat().st_mtime_ns)
    assert hashlib.sha256(image.read_bytes()).hexdigest() == metadata["artifact_sha256"]
    assert "sp2_geom_scatter" in metadata["symbols"]
    assert any(name.startswith("sp2_geom_radial_basis") for name in metadata["symbols"])


def test_jit_planner_profiles_and_auto_cache_hit(native_case, artifacts, monkeypatch):
    case = CASES[0]
    fixture = native_case(case)
    artifact, library = artifacts(fixture)
    data = fixture.data
    # Many disconnected centers exercise the default admission widths without
    # requiring a different checkpoint or a large contraction workspace.
    n = 513
    graph = dict(
        data,
        pos=np.concatenate((data["pos"], np.zeros((n - len(data["pos"]), 3)))),
        spin=np.concatenate((data["spin"], np.zeros((n - len(data["spin"]), 3)))),
        atom_types=np.pad(data["atom_types"], (0, n - len(data["atom_types"]))),
        batch=np.zeros(n, dtype=np.int64),
    )
    metadata = json.loads(artifact.read_text())
    image = artifact.with_name(metadata["artifact"])
    times = (artifact.stat().st_mtime_ns, image.stat().st_mtime_ns)
    # If auto recompiles on a cache hit rather than finding the verified
    # artifact, it will try one of these deliberately unavailable compilers.
    monkeypatch.setenv("CXX", "sp2-compiler-disabled-for-cache-hit")
    monkeypatch.setenv("SPINGNN2_NVRTC_LIBRARY", "sp2-nvrtc-disabled-for-cache-hit")
    with _open(fixture, None, library, "auto", cache_dir=artifact.parent) as capacity:
        expected = capacity.evaluate(graph)
        _assert_direct(capacity, artifact, "capacity")
        assert capacity.execution_plan["tile_atoms"] == 256
    with _open(
        fixture, artifact, library, "direct", execution_profile="speed"
    ) as speed:
        actual = speed.evaluate(graph)
        _assert_direct(speed, artifact, "speed")
        assert speed.execution_plan["tile_atoms"] == 512
        explicit = speed.evaluate(graph, tile_atoms=31)
        assert speed.execution_plan["tile_atoms"] == 31
        _assert_parity(case, "profile-width", actual, expected, FP64)
        _assert_parity(case, "profile-explicit-width", explicit, expected, FP64)
    assert times == (artifact.stat().st_mtime_ns, image.stat().st_mtime_ns)


def test_jit_digest_and_arch_rejections(native_case, artifacts, tmp_path):
    fixture = native_case(CASES[0])
    artifact, library = artifacts(fixture)

    copied, metadata = _copy_artifact(artifact, tmp_path / "tampered")
    image = copied.with_name(metadata["artifact"])
    contents = bytearray(image.read_bytes())
    contents[len(contents) // 2] ^= 1
    image.write_bytes(contents)
    with pytest.raises(ValueError, match=r"(?i)sha-?256|digest"):
        _open(fixture, copied, library, "direct")

    wrong, metadata = _copy_artifact(artifact, tmp_path / "wrong-architecture")
    metadata["gpu_arch"] = "sm_89" if metadata["gpu_arch"] != "sm_89" else "sm_120"
    wrong.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=r"(?i)arch"):
        _open(fixture, wrong, library, "direct")

    with pytest.raises(ValueError, match=r"(?i)prepared artifact"):
        _open(fixture, None, library, "direct")


def test_jit_prepared_inference_without_compilers(native_case, artifacts, tmp_path):
    fixture = native_case(CASES[-1])
    artifact, library = artifacts(fixture)
    graph = tmp_path / "data.npz"
    np.savez(graph, **fixture.data)
    child = """
import importlib.abc
import sys

class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'e3nn', 'allegro', 'nequip'):
            raise AssertionError(fullname)

sys.meta_path.insert(0, Forbidden())
import numpy as np
from spingnn2.native import NativeModel

with NativeModel.load(sys.argv[1], backend=sys.argv[2], library=sys.argv[3],
                      execution_mode='direct', artifact=sys.argv[4], tile_atoms=3) as model:
    with np.load(sys.argv[5]) as data:
        result = model.evaluate(dict(data), radial_mode='spline')
    assert np.isfinite(result['total_energy']).all()
    assert np.isfinite(result['sf_Uepi']).all()
    assert model.execution_plan['selected_mode'] == 'direct'
    assert model.execution_plan['fallback_count'] == 0
"""
    env = {
        **os.environ,
        "CXX": str(tmp_path / "missing-cxx"),
        "SPINGNN2_NVRTC_LIBRARY": str(tmp_path / "missing-nvrtc.so"),
        "PATH": str(tmp_path / "empty"),
        "PYTHONPATH": str(REPO / "src"),
    }
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            child,
            str(fixture.manifest),
            BACKEND,
            str(library),
            str(artifact),
            str(graph),
        ],
        env=env,
        check=True,
        timeout=180,
    )


def test_blas_precision_is_explicit_and_scoped(native_case, artifacts):
    """One cache must retain its math policy when other models run on the thread."""
    case = MAGNETIC_CASES[0]
    fp64 = native_case(case, "float64")
    _, library = artifacts(fp64)
    with pytest.raises(ValueError, match="CUDA FP32"):
        NativeModel.load(
            fp64.manifest, backend=BACKEND, library=library, allow_tf32=True
        )
    fixture = native_case(case, "float32")
    manifest, _ = artifacts(fixture)
    settings = dict(
        backend=BACKEND,
        library=library,
        artifact=manifest,
        execution_mode="direct",
        radial_mode="spline",
        tile_atoms=3,
    )
    if BACKEND == "cpu":
        with pytest.raises(ValueError, match="CUDA FP32"):
            NativeModel.load(fixture.manifest, allow_tf32=True, **settings)
        return
    with NativeModel.load(fixture.manifest, **settings) as strict, NativeModel.load(
        fixture.manifest, allow_tf32=True, **settings
    ) as fast:
        reference = strict.evaluate(fixture.data)
        assert strict.execution_plan["math_mode"] == "strict_fp32"
        with fast.evaluate_stage1_lattice(fixture.data) as saved:
            assert fast.execution_plan["math_mode"] == "tf32"
            strict.evaluate(fixture.data)
            cached = fast.evaluate_stage2_spin_cached(saved, fixture.data["spin"])
            assert fast.execution_plan["math_mode"] == "tf32"
            expected = fast.evaluate(fixture.data)
            for field in ("total_energy", "forces", "spin_forces"):
                np.testing.assert_allclose(
                    cached[field], expected[field], atol=5e-5, rtol=1e-6
                )
        # Calling into the strict model restores FP32 after the TF32 launches.
        repeated = strict.evaluate(fixture.data)
        assert strict.execution_plan["math_mode"] == "strict_fp32"
        for field in ("total_energy", "forces", "spin_forces"):
            np.testing.assert_allclose(
                repeated[field], reference[field], atol=5e-5, rtol=1e-6
            )
