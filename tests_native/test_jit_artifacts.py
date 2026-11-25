"""Publication and trust-boundary checks for the prepared-artifact loader."""

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
from test_contract_oracle import CASES, REPO
from test_native_engine import BACKEND, native_case as _native_case  # noqa: F401
from test_jit_runtime import artifacts as _artifacts  # noqa: F401


@pytest.fixture
def native_case(request):
    return request.getfixturevalue("_native_case")


@pytest.fixture
def artifacts(request):
    return request.getfixturevalue("_artifacts")


def test_jit_concurrent_publication_and_driver_free_cache_hit(
    native_case, artifacts, tmp_path, monkeypatch
):
    fixture = native_case(CASES[0])
    _, library = artifacts(fixture)
    source = """
import sys
from pathlib import Path
from spingnn2.jit import prepare_jit_artifact
print(prepare_jit_artifact(sys.argv[1], backend=sys.argv[2], gpu_arch=sys.argv[3] or None,
                           library=sys.argv[4], cache_dir=Path(sys.argv[5])))
"""
    arch = os.environ.get("SPINGNN2_ARCH", "") if BACKEND == "cuda" else ""
    cache = tmp_path / "concurrent"
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    cmd = [
        sys.executable,
        "-B",
        "-c",
        source,
        str(fixture.manifest),
        BACKEND,
        arch,
        str(library),
        str(cache),
    ]
    processes = [
        subprocess.Popen(
            cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        for _ in range(3)
    ]
    manifests = set()
    try:
        for process in processes:
            out, err = process.communicate(timeout=180)
            assert process.returncode == 0, err
            manifests.add(Path(out.strip()))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert len(manifests) == 1
    a = manifests.pop()
    image = a.with_name(json.loads(a.read_text())["artifact"])
    assert set(cache.iterdir()) == {a, image}

    # A fresh process sees only the inference library. Its auto policy must
    # select the cached image without loading the preparation companion or
    # invoking an executable from PATH.
    private_lib = tmp_path / "inference.so"
    shutil.copyfile(library, private_lib)
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, **fixture.data)
    code = """
import ctypes, sys
import numpy as np

native_load = ctypes.CDLL

def guarded(path, *args, **kwargs):
    assert "jit_prepare" not in str(path), ("Loaded a compiler library", path)
    return native_load(path, *args, **kwargs)

ctypes.CDLL = guarded

from spingnn2.native import NativeModel

contract, backend, library, cache, graph = sys.argv[1:]
with NativeModel.load(contract, library=library, backend=backend, cache_dir=cache,
                      execution_mode="auto") as model:
    with np.load(graph) as data:
        assert np.isfinite(model.evaluate(dict(data))["total_energy"]).all()
    assert model.execution_plan["selected_mode"] == "direct"
    assert model.execution_plan["fallback_count"] == 0
"""
    monkeypatch.setenv(
        "SPINGNN2_NVRTC_LIBRARY", os.fspath(tmp_path / "missing-nvrtc.so")
    )
    monkeypatch.setenv("CXX", os.fspath(tmp_path / "missing-cxx"))
    monkeypatch.setenv("PATH", os.fspath(tmp_path / "empty"))
    monkeypatch.setenv("PYTHONPATH", os.fspath(REPO / "src"))
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            os.fspath(fixture.manifest),
            BACKEND,
            os.fspath(private_lib),
            os.fspath(cache),
            os.fspath(inputs),
        ],
        check=True,
        timeout=120,
    )


def test_jit_rejects_unsafe_artifact_files_and_cache(native_case, artifacts, tmp_path):
    fixture = native_case(CASES[0])
    manifest, library = artifacts(fixture)
    cache = tmp_path / "unsafe"
    cache.mkdir()
    document = json.loads(manifest.read_text())
    image = cache / document["artifact"]
    copied = cache / manifest.name
    shutil.copyfile(manifest, copied)
    shutil.copyfile(manifest.with_name(image.name), image)

    def load():
        return NativeModel.load(
            fixture.manifest,
            backend=BACKEND,
            library=library,
            artifact=copied,
            execution_mode="direct",
        )

    for path in (cache, copied, image):
        mode = path.stat().st_mode
        path.chmod(mode | 0o020)
        try:
            with pytest.raises(ValueError, match="permissions"):
                load()
            if path == cache:
                with pytest.raises(ValueError, match="permissions"):
                    prepare_jit_artifact(
                        fixture.manifest,
                        backend=BACKEND,
                        library=library,
                        gpu_arch=document["gpu_arch"] or None,
                        cache_dir=cache,
                    )
        finally:
            path.chmod(mode)

    link = cache / "manifest-target.json"
    copied.rename(link)
    copied.symlink_to(Path(link.name))
    with pytest.raises(ValueError, match=r"(?i)open jit path"):
        load()

    copied.unlink()
    link.rename(copied)
    image_target = cache / "image-target"
    image.rename(image_target)
    image.symlink_to(image_target.name)
    with pytest.raises(ValueError, match=r"(?i)open jit path"):
        load()


def test_jit_manifest_abi_precision_and_artifact_bindings(
    native_case, artifacts, tmp_path
):
    fixture = native_case(CASES[0])
    manifest, library = artifacts(fixture)
    metadata = json.loads(manifest.read_text())
    cache = tmp_path / "manifest"
    cache.mkdir()
    copy = cache / manifest.name
    options = dict(backend=BACKEND, library=library, execution_mode="direct")
    other_precision = "float64" if metadata["precision"] == "float32" else "float32"
    other = native_case(CASES[2])
    with pytest.raises(ValueError, match="contract SHA-256 mismatch"):
        NativeModel.load(other.manifest, artifact=manifest, **options)
    with pytest.raises(ValueError, match="precision mismatch"):
        NativeModel.load(
            fixture.manifest, artifact=manifest, precision=other_precision, **options
        )
    for field, value, message in [
        ("abi_version", 99, "ABI version mismatch"),
        ("backend", "cuda" if BACKEND == "cpu" else "cpu", "backend mismatch"),
        ("artifact", "../kernel.so", "artifact filename"),
        ("contract_sha256", "0" * 64, "contract SHA-256 mismatch"),
        ("cache_key", "1" * 64, "cache identity"),
    ]:
        copy.write_text(json.dumps({**metadata, field: value}))
        with pytest.raises(ValueError, match=message):
            NativeModel.load(fixture.manifest, artifact=copy, **options)
