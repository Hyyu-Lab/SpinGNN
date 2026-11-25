"""Unit, security, and multi-arch fat-binary tests for SpinGNN2 multi-GPU support."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from spingnn2 import jit, native

ROOT = Path(__file__).resolve().parents[1]


def _load_build_native():
    spec = importlib.util.spec_from_file_location(
        "build_native_mod", ROOT / "tools/native/build_native.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_cuda_arches_registry_and_validation():
    bn = _load_build_native()
    assert bn.parse_cuda_arches("all-gpu") == (
        "sm70",
        "sm80",
        "sm86",
        "sm89",
        "sm90",
        "sm120",
    )
    assert bn.parse_cuda_arches("sm120,sm80,sm_70,sm_89,sm80") == (
        "sm70",
        "sm80",
        "sm89",
        "sm120",
    )
    assert jit.parse_cuda_arches("all-gpu") == (
        "sm_70",
        "sm_80",
        "sm_86",
        "sm_89",
        "sm_90",
        "sm_120",
    )
    assert jit.parse_cuda_arches("sm70;sm80;sm_90,sm120") == (
        "sm_70",
        "sm_80",
        "sm_90",
        "sm_120",
    )
    for bad in ("", "sm60", "sm75", "sm99", "sm_50", "foo"):
        with pytest.raises(ValueError):
            bn.parse_cuda_arches(bad)
        with pytest.raises(ValueError):
            jit.parse_cuda_arches(bad)


def test_sm70_launch_bounds_and_cuda_codegen():
    """Volta V100 (sm_70) uses conservative (128, 4) launch bounds for JIT kernels."""
    from spingnn2 import jit_codegen

    assert jit_codegen._launch_bounds("sm_70", heavy=False) == "128, 4"
    assert jit_codegen._launch_bounds("sm_70", heavy=True) == "128, 4"
    src = jit_codegen.Source(backend="cuda", precision="float32", gpu_arch="sm_70")
    src.kernel("sp2_test_volta", [("int", "n")], "if (i < n) {}", heavy=True)
    emitted = "\n".join(src.parts)
    assert "__launch_bounds__(128, 4)" in emitted


def test_nvcc_fatbin_shim_compile_dlink_and_passthrough(tmp_path):
    """Fat-binary nvcc shim injects exact -gencode flags for -c, omits PTX for -dlink,
    and passes --version untouched."""
    bn = _load_build_native()
    fake_nvcc = tmp_path / "real_nvcc"
    fake_nvcc.write_text(
        f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    fake_nvcc.chmod(0o755)
    shim = bn.write_nvcc_fatbin_shim(
        tmp_path / "shim-bin",
        fake_nvcc,
        ("sm70", "sm80", "sm86", "sm89", "sm90", "sm120"),
    )

    def run_shim(*args):
        out = subprocess.check_output([str(shim), *args], text=True)
        return json.loads(out)

    # 1. --version passthrough (no gencode injected)
    assert run_shim("--version") == ["--version"]

    # 2. Normal compile (-c) replaces single Kokkos -arch=sm_70 with all 6 sm + compute_120 PTX
    compiled = run_shim("-arch=sm_70", "-O3", "-c", "foo.cpp", "-o", "foo.o")
    assert "-arch=sm_70" not in compiled
    for n in (70, 80, 86, 89, 90, 120):
        assert f"-gencode=arch=compute_{n},code=sm_{n}" in compiled
    assert "-gencode=arch=compute_120,code=compute_120" in compiled
    assert compiled[-4:] == ["-O3", "-c", "foo.cpp", "-o", "foo.o"][-4:]

    # 3. Device link (-dlink / -rdc=true) injects all 6 real sm_XX targets and NO virtual PTX
    dlinked = run_shim("-arch", "sm_70", "-dlink", "a.o", "b.o", "-o", "dlink.o")
    assert "-arch" not in dlinked and "sm_70" not in dlinked
    for n in (70, 80, 86, 89, 90, 120):
        assert f"-gencode=arch=compute_{n},code=sm_{n}" in dlinked
    assert "-gencode=arch=compute_120,code=compute_120" not in dlinked


def test_detect_cuda_arch_degrades_gracefully_without_driver(monkeypatch):
    """native._detect_cuda_arch() returns None silently when libcuda.so.1 is absent."""

    def missing_cdll(_name):
        raise OSError("libcuda.so.1: cannot open shared object file")

    monkeypatch.setattr(native.ct, "CDLL", missing_cdll)
    assert native._detect_cuda_arch() is None


def test_jit_bundle_manifest_rejects_unsafe_member_paths(tmp_path):
    """Verify libspingnn2 rejects unsafe traversal/symlink/invalid member paths in bundle_abi=1."""
    lib_path = os.environ.get("SPINGNN2_LIBRARY")
    if not lib_path or not Path(lib_path).is_file():
        pytest.skip("Set SPINGNN2_LIBRARY to test C++ bundle manifest validation")
    lib = native._load_library(lib_path, os.environ.get("SPINGNN2_TEST_BACKEND", "cpu"))
    if lib.sp2_backend().decode() != "cuda":
        pytest.skip("Bundle manifest validation requires CUDA libspingnn2")

    # Locate any existing contract to construct a model handle for artifact loading tests
    models_json = os.environ.get("SPINGNN2_LAMMPS_MODELS")
    if not models_json or not Path(models_json).is_file():
        pytest.skip("Set SPINGNN2_LAMMPS_MODELS to test bundle security")
    spec = json.loads(Path(models_json).read_text())["nosoc"]["float32"]
    contract_path = Path(spec["contract"])
    from spingnn2.contract import load_contract_with_digest

    _, digest = load_contract_with_digest(contract_path)

    cache = tmp_path / "bundle_sec"
    cache.mkdir(mode=0o700)

    # Create a legitimate dummy file outside cache to test traversal / symlink escape
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n")
    escape_link = cache / "escape.json"
    escape_link.symlink_to(outside)

    bad_members = [
        "../outside.json",
        "/etc/passwd",
        "sub/member.json",
        "escape.json",
        "not_a_json.txt",
    ]
    for idx, bad in enumerate(bad_members):
        bundle_file = cache / f"bad_{idx}.bundle.json"
        bundle = {
            "abi_version": 1,
            "bundle_abi": 1,
            "backend": "cuda",
            "contract_sha256": digest,
            "precision": "float32",
            "artifacts": {
                "sm_80": bad,
                "sm_86": bad,
                "sm_89": bad,
                "sm_90": bad,
                "sm_120": bad,
            },
        }
        bundle_file.write_text(json.dumps(bundle))
        with pytest.raises(ValueError):
            native.NativeModel.load(
                contract_path,
                backend="cuda",
                precision="float32",
                execution_mode="direct",
                artifact=bundle_file,
                library=lib_path,
            )
