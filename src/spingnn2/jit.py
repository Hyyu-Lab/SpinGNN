"""Prepare and publish model-specialized host plugins and CUDA cubins."""

import argparse
from contextlib import contextmanager
import ctypes as ct
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
from tempfile import TemporaryDirectory

from .contract import load_contract_with_digest


_ABI = 1
_KERNEL_ABI = 1
_BUNDLE_ABI = 1
_ARCHES = ("sm_70", "sm_80", "sm_86", "sm_89", "sm_90", "sm_120")
_ARCH_ALIASES = {
    "sm70": "sm_70",
    "sm80": "sm_80",
    "sm86": "sm_86",
    "sm89": "sm_89",
    "sm90": "sm_90",
    "sm120": "sm_120",
}
_HOST_FLAGS = ("-std=c++17", "-O3", "-fopenmp", "-fPIC", "-shared", "-ffp-contract=off")
_CACHE_FIELDS = (
    "abi_version",
    "artifact_abi",
    "backend",
    "contract_sha256",
    "gpu_arch",
    "precision",
    "source_sha256",
    "symbols",
)


def _arch(value):
    if value is None:
        return None
    text = str(value).strip()
    return _ARCH_ALIASES.get(text, text)


def parse_cuda_arches(value):
    """Normalize a single arch, comma/semicolon list, or 'all-gpu' to canonical sm_XX tuple."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("CUDA architecture specification must not be empty")
        if text == "all-gpu":
            return _ARCHES
        tokens = [tok.strip() for tok in re.split(r"[,;]+", text) if tok.strip()]
    else:
        tokens = [str(tok).strip() for tok in value if str(tok).strip()]
    if not tokens:
        raise ValueError("CUDA architecture specification must not be empty")
    result = []
    for token in tokens:
        if token == "all-gpu":
            for item in _ARCHES:
                if item not in result:
                    result.append(item)
            continue
        norm = _arch(token)
        if norm not in _ARCHES:
            raise ValueError(
                f"Unsupported CUDA architecture {token!r}; expected one of "
                f"{', '.join(_ARCHES)} or 'all-gpu'"
            )
        if norm not in result:
            result.append(norm)
    return tuple(result)


def _call_generate_source(generate_source, contract, backend, precision, gpu_arch):
    params = inspect.signature(generate_source).parameters
    if "gpu_arch" in params:
        return generate_source(contract, backend, precision, gpu_arch=gpu_arch)
    return generate_source(contract, backend, precision)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _key(metadata):
    body = {key: metadata[key] for key in _CACHE_FIELDS}
    return _sha(json.dumps(body, separators=(",", ":"), sort_keys=True).encode())


def _require_safe(info, kind):
    if (
        stat.S_IFMT(info.st_mode) != kind
        or info.st_uid not in (0, os.geteuid())
        or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise ValueError("Unsafe JIT cache permissions or file type")


@contextmanager
def _locked_cache(directory):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        _require_safe(os.fstat(fd), stat.S_IFDIR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        os.close(fd)


def _read(directory_fd, name, limit):
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(name, flags, dir_fd=directory_fd), "rb") as stream:
        info = os.fstat(stream.fileno())
        _require_safe(info, stat.S_IFREG)
        if not 0 < info.st_size <= limit:
            raise ValueError("Invalid JIT cache file size")
        contents = stream.read(limit + 1)
        if len(contents) != info.st_size:
            raise ValueError("JIT cache file changed during read")
        return contents


def _cached(directory_fd, metadata_path, expected):
    try:
        header = json.loads(_read(directory_fd, metadata_path, 1024 * 1024))
    except FileNotFoundError:
        return False
    if set(header) != {*expected, "artifact_sha256", "artifact"} or any(
        header[key] != value for key, value in expected.items()
    ):
        raise ValueError("Invalid cached JIT manifest metadata")
    digest = header["artifact_sha256"]
    suffix = ".cubin" if expected["backend"] == "cuda" else ".so"
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("Invalid cached JIT artifact digest")
    name = "kernel-" + digest + suffix
    if header["artifact"] != name:
        raise ValueError("Invalid cached JIT artifact filename")
    if _sha(_read(directory_fd, name, 512 * 1024 * 1024)) != digest:
        raise ValueError("Cached JIT artifact SHA-256 mismatch")
    return True


def _native_cuda(library, gpu_arch):
    first = None
    if gpu_arch:
        parsed = parse_cuda_arches(gpu_arch)
        first = parsed[0] if parsed else None
    suffix = (first or _arch(os.environ.get("SPINGNN2_ARCH", "sm120"))).replace("_", "")
    location = library or os.environ.get("SPINGNN2_LIBRARY")
    if location:
        path = Path(location).with_name("libspingnn2_jit_prepare.so")
    else:
        native_root = Path(__file__).resolve().parents[2] / "validation/native"
        candidates = [
            native_root / "build-cuda-all-gpu" / "libspingnn2_jit_prepare.so",
            native_root / ("build-cuda-" + suffix) / "libspingnn2_jit_prepare.so",
        ]
        path = next(
            (c for c in candidates if c.is_file()), "libspingnn2_jit_prepare.so"
        )
    lib = ct.CDLL(os.fspath(path))
    lib.sp2_jit_target_arch.restype = ct.c_char_p
    lib.sp2_jit_prepare_error.restype = ct.c_char_p
    lib.sp2_jit_compile_cuda.argtypes = (ct.c_char_p, ct.c_char_p, ct.c_char_p)
    lib.sp2_jit_compile_cuda.restype = ct.c_int
    return lib


def _artifact(directory_fd, data, filename, staging, manifest):
    # Artifact names are immutable hashes. Publishing the manifest last makes
    # it the sole ready marker, even after a failed or interrupted compilation.
    try:
        existing = _read(directory_fd, filename, 512 * 1024 * 1024)
    except FileNotFoundError:
        existing = None
    if existing is None:
        blob = staging / filename
        with blob.open("xb") as stream:
            stream.write(data)
            os.fsync(stream.fileno())
        os.replace(blob, filename, dst_dir_fd=directory_fd)
    elif data != existing:
        raise ValueError("JIT cache contains inconsistent content-addressed data")
    marker = staging / "manifest.json"
    with marker.open("x") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(marker, manifest["cache_key"] + ".json", dst_dir_fd=directory_fd)
    os.fsync(directory_fd)


def prepare_jit_artifact(
    model,
    *,
    backend,
    precision=None,
    gpu_arch=None,
    cache_dir=None,
    library=None,
    cxx=None,
):
    """Compile a verified M1 model once and return its ready manifest path.

    `backend="cuda"` uses NVRTC only on a cache miss. Its compiler runs on the
    host and targets the architecture selected at native-library build time.
    A prepared artifact can be loaded during inference without NVRTC or a C++
    compiler installed on the execution host.
    """
    from .jit_codegen import generate_source

    if backend not in ("cpu", "cuda"):
        raise ValueError("JIT backend must be cpu or cuda")
    if backend == "cpu" and gpu_arch is not None:
        raise ValueError("A CPU artifact does not have a GPU architecture")
    lib = None
    if backend == "cpu":
        arch = ""
    else:
        if gpu_arch is None:
            lib = _native_cuda(library, None)
            arches = parse_cuda_arches(lib.sp2_jit_target_arch().decode())
        else:
            arches = parse_cuda_arches(gpu_arch)
        if len(arches) > 1:
            return prepare_jit_bundle(
                model,
                arches=arches,
                precision=precision,
                cache_dir=cache_dir,
                library=library,
            )
        arch = arches[0]
    contract, digest = load_contract_with_digest(model)
    precision = precision or contract["dtype"]
    if str(precision) not in ("float32", "float64"):
        raise ValueError("JIT precision must be float32 or float64")
    precision = str(precision)
    source, symbols = _call_generate_source(
        generate_source, contract, backend, precision, arch or None
    )
    symbols = sorted(symbols)
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not re.fullmatch(r"sp2_[a-zA-Z0-9_]{1,124}", name) for name in symbols)
    ):
        raise ValueError("Invalid generated JIT kernel names")
    # Include the compiler policy and host machine in the source hash. Cached
    # CUDA cubins remain usable after the preparing NVRTC is removed.
    policy = (
        "cuda-c++17-fmad-on"
        if backend == "cuda"
        else " ".join(_HOST_FLAGS) + " " + platform.machine()
    )
    source = "// SpinGNN2 compilation policy: " + policy + "\n" + source
    header = dict(
        abi_version=_ABI,
        artifact_abi=_KERNEL_ABI,
        backend=backend,
        contract_sha256=digest,
        gpu_arch=arch,
        precision=precision,
        source_sha256=_sha(source.encode()),
        symbols=symbols,
    )
    key = _key(header)
    header["cache_key"] = key
    marker = key + ".json"
    directory = (
        Path(
            cache_dir
            or os.environ.get("SPINGNN2_JIT_CACHE")
            or Path.home() / ".cache" / "spingnn2" / "jit"
        )
        .expanduser()
        .absolute()
    )
    with _locked_cache(directory) as fd:
        if _cached(fd, marker, header):
            return directory / marker
        decl = "__global__ " if backend == "cuda" else ""
        source += f'\nextern "C" {decl}void sp2_identity_{key}() {{}}\n'
        with TemporaryDirectory(prefix=".prepare-", dir=directory) as temp:
            stage = Path(temp)
            output = stage / "compiled-image"
            if backend == "cuda":
                lib = lib or _native_cuda(library, arch)
                if lib.sp2_jit_compile_cuda(
                    source.encode(), arch.encode(), os.fsencode(output)
                ):
                    raise RuntimeError(
                        lib.sp2_jit_prepare_error().decode(errors="replace")
                    )
            else:
                compiler = cxx or os.environ.get("CXX") or shutil.which("c++")
                if not compiler:
                    raise RuntimeError("No C++ compiler found for host JIT preparation")
                translation_unit = stage / "kernel.cpp"
                translation_unit.write_text(source)
                run = subprocess.run(
                    [compiler, *_HOST_FLAGS, str(translation_unit), "-o", str(output)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if run.returncode:
                    raise RuntimeError("Host JIT compilation failed:\n" + run.stderr)
            binary = output.read_bytes()
            digest = _sha(binary)
            suffix = ".cubin" if backend == "cuda" else ".so"
            name = "kernel-" + digest + suffix
            _artifact(
                fd,
                binary,
                name,
                stage,
                dict(header, artifact=name, artifact_sha256=digest),
            )
    return directory / marker


def prepare_jit_bundle(
    model,
    *,
    arches="all-gpu",
    precision=None,
    cache_dir=None,
    library=None,
):
    """Compile CUDA cubins for multiple GPU architectures and publish a bundle manifest."""
    parsed = parse_cuda_arches(arches)
    if not parsed:
        raise ValueError("At least one CUDA architecture is required for a JIT bundle")
    contract, contract_digest = load_contract_with_digest(model)
    precision = str(precision or contract["dtype"])
    if precision not in ("float32", "float64"):
        raise ValueError("JIT precision must be float32 or float64")
    directory = (
        Path(
            cache_dir
            or os.environ.get("SPINGNN2_JIT_CACHE")
            or Path.home() / ".cache" / "spingnn2" / "jit"
        )
        .expanduser()
        .absolute()
    )
    members = {}
    for arch in parsed:
        member_path = prepare_jit_artifact(
            model,
            backend="cuda",
            precision=precision,
            gpu_arch=arch,
            cache_dir=directory,
            library=library,
        )
        members[arch] = member_path.name
    bundle = {
        "abi_version": _ABI,
        "bundle_abi": _BUNDLE_ABI,
        "backend": "cuda",
        "contract_sha256": contract_digest,
        "precision": precision,
        "artifacts": dict(sorted(members.items())),
    }
    bundle_key = _sha(
        json.dumps(bundle, separators=(",", ":"), sort_keys=True).encode()
    )
    marker = f"{bundle_key}.bundle.json"
    with _locked_cache(directory) as fd:
        with TemporaryDirectory(prefix=".bundle-", dir=directory) as temp:
            tmp_file = Path(temp) / "bundle.json"
            with tmp_file.open("x") as stream:
                json.dump(bundle, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_file, marker, dst_dir_fd=fd)
        os.fsync(fd)
    return directory / marker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--precision", choices=("float32", "float64"))
    parser.add_argument(
        "--arch",
        help="Target CUDA arch (sm70|sm80|sm86|sm89|sm90|sm120, comma-separated list, or all-gpu)",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--cxx")
    args = parser.parse_args(argv)
    manifest = prepare_jit_artifact(
        args.model,
        backend=args.backend,
        precision=args.precision,
        gpu_arch=args.arch,
        cache_dir=args.cache_dir,
        library=args.library,
        cxx=args.cxx,
    )
    print(manifest)


if __name__ == "__main__":
    main()
