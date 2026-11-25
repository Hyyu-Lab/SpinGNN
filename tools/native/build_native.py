#!/usr/bin/env python3
"""Configure and build the SpinGNN2 C ABI library in an out-of-tree directory."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
KOKKOS_VERSION = "4.7.02"
KOKKOS_HASH = "de8a1a137cf2110d391a21454406f4f5ea367c7610b540a3961f7bf1246b4ff8"
JSON_VERSION = "3.11.3"
JSON_HASH = "0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406"
CUDA_ARCHITECTURES = {
    "sm70": "VOLTA70",
    "sm80": "AMPERE80",
    "sm86": "AMPERE86",
    "sm89": "ADA89",
    "sm90": "HOPPER90",
    "sm120": "BLACKWELL120",
}
ALL_GPU_ARCHES = ("sm70", "sm80", "sm86", "sm89", "sm90", "sm120")
CLUSTER_GCC = Path("/data/app/gcc/10.5.0")
CLUSTER_MKL = Path("/data/app/intel/oneapi-2024.2.1/mkl/2024.2")
CLUSTER_CUDA = Path("/data/app/cuda/12.8")


def parse_cuda_arches(spec):
    """Parse single arch, comma/semicolon list, or 'all-gpu' into sorted smXX tuple."""
    if not spec or not str(spec).strip():
        raise ValueError("CUDA architecture specification must not be empty")
    text = str(spec).strip()
    if text == "all-gpu":
        return ALL_GPU_ARCHES
    seen = []
    for raw in text.replace(";", ",").split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok == "all-gpu":
            for item in ALL_GPU_ARCHES:
                if item not in seen:
                    seen.append(item)
            continue
        norm = tok.replace("_", "")
        if norm not in CUDA_ARCHITECTURES:
            raise ValueError(
                f"Unsupported CUDA architecture {tok!r}; expected one of "
                f"{', '.join(ALL_GPU_ARCHES)} or 'all-gpu'"
            )
        if norm not in seen:
            seen.append(norm)
    if not seen:
        raise ValueError("CUDA architecture specification must not be empty")
    return tuple(sorted(seen, key=lambda a: int(a[2:])))


def canonical_sm_name(arch):
    """Convert 'sm80' -> 'sm_80'."""
    return f"sm_{arch[2:]}"


def write_nvcc_fatbin_shim(shim_dir, real_nvcc, arches):
    """Write an nvcc shim injecting multi-arch -gencode flags for compile and -dlink."""
    shim_dir = Path(shim_dir)
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "nvcc"
    nums = [int(a[2:]) for a in arches]
    compile_gencodes = [f"-gencode=arch=compute_{n},code=sm_{n}" for n in nums]
    compile_gencodes.append(f"-gencode=arch=compute_{nums[-1]},code=compute_{nums[-1]}")
    dlink_gencodes = [f"-gencode=arch=compute_{n},code=sm_{n}" for n in nums]
    script = f"""#!/usr/bin/env python3
import os
import sys

REAL_NVCC = {str(Path(real_nvcc).resolve())!r}
COMPILE_GENCODES = {compile_gencodes!r}
DLINK_GENCODES = {dlink_gencodes!r}

args = sys.argv[1:]
passthrough = any(
    a in ("--version", "-V", "--help", "-h", "-E", "--preprocess", "--dryrun")
    for a in args
)
if passthrough:
    os.execv(REAL_NVCC, [REAL_NVCC, *args])

is_dlink = any(a in ("-dlink", "--device-link") for a in args)
filtered = []
saw_arch = False
i = 0
while i < len(args):
    arg = args[i]
    if arg in ("-arch", "--gpu-architecture", "-gencode", "--generate-code"):
        saw_arch = True
        i += 2
        continue
    if arg.startswith(("-arch=", "--gpu-architecture=", "-gencode=", "--generate-code=")):
        saw_arch = True
        i += 1
        continue
    filtered.append(arg)
    i += 1

needs_gencode = saw_arch or is_dlink or any(a in ("-c", "--compile") for a in filtered)
if needs_gencode:
    gencodes = DLINK_GENCODES if is_dlink else COMPILE_GENCODES
    filtered = [*gencodes, *filtered]

os.execv(REAL_NVCC, [REAL_NVCC, *filtered])
"""
    shim.write_text(script, encoding="utf-8")
    shim.chmod(0o755)
    return shim


def fetch_source(cache, repository, tag, dirname, expected_hash):
    source = cache / dirname
    if (source / "CMakeLists.txt").is_file():
        return source
    cache.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{repository}/tar.gz/refs/tags/{tag}"
    with tempfile.TemporaryDirectory(dir=cache) as work:
        work = Path(work)
        archive = work / "source.tar.gz"
        digest = hashlib.sha256()
        with urlopen(url, timeout=60) as response, archive.open("wb") as output:
            while data := response.read(1024 * 1024):
                output.write(data)
                digest.update(data)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"{repository} archive failed SHA-256 verification")
        with tarfile.open(archive) as source_archive:
            # The source archive has a pinned checksum.
            source_archive.extractall(work)
        try:
            (work / source.name).rename(source)
        except OSError:
            if not (source / "CMakeLists.txt").is_file():
                raise
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"))
    parser.add_argument(
        "--arch",
        help="CUDA arch: sm70|sm80|sm86|sm89|sm90|sm120, comma-separated list, or all-gpu",
    )
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument(
        "--dependency-dir", type=Path, default=ROOT / "validation/native/deps"
    )
    parser.add_argument("--kokkos-source", type=Path)
    parser.add_argument("--json-source", type=Path)
    parser.add_argument("--cxx", type=Path, help="C++ compiler (CUDA host compiler)")
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--mkl-root", type=Path)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument(
        "--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    )
    parser.add_argument(
        "--probe", action="store_true", help="Build the backend smoke executable"
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Configure before engine sources exist",
    )
    parser.add_argument("--install-prefix", type=Path)
    parser.add_argument(
        "--prepare-dependencies",
        action="store_true",
        help="Populate the source cache without configuring or building",
    )
    args = parser.parse_args()

    if args.backend is None and not args.prepare_dependencies:
        parser.error("--backend is required for a library or probe build")
    cuda = args.backend == "cuda"
    if cuda != (args.arch is not None):
        parser.error("--arch is required for CUDA and only applies to CUDA")
    if args.probe_only and not args.probe:
        parser.error("--probe-only requires --probe")
    if args.probe and args.install_prefix is not None:
        parser.error("--install-prefix requires a library build")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    cache = args.dependency_dir.resolve()
    kokkos = args.kokkos_source or fetch_source(
        cache, "kokkos/kokkos", KOKKOS_VERSION, f"kokkos-{KOKKOS_VERSION}", KOKKOS_HASH
    )
    json_source = args.json_source or fetch_source(
        cache, "nlohmann/json", f"v{JSON_VERSION}", f"json-{JSON_VERSION}", JSON_HASH
    )
    for source in (kokkos, json_source):
        if not (source / "CMakeLists.txt").is_file():
            parser.error(
                f"No CMakeLists.txt under {source}. Check the dependency source path."
            )
    if args.prepare_dependencies:
        print(kokkos.resolve())
        print(json_source.resolve())
        return

    arches = ()
    if cuda:
        try:
            arches = parse_cuda_arches(args.arch)
        except ValueError as exc:
            parser.error(str(exc))
    if not cuda:
        label = args.backend
    elif args.arch.strip() == "all-gpu":
        label = "cuda-all-gpu"
    elif len(arches) == 1:
        label = f"cuda-{arches[0]}"
    else:
        label = "cuda-" + "-".join(arches)
    build = (args.build_dir or ROOT / f"validation/native/build-{label}").resolve()
    cxx = args.cxx or (
        CLUSTER_GCC / "bin/g++" if CLUSTER_GCC.is_dir() else shutil.which("g++")
    )
    if cxx is None or not Path(cxx).is_file():
        parser.error("No C++ compiler found. Specify --cxx.")
    cxx = Path(cxx).absolute()
    environment = os.environ.copy()
    command = [
        args.cmake,
        "-S",
        str(ROOT / "libspingnn2"),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DSP2_ENABLE_CUDA={'ON' if cuda else 'OFF'}",
        f"-DSP2_BUILD_PROBE_ONLY={'ON' if args.probe_only else 'OFF'}",
        f"-DSP2_KOKKOS_SOURCE={kokkos.resolve()}",
        f"-DSP2_JSON_SOURCE={json_source.resolve()}",
    ]
    if shutil.which("ninja", path=environment.get("PATH")):
        command += ["-G", "Ninja"]
    gcc_lib = cxx.parent.parent / "lib64"
    if (gcc_lib / "libstdc++.so.6").is_file():
        command.extend(
            (f"-DCMAKE_BUILD_RPATH={gcc_lib}", f"-DCMAKE_INSTALL_RPATH={gcc_lib}")
        )
    if cuda:
        cuda_root = args.cuda_root
        if cuda_root is None:
            cuda_root = (
                CLUSTER_CUDA if CLUSTER_CUDA.is_dir() else Path("/usr/local/cuda")
            )
        cuda_root = cuda_root.resolve()
        if not (cuda_root / "bin/nvcc").is_file():
            parser.error(f"No nvcc under {cuda_root}. Specify --cuda-root.")
        wrapper = kokkos.resolve() / "bin/nvcc_wrapper"
        if not wrapper.is_file():
            parser.error(f"No nvcc_wrapper under {kokkos}. Check --kokkos-source.")
        path_entries = [str(cuda_root / "bin"), environment.get("PATH", "")]
        if len(arches) > 1:
            shim_dir = build / "shim-bin"
            write_nvcc_fatbin_shim(shim_dir, cuda_root / "bin/nvcc", arches)
            path_entries.insert(0, str(shim_dir))
            # Kokkos nvcc_wrapper prefers $CUDA_ROOT/bin/nvcc over PATH; unset
            # CUDA_ROOT so it invokes the fat-binary shim on PATH while CMake
            # still receives -DCUDAToolkit_ROOT for libraries and headers.
            environment.pop("CUDA_ROOT", None)
        else:
            environment["CUDA_ROOT"] = str(cuda_root)
        environment.update(
            NVCC_WRAPPER_DEFAULT_COMPILER=str(cxx),
            PATH=os.pathsep.join(path_entries),
        )
        # nvcc_wrapper may not add paths inferred from its implicit CUDA libs.
        cuda_lib = cuda_root / "lib64"
        runtime_paths = ";".join(
            str(path) for path in (gcc_lib, cuda_lib) if path.is_dir()
        )
        base_arch = arches[0]
        sp2_cuda_arches = ";".join(canonical_sm_name(a) for a in arches)
        command.extend(
            (
                f"-DCMAKE_CXX_COMPILER={wrapper}",
                f"-DCUDAToolkit_ROOT={cuda_root}",
                f"-DCMAKE_BUILD_RPATH={runtime_paths}",
                f"-DCMAKE_INSTALL_RPATH={runtime_paths}",
                f"-DSP2_CUDA_ARCHES={sp2_cuda_arches}",
            )
        )
        for arch, flag in CUDA_ARCHITECTURES.items():
            command.append(
                f"-DKokkos_ARCH_{flag}={'ON' if base_arch == arch else 'OFF'}"
            )
    else:
        command.append(f"-DCMAKE_CXX_COMPILER={cxx}")
        mkl_root = args.mkl_root or (CLUSTER_MKL if CLUSTER_MKL.is_dir() else None)
        if mkl_root:
            command.append(f"-DSP2_CPU_BLAS_ROOT={mkl_root.resolve()}")

    subprocess.run(command, check=True, env=environment)
    target = "sp2_toolchain_probe" if args.probe else "spingnn2"
    subprocess.run(
        [
            args.cmake,
            "--build",
            str(build),
            "--target",
            target,
            "--parallel",
            str(args.jobs),
        ],
        check=True,
        env=environment,
    )
    if args.install_prefix is not None:
        subprocess.run(
            [
                args.cmake,
                "--install",
                str(build),
                "--prefix",
                str(args.install_prefix.resolve()),
            ],
            check=True,
            env=environment,
        )
    print(build / ("sp2_toolchain_probe" if args.probe else "libspingnn2.so"))


if __name__ == "__main__":
    main()
