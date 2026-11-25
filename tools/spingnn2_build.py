#!/usr/bin/env python3
"""Build a pinned LAMMPS executable, native library and Python wheel with Slurm."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/lammps"
MPI = Path("/data/app/intel/oneapi-2024.2.1/mpi/2021.13")
GCC = Path("/data/app/gcc/10.5.0")
LOCAL = Path.home() / ".local/bin"
SPEC = importlib.util.spec_from_file_location(
    "lammps_overlay", ROOT / "lammps_spingnn2/apply.py"
)
OVERLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OVERLAY)

NATIVE_SPEC = importlib.util.spec_from_file_location(
    "build_native", ROOT / "tools/native/build_native.py"
)
BUILD_NATIVE = importlib.util.module_from_spec(NATIVE_SPEC)
NATIVE_SPEC.loader.exec_module(BUILD_NATIVE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    files = {}
    for name in ("libspingnn2", "src/spingnn2", "lammps_spingnn2"):
        for path in sorted((ROOT / name).rglob("*")):
            if path.is_file() and not any(
                p == "__pycache__"
                or p.endswith(".egg-info")
                or p.endswith(".so")
                or p in {"build", "dist", "symmetry", "v2"}
                for p in path.relative_to(ROOT / name).parts
            ):
                files[str(path.relative_to(ROOT))] = digest(path)
    for name in (
        "pyproject.toml",
        "README.md",
        "tools/spingnn2_build.py",
        "tools/native/build_native.py",
    ):
        files[name] = digest(ROOT / name)
    return files


def version(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def _build_gpst_kernel(wheel_src, env):
    helper = wheel_src / "src/spingnn2/gpst/build_ext.py"
    if not helper.is_file():
        return
    spec = importlib.util.spec_from_file_location("spingnn2_gpst_build_ext", helper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_edge_kernel(output_dir=helper.parent, cc=str(GCC / "bin/gcc"))


def run(args):
    gpu = args.target != "cpu"
    arches = BUILD_NATIVE.parse_cuda_arches(args.target) if gpu else ()
    if args.cuda_root is not None:
        cuda = args.cuda_root.resolve()
    else:
        cuda = Path("/data/app/cuda") / ("12.9" if args.target == "sm89" else "12.8")
    cmake = str(LOCAL / "cmake")
    compiler = GCC / "bin/g++"
    env = os.environ.copy()
    env.update(
        PATH=os.pathsep.join(
            map(
                str,
                [
                    GCC / "bin",
                    MPI / "bin",
                    cuda / "bin",
                    Path(sys.executable).parent,
                    LOCAL,
                    env["PATH"],
                ],
            )
        ),
        I_MPI_CXX=str(compiler),
        I_MPI_CC=str(GCC / "bin/gcc"),
        I_MPI_ROOT=str(MPI),
        NVCC_WRAPPER_DEFAULT_COMPILER=str(compiler),
        PYTHONDONTWRITEBYTECODE="1",
        LD_LIBRARY_PATH=os.pathsep.join(
            map(
                str,
                [
                    GCC / "lib64",
                    MPI / "lib",
                    cuda / "lib64",
                    env.get("LD_LIBRARY_PATH", ""),
                ],
            )
        ),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    target_label = (
        "cpu"
        if not gpu
        else ("all-gpu" if args.target.strip() == "all-gpu" else "-".join(arches))
    )
    descriptor = {
        "target": target_label,
        "arches": list(arches),
        "tag": OVERLAY.TAG_OBJECT,
        "commit": OVERLAY.COMMIT,
        "python": [sys.executable, sys.version, sysconfig.get_config_var("SOABI")],
        "cxx": version([compiler, "--version"]),
        "cmake": version([cmake, "--version"]),
        "mpi": version([MPI / "bin/mpirun", "--version"]),
        "cuda": version([cuda / "bin/nvcc", "--version"]) if gpu else None,
        "files": sources(),
    }
    key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[
        :20
    ]
    build = OUT / "builds" / f"{target_label}-{key}"
    build.mkdir(parents=True, exist_ok=True)
    with (build / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (OUT / "upstream.lock").open("w") as base_lock:
            fcntl.flock(base_lock, fcntl.LOCK_EX)
            if not args.upstream.exists():
                subprocess.run(
                    [
                        "gh",
                        "repo",
                        "clone",
                        "lammps/lammps",
                        args.upstream,
                        "--",
                        "--depth",
                        "1",
                        "--branch",
                        OVERLAY.TAG,
                    ],
                    env=env,
                    check=True,
                )
            tree = OVERLAY.materialize(args.upstream.resolve(), build / "source")
        prefix = build / "install"
        prefix.mkdir(exist_ok=True)
        jobs = str(args.jobs)
        command = [
            sys.executable,
            "-B",
            ROOT / "tools/native/build_native.py",
            "--backend",
            "cuda" if gpu else "cpu",
            "--cmake",
            cmake,
            "--cxx",
            compiler,
            "--build-dir",
            build / "native",
            "--install-prefix",
            prefix,
            "--jobs",
            jobs,
        ]
        if gpu:
            command.extend(["--arch", args.target, "--cuda-root", cuda])
        subprocess.run(command, env=env, check=True)
        wheel_src = build / "python"
        if wheel_src.exists():
            shutil.rmtree(wheel_src)
        wheel_src.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "pyproject.toml", wheel_src / "pyproject.toml")
        shutil.copy2(ROOT / "README.md", wheel_src / "README.md")
        shutil.copytree(
            ROOT / "src/spingnn2",
            wheel_src / "src/spingnn2",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.egg-info", "dist", "build", "_edge_kernel.so"
            ),
        )
        _build_gpst_kernel(wheel_src, env)
        subprocess.run(
            [
                LOCAL / "uv",
                "build",
                "--wheel",
                "--no-build-isolation",
                "--python",
                sys.executable,
                "--out-dir",
                prefix / "wheel",
                wheel_src,
            ],
            env=env,
            check=True,
        )
        upstream = build / "source"
        # The loader must find the matching C++ runtime before MPI or the
        # executable pulls in the system libstdc++. Native libraries travel
        # with the prefix; MPI and the selected toolchains live on the host.
        runtime_dirs = [GCC / "lib64", MPI / "lib"]
        if gpu:
            runtime_dirs.append(cuda / "lib64")
        command = [
            cmake,
            "-S",
            upstream / "cmake",
            "-B",
            build / "lammps",
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_MPI=ON",
            "-DBUILD_OMP=ON",
            "-DPKG_SPIN=ON",
            "-DPKG_KOKKOS=ON",
            "-DKokkos_ENABLE_OPENMP=ON",
            "-DCMAKE_CXX_STANDARD=20",
            "-DLAMMPS_MACHINE=spingnn2",
            f"-DCMAKE_PREFIX_PATH={prefix}",
            f"-DMPI_CXX_COMPILER={MPI}/bin/mpicxx",
            f"-DCMAKE_C_COMPILER={GCC}/bin/gcc",
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            "-DCMAKE_INSTALL_RPATH=" + ";".join(map(str, runtime_dirs)),
            f"-DCMAKE_CXX_COMPILER={upstream / 'lib/kokkos/bin/nvcc_wrapper' if gpu else compiler}",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DWITH_PNG=OFF",
            "-DWITH_JPEG=OFF",
            "-DWITH_FFMPEG=OFF",
            "-DPKG_OPENMP=ON" if not gpu else "-DPKG_OPENMP=OFF",
        ]
        lammps_env = env.copy()
        if gpu:
            base_kokkos_arch = BUILD_NATIVE.CUDA_ARCHITECTURES[arches[0]]
            if len(arches) > 1:
                shim_dir = build / "lammps-shim-bin"
                BUILD_NATIVE.write_nvcc_fatbin_shim(shim_dir, cuda / "bin/nvcc", arches)
                lammps_env.pop("CUDA_ROOT", None)
                lammps_env["PATH"] = os.pathsep.join(
                    [str(shim_dir), lammps_env["PATH"]]
                )
            command.extend(
                [
                    "-DKokkos_ENABLE_CUDA=ON",
                    f"-DCMAKE_USER_MAKE_RULES_OVERRIDE_CXX={ROOT}/lammps_spingnn2/cmake/cuda-cxx20.cmake",
                    f"-DKokkos_ARCH_{base_kokkos_arch}=ON",
                    f"-DCUDAToolkit_ROOT={cuda}",
                ]
            )
        subprocess.run(command, env=lammps_env, check=True)
        subprocess.run(
            [cmake, "--build", build / "lammps", "--target", "lmp", "-j", jobs],
            env=lammps_env,
            check=True,
        )
        subprocess.run(
            [cmake, "--install", build / "lammps", "--component", "spingnn2_runtime"],
            env=lammps_env,
            check=True,
        )
        binary = prefix / "bin/lmp_spingnn2"
        if sources() != descriptor["files"]:
            raise RuntimeError("Build inputs changed before completion")
        products = [
            binary,
            *prefix.glob("lib*/libspingnn2*.so"),
            *prefix.glob("wheel/*.whl"),
        ]
        descriptor.update(
            overlay_tree=tree, products={str(p): digest(p) for p in products}
        )
        manifest = build / "manifest.json"
        tmp = build / "manifest.tmp"
        tmp.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")
        tmp.replace(manifest)
        print(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        required=True,
        help="Target: cpu, sm70, sm80, sm86, sm89, sm90, sm120, comma-separated list, or all-gpu",
    )
    parser.add_argument("--upstream", type=Path, default=OUT / "upstream")
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--partition", default="xiang")
    parser.add_argument(
        "--jobs", type=int, default=int(os.getenv("SLURM_CPUS_PER_TASK", "8"))
    )
    args = parser.parse_args()
    if args.target != "cpu":
        try:
            BUILD_NATIVE.parse_cuda_arches(args.target)
        except ValueError as exc:
            parser.error(str(exc))
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if "SLURM_JOB_ID" in os.environ:
        run(args)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    label = args.target.replace(",", "-")
    command = [
        "sbatch",
        "--wait",
        "--parsable",
        "-p",
        args.partition,
        "-c",
        str(args.jobs),
        "--mem=60G",
        "-N1",
        "-n1",
        "-J",
        f"spingnn2-{label}",
        "-o",
        str(OUT / f"build-{label}-%j.log"),
    ]
    if args.partition not in ("xiang",):
        command.append("--gres=gpu:1")
    command += [
        "--wrap",
        shlex.join(
            [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]]
        ),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
