"""Build helper for the optional C99 OpenMP GPST edge kernel (_edge_kernel.so)."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def build_edge_kernel(
    output_dir: Path | None = None,
    *,
    cc: str | None = None,
    openmp: bool = True,
) -> Path:
    """Compile ``edge_kernel.c`` into ``_edge_kernel.so`` with strict IEEE flags."""
    pkg_dir = Path(__file__).resolve().parent
    src = pkg_dir / "edge_kernel.c"
    out_dir = pkg_dir if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "_edge_kernel.so"

    compiler = cc or os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if not compiler:
        raise RuntimeError("No C compiler found to build GPST _edge_kernel.so")

    base_cmd = [
        compiler,
        "-std=c99",
        "-O3",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-shared",
        "-fPIC",
    ]
    fd, tmp_name = tempfile.mkstemp(
        dir=str(out_dir), prefix=".tmp_edge_kernel_", suffix=".so"
    )
    os.close(fd)
    tmp_dst = Path(tmp_name)
    try:
        cmd = [
            *base_cmd,
            *(("-fopenmp",) if openmp else ()),
            str(src),
            "-lm",
            "-o",
            str(tmp_dst),
        ]
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if proc.returncode != 0 and openmp:
            # Retry without OpenMP if toolchain lacks libgomp.
            cmd_no_omp = [*base_cmd, str(src), "-lm", "-o", str(tmp_dst)]
            proc = subprocess.run(
                cmd_no_omp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to compile {src}:\n{proc.stdout}")
        os.replace(tmp_dst, dst)
    except Exception:
        tmp_dst.unlink(missing_ok=True)
        raise
    return dst


if __name__ == "__main__":
    path = build_edge_kernel()
    print(f"Built {path}", file=sys.stdout)
