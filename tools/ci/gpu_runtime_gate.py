#!/usr/bin/env python3
"""Require actual CUDA execution evidence, including when reusing a binary."""

import argparse
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def require_cuda_binary(path: Path) -> None:
    result = subprocess.run(
        ["ldd", str(path)], text=True, capture_output=True, check=False
    )
    if (
        result.returncode
        or "not found" in result.stdout
        or not any(name in result.stdout for name in ("libtorch_cuda", "libc10_cuda"))
    ):
        raise ValueError("GPU lane requires resolved CUDA LibTorch binary linkage")


def require_runtime_passes(path: Path) -> None:
    # Each parity test runs both backends and checks the actual CUDA device.
    expected = {
        "test_pt2_lammps_matches_torchscript_run0[nosoc]",
        "test_pt2_lammps_matches_torchscript_run0[soc]",
        "test_bec_pt2_lammps_matches_torchscript_and_python_run0",
    }
    actual = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        name = case.get("name")
        if (
            case.get("classname", "").rsplit(".", 1)[-1] != "test_pt2_parity"
            or name not in expected
        ):
            continue
        if name in actual:
            raise ValueError(f"duplicate CUDA runtime result: {name}")
        actual[name] = not any(
            case.find(tag) is not None for tag in ("skipped", "failure", "error")
        )
    if not expected.issubset(actual) or any(not actual[name] for name in expected):
        raise ValueError("GPU lane requires CUDA parity passes for NoSOC, SOC, and BEC")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    if not args.binary and not args.junit:
        parser.error("--binary or --junit is required")
    try:
        if args.binary:
            require_cuda_binary(args.binary)
        if args.junit:
            require_runtime_passes(args.junit)
    except (ValueError, OSError, ET.ParseError) as error:
        print(f"GPU_RUNTIME_GATE_FAIL: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
