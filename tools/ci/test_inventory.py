#!/usr/bin/env python3
"""Validate test-file ownership and emit the GitHub Python shard matrix.

This static gate does not import Torch or collect tests. Runtime parameter,
marker and skip coverage is recorded separately by pytest_evidence.py.
"""

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOTS = (
    "src/core/tests",
    "src/core/nequip",
    "src/models/tests",
    "src/models/allegro",
    "tests_lammps",
    "tests_ci",
)


def marker_matches(expression, markers):
    """Evaluate the manifest's boolean marker subset without executing code."""

    def evaluate(node):
        if isinstance(node, ast.Name):
            return node.id in markers
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not evaluate(node.operand)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [evaluate(value) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        raise ValueError(f"unsupported marker expression: {expression}")

    return evaluate(ast.parse(expression, mode="eval").body)


def validate_evidence(manifest, paths):
    """Check runtime node identities, including inherited and parameterized tests."""
    nodes = {}
    for path in paths:
        evidence = json.loads(path.read_text())
        for row in evidence["selected"] + evidence["deselected"]:
            nodes[row["nodeid"]] = set(row["markers"])
    lanes = [
        dict(lane, markers=manifest["python_markers"])
        for lane in manifest["python_lanes"]
    ] + manifest.get("additional_lanes", [])
    missing = []
    for nodeid, markers in nodes.items():
        filename = nodeid.split("::", 1)[0]
        if not any(
            any(
                filename == path or filename.startswith(path.rstrip("/") + "/")
                for path in lane["tests"]
            )
            and marker_matches(lane["markers"], markers)
            for lane in lanes
        ):
            missing.append(nodeid)
    if missing:
        raise ValueError(
            f"collected nodes without an execution lane: {sorted(missing)}"
        )
    if not nodes:
        raise ValueError("evidence contains no collected node identities")
    return len(nodes)


def validate_inventory(root, manifest):
    """Every test file must have exactly one base Python shard owner."""
    owners = defaultdict(list)
    names = set()
    for lane in manifest["python_lanes"]:
        name = lane["suite"]
        if name in names:
            raise ValueError(f"duplicate suite: {name}")
        names.add(name)
        if not isinstance(lane["xdist"], int) or lane["xdist"] < 0:
            raise ValueError(f"invalid xdist worker count: {name}")
        for selection in lane["tests"]:
            path = root / selection
            if not path.exists():
                raise ValueError(f"missing selection: {selection}")
            paths = path.rglob("test_*.py") if path.is_dir() else (path,)
            for test in paths:
                owners[test.relative_to(root).as_posix()].append(name)
    actual = {
        path.relative_to(root).as_posix()
        for directory in TEST_ROOTS
        for path in (root / directory).rglob("test_*.py")
    }
    missing = sorted(actual - owners.keys())
    duplicate = {path: lanes for path, lanes in owners.items() if len(lanes) != 1}
    if missing or duplicate:
        raise ValueError(f"unowned test files={missing}; duplicate owners={duplicate}")
    for lane in manifest.get("additional_lanes", []):
        if not (root / lane["entry"]).is_file():
            raise ValueError(f"missing lane entry: {lane['entry']}")
        for selection in lane["tests"]:
            if not (root / selection).exists():
                raise ValueError(f"missing selection in {lane['suite']}: {selection}")
    return owners


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--evidence", type=Path, nargs="+", help="Check pytest JSON collection evidence"
    )
    args = parser.parse_args(argv)
    manifest = json.loads((args.root / "tools/ci/test-lanes.json").read_text())
    if manifest.get("schema_version") != 1:
        parser.error("unsupported test inventory schema")
    try:
        owners = validate_inventory(args.root, manifest)
        node_count = (
            validate_evidence(manifest, args.evidence) if args.evidence else None
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.matrix:
        print(
            json.dumps(
                {
                    "include": [
                        {
                            "suite": lane["suite"],
                            "tests": " ".join(lane["tests"]),
                            "xdist": str(lane["xdist"]),
                            "markers": manifest["python_markers"],
                            "environment": lane.get("environment", "cpu"),
                        }
                        for lane in manifest["python_lanes"]
                    ]
                }
            )
        )
    else:
        print(
            f"TEST_INVENTORY_OK files={len(owners)} shards={len(manifest['python_lanes'])}"
        )
        if node_count is not None:
            print(f"TEST_NODE_INVENTORY_OK nodes={node_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
