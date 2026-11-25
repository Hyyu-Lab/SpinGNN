#!/usr/bin/env python3
"""Check a pytest warnings count against the committed CI budget."""

import argparse
import json
import re
import sys
from pathlib import Path


SUMMARY_RE = re.compile(
    r"(?:^|, )\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed)"
)
WARNING_RE = re.compile(r"(?:^|, )(\d+) warnings?(?:,| in )")


def _parse_warning_count(log_path: Path) -> int:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read pytest log {log_path}: {exc}") from exc

    for line in reversed(lines):
        if " in " not in line or SUMMARY_RE.search(line) is None:
            continue
        match = WARNING_RE.search(line)
        return int(match.group(1)) if match is not None else 0
    raise ValueError(f"cannot find a pytest result summary in {log_path}")


def _load_budget(budget_path: Path, suite: str) -> int:
    try:
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        schema_version = budget["schema_version"]
        maximum = budget["suites"][suite]["max_warnings"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid warning budget {budget_path}: {exc}") from exc

    if schema_version != 1:
        raise ValueError(f"unsupported warning budget schema: {schema_version!r}")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise ValueError(f"invalid max_warnings for {suite}: {maximum!r}")
    return maximum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--budget", required=True, type=Path)
    args = parser.parse_args()

    try:
        count = _parse_warning_count(args.log)
        maximum = _load_budget(args.budget, args.suite)
    except ValueError as exc:
        print(f"CI_WARNING_ERROR name={args.suite} message={exc}", file=sys.stderr)
        return 2

    result = "PASS" if count <= maximum else "FAIL"
    print(
        f"CI_WARNING_REPORT name={args.suite} count={count} "
        f"budget={maximum} result={result}"
    )
    if count > maximum:
        print(
            "ERROR: warning budget exceeded; fix the regression or update "
            "tools/ci/warnings-budget.json and tools/ci/warnings-inventory.md "
            "together",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
