#!/usr/bin/env python3
"""Materialize or verify a modern LAMMPS source tree from the pinned overlay."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
TAG = "stable_30Sep2026_update1"
TAG_OBJECT = "17ea1b9effad496ec60d0f079e8b0aaa4c5b6d89"
COMMIT = "098ffdefa525cabc82272925de02e9884f58c657"


def git(repo, *args, env=None, data=None):
    return subprocess.run(
        ["git", "-C", repo, *args],
        input=data,
        stdout=subprocess.PIPE,
        env=env,
        check=True,
    ).stdout


def check_pin(repo):
    for ref, expected in (
        (f"refs/tags/{TAG}", TAG_OBJECT),
        (f"{TAG}^{{commit}}", COMMIT),
    ):
        if git(repo, "rev-parse", "--verify", ref).decode().strip() != expected:
            raise ValueError(f"LAMMPS pin mismatch for {ref}, expected {expected}")


def overlay_tree(repo, index):
    # Build the expected index independently of the working files. Applying
    # patches here verifies their context against the exact upstream blobs.
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    git(repo, "read-tree", COMMIT, env=env)
    for patch in sorted((ROOT / "patches").glob("*.patch")):
        git(repo, "apply", "--cached", "--whitespace=error", patch, env=env)
    for source in sorted((ROOT / "overlay").rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(ROOT / "overlay").as_posix()
        if git(repo, "ls-files", "--", relative, env=env):
            raise ValueError(f"Use a patch for upstream file {relative}")
        blob = git(repo, "hash-object", "-w", "--", source).decode().strip()
        git(
            repo,
            "update-index",
            "--add",
            "--cacheinfo",
            f"100644,{blob},{relative}",
            env=env,
        )
    return git(repo, "write-tree", env=env).decode().strip()


def verify(repo):
    check_pin(repo)
    if git(repo, "rev-parse", "HEAD").decode().strip() != COMMIT:
        raise ValueError("Unexpected upstream HEAD in overlay checkout")
    with tempfile.TemporaryDirectory(dir=repo.parent) as tmp:
        expected = overlay_tree(repo, Path(tmp) / "index")
    git(repo, "diff", "--quiet", expected, "--")
    if git(repo, "ls-files", "--others", "-z"):
        raise ValueError("Unexpected untracked files in overlay checkout")
    return expected


def materialize(upstream, destination):
    check_pin(upstream)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return verify(destination)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        checkout = Path(tmp) / "source"
        subprocess.run(
            ["git", "clone", "--local", "--no-checkout", upstream, checkout], check=True
        )
        git(checkout, "checkout", "--quiet", "--detach", COMMIT)
        tree = overlay_tree(checkout, Path(tmp) / "index")
        git(checkout, "read-tree", "--reset", "-u", tree)
        verify(checkout)
        checkout.rename(destination)
    return tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    dst = args.destination.resolve()
    if args.check:
        tree = verify(dst)
    else:
        if not args.upstream:
            parser.error("--upstream is required to apply the overlay")
        tree = materialize(args.upstream.resolve(), dst)
    print(f"{TAG} {COMMIT} overlay-tree {tree}")


if __name__ == "__main__":
    main()
