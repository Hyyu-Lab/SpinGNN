"""Protect source-checkout CLI calls from an installed package snapshot."""

import importlib.util
from pathlib import Path


def guard_source_entry():
    repo_root = Path(__file__).resolve().parents[4]
    bootstrap = repo_root / "_spingnn_source.py"
    if not bootstrap.is_file():
        # Wheels and standalone NequIP checkouts have no monorepo bootstrap.
        return
    spec = importlib.util.spec_from_file_location("_spingnn_source", bootstrap)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.activate_source_tree(repo_root)
