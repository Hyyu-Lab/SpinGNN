#!/usr/bin/env python3
"""Run a Python module, file, or registered SpinGNN2 command from this checkout."""

import ast
from importlib.metadata import EntryPoint
from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(REPO_ROOT / "_spingnn_source.py"))["activate_source_tree"](
    include_spingnn2=(REPO_ROOT / "src/spingnn2").is_dir()
)


def console_scripts():
    """Read the source registry without executing setup.py or stale metadata."""
    tree = ast.parse((REPO_ROOT / "src/core/setup.py").read_text())
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id != "setup":
            continue
        for keyword in call.keywords:
            if keyword.arg == "entry_points":
                registry = ast.literal_eval(keyword.value)["console_scripts"]
                entries = {}
                for entry in registry:
                    name, separator, value = entry.partition("=")
                    if not separator:
                        raise RuntimeError(f"Invalid console script: {entry!r}")
                    entries[name.strip()] = EntryPoint(
                        name=name.strip(), value=value.strip(), group="console_scripts"
                    )
                if (REPO_ROOT / "src/spingnn2").is_dir():
                    for name, target in (
                        ("spingnn-data", "spingnn2.data.vasp_spin:main"),
                        ("spingnn2-export", "spingnn2.cli:main"),
                        ("spingnn2-prepare-jit-artifact", "spingnn2.jit:main"),
                        ("spingnn2-gpst-moments", "spingnn2.gpst.cli:main_moments"),
                        ("spingnn2-gpst-infer", "spingnn2.gpst.cli:main_infer"),
                    ):
                        entries[name] = EntryPoint(name, target, "console_scripts")
                return entries
    raise RuntimeError("Source setup.py has no literal console_scripts registry")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(
            "usage: python tools/ci/source_python.py "
            "{COMMAND | -m MODULE | -c CODE | SCRIPT.py} [args ...]"
        )
        print("commands: " + ", ".join(console_scripts()))
        return 0 if args else 2
    command, *tail = args
    if command in ("-m", "-c"):
        if not tail:
            raise SystemExit(f"{command} requires an argument")
        target, *tail = tail
        sys.argv = [target if command == "-m" else "-c", *tail]
        if command == "-m":
            runpy.run_module(target, run_name="__main__", alter_sys=True)
        else:
            exec(compile(target, "<string>", "exec"), {"__name__": "__main__"})
        return 0
    entries = console_scripts()
    if command in entries:
        sys.argv = [command, *tail]
        return entries[command].load()()
    script = Path(command).resolve()
    if not script.is_file():
        raise SystemExit(f"Unknown source command or Python file: {command}")
    sys.argv = [str(script), *tail]
    sys.path.insert(
        3 if (REPO_ROOT / "src/spingnn2").is_dir() else 2, str(script.parent)
    )
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
