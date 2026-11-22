"""Select this checkout before importing either SpinGNN Python package."""

import importlib.util
import os
from pathlib import Path
import sys


def activate_source_tree(repo_root=None):
    """Prefix both package roots, rejecting any already imported foreign code."""
    repo = Path(repo_root or __file__).resolve()
    if repo_root is None:
        repo = repo.parent
    packages = {
        "nequip": repo / "src" / "core" / "nequip",
        "allegro": repo / "src" / "models" / "allegro",
    }
    for name, package in packages.items():
        if not (package / "__init__.py").is_file():
            raise RuntimeError(f"SpinGNN source tree is incomplete: {package}")
        for loaded_name, module in tuple(sys.modules.items()):
            if loaded_name != name and not loaded_name.startswith(name + "."):
                continue
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(package):
                raise RuntimeError(
                    f"SpinGNN source-exactness violation: {loaded_name} already "
                    f"loaded from {origin!r}; expected {package}. Restart through "
                    f"{repo / 'tools/ci/source_python.py'} or put both checkout "
                    "package roots first in PYTHONPATH before starting Python."
                )

    roots = [str(package.parent) for package in packages.values()]
    sys.path[:] = roots + [entry for entry in sys.path if entry not in roots]
    inherited = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        roots + [entry for entry in inherited if entry and entry not in roots]
    )
    importlib.invalidate_caches()
    origins = {}
    for name, package in packages.items():
        spec = importlib.util.find_spec(name)
        origin = spec.origin if spec is not None else None
        if origin is None or Path(origin).resolve() != package / "__init__.py":
            raise RuntimeError(
                f"SpinGNN source-exactness violation: {name} resolves to "
                f"{origin!r}; expected {package / '__init__.py'}"
            )
        origins[name] = str(Path(origin).resolve())
    return origins
