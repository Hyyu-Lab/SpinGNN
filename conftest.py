"""Apply source selection before any package's shared fixtures are imported."""

from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).with_name("_spingnn_source.py")))[
    "activate_source_tree"
]()
