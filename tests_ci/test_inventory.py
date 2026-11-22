"""The CI manifest must not silently drop new test modules."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "test_inventory_tool", ROOT / "tools/ci/test_inventory.py"
)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def test_repository_inventory_is_complete():
    manifest = json.loads((ROOT / "tools/ci/test-lanes.json").read_text())
    assert inventory.validate_inventory(ROOT, manifest)


def test_missing_additional_lane_selection_is_rejected(tmp_path):
    entry = tmp_path / "run-tests.sh"
    entry.touch()
    manifest = {
        "python_lanes": [],
        "additional_lanes": [
            {"suite": "extra", "entry": entry.name, "tests": ["tests_ci/test_old.py"]}
        ],
    }
    with pytest.raises(ValueError, match="missing selection in extra: tests_ci/"):
        inventory.validate_inventory(tmp_path, manifest)


def test_new_module_requires_a_lane(tmp_path):
    test = tmp_path / "src/models/tests/model/test_new.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_new(): pass\n")
    with pytest.raises(ValueError, match="unowned.*test_new.py"):
        inventory.validate_inventory(tmp_path, {"python_lanes": []})


@pytest.mark.parametrize("package", ["src/core/nequip", "src/models/allegro"])
def test_library_embedded_regression_requires_a_lane(tmp_path, package):
    test = tmp_path / package / "utils" / "test_internal.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_internal(): pass\n")
    with pytest.raises(ValueError, match="unowned.*test_internal.py"):
        inventory.validate_inventory(tmp_path, {"python_lanes": []})


def test_duplicate_ownership_is_rejected(tmp_path):
    test = tmp_path / "tests_ci/test_control.py"
    test.parent.mkdir()
    test.write_text("def test_control(): pass\n")
    lanes = [{"suite": name, "tests": ["tests_ci"], "xdist": 0} for name in ("a", "b")]
    with pytest.raises(ValueError, match="duplicate owners"):
        inventory.validate_inventory(tmp_path, {"python_lanes": lanes})


def test_runtime_node_marker_gap_is_rejected(tmp_path):
    path = tmp_path / "collection.json"
    path.write_text(
        json.dumps(
            {
                "selected": [
                    {"nodeid": "tests_ci/test_example.py::test_gpu", "markers": ["gpu"]}
                ],
                "deselected": [],
            }
        )
    )
    manifest = {
        "python_markers": "not gpu",
        "python_lanes": [{"suite": "cpu", "tests": ["tests_ci"], "xdist": 0}],
    }
    with pytest.raises(ValueError, match="without an execution lane"):
        inventory.validate_evidence(manifest, [path])


@pytest.mark.parametrize(
    "expression,markers,expected",
    [
        ("gpu or compile", {"compile"}, True),
        ("gpu and not slow", {"gpu", "slow"}, False),
        ("(gpu and not slow) or (compile and not gpu)", {"compile"}, True),
    ],
)
def test_marker_boolean_expression(expression, markers, expected):
    assert inventory.marker_matches(expression, markers) is expected


@pytest.mark.parametrize(
    "filename,markers",
    [
        ("src/core/tests/unit/model/test_nequip_model.py", ["slow"]),
        ("src/models/tests/model/test_allegro.py", ["slow"]),
        (
            "src/core/tests/unit/trainer/test_compiled_optional_ddp.py",
            ["gpu", "slow"],
        ),
    ],
)
def test_slow_runtime_nodes_have_execution_lanes(tmp_path, filename, markers):
    manifest = json.loads((ROOT / "tools/ci/test-lanes.json").read_text())
    path = tmp_path / "slow-collection.json"
    path.write_text(
        json.dumps(
            {
                "selected": [],
                "deselected": [
                    {
                        "nodeid": filename + "::test_inherited[float64]",
                        "markers": markers,
                    }
                ],
            }
        )
    )
    assert inventory.validate_evidence(manifest, [path]) == 1
