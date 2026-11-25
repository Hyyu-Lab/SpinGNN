"""Native SpinGNN2 parity, cache, and Python ABI integration."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress
import numpy as np
import pytest
import torch

from nequip.utils._global_options import global_options_context
from spingnn2.calculator import SpinGNN2Calculator
from spingnn2.export import export_model
from spingnn2.native import NativeModel
from spingnn2.reference import ReferenceModel
from test_contract_oracle import (
    CASES,
    MAGNETIC_CASES,
    REPO,
    _config,
    _model,
    _observables,
    _sample,
)


BACKEND = os.environ.get("SPINGNN2_TEST_BACKEND", "cpu")
FP32_ABS = 5e-5
FP64_ABS = 1e-10
FP32_EVIDENTIAL_REL = 1e-4


@dataclass
class NativeCase:
    data: dict
    oracle: ReferenceModel
    eager: dict
    manifest: Path
    binary: Path


def _eager(model, data):
    device = "cuda" if BACKEND == "cuda" else "cpu"
    output = model(
        {key: torch.from_numpy(value.copy()).to(device) for key, value in data.items()}
    )
    return {key: value.detach().cpu().numpy() for key, value in output.items()}


@pytest.fixture(scope="module")
def native_case(tmp_path_factory):
    if BACKEND not in ("cpu", "cuda"):
        pytest.fail(f"Unknown native backend {BACKEND}")
    suffix = (
        "cpu"
        if BACKEND == "cpu"
        else "cuda-" + os.environ.get("SPINGNN2_ARCH", "sm120")
    )
    build = REPO / "validation" / "native" / ("build-" + suffix)
    library = Path(os.environ.get("SPINGNN2_LIBRARY", build / "libspingnn2.so"))
    if not library.is_file():
        if "SPINGNN2_TEST_BACKEND" in os.environ or "SPINGNN2_LIBRARY" in os.environ:
            pytest.fail(f"Requested native library does not exist: {library}")
        pytest.skip(
            "Build libspingnn2 or specify SPINGNN2_LIBRARY to exercise native tests"
        )
    if BACKEND == "cuda" and not torch.cuda.is_available():
        pytest.fail("CUDA parity requires an allocated CUDA device")
    built = {}

    def fixture(case, dtype="float64"):
        key = (case.name, dtype)
        if key not in built:
            config = _config(case, dtype)
            data = _sample(config)
            stem = tmp_path_factory.mktemp(case.name) / dtype
            with global_options_context(config):
                model = _model(case, config)
                manifest, binary = export_model(model, stem)
                model.to("cuda" if BACKEND == "cuda" else "cpu")
                eager = _eager(model, data)
            built[key] = NativeCase(
                data, ReferenceModel.load(manifest), eager, manifest, binary
            )
        return built[key]

    return fixture


def _record(case, what, values):
    payload = {"backend": BACKEND, "case": case.name, "check": what, **values}
    print("M2_RESULT", json.dumps(payload, sort_keys=True))


def _assert_results(case, what, actual, expected, tolerance, amp_aware=False):
    errors = {}
    maxima = {}
    for key in _observables(case):
        value = actual[key]
        truth = expected[key]
        assert value.shape == truth.shape, key
        assert value.dtype == truth.dtype, key
        assert np.isfinite(value).all(), key
        delta = np.abs(value - truth)
        errors[key] = float(np.max(delta, initial=0))
        evidence = key.startswith(("f_", "sf_", "total_f_", "total_sf_"))
        if evidence and amp_aware:
            maxima[key + "_normalized"] = float(
                np.max(delta / np.maximum(1, np.abs(truth)), initial=0)
            )
        else:
            maxima[key] = errors[key]
    _record(case, what, {**errors, **maxima})
    for key, value in maxima.items():
        limit = FP32_EVIDENTIAL_REL if key.endswith("_normalized") else tolerance
        assert value < limit, (case.name, what, key, value, limit)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_native_parity(native_case, case, dtype):
    fixture = native_case(case, dtype)
    is_double = dtype == "float64"
    tolerance = FP64_ABS if is_double else FP32_ABS
    with NativeModel.load(
        fixture.manifest, backend=BACKEND, precision=dtype, tile_atoms=3
    ) as model:
        actual = model.evaluate(fixture.data)
        _assert_results(
            case, f"eager-{dtype}", actual, fixture.eager, tolerance, not is_double
        )
        _assert_results(
            case,
            f"oracle-{dtype}",
            actual,
            fixture.oracle.evaluate(fixture.data),
            tolerance,
            not is_double,
        )
        if case.magnetic:
            zero = np.all(fixture.data["spin"] == 0, axis=1)
            assert not np.any(actual["spin_forces"][zero])
            assert not np.any(actual["spin_forces_vert"][zero])
            if case.evidence == "sf":
                assert not np.any(actual["sf_sigma"][zero])
        if case.family == "born" and case.asr == "hard":
            np.testing.assert_allclose(
                actual["born_effective_charges"].sum(0), 0, atol=tolerance
            )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_native_spline_and_tiling(native_case, case):
    fixture = native_case(case)
    with NativeModel.load(
        fixture.manifest, backend=BACKEND, radial_mode="spline"
    ) as model:
        actual = model.evaluate(fixture.data, tile_atoms=3)
        expected = fixture.oracle.evaluate(
            fixture.data, tile_atoms=3, radial_mode="spline"
        )
        _assert_results(case, "spline-oracle-float64", actual, expected, FP64_ABS)
        for mode in ("analytic", "spline"):
            whole = model.evaluate(fixture.data, radial_mode=mode, tile_atoms=None)
            for size in (1, 3):
                tiled = model.evaluate(fixture.data, radial_mode=mode, tile_atoms=size)
                _assert_results(case, f"tile-{mode}-{size}", tiled, whole, FP64_ABS)
                assert model.diagnostics["peak_tile_edges"] > 0
                assert model.diagnostics["workspace_bytes"] > 0


@pytest.mark.parametrize("case", MAGNETIC_CASES, ids=lambda case: case.name)
def test_native_lattice_cache(native_case, case):
    fixture = native_case(case)
    options = {"backend": BACKEND, "tile_atoms": 4, "radial_mode": "spline"}
    lattice = {key: val for key, val in fixture.data.items() if key != "spin"}
    with NativeModel.load(fixture.manifest, **options) as model, NativeModel.load(
        fixture.manifest, cache_device_limit_bytes=0, **options
    ) as spill:
        with model.evaluate_stage1_lattice(
            lattice
        ) as cache, spill.evaluate_stage1_lattice(lattice) as host_cache:
            for step in (1, 2):
                spin = np.roll(fixture.data["spin"], step, axis=0)
                direct = model.evaluate({**lattice, "spin": spin})
                cached = model.evaluate_stage2_spin_cached(cache, spin)
                host = spill.evaluate_stage2_spin_cached(host_cache, spin)
                _assert_results(case, f"cache-{step}", cached, direct, FP64_ABS)
                _assert_results(
                    case, f"cache-forced-host-{step}", host, cached, FP64_ABS
                )
                assert spill.diagnostics["cache_host_bytes"] > 0
                assert spill.diagnostics["cache_device_bytes"] == 0
                if BACKEND == "cpu":
                    assert model.diagnostics["cache_host_bytes"] > 0
                    assert model.diagnostics["cache_device_bytes"] == 0
                else:
                    assert model.diagnostics["cache_host_bytes"] == 0
                    assert model.diagnostics["cache_device_bytes"] > 0


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c.name in ("born-hard", "soc-e2ip-spin")],
    ids=lambda case: case.name,
)
def test_native_ase_calculator(native_case, case):
    fixture = native_case(case)
    graph = fixture.data
    config = _config(case, "float64")
    species = sorted(
        config["chemical_symbol_to_type"], key=config["chemical_symbol_to_type"].get
    )
    atoms = Atoms(
        [species[i] for i in graph["atom_types"]],
        positions=graph["pos"],
        cell=graph["cell"][0],
        pbc=True,
    )
    atoms.set_array("spin", graph["spin"])
    with NativeModel.load(fixture.manifest, backend=BACKEND) as model:
        atoms.calc = SpinGNN2Calculator(model, type_names=species)
        expected = model.evaluate(graph)
        assert (
            abs(atoms.get_potential_energy() - expected["total_energy"].item())
            < FP64_ABS
        )
        np.testing.assert_allclose(
            atoms.get_forces(), expected["forces"], rtol=0, atol=FP64_ABS
        )
        np.testing.assert_allclose(
            atoms.get_stress(),
            full_3x3_to_voigt_6_stress(expected["stress"][0]),
            rtol=0,
            atol=FP64_ABS,
        )
        if case.magnetic:
            np.testing.assert_allclose(
                atoms.calc.get_property("spin_forces", atoms),
                expected["spin_forces"],
                rtol=0,
                atol=FP64_ABS,
            )
            np.testing.assert_allclose(
                atoms.calc.get_property("sf_Uepi", atoms),
                expected["sf_Uepi"],
                rtol=0,
                atol=FP64_ABS,
            )
            atoms.set_array("spin", np.roll(graph["spin"], 1, axis=0))
            updated = atoms.get_potential_energy()
            assert abs(updated - expected["total_energy"].item()) > 1e-8
        else:
            np.testing.assert_allclose(
                atoms.calc.get_property("born_effective_charges", atoms),
                expected["born_effective_charges"].reshape(-1, 3, 3),
                rtol=0,
                atol=FP64_ABS,
            )


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c.name in ("born-hard", "soc-e2ip-spin")],
    ids=lambda case: case.name,
)
def test_native_batches(native_case, case):
    fixture = native_case(case)
    first = fixture.data
    config = _config(case, "float64")
    second = _sample(config, cell_transform=np.diag([1.01, 0.99, 1.03]))
    second["spin"] = np.roll(second["spin"], 1, axis=0)
    batch = {
        key: np.concatenate((value, second[key]), axis=0)
        for key, value in first.items()
        if key not in ("edge_index", "batch")
    }
    batch["edge_index"] = np.column_stack(
        (first["edge_index"], second["edge_index"] + len(first["pos"]))
    )
    batch["batch"] = np.concatenate((first["batch"], second["batch"] + 1))
    with NativeModel.load(fixture.manifest, backend=BACKEND) as model:
        native = model.evaluate(batch, tile_atoms=4)
        direct = fixture.oracle.evaluate(batch, tile_atoms=4)
    _assert_results(case, "batch-oracle", native, direct, FP64_ABS)


def test_native_sha256(native_case, tmp_path):
    fixture = native_case(CASES[0])
    manifest = tmp_path / fixture.manifest.name
    manifest.write_bytes(fixture.manifest.read_bytes())
    contents = bytearray(fixture.binary.read_bytes())
    contents[len(contents) // 2] ^= 1
    (tmp_path / fixture.binary.name).write_bytes(contents)
    with pytest.raises(ValueError, match=r"(?i)sha-?256|digest"):
        NativeModel.load(manifest, backend=BACKEND)


def test_native_requires_no_training_modules(native_case, tmp_path):
    fixture = native_case(CASES[-1])
    graph = tmp_path / "data.npz"
    np.savez(graph, **fixture.data)
    script = """
import importlib.abc
import sys

class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'e3nn', 'allegro', 'nequip'):
            raise AssertionError(fullname)

sys.meta_path.insert(0, Forbidden())
import numpy as np
from spingnn2.native import NativeModel

with NativeModel.load(sys.argv[1], backend=sys.argv[3]) as model:
    with np.load(sys.argv[2]) as graph:
        output = model.evaluate(dict(graph), tile_atoms=3, radial_mode="spline")
    assert np.isfinite(output["total_energy"]).all()
    assert np.isfinite(output["sf_Uepi"]).all()
"""
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            str(fixture.manifest),
            str(graph),
            BACKEND,
        ],
        env=env,
        check=True,
        timeout=150,
    )
