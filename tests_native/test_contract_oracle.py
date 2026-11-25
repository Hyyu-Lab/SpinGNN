"""Exit gates for the version 1 SpinGNN2 contract and reference evaluator."""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml
from ase import Atoms
from ase.neighborlist import neighbor_list

from nequip.model import model_from_config
from nequip.utils import Config
from nequip.utils._global_options import global_options_context

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "nosoc": "02_spin_nosoc.yaml",
    "soc": "03_spin_soc.yaml",
    "born": "04_born_effective_charge.yaml",
    "e2ip": "06_allegro_e2ip.yaml",
    "soc-e2ip": "07_allegro_soc_e2ip.yaml",
}


@dataclass(frozen=True)
class ModelCase:
    name: str
    family: str
    overrides: dict
    asr: str = "soft"

    @property
    def magnetic(self):
        return self.family in {"nosoc", "soc", "soc-e2ip"}

    @property
    def evidence(self):
        if "e2ip" in self.family:
            return (
                "sf"
                if self.overrides.get("evidential_key") == "spin_forces_vert"
                else "f"
            )
        return None


CASES = (
    ModelCase("nosoc-te3nn", "nosoc", {"lbiquadratic": True}),
    ModelCase(
        "nosoc-plain",
        "nosoc",
        {
            "lte3nn": False,
            "normalize_basis": False,
            "ledge_cutoff": True,
            "per_edge_species_scale": True,
        },
    ),
    ModelCase("soc-u2", "soc", {}),
    ModelCase("soc-spin-harmonics", "soc", {"tenn_soc": False}),
    ModelCase("born-soft", "born", {}),
    ModelCase("born-hard", "born", {"edge_prune": "dense"}, asr="hard"),
    ModelCase("born-none", "born", {}, asr="none"),
    ModelCase("e2ip-force", "e2ip", {}),
    ModelCase("soc-e2ip-force", "soc-e2ip", {"evidential_key": "forces"}),
    ModelCase("soc-e2ip-spin", "soc-e2ip", {"evidential_key": "spin_forces_vert"}),
)
MAGNETIC_CASES = tuple(case for case in CASES if case.magnetic)


def _config(case, dtype):
    config = yaml.safe_load((REPO / "example" / EXAMPLES[case.family]).read_text())
    # Preserve the actual builder and cutoff options while keeping eager codegen
    # and central differences small enough for routine CPU validation.
    config.update(
        {
            name: [8]
            for name, value in config.items()
            if name.endswith("_mlp_latent_dimensions") and value
        }
    )
    config.update(
        default_dtype=dtype,
        allow_tf32=False,
        float32_matmul_precision="highest",
        spingnn_compile={"enabled": False},
        train_autocast={"enabled": False},
        avg_num_neighbors=4.25,
        env_embed_multiplicity=3,
        allegro_onlyr_num_layers=2,
        allegro_soc_on_num_layers=2,
        allegro_nosoc_num_layers=2,
        e2ip_latent_channels=2,
        edge_biquad_coeff_mlp_latent_dimensions=[8],
        per_species_rescale_scales=[0.85, 1.12, 1.04][
            : len(config["chemical_symbol_to_type"])
        ],
        per_species_rescale_shifts=[0.07, -0.03, 0.04][
            : len(config["chemical_symbol_to_type"])
        ],
        global_rescale_scale=1.125,
    )
    config.update(case.overrides)
    if case.family == "born":
        config["born_scale_asr"] = case.asr
    return config


def _model(case, config):
    torch.manual_seed(1793)
    model = model_from_config(Config.from_dict(config), initialize=True).eval()
    if config.get("per_edge_species_scale"):
        edge_sum = model.model.func.edge_eng_sum
        scales = edge_sum.per_edge_scales
        with torch.no_grad():
            scales.copy_(torch.linspace(0.75, 1.22, scales.numel()).reshape_as(scales))
    if case.family == "born":
        reducer = next(
            m for m in model.modules() if type(m).__name__ == "CartesianBECReduce"
        )
        assert reducer.asr == case.asr
    return model


def _sample(config, cell_transform=None):
    cell = np.array([[10.0, 0.0, 0.0], [0.8, 17.0, 0.0], [0.2, -0.6, 17.0]])
    if cell_transform is not None:
        cell = cell @ cell_transform
    fractional = np.array(
        [
            [0.04, 0.07, 0.10],
            [0.91, 0.05, 0.12],
            [0.19, 0.17, 0.12],
            [0.29, 0.05, 0.22],
            [0.54, 0.08, 0.13],
            [0.63, 0.62, 0.62],
        ]
    )
    pos = fractional @ cell
    centers, neighbors, shifts = neighbor_list(
        "ijS", Atoms("H6", positions=pos, cell=cell, pbc=True), config["r_max"]
    )
    assert np.any(shifts)
    assert len(pos) - 1 not in centers
    order = np.random.default_rng(183).permutation(len(centers))
    atom_types = np.array([0, 1, 0, 1, 2, 0]) % len(config["chemical_symbol_to_type"])
    dtype = np.dtype(config["default_dtype"])
    return {
        "pos": pos.astype(dtype),
        "cell": cell[None, :, :].astype(dtype),
        "edge_index": np.stack((centers[order], neighbors[order])),
        "edge_cell_shift": shifts[order].astype(dtype),
        "atom_types": atom_types,
        "pbc": np.ones((1, 3), dtype=bool),
        "batch": np.zeros(len(pos), dtype=np.int64),
        "spin": np.array(
            [
                [0.81, 0.32, -0.15],
                [-0.36, 0.68, 0.46],
                [0.00, 0.00, 0.00],
                [0.26, -0.75, 0.37],
                [0.12, -0.34, -0.82],
                [-0.49, 0.25, 0.76],
            ],
            dtype=dtype,
        ),
    }


def _eager(model, data):
    output = model({key: torch.from_numpy(value.copy()) for key, value in data.items()})
    return {key: value.detach().numpy() for key, value in output.items()}


def _observables(case):
    fields = [
        "total_energy",
        "atomic_energy",
        "forces",
        "virial",
        "stress",
        "atom_virial",
    ]
    if case.magnetic:
        fields.extend(("spin_forces", "spin_forces_vert"))
    if case.family == "born":
        fields.extend(("born_effective_charges", "total_born_effective_charges"))
    if case.evidence:
        prefix = case.evidence
        fields.extend(
            [
                prefix + "_" + field
                for field in ("nu", "kappa", "sigma", "Uepi", "uncertainty")
            ]
        )
        fields.append("total_" + prefix + "_Uepi")
    return fields


@dataclass
class OracleCase:
    model: object
    data: dict
    eager: dict
    json_path: Path
    bin_path: Path


@pytest.fixture(scope="module")
def make_oracle(tmp_path_factory):
    from spingnn2.export import export_model
    from spingnn2.reference import ReferenceModel

    built = {}

    def fixture(case, dtype="float64"):
        tag = (case.name, dtype)
        if tag not in built:
            config = _config(case, dtype)
            data = _sample(config)
            stem = tmp_path_factory.mktemp(case.name) / dtype
            with global_options_context(config):
                model = _model(case, config)
                eager = _eager(model, data)
                json_path, bin_path = export_model(model, stem)
            built[tag] = OracleCase(
                ReferenceModel.load(json_path), data, eager, json_path, bin_path
            )
        return built[tag]

    return fixture


def _report(check, case, errors):
    print(
        "M1_RESULT "
        + json.dumps({"check": check, "case": case.name, **errors}, sort_keys=True)
    )


def _check_errors(check, case, observed, expected, limit):
    errors = {}
    scaled_errors = {}
    dtype_mismatches = {}
    for field in _observables(case):
        actual = observed[field]
        target = expected[field]
        assert actual.shape == target.shape, (field, actual.shape, target.shape)
        if actual.dtype != target.dtype:
            dtype_mismatches[field] = (actual.dtype, target.dtype)
        diff = np.abs(actual - target)
        errors[field] = float(np.max(diff))
        # Relative+absolute parity: prevents mathematically impossible sub-ULP float32
        # failures on large-magnitude sums (e.g. total_f_Uepi ~ 400 where 1 ULP > 1e-5).
        scaled_errors[field] = float(np.max(diff / (1.0 + np.abs(target))))
    _report(check, case, errors)
    assert not dtype_mismatches, dtype_mismatches
    assert all(error < limit for error in scaled_errors.values()), (
        limit,
        errors,
        scaled_errors,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype,limit", [("float64", 1e-11), ("float32", 1e-5)])
def test_analytic_parity(make_oracle, case, dtype, limit):
    fixture = make_oracle(case, dtype)
    observed = fixture.model.evaluate(
        fixture.data, tile_atoms=3, radial_mode="analytic"
    )
    _check_errors("analytic-" + dtype, case, observed, fixture.eager, limit)
    if case.magnetic:
        zero_spin = np.all(fixture.data["spin"] == 0, axis=1)
        assert np.all(observed["spin_forces"][zero_spin] == 0.0)
        assert np.all(observed["spin_forces_vert"][zero_spin] == 0.0)
        if case.evidence == "sf":
            assert np.all(observed["sf_sigma"][zero_spin] == 0.0)
    if case.family == "born" and case.asr == "hard":
        assert np.max(np.abs(observed["total_born_effective_charges"])) < limit


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_spline_parity(make_oracle, case):
    from spingnn2.splines import evaluate_basis

    fixture = make_oracle(case)
    observed = fixture.model.evaluate(fixture.data, tile_atoms=3, radial_mode="spline")
    _check_errors("spline-float64", case, observed, fixture.eager, 1e-6)
    calibrations = {}
    for layer in fixture.model.contract["layers"]:
        spec = layer.get("basis")
        if spec is None or "spline" not in spec:
            continue
        calibration = spec["spline"]
        x = np.random.default_rng(2781).uniform(
            calibration["lower"], calibration["upper"], size=8192
        )
        exact = evaluate_basis(spec, x)
        spline = evaluate_basis(spec, x, radial_mode="spline")
        calibrations[layer["name"]] = {
            "p": spec["cutoff"]["p"],
            "kind": spec["kind"],
            "sample_value_error": float(np.max(np.abs(exact[0] - spline[0]))),
            "sample_derivative_error": float(np.max(np.abs(exact[1] - spline[1]))),
            **{
                field: calibration[field]
                for field in ("max_value_error", "max_derivative_error")
            },
        }
    _report("spline-calibration", case, calibrations)
    assert calibrations
    assert all(
        result[field] < 1e-7
        for result in calibrations.values()
        for field in (
            "max_value_error",
            "max_derivative_error",
            "sample_value_error",
            "sample_derivative_error",
        )
    )
    if case.family in {"nosoc", "soc"}:
        assert any(result["p"] == 48 for result in calibrations.values())


def _finite_differences(fixture, field, active):
    data = fixture.data
    step = 2.0e-5
    gradient = np.zeros_like(data[field])
    for atom in np.flatnonzero(active):
        for component in range(3):
            plus = data[field].copy()
            minus = data[field].copy()
            plus[atom, component] += step
            minus[atom, component] -= step
            a = fixture.model.evaluate({**data, field: plus})["total_energy"].item()
            b = fixture.model.evaluate({**data, field: minus})["total_energy"].item()
            gradient[atom, component] = (a - b) / (2.0 * step)
    return gradient


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_finite_difference(make_oracle, case):
    fixture = make_oracle(case)
    result = fixture.model.evaluate(fixture.data, tile_atoms=3)
    active_pos = np.ones(len(fixture.data["pos"]), dtype=bool)
    pos_grad = _finite_differences(fixture, "pos", active_pos)
    errors = {"pos": float(np.max(np.abs(pos_grad + result["forces"])))}
    if case.magnetic:
        spin_norm_squared = np.sum(fixture.data["spin"] ** 2, axis=-1, keepdims=True)
        # The practical finite-difference step crosses the normalization floor
        # at zero spin, where eager parity instead checks the physical force.
        # The reported force includes the spin norm squared.
        spin_grad = _finite_differences(fixture, "spin", spin_norm_squared[:, 0] > 0)
        errors["spin"] = float(
            np.max(np.abs(spin_grad * spin_norm_squared + result["spin_forces"]))
        )
    _report("finite-difference", case, errors)
    assert all(error < 1e-8 for error in errors.values()), errors


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("mode", ["analytic", "spline"])
def test_tiling_invariance(make_oracle, case, mode):
    fixture = make_oracle(case)
    full = fixture.model.evaluate(
        fixture.data, tile_atoms=len(fixture.data["pos"]), radial_mode=mode
    )
    for tile_atoms in (1, 3):
        tiled = fixture.model.evaluate(
            fixture.data, tile_atoms=tile_atoms, radial_mode=mode
        )
        _check_errors(f"tiling-{mode}-{tile_atoms}", case, tiled, full, 1e-13)


@pytest.mark.parametrize("case", MAGNETIC_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("mode", ["analytic", "spline"])
def test_stage1_cache(make_oracle, case, mode):
    fixture = make_oracle(case)
    model, data = fixture.model, fixture.data
    cache = model.evaluate_stage1_lattice(data, tile_atoms=3, radial_mode=mode)
    for shift in (1, 2):
        spin = np.roll(data["spin"], shift, axis=0)
        updated = {**data, "spin": spin}
        full = model.evaluate(updated, tile_atoms=3, radial_mode=mode)
        cached = model.evaluate_stage2_spin_cached(cache, spin)
        _check_errors(f"cache-{mode}-{shift}", case, cached, full, 1e-13)
        assert (
            abs(full["total_energy"].item() - fixture.eager["total_energy"].item())
            > 1e-8
        )


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case.name in {"born-hard", "soc-e2ip-spin"}],
    ids=lambda case: case.name,
)
def test_batched_frame_reductions(tmp_path, case):
    from spingnn2.export import export_model
    from spingnn2.reference import ReferenceModel

    config = _config(case, "float64")
    first = _sample(config)
    second = _sample(config, cell_transform=np.diag([1.01, 0.99, 1.03]))
    second["spin"] = np.roll(second["spin"], 1, axis=0)
    second["atom_types"] = np.roll(second["atom_types"], 1)
    n_atoms = len(first["pos"])
    batch = {
        field: np.concatenate((value, second[field]), axis=0)
        for field, value in first.items()
        if field not in {"edge_index", "batch"}
    }
    batch["edge_index"] = np.concatenate(
        (first["edge_index"], second["edge_index"] + n_atoms), axis=1
    )
    batch["batch"] = np.concatenate((first["batch"], second["batch"] + 1))
    batch["num_batch"] = np.array(2, dtype=np.int64)

    with global_options_context(config):
        model = _model(case, config)
        eager_frames = [_eager(model, graph) for graph in (first, second)]
        eager_batch = _eager(model, batch)
        manifest, _ = export_model(model, tmp_path / "batch")
    expected = {
        field: np.concatenate([frame[field] for frame in eager_frames], axis=0)
        for field in _observables(case)
    }
    assert np.max(np.abs(expected["virial"][0] - expected["virial"][1])) > 1e-8
    _check_errors("batched-eager", case, eager_batch, expected, 1e-13)

    oracle = ReferenceModel.load(manifest)
    # Six atoms per frame put centers from both frames in the middle tile.
    actual = oracle.evaluate(batch, tile_atoms=4)
    _check_errors("batched-oracle", case, actual, expected, 1e-11)
    if case.family == "born":
        for frame in range(2):
            total = actual["born_effective_charges"][batch["batch"] == frame].sum(
                axis=0
            )
            assert np.max(np.abs(total)) < 1e-13
    else:
        assert abs(np.diff(expected["sf_uncertainty"], axis=0).item()) > 1e-8


@pytest.mark.parametrize("normalize", [False, True], ids=["solid", "unit"])
@pytest.mark.parametrize("normalization", ["component", "integral", "norm"])
def test_spherical_harmonics(normalize, normalization):
    from nequip.nn.embedding._edge import _SafeSphericalHarmonics
    from spingnn2.harmonics import spherical_harmonics

    vectors = np.array([[0.31, 0.72, -0.23], [1.12, -0.16, -0.83], [0.0, 0.0, 0.0]])
    degrees = (2, 0, 1, 3)
    irreps = "2e + 0e + 1o + 3o"
    eager = _SafeSphericalHarmonics(irreps, normalize, normalization)
    x = torch.tensor(vectors, requires_grad=True)
    expected = eager(x).detach().numpy()
    jacobian = torch.autograd.functional.jacobian(eager, x).numpy()
    expected_jacobian = np.stack([jacobian[i, :, i, :] for i in range(len(x))])
    value, derivative = spherical_harmonics(
        vectors, degrees=degrees, normalize=normalize, normalization=normalization
    )
    np.testing.assert_allclose(value, expected, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(derivative, expected_jacobian, atol=1e-13, rtol=0.0)
    if normalize:
        np.testing.assert_array_equal(derivative[-1], 0.0)


def test_evidence_softplus_threshold():
    from spingnn2._evidence import _softplus

    values = np.array([-1000.0, -20.0, 0.0, 19.9, 20.0, 20.1, 1000.0])
    expected = torch.nn.functional.softplus(torch.from_numpy(values)).numpy()
    observed = _softplus(values)
    np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=0.0)
    np.testing.assert_array_equal(observed[values > 20.0], values[values > 20.0])


def test_numpy_runtime_imports(make_oracle, tmp_path):
    fixture = make_oracle(CASES[-1])
    graph_path = tmp_path / "graph.npz"
    np.savez(graph_path, **fixture.data)
    script = """
import importlib.abc
from pathlib import Path
import sys

class ForbidTrainingPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'e3nn', 'te3nn', 'nequip', 'allegro'}:
            raise AssertionError(f'Runtime imported training package {fullname}')

source, contract, graph = sys.argv[1:]
sys.path.insert(0, str(Path(source) / 'src'))
sys.meta_path.insert(0, ForbidTrainingPackages())
import numpy as np
from spingnn2.reference import ReferenceModel

model = ReferenceModel.load(contract)
with np.load(graph) as arrays:
    for mode in ('analytic', 'spline'):
        out = model.evaluate(dict(arrays), tile_atoms=3, radial_mode=mode)
        for name in ('total_energy', 'forces', 'spin_forces', 'virial', 'sf_sigma'):
            assert np.isfinite(out[name]).all(), name
assert not {'torch', 'e3nn', 'te3nn', 'nequip', 'allegro'} & sys.modules.keys()
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(REPO),
            str(fixture.json_path),
            str(graph_path),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_corrupted_binary_digest(make_oracle, tmp_path):
    from spingnn2.reference import ReferenceModel

    fixture = make_oracle(CASES[0])
    payload = bytearray(fixture.bin_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    (tmp_path / fixture.bin_path.name).write_bytes(payload)
    contract = tmp_path / fixture.json_path.name
    contract.write_bytes(fixture.json_path.read_bytes())
    with pytest.raises(ValueError, match=r"(?i)sha-?256|digest"):
        ReferenceModel.load(contract)


@pytest.mark.parametrize("input_kind", ["train-directory", "checkpoint"])
def test_export_checkpoint_cli(input_kind, tmp_path):
    from spingnn2.cli import main
    from spingnn2.reference import ReferenceModel

    case = CASES[0]
    config = _config(case, "float64")
    graph = _sample(config)
    with global_options_context(config):
        model = _model(case, config)
        eager = _eager(model, graph)
    checkpoint = tmp_path / "best_model.pth"
    config_path = tmp_path / "config.yaml"
    torch.save(model.state_dict(), checkpoint)
    config_path.write_text(yaml.safe_dump(config))
    source = tmp_path if input_kind == "train-directory" else checkpoint
    stem = tmp_path / "exported"
    argv = [str(source), "--output", str(stem)]
    if input_kind == "checkpoint":
        argv.extend(["--config", str(config_path)])
    main(argv)
    oracle = ReferenceModel.load(stem.with_suffix(".sp2.json"))
    assert stem.with_suffix(".sp2.bin").is_file()
    _check_errors("cli-" + input_kind, case, oracle.evaluate(graph), eager, 1e-11)


def test_unsupported_nonlocal_model(tmp_path):
    from spingnn2.export import export_model

    config = dict(
        model_builders=[
            "SimpleIrrepsConfig",
            "EnergyModel",
            "ParaStressForceSpinOutput",
        ],
        avg_num_neighbors=4.25,
        r_max=5.0,
        num_types=2,
        type_names=["Fe", "O"],
        num_layers=2,
        num_features=4,
        l_max=1,
        include_spin=True,
        soc=False,
    )
    with global_options_context({"default_dtype": "float64"}):
        unsupported = model_from_config(config).eval()
        with pytest.raises(ValueError, match=r"(?i)unsupported|only (local |allegro)"):
            export_model(unsupported, tmp_path / "unsupported")
    assert not list(tmp_path.iterdir())


def test_unsupported_allegro_layer(tmp_path):
    from spingnn2.export import export_model

    case = CASES[0]
    config = _config(case, "float64")
    with global_options_context(config):
        model = _model(case, config)
        model.model.func.add_module("unsupported_layer", torch.nn.Identity())
        with pytest.raises(ValueError, match=r"(?i)unsupported.*Identity"):
            export_model(model, tmp_path / "unsupported")
    assert not list(tmp_path.iterdir())
