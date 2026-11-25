"""Mathematical and archive/CLI contracts for e2SLP evaluation."""
import importlib.util
import json
from pathlib import Path
from typing import Dict

import torch

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "e2slp_evaluation", ROOT / "tools/e2slp/evaluate.py"
)
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)


def record(cov, spin=None, **kwargs):
    result = dict(frame=0, source="synthetic", timestep="50", covariance=cov)
    if spin is not None:
        result["spin"] = spin
    result.update(kwargs)
    return result


def test_projection_is_rotation_invariant_and_masks_nonmagnetic_atoms():
    rng = np.random.default_rng(4)
    spin = rng.normal(size=(8, 3))
    spin[-2:] = 0
    raw = rng.normal(size=(8, 3, 3))
    cov = raw @ raw.swapaxes(-1, -2)
    trace, mask = score.atom_traces(cov, spin)
    unit = spin[mask] / np.linalg.norm(spin[mask], axis=-1, keepdims=True)
    projector = np.eye(3) - unit[:, :, None] * unit[:, None, :]
    expected = np.trace(projector @ cov[mask] @ projector, axis1=-2, axis2=-1)
    np.testing.assert_allclose(trace[mask], expected)
    np.testing.assert_array_equal(trace[~mask], 0)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated, _ = score.atom_traces(rotation @ cov @ rotation.T, spin @ rotation.T)
    np.testing.assert_allclose(rotated, trace, atol=1e-12)


def test_known_trace_and_rmse_normalization():
    cov = np.tile(np.diag([1.0, 2.0, 100.0]), (4, 1, 1))
    spin = np.array([[0, 0, 1], [0, 0, -3], [0, 0, 0], [0, 0, 0]])
    pred = np.zeros((4, 3))
    pred[:2] = 2
    r = record(cov, spin, spin_forces=pred, spin_force_label=np.zeros_like(pred))
    default = score.score_frame(r)
    assert default["uncertainty"] == 1.5
    assert default["spin_force_rmse"] == pytest.approx(np.sqrt(2))
    magnetic = score.score_frame(r, atom_scope="magnetic")
    assert magnetic["uncertainty"] == 3
    assert magnetic["spin_force_rmse"] == 2
    assert score.score_frame(r, reduction="max")["uncertainty"] == 3
    assert score.score_frame(r, target="forces")["uncertainty"] == 103


@pytest.mark.parametrize(
    "kind", ["nan", "asymmetric", "negative", "wrong_shape", "missing_spin"]
)
def test_invalid_covariances_are_rejected(kind):
    cov = np.eye(3)[None].copy()
    spin = np.array([[0.0, 0.0, 1.0]])
    if kind == "nan":
        cov[0, 0, 0] = np.nan
    elif kind == "asymmetric":
        cov[0, 0, 1] = 1
    elif kind == "negative":
        cov[0, 0, 0] = -1
    elif kind == "wrong_shape":
        cov = cov[:, :2]
    else:
        spin = None
    with pytest.raises(ValueError):
        score.atom_traces(cov, spin)


def test_zero_spin_and_undefined_correlation():
    r = record(np.eye(3)[None], np.zeros((1, 3)))
    assert score.score_frame(r)["uncertainty"] == 0
    with pytest.raises(ValueError, match="at least one"):
        score.score_frame(r, atom_scope="magnetic")
    result = score.correlations([dict(uncertainty=1, force_rmse=2)] * 4)
    assert result["force_rmse"]["spearman_rho"] is None
    json.dumps(result, allow_nan=False)


def test_variable_size_archive_roundtrip_and_missing_labels(tmp_path):
    records = []
    for i in range(3):
        n = i + 1
        r = record(np.tile(np.eye(3) * n, (n, 1, 1)), np.tile([0, 0, 1], (n, 1)))
        r.update(frame=i, forces=np.ones((n, 3)))
        if i != 1:
            r["force_label"] = np.zeros((n, 3))
        records.append(r)
    path = tmp_path / "predictions.npz"
    score.save_predictions(path, records, "spin-forces")
    loaded = list(score.archive_frames(path, "spin-forces"))
    assert [len(r["covariance"]) for r in loaded] == [1, 2, 3]
    assert "force_label" not in loaded[1]
    np.testing.assert_allclose(
        [score.score_frame(r)["uncertainty"] for r in loaded], [2, 4, 6]
    )
    with np.load(path, allow_pickle=False) as archive:
        assert archive["frame_ptr"].tolist() == [0, 1, 3, 6]


def test_legacy_cli_reproduces_ordering_and_preserves_outputs(tmp_path):
    path = tmp_path / "legacy.npz"
    cov = np.array([np.eye(3)[None] * x for x in (1, 2, 3, 4)])
    prediction = np.array([np.ones((1, 3)) * x for x in (1, 2, 3, 4)])
    np.savez(
        path,
        Uepi=cov,
        spin=np.tile([0, 0, 1], (4, 1, 1)),
        force_hat=prediction,
        force_label=np.zeros_like(prediction),
    )
    output = tmp_path / "result"
    assert (
        score.main(["--predictions", str(path), "--output", str(output), "--no-plot"])
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    assert report["correlations"]["force_rmse"]["spearman_rho"] == 1
    assert report["frames"] == 4
    with pytest.raises(SystemExit):
        score.main(["--predictions", str(path), "--output", str(output)])
    with pytest.raises(ValueError, match="missing required"):
        score.main(
            [
                "--predictions",
                str(path),
                "--output",
                str(tmp_path / "missing"),
                "--labels",
                "required",
                "--no-plot",
            ]
        )


def test_force_cli_without_spin_or_labels(tmp_path):
    path = tmp_path / "force.npz"
    np.savez(path, f_Uepi=np.tile(np.eye(3), (3, 2, 1, 1)))
    output = tmp_path / "scores"
    score.main(
        [
            "--predictions",
            str(path),
            "--target",
            "forces",
            "--labels",
            "none",
            "--output",
            str(output),
            "--no-plot",
        ]
    )
    report = json.loads((output / "report.json").read_text())
    assert report["correlations"] == {}
    assert report["input_covariance_keys"] == ["f_Uepi"]
    assert report["projection"] == "3D Cartesian"


def test_malformed_archive_offsets_are_rejected(tmp_path):
    path = tmp_path / "broken.npz"
    np.savez(
        path, sf_Uepi=np.tile(np.eye(3), (3, 1, 1)), frame_ptr=np.array([0, 3, 2, 3])
    )
    with pytest.raises(ValueError, match="frame_ptr"):
        list(score.archive_frames(path, "spin-forces"))


def test_archive_metadata_is_extracted_once(tmp_path, monkeypatch):
    from collections import Counter

    path = tmp_path / "metadata.npz"
    np.savez_compressed(
        path,
        f_Uepi=np.tile(np.eye(3), (12, 2, 1, 1)),
        forces=np.ones((12, 2, 3)),
        source=np.array(["source"] * 12),
        timestep=np.arange(12),
        forces_present=np.ones(12, dtype=bool),
    )
    counts = Counter()
    getitem = np.lib.npyio.NpzFile.__getitem__

    def counted(archive, key):
        counts[key] += 1
        return getitem(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", counted)
    assert len(list(score.archive_frames(path, "forces"))) == 12
    assert counts == dict(f_Uepi=1, forces=1, source=1, timestep=1, forces_present=1)


def test_streaming_archive_preserves_concatenate_dtype_and_values(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(score.PredictionArchive, "BLOCK_BYTES", 32)
    records = []
    for i, dtype in enumerate((np.float32, np.float64, np.float32)):
        n = i + 2
        item = record(np.arange(n * 9, dtype=dtype).reshape(n, 3, 3))
        item.update(source="source" + "x" * i, timestep=str(i))
        item["forces"] = np.arange(n * 3, dtype=np.int16).reshape(n, 3)
        if i != 0:
            item["spin"] = np.ones((n, 3), dtype=np.float32)
        if i != 1:
            item["force_label"] = np.ones((n, 3), dtype=np.float32)
        records.append(item)
    path = tmp_path / "roundtrip.npz"
    score.save_predictions(path, iter(records), "forces")
    expected = {"f_Uepi": np.concatenate([r["covariance"] for r in records])}
    for name in ("forces", "spin", "force_label"):
        expected[name] = np.concatenate(
            [r.get(name, np.zeros((len(r["covariance"]), 3))) for r in records]
        )
        expected[name + "_present"] = np.asarray([name in r for r in records])
    expected.update(
        frame_ptr=np.array([0, 2, 5, 9]),
        source=np.asarray([r["source"] for r in records]),
        timestep=np.asarray([r["timestep"] for r in records]),
    )
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(expected)
        for name, value in expected.items():
            assert loaded[name].dtype == value.dtype
            np.testing.assert_array_equal(loaded[name], value)
    assert list(tmp_path.iterdir()) == [path]


def test_streaming_archive_releases_arrays_before_consuming_next_frame(tmp_path):
    import weakref

    references = []

    def frames():
        for i in range(8):
            assert all(ref() is None for ref in references)
            covariance = np.tile(np.eye(3), (10, 1, 1))
            references.append(weakref.ref(covariance))
            yield record(covariance)
            del covariance

    score.save_predictions(tmp_path / "stream.npz", frames(), "forces")
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize("failure", ["source", "append", "write"])
def test_streaming_archive_failure_cleans_owned_files(tmp_path, monkeypatch, failure):
    path = tmp_path / "predictions.npz"
    path.write_bytes(b"existing archive")

    def frames():
        yield record(np.eye(3)[None])
        if failure == "source":
            raise RuntimeError("source failed")
        yield record(np.zeros((2, 2, 2)) if failure == "append" else np.eye(3)[None])

    if failure == "write":

        def fail(*args, **kwargs):
            raise RuntimeError("write failed")

        monkeypatch.setattr(np.lib.format, "write_array_header_2_0", fail)
    with pytest.raises((RuntimeError, ValueError)):
        score.save_predictions(path, frames(), "forces")
    assert path.read_bytes() == b"existing archive"
    assert list(tmp_path.iterdir()) == [path]


def test_direct_main_streams_and_limit_does_not_prefetch(tmp_path, monkeypatch):
    import weakref

    references, consumed = [], []

    def frames(args):
        try:
            for i in range(4):
                assert all(ref() is None for ref in references)
                consumed.append(i)
                covariance = np.eye(3)[None]
                references.append(weakref.ref(covariance))
                yield record(covariance, input_covariance_key="f_Uepi")
                del covariance
        finally:
            consumed.append("closed")

    monkeypatch.setattr(score, "deployed_frames", frames)
    model, dataset = tmp_path / "model", tmp_path / "dataset"
    model.touch()
    dataset.touch()
    output = tmp_path / "result"
    score.main(
        [
            "--model",
            str(model),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--target",
            "forces",
            "--limit",
            "2",
            "--no-plot",
        ]
    )
    assert consumed == [0, 1, "closed"]
    assert all(ref() is None for ref in references)
    assert len(list(score.archive_frames(output / "predictions.npz", "forces"))) == 2


def test_offline_spin_threshold_is_unchanged():
    norms = np.array([0, 1e-8, 1e-6, 1, 1.01e-6])
    spin = np.column_stack([np.zeros_like(norms), np.zeros_like(norms), norms])
    traces, magnetic = score.atom_traces(np.tile(np.eye(3), (5, 1, 1)), spin)
    np.testing.assert_array_equal(magnetic, [False, False, False, True, True])
    np.testing.assert_array_equal(traces, [0, 0, 0, 2, 2])


# A real serialized deployment exercises the actual loader and AtomicData path;
# an analytic model keeps precision and label-leak failures easy to diagnose.
class _DeployedGeometryProbe(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.register_buffer("weight", torch.eye(3, dtype=dtype))

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert data["edge_index"].dtype == torch.int64
        assert data["atom_types"].dtype == torch.int64
        assert "forces" not in data
        assert "spin_forces_vert" not in data
        assert "total_energy" not in data
        positions = data["pos"] @ self.weight
        # Preserve explicit physical magnitudes independently of unit directions.
        assert torch.all(data["spin_length"] == 2.0)
        covariance = self.weight.unsqueeze(0).expand(positions.shape[0], 3, 3)
        return {
            "sf_Uepi": covariance,
            "forces": positions,
            "spin_forces_vert": data["spin"],
        }


def test_real_deployment_dtype_spin_and_cli(tmp_path, float_tolerance):
    import subprocess
    import sys
    from types import SimpleNamespace
    from ase import Atoms
    from ase.io import write
    from ase.calculators.singlepoint import SinglePointCalculator
    from nequip.scripts.deploy import (
        DEFAULT_DTYPE_KEY,
        NEQUIP_VERSION_KEY,
        R_MAX_KEY,
        TYPE_NAMES_KEY,
        TF32_KEY,
    )
    from nequip.utils._global_options import global_options_context

    ambient = torch.get_default_dtype()
    dtype = torch.float64 if ambient == torch.float32 else torch.float32
    dtype_name = str(dtype).split(".")[-1]
    path = tmp_path / "probe.pth"
    model = torch.jit.script(_DeployedGeometryProbe(dtype).eval())
    torch.jit.save(
        model,
        str(path),
        _extra_files={
            DEFAULT_DTYPE_KEY: dtype_name,
            NEQUIP_VERSION_KEY: "0.0.test",
            R_MAX_KEY: "3.0",
            TYPE_NAMES_KEY: "H",
            TF32_KEY: "0",
        },
    )
    positions = np.array(
        [
            [0.123456789123, 0.0, 0.0],
            [0.876543219876, 0, 0],
            [1.234567891234, 0, 0],
            [1.876543219876, 0, 0],
        ]
    )
    atoms = Atoms("H4", positions=positions, magmoms=[0, 1e-8, 1e-6, 1])
    atoms.set_array("spin_length", np.full((4, 1), 2.0))
    atoms.set_array("spin_forces_vert", np.ones((4, 3)))
    atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=np.ones((4, 3)))
    dataset = tmp_path / "input.extxyz"
    write(
        dataset,
        [atoms, atoms],
        format="extxyz",
        columns=[
            "symbols",
            "positions",
            "initial_magmoms",
            "spin_length",
            "spin_forces_vert",
        ],
    )
    # extxyz uses fixed decimal output; read the exact stored geometry as the oracle.
    from ase.io import read

    stored = read(dataset).positions
    args = SimpleNamespace(
        model=path,
        device="cpu",
        species_map=None,
        covariance_key=None,
        target="spin-forces",
        dataset=dataset,
        limit=None,
        labels="auto",
    )
    with global_options_context({"default_dtype": str(ambient).split(".")[-1]}):
        frames = score.deployed_frames(args)
        first = next(frames)
        assert torch.get_default_dtype() == ambient
        frames.close()
        assert torch.get_default_dtype() == ambient
    assert first["covariance"].dtype == np.dtype(dtype_name)
    np.testing.assert_array_equal(first["forces"], stored.astype(dtype_name))
    np.testing.assert_array_equal(
        first["spin"], [[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    )
    assert score.score_frame(first)["n_magnetic"] == 3
    assert score.score_frame(first)["uncertainty"] == 1.5
    output = tmp_path / "cli"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/ci/source_python.py"),
            str(ROOT / "tools/e2slp/evaluate.py"),
            "--model",
            str(path),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--no-plot",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    loaded = list(score.archive_frames(output / "predictions.npz", "spin-forces"))
    assert len(loaded) == 2
    np.testing.assert_array_equal(loaded[0]["forces"], first["forces"])
    np.testing.assert_array_equal(loaded[0]["spin"], first["spin"])
