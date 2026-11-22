"""D0 metadata assembly and PT2 sidecar contract tests."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
import torch
from nequip.scripts import deploy


TORCHSCRIPT_BASELINE_KEYS = {
    "_jit_bailout_depth",
    "_jit_fusion_strategy",
    "allow_tf32",
    "config",
    "default_dtype",
    "e3nn_version",
    "energy_units",
    "edge_prune",
    "has_bec",
    "has_spin",
    "length_units",
    "n_species",
    "nequip_version",
    "provided_outputs",
    "r_max",
    "required_inputs",
    "schema_version",
    "torch_version",
    "type_names",
}


def _write_mock_pt2_sidecar(output, device, rescale_layers="0"):
    output = Path(output)
    output.write_bytes(b"mock PT2 package")
    metadata = {
        deploy.INPUT_SIGNATURE_KEY: "pos spin edge_index atom_types",
        deploy.OUTPUT_SIGNATURE_KEY: "total_energy forces spin_forces",
        deploy.TARGET_DEVICE_KEY: device,
        deploy.RESCALE_LAYERS_KEY: rescale_layers,
        deploy.PACKAGE_SHA256_KEY: hashlib.sha256(output.read_bytes()).hexdigest(),
        deploy.PACKAGE_PAYLOAD_SHA256_KEY: hashlib.sha256(
            output.read_bytes()
        ).hexdigest(),
        deploy.EDGE_PRUNE_KEY: "dense",
    }
    if device == "cuda":
        metadata[deploy.CUDA_COMPUTE_CAPABILITY_KEY] = "9.0"
        metadata[deploy.CUDA_CODE_KIND_KEY] = "cubin"
        metadata[deploy.TORCH_CUDA_VERSION_KEY] = "12.8"
    deploy.write_deploy_metadata_sidecar(output, metadata)


def test_torchscript_build_uses_shared_metadata_without_key_drift(
    monkeypatch, tmp_path
):
    calls = []
    original = deploy.assemble_deploy_metadata

    def recording_assembler(config, model, provided_outputs, required_inputs):
        metadata = original(config, model, provided_outputs, required_inputs)
        calls.append(metadata)
        return metadata

    monkeypatch.setattr(deploy, "assemble_deploy_metadata", recording_assembler)
    output = tmp_path / "nosoc.pth"
    deploy.main(
        [
            "build",
            "--model",
            str(Path(__file__).parent / "fixtures/configs/nosoc.yaml"),
            str(output),
        ]
    )
    assert len(calls) == 1
    # Git provenance is optional outside a Git checkout; code versions are
    # required so both deployment readers can recognize the CLI artifact.
    assert set(calls[0]) - {deploy.CODE_COMMITS_KEY} == TORCHSCRIPT_BASELINE_KEYS
    for key in (
        deploy.NEQUIP_VERSION_KEY,
        deploy.E3NN_VERSION_KEY,
        deploy.TORCH_VERSION_KEY,
    ):
        assert calls[0][key]
    with zipfile.ZipFile(output) as archive:
        extras = {
            name.split("extra/", 1)[1]: archive.read(name).decode("ascii")
            for name in archive.namelist()
            if "extra/" in name
        }
    assert extras == calls[0]
    assert extras[deploy.EDGE_PRUNE_KEY] == "dense"

    loaded, metadata = deploy.load_deployed_model(
        output, freeze=False, set_global_options=False
    )
    mismatch = tmp_path / "nosoc-edge-prune-mismatch.pth"
    metadata[deploy.EDGE_PRUNE_KEY] = "nonzero"
    torch.jit.save(
        loaded,
        mismatch,
        _extra_files={key: value.encode("ascii") for key, value in metadata.items()},
    )
    with pytest.raises(ValueError, match="does not match embedded config"):
        deploy.load_deployed_model(mismatch, freeze=False, set_global_options=False)


@pytest.mark.parametrize(
    ("allowed_species", "expected_type_names"),
    (([25, 32], "Mn Ge"), ([32, 25], "Ge Mn")),
)
def test_deploy_metadata_preserves_legacy_species_type_order(
    allowed_species, expected_type_names
):
    config = {
        "allowed_species": allowed_species,
        "r_max": 4.0,
        "_jit_bailout_depth": 2,
        "allow_tf32": False,
        "default_dtype": "float32",
    }
    metadata = deploy.assemble_deploy_metadata(
        config,
        object(),
        ("total_energy", "atomic_energy", "forces"),
        ("pos", "edge_index", "atom_types"),
    )
    assert metadata[deploy.N_SPECIES_KEY] == "2"
    assert metadata[deploy.TYPE_NAMES_KEY] == expected_type_names


@pytest.mark.parametrize(
    ("provided_outputs", "required_inputs", "expected_has_spin"),
    (
        (
            ("total_energy", "forces", "atomic_energy", "born_effective_charges"),
            ("pos", "spin", "edge_index", "atom_types"),
            0,
        ),
        (
            ("total_energy", "forces", "spin_forces", "atomic_energy"),
            ("pos", "edge_index", "atom_types"),
            1,
        ),
    ),
)
def test_deploy_metadata_rejects_spin_signature_capability_mismatch(
    monkeypatch, provided_outputs, required_inputs, expected_has_spin
):
    monkeypatch.setattr(deploy, "get_config_code_versions", lambda config: ({}, {}))
    config = {
        "r_max": 4.0,
        "num_types": 2,
        "type_names": ("Mn", "Ge"),
        "_jit_bailout_depth": 2,
        "allow_tf32": False,
        "default_dtype": "float32",
    }

    with pytest.raises(ValueError, match=f"has_spin={expected_has_spin}"):
        deploy.assemble_deploy_metadata(
            config, object(), provided_outputs, required_inputs
        )


def test_pt2_sidecar_round_trip_escapes_multiline_values(tmp_path):
    model_path = tmp_path / "model.pt2"
    model_path.write_bytes(b"round-trip package")
    metadata = {
        "config": "line one\nline two\\nwith literal slash-n\tand tab",
        deploy.INPUT_SIGNATURE_KEY: "pos spin edge_index atom_types",
        deploy.OUTPUT_SIGNATURE_KEY: "total_energy forces spin_forces",
        deploy.TARGET_DEVICE_KEY: "cpu",
        deploy.PACKAGE_SHA256_KEY: hashlib.sha256(model_path.read_bytes()).hexdigest(),
        deploy.PACKAGE_PAYLOAD_SHA256_KEY: hashlib.sha256(
            model_path.read_bytes()
        ).hexdigest(),
        deploy.EDGE_PRUNE_KEY: "dense",
    }
    sidecar_path = deploy.write_deploy_metadata_sidecar(model_path, metadata)
    assert sidecar_path == tmp_path / "model.pt2.meta"
    assert deploy.read_deploy_metadata_sidecar(model_path) == metadata
    assert len(sidecar_path.read_text().splitlines()) == len(metadata)


@pytest.mark.parametrize(
    "missing_key",
    (
        deploy.INPUT_SIGNATURE_KEY,
        deploy.OUTPUT_SIGNATURE_KEY,
        deploy.TARGET_DEVICE_KEY,
        deploy.PACKAGE_SHA256_KEY,
        deploy.PACKAGE_PAYLOAD_SHA256_KEY,
        deploy.EDGE_PRUNE_KEY,
    ),
)
def test_pt2_sidecar_requires_deployment_contract(tmp_path, missing_key):
    metadata = {
        deploy.INPUT_SIGNATURE_KEY: "pos edge_index atom_types",
        deploy.OUTPUT_SIGNATURE_KEY: "total_energy forces",
        deploy.TARGET_DEVICE_KEY: "cpu",
        deploy.PACKAGE_SHA256_KEY: "0" * 64,
        deploy.PACKAGE_PAYLOAD_SHA256_KEY: "0" * 64,
        deploy.EDGE_PRUNE_KEY: "dense",
    }
    del metadata[missing_key]
    with pytest.raises(ValueError, match=missing_key):
        deploy.write_deploy_metadata_sidecar(tmp_path / "model.pt2", metadata)


@pytest.mark.parametrize("package_hash", ("", "abc", "G" * 64, "A" * 64))
def test_pt2_sidecar_rejects_malformed_package_hash(tmp_path, package_hash):
    metadata = {
        deploy.INPUT_SIGNATURE_KEY: "pos edge_index atom_types",
        deploy.OUTPUT_SIGNATURE_KEY: "total_energy forces",
        deploy.TARGET_DEVICE_KEY: "cpu",
        deploy.PACKAGE_SHA256_KEY: package_hash,
        deploy.PACKAGE_PAYLOAD_SHA256_KEY: "0" * 64,
        deploy.EDGE_PRUNE_KEY: "dense",
    }
    with pytest.raises(ValueError, match="package_sha256"):
        deploy.write_deploy_metadata_sidecar(tmp_path / "model.pt2", metadata)


@pytest.mark.parametrize(
    "missing_key",
    (
        deploy.CUDA_COMPUTE_CAPABILITY_KEY,
        deploy.CUDA_CODE_KIND_KEY,
        deploy.TORCH_CUDA_VERSION_KEY,
    ),
)
def test_cuda_pt2_sidecar_requires_architecture_contract(tmp_path, missing_key):
    metadata = {
        deploy.INPUT_SIGNATURE_KEY: "pos edge_index atom_types",
        deploy.OUTPUT_SIGNATURE_KEY: "total_energy forces",
        deploy.TARGET_DEVICE_KEY: "cuda",
        deploy.PACKAGE_SHA256_KEY: "0" * 64,
        deploy.PACKAGE_PAYLOAD_SHA256_KEY: "0" * 64,
        deploy.EDGE_PRUNE_KEY: "dense",
        deploy.CUDA_COMPUTE_CAPABILITY_KEY: "9.0",
        deploy.CUDA_CODE_KIND_KEY: "cubin",
        deploy.TORCH_CUDA_VERSION_KEY: "12.8",
    }
    del metadata[missing_key]
    with pytest.raises(ValueError, match=missing_key):
        deploy.write_deploy_metadata_sidecar(tmp_path / "model.pt2", metadata)


@pytest.mark.parametrize("edge_prune", ("", "all", "Dense"))
def test_pt2_sidecar_rejects_invalid_edge_prune(tmp_path, edge_prune):
    metadata = {
        deploy.INPUT_SIGNATURE_KEY: "pos edge_index atom_types",
        deploy.OUTPUT_SIGNATURE_KEY: "total_energy forces",
        deploy.TARGET_DEVICE_KEY: "cpu",
        deploy.PACKAGE_SHA256_KEY: "0" * 64,
        deploy.PACKAGE_PAYLOAD_SHA256_KEY: "0" * 64,
        deploy.EDGE_PRUNE_KEY: edge_prune,
    }
    with pytest.raises(ValueError, match="edge_prune"):
        deploy.write_deploy_metadata_sidecar(tmp_path / "model.pt2", metadata)


def test_build_pt2_cli_forwards_target_device(monkeypatch, tmp_path, capsys):
    from nequip.utils import pt2_deploy

    calls = []

    def recording_builder(source, output, device):
        calls.append((source, output, device))
        _write_mock_pt2_sidecar(output, device)
        return output

    monkeypatch.setattr(pt2_deploy, "build_pt2", recording_builder)
    config = tmp_path / "model.yaml"
    output = tmp_path / "model.pt2"
    deploy.main(
        [
            "build-pt2",
            "--model",
            str(config),
            "--device",
            "cuda",
            str(output),
        ]
    )
    assert calls == [(config, output, "cuda")]
    stdout = capsys.readouterr().out
    assert f"PT2 package: {output.resolve()}" in stdout
    assert f"PT2 sidecar: {output.resolve()}.meta" in stdout
    assert "PT2 target_device: cuda" in stdout
    assert f"pair_coeff * * {output.resolve()} <ELEMENTS...>" in stdout
    assert "PT2 learned scaling: not present (0 layer(s))" in stdout


@pytest.mark.parametrize(
    ("cuda_available", "expected_device"),
    ((False, "cpu"), (True, "cuda")),
)
def test_build_pt2_cli_auto_selects_current_machine(
    monkeypatch, tmp_path, capsys, cuda_available, expected_device
):
    from nequip.utils import pt2_deploy

    calls = []

    def recording_builder(source, output, device):
        calls.append((source, output, device))
        _write_mock_pt2_sidecar(output, device)
        return output

    monkeypatch.setattr(pt2_deploy, "build_pt2", recording_builder)
    monkeypatch.setattr(deploy.torch.cuda, "is_available", lambda: cuda_available)
    config = tmp_path / "model.yaml"
    output = tmp_path / "model.pt2"
    deploy.main(["build-pt2", "--model", str(config), str(output)])

    assert calls == [(config, output, expected_device)]
    stdout = capsys.readouterr().out
    assert f"PT2 device auto-selected: {expected_device}" in stdout
    assert f"PT2 target_device: {expected_device}" in stdout
    if cuda_available:
        assert "CUDA LAMMPS requires" not in stdout
    else:
        assert "CUDA LAMMPS requires a CUDA-target package" in stdout


def test_build_pt2_cli_reports_included_learned_scaling(monkeypatch, tmp_path, capsys):
    from nequip.utils import pt2_deploy

    def rescaled_builder(source, output, device):
        del source
        _write_mock_pt2_sidecar(output, device, rescale_layers="2")
        return output

    monkeypatch.setattr(pt2_deploy, "build_pt2", rescaled_builder)
    output = tmp_path / "model.pt2"
    deploy.main(
        [
            "build-pt2",
            "--model",
            str(tmp_path / "model.yaml"),
            "--device",
            "cpu",
            str(output),
        ]
    )
    assert "PT2 learned scaling: included (2 layer(s))" in capsys.readouterr().out


def test_build_pt2_cli_requires_builder_sidecar(monkeypatch, tmp_path):
    from nequip.utils import pt2_deploy

    monkeypatch.setattr(
        pt2_deploy,
        "build_pt2",
        lambda source, output, device: output,
    )
    with pytest.raises(FileNotFoundError, match="model.pt2.meta"):
        deploy.main(
            [
                "build-pt2",
                "--model",
                str(tmp_path / "model.yaml"),
                str(tmp_path / "model.pt2"),
            ]
        )


def test_build_pt2_cli_preserves_unsupported_wrapper_error(monkeypatch, tmp_path):
    from nequip.utils import pt2_deploy

    def unsupported_builder(source, output, device):
        del source, output, device
        raise TypeError("PT2 export does not support output wrapper custom.Unknown")

    monkeypatch.setattr(pt2_deploy, "build_pt2", unsupported_builder)
    with pytest.raises(TypeError, match="custom.Unknown"):
        deploy.main(
            [
                "build-pt2",
                "--model",
                str(tmp_path / "model.yaml"),
                str(tmp_path / "model.pt2"),
            ]
        )
