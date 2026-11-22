"""TorchScript versus packaged PT2 inference parity."""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import shutil
import zipfile

import numpy as np
import pytest
import torch

from helpers import (
    CASES,
    CASE_NAMES,
    absorb_aoti_lapack_poison,
    atoms_from_fixture,
    cloned,
    python_reference,
    run_case,
    run_lammps,
    write_cutoff_boundary_data,
)
from nequip.data import AtomicDataDict
from nequip.nn import RescaleOutput
from nequip.scripts import deploy
from nequip.utils.fx_compat import expand_deploy_graph
from tolerances import (
    BEC_ATOL_E,
    E_ATOL_EV,
    F_ATOL_EV_A,
    PT2_ATOL,
    PT2_CPU_ATOL,
    PT2_CPU_RTOL,
    PT2_LAMMPS_F_ATOL_EV_A,
    PT2_LAMMPS_SPIN_F_ATOL_EV,
    PT2_RTOL,
    SERIAL_RTOL,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_RESCALED_TRAIN_DIR = REPO_ROOT.parent / "validation/results/nosoc/smoke_nosoc"


class _MalformedBECOutputs(torch.nn.Module):
    def __init__(self, mutation):
        super().__init__()
        self.mutation = mutation

    def forward(self, pos, edge_index, atom_types):
        dependency = (
            pos.sum() + edge_index.sum().to(pos.dtype) + atom_types.sum().to(pos.dtype)
        ) * 0.0
        total_energy = dependency.reshape(1, 1)
        forces = pos * 0.0
        atomic_energy = pos[:, :1] * 0.0
        virial = pos.new_zeros((1, 3, 3)) + dependency
        atom_virial = pos.unsqueeze(-1).expand(-1, -1, 3) * 0.0
        width = 8 if self.mutation == "shape" else 9
        born_effective_charges = pos.new_zeros((pos.shape[0], width)) + dependency
        if self.mutation == "dtype":
            born_effective_charges = born_effective_charges.to(torch.float64)
        total_born_effective_charges = born_effective_charges.sum(dim=0, keepdim=True)
        return (
            total_energy,
            forces,
            atomic_energy,
            virial,
            atom_virial,
            born_effective_charges,
            total_born_effective_charges,
        )


def _rewrite_embedded_manifest(package, updates):
    package = pathlib.Path(package)
    replacement = package.with_suffix(".rewritten.pt2")
    with zipfile.ZipFile(package, "r") as source:
        records = [
            name for name in source.namelist() if name.endswith("/spingnn_deploy.meta")
        ]
        assert len(records) == 1
        manifest_record = records[0]
        manifest = dict(
            line.split("\t", 1)
            for line in source.read(manifest_record).decode("utf-8").splitlines()
        )
        manifest.update(updates)
        with zipfile.ZipFile(replacement, "w") as destination:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename == manifest_record:
                    data = deploy.encode_deploy_metadata(manifest).encode("utf-8")
                destination.writestr(info, data)
    replacement.replace(package)


def _file_sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _corrupt_aoti_library(package):
    package = pathlib.Path(package)
    replacement = package.with_suffix(".corrupted.pt2")
    corrupted = 0
    with zipfile.ZipFile(package, "r") as source, zipfile.ZipFile(
        replacement, "w"
    ) as destination:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.endswith(".so"):
                data = b"NOT-AN-ELF" + data[10:]
                corrupted += 1
            destination.writestr(info, data)
    assert corrupted >= 1
    replacement.replace(package)


@pytest.fixture(scope="session")
def malformed_bec_pt2_factory(
    tmp_path_factory, pt2_artifact_factory, lammps_pt2_target_device
):
    from nequip.utils.pt2_deploy import _embed_pt2_manifest

    output_dir = tmp_path_factory.mktemp("malformed_bec_pt2")
    artifacts = {}

    def build(mutation):
        if mutation in artifacts:
            return artifacts[mutation]
        _, source = pt2_artifact_factory("bec", lammps_pt2_target_device)
        metadata = deploy.read_deploy_metadata_sidecar(source)
        device = torch.device(lammps_pt2_target_device)
        module = _MalformedBECOutputs(mutation).to(device).eval()
        pos = torch.zeros((5, 3), dtype=torch.float32, device=device)
        edge_index = torch.tensor(
            ((0, 1, 2, 3), (1, 2, 3, 4)), dtype=torch.long, device=device
        )
        atom_types = torch.tensor((0, 1, 2, 0, 1), dtype=torch.long, device=device)
        sample = tuple(tensor.contiguous() for tensor in (pos, edge_index, atom_types))
        n_atoms = torch.export.Dim("n_atoms", min=1)
        n_edges = torch.export.Dim("n_edges", min=2, max=2**63 - 2)
        exported = torch.export.export(
            module,
            sample,
            dynamic_shapes=({0: n_atoms}, {1: n_edges}, {0: n_atoms}),
            strict=False,
        )
        output = output_dir / f"bec-{mutation}-{lammps_pt2_target_device}.pt2"
        package = pathlib.Path(
            torch._inductor.aoti_compile_and_package(exported, package_path=str(output))
        )
        assert package.resolve() == output.resolve()
        metadata[deploy.PACKAGE_PAYLOAD_SHA256_KEY] = _file_sha256(output)
        _embed_pt2_manifest(output, metadata)
        metadata[deploy.PACKAGE_SHA256_KEY] = _file_sha256(output)
        deploy.write_deploy_metadata_sidecar(output, metadata)
        artifacts[mutation] = output
        return output

    return build


@pytest.fixture(scope="session")
def real_rescaled_artifact_factory(tmp_path_factory):
    if not (REAL_RESCALED_TRAIN_DIR / "best_model.pth").is_file():
        pytest.skip(
            "real rescaled smoke checkpoint is unavailable: "
            f"{REAL_RESCALED_TRAIN_DIR}"
        )
    output_dir = tmp_path_factory.mktemp("real_rescaled_pt2")
    scripted_path = output_dir / "smoke_nosoc.pth"
    artifacts = {}

    def build(target_device="cpu"):
        from nequip.utils.pt2_deploy import build_pt2

        if target_device == "cuda" and not torch.cuda.is_available():
            pytest.skip(
                "CUDA-target real-checkpoint PT2 export required by LAMMPS, "
                "but torch.cuda is unavailable"
            )
        if not scripted_path.is_file():
            deploy.main(
                [
                    "build",
                    "--train-dir",
                    str(REAL_RESCALED_TRAIN_DIR),
                    str(scripted_path),
                ]
            )
        if target_device not in artifacts:
            pt2_path = output_dir / f"smoke_nosoc-{target_device}.pt2"
            build_pt2(
                REAL_RESCALED_TRAIN_DIR,
                pt2_path,
                device=target_device,
            )
            artifacts[target_device] = (scripted_path, pt2_path)
        return artifacts[target_device]

    return build


def _parity_sample(case_name):
    case = CASES[case_name]
    n_atoms = 11
    n_edges = 37
    generator = torch.Generator().manual_seed(20260824)
    pos = 0.25 * torch.randn((n_atoms, 3), generator=generator)
    spin = torch.nn.functional.normalize(
        torch.randn((n_atoms, 3), generator=generator), dim=-1
    )
    edge_ids = torch.arange(n_edges, dtype=torch.long)
    source = edge_ids.remainder(n_atoms)
    destination = (source + 1).remainder(n_atoms)
    edge_index = torch.stack((source, destination), dim=0)
    atom_types = torch.arange(n_atoms, dtype=torch.long).remainder(len(case.type_names))
    sample = tuple(
        tensor.contiguous() for tensor in (pos, spin, edge_index, atom_types)
    )
    if case_name == "bec":
        return tuple(tensor.contiguous() for tensor in (pos, edge_index, atom_types))
    if case_name != "nequip_spin":
        return sample
    edge_cell_shift = torch.zeros((n_edges, 3), dtype=pos.dtype)
    edge_cell_shift[::3, 0] = 1.0
    cell = 0.75 * torch.eye(3, dtype=pos.dtype)
    pbc = torch.ones((1, 3), dtype=torch.bool)
    return tuple(
        tensor.contiguous()
        for tensor in (
            pos,
            spin,
            edge_index,
            edge_cell_shift,
            cell,
            pbc,
            atom_types,
        )
    )


def _small_edge_sample(n_edges):
    n_atoms = 11
    generator = torch.Generator().manual_seed(20260829)
    pos = 0.1 * torch.randn((n_atoms, 3), generator=generator)
    spin = torch.nn.functional.normalize(
        torch.randn((n_atoms, 3), generator=generator), dim=-1
    )
    edge_ids = torch.arange(n_edges, dtype=torch.long)
    source = edge_ids.remainder(n_atoms)
    destination = (source + 1).remainder(n_atoms)
    return tuple(
        tensor.contiguous()
        for tensor in (
            pos,
            spin,
            torch.stack((source, destination)),
            torch.arange(n_atoms, dtype=torch.long).remainder(2),
        )
    )


def _real_rescaled_sample():
    case = CASES["nosoc"]
    atoms = atoms_from_fixture(case)
    pos = 1.05 * torch.as_tensor(atoms.positions, dtype=torch.float32)
    spin = torch.nn.functional.normalize(
        torch.as_tensor(atoms.get_initial_magnetic_moments(), dtype=torch.float32),
        dim=-1,
    )
    distances = torch.cdist(pos, pos)
    edge_index = ((distances < case.r_max) & (distances > 0.0)).nonzero().T
    atom_types = torch.tensor(
        [case.type_names.index(symbol) for symbol in atoms.symbols],
        dtype=torch.long,
    )
    sample = tuple(
        tensor.contiguous() for tensor in (pos, spin, edge_index, atom_types)
    )
    assert sample[0].shape[0] == 8
    assert sample[2].shape[1] == 36
    return sample


def _real_rescaled_scripted_input(input_signature, sample):
    data = dict(zip(input_signature, (tensor.clone() for tensor in sample)))
    data[AtomicDataDict.CELL_KEY] = 10.0 * torch.eye(3, dtype=sample[0].dtype)
    data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = torch.zeros(
        (sample[2].shape[1], 3), dtype=sample[0].dtype
    )
    data[AtomicDataDict.PBC_KEY] = torch.ones((1, 3), dtype=torch.bool)
    return data


def _assert_linf_close(label, actual, reference, *, atol=PT2_ATOL):
    max_abs_error = np.max(np.abs(actual - reference))
    reference_scale = np.max(np.abs(reference))
    assert max_abs_error <= atol + PT2_RTOL * reference_scale, (
        f"{label}: max_abs_error={max_abs_error}, " f"reference_scale={reference_scale}"
    )


@pytest.mark.compile
@pytest.mark.slow
def test_real_checkpoint_dropped_scaling_exceeds_pt2_gate():
    if not (REAL_RESCALED_TRAIN_DIR / "best_model.pth").is_file():
        pytest.skip(
            "real rescaled smoke checkpoint is unavailable: "
            f"{REAL_RESCALED_TRAIN_DIR}"
        )
    model, _ = deploy.load_model_for_deploy(
        train_dir=REAL_RESCALED_TRAIN_DIR,
        device="cpu",
        fx_compatible=True,
    )
    assert isinstance(model, RescaleOutput)
    assert model.has_scale
    assert not torch.allclose(model.scale_by, torch.ones_like(model.scale_by))

    sample = _real_rescaled_sample()
    input_signature = (
        AtomicDataDict.POSITIONS_KEY,
        AtomicDataDict.SPIN_KEY,
        AtomicDataDict.EDGE_INDEX_KEY,
        AtomicDataDict.ATOM_TYPE_KEY,
    )
    eager = model(dict(zip(input_signature, (value.clone() for value in sample))))
    dropped_graph = expand_deploy_graph(model.model, sample)
    dropped = dict(zip(dropped_graph.output_signature, dropped_graph(*sample)))
    guarded_keys = (
        AtomicDataDict.TOTAL_ENERGY_KEY,
        AtomicDataDict.FORCE_KEY,
        AtomicDataDict.SPIN_FORCE_KEY,
        AtomicDataDict.VIRIAL_KEY,
        AtomicDataDict.ATOM_VIRIAL_KEY,
    )
    for key in guarded_keys:
        max_abs_error = torch.max(torch.abs(dropped[key] - eager[key])).item()
        reference_scale = torch.max(torch.abs(eager[key])).item()
        assert max_abs_error > PT2_ATOL + PT2_RTOL * reference_scale, key


@pytest.mark.compile
@pytest.mark.slow
def test_real_rescaled_checkpoint_pt2_matches_torchscript(
    real_rescaled_artifact_factory,
):
    scripted_path, pt2_path = real_rescaled_artifact_factory("cpu")
    metadata = deploy.read_deploy_metadata_sidecar(pt2_path)
    assert metadata[deploy.RESCALE_LAYERS_KEY] == "1"
    input_signature = tuple(metadata[deploy.INPUT_SIGNATURE_KEY].split())
    output_signature = tuple(metadata[deploy.OUTPUT_SIGNATURE_KEY].split())
    sample = _real_rescaled_sample()
    scripted, _ = deploy.load_deployed_model(
        scripted_path,
        device="cpu",
        freeze=True,
        set_global_options=False,
    )
    compiled = torch._inductor.aoti_load_package(pt2_path, run_single_threaded=True)

    for _ in range(2):
        scripted(_real_rescaled_scripted_input(input_signature, sample))
        compiled(*(tensor.clone() for tensor in sample))
    absorb_aoti_lapack_poison()
    reference = scripted(_real_rescaled_scripted_input(input_signature, sample))
    actual = dict(
        zip(
            output_signature,
            compiled(*(tensor.clone() for tensor in sample)),
        )
    )
    absorb_aoti_lapack_poison()
    volume = torch.det(
        _real_rescaled_scripted_input(input_signature, sample)[AtomicDataDict.CELL_KEY]
    ).abs()
    actual[AtomicDataDict.STRESS_KEY] = -actual[AtomicDataDict.VIRIAL_KEY] / volume

    guarded_keys = (
        AtomicDataDict.TOTAL_ENERGY_KEY,
        AtomicDataDict.FORCE_KEY,
        AtomicDataDict.SPIN_FORCE_KEY,
        AtomicDataDict.STRESS_KEY,
        AtomicDataDict.VIRIAL_KEY,
        AtomicDataDict.ATOM_VIRIAL_KEY,
    )
    strict_atol, strict_rtol = (
        (PT2_CPU_ATOL, PT2_CPU_RTOL)
        if metadata[deploy.TARGET_DEVICE_KEY] == "cpu"
        else (PT2_ATOL, PT2_RTOL)
    )
    for key in guarded_keys:
        max_abs_error = torch.max(torch.abs(actual[key] - reference[key])).item()
        reference_scale = torch.max(torch.abs(reference[key])).item()
        assert max_abs_error <= strict_atol + strict_rtol * reference_scale, (
            f"real-rescaled/{key}: max_abs_error={max_abs_error}, "
            f"reference_scale={reference_scale}"
        )


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.slow
def test_real_rescaled_checkpoint_lammps_matches_torchscript_run0(
    real_rescaled_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES["nosoc"]
    scripted_path, pt2_path = real_rescaled_artifact_factory(lammps_pt2_target_device)
    scripted = run_case(
        case=case,
        model_path=scripted_path,
        binary=lammps_bin,
        workdir=tmp_path / "real-rescaled-torchscript",
        env=lammps_env,
    )
    compiled = run_case(
        case=case,
        model_path=pt2_path,
        binary=lammps_bin,
        workdir=tmp_path / "real-rescaled-pt2",
        env=lammps_env,
    )

    _assert_linf_close(
        "real-rescaled/energy",
        np.asarray(compiled.energy_ev),
        np.asarray(scripted.energy_ev),
    )
    _assert_linf_close(
        "real-rescaled/forces",
        compiled.forces_ev_a,
        scripted.forces_ev_a,
        atol=PT2_LAMMPS_F_ATOL_EV_A,
    )
    assert compiled.spin_forces_ev is not None
    assert scripted.spin_forces_ev is not None
    _assert_linf_close(
        "real-rescaled/spin_forces",
        compiled.spin_forces_ev,
        scripted.spin_forces_ev,
        atol=PT2_LAMMPS_SPIN_F_ATOL_EV,
    )
    assert compiled.atom_virial_ev is not None
    assert scripted.atom_virial_ev is not None
    _assert_linf_close(
        "real-rescaled/atom_virial",
        compiled.atom_virial_ev,
        scripted.atom_virial_ev,
        atol=PT2_LAMMPS_F_ATOL_EV_A,
    )


@pytest.mark.compile
@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_pt2_compiled_model_matches_torchscript(
    case_name,
    pt2_artifact_factory,
):
    sample = _parity_sample(case_name)
    scripted_path, pt2_path = pt2_artifact_factory(case_name)
    scripted, _ = deploy.load_deployed_model(
        scripted_path,
        device="cpu",
        freeze=True,
        set_global_options=False,
    )
    expected_input_signature = (
        (
            AtomicDataDict.POSITIONS_KEY,
            AtomicDataDict.SPIN_KEY,
            AtomicDataDict.EDGE_INDEX_KEY,
            AtomicDataDict.EDGE_CELL_SHIFT_KEY,
            AtomicDataDict.CELL_KEY,
            AtomicDataDict.PBC_KEY,
            AtomicDataDict.ATOM_TYPE_KEY,
        )
        if case_name == "nequip_spin"
        else (
            AtomicDataDict.POSITIONS_KEY,
            AtomicDataDict.EDGE_INDEX_KEY,
            AtomicDataDict.ATOM_TYPE_KEY,
        )
        if case_name == "bec"
        else (
            AtomicDataDict.POSITIONS_KEY,
            AtomicDataDict.SPIN_KEY,
            AtomicDataDict.EDGE_INDEX_KEY,
            AtomicDataDict.ATOM_TYPE_KEY,
        )
    )
    scripted_input = dict(zip(expected_input_signature, sample))
    package = pathlib.Path(pt2_path)
    metadata = deploy.read_deploy_metadata_sidecar(package)
    assert metadata[deploy.TARGET_DEVICE_KEY] == "cpu"
    assert (
        tuple(metadata[deploy.INPUT_SIGNATURE_KEY].split()) == expected_input_signature
    )
    output_signature = tuple(metadata[deploy.OUTPUT_SIGNATURE_KEY].split())
    compiled = torch._inductor.aoti_load_package(package, run_single_threaded=True)
    scripted(cloned(scripted_input))
    compiled(*(tensor.clone() for tensor in sample))
    scripted_output = scripted(cloned(scripted_input))
    compiled_output = tuple(compiled(*(tensor.clone() for tensor in sample)))
    absorb_aoti_lapack_poison()
    assert len(compiled_output) == len(output_signature)
    strict_atol, strict_rtol = (
        (PT2_CPU_ATOL, PT2_CPU_RTOL)
        if metadata[deploy.TARGET_DEVICE_KEY] == "cpu"
        else (PT2_ATOL, PT2_RTOL)
    )
    for key, pt2_value in zip(output_signature, compiled_output):
        reference = scripted_output[key]
        max_abs_error = torch.max(torch.abs(pt2_value - reference)).item()
        reference_scale = torch.max(torch.abs(reference)).item()
        print(
            "PT2_STRICT_PARITY "
            f"case={case_name} output={key} target_device="
            f"{metadata[deploy.TARGET_DEVICE_KEY]} max_abs_error={max_abs_error:.9e} "
            f"reference_scale={reference_scale:.9e} atol={strict_atol:.1e} "
            f"rtol={strict_rtol:.1e}"
        )
        assert max_abs_error <= strict_atol + strict_rtol * reference_scale, (
            f"{case_name}/{key}: max_abs_error={max_abs_error}, "
            f"reference_scale={reference_scale}"
        )


@pytest.mark.compile
@pytest.mark.lammps
def test_bec_pt2_lammps_matches_torchscript_and_python_run0(
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES["bec"]
    scripted_path, pt2_path = pt2_artifact_factory(case.name, lammps_pt2_target_device)
    reference = python_reference(scripted_path, case)
    results = {
        "torchscript": run_case(
            case=case,
            model_path=scripted_path,
            binary=lammps_bin,
            workdir=tmp_path / "torchscript",
            env=lammps_env,
        ),
        "pt2": run_case(
            case=case,
            model_path=pt2_path,
            binary=lammps_bin,
            workdir=tmp_path / "pt2",
            env=lammps_env,
        ),
    }

    for backend, result in results.items():
        assert abs(result.energy_ev - reference.energy_ev) <= E_ATOL_EV, backend
        np.testing.assert_allclose(
            result.forces_ev_a,
            reference.forces_ev_a,
            rtol=SERIAL_RTOL,
            atol=F_ATOL_EV_A,
            err_msg=backend,
        )
        assert result.bec_e is not None
        assert reference.bec_e is not None
        np.testing.assert_allclose(
            result.bec_e,
            reference.bec_e,
            rtol=SERIAL_RTOL,
            atol=BEC_ATOL_E,
            err_msg=backend,
        )


@pytest.mark.compile
@pytest.mark.lammps
def test_bec_non_extensive_energy_shift_matches_python_for_both_backends(
    shifted_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES["bec"]
    scripted_path, pt2_path = shifted_artifact_factory(
        lammps_pt2_target_device, case_name=case.name
    )
    reference = python_reference(scripted_path, case)

    for backend, artifact in (("torchscript", scripted_path), ("pt2", pt2_path)):
        result = run_case(
            case=case,
            model_path=artifact,
            binary=lammps_bin,
            workdir=tmp_path / backend,
            env=lammps_env,
        )
        assert abs(result.energy_ev - reference.energy_ev) <= E_ATOL_EV, (
            backend,
            result.energy_ev,
            reference.energy_ev,
        )


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.parametrize("mutation", ("shape", "dtype"))
def test_bec_pt2_rejects_invalid_output_shape_or_dtype(
    mutation,
    malformed_bec_pt2_factory,
    lammps_bin,
    lammps_env,
    tmp_path,
):
    case = CASES["bec"]
    package = malformed_bec_pt2_factory(mutation)
    result = run_lammps(
        binary=lammps_bin,
        input_path=case.input_path,
        workdir=tmp_path / mutation,
        env=lammps_env,
        variables={
            "data": case.data_path,
            "bec": case.bec_path,
            "model": package,
            "dump": tmp_path / f"{mutation}.dump",
        },
        check=False,
    )
    assert result.returncode != 0
    if mutation == "shape":
        assert (
            "PT2 output 'born_effective_charges' has invalid size 8 on axis 1, "
            "expected 9" in result.output
        )
    else:
        assert (
            "PT2 output 'born_effective_charges' must have dtype float32"
            in result.output
        )


@pytest.mark.compile
@pytest.mark.parametrize("n_edges", (2, 4, 24, 25))
def test_pt2_accepts_declared_small_edge_range(n_edges, pt2_artifact_factory):
    _, pt2_path = pt2_artifact_factory("nosoc", "cpu")
    compiled = torch._inductor.aoti_load_package(pt2_path, run_single_threaded=True)
    outputs = compiled(*_small_edge_sample(n_edges))
    absorb_aoti_lapack_poison()
    assert outputs
    assert all(torch.isfinite(output).all() for output in outputs)


@pytest.mark.compile
def test_pt2_package_embeds_the_sidecar_abi_manifest(pt2_artifact_factory):
    _, pt2_path = pt2_artifact_factory("nosoc", "cpu")
    sidecar = deploy.read_deploy_metadata_sidecar(pt2_path)
    with zipfile.ZipFile(pt2_path) as archive:
        records = [
            name for name in archive.namelist() if name.endswith("/spingnn_deploy.meta")
        ]
        assert len(records) == 1
        embedded = dict(
            line.split("\t", 1)
            for line in archive.read(records[0]).decode("utf-8").splitlines()
        )
    for key in (
        deploy.SCHEMA_VERSION_KEY,
        deploy.INPUT_SIGNATURE_KEY,
        deploy.OUTPUT_SIGNATURE_KEY,
        deploy.PROVIDED_OUTPUTS_KEY,
        deploy.DEFAULT_DTYPE_KEY,
        deploy.TARGET_DEVICE_KEY,
        deploy.R_MAX_KEY,
        deploy.N_SPECIES_KEY,
        deploy.TYPE_NAMES_KEY,
        deploy.TF32_KEY,
        deploy.ENERGY_UNITS_KEY,
        deploy.LENGTH_UNITS_KEY,
        deploy.HAS_SPIN_KEY,
        deploy.HAS_BEC_KEY,
        deploy.EDGE_PRUNE_KEY,
        "package_payload_sha256",
    ):
        assert embedded[key] == sidecar[key]


@pytest.mark.compile
@pytest.mark.lammps
def test_pt2_two_atom_two_edge_lammps_run0(
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = dataclasses.replace(CASES["nosoc"], data_name="two_atom.data")
    _, pt2_path = pt2_artifact_factory("nosoc", lammps_pt2_target_device)
    result = run_case(
        case=case,
        model_path=pt2_path,
        binary=lammps_bin,
        workdir=tmp_path,
        env=lammps_env,
    )
    assert np.isfinite(result.energy_ev)
    assert np.isfinite(result.forces_ev_a).all()


@pytest.mark.compile
@pytest.mark.lammps
def test_allegro_cutoff_boundary_torchscript_pt2_parity(
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES["nosoc"]
    scripted_path, pt2_path = pt2_artifact_factory(case.name, lammps_pt2_target_device)
    artifacts = {"torchscript": scripted_path, "pt2": pt2_path}
    results = {name: {} for name in artifacts}
    for boundary_distance in (3.999, 4.0, 4.001):
        boundary_case = dataclasses.replace(
            case,
            data_name=str(
                write_cutoff_boundary_data(
                    tmp_path / f"cutoff-{boundary_distance}.data",
                    boundary_distance,
                )
            ),
        )
        for name, artifact in artifacts.items():
            results[name][boundary_distance] = run_case(
                case=boundary_case,
                model_path=artifact,
                binary=lammps_bin,
                workdir=tmp_path / f"{name}-{boundary_distance}",
                env=lammps_env,
            )

    for name in artifacts:
        exact = results[name][4.0]
        above = results[name][4.001]
        assert exact.energy_ev == above.energy_ev
        np.testing.assert_array_equal(exact.forces_ev_a, above.forces_ev_a)
        np.testing.assert_array_equal(exact.spin_forces_ev, above.spin_forces_ev)

    for boundary_distance in (3.999, 4.0, 4.001):
        scripted = results["torchscript"][boundary_distance]
        compiled = results["pt2"][boundary_distance]
        assert abs(compiled.energy_ev - scripted.energy_ev) <= E_ATOL_EV
        _assert_linf_close(
            "cutoff-boundary/forces",
            compiled.forces_ev_a,
            scripted.forces_ev_a,
            atol=PT2_LAMMPS_F_ATOL_EV_A,
        )
        assert compiled.spin_forces_ev is not None
        assert scripted.spin_forces_ev is not None
        _assert_linf_close(
            "cutoff-boundary/spin-forces",
            compiled.spin_forces_ev,
            scripted.spin_forces_ev,
            atol=PT2_LAMMPS_SPIN_F_ATOL_EV,
        )


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.parametrize("case_name", ("nosoc", "soc"))
def test_pt2_lammps_matches_torchscript_run0(
    case_name,
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES[case_name]
    scripted_path, pt2_path = pt2_artifact_factory(case.name, lammps_pt2_target_device)
    scripted = run_case(
        case=case,
        model_path=scripted_path,
        binary=lammps_bin,
        workdir=tmp_path / "torchscript",
        env=lammps_env,
    )
    compiled = run_case(
        case=case,
        model_path=pt2_path,
        binary=lammps_bin,
        workdir=tmp_path / "pt2",
        env=lammps_env,
    )

    _assert_linf_close(
        "energy",
        np.asarray(compiled.energy_ev),
        np.asarray(scripted.energy_ev),
    )
    _assert_linf_close(
        "forces",
        compiled.forces_ev_a,
        scripted.forces_ev_a,
        atol=PT2_LAMMPS_F_ATOL_EV_A,
    )
    assert compiled.spin_forces_ev is not None
    assert scripted.spin_forces_ev is not None
    _assert_linf_close(
        "spin_forces",
        compiled.spin_forces_ev,
        scripted.spin_forces_ev,
        atol=PT2_LAMMPS_SPIN_F_ATOL_EV,
    )


@pytest.mark.compile
@pytest.mark.lammps
def test_pt2_nequip_spin_lammps_matches_torchscript_run0(
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES["nequip_spin"]
    scripted_path, pt2_path = pt2_artifact_factory(case.name, lammps_pt2_target_device)
    scripted = run_case(
        case=case,
        model_path=scripted_path,
        binary=lammps_bin,
        workdir=tmp_path / "torchscript",
        env=lammps_env,
    )
    compiled = run_case(
        case=case,
        model_path=pt2_path,
        binary=lammps_bin,
        workdir=tmp_path / "pt2",
        env=lammps_env,
    )

    _assert_linf_close(
        "nequip_spin/energy",
        np.asarray(compiled.energy_ev),
        np.asarray(scripted.energy_ev),
    )
    _assert_linf_close(
        "nequip_spin/forces",
        compiled.forces_ev_a,
        scripted.forces_ev_a,
        atol=PT2_LAMMPS_F_ATOL_EV_A,
    )
    assert compiled.spin_forces_ev is not None
    assert scripted.spin_forces_ev is not None
    _assert_linf_close(
        "nequip_spin/spin_forces",
        compiled.spin_forces_ev,
        scripted.spin_forces_ev,
        atol=PT2_LAMMPS_SPIN_F_ATOL_EV,
    )


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.parametrize(
    ("case_name", "pair_style"),
    (
        ("nosoc", "spin/allegro_spin"),
        ("nequip_spin", "spin/nequip_spin"),
        ("bec", "allegro"),
    ),
)
def test_pt2_target_device_mismatch_fails_before_loading(
    case_name,
    pair_style,
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES[case_name]
    _, source_package = pt2_artifact_factory(case_name, lammps_pt2_target_device)
    mismatched_package = tmp_path / f"{case_name}-mismatched.pt2"
    shutil.copy2(source_package, mismatched_package)
    metadata = deploy.read_deploy_metadata_sidecar(source_package)
    mismatched_target = "cpu" if lammps_pt2_target_device == "cuda" else "cuda"
    metadata[deploy.TARGET_DEVICE_KEY] = mismatched_target
    manifest_updates = {deploy.TARGET_DEVICE_KEY: mismatched_target}
    if mismatched_target == "cuda":
        metadata[deploy.CUDA_COMPUTE_CAPABILITY_KEY] = "9.0"
        metadata[deploy.CUDA_CODE_KIND_KEY] = "cubin"
        metadata[deploy.TORCH_CUDA_VERSION_KEY] = "12.8"
        manifest_updates.update(
            {
                deploy.CUDA_COMPUTE_CAPABILITY_KEY: "9.0",
                deploy.CUDA_CODE_KIND_KEY: "cubin",
                deploy.TORCH_CUDA_VERSION_KEY: "12.8",
            }
        )
    _rewrite_embedded_manifest(mismatched_package, manifest_updates)
    metadata[deploy.PACKAGE_SHA256_KEY] = _file_sha256(mismatched_package)
    deploy.write_deploy_metadata_sidecar(mismatched_package, metadata)

    result = run_lammps(
        binary=lammps_bin,
        input_path=case.input_path,
        workdir=tmp_path / "lammps",
        env=lammps_env,
        variables={
            "data": case.data_path,
            "model": mismatched_package,
            "dump": tmp_path / f"{case_name}.dump",
            **({"bec": case.bec_path} if case.bec_path is not None else {}),
        },
        check=False,
    )
    assert result.returncode != 0
    assert (
        f"Pair style {pair_style} PT2 package targets {mismatched_target} "
        f"but this pair runs on {lammps_pt2_target_device}; rebuild with "
        f"`nequip-deploy build-pt2 --device {lammps_pt2_target_device}`"
    ) in result.output


@pytest.mark.compile
@pytest.mark.lammps
def test_bec_pt2_missing_sidecar_fails_without_torchscript_fallback(
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES["bec"]
    _, source = pt2_artifact_factory(case.name, lammps_pt2_target_device)
    package = tmp_path / "missing-sidecar.pt2"
    shutil.copy2(source, package)
    result = run_lammps(
        binary=lammps_bin,
        input_path=case.input_path,
        workdir=tmp_path / "lammps",
        env=lammps_env,
        variables={
            "data": case.data_path,
            "bec": case.bec_path,
            "model": package,
            "dump": tmp_path / "missing-sidecar.dump",
        },
        check=False,
    )
    assert result.returncode != 0
    assert f"PT2 deploy metadata sidecar is missing: {package}.meta" in result.output


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.parametrize(
    ("case_name", "pair_style", "wrong_signature", "expected_has_spin"),
    (
        ("bec", "allegro", "pos spin edge_index atom_types", 0),
        ("nosoc", "spin/allegro_spin", "pos edge_index atom_types", 1),
    ),
)
def test_pt2_rejects_input_signature_spin_capability_mismatch(
    case_name,
    pair_style,
    wrong_signature,
    expected_has_spin,
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES[case_name]
    _, source = pt2_artifact_factory(case.name, lammps_pt2_target_device)
    package = tmp_path / f"{case_name}-wrong-input-signature.pt2"
    shutil.copy2(source, package)
    metadata = deploy.read_deploy_metadata_sidecar(source)
    metadata[deploy.INPUT_SIGNATURE_KEY] = wrong_signature
    _rewrite_embedded_manifest(package, {deploy.INPUT_SIGNATURE_KEY: wrong_signature})
    metadata[deploy.PACKAGE_SHA256_KEY] = _file_sha256(package)
    deploy.write_deploy_metadata_sidecar(package, metadata)
    result = run_lammps(
        binary=lammps_bin,
        input_path=case.input_path,
        workdir=tmp_path / "lammps",
        env=lammps_env,
        variables={
            "data": case.data_path,
            "model": package,
            "dump": tmp_path / f"{case_name}.dump",
            **({"bec": case.bec_path} if case.bec_path is not None else {}),
        },
        check=False,
    )
    assert result.returncode != 0
    assert f"Pair style {pair_style} PT2 has_spin={expected_has_spin}" in result.output
    assert "input_signature" in result.output


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.parametrize(
    ("metadata_key", "mutation"),
    ((deploy.R_MAX_KEY, "3.0"), (deploy.HAS_SPIN_KEY, "1"), (deploy.HAS_BEC_KEY, "0")),
)
def test_bec_pt2_rejects_sidecar_only_protected_metadata_mutation(
    metadata_key,
    mutation,
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES["bec"]
    _, source = pt2_artifact_factory(case.name, lammps_pt2_target_device)
    package = tmp_path / f"bec-{metadata_key}.pt2"
    shutil.copy2(source, package)
    metadata = deploy.read_deploy_metadata_sidecar(source)
    metadata[metadata_key] = mutation
    deploy.write_deploy_metadata_sidecar(package, metadata)
    result = run_lammps(
        binary=lammps_bin,
        input_path=case.input_path,
        workdir=tmp_path / "lammps",
        env=lammps_env,
        variables={
            "data": case.data_path,
            "bec": case.bec_path,
            "model": package,
            "dump": tmp_path / "protected-mutation.dump",
        },
        check=False,
    )
    assert result.returncode != 0
    assert (
        f"Pair style allegro PT2 sidecar does not match embedded ABI manifest "
        f"key '{metadata_key}'" in result.output
    )


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.parametrize(
    ("case_name", "pair_style"),
    (("nosoc", "spin/allegro_spin"), ("bec", "allegro")),
)
@pytest.mark.parametrize("mutation", ("swapped", "duplicate"))
def test_pt2_rejects_untrusted_output_signature(
    case_name,
    pair_style,
    mutation,
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES[case_name]
    _, source = pt2_artifact_factory(case.name, lammps_pt2_target_device)
    package = tmp_path / f"{case_name}-{mutation}.pt2"
    shutil.copy2(source, package)
    metadata = deploy.read_deploy_metadata_sidecar(source)
    signature = metadata[deploy.OUTPUT_SIGNATURE_KEY].split()
    second_name = "spin_forces" if "spin_forces" in signature else "atomic_energy"
    if mutation == "swapped":
        first = signature.index("forces")
        second = signature.index(second_name)
        signature[first], signature[second] = signature[second], signature[first]
    else:
        signature[signature.index(second_name)] = "forces"
    metadata[deploy.PROVIDED_OUTPUTS_KEY] = " ".join(signature)
    metadata[deploy.OUTPUT_SIGNATURE_KEY] = " ".join(signature)
    _rewrite_embedded_manifest(
        package,
        {
            deploy.PROVIDED_OUTPUTS_KEY: " ".join(signature),
            deploy.OUTPUT_SIGNATURE_KEY: " ".join(signature),
        },
    )
    metadata[deploy.PACKAGE_SHA256_KEY] = _file_sha256(package)
    deploy.write_deploy_metadata_sidecar(package, metadata)

    private_tmp = tmp_path / "private-tmp"
    private_tmp.mkdir()
    run_env = dict(lammps_env)
    run_env["TMPDIR"] = str(private_tmp)

    result = run_lammps(
        binary=lammps_bin,
        input_path=case.input_path,
        workdir=tmp_path / "lammps",
        env=run_env,
        variables={
            "data": case.data_path,
            "model": package,
            "dump": tmp_path / "rejected.dump",
            **({"bec": case.bec_path} if case.bec_path is not None else {}),
        },
        check=False,
    )
    assert result.returncode != 0
    assert not tuple(private_tmp.iterdir())
    expected = (
        "PT2 output_signature must follow the canonical ABI order"
        if mutation == "swapped"
        else "PT2 output_signature contains duplicate output name 'forces'"
    )
    assert f"Pair style {pair_style} {expected}" in result.output


_OPERATIONAL_METADATA_MUTATIONS = {
    deploy.R_MAX_KEY: "3.0",
    deploy.N_SPECIES_KEY: "3",
    deploy.TYPE_NAMES_KEY: "Ge Mn",
    deploy.TF32_KEY: "1",
    deploy.ENERGY_UNITS_KEY: "meV",
    deploy.LENGTH_UNITS_KEY: "nm",
    deploy.HAS_SPIN_KEY: "0",
    deploy.HAS_BEC_KEY: "1",
    deploy.EDGE_PRUNE_KEY: "__opposite__",
}


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.parametrize(
    ("case_name", "pair_style"),
    (("nosoc", "spin/allegro_spin"), ("nequip_spin", "spin/nequip_spin")),
)
@pytest.mark.parametrize("metadata_key", tuple(_OPERATIONAL_METADATA_MUTATIONS))
def test_pt2_rejects_untrusted_operational_sidecar_metadata(
    case_name,
    pair_style,
    metadata_key,
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES[case_name]
    _, source = pt2_artifact_factory(case_name, lammps_pt2_target_device)
    package = tmp_path / f"{case_name}-{metadata_key}.pt2"
    shutil.copy2(source, package)
    metadata = deploy.read_deploy_metadata_sidecar(source)
    mutation = _OPERATIONAL_METADATA_MUTATIONS[metadata_key]
    if metadata_key == deploy.EDGE_PRUNE_KEY:
        mutation = "dense" if metadata[metadata_key] == "nonzero" else "nonzero"
    metadata[metadata_key] = mutation
    deploy.write_deploy_metadata_sidecar(package, metadata)
    private_tmp = tmp_path / "private-tmp"
    private_tmp.mkdir()
    run_env = dict(lammps_env)
    run_env["TMPDIR"] = str(private_tmp)

    result = run_lammps(
        binary=lammps_bin,
        input_path=case.input_path,
        workdir=tmp_path / "lammps",
        env=run_env,
        variables={
            "data": case.data_path,
            "model": package,
            "dump": tmp_path / "rejected.dump",
        },
        check=False,
    )
    assert result.returncode > 0, result.output
    assert (
        f"Pair style {pair_style} PT2 sidecar does not match embedded ABI "
        f"manifest key '{metadata_key}'"
    ) in result.output
    assert not tuple(private_tmp.iterdir())


@pytest.mark.compile
@pytest.mark.lammps
def test_pt2_loader_construction_failure_is_a_lammps_error(
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES["nosoc"]
    _, source = pt2_artifact_factory(case.name, lammps_pt2_target_device)
    package = tmp_path / "corrupt-native-library.pt2"
    shutil.copy2(source, package)
    _corrupt_aoti_library(package)
    metadata = deploy.read_deploy_metadata_sidecar(source)
    metadata[deploy.PACKAGE_SHA256_KEY] = _file_sha256(package)
    deploy.write_deploy_metadata_sidecar(package, metadata)
    private_tmp = tmp_path / "private-tmp"
    private_tmp.mkdir()
    run_env = dict(lammps_env)
    run_env["TMPDIR"] = str(private_tmp)

    result = run_lammps(
        binary=lammps_bin,
        input_path=case.input_path,
        workdir=tmp_path / "lammps",
        env=run_env,
        variables={
            "data": case.data_path,
            "model": package,
            "dump": tmp_path / "corrupt.dump",
        },
        check=False,
    )
    assert result.returncode > 0, result.output
    assert "PT2 loader construction failed" in result.output
    assert not tuple(private_tmp.iterdir())


@pytest.mark.compile
@pytest.mark.lammps
def test_successful_repeated_pt2_load_releases_private_copy_immediately(
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES["nosoc"]
    _, package = pt2_artifact_factory(case.name, lammps_pt2_target_device)
    fd_report = tmp_path / "lammps-fds.txt"
    probe = tmp_path / "probe-parent-fds.sh"
    probe.write_text(
        """#!/usr/bin/env bash
set -eu
pid=$PPID
for _ in 1 2 3; do
  [[ "$(cat "/proc/$pid/comm")" == lmp ]] && break
  pid=$(awk '/^PPid:/ { print $2 }' "/proc/$pid/status")
done
for fd in "/proc/$pid/fd"/*; do
  readlink "$fd" 2>/dev/null || true
done >"$1"
"""
    )
    probe.chmod(0o755)
    input_path = tmp_path / "repeated-load.in"
    input_path.write_text(
        f"""units metal
boundary p p p
atom_style spin
atom_modify map array
newton on
read_data ${{data}}
pair_style spin/allegro_spin
pair_coeff * * ${{model}} Mn Ge
pair_coeff * * ${{model}} Mn Ge
pair_coeff * * ${{model}} Mn Ge
shell {probe} {fd_report}
run 0
"""
    )
    private_tmp = tmp_path / "private-tmp"
    private_tmp.mkdir()
    run_env = dict(lammps_env)
    run_env["TMPDIR"] = str(private_tmp)
    result = run_lammps(
        binary=lammps_bin,
        input_path=input_path,
        workdir=tmp_path / "lammps",
        env=run_env,
        variables={"data": case.data_path, "model": package},
        check=False,
    )
    assert result.returncode == 0, result.output
    assert fd_report.is_file()
    assert "spingnn-pt2-" not in fd_report.read_text()
    assert not tuple(private_tmp.iterdir())


@pytest.mark.compile
@pytest.mark.mpi
def test_rank_local_private_copy_failure_does_not_hang_mpi(
    pt2_artifact_factory,
    lammps_mpi_bin,
    mpi_prefix,
    lammps_env,
    tmp_path,
):
    case = CASES["nosoc"]
    _, package = pt2_artifact_factory(case.name, "cpu")
    good_tmp = tmp_path / "rank0-tmp"
    good_tmp.mkdir()
    wrapper = tmp_path / "rank-local-tmpdir.sh"
    wrapper.write_text(
        """#!/usr/bin/env bash
if [[ "${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}" == 1 ]]; then
  export TMPDIR=/sys
else
  export TMPDIR="$SPINGNN_GOOD_TMPDIR"
fi
exec "$SPINGNN_REAL_LAMMPS" "$@"
"""
    )
    wrapper.chmod(0o755)
    run_env = dict(lammps_env)
    run_env["SPINGNN_REAL_LAMMPS"] = str(lammps_mpi_bin)
    run_env["SPINGNN_GOOD_TMPDIR"] = str(good_tmp)

    result = run_lammps(
        binary=wrapper,
        input_path=case.input_path,
        workdir=tmp_path / "lammps",
        env=run_env,
        prefix=mpi_prefix(2),
        variables={
            "data": case.data_path,
            "model": package,
            "dump": tmp_path / "rank-local.dump",
        },
        check=False,
        timeout=20,
    )
    assert result.returncode != 0
    assert "Cannot create private PT2 package copy" in result.output
    assert not tuple(good_tmp.iterdir())


@pytest.mark.compile
@pytest.mark.lammps
@pytest.mark.parametrize("case_name", ("nosoc", "bec"))
def test_pt2_rejects_package_sidecar_sha256_mismatch(
    case_name,
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    case = CASES[case_name]
    _, source = pt2_artifact_factory(case.name, lammps_pt2_target_device)
    package = tmp_path / "tampered.pt2"
    shutil.copy2(source, package)
    shutil.copy2(
        pathlib.Path(str(source) + ".meta"),
        pathlib.Path(str(package) + ".meta"),
    )
    with package.open("ab") as stream:
        stream.write(b"tampered trailing bytes")
    result = run_lammps(
        binary=lammps_bin,
        input_path=case.input_path,
        workdir=tmp_path / "lammps",
        env=lammps_env,
        variables={
            "data": case.data_path,
            "model": package,
            "dump": tmp_path / "tampered.dump",
            **({"bec": case.bec_path} if case.bec_path is not None else {}),
        },
        check=False,
    )
    assert result.returncode != 0
    assert "PT2 package SHA-256 does not match its sidecar" in result.output


@pytest.mark.compile
@pytest.mark.lammps
def test_pt2_cuda_architecture_mismatch_fails_before_loading(
    pt2_artifact_factory,
    lammps_bin,
    lammps_env,
    lammps_pt2_target_device,
    tmp_path,
):
    if lammps_pt2_target_device != "cuda":
        pytest.skip("CUDA architecture contract requires a CUDA LAMMPS binary")
    case = CASES["nosoc"]
    _, source = pt2_artifact_factory(case.name, "cuda")
    package = tmp_path / "wrong-architecture.pt2"
    shutil.copy2(source, package)
    metadata = deploy.read_deploy_metadata_sidecar(source)
    metadata[deploy.CUDA_COMPUTE_CAPABILITY_KEY] = "99.0"
    _rewrite_embedded_manifest(package, {deploy.CUDA_COMPUTE_CAPABILITY_KEY: "99.0"})
    metadata[deploy.PACKAGE_SHA256_KEY] = _file_sha256(package)
    deploy.write_deploy_metadata_sidecar(package, metadata)
    result = run_lammps(
        binary=lammps_bin,
        input_path=case.input_path,
        workdir=tmp_path / "lammps",
        env=lammps_env,
        variables={
            "data": case.data_path,
            "model": package,
            "dump": tmp_path / "wrong-architecture.dump",
        },
        check=False,
    )
    assert result.returncode != 0
    assert "targets CUDA compute capability 99.0" in result.output


def test_pt2_pairs_load_only_the_verified_private_package_copy():
    metadata_header = (REPO_ROOT / "lammps/src/deploy_metadata.h").read_text()
    assert "class VerifiedPT2PrivateCopy" in metadata_header
    assert '"/proc/self/fd/"' in metadata_header
    assert "loader_path()" in metadata_header
    loader_header_path = REPO_ROOT / "lammps/src/pt2_loader.h"
    assert loader_header_path.is_file()
    loader_header = loader_header_path.read_text()
    assert loader_header.count("create_verified_pt2_private_copy(") == 1
    assert (
        loader_header.count(
            "std::make_unique<torch::inductor::AOTIModelPackageLoader>("
        )
        == 1
    )
    assert "private_copy.loader_path()" in loader_header
    assert "private_copy.cleanup()" in loader_header
    assert "PT2 loader construction failed" in loader_header
    for relative_path in (
        "lammps/src/pair_allegro.cpp",
        "lammps/src/SPIN/pair_allegro_spin.cpp",
        "lammps/src/SPIN/pair_nequip_spin.cpp",
    ):
        source = (REPO_ROOT / relative_path).read_text()
        assert '#include "pt2_loader.h"' in source
        assert "load_verified_pt2_model(" in source
        assert "create_verified_pt2_private_copy" not in source
        assert "private_copy.loader_path()" not in source
        header = (REPO_ROOT / relative_path.replace(".cpp", ".h")).read_text()
        assert "pt2_package_path" not in header
        assert "pt2_package_fd" not in header
