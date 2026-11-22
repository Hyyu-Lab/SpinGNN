import pytest

import warnings
import tempfile
import functools
import torch

import numpy as np

from e3nn.util.jit import script
from e3nn.o3 import Irreps

from nequip.data import (
    AtomicDataDict,
    AtomicData,
    Collater,
    _GRAPH_FIELDS,
    _NODE_FIELDS,
    _EDGE_FIELDS,
)
from nequip.data.transforms import TypeMapper
from nequip.model import model_from_config
from nequip.nn import GraphModuleMixin
from nequip.utils.test import assert_AtomicData_equivariant, add_spin


def _clone_data_dict(data):
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


def _make_physics_data(config, device, include_spin, positions):
    """Build a deterministic, fully periodic small system for physics tests."""
    from ase import Atoms

    symbols = ("H", "C", "O")[: len(positions)]
    atoms = Atoms(
        symbols=symbols,
        positions=np.asarray(positions),
        cell=np.eye(3) * 12.0,
        pbc=True,
    )
    mapper = TypeMapper(chemical_symbols=["H", "C", "O"])
    data = mapper(AtomicData.from_ase(atoms, r_max=config["r_max"])).to(device)
    data = AtomicData.to_AtomicDataDict(data)
    if include_spin:
        spin = torch.as_tensor(
            [[0.36, -0.48, 0.80], [-0.80, 0.60, 0.00], [0.48, 0.64, 0.60]][
                : len(positions)
            ],
            dtype=torch.get_default_dtype(),
            device=device,
        )
        data[AtomicDataDict.SPIN_KEY] = spin
    return data


def _is_soc_config(config):
    builders = config.get("model_builders", ())
    if any("Allegro_soc" in builder for builder in builders):
        return True
    if any("Allegro_nosoc" in builder for builder in builders):
        return False
    return config.get("soc", config.get("include_spin", False))


# see https://github.com/pytest-dev/pytest/issues/421#issuecomment-943386533
# to allow external packages to import tests through subclassing
class BaseModelTests:
    @pytest.fixture(autouse=True)
    def _restore_model_training_mode(self, request):
        if "model" not in request.fixturenames:
            yield
            return
        instance, _ = request.getfixturevalue("model")
        was_training = instance.training
        try:
            yield
        finally:
            instance.train(was_training)

    @pytest.fixture
    def add_spin(self):
        return False

    @pytest.fixture(scope="class")
    def config(self):
        """Implemented by subclasses.

        Return a tuple of config, out_field
        """
        raise NotImplementedError

    @pytest.fixture(
        scope="class",
        params=(
            [torch.device("cuda"), torch.device("cpu")]
            if torch.cuda.is_available()
            else [torch.device("cpu")]
        ),
    )
    def device(self, request):
        return request.param

    @staticmethod
    def make_model(config, device, initialize: bool = True, deploy: bool = False):
        torch.manual_seed(127)
        np.random.seed(193)
        config = config.copy()
        config.update(
            {
                "num_types": 3,
                "types_names": ["H", "C", "O"],
            }
        )
        model = model_from_config(config, initialize=initialize, deploy=deploy)
        model = model.to(device)
        return model

    @pytest.fixture(scope="class")
    def model(self, config, device):
        config, out_fields = config
        model = self.make_model(config, device=device)
        return model, out_fields

    # == common tests for all models ==
    def test_init(self, model):
        instance, _ = model
        assert isinstance(instance, GraphModuleMixin)

    def test_jit(self, model, atomic_batch, device):
        instance, out_fields = model
        data = AtomicData.to_AtomicDataDict(atomic_batch.to(device=device))
        instance = instance.to(device=device)
        model_script = script(instance)

        for out_field in out_fields:
            assert torch.allclose(
                instance(data)[out_field],
                model_script(data)[out_field],
                atol=1e-6,
            )

        # Save, load, and run in the same process. A fresh interpreter would
        # cover cold TorchScript profiling state, but CUDA + torch subprocesses
        # are unreliable in this shared test process; the dedicated Slurm
        # deployment lane owns that cold-process contract.
        with tempfile.TemporaryDirectory() as tmpdir:
            model_script.save(tmpdir + "/model.pt")
            torch.save(data, tmpdir + "/dat.pt")
            load_model = torch.jit.load(tmpdir + "/model.pt")
            load_dat = torch.load(tmpdir + "/dat.pt")

            atol = {
                # tight, but not that tight, since GPU nondet has to pass
                torch.float32: 1e-6,
                torch.float64: 1e-10,
            }[torch.get_default_dtype()]

            for out_field in out_fields:
                assert torch.allclose(
                    model_script(data)[out_field],
                    load_model(load_dat)[out_field],
                    atol=atol,
                )

    def test_forward(self, model, atomic_batch, device):
        instance, out_fields = model
        instance.to(device)
        data = atomic_batch.to(device)
        output = instance(AtomicData.to_AtomicDataDict(data))
        for out_field in out_fields:
            assert out_field in output

    def test_batch(self, model, atomic_batch, device, float_tolerance):
        """Confirm that the results for individual examples are the same regardless of whether they are batched."""
        allclose = functools.partial(torch.allclose, atol=float_tolerance)
        instance, out_fields = model
        instance.to(device)
        data = atomic_batch.to(device)
        data1 = data.get_example(0)
        data2 = data.get_example(1)
        output1 = instance(AtomicData.to_AtomicDataDict(data1))
        output2 = instance(AtomicData.to_AtomicDataDict(data2))
        output = instance(AtomicData.to_AtomicDataDict(data))
        for out_field in out_fields:
            if out_field in _GRAPH_FIELDS:
                assert allclose(
                    output1[out_field],
                    output[out_field][0],
                )
                assert allclose(
                    output2[out_field],
                    output[out_field][1],
                )
            elif out_field in _NODE_FIELDS:
                assert allclose(
                    output1[out_field],
                    output[out_field][output[AtomicDataDict.BATCH_KEY] == 0],
                )
                assert allclose(
                    output2[out_field],
                    output[out_field][output[AtomicDataDict.BATCH_KEY] == 1],
                )
            elif out_field in _EDGE_FIELDS:
                assert allclose(
                    output1[out_field],
                    output[out_field][
                        output[AtomicDataDict.BATCH_KEY][
                            output[AtomicDataDict.EDGE_INDEX_KEY][0]
                        ]
                        == 0
                    ],
                )
                assert allclose(
                    output2[out_field],
                    output[out_field][
                        output[AtomicDataDict.BATCH_KEY][
                            output[AtomicDataDict.EDGE_INDEX_KEY][0]
                        ]
                        == 1
                    ],
                )
            else:
                raise NotImplementedError

    def test_equivariance(self, model, atomic_batch, device):
        instance, out_fields = model
        instance = instance.to(device=device)
        atomic_batch = atomic_batch.to(device=device)
        assert_AtomicData_equivariant(func=instance, data_in=atomic_batch)

    def test_embedding_cutoff(self, model, config, device, add_spin):
        instance, out_fields = model
        config, out_fields = config
        r_max = config["r_max"]

        # make a synthetic three atom example
        data = AtomicData(
            atom_types=np.random.choice([0, 1, 2], size=3),
            pos=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            edge_index=np.array([[0, 1, 0, 2], [1, 0, 2, 0]]),
        )
        if add_spin:
            data[AtomicDataDict.SPIN_KEY] = torch.as_tensor(
                np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                dtype=torch.get_default_dtype(),
            )
            data.irreps[AtomicDataDict.SPIN_KEY] = Irreps("1x1eo")
        data = data.to(device)
        edge_embed = instance(AtomicData.to_AtomicDataDict(data))
        if AtomicDataDict.EDGE_FEATURES_KEY in edge_embed:
            key = AtomicDataDict.EDGE_FEATURES_KEY
        else:
            key = AtomicDataDict.EDGE_EMBEDDING_KEY
        edge_embed = edge_embed[key]
        data.pos[2, 1] = r_max  # put it past the cutoff
        edge_embed2 = instance(AtomicData.to_AtomicDataDict(data))[key]

        if key == AtomicDataDict.EDGE_EMBEDDING_KEY:
            # we can only check that other edges are unaffected if we know it's an embedding
            # For example, an Allegro edge feature is many body so will be affected
            assert torch.allclose(edge_embed[:2], edge_embed2[:2])
        if add_spin:
            feat_rij_num = int(edge_embed.size(1) / 2)
        else:
            feat_rij_num = int(edge_embed.size(1))
        assert edge_embed[2:, :feat_rij_num].abs().sum() > 1e-6  # some nonzero terms
        assert torch.allclose(
            edge_embed2[2:, :feat_rij_num], torch.zeros(1, device=device)
        )

        # test gradients
        in_dict = AtomicData.to_AtomicDataDict(data)
        in_dict[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)

        with torch.autograd.set_detect_anomaly(True):
            out = instance(in_dict)

            # is the edge embedding of the cutoff length edge unchanged at the cutoff?
            grads = torch.autograd.grad(
                outputs=out[key][2:, :feat_rij_num].sum(),
                inputs=in_dict[AtomicDataDict.POSITIONS_KEY],
                retain_graph=True,
            )[0]
            assert torch.allclose(grads, torch.zeros(1, device=device))

            if AtomicDataDict.PER_ATOM_ENERGY_KEY in out:
                # are the first two atom's energies unaffected by atom at the cutoff?
                grads = torch.autograd.grad(
                    outputs=out[AtomicDataDict.PER_ATOM_ENERGY_KEY][:2].sum(),
                    inputs=in_dict[AtomicDataDict.POSITIONS_KEY],
                )[0]
                print(grads)
                # only care about gradient wrt moved atom
                assert grads.shape == (3, 3)
                assert torch.allclose(grads[2], torch.zeros(1, device=device))


class BaseEnergyModelTests(BaseModelTests):
    def test_cutoff_smoothness(self, model, config, device, add_spin, float_tolerance):
        instance, _ = model
        config, _ = config
        instance = instance.to(device)
        instance.eval()
        r_max = float(config["r_max"])
        deltas = (
            -0.05,
            -0.025,
            -0.01,
            -0.001,
            -1.0e-5,
            0.0,
            1.0e-5,
            0.001,
            0.01,
            0.025,
            0.05,
        )

        outputs = []
        for delta in deltas:
            data = _make_physics_data(
                config,
                device,
                add_spin,
                [[2.0, 2.0, 2.0], [2.0 + r_max + delta, 2.0, 2.0]],
            )
            outputs.append(instance(_clone_data_dict(data)))

        isolated = outputs[-1]
        energy_key = AtomicDataDict.TOTAL_ENERGY_KEY
        per_atom_key = AtomicDataDict.PER_ATOM_ENERGY_KEY
        force_key = AtomicDataDict.FORCE_KEY
        spin_force_key = AtomicDataDict.SPIN_FORCE_KEY
        tolerance = float(float_tolerance)

        # At and immediately inside the boundary, both E and F approach the
        # exactly isolated result; this is the direct smooth-cutoff check.
        for index in (4, 5):
            assert torch.allclose(
                outputs[index][energy_key],
                isolated[energy_key],
                atol=tolerance,
                rtol=0.0,
            )
            if force_key in outputs[index]:
                assert torch.allclose(
                    outputs[index][force_key],
                    isolated[force_key],
                    atol=tolerance,
                    rtol=0.0,
                )

        # Every strictly outside point has an empty neighbor graph, hence the
        # interaction contribution must be bit-exactly zero.
        for index, delta in enumerate(deltas):
            if delta <= 0.0:
                continue
            assert torch.equal(outputs[index][energy_key], isolated[energy_key])
            assert torch.equal(outputs[index][per_atom_key], isolated[per_atom_key])
            if force_key in outputs[index]:
                assert torch.equal(outputs[index][force_key], isolated[force_key])
            if spin_force_key in outputs[index]:
                assert torch.equal(
                    outputs[index][spin_force_key], isolated[spin_force_key]
                )

        # The scan itself must be finite and free of an energy jump larger than
        # the work permitted by the sampled radial forces.
        energies = torch.stack([output[energy_key].reshape(()) for output in outputs])
        assert torch.isfinite(energies).all()
        if force_key in outputs[0]:
            radial_forces = torch.stack(
                [output[force_key][1, 0].abs() for output in outputs]
            )
            assert torch.isfinite(radial_forces).all()
            for index in range(len(deltas) - 1):
                width = deltas[index + 1] - deltas[index]
                force_bound = radial_forces[index : index + 2].max() * width * 2.0
                assert (energies[index + 1] - energies[index]).abs() <= (
                    force_bound + tolerance
                )

    @pytest.mark.slow
    def test_numeric_stress(self, model, config, device, add_spin, float_tolerance):
        if torch.get_default_dtype() is not torch.float64:
            pytest.skip("numeric stress uses the strict float64 lane")

        instance, _ = model
        config, _ = config
        instance = instance.to(device)
        instance.eval()
        data = _make_physics_data(
            config,
            device,
            add_spin,
            [[2.0, 2.0, 2.0], [3.1, 2.4, 2.2], [2.4, 3.2, 2.7]],
        )
        analytical_output = instance(_clone_data_dict(data))
        if AtomicDataDict.ATOM_VIRIAL_KEY not in analytical_output:
            pytest.skip("model is not a ParaStress output model")

        epsilon = 1.0e-5
        identity = torch.eye(3, dtype=torch.float64, device=device)
        voigt = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        numeric = []
        base_cell = data[AtomicDataDict.CELL_KEY].reshape(-1, 3, 3)
        volume = torch.det(base_cell[0]).abs()
        for row, column in voigt:
            basis = torch.zeros((3, 3), dtype=torch.float64, device=device)
            basis[row, column] = 1.0 if row == column else 0.5
            basis[column, row] = 1.0 if row == column else 0.5
            energies = []
            for sign in (1.0, -1.0):
                deformation = identity + sign * epsilon * basis
                strained = _clone_data_dict(data)
                strained[AtomicDataDict.POSITIONS_KEY] = data[
                    AtomicDataDict.POSITIONS_KEY
                ] @ deformation.transpose(0, 1)
                strained[AtomicDataDict.CELL_KEY] = base_cell @ deformation.transpose(
                    0, 1
                )
                energies.append(
                    instance(strained)[AtomicDataDict.TOTAL_ENERGY_KEY].sum()
                )
            numeric.append((energies[0] - energies[1]) / (2.0 * epsilon * volume))

        stress = analytical_output[AtomicDataDict.STRESS_KEY].reshape(-1, 3, 3)[0]
        analytical = torch.stack([stress[row, column] for row, column in voigt])
        numeric = torch.stack(numeric)
        assert torch.allclose(numeric, analytical, atol=2.0e-7, rtol=2.0e-5)

    def test_atom_virial_sum(self, model, config, device, add_spin, float_tolerance):
        if torch.get_default_dtype() is not torch.float64:
            pytest.skip("atom virial contract uses the strict float64 lane")

        instance, _ = model
        config, _ = config
        instance = instance.to(device)
        instance.eval()
        data = _make_physics_data(
            config,
            device,
            add_spin,
            [[2.0, 2.0, 2.0], [3.1, 2.4, 2.2], [2.4, 3.2, 2.7]],
        )
        output = instance(_clone_data_dict(data))
        if AtomicDataDict.ATOM_VIRIAL_KEY not in output:
            pytest.skip("model is not a ParaStress output model")

        atom_sum = output[AtomicDataDict.ATOM_VIRIAL_KEY].sum(dim=0)
        virial = output[AtomicDataDict.VIRIAL_KEY].reshape(-1, 3, 3)[0]
        tolerance = float(float_tolerance)
        if not torch.allclose(atom_sum, virial, atol=tolerance, rtol=0.0):
            symmetric_atom_sum = (atom_sum + atom_sum.transpose(-1, -2)) / 2.0
            assert torch.allclose(symmetric_atom_sum, virial, atol=tolerance, rtol=0.0)
            pytest.skip(
                "ParaStress atom_virial intentionally retains the asymmetric "
                "edge-centered tensor consumed by LAMMPS cvatom[9]; global "
                "virial is the symmetric strain response, so SOC models agree "
                "only after symmetrization"
            )

    def test_determinism(self, config, device, add_spin):
        config_dict, out_fields = config
        first = self.make_model(config_dict, device=device)
        second = self.make_model(config_dict, device=device)
        assert first.state_dict().keys() == second.state_dict().keys()
        for key in first.state_dict():
            assert torch.equal(first.state_dict()[key], second.state_dict()[key]), key

        data = _make_physics_data(
            config_dict,
            device,
            add_spin,
            [[2.0, 2.0, 2.0], [3.1, 2.4, 2.2], [2.4, 3.2, 2.7]],
        )

        def forward_backward(instance):
            torch.manual_seed(449)
            np.random.seed(449)
            instance.train()
            instance.zero_grad(set_to_none=True)
            output = instance(_clone_data_dict(data))
            output[AtomicDataDict.TOTAL_ENERGY_KEY].sum().backward()
            gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in instance.named_parameters()
                if parameter.grad is not None
            }
            return output, gradients

        previous_threads = torch.get_num_threads()
        try:
            if device.type == "cpu":
                torch.set_num_threads(1)
            first_output, first_gradients = forward_backward(first)
            second_output, second_gradients = forward_backward(second)
        finally:
            if device.type == "cpu":
                torch.set_num_threads(previous_threads)
        for key in out_fields:
            first_value = first_output[key]
            second_value = second_output[key]
            if first_value.is_floating_point():
                tolerance = torch.finfo(first_value.dtype).eps
                assert torch.allclose(
                    first_value, second_value, atol=tolerance, rtol=0.0
                ), f"output {key}"
            else:
                assert torch.equal(first_value, second_value), f"output {key}"
        assert first_gradients.keys() == second_gradients.keys()
        assert first_gradients
        for key in first_gradients:
            assert torch.equal(
                first_gradients[key], second_gradients[key]
            ), f"gradient {key}"

    def test_large_separation(self, model, config, molecules, device):
        atol = {torch.float32: 1e-4, torch.float64: 1e-10}[torch.get_default_dtype()]
        instance, _ = model
        instance.to(device)
        config, out_fields = config
        r_max = config["r_max"]
        atoms1 = molecules[0].copy()
        atoms2 = molecules[1].copy()
        # translate atoms2 far away
        atoms2.positions += 40.0 + np.random.randn(3)
        atoms_both = atoms1.copy()
        atoms_both.extend(atoms2)
        tm = TypeMapper(chemical_symbols=["H", "C", "O"])
        data1 = tm(AtomicData.from_ase(atoms1, r_max=r_max).to(device=device))
        data2 = tm(AtomicData.from_ase(atoms2, r_max=r_max).to(device=device))
        data_both = tm(AtomicData.from_ase(atoms_both, r_max=r_max).to(device=device))
        assert (
            data_both[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
            == data1[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
            + data2[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        )

        out1 = instance(AtomicData.to_AtomicDataDict(data1))
        out2 = instance(AtomicData.to_AtomicDataDict(data2))
        out_both = instance(AtomicData.to_AtomicDataDict(data_both))

        assert torch.allclose(
            out1[AtomicDataDict.TOTAL_ENERGY_KEY]
            + out2[AtomicDataDict.TOTAL_ENERGY_KEY],
            out_both[AtomicDataDict.TOTAL_ENERGY_KEY],
            atol=atol,
        )

        atoms_both2 = atoms1.copy()
        atoms3 = atoms2.copy()
        atoms3.positions += np.random.randn(3)
        atoms_both2.extend(atoms3)
        data_both2 = tm(AtomicData.from_ase(atoms_both2, r_max=r_max).to(device=device))
        out_both2 = instance(AtomicData.to_AtomicDataDict(data_both2))
        assert torch.allclose(
            out_both2[AtomicDataDict.TOTAL_ENERGY_KEY],
            out_both[AtomicDataDict.TOTAL_ENERGY_KEY],
            atol=atol,
        )
        assert torch.allclose(
            out_both2[AtomicDataDict.PER_ATOM_ENERGY_KEY],
            out_both[AtomicDataDict.PER_ATOM_ENERGY_KEY],
            atol=atol,
        )

    def test_cross_frame_grad(self, model, device, nequip_dataset):
        c = Collater.for_dataset(nequip_dataset)
        batch = c([nequip_dataset[i] for i in range(len(nequip_dataset))])
        energy_model, out_fields = model
        energy_model.to(device)
        data = AtomicData.to_AtomicDataDict(batch.to(device))
        data[AtomicDataDict.POSITIONS_KEY].requires_grad = True

        output = energy_model(data)
        grads = torch.autograd.grad(
            outputs=output[AtomicDataDict.TOTAL_ENERGY_KEY][-1],
            inputs=data[AtomicDataDict.POSITIONS_KEY],
            allow_unused=True,
        )[0]

        last_frame_n_atom = batch.ptr[-1] - batch.ptr[-2]

        in_frame_grad = grads[-last_frame_n_atom:]
        cross_frame_grad = grads[:-last_frame_n_atom]

        assert cross_frame_grad.abs().max().item() == 0
        assert in_frame_grad.abs().max().item() > 0

    def test_numeric_gradient(self, model, atomic_batch, device):
        model, out_fields = model
        if AtomicDataDict.FORCE_KEY not in out_fields:
            pytest.skip()
        model.to(device)
        data = atomic_batch.to(device)
        output = model(AtomicData.to_AtomicDataDict(data))

        forces = output[AtomicDataDict.FORCE_KEY]

        epsilon = torch.as_tensor(1e-3)
        epsilon2 = torch.as_tensor(2e-3)
        iatom = 1
        for idir in range(3):
            pos = data[AtomicDataDict.POSITIONS_KEY][iatom, idir]
            data[AtomicDataDict.POSITIONS_KEY][iatom, idir] = pos + epsilon
            output = model(AtomicData.to_AtomicDataDict(data.to(device)))
            e_plus = output[AtomicDataDict.TOTAL_ENERGY_KEY].sum()

            data[AtomicDataDict.POSITIONS_KEY][iatom, idir] -= epsilon2
            output = model(AtomicData.to_AtomicDataDict(data.to(device)))
            e_minus = output[AtomicDataDict.TOTAL_ENERGY_KEY].sum()

            numeric = -(e_plus - e_minus) / epsilon2
            analytical = forces[iatom, idir]
            print(numeric.item(), analytical.item())
            assert torch.isclose(numeric, analytical, atol=2e-2) or torch.isclose(
                numeric, analytical, rtol=5e-2
            )

    def test_partial_forces(self, config, atomic_batch, device, strict_locality):
        config, out_fields = config
        if "ForceOutput" not in config["model_builders"]:
            pytest.skip()
        config = config.copy()
        partial_config = config.copy()
        partial_config["model_builders"] = [
            "PartialForceOutput" if b == "ForceOutput" else b
            for b in partial_config["model_builders"]
        ]
        model = model_from_config(config=config, initialize=True)
        partial_model = model_from_config(config=partial_config, initialize=True)
        model.to(device)
        partial_model.to(device)
        partial_model.load_state_dict(model.state_dict())
        data = atomic_batch.to(device)
        output = model(AtomicData.to_AtomicDataDict(data))
        output_partial = partial_model(AtomicData.to_AtomicDataDict(data))
        # everything should be the same
        # including the
        for k in output:
            assert k != AtomicDataDict.PARTIAL_FORCE_KEY
            assert k in output_partial
            if output[k].is_floating_point():
                assert torch.allclose(
                    output[k],
                    output_partial[k],
                    atol=1e-8 if k == AtomicDataDict.TOTAL_ENERGY_KEY else 1e-6,
                )
            else:
                assert torch.equal(output[k], output_partial[k])
        n_at = data[AtomicDataDict.POSITIONS_KEY].shape[0]
        partial_forces = output_partial[AtomicDataDict.PARTIAL_FORCE_KEY]
        assert partial_forces.shape == (n_at, n_at, 3)
        # confirm that sparsity matches graph topology:
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        adjacency = torch.zeros(
            n_at, n_at, dtype=torch.bool, device=partial_forces.device
        )
        if strict_locality:
            # only adjacent for nonzero deriv to neighbors
            adjacency[edge_index[0], edge_index[1]] = True
            arange = torch.arange(n_at, device=partial_forces.device)
            adjacency[arange, arange] = True  # diagonal is ofc True
        else:
            # technically only adjacent to n-th degree neighbor, but in this tiny test system that is same as all-to-all and easier to program
            adjacency = data[AtomicDataDict.BATCH_KEY].view(-1, 1) == data[
                AtomicDataDict.BATCH_KEY
            ].view(1, -1)
        assert torch.equal(adjacency, torch.any(partial_forces != 0, dim=-1))


class BaseEnergySpinModelTests(BaseEnergyModelTests):
    @pytest.fixture(scope="class")
    def model_config(self, config, device):
        config, out_fields = config
        torch.manual_seed(0)
        np.random.seed(0)
        config = config.copy()
        config.update(
            {
                "num_types": 3,
                "types_names": ["H", "C", "O"],
            }
        )
        model = model_from_config(config)
        model = model.to(device)
        return model, out_fields, config

    @pytest.fixture
    def add_spin(self):
        return True

    def test_jit(self, model, atomic_batch_spin, device):
        super().test_jit(model, atomic_batch_spin, device)

    def test_forward(self, model, atomic_batch_spin, device):
        super().test_forward(model, atomic_batch_spin, device)

    def test_batch(self, model, atomic_batch_spin, device, float_tolerance):
        super().test_batch(model, atomic_batch_spin, device, float_tolerance)

    def test_large_separation(self, model, config, molecules_spin, device):
        super().test_large_separation(model, config, molecules_spin, device)

    def test_numeric_gradient(self, model, atomic_batch_spin, device):
        super().test_numeric_gradient(model, atomic_batch_spin, device)

    def test_partial_forces(self, config, atomic_batch_spin, device, strict_locality):
        super().test_partial_forces(config, atomic_batch_spin, device, strict_locality)

    def test_equivariance(self, model_config, atomic_batch_spin, device):
        instance, out_field, config = model_config
        parity = config["parity"] in ("o3", "u2", "o3_full", "o3_restricted")
        time_reversal = config["parity"] in ("su2", "u2")
        soc = config.get("soc", False)

        instance = instance.to(device=device)
        atomic_batch_spin = atomic_batch_spin.to(device=device)
        assert_AtomicData_equivariant(
            func=instance,
            data_in=atomic_batch_spin,
            do_parity=parity,
            do_time_reversal=time_reversal,
            do_only_rot_spin=(not soc),
        )

    def test_lattice_spin_joint_rotation_distinction(
        self, model, config, device, float_tolerance
    ):
        instance, _ = model
        config, _ = config
        if config.get("lsoc_layer") is False:
            pytest.skip("SOC builder has spin-orbit layers explicitly disabled")

        instance = instance.to(device)
        instance.eval()
        data = _make_physics_data(
            config,
            device,
            True,
            [[2.0, 2.0, 2.0], [3.1, 2.4, 2.2], [2.4, 3.2, 2.7]],
        )
        rotation = torch.as_tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.get_default_dtype(),
            device=device,
        )
        spin_only = _clone_data_dict(data)
        spin_only[AtomicDataDict.SPIN_KEY] = spin_only[
            AtomicDataDict.SPIN_KEY
        ] @ rotation.transpose(0, 1)
        reference_energy = instance(_clone_data_dict(data))[
            AtomicDataDict.TOTAL_ENERGY_KEY
        ]
        rotated_energy = instance(spin_only)[AtomicDataDict.TOTAL_ENERGY_KEY]

        difference = (reference_energy - rotated_energy).abs().max()
        if _is_soc_config(config):
            threshold = 100.0 * torch.finfo(reference_energy.dtype).eps
            assert difference > threshold
        else:
            assert torch.allclose(
                reference_energy,
                rotated_energy,
                atol=float(float_tolerance),
                rtol=0.0,
            )

    def test_cross_frame_grad(self, model, device, nequip_dataset):
        c = Collater.for_dataset(nequip_dataset)
        batch = c([nequip_dataset[i] for i in range(len(nequip_dataset))])
        energy_model, out_fields = model
        energy_model.to(device)
        data = AtomicData.to_AtomicDataDict(batch.to(device))
        data[AtomicDataDict.POSITIONS_KEY].requires_grad = True
        data = add_spin(
            [{key: value.clone() for key, value in data.items()}],
            spin_precent=1.0,
        )[0]
        data[AtomicDataDict.SPIN_KEY].requires_grad = True

        output = energy_model(data)
        grads = torch.autograd.grad(
            outputs=output[AtomicDataDict.TOTAL_ENERGY_KEY][-1],
            inputs=data[AtomicDataDict.SPIN_KEY],
            allow_unused=True,
        )[0]

        last_frame_n_atom = batch.ptr[-1] - batch.ptr[-2]

        in_frame_grad = grads[-last_frame_n_atom:]
        cross_frame_grad = grads[:-last_frame_n_atom]

        assert cross_frame_grad.abs().max().item() == 0
        assert in_frame_grad.abs().max().item() > 0

    def test_numeric_gradient_spin(self, model, atomic_batch_spin, device):
        model, out_fields = model
        if AtomicDataDict.SPIN_FORCE_KEY not in out_fields:
            pytest.skip()
        model.to(device)
        data = atomic_batch_spin.clone().to(device)
        output = model(AtomicData.to_AtomicDataDict(data))

        forces = output[AtomicDataDict.SPIN_FORCE_KEY]

        epsilon = torch.as_tensor(1e-3)
        epsilon2 = torch.as_tensor(2e-3)

        spin = data[AtomicDataDict.SPIN_KEY]
        spin_norm = torch.norm(spin, dim=-1)
        node_lspin = spin_norm > 1e-6

        # Find atom with spin
        iatom = 0
        while iatom < spin.size(0) and not bool(node_lspin[iatom]):
            iatom = iatom + 1

        if iatom == spin.size(0):
            raise ValueError("data has no spin!")

        original_spin = data[AtomicDataDict.SPIN_KEY].clone()
        for idir in range(3):
            data_plus = data.clone()
            data_plus[AtomicDataDict.SPIN_KEY][iatom, idir] += epsilon
            output = model(AtomicData.to_AtomicDataDict(data_plus))
            e_plus = output[AtomicDataDict.TOTAL_ENERGY_KEY].sum()

            data_minus = data.clone()
            data_minus[AtomicDataDict.SPIN_KEY][iatom, idir] -= epsilon
            output = model(AtomicData.to_AtomicDataDict(data_minus))
            e_minus = output[AtomicDataDict.TOTAL_ENERGY_KEY].sum()

            numeric = -(e_plus - e_minus) / epsilon2
            analytical = forces[iatom, idir]
            print(numeric.item(), analytical.item())
            assert torch.isclose(numeric, analytical, atol=2e-2) or torch.isclose(
                numeric, analytical, rtol=5e-2
            )
            assert torch.equal(data[AtomicDataDict.SPIN_KEY], original_spin)

    def test_partial_forces_spin(
        self, config, atomic_batch_spin, device, strict_locality
    ):
        config, out_fields = config
        if "SpinForceOutput" not in config["model_builders"]:
            pytest.skip()
        config = config.copy()
        partial_config = config.copy()
        partial_config["model_builders"] = [
            "PartialSpinForceOutput" if b == "SpinForceOutput" else b
            for b in partial_config["model_builders"]
        ]
        model = model_from_config(config=config, initialize=True)
        partial_model = model_from_config(config=partial_config, initialize=True)
        model.to(device)
        partial_model.to(device)
        partial_model.load_state_dict(model.state_dict())
        data = atomic_batch_spin.to(device)
        output = model(AtomicData.to_AtomicDataDict(data))
        output_partial = partial_model(AtomicData.to_AtomicDataDict(data))
        # everything should be the same
        # including the
        for k in output:
            assert k != AtomicDataDict.SPIN_PARTIAL_FORCE_KEY
            assert k in output_partial
            if output[k].is_floating_point():
                assert torch.allclose(
                    output[k],
                    output_partial[k],
                    atol=1e-8 if k == AtomicDataDict.TOTAL_ENERGY_KEY else 1e-6,
                )
            else:
                assert torch.equal(output[k], output_partial[k])
        n_at = data[AtomicDataDict.SPIN_KEY].shape[0]
        partial_forces = output_partial[AtomicDataDict.SPIN_PARTIAL_FORCE_KEY]
        assert partial_forces.shape == (n_at, n_at, 3)
        # confirm that sparsity matches graph topology:
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        adjacency = torch.zeros(n_at, n_at, dtype=torch.bool, device=edge_index.device)

        if strict_locality:
            # only adjacent for nonzero deriv to neighbors
            adjacency[edge_index[0], edge_index[1]] = True
            adjacency[
                torch.arange(n_at), torch.arange(n_at)
            ] = True  # diagonal is ofc True
        else:
            # technically only adjacent to n-th degree neighbor, but in this tiny test system that is same as all-to-all and easier to program
            adjacency = data[AtomicDataDict.BATCH_KEY].view(-1, 1) == data[
                AtomicDataDict.BATCH_KEY
            ].view(1, -1)

        spin = data[AtomicDataDict.SPIN_KEY]
        spin_norm = torch.norm(spin, dim=-1)
        node_lspin = spin_norm > 1e-6

        adjacency[:, ~node_lspin] = False  # atoms without spin always zero

        partial_forces_adjacency = torch.any(partial_forces != 0, dim=-1)
        partial_forces_adjacency[
            torch.arange(n_at)[node_lspin], torch.arange(n_at)[node_lspin]
        ] = True  # manually set diagonal is ofc True. In cases of not considering soc but normalize spin, this term could be False. Now Force it to be True to fit with adjacency
        partial_forces_adjacency[:, ~node_lspin] = False
        try:
            assert torch.equal(adjacency, partial_forces_adjacency)
        except AssertionError:
            warnings.warn(
                "Fail in torch.equal(adjacency, partial_forces_adjacency)."
                + "Try to torch.equal(adjacency, partial_forces_adjacency | adjacency)"
            )
            assert torch.equal(adjacency, partial_forces_adjacency | adjacency)
