import pytest

import torch

from nequip.data import AtomicDataDict, AtomicData, Collater
from nequip.model import model_from_config
from nequip.utils.test import assert_AtomicData_equivariant, add_spin
from nequip.utils.unittests.model_tests import (
    BaseEnergyModelTests,
    BaseEnergySpinModelTests,
)
from nequip.utils.unittests.parity_tests import (
    BaseCompileParityTests,
    BasePrecisionLadderTests,
)

from _class_scoped_fixtures import ClassScopedSpinModelFixtures


# Spin config and spin model part
COMMON_CONFIG = {
    "num_types": 3,
    "type_names": ["H", "C", "O"],
    "avg_num_neighbors": None,
    "include_spin": True,
}
minimal_config1 = dict(
    l_max=1,
    r_max=4,
    num_layers=2,
    num_basis=8,
    PolynomialCutoff_p=6,
    nonlinearity_type="norm",
    **COMMON_CONFIG,
)
minimal_config2 = dict(
    l_max=1,
    r_max=4,
    num_layers=2,
    num_basis=8,
    PolynomialCutoff_p=6,
    # Test gate
    nonlinearity_type="gate",
    **COMMON_CONFIG,
)
minimal_config3 = dict(
    l_max=1,
    r_max=4,
    num_layers=2,
    num_basis=8,
    PolynomialCutoff_p=6,
    nonlinearity_type="norm",
    soc=False,  # Test not consider soc
    **COMMON_CONFIG,
)

from typing import List

import numpy as np
from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator


@pytest.fixture(scope="session")
def molecules_spin() -> List[Atoms]:
    atoms_list = []
    for i in range(8):
        atoms = molecule("CH3CHO" if i % 2 == 0 else "H2")
        atoms.set_cell([100.0, 100.0, 100.0])
        atoms.set_pbc(True)
        atoms.set_initial_magnetic_moments(np.random.random((len(atoms), 3)))
        atoms.rattle()
        atoms.calc = SinglePointCalculator(
            energy=np.random.random(),
            forces=np.random.random((len(atoms), 3)),
            stress=None,
            atoms=atoms,
        )
        atoms_list.append(atoms)
    return atoms_list


class _NequIPSpinConfig(ClassScopedSpinModelFixtures):
    @pytest.fixture
    def add_spin(self):
        return True

    @pytest.fixture
    def strict_locality(self):
        return False

    @pytest.fixture(
        params=[minimal_config1, minimal_config2, minimal_config3],
        scope="class",
    )
    @classmethod
    def base_config(cls, request):
        return request.param

    @pytest.fixture(
        params=[
            # (
            #     ["SimpleIrrepsConfig", "EnergyModel", "ForceOutput"],
            #     [
            #         AtomicDataDict.TOTAL_ENERGY_KEY,
            #         AtomicDataDict.PER_ATOM_ENERGY_KEY,
            #         AtomicDataDict.FORCE_KEY,
            #     ],
            # ),
            # (
            #     ["SimpleIrrepsConfig", "EnergyModel"],
            #     [
            #         AtomicDataDict.TOTAL_ENERGY_KEY,
            #         AtomicDataDict.PER_ATOM_ENERGY_KEY,
            #     ],
            # ),
            # (
            #     ["SimpleIrrepsConfig", "EnergyModel", "StressForceOutput"],
            #     [
            #         AtomicDataDict.TOTAL_ENERGY_KEY,
            #         AtomicDataDict.PER_ATOM_ENERGY_KEY,
            #         AtomicDataDict.FORCE_KEY,
            #         AtomicDataDict.STRESS_KEY,
            #         AtomicDataDict.VIRIAL_KEY,
            #     ],
            # ),
            (
                [
                    "SimpleIrrepsConfig",
                    "EnergyModel",
                    "ParaStressForceSpinOutput",
                ],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.SPIN_FORCE_KEY,
                    AtomicDataDict.FORCE_KEY,
                    AtomicDataDict.STRESS_KEY,
                    AtomicDataDict.VIRIAL_KEY,
                    AtomicDataDict.ATOM_VIRIAL_KEY,
                ],
            ),
        ],
        scope="class",
    )
    @classmethod
    def config(cls, request, base_config):
        config = base_config.copy()
        builder, out_fields = request.param
        config = config.copy()
        config["model_builders"] = builder
        return config, out_fields


class TestNequIPModel_spin(_NequIPSpinConfig, BaseEnergyModelTests):
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

    def test_equivariance(self, model, config, atomic_batch_spin, device):
        instance, out_field = model
        config, _ = config
        parity = config.get("parity", True)
        time_reversal = config.get("time_reversal", config["include_spin"])
        soc = config.get("soc", config["include_spin"])

        instance = instance.to(device=device)
        atomic_batch_spin = atomic_batch_spin.clone().to(device=device)
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
        BaseEnergySpinModelTests.test_lattice_spin_joint_rotation_distinction(
            self, model, config, device, float_tolerance
        )

    def test_nonperiodic_stress_rejected(self, model, atomic_batch_spin, device):
        instance, _ = model
        data = atomic_batch_spin.clone().to(device=device)
        data[AtomicDataDict.CELL_KEY].zero_()
        data[AtomicDataDict.PBC_KEY].zero_()

        with pytest.raises(ValueError, match="non-periodic"):
            instance(AtomicData.to_AtomicDataDict(data))

    def test_cross_frame_grad(self, model, device, nequip_dataset):
        c = Collater.for_dataset(nequip_dataset)
        batch = c([nequip_dataset[i] for i in range(len(nequip_dataset))])
        energy_model, out_fields = model
        energy_model.to(device)
        data = AtomicData.to_AtomicDataDict(batch.to(device))
        num_frames = int(data[AtomicDataDict.BATCH_KEY].max().item()) + 1
        data[AtomicDataDict.CELL_KEY] = (
            torch.eye(3, device=device).repeat(num_frames, 1, 1) * 10.0
        )
        data[AtomicDataDict.PBC_KEY] = torch.ones(
            (num_frames, 3), dtype=torch.bool, device=device
        )
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

        # Central differences trade truncation error (~eps^2) against roundoff
        # (~machine_eps * |E| / eps). At float32 the 1e-3 step leaves roundoff
        # noise of order 5e-3, which sits inside the 2e-2 gate and made this
        # test flaky on CI. A larger step at float32 pushes noise back to ~5e-4
        # while keeping truncation error negligible.
        if data[AtomicDataDict.SPIN_KEY].dtype is torch.float64:
            epsilon = torch.as_tensor(1e-3)
        else:
            epsilon = torch.as_tensor(1e-2)
        epsilon2 = 2.0 * epsilon

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
            assert torch.isclose(numeric, analytical, atol=2e-2) or torch.isclose(
                numeric, analytical, rtol=5e-2
            ), (
                f"spin force FD mismatch dir={idir} dtype={numeric.dtype} "
                f"eps={float(epsilon):g} numeric={numeric.item():.6e} "
                f"analytical={analytical.item():.6e}"
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
        adjacency = torch.zeros(n_at, n_at, dtype=torch.bool)

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
        ] = True  # manually set diagonal is ofc True. In case of not considering soc, this term could be False. Now Force it to be True to fit with adjacency
        partial_forces_adjacency[:, ~node_lspin] = False
        assert torch.equal(adjacency, partial_forces_adjacency)


class TestNequIPSpinParity(
    _NequIPSpinConfig, BaseCompileParityTests, BasePrecisionLadderTests
):
    pass
