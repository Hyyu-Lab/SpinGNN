import pytest
import copy

import numpy as np
import torch
from nequip.utils.torch_geometric import Batch

import ase.build
import ase.geometry
import ase.io
from ase.calculators.singlepoint import SinglePointCalculator

from nequip.data import AtomicData, AtomicDataDict
from nequip.data.AtomicData import neighbor_list_and_relative_vec
from nequip.utils.test import add_spin

# skip_ase = pytest.mark.skipif(not has_ase, reason="ASE library is not installed")


@pytest.mark.parametrize("include_spin", [True, False])
def test_from_ase(CuFcc, include_spin):
    atoms, data = CuFcc
    for key in [AtomicDataDict.FORCE_KEY, AtomicDataDict.POSITIONS_KEY]:
        assert data[key].shape == (len(atoms), 3)  # 4 species in this atoms
    if include_spin:
        data = add_spin([data.clone()])[0]
        for key in [AtomicDataDict.SPIN_KEY]:
            assert data[key].shape == (len(atoms), 3)  # 4 species in this atoms


def test_to_ase(CH3CHO_no_typemap):
    atoms, data = CH3CHO_no_typemap
    to_ase_atoms = data.to_ase()
    assert np.allclose(atoms.get_positions(), to_ase_atoms.get_positions())
    assert np.array_equal(atoms.get_atomic_numbers(), to_ase_atoms.get_atomic_numbers())
    assert np.array_equal(atoms.get_pbc(), to_ase_atoms.get_pbc())
    assert np.array_equal(atoms.get_cell(), to_ase_atoms.get_cell())


@pytest.mark.parametrize("include_spin", [True, False])
def test_to_ase_batches(atomic_batch, include_spin):
    if include_spin:
        atomic_batch = add_spin([atomic_batch.clone()])[0]
    atomic_data = AtomicData.from_dict(atomic_batch.to_dict())
    to_ase_atoms_batch = atomic_data.to_ase()
    for batch_idx, atoms in enumerate(to_ase_atoms_batch):
        mask = atomic_data.batch == batch_idx
        if include_spin:
            assert atoms.get_initial_magnetic_moments().shape == (len(atoms), 3)
            assert np.allclose(
                atoms.get_initial_magnetic_moments(),
                atomic_data[AtomicDataDict.SPIN_KEY][mask],
            )
        assert atoms.get_positions().shape == (len(atoms), 3)
        assert np.allclose(atoms.get_positions(), atomic_data.pos[mask])
        assert atoms.get_atomic_numbers().shape == (len(atoms),)
        assert np.array_equal(
            atoms.get_atomic_numbers(),
            atomic_data[AtomicDataDict.ATOM_TYPE_KEY][mask].view(-1),
        )

        assert (
            np.max(np.abs(atoms.get_cell()[:] - atomic_data.cell[batch_idx].numpy()))
            == 0
        )
        assert not np.logical_xor(
            atoms.get_pbc(), atomic_data.pbc[batch_idx].numpy()
        ).all()


def test_ase_roundtrip(CuFcc):
    atoms, data = CuFcc
    atoms2 = data.to_ase()
    assert np.allclose(atoms.get_positions(), atoms2.get_positions())
    assert np.array_equal(atoms.get_atomic_numbers(), atoms2.get_atomic_numbers())
    assert np.array_equal(atoms.get_pbc(), atoms2.get_pbc())
    assert np.allclose(atoms.get_cell(), atoms2.get_cell())
    assert np.allclose(atoms.calc.results["forces"], atoms2.calc.results["forces"])


@pytest.mark.parametrize("batched", [False, True])
def test_spin_custom_tensor_ase_roundtrip(batched):
    def make_atoms(scale):
        atoms = ase.Atoms(
            numbers=[26, 27],
            positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            cell=np.eye(3) * 8.0,
            pbc=True,
        )
        magmoms = scale * np.array([[2.0, 0.0, 0.0], [0.0, -3.0, 4.0]])
        born = scale * np.arange(18, dtype=float).reshape(2, 9)
        dipole = scale * np.array([0.1, 0.2, 0.3])
        spin_forces = scale * np.arange(6, dtype=float).reshape(2, 3)
        spin_forces_vert = spin_forces + scale
        atomic_dipole = spin_forces - scale
        piezoelectric = scale * np.arange(27, dtype=float).reshape(3, 3, 3)

        atoms.set_initial_magnetic_moments(magmoms)
        atoms.set_array("spin_forces", spin_forces)
        atoms.set_array("spin_forces_vert", spin_forces_vert)
        atoms.set_array("atomic_dipole", atomic_dipole)
        atoms.info["piezoelectric"] = piezoelectric
        atoms.calc = SinglePointCalculator(
            atoms, born_effective_charges=born, dipole=dipole
        )
        return atoms

    inputs = [make_atoms(1.0), make_atoms(2.0)] if batched else [make_atoms(1.0)]
    data_list = [AtomicData.from_ase(atoms, r_max=3.0) for atoms in inputs]
    for atoms, data in zip(inputs, data_list):
        assert data[AtomicDataDict.SPIN_LENGTH_KEY].shape == (len(atoms), 1)
        assert torch.allclose(
            data[AtomicDataDict.SPIN_LENGTH_KEY].view(-1),
            torch.as_tensor(
                np.linalg.norm(atoms.get_initial_magnetic_moments(), axis=-1),
                dtype=data.pos.dtype,
            ),
        )
        assert data[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY].shape == (
            len(atoms),
            9,
        )

    if batched:
        data = AtomicData.from_dict(Batch.from_data_list(data_list).to_dict())
    else:
        data = data_list[0]
    outputs = data.to_ase()
    outputs = outputs if batched else [outputs]

    for original, output in zip(inputs, outputs):
        assert np.allclose(
            output.get_initial_magnetic_moments(),
            original.get_initial_magnetic_moments(),
        )
        assert output.calc.results["born_effective_charges"].shape == (
            len(output),
            9,
        )
        assert np.allclose(
            output.calc.results["born_effective_charges"],
            original.calc.results["born_effective_charges"],
        )
        assert np.allclose(
            output.calc.results["dipole"], original.calc.results["dipole"]
        )
        for key in ["spin_forces", "spin_forces_vert", "atomic_dipole"]:
            assert key in output.arrays
            assert np.allclose(output.arrays[key], original.arrays[key])
            assert key not in output.calc.results
        assert np.allclose(output.info["piezoelectric"], original.info["piezoelectric"])
        assert "piezoelectric" not in output.calc.results

        roundtrip_data = AtomicData.from_ase(output, r_max=3.0)
        assert roundtrip_data[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY].shape == (
            len(output),
            9,
        )
        assert torch.allclose(
            roundtrip_data[AtomicDataDict.SPIN_LENGTH_KEY].view(-1),
            torch.as_tensor(
                np.linalg.norm(original.get_initial_magnetic_moments(), axis=-1),
                dtype=roundtrip_data.pos.dtype,
            ),
        )


def _two_atom_frame():
    return ase.Atoms(
        numbers=[26, 27],
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        cell=np.eye(3) * 8.0,
        pbc=True,
    )


@pytest.mark.parametrize("bec_shape", [(2, 9), (2, 3, 3)])
def test_from_ase_born_effective_charge_shapes(bec_shape):
    bec = np.arange(18, dtype=float)
    atoms = _two_atom_frame()
    atoms.calc = SinglePointCalculator(
        atoms, born_effective_charges=bec.reshape(bec_shape)
    )
    data = AtomicData.from_ase(atoms, r_max=3.0)
    assert data[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY].shape == (len(atoms), 9)
    assert torch.equal(
        data[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY],
        torch.as_tensor(bec.reshape(len(atoms), 9), dtype=data.pos.dtype),
    )


@pytest.mark.parametrize("piezo_shape", [(3, 3, 3), (27,)])
def test_from_ase_piezoelectric_shapes(piezo_shape):
    piezo = np.arange(27, dtype=float).reshape(3, 3, 3)
    atoms = _two_atom_frame()
    atoms.info["piezoelectric"] = piezo.reshape(piezo_shape)
    data = AtomicData.from_ase(atoms, r_max=3.0)
    # graph fields carry no frame dim per frame; Batch adds it
    assert data[AtomicDataDict.PIEZOELECTRIC_KEY].shape == (3, 3, 3)
    output = data.to_ase()
    assert output.info["piezoelectric"].shape == (3, 3, 3)
    assert np.array_equal(output.info["piezoelectric"], piezo)


def test_from_ase_rejects_bad_born_and_piezo_shapes():
    atoms = _two_atom_frame()
    atoms.set_array("born_effective_charges", np.zeros((2, 4)))
    with pytest.raises(RuntimeError, match="bad shape for born_effective_charges"):
        AtomicData.from_ase(atoms, r_max=3.0)

    atoms = _two_atom_frame()
    atoms.info["piezoelectric"] = np.zeros(5)
    with pytest.raises(RuntimeError, match="bad shape for piezoelectric"):
        AtomicData.from_ase(atoms, r_max=3.0)


def test_born_piezo_batch_shapes():
    def make_atoms(scale):
        atoms = _two_atom_frame()
        atoms.calc = SinglePointCalculator(
            atoms,
            born_effective_charges=scale * np.arange(18, dtype=float).reshape(2, 3, 3),
        )
        atoms.info["piezoelectric"] = scale * np.arange(27, dtype=float)
        return atoms

    inputs = [make_atoms(1.0), make_atoms(2.0)]
    batch = Batch.from_data_list([AtomicData.from_ase(a, r_max=3.0) for a in inputs])
    assert batch[AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY].shape == (4, 9)
    assert batch[AtomicDataDict.PIEZOELECTRIC_KEY].shape == (2, 3, 3, 3)

    # the validating constructor enforces the batched (n_frames, 3, 3, 3) contract
    data = AtomicData(**batch.to_dict())
    for original, output in zip(inputs, data.to_ase()):
        assert np.array_equal(
            output.calc.results["born_effective_charges"],
            original.calc.results["born_effective_charges"].reshape(2, 9),
        )
        assert np.array_equal(
            output.info["piezoelectric"],
            original.info["piezoelectric"].reshape(3, 3, 3),
        )


def test_non_periodic_edge(CH3CHO):
    atoms, data = CH3CHO
    # check edges
    for edge in range(data.num_edges):
        real_displacement = (
            atoms.positions[data.edge_index[1, edge]]
            - atoms.positions[data.edge_index[0, edge]]
        )
        assert torch.allclose(
            data.get_edge_vectors()[edge],
            torch.as_tensor(real_displacement, dtype=torch.get_default_dtype()),
        )


def test_edges_missing():
    with pytest.warns(UserWarning, match="no neighbors within the cutoff"):
        atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
        data = AtomicData.from_ase(atoms, r_max=2.5)

    assert data[AtomicDataDict.EDGE_INDEX_KEY].shape == (2, 0)
    assert data[AtomicDataDict.EDGE_CELL_SHIFT_KEY].shape == (0, 3)
    assert data.get_edge_vectors().shape == (0, 3)


def test_periodic_edge():
    atoms = ase.build.bulk("Cu", "fcc")
    dist = np.linalg.norm(atoms.cell[0])
    data = AtomicData.from_ase(atoms, r_max=1.05 * dist)
    edge_vecs = data.get_edge_vectors()
    assert edge_vecs.shape == (12, 3)  # 12 neighbors in close-packed bulk
    assert torch.allclose(
        edge_vecs.norm(dim=-1), torch.as_tensor(dist, dtype=torch.get_default_dtype())
    )


def test_without_nodes(CH3CHO):
    # Non-periodic
    atoms, data = CH3CHO
    which_nodes = [0, 5, 6]
    new_data = data.without_nodes(which_nodes=which_nodes)
    assert new_data.num_nodes == len(atoms) - len(which_nodes)
    new_data.debug()
    assert new_data.edge_index.min() >= 0
    assert new_data.edge_index.max() == new_data.num_nodes - 1

    which_nodes_mask = np.zeros(len(atoms), dtype=bool)
    which_nodes_mask[[0, 1, 2, 4]] = True
    new_data = data.without_nodes(which_nodes=which_nodes_mask)
    assert new_data.num_nodes == len(atoms) - np.sum(which_nodes_mask)
    new_data.debug()
    assert new_data.edge_index.min() >= 0
    assert new_data.edge_index.max() == new_data.num_nodes - 1


@pytest.mark.parametrize("periodic", [True, False])
def test_positions_grad(periodic, CH3CHO, CuFcc):

    if periodic:
        atoms, data = CuFcc
    else:
        atoms, data = CH3CHO

    data.pos.requires_grad_(True)
    assert data.get_edge_vectors().requires_grad

    torch.autograd.grad(data.get_edge_vectors().sum(), data.pos, create_graph=True)

    if periodic:
        # Test grad cell
        data.cell.requires_grad_(True)
        assert data.get_edge_vectors().requires_grad
        torch.autograd.grad(data.get_edge_vectors().sum(), data.cell, create_graph=True)


@pytest.mark.parametrize("periodic", [True, False])
def test_spin_grad(periodic, CH3CHO, CuFcc):

    if periodic:
        atoms, data = CuFcc
    else:
        atoms, data = CH3CHO
    data = add_spin([data.clone()])[0]
    data[AtomicDataDict.SPIN_KEY].requires_grad_(True)

    spin = data[AtomicDataDict.SPIN_KEY]

    col = data[AtomicDataDict.EDGE_INDEX_KEY][1]
    row = data[AtomicDataDict.EDGE_INDEX_KEY][0]

    si_sj = torch.sum(spin[row] * spin[col], dim=-1)

    assert si_sj.requires_grad

    torch.autograd.grad(si_sj.sum(), data[AtomicDataDict.SPIN_KEY], create_graph=True)


def test_some_periodic():
    # monolayer in xy,
    # only periodic in xy
    atoms = ase.build.fcc111("Al", size=(3, 3, 1), vacuum=0.0)
    assert all(atoms.pbc == (True, True, False))
    data = AtomicData.from_ase(atoms, r_max=2.9)  # first shell dist is 2.864A
    # Check number of neighbors:
    _, neighbor_count = np.unique(data.edge_index[0].numpy(), return_counts=True)
    assert (neighbor_count == 6).all()  # 6 neighbors
    # Check not periodic in z
    assert torch.allclose(
        data.get_edge_vectors()[:, 2],
        torch.zeros(data.num_edges, dtype=data.get_edge_vectors().dtype),
    )


def test_relative_vecs(H2):
    atoms, data = H2
    coords = data.pos
    edge_index_true = torch.LongTensor([[0, 1], [1, 0]])
    assert edge_index_set_equiv(data.edge_index, edge_index_true)
    assert torch.allclose(
        coords[1] - coords[0],
        data.get_edge_vectors()[(data.edge_index[0] == 0) & (data.edge_index[1] == 1)][
            0
        ],
    )
    assert torch.allclose(
        coords[0] - coords[1],
        data.get_edge_vectors()[(data.edge_index[0] == 1) & (data.edge_index[1] == 0)][
            0
        ],
    )


@pytest.mark.parametrize("interact", [True, False])
def test_self_interaction(interact, Si):
    points, _ = Si
    data = AtomicData.from_points(
        **points,
        self_interaction=interact,
    )
    if interact:
        true = torch.LongTensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    else:
        true = torch.LongTensor([[0, 1], [1, 0]])
    assert edge_index_set_equiv(data.edge_index, true)


def test_silicon_neighbors(Si):

    points, data = Si
    edge_index, cell_shifts, cell = neighbor_list_and_relative_vec(
        points["pos"],
        pbc=True,
        cell=points["cell"],
        r_max=points["r_max"],
        self_interaction=False,
    )
    edge_index_true = torch.LongTensor(
        [[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]]
    )
    assert edge_index_set_equiv(edge_index, edge_index_true)
    assert edge_index_set_equiv(data.edge_index, edge_index_true)


def test_batching(Si):
    _, orig = Si
    N = 4
    datas = []
    for _ in range(N):
        new = copy.deepcopy(orig)
        new.pos += torch.randn_like(new.pos)
        datas.append(new)
    batch = Batch.from_data_list(datas)
    for i, orig in enumerate(datas):
        new = batch.get_example(i)
        for k, v in orig:
            assert torch.equal(v, new[k])


@pytest.mark.parametrize(
    "value", [3, np.int64(3), np.int32(3)], ids=["int", "np.int64", "np.int32"]
)
def test_integer_scalar_field(value):
    data = AtomicData.from_points(
        pos=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        r_max=2.0,
        fold=value,
        weight=np.float64(0.5),
    )
    assert data["fold"].dtype == torch.long
    assert data["fold"].shape == (1,)
    assert data["fold"].item() == 3
    # floating scalars keep their existing handling
    assert data["weight"].dtype == torch.get_default_dtype()
    assert data["weight"].shape == (1,)


def test_extxyz_integer_info_keys(tmp_path):
    from nequip.data import ASEDataset, Collater

    frames = [
        (3, 0.25, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        (-1, 0.5, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.80], [0.0, 0.9, 0.0]]),
    ]
    lines = []
    for fold, weight, pos in frames:
        lines.append(f"{len(pos)}")
        lines.append(
            f'Properties=species:S:1:pos:R:3 fold={fold} weight={weight} pbc="F F F"'
        )
        lines.extend(f"H {x} {y} {z}" for x, y, z in pos)
    file_name = tmp_path / "frames.xyz"
    file_name.write_text("\n".join(lines) + "\n")

    atoms = ase.io.read(str(file_name), index=":", format="extxyz")
    # ASE parses integer info values as numpy integer scalars
    assert isinstance(atoms[0].info["fold"], np.integer)

    dataset = ASEDataset(
        root=str(tmp_path / "root"),
        file_name=str(file_name),
        ase_args=dict(format="extxyz"),
        extra_fixed_fields={"r_max": 2.0},
        include_keys=["fold", "weight"],
    )
    assert len(dataset) == len(frames)
    for i, (fold, weight, _) in enumerate(frames):
        assert dataset[i]["fold"].dtype == torch.long
        assert dataset[i]["fold"].tolist() == [fold]
        assert dataset[i]["weight"].dtype == torch.get_default_dtype()

    batch = Collater.for_dataset(dataset)([dataset[0], dataset[1]])
    assert batch["fold"].dtype == torch.long
    assert batch["fold"].tolist() == [fold for fold, _, _ in frames]


def edge_index_set_equiv(a, b):
    """Compare edge_index arrays in an unordered way."""
    # [[0, 1], [1, 0]] -> {(0, 1), (1, 0)}
    a = (
        a.numpy()
    )  # numpy gives ints when iterated, tensor gives non-identical scalar tensors.
    b = b.numpy()
    return set(zip(a[0], a[1])) == set(zip(b[0], b[1]))


@pytest.fixture(scope="session")
def H2(float_tolerance):
    atoms = ase.build.molecule("H2")
    data = AtomicData.from_ase(atoms, r_max=2.0)
    return atoms, data


@pytest.fixture(scope="session")
def CuFcc(float_tolerance):
    atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = SinglePointCalculator(
        atoms, **{"forces": np.random.random((len(atoms), 3))}
    )
    data = AtomicData.from_ase(atoms, r_max=4.0)
    return atoms, data


@pytest.fixture(scope="session")
def Si(float_tolerance):
    lattice = torch.tensor(
        [
            [3.34939851, 0, 1.93377613],
            [1.11646617, 3.1578432, 1.93377613],
            [0, 0, 3.86755226],
        ]
    )
    coords = torch.tensor([[0, 0, 0], [1.11646617, 0.7894608, 1.93377613]])
    r_max = 2.5
    points = dict(
        # z=torch.zeros(size=(len(coords), 1)),
        pos=coords,
        r_max=r_max,
        cell=lattice,
        pbc=True,
    )
    data = AtomicData.from_points(**points)
    return points, data


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("flat", [False, True])
def test_response_from_ase_shapes_and_batch(rank, flat, tmp_path):
    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[3, 4, 5], pbc=True)
    node = "ipv" if rank == 2 else "ipd"
    graph = ("epsilon_ion", "epsilon_el") if rank == 2 else ("piezo_ion", "piezo_el")
    width = 3**rank
    values = np.arange(2 * width, dtype=float).reshape(2, width)
    atoms.arrays[node] = values if flat else values.reshape((2,) + (3,) * rank)
    for key in graph:
        atoms.info[key] = np.arange(width, dtype=float).reshape(
            (width,) if flat else (3,) * rank
        )
    atoms.info["piezoelectric"] = np.arange(27, dtype=float).reshape(3, 3, 3)
    data = AtomicData.from_ase(atoms, r_max=1.5)
    assert torch.equal(data[node], torch.as_tensor(values, dtype=data[node].dtype))
    for key in graph:
        assert torch.equal(data[key], torch.arange(width, dtype=data[key].dtype))
    assert data["piezoelectric"].shape == (3, 3, 3)
    batch = Batch.from_data_list([data, data.clone()])
    assert batch[node].shape == (4, width)
    for key in graph:
        assert batch[key].shape == (2, width)
        assert torch.equal(batch[key][1], data[key])
    # extxyz's default parser must preserve all flat Cartesian components.
    atoms.arrays[node] = values
    for key in graph:
        atoms.info[key] = np.arange(width, dtype=float)
    path = tmp_path / "response.extxyz"
    ase.io.write(path, atoms)
    parsed = AtomicData.from_ase(ase.io.read(path), r_max=1.5)
    for key in (node,) + graph:
        assert torch.equal(parsed[key], data[key])


@pytest.mark.parametrize(
    "key,shape",
    [
        ("ipv", (2, 8)),
        ("ipd", (2, 3, 9)),
        ("epsilon_el", (1, 9)),
        ("epsilon_ion", (8,)),
        ("piezo_el", (3, 9)),
        ("piezo_ion", (26,)),
    ],
)
def test_response_from_ase_bad_shapes(key, shape):
    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.info[key] = np.zeros(shape)
    with pytest.raises(RuntimeError, match=f"bad shape for {key}"):
        AtomicData.from_ase(atoms, r_max=1.5)


@pytest.mark.parametrize(
    "key,width", [("ipv", 9), ("ipd", 27), ("epsilon_el", 9), ("piezo_el", 27)]
)
def test_response_direct_shape_check(key, width):
    kwargs = {
        key: torch.zeros((2, width - 1) if key in ("ipv", "ipd") else (width - 1,))
    }
    with pytest.raises(AssertionError):
        AtomicData.from_points(
            pos=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), r_max=1.5, **kwargs
        )


@pytest.mark.parametrize(
    "key,width",
    [
        (AtomicDataDict.EDGE_IPV_CONTRIBUTION_KEY, 9),
        (AtomicDataDict.EDGE_IPD_CONTRIBUTION_KEY, 18),
        (AtomicDataDict.EDGE_EPS_EL_CONTRIBUTION_KEY, 6),
        (AtomicDataDict.EDGE_PIEZO_EL_CONTRIBUTION_KEY, 18),
    ],
)
def test_response_edge_registration_and_batch(key, width):
    from nequip.data.AtomicData import _EDGE_FIELDS

    assert key in _EDGE_FIELDS
    frames = []
    for nodes in (2, 3):
        edges = torch.stack([torch.arange(nodes), torch.arange(nodes).roll(1)])
        frames.append(
            AtomicData(
                pos=torch.randn(nodes, 3),
                edge_index=edges,
                **{key: torch.arange(nodes * width).reshape(nodes, width).float()},
            )
        )
    batch = Batch.from_data_list(frames)
    assert torch.equal(batch[key], torch.cat([frame[key] for frame in frames]))
    for actual, expected in zip(batch.to_data_list(), frames):
        assert torch.equal(actual[key], expected[key])
        assert torch.equal(actual.edge_index, expected.edge_index)
