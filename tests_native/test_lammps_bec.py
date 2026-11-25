"""DREAM force, polarization, migration, and model-to-field integration."""

import io
import json
import os

import numpy as np
import pytest

from spingnn2.reference import ReferenceModel
from test_lammps_pair import POS, SPIN, TYPES, _reference_graph


FIELD = np.array([0.2, -0.1, 0.3])
TENSOR = np.array([[1.0, 0.2, -0.4], [0.7, 1.3, 0.5], [-0.2, 0.8, 1.7]])
TENSOR_FIELDS = "becxx becxy becxz becyx becyy becyz beczx beczy beczz"
DIPOLE_FIELDS = "becdipolex becdipoley becdipolez"


def _frames(path):
    return [
        np.loadtxt(io.StringIO(block.split("ITEM: ATOMS ", 1)[1]), skiprows=1, ndmin=2)
        for block in path.read_text().split("ITEM: TIMESTEP\n")[1:]
    ]


def _marker(output, name):
    line = next(line for line in output.splitlines() if line.startswith(name + " "))
    return np.array([float(value) for value in line.split()[1:]])


def _fixed_run(lmp_run, path, coupling, ranks):
    path.mkdir()
    bec = path / "fixed.bec"
    tensors = np.stack([TENSOR, -TENSOR])
    np.savetxt(bec, np.column_stack([[1, 2], tensors.reshape(2, 9)]), fmt="%g")
    atoms, charges = "bec", ""
    if coupling == "both":
        atoms = "hybrid charge bec"
        charges = "set atom 1 charge 0.4\nset atom 2 charge -0.4"
    field = f"fix field all efield 0.2 -0.1 0.3 coupling {coupling} piezo no"
    columns = f"id xu yu zu fx fy fz {TENSOR_FIELDS} {DIPOLE_FIELDS} c_dip[1] c_dip[2] c_dip[3]"
    script = f"""units metal
boundary p p p
atom_style {atoms}
atom_modify map hash sort 1 0.0
newton off
processors {ranks} 1 1
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 9.99 2 2 units box
create_atoms 1 single 4.99 6 6 units box
mass 1 100
{charges}
read_bec "{bec}"
pair_style lj/cut 2.0
pair_coeff * * 0.0 1.0
{field}
fix_modify field energy yes virial yes
compute pol all bec
compute dip all property/atom {DIPOLE_FIELDS}
neighbor 0.1 bin
neigh_modify every 1 delay 0 check no
velocity all set 20 -10 5 units box
fix integrate all nve
timestep 0.001
dump trajectory all custom 1 "{path / 'trajectory.dump'}" {columns}
dump_modify trajectory sort id format float %.17g
thermo 1
thermo_style custom step etotal f_field c_pol c_pol[1] c_pol[2] c_pol[3] c_pol[4]
run 2
print "BEC_FIELD $(f_field:%.17g) $(c_pol[1]:%.17g) $(c_pol[2]:%.17g) $(c_pol[3]:%.17g)"
write_restart "{path / 'state.restart'}"
undump trajectory
bec_dipole_reset
write_dump all custom "{path / 'reset.dump'}" id {DIPOLE_FIELDS} modify sort id
clear
newton off
read_restart "{path / 'state.restart'}"
{field}
fix_modify field energy yes virial yes
compute dip all property/atom {DIPOLE_FIELDS}
dump restored all custom 1 "{path / 'restored.dump'}" {columns}
dump_modify restored sort id format float %.17g
run 0
"""
    output = lmp_run(script, ranks=ranks)
    trajectory = np.stack(_frames(path / "trajectory.dump"))
    restored = _frames(path / "restored.dump")[0]
    np.testing.assert_allclose(restored, trajectory[-1], rtol=0, atol=1e-11)
    np.testing.assert_array_equal(
        _frames(path / "reset.dump")[0][:, 1:], np.zeros((2, 3))
    )
    np.testing.assert_array_equal(
        trajectory[:, :, 7:16],
        np.broadcast_to(tensors.reshape(2, 9), (len(trajectory), 2, 9)),
    )
    force = tensors.transpose(0, 2, 1) @ FIELD
    if coupling == "both":
        force += np.array([0.4, -0.4])[:, None] * FIELD
    np.testing.assert_allclose(
        trajectory[:, :, 4:7],
        np.broadcast_to(force, (len(trajectory), 2, 3)),
        rtol=0,
        atol=1e-11,
    )
    displacement = trajectory[:, :, 1:4] - trajectory[0, :, 1:4]
    assert trajectory[-1, 0, 1] > 10.0 and trajectory[-1, 1, 1] > 5.0
    dipoles = np.einsum("nij,tnj->tni", tensors, displacement)
    np.testing.assert_allclose(trajectory[:, :, 16:19], dipoles, rtol=0, atol=1e-11)
    np.testing.assert_array_equal(trajectory[:, :, 19:22], trajectory[:, :, 16:19])
    final = _marker(output, "BEC_FIELD")
    np.testing.assert_allclose(final[1:], dipoles[-1].sum(axis=0), rtol=0, atol=1e-11)
    if coupling == "bec":
        assert abs(final[0] + float(dipoles[-1].sum(axis=0) @ FIELD)) <= 1e-11
    print(
        "M4_BEC_FIELD",
        json.dumps(
            dict(
                coupling=coupling,
                ranks=ranks,
                force_max_abs=float(np.max(np.abs(trajectory[:, :, 4:7] - force))),
                dipole_max_abs=float(np.max(np.abs(trajectory[:, :, 16:19] - dipoles))),
                restart_max_abs=float(np.max(np.abs(restored - trajectory[-1]))),
            ),
            sort_keys=True,
        ),
    )
    return trajectory


@pytest.mark.parametrize("coupling", ("bec", "both"))
def test_fixed_bec_field_and_restart(lmp_run, tmp_path, coupling):
    one = _fixed_run(lmp_run, tmp_path / "single", coupling, 1)
    ranks = os.environ.get("SPINGNN2_M4_MPI_RANKS", "1").split(",")
    for n in map(int, ranks):
        if n == 1:
            continue
        multi = _fixed_run(lmp_run, tmp_path / f"ranks-{n}", coupling, n)
        error = float(np.max(np.abs(one - multi)))
        print("M4_BEC_MPI", json.dumps(dict(ranks=n, coupling=coupling, max_abs=error)))
        assert error <= 1e-11


def test_dream_pair_feeds_bec_field(lmp_run, m4_pair, tmp_path):
    pair = m4_pair("born")
    forces = []
    tensors = []
    for field_on in (False, True):
        dump = tmp_path / f"model-{int(field_on)}.dump"
        field = "fix field all efield 0.2 -0.1 0.3 coupling bec" if field_on else ""
        lmp_run(
            f"""units metal
boundary p p p
atom_style bec
atom_modify map array
lattice sc 2.4
region box block 0 5 0 5 0 5
create_box 2 box
create_atoms 1 box
set atom 12 type 2
displace_atoms all random 0.08 0.08 0.08 5823 units box
mass 1 54.938
mass 2 72.630
{pair}
{field}
run 0
write_dump all custom "{dump}" id fx fy fz {TENSOR_FIELDS} modify sort id format float %.17g
"""
        )
        frame = _frames(dump)[0]
        forces.append(frame[:, 1:4])
        tensors.append(frame[:, 4:].reshape(-1, 3, 3))
    np.testing.assert_allclose(tensors[0], tensors[1], rtol=0, atol=1e-11)
    assert np.max(np.abs(tensors[0])) > 1e-12
    observed = forces[1] - forces[0]
    expected = tensors[0].transpose(0, 2, 1) @ FIELD
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-11)
    print("M4_DREAM_MODEL_FIELD", float(np.max(np.abs(observed - expected))))


def test_dream_hard_asr_without_gpu_aware_mpi(
    lmp_run, m4_backend, m4_pair, m4_models, tmp_path
):
    if m4_backend not in ("cuda", "kk-cpu"):
        pytest.skip("Kokkos is required for the GPU-aware MPI setting")
    if "2" not in os.getenv("SPINGNN2_M4_MPI_RANKS", "1").split(","):
        pytest.skip("Enable two ranks in SPINGNN2_M4_MPI_RANKS")

    model = ReferenceModel.load(m4_models["born"]["float64"]["contract"])
    assert model.bec["asr"] == "hard"
    key = model.bec["out_field"]
    reference = _reference_graph(model, POS, TYPES, SPIN, 16.0)[key].reshape(-1, 9)
    # The global projection must retain distinct means in the two domains.
    assert (
        np.max(np.abs(reference[POS[:, 0] < 8].mean(axis=0) - reference.mean(axis=0)))
        > 1e-8
    )
    atoms = "\n".join(
        f"create_atoms {species + 1} single {x} {y} {z} units box"
        for species, (x, y, z) in zip(TYPES, POS)
    )
    pair = m4_pair("born").replace("pair_style spingnn2 ", "pair_style spingnn2/kk ")
    results = []
    for ranks in (1, 2):
        dump = tmp_path / f"asr-{ranks}.dump"
        script = f"""units metal
boundary p p p
atom_style bec
atom_modify map array
processors {ranks} 1 1
region box block 0 16 0 16 0 16 units box
create_box 2 box
{atoms}
mass 1 55
mass 2 81
{pair}
run 0
write_dump all custom "{dump}" id {TENSOR_FIELDS} modify sort id format float %.17g
"""
        output = lmp_run(script, ranks=ranks, gpu_aware=False, allow_shared_gpu=True)
        assert f"{ranks} by 1 by 1 MPI processor grid" in output
        frame = _frames(dump)[0]
        np.testing.assert_array_equal(frame[:, 0], np.arange(1, len(POS) + 1))
        tensors = frame[:, 1:]
        np.testing.assert_allclose(tensors, reference, rtol=0, atol=1e-11)
        np.testing.assert_allclose(
            tensors.mean(axis=0), reference.mean(axis=0), rtol=0, atol=1e-11
        )
        results.append(tensors)
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-11)
    print(
        "M4_DREAM_HARD_ASR",
        json.dumps(
            dict(
                ranks=2,
                gpu_aware=False,
                oracle_max_abs=float(np.max(np.abs(results[1] - reference))),
                rank_max_abs=float(np.max(np.abs(results[1] - results[0]))),
            ),
            sort_keys=True,
        ),
    )
