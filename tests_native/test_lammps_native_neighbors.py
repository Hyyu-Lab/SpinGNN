"""Native partition outputs and CUDA skin-list/stream interop for LAMMPS."""

import ctypes as ct
import os

import numpy as np
import pytest
import torch

from spingnn2.native import _checked, _Graph, _Options, _prepare
from test_contract_oracle import CASES
from test_jit_runtime import _open, artifacts as _artifact_fixture  # noqa: F401
from test_native_engine import BACKEND, native_case as _native_fixture  # noqa: F401


MODELS = ("nosoc-te3nn", "soc-u2", "born-hard", "e2ip-force", "soc-e2ip-spin")
PARTITION_CASES = [case for case in CASES if case.name in MODELS]
SKIP_LATTICE_FORCES = 1


@pytest.fixture
def build_case(request):
    return request.getfixturevalue("_native_fixture")


@pytest.fixture
def prepare_artifact(request):
    return request.getfixturevalue("_artifact_fixture")


class _Partition(_Graph):
    _fields_ = [("num_owned", ct.c_int32), ("num_global", ct.c_int64)]


class _Neighbors(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("num_atoms", ct.c_int32),
        ("num_owned", ct.c_int32),
        ("position_precision", ct.c_int32),
        ("num_global", ct.c_int64),
        *((name, ct.c_void_p) for name in ("positions", "spins")),
        *(
            (name, ct.c_int64)
            for name in (
                "position_atom_stride",
                "position_component_stride",
                "spin_atom_stride",
                "spin_component_stride",
            )
        ),
        *(
            (name, ct.c_void_p)
            for name in ("types", "type_map", "ilist", "numneigh", "firstneigh")
        ),
        *((name, ct.c_int32) for name in ("type_map_size", "neighbor_mask", "slots")),
        *(
            (name, ct.c_int64)
            for name in (
                "ilist_stride",
                "numneigh_stride",
                "neighbor_atom_stride",
                "neighbor_slot_stride",
            )
        ),
        ("topology_epoch", ct.c_uint64),
        ("lattice_epoch", ct.c_uint64),
        ("stream", ct.c_void_p),
    ]


class _Diagnostics(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        *(
            (name, ct.c_uint64)
            for name in ("workspace", "host_cache", "peak_tile_edges")
        ),
        *((name, ct.c_double) for name in ("geometry", "mlp", "tp", "reverse")),
        ("device_cache", ct.c_uint64),
        ("stage1", ct.c_double),
        ("stage2", ct.c_double),
        ("cache_hits", ct.c_uint64),
        ("metadata_bytes", ct.c_uint64),
    ]


class _Evidence(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in ("sigma", "Uepi", "nu", "kappa")]


class _DeviceResult(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        *(
            (name, ct.c_void_p)
            for name in ("forces", "spin_forces", "owned_energy", "atom_virial", "bec")
        ),
        ("diagnostics", _Diagnostics),
        ("f", _Evidence),
        ("sf", _Evidence),
    ]


class _GraphNeighbors(_Neighbors):
    _fields_ = [("flags", ct.c_uint32)]


class _GraphDeviceResult(_DeviceResult):
    _fields_ = [("cuda_graph_status", ct.c_uint32)]


def _m4_library(model):
    if hasattr(model._lib, "sp2_evaluate_neighbors"):
        return
    if "SPINGNN2_LIBRARY" in os.environ:
        pytest.fail("The requested library lacks the M4 neighbor API")
    pytest.skip("Build the M4 native library to exercise partition outputs")


def _data(model):
    return {
        "pos": model.info["r_max"]
        * np.array(
            [[0, 0, 0], [0.42, 0.16, 0], [0.79, 0.04, 0.19], [1.1, 0.12, 0], [3, 0, 0]]
        ),
        "atom_types": np.arange(5, dtype=np.int32) % model.info["num_types"],
        "spin": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, -1, 0]]),
    }


def _with_edges(data, model, owners):
    x = data["pos"].astype(model.dtype)
    rsq = ((x[:owners, None] - x[None, :]) ** 2).sum(-1)
    cutoff = model.dtype.type(model.info["r_max"])
    enabled = (rsq < cutoff * cutoff) & (rsq > 0)
    return {**data, "edge_index": np.stack(np.where(enabled))}


def _partition(model, data, owners, global_atoms=5):
    legacy, keepalive = _prepare(_with_edges(data, model, owners), model.dtype)
    graph = _Partition(
        *(getattr(legacy, name) for name, _ in legacy._fields_),
        num_owned=owners,
        num_global=global_atoms,
    )
    graph.struct_size = ct.sizeof(graph)
    opts = model._opts(model.tile_atoms, None, legacy.num_atoms)
    out = ct.c_void_p()
    _checked(
        model._lib,
        model._lib.sp2_evaluate(
            model._handle, ct.byref(graph), ct.byref(opts), ct.byref(out)
        ),
    )
    return model._read(out)


@pytest.mark.parametrize("case", PARTITION_CASES, ids=lambda case: case.name)
def test_m4_owned_centers_and_ghost_adjoint(build_case, prepare_artifact, case):
    fixture = build_case(case)
    artifact, library = prepare_artifact(fixture)
    with _open(fixture, artifact, library, "direct", tile_atoms=2) as model:
        _m4_library(model)
        data = _data(model)
        permutations = [np.arange(5), np.array([3, 4, 0, 1, 2])]
        whole = model.evaluate(_with_edges(data, model, 5))
        parts = [
            _partition(model, {key: value[perm] for key, value in data.items()}, count)
            for count, perm in zip((3, 2), permutations)
        ]
        energy = sum(np.sum(part["owned_energy"]) for part in parts)
        np.testing.assert_allclose(energy, whole["total_energy"], rtol=0, atol=1e-11)
        for key in ("forces", "spin_forces"):
            if key in whole:
                expected = sum(
                    part[key][np.argsort(perm)]
                    for part, perm in zip(parts, permutations)
                )
                np.testing.assert_allclose(expected, whole[key], rtol=0, atol=1e-11)
        np.testing.assert_allclose(
            sum(part["virial"] for part in parts), whole["virial"], rtol=0, atol=1e-11
        )
        for key in whole:
            if (
                key.startswith(("f_", "sf_"))
                and key.endswith(("_sigma", "_Uepi", "_nu", "_kappa"))
                or key == "atom_virial"
            ):
                np.testing.assert_allclose(
                    np.concatenate([part[key] for part in parts]),
                    whole[key],
                    rtol=0,
                    atol=1e-11,
                )
        if model.info.get("bec_asr") == "hard":
            key = model.info["bec"]
            raw = np.concatenate([part[key] for part in parts])
            projected = raw - np.mean(raw, axis=0) + model.info["bec_asr_shift"]
            np.testing.assert_allclose(projected, whole[key], rtol=0, atol=1e-11)
        empty = _partition(model, {key: value[:0] for key, value in data.items()}, 0)
        assert empty["owned_energy"].shape == (0, 1)
        assert empty["forces"].shape == (0, 3)
        assert empty["total_energy"].item() == 0.0


def _device_arrays(data, owners, skin_radius=np.inf):
    n = len(data["pos"])
    spins = np.column_stack(
        (data["spin"], np.linalg.norm(data["spin"], axis=1))
    ).astype(data["pos"].dtype)
    distances = np.linalg.norm(data["pos"][None] - data["pos"][:, None], axis=2)
    rows = [np.flatnonzero((row < skin_radius) & (row > 0)) for row in distances]
    lengths = np.array([len(row) for row in rows], dtype=np.int32)
    neighbor_rows = np.zeros((n, max(lengths)), dtype=np.int32, order="F")
    for i, row in enumerate(rows):
        neighbor_rows[i, : len(row)] = row | (1 << 30)
    values = (
        np.ascontiguousarray(data["pos"].T).T,
        np.ascontiguousarray(spins.T).T,
        data["atom_types"] + 1,
        np.arange(-1, 1 + np.max(data["atom_types"]), dtype=np.int32),
        np.arange(owners - 1, -1, -1, dtype=np.int32),
        lengths,
        neighbor_rows,
    )
    return tuple(torch.from_numpy(v).to("cuda", non_blocking=True) for v in values)


def _neighbor_descriptor(device_arrays, owners, stream, *, graph=False):
    x, sp, types, mapping, ilist, degrees, edges = device_arrays
    cls = _GraphNeighbors if graph else _Neighbors
    return cls(
        ct.sizeof(cls),
        len(x),
        owners,
        8 * x.element_size(),
        len(x),
        x.data_ptr(),
        sp.data_ptr(),
        *x.stride(),
        *sp.stride(),
        types.data_ptr(),
        mapping.data_ptr(),
        ilist.data_ptr(),
        degrees.data_ptr(),
        edges.data_ptr(),
        mapping.numel(),
        0x3FFFFFFF,
        edges.shape[1],
        ilist.stride(0),
        degrees.stride(0),
        *edges.stride(),
        1,
        0,
        stream.cuda_stream,
        **({"flags": 1} if graph else {}),
    )


def _evaluate_neighbors(model, descriptor, opts):
    output_type = (
        _GraphDeviceResult if isinstance(descriptor, _GraphNeighbors) else _DeviceResult
    )
    output = output_type(ct.sizeof(output_type))
    evaluate = model._lib.sp2_evaluate_neighbors
    evaluate.argtypes = (
        ct.c_void_p,
        ct.POINTER(_Neighbors),
        ct.POINTER(_Options),
        ct.POINTER(_DeviceResult),
    )
    evaluate.restype = ct.c_int
    _checked(
        model._lib,
        evaluate(model._handle, ct.byref(descriptor), ct.byref(opts), ct.byref(output)),
    )
    return output


def _copy_cuda_results(out, model, descriptor, opts):
    runtime = ct.CDLL("libcudart.so.12")
    copy = runtime.cudaMemcpyAsync
    copy.argtypes = (ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_void_p)
    copy.restype = ct.c_int
    cuda_stream = ct.c_void_p(descriptor.stream)
    result = {}
    owners, atoms = descriptor.num_owned, descriptor.num_atoms
    lattice_forces = not (opts.eval_flags & SKIP_LATTICE_FORCES)

    def store(name, pointer, shape, *, expected=True):
        assert bool(pointer) == expected, f"Unexpected device pointer for {name}"
        if expected:
            value = np.empty(shape, dtype=model.dtype)
            assert copy(value.ctypes.data, pointer, value.nbytes, 2, cuda_stream) == 0
            result[name] = value

    store("forces", out.forces, (atoms, 3), expected=lattice_forces)
    store("spin_forces", out.spin_forces, (atoms, 3), expected=model.info["magnetic"])
    store("owned_energy", out.owned_energy, (owners, 1))
    store("atom_virial", out.atom_virial, (owners, 3, 3), expected=lattice_forces)
    bec = model.info.get("bec")
    store(bec or "bec", out.bec, (owners, 9), expected=bec is not None)
    for prefix in ("f", "sf"):
        evidence = getattr(out, prefix)
        for key, shape in (
            ("sigma", (3, 3)),
            ("Uepi", (3, 3)),
            ("nu", (1,)),
            ("kappa", (1,)),
        ):
            store(
                prefix + "_" + key,
                getattr(evidence, key),
                (owners, *shape),
                expected=prefix in model.info["prefixes"],
            )
    runtime.cudaStreamSynchronize.argtypes = (ct.c_void_p,)
    assert runtime.cudaStreamSynchronize(cuda_stream) == 0
    return result


@pytest.mark.skipif(BACKEND != "cuda", reason="CUDA device-pointer evaluation")
@pytest.mark.parametrize("case", PARTITION_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_m4_cuda_neighbor_skin_and_cache(build_case, prepare_artifact, case, dtype):
    fixture = build_case(case, dtype)
    artifact, library = prepare_artifact(fixture)
    with _open(fixture, artifact, library, "direct", tile_atoms=2) as model:
        _m4_library(model)
        owners = 3
        data = _data(model)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            arrays = _device_arrays(data, owners)
            x, sp = arrays[:2]
            descriptor = _neighbor_descriptor(arrays, owners, stream)
            opts = model._opts(model.tile_atoms, None, descriptor.num_atoms)

            def check(moved, *, graph_status=None):
                expected = _partition(model, moved, owners)
                out = _evaluate_neighbors(model, descriptor, opts)
                if graph_status is not None:
                    assert out.cuda_graph_status == graph_status
                centers, _ = _with_edges(moved, model, owners)["edge_index"]
                counts = np.bincount(centers // opts.tile_atoms, minlength=1)
                assert out.diagnostics.peak_tile_edges == counts.max()
                actual = _copy_cuda_results(out, model, descriptor, opts)
                for name, value in actual.items():
                    tolerance = 1e-11 if dtype == "float64" else 5e-5
                    target = expected[name]
                    if name.startswith(("f_", "sf_")) and dtype == "float32":
                        scale = np.maximum(1, np.abs(target))
                        np.testing.assert_allclose(
                            value / scale, target / scale, atol=1e-4, rtol=0
                        )
                    else:
                        np.testing.assert_allclose(
                            value, target, atol=tolerance, rtol=0, err_msg=name
                        )
                return out.diagnostics

            diag = check(data)
            assert diag.metadata_bytes == 44 and diag.cache_hits == 0
            moved = {key: value.copy() for key, value in data.items()}
            moved["pos"][3, 0] *= 0.8
            x.copy_(torch.from_numpy(moved["pos"]), non_blocking=True)
            descriptor.lattice_epoch = 6
            diag = check(moved)
            assert diag.metadata_bytes == 12 and diag.cache_hits == 0
            if model.info["magnetic"]:
                # Advance the geometry twice without rebuilding the skin list.
                # Every refresh must use the new displacements before reuse.
                for epoch in (7, 8):
                    moved["pos"][2, 1] += 0.07
                    x.copy_(torch.from_numpy(moved["pos"]), non_blocking=True)
                    descriptor.lattice_epoch = epoch
                    refreshed = check(moved)
                    assert refreshed.cache_hits == 0
                    assert refreshed.metadata_bytes == 12
                    assert refreshed.device_cache == diag.device_cache
                moved["spin"] = np.roll(moved["spin"], 1, axis=0)
                spins = np.column_stack(
                    (moved["spin"], np.linalg.norm(moved["spin"], axis=1))
                )
                sp.copy_(torch.from_numpy(spins), non_blocking=True)
                opts.eval_flags, opts.profile = 1, 1
                diag = check(moved)
                assert diag.cache_hits == 2 and diag.device_cache
                assert (
                    diag.stage1 == 0.0
                    and diag.stage2 > 0.0
                    and diag.metadata_bytes == 0
                )
                full_bytes = diag.device_cache
                opts.eval_flags = 2
                lean = check(moved)
                assert lean.cache_hits == 0
                assert 0 < lean.device_cache < full_bytes
                for _ in range(2):
                    opts.eval_flags = 3
                    replay = check(moved)
                    assert replay.cache_hits == 2
                    assert replay.stage1 == 0
                    assert replay.device_cache == lean.device_cache
                    opts.eval_flags = 2
                    recomputed = check(moved)
                    assert recomputed.cache_hits == 0
                    assert recomputed.device_cache == lean.device_cache

            # Exercise both legacy and append-only ABI views in one evaluation
            # sequence and assert that the graph actually launches.
            descriptor = _GraphNeighbors(
                *(getattr(descriptor, name) for name, _ in _Neighbors._fields_), flags=1
            )
            descriptor.struct_size = ct.sizeof(descriptor)
            descriptor.lattice_epoch = 0
            opts.profile = opts.eval_flags = 0
            assert check(moved, graph_status=0).cache_hits == 0
            check(moved, graph_status=1)
            moved["pos"][2, 1] += 0.015
            x.copy_(torch.from_numpy(moved["pos"]), non_blocking=True)
            check(moved, graph_status=2)

            new_x = x.clone()
            descriptor.positions = new_x.data_ptr()
            (
                descriptor.position_atom_stride,
                descriptor.position_component_stride,
            ) = new_x.stride()
            check(moved, graph_status=0)
            check(moved, graph_status=1)
            descriptor.topology_epoch += 1
            assert check(moved, graph_status=0).metadata_bytes > 0
            check(moved, graph_status=1)
            check(moved, graph_status=2)
            opts.tile_atoms = 3
            assert check(moved, graph_status=0).metadata_bytes > 0
            check(moved, graph_status=1)
            opts.tile_atoms = 2

            if model.info["magnetic"]:
                descriptor.lattice_epoch = 41
                opts.eval_flags = 2
                check(moved, graph_status=0)
                opts.eval_flags = 3
                assert check(moved, graph_status=0).cache_hits == 2
                assert check(moved, graph_status=1).cache_hits == 2
                moved["spin"] = np.roll(moved["spin"], 1, axis=0)
                spins = np.column_stack(
                    (moved["spin"], np.linalg.norm(moved["spin"], axis=1))
                )
                sp.copy_(torch.from_numpy(spins), non_blocking=True)
                assert check(moved, graph_status=2).cache_hits == 2


@pytest.mark.skipif(BACKEND != "cuda", reason="CUDA device-pointer evaluation")
@pytest.mark.parametrize(
    "case",
    [case for case in PARTITION_CASES if case.name in ("soc-u2", "born-hard")],
    ids=lambda case: case.name,
)
@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_m4_cuda_cutoff_crossing(build_case, prepare_artifact, case, dtype):
    fixture = build_case(case, dtype)
    artifact, library = prepare_artifact(fixture)
    with _open(fixture, artifact, library, "direct", tile_atoms=2) as model:
        _m4_library(model)
        owners = 3
        data = _data(model)
        cutoff = model.dtype.type(model.info["r_max"])
        data["pos"] = np.array(
            [
                [0, 0, 0],
                [0, 0, 10],
                [0, 0, 20],
                [cutoff - 0.5, 0, 0],
                [0, cutoff + 0.125, 0],
            ],
            dtype=model.dtype,
        )
        stream = torch.cuda.Stream()
        tolerance = 1e-11 if dtype == "float64" else 5e-5
        max_error = 0.0
        with torch.cuda.stream(stream):
            arrays = _device_arrays(data, owners, cutoff + 1.5)
            x, sp = arrays[:2]
            graph = _neighbor_descriptor(arrays, owners, stream, graph=True)
            opts = model._opts(model.tile_atoms, None, graph.num_atoms)

            def check(num_edges, status=None, hits=0):
                nonlocal max_error
                x.copy_(torch.from_numpy(data["pos"]), non_blocking=True)
                out = _evaluate_neighbors(model, graph, opts)
                if status is not None:
                    assert out.cuda_graph_status == status
                assert out.diagnostics.peak_tile_edges == num_edges
                assert out.diagnostics.cache_hits == hits
                values = _copy_cuda_results(out, model, graph, opts)
                reference = _partition(model, data, owners)
                for name, value in values.items():
                    error = np.max(np.abs(value - reference[name]))
                    assert (
                        error <= tolerance if dtype == "float64" else error < tolerance
                    ), f"{name}: {error:.6g}"
                    max_error = max(max_error, error)
                return values

            before = check(1, 0)
            check(1, 1)
            check(1, 2)
            # One edge leaves and one enters the same tile. Its dimensions
            # permit replay, so stale CSR data would lead to a missed force.
            data["pos"][3, 0] = cutoff + 0.125
            data["pos"][4, 1] = cutoff - 0.5
            after = check(1, 2)
            assert np.linalg.norm(before["forces"][3]) > 0
            assert np.linalg.norm(after["forces"][4]) > 0
            np.testing.assert_array_equal(after["forces"][[1, 2, 3]], 0)

            near = np.nextafter(cutoff, model.dtype.type(0))
            data["pos"][4, 1] = near
            inside = check(1, 2)
            data["pos"][4, 1] = cutoff
            boundary = check(0, 0)
            for name, value in boundary.items():
                error = np.max(np.abs(value - inside[name]))
                assert (
                    error <= tolerance if dtype == "float64" else error < tolerance
                ), f"{name}: {error:.6g}"
            data["pos"][4, 1] = cutoff + 0.125
            check(0, 1)
            check(0, 2)

            if model.info["magnetic"]:
                # A moving evaluation recompacts, then spin-only midpoints
                # read that geometry and its lean Stage-1 tape in-place.
                data["pos"][4, 1] = cutoff - 0.5
                graph.lattice_epoch = 2
                opts.eval_flags = 2
                check(1, 0)
                opts.eval_flags = 3
                check(1, 0, hits=2)
                check(1, 1, hits=2)
                data["spin"][0] = [0, 1, 0]
                spins = np.column_stack(
                    (data["spin"], np.linalg.norm(data["spin"], axis=1))
                )
                sp.copy_(torch.from_numpy(spins), non_blocking=True)
                check(1, 2, hits=2)
        print(f"SPINGNN2_CUTOFF_CROSSING {case.name} {dtype} max_abs={max_error:.6g}")
