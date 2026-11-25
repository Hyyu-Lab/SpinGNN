"""Regression tests for P4 bucketed CUDA Graph padding, density-spike reallocation,
and tile_atoms guardrails."""

import numpy as np
import pytest
import torch

from test_contract_oracle import CASES
from test_jit_runtime import _open, artifacts as _artifact_fixture  # noqa: F401
from test_lammps_native_neighbors import (
    _copy_cuda_results,
    _device_arrays,
    _evaluate_neighbors,
    _m4_library,
    _neighbor_descriptor,
    _partition,
    _with_edges,
)
from test_native_engine import BACKEND, native_case as _native_fixture  # noqa: F401


BUCKET_CASES = [
    case
    for case in CASES
    if case.name in ("nosoc-te3nn", "soc-u2", "born-hard", "soc-e2ip-spin")
]


@pytest.fixture
def build_case(request):
    return request.getfixturevalue("_native_fixture")


@pytest.fixture
def prepare_artifact(request):
    return request.getfixturevalue("_artifact_fixture")


def _cluster_system(model, n_atoms=64, scale=0.72):
    """Build a deterministic 3D cluster with controllable density inside a skin."""
    cutoff = float(model.info["r_max"])
    rng = np.random.default_rng(2026)
    grid = np.mgrid[0:4, 0:4, 0:4].reshape(3, -1).T.astype(np.float64)
    pos = (cutoff * (grid * scale + rng.uniform(-0.04, 0.04, size=grid.shape))).astype(
        model.dtype
    )
    types = np.arange(n_atoms, dtype=np.int32) % model.info["num_types"]
    angles = 0.27 * np.arange(n_atoms, dtype=np.float64)
    spin = np.stack(
        (np.cos(angles), np.sin(angles), 0.25 * np.sin(2 * angles)), axis=1
    ).astype(model.dtype)
    spin[::7] = 0.0
    return {"pos": pos, "atom_types": types, "spin": spin}


@pytest.mark.skipif(BACKEND != "cuda", reason="CUDA bucketed graph evaluation")
@pytest.mark.parametrize("case", BUCKET_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_bucketed_cuda_graph_growing_shrinking_and_density_spike(
    build_case, prepare_artifact, case, dtype
):
    fixture = build_case(case, dtype)
    artifact, library = prepare_artifact(fixture)
    with _open(fixture, artifact, library, "direct", tile_atoms=16) as model:
        _m4_library(model)
        cutoff = float(model.info["r_max"])
        owners = 48
        data = _cluster_system(model, n_atoms=64, scale=0.72)
        tolerance = 1e-11 if dtype == "float64" else 5e-5
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            # Large skin accommodates both thermal crossings and >1.5x density spike.
            arrays = _device_arrays(data, owners, cutoff * 3.0)
            x = arrays[0]
            graph_desc = _neighbor_descriptor(arrays, owners, stream, graph=True)
            opts = model._opts(model.tile_atoms, None, graph_desc.num_atoms)

            def eval_and_verify(pos_np):
                cur = {**data, "pos": pos_np.astype(model.dtype)}
                x.copy_(torch.from_numpy(cur["pos"]), non_blocking=True)
                out = _evaluate_neighbors(model, graph_desc, opts)
                got = _copy_cuda_results(out, model, graph_desc, opts)
                ref = _partition(model, cur, owners, global_atoms=len(cur["pos"]))
                centers, _ = _with_edges(cur, model, owners)["edge_index"]
                counts = (
                    np.bincount(centers // opts.tile_atoms, minlength=1)
                    if centers.size
                    else np.array([0])
                )
                assert out.diagnostics.peak_tile_edges == int(counts.max())
                for name, value in got.items():
                    scale = np.maximum(1.0, np.abs(ref[name]))
                    scaled = np.abs(value - ref[name]) / scale
                    idx = np.unravel_index(int(np.argmax(scaled)), scaled.shape)
                    err = float(scaled[idx])
                    lim = (
                        tolerance
                        if dtype == "float64"
                        else (1e-4 if name == "atom_virial" else tolerance)
                    )
                    assert err <= lim if dtype == "float64" else err < lim, (
                        f"{case.name} {dtype} {name}: max_scaled_err={err:.6g} "
                        f"at {idx} got={value[idx]} ref={ref[name][idx]}"
                    )
                return int(out.cuda_graph_status), int(out.diagnostics.peak_tile_edges)

            # Warmup -> Capture -> Replay on baseline positions
            s0, e0 = eval_and_verify(data["pos"])
            s1, e1 = eval_and_verify(data["pos"])
            assert s0 == 0 and s1 == 1 and e0 == e1

            # Thermal growing AND shrinking edge counts within the 256-edge bucket:
            # EVERY step MUST replay (status == 2) with zero recaptures and strict parity!
            replays = 0
            edge_history = [e0]
            for factor in (0.985, 0.965, 0.950, 0.975, 1.015, 1.035, 1.000):
                st, ec = eval_and_verify(data["pos"] * factor)
                edge_history.append(ec)
                assert st == 2, (
                    f"Expected bucketed CUDA Graph replay (2) for {case.name} {dtype} "
                    f"at factor={factor} (edges={ec}, base={e0}), got {st}"
                )
                replays += 1
            assert replays == 7
            # Confirm we actually exercised BOTH growing and shrinking active edge counts
            assert max(edge_history) > e0 and min(edge_history) < e0

            # Sudden density spike (>2.2x edge jump past 256-edge bucket & 1.125x capacity)
            # at physical bond spacing (>= 0.50 * r_max):
            # Must dynamically reallocate NeighborTopology, reset graph safely, and match eager.
            compressed = data["pos"] * 0.78
            st_spike, e_spike = eval_and_verify(compressed)
            assert e_spike > int(
                e0 * 1.5
            ), f"Spike did not jump >1.5x: {e0} -> {e_spike}"
            # Re-warm -> capture -> replay at the spiked density
            if st_spike == 0:
                st_cap, _ = eval_and_verify(compressed)
                assert st_cap == 1
            st_rep, _ = eval_and_verify(compressed * 1.005)
            assert st_rep == 2


def test_tile_atoms_uint16_guardrail(build_case, prepare_artifact):
    """Ensure tile_atoms > 65535 is rejected cleanly before execution."""
    fixture = build_case(BUCKET_CASES[0], "float32")
    artifact, library = prepare_artifact(fixture)
    with pytest.raises(Exception, match="65535"):
        with _open(fixture, artifact, library, "direct", tile_atoms=65536) as model:
            model.evaluate(
                {
                    "pos": np.zeros((2, 3), dtype=np.float32),
                    "atom_types": np.zeros(2, dtype=np.int32),
                    "edge_index": np.array([[0], [1]], dtype=np.int64),
                }
            )
