"""Adversarial stress & limit tests for native SpinGNN2 CUDA/CPU/JIT/LAMMPS."""

import ctypes as ct
import json
import os
from pathlib import Path

from ase import Atoms
from ase.neighborlist import neighbor_list
import numpy as np
import pytest
import torch
import yaml

from nequip.model import model_from_config
from nequip.utils import Config
from nequip.utils._global_options import global_options_context
from spingnn2.export import export_model
from spingnn2.jit import prepare_jit_artifact
from spingnn2.native import NativeModel
from test_contract_oracle import EXAMPLES, REPO
from test_lammps_native_neighbors import (
    _GraphNeighbors,
    _copy_cuda_results,
    _device_arrays,
    _evaluate_neighbors,
    _m4_library,
    _neighbor_descriptor,
    _partition,
)
from test_native_engine import BACKEND

_ELEMENTS = ["Fe", "Cr", "Ni", "Co", "Mn", "O", "I", "Cl", "Br", "S", "Se", "Te"]


def _native_library():
    suffix = (
        "cpu"
        if BACKEND == "cpu"
        else "cuda-" + os.environ.get("SPINGNN2_ARCH", "sm120")
    )
    build = REPO / "validation" / "native" / ("build-" + suffix)
    library = Path(os.environ.get("SPINGNN2_LIBRARY", build / "libspingnn2.so"))
    if not library.is_file():
        pytest.skip(f"Native library not found: {library}")
    return library


def _active_cuda_arch():
    if BACKEND != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm_{10 * major + minor}"


def _make_config(
    family,
    dtype,
    *,
    mul=5,
    mlp_dims=(33,),
    l_max=2,
    num_layers=2,
    num_types=2,
    extra=None,
):
    cfg = yaml.safe_load((REPO / "example" / EXAMPLES[family]).read_text())
    cfg["chemical_symbol_to_type"] = {_ELEMENTS[i]: i for i in range(num_types)}
    cfg["num_types"] = num_types
    for key, val in list(cfg.items()):
        if key.endswith("_mlp_latent_dimensions") and val:
            cfg[key] = list(mlp_dims)
    cfg.update(
        default_dtype=dtype,
        allow_tf32=False,
        float32_matmul_precision="highest",
        spingnn_compile={"enabled": False},
        train_autocast={"enabled": False},
        avg_num_neighbors=4.25,
        env_embed_multiplicity=mul,
        l_max=l_max,
        allegro_onlyr_num_layers=num_layers,
        allegro_soc_on_num_layers=num_layers,
        allegro_nosoc_num_layers=num_layers,
        e2ip_latent_channels=max(2, min(4, mul)),
        per_species_rescale_scales=[0.85 + 0.03 * i for i in range(num_types)],
        per_species_rescale_shifts=[
            0.02 * (i - num_types / 2) for i in range(num_types)
        ],
        global_rescale_scale=0.125,
    )
    if "l_max_spin" in cfg:
        cfg["l_max_spin"] = min(l_max, cfg["l_max_spin"])
    if family == "born":
        cfg["born_scale_list"] = [
            (-1.0 if i % 2 == 0 else 1.0) * (1.5 + 0.5 * i) for i in range(num_types)
        ]
    if extra:
        cfg.update(extra)
    return cfg


def _make_sample(cfg, n_atoms=7, seed=183):
    rng = np.random.default_rng(seed)
    cell = np.array([[9.5, 0.0, 0.0], [0.7, 10.5, 0.0], [0.2, -0.5, 11.0]])
    frac = rng.uniform(0.05, 0.45, size=(n_atoms, 3))
    pos = frac @ cell
    centers, neighbors, shifts = neighbor_list(
        "ijS", Atoms(f"H{n_atoms}", positions=pos, cell=cell, pbc=True), cfg["r_max"]
    )
    num_types = len(cfg["chemical_symbol_to_type"])
    atom_types = np.arange(n_atoms, dtype=np.int64) % num_types
    dtype = np.dtype(cfg["default_dtype"])
    angles = np.linspace(0.2, 2.7, n_atoms)
    spin = np.stack(
        (np.cos(angles), np.sin(angles), 0.3 * np.sin(2 * angles)), axis=1
    ).astype(dtype)
    if n_atoms >= 3:
        spin[2] = 0.0
    return {
        "pos": pos.astype(dtype),
        "cell": cell[None, :, :].astype(dtype),
        "edge_index": np.stack((centers, neighbors)).astype(np.int64),
        "edge_cell_shift": shifts.astype(dtype),
        "atom_types": atom_types,
        "pbc": np.ones((1, 3), dtype=bool),
        "batch": np.zeros(n_atoms, dtype=np.int64),
        "spin": spin,
    }


def _fields_for(family):
    fields = ["total_energy", "atomic_energy", "forces", "virial", "atom_virial"]
    if family in ("nosoc", "soc", "soc-e2ip"):
        fields.extend(["spin_forces", "spin_forces_vert"])
    if family == "born":
        fields.append("born_effective_charges")
    return fields


def _assert_parity(got, ref, dtype, fields, *, label=""):
    tol = 1e-10 if dtype == "float64" else 5e-5
    for key in fields:
        actual = got[key]
        expected = ref[key]
        assert actual.shape == expected.shape, f"{label} {key} shape mismatch"
        assert np.isfinite(actual).all(), f"{label} {key} contains non-finite values"
        scale = np.maximum(1.0, np.abs(expected))
        err = float(np.max(np.abs(actual - expected) / scale, initial=0.0))
        lim = tol if dtype == "float64" else (1e-4 if key == "atom_virial" else tol)
        assert err <= lim, f"{label} {key}: max_scaled_err={err:.3e} > {lim:.3e}"


def _build_and_verify_shape(tmp_path, name, family, dtype, **kw):
    library = _native_library()
    cfg = _make_config(family, dtype, **kw)
    n_atoms = max(7, kw.get("num_types", 2) + 2)
    data = _make_sample(cfg, n_atoms=n_atoms)
    stem = tmp_path / name / dtype
    stem.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if BACKEND == "cuda" else "cpu"
    with global_options_context(cfg):
        torch.manual_seed(1793)
        model = model_from_config(Config.from_dict(cfg), initialize=True).eval()
        manifest, _ = export_model(model, stem)
        model.to(device)
        eager_out = model(
            {k: torch.from_numpy(v.copy()).to(device) for k, v in data.items()}
        )
        eager = {k: v.detach().cpu().numpy() for k, v in eager_out.items()}

    art_path = prepare_jit_artifact(
        manifest,
        cache_dir=tmp_path / name / "jit",
        backend=BACKEND,
        precision=dtype,
        gpu_arch=_active_cuda_arch(),
        library=library,
    )
    fields = _fields_for(family)
    with NativeModel.load(
        manifest,
        precision=dtype,
        backend=BACKEND,
        library=library,
        artifact=art_path,
        execution_mode="direct",
        tile_atoms=None,
    ) as nm_dir:
        got_dir = nm_dir.evaluate(data)
    with NativeModel.load(
        manifest,
        precision=dtype,
        backend=BACKEND,
        library=library,
        execution_mode="generic",
        tile_atoms=None,
    ) as nm_gen:
        got_gen = nm_gen.evaluate(data)

    _assert_parity(got_dir, eager, dtype, fields, label=f"{name}:direct")
    _assert_parity(got_gen, eager, dtype, fields, label=f"{name}:generic")

    if BACKEND == "cuda" and dtype == "float32":
        with NativeModel.load(
            manifest,
            precision="float32",
            backend="cuda",
            library=library,
            artifact=art_path,
            execution_mode="direct",
            allow_tf32=True,
            tile_atoms=None,
        ) as nm_tf:
            got_tf = nm_tf.evaluate(data)
        for key in fields:
            assert np.isfinite(got_tf[key]).all(), f"{name}:tf32 non-finite {key}"
    return manifest, art_path


_SHAPE_CASES = [
    # Non-divisible-by-4 and wide multiplicities + odd MLP widths (float4 & cublasLt safety)
    ("nosoc_mul1_odd33", "nosoc", "float32", dict(mul=1, mlp_dims=(33,))),
    ("nosoc_mul3_odd33_65", "nosoc", "float32", dict(mul=3, mlp_dims=(33, 65))),
    ("nosoc_mul5_odd65", "nosoc", "float32", dict(mul=5, mlp_dims=(65,))),
    ("nosoc_mul7_odd33", "nosoc", "float32", dict(mul=7, mlp_dims=(33,))),
    ("nosoc_mul13_odd65", "nosoc", "float32", dict(mul=13, mlp_dims=(65,))),
    ("nosoc_mul17_w96", "nosoc", "float32", dict(mul=17, mlp_dims=(96,))),
    ("nosoc_mul24_w128", "nosoc", "float32", dict(mul=24, mlp_dims=(128,))),
    ("nosoc_mul32_w128", "nosoc", "float32", dict(mul=32, mlp_dims=(128,))),
    ("nosoc_mul48_w128", "nosoc", "float32", dict(mul=48, mlp_dims=(128,))),
    ("nosoc_mul64_w256", "nosoc", "float32", dict(mul=64, mlp_dims=(256,))),
    ("nosoc_mul64_fp64", "nosoc", "float64", dict(mul=64, mlp_dims=(65,))),
    ("soc_mul1_odd33", "soc", "float32", dict(mul=1, mlp_dims=(33,))),
    ("soc_mul5_odd65_33", "soc", "float32", dict(mul=5, mlp_dims=(65, 33))),
    ("soc_mul17_odd65", "soc", "float32", dict(mul=17, mlp_dims=(65,))),
    ("soc_mul32_w128", "soc", "float32", dict(mul=32, mlp_dims=(128,))),
    ("soc_mul48_w128", "soc", "float32", dict(mul=48, mlp_dims=(128,))),
    ("soc_mul64_w128", "soc", "float32", dict(mul=64, mlp_dims=(128,))),
    # l_max = 1, 3 (including mul=16 l_max=3 non-unrolled heavy path)
    ("nosoc_lmax1", "nosoc", "float32", dict(mul=7, mlp_dims=(33,), l_max=1)),
    ("soc_lmax1", "soc", "float32", dict(mul=5, mlp_dims=(33,), l_max=1)),
    ("nosoc_lmax3_mul16", "nosoc", "float32", dict(mul=16, mlp_dims=(65,), l_max=3)),
    ("nosoc_lmax3_mul32", "nosoc", "float32", dict(mul=32, mlp_dims=(65,), l_max=3)),
    ("soc_lmax3_mul32", "soc", "float32", dict(mul=32, mlp_dims=(65,), l_max=3)),
    # num_layers = 1, 3, 4
    ("nosoc_L1", "nosoc", "float32", dict(mul=7, mlp_dims=(33,), num_layers=1)),
    ("nosoc_L3", "nosoc", "float32", dict(mul=7, mlp_dims=(33,), num_layers=3)),
    (
        "nosoc_L4",
        "nosoc",
        "float32",
        dict(mul=7, mlp_dims=(33,), num_layers=4, extra={"avg_num_neighbors": 6.0}),
    ),
    ("soc_L1", "soc", "float32", dict(mul=5, mlp_dims=(33,), num_layers=1)),
    ("soc_L4", "soc", "float32", dict(mul=5, mlp_dims=(33,), num_layers=4)),
    # num_types = 1, 4, 8, 12 (with per_edge_species_scale & ledge_cutoff)
    ("nosoc_T1", "nosoc", "float32", dict(mul=5, mlp_dims=(33,), num_types=1)),
    (
        "nosoc_T4_scale",
        "nosoc",
        "float32",
        dict(
            mul=5,
            mlp_dims=(33,),
            num_types=4,
            extra={"per_edge_species_scale": True, "ledge_cutoff": True},
        ),
    ),
    ("nosoc_T8", "nosoc", "float32", dict(mul=5, mlp_dims=(33,), num_types=8)),
    (
        "nosoc_T12_scale",
        "nosoc",
        "float32",
        dict(
            mul=5,
            mlp_dims=(33,),
            num_types=12,
            extra={"per_edge_species_scale": True},
        ),
    ),
    # DREAM (born), e2IP, soc-e2IP with odd widths/multiplicities
    (
        "born_odd_T4",
        "born",
        "float32",
        dict(mul=7, mlp_dims=(33, 65), num_types=4),
    ),
    ("e2ip_odd_T3", "e2ip", "float32", dict(mul=7, mlp_dims=(33,), num_types=3)),
    (
        "soc_e2ip_odd_T3",
        "soc-e2ip",
        "float32",
        dict(mul=7, mlp_dims=(33,), num_types=3),
    ),
]


@pytest.mark.parametrize(
    ("name", "family", "dtype", "kw"),
    _SHAPE_CASES,
    ids=[c[0] for c in _SHAPE_CASES],
)
def test_adversarial_network_shapes(tmp_path, name, family, dtype, kw):
    """Verify odd widths, extreme mul=1..64, l_max=1..3, L=1..4, types=1..12."""
    _build_and_verify_shape(tmp_path, name, family, dtype, **kw)


@pytest.mark.skipif(BACKEND != "cuda", reason="CUDA shared-memory opt-in & fallback")
@pytest.mark.parametrize(
    ("name", "mul", "min_bytes", "max_bytes"),
    [
        # nosoc l_max=2 allegro_nosoc has d2=18 -> 18 * 400 * 8 = 57,600 B in (48 KiB, 96 KiB]
        ("smem_optin_56k", 400, 48 * 1024 + 1, 96 * 1024),
        # nosoc l_max=2 allegro_nosoc has d2=18 -> 18 * 760 * 8 = 109,440 B > 99 KiB (global fallback!)
        ("smem_global_fallback_109k", 760, 101376 + 1, 1024 * 1024),
    ],
)
def test_shared_memory_optin_and_global_fallback(
    tmp_path, name, mul, min_bytes, max_bytes
):
    """Verify >48 KiB dynamic opt-in smem and >96 KiB global-env fallback never crash."""
    manifest, _ = _build_and_verify_shape(
        tmp_path,
        name,
        "nosoc",
        "float64",
        mul=mul,
        mlp_dims=(8,),
        l_max=2,
        num_layers=2,
        num_types=2,
    )
    spec = json.loads(manifest.read_text())
    max_ew_bytes = 0
    for layer in spec["model"]["layers"]:
        if "backbone" in layer:
            for bl in layer["backbone"]["layers"]:
                tp = bl["tensor_product"]
                ew_b = int(tp["mul_out"]) * int(tp["dim_in2"]) * 8
                max_ew_bytes = max(max_ew_bytes, ew_b)
    assert min_bytes <= max_ew_bytes <= max_bytes


@pytest.mark.skipif(BACKEND != "cuda", reason="CUDA IrrepLinear >16M fallback test")
def test_irreplinear_large_tile_16m_fallback(tmp_path):
    """Force IrrepLinear::apply >16M per-group fallback and verify strict parity."""
    library = _native_library()
    models_idx = os.environ.get("SPINGNN2_LAMMPS_MODELS")
    manifest = art = None
    if models_idx and Path(models_idx).is_file():
        idx = json.loads(Path(models_idx).read_text())
        if "soc-e2ip" in idx and "float32" in idx["soc-e2ip"]:
            cand_m = Path(idx["soc-e2ip"]["float32"]["contract"])
            cand_a = Path(idx["soc-e2ip"]["float32"]["manifest"])
            if cand_m.is_file() and cand_a.is_file():
                manifest, art = cand_m, cand_a
    if manifest is None:
        cfg = _make_config(
            "e2ip",
            "float32",
            mul=24,
            mlp_dims=(16,),
            l_max=2,
            num_layers=2,
            num_types=2,
        )
        stem = tmp_path / "e2ip_16m" / "float32"
        stem.parent.mkdir(parents=True, exist_ok=True)
        with global_options_context(cfg):
            torch.manual_seed(1793)
            model = model_from_config(Config.from_dict(cfg), initialize=True).eval()
            manifest, _ = export_model(model, stem)
        art = prepare_jit_artifact(
            manifest,
            cache_dir=tmp_path / "e2ip_16m" / "jit",
            backend="cuda",
            precision="float32",
            gpu_arch=_active_cuda_arch(),
            library=library,
        )
    with NativeModel.load(
        manifest,
        precision="float32",
        backend="cuda",
        library=library,
        artifact=art,
        execution_mode="direct",
        tile_atoms=32,
    ) as nm_big, NativeModel.load(
        manifest,
        precision="float32",
        backend="cuda",
        library=library,
        artifact=art,
        execution_mode="direct",
        tile_atoms=1,
    ) as nm_small:
        cutoff = float(nm_big.info["r_max"])
        n_atoms = 32
        rng = np.random.default_rng(77)
        grid = np.mgrid[0:4, 0:4, 0:2].reshape(3, -1).T.astype(np.float64)
        pos = (
            cutoff * (grid * 0.42 + rng.uniform(-0.02, 0.02, size=grid.shape))
        ).astype(np.float32)
        types = np.arange(n_atoms, dtype=np.int64) % nm_big.info["num_types"]
        spin = np.tile(np.array([[0.6, 0.8, 0.0]], dtype=np.float32), (n_atoms, 1))
        cell = np.diag([cutoff * 4.0] * 3).astype(np.float32)
        ci, cj, _ = neighbor_list(
            "ijS", Atoms(f"H{n_atoms}", positions=pos, cell=cell, pbc=False), cutoff
        )
        assert len(ci) >= 250
        data = {
            "pos": pos,
            "atom_types": types,
            "spin": spin,
            "edge_index": np.stack((ci, cj)).astype(np.int64),
        }
        got_big = nm_big.evaluate(data)
        got_small = nm_small.evaluate(data)
        fields = ["total_energy", "atomic_energy", "forces", "atom_virial"]
        if nm_big.info["magnetic"]:
            fields.append("spin_forces")
        _assert_parity(
            got_big,
            got_small,
            "float32",
            fields,
            label="irreplinear_16m",
        )


@pytest.mark.parametrize("family", ("nosoc", "soc", "born"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_degenerate_graphs_isolated_single_and_empty_owned(tmp_path, family, dtype):
    """Test E=0 (N=1, N=8), N=1 periodic self-image, and N_owned=0 evaluate()."""
    library = _native_library()
    cfg = _make_config(family, dtype, mul=5, mlp_dims=(16,), num_layers=2, num_types=2)
    stem = tmp_path / f"{family}_{dtype}" / dtype
    stem.parent.mkdir(parents=True, exist_ok=True)
    with global_options_context(cfg):
        torch.manual_seed(1793)
        model = model_from_config(Config.from_dict(cfg), initialize=True).eval()
        manifest, _ = export_model(model, stem)
    art = prepare_jit_artifact(
        manifest,
        cache_dir=tmp_path / f"{family}_{dtype}" / "jit",
        backend=BACKEND,
        precision=dtype,
        gpu_arch=_active_cuda_arch(),
        library=library,
    )
    np_dtype = np.float32 if dtype == "float32" else np.float64
    with NativeModel.load(
        manifest,
        precision=dtype,
        backend=BACKEND,
        library=library,
        artifact=art,
        execution_mode="direct",
        tile_atoms=8,
    ) as nm_dir, NativeModel.load(
        manifest,
        precision=dtype,
        backend=BACKEND,
        library=library,
        execution_mode="generic",
        tile_atoms=8,
    ) as nm_gen:
        cutoff = float(nm_dir.info["r_max"])
        # 1. E=0 isolated atoms (N=1 and N=8)
        for n_iso in (1, 8):
            pos = np.zeros((n_iso, 3), dtype=np_dtype)
            pos[:, 0] = np.arange(n_iso) * (cutoff * 3.0)
            data = {
                "pos": pos,
                "atom_types": np.zeros(n_iso, dtype=np.int64),
                "edge_index": np.zeros((2, 0), dtype=np.int64),
                "spin": np.tile(
                    np.array([[0.0, 0.0, 1.0]], dtype=np_dtype), (n_iso, 1)
                ),
            }
            out_d = nm_dir.evaluate(data)
            out_g = nm_gen.evaluate(data)
            _assert_parity(
                out_d, out_g, dtype, ["total_energy", "atomic_energy", "forces"]
            )
            assert np.allclose(out_d["forces"], 0.0)

        # 2. N=1 periodic self-image (E > 0)
        cell = np.diag([cutoff * 0.75] * 3).astype(np_dtype)
        pos1 = np.zeros((1, 3), dtype=np_dtype)
        ci, cj, sh = neighbor_list(
            "ijS", Atoms("H", positions=pos1, cell=cell, pbc=True), cutoff
        )
        assert len(ci) > 0
        data_self = {
            "pos": pos1,
            "cell": cell[None, :, :],
            "edge_index": np.stack((ci, cj)).astype(np.int64),
            "edge_cell_shift": sh.astype(np_dtype),
            "atom_types": np.zeros(1, dtype=np.int64),
            "pbc": np.ones((1, 3), dtype=bool),
            "batch": np.zeros(1, dtype=np.int64),
            "spin": np.array([[0.6, 0.8, 0.0]], dtype=np_dtype),
        }
        out_sd = nm_dir.evaluate(data_self)
        out_sg = nm_gen.evaluate(data_self)
        _assert_parity(out_sd, out_sg, dtype, _fields_for(family), label="N1_self")


@pytest.mark.skipif(BACKEND != "cuda", reason="CUDA evaluate_neighbors boundary tests")
@pytest.mark.parametrize("family", ("nosoc", "soc", "born"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_cuda_neighbors_zero_slot_capacity_and_empty_rank(tmp_path, family, dtype):
    """Test E=0 with neighbor_slot_capacity=0, empty MPI rank (nlocal=0), and owner->0 reset."""
    library = _native_library()
    cfg = _make_config(family, dtype, mul=5, mlp_dims=(16,), num_layers=2, num_types=2)
    stem = tmp_path / f"neigh_{family}_{dtype}" / dtype
    stem.parent.mkdir(parents=True, exist_ok=True)
    with global_options_context(cfg):
        torch.manual_seed(1793)
        model = model_from_config(Config.from_dict(cfg), initialize=True).eval()
        manifest, _ = export_model(model, stem)
    art = prepare_jit_artifact(
        manifest,
        cache_dir=tmp_path / f"neigh_{family}_{dtype}" / "jit",
        backend="cuda",
        precision=dtype,
        gpu_arch=_active_cuda_arch(),
        library=library,
    )
    np_dtype = np.float32 if dtype == "float32" else np.float64
    with NativeModel.load(
        manifest,
        precision=dtype,
        backend="cuda",
        library=library,
        artifact=art,
        execution_mode="direct",
        tile_atoms=16,
    ) as nm:
        _m4_library(nm)
        cutoff = float(nm.info["r_max"])
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # 1. Isolated atoms with neighbor_slot_capacity == 0 AND zero neighbor strides
            n_iso = 8
            pos = np.zeros((n_iso, 3), dtype=np_dtype)
            pos[:, 0] = np.arange(n_iso) * (cutoff * 3.0)
            cur = {
                "pos": pos,
                "atom_types": np.zeros(n_iso, dtype=np.int32),
                "spin": np.tile(
                    np.array([[0.0, 0.0, 1.0]], dtype=np_dtype), (n_iso, 1)
                ),
            }
            arrays = _device_arrays(cur, n_iso, cutoff * 1.5)
            desc = _neighbor_descriptor(arrays, n_iso, stream, graph=True)
            desc.neighbor_slot_capacity = 0
            desc.firstneigh = 0
            desc.neighbor_atom_stride = 0
            desc.neighbor_slot_stride = 0
            opts = nm._opts(nm.tile_atoms, None, desc.num_atoms)
            r0 = _evaluate_neighbors(nm, desc, opts)
            r1 = _evaluate_neighbors(nm, desc, opts)
            r2 = _evaluate_neighbors(nm, desc, opts)
            assert (
                r0.cuda_graph_status,
                r1.cuda_graph_status,
                r2.cuda_graph_status,
            ) == (
                0,
                1,
                2,
            )
            got = _copy_cuda_results(r2, nm, desc, opts)
            assert all(np.isfinite(v).all() for v in got.values())
            assert np.allclose(got["forces"], 0.0)

            # 2. Non-empty step followed by empty rank (nlocal=0, nghost=0 with non-null empty spin view)
            close_pos = np.zeros((4, 3), dtype=np_dtype)
            close_pos[:, 0] = np.arange(4) * (cutoff * 0.4)
            cur_close = {
                "pos": close_pos,
                "atom_types": np.zeros(4, dtype=np.int32),
                "spin": np.tile(np.array([[0.6, 0.8, 0.0]], dtype=np_dtype), (4, 1)),
            }
            arr_close = _device_arrays(cur_close, 4, cutoff * 1.5)
            d_close = _neighbor_descriptor(arr_close, 4, stream, graph=True)
            d_close.topology_epoch = 2
            opts_close = nm._opts(nm.tile_atoms, None, 16)
            rc = _evaluate_neighbors(nm, d_close, opts_close)
            assert rc.diagnostics.peak_tile_edges > 0

            # Now transition to empty MPI rank (0 owned, 0 ghosts, non-null empty spin ptr with 0 strides!)
            empty_sp = torch.empty(
                (0, 4),
                dtype=torch.float32 if dtype == "float32" else torch.float64,
                device="cuda",
            )
            tmap = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
            d_empty = _GraphNeighbors()
            d_empty.struct_size = ct.sizeof(_GraphNeighbors)
            d_empty.num_atoms = 0
            d_empty.num_owned = 0
            d_empty.num_global = 16
            d_empty.positions = 0
            d_empty.spins = empty_sp.data_ptr() if nm.info["magnetic"] else 0
            d_empty.position_precision = 32 if dtype == "float32" else 64
            d_empty.position_atom_stride = 0
            d_empty.position_component_stride = 0
            d_empty.spin_atom_stride = 0
            d_empty.spin_component_stride = 0
            d_empty.types = 0
            d_empty.type_map = tmap.data_ptr()
            d_empty.type_map_size = 2
            d_empty.neighbor_mask = 0x3FFFFFFF
            d_empty.neighbor_slot_capacity = 0
            d_empty.topology_epoch = 3
            d_empty.lattice_epoch = 0
            d_empty.stream = stream.cuda_stream
            d_empty.flags = 1
            re_empty = _evaluate_neighbors(nm, d_empty, opts_close)
            assert re_empty.diagnostics.peak_tile_edges == 0


@pytest.mark.skipif(BACKEND != "cuda", reason="CUDA bucket boundary & VRAM leak test")
@pytest.mark.parametrize("family", ("nosoc", "soc"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_cuda_graph_exact_256_boundaries_5x_spike_and_vram_leak(
    tmp_path, family, dtype
):
    """Test exact E_tile=0,1,128,255,256,257, 5.25x spike, poisoned VRAM, and 500-step leak."""
    library = _native_library()
    cfg = _make_config(
        family,
        dtype,
        mul=5,
        mlp_dims=(16,),
        num_layers=2,
        num_types=2,
        extra={"global_rescale_scale": 0.05, "avg_num_neighbors": 16.0},
    )
    stem = tmp_path / f"bkt_{family}_{dtype}" / dtype
    stem.parent.mkdir(parents=True, exist_ok=True)
    with global_options_context(cfg):
        torch.manual_seed(1793)
        model = model_from_config(Config.from_dict(cfg), initialize=True).eval()
        manifest, _ = export_model(model, stem)
    art = prepare_jit_artifact(
        manifest,
        cache_dir=tmp_path / f"bkt_{family}_{dtype}" / "jit",
        backend="cuda",
        precision=dtype,
        gpu_arch=_active_cuda_arch(),
        library=library,
    )
    # Poison CUDA memory pool with NaNs before evaluation to catch uninitialized padded-edge reads
    poison = torch.full(
        (8 * 1024 * 1024,), float("nan"), dtype=torch.float32, device="cuda"
    )
    del poison
    torch.cuda.empty_cache()

    np_dtype = np.float32 if dtype == "float32" else np.float64
    with NativeModel.load(
        manifest,
        precision=dtype,
        backend="cuda",
        library=library,
        artifact=art,
        execution_mode="direct",
        tile_atoms=32,
    ) as nm:
        _m4_library(nm)
        cutoff = float(nm.info["r_max"])
        # 16 well-separated owned centers (spaced 4*cutoff apart), each with up to 32
        # ghost neighbors -> physical per-atom degree <= 32 while dialing ANY exact
        # total tile edge count K in [0, 512] across the 256-edge bucket boundary!
        n_owned = 16
        per_center = 32
        max_ghosts = n_owned * per_center
        n_atoms = n_owned + max_ghosts
        rng = np.random.default_rng(999)
        centers_pos = np.zeros((n_owned, 3), dtype=np_dtype)
        centers_pos[:, 0] = np.arange(n_owned, dtype=np_dtype) * (cutoff * 4.0)
        dirs = rng.normal(size=(max_ghosts, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        active_r = cutoff * rng.uniform(0.55, 0.85, size=(max_ghosts, 1))
        inactive_r = cutoff * np.full((max_ghosts, 1), 1.8)
        types = np.arange(n_atoms, dtype=np.int32) % nm.info["num_types"]
        spin = np.zeros((n_atoms, 3), dtype=np_dtype)
        spin[:, 0] = np.cos(0.1 * np.arange(n_atoms))
        spin[:, 1] = np.sin(0.1 * np.arange(n_atoms))

        def make_pos(k_active):
            p = np.zeros((n_atoms, 3), dtype=np_dtype)
            p[:n_owned] = centers_pos
            radii = inactive_r.copy()
            # Round-robin across the 16 centers so per-atom degree stays balanced (<= 27)
            for idx_e in range(k_active):
                owner = idx_e % n_owned
                slot = idx_e // n_owned
                radii[owner * per_center + slot] = active_r[owner * per_center + slot]
            owner_idx = np.repeat(np.arange(n_owned), per_center)
            p[n_owned:] = (centers_pos[owner_idx] + dirs * radii).astype(np_dtype)
            return p

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            arrays = _device_arrays(
                {"pos": make_pos(255), "atom_types": types, "spin": spin},
                n_owned,
                cutoff * 2.2,
            )
            x = arrays[0]
            desc = _neighbor_descriptor(arrays, n_owned, stream, graph=True)
            opts = nm._opts(nm.tile_atoms, None, desc.num_atoms)
            seq = [
                0,
                0,
                1,
                1,
                128,
                255,
                256,
                257,
                257,
                256,
                255,
                1,
                0,
                80,
                80,
                420,
                420,
                421,
                80,
            ]
            fields = ["owned_energy", "forces", "atom_virial", "spin_forces"]
            for idx, k_e in enumerate(seq):
                p = make_pos(k_e)
                cur = {"pos": p, "atom_types": types, "spin": spin}
                x.copy_(torch.from_numpy(p), non_blocking=True)
                out = _evaluate_neighbors(nm, desc, opts)
                got = _copy_cuda_results(out, nm, desc, opts)
                ref = _partition(nm, cur, n_owned, global_atoms=n_atoms)
                assert out.diagnostics.peak_tile_edges == k_e
                _assert_parity(
                    got, ref, dtype, fields, label=f"{family}_{dtype}_e{k_e}"
                )
                # Check expected replays within steady bucket (128, 255, 256 after cap=256 capture at k_e=1; 421 after 420)
                if idx in (4, 5, 6, 9, 10, 11, 17):
                    assert (
                        out.cuda_graph_status == 2
                    ), f"Expected replay at step {idx} (k_e={k_e})"

            # 500-step VRAM leak check on float32 soc
            if family == "soc" and dtype == "float32":
                torch.cuda.synchronize()
                free_before, _ = torch.cuda.mem_get_info()
                for step in range(500):
                    k_e = 100 + (step % 60)
                    x.copy_(torch.from_numpy(make_pos(k_e)), non_blocking=True)
                    _evaluate_neighbors(nm, desc, opts)
                torch.cuda.synchronize()
                free_after, _ = torch.cuda.mem_get_info()
                leaked_mb = (free_before - free_after) / (1024**2)
                assert leaked_mb < 2.0, f"Detected VRAM leak: {leaked_mb:.2f} MB"


def _parse_lammps_thermo(stdout, ncols):
    rows = []
    in_table = False
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("Step "):
            in_table = True
            continue
        if in_table:
            if s.startswith("Loop time"):
                in_table = False
                continue
            parts = s.split()
            if len(parts) == ncols and parts[0].isdigit():
                rows.append([float(x) for x in parts])
    return rows


def test_lammps_triclinic_isolated_nve500_highT_and_empty_mpi_rank(
    lmp_run, m4_backend, m4_models, tmp_path
):
    """LAMMPS stress: triclinic box, isolated E=0, high-T crossings, 500-step NVE, empty MPI rank."""
    if m4_backend != "cuda":
        pytest.skip("CUDA LAMMPS stress lane")
    library = _native_library()
    arch = _active_cuda_arch()
    # Export a gentle small-scale nosoc model for stable 500-step NVE & triclinic MD
    cfg = _make_config(
        "nosoc",
        "float32",
        mul=8,
        mlp_dims=(32,),
        num_layers=2,
        num_types=2,
        extra={"global_rescale_scale": 0.01},
    )
    stem = tmp_path / "lmp_smooth" / "nosoc"
    stem.parent.mkdir(parents=True, exist_ok=True)
    with global_options_context(cfg):
        torch.manual_seed(1793)
        model = model_from_config(Config.from_dict(cfg), initialize=True).eval()
        manifest, _ = export_model(model, stem)
    art = prepare_jit_artifact(
        manifest,
        cache_dir=tmp_path / "lmp_smooth" / "jit",
        backend="cuda",
        precision="float32",
        gpu_arch=arch,
        library=library,
    )

    # 1. Triclinic box parity (cuda_graph yes vs no)
    tri_etot = {}
    for cg in ("no", "yes"):
        out = lmp_run(
            f"""
units metal
atom_style spin
boundary p p p
lattice sc 4.2
region box prism 0 4 0 4 0 4 0.8 0.5 -0.4
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.5 12345
mass * 55.845
set atom * spin/random 42 1.0
pair_style spingnn2 precision float32 cuda_graph {cg}
pair_coeff * * "{manifest}" "{art}" 0 1
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
velocity all create 300.0 99123
fix 1 all nve
timestep 0.0005
thermo 1
thermo_style custom step pe ke etotal temp press
run 10
""",
            gpu_aware=False,
        )
        rows = _parse_lammps_thermo(out, 6)
        assert len(rows) == 11
        tri_etot[cg] = [r[3] for r in rows]
    assert max(abs(a - b) for a, b in zip(tri_etot["no"], tri_etot["yes"])) < 1e-4

    # 2. Isolated atoms (E=0) with cuda_graph yes
    lmp_run(
        f"""
units metal
atom_style spin
boundary f f f
region box block 0 100 0 100 0 100
create_box 2 box
create_atoms 1 single 10 10 10
create_atoms 2 single 50 50 50
create_atoms 1 single 90 90 90
mass * 55.845
set atom * spin/random 42 1.0
pair_style spingnn2 precision float32 cuda_graph yes
pair_coeff * * "{manifest}" "{art}" 0 1
neighbor 1.0 bin
fix 1 all nve
timestep 0.001
thermo 1
run 5
""",
        gpu_aware=False,
    )

    # 3. 500-step NVE energy conservation with cuda_graph yes
    out_nve = lmp_run(
        f"""
units metal
atom_style spin
boundary p p p
lattice sc 4.2
region box block 0 4 0 4 0 4
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.5 12345
mass * 55.845
set atom * spin/random 42 1.0
pair_style spingnn2 precision float32 cuda_graph yes
pair_coeff * * "{manifest}" "{art}" 0 1
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
velocity all create 300.0 77711
fix 1 all nve
timestep 0.0005
thermo 50
thermo_style custom step pe ke etotal temp
run 500
""",
        gpu_aware=False,
    )
    nve_rows = _parse_lammps_thermo(out_nve, 5)
    assert len(nve_rows) == 11
    etots = [r[3] for r in nve_rows]
    assert np.isfinite(etots).all()
    assert (max(etots) - min(etots)) / 64.0 < 1e-3

    # 4. High-T 2000 K rapid cutoff crossings
    out_ht = lmp_run(
        f"""
units metal
atom_style spin
boundary p p p
lattice sc 4.2
region box block 0 4 0 4 0 4
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.5 12345
mass * 55.845
set atom * spin/random 42 1.0
pair_style spingnn2 precision float32 cuda_graph yes
pair_coeff * * "{manifest}" "{art}" 0 1
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
velocity all create 2000.0 55511
fix 1 all nve
timestep 0.0005
thermo 20
thermo_style custom step pe ke etotal temp
run 100
""",
        gpu_aware=False,
    )
    ht_rows = _parse_lammps_thermo(out_ht, 5)
    assert len(ht_rows) == 6
    assert np.isfinite([r[3] for r in ht_rows]).all()

    # 5. Empty MPI rank (nlocal=0 on rank 1) matching 1-rank reference (if >=2 GPUs or shared allowed)
    spec_born = m4_models["born"]["float32"]
    for family, style, extra_init, c_path, a_path in (
        ("nosoc", "spin", "set atom * spin/random 42 1.0", manifest, art),
        ("born", "bec", "", spec_born["contract"], spec_born["manifest"]),
    ):
        for boundary, x_hi in (("f f f", 15.0), ("p p p", 19.5)):
            pes = {}
            for ranks in (1, 2):
                out_m = lmp_run(
                    f"""
units metal
atom_style {style}
boundary {boundary}
processors {ranks} 1 1
region box block 0 40 0 20 0 20
create_box 2 box
lattice sc 4.5
region left block 2 {x_hi} 2 12 2 12 units box
create_atoms 1 region left
set type 1 type/fraction 2 0.5 12345
mass * 55.845
{extra_init}
pair_style spingnn2 precision float32 cuda_graph yes
pair_coeff * * "{c_path}" "{a_path}" 0 1
neighbor 1.0 bin
fix 1 all nve
timestep 0.00001
thermo 1
thermo_style custom step pe ke etotal temp
run 3
""",
                    ranks=ranks,
                    gpu_aware=False,
                    allow_shared_gpu=True,
                )
                rows_m = _parse_lammps_thermo(out_m, 5)
                assert len(rows_m) == 4
                pes[ranks] = rows_m[0][1]
            assert abs(pes[1] - pes[2]) <= 1e-4 * max(
                1.0, abs(pes[1])
            ), f"{family} {boundary}: 1-rank PE={pes[1]} != 2-rank PE={pes[2]}"
