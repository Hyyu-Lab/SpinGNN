"""PackagedBackbone tests on a toy model packaged with torch.package at test time."""

import copy
import importlib
import math
import os
import sys
import textwrap

import pytest
import torch
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp
from e3nn import o3
from torch.package import PackageExporter, PackageImporter

from nequip.data import AtomicData, AtomicDataDict
from nequip.model import model_from_config
from nequip.nn import PackagedBackbone
from nequip.nn._packaged_backbone import _torch_compile_kwargs
from nequip.utils._global_options import cast_model_dtype
from nequip.utils.torch_geometric import Batch

# torch.package's exporter still pickles through TypedStorage (torch 2.11).
pytestmark = pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")

TYPE_NAMES = ["H", "C", "O"]
R_MAX = 3.0
MUL = 4

# A pure-torch imitation of a nequip>=0.7 package: ModuleDict["sole_model"] ->
# GraphModel(.model) -> ForceOutput(.func) -> sequential stages operating on
# dicts. The "allegro" stage owns ``tps`` (strided outputs) and ``latents``.
_TOY_SOURCE = textwrap.dedent(
    """
    import math

    import torch


    class EdgeEmbed(torch.nn.Module):
        def __init__(self, r_max):
            super().__init__()
            # float64 geometry, cast to float32 at the output (as packaged models do)
            self.register_buffer("_rmax_recip", torch.tensor(1.0 / r_max, dtype=torch.float64))
            self.bessel_weights = torch.nn.Parameter(
                torch.linspace(1.0, 4.0, 4, dtype=torch.float64)
            )
            self.irreps_out = {"edge_embedding": "4x0e", "edge_attrs": "1x0e+1x1o"}

        def forward(self, data):
            pos, edge_index = data["pos"], data["edge_index"]
            vec = pos[edge_index[1]] - pos[edge_index[0]]
            if "cell" in data:
                edge_batch = data["batch"][edge_index[0]]
                vec = vec + torch.einsum(
                    "ni,nij->nj", data["edge_cell_shift"], data["cell"][edge_batch]
                )
            r = vec.norm(dim=-1, keepdim=True)
            x = r * self._rmax_recip
            basis = torch.sin(self.bessel_weights * math.pi * x) / r
            data["edge_embedding"] = basis.to(torch.float32)
            sh = torch.cat([torch.ones_like(r), vec / r], dim=-1)
            data["edge_attrs"] = sh.to(torch.float32)
            return data


    class StridedProduct(torch.nn.Module):
        def __init__(self, mul):
            super().__init__()
            self.mul = mul
            self.irreps_out = "1x0e+1x1o"
            self.weights = torch.nn.Parameter(torch.linspace(-1.0, 1.0, mul))

        def forward(self, x, w):
            return x * (w * self.weights).unsqueeze(-1)


    class Core(torch.nn.Module):
        def __init__(self, num_types, mul):
            super().__init__()
            self.mul = mul
            self.type_embed = torch.nn.Embedding(num_types, 4)
            self.latents = torch.nn.ModuleList(
                [torch.nn.Linear(12, 8), torch.nn.Linear(8 + mul, 8)]
            )
            self.env = torch.nn.Linear(8, mul)
            self.tps = torch.nn.ModuleList([StridedProduct(mul), StridedProduct(mul)])
            self.irreps_out = {
                "edge_embedding": "4x0e",
                "edge_attrs": "1x0e+1x1o",
                "edge_features": "8x0e",
            }

        def forward(self, data):
            edge_index = data["edge_index"]
            z = self.type_embed(data["atom_types"])
            # exercise the translated num_atoms: per-edge size of the centre's frame
            frame_size = data["num_atoms"][data["batch"][edge_index[0]]]
            scale = frame_size.to(torch.float32).rsqrt().unsqueeze(-1)
            h = torch.cat([z[edge_index[0]], z[edge_index[1]], data["edge_embedding"]], -1)
            h = torch.tanh(self.latents[0](h)) * scale
            feats = data["edge_attrs"].unsqueeze(1).expand(-1, self.mul, -1)
            for i, tp in enumerate(self.tps):
                feats = tp(feats, self.env(h))
                if i == 0:
                    h = torch.tanh(self.latents[1](torch.cat([h, feats[:, :, 0]], -1)))
            data["edge_features"] = h
            return data


    class Readout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.edge_readout = torch.nn.Linear(8, 1)
            self.irreps_out = {"total_energy": "0e"}

        def forward(self, data):
            edge_energy = self.edge_readout(data["edge_features"]).to(torch.float64)
            n_frames = data["num_atoms"].shape[0]
            edge_batch = data["batch"][data["edge_index"][0]]
            energy = torch.zeros(n_frames, 1, dtype=torch.float64)
            data["total_energy"] = energy.index_add(0, edge_batch, edge_energy)
            return data


    class Sequential(torch.nn.Sequential):
        def forward(self, data):
            for module in self:
                data = module(data)
            return data


    class ForceOutput(torch.nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func

        def forward(self, data):
            raise RuntimeError("the wrapper must not call the force output")


    class GraphModel(torch.nn.Module):
        def __init__(self, model, type_names, r_max, extra_inputs=()):
            super().__init__()
            self.model = model
            self.type_names = list(type_names)
            self._metadata = {"type_names": " ".join(type_names), "r_max": str(r_max)}
            self.model_input_fields = [
                "pos", "edge_index", "edge_cell_shift", "edge_vectors", "cell",
                "batch", "num_atoms", "atom_types",
            ] + list(extra_inputs)

        @property
        def metadata(self):
            return dict(self._metadata)

        def forward(self, data):
            return self.model(data)


    def build(type_names, r_max, mul, extra_inputs=()):
        func = Sequential()
        func.add_module("edge_embed", EdgeEmbed(r_max))
        func.add_module("allegro", Core(len(type_names), mul))
        func.add_module("readout", Readout())
        model = GraphModel(ForceOutput(func), type_names, r_max, extra_inputs)
        return torch.nn.ModuleDict({"sole_model": model})
    """
)


def _toy_example_data(extra=()):
    data = {
        "pos": torch.zeros(2, 3, dtype=torch.float64),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_cell_shift": torch.zeros(2, 3, dtype=torch.float64),
        "cell": torch.eye(3, dtype=torch.float64).unsqueeze(0),
        "pbc": torch.ones(1, 3, dtype=torch.bool),
        "atom_types": torch.tensor([0, 1]),
        "atomic_numbers": torch.tensor([1, 6]),
        "batch": torch.zeros(2, dtype=torch.long),
        "num_atoms": torch.tensor([2]),
        "total_energy": torch.zeros(1, 1, dtype=torch.float64),
    }
    for key in extra:
        data[key] = torch.zeros(2)
    return data


def _write_package(directory, name, module_name, extra_inputs=(), type_names=None):
    if module_name not in sys.modules:
        source_dir = directory / f"src_{module_name}"
        source_dir.mkdir()
        (source_dir / f"{module_name}.py").write_text(_TOY_SOURCE)
        sys.path.insert(0, str(source_dir))
        importlib.invalidate_caches()
    toy = importlib.import_module(module_name)
    torch.manual_seed(0)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)  # packaged models carry their own dtypes
    try:
        model = toy.build(type_names or TYPE_NAMES, R_MAX, MUL, extra_inputs)
    finally:
        torch.set_default_dtype(previous)
    path = directory / name
    with PackageExporter(str(path)) as exporter:
        exporter.intern(module_name)
        exporter.extern(["math"])
        exporter.save_pickle("model", "eager_model.pkl", model)
        exporter.save_pickle(
            "model", "example_data.pkl", _toy_example_data(extra_inputs)
        )
    return path


@pytest.fixture(scope="module")
def package_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("packaged_backbone")
    _write_package(directory, "toy.zip", "spingnn_toy_backbone")
    _write_package(
        directory,
        "toy_extra_input.zip",
        "spingnn_toy_backbone",
        extra_inputs=("magic_field",),
    )
    _write_package(
        directory,
        "toy_other_types.zip",
        "spingnn_toy_backbone",
        type_names=["C", "H", "O"],
    )
    yield directory
    sys.modules.pop("spingnn_toy_backbone", None)
    sys.path[:] = [p for p in sys.path if not p.startswith(str(directory))]


TAPS = {
    "bb_tensor": {"module": "allegro.tps.1"},
    "bb_scalars": "edge_features",
    "bb_basis": {"key": "edge_attrs"},
}


def _backbone(package_dir, name="toy.zip", **kwargs):
    kwargs.setdefault("taps", TAPS)
    kwargs.setdefault("stop_after", "allegro")
    kwargs.setdefault("type_names", TYPE_NAMES)
    kwargs.setdefault("r_max", R_MAX)
    return PackagedBackbone(package_path=str(package_dir / name), **kwargs)


def _batch():
    dtype = torch.get_default_dtype()
    first = AtomicData.from_points(
        pos=torch.tensor(
            [[0.1, 0.2, 0.0], [1.2, 0.1, 0.4], [0.3, 1.4, 0.9], [1.6, 1.5, 1.7]],
            dtype=dtype,
        ),
        r_max=R_MAX,
        cell=torch.eye(3, dtype=dtype) * 3.2,
        pbc=True,
        atom_types=torch.tensor([[0], [1], [2], [1]]),
    )
    second = AtomicData.from_points(
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.3, 0.0], [0.2, 1.1, 0.5]], dtype=dtype
        ),
        r_max=R_MAX,
        cell=torch.eye(3, dtype=dtype) * 12.0,
        pbc=True,
        atom_types=torch.tensor([[2], [0], [0]]),
    )
    return AtomicData.to_AtomicDataDict(Batch.from_data_list([first, second]))


def _direct_reference(package_path, data, hook_path="allegro.tps.1"):
    """Run the packaged stages by hand, the way the wrapper is meant to."""
    importer = PackageImporter(str(package_path))
    model = importer.load_pickle("model", "eager_model.pkl", map_location="cpu")
    func = model["sole_model"].model.func
    func.eval()
    batch = data[AtomicDataDict.BATCH_KEY]
    packaged = {
        "pos": data[AtomicDataDict.POSITIONS_KEY],
        "edge_index": data[AtomicDataDict.EDGE_INDEX_KEY],
        "edge_cell_shift": data[AtomicDataDict.EDGE_CELL_SHIFT_KEY],
        "cell": data[AtomicDataDict.CELL_KEY].view(-1, 3, 3),
        "atom_types": data[AtomicDataDict.ATOM_TYPE_KEY].view(-1),
        "batch": batch,
        "num_atoms": torch.bincount(batch),
    }
    captured = {}
    handle = func.get_submodule(hook_path).register_forward_hook(
        lambda m, i, o: captured.setdefault("out", o)
    )
    packaged = func.edge_embed(packaged)
    packaged = func.allegro(packaged)
    handle.remove()
    return packaged, captured["out"]


def test_taps_match_direct_packaged_run(package_dir):
    backbone = _backbone(package_dir)
    data = _batch()
    out = backbone(dict(data))
    packaged, hooked = _direct_reference(package_dir / "toy.zip", data)
    assert out["bb_tensor"].shape == (data["edge_index"].shape[1], MUL, 4)
    assert torch.equal(out["bb_tensor"], hooked)
    assert torch.equal(out["bb_scalars"], packaged["edge_features"])
    assert torch.equal(out["bb_basis"], packaged["edge_attrs"])
    # the packaged keys themselves are not leaked into the SpinGNN data
    assert "edge_features" not in out and "num_atoms" not in out
    # num_atoms is derived from the batch: frames of 4 and 3 atoms
    assert backbone._translate(data)["num_atoms"].tolist() == [4, 3]


def test_module_tap_hook_is_fixed_and_records_only_in_forward(package_dir):
    backbone = _backbone(package_dir)
    (tap,) = backbone._module_taps
    hooked = backbone.func.get_submodule("allegro.tps.1")
    hooks = list(hooked._forward_hooks.items())
    assert [hook for _, hook in hooks] == [tap]  # registered once, at construction
    data = _batch()
    first = backbone(dict(data))["bb_tensor"]
    assert torch.equal(backbone(dict(data))["bb_tensor"], first)
    assert list(hooked._forward_hooks.items()) == hooks
    assert tap.output is None and not tap.recording  # nothing kept after a forward
    # running the packaged stages outside the backbone's forward records nothing
    backbone._forward_stages(backbone._translate(data))
    assert tap.output is None
    # an error inside the stages also resets the tap
    handle = backbone.func["allegro"].register_forward_hook(lambda *args: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        backbone(dict(data))
    handle.remove()
    assert tap.output is None and not tap.recording
    # a copy has its own hook on its own module
    clone = copy.deepcopy(backbone)
    (clone_tap,) = clone._module_taps
    assert clone_tap is not tap
    clone_hooked = clone.func.get_submodule("allegro.tps.1")
    assert list(clone_hooked._forward_hooks.values()) == [clone_tap]
    assert torch.equal(clone(dict(data))["bb_tensor"], first)


def test_registered_irreps(package_dir):
    backbone = _backbone(package_dir)
    irreps = backbone.irreps_out
    # strided (E, MUL, 4) output: the per-channel irreps of the last dimension
    assert irreps["bb_tensor"] == o3.Irreps("1x0e+1x1o")
    assert irreps["bb_scalars"] == o3.Irreps("8x0e")
    assert irreps["bb_basis"] == o3.Irreps("1x0e+1x1o")
    explicit = _backbone(
        package_dir, taps={"bb_basis": {"key": "edge_attrs", "irreps": "0e+1o"}}
    )
    assert str(explicit.irreps_out["bb_basis"]) == str(irreps["bb_basis"])


def test_stage_cut_off(package_dir):
    data = _batch()
    backbone = _backbone(package_dir)
    assert list(backbone.func.keys()) == ["edge_embed", "allegro"]
    assert not any(n.startswith("readout") for n, _ in backbone.named_parameters())
    kept = _backbone(package_dir, keep_skipped_stages=True)
    assert list(kept.func.keys()) == ["edge_embed", "allegro", "readout"]
    calls = []
    kept.func["readout"].register_forward_hook(lambda *args: calls.append(1))
    kept(dict(data))
    assert calls == []  # kept, but never executed
    full = _backbone(
        package_dir, stop_after=None, taps={"energy": {"key": "total_energy"}}
    )
    assert full(dict(data))["energy"].shape == (2, 1)
    with pytest.raises(ValueError, match="not produced by the stages"):
        _backbone(package_dir, taps={"energy": "total_energy"})
    with pytest.raises(ValueError, match="not inside the stages"):
        _backbone(package_dir, taps={"x": {"module": "readout.edge_readout"}})
    with pytest.raises(ValueError, match="not a stage"):
        _backbone(package_dir, stop_after="nonexistent")
    with pytest.raises(ValueError, match="exactly one of"):
        _backbone(package_dir, taps={"x": {"key": "edge_attrs", "module": "allegro"}})


_LATENT0_AND_BESSEL = [
    "func.allegro.latents.0.bias",
    "func.allegro.latents.0.weight",
    "func.edge_embed.bessel_weights",
]


@pytest.mark.parametrize(
    "prefixes, expected",
    [
        # full names relative to the packaged network
        (["edge_embed.bessel", "allegro.latents.0."], _LATENT0_AND_BESSEL),
        # the same names without their stage component, in any stage
        (["bessel_weights", "latents.0."], _LATENT0_AND_BESSEL),
        (
            ["type_embed.", "tps."],
            [
                "func.allegro.tps.0.weights",
                "func.allegro.tps.1.weights",
                "func.allegro.type_embed.weight",
            ],
        ),
    ],
)
def test_trainable_prefixes(package_dir, prefixes, expected):
    backbone = _backbone(package_dir, trainable_prefixes=prefixes)
    trainable = sorted(n for n, p in backbone.named_parameters() if p.requires_grad)
    assert trainable == expected
    out = backbone(_batch())
    (out["bb_tensor"].sum() + out["bb_scalars"].sum()).backward()
    for name, param in backbone.named_parameters():
        assert (param.grad is not None) == param.requires_grad, name


def test_trainable_prefix_rule_limits(package_dir):
    frozen = _backbone(package_dir)
    assert not any(p.requires_grad for p in frozen.parameters())
    with pytest.raises(ValueError, match="match no parameter"):
        _backbone(package_dir, trainable_prefixes=["no_such_module."])
    # only the stage component is dropped, not deeper ones
    with pytest.raises(ValueError, match=r"\['0\.weight'\]"):
        _backbone(package_dir, trainable_prefixes=["latents.", "0.weight"])
    # parameters of dropped stages do not exist
    with pytest.raises(ValueError, match="match no parameter"):
        _backbone(package_dir, trainable_prefixes=["edge_readout."])
    kept = _backbone(
        package_dir, trainable_prefixes=["edge_readout."], keep_skipped_stages=True
    )
    trainable = sorted(n for n, p in kept.named_parameters() if p.requires_grad)
    assert trainable == [
        "func.readout.edge_readout.bias",
        "func.readout.edge_readout.weight",
    ]


def test_packaged_modules_stay_in_eval_mode(package_dir):
    backbone = _backbone(package_dir)
    backbone.train()
    assert backbone.training
    assert not any(m.training for m in backbone.func.modules())
    backbone.eval()
    backbone.train(True)
    assert not any(m.training for m in backbone.func.modules())


def test_dtypes_are_kept(package_dir):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        backbone = _backbone(package_dir)
    finally:
        torch.set_default_dtype(previous)
    assert PackagedBackbone._spingnn_keep_dtype is True
    assert backbone._spingnn_keep_dtype is True
    params = dict(backbone.named_parameters())
    buffers = dict(backbone.named_buffers())
    assert params["func.edge_embed.bessel_weights"].dtype == torch.float64
    assert buffers["func.edge_embed._rmax_recip"].dtype == torch.float64
    assert params["func.allegro.latents.0.weight"].dtype == torch.float32
    out = backbone(_batch())
    assert out["bb_tensor"].dtype == torch.float32
    assert out["bb_scalars"].dtype == torch.float32


def test_model_dtype_cast_skips_backbone(package_dir):
    """``cast_model_dtype`` (``model_dtype``) leaves the packaged dtypes alone."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        backbone = _backbone(package_dir)
        head = torch.nn.Linear(4, 4)
    finally:
        torch.set_default_dtype(previous)
    model = torch.nn.ModuleDict({"backbone": backbone, "head": head})
    cast_model_dtype(model, torch.float32)
    assert head.weight.dtype == torch.float32
    params = dict(backbone.named_parameters())
    buffers = dict(backbone.named_buffers())
    assert params["func.edge_embed.bessel_weights"].dtype == torch.float64
    assert buffers["func.edge_embed._rmax_recip"].dtype == torch.float64
    assert params["func.allegro.latents.0.weight"].dtype == torch.float32
    out = backbone(_batch())
    assert out["bb_tensor"].dtype == torch.float32


def test_state_dict_round_trip(package_dir):
    data = _batch()
    source = _backbone(package_dir, trainable_prefixes=["latents."])
    with torch.no_grad():
        for param in source.parameters():
            if param.requires_grad:
                param.add_(0.25)
    target = _backbone(package_dir, trainable_prefixes=["latents."])
    before = target(dict(data))["bb_scalars"]
    state = source.state_dict()
    assert "func.allegro.latents.1.weight" in state
    assert "func.edge_embed._rmax_recip" in state
    assert state["func.edge_embed.bessel_weights"].dtype == torch.float64
    target.load_state_dict(state, strict=True)
    after = target(dict(data))
    expected = source(dict(data))
    assert not torch.equal(before, after["bb_scalars"])
    for key in TAPS:
        assert torch.equal(after[key], expected[key])


def test_type_names_r_max_and_inputs_are_checked(package_dir):
    with pytest.raises(ValueError, match="type names"):
        _backbone(package_dir, type_names=["H", "O", "C"])
    with pytest.raises(ValueError, match="type names"):
        _backbone(package_dir, name="toy_other_types.zip")
    with pytest.raises(ValueError, match="type_names"):
        _backbone(package_dir, type_names=None)
    with pytest.raises(ValueError, match="r_max"):
        _backbone(package_dir, r_max=R_MAX + 1.0)
    with pytest.raises(ValueError, match="magic_field"):
        _backbone(package_dir, name="toy_extra_input.zip")
    with pytest.raises(KeyError, match="no key"):
        _backbone(package_dir, model_key="other")


def test_builder(package_dir):
    config = {
        "model_builders": ["PackagedBackboneModel"],
        "type_names": TYPE_NAMES,
        "num_types": len(TYPE_NAMES),
        "r_max": R_MAX,
        "packaged_backbone_path": str(package_dir / "toy.zip"),
        "packaged_backbone_stop_after": "allegro",
        "packaged_backbone_taps": TAPS,
        "packaged_backbone_trainable_prefixes": ["latents."],
    }
    model = model_from_config(config)
    assert list(dict(model.named_children())) == ["packaged_backbone"]
    assert isinstance(model.packaged_backbone, PackagedBackbone)
    out = model(_batch())
    assert set(TAPS) <= set(out)
    assert set(TAPS) <= set(model.irreps_out)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(".latents." in n for n in trainable)
    config.pop("packaged_backbone_taps")
    with pytest.raises(KeyError, match="packaged_backbone_taps"):
        model_from_config(config)


# -- DDP: the wrapped backbone trains like any other module --------------------

_WORLD_SIZE = 2
_JOIN_TIMEOUT = 120.0


def _ddp_worker(rank, world_size, init_file, package_path, result_dir, dtype):
    from torch.nn.parallel import DistributedDataParallel

    from nequip.train.trainer import warmup_process_group_for_ddp

    torch.set_default_dtype(getattr(torch, dtype))
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        warmup_process_group_for_ddp()
        model = PackagedBackbone(
            package_path=package_path,
            taps=TAPS,
            stop_after="allegro",
            trainable_prefixes=["latents."],
            type_names=TYPE_NAMES,
            r_max=R_MAX,
        )
        ddp = DistributedDataParallel(model)
        ddp.train()
        assert not any(m.training for m in ddp.module.func.modules())
        data = _batch()
        # rank-dependent objective, so the reduced gradient differs from both locals
        out = ddp(dict(data))
        loss = (rank + 1.0) * out["bb_scalars"].square().sum() + out["bb_tensor"].sum()
        loss.backward()
        grads = {
            n: p.grad.detach().clone()
            for n, p in ddp.module.named_parameters()
            if p.requires_grad
        }
        torch.save(grads, str(result_dir / f"grads-{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="torch distributed gloo backend unavailable",
)
def test_ddp_two_rank_gradients(package_dir, tmp_path):
    package_path = str(package_dir / "toy.zip")
    context = mp.start_processes(
        _ddp_worker,
        args=(
            _WORLD_SIZE,
            str(tmp_path / "init"),
            package_path,
            tmp_path,
            str(torch.get_default_dtype()).replace("torch.", ""),
        ),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    for process in context.processes:
        process.join(timeout=_JOIN_TIMEOUT)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("gloo worker hung; terminated after timeout")
    assert [p.exitcode for p in context.processes] == [0] * _WORLD_SIZE
    grads = [torch.load(str(tmp_path / f"grads-{r}.pt")) for r in range(_WORLD_SIZE)]
    assert sorted(grads[0]) == sorted(grads[1])
    assert grads[0] and all(".latents." in name for name in grads[0])

    # single-process reference: mean of the two rank objectives
    reference = _backbone(package_dir, trainable_prefixes=["latents."])
    out = reference(_batch())
    loss = 0.5 * (3.0 * out["bb_scalars"].square().sum() + 2.0 * out["bb_tensor"].sum())
    loss.backward()
    for name, param in reference.named_parameters():
        if param.requires_grad:
            assert torch.equal(grads[0][name], grads[1][name]), name
            assert math.isclose(
                float((grads[0][name] - param.grad).abs().max()), 0.0, abs_tol=1e-5
            ), name


# -- torch_compile: opt-in compilation of the packaged stages -----------------

# Dynamo + AOTAutograd without Inductor's C++ build keeps the default CPU lane
# fast; `test_torch_compile_matches_eager_over_graph_sizes[inductor]` runs the
# production backend behind the SPINGNN_TEST_COMPILE gate.
_CPU_BACKEND = "aot_eager"
_COMPILE_PREFIXES = ["latents.", "tps.", "bessel_weights"]
# atoms per frame; one- and multi-frame batches, and a one-atom frame whose
# only edges are its periodic images
_GRAPH_SIZES = [(4, 3), (5,), (2, 6, 3), (1,), (7, 7), (3, 1, 3, 3)]


def _sized_batch(frame_atoms, seed):
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.get_default_dtype()
    frames = []
    for n_atoms in frame_atoms:
        length = 2.5 if n_atoms == 1 else 4.0
        frames.append(
            AtomicData.from_points(
                pos=torch.rand(n_atoms, 3, generator=generator, dtype=dtype) * length,
                r_max=R_MAX,
                cell=torch.eye(3, dtype=dtype) * length,
                pbc=True,
                atom_types=torch.randint(
                    0, len(TYPE_NAMES), (n_atoms, 1), generator=generator
                ),
            )
        )
    return AtomicData.to_AtomicDataDict(Batch.from_data_list(frames))


def _dynamo_counts():
    counters = torch._dynamo.utils.counters
    return counters["stats"]["unique_graphs"], sum(counters["graph_break"].values())


@pytest.fixture
def fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _compiled_pair(package_dir, backend=_CPU_BACKEND, **kwargs):
    kwargs.setdefault("trainable_prefixes", _COMPILE_PREFIXES)
    eager = _backbone(package_dir, **kwargs)
    compiled = _backbone(
        package_dir, torch_compile={"enabled": True, "backend": backend}, **kwargs
    )
    compiled.load_state_dict(eager.state_dict(), strict=True)
    return eager, compiled


def _perturb_trainable(module, amount):
    with torch.no_grad():
        for param in module.parameters():
            if param.requires_grad:
                param.add_(amount)


def _assert_taps_equal(out, expected, rtol, atol):
    for key in TAPS:
        torch.testing.assert_close(out[key], expected[key], rtol=rtol, atol=atol)


def _check_compiled_matches_eager(eager, compiled, rtol, atol):
    """Compiled vs eager taps and gradients over `_GRAPH_SIZES`, then repeats."""
    assert compiled._compiled_stages is None  # compiled lazily on first forward
    hooked = compiled.func.get_submodule("allegro.tps.1")
    hooks = list(hooked._forward_hooks.items())
    graphs0, breaks0 = _dynamo_counts()
    for seed, frame_atoms in enumerate(_GRAPH_SIZES + _GRAPH_SIZES[:3]):
        if seed == len(_GRAPH_SIZES):
            graphs_first_pass = _dynamo_counts()[0] - graphs0
        data = _sized_batch(frame_atoms, seed=seed % len(_GRAPH_SIZES))
        out, expected = compiled(dict(data)), eager(dict(data))
        assert list(hooked._forward_hooks.items()) == hooks  # tap hook unchanged
        _assert_taps_equal(out, expected, rtol, atol)
        objective = out["bb_tensor"].square().sum() + out["bb_scalars"].sum()
        objective.backward()
        (expected["bb_tensor"].square().sum() + expected["bb_scalars"].sum()).backward()
        for (name, param), reference in zip(
            compiled.named_parameters(), eager.parameters()
        ):
            if param.requires_grad:
                scale = float(reference.grad.abs().max())
                torch.testing.assert_close(
                    param.grad, reference.grad, rtol=rtol, atol=atol * max(scale, 1.0)
                )
            else:
                assert param.grad is None, name
        compiled.zero_grad(set_to_none=True)
        eager.zero_grad(set_to_none=True)
    graphs, breaks = _dynamo_counts()
    assert breaks == breaks0  # module-tap hooks are traced without graph breaks
    # dynamic shapes: one graph for multi-frame batches plus the size-1
    # specializations (one frame; one atom), and no recompiles on repeats
    assert 1 <= graphs_first_pass <= 4
    assert graphs - graphs0 == graphs_first_pass


def _check_instances_keep_their_own_taps(package_dir):
    settings = {"enabled": True, "backend": _CPU_BACKEND}
    taps_a = {"t": {"module": "allegro.tps.1"}, "s": "edge_features"}
    taps_b = {"t": {"module": "allegro.tps.0"}, "s": "edge_features"}
    data = _batch()
    runs = []
    for taps in (taps_a, taps_b, {"s": "edge_features"}):
        compiled = _backbone(package_dir, taps=taps, torch_compile=settings)
        runs.append((compiled, _backbone(package_dir, taps=taps)))
    for compiled, eager in runs + runs[:1]:
        out, expected = compiled(dict(data)), eager(dict(data))
        assert set(out) == set(expected)
        for key in [k for k in ("t", "s") if k in expected]:
            torch.testing.assert_close(out[key], expected[key], rtol=1e-5, atol=1e-6)


def _check_copy_runs_its_own_parameters(package_dir, eager, compiled, data):
    clone = copy.deepcopy(compiled)
    assert clone._compiled_stages is None
    assert clone._torch_compile_kwargs == compiled._torch_compile_kwargs
    _perturb_trainable(clone, 0.125)
    reference = _backbone(package_dir, trainable_prefixes=_COMPILE_PREFIXES)
    reference.load_state_dict(clone.state_dict(), strict=True)
    _assert_taps_equal(clone(dict(data)), reference(dict(data)), 1e-5, 1e-6)
    assert clone._compiled_stages is not None
    _assert_taps_equal(compiled(dict(data)), eager(dict(data)), 1e-5, 1e-6)


@pytest.mark.parametrize(
    "backend",
    [
        _CPU_BACKEND,
        pytest.param(
            "inductor",
            marks=[
                pytest.mark.compile,
                # emitted by Inductor's own CPU compile path (torch 2.11)
                pytest.mark.filterwarnings(
                    "ignore:`torch.jit.script_method` is deprecated:DeprecationWarning"
                ),
            ],
        ),
    ],
)
def test_torch_compile_matches_eager_over_graph_sizes(
    package_dir, fresh_dynamo, backend
):
    if backend == "inductor" and os.getenv("SPINGNN_TEST_COMPILE") != "1":
        pytest.skip("compile gate disabled")
    rtol, atol = (1e-5, 1e-6) if backend == _CPU_BACKEND else (1e-4, 1e-5)
    eager, compiled = _compiled_pair(package_dir, backend=backend)
    _check_compiled_matches_eager(eager, compiled, rtol, atol)


def test_torch_compile_instances_keep_their_own_taps(package_dir, fresh_dynamo):
    """Backbones tapping different submodules do not share compiled tap code."""
    _check_instances_keep_their_own_taps(package_dir)


def test_torch_compile_keeps_state_dict_and_copies(package_dir, fresh_dynamo):
    eager, compiled = _compiled_pair(package_dir)
    keys = list(eager.state_dict())
    names = [name for name, _ in eager.named_parameters()]
    data = _batch()
    compiled(dict(data))
    assert compiled._compiled_stages is not None
    assert list(compiled.state_dict()) == keys
    assert [name for name, _ in compiled.named_parameters()] == names
    assert not any("_orig_mod" in key for key in keys)
    # compiled -> eager and eager -> compiled checkpoints, strict
    _perturb_trainable(compiled, 0.25)
    eager.load_state_dict(compiled.state_dict(), strict=True)
    _assert_taps_equal(compiled(dict(data)), eager(dict(data)), 1e-5, 1e-6)
    _perturb_trainable(eager, -0.5)
    compiled.load_state_dict(eager.state_dict(), strict=True)
    _assert_taps_equal(compiled(dict(data)), eager(dict(data)), 1e-5, 1e-6)
    # a copy recompiles for itself and runs its own parameters
    _check_copy_runs_its_own_parameters(package_dir, eager, compiled, data)


@pytest.fixture
def spingnn_compile_dynamo_config(fresh_dynamo):
    """Install the process-wide Dynamo settings of `spingnn_compile`; restore them after.

    `configure_model_for_compile` sets them (e.g. `inline_inbuilt_nn_modules`
    and `guard_nn_modules` off) for the whole process once a model has a
    `ParaStressForceSpinOutput`, so a compiled backbone in the same process
    runs under them.
    """
    import e3nn
    import torch._functorch.config as functorch_config

    from nequip.utils.compile_utils import configure_model_for_compile

    configs = (torch._dynamo.config, functorch_config)
    saved = [config.get_config_copy() for config in configs]
    # the production selector matches this class name
    target = type(
        "ParaStressForceSpinOutput",
        (torch.nn.Module,),
        {"forward": lambda self, data: data},
    )()
    jit_script_fx = e3nn.get_optimization_defaults()["jit_script_fx"]
    try:
        e3nn.set_optimization_defaults(jit_script_fx=False)
        try:
            configure_model_for_compile(target, {"spingnn_compile": {"enabled": True}})
        finally:
            e3nn.set_optimization_defaults(jit_script_fx=jit_script_fx)
        assert not torch._dynamo.config.inline_inbuilt_nn_modules
        assert not torch._dynamo.config.guard_nn_modules
        yield
    finally:
        for config, before in zip(configs, saved):
            for key, value in before.items():
                if getattr(config, key) != value:
                    setattr(config, key, value)


def test_torch_compile_under_spingnn_compile_dynamo_config(
    package_dir, spingnn_compile_dynamo_config
):
    """Regression: the compiled stages under the `spingnn_compile` Dynamo settings.

    Without inlining of nn modules, Dynamo addresses a forward hook by its
    handle id. A tap hook registered anew in every forward then recompiled on
    every call; with `guard_nn_modules=False` the cached code looked up the
    previous forward's handle id and raised `KeyError`.
    """
    eager, compiled = _compiled_pair(package_dir)
    _check_compiled_matches_eager(eager, compiled, rtol=1e-5, atol=1e-6)
    _check_copy_runs_its_own_parameters(package_dir, eager, compiled, _batch())
    _check_instances_keep_their_own_taps(package_dir)


def test_torch_compile_default_off_is_eager(package_dir, monkeypatch):
    config = {
        "model_builders": ["PackagedBackboneModel"],
        "type_names": TYPE_NAMES,
        "num_types": len(TYPE_NAMES),
        "r_max": R_MAX,
        "packaged_backbone_path": str(package_dir / "toy.zip"),
        "packaged_backbone_stop_after": "allegro",
        "packaged_backbone_taps": TAPS,
    }
    default = model_from_config(config).packaged_backbone
    config["packaged_backbone_torch_compile"] = {"enabled": False, "backend": "eager"}
    disabled = model_from_config(config).packaged_backbone
    for backbone in (default, disabled):
        assert backbone._torch_compile_kwargs is None
    assert list(disabled.state_dict()) == list(default.state_dict())

    def no_compile(*args, **kwargs):
        raise AssertionError("torch.compile must not be called when disabled")

    monkeypatch.setattr(torch, "compile", no_compile)
    graphs0, _ = _dynamo_counts()
    data = _batch()
    out = default(dict(data))
    assert default._compiled_stages is None
    assert _dynamo_counts()[0] == graphs0
    packaged, hooked = _direct_reference(package_dir / "toy.zip", data)
    assert torch.equal(out["bb_tensor"], hooked)
    assert torch.equal(out["bb_scalars"], packaged["edge_features"])


def test_torch_compile_settings(package_dir):
    config = {
        "model_builders": ["PackagedBackboneModel"],
        "type_names": TYPE_NAMES,
        "num_types": len(TYPE_NAMES),
        "r_max": R_MAX,
        "packaged_backbone_path": str(package_dir / "toy.zip"),
        "packaged_backbone_stop_after": "allegro",
        "packaged_backbone_taps": TAPS,
        "packaged_backbone_torch_compile": {"enabled": True, "backend": "aot_eager"},
    }
    backbone = model_from_config(config).packaged_backbone
    assert backbone._torch_compile_kwargs == {"backend": "aot_eager", "dynamic": True}
    assert backbone._compiled_stages is None  # nothing is compiled at build time

    defaults = _torch_compile_kwargs(True)
    assert defaults["backend"] == "inductor" and defaults["dynamic"] is True
    # the `spingnn_compile` inductor policy: no CUDA graphs, no max-autotune
    assert defaults["options"]["triton.cudagraphs"] is False
    assert defaults["options"]["max_autotune"] is False
    assert _torch_compile_kwargs({"enabled": True, "dynamic": None})["dynamic"] is None
    assert _torch_compile_kwargs(
        {"enabled": True, "mode": "max-autotune-no-cudagraphs"}
    ) == {"backend": "inductor", "dynamic": True, "mode": "max-autotune-no-cudagraphs"}
    with pytest.warns(UserWarning, match="no_such_inductor_option"):
        filtered = _torch_compile_kwargs(
            {"enabled": True, "inductor_options": {"no_such_inductor_option": 1}}
        )
    assert "no_such_inductor_option" not in filtered["options"]
    for disabled in (None, False, {}, {"enabled": False, "mode": "x"}):
        assert _torch_compile_kwargs(disabled) is None

    with pytest.raises(ValueError, match="Unknown torch_compile option"):
        _torch_compile_kwargs({"enabled": True, "fullgraph": True})
    with pytest.raises(ValueError, match="either `mode` or `inductor_options`"):
        _torch_compile_kwargs(
            {
                "enabled": True,
                "mode": "default",
                "inductor_options": {"max_autotune": True},
            }
        )
    with pytest.raises(ValueError, match="inductor backend only"):
        _torch_compile_kwargs({"enabled": True, "backend": "eager", "mode": "default"})
    with pytest.raises(ValueError, match="inductor backend only"):
        _torch_compile_kwargs(
            {
                "enabled": True,
                "backend": "eager",
                "inductor_options": {"max_autotune": True},
            }
        )
    with pytest.raises(ValueError, match="dynamic"):
        _torch_compile_kwargs({"enabled": True, "dynamic": "yes"})
    with pytest.raises(TypeError, match="bool or a mapping"):
        _torch_compile_kwargs("inductor")
    config["packaged_backbone_torch_compile"] = {"enabled": True, "dynamics": True}
    with pytest.raises(ValueError, match="dynamics"):
        model_from_config(config)


# The opt-in DDP options (trainer keys `ddp_static_graph`,
# `ddp_gradient_as_bucket_view`, `ddp_broadcast_buffers`,
# `ddp_find_unused_parameters`) together with a compiled backbone.
_DDP_OPTIONS = {
    "static_graph": True,
    "gradient_as_bucket_view": True,
    "broadcast_buffers": False,
    "find_unused_parameters": False,
}
_DDP_STEPS = [((4, 3), (5,)), ((2, 6), (1, 3)), ((3, 3, 2), (6,))]  # per step, per rank
_DDP_LR = 0.01


def _ddp_objective(out, rank):
    return (rank + 1.0) * out["bb_scalars"].square().sum() + out["bb_tensor"].sum()


def _ddp_compile_worker(rank, world_size, init_file, package_path, result_dir, dtype):
    from torch.nn.parallel import DistributedDataParallel

    from nequip.train.trainer import warmup_process_group_for_ddp

    torch.set_default_dtype(getattr(torch, dtype))
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        warmup_process_group_for_ddp()
        model = PackagedBackbone(
            package_path=package_path,
            taps=TAPS,
            stop_after="allegro",
            trainable_prefixes=_COMPILE_PREFIXES,
            type_names=TYPE_NAMES,
            r_max=R_MAX,
            torch_compile={"enabled": True, "backend": _CPU_BACKEND},
        )
        ddp = DistributedDataParallel(model, **_DDP_OPTIONS)
        ddp.train()
        optim = torch.optim.SGD(
            [p for p in ddp.parameters() if p.requires_grad], lr=_DDP_LR
        )
        grads = []
        for step, per_rank in enumerate(_DDP_STEPS):
            optim.zero_grad(set_to_none=True)
            out = ddp(dict(_sized_batch(per_rank[rank], seed=10 * step + rank)))
            _ddp_objective(out, rank).backward()
            grads.append(
                {
                    n: p.grad.detach().clone()
                    for n, p in ddp.module.named_parameters()
                    if p.requires_grad
                }
            )
            optim.step()
        assert ddp.module._compiled_stages is not None
        params = {n: p.detach().clone() for n, p in ddp.module.named_parameters()}
        torch.save(
            {"grads": grads, "params": params}, str(result_dir / f"ddp-{rank}.pt")
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="torch distributed gloo backend unavailable",
)
def test_torch_compile_ddp_options_two_ranks(package_dir, tmp_path):
    if torch.get_default_dtype() is not torch.float64:
        pytest.skip("compiled DDP runs once, with float64 data and the packaged dtypes")
    package_path = str(package_dir / "toy.zip")
    context = mp.start_processes(
        _ddp_compile_worker,
        args=(
            _WORLD_SIZE,
            str(tmp_path / "init"),
            package_path,
            tmp_path,
            str(torch.get_default_dtype()).replace("torch.", ""),
        ),
        nprocs=_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    for process in context.processes:
        process.join(timeout=_JOIN_TIMEOUT)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("gloo worker hung; terminated after timeout")
    assert [p.exitcode for p in context.processes] == [0] * _WORLD_SIZE
    results = [torch.load(str(tmp_path / f"ddp-{r}.pt")) for r in range(_WORLD_SIZE)]

    # single-process eager reference: SGD on the mean of the two rank objectives
    reference = _backbone(package_dir, trainable_prefixes=_COMPILE_PREFIXES)
    optim = torch.optim.SGD(
        [p for p in reference.parameters() if p.requires_grad], lr=_DDP_LR
    )
    for step, per_rank in enumerate(_DDP_STEPS):
        optim.zero_grad(set_to_none=True)
        for rank in range(_WORLD_SIZE):
            out = reference(dict(_sized_batch(per_rank[rank], seed=10 * step + rank)))
            (_ddp_objective(out, rank) / _WORLD_SIZE).backward()
        for name, param in reference.named_parameters():
            if not param.requires_grad:
                continue
            got = [result["grads"][step][name] for result in results]
            assert torch.equal(got[0], got[1]), name
            scale = max(float(param.grad.abs().max()), 1.0)
            torch.testing.assert_close(
                got[0], param.grad, rtol=1e-5, atol=1e-5 * scale, msg=name
            )
        optim.step()
    for name, param in reference.named_parameters():
        assert torch.equal(results[0]["params"][name], results[1]["params"][name]), name
        torch.testing.assert_close(
            results[0]["params"][name], param.detach(), rtol=1e-5, atol=1e-6
        )
