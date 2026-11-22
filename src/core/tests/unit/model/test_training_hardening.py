import contextlib
import os
import threading
import warnings

import e3nn
import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.nn import (
    AtomwiseReduce,
    GraphModuleMixin,
    ParaStressForceSpinOutput,
    RescaleOutput,
    StressOutput,
)
import nequip.utils.compile_utils as compile_utils
import nequip.utils.fx_compat as fx_compat
import nequip.utils._global_options as global_options
import nequip.scripts.deploy as deploy_script
import nequip.scripts.evaluate as evaluate_script
from nequip.utils.fx_compat import expand_training_graph
from nequip.utils.unittests.model_tests import BaseEnergyModelTests
from nequip.utils.unittests.parity_tests import _compile_without_global_debug_hooks
from nequip.utils.unittests.parity_tests import (
    BF16_FORCE_SUPERVISION_REL_LIMIT,
    BasePrecisionLadderTests,
    PRECISION_TOLERANCES,
    _assert_bf16_force_supervision,
)


_PREBOUND_TORCH_DETACH = torch.detach
_PREBOUND_TENSOR_DETACH = torch.Tensor.detach


@contextlib.contextmanager
def _e3nn_jit_script_fx(enabled):
    previous = e3nn.get_optimization_defaults()["jit_script_fx"]
    e3nn.set_optimization_defaults(jit_script_fx=enabled)
    try:
        yield
    finally:
        e3nn.set_optimization_defaults(jit_script_fx=previous)


def _clone_data(data):
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


def _data_to_device(data, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


class _ToySpinEnergy(GraphModuleMixin, torch.nn.Module):
    def __init__(self, detach_style=None, trace_barrier=None):
        super().__init__()
        self.live_weight = torch.nn.Parameter(torch.tensor(0.7))
        self.stopped_weight = torch.nn.Parameter(torch.tensor(1.3))
        self.detach_style = detach_style
        self.trace_barrier = trace_barrier
        self._init_irreps(
            irreps_in={
                AtomicDataDict.POSITIONS_KEY: "1o",
                AtomicDataDict.SPIN_KEY: "1eo",
                AtomicDataDict.EDGE_INDEX_KEY: None,
                AtomicDataDict.EDGE_VECTORS_KEY: "1o",
                AtomicDataDict.EDGE_CELL_SHIFT_KEY: None,
                AtomicDataDict.ATOM_TYPE_KEY: None,
                AtomicDataDict.BATCH_KEY: None,
                AtomicDataDict.NUM_BATCH_KEY: None,
                AtomicDataDict.CELL_KEY: None,
                AtomicDataDict.PBC_KEY: None,
            },
            irreps_out={
                AtomicDataDict.PER_ATOM_ENERGY_KEY: "0e",
                AtomicDataDict.TOTAL_ENERGY_KEY: "0e",
            },
        )
        reduce_irreps = dict(self.irreps_in)
        reduce_irreps[AtomicDataDict.PER_ATOM_ENERGY_KEY] = "0e"
        self.reduce = AtomwiseReduce(
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            irreps_in=reduce_irreps,
        )

    def forward(self, data):
        if self.trace_barrier is not None:
            self.trace_barrier()
        pos = data[AtomicDataDict.POSITIONS_KEY]
        spin = data[AtomicDataDict.SPIN_KEY]
        edge_vectors = data[AtomicDataDict.EDGE_VECTORS_KEY]
        edge_source = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_energy = torch.zeros(
            pos.shape[0], dtype=pos.dtype, device=pos.device
        ).scatter_add(0, edge_source, edge_vectors.square().sum(dim=-1))
        stopped = self.stopped_weight * pos.square().sum(dim=-1)
        if self.detach_style == "method":
            stopped = stopped.detach()
        elif self.detach_style == "torch_alias":
            stopped = _PREBOUND_TORCH_DETACH(stopped)
        elif self.detach_style == "tensor_alias":
            stopped = _PREBOUND_TENSOR_DETACH(stopped)
        elif self.detach_style == "aten":
            stopped = torch.ops.aten.detach.default(stopped)
        atomic_energy = (
            self.live_weight * pos.square().sum(dim=-1)
            + stopped
            + 0.1 * spin.square().sum(dim=-1)
            + 0.05 * edge_energy
        ).unsqueeze(-1)
        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = atomic_energy
        return self.reduce(data)


def _training_batch(atom_counts, edges):
    n_frames = len(atom_counts)
    n_atoms = sum(atom_counts)
    batch = torch.cat(
        [
            torch.full((count,), index, dtype=torch.long)
            for index, count in enumerate(atom_counts)
        ]
    )
    positions = torch.arange(n_atoms * 3, dtype=torch.float32).reshape(n_atoms, 3)
    positions = positions * 0.07 + 0.2
    spins = torch.arange(n_atoms * 3, dtype=torch.float32).reshape(n_atoms, 3)
    spins = torch.nn.functional.normalize(spins + 0.5, dim=-1)
    edge_index = torch.tensor(edges, dtype=torch.long)
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return {
        AtomicDataDict.POSITIONS_KEY: positions,
        AtomicDataDict.SPIN_KEY: spins,
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.zeros((edge_index.shape[1], 3)),
        AtomicDataDict.ATOM_TYPE_KEY: torch.zeros(n_atoms, dtype=torch.long),
        AtomicDataDict.BATCH_KEY: batch,
        AtomicDataDict.NUM_BATCH_KEY: torch.tensor(n_frames, dtype=torch.long),
        AtomicDataDict.CELL_KEY: torch.eye(3).repeat(n_frames, 1, 1) * 8.0,
        AtomicDataDict.PBC_KEY: torch.ones((n_frames, 3), dtype=torch.bool),
    }


class _GroupedSpinEnergy(_ToySpinEnergy):
    def __init__(self):
        super().__init__()
        # Five learned channels make seven the first admissible frame prime,
        # which deliberately collides with the fixed stack width below.
        self.channels = torch.nn.Parameter(torch.linspace(0.1, 0.5, 5))

    def forward(self, data):
        data = super().forward(data)
        pos = data[AtomicDataDict.POSITIONS_KEY][:, :2]
        spin = data[AtomicDataDict.SPIN_KEY][:, :2]
        packed = torch.stack(
            [pos * ((index + 1) / 7) + spin.square() for index in range(7)],
            dim=1,
        ).flatten(1)
        extra = packed.sin().square().sum(dim=1, keepdim=True)
        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = (
            data[AtomicDataDict.PER_ATOM_ENERGY_KEY]
            + self.channels.square().sum() * extra
        )
        return self.reduce(data)


def test_full_fx_group_width_collision_preserves_dynamic_frames(monkeypatch):
    from torch._dynamo.backends.registry import lookup_backend

    torch._dynamo.reset()
    captures = []
    initial_capture_count = None
    steady_capture_count = None
    aot_backend = lookup_backend("aot_eager")
    compile_torch = torch.compile

    def backend(graph, inputs):
        captures.append(graph)
        return aot_backend(graph, inputs)

    monkeypatch.setattr(
        compile_utils.torch,
        "compile",
        lambda module, **kwargs: compile_torch(
            module, backend=backend, fullgraph=True, dynamic=kwargs["dynamic"]
        ),
    )
    eager = ParaStressForceSpinOutput(_GroupedSpinEnergy()).train()
    compiled = ParaStressForceSpinOutput(_GroupedSpinEnergy()).train()
    compiled.load_state_dict(eager.state_dict(), strict=True)
    parameter_ids = tuple(map(id, compiled.parameters()))
    dtype = torch.get_default_dtype()
    output_tolerance = 1e-5 if dtype == torch.float32 else 1e-10
    gradient_tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    fields = (
        AtomicDataDict.TOTAL_ENERGY_KEY,
        AtomicDataDict.FORCE_KEY,
        AtomicDataDict.SPIN_FORCE_VERT_KEY,
        AtomicDataDict.STRESS_KEY,
    )
    try:
        with _e3nn_jit_script_fx(False), _compile_without_global_debug_hooks():
            compile_utils.configure_model_for_compile(
                compiled,
                {
                    "spingnn_compile": {
                        "enabled": True,
                        "scope": "full_fx",
                        "dynamic": True,
                    }
                },
            )
            for sample_index, frames in enumerate((2, 3, 7, 8, 5, 2, 3, 7, 8, 5)):
                sources = tuple(
                    index
                    for frame in range(frames)
                    for index in (2 * frame, 2 * frame + 1)
                )
                targets = tuple(
                    index
                    for frame in range(frames)
                    for index in (2 * frame + 1, 2 * frame)
                )
                data = _training_batch((2,) * frames, (sources, targets))
                data = {
                    key: value.to(dtype=dtype) if value.is_floating_point() else value
                    for key, value in data.items()
                }
                expected = eager(_clone_data(data))
                actual = compiled(_clone_data(data))
                for key in fields:
                    torch.testing.assert_close(
                        actual[key],
                        expected[key],
                        rtol=output_tolerance,
                        atol=output_tolerance,
                    )
                losses = [
                    sum(output[key].square().mean() for key in fields)
                    for output in (expected, actual)
                ]
                gradients = [
                    torch.autograd.grad(
                        loss, tuple(model.parameters()), allow_unused=True
                    )
                    for loss, model in zip(losses, (eager, compiled))
                ]
                for left, right in zip(*gradients):
                    assert (left is None) == (right is None)
                    if left is not None:
                        torch.testing.assert_close(
                            right,
                            left,
                            rtol=gradient_tolerance,
                            atol=gradient_tolerance,
                        )
                if initial_capture_count is None:
                    initial_capture_count = len(captures)
                    assert initial_capture_count > 0
                elif sample_index < 4:
                    # Plain Torch 2.11 has the same three cold duck guards when
                    # a dynamic edge-view offset equals fixed offsets 4/6 or
                    # stride 14. Retain all collisions, then require reuse for
                    # an unseen size and repeated sizes after guard refinement.
                    assert len(captures) <= initial_capture_count + sample_index
                    if sample_index == 3:
                        steady_capture_count = len(captures)
                else:
                    assert len(captures) == steady_capture_count
        stats = compile_utils.get_compile_stats(compiled)
        assert (
            stats["compile_count"]
            == stats["full_fx_expand_count"]
            == stats["cache_entries"]
            == 1
        )
        assert all(sizes[0] == 7 for sizes in stats["trace_sizes"].values())
        assert tuple(map(id, compiled.parameters())) == parameter_ids
    finally:
        torch._dynamo.reset()


_EMPTY_EDGE_CASES = (
    pytest.param((1,), ((), ()), id="single-frame-empty"),
    pytest.param((2, 1), ((0, 1), (1, 0)), id="last-frame-empty"),
    pytest.param((1, 1), ((), ()), id="all-frames-empty"),
)


@pytest.mark.compile
@pytest.mark.parametrize("atom_counts,edges", _EMPTY_EDGE_CASES)
def test_full_fx_first_batch_accepts_empty_edge_boundaries(atom_counts, edges):
    if os.getenv("SPINGNN_TEST_COMPILE") != "1":
        pytest.skip("compile gate disabled")
    if torch.get_default_dtype() is not torch.float32:
        pytest.skip("empty-edge compile regression runs once in fp32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _data_to_device(_training_batch(atom_counts, edges), device)
    model = ParaStressForceSpinOutput(_ToySpinEnergy()).to(device).train()
    with _e3nn_jit_script_fx(False):
        compile_utils.configure_model_for_compile(
            model,
            {
                "spingnn_compile": {
                    "enabled": True,
                    "scope": "full_fx",
                    "dynamic": True,
                    "inductor_options": {},
                }
            },
        )
    with _compile_without_global_debug_hooks():
        output = model(_clone_data(data))

    assert output[AtomicDataDict.TOTAL_ENERGY_KEY].shape[0] == len(atom_counts)
    assert output[AtomicDataDict.STRESS_KEY].shape[0] == len(atom_counts)
    assert torch.isfinite(output[AtomicDataDict.TOTAL_ENERGY_KEY]).all()
    assert torch.isfinite(output[AtomicDataDict.FORCE_KEY]).all()


class _PerFrameEnergy(GraphModuleMixin, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._init_irreps(
            irreps_in={AtomicDataDict.POSITIONS_KEY: "1o"},
            irreps_out={AtomicDataDict.TOTAL_ENERGY_KEY: "0e"},
        )

    def forward(self, data):
        pos = data[AtomicDataDict.POSITIONS_KEY]
        batch = data[AtomicDataDict.BATCH_KEY]
        atomic = pos.sum(dim=-1)
        total = torch.zeros(
            int(batch.max().item()) + 1, dtype=pos.dtype, device=pos.device
        ).scatter_add(0, batch, atomic)
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = total.unsqueeze(-1)
        return data


class _ToySpinEnergyWithIntermediate(_ToySpinEnergy):
    def __init__(self):
        super().__init__()
        self.irreps_out[AtomicDataDict.EDGE_ENERGY_KEY] = "0e"

    def forward(self, data):
        edge_vectors = data[AtomicDataDict.EDGE_VECTORS_KEY]
        data[AtomicDataDict.EDGE_ENERGY_KEY] = edge_vectors.square().sum(
            dim=-1, keepdim=True
        )
        return super().forward(data)


def test_num_batch_mismatch_is_rejected_before_phantom_stress():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = StressOutput(_PerFrameEnergy()).eval()
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.tensor(
            [[0.2, 0.1, 0.3], [0.7, 0.2, 0.5], [0.4, 0.9, 0.8]]
        ),
        AtomicDataDict.BATCH_KEY: torch.tensor([0, 0, 1], dtype=torch.long),
        AtomicDataDict.NUM_BATCH_KEY: torch.tensor(3, dtype=torch.long),
        AtomicDataDict.EDGE_INDEX_KEY: torch.empty((2, 0), dtype=torch.long),
        AtomicDataDict.CELL_KEY: torch.eye(3).reshape(1, 3, 3) * 8.0,
        AtomicDataDict.PBC_KEY: torch.ones((1, 3), dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="num_batch.*batch indices"):
        model(data)


def _invalid_training_batch(batch_values, num_batch):
    batch = torch.tensor(batch_values, dtype=torch.long)
    n_atoms = batch.numel()
    n_frames = num_batch
    if n_frames is None:
        n_frames = int(batch.max().item()) + 1 if n_atoms else 1
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.arange(
            n_atoms * 3, dtype=torch.float32
        ).reshape(n_atoms, 3),
        AtomicDataDict.SPIN_KEY: torch.ones((n_atoms, 3)),
        AtomicDataDict.EDGE_INDEX_KEY: torch.empty((2, 0), dtype=torch.long),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.empty((0, 3)),
        AtomicDataDict.ATOM_TYPE_KEY: torch.zeros(n_atoms, dtype=torch.long),
        AtomicDataDict.BATCH_KEY: batch,
        AtomicDataDict.CELL_KEY: torch.eye(3).repeat(n_frames, 1, 1) * 8.0,
        AtomicDataDict.PBC_KEY: torch.ones((n_frames, 3), dtype=torch.bool),
    }
    if num_batch is not None:
        data[AtomicDataDict.NUM_BATCH_KEY] = torch.tensor(num_batch, dtype=torch.long)
    return data


_INVALID_BATCH_CASES = (
    pytest.param((1, 1), 2, id="missing-first-frame"),
    pytest.param((0, 2), 3, id="missing-middle-frame"),
    pytest.param((0, 2), None, id="missing-middle-frame-inferred"),
    pytest.param((-1, 0), 1, id="negative-frame"),
    pytest.param((), 1, id="empty-batch"),
)


@pytest.mark.parametrize("batch_values,num_batch", _INVALID_BATCH_CASES)
def test_invalid_batch_indices_are_rejected_in_eager_and_full_fx(
    batch_values, num_batch
):
    data = _invalid_training_batch(batch_values, num_batch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        eager_model = StressOutput(_PerFrameEnergy()).eval()
    with pytest.raises(ValueError, match="batch"):
        eager_model(_clone_data(data))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        full_fx_model = ParaStressForceSpinOutput(_ToySpinEnergy()).train()
    with pytest.raises(ValueError, match="batch"):
        expand_training_graph(full_fx_model, _clone_data(data), create_graph=True)


def _parameter_gradients(total_energy, model):
    named_parameters = tuple(model.named_parameters())
    gradients = torch.autograd.grad(
        total_energy.sum(),
        tuple(parameter for _, parameter in named_parameters),
        allow_unused=True,
    )
    return dict(zip((name for name, _ in named_parameters), gradients))


@pytest.mark.compile
@pytest.mark.parametrize("detach_style", ("torch_alias", "tensor_alias", "aten"))
def test_full_fx_preserves_all_model_author_detach_forms(detach_style):
    if os.getenv("SPINGNN_TEST_COMPILE") != "1":
        pytest.skip("compile gate disabled")
    if torch.get_default_dtype() is not torch.float32:
        pytest.skip("detach regression runs once in fp32")

    data = _training_batch((3,), ((0, 1, 1, 2, 2, 0), (1, 0, 2, 1, 0, 2)))
    model = ParaStressForceSpinOutput(_ToySpinEnergy(detach_style=detach_style)).train()

    eager_input = AtomicDataDict.with_edge_vectors(
        _clone_data(data), with_lengths=False
    )
    eager_output = model.func(eager_input)
    eager_gradients = _parameter_gradients(
        eager_output[AtomicDataDict.TOTAL_ENERGY_KEY], model
    )

    graph = expand_training_graph(model, data, create_graph=True)
    graph_input = {key: data[key] for key in graph.input_signature}
    outputs = graph(graph_input)
    energy_index = graph.output_signature.index(AtomicDataDict.TOTAL_ENERGY_KEY)
    full_fx_gradients = _parameter_gradients(outputs[energy_index], model)

    assert eager_gradients.keys() == full_fx_gradients.keys()
    for name in eager_gradients:
        eager_gradient = eager_gradients[name]
        full_fx_gradient = full_fx_gradients[name]
        if eager_gradient is None:
            assert full_fx_gradient is None, name
        else:
            assert full_fx_gradient is not None, name
            assert torch.allclose(eager_gradient, full_fx_gradient), name


@pytest.mark.compile
def test_full_fx_tracing_is_thread_safe_and_restores_global_torch_api():
    if os.getenv("SPINGNN_TEST_COMPILE") != "1":
        pytest.skip("compile gate disabled")
    if torch.get_default_dtype() is not torch.float32:
        pytest.skip("threaded trace regression runs once in fp32")

    first_entered = threading.Event()
    second_entered = threading.Event()
    first_finished = threading.Event()
    errors = []
    trace_sizes = []

    def first_barrier():
        first_entered.set()
        second_entered.wait(timeout=0.25)

    def second_barrier():
        second_entered.set()
        if not first_finished.wait(timeout=30.0):
            raise RuntimeError("first trace did not finish")

    first_data = _training_batch((3,), ((0, 1, 1, 2, 2, 0), (1, 0, 2, 1, 0, 2)))
    second_edges = []
    for atom_offset in range(0, 10, 2):
        second_edges.append((atom_offset, atom_offset + 1))
        second_edges.append((atom_offset + 1, atom_offset))
    second_data = _training_batch((2, 2, 2, 2, 2), tuple(zip(*second_edges)))
    models_and_data = (
        (
            ParaStressForceSpinOutput(
                _ToySpinEnergy(detach_style="method", trace_barrier=first_barrier)
            ).train(),
            first_data,
        ),
        (
            ParaStressForceSpinOutput(
                _ToySpinEnergy(detach_style="method", trace_barrier=second_barrier)
            ).train(),
            second_data,
        ),
    )
    original_functions = (
        torch.Tensor.detach,
        torch.detach,
        torch.Tensor.nonzero,
        fx_compat.scatter,
    )

    def expand(index):
        try:
            model, data = models_and_data[index]
            graph = expand_training_graph(model, data, create_graph=True)
            trace_sizes.append(graph.trace_sizes)
        except BaseException as exc:
            errors.append(exc)
        finally:
            if index == 0:
                first_finished.set()

    first_thread = threading.Thread(target=expand, args=(0,))
    second_thread = threading.Thread(target=expand, args=(1,))
    first_thread.start()
    assert first_entered.wait(timeout=30.0)
    second_thread.start()
    first_thread.join(timeout=60.0)
    second_thread.join(timeout=60.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not errors, errors
    assert len(trace_sizes) == 2
    assert trace_sizes[0] != trace_sizes[1]
    assert torch.Tensor.detach is original_functions[0]
    assert torch.detach is original_functions[1]
    assert torch.Tensor.nonzero is original_functions[2]
    assert fx_compat.scatter is original_functions[3]


@pytest.mark.compile
def test_full_fx_preserves_outputs_consumed_by_outer_rescale():
    if os.getenv("SPINGNN_TEST_COMPILE") != "1":
        pytest.skip("compile gate disabled")
    if torch.get_default_dtype() is not torch.float32:
        pytest.skip("outer rescale regression runs once in fp32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _data_to_device(
        _training_batch((3,), ((0, 1, 1, 2, 2, 0), (1, 0, 2, 1, 0, 2))),
        device,
    )
    target = ParaStressForceSpinOutput(_ToySpinEnergyWithIntermediate()).to(device)
    model = (
        RescaleOutput(
            model=target,
            scale_keys=[
                AtomicDataDict.TOTAL_ENERGY_KEY,
                AtomicDataDict.EDGE_ENERGY_KEY,
            ],
            scale_by=2.0,
            shift_keys=[],
            shift_by=None,
        )
        .to(device)
        .eval()
    )
    with _compile_without_global_debug_hooks():
        eager = model(
            AtomicDataDict.with_edge_vectors(_clone_data(data), with_lengths=False)
        )

    with _e3nn_jit_script_fx(False):
        compile_utils.configure_model_for_compile(
            model,
            {
                "spingnn_compile": {
                    "enabled": True,
                    "scope": "full_fx",
                    "dynamic": True,
                    "inductor_options": {},
                }
            },
        )
    with _compile_without_global_debug_hooks():
        compiled = model(_clone_data(data))

    for key in (
        AtomicDataDict.TOTAL_ENERGY_KEY,
        AtomicDataDict.EDGE_ENERGY_KEY,
    ):
        assert torch.allclose(compiled[key], eager[key]), key


@pytest.mark.compile
def test_full_fx_dynamic_batches_support_smaller_tail_and_empty_edges():
    if os.getenv("SPINGNN_TEST_COMPILE") != "1":
        pytest.skip("compile gate disabled")
    if torch.get_default_dtype() is not torch.float32:
        pytest.skip("dynamic batch regression runs once in fp32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = tuple(
        _data_to_device(data, device)
        for data in (
            _training_batch(
                (2, 2),
                ((0, 1, 1, 0, 2, 3, 3, 2), (1, 0, 0, 1, 3, 2, 2, 3)),
            ),
            _training_batch((1,), ((), ())),
            _training_batch(
                (3, 1),
                ((0, 1, 1, 2, 2, 0), (1, 0, 2, 1, 0, 2)),
            ),
        )
    )
    eager = ParaStressForceSpinOutput(_ToySpinEnergy()).to(device).train()
    compiled = ParaStressForceSpinOutput(_ToySpinEnergy()).to(device).train()
    compiled.load_state_dict(eager.state_dict())
    with _e3nn_jit_script_fx(False):
        compile_utils.configure_model_for_compile(
            compiled,
            {
                "spingnn_compile": {
                    "enabled": True,
                    "scope": "full_fx",
                    "dynamic": True,
                    "inductor_options": {},
                }
            },
        )

    with _compile_without_global_debug_hooks():
        for data in batches:
            eager_output = eager(_clone_data(data))
            compiled_output = compiled(_clone_data(data))
            for key in (
                AtomicDataDict.TOTAL_ENERGY_KEY,
                AtomicDataDict.FORCE_KEY,
                AtomicDataDict.STRESS_KEY,
            ):
                assert torch.allclose(
                    compiled_output[key], eager_output[key], atol=1.0e-5, rtol=1.0e-5
                ), key

    stats = compile_utils.get_compile_stats(compiled)
    assert stats["compile_count"] == 2
    assert stats["full_fx_expand_count"] == 2
    assert stats["cache_entries"] == 2
    assert stats["cache_entries"] <= compile_utils._MAX_FULL_FX_CACHE_ENTRIES


def test_reconfiguring_full_fx_rebinds_forward_to_new_state(monkeypatch):
    # The production selector uses the exact class name.
    target_type = type(
        "ParaStressForceSpinOutput",
        (torch.nn.Module,),
        {"forward": lambda self, data: data},
    )
    model = target_type()
    config = {
        "spingnn_compile": {
            "enabled": True,
            "scope": "full_fx",
            "dynamic": True,
            "inductor_options": {},
        }
    }
    with _e3nn_jit_script_fx(False):
        compile_utils.configure_model_for_compile(model, config)
        config["spingnn_compile"]["dynamic"] = False
        compile_utils.configure_model_for_compile(model, config)

    monkeypatch.setattr(
        compile_utils,
        "_call_full_fx",
        lambda output_module, data, state: state,
    )
    state_used_by_forward = model({})
    assert state_used_by_forward is model._spingnn_compile_state
    assert state_used_by_forward.dynamic is False

    config["spingnn_compile"]["enabled"] = False
    with _e3nn_jit_script_fx(False):
        compile_utils.configure_model_for_compile(model, config)
    payload = {"sentinel": object()}
    assert model(payload) is payload
    assert not hasattr(model, "_spingnn_compile_state")
    assert not hasattr(model, "_spingnn_full_fx_wrapped")


@pytest.mark.parametrize("action", ("reconfigure", "disable"))
def test_inflight_forward_cannot_return_stale_compile_state(monkeypatch, action):
    target_type = type(
        "ParaStressForceSpinOutput",
        (torch.nn.Module,),
        {"forward": lambda self, data: data},
    )
    model = target_type()
    config = {
        "spingnn_compile": {
            "enabled": True,
            "scope": "full_fx",
            "dynamic": True,
            "inductor_options": {},
        }
    }
    with _e3nn_jit_script_fx(False):
        compile_utils.configure_model_for_compile(model, config)

    entered = threading.Event()
    release = threading.Event()
    results = []
    errors = []

    def blocked_call(output_module, data, state):
        entered.set()
        if not release.wait(timeout=10.0):
            raise RuntimeError("in-flight test timed out")
        return state

    monkeypatch.setattr(compile_utils, "_call_full_fx", blocked_call)

    def run_forward():
        try:
            results.append(model({}))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_forward)
    thread.start()
    assert entered.wait(timeout=10.0)
    if action == "reconfigure":
        config["spingnn_compile"]["dynamic"] = False
    else:
        config["spingnn_compile"]["enabled"] = False
    with _e3nn_jit_script_fx(False):
        compile_utils.configure_model_for_compile(model, config)
    release.set()
    thread.join(timeout=10.0)

    assert not thread.is_alive()
    assert not results
    assert len(errors) == 1
    assert "configuration changed during forward" in str(errors[0])


def test_bf16_recording_and_production_gates_are_distinct():
    errors = {
        "energy_per_atom_ev": 1.0e-3,
        AtomicDataDict.FORCE_KEY: 7.755e-2,
        f"{AtomicDataDict.FORCE_KEY}_absolute": 1.0e-2,
    }
    BasePrecisionLadderTests._assert_precision_errors(
        errors, PRECISION_TOLERANCES["bf16"], "bf16_recording"
    )
    assert errors[AtomicDataDict.FORCE_KEY] > BF16_FORCE_SUPERVISION_REL_LIMIT
    with pytest.raises(AssertionError, match="bf16 production"):
        _assert_bf16_force_supervision(errors)


def test_bf16_gpu_lane_accepts_production_quality_improvement(monkeypatch):
    if torch.get_default_dtype() is not torch.float32:
        pytest.skip("bf16 production improvement runs once in fp32")
    safe_errors = {
        "energy_per_atom_ev": 1.0e-3,
        AtomicDataDict.FORCE_KEY: 5.0e-3,
        f"{AtomicDataDict.FORCE_KEY}_absolute": 1.0e-3,
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        BasePrecisionLadderTests,
        "_run_precision_case",
        lambda *args, **kwargs: safe_errors,
    )

    BasePrecisionLadderTests().test_bf16_gpu_precision(None, None)


class _InjectedNondeterministicModel(torch.nn.Module):
    calls = 0

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, data):
        type(self).calls += 1
        jitter = data[AtomicDataDict.POSITIONS_KEY].new_tensor(float(type(self).calls))
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = (
            self.weight * data[AtomicDataDict.POSITIONS_KEY].sum() + jitter
        ).reshape(1, 1)
        return data


class _DeterminismGateWithInjectedFault(BaseEnergyModelTests):
    @staticmethod
    def make_model(config, device, initialize=True, deploy=False):
        return _InjectedNondeterministicModel().to(device)


class _InjectedGradientNondeterministicModel(torch.nn.Module):
    calls = 0

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, data):
        type(self).calls += 1
        gradient_only_jitter = float(type(self).calls) * (
            self.weight - self.weight.detach()
        )
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = (
            self.weight * data[AtomicDataDict.POSITIONS_KEY].sum()
            + gradient_only_jitter
        ).reshape(1, 1)
        return data


class _DeterminismGateWithInjectedGradientFault(BaseEnergyModelTests):
    @staticmethod
    def make_model(config, device, initialize=True, deploy=False):
        return _InjectedGradientNondeterministicModel().to(device)


def _assert_expected_determinism_failure(test_case, config, expected_message):
    try:
        test_case.test_determinism(
            config=(config, (AtomicDataDict.TOTAL_ENERGY_KEY,)),
            device=torch.device("cpu"),
            add_spin=False,
        )
    except AssertionError as exc:
        if str(exc) != expected_message:
            raise
    else:
        raise AssertionError("determinism mismatch did not fail")


def test_allegro_determinism_mismatch_fails_instead_of_skips():
    _InjectedNondeterministicModel.calls = 0
    config = {
        "r_max": 4.0,
        "model_builders": ["allegro.model.InjectedNondeterministic"],
    }
    _assert_expected_determinism_failure(
        _DeterminismGateWithInjectedFault(), config, "output total_energy"
    )


def test_allegro_gradient_determinism_mismatch_fails():
    _InjectedGradientNondeterministicModel.calls = 0
    config = {
        "r_max": 4.0,
        "model_builders": ["allegro.model.InjectedGradientNondeterministic"],
    }
    _assert_expected_determinism_failure(
        _DeterminismGateWithInjectedGradientFault(), config, "gradient weight"
    )


def test_determinism_regression_does_not_swallow_unrelated_assertion(monkeypatch):
    config = {
        "r_max": 4.0,
        "model_builders": ["allegro.model.InjectedNondeterministic"],
    }
    monkeypatch.setattr(
        _DeterminismGateWithInjectedFault,
        "test_determinism",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unrelated setup failure")
        ),
    )
    with pytest.raises(AssertionError, match="unrelated setup failure"):
        _assert_expected_determinism_failure(
            _DeterminismGateWithInjectedFault(), config, "output total_energy"
        )


def test_global_precision_context_restores_matmul_and_tf32(monkeypatch):
    original_precision = torch.get_float32_matmul_precision()
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with global_options.global_options_context(
            {
                "float32_matmul_precision": "highest",
                "allow_tf32": False,
            }
        ):
            assert torch.get_float32_matmul_precision() == "highest"
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


@pytest.mark.parametrize("script_module", (evaluate_script, deploy_script))
def test_nontraining_cli_restores_global_precision(monkeypatch, script_module):
    original_precision = torch.get_float32_matmul_precision()
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    sentinel = object()

    def set_validation_precision(*args, **kwargs):
        global_options._set_global_options(
            {
                "float32_matmul_precision": "highest",
                "allow_tf32": False,
            }
        )
        return sentinel

    monkeypatch.setattr(script_module, "_main", set_validation_precision)
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        assert script_module.main() is sentinel
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


# --- GV3-F1 regression: mixed legacy/new TF32 APIs must not break global setup ---
#
# On CUDA with torch >= 2.11, `torch.get_float32_matmul_precision()` raises
# RuntimeError once any per-backend fp32 precision has been configured, which is
# what the `allow_tf32` switch does via `torch.backends.cuda.matmul.allow_tf32`.
# Within one `_apply_global_options` call the new-API read happens before the
# legacy write, so the FIRST call succeeds and every SUBSEQUENT call raises.
# `torch.cuda.is_available()` gates the legacy write, so CPU CI cannot reach the
# real tainted state -- these tests install the taint explicitly so the contract
# is enforced on every runner, not only on a GPU.

_TAINTED_MATMUL_PRECISION_MESSAGE = (
    "PyTorch is checking the matmul precision without a specific backend name,"
    "Current status indicate that you have used mix of the legacy and new APIs "
    "to set the matmul precision."
)


def _taint_global_matmul_precision_getter(monkeypatch):
    """Make the global getter behave as it does on a tainted CUDA process."""

    def _raise():
        raise RuntimeError(_TAINTED_MATMUL_PRECISION_MESSAGE)

    monkeypatch.setattr(torch, "get_float32_matmul_precision", _raise)


def test_set_global_options_survives_tainted_matmul_precision_getter(monkeypatch):
    """A second `_set_global_options` call must not raise on CUDA."""
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    config = {"float32_matmul_precision": "high", "allow_tf32": True}
    try:
        _taint_global_matmul_precision_getter(monkeypatch)
        # Old code raised RuntimeError here; the fix degrades the read to "unknown".
        global_options._set_global_options(config)
        global_options._set_global_options(config)
        # The TF32 switch must still have been applied.
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def test_set_global_options_warns_only_when_previous_precision_is_readable(
    monkeypatch,
):
    """An unreadable previous value must not produce a bogus override warning."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _taint_global_matmul_precision_getter(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        global_options._set_global_options(
            {"float32_matmul_precision": "high"}, warn_on_override=True
        )


def test_global_options_context_survives_tainted_matmul_precision_getter(
    monkeypatch,
):
    """The scoped precision context must enter and exit on a tainted process."""
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        _taint_global_matmul_precision_getter(monkeypatch)
        # Old code raised RuntimeError on entry, before yielding.
        with global_options.global_options_context({"allow_tf32": False}):
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
        # The legacy flags stay readable, so they are still restored exactly.
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


# --- RV4A-M1/M2: tolerant-restore granularity and bookkeeping honesty ---
#
# M1: `_read_legacy_allow_tf32` collapsed BOTH backend flags into one tuple, so
# when only one getter raised, the other backend's readable value was discarded
# and neither flag was restored on context exit -- a readable knob leaked.
# M2: the context's `finally` always restored `_latest_global_config` to the
# pre-context dict, even when the real knob could NOT be restored; worker
# processes initialised from that dict then disagree with the parent's real
# torch state.


class _ReadableTF32Backend:
    def __init__(self, value):
        self._value = value
        self.writes = []
        self.deterministic = False
        self.benchmark = False

    @property
    def allow_tf32(self):
        return self._value

    @allow_tf32.setter
    def allow_tf32(self, value):
        self._value = value
        self.writes.append(value)


class _RaisingTF32Backend:
    def __init__(self):
        self._value = None
        self.writes = []
        self.deterministic = False
        self.benchmark = False

    @property
    def allow_tf32(self):
        raise RuntimeError(_TAINTED_MATMUL_PRECISION_MESSAGE)

    @allow_tf32.setter
    def allow_tf32(self, value):
        self._value = value
        self.writes.append(value)


@pytest.mark.parametrize("raising_side", ("cudnn", "matmul"))
def test_partial_legacy_getter_failure_still_restores_the_readable_flag(
    monkeypatch, raising_side
):
    """RV4A-M1: a readable backend flag must survive the scoped context."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    readable = _ReadableTF32Backend(True)
    raising = _RaisingTF32Backend()
    if raising_side == "cudnn":
        monkeypatch.setattr(torch.backends.cuda, "matmul", readable)
        monkeypatch.setattr(torch.backends, "cudnn", raising)
    else:
        monkeypatch.setattr(torch.backends.cuda, "matmul", raising)
        monkeypatch.setattr(torch.backends, "cudnn", readable)

    with global_options.global_options_context({"allow_tf32": False}):
        assert readable._value is False
        assert raising._value is False

    assert readable._value is True, (
        "the readable backend's pre-context value must be restored even when "
        "the other backend's getter raises"
    )


def test_unrestorable_precision_does_not_falsify_worker_bookkeeping(monkeypatch):
    """RV4A-M2: `_latest_global_config` must describe the SURVIVING state."""
    recorded = []
    monkeypatch.setattr(
        torch,
        "get_float32_matmul_precision",
        lambda: (_ for _ in ()).throw(RuntimeError(_TAINTED_MATMUL_PRECISION_MESSAGE)),
    )
    monkeypatch.setattr(
        torch, "set_float32_matmul_precision", lambda value: recorded.append(value)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        global_options,
        "_latest_global_config",
        {"float32_matmul_precision": "highest"},
    )

    with global_options.global_options_context({"float32_matmul_precision": "medium"}):
        pass

    # The old value could not be read, so it was not restored: the last real
    # write is the body's "medium" and it survives context exit.
    assert recorded[-1] == "medium"
    surviving = global_options._get_latest_global_options().get(
        "float32_matmul_precision"
    )
    assert surviving != "highest", (
        "bookkeeping claims the pre-context precision although the real torch "
        "state kept the body's value; workers initialised from this dict would "
        "diverge from the parent"
    )
    assert surviving in ("medium", None)
