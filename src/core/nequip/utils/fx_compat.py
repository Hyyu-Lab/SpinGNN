"""FX compatibility helpers for tensor-only SpinGNN inference graphs."""

from __future__ import annotations

import copy
import builtins
import contextlib
import importlib
import math
import sys
import threading
from typing import Dict, Iterator, Tuple, Union

import torch
from torch import fx
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx
from torch_runstats.scatter import scatter

from nequip.data import AtomicDataDict


SPIN_INPUT_SIGNATURE: Tuple[str, ...] = (
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.SPIN_KEY,
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
)

# Backward-compatible name for the established spin pair ABI.
INPUT_SIGNATURE = SPIN_INPUT_SIGNATURE

NON_SPIN_INPUT_SIGNATURE: Tuple[str, ...] = (
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
)

PERIODIC_INPUT_SIGNATURE: Tuple[str, ...] = (
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.SPIN_KEY,
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
    AtomicDataDict.CELL_KEY,
    AtomicDataDict.PBC_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
)

_ORDERED_OUTPUT_KEYS: Tuple[str, ...] = (
    AtomicDataDict.TOTAL_ENERGY_KEY,
    AtomicDataDict.FORCE_KEY,
    AtomicDataDict.SPIN_FORCE_KEY,
    AtomicDataDict.PER_ATOM_ENERGY_KEY,
    AtomicDataDict.VIRIAL_KEY,
    AtomicDataDict.ATOM_VIRIAL_KEY,
    AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY,
    AtomicDataDict.TOTAL_BORN_EFFECTIVE_CHARGE_KEY,
)

_TRAINING_OUTPUT_KEYS: Tuple[str, ...] = (
    AtomicDataDict.TOTAL_ENERGY_KEY,
    AtomicDataDict.PER_ATOM_ENERGY_KEY,
    AtomicDataDict.FORCE_KEY,
    AtomicDataDict.SPIN_FORCE_KEY,
    AtomicDataDict.SPIN_FORCE_VERT_KEY,
    AtomicDataDict.STRESS_KEY,
    AtomicDataDict.VIRIAL_KEY,
    AtomicDataDict.ATOM_VIRIAL_KEY,
    AtomicDataDict.SPIN_LENGTH_KEY,
)

_TRAINING_STRUCTURE_KEYS = {
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.SPIN_KEY,
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
    AtomicDataDict.CELL_KEY,
    AtomicDataDict.PBC_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
    AtomicDataDict.ATOMIC_NUMBERS_KEY,
    AtomicDataDict.SPIN_LENGTH_KEY,
    AtomicDataDict.BATCH_KEY,
    AtomicDataDict.NUM_BATCH_KEY,
}

_ATOM_INPUT_KEYS = {
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.SPIN_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
    AtomicDataDict.ATOMIC_NUMBERS_KEY,
    AtomicDataDict.SPIN_LENGTH_KEY,
}

_EDGE_INPUT_KEYS = {
    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
}

_FRAME_INPUT_KEYS = {
    AtomicDataDict.CELL_KEY,
    AtomicDataDict.PBC_KEY,
}

SpinSample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
NonSpinSample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Sample = Union[SpinSample, NonSpinSample]
PeriodicSample = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
_TRACE_N_ATOMS = None
_TRACE_N_FRAMES = None
_FX_TRACE_LOCK = threading.RLock()


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    for divisor in range(2, math.isqrt(value) + 1):
        if value % divisor == 0:
            return False
    return True


def _next_allowed_prime(start: int, forbidden: set[int]) -> int:
    candidate = max(5, start)
    while not _is_prime(candidate) or candidate in forbidden:
        candidate += 1
    return candidate


def _model_dimensions(model: torch.nn.Module) -> set[int]:
    dimensions: set[int] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        dimensions.update(int(size) for size in tensor.shape)
    return dimensions


def _model_dimension_prime_factors(model: torch.nn.Module) -> set[int]:
    factors: set[int] = set()
    for dimension in _model_dimensions(model):
        candidate = 2
        remaining = dimension
        while candidate * candidate <= remaining:
            if remaining % candidate == 0:
                factors.add(candidate)
                while remaining % candidate == 0:
                    remaining //= candidate
            candidate += 1
        if remaining > 1:
            factors.add(remaining)
    return factors


def _model_num_types(model: torch.nn.Module) -> int:
    candidates = [
        int(module.num_types)
        for module in model.modules()
        if hasattr(module, "num_types") and int(module.num_types) > 0
    ]
    if not candidates:
        raise ValueError("cannot infer num_types from model")
    return min(candidates)


def _model_device_dtype(model: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_floating_point():
            return tensor.device, tensor.dtype
    return torch.device("cpu"), torch.get_default_dtype()


def _model_has_spin(model: torch.nn.Module) -> bool:
    input_has_spin = AtomicDataDict.SPIN_KEY in getattr(model, "irreps_in", {})
    output_has_spin = AtomicDataDict.SPIN_FORCE_KEY in getattr(model, "irreps_out", {})
    if input_has_spin != output_has_spin:
        raise ValueError(
            "PT2 export requires spin input and spin_forces output capabilities "
            f"to match; got spin input={input_has_spin}, "
            f"spin_forces output={output_has_spin}"
        )
    return output_has_spin


def _make_spin_deploy_sample(model: torch.nn.Module) -> SpinSample:
    """Create a deterministic contiguous sample with collision-free prime sizes."""
    forbidden = _model_dimensions(model)
    # Small primes commonly collide with non-persistent e3nn codegen constants.
    n_atoms = _next_allowed_prime(101, forbidden)
    n_edges = _next_allowed_prime(n_atoms + 1, forbidden | {n_atoms})
    device, dtype = _model_device_dtype(model)
    generator = torch.Generator(device=device)
    generator.manual_seed(20260827)
    pos = 0.25 * torch.randn(
        (n_atoms, 3), generator=generator, device=device, dtype=dtype
    )
    spin = torch.randn((n_atoms, 3), generator=generator, device=device, dtype=dtype)
    spin = torch.nn.functional.normalize(spin, dim=-1)
    edge_ids = torch.arange(n_edges, device=device, dtype=torch.long)
    source = edge_ids.remainder(n_atoms)
    destination = (source + 1 + edge_ids.div(n_atoms, rounding_mode="floor")).remainder(
        n_atoms
    )
    edge_index = torch.stack((source, destination), dim=0)
    atom_types = torch.arange(n_atoms, device=device, dtype=torch.long).remainder(
        _model_num_types(model)
    )
    return tuple(
        tensor.contiguous() for tensor in (pos, spin, edge_index, atom_types)
    )  # type: ignore[return-value]


def make_deploy_sample(model: torch.nn.Module) -> Sample:
    """Create the model-specific non-periodic PT2 ABI sample."""
    pos, spin, edge_index, atom_types = _make_spin_deploy_sample(model)
    if _model_has_spin(model):
        return pos, spin, edge_index, atom_types
    return pos, edge_index, atom_types


def make_periodic_deploy_sample(model: torch.nn.Module) -> PeriodicSample:
    """Create the tensor-only sample used by periodic NequIP deployment."""
    pos, spin, edge_index, atom_types = _make_spin_deploy_sample(model)
    edge_cell_shift = torch.zeros(
        (edge_index.shape[1], 3), dtype=pos.dtype, device=pos.device
    )
    cell = torch.eye(3, dtype=pos.dtype, device=pos.device) * 8.0
    pbc = torch.ones((1, 3), dtype=torch.bool, device=pos.device)
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
    )  # type: ignore[return-value]


def output_signature(model: torch.nn.Module) -> Tuple[str, ...]:
    available = set(getattr(model, "irreps_out", {}))
    return tuple(key for key in _ORDERED_OUTPUT_KEYS if key in available)


def training_output_signature(model: torch.nn.Module) -> Tuple[str, ...]:
    """Return the stable output order used across the full-FX boundary."""
    available = set(getattr(model, "irreps_out", {}))
    signature = tuple(key for key in _TRAINING_OUTPUT_KEYS if key in available)
    signature += tuple(
        key
        for key in AtomicDataDict.ALL_ENERGY_KEYS
        if key in available and key not in signature
    )
    required = {
        AtomicDataDict.TOTAL_ENERGY_KEY,
        AtomicDataDict.PER_ATOM_ENERGY_KEY,
        AtomicDataDict.FORCE_KEY,
        AtomicDataDict.SPIN_FORCE_KEY,
        AtomicDataDict.SPIN_FORCE_VERT_KEY,
        AtomicDataDict.STRESS_KEY,
        AtomicDataDict.VIRIAL_KEY,
    }
    missing = sorted(required - set(signature))
    if missing:
        raise ValueError(
            "full-FX training model is missing required outputs: " + ", ".join(missing)
        )
    return signature


def _training_inputs(model: torch.nn.Module, sample_batch) -> Dict[str, torch.Tensor]:
    allowed = _TRAINING_STRUCTURE_KEYS | set(getattr(model, "irreps_in", {}))
    sample = {
        key: value
        for key, value in sample_batch.items()
        if key in allowed and isinstance(value, torch.Tensor)
    }
    required = {
        AtomicDataDict.POSITIONS_KEY,
        AtomicDataDict.SPIN_KEY,
        AtomicDataDict.EDGE_INDEX_KEY,
        AtomicDataDict.ATOM_TYPE_KEY,
        AtomicDataDict.BATCH_KEY,
        AtomicDataDict.NUM_BATCH_KEY,
        AtomicDataDict.CELL_KEY,
        AtomicDataDict.PBC_KEY,
    }
    missing = sorted(required - set(sample))
    if missing:
        raise ValueError(
            "full-FX training batch is missing required keys: " + ", ".join(missing)
        )
    if AtomicDataDict.EDGE_CELL_SHIFT_KEY not in sample:
        raise ValueError("full-FX periodic training requires edge_cell_shift")
    if sample[AtomicDataDict.EDGE_INDEX_KEY].dtype is not torch.int64:
        raise TypeError("full-FX edge_index must remain int64")
    if sample[AtomicDataDict.BATCH_KEY].dtype is not torch.int64:
        raise TypeError("full-FX batch indices must remain int64")
    if sample[AtomicDataDict.NUM_BATCH_KEY].dtype is not torch.int64:
        raise TypeError("full-FX num_batch must remain int64")
    return sample


def _repeat_rows(value: torch.Tensor, count: int) -> torch.Tensor:
    if count == 0:
        return value.new_empty((0, *value.shape[1:]))
    if value.shape[0] == 0:
        raise ValueError("cannot pad a training dimension from an empty template")
    indices = torch.arange(count, device=value.device, dtype=torch.long)
    return value.index_select(0, indices.remainder(value.shape[0]))


def _distributed_counts(total: int, groups: int, minimum: int) -> list[int]:
    if groups <= 0:
        if total != 0:
            raise ValueError("padding count has no destination frames")
        return []
    if total < groups * minimum:
        raise ValueError("prime padding is too small for valid frame replicas")
    counts = [minimum] * groups
    for index in range(total - groups * minimum):
        counts[index % groups] += 1
    return counts


def _training_prime_sizes(
    model: torch.nn.Module,
    n_frames: int,
    n_atoms: int,
    n_edges: int,
) -> tuple[int, int, int]:
    forbidden = _model_dimensions(model) | _model_dimension_prime_factors(model)
    padded_frames = _next_allowed_prime(n_frames + 1, forbidden)
    new_frames = padded_frames - n_frames
    padded_atoms = _next_allowed_prime(
        n_atoms + 2 * new_frames,
        forbidden | {padded_frames},
    )
    padded_edges = _next_allowed_prime(
        n_edges + 2 * new_frames,
        forbidden | {padded_frames, padded_atoms},
    )
    return padded_frames, padded_atoms, padded_edges


def validate_training_batch(model, sample_batch):
    """Validate every full-FX call, including hits on an already captured graph.

    Frame count comes from the required per-frame cell metadata. Aggregate
    value checks before the host transfer so warm CUDA calls do not perform
    a separate synchronization for every index/periodicity/volume predicate.
    """
    sample = _training_inputs(model, sample_batch)
    pos = sample[AtomicDataDict.POSITIONS_KEY]
    spin = sample[AtomicDataDict.SPIN_KEY]
    edge_index = sample[AtomicDataDict.EDGE_INDEX_KEY]
    batch = sample[AtomicDataDict.BATCH_KEY]
    num_batch = sample[AtomicDataDict.NUM_BATCH_KEY]
    cell = sample[AtomicDataDict.CELL_KEY]
    pbc = sample[AtomicDataDict.PBC_KEY]
    if num_batch.ndim != 0:
        raise ValueError("full-FX num_batch must be a scalar tensor")
    if pos.ndim != 2 or pos.shape[1] != 3 or spin.shape != pos.shape:
        raise ValueError("full-FX positions and spin must have shape (atoms, 3)")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("full-FX edge_index must have shape (2, edges)")
    if cell.ndim != 3 or cell.shape[1:] != (3, 3) or cell.shape[0] == 0:
        raise ValueError("full-FX cell must have shape (frames, 3, 3)")
    if pbc.shape != (cell.shape[0], 3):
        raise ValueError("full-FX pbc must have shape (frames, 3)")
    if not pos.is_floating_point() or not cell.is_floating_point():
        raise TypeError("full-FX positions and cell must have floating point dtype")
    n_frames, n_atoms, n_edges = cell.shape[0], pos.shape[0], edge_index.shape[1]
    if batch.ndim != 1 or batch.numel() == 0:
        raise ValueError("full-FX batch/num_batch requires a non-empty atom batch")
    if batch.shape[0] != n_atoms:
        raise ValueError("full-FX batch length must match the number of atoms")
    for key, value in sample.items():
        if value.device != pos.device:
            raise ValueError("full-FX structure tensors must share one device")
        size = (
            n_atoms
            if key in _ATOM_INPUT_KEYS
            else (
                n_edges
                if key in _EDGE_INPUT_KEYS
                else (n_frames if key in _FRAME_INPUT_KEYS else None)
            )
        )
        if size is not None and (value.ndim == 0 or value.shape[0] != size):
            raise ValueError(f"full-FX tensor {key!r} has an invalid leading size")
    if sample[AtomicDataDict.EDGE_CELL_SHIFT_KEY].shape != (n_edges, 3):
        raise ValueError("full-FX edge_cell_shift must have shape (edges, 3)")
    # Clamp only the validation input: malformed indices must never reach a
    # scatter kernel or cause an unbounded bincount allocation on the device.
    counts = torch.bincount(batch.clamp(0, n_frames - 1), minlength=n_frames)
    volume = torch.abs(
        torch.einsum(
            "zi,zi->z",
            cell[:, 0, :],
            torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
        )
    )
    checks = (
        (num_batch != n_frames, "full-FX num_batch does not match cell frames"),
        (
            ((batch < 0) | (batch >= n_frames)).any(),
            "full-FX batch indices must start at zero and match num_batch",
        ),
        ((counts == 0).any(), "full-FX batch indices must densely cover every frame"),
        (
            ((edge_index < 0) | (edge_index >= n_atoms)).any(),
            "full-FX edge_index contains an out-of-range atom index",
        ),
        ((~pbc.bool()).any(), "full-FX stress requires fully periodic training frames"),
        (
            (~torch.isfinite(volume) | (volume <= torch.finfo(cell.dtype).eps)).any(),
            "full-FX stress requires finite, non-zero cell volumes",
        ),
    )
    failed = torch.stack([condition for condition, _ in checks]).tolist()
    for invalid, (_, message) in zip(failed, checks):
        if invalid:
            raise ValueError(message)
    return sample, (n_frames, n_atoms, n_edges)


def _pad_training_batch(
    model: torch.nn.Module, sample_batch
) -> Dict[str, torch.Tensor]:
    """Pad frame, atom, and edge axes to distinct collision-free primes."""
    sample, (n_frames, n_atoms, n_edges) = validate_training_batch(model, sample_batch)
    edge_index = sample[AtomicDataDict.EDGE_INDEX_KEY]
    batch = sample[AtomicDataDict.BATCH_KEY]
    padded_frames, padded_atoms, padded_edges = _training_prime_sizes(
        model, n_frames, n_atoms, n_edges
    )
    extra_frames = padded_frames - n_frames
    atom_counts = _distributed_counts(padded_atoms - n_atoms, extra_frames, 2)
    edge_counts = _distributed_counts(padded_edges - n_edges, extra_frames, 2)

    if n_edges > 0:
        edge_frames = batch.index_select(0, edge_index[0])
        template_frame = int(edge_frames[0].item())
        template_edge_indices = torch.nonzero(edge_frames == template_frame).flatten()
    else:
        template_frame = n_frames - 1
        template_edge_indices = torch.empty(
            (0,), dtype=torch.long, device=edge_index.device
        )
    template_atom_indices = torch.nonzero(batch == template_frame).flatten()
    if template_atom_indices.numel() == 0:
        raise ValueError("full-FX padding template frame has no atoms")

    atom_lookup = torch.full((n_atoms,), -1, dtype=torch.long, device=edge_index.device)
    atom_lookup[template_atom_indices] = torch.arange(
        template_atom_indices.numel(), device=edge_index.device, dtype=torch.long
    )
    if template_edge_indices.numel() > 0:
        template_edges = atom_lookup[edge_index.index_select(1, template_edge_indices)]
        if bool((template_edges < 0).any()):
            raise ValueError("padding template contains cross-frame edges")
    else:
        template_edges = torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.long, device=edge_index.device
        )

    appended_batch = []
    appended_edges = []
    appended_edge_templates = []
    atom_offset = n_atoms
    for frame_offset, (atom_count, edge_count) in enumerate(
        zip(atom_counts, edge_counts), start=1
    ):
        frame_index = n_frames + frame_offset - 1
        appended_batch.append(
            torch.full(
                (atom_count,), frame_index, dtype=torch.long, device=batch.device
            )
        )
        template_indices = torch.arange(
            edge_count, device=edge_index.device, dtype=torch.long
        )
        template_slots = template_indices.remainder(template_edges.shape[1])
        replica_shift = template_indices.div(
            template_edges.shape[1], rounding_mode="floor"
        )
        local_edges = (
            template_edges.index_select(1, template_slots) + replica_shift.unsqueeze(0)
        ).remainder(atom_count)
        appended_edges.append(local_edges + atom_offset)
        appended_edge_templates.append(
            template_edge_indices.index_select(0, template_slots)
            if template_edge_indices.numel() > 0
            else None
        )
        atom_offset += atom_count

    atom_template = {
        key: value.index_select(0, template_atom_indices)
        for key, value in sample.items()
        if key in _ATOM_INPUT_KEYS
    }
    padded: Dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if key == AtomicDataDict.BATCH_KEY:
            padded[key] = torch.cat((value, *appended_batch), dim=0).contiguous()
        elif key == AtomicDataDict.NUM_BATCH_KEY:
            padded[key] = value.new_tensor(padded_frames).contiguous()
        elif key == AtomicDataDict.EDGE_INDEX_KEY:
            padded[key] = torch.cat((value, *appended_edges), dim=1).contiguous()
        elif key in _ATOM_INPUT_KEYS:
            extras = [_repeat_rows(atom_template[key], count) for count in atom_counts]
            padded[key] = torch.cat((value, *extras), dim=0).contiguous()
        elif key in _EDGE_INPUT_KEYS:
            extras = [
                value.index_select(0, indices)
                if indices is not None
                else value.new_zeros((count, *value.shape[1:]))
                for count, indices in zip(edge_counts, appended_edge_templates)
            ]
            padded[key] = torch.cat((value, *extras), dim=0).contiguous()
        elif key in _FRAME_INPUT_KEYS:
            if value.shape[0] != n_frames:
                raise ValueError(
                    f"full-FX frame tensor {key!r} must have leading size {n_frames}"
                )
            padded[key] = torch.cat(
                (value, _repeat_rows(value[-1:], extra_frames)), dim=0
            ).contiguous()
        elif value.ndim == 0:
            padded[key] = value.clone().contiguous()
        else:
            matches = [
                name
                for name, size in (
                    ("atom", n_atoms),
                    ("edge", n_edges),
                    ("frame", n_frames),
                )
                if value.shape[0] == size
            ]
            if matches:
                raise ValueError(
                    f"full-FX cannot classify training tensor {key!r} on {matches} axis"
                )
            padded[key] = value.clone().contiguous()

    assert padded[AtomicDataDict.POSITIONS_KEY].shape[0] == padded_atoms
    assert padded[AtomicDataDict.EDGE_INDEX_KEY].shape[1] == padded_edges
    return padded


def _validate_cutoff_contract(model: torch.nn.Module) -> None:
    for module in model.modules():
        cutoffs = getattr(module, "per_layer_cutoffs", None)
        if cutoffs is None or cutoffs.numel() < 2:
            continue
        if not torch.equal(cutoffs, cutoffs[0].expand_as(cutoffs)):
            if getattr(module, "edge_prune", "nonzero") == "dense":
                continue
            raise ValueError(
                "PT2 export does not yet support decreasing per-layer cutoffs"
            )


def _uses_source_dense_edge_prune(model: torch.nn.Module) -> bool:
    modes = {
        module.edge_prune for module in model.modules() if hasattr(module, "edge_prune")
    }
    if len(modes) > 1:
        raise ValueError(f"model has inconsistent edge_prune modes: {sorted(modes)}")
    return modes == {"dense"}


def _all_indices_nonzero(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() != 1:
        raise ValueError("PT2 nonzero decomposition only supports 1D cutoff masks")
    return torch.arange(mask.shape[0], device=mask.device).unsqueeze(-1)


def _nequip_polynomial_cutoff(x: torch.Tensor, factor: float, p: float = 6.0):
    x = x * factor
    assert p >= 2.0
    p_int = int(p)
    assert float(p_int) == p
    out = float(p_int * (p_int + 1) // 2) * torch.ones_like(x)
    for k in range(p_int - 2, -1, -1):
        out = out * x + float((k + 1) * (k + 2) // 2)
    tail = 1.0 - x
    return tail * tail * tail * out * (x < 1.0)


def _allegro_polynomial_cutoff(x: torch.Tensor, r_max: torch.Tensor, p: float = 6.0):
    assert p >= 2.0
    p_int = int(p)
    assert float(p_int) == p
    r_max, x = torch.broadcast_tensors(r_max.unsqueeze(-1), x.unsqueeze(0))
    x = x / r_max
    out = float(p_int * (p_int + 1) // 2) * torch.ones_like(x)
    for k in range(p_int - 2, -1, -1):
        out = out * x + float((k + 1) * (k + 2) // 2)
    tail = 1.0 - x
    return tail * tail * tail * out * (x < 1.0)


def _allegro_cosine_cutoff(
    x: torch.Tensor, r_max: torch.Tensor, r_start_cos_ratio: float = 0.8
):
    r_max, x = torch.broadcast_tensors(r_max.unsqueeze(-1), x.unsqueeze(0))
    r_decay = r_start_cos_ratio * r_max
    x = x.clamp(r_decay, r_max)
    return 0.5 * (torch.cos((math.pi / (r_max - r_decay)) * (x - r_decay)) + 1.0)


def _spherical_harmonics_l2(
    lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    if lmax == 0:
        return torch.stack((sh_0_0,), dim=-1)
    sh_1_0 = math.sqrt(3) * x
    sh_1_1 = math.sqrt(3) * y
    sh_1_2 = math.sqrt(3) * z
    if lmax == 1:
        return torch.stack((sh_0_0, sh_1_0, sh_1_1, sh_1_2), dim=-1)
    sh_2_0 = math.sqrt(15) * x * z
    sh_2_1 = math.sqrt(15) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = math.sqrt(5) * (y2 - (1 / 2) * x2z2)
    sh_2_3 = math.sqrt(15) * y * z
    sh_2_4 = (1 / 2) * math.sqrt(15) * (z.pow(2) - x.pow(2))
    if lmax == 2:
        return torch.stack(
            (
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
            ),
            dim=-1,
        )
    raise ValueError(f"PT2 spherical harmonics supports lmax <= 2, got {lmax}")


def _with_edge_sisj(data):
    if AtomicDataDict.EDGE_SISJ_KEY in data:
        return data
    if AtomicDataDict.SPIN_NONLEAF_KEY not in data:
        data[AtomicDataDict.SPIN_NONLEAF_KEY] = data[AtomicDataDict.SPIN_KEY]
    spin = data[AtomicDataDict.SPIN_NONLEAF_KEY]
    neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]
    center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
    data[AtomicDataDict.EDGE_SISJ_KEY] = torch.sum(
        spin[center] * spin[neighbor], dim=-1
    ).view(-1, 1)
    return data


def _broadcast_index(
    index: torch.Tensor, other: torch.Tensor, dim: int
) -> torch.Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if index.dim() == 1:
        for _ in range(dim):
            index = index.unsqueeze(0)
    for _ in range(index.dim(), other.dim()):
        index = index.unsqueeze(-1)
    return index.expand_as(other)


@contextlib.contextmanager
def _symbolic_scatter_context(replace_nonzero: bool = True) -> Iterator[None]:
    scatter_module = importlib.import_module("torch_runstats.scatter")
    original = scatter_module.scatter

    def make_symbolic_scatter(atom_reduction: bool):
        def symbolic_scatter(
            src: torch.Tensor,
            index: torch.Tensor,
            dim: int = -1,
            out: torch.Tensor | None = None,
            dim_size=None,
            reduce: str = "sum",
        ) -> torch.Tensor:
            if reduce != "sum":
                raise ValueError(f"unsupported PT2 scatter reduction: {reduce}")
            if dim < 0:
                dim += src.dim()
            if dim_size is None:
                if atom_reduction:
                    if _TRACE_N_ATOMS is None:
                        raise RuntimeError("symbolic atom dimension is not initialized")
                    dim_size = _TRACE_N_ATOMS
                else:
                    if _TRACE_N_FRAMES is None:
                        raise RuntimeError(
                            "symbolic frame dimension is not initialized"
                        )
                    dim_size = _TRACE_N_FRAMES
            expanded_index = _broadcast_index(index, src, dim)
            if out is None:
                size = list(src.shape)
                size[dim] = dim_size
                out = torch.zeros(size, dtype=src.dtype, device=src.device)
            return out.scatter_add(dim, expanded_index, src)

        return symbolic_scatter

    patched = []
    for module_name, module in tuple(sys.modules.items()):
        if module is None or getattr(module, "scatter", None) is not original:
            continue
        if module_name.startswith("allegro."):
            replacement = make_symbolic_scatter(atom_reduction=True)
        elif module_name == "nequip.utils.fx_compat":
            replacement = make_symbolic_scatter(atom_reduction=True)
        elif module_name == "nequip.nn._interaction_block":
            replacement = make_symbolic_scatter(atom_reduction=True)
        elif module_name == "nequip.nn._atomwise":
            replacement = make_symbolic_scatter(atom_reduction=False)
        else:
            continue
        patched.append(module)
        setattr(module, "scatter", replacement)

    def symbolic_len(value):
        if isinstance(value, torch.Tensor):
            return value.shape[0]
        return builtins.len(value)

    len_module_names = {
        "allegro.nn._allegro",
        "allegro.nn._allegro_born",
        "allegro.nn._allegro_dipole",
        "allegro.nn._allegro_og",
        "allegro.nn._allegro_spin",
        "allegro.nn._edgewise",
        "nequip.nn._atomwise",
        "nequip.nn._interaction_block",
    }
    len_patched = []
    for module_name in len_module_names:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        previous = module.__dict__.get("len")
        len_patched.append((module, previous))
        module.__dict__["len"] = symbolic_len

    scatter_mean_patched = []
    atomwise_module = sys.modules.get("nequip.nn._atomwise")
    if atomwise_module is not None and "scatter_mean" in atomwise_module.__dict__:
        previous_scatter_mean = atomwise_module.__dict__["scatter_mean"]
        batch_scatter = make_symbolic_scatter(atom_reduction=False)

        def symbolic_scatter_mean(
            src: torch.Tensor, index: torch.Tensor, dim: int = -1, **kwargs
        ) -> torch.Tensor:
            summed = batch_scatter(src, index, dim=dim, reduce="sum", **kwargs)
            count = batch_scatter(
                torch.ones_like(index, dtype=src.dtype),
                index,
                dim=0,
                reduce="sum",
            ).clamp_min(1)
            while count.dim() < summed.dim():
                count = count.unsqueeze(-1)
            return summed / count

        scatter_mean_patched.append((atomwise_module, previous_scatter_mean))
        atomwise_module.__dict__["scatter_mean"] = symbolic_scatter_mean

    function_patched = []

    def patch_function(module_name, function_name, replacement):
        module = sys.modules.get(module_name)
        if module is None or function_name not in module.__dict__:
            return
        function_patched.append((module, function_name, module.__dict__[function_name]))
        module.__dict__[function_name] = replacement

    patch_function("nequip.nn.cutoffs", "_poly_cutoff", _nequip_polynomial_cutoff)
    patch_function("nequip.data.AtomicDataDict", "with_edge_sisj", _with_edge_sisj)
    patch_function(
        "e3nn.o3._spherical_harmonics",
        "_spherical_harmonics",
        _spherical_harmonics_l2,
    )
    for module_name in len_module_names:
        if module_name.startswith("allegro.nn._allegro"):
            patch_function(module_name, "polynomial_cutoff", _allegro_polynomial_cutoff)
            patch_function(module_name, "cosine_cutoff", _allegro_cosine_cutoff)

    original_tensor_nonzero = None
    if replace_nonzero:
        original_tensor_nonzero = torch.Tensor.nonzero
        torch.Tensor.nonzero = lambda tensor, *args, **kwargs: _all_indices_nonzero(
            tensor
        )
    try:
        yield
    finally:
        if original_tensor_nonzero is not None:
            torch.Tensor.nonzero = original_tensor_nonzero
        for module, function_name, previous in function_patched:
            module.__dict__[function_name] = previous
        for module in patched:
            setattr(module, "scatter", original)
        for module, previous in len_patched:
            if previous is None:
                module.__dict__.pop("len", None)
            else:
                module.__dict__["len"] = previous
        for module, previous in scatter_mean_patched:
            module.__dict__["scatter_mean"] = previous
        global _TRACE_N_ATOMS, _TRACE_N_FRAMES
        _TRACE_N_ATOMS = None
        _TRACE_N_FRAMES = None


class _DeployRescaleLayer(torch.nn.Module):
    """Inference-only copy of one ``RescaleOutput`` runtime transform.

    ``related_*_keys`` drive construction-time callbacks on the already-loaded
    inner model; they are not operations performed by ``RescaleOutput.forward``.
    PT2 therefore preserves those updated inner parameters and replays only the
    runtime ``scale_keys`` and ``shift_keys`` here.
    """

    def __init__(self, rescale):
        super().__init__()
        self.scale_keys = tuple(rescale.scale_keys)
        self.shift_keys = tuple(rescale.shift_keys)
        self.has_scale = bool(rescale.has_scale)
        self.has_shift = bool(rescale.has_shift)
        self.register_buffer("scale_by", rescale.scale_by.detach().clone())
        self.register_buffer("shift_by", rescale.shift_by.detach().clone())

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.has_scale:
            for field in self.scale_keys:
                if field in data:
                    data[field] = data[field] * self.scale_by
        if self.has_shift:
            for field in self.shift_keys:
                if field in data:
                    data[field] = data[field] + self.shift_by
        return data


def _deploy_model_and_rescale_layers(model: torch.nn.Module):
    from nequip.nn._rescale import RescaleOutput

    current = model
    outer_to_inner = []
    while isinstance(current, RescaleOutput):
        outer_to_inner.append(_DeployRescaleLayer(current))
        current = current.model
    if not hasattr(current, "func"):
        model_type = f"{type(current).__module__}.{type(current).__name__}"
        raise TypeError(
            "PT2 export does not support output wrapper "
            f"{model_type}; expected RescaleOutput layer(s) around a "
            "ParaStressForce*Output model with a func module"
        )
    return current, tuple(reversed(outer_to_inner))


def _strip_deploy_edge_filters(
    graph_module: fx.GraphModule, input_signature: Tuple[str, ...]
) -> fx.GraphModule:
    """Remove edge masks made redundant by the deployment input contract.

    LAMMPS passes only edges strictly inside the model cutoff, and deployment
    rejects decreasing per-layer cutoffs. Boolean filtering on that edge axis
    is therefore an identity operation. Leaving it in the exported graph
    creates unbacked active-edge symbols that AOTInductor specializes around
    the large tracing sample instead of the declared ``n_edges >= 2`` range.
    """
    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    edge_placeholder = input_signature.index(AtomicDataDict.EDGE_INDEX_KEY)
    edge_value = placeholders[edge_placeholder].meta.get("val")
    edge_dimension = getattr(edge_value, "shape", (None, None))[1]
    removed = 0
    for node in list(graph_module.graph.nodes):
        if node.target != torch.ops.aten.index.Tensor:
            continue
        source, indices = node.args
        if not isinstance(indices, (list, tuple)) or len(indices) != 1:
            continue
        index = indices[0]
        if not isinstance(source, fx.Node) or not isinstance(index, fx.Node):
            continue
        source_value = source.meta.get("val")
        index_value = index.meta.get("val")
        source_shape = getattr(source_value, "shape", ())
        index_shape = getattr(index_value, "shape", ())
        if (
            getattr(index_value, "dtype", None) is torch.bool
            and source_shape
            and index_shape
            and str(source_shape[0]) == str(edge_dimension)
            and str(index_shape[0]) == str(edge_dimension)
        ):
            node.replace_all_uses_with(source)
            graph_module.graph.erase_node(node)
            removed += 1
    graph_module.graph.eliminate_dead_code()
    graph_module.graph.lint()
    graph_module.recompile()
    graph_module.removed_edge_filters = removed
    return graph_module


class _TensorOnlyDeployWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        inner_model, rescale_layers = _deploy_model_and_rescale_layers(model)
        self.func = inner_model.func
        self.rescale_layers = torch.nn.ModuleList(rescale_layers)
        self.num_rescale_layers = len(rescale_layers)
        self.has_spin = AtomicDataDict.SPIN_FORCE_KEY in inner_model.irreps_out
        self.has_atom_virial = AtomicDataDict.ATOM_VIRIAL_KEY in inner_model.irreps_out
        self.signature = output_signature(model)

    def forward(
        self,
        pos: torch.Tensor,
        spin: torch.Tensor,
        edge_index: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        pos = pos.detach().requires_grad_(True)
        spin = spin.detach().requires_grad_(self.has_spin)
        edge_vectors = pos[edge_index[1]] - pos[edge_index[0]]
        edge_vectors.requires_grad_(True)
        return self._forward_with_edge_vectors(
            pos, spin, edge_index, atom_types, edge_vectors
        )

    def _forward_with_edge_vectors(
        self,
        pos: torch.Tensor,
        spin: torch.Tensor,
        edge_index: torch.Tensor,
        atom_types: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        global _TRACE_N_ATOMS, _TRACE_N_FRAMES
        _TRACE_N_ATOMS = pos.shape[0]
        _TRACE_N_FRAMES = 1
        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        data = {
            AtomicDataDict.POSITIONS_KEY: pos,
            AtomicDataDict.SPIN_KEY: spin,
            AtomicDataDict.EDGE_INDEX_KEY: edge_index,
            AtomicDataDict.ATOM_TYPE_KEY: atom_types,
            AtomicDataDict.BATCH_KEY: batch,
            AtomicDataDict.EDGE_VECTORS_KEY: edge_vectors,
        }
        data = self.func(data)

        grad_inputs = [pos, edge_vectors]
        if self.has_spin:
            grad_inputs.append(spin)
        gradients = torch.autograd.grad(
            data[AtomicDataDict.TOTAL_ENERGY_KEY].sum(),
            grad_inputs,
            create_graph=False,
        )
        data[AtomicDataDict.FORCE_KEY] = -gradients[0]
        vector_force = gradients[1]
        if self.has_spin:
            spin_forces = -gradients[2]
            data[AtomicDataDict.SPIN_FORCE_KEY] = spin_forces * torch.sum(
                spin * spin, dim=-1
            ).unsqueeze(-1)

        if self.has_atom_virial:
            edge_virial = torch.einsum("zi,zj->zij", vector_force, edge_vectors)
            atom_virial = scatter(
                edge_virial,
                edge_index[0],
                dim=0,
                reduce="sum",
                dim_size=pos.shape[0],
            )
            virial = atom_virial.sum(dim=0, keepdim=True)
            virial = (virial + virial.transpose(-1, -2)) / 2
            data[AtomicDataDict.VIRIAL_KEY] = -virial
            data[AtomicDataDict.ATOM_VIRIAL_KEY] = -atom_virial

        for rescale in self.rescale_layers:
            data = rescale(data)

        return tuple(data[key] for key in self.signature)


class _NonSpinTensorOnlyDeployWrapper(_TensorOnlyDeployWrapper):
    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        original_pos = pos
        pos = pos.detach().requires_grad_(True)
        # Preserve the established lowering while removing spin from the
        # external ABI. The non-spin model ignores this allocation-free view.
        compatibility_spin = original_pos.detach()
        edge_vectors = pos[edge_index[1]] - pos[edge_index[0]]
        edge_vectors.requires_grad_(True)
        return self._forward_with_edge_vectors(
            pos, compatibility_spin, edge_index, atom_types, edge_vectors
        )


class _PeriodicTensorOnlyDeployWrapper(_TensorOnlyDeployWrapper):
    def forward(
        self,
        pos: torch.Tensor,
        spin: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cell_shift: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        pos = pos.detach().requires_grad_(True)
        spin = spin.detach().requires_grad_(self.has_spin)
        edge_vectors = pos[edge_index[1]] - pos[edge_index[0]]
        edge_vectors = edge_vectors + torch.matmul(edge_cell_shift, cell)
        # Keep pbc in the exported signature; shifts are already zero on open axes.
        edge_vectors = edge_vectors + pbc.to(dtype=pos.dtype).sum() * 0.0
        edge_vectors.requires_grad_(True)
        return self._forward_with_edge_vectors(
            pos, spin, edge_index, atom_types, edge_vectors
        )


def _mark_energy_graph_detach(graph_module: fx.GraphModule) -> fx.GraphModule:
    """Carry energy-graph detach provenance through the derivative trace."""
    for detach in list(graph_module.graph.nodes):
        if not _is_detach_node(detach):
            continue
        original_users = list(detach.users)
        with graph_module.graph.inserting_after(detach):
            first_alias = graph_module.graph.call_function(
                torch.ops.aten.alias.default, args=(detach,)
            )
        with graph_module.graph.inserting_after(first_alias):
            second_alias = graph_module.graph.call_function(
                torch.ops.aten.alias.default, args=(first_alias,)
            )
        for user in original_users:
            user.replace_input_with(detach, second_alias)
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


def _training_energy_core_for_trace(energy_core: torch.nn.Module) -> torch.nn.Module:
    """Trace the equivalent dense source path without changing the live model.

    Full-FX already expands cutoff nonzero operations over all supplied edges.
    Keeping the accompanying boolean indexing creates data-dependent shapes
    across AOTAutograd boundaries. The source dense implementation retains
    cutoff factors and their derivatives without those shape-changing masks.
    """
    supported = {
        "Allegro_og_Module",
        "Allegro_Module",
        "Allegro_spin_Module",
        "Allegro_born_Module",
    }
    if not any(
        type(module).__name__ in supported
        and getattr(module, "edge_prune", None) == "nonzero"
        for module in energy_core.modules()
    ):
        return energy_core

    def clone_path(module):
        # ScriptModule deepcopy can duplicate C++ parameter wrappers despite
        # a tensor memo. Copy only Python ancestors of changed pruning flags;
        # untouched children and every parameter/buffer remain the same object.
        children = {
            name: clone_path(child) if child is not None else None
            for name, child in module._modules.items()
        }
        change_pruning = (
            type(module).__name__ in supported
            and getattr(module, "edge_prune", None) == "nonzero"
        )
        if not change_pruning and all(
            children[name] is child for name, child in module._modules.items()
        ):
            return module
        clone = copy.copy(module)
        clone._modules = children
        if change_pruning:
            clone.edge_prune = "dense"
        return clone

    traced_core = clone_path(energy_core)
    if tuple(map(id, traced_core.parameters())) != tuple(
        map(id, energy_core.parameters())
    ):
        raise RuntimeError("training FX capture must preserve parameter identity")
    if tuple(map(id, traced_core.buffers())) != tuple(map(id, energy_core.buffers())):
        raise RuntimeError("training FX capture must preserve buffer identity")
    return traced_core


def _capture_training_energy_graph(
    model: torch.nn.Module, sample: Dict[str, torch.Tensor]
) -> fx.GraphModule:
    """Capture the author-owned energy graph before expanding derivatives."""
    energy_input = AtomicDataDict.with_edge_vectors(dict(sample), with_lengths=False)
    decomposition_table = {
        torch.ops.aten.nonzero.default: _all_indices_nonzero,
    }
    energy_core = _training_energy_core_for_trace(model.func)

    def dynamic_energy(input_data):
        global _TRACE_N_ATOMS, _TRACE_N_FRAMES
        # Read these axes inside make_fx so scatter sizes retain their input
        # provenance. Real sample integers cannot distinguish a frame count
        # from an equal fixed feature width in a later shape-recovery pass.
        _TRACE_N_ATOMS = input_data[AtomicDataDict.POSITIONS_KEY].shape[0]
        _TRACE_N_FRAMES = input_data[AtomicDataDict.CELL_KEY].shape[0]
        return energy_core(input_data)

    with _symbolic_scatter_context():
        energy_graph = make_fx(
            dynamic_energy,
            decomposition_table=decomposition_table,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(energy_input)
    energy_graph.train(model.training)
    return _mark_energy_graph_detach(energy_graph)


class _TrainingGraphWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        energy_graph: fx.GraphModule,
        create_graph: bool,
    ):
        super().__init__()
        if type(model).__name__ != "ParaStressForceSpinOutput":
            raise TypeError(
                "full-FX training requires a ParaStressForceSpinOutput model"
            )
        self.func = energy_graph
        self.register_parameter("spin_fv_coeff", model.spin_fv_coeff)
        self.create_graph = bool(create_graph)
        self.signature = training_output_signature(model)

    def forward(self, input_data: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        global _TRACE_N_ATOMS, _TRACE_N_FRAMES
        data = dict(input_data)
        pos = data[AtomicDataDict.POSITIONS_KEY].detach().requires_grad_(True)
        spin = data[AtomicDataDict.SPIN_KEY].detach().requires_grad_(True)
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        batch = data[AtomicDataDict.BATCH_KEY]
        cell = data[AtomicDataDict.CELL_KEY].view(-1, 3, 3)
        _TRACE_N_ATOMS = pos.shape[0]
        _TRACE_N_FRAMES = cell.shape[0]

        data[AtomicDataDict.POSITIONS_KEY] = pos
        data[AtomicDataDict.SPIN_KEY] = spin
        data[AtomicDataDict.CELL_KEY] = cell
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
        edge_vectors = data[AtomicDataDict.EDGE_VECTORS_KEY]
        edge_vectors.requires_grad_(True)
        data = self.func(data)

        forces_grad, vector_force, spin_grad = torch.autograd.grad(
            data[AtomicDataDict.TOTAL_ENERGY_KEY].sum(),
            (pos, edge_vectors, spin),
            create_graph=self.create_graph,
        )
        forces = -forces_grad
        spin_forces = -spin_grad
        spin_forces = spin_forces * torch.sum(spin * spin, dim=-1).unsqueeze(-1)
        spin_forces_vert = (
            spin_forces - torch.sum(spin_forces * spin, dim=-1).view(-1, 1) * spin
        )
        spin_forces_vert = spin_forces_vert * self.spin_fv_coeff

        edge_virial = torch.einsum("zi,zj->zij", vector_force, edge_vectors)
        atom_index = edge_index[0].view(-1, 1, 1).expand_as(edge_virial)
        atom_virial = torch.zeros(
            (pos.shape[0], 3, 3), dtype=pos.dtype, device=pos.device
        ).scatter_add(0, atom_index, edge_virial)
        frame_index = batch.view(-1, 1, 1).expand_as(atom_virial)
        virial = torch.zeros(
            (cell.shape[0], 3, 3), dtype=pos.dtype, device=pos.device
        ).scatter_add(0, frame_index, atom_virial)
        virial = (virial + virial.transpose(-1, -2)) / 2

        volume = torch.abs(
            torch.einsum(
                "zi,zi->z",
                cell[:, 0, :],
                torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
            )
        )
        data[AtomicDataDict.FORCE_KEY] = forces
        data[AtomicDataDict.SPIN_FORCE_KEY] = spin_forces
        data[AtomicDataDict.SPIN_FORCE_VERT_KEY] = spin_forces_vert
        data[AtomicDataDict.STRESS_KEY] = virial / volume.view(-1, 1, 1)
        data[AtomicDataDict.VIRIAL_KEY] = -virial
        data[AtomicDataDict.ATOM_VIRIAL_KEY] = -atom_virial
        return tuple(data[key] for key in self.signature)


def expand_training_graph(
    model: torch.nn.Module,
    sample_batch,
    *,
    create_graph: bool = True,
) -> fx.GraphModule:
    """Expand conservative training autograd into a first-order FX graph."""
    _validate_cutoff_contract(model)
    sample = _pad_training_batch(model, sample_batch)
    with _FX_TRACE_LOCK:
        energy_graph = _capture_training_energy_graph(model, sample)
        wrapper = _TrainingGraphWrapper(
            model,
            energy_graph=energy_graph,
            create_graph=create_graph,
        )
        wrapper.train(model.training)
        decomposition_table = get_decompositions([torch.ops.aten.silu_backward])
        decomposition_table[torch.ops.aten.nonzero.default] = _all_indices_nonzero
        with torch.enable_grad(), _symbolic_scatter_context():
            graph_module = make_fx(
                wrapper,
                decomposition_table=decomposition_table,
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
            )(sample)
        input_signature = tuple(sample)
        trace_sizes = (
            int(sample[AtomicDataDict.NUM_BATCH_KEY].item()),
            int(sample[AtomicDataDict.POSITIONS_KEY].shape[0]),
            int(sample[AtomicDataDict.EDGE_INDEX_KEY].shape[1]),
        )
        graph_module.input_signature = input_signature
        graph_module.output_signature = wrapper.signature
        graph_module.trace_sizes = trace_sizes
        graph_module.create_graph = bool(create_graph)
        return strip_saved_tensor_detach(graph_module, remove_all=True)


def expand_deploy_graph(model: torch.nn.Module, sample: Sample) -> fx.GraphModule:
    """Trace inference autograd into an explicit tensor-only FX graph."""
    input_signature = (
        SPIN_INPUT_SIGNATURE if _model_has_spin(model) else NON_SPIN_INPUT_SIGNATURE
    )
    if len(sample) != len(input_signature):
        raise ValueError(
            f"expected {len(input_signature)} sample tensors, got {len(sample)}"
        )
    if not all(tensor.is_contiguous() for tensor in sample):
        raise ValueError("all PT2 sample tensors must be contiguous")
    _validate_cutoff_contract(model)
    source_dense = _uses_source_dense_edge_prune(model)
    with _FX_TRACE_LOCK:
        model.eval()
        wrapper_type = (
            _TensorOnlyDeployWrapper
            if _model_has_spin(model)
            else _NonSpinTensorOnlyDeployWrapper
        )
        wrapper = wrapper_type(model).eval()
        decomposition_table = {}
        if not source_dense:
            decomposition_table[torch.ops.aten.nonzero.default] = _all_indices_nonzero
        with _symbolic_scatter_context(replace_nonzero=not source_dense):
            graph_module = make_fx(
                wrapper,
                decomposition_table=decomposition_table,
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
            )(*sample)
        if not source_dense:
            graph_module = _strip_deploy_edge_filters(graph_module, input_signature)
        graph_module.input_signature = input_signature
        graph_module.output_signature = wrapper.signature
        graph_module.num_rescale_layers = wrapper.num_rescale_layers
        return graph_module


def expand_periodic_deploy_graph(
    model: torch.nn.Module, sample: PeriodicSample
) -> fx.GraphModule:
    """Trace a periodic tensor-only graph for the NequIP LAMMPS pair style."""
    if len(sample) != len(PERIODIC_INPUT_SIGNATURE):
        raise ValueError(
            f"expected {len(PERIODIC_INPUT_SIGNATURE)} periodic sample tensors, "
            f"got {len(sample)}"
        )
    if not all(tensor.is_contiguous() for tensor in sample):
        raise ValueError("all periodic PT2 sample tensors must be contiguous")
    _validate_cutoff_contract(model)
    source_dense = _uses_source_dense_edge_prune(model)
    with _FX_TRACE_LOCK:
        model.eval()
        wrapper = _PeriodicTensorOnlyDeployWrapper(model).eval()
        decomposition_table = {}
        if not source_dense:
            decomposition_table[torch.ops.aten.nonzero.default] = _all_indices_nonzero
        with _symbolic_scatter_context(replace_nonzero=not source_dense):
            graph_module = make_fx(
                wrapper,
                decomposition_table=decomposition_table,
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
            )(*sample)
        if not source_dense:
            graph_module = _strip_deploy_edge_filters(
                graph_module, PERIODIC_INPUT_SIGNATURE
            )
        graph_module.input_signature = PERIODIC_INPUT_SIGNATURE
        graph_module.output_signature = wrapper.signature
        graph_module.num_rescale_layers = wrapper.num_rescale_layers
        return graph_module


def _is_detach_node(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    schema = getattr(node.target, "_schema", None)
    return schema is not None and schema.name == "aten::detach"


def _is_alias_node(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    schema = getattr(node.target, "_schema", None)
    return schema is not None and schema.name == "aten::alias"


def _restore_model_detach_markers(graph: fx.Graph) -> set[fx.Node]:
    semantic_detaches = set()
    for detach in list(graph.nodes):
        if not _is_detach_node(detach):
            continue
        first_aliases = list(detach.users)
        if len(first_aliases) != 1 or not _is_alias_node(first_aliases[0]):
            continue
        first_alias = first_aliases[0]
        second_aliases = list(first_alias.users)
        if len(second_aliases) != 1 or not _is_alias_node(second_aliases[0]):
            continue
        second_alias = second_aliases[0]
        semantic_detaches.add(detach)
        second_alias.replace_all_uses_with(detach)
        graph.erase_node(second_alias)
        graph.erase_node(first_alias)
    return semantic_detaches


def strip_saved_tensor_detach(
    graph_module: fx.GraphModule, *, remove_all: bool = True
) -> fx.GraphModule:
    """Remove generated detach nodes while preserving model-authored detach."""
    stripped = fx.GraphModule(graph_module, copy.deepcopy(graph_module.graph))
    semantic_detaches = _restore_model_detach_markers(stripped.graph)
    for node in list(stripped.graph.nodes):
        if (
            node not in semantic_detaches
            and _is_detach_node(node)
            and (remove_all or getattr(node.args[0], "op", None) != "placeholder")
        ):
            node.replace_all_uses_with(node.args[0])
            stripped.graph.erase_node(node)
    stripped.graph.lint()
    stripped.recompile()
    for attribute in (
        "_in_spec",
        "_out_spec",
        "input_signature",
        "output_signature",
        "trace_sizes",
        "create_graph",
        "num_rescale_layers",
        "removed_edge_filters",
    ):
        if hasattr(graph_module, attribute):
            setattr(stripped, attribute, getattr(graph_module, attribute))
    return stripped
