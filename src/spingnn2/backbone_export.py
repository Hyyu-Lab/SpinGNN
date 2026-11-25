"""Resolve checkout Allegro operators into static NumPy model records."""

from collections import Counter
import math
import operator

import numpy as np
import torch
from torch import fx
from e3nn import o3
from e3nn.nn import NormActivation

from allegro.nn._allegro_born import Allegro_born_Module
from allegro.nn._allegro_og import Allegro_og_Module
from allegro.nn._allegro_spin import Allegro_spin_Module
from allegro.nn._fc import ScalarMLPFunction, _mlp_shifted_softplus
from allegro.nn._strided._channels import MakeWeightedChannels
from allegro.nn._strided._linear import _instruction_groups_for
from nequip.data import AtomicDataDict


def _array(tensor):
    return tensor.detach().cpu().numpy().copy()


def _mlp_ops_fx(forward):
    features = None
    for node in forward.graph.nodes:
        if node.op == "placeholder":
            features = node
            continue
        if node.op == "get_attr" or (
            node.op == "call_function" and node.target is operator.mul
        ):
            continue
        if node.op == "output" and node.args == (features,):
            return
        if not node.args or node.args[0] is not features:
            raise ValueError(f"Unsupported ScalarMLPFunction input to {node.name}")
        features = node
        if (
            node.op == "call_module"
            and node.target == "_dropout"
            and isinstance(forward._dropout, torch.nn.AlphaDropout)
        ):
            # The exported model implements evaluation semantics.
            continue
        if node.op == "call_function":
            if node.target is torch.matmul:
                scaled = node.args[1]
                if (
                    isinstance(scaled, fx.Node)
                    and scaled.op == "call_function"
                    and scaled.target is operator.mul
                ):
                    weight_node, scale = scaled.args
                    if weight_node.op == "get_attr" and isinstance(scale, (float, int)):
                        yield getattr(forward, weight_node.target).detach() * scale
                        continue
            if node.target is torch.nn.functional.silu:
                yield "silu"
                continue
            if (
                node.target is _mlp_shifted_softplus
                or getattr(node.target, "name", None) == "ShiftedSoftPlus"
            ):
                yield "ssp"
                continue
        raise ValueError(
            f"Unsupported ScalarMLPFunction operation: {node.op} {node.target}"
        )
    raise ValueError("ScalarMLPFunction has no output")


def _mlp_ops_script(forward):
    _, features = forward.graph.inputs()
    for node in forward.graph.nodes():
        kind = node.kind()
        if kind in ("prim::Constant", "prim::GetAttr", "aten::mul"):
            continue
        input_index = 1 if kind in ("prim::CallFunction", "prim::CallMethod") else 0
        if (
            node.inputsSize() <= input_index
            or node.inputsAt(input_index).unique() != features.unique()
        ):
            raise ValueError(f"Unsupported scripted ScalarMLPFunction input to {kind}")
        features = node.output()
        if kind == "aten::matmul":
            weight_scale = node.inputsAt(1).node()
            if weight_scale.kind() == "aten::mul":
                weight = weight_scale.inputsAt(0).node()
                scale = weight_scale.inputsAt(1).toIValue()
                if weight.kind() == "prim::GetAttr" and isinstance(scale, (int, float)):
                    yield getattr(forward, weight.s("name")).detach() * scale
                    continue
        if kind == "aten::silu":
            yield "silu"
            continue
        if kind == "prim::CallFunction":
            function = node.inputsAt(0).node().s("name")
            if function in ("silu", "ShiftedSoftPlus", "_mlp_shifted_softplus"):
                yield "silu" if function == "silu" else "ssp"
                continue
        if kind == "prim::CallMethod":
            target = node.inputsAt(0).node()
            if (
                target.kind() == "prim::GetAttr"
                and target.s("name") == "_dropout"
                and forward._dropout.original_name == "AlphaDropout"
            ):
                continue
        raise ValueError(f"Unsupported scripted ScalarMLPFunction operation: {kind}")
    if forward.graph.return_node().inputsAt(0).unique() != features.unique():
        raise ValueError("Unsupported scripted ScalarMLPFunction output")


def export_mlp(module):
    """Extract each generated MLP's exact variance factor and fold it into W."""
    if not isinstance(module, ScalarMLPFunction):
        raise ValueError(f"Unsupported scalar MLP {type(module).__name__}")
    forward = module._forward
    ops = (
        _mlp_ops_fx(forward)
        if isinstance(forward, fx.GraphModule)
        else _mlp_ops_script(forward)
    )
    layers = []
    for op in ops:
        if isinstance(op, str):
            if not layers or layers[-1]["activation"] is not None:
                raise ValueError("Unsupported ScalarMLPFunction activation placement")
            layers[-1]["activation"] = op
        else:
            layers.append({"weight": _array(op), "activation": None})
    if not layers or layers[-1]["activation"] is not None:
        raise ValueError("Unsupported ScalarMLPFunction output")
    return {"layers": layers}


def _offsets(irreps):
    return np.cumsum([0, *(ir.dim for _, ir in irreps)], dtype=np.int32)


def _irreps_record(irreps):
    return [[mul, ir.l, ir.p, getattr(ir, "t", 1)] for mul, ir in irreps]


def _linear(module):
    if isinstance(module, torch.nn.Identity):
        return None
    if not all(
        hasattr(module, attr) for attr in ("w", "dim_in", "dim_out", "instructions")
    ):
        raise ValueError(f"Unsupported Allegro Linear layer {type(module).__name__}")

    irreps_in = o3.Irreps(module.irreps_in)
    irreps_out = o3.Irreps(module.irreps_out)
    offsets_in, offsets_out = _offsets(irreps_in), _offsets(irreps_out)
    packed = _array(module.w)
    counts = Counter(out_index for _, out_index in module.instructions)
    blocks = [
        (
            {
                "out": int(offsets_out[i]),
                "components": ir.dim,
                "normalization": 1.0 / math.sqrt(module.mul_in * counts[i]),
                "groups": [],
            }
            if counts[i]
            else None
        )
        for i, (_, ir) in enumerate(irreps_out)
    ]
    position = 0
    for out_index, first, last, _ in _instruction_groups_for(module):
        count = last - first + 1
        end = position + module.mul_out * module.mul_in * count
        blocks[out_index]["groups"].append(
            {
                "start": int(offsets_in[first]),
                "stop": int(offsets_in[last + 1]),
                "weight": packed[position:end]
                .reshape(module.mul_out, module.mul_in, count)
                .copy(),
            }
        )
        position = end
    if position != packed.size:
        raise ValueError(
            "Allegro Linear instruction layout does not match its stored weights"
        )
    return {
        "mul_in": module.mul_in,
        "mul_out": module.mul_out,
        "dim_in": module.dim_in,
        "dim_out": module.dim_out,
        "irreps_in": _irreps_record(irreps_in),
        "irreps_out": _irreps_record(irreps_out),
        "blocks": [block for block in blocks if block is not None],
    }


def _w3j_coordinates(module):
    if hasattr(module, "_w3j_mm"):
        sparse = module._w3j_mm
        values = _array(sparse._v)
        if hasattr(sparse, "_i"):
            k, pair = _array(sparse._i)
        else:
            rowptr = _array(sparse._crow)
            k = np.repeat(np.arange(len(rowptr) - 1), np.diff(rowptr))
            pair = _array(sparse._col)
        if sparse.shape[-1] == module._dim_in1:
            i = j = pair
        else:
            i, j = np.divmod(pair, module._dim_in2)
        return i, j, k, values
    if not hasattr(module, "_big_w3j"):
        raise ValueError("Allegro tensor product has no static Wigner-3j buffer")
    dense = _array(module._big_w3j)
    index = np.nonzero(dense)
    values = dense[index]
    if dense.ndim == 2:
        k, i = index
        j = i
    elif dense.ndim == 3:
        k, i, j = index
    else:
        raise ValueError(f"Unsupported Allegro Wigner-3j shape: {dense.shape}")
    return i, j, k, values


def _tensor_product(module):
    if getattr(module, "has_weight", True) or getattr(
        module, "connection_mode", ""
    ) not in ("uuu", "uvv"):
        raise ValueError(
            "Expected a static unweighted Allegro uuu or uvv tensor product"
        )
    irreps_in1, irreps_in2, irreps_out = (
        o3.Irreps(module.irreps_in1),
        o3.Irreps(module.irreps_in2),
        o3.Irreps(module.irreps_out),
    )
    if module.connection_mode == "uvv" and irreps_in1[0].mul != 1:
        raise ValueError(
            "An Allegro uvv tensor product requires one first-input channel"
        )
    offsets = [_offsets(irreps) for irreps in (irreps_in1, irreps_in2, irreps_out)]
    i, j, k, values = _w3j_coordinates(module)
    absolute = (i, j, k)
    irrep_ids = [
        np.searchsorted(boundaries[1:], components, side="right")
        for boundaries, components in zip(offsets, absolute)
    ]
    local_components = [
        components - boundaries[ids]
        for boundaries, components, ids in zip(offsets, absolute, irrep_ids)
    ]
    # The upstream buffer already includes path normalization and the resolved
    # parity/time-reversal rules. Component offsets within an irrep are zero-based.
    indices = np.stack((*irrep_ids, *local_components), axis=1).astype(np.int32)
    return {
        "connection_mode": module.connection_mode,
        "dim_in1": module._dim_in1,
        "dim_in2": module._dim_in2,
        "dim_out": module._dim_out,
        "mul_in1": irreps_in1[0].mul,
        "mul_out": module._mul_out,
        "irreps_in1": _irreps_record(irreps_in1),
        "irreps_in2": _irreps_record(irreps_in2),
        "irreps_out": _irreps_record(irreps_out),
        "offsets_in1": offsets[0],
        "offsets_in2": offsets[1],
        "offsets_out": offsets[2],
        "w3j_indices": indices,
        "scaled_w3j": values,
    }


def _head(module, basis_name=None, basis=None):
    if not isinstance(module, torch.nn.Sequential) or len(module) != 3:
        raise ValueError(f"Unsupported Allegro tensor head {type(module).__name__}")
    linear_in, activation, linear_out = module
    if (
        not isinstance(activation, NormActivation)
        or activation.bias
        or not activation.normalize
        or activation.scalar_nonlinearity is not torch.nn.functional.silu
    ):
        raise ValueError(
            "Unsupported Allegro head normalization; expected normalized bias-free SiLU"
        )
    norm_widths = [ir.dim for count, ir in activation.irreps_in for _ in range(count)]
    flat_dim = linear_out.dim_out * linear_out.mul_out
    if linear_in.mul_out != 1 or sum(norm_widths) != linear_in.dim_out:
        raise ValueError("Unsupported padded Allegro tensor head")
    spec = {
        "linear_in": _linear(linear_in),
        "linear_out": _linear(linear_out),
        "norm_components": norm_widths,
        "epsilon": activation.epsilon,
        "flat_dim": flat_dim,
        "out_shape": [flat_dim],
        "cartesian_basis": basis_name,
    }
    if basis is not None:
        if basis.shape != (flat_dim, 9):
            raise ValueError(
                f"Unsupported Cartesian basis {basis_name} with shape {tuple(basis.shape)}"
            )
        spec["out_shape"] = [3, 3]
        spec[basis_name] = _array(basis)
    return spec


def export_backbone(module):
    """Export an Allegro trunk and its optional edge-local tensor heads."""
    if not isinstance(
        module, (Allegro_og_Module, Allegro_spin_Module, Allegro_born_Module)
    ):
        raise ValueError(f"Unsupported Allegro backbone {type(module).__name__}")
    if getattr(module, "use_field_nonscalar_features", False):
        raise ValueError(
            "Allegro backbones with external nonscalar features are unsupported"
        )
    if module.cutoff_type not in ("polynomial", "cosine"):
        raise ValueError(f"Unsupported backbone cutoff {module.cutoff_type!r}")
    weighter = module._env_weighter
    if not isinstance(weighter, MakeWeightedChannels):
        raise ValueError(
            f"Unsupported Allegro environment weighter {type(weighter).__name__}"
        )

    heads = {}
    if isinstance(module, Allegro_spin_Module):
        heads[AtomicDataDict.EDGE_HB_MATRIX_KEY] = _head(
            module.final_nonscalar_J, "Q_J", module.nonscalr_transform_last_Q_J
        )
        heads[AtomicDataDict.NODE_SPINANI_MATRIX_KEY] = _head(
            module.final_nonscalar_ani, "Q_ani", module.nonscalr_transform_last_Q_ani
        )
    if isinstance(module, Allegro_born_Module):
        if not module.output_nonscalar or not module.nonscalar_transform_last:
            raise ValueError(
                "Allegro Born backbone must use its Cartesian tensor output"
            )
        heads[module.latent_out_nonscalar_field] = _head(
            module.final_nonscalar, "Q_bec", module.nonscalar_transform_last_Q
        )
    e2ip_field = getattr(module, "e2IP_out_field", "")
    if e2ip_field:
        heads[e2ip_field] = _head(module.e2ip_vector_head)

    layers = [
        {
            "latent": export_mlp(latent),
            "env_embed_mlp": export_mlp(embed),
            "env_linear": _linear(env_linear),
            "tensor_product": _tensor_product(tp),
            "linear": _linear(linear),
            "n_scalar_outs": scalars,
        }
        for latent, embed, env_linear, tp, linear, scalars in zip(
            module.latents,
            module.env_embed_mlps,
            module.env_linears,
            module.tps,
            module.linears,
            module._n_scalar_outs,
        )
    ]
    ratio = module._latent_resnet_update_params.detach().sigmoid()
    old = torch.rsqrt(ratio.square() + 1.0)
    return {
        "field": module.field,
        "edge_invariant_field": module.edge_invariant_field,
        "node_invariant_field": module.node_invariant_field,
        "latent_out_field": module.latent_out_field,
        "input_pad": module._input_pad,
        "weighted_channels": {
            "w_index": _array(weighter._w_index).astype(np.int32),
            "multiplicity": weighter.multiplicity_out,
            "irreps": weighter._num_irreps,
            "weight_numel": weighter.weight_numel,
        },
        "embed_initial_edge": module.embed_initial_edge,
        "env_sum_normalizations": _array(module.env_sum_normalizations),
        "cutoffs": _array(module.per_layer_cutoffs),
        "cutoff_type": module.cutoff_type,
        "edge_prune": module.edge_prune,
        "polynomial_cutoff_p": int(module.polynomial_cutoff_p),
        "r_start_cos_ratio": module.r_start_cos_ratio,
        "latent_resnet": module.latent_resnet,
        "residuals": _array(torch.stack((old, ratio * old), dim=1)),
        "layers": layers,
        "final_latent": export_mlp(module.final_latent),
        "heads": heads,
    }
