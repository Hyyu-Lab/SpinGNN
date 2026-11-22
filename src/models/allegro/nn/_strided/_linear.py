from typing import List, Optional, Tuple, NamedTuple
from math import sqrt

import torch
from torch import fx

import e3nn
from e3nn import o3
from e3nn.util.jit import compile
from e3nn.o3._tensor_product._codegen import _sum_tensors

from opt_einsum_fx import jitable, optimize_einsums_full

from ._layout import StridedLayout


class Instruction(NamedTuple):
    i_in: int
    i_out: int


class _InstructionGroup(NamedTuple):
    """A rectangular block of input irreps contributing to one output irrep.

    ``instruction_indices`` refer to the original instruction list.  Keeping
    these indices explicitly is important when instructions are supplied in an
    order that differs from their input-irrep order (which is common for
    generated l=3 instruction sets).
    """

    i_out: int
    i_start: int
    i_end: int
    instruction_indices: Tuple[int, ...]


def codegen_strided_linear(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List[Instruction],
    normalization: str = "component",
    internal_weights: bool = False,
    shared_weights: bool = False,
    pad_to_alignment: int = 1,
) -> Optional[fx.GraphModule]:
    """Returns None if strided doesn't make sense for this TP."""
    # Check if irreps can be strided
    try:
        layout_in = StridedLayout(irreps_in, pad_to_multiple=pad_to_alignment)
        layout_out = StridedLayout(irreps_out, pad_to_multiple=pad_to_alignment)
    except ValueError:
        # one cannot be strided
        return None

    assert normalization == "component"

    if internal_weights:
        assert shared_weights

    # Explicit instructions must remain irrep-preserving.  Checking before
    # grouping also prevents an invalid output index from being silently
    # ignored and an incompatible singleton run from broadcasting as a vector.
    for instruction_index, ins in enumerate(instructions):
        if not 0 <= ins.i_in < len(irreps_in):
            raise IndexError(
                f"{ins.i_in} is not a valid index for irreps_in "
                f"(instruction {instruction_index})"
            )
        if not 0 <= ins.i_out < len(irreps_out):
            raise IndexError(
                f"{ins.i_out} is not a valid index for irreps_out "
                f"(instruction {instruction_index})"
            )
        if irreps_in[ins.i_in].ir != irreps_out[ins.i_out].ir:
            raise ValueError(
                f"Instruction {instruction_index} connects irreps "
                f"{irreps_in[ins.i_in].ir} and {irreps_out[ins.i_out].ir}"
            )

    # Group instructions by output into maximal consecutive input-irrep runs.
    # The historical weight layout is (mul_out, mul_in, inputs_in_run), with
    # output groups in output-index order.  Keep that layout for previously
    # supported groups; the packing helpers map the published instruction
    # order to these blocks.  It is intentionally not e3nn's flat weight ABI.
    ins_per_output: List[List[Instruction]] = [
        [ins for ins in instructions if ins.i_out == i]
        for i in range(len(layout_out.base_irreps))
    ]
    ins_group_irrep_slice: List[Optional[Tuple[int, int]]] = []
    instruction_groups: List[_InstructionGroup] = []
    for i_out, ins_group in enumerate(ins_per_output):
        if len(ins_group) == 0:
            ins_group_irrep_slice.append(None)
            continue

        by_input = {}
        for instruction_index, ins in enumerate(instructions):
            if ins.i_out == i_out:
                if ins.i_in in by_input:
                    raise ValueError(
                        "Duplicate Linear instruction for input/output pair "
                        f"({ins.i_in}, {i_out})"
                    )
                by_input[ins.i_in] = instruction_index

        runs: List[List[int]] = []
        for input_index in sorted(by_input):
            same_irrep_as_previous = (
                bool(runs)
                and input_index == runs[-1][-1] + 1
                and layout_in.base_irreps[input_index]
                == layout_in.base_irreps[runs[-1][-1]]
            )
            if not same_irrep_as_previous:
                runs.append([input_index])
            else:
                runs[-1].append(input_index)

        # Retain the historical per-output metadata for the common one-run
        # case.  The complete representation is ``_instruction_groups``.
        if len(runs) == 1:
            ins_group_irrep_slice.append((runs[0][0], runs[0][-1]))
        else:
            ins_group_irrep_slice.append(None)

        for run in runs:
            i_start, i_end = run[0], run[-1]
            instruction_groups.append(
                _InstructionGroup(
                    i_out=i_out,
                    i_start=i_start,
                    i_end=i_end,
                    instruction_indices=tuple(by_input[i] for i in run),
                )
            )

    # generate actual code
    graph_out = fx.Graph()
    tracer = fx.proxy.GraphAppendingTracer(graph_out)

    def Proxy(n):
        return fx.Proxy(n, tracer=tracer)

    # = Function definitions =
    x = Proxy(graph_out.placeholder("x", torch.Tensor))
    x = x.reshape(-1, layout_in.mul, layout_in.base_dim)
    if internal_weights:
        ws = Proxy(graph_out.get_attr("w", torch.Tensor))
    else:
        ws = Proxy(graph_out.placeholder("w", torch.Tensor))

    outs = [[] for _ in range(len(layout_out.base_irreps))]

    z = "" if shared_weights else "z"

    w_index: int = 0
    # Pack one rectangular weight block per maximal run.  The generated graph
    # is independent of the user-provided instruction ordering; the published
    # ``instructions`` list remains unchanged.
    for group in instruction_groups:
        ins_grp_i = group.i_out
        ins_grp_ins = (group.i_start, group.i_end)
        # for a given group, which mixes a consecutive set of irreps of the same irrep,
        # we can reduce it to a rectangular operation:
        to_mix = x[
            :,
            :,
            layout_in.base_irreps[: ins_grp_ins[0]]
            .dim : layout_in.base_irreps[: ins_grp_ins[1] + 1]
            .dim,
        ]  # index the i dim in z u i
        # ^ has i index ranging over ins_grp_ins inputs *of same irrep*, so we can rectangularize with a new "n" dimension:
        n: int = 1 + ins_grp_ins[1] - ins_grp_ins[0]
        to_mix = to_mix.reshape(
            -1, layout_in.mul, n, layout_in.base_irreps[ins_grp_ins[0]].ir.dim
        )  # z u n i
        n_weight = layout_in.mul * n * layout_out.mul
        if shared_weights:
            this_w = ws[w_index : w_index + n_weight].reshape(
                layout_out.mul, layout_in.mul, n
            )
        else:
            this_w = ws[:, w_index : w_index + n_weight].reshape(
                -1, layout_out.mul, layout_in.mul, n
            )
        outs[ins_grp_i].append(torch.einsum(f"{z}vun,zuni->zvi", this_w, to_mix))
        w_index += n_weight

    outs = [
        _sum_tensors(
            o,
            shape=(
                x.shape[0],
                layout_out.mul,
                layout_out.base_irreps[i].dim,
            ),
            like=x,
        )
        for i, (ins_grp, o) in enumerate(zip(ins_per_output, outs))
    ]
    outs = [
        (
            (1.0 / sqrt(sum(layout_in.mul for ins in instructions if ins.i_out == i)))
            * out
        )
        if len(ins_per_output[i]) > 0
        else out
        for i, out in enumerate(outs)
    ]
    if len(outs) > 1:
        out = torch.cat(outs, dim=-1)
    else:
        out = outs[0]

    # pad output
    padding: int = layout_out.base_dim - layout_out.base_irreps.dim
    if padding > 0:
        out = torch.nn.functional.pad(
            out,
            (0, padding),
        )

    graph_out.output(out.node)

    # check graphs
    graph_out.lint()

    # Make GraphModules
    # By putting the constants in a Module rather than a dict,
    # we force FX to copy them as buffers instead of as attributes.
    #
    # FX seems to have resolved this issue for dicts in 1.9, but we support all the way back to 1.8.0.
    constants_root = torch.nn.Module()
    if internal_weights:
        constants_root.w = torch.nn.Parameter(torch.randn(w_index))
    graphmod_out = fx.GraphModule(
        constants_root, graph_out, class_name="linear_forward"
    )

    if True:  # optimize_einsums
        # Note that for our einsums, we can optimize _once_ for _any_ batch dimension
        # and still get the right path for _all_ batch dimensions.
        # This is because our einsums are essentially of the form:
        #    zuvw,ijk,zuvij->zwk    OR     uvw,ijk,zuvij->zwk
        # In the first case, all but one operands have the batch dimension
        #    => The first contraction gains the batch dimension
        #    => All following contractions have batch dimension
        #    => All possible contraction paths have cost that scales linearly in batch size
        #    => The optimal path is the same for all batch sizes
        # For the second case, this logic follows as long as the first contraction is not between the first two operands. Since those two operands do not share any indexes, contracting them first is a rare pathological case. See
        # https://github.com/dgasmith/opt_einsum/issues/158
        # for more details.
        #
        # TODO: consider the impact maximum intermediate result size on this logic
        #         \- this is the `memory_limit` option in opt_einsum
        # TODO: allow user to choose opt_einsum parameters?
        #
        # We use float32 and zeros to save memory and time, since opt_einsum_fx looks only at traced shapes, not values or dtypes.
        batchdim = 4
        example_inputs = (
            torch.zeros((batchdim, layout_in.dim)),
            torch.zeros((tuple() if shared_weights else (batchdim,)) + (w_index,)),
        )
        graphmod_out = jitable(optimize_einsums_full(graphmod_out, example_inputs))

    graphmod_out.weight_numel = w_index
    graphmod_out.dim_in = layout_in.base_dim
    graphmod_out.mul_in = layout_in.mul
    graphmod_out.dim_out = layout_out.base_dim
    graphmod_out.mul_out = layout_out.mul
    graphmod_out.irreps_in = str(irreps_in)
    graphmod_out.irreps_out = str(irreps_out)
    graphmod_out.instructions = [tuple(ins) for ins in instructions]
    # ``_ins_group_irrep_slice`` is retained for callers that inspect the
    # historical one-run-per-output metadata.  For split groups it contains
    # None and the complete, lossless representation is exposed separately.
    graphmod_out._ins_group_irrep_slice = ins_group_irrep_slice
    graphmod_out._instruction_groups = [
        (g.i_out, g.i_start, g.i_end, g.instruction_indices) for g in instruction_groups
    ]

    return graphmod_out


def Linear(
    irreps_in,
    irreps_out,
    shared_weights: Optional[bool] = None,
    internal_weights: bool = False,
    instructions: Optional[List[Tuple[int, int]]] = None,
    pad_to_alignment: int = 1,
):
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)
    # == Instructions ==
    if instructions is None:
        # By default, make all possible connections
        instructions = [
            (i_in, i_out)
            for i_in, (_, ir_in) in enumerate(irreps_in)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_in == ir_out
        ]
        # note that "empty" instructions to/from empty irreps are dealt with in the codegen

    instructions = [Instruction(i_in=e[0], i_out=e[1]) for e in instructions]

    mod = codegen_strided_linear(
        irreps_in,
        irreps_out,
        instructions=instructions,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        pad_to_alignment=pad_to_alignment,
    )

    if mod is None:
        raise ValueError
    if e3nn.get_optimization_defaults()["jit_script_fx"]:
        mod = compile(mod)
    return mod


def _instruction_groups_for(linear):
    """Return packed groups, rebuilding metadata after optional TorchScript."""
    if hasattr(linear, "_instruction_groups"):
        return linear._instruction_groups
    irreps_in = o3.Irreps(linear.irreps_in)
    irreps_out = o3.Irreps(linear.irreps_out)
    instructions = [Instruction(*ins) for ins in linear.instructions]
    groups = []
    for i_out in range(len(irreps_out)):
        by_input = {}
        for instruction_index, ins in enumerate(instructions):
            if ins.i_out == i_out:
                by_input[ins.i_in] = instruction_index
        runs = []
        for i_in in sorted(by_input):
            if (
                runs
                and i_in == runs[-1][-1] + 1
                and irreps_in[i_in].ir == irreps_in[runs[-1][-1]].ir
            ):
                runs[-1].append(i_in)
            else:
                runs.append([i_in])
        for run in runs:
            groups.append((i_out, run[0], run[-1], tuple(by_input[i] for i in run)))
    return groups


def weights_per_instruction(
    linear,
) -> Tuple[o3.Irreps, o3.Irreps, List[Tuple[int, int]], List[torch.Tensor]]:
    """Extract the per-instruction weights from a `Linear`.

    Args:
        linear

    Returns:
        irreps_in
        irreps_out
        instructions: list of tuples (in_index, out_index)
        weights: list of per-instruction weight tensors of shape (mul_out, mul_in)
    """
    irreps_in = o3.Irreps(linear.irreps_in)
    irreps_out = o3.Irreps(linear.irreps_out)
    instructions = linear.instructions

    layout_in = StridedLayout(irreps_in)
    layout_out = StridedLayout(irreps_out)

    weights = []
    ws = linear.w.detach()

    # In the code for efficiency the weight shape is vun, where n is over
    # instructions within an instruction group.  Groups may be split into
    # multiple runs for one output, so use the explicit instruction-index
    # metadata rather than assuming one group per output.
    # adapted from _linear.py#L103-L111
    w_index: int = 0
    n: int
    for _, i_start, i_end, instruction_indices in _instruction_groups_for(linear):
        n = 1 + i_end - i_start
        n_weight = layout_in.mul * n * layout_out.mul
        this_w = ws[w_index : w_index + n_weight].reshape(
            layout_out.mul, layout_in.mul, n
        )
        for i, instruction_index in enumerate(instruction_indices):
            weights.append((instruction_index, this_w[:, :, i]))
        w_index += n_weight

    weights = torch.vstack(
        [w.unsqueeze(0) for _, w in sorted(weights, key=lambda item: item[0])]
    )
    assert weights.shape == (len(instructions), layout_out.mul, layout_in.mul)

    return irreps_in, irreps_out, instructions, weights


def weights_from_per_instruction(linear, weights) -> torch.Tensor:
    """Re-pack the per-instruction weights from a `Linear`.

    Args:
        linear
        weights: list of per-instruction weight tensors of shape (mul_out, mul_in)

    Returns:
        weight: a flattened, packed weight tensor that can be assigned to `linear.w` or used in a state dict
    """
    instructions = linear.instructions

    assert len(weights) == len(instructions)
    assert frozenset(w.shape for w in weights) == {(linear.mul_out, linear.mul_in)}

    # Need to group them back into the exact rectangular runs used by codegen.
    # ``weights`` follows the published instruction order.
    to_cat = []
    for _, _, _, instruction_indices in _instruction_groups_for(linear):
        to_cat.append(
            torch.cat([weights[i].unsqueeze(-1) for i in instruction_indices], dim=-1)
        )

    return torch.cat([w.view(-1) for w in to_cat])
