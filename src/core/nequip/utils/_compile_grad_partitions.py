"""Keep optional output branches separate at AOTAutograd boundaries.

AOTAutograd materializes unused output gradients as zeros. Compiling heads
with different parameter dependencies in one region therefore changes an
unused parameter's gradient from None to zero, which changes AdamW updates.
Partitioning by output reachability keeps the common trunk shared while
allowing autograd to skip an unconsumed head's compiled backward entirely.
"""

import copy

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_leaves


def _preserve_absent_output_gradients(gradient_inputs, gradient_outputs):
    # Node hooks receive the original engine inputs, before a Python
    # autograd.Function materializes None. DDPSink visits unconsumed outputs;
    # such a visit must not turn an unused parameter into an optimizer update.
    # A real tensor of zeros remains a real gradient and is never discarded.
    if all(gradient is None for gradient in gradient_outputs):
        return tuple(None for _ in gradient_inputs)
    return gradient_inputs


class _GradientPresenceBoundary(torch.nn.Module):
    """Preserve an unconsumed compiled region across autograd output sinks."""

    def __init__(self, compiled):
        super().__init__()
        self.compiled = compiled

    def forward(self, *args, **kwargs):
        frontier = {
            value.grad_fn
            for value in tree_leaves((args, kwargs))
            if isinstance(value, torch.Tensor) and value.grad_fn is not None
        }
        output = self.compiled(*args, **kwargs)
        pending = [
            value.grad_fn
            for value in tree_leaves(output)
            if isinstance(value, torch.Tensor) and value.grad_fn is not None
        ]
        seen = set()
        while pending:
            node = pending.pop()
            if node is None or node in seen or node in frontier:
                continue
            seen.add(node)
            if node.name() == "CompiledFunctionBackward":
                node.register_hook(_preserve_absent_output_gradients)
                continue
            # AOT may reconstruct output aliases with a view node. Walk back
            # to this region's compiled node, but never past its input graph.
            pending.extend(parent for parent, _ in node.next_functions)
        return output


def _stops_gradient(node):
    schema = getattr(node.target, "_schema", None)
    return (node.op == "call_method" and node.target == "detach") or (
        schema is not None and schema.name == "aten::detach"
    )


def _bind_partition_constants(split):
    """Keep captured weights/buffers as attributes, not dynamic tensor inputs."""
    for call in split.graph.nodes:
        if call.op != "call_module":
            continue
        child = split.get_submodule(call.target)
        placeholders = [node for node in child.graph.nodes if node.op == "placeholder"]
        arguments = []
        for index, (placeholder, argument) in enumerate(zip(placeholders, call.args)):
            if not isinstance(argument, torch.fx.Node) or argument.op != "get_attr":
                arguments.append(argument)
                continue
            value = split
            for part in argument.target.split("."):
                value = getattr(value, part)
            name = f"_spingnn_bound_{index}"
            if isinstance(value, torch.nn.Parameter):
                child.register_parameter(name, value)
            elif isinstance(value, torch.Tensor):
                child.register_buffer(name, value)
            else:
                setattr(child, name, value)
            with child.graph.inserting_after(placeholder):
                bound = child.graph.get_attr(name)
                bound.meta = dict(placeholder.meta)
            placeholder.replace_all_uses_with(bound)
            child.graph.erase_node(placeholder)
        call.args = tuple(arguments)
        child.recompile()
    split.graph.eliminate_dead_code()


def _differentiable_copy(destination, source, non_blocking=False):
    # Functionalization emits aten.copy, which has no autograd derivative in
    # torch 2.11. Keep copy_'s real zero gradient for the overwritten input,
    # without multiplying potentially uninitialized/NaN destination data by zero.
    # Expand before casting so backward converts to source dtype before reducing
    # broadcast axes, matching copy_'s rounding order.
    converted = source.expand_as(destination).to(
        dtype=destination.dtype,
        device=destination.device,
        non_blocking=non_blocking,
    )
    return torch.where(
        torch.ones_like(destination, dtype=torch.bool), converted, destination
    )


def _functionalize_graph(graph_module):
    """Make alias writes explicit before splitting the graph's dataflow.

    split_module does not order aliasing reads against writes. A slice copy_
    whose return value is unused can otherwise move after its consumers and
    silently change model outputs. Reuse symbolic capture inputs and original
    parameters/buffers rather than specializing runtime batches or copying weights.
    """
    if not any(
        getattr(getattr(node.target, "_schema", None), "is_mutable", False)
        for node in graph_module.graph.nodes
    ):
        return graph_module
    values = [
        node.meta["val"]
        for node in graph_module.graph.nodes
        if node.op == "placeholder"
    ]
    pytree = getattr(graph_module.graph._codegen, "pytree_info", None)
    args = pytree.in_spec.unflatten(values) if pytree is not None else tuple(values)
    result = make_fx(
        torch.func.functionalize(graph_module, remove="mutations"),
        decomposition_table={torch.ops.aten.copy.default: _differentiable_copy},
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
    )(*args)
    if set(map(id, result.parameters())) != set(map(id, graph_module.parameters())):
        raise RuntimeError("FX functionalization must preserve parameter identity")
    if set(map(id, result.buffers())) != set(map(id, graph_module.buffers())):
        raise RuntimeError("FX functionalization must preserve buffer identity")
    for name in ("input_signature", "output_signature", "trace_sizes", "create_graph"):
        if hasattr(graph_module, name):
            setattr(result, name, getattr(graph_module, name))
    return result


def compile_gradient_partitions(graph_module, compiler):
    """Compile independent gradient regions without duplicating shared work."""
    graph_module = _functionalize_graph(graph_module)
    parameters = dict(graph_module.named_parameters(remove_duplicate=False))
    dependencies = {}
    output = None
    for node in graph_module.graph.nodes:
        if node.op == "get_attr":
            parameter = parameters.get(node.target)
            dependencies[node] = (
                frozenset((id(parameter),))
                if parameter is not None and parameter.requires_grad
                else frozenset()
            )
        elif _stops_gradient(node):
            dependencies[node] = frozenset()
        else:
            dependencies[node] = frozenset().union(
                *(dependencies[parent] for parent in node.all_input_nodes)
            )
            if node.op == "call_module":
                dependencies[node] |= frozenset(
                    id(parameter)
                    for parameter in graph_module.get_submodule(
                        node.target
                    ).parameters()
                    if parameter.requires_grad
                )
        if node.op == "output":
            output = node

    output_nodes = output.all_input_nodes
    groups = {}
    for node in output_nodes:
        dependency = dependencies[node]
        if dependency and dependency not in groups:
            groups[dependency] = len(groups)
    if len(groups) <= 1:
        return _GradientPresenceBoundary(compiler(graph_module))

    # Every operation belongs to the set of output groups which consume it.
    # Shared ancestors remain in one region, so splitting optional heads does
    # not recompute the common message-passing trunk.
    graph_copy = torch.fx.GraphModule(graph_module, copy.deepcopy(graph_module.graph))
    graph_copy.graph.eliminate_dead_code()
    copied_output = next(node for node in graph_copy.graph.nodes if node.op == "output")
    original_by_name = {node.name: node for node in graph_module.graph.nodes}
    consumers = {node: set() for node in graph_copy.graph.nodes}
    for node in copied_output.all_input_nodes:
        dependency = dependencies[original_by_name[node.name]]
        consumers[node].add(groups.get(dependency, 0))
    for node in reversed(list(graph_copy.graph.nodes)):
        if node.op == "output":
            continue
        for parent in node.all_input_nodes:
            consumers[parent].update(consumers[node])
    partitions = {}

    def partition(node):
        key = frozenset(consumers[node])
        if key not in partitions:
            partitions[key] = len(partitions)
        return partitions[key]

    split = split_module(graph_copy, graph_copy, partition)
    _bind_partition_constants(split)
    # make_fx flattens the input mapping into tensor placeholders. Preserve
    # its public pytree boundary instead of exposing split_module's flat
    # positional signature to the model wrapper.
    split.graph.set_codegen(copy.deepcopy(graph_module.graph._codegen))
    split.recompile()
    split._spingnn_gradient_partition_count = sum(
        isinstance(child, torch.fx.GraphModule) for child in split.children()
    )
    for name, child in list(split.named_children()):
        if isinstance(child, torch.fx.GraphModule):
            setattr(split, name, _GradientPresenceBoundary(compiler(child)))
    return split
