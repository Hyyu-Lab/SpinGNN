from typing import Optional, List
import contextlib
import math
import functools
import warnings

import torch
from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.utils.tp_utils import tp_path_exists

from ._fc import ScalarMLPFunction
from ._e2ip import EquivariantVectorHead

# from .. import _keys
from ._strided import Contracter, MakeWeightedChannels, Linear
from .cutoffs import cosine_cutoff, polynomial_cutoff


class _RebindToOriginal(torch.autograd.Function):
    """Identity that lets a recomputed graph end at its own boundary leaves
    while routing gradients to the original boundary tensors.

    The checkpoint recompute must be an island. Recomputing directly on the
    saved (graph-carrying) inputs would let the inner ``autograd.grad`` walk
    past the boundary into upstream history: an input that is an ancestor of
    another input (e.g. positions upstream of edge vectors and edge lengths)
    would then receive a gradient through that history AND again when the
    engine continues from the boundary tensors' grad_fns — double counting.
    """

    @staticmethod
    def forward(ctx, original):
        return original.detach()

    @staticmethod
    def backward(ctx, grad):
        return grad


class _CheckpointReplayState:
    """Replay stochastic/mixed-precision work without advancing live module state."""

    def __init__(self, fn, tensors):
        self.cpu_rng = torch.get_rng_state()
        self.cuda_rng = {
            device: torch.cuda.get_rng_state(device)
            for device in sorted(
                {tensor.device.index for tensor in tensors if tensor.is_cuda}
            )
        }
        self.autocast = {
            device: dict(
                enabled=torch.is_autocast_enabled(device),
                dtype=torch.get_autocast_dtype(device),
                cache_enabled=torch.is_autocast_cache_enabled(),
            )
            for device in ("cpu", "cuda")
        }
        owner = fn if isinstance(fn, torch.nn.Module) else getattr(fn, "__self__", None)
        modules = tuple(owner.modules()) if isinstance(owner, torch.nn.Module) else ()
        self.training = [(module, module.training) for module in modules]
        self.batchnorm = [
            (
                module,
                {
                    name: buffer.detach().clone()
                    for name, buffer in module.named_buffers(recurse=False)
                    if buffer is not None
                },
            )
            for module in modules
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            or (
                isinstance(module, torch.jit.ScriptModule)
                and getattr(module, "original_name", "")
                in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "SyncBatchNorm")
            )
        ]

    @contextlib.contextmanager
    def replay(self):
        with torch.random.fork_rng(
            devices=list(self.cuda_rng)
        ), contextlib.ExitStack() as stack:
            torch.set_rng_state(self.cpu_rng)
            for device, state in self.cuda_rng.items():
                torch.cuda.set_rng_state(state, device)
            for device, settings in self.autocast.items():
                stack.enter_context(torch.autocast(device, **settings))
            training = [(module, module.training) for module, _ in self.training]
            buffers = [(module, dict(module._buffers)) for module, _ in self.batchnorm]
            try:
                for module, enabled in self.training:
                    module.training = enabled
                for module, saved in self.batchnorm:
                    # Rebind private copies instead of copy_ restoring live
                    # buffers: retained derivative graphs may save their versions.
                    for name, buffer in saved.items():
                        module._buffers[name] = buffer.clone()
                yield
            finally:
                for module, saved in buffers:
                    # ScriptModule exposes an OrderedDictWrapper without update.
                    for name, buffer in saved.items():
                        module._buffers[name] = buffer
                for module, enabled in training:
                    module.training = enabled


class _ActivationCheckpointFunction(torch.autograd.Function):
    """Recompute ``fn`` once in backward instead of storing its activations.

    Force/spin-force training propagates through this region twice per step:
    the inner ``autograd.grad(..., create_graph=True)`` for derivatives and
    the outer loss backward. Recomputing with
    ``create_graph=torch.is_grad_enabled()`` keeps both paths exact.
    ``torch.utils.checkpoint`` cannot serve this case here: its non-reentrant
    path re-enters one-shot per-frame state on the second unpack and
    recomputes with corrupted inputs.

    Module parameters are explicit inputs so their gradients flow back
    through the engine (parameter .grad accumulation and DDP hooks).

    The backward recompute runs on boundary leaves rebound through
    ``_RebindToOriginal`` so the recomputed graph cannot leak into upstream
    history (which would double-count shared ancestors such as positions).

    Under exact double-backward force training the recomputed graph must
    stay alive until the outer loss backward consumes it, so recomputation
    alone cannot lower peak device memory. With ``offload`` enabled the
    backward runs under ``torch.autograd.graph.save_on_cpu(pin_memory=True)``:
    tensors saved by the recomputed graph AND by the derivative graph built
    from it are staged in pinned host memory and fetched back on demand.
    Only the storage location changes, never the values, so results are
    identical to on-device storage while the throughput cost is whatever the
    copy engine imposes. Offloading both graphs is disabled by default because
    it can substantially increase host memory use per rank.

    The first return value is metadata: (out_extra, new_keys,
    deleted_keys). Only entries whose tensor is new (absent from the input
    or a different object) are returned as tensor outputs. Passthrough
    entries (output value is the identical input tensor) are taken from
    the original mapping by the caller: forwarding them through this
    function's outputs would leave unconsumed alias edges that break
    ``autograd.grad`` connectivity when downstream code differentiates
    wrt the original dict entry.
    """

    @staticmethod
    def forward(ctx, fn, extra, keys, offload, *tensors):
        data = dict(extra)
        data.update(zip(keys, tensors))
        # ``fn`` may mutate ``data`` in place and return it, so key
        # bookkeeping must compare against this input snapshot.
        input_keys = tuple(data.keys())
        input_tensors = dict(zip(keys, tensors))
        ctx.replay_state = _CheckpointReplayState(fn, tensors)
        output = fn(data)
        ctx.fn = fn
        ctx.extra = extra
        ctx.keys = keys
        ctx.offload = offload
        out_extra = tuple(
            (key, value) for key, value in output.items() if not torch.is_tensor(value)
        )
        new_keys = tuple(
            key
            for key, value in output.items()
            if torch.is_tensor(value)
            and (key not in input_tensors or value is not input_tensors[key])
        )
        deleted_keys = tuple(key for key in input_keys if key not in output)
        ctx.new_keys = new_keys
        ctx.save_for_backward(*tensors)
        meta = (out_extra, new_keys, deleted_keys)
        return (meta,) + tuple(output[key] for key in new_keys)

    @staticmethod
    def backward(ctx, *grads):
        # Recompute needs grad mode even on the final backward. Capture the
        # caller's derivative request first so that final backward does not
        # retain a third-order graph which the training loss never consumes.
        create_graph = torch.is_grad_enabled()
        saved = ctx.saved_tensors
        n_data = len(ctx.keys)
        # With offload enabled, the saved tensors of BOTH the recomputed graph
        # and the derivative graph built below are staged in pinned host
        # memory to reduce peak device memory during force training.
        if ctx.offload:
            offload_scope = torch.autograd.graph.save_on_cpu(pin_memory=True)
        else:
            offload_scope = contextlib.nullcontext()
        with torch.enable_grad(), offload_scope:
            # Recompute on rebound boundary leaves so the recompute graph is
            # an island (see _RebindToOriginal). Parameters are graph leaves
            # without history, so they can be used directly.
            rebound = [
                _RebindToOriginal.apply(tensor) if tensor.requires_grad else tensor
                for tensor in saved[:n_data]
            ]
            data = dict(ctx.extra)
            data.update(zip(ctx.keys, rebound))
            with ctx.replay_state.replay():
                output = ctx.fn(data)
            pairs = [
                (output[key], grad)
                for key, grad in zip(ctx.new_keys, grads[1:])
                if output[key].requires_grad
            ]
            targets = [tensor for tensor in rebound if tensor.requires_grad]
            targets.extend(tensor for tensor in saved[n_data:] if tensor.requires_grad)
            if pairs and targets:
                input_grads = torch.autograd.grad(
                    [value for value, _ in pairs],
                    targets,
                    [
                        grad if grad is not None else torch.zeros_like(value)
                        for value, grad in pairs
                    ],
                    create_graph=create_graph,
                    allow_unused=True,
                )
            else:
                input_grads = [None] * len(targets)
        grads_by_position = []
        position = 0
        for tensor in saved:
            if tensor.requires_grad:
                grads_by_position.append(input_grads[position])
                position += 1
            else:
                grads_by_position.append(None)
        return (None, None, None, None, *grads_by_position)


@compile_mode("script")
class Allegro_og_Module(GraphModuleMixin, torch.nn.Module):
    __constants__ = ["edge_prune", "e2IP_out_field"]

    # saved params
    num_layers: int
    field: str
    out_field: str
    num_types: int
    env_embed_mul: int
    weight_numel: int
    latent_resnet: bool
    embed_initial_edge: bool
    edge_prune: str
    allegro_activation_checkpoint: bool
    allegro_checkpoint_offload: bool

    # internal values
    _env_builder_w_index: List[int]
    _env_builder_n_irreps: int
    _input_pad: int

    def __init__(
        self,
        # required params
        num_layers: int,
        num_types: int,
        r_max: float,
        avg_num_neighbors: Optional[float] = None,
        # cutoffs
        r_start_cos_ratio: float = 0.8,
        PolynomialCutoff_p: float = 6,
        per_layer_cutoffs: Optional[List[float]] = None,
        cutoff_type: str = "polynomial",
        # general hyperparameters:
        field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        edge_invariant_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        node_invariant_field: str = AtomicDataDict.NODE_ATTRS_KEY,
        env_embed_multiplicity: int = 32,
        embed_initial_edge: bool = True,
        linear_after_env_embed: bool = False,
        nonscalars_include_parity: bool = True,
        nonscalars_time_reversal: bool = True,
        # MLP parameters:
        two_body_latent=ScalarMLPFunction,
        two_body_latent_kwargs={},
        env_embed=ScalarMLPFunction,
        env_embed_kwargs={},
        latent=ScalarMLPFunction,
        latent_kwargs={},
        latent_resnet: bool = True,
        latent_resnet_update_ratios: Optional[List[float]] = None,
        latent_resnet_update_ratios_learnable: bool = False,
        latent_out_field: Optional[str] = AtomicDataDict.EDGE_FEATURES_KEY,
        del_edge_attrs: bool = False,
        # Performance parameters:
        edge_prune: str = "nonzero",
        pad_to_alignment: int = 1,
        sparse_mode: Optional[str] = None,
        allegro_activation_checkpoint: bool = False,
        allegro_checkpoint_offload: bool = False,
        # Optional evidential vector readout; disabled for existing models.
        e2IP_out_field: str = "",
        e2ip_latent_channels: int = 8,
        # Other:
        irreps_in=None,
    ):
        super().__init__()
        SCALAR = o3.Irrep("0e")  # define for convinience

        # save parameters
        assert (
            num_layers >= 1
        )  # zero layers is "two body", but we don't need to support that fallback case
        self.num_layers = num_layers
        self.nonscalars_include_parity = nonscalars_include_parity
        self.nonscalars_time_reversal = nonscalars_time_reversal
        self.e2IP_out_field = e2IP_out_field
        self.field = field
        self.latent_out_field = latent_out_field
        self.edge_invariant_field = edge_invariant_field
        self.node_invariant_field = node_invariant_field
        self.latent_resnet = latent_resnet
        self.env_embed_mul = env_embed_multiplicity
        self.r_start_cos_ratio = r_start_cos_ratio
        self.polynomial_cutoff_p = float(PolynomialCutoff_p)
        self.cutoff_type = cutoff_type
        assert cutoff_type in ("cosine", "polynomial")
        self.embed_initial_edge = embed_initial_edge
        self.avg_num_neighbors = avg_num_neighbors
        self.linear_after_env_embed = linear_after_env_embed
        self.num_types = num_types
        self.del_edge_attrs = del_edge_attrs
        if edge_prune not in ("nonzero", "dense"):
            raise ValueError(
                f"edge_prune must be 'nonzero' or 'dense', got {edge_prune!r}"
            )
        self.edge_prune = edge_prune
        self.allegro_activation_checkpoint = allegro_activation_checkpoint
        self.allegro_checkpoint_offload = allegro_checkpoint_offload

        # set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                self.field,
                self.edge_invariant_field,
                self.node_invariant_field,
            ],
        )

        # for normalization of env embed sums
        # one per layer
        self.register_buffer(
            "env_sum_normalizations",
            # dividing by sqrt(N)
            torch.as_tensor([avg_num_neighbors] * num_layers).rsqrt(),
        )

        latent = functools.partial(latent, **latent_kwargs)
        env_embed = functools.partial(env_embed, **env_embed_kwargs)

        self.latents = torch.nn.ModuleList([])
        self.env_embed_mlps = torch.nn.ModuleList([])
        self.tps = torch.nn.ModuleList([])
        self.linears = torch.nn.ModuleList([])
        self.env_linears = torch.nn.ModuleList([])

        # Embed to the spharm * it as mul
        input_irreps = self.irreps_in[self.field]
        # this is not inherant, but no reason to fix right now:
        assert all(mul == 1 for mul, ir in input_irreps)
        env_embed_irreps = o3.Irreps([(1, ir) for _, ir in input_irreps])
        assert (
            env_embed_irreps[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"
        self._input_pad = (
            int(math.ceil(env_embed_irreps.dim / pad_to_alignment)) * pad_to_alignment
        ) - env_embed_irreps.dim
        self.register_buffer("_zero", torch.zeros(1, 1))

        # Initially, we have the B(r)Y(\vec{r})-projection of the edges
        # (possibly embedded)
        if self.embed_initial_edge:
            arg_irreps = env_embed_irreps
        else:
            arg_irreps = input_irreps

        # - begin irreps -
        # start to build up the irreps for the iterated TPs
        tps_irreps = [arg_irreps]

        for layer_idx in range(num_layers):
            # Create higher order terms cause there are more TPs coming
            if layer_idx == 0:
                # Add parity irreps
                ir_out = []
                for (mul, ir) in env_embed_irreps:
                    if self.nonscalars_include_parity:
                        if self.nonscalars_time_reversal:
                            ir_out.append((1, (ir.l, 1, 1)))
                            ir_out.append((1, (ir.l, 1, -1)))
                            ir_out.append((1, (ir.l, -1, 1)))
                            ir_out.append((1, (ir.l, -1, -1)))
                        else:
                            ir_out.append((1, (ir.l, 1, ir.t)))
                            ir_out.append((1, (ir.l, -1, ir.t)))
                    else:
                        if self.nonscalars_time_reversal:
                            ir_out.append((1, (ir.l, ir.p, 1)))
                            ir_out.append((1, (ir.l, ir.p, -1)))
                        else:
                            ir_out.append((1, ir))

                ir_out = o3.Irreps(ir_out)

            if layer_idx == self.num_layers - 1 and not self.e2IP_out_field:
                # ^ means we're doing the last layer
                # No more TPs follow this, so only need scalars
                ir_out = o3.Irreps([(1, (0, 1))])

            # Prune impossible paths
            ir_out = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in ir_out
                    if tp_path_exists(arg_irreps, env_embed_irreps, ir)
                ]
            )

            if self.e2IP_out_field:
                ir_out = ir_out.sort().irreps

            # the argument to the next tensor product is the output of this one
            arg_irreps = ir_out
            tps_irreps.append(ir_out)
        # - end build irreps -

        # == Remove unneeded paths ==
        out_irreps = tps_irreps[-1]
        new_tps_irreps = [out_irreps]
        for arg_irreps in reversed(tps_irreps[:-1]):
            new_arg_irreps = []
            for mul, arg_ir in arg_irreps:
                for _, env_ir in env_embed_irreps:
                    if any(i in out_irreps for i in arg_ir * env_ir):
                        # arg_ir is useful: arg_ir * env_ir has a path to something we want
                        new_arg_irreps.append((mul, arg_ir))
                        # once its useful once, we keep it no matter what
                        break
            new_arg_irreps = o3.Irreps(new_arg_irreps)
            new_tps_irreps.append(new_arg_irreps)
            out_irreps = new_arg_irreps

        assert len(new_tps_irreps) == len(tps_irreps)
        tps_irreps = list(reversed(new_tps_irreps))
        del new_tps_irreps

        if not self.e2IP_out_field:
            assert tps_irreps[-1].lmax == 0

        tps_irreps_in = tps_irreps[:-1]
        tps_irreps_out = tps_irreps[1:]
        del tps_irreps

        # Environment builder:
        self._env_weighter = MakeWeightedChannels(
            irreps_in=input_irreps,
            multiplicity_out=env_embed_multiplicity,
            pad_to_alignment=pad_to_alignment,
        )

        self._n_scalar_outs = []

        # == Build TPs ==
        for layer_idx, (arg_irreps, out_irreps) in enumerate(
            zip(tps_irreps_in, tps_irreps_out)
        ):
            # Make the env embed linear
            if self.linear_after_env_embed:
                self.env_linears.append(
                    Linear(
                        [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps],
                        [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps],
                        shared_weights=True,
                        internal_weights=True,
                    )
                )
            else:
                self.env_linears.append(torch.nn.Identity())
            # Make TP
            tmp_i_out: int = 0
            instr = []
            n_scalar_outs: int = 0
            full_out_irreps = []
            for i_out, (_, ir_out) in enumerate(out_irreps):
                for i_1, (_, ir_1) in enumerate(arg_irreps):
                    for i_2, (_, ir_2) in enumerate(env_embed_irreps):
                        if ir_out in ir_1 * ir_2:
                            if ir_out == SCALAR:
                                n_scalar_outs += 1
                            instr.append((i_1, i_2, tmp_i_out))
                            full_out_irreps.append((env_embed_multiplicity, ir_out))
                            tmp_i_out += 1
            full_out_irreps = o3.Irreps(full_out_irreps)
            self._n_scalar_outs.append(n_scalar_outs)
            assert all(ir == SCALAR for _, ir in full_out_irreps[:n_scalar_outs])
            tp = Contracter(
                irreps_in1=o3.Irreps(
                    [
                        (
                            (
                                env_embed_multiplicity
                                if layer_idx > 0 or self.embed_initial_edge
                                else 1
                            ),
                            ir,
                        )
                        for _, ir in arg_irreps
                    ]
                ),
                irreps_in2=o3.Irreps(
                    [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps]
                ),
                irreps_out=o3.Irreps(
                    [(env_embed_multiplicity, ir) for _, ir in full_out_irreps]
                ),
                instructions=instr,
                # For the first layer, we have the unprocessed edges
                # coming in from the input if `not self.embed_initial_edge`.
                # These don't match the embedding in mul, so we have
                # to use uvv --- since the input edges should be mul
                # of one in normal circumstances, this is still plenty fast.
                # For this reason it also doesn't increase the number of weights.
                connection_mode=(
                    "uuu" if layer_idx > 0 or self.embed_initial_edge else "uvv"
                ),
                shared_weights=False,
                has_weight=False,
                pad_to_alignment=pad_to_alignment,
                sparse_mode=sparse_mode,
            )
            self.tps.append(tp)
            # we extract the scalars from the first irrep of the tp
            assert out_irreps[0].ir == SCALAR

            # Make env embed mlp
            generate_n_weights = (
                self._env_weighter.weight_numel
            )  # the weight for the edge embedding
            if layer_idx == 0 and self.embed_initial_edge:
                # also need weights to embed the edge itself
                # this is because the 2 body latent is mixed in with the first layer
                # in terms of code
                generate_n_weights += self._env_weighter.weight_numel

            # the linear acts after the extractor
            self.linears.append(
                Linear(
                    full_out_irreps,
                    [(env_embed_multiplicity, ir) for _, ir in out_irreps],
                    shared_weights=True,
                    internal_weights=True,
                    pad_to_alignment=pad_to_alignment,
                )
            )

            if layer_idx == 0:
                # at the first layer, we have no invariants from previous TPs
                self.latents.append(
                    two_body_latent(
                        mlp_input_dimension=(
                            (
                                # Node invariants for center and neighbor (chemistry)
                                2 * self.irreps_in[self.node_invariant_field].num_irreps
                                # Plus edge invariants for the edge (radius).
                                + self.irreps_in[self.edge_invariant_field].num_irreps
                            )
                        ),
                        mlp_output_dimension=None,
                        **two_body_latent_kwargs,
                    )
                )
            else:
                self.latents.append(
                    latent(
                        mlp_input_dimension=(
                            (
                                # the embedded latent invariants from the previous layer(s)
                                self.latents[-1].out_features
                                # and the invariants extracted from the last layer's TP:
                                # above, we already appended the n_scalar_out for the new TP for
                                # the layer we are building right now. So, we need -2
                                # to get the n_scalar_out for the previous TP, which are what we are actually integrating:
                                # in forward(), the `latent` is called _first_ before the TP
                                # of this layer we are building.
                                + env_embed_multiplicity * self._n_scalar_outs[-2]
                            )
                        ),
                        mlp_output_dimension=None,
                    )
                )
            # the env embed MLP takes the last latent's output as input
            # and outputs enough weights for the env embedder
            self.env_embed_mlps.append(
                env_embed(
                    mlp_input_dimension=self.latents[-1].out_features,
                    mlp_output_dimension=generate_n_weights,
                )
            )

        # For the final layer, we specialize:
        # we don't need to propagate nonscalars, so there is no TP
        # thus we only need the latent:
        self.final_latent = latent(
            mlp_input_dimension=self.latents[-1].out_features
            # here we use self._n_scalar_outs[-1] since we haven't appended anything to it
            # so it is still the correct n_scalar_outs for the previous (and last) TP
            + env_embed_multiplicity * self._n_scalar_outs[-1],
            mlp_output_dimension=None,
        )
        self.e2ip_vector_head = torch.nn.Identity()
        if self.e2IP_out_field:
            self.e2ip_vector_head = EquivariantVectorHead(
                tps_irreps_out[-1],
                env_embed_multiplicity,
                e2ip_latent_channels,
                pad_to_alignment,
            )
            self.irreps_out[self.e2IP_out_field] = o3.Irreps(
                f"{e2ip_latent_channels}x1o"
            )
        # - end build modules -

        # - layer resnet update weights -
        if latent_resnet_update_ratios is None:
            # We initialize to zeros, which under the sigmoid() become 0.5
            # so 1/2 * layer_1 + 1/4 * layer_2 + ...
            # note that the sigmoid of these are the factor _between_ layers
            # so the first entry is the ratio for the latent resnet of the first and second layers, etc.
            # e.g. if there are 3 layers, there are 2 ratios: l1:l2, l2:l3
            latent_resnet_update_params = torch.zeros(self.num_layers)
        else:
            latent_resnet_update_ratios = torch.as_tensor(
                latent_resnet_update_ratios, dtype=torch.get_default_dtype()
            )
            assert latent_resnet_update_ratios.min() > 0.0
            assert latent_resnet_update_ratios.min() < 1.0
            latent_resnet_update_params = torch.special.logit(
                latent_resnet_update_ratios
            )
            # The sigmoid is mostly saturated at ±6, keep it in a reasonable range
            latent_resnet_update_params.clamp_(-6.0, 6.0)
        assert latent_resnet_update_params.shape == (
            num_layers,
        ), f"There must be {num_layers} layer resnet update ratios (layer0:layer1, layer1:layer2)"
        if latent_resnet_update_ratios_learnable:
            self._latent_resnet_update_params = torch.nn.Parameter(
                latent_resnet_update_params
            )
        else:
            self.register_buffer(
                "_latent_resnet_update_params", latent_resnet_update_params
            )

        # - Per-layer cutoffs -
        if per_layer_cutoffs is None:
            per_layer_cutoffs = torch.full((num_layers + 1,), r_max)
        self.register_buffer("per_layer_cutoffs", torch.as_tensor(per_layer_cutoffs))
        assert torch.all(self.per_layer_cutoffs <= r_max)
        assert self.per_layer_cutoffs.shape == (
            num_layers + 1,
        ), "Must be one per-layer cutoff for layer 0 and every layer for a total of {num_layers} cutoffs (the first applies to the two body latent, which is 'layer 0')"
        assert (
            self.per_layer_cutoffs[1:] <= self.per_layer_cutoffs[:-1]
        ).all(), "Per-layer cutoffs must be equal or decreasing"
        assert (
            self.per_layer_cutoffs.min() > 0
        ), "Per-layer cutoffs must be >0. To remove higher layers entirely, lower `num_layers`."
        if self.edge_prune == "dense":
            warnings.warn(
                "edge_prune='dense' computes every supplied edge; without strict "
                "LAMMPS prefiltering this is computation amplification",
                UserWarning,
                stacklevel=2,
            )
        self._latent_dim = self.final_latent.out_features
        self.register_buffer("_zero", torch.as_tensor(0.0))

        self.irreps_out.update(
            {
                self.latent_out_field: o3.Irreps(
                    [(self.final_latent.out_features, (0, 1))]
                ),
            }
        )

        if self.del_edge_attrs:
            self.irreps_out.pop(self.field)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """Evaluate, optionally recomputing activations in backward."""
        if torch.jit.is_scripting():
            return self._forward_impl(data)
        if (
            self.allegro_activation_checkpoint
            and self.training
            and torch.is_grad_enabled()
        ):
            return self._checkpointed_forward(data)
        return self._forward_impl(data)

    @torch.jit.unused
    def _checkpointed_forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        keys = tuple(key for key, value in data.items() if torch.is_tensor(value))
        extra = tuple(
            (key, value) for key, value in data.items() if not torch.is_tensor(value)
        )
        tensors = [data[key] for key in keys]
        tensors.extend(self.parameters())
        outputs = _ActivationCheckpointFunction.apply(
            self._forward_impl, extra, keys, self.allegro_checkpoint_offload, *tensors
        )
        out_extra, new_keys, deleted_keys = outputs[0]
        # Rebuild on top of the caller's mapping: passthrough entries keep
        # their original tensors (see the Function's docstring).
        result = dict(data)
        for key in deleted_keys:
            result.pop(key, None)
        result.update(out_extra)
        result.update(zip(new_keys, outputs[1:]))
        return result

    def _forward_impl(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """Evaluate.

        :param data: AtomicDataDict.Type
        :return: AtomicDataDict.Type
        """
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        edge_attr = data[self.field]
        # pad edge_attr
        if self._input_pad > 0:
            edge_attr = torch.cat(
                (
                    edge_attr,
                    self._zero.expand(len(edge_attr), self._input_pad),
                ),
                dim=-1,
            )

        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
        num_edges: int = len(edge_attr)
        edge_invariants = data[self.edge_invariant_field]
        node_invariants = data[self.node_invariant_field]
        # pre-declare variables as Tensors for TorchScript
        scalars = self._zero
        coefficient_old = scalars
        coefficient_new = scalars
        # Initialize state
        latents = torch.zeros(
            (num_edges, self._latent_dim),
            dtype=edge_attr.dtype,
            device=edge_attr.device,
        )
        active_edges = torch.empty((0,), dtype=torch.long, device=edge_attr.device)
        if self.edge_prune == "nonzero":
            active_edges = torch.arange(
                num_edges,
                device=edge_attr.device,
            )

        # For the first layer, we use the input invariants:
        # The center and neighbor invariants and edge invariants
        latent_inputs_to_cat = [
            node_invariants[edge_center],
            node_invariants[edge_neighbor],
            edge_invariants,
        ]
        # The nonscalar features. Initially, the edge data.
        features = edge_attr

        layer_index: int = 0
        # compute the sigmoids vectorized instead of each loop
        layer_update_coefficients = self._latent_resnet_update_params.sigmoid()

        # Vectorized precompute per layer cutoffs
        if self.cutoff_type == "cosine":
            cutoff_coeffs_all = cosine_cutoff(
                edge_length,
                self.per_layer_cutoffs,
                r_start_cos_ratio=self.r_start_cos_ratio,
            )
        elif self.cutoff_type == "polynomial":
            cutoff_coeffs_all = polynomial_cutoff(
                edge_length, self.per_layer_cutoffs, p=self.polynomial_cutoff_p
            )
        else:
            # This branch is unreachable (cutoff type is checked in __init__)
            # But TorchScript doesn't know that, so we need to make it explicitly
            # impossible to make it past so it doesn't throw
            # "cutoff_coeffs_all is not defined in the false branch"
            assert False, "Invalid cutoff type"

        # !!!! REMEMBER !!!! update final layer if update the code in main loop!!!
        # This goes through layer0, layer1, ..., layer_max-1
        for latent, env_embed_mlp, env_linear, tp, linear in zip(
            self.latents, self.env_embed_mlps, self.env_linears, self.tps, self.linears
        ):
            cutoff_coeffs = cutoff_coeffs_all[layer_index]
            w_index: int = 0
            if self.edge_prune == "dense":
                edge_mask = (cutoff_coeffs > 0).unsqueeze(-1)
                new_latents = latent(torch.cat(latent_inputs_to_cat, dim=-1))
                new_latents = cutoff_coeffs.unsqueeze(-1) * new_latents

                if self.latent_resnet and layer_index > 0:
                    this_layer_update_coeff = layer_update_coefficients[layer_index - 1]
                    coefficient_old = torch.rsqrt(this_layer_update_coeff.square() + 1)
                    coefficient_new = this_layer_update_coeff * coefficient_old
                    latents = coefficient_old * latents + coefficient_new * new_latents
                else:
                    latents = torch.where(edge_mask, new_latents, latents)

                weights = env_embed_mlp(latents)
                weights = torch.where(edge_mask, weights, torch.zeros_like(weights))
                if self.embed_initial_edge and layer_index == 0:
                    env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
                    w_index += self._env_weighter.weight_numel
                    features = self._env_weighter(features, env_w)
                if features.dim() == 2:
                    features = torch.where(
                        edge_mask, features, torch.zeros_like(features)
                    )
                else:
                    features = torch.where(
                        edge_mask.unsqueeze(-1),
                        features,
                        torch.zeros_like(features),
                    )

                env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
                w_index += self._env_weighter.weight_numel
                local_env_per_edge = scatter(
                    self._env_weighter(edge_attr, env_w),
                    edge_center,
                    dim=0,
                    dim_size=len(data[AtomicDataDict.ATOM_TYPE_KEY]),
                )
            else:
                # Determine which edges are still in play
                prev_mask = cutoff_coeffs[active_edges] > 0
                active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)

                # Compute latents
                new_latents = latent(torch.cat(latent_inputs_to_cat, dim=-1)[prev_mask])
                # Apply cutoff, which propagates through to everything else
                new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents

                if self.latent_resnet and layer_index > 0:
                    this_layer_update_coeff = layer_update_coefficients[layer_index - 1]
                    # At init, we assume new and old to be approximately uncorrelated
                    # Thus their variances add
                    # we always want the latent space to be normalized to variance = 1.0,
                    # because it is critical for learnability. Still, we want to preserve
                    # the _relative_ magnitudes of the current latent and the residual update
                    # to be controled by `this_layer_update_coeff`
                    # Solving the simple system for the two coefficients:
                    #   a^2 + b^2 = 1  (variances add)   &    a * this_layer_update_coeff = b
                    # gives
                    #   a = 1 / sqrt(1 + this_layer_update_coeff^2)  &  b = this_layer_update_coeff / sqrt(1 + this_layer_update_coeff^2)
                    # rsqrt is reciprocal sqrt
                    coefficient_old = torch.rsqrt(this_layer_update_coeff.square() + 1)
                    coefficient_new = this_layer_update_coeff * coefficient_old
                    # Residual update
                    # Note that it only runs when there are latents to resnet with, so not at the first layer
                    # index_add adds only to the edges for which we have something to contribute
                    latents = torch.index_add(
                        coefficient_old * latents,
                        0,
                        active_edges,
                        coefficient_new * new_latents,
                    )
                else:
                    # Normal (non-residual) update
                    # index_copy replaces, unlike index_add
                    latents = torch.index_copy(latents, 0, active_edges, new_latents)

                # From the latents, compute the weights for active edges:
                weights = env_embed_mlp(latents[active_edges])
                w_index = 0

                if self.embed_initial_edge and layer_index == 0:
                    # embed initial edge
                    env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
                    w_index += self._env_weighter.weight_numel
                    features = self._env_weighter(
                        features[prev_mask], env_w
                    )  # features is edge_attr
                else:
                    # just take the previous features that we still need
                    features = features[prev_mask]

                # Extract weights for the environment builder
                env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
                w_index += self._env_weighter.weight_numel

                # Build the local environments
                # This local environment should only be a sum over neighbors
                # who are within the cutoff of the _current_ layer
                # Those are the active edges, which are the only ones we
                # have weights for (env_w) anyway.
                # So we mask out the edges in the sum:
                local_env_per_edge = scatter(
                    self._env_weighter(edge_attr[active_edges], env_w),
                    edge_center[active_edges],
                    dim=0,
                    dim_size=len(data[AtomicDataDict.ATOM_TYPE_KEY]),
                )
            if self.env_sum_normalizations.ndim < 2:
                # it's a scalar per layer
                norm_const = self.env_sum_normalizations[layer_index]
            else:
                # it's per type
                # get shape [N_atom, 1] for broadcasting
                norm_const = self.env_sum_normalizations[
                    layer_index, data[AtomicDataDict.ATOM_TYPE_KEY]
                ].unsqueeze(-1)
            local_env_per_edge = local_env_per_edge * norm_const
            local_env_per_edge = env_linear(local_env_per_edge)
            # Copy to get per-edge
            # Large allocation, but no better way to do this:
            if self.edge_prune == "dense":
                local_env_per_edge = local_env_per_edge[edge_center]
            else:
                local_env_per_edge = local_env_per_edge[edge_center[active_edges]]

            # Now do the TP
            # recursively tp current features with the environment embeddings
            features = tp(features, local_env_per_edge)

            # Get invariants
            # features has shape [z][mul][k]
            # we know scalars are first
            scalars = features[:, :, : self._n_scalar_outs[layer_index]].reshape(
                features.shape[0],
                features.shape[1] * self._n_scalar_outs[layer_index],
            )

            # do the linear
            features = linear(features)

            # For layer2+, use the previous latents and scalars
            # This makes it deep
            if self.edge_prune == "dense":
                latent_inputs_to_cat = [latents, scalars]
            else:
                latent_inputs_to_cat = [latents[active_edges], scalars]

            # increment counter
            layer_index += 1

        # - final layer -
        # due to TorchScript limitations, we have to
        # copy and repeat the code here --- no way to
        # escape the final iteration of the loop early
        cutoff_coeffs = cutoff_coeffs_all[layer_index]
        prev_mask = torch.empty((0,), dtype=torch.bool, device=edge_attr.device)
        if self.edge_prune == "dense":
            edge_mask = (cutoff_coeffs > 0).unsqueeze(-1)
            new_latents = self.final_latent(torch.cat(latent_inputs_to_cat, dim=-1))
            new_latents = cutoff_coeffs.unsqueeze(-1) * new_latents
            if self.latent_resnet:
                this_layer_update_coeff = layer_update_coefficients[layer_index - 1]
                coefficient_old = torch.rsqrt(this_layer_update_coeff.square() + 1)
                coefficient_new = this_layer_update_coeff * coefficient_old
                latents = coefficient_old * latents + coefficient_new * new_latents
            else:
                latents = torch.where(edge_mask, new_latents, latents)
        else:
            prev_mask = cutoff_coeffs[active_edges] > 0
            active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)
            new_latents = self.final_latent(
                torch.cat(latent_inputs_to_cat, dim=-1)[prev_mask]
            )
            new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents
            if self.latent_resnet:
                this_layer_update_coeff = layer_update_coefficients[layer_index - 1]
                coefficient_old = torch.rsqrt(this_layer_update_coeff.square() + 1)
                coefficient_new = this_layer_update_coeff * coefficient_old
                latents = torch.index_add(
                    coefficient_old * latents,
                    0,
                    active_edges,
                    coefficient_new * new_latents,
                )
            else:
                latents = torch.index_copy(latents, 0, active_edges, new_latents)
        # - end final layer -

        if self.e2IP_out_field:
            e2ip_vectors = self.e2ip_vector_head(features).flatten(start_dim=1)
            if self.edge_prune == "dense":
                e2ip_vectors = e2ip_vectors * cutoff_coeffs.unsqueeze(-1)
            else:
                e2ip_vectors = torch.index_copy(
                    edge_attr.new_zeros((num_edges, e2ip_vectors.shape[-1])),
                    0,
                    active_edges,
                    e2ip_vectors[prev_mask] * cutoff_coeffs[active_edges].unsqueeze(-1),
                )
            data[self.e2IP_out_field] = e2ip_vectors

        # final latents
        data[self.latent_out_field] = latents

        if self.del_edge_attrs:
            del data[self.field]
        return data
