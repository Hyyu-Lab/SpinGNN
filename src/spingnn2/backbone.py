"""NumPy Allegro contractions and their local analytical pullbacks."""

import numpy as np
from scipy.special import expit

from .splines import polynomial_cutoff


def mlp_forward(spec, x):
    """Evaluate a bias-free MLP and return its input pullback."""
    slopes = []
    for layer in spec["layers"]:
        x = x @ layer["weight"]
        activation = layer["activation"]
        if activation == "silu":
            sigmoid = expit(x)
            # Match the eager CPU division order in the activation. Rounding
            # the reciprocal first can compound across the hidden layers.
            with np.errstate(over="ignore"):
                x = x / (1.0 + np.exp(-x))
            slopes.append(sigmoid + x * (1.0 - sigmoid))
        elif activation == "ssp":
            slopes.append(np.where(x > 20.0, 1.0, expit(x)))
            x = np.where(x > 20.0, x, np.logaddexp(0.0, x)) - np.asarray(
                np.log(2.0), dtype=x.dtype
            )
        elif activation is None:
            slopes.append(None)
        else:
            raise ValueError(f"Unsupported scalar activation {activation!r}")

    def pullback(grad):
        for layer, slope in zip(reversed(spec["layers"]), reversed(slopes)):
            if slope is not None:
                grad = grad * slope
            grad = grad @ layer["weight"].T
        return grad

    return x, pullback


def _linear_forward(spec, x):
    if spec is None:
        return x, lambda grad: grad

    x = x.reshape(len(x), spec["mul_in"], spec["dim_in"])
    out = np.zeros((len(x), spec["mul_out"], spec["dim_out"]), dtype=x.dtype)
    for block in spec["blocks"]:
        start = block["out"]
        stop = start + block["components"]
        for group in block["groups"]:
            source = x[:, :, group["start"] : group["stop"]]
            # The packed source Linear forms a matrix over its channel and
            # irrep axes. Use the same contraction to preserve that grouping.
            inputs = spec["mul_in"] * group["weight"].shape[-1]
            weights = group["weight"].reshape(spec["mul_out"], inputs)
            source = source.reshape(len(x), inputs, block["components"])
            out[:, :, start:stop] += weights @ source
        out[:, :, start:stop] *= block["normalization"]

    def pullback(grad):
        d_x = np.zeros_like(x)
        for block in spec["blocks"]:
            start = block["out"]
            stop = start + block["components"]
            d_out = grad[:, :, start:stop] * block["normalization"]
            for group in block["groups"]:
                weights = group["weight"].reshape(spec["mul_out"], -1)
                d_group = weights.T @ d_out
                begin, end = group["start"], group["stop"]
                d_x[:, :, begin:end] += d_group.reshape(
                    len(x), spec["mul_in"], end - begin
                )
        return d_x

    return out, pullback


def _tensor_product(spec, features, environment, center):
    features = features.reshape(len(center), spec["mul_in1"], spec["dim_in1"])
    out = np.zeros(
        (len(center), spec["mul_out"], spec["dim_out"]), dtype=features.dtype
    )
    paths = spec["w3j_indices"]
    left = paths[:, 3] + spec["offsets_in1"][paths[:, 0]]
    right = paths[:, 4] + spec["offsets_in2"][paths[:, 1]]
    result = paths[:, 5] + spec["offsets_out"][paths[:, 2]]
    terms = tuple(zip(left, right, result, spec["scaled_w3j"]))
    # Gather only the environment component used by the current sparse term.
    for i, j, k, coefficient in terms:
        out[:, :, k] += coefficient * (features[:, :, i] * environment[center, :, j])

    def pullback(grad):
        d_features = np.zeros_like(features)
        d_environment = np.zeros_like(environment)
        for i, j, k, coefficient in terms:
            d_term = coefficient * grad[:, :, k]
            d_left = d_term * environment[center, :, j]
            if spec["mul_in1"] == 1 and spec["mul_out"] != 1:
                d_left = d_left.sum(axis=1, keepdims=True)
            d_features[:, :, i] += d_left
            np.add.at(d_environment[:, :, j], center, d_term * features[:, :, i])
        return d_features, d_environment

    return out, pullback


def _weigh_channels(spec, attrs, weights):
    component_index = spec["w_index"]
    unexpanded = weights.reshape(len(attrs), spec["multiplicity"], spec["irreps"])
    expanded = unexpanded[:, :, component_index]
    out = attrs[:, None, :] * expanded

    def pullback(grad):
        d_attrs = (grad * expanded).sum(axis=1)
        d_weights = np.zeros_like(unexpanded)
        np.add.at(
            d_weights,
            (slice(None), slice(None), component_index),
            grad * attrs[:, None, :],
        )
        return d_attrs, d_weights.reshape(weights.shape)

    return out, pullback


def _cutoffs(spec, lengths, index):
    radius = np.asarray(spec["cutoffs"][index], dtype=lengths.dtype)
    if spec["cutoff_type"] == "polynomial":
        value, derivative = polynomial_cutoff(
            lengths / radius, 1.0, spec["polynomial_cutoff_p"]
        )
        return value[:, None], derivative / radius
    if spec["cutoff_type"] == "cosine":
        start = radius * spec["r_start_cos_ratio"]
        frequency = np.asarray(np.pi, dtype=lengths.dtype) / (radius - start)
        phase = frequency * (np.clip(lengths, start, radius) - start)
        derivative = np.where(
            (lengths >= start) & (lengths <= radius),
            -0.5 * frequency * np.sin(phase),
            0.0,
        )
        value = 0.5 * (np.cos(phase) + 1.0)
        if spec["edge_prune"] == "nonzero":
            # In float32 the cutoff can round to zero while its sine stays
            # nonzero. The source prunes those edges from the residual update.
            derivative = np.where(value > 0.0, derivative, 0.0)
        return value[:, None], derivative
    raise ValueError(f"Unsupported backbone cutoff {spec['cutoff_type']!r}")


def _update_latents(mlp, inputs, previous, cutoff, residual):
    update, reverse_mlp = mlp_forward(mlp, inputs)
    active = cutoff > 0
    if residual is None:
        old_weight = ~active
        new_weight = active
    else:
        old_weight, new_weight = residual
    current = new_weight * (cutoff * update)
    if previous is not None:
        current = old_weight * previous + current

    def pullback(grad):
        d_update = new_weight * grad
        d_inputs = reverse_mlp(cutoff * d_update)
        d_cutoff = (d_update * update).sum(axis=1)
        return d_inputs, old_weight * grad, d_cutoff

    return current, pullback


def _head_forward(spec, features):
    intermediate, reverse_input = _linear_forward(spec["linear_in"], features)
    activated = np.empty_like(intermediate)
    activation_blocks = []
    start = 0
    epsilon_squared = spec["epsilon"] ** 2
    for width in spec["norm_components"]:
        stop = start + width
        block = intermediate[:, :, start:stop]
        squared = (block * block).sum(axis=-1, keepdims=True)
        norm = np.sqrt(np.maximum(squared, epsilon_squared))
        scale = expit(norm)
        activated[:, :, start:stop] = scale * block
        d_scale = np.where(
            squared >= epsilon_squared, scale * (1.0 - scale) / norm, 0.0
        )
        activation_blocks.append((start, stop, block, scale, d_scale))
        start = stop

    out, reverse_output = _linear_forward(spec["linear_out"], activated)
    out_layout = out.shape
    out = out.reshape(len(out), spec["flat_dim"])
    basis = spec.get(spec["cartesian_basis"])
    if basis is not None:
        out = out @ basis

    def pullback(grad):
        if basis is not None:
            grad = grad @ basis.T
        d_activated = reverse_output(grad.reshape(out_layout))
        d_intermediate = np.empty_like(intermediate)
        for start, stop, block, scale, d_scale in activation_blocks:
            d_block = d_activated[:, :, start:stop]
            d_intermediate[:, :, start:stop] = scale * d_block + block * d_scale * (
                d_block * block
            ).sum(axis=-1, keepdims=True)
        return reverse_input(d_intermediate)

    return out, pullback


def backbone_forward(
    spec,
    attrs,
    invariants,
    node_center,
    node_neighbor,
    lengths,
    center,
    n_centers,
    species_center=None,
):
    """Evaluate an Allegro core for the incident edges of a center-atom tile."""
    if spec["input_pad"]:
        edge_attrs = np.pad(attrs, ((0, 0), (0, spec["input_pad"])))
    else:
        edge_attrs = attrs
    norm = spec["env_sum_normalizations"]
    if norm.ndim == 2:
        if species_center is None:
            raise ValueError("This backbone requires the species at each edge center")
        center_types = np.zeros(n_centers, dtype=np.intp)
        center_types[center] = np.asarray(species_center).reshape(len(center))
        norm = norm[:, center_types, None, None]

    inputs = np.concatenate((node_center, node_neighbor, invariants), axis=-1)
    features = edge_attrs[:, None, :]
    latents = None
    layers = []
    for i, layer in enumerate(spec["layers"]):
        cutoff, d_cutoff = _cutoffs(spec, lengths, i)
        update = spec["residuals"][i - 1] if i and spec["latent_resnet"] else None
        latents, reverse_latents = _update_latents(
            layer["latent"], inputs, latents, cutoff, update
        )
        weights, reverse_weights = mlp_forward(layer["env_embed_mlp"], latents)
        active = cutoff > 0
        weights = np.where(active, weights, 0.0)
        weighter = spec["weighted_channels"]
        split = weighter["weight_numel"] if i == 0 and spec["embed_initial_edge"] else 0
        reverse_initial = None
        if split:
            features, reverse_initial = _weigh_channels(
                weighter, edge_attrs, weights[:, :split]
            )
        features = np.where(active[:, :, None], features, 0.0)
        environment_edges, reverse_environment_edges = _weigh_channels(
            weighter, edge_attrs, weights[:, split:]
        )
        environment = np.zeros(
            (n_centers, *environment_edges.shape[1:]), dtype=attrs.dtype
        )
        np.add.at(environment, center, environment_edges)
        environment, reverse_environment = _linear_forward(
            layer["env_linear"], environment * norm[i]
        )
        contracted, reverse_tp = _tensor_product(
            layer["tensor_product"], features, environment, center
        )
        scalar_count = layer["n_scalar_outs"]
        scalars = contracted[:, :, :scalar_count].reshape(
            len(center), contracted.shape[1] * scalar_count
        )
        features, reverse_linear = _linear_forward(layer["linear"], contracted)
        inputs = np.concatenate((latents, scalars), axis=-1)
        layers.append(
            (
                layer,
                d_cutoff,
                active,
                reverse_latents,
                reverse_weights,
                reverse_initial,
                reverse_environment_edges,
                reverse_environment,
                reverse_tp,
                reverse_linear,
                contracted.shape,
            )
        )

    cutoff, d_cutoff_final = _cutoffs(spec, lengths, len(layers))
    update = spec["residuals"][-1] if spec["latent_resnet"] else None
    latent_dim = latents.shape[1]
    latents, reverse_final = _update_latents(
        spec["final_latent"], inputs, latents, cutoff, update
    )
    outputs = {spec["latent_out_field"]: latents}
    heads = {}
    for field, head in spec["heads"].items():
        values, reverse = _head_forward(head, features)
        outputs[field] = (values * cutoff).reshape((len(center), *head["out_shape"]))
        heads[field] = values, reverse

    def pullback(grads):
        d_inputs, d_latents, d_cutoff = reverse_final(
            grads.get(spec["latent_out_field"], np.zeros_like(latents))
        )
        d_features = np.zeros_like(features)
        for field, grad in grads.items():
            if field in heads:
                values, reverse = heads[field]
                d_head = grad.reshape(values.shape)
                d_features += reverse(d_head * cutoff)
                d_cutoff += (d_head * values).sum(axis=1)
        d_lengths = d_cutoff * d_cutoff_final
        d_attrs = np.zeros_like(edge_attrs)
        for i in range(len(layers) - 1, -1, -1):
            (
                layer,
                d_cutoff_layer,
                active,
                reverse_latents,
                reverse_weights,
                reverse_initial,
                reverse_environment_edges,
                reverse_environment,
                reverse_tp,
                reverse_linear,
                contracted_shape,
            ) = layers[i]
            d_latents += d_inputs[:, :latent_dim]
            d_contracted = reverse_linear(d_features)
            scalar_count = layer["n_scalar_outs"]
            d_contracted[:, :, :scalar_count] += d_inputs[:, latent_dim:].reshape(
                len(center), contracted_shape[1], scalar_count
            )
            d_features, d_environment = reverse_tp(d_contracted)
            d_features *= active[:, :, None]
            d_environment = reverse_environment(d_environment) * norm[i]
            d_attr_env, d_weights = reverse_environment_edges(d_environment[center])
            d_attrs += d_attr_env
            if reverse_initial is not None:
                d_attr_initial, d_initial_weights = reverse_initial(d_features)
                d_attrs += d_attr_initial
                d_weights = np.concatenate((d_initial_weights, d_weights), axis=-1)
            elif i == 0:
                d_attrs += d_features[:, 0, :]
            d_latents += reverse_weights(d_weights * active)
            d_inputs, d_latents, d_cutoff = reverse_latents(d_latents)
            d_lengths += d_cutoff * d_cutoff_layer

        first = node_center.shape[1]
        second = first + node_neighbor.shape[1]
        return (
            d_attrs[:, : attrs.shape[1]],
            d_inputs[:, second:],
            d_inputs[:, :first],
            d_inputs[:, first:second],
            d_lengths,
        )

    return outputs, pullback
