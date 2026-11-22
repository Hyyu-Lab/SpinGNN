"""Builder helpers; reuse current energy paths and derivative wrappers."""
from allegro.nn import EdgewiseReduce, ScalarMLP, e2IP_nodewise
from nequip.data import AtomicDataDict


def validate_e2ip(config, soc):
    if config.get("pad_to_alignment", 1) != 1:
        raise ValueError("e2IP currently requires pad_to_alignment: 1")
    key = config.get("evidential_key", "spin_forces_vert" if soc else "forces")
    if key not in (("forces", "spin_forces_vert") if soc else ("forces",)):
        raise ValueError(
            "evidential_key must be forces (onlyr/SOC) or spin_forces_vert (SOC)"
        )
    config["evidential_key"] = key
    if (
        int(config.get("l_max", 0)) < 1
        or int(config.get("e2ip_latent_channels", 8)) < 1
    ):
        raise ValueError("e2IP requires l_max >= 1 and e2ip_latent_channels >= 1")
    if soc and not config.get("lsoc_layer", True):
        raise ValueError("Allegro_soc_e2IP requires lsoc_layer: true")
    # matrix_exp and its derivatives have not been calibrated in mixed precision/full FX.
    if config.get("spingnn_compile", {}).get("enabled", False):
        raise ValueError("e2IP currently requires spingnn_compile.enabled: false")
    if config.get("train_autocast", {}).get("enabled", False):
        raise ValueError("e2IP currently requires train_autocast.enabled: false")


def add_evidence_branch(model, config, after, prefix):
    layers = [
        (
            prefix + "_vector_sum",
            EdgewiseReduce,
            dict(field=prefix + "_edge_vectors", out_field=prefix + "_sigma_v_raw"),
        ),
        (
            prefix + "_feature_sum",
            EdgewiseReduce,
            dict(
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=prefix + "_node_features",
            ),
        ),
    ]
    for suffix, channels in (
        ("nu", 1),
        ("kappa", 1),
        ("sigma_s_raw", int(config.get("e2ip_latent_channels", 8))),
    ):
        layers.append(
            (
                prefix + "_node_" + suffix,
                ScalarMLP,
                dict(
                    field=prefix + "_node_features",
                    out_field=prefix + "_" + suffix,
                    mlp_latent_dimensions=config.get(
                        "e2ip_mlp_latent_dimensions", [32, 32]
                    ),
                    mlp_output_dimension=channels,
                    add_collect=False,
                ),
            )
        )
    for name, builder, params in layers:
        model.insert_from_parameters(
            shared_params=config, name=name, builder=builder, params=params, after=after
        )
        after = name


def finish_evidence(model, config, prefix):
    model.append_from_parameters(
        shared_params=config,
        name="e2IP_nodewise",
        builder=e2IP_nodewise,
        params=dict(include_f=prefix == "f", include_sf=prefix == "sf"),
    )
    return model
