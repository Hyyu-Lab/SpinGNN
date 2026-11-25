from typing import Optional
import logging

from e3nn import o3
from nequip.data import AtomicDataDict, AtomicDataset

from nequip.nn import SequentialGraphNetwork, AtomwiseReduce
from nequip.nn.radial_basis import BesselBasis

from nequip.nn.embedding import (
    OneHotAtomEncoding,
    SphericalHarmonicEdgeAttrs,
    RadialBasisEdgeEncoding,
)

from allegro.nn import (
    NormalizedBasis,
    EdgewiseEnergySum,
    Allegro_born_Module,
    CartesianBECReduce,
    ScalarMLP,
)

from nequip.model import builder_utils


# The BEC reduction used to be three layers (`edge_born_sum`: EdgewiseReduce,
# `born_scale`: PerSpeciesScaleShift, `total_born_sum`: AtomwiseReduce) and is
# now the single `born_scale` layer (CartesianBECReduce). Layer-prefixed config
# keys of the replaced layers would be silently ignored, so they are rejected.
_REMOVED_LAYER_PREFIXES = ("edge_born_sum_", "total_born_sum_", "EdgewiseReduce_")
_REMOVED_BORN_SCALE_OPTIONS = (
    "field",
    "scales",
    "scales_trainable",
    "shifts",
    "shifts_trainable",
    "arguments_in_dataset_units",
    "type_names",
)
_REMOVED_OPTION_REPLACEMENTS = {
    "edge_born_sum_avg_num_neighbors": "born_scale_avg_num_neighbors",
    "EdgewiseReduce_avg_num_neighbors": "born_scale_avg_num_neighbors",
    "born_scale_scales": "born_scale_list",
    "born_scale_scales_trainable": "born_scale_trainable",
}


def _reject_removed_layer_options(config) -> None:
    removed_born_scale = {f"born_scale_{o}" for o in _REMOVED_BORN_SCALE_OPTIONS}
    for key in config.keys():
        if not (key.startswith(_REMOVED_LAYER_PREFIXES) or key in removed_born_scale):
            continue
        replacement = _REMOVED_OPTION_REPLACEMENTS.get(key, None)
        hint = (
            f"use `{replacement}` instead"
            if replacement is not None
            else "this option no longer exists; remove the key"
        )
        raise ValueError(
            f"Config key `{key}` configured a BEC reduction layer of Allegro_born "
            "that was merged into the `born_scale` layer (CartesianBECReduce: "
            "normalized edge sum, per-species scale and the per-structure mean "
            f"`total_born_effective_charges`); {hint}."
        )


def Allegro_born(config, initialize: bool, dataset: Optional[AtomicDataset] = None):
    logging.debug("Building Allegro model...")
    _reject_removed_layer_options(config)

    # Handle avg num neighbors auto
    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    # Handle simple irreps
    if "l_max" in config:
        l_max = int(config["l_max"])
        assert l_max >= 2, "l_max must be at least 2 for BEC tensors"
        parity_setting = config["parity"]
        assert parity_setting in ("o3_full", "o3_restricted", "so3")
        irreps_edge_sh = repr(
            o3.Irreps.spherical_harmonics(
                l_max, p=(1 if parity_setting == "so3" else -1)
            )
        )
        nonscalars_include_parity = parity_setting == "o3_full"
        # check consistant
        assert config.get("irreps_edge_sh", irreps_edge_sh) == irreps_edge_sh
        assert (
            config.get("nonscalars_include_parity", nonscalars_include_parity)
            == nonscalars_include_parity
        )
        config["irreps_edge_sh"] = irreps_edge_sh
        config["nonscalars_include_parity"] = nonscalars_include_parity
    born_scale_list = config.get("born_scale_list", 1.0)
    born_scale_trainable = config.get("born_scale_trainable", True)

    layers = {
        # -- Encode --
        # Get various edge invariants
        "one_hot": OneHotAtomEncoding,
        "radial_basis": (
            RadialBasisEdgeEncoding,
            dict(
                basis=(
                    NormalizedBasis
                    if config.get("normalize_basis", True)
                    else BesselBasis
                ),
                out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            ),
        ),
        # Get edge nonscalars
        "spharm": SphericalHarmonicEdgeAttrs,
        # The core allegro model:
        "allegro": (
            Allegro_born_Module,
            dict(
                field=AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is the edge SH
                edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
                node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
                output_nonscalar=True,
                output_nonscalar_irreps="ij=ij",
                latent_out_nonscalar_field=AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY,
            ),
        ),
        # Process Born Effective Charge: normalized edge sum of the per-edge
        # Cartesian tensors, per-type scale (no shift, to keep the symmetry),
        # and the soft acoustic sum rule (the per-structure mean is the
        # `total_born_effective_charges` target, which should be zero).
        # The layer keeps the name of its only stateful step, `born_scale`,
        # so checkpoints keep their `born_scale.{scales,shifts}` keys.
        "born_scale": (
            CartesianBECReduce,
            dict(
                fields={AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY: "cartesian"},
                out_field=AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY,
                reduce="normalized_sum",
                asr=config.get("born_scale_asr", "soft"),
                total_out_field=AtomicDataDict.TOTAL_BORN_EFFECTIVE_CHARGE_KEY,
                pre_asr_out_field=None,
                per_species_scale=born_scale_list,
                per_species_scale_trainable=born_scale_trainable,
                per_species_shift=0.0,
            ),
        ),
        # Sum edgewise energies -> per-atom energies:
        "edge_eng": (
            ScalarMLP,
            dict(
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=AtomicDataDict.EDGE_ENERGY_KEY,
                mlp_output_dimension=1,
            ),
        ),
        # Sum edgewise energies -> per-atom energies:
        "edge_eng_sum": EdgewiseEnergySum,
        # Sum system energy:
        "total_energy_sum": (
            AtomwiseReduce,
            dict(
                reduce="sum",
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            ),
        ),
    }

    model = SequentialGraphNetwork.from_parameters(shared_params=config, layers=layers)

    return model
