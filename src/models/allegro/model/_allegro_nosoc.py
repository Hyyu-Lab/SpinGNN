from typing import Optional
import logging

from e3nn import o3

from nequip.data import AtomicDataDict, AtomicDataset

from nequip.nn import SequentialGraphNetwork, AtomwiseReduce
from nequip.nn.radial_basis import BesselBasis

from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
    OneHotAtomEncoding_spin,
    SphericalHarmonicEdgeAttrs_spin_nosoc,
    EdgeScalarAddSpin,
)

from allegro.nn import (
    NormalizedBasis,
    EdgewiseEnergySum,
    Allegro_og_Module,
    ScalarMLP,
    HB_Energy,
    Biquad_Energy,
    EdgewiseReduce,
    Delta_spin_energy_sum,
)

from nequip.model import builder_utils


def Allegro_nosoc(config, initialize: bool, dataset: Optional[AtomicDataset] = None):
    logging.info("Building Allegro spin nosoc model...")

    # Handle avg num neighbors auto
    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    # Handle simple irreps
    assert "l_max" in config
    if "l_max" in config:
        l_max = int(config["l_max"])
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

    lhb = config.get("lhb", True)
    lbiquadratic = config.get("lbiquadratic", True)
    lnosoc_layer = config.get("lnosoc_layer", True)
    lte3nn = config.get("lte3nn", True)
    ene_coeff_trainable = config.get("ene_coeff_trainable", False)
    ldelta_learning = config.get("ldelta_learning", False)
    ldelta_spin_fm = config.get("ldelta_spin_fm", False)
    if config.get("train_spin_ladder", False):
        assert (
            ene_coeff_trainable is False
        ), "When using train_spin_ladder, ene_coeff_trainable must be False"

    layers = {
        # -- Encode --
        # Get various edge invariants
        "one_hot_onlyr": OneHotAtomEncoding,
        "radial_basis_onlyr": (
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
        "spharm_onlyr": SphericalHarmonicEdgeAttrs,
    }
    # The core allegro model:
    layers["allegro_onlyr"] = (
        Allegro_og_Module,
        dict(
            field=AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is the edge SH
            edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
        ),
    )
    layers["edge_onlyr_eng"] = (
        ScalarMLP,
        dict(
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=AtomicDataDict.EDGE_ENERGY_KEY,
            mlp_output_dimension=1,
            add_collect=False,
            coeff_trainable=ene_coeff_trainable,
        ),
    )

    if ldelta_learning:
        # Sum edgewise energies -> per-atom energies:
        layers["edge_eng_sum_fix_spin"] = (
            EdgewiseEnergySum,  # per_edge_species_scale(False)
            dict(
                per_edge_species_scale=False,
                lrescale_spin=False,
                ldelta_spin=False,
            ),
        )
        # Sum system energy:
        layers["energy_sum_fix_spin"] = (
            AtomwiseReduce,
            dict(
                reduce="sum",
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.TOTAL_ENERGY_FIX_SPIN_KEY,
            ),
        )
        ldelta_learning_add_collect = False
    else:
        ldelta_learning_add_collect = True

    if lhb:
        layers["edge_hb_coeff"] = (
            ScalarMLP,
            dict(
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=AtomicDataDict.EDGE_HB_COEFF_KEY,
                mlp_output_dimension=1,
            ),
        )
        layers["edge_hb_eng"] = (
            HB_Energy,
            dict(
                field=AtomicDataDict.EDGE_HB_COEFF_KEY,
                out_field=AtomicDataDict.EDGE_ENERGY_KEY,
                coeff_trainable=ene_coeff_trainable,
                add_collect=ldelta_learning_add_collect,
            ),
        )  # coeff(1.0) & coeff_trainable (False) & add_collect (True)
        ldelta_learning_add_collect = True

    if lbiquadratic:
        layers["edge_biquad_coeff"] = (
            ScalarMLP,
            dict(
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=AtomicDataDict.EDGE_BIQUAD_COEFF_KEY,
                mlp_output_dimension=1,
            ),
        )
        layers["edge_biquad_eng"] = (
            Biquad_Energy,
            dict(
                field=AtomicDataDict.EDGE_BIQUAD_COEFF_KEY,
                out_field=AtomicDataDict.EDGE_ENERGY_KEY,
                coeff_trainable=ene_coeff_trainable,
                add_collect=ldelta_learning_add_collect,
            ),
        )  # coeff(1.0) & coeff_trainable (False) & add_collect (True)
        ldelta_learning_add_collect = True

    if lnosoc_layer:
        # Nosoc part
        # Modify NODE_ATTRS_KEY and EDGE_EMBEDDING_KEY with |si| and si*sj. EDGE_ATTRS_KEY remain the same
        layers["one_hot_nosoc"] = OneHotAtomEncoding_spin
        layers["radial_basis_nosoc"] = (
            EdgeScalarAddSpin,
            dict(
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            ),
        )
        if lte3nn:
            layers["spharm_nosoc"] = (
                SphericalHarmonicEdgeAttrs_spin_nosoc,
                dict(
                    time_reversal=True,
                ),
            )
        layers["allegro_nosoc"] = (
            Allegro_og_Module,
            dict(
                field=AtomicDataDict.EDGE_ATTRS_KEY,
                edge_invariant_field=AtomicDataDict.EDGE_FEATURES_KEY,
                node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
                nonscalars_time_reversal=True,
                del_edge_attrs=True,
            ),
        )
        layers["edge_eng_nosoc"] = (
            ScalarMLP,
            dict(
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=AtomicDataDict.EDGE_ENERGY_KEY,
                mlp_output_dimension=1,
                coeff_trainable=ene_coeff_trainable,
                add_collect=ldelta_learning_add_collect,
            ),
        )
        ldelta_learning_add_collect = True

    lpredict_spin_length = config.get("lpredict_spin_length", False)
    if lpredict_spin_length:
        layers["edge_spin_sum"] = (
            EdgewiseReduce,
            dict(
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=AtomicDataDict.NODE_FEATURES_KEY,
                normalize_edge_reduce=True,
                reduce="sum",
            ),
        )
        layers["node_spin_length"] = (
            ScalarMLP,
            dict(
                field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=AtomicDataDict.SPIN_LENGTH_KEY,
                mlp_output_dimension=1,
                add_collect=False,
            ),
        )

    # Sum edgewise energies -> per-atom energies:
    if ldelta_learning:
        layers["edge_eng_sum"] = (
            EdgewiseEnergySum,  # per_edge_species_scale(False)
            dict(
                per_edge_species_scale=False,
                ldelta_spin=ldelta_spin_fm,
            ),
        )
    else:
        layers["edge_eng_sum"] = (
            EdgewiseEnergySum,  # per_edge_species_scale(False)
            dict(
                ldelta_spin=False,
            ),  # per_edge_species_scale(False)
        )
    # Sum system energy:

    if ldelta_learning:
        layers["total_energy_sum"] = (
            AtomwiseReduce,
            dict(
                reduce="sum",
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.TOTAL_ENERGY_DELTA_SPIN_KEY,
            ),
        )
        layers["delta_spin_energy_sum"] = Delta_spin_energy_sum
    else:
        layers["total_energy_sum"] = (
            AtomwiseReduce,
            dict(
                reduce="sum",
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            ),
        )

    model = SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
        irreps_in={AtomicDataDict.SPIN_KEY: o3.Irreps("1eo")},
    )

    return model
