from typing import Optional
import logging

from e3nn import o3

from nequip.data import AtomicDataDict, AtomicDataset

from nequip.nn import SequentialGraphNetwork, AtomwiseReduce
from nequip.nn.radial_basis import BesselBasis

from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    RadialBasisEdgeEncoding_spin,
    SphericalHarmonicEdgeAttrs,
    SphericalHarmonicEdgeAttrs_spin_soc_simple,
    SphericalHarmonicEdgeAttrs_spin_nosoc,
    OneHotAtomEncoding_spin,
    EdgeScalarAddSpin,
)

from allegro.nn import (
    NormalizedBasis,
    EdgewiseEnergySum,
    Allegro_og_Module,
    Allegro_spin_Module,
    Biquad_Energy,
    ScalarMLP,
    Spin_ani_Energy,
    Spin_JMatrix_Energy,
    EdgewiseReduce,
    Delta_spin_energy_sum,
)

from nequip.model import builder_utils


def Allegro_soc(config, initialize: bool, dataset: Optional[AtomicDataset] = None):
    logging.debug("Building Allegro spin soc model...")

    # Handle avg num neighbors auto
    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    # Handle simple irreps
    assert "l_max" in config
    config["irreps_edge_sh"] = int(config["l_max"])

    symmetry = config["symmetry"]
    assert symmetry in (
        "u2",
        "o3",
        "su2",
        "so3",
        "o3_full",
        "o3_restricted",
    )  # "o3_full" is equal to o3
    config["parity"] = symmetry in ("o3", "u2", "o3_full", "o3_restricted")
    config["time_reversal"] = symmetry in ("su2", "u2")
    config["nonscalars_time_reversal"] = symmetry in ("su2", "u2")
    config["nonscalars_include_parity"] = symmetry in ("o3", "u2", "o3_full")

    ljmatrix = config.get("ljmatrix", True)
    lani = config.get("lani", True)
    lbiquadratic = config.get("lbiquadratic", True)
    lsoc_layer = config.get("lsoc_layer", True)
    lonly_soc_layer = config.get("lonly_soc_layer", False)
    ene_coeff_trainable = config.get("ene_coeff_trainable", False)
    tenn_soc = config.get("tenn_soc", True)
    ldelta_learning = config.get("ldelta_learning", False)
    # lsimple_soc = conly_onfig.get("lsimple_soc", True)

    if not lonly_soc_layer:
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
            Allegro_spin_Module,
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

        if ljmatrix:
            layers["edge_jmatrix_eng"] = (
                Spin_JMatrix_Energy,
                dict(
                    field=AtomicDataDict.EDGE_HB_MATRIX_KEY,
                    out_field=AtomicDataDict.EDGE_ENERGY_KEY,
                    add_collect=ldelta_learning_add_collect,
                    coeff_trainable=ene_coeff_trainable,
                ),
            )  # coeff(1.0) & coeff_trainable (False) & add_collect (True)
            ldelta_learning_add_collect = True

        if lani:
            layers["edge_ani_eng"] = (
                Spin_ani_Energy,
                dict(
                    field=AtomicDataDict.NODE_SPINANI_MATRIX_KEY,
                    out_field=AtomicDataDict.EDGE_ENERGY_KEY,
                    add_collect=ldelta_learning_add_collect,
                    coeff_trainable=ene_coeff_trainable,
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
                    add_collect=False,
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
    else:
        assert lsoc_layer is True, "if lonly_soc_layer turn on ,must turn on lsoc_layer"
        assert (
            ldelta_learning is False
        ), "delta learning is not suitable for lonly_soc_layer"
        ldelta_learning_add_collect = False
        layers = {}

    if lsoc_layer:
        # soc part
        # Modify NODE_ATTRS_KEY and EDGE_EMBEDDING_KEY with |si| and si*sj. EDGE_ATTRS_KEY remain the same
        layers["one_hot_soc_on"] = OneHotAtomEncoding_spin
        layers["radial_basis_soc_on"] = (
            RadialBasisEdgeEncoding_spin if lonly_soc_layer else EdgeScalarAddSpin,
            dict(
                field=AtomicDataDict.EDGE_EMBEDDING_KEY
                if lonly_soc_layer
                else AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            ),
        )
        layers["spharm_soc_on"] = (
            SphericalHarmonicEdgeAttrs_spin_soc_simple
            if tenn_soc
            else SphericalHarmonicEdgeAttrs_spin_nosoc
        )  # if lsimple_soc else SphericalHarmonicEdgeAttrs_spin_soc
        # The core allegro_nosoc model:
        layers["allegro_soc_on"] = (
            Allegro_og_Module,
            dict(
                field=AtomicDataDict.EDGE_ATTRS_KEY,
                edge_invariant_field=AtomicDataDict.EDGE_FEATURES_KEY,
                node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
                nonscalars_time_reversal=False,
            ),
        )
        layers["edge_eng_soc_on"] = (
            ScalarMLP,
            dict(
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=AtomicDataDict.EDGE_ENERGY_KEY,
                mlp_output_dimension=1,
                add_collect=ldelta_learning_add_collect,
                coeff_trainable=ene_coeff_trainable,
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
                ldelta_spin=False,
            ),
        )
    else:
        layers["edge_eng_sum"] = (
            EdgewiseEnergySum,  # per_edge_species_scale(False)
            dict(
                ldelta_spin=False,
            ),
        )  # per_edge_species_scale(False)
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
