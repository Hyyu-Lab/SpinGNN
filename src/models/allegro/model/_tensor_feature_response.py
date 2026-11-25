"""Response heads appended to a backbone exposing strided tensor features."""

import logging

from nequip.data import AtomicDataDict
from nequip.nn import SequentialGraphNetwork, VolumeNormalization
from allegro.nn import (
    TensorFeatureCartesianReadout,
    EquivariantDielectricPredictor,
    EquivariantPiezoelectricPredictor,
)
from ._tensor_feature_bec import append_layer


def TensorFeatureDielectric(
    config, model: SequentialGraphNetwork
) -> SequentialGraphNetwork:
    """Append ionic and electronic dielectric heads; no energy outputs required.

    Shared readout settings use ``TensorFeatureCartesianReadout_*``; layer
    settings use ``ipv_readout_*`` and ``eps_el_readout_*``. Formulas, output
    fields, frame reductions and ASR semantics are fixed by this builder.
    """
    append_layer(
        config,
        model,
        "ipv_readout",
        TensorFeatureCartesianReadout,
        dict(
            cartesian_formula="ij",
            out_field=AtomicDataDict.IPV_KEY,
            total_out_field=AtomicDataDict.TOTAL_IPV_KEY,
            total_reduce="sum",
            asr="none",
            edge_symmetric_field=AtomicDataDict.EDGE_IPV_CONTRIBUTION_KEY,
            edge_antisymmetric_field="edge_ipv_antisymmetric",
        ),
    )
    if model.ipv_readout.antisymmetric_channels is None:
        logging.warning(
            "The same-edge ij lift has no antisymmetric (1e) part; ipv will be "
            "symmetric without ipv_readout_antisymmetric_channels."
        )
    append_layer(
        config,
        model,
        "epsilon_ion_calc",
        VolumeNormalization,
        dict(
            field=AtomicDataDict.TOTAL_IPV_KEY,
            out_field=AtomicDataDict.EPSILON_ION_KEY,
        ),
    )
    append_layer(
        config,
        model,
        "eps_el_readout",
        TensorFeatureCartesianReadout,
        dict(
            cartesian_formula="ij=ji",
            out_field=AtomicDataDict.ATOMIC_EPS_EL_KEY,
            total_out_field="system_eps_el_features",
            total_reduce="mean",
            asr="none",
            edge_symmetric_field=AtomicDataDict.EDGE_EPS_EL_CONTRIBUTION_KEY,
            antisymmetric_channels=None,
        ),
    )
    return append_layer(
        config,
        model,
        "system_eps_el_predict",
        EquivariantDielectricPredictor,
        dict(
            field="system_eps_el_features",
            out_field=AtomicDataDict.EPSILON_EL_KEY,
        ),
    )


def TensorFeaturePiezo(config, model: SequentialGraphNetwork) -> SequentialGraphNetwork:
    """Append rank-3 ionic and electronic piezoelectric heads (last two indices symmetric)."""
    append_layer(
        config,
        model,
        "ipd_readout",
        TensorFeatureCartesianReadout,
        dict(
            cartesian_formula="ijk=ikj",
            out_field=AtomicDataDict.IPD_KEY,
            total_out_field=AtomicDataDict.TOTAL_IPD_KEY,
            total_reduce="sum",
            asr="none",
            edge_symmetric_field=AtomicDataDict.EDGE_IPD_CONTRIBUTION_KEY,
            antisymmetric_channels=None,
        ),
    )
    append_layer(
        config,
        model,
        "piezo_ion_calc",
        VolumeNormalization,
        dict(
            field=AtomicDataDict.TOTAL_IPD_KEY,
            out_field=AtomicDataDict.PIEZO_ION_KEY,
        ),
    )
    append_layer(
        config,
        model,
        "piezo_el_readout",
        TensorFeatureCartesianReadout,
        dict(
            cartesian_formula="ijk=ikj",
            out_field=AtomicDataDict.ATOMIC_PIEZO_EL_KEY,
            total_out_field="system_piezo_el_features",
            total_reduce="mean",
            asr="none",
            edge_symmetric_field=AtomicDataDict.EDGE_PIEZO_EL_CONTRIBUTION_KEY,
            antisymmetric_channels=None,
        ),
    )
    return append_layer(
        config,
        model,
        "system_piezo_el_predict",
        EquivariantPiezoelectricPredictor,
        dict(
            field="system_piezo_el_features",
            out_field=AtomicDataDict.PIEZO_EL_KEY,
        ),
    )
