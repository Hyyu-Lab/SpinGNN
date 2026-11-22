from e3nn.util.test import assert_auto_jitable
from e3nn import o3

from nequip.utils.test import assert_AtomicData_equivariant
from nequip.nn.radial_basis import BesselBasis
from nequip.nn.cutoffs import PolynomialCutoff
from nequip.nn.embedding import (
    OneHotAtomEncoding_spin,
    SphericalHarmonicEdgeAttrs_spin_soc,
    SphericalHarmonicEdgeAttrs_spin_soc_simple,
    RadialBasisEdgeEncoding_spin,
)
from nequip.utils.test import add_spin
from nequip.data import AtomicDataDict
import pytest


def test_spin_onehot(CH3CHO):
    _, data = CH3CHO
    data = add_spin([data.clone()])[0]
    data[AtomicDataDict.SPIN_NONLEAF_KEY] = data[AtomicDataDict.SPIN_KEY]
    irreps_in = data.irreps.copy()
    irreps_in[AtomicDataDict.SPIN_NONLEAF_KEY] = o3.Irreps("1eo")
    oh = OneHotAtomEncoding_spin(
        num_types=3,
        irreps_in=irreps_in,
    )
    assert_auto_jitable(oh)
    assert_AtomicData_equivariant(
        oh, data, do_parity=True, do_time_reversal=True, do_only_rot_spin=True
    )


@pytest.mark.parametrize("parity", [True, False])
@pytest.mark.parametrize("time_reversal", [True, False])
def test_spin_spharm(CH3CHO, parity, time_reversal):
    _, data = CH3CHO
    data = add_spin([data.clone()])[0]
    data[AtomicDataDict.SPIN_NONLEAF_KEY] = data[AtomicDataDict.SPIN_KEY]
    irreps_in = data.irreps.copy()
    irreps_in[AtomicDataDict.SPIN_NONLEAF_KEY] = o3.Irreps("1eo")
    sph = SphericalHarmonicEdgeAttrs_spin_soc(
        l_max=2,
        irreps_in=irreps_in,
        parity=parity,
        time_reversal=time_reversal,
    )
    assert_auto_jitable(sph)
    assert_AtomicData_equivariant(
        sph,
        data,
        do_parity=parity,
        do_time_reversal=time_reversal,
        do_only_rot_spin=False,
    )


@pytest.mark.parametrize("parity", [True, False])
@pytest.mark.parametrize("time_reversal", [True, False])
def test_spin_spharm_simple(CH3CHO, parity, time_reversal):
    _, data = CH3CHO
    data = add_spin([data.clone()])[0]
    data[AtomicDataDict.SPIN_NONLEAF_KEY] = data[AtomicDataDict.SPIN_KEY]
    irreps_in = data.irreps.copy()
    irreps_in[AtomicDataDict.SPIN_NONLEAF_KEY] = o3.Irreps("1eo")
    sph = SphericalHarmonicEdgeAttrs_spin_soc_simple(
        l_max=2,
        irreps_in=irreps_in,
        parity=parity,
        time_reversal=time_reversal,
    )
    assert_auto_jitable(sph)
    assert_AtomicData_equivariant(
        sph,
        data,
        do_parity=parity,
        do_time_reversal=time_reversal,
        do_only_rot_spin=False,
    )


def test_spin_radial_basis(CH3CHO):
    _, data = CH3CHO
    data = add_spin([data.clone()])[0]
    data[AtomicDataDict.SPIN_NONLEAF_KEY] = data[AtomicDataDict.SPIN_KEY]
    irreps_in = data.irreps.copy()
    irreps_in[AtomicDataDict.SPIN_NONLEAF_KEY] = o3.Irreps("1eo")
    basis = BesselBasis
    cutoff = PolynomialCutoff
    rad = RadialBasisEdgeEncoding_spin(
        basis,
        cutoff,
        basis_kwargs={"r_max": 5.0},
        cutoff_kwargs={"r_max": 5.0},
        irreps_in=irreps_in,
    )
    assert_auto_jitable(rad)
    assert_AtomicData_equivariant(
        rad, data, do_parity=True, do_time_reversal=True, do_only_rot_spin=True
    )
