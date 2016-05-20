# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.tests.helper import pytest


from ..nduncertainty_unknown import UnknownUncertainty
from ..nduncertainty_stddev import StdDevUncertainty
from ..nduncertainty_relstd import RelativeUncertainty
from ..nduncertainty_var import VarianceUncertainty

from ..meta import NDUncertainty
from ..exceptions import IncompatibleUncertaintiesException
from ..exceptions import MissingDataAssociationException
from ..nddata import NDDataBase


def test_std_to_var():
    unc1 = StdDevUncertainty(np.ones(5), unit='m')
    # Two ways to convert, from_uncertainty and directly during init:
    unc2 = VarianceUncertainty.from_uncertainty(unc1)
    unc3 = VarianceUncertainty(unc1)

    # Test if both variants produce the same result
    np.testing.assert_array_equal(unc2.data, unc3.data)
    assert unc2.unit == unc3.unit

    # Now test that the data and unit was squared:
    np.testing.assert_array_equal(unc1.data ** 2, unc3.data)
    assert unc1.unit ** 2 == unc3.unit

    # Quick check with parent
    parent = NDDataBase(1)
    parent.uncertainty = unc1
    unc2 = VarianceUncertainty.from_uncertainty(unc1)
    unc3 = VarianceUncertainty(unc1)

    # Test if both variants produce the same result
    np.testing.assert_array_equal(unc2.data, unc3.data)
    assert unc2.unit == unc3.unit
    assert unc2.parent_nddata is unc3.parent_nddata

    # Now test that the data and unit was squared:
    np.testing.assert_array_equal(unc1.data ** 2, unc3.data)
    assert unc1.unit ** 2 == unc3.unit
    assert unc1.parent_nddata is unc3.parent_nddata

    # Quick check to see if data=None or unit=None makes trouble:
    unc1 = StdDevUncertainty(None, unit=None)
    unc2 = VarianceUncertainty.from_uncertainty(unc1)
    assert unc1.data == unc2.data
    assert unc1.unit == unc2.unit


def test_var_to_std():
    unc1 = VarianceUncertainty(np.ones(5), unit='m')
    # Two ways to convert, from_uncertainty and directly during init:
    unc2 = StdDevUncertainty.from_uncertainty(unc1)
    unc3 = StdDevUncertainty(unc1)

    # Test if both variants produce the same result
    np.testing.assert_array_equal(unc2.data, unc3.data)
    assert unc2.unit == unc3.unit

    # Now test that the data and unit was square rooted:
    np.testing.assert_array_equal(np.sqrt(unc1.data), unc3.data)
    assert unc1.unit ** (1/2) == unc3.unit

    # Quick check with parent
    parent = NDDataBase(1)
    parent.uncertainty = unc1
    unc2 = StdDevUncertainty.from_uncertainty(unc1)
    unc3 = StdDevUncertainty(unc1)

    # Test if both variants produce the same result
    np.testing.assert_array_equal(unc2.data, unc3.data)
    assert unc2.unit == unc3.unit
    assert unc2.parent_nddata is unc3.parent_nddata

    # Now test that the data and unit was squared:
    np.testing.assert_array_equal(np.sqrt(unc1.data), unc3.data)
    assert unc1.unit ** (1/2) == unc3.unit
    assert unc1.parent_nddata is unc3.parent_nddata

    # Quick check to see if data=None or unit=None makes trouble:
    unc1 = VarianceUncertainty(None, unit=None)
    unc2 = StdDevUncertainty.from_uncertainty(unc1)
    assert unc1.data == unc2.data
    assert unc1.unit == unc2.unit


def test_std_to_rel():
    unc1 = StdDevUncertainty(np.ones(5)*3, unit='m')
    # Without parent conversion is impossible
    with pytest.raises(MissingDataAssociationException):
        RelativeUncertainty.from_uncertainty(unc1)
    with pytest.raises(MissingDataAssociationException):
        RelativeUncertainty(unc1)

    parent = NDDataBase(np.ones(5)*100, unit='m')
    parent.uncertainty = unc1

    unc2 = RelativeUncertainty.from_uncertainty(unc1)
    unc3 = RelativeUncertainty(unc1)

    np.testing.assert_array_equal(unc2.data, np.ones(5)*3/100)
    np.testing.assert_array_equal(unc2.data, unc3.data)
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc2.parent_nddata is unc3.parent_nddata
    assert unc2.unit is None
    assert unc3.unit is None

    # Let's see if it works if uncertainty and parent have different units.
    unc1 = StdDevUncertainty(np.ones(5)*3, unit='m')
    parent = NDDataBase(np.ones(5)/10, unit='km')
    parent.uncertainty = unc1

    unc2 = RelativeUncertainty.from_uncertainty(unc1)
    unc3 = RelativeUncertainty(unc1)

    np.testing.assert_array_equal(unc2.data, np.ones(5)*3/100)
    np.testing.assert_array_equal(unc2.data, unc3.data)
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc2.parent_nddata is unc3.parent_nddata
    assert unc2.unit is None
    assert unc3.unit is None

    # and no units
    unc1 = StdDevUncertainty(0.1)
    parent = NDDataBase(5)
    parent.uncertainty = unc1

    unc2 = RelativeUncertainty.from_uncertainty(unc1)
    unc3 = RelativeUncertainty(unc1)

    np.testing.assert_array_equal(unc2.data, 0.02)
    np.testing.assert_array_equal(unc2.data, unc3.data)
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc2.parent_nddata is unc3.parent_nddata
    assert unc2.unit is None
    assert unc3.unit is None


def test_rel_to_std():
    unc1 = RelativeUncertainty(0.02)
    # Without parent conversion is impossible
    with pytest.raises(MissingDataAssociationException):
        StdDevUncertainty.from_uncertainty(unc1)
    with pytest.raises(MissingDataAssociationException):
        StdDevUncertainty(unc1)

    parent = NDDataBase(100, unit='m')
    parent.uncertainty = unc1

    unc2 = StdDevUncertainty.from_uncertainty(unc1)
    unc3 = StdDevUncertainty(unc1)

    np.testing.assert_array_equal(unc2.data, 2)
    np.testing.assert_array_equal(unc2.data, unc3.data)
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc2.parent_nddata is unc3.parent_nddata
    assert unc2.unit is None
    assert unc3.unit is None

    # and parent with no unit
    unc1 = RelativeUncertainty(0.02)
    parent = NDDataBase(100)
    parent.uncertainty = unc1

    unc2 = StdDevUncertainty.from_uncertainty(unc1)
    unc3 = StdDevUncertainty(unc1)

    np.testing.assert_array_equal(unc2.data, 2)
    np.testing.assert_array_equal(unc2.data, unc3.data)
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc2.parent_nddata is unc3.parent_nddata
    assert unc2.unit is None
    assert unc3.unit is None


def test_var_to_rel():
    unc1 = VarianceUncertainty(np.ones(5)*9, unit='m*m')
    # Without parent conversion is impossible
    with pytest.raises(MissingDataAssociationException):
        RelativeUncertainty.from_uncertainty(unc1)
    with pytest.raises(MissingDataAssociationException):
        RelativeUncertainty(unc1)

    parent = NDDataBase(np.ones(5)*100, unit='m')
    parent.uncertainty = unc1

    unc2 = RelativeUncertainty.from_uncertainty(unc1)
    unc3 = RelativeUncertainty(unc1)

    np.testing.assert_array_almost_equal(unc2.data, np.ones(5)*3/100)
    np.testing.assert_array_almost_equal(unc2.data, unc3.data)
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc2.parent_nddata is unc3.parent_nddata
    assert unc2.unit is None
    assert unc3.unit is None

    # Let's see if it works if uncertainty and parent have different units.
    unc1 = VarianceUncertainty(np.ones(5)*9, unit='m*m')
    parent = NDDataBase(np.ones(5)/10, unit='km')
    parent.uncertainty = unc1

    unc2 = RelativeUncertainty.from_uncertainty(unc1)
    unc3 = RelativeUncertainty(unc1)

    np.testing.assert_array_almost_equal(unc2.data, np.ones(5)*3/100)
    np.testing.assert_array_almost_equal(unc2.data, unc3.data)
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc2.parent_nddata is unc3.parent_nddata
    assert unc2.unit is None
    assert unc3.unit is None

    # and no units
    unc1 = VarianceUncertainty(0.01)
    parent = NDDataBase(5)
    parent.uncertainty = unc1

    unc2 = RelativeUncertainty.from_uncertainty(unc1)
    unc3 = RelativeUncertainty(unc1)

    np.testing.assert_array_almost_equal(unc2.data, 0.02)
    np.testing.assert_array_almost_equal(unc2.data, unc3.data)
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc2.parent_nddata is unc3.parent_nddata
    assert unc2.unit is None
    assert unc3.unit is None


def test_rel_to_var():
    unc1 = RelativeUncertainty(0.02)
    # Without parent conversion is impossible
    with pytest.raises(MissingDataAssociationException):
        VarianceUncertainty.from_uncertainty(unc1)
    with pytest.raises(MissingDataAssociationException):
        VarianceUncertainty(unc1)

    parent = NDDataBase(100, unit='m')
    parent.uncertainty = unc1

    unc2 = VarianceUncertainty.from_uncertainty(unc1)
    unc3 = VarianceUncertainty(unc1)

    np.testing.assert_array_equal(unc2.data, 4)
    np.testing.assert_array_equal(unc2.data, unc3.data)
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc2.parent_nddata is unc3.parent_nddata
    assert unc2.unit is None
    assert unc3.unit is None

    # and parent with no unit
    unc1 = RelativeUncertainty(0.02)
    parent = NDDataBase(100)
    parent.uncertainty = unc1

    unc2 = VarianceUncertainty.from_uncertainty(unc1)
    unc3 = VarianceUncertainty(unc1)

    np.testing.assert_array_equal(unc2.data, 4)
    np.testing.assert_array_equal(unc2.data, unc3.data)
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc2.parent_nddata is unc3.parent_nddata
    assert unc2.unit is None
    assert unc3.unit is None


def test_unknown_to_stddev():
    # This conversion does:
    # - not copy the data
    # - keep the values/unit/parent_nddata
    unc1 = UnknownUncertainty(np.ones(5), unit='m')
    unc2 = StdDevUncertainty.from_uncertainty(unc1)
    assert unc1.data is unc2.data
    assert unc1.unit == unc2.unit
    with pytest.raises(MissingDataAssociationException):
        unc1.parent_nddata
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # with parent
    parent = NDDataBase(1)
    parent.uncertainty = unc1

    unc3 = StdDevUncertainty.from_uncertainty(unc1)
    assert unc3.data is unc1.data
    assert unc3.unit == unc1.unit
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc1.parent_nddata is parent

    # make sure unc2 wasn't modified ...
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # Less efficient direct usage during init
    unc4 = StdDevUncertainty(unc1)
    assert unc4.data is unc1.data
    assert unc4.unit == unc1.unit
    assert unc1.parent_nddata is unc4.parent_nddata
    assert unc1.parent_nddata is parent


def test_stddev_to_unknown():
    # This conversion does:
    # - not copy the data
    # - keep the values/unit/parent_nddata
    unc1 = StdDevUncertainty(np.ones(5), unit='m')
    unc2 = UnknownUncertainty.from_uncertainty(unc1)
    assert unc1.data is unc2.data
    assert unc1.unit == unc2.unit
    with pytest.raises(MissingDataAssociationException):
        unc1.parent_nddata
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # with parent
    parent = NDDataBase(1)
    parent.uncertainty = unc1

    unc3 = UnknownUncertainty.from_uncertainty(unc1)
    assert unc3.data is unc1.data
    assert unc3.unit == unc1.unit
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc1.parent_nddata is parent

    # make sure unc2 wasn't modified ...
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # Less efficient direct usage during init
    unc4 = UnknownUncertainty(unc1)
    assert unc4.data is unc1.data
    assert unc4.unit == unc1.unit
    assert unc1.parent_nddata is unc4.parent_nddata
    assert unc1.parent_nddata is parent


def test_unknown_to_var():
    # This conversion does:
    # - not copy the data
    # - keep the values/unit/parent_nddata
    unc1 = UnknownUncertainty(np.ones(5), unit='m')
    unc2 = VarianceUncertainty.from_uncertainty(unc1)
    assert unc1.data is unc2.data
    assert unc1.unit == unc2.unit
    with pytest.raises(MissingDataAssociationException):
        unc1.parent_nddata
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # with parent
    parent = NDDataBase(1)
    parent.uncertainty = unc1

    unc3 = VarianceUncertainty.from_uncertainty(unc1)
    assert unc3.data is unc1.data
    assert unc3.unit == unc1.unit
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc1.parent_nddata is parent

    # make sure unc2 wasn't modified ...
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # Less efficient direct usage during init
    unc4 = VarianceUncertainty(unc1)
    assert unc4.data is unc1.data
    assert unc4.unit == unc1.unit
    assert unc1.parent_nddata is unc4.parent_nddata
    assert unc1.parent_nddata is parent


def test_var_to_unknown():
    # This conversion does:
    # - not copy the data
    # - keep the values/unit/parent_nddata
    unc1 = VarianceUncertainty(np.ones(5), unit='m')
    unc2 = UnknownUncertainty.from_uncertainty(unc1)
    assert unc1.data is unc2.data
    assert unc1.unit == unc2.unit
    with pytest.raises(MissingDataAssociationException):
        unc1.parent_nddata
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # with parent
    parent = NDDataBase(1)
    parent.uncertainty = unc1

    unc3 = UnknownUncertainty.from_uncertainty(unc1)
    assert unc3.data is unc1.data
    assert unc3.unit == unc1.unit
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc1.parent_nddata is parent

    # make sure unc2 wasn't modified ...
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # Less efficient direct usage during init
    unc4 = UnknownUncertainty(unc1)
    assert unc4.data is unc1.data
    assert unc4.unit == unc1.unit
    assert unc1.parent_nddata is unc4.parent_nddata
    assert unc1.parent_nddata is parent


def test_unknown_to_relstd():
    # Different from the other ones this will fail if the unknown uncertainty
    # has a unit because relative uncertainties must not have a unit.
    unc1 = UnknownUncertainty(np.ones(5), unit='m')
    with pytest.raises(TypeError):
        RelativeUncertainty.from_uncertainty(unc1)
    with pytest.raises(TypeError):
        RelativeUncertainty(unc1)

    unc1 = UnknownUncertainty(np.ones(5))
    unc2 = RelativeUncertainty.from_uncertainty(unc1)
    assert unc1.data is unc2.data
    assert unc1.unit == unc2.unit
    with pytest.raises(MissingDataAssociationException):
        unc1.parent_nddata
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # with parent
    parent = NDDataBase(1)
    parent.uncertainty = unc1

    unc3 = RelativeUncertainty.from_uncertainty(unc1)
    assert unc3.data is unc1.data
    assert unc3.unit == unc1.unit
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc1.parent_nddata is parent

    # make sure unc2 wasn't modified ...
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # Less efficient direct usage during init
    unc4 = RelativeUncertainty(unc1)
    assert unc4.data is unc1.data
    assert unc4.unit == unc1.unit
    assert unc1.parent_nddata is unc4.parent_nddata
    assert unc1.parent_nddata is parent


def test_relstd_to_unknown():
    # relative uncertainties don't have a unit, so don't try it
    unc1 = RelativeUncertainty(np.ones(5))
    unc2 = UnknownUncertainty.from_uncertainty(unc1)
    assert unc1.data is unc2.data
    assert unc1.unit == unc2.unit
    with pytest.raises(MissingDataAssociationException):
        unc1.parent_nddata
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # with parent
    parent = NDDataBase(1)
    parent.uncertainty = unc1

    unc3 = UnknownUncertainty.from_uncertainty(unc1)
    assert unc3.data is unc1.data
    assert unc3.unit == unc1.unit
    assert unc1.parent_nddata is unc3.parent_nddata
    assert unc1.parent_nddata is parent

    # make sure unc2 wasn't modified ...
    with pytest.raises(MissingDataAssociationException):
        unc2.parent_nddata

    # Less efficient direct usage during init
    unc4 = UnknownUncertainty(unc1)
    assert unc4.data is unc1.data
    assert unc4.unit == unc1.unit
    assert unc1.parent_nddata is unc4.parent_nddata
    assert unc1.parent_nddata is parent
