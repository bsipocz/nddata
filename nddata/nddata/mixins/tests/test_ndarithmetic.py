# Licensed under a 3-clause BSD style license - see LICENSE.rst

# TEST_UNICODE_LITERALS

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ...nduncertainty_stddev import StdDevUncertainty
from ...nduncertainty_unknown import UnknownUncertainty
from ...nduncertainty_var import VarianceUncertainty
from ...exceptions import IncompatibleUncertaintiesException

from ... import NDData
from astropy.units import UnitsError, Quantity
from astropy.tests.helper import pytest
from astropy import units as u

import astropy
from distutils.version import LooseVersion
astropy_1_2 = LooseVersion(astropy.__version__) >= LooseVersion('1.2')


# Alias NDDataAllMixins in case this will be renamed ... :-)
NDDataArithmetic = NDData


# Test with Data covers:
# scalars, 1D, 2D and 3D
# broadcasting between them
@pytest.mark.parametrize(('data1', 'data2'), [
                         (np.array(5), np.array(10)),
                         (np.array(5), np.arange(10)),
                         (np.array(5), np.arange(10).reshape(2, 5)),
                         (np.arange(10), np.ones(10) * 2),
                         (np.arange(10), np.ones((10, 10)) * 2),
                         (np.arange(10).reshape(2, 5), np.ones((2, 5)) * 3),
                         (np.arange(1000).reshape(20, 5, 10),
                          np.ones((20, 5, 10)) * 3)
                         ])
def test_arithmetics_data(data1, data2):

    nd1 = NDDataArithmetic(data1)
    nd2 = NDDataArithmetic(data2)

    # Addition
    nd3 = nd1.add(nd2)
    assert_array_equal(data1+data2, nd3.data)
    # Subtraction
    nd4 = nd1.subtract(nd2)
    assert_array_equal(data1-data2, nd4.data)
    # Multiplication
    nd5 = nd1.multiply(nd2)
    assert_array_equal(data1*data2, nd5.data)
    # Division
    nd6 = nd1.divide(nd2)
    assert_array_equal(data1/data2, nd6.data)
    for nd in [nd3, nd4, nd5, nd6]:
        # Check that broadcasting worked as expected
        if data1.ndim > data2.ndim:
            assert data1.shape == nd.data.shape
        else:
            assert data2.shape == nd.data.shape
        # Check all other attributes are not set
        assert nd.unit is None
        assert nd.uncertainty is None
        assert nd.mask is None
        assert len(nd.meta) == 0
        assert nd.wcs is None


# Invalid arithmetic operations for data covering:
# not broadcastable data
def test_arithmetics_data_invalid():
    nd1 = NDDataArithmetic([1, 2, 3])
    nd2 = NDDataArithmetic([1, 2])
    with pytest.raises(ValueError):
        nd1.add(nd2)


# Test with Data and unit and covers:
# identical units (even dimensionless unscaled vs. no unit),
# equivalent units (such as meter and kilometer)
# equivalent composite units (such as m/s and km/h)
@pytest.mark.parametrize(('data1', 'data2'), [
    (np.array(5) * u.s, np.array(10) * u.s),
    (np.array(5) * u.s, np.arange(10) * u.h),
    (np.array(5) * u.s, np.arange(10).reshape(2, 5) * u.min),
    (np.arange(10) * u.m / u.s, np.ones(10) * 2 * u.km / u.s),
    (np.arange(10) * u.m / u.s, np.ones((10, 10)) * 2 * u.m / u.h),
    (np.arange(10).reshape(2, 5) * u.m / u.s,
     np.ones((2, 5)) * 3 * u.km / u.h),
    (np.arange(1000).reshape(20, 5, 10),
     np.ones((20, 5, 10)) * 3 * u.dimensionless_unscaled),
    (np.array(5), np.array(10) * u.s / u.h),
    ])
def test_arithmetics_data_unit_identical(data1, data2):

    nd1 = NDDataArithmetic(data1)
    nd2 = NDDataArithmetic(data2)

    # Addition
    nd3 = nd1.add(nd2)
    ref = data1 + data2
    ref_unit, ref_data = ref.unit, ref.value
    assert_array_equal(ref_data, nd3.data)
    assert nd3.unit == ref_unit
    # Subtraction
    nd4 = nd1.subtract(nd2)
    ref = data1 - data2
    ref_unit, ref_data = ref.unit, ref.value
    assert_array_equal(ref_data, nd4.data)
    assert nd4.unit == ref_unit
    # Multiplication
    nd5 = nd1.multiply(nd2)
    ref = data1 * data2
    ref_unit, ref_data = ref.unit, ref.value
    assert_array_equal(ref_data, nd5.data)
    assert nd5.unit == ref_unit
    # Division
    nd6 = nd1.divide(nd2)
    ref = data1 / data2
    ref_unit, ref_data = ref.unit, ref.value
    assert_array_equal(ref_data, nd6.data)
    assert nd6.unit == ref_unit
    for nd in [nd3, nd4, nd5, nd6]:
        # Check that broadcasting worked as expected
        if data1.ndim > data2.ndim:
            assert data1.shape == nd.data.shape
        else:
            assert data2.shape == nd.data.shape
        # Check all other attributes are not set
        assert nd.uncertainty is None
        assert nd.mask is None
        assert len(nd.meta) == 0
        assert nd.wcs is None


# Test with Data and unit and covers:
# not identical not convertible units
# one with unit (which is not dimensionless) and one without
@pytest.mark.parametrize(('data1', 'data2'), [
    (np.array(5) * u.s, np.array(10) * u.m),
    (np.array(5) * u.Mpc, np.array(10) * u.km / u.s),
    (np.array(5) * u.Mpc, np.array(10)),
    (np.array(5), np.array(10) * u.s),
    ])
def test_arithmetics_data_unit_not_identical(data1, data2):

    nd1 = NDDataArithmetic(data1)
    nd2 = NDDataArithmetic(data2)

    # Addition should not be possible
    with pytest.raises(UnitsError):
        nd1.add(nd2)
    # Subtraction should not be possible
    with pytest.raises(UnitsError):
        nd1.subtract(nd2)
    # Multiplication is possible
    nd3 = nd1.multiply(nd2)
    ref = data1 * data2
    ref_unit, ref_data = ref.unit, ref.value
    assert_array_equal(ref_data, nd3.data)
    assert nd3.unit == ref_unit
    # Division is possible
    nd4 = nd1.divide(nd2)
    ref = data1 / data2
    ref_unit, ref_data = ref.unit, ref.value
    assert_array_equal(ref_data, nd4.data)
    assert nd4.unit == ref_unit
    for nd in [nd3, nd4]:
        # Check all other attributes are not set
        assert nd.uncertainty is None
        assert nd.mask is None
        assert len(nd.meta) == 0
        assert nd.wcs is None


# Tests with wcs (not very senseable because there is no operation between them
# covering:
# both set and identical/not identical
# one set
# None set
@pytest.mark.parametrize(('wcs1', 'wcs2'), [
    (None, None),
    (None, 5),
    (5, None),
    (5, 5),
    (7, 5),
    ])
def test_arithmetics_data_wcs(wcs1, wcs2):

    nd1 = NDDataArithmetic(1, wcs=wcs1)
    nd2 = NDDataArithmetic(1, wcs=wcs2)

    if wcs1 is None and wcs2 is None:
        ref_wcs = None
    elif wcs1 is None:
        ref_wcs = wcs2
    elif wcs2 is None:
        ref_wcs = wcs1
    else:
        ref_wcs = wcs1

    # Addition
    nd3 = nd1.add(nd2)
    assert ref_wcs == nd3.wcs
    # Subtraction
    nd4 = nd1.subtract(nd2)
    assert ref_wcs == nd3.wcs
    # Multiplication
    nd5 = nd1.multiply(nd2)
    assert ref_wcs == nd3.wcs
    # Division
    nd6 = nd1.divide(nd2)
    assert ref_wcs == nd3.wcs
    for nd in [nd3, nd4, nd5, nd6]:
        # Check all other attributes are not set
        assert nd.unit is None
        assert nd.uncertainty is None
        assert len(nd.meta) == 0
        assert nd.mask is None


# Masks are completely seperated in the NDArithmetics from the data so we need
# no correlated tests but covering:
# masks 1D, 2D and mixed cases with broadcasting
@pytest.mark.parametrize(('mask1', 'mask2'), [
    (None, None),
    (None, False),
    (True, None),
    (False, False),
    (True, False),
    (False, True),
    (True, True),
    (np.array(False), np.array(True)),
    (np.array(False), np.array([0, 1, 0, 1, 1], dtype=np.bool_)),
    (np.array(True),
     np.array([[0, 1, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=np.bool_)),
    (np.array([0, 1, 0, 1, 1], dtype=np.bool_),
     np.array([1, 1, 0, 0, 1], dtype=np.bool_)),
    (np.array([0, 1, 0, 1, 1], dtype=np.bool_),
     np.array([[0, 1, 0, 1, 1], [1, 0, 0, 1, 1]], dtype=np.bool_)),
    (np.array([[0, 1, 0, 1, 1], [1, 0, 0, 1, 1]], dtype=np.bool_),
     np.array([[0, 1, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=np.bool_)),
    ])
def test_arithmetics_data_masks(mask1, mask2):

    nd1 = NDDataArithmetic(1, mask=mask1)
    nd2 = NDDataArithmetic(1, mask=mask2)

    if mask1 is None and mask2 is None:
        ref_mask = None
    elif mask1 is None:
        ref_mask = mask2
    elif mask2 is None:
        ref_mask = mask1
    else:
        ref_mask = mask1 | mask2

    # Addition
    nd3 = nd1.add(nd2)
    assert_array_equal(ref_mask, nd3.mask)
    # Subtraction
    nd4 = nd1.subtract(nd2)
    assert_array_equal(ref_mask, nd4.mask)
    # Multiplication
    nd5 = nd1.multiply(nd2)
    assert_array_equal(ref_mask, nd5.mask)
    # Division
    nd6 = nd1.divide(nd2)
    assert_array_equal(ref_mask, nd6.mask)
    for nd in [nd3, nd4, nd5, nd6]:
        # Check all other attributes are not set
        assert nd.unit is None
        assert nd.uncertainty is None
        assert len(nd.meta) == 0
        assert nd.wcs is None


# One additional case which can not be easily incorporated in the test above
# what happens if the masks are numpy ndarrays are not broadcastable
def test_arithmetics_data_masks_invalid():

    nd1 = NDDataArithmetic(1, mask=np.array([1, 0], dtype=np.bool_))
    nd2 = NDDataArithmetic(1, mask=np.array([1, 0, 1], dtype=np.bool_))

    with pytest.raises(ValueError):
        nd1.add(nd2)
    with pytest.raises(ValueError):
        nd1.multiply(nd2)
    with pytest.raises(ValueError):
        nd1.subtract(nd2)
    with pytest.raises(ValueError):
        nd1.divide(nd2)


def test_uncertainty_fail():
    class FakeUncertainty(object):
        def __init__(self):
            pass

        uncertainty_type = 'std'

    ndd1 = NDDataArithmetic([0, 1, 3], uncertainty=FakeUncertainty())
    ndd2 = NDDataArithmetic([0, 1, 1])
    with pytest.raises(TypeError):
        ndd1.add(ndd2)

    with pytest.raises(TypeError):
        ndd2.add(ndd1)


# Covering:
# both have uncertainties (data and uncertainty without unit)
# tested against manually determined resulting uncertainties to verify the
# implemented formulas
# this test only works as long as data1 and data2 do not contain any 0
def test_arithmetics_stddevuncertainty_basic():
    nd1 = NDDataArithmetic([1, 2, 3], uncertainty=StdDevUncertainty([1, 1, 3]))
    nd2 = NDDataArithmetic([2, 2, 2], uncertainty=StdDevUncertainty([2, 2, 2]))
    nd3 = nd1.add(nd2)
    nd4 = nd2.add(nd1)
    # Inverse operation should result in the same uncertainty
    assert_array_equal(nd3.uncertainty.data, nd4.uncertainty.data)
    # Compare it to the theoretical uncertainty
    ref_uncertainty = np.sqrt(np.array([1, 1, 3])**2 + np.array([2, 2, 2])**2)
    assert_array_equal(nd3.uncertainty.data, ref_uncertainty)

    nd3 = nd1.subtract(nd2)
    nd4 = nd2.subtract(nd1)
    # Inverse operation should result in the same uncertainty
    assert_array_equal(nd3.uncertainty.data, nd4.uncertainty.data)
    # Compare it to the theoretical uncertainty (same as for add)
    assert_array_equal(nd3.uncertainty.data, ref_uncertainty)

    # Multiplication and Division only work with almost equal array comparisons
    # since the formula implemented and the formula used as reference are
    # slightly different.
    nd3 = nd1.multiply(nd2)
    nd4 = nd2.multiply(nd1)
    # Inverse operation should result in the same uncertainty
    assert_array_almost_equal(nd3.uncertainty.data, nd4.uncertainty.data)
    # Compare it to the theoretical uncertainty
    ref_uncertainty = np.abs(np.array([2, 4, 6])) * np.sqrt(
        (np.array([1, 1, 3]) / np.array([1, 2, 3]))**2 +
        (np.array([2, 2, 2]) / np.array([2, 2, 2]))**2)
    assert_array_almost_equal(nd3.uncertainty.data, ref_uncertainty)

    nd3 = nd1.divide(nd2)
    nd4 = nd2.divide(nd1)
    # Inverse operation gives a different uncertainty!
    # Compare it to the theoretical uncertainty
    ref_uncertainty_1 = np.abs(np.array([1/2, 2/2, 3/2])) * np.sqrt(
        (np.array([1, 1, 3]) / np.array([1, 2, 3]))**2 +
        (np.array([2, 2, 2]) / np.array([2, 2, 2]))**2)
    assert_array_almost_equal(nd3.uncertainty.data, ref_uncertainty_1)
    ref_uncertainty_2 = np.abs(np.array([2, 1, 2/3])) * np.sqrt(
        (np.array([1, 1, 3]) / np.array([1, 2, 3]))**2 +
        (np.array([2, 2, 2]) / np.array([2, 2, 2]))**2)
    assert_array_almost_equal(nd4.uncertainty.data, ref_uncertainty_2)


# Tests for correlation, covering
# correlation between -1 and 1 with correlation term being positive / negative
# also with one data being once positive and once completly negative
# The point of this test is to compare the used formula to the theoretical one.
# TODO: Maybe covering units too but I think that should work because of
# the next tests. Also this may be reduced somehow.
@pytest.mark.parametrize(('cor', 'uncert1', 'data2'), [
    (-1, [1, 1, 3], [2, 2, 7]),
    (-0.5, [1, 1, 3], [2, 2, 7]),
    (-0.25, [1, 1, 3], [2, 2, 7]),
    (0, [1, 1, 3], [2, 2, 7]),
    (0.25, [1, 1, 3], [2, 2, 7]),
    (0.5, [1, 1, 3], [2, 2, 7]),
    (1, [1, 1, 3], [2, 2, 7]),
    (-1, [-1, -1, -3], [2, 2, 7]),
    (-0.5, [-1, -1, -3], [2, 2, 7]),
    (-0.25, [-1, -1, -3], [2, 2, 7]),
    (0, [-1, -1, -3], [2, 2, 7]),
    (0.25, [-1, -1, -3], [2, 2, 7]),
    (0.5, [-1, -1, -3], [2, 2, 7]),
    (1, [-1, -1, -3], [2, 2, 7]),
    (-1, [1, 1, 3], [-2, -3, -2]),
    (-0.5, [1, 1, 3], [-2, -3, -2]),
    (-0.25, [1, 1, 3], [-2, -3, -2]),
    (0, [1, 1, 3], [-2, -3, -2]),
    (0.25, [1, 1, 3], [-2, -3, -2]),
    (0.5, [1, 1, 3], [-2, -3, -2]),
    (1, [1, 1, 3], [-2, -3, -2]),
    (-1, [-1, -1, -3], [-2, -3, -2]),
    (-0.5, [-1, -1, -3], [-2, -3, -2]),
    (-0.25, [-1, -1, -3], [-2, -3, -2]),
    (0, [-1, -1, -3], [-2, -3, -2]),
    (0.25, [-1, -1, -3], [-2, -3, -2]),
    (0.5, [-1, -1, -3], [-2, -3, -2]),
    (1, [-1, -1, -3], [-2, -3, -2]),
    ])
def test_arithmetics_stddevuncertainty_basic_with_correlation(
        cor, uncert1, data2):
    data1 = np.array([1, 2, 3])
    data2 = np.array(data2)
    uncert1 = np.array(uncert1)
    uncert2 = np.array([2, 2, 2])
    nd1 = NDDataArithmetic(data1, uncertainty=StdDevUncertainty(uncert1))
    nd2 = NDDataArithmetic(data2, uncertainty=StdDevUncertainty(uncert2))
    nd3 = nd1.add(nd2, uncertainty_correlation=cor)
    nd4 = nd2.add(nd1, uncertainty_correlation=cor)
    # Inverse operation should result in the same uncertainty
    assert_array_equal(nd3.uncertainty.data, nd4.uncertainty.data)
    # Compare it to the theoretical uncertainty
    ref_uncertainty = np.sqrt(uncert1**2 + uncert2**2 +
                              2 * cor * uncert1 * uncert2)
    assert_array_equal(nd3.uncertainty.data, ref_uncertainty)

    nd3 = nd1.subtract(nd2, uncertainty_correlation=cor)
    nd4 = nd2.subtract(nd1, uncertainty_correlation=cor)
    # Inverse operation should result in the same uncertainty
    assert_array_equal(nd3.uncertainty.data, nd4.uncertainty.data)
    # Compare it to the theoretical uncertainty
    ref_uncertainty = np.sqrt(uncert1**2 + uncert2**2 -
                              2 * cor * uncert1 * uncert2)
    assert_array_equal(nd3.uncertainty.data, ref_uncertainty)

    # Multiplication and Division only work with almost equal array comparisons
    # since the formula implemented and the formula used as reference are
    # slightly different.
    nd3 = nd1.multiply(nd2, uncertainty_correlation=cor)
    nd4 = nd2.multiply(nd1, uncertainty_correlation=cor)
    # Inverse operation should result in the same uncertainty
    assert_array_almost_equal(nd3.uncertainty.data, nd4.uncertainty.data)
    # Compare it to the theoretical uncertainty
    ref_uncertainty = (np.abs(data1 * data2)) * np.sqrt(
        (uncert1 / data1)**2 + (uncert2 / data2)**2 +
        (2 * cor * uncert1 * uncert2 / (data1 * data2)))
    assert_array_almost_equal(nd3.uncertainty.data, ref_uncertainty)

    nd3 = nd1.divide(nd2, uncertainty_correlation=cor)
    nd4 = nd2.divide(nd1, uncertainty_correlation=cor)
    # Inverse operation gives a different uncertainty!
    # Compare it to the theoretical uncertainty
    ref_uncertainty_1 = (np.abs(data1 / data2)) * np.sqrt(
        (uncert1 / data1)**2 + (uncert2 / data2)**2 -
        (2 * cor * uncert1 * uncert2 / (data1 * data2)))
    assert_array_almost_equal(nd3.uncertainty.data, ref_uncertainty_1)
    ref_uncertainty_2 = (np.abs(data2 / data1)) * np.sqrt(
        (uncert1 / data1)**2 + (uncert2 / data2)**2 -
        (2 * cor * uncert1 * uncert2 / (data1 * data2)))
    assert_array_almost_equal(nd4.uncertainty.data, ref_uncertainty_2)


# Covering:
# just an example that a np.ndarray works as correlation, no checks for
# the right result since these were basically done in the function above.
def test_arithmetics_stddevuncertainty_basic_with_correlation_array():
    data1 = np.array([1, 2, 3])
    data2 = np.array([1, 1, 1])
    uncert1 = np.array([1, 1, 1])
    uncert2 = np.array([2, 2, 2])
    cor = np.array([0, 0.25, 0])
    nd1 = NDDataArithmetic(data1, uncertainty=StdDevUncertainty(uncert1))
    nd2 = NDDataArithmetic(data2, uncertainty=StdDevUncertainty(uncert2))
    nd1.add(nd2, uncertainty_correlation=cor)


# Covering:
# That propagate throws an exception when correlation is given but the
# uncertainty does not support correlation.
def test_arithmetics_with_correlation_unsupported():

    class StdDevUncertaintyUncorrelated(StdDevUncertainty):
        @property
        def supports_correlated(self):
            return False

    class VarianceUncertaintyUncorrelated(VarianceUncertainty):
        @property
        def supports_correlated(self):
            return False

    data1 = np.array([1, 2, 3])
    data2 = np.array([1, 1, 1])
    uncert1 = np.array([1, 1, 1])
    uncert2 = np.array([2, 2, 2])
    cor = 3
    nd1 = NDDataArithmetic(data1,
                           uncertainty=StdDevUncertaintyUncorrelated(uncert1))
    nd2 = NDDataArithmetic(data2,
                           uncertainty=StdDevUncertaintyUncorrelated(uncert2))

    with pytest.raises(ValueError):
        nd1.add(nd2, uncertainty_correlation=cor)

    unc1 = VarianceUncertaintyUncorrelated(uncert1)
    unc2 = VarianceUncertaintyUncorrelated(uncert2)
    nd1 = NDDataArithmetic(data1, uncertainty=unc1)
    nd2 = NDDataArithmetic(data2, uncertainty=unc2)

    with pytest.raises(ValueError):
        nd1.add(nd2, uncertainty_correlation=cor)


# Covering:
# only one has an uncertainty (data and uncertainty without unit)
# tested against the case where the other one has zero uncertainty. (this case
# must be correct because we tested it in the last case)
# Also verify that if the result of the data has negative values the resulting
# uncertainty has no negative values.
def test_arithmetics_stddevuncertainty_one_missing():
    nd1 = NDDataArithmetic([1, -2, 3])
    nd1_ref = NDDataArithmetic([1, -2, 3],
                               uncertainty=StdDevUncertainty([0, 0, 0]))
    nd2 = NDDataArithmetic([2, 2, -2],
                           uncertainty=StdDevUncertainty([2, 2, 2]))

    # Addition
    nd3 = nd1.add(nd2)
    nd3_ref = nd1_ref.add(nd2)
    assert_array_equal(nd3.uncertainty.data, nd3_ref.uncertainty.data)
    assert_array_equal(np.abs(nd3.uncertainty.data), nd3.uncertainty.data)

    nd3 = nd2.add(nd1)
    nd3_ref = nd2.add(nd1_ref)
    assert_array_equal(nd3.uncertainty.data, nd3_ref.uncertainty.data)
    assert_array_equal(np.abs(nd3.uncertainty.data), nd3.uncertainty.data)

    # Subtraction
    nd3 = nd1.subtract(nd2)
    nd3_ref = nd1_ref.subtract(nd2)
    assert_array_equal(nd3.uncertainty.data, nd3_ref.uncertainty.data)
    assert_array_equal(np.abs(nd3.uncertainty.data), nd3.uncertainty.data)

    nd3 = nd2.subtract(nd1)
    nd3_ref = nd2.subtract(nd1_ref)
    assert_array_equal(nd3.uncertainty.data, nd3_ref.uncertainty.data)
    assert_array_equal(np.abs(nd3.uncertainty.data), nd3.uncertainty.data)

    # Multiplication
    nd3 = nd1.multiply(nd2)
    nd3_ref = nd1_ref.multiply(nd2)
    assert_array_equal(nd3.uncertainty.data, nd3_ref.uncertainty.data)
    assert_array_equal(np.abs(nd3.uncertainty.data), nd3.uncertainty.data)

    nd3 = nd2.multiply(nd1)
    nd3_ref = nd2.multiply(nd1_ref)
    assert_array_equal(nd3.uncertainty.data, nd3_ref.uncertainty.data)
    assert_array_equal(np.abs(nd3.uncertainty.data), nd3.uncertainty.data)

    # Division
    nd3 = nd1.divide(nd2)
    nd3_ref = nd1_ref.divide(nd2)
    assert_array_equal(nd3.uncertainty.data, nd3_ref.uncertainty.data)
    assert_array_equal(np.abs(nd3.uncertainty.data), nd3.uncertainty.data)

    nd3 = nd2.divide(nd1)
    nd3_ref = nd2.divide(nd1_ref)
    assert_array_equal(nd3.uncertainty.data, nd3_ref.uncertainty.data)
    assert_array_equal(np.abs(nd3.uncertainty.data), nd3.uncertainty.data)


# Covering:
# data with unit and uncertainty with unit (but equivalent units)
# compared against correctly scaled NDDatas
@pytest.mark.parametrize(('uncert1', 'uncert2'), [
    (np.array([1, 2, 3]) * u.m, None),
    (np.array([1, 2, 3]) * u.cm, None),
    (None, np.array([1, 2, 3]) * u.m),
    (None, np.array([1, 2, 3]) * u.cm),
    (np.array([1, 2, 3]), np.array([2, 3, 4])),
    (np.array([1, 2, 3]) * u.m, np.array([2, 3, 4])),
    (np.array([1, 2, 3]), np.array([2, 3, 4])) * u.m,
    (np.array([1, 2, 3]) * u.m, np.array([2, 3, 4])) * u.m,
    (np.array([1, 2, 3]) * u.cm, np.array([2, 3, 4])),
    (np.array([1, 2, 3]), np.array([2, 3, 4])) * u.cm,
    (np.array([1, 2, 3]) * u.cm, np.array([2, 3, 4])) * u.cm,
    (np.array([1, 2, 3]) * u.km, np.array([2, 3, 4])) * u.cm,
    ])
def test_arithmetics_stddevuncertainty_with_units(uncert1, uncert2):
    # Data has same units
    data1 = np.array([1, 2, 3]) * u.m
    data2 = np.array([-4, 7, 0]) * u.m
    if uncert1 is not None:
        uncert1 = StdDevUncertainty(uncert1)
        if isinstance(uncert1, Quantity):
            uncert1_ref = uncert1.to(data1.unit).value
        else:
            uncert1_ref = uncert1
        uncert_ref1 = StdDevUncertainty(uncert1_ref, copy=True)
    else:
        uncert1 = None
        uncert_ref1 = None

    if uncert2 is not None:
        uncert2 = StdDevUncertainty(uncert2)
        if isinstance(uncert2, Quantity):
            uncert2_ref = uncert2.to(data2.unit).value
        else:
            uncert2_ref = uncert2
        uncert_ref2 = StdDevUncertainty(uncert2_ref, copy=True)
    else:
        uncert2 = None
        uncert_ref2 = None

    nd1 = NDDataArithmetic(data1, uncertainty=uncert1)
    nd2 = NDDataArithmetic(data2, uncertainty=uncert2)

    nd1_ref = NDDataArithmetic(data1, uncertainty=uncert_ref1)
    nd2_ref = NDDataArithmetic(data2, uncertainty=uncert_ref2)

    # Let's start the tests
    # Addition
    nd3 = nd1.add(nd2)
    nd3_ref = nd1_ref.add(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.data, nd3.uncertainty.data)

    nd3 = nd2.add(nd1)
    nd3_ref = nd2_ref.add(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.data, nd3.uncertainty.data)

    # Subtraction
    nd3 = nd1.subtract(nd2)
    nd3_ref = nd1_ref.subtract(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.data, nd3.uncertainty.data)

    nd3 = nd2.subtract(nd1)
    nd3_ref = nd2_ref.subtract(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.data, nd3.uncertainty.data)

    # Multiplication
    nd3 = nd1.multiply(nd2)
    nd3_ref = nd1_ref.multiply(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.data, nd3.uncertainty.data)

    nd3 = nd2.multiply(nd1)
    nd3_ref = nd2_ref.multiply(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.data, nd3.uncertainty.data)

    # Division
    nd3 = nd1.divide(nd2)
    nd3_ref = nd1_ref.divide(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.data, nd3.uncertainty.data)

    nd3 = nd2.divide(nd1)
    nd3_ref = nd2_ref.divide(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.data, nd3.uncertainty.data)


# Test abbreviation and long name for taking the first found meta, mask, wcs
@pytest.mark.parametrize(('use_abbreviation'), ['ff', 'first_found'])
def test_arithmetics_handle_switches(use_abbreviation):
    meta1 = {'a': 1}
    meta2 = {'b': 2}
    mask1 = True
    mask2 = False
    flags1 = 20
    flags2 = 5
    uncertainty1 = StdDevUncertainty([1, 2, 3])
    uncertainty2 = StdDevUncertainty([1, 2, 3])
    wcs1 = 5
    wcs2 = 100
    data1 = [1, 1, 1]
    data2 = [1, 1, 1]

    nd1 = NDDataArithmetic(data1, meta=meta1, mask=mask1, wcs=wcs1,
                           uncertainty=uncertainty1, flags=flags1)
    nd2 = NDDataArithmetic(data2, meta=meta2, mask=mask2, wcs=wcs2,
                           uncertainty=uncertainty2, flags=flags2)
    nd3 = NDDataArithmetic(data1)

    # Both have the attributes but option None is chosen
    nd_ = nd1.add(nd2, propagate_uncertainties=None, handle_meta=None,
                  handle_mask=None, compare_wcs=None, handle_flags=None)
    assert nd_.wcs is None
    assert len(nd_.meta) == 0
    assert nd_.mask is None
    assert nd_.uncertainty is None
    assert nd_.flags is None

    # Only second has attributes and False/firstfound is chosen
    nd_ = nd3.add(nd2, propagate_uncertainties=False,
                  handle_meta=use_abbreviation, handle_mask=use_abbreviation,
                  compare_wcs=use_abbreviation, handle_flags=use_abbreviation)
    assert nd_.wcs == wcs2
    assert nd_.meta == meta2
    assert nd_.mask == mask2
    assert nd_.flags == flags2
    assert_array_equal(nd_.uncertainty.data, uncertainty2.data)

    # Only first has attributes and False is chosen
    nd_ = nd1.add(nd3, propagate_uncertainties=False,
                  handle_meta=use_abbreviation, handle_mask=use_abbreviation,
                  compare_wcs=use_abbreviation, handle_flags=use_abbreviation)
    assert nd_.wcs == wcs1
    assert nd_.meta == meta1
    assert nd_.mask == mask1
    assert nd_.flags == flags1
    assert_array_equal(nd_.uncertainty.data, uncertainty1.data)


def test_arithmetics_meta_func():
    def meta_fun_func(meta1, meta2, take='first'):
        if take == 'first':
            return meta1
        else:
            return meta2

    meta1 = {'a': 1}
    meta2 = {'a': 3, 'b': 2}
    mask1 = True
    mask2 = False
    uncertainty1 = StdDevUncertainty([1, 2, 3])
    uncertainty2 = StdDevUncertainty([1, 2, 3])
    wcs1 = 5
    wcs2 = 100
    data1 = [1, 1, 1]
    data2 = [1, 1, 1]

    nd1 = NDDataArithmetic(data1, meta=meta1, mask=mask1, wcs=wcs1,
                           uncertainty=uncertainty1)
    nd2 = NDDataArithmetic(data2, meta=meta2, mask=mask2, wcs=wcs2,
                           uncertainty=uncertainty2)

    nd3 = nd1.add(nd2, handle_meta=meta_fun_func)
    assert nd3.meta['a'] == 1
    assert 'b' not in nd3.meta

    nd4 = nd1.add(nd2, handle_meta=meta_fun_func, meta_take='second')
    assert nd4.meta['a'] == 3
    assert nd4.meta['b'] == 2

    with pytest.raises(KeyError):
        nd1.add(nd2, handle_meta=meta_fun_func, take='second')


def test_arithmetics_flags_func():
    def flags_func(flags1, flags2, bitwise=True):
        if bitwise:
            return np.bitwise_or(flags1, flags2)
        else:
            return np.logical_or(flags1, flags2)

    meta1 = {'a': 1}
    meta2 = {'a': 3, 'b': 2}
    mask1 = True
    mask2 = False
    flags1 = np.array([0, 1, 3])
    flags2 = np.array([0, 2, 1])
    uncertainty1 = StdDevUncertainty([1, 2, 3])
    uncertainty2 = StdDevUncertainty([1, 2, 3])
    wcs1 = 5
    wcs2 = 100
    data1 = [1, 1, 1]
    data2 = [1, 1, 1]

    nd1 = NDDataArithmetic(data1, meta=meta1, mask=mask1, wcs=wcs1,
                           uncertainty=uncertainty1, flags=flags1)
    nd2 = NDDataArithmetic(data2, meta=meta2, mask=mask2, wcs=wcs2,
                           uncertainty=uncertainty2, flags=flags2)

    nd3 = nd1.add(nd2, handle_flags=flags_func)
    assert_array_equal(nd3.flags, [0, 3, 3])

    nd4 = nd2.add(nd1, handle_flags=flags_func)
    assert_array_equal(nd4.flags, [0, 3, 3])

    nd5 = nd1.add(nd2, handle_flags=flags_func, flags_bitwise=False)
    assert_array_equal(nd5.flags, [False, True, True])


def test_arithmetics_wcs_func():
    def wcs_comp_func(wcs1, wcs2, tolerance=0.1):
        if abs(wcs1 - wcs2) <= tolerance:
            return True
        else:
            return False

    meta1 = {'a': 1}
    meta2 = {'a': 3, 'b': 2}
    mask1 = True
    mask2 = False
    uncertainty1 = StdDevUncertainty([1, 2, 3])
    uncertainty2 = StdDevUncertainty([1, 2, 3])
    wcs1 = 99.99
    wcs2 = 100
    data1 = [1, 1, 1]
    data2 = [1, 1, 1]

    nd1 = NDDataArithmetic(data1, meta=meta1, mask=mask1, wcs=wcs1,
                           uncertainty=uncertainty1)
    nd2 = NDDataArithmetic(data2, meta=meta2, mask=mask2, wcs=wcs2,
                           uncertainty=uncertainty2)

    nd3 = nd1.add(nd2, compare_wcs=wcs_comp_func)
    assert nd3.wcs == 99.99

    with pytest.raises(ValueError):
        nd1.add(nd2, compare_wcs=wcs_comp_func, wcs_tolerance=0.00001)

    with pytest.raises(KeyError):
        nd1.add(nd2, compare_wcs=wcs_comp_func, tolerance=1)


def test_arithmetics_mask_func():
    def mask_sad_func(mask1, mask2, fun=0):
        if fun > 0.5:
            return mask2
        else:
            return mask1

    meta1 = {'a': 1}
    meta2 = {'a': 3, 'b': 2}
    mask1 = [True, False, True]
    mask2 = [True, False, False]
    uncertainty1 = StdDevUncertainty([1, 2, 3])
    uncertainty2 = StdDevUncertainty([1, 2, 3])
    wcs1 = 99.99
    wcs2 = 100
    data1 = [1, 1, 1]
    data2 = [1, 1, 1]

    nd1 = NDDataArithmetic(data1, meta=meta1, mask=mask1, wcs=wcs1,
                           uncertainty=uncertainty1)
    nd2 = NDDataArithmetic(data2, meta=meta2, mask=mask2, wcs=wcs2,
                           uncertainty=uncertainty2)

    nd3 = nd1.add(nd2, handle_mask=mask_sad_func)
    assert_array_equal(nd3.mask, nd1.mask)

    nd4 = nd1.add(nd2, handle_mask=mask_sad_func, mask_fun=1)
    assert_array_equal(nd4.mask, nd2.mask)

    with pytest.raises(KeyError):
        nd1.add(nd2, handle_mask=mask_sad_func, fun=1)


def test_classmethod_fail():
    with pytest.raises(TypeError):
        NDDataArithmetic.add([1, 2, 3])


@pytest.mark.parametrize('meth', ['add', 'subtract', 'divide', 'multiply'])
def test_two_argument_useage(meth):
    ndd1 = NDDataArithmetic(np.ones((3, 3)))
    ndd2 = NDDataArithmetic(np.ones((3, 3)))

    # Call add on the class (not the instance) and compare it with already
    # tested useage:
    ndd3 = getattr(NDDataArithmetic, meth)(ndd1, ndd2)
    ndd4 = getattr(ndd1, meth)(ndd2)
    np.testing.assert_array_equal(ndd3.data, ndd4.data)

    # And the same done on an unrelated instance...
    ndd3 = getattr(NDDataArithmetic(-100), meth)(ndd1, ndd2)
    ndd4 = getattr(ndd1, meth)(ndd2)
    np.testing.assert_array_equal(ndd3.data, ndd4.data)


@pytest.mark.parametrize('meth', ['add', 'subtract', 'divide', 'multiply'])
def test_two_argument_useage_non_nddata_first_arg(meth):
    data1 = 50
    data2 = 100

    # Call add on the class (not the instance)
    ndd3 = getattr(NDDataArithmetic, meth)(data1, data2)

    # Compare it with the instance-useage and two identical NDData-like
    # classes:
    ndd1 = NDDataArithmetic(data1)
    ndd2 = NDDataArithmetic(data2)
    ndd4 = getattr(ndd1, meth)(ndd2)
    np.testing.assert_array_equal(ndd3.data, ndd4.data)

    # and check it's also working when called on an instance
    ndd3 = getattr(NDDataArithmetic(-100), meth)(data1, data2)
    ndd4 = getattr(ndd1, meth)(ndd2)
    np.testing.assert_array_equal(ndd3.data, ndd4.data)


def test_arithmetics_unknown_uncertainties():
    # Not giving any uncertainty class means it is saved as UnknownUncertainty
    ndd1 = NDDataArithmetic(np.ones((3, 3)),
                            uncertainty=UnknownUncertainty(np.ones((3, 3))))
    ndd2 = NDDataArithmetic(np.ones((3, 3)),
                            uncertainty=UnknownUncertainty(np.ones((3, 3))*2))
    # There is no way to propagate uncertainties:
    with pytest.raises(TypeError):
        ndd1.add(ndd2)
    # But it should be possible without propagation
    ndd3 = ndd1.add(ndd2, propagate_uncertainties=False)
    np.testing.assert_array_equal(ndd1.uncertainty.data, ndd3.uncertainty.data)

    ndd4 = ndd1.add(ndd2, propagate_uncertainties=None)
    assert ndd4.uncertainty is None


def test_uncertainty_impossible_operation():
    impossible_operation = np.log

    ndd2 = NDDataArithmetic([1, 2, 3], StdDevUncertainty(None))
    uncert1 = StdDevUncertainty([1, 2, 3])
    # TODO: One shouldn't call propagate directly but for now I have no
    # unsupported arithmetic operations that cannot be propagated. So better
    # to do this than don't test it.
    with pytest.raises(ValueError):
        # Parameter 3 and 4 are just stubs... they shouldn't affect the
        # behaviour in THIS case.
        uncert1.propagate(impossible_operation, ndd2, np.array([1, 2, 3]), 0)

    ndd2 = NDDataArithmetic([1, 2, 3], VarianceUncertainty(None))
    uncert1 = VarianceUncertainty([1, 2, 3])
    with pytest.raises(ValueError):
        uncert1.propagate(impossible_operation, ndd2, np.array([1, 2, 3]), 0)


# Power operation.

def test_power_data_unit():
    # Test it with classmethods, other uses should be the same.

    # Case 1: Two scalars
    ndd = NDDataArithmetic.power(2, 2)
    assert ndd.data == 4

    # Case 2: Scalar Quantity and scalar
    ndd = NDDataArithmetic.power(2 * u.m, 4)
    assert ndd.data == 2 ** 4
    assert ndd.unit == u.m ** 4

    # Case 3: Quantity Array and scalar
    ndd = NDDataArithmetic.power(np.array([1, 2, 3]) * u.cm, 0.5)
    np.testing.assert_almost_equal(ndd.data, np.sqrt([1, 2, 3]))
    assert ndd.unit == u.cm ** 0.5

    # Case 4: Array and array
    ndd = NDDataArithmetic.power(np.array([1, 2, 3]), np.array([1, 2, 3]))
    np.testing.assert_almost_equal(ndd.data, np.array([1, 4, 27]))

    # Cases that don't work because of Quantity framework:
    # Case 5: Quantity Array and array
    with pytest.raises(ValueError):
        NDDataArithmetic.power(np.array([1, 2, 3]) * u.cm, [1, 2, 3])

    # Case 6: Exponent has unit
    with pytest.raises(TypeError):
        NDDataArithmetic.power(2, 4 * u.m)


# TODO: Check if this test needs to be skipped if astropy < 1.2
def test_power_first_op_uncertainty():
    # Case 1: Only first operand has uncertainty and second operand is really
    # dimensionless
    ndd = NDDataArithmetic(10, uncertainty=StdDevUncertainty(5))
    ndd2 = ndd.power(3)
    assert ndd2.data == 1000
    assert ndd2.unit is None
    assert ndd2.uncertainty.data == 1500
    assert ndd2.uncertainty.unit is None
    # Check that this is identical to dimensionless exponent (with unit)
    ndd3 = ndd.power(3*u.dimensionless_unscaled)
    assert ndd2.data == ndd3.data
    assert ndd2.uncertainty.data == ndd3.uncertainty.data
    assert ndd2.uncertainty.unit == ndd3.uncertainty.unit

    # Case 2: Like case 1 but this time the exponent has a unit that can be
    # converted to dimensionless
    ndd = NDDataArithmetic(10, uncertainty=StdDevUncertainty(5))
    ndd2 = ndd.power(3 * u.cm / u.mm)
    assert ndd2.data == 1e30
    assert ndd2.unit == u.dimensionless_unscaled
    # really big values so only compare like 8 valid decimals
    # unfortunatly numpy testing wasn't very useful here...
    assert np.abs(ndd2.uncertainty.data - 1.5e31) < 1e23
    assert ndd2.uncertainty.unit is None
    # Check this against a converted case
    ndd3 = ndd.power(30.)
    assert ndd2.data == ndd3.data
    assert ndd2.uncertainty.data == ndd3.uncertainty.data
    assert ndd2.uncertainty.unit == ndd3.uncertainty.unit

    # Case 3: Base has different unit for data and uncertainty
    ndd = NDDataArithmetic(10, unit='m',
                           uncertainty=StdDevUncertainty(5, unit='cm'))
    ndd2 = ndd.power(2)
    assert ndd2.data == 100
    assert ndd2.unit == u.m ** 2
    assert ndd2.uncertainty.data == 10000
    assert ndd2.uncertainty.unit == u.cm ** 2
    # Compare this against converted case
    ndd = NDDataArithmetic(10, unit='m',
                           uncertainty=StdDevUncertainty(0.05))
    ndd3 = ndd.power(2)
    assert ndd2.data == ndd3.data
    assert ndd2.unit == ndd3.unit
    assert ndd2.uncertainty.data == ndd3.uncertainty.data * 10000
    assert ndd3.uncertainty.unit is None  # the uncertainty should have no unit

    # Case 4: Base has different unit for data and uncertainty
    ndd = NDDataArithmetic(10, uncertainty=StdDevUncertainty(500, unit='cm/m'))
    ndd2 = ndd.power(2)
    assert ndd2.data == 100
    assert ndd2.uncertainty.data == 100
    # Compare this against converted case
    ndd = NDDataArithmetic(10, unit='m',
                           uncertainty=StdDevUncertainty(5))
    ndd3 = ndd.power(2)
    assert ndd2.data == ndd3.data
    assert ndd2.uncertainty.data == ndd3.uncertainty.data

    # Test the trivial case that ** 1 leaves the uncertainty be and 0 sets it
    # to zero
    ndd = NDDataArithmetic(10, unit='m', uncertainty=StdDevUncertainty(0.05))
    assert ndd.power(1).uncertainty.data == ndd.uncertainty.data
    assert ndd.power(0).uncertainty.data == 0


def test_power_second_op_uncertainty():
    # The second operand must be dimensionless or convertable to dimensionless
    # So let's check:

    # Case 1: Exponent unitless and unitless uncertainty
    ndd = NDDataArithmetic(10, uncertainty=StdDevUncertainty(5))
    ndd2 = ndd.power(2, ndd)
    assert ndd2.data == 1024
    assert ndd2.unit is None
    np.testing.assert_almost_equal(ndd2.uncertainty.data,
                                   1024 * np.log(2) * 5)
    # Formula is result (1024) * ln(operand1) (ln(2)) * uncertainty2 (5)
    assert ndd2.uncertainty.unit is None

    # Compare this to dimensionless exponent
    ndd = NDDataArithmetic(10, unit='', uncertainty=StdDevUncertainty(5))
    ndd3 = ndd.power(2, ndd)
    assert ndd2.data == ndd3.data
    # The unit is somehow convolved because of power ... don't compare it.
    np.testing.assert_almost_equal(ndd2.uncertainty.data,
                                   ndd3.uncertainty.data)
    assert ndd2.uncertainty.unit is None

    # and to an exponent with a unit that can be converted to dimensionless
    ndd = NDDataArithmetic(10, uncertainty=StdDevUncertainty(500, unit='cm/m'))
    ndd3 = ndd.power(2, ndd)
    assert ndd2.data == ndd3.data
    # The unit is somehow convolved because of power ... don't compare it.
    np.testing.assert_almost_equal(ndd2.uncertainty.data,
                                   ndd3.uncertainty.data)
    assert ndd2.uncertainty.unit is None


def test_power_both_uncertainty():
    # Easy test first: Both uncertainties but both of them zero
    ndd1 = NDDataArithmetic(2, uncertainty=StdDevUncertainty(0))
    ndd2 = NDDataArithmetic(3, uncertainty=StdDevUncertainty(0))

    nddres = ndd1.power(ndd2)
    assert nddres.data == 8
    assert nddres.uncertainty.data == 0

    # Both of them set
    ndd1 = NDDataArithmetic(2, uncertainty=StdDevUncertainty(2))
    ndd2 = NDDataArithmetic(3, uncertainty=StdDevUncertainty(2))

    nddres = ndd1.power(ndd2)
    assert nddres.data == 8
    np.testing.assert_almost_equal(nddres.uncertainty.data,
                                   np.sqrt((8*3*2/2)**2+(8*np.log(2)*2)**2))

    # Compare this to dimensionless operands
    ndd1 = NDDataArithmetic(2, unit='', uncertainty=StdDevUncertainty(2))
    ndd2 = NDDataArithmetic(3, unit='', uncertainty=StdDevUncertainty(2))
    nddres2 = ndd1.power(ndd2)
    assert nddres.data == nddres2.data
    np.testing.assert_almost_equal(nddres.uncertainty.data,
                                   nddres2.uncertainty.data)

    # and to only second nominally unitless operands
    ndd1 = NDDataArithmetic(2,
                            uncertainty=StdDevUncertainty(2))
    ndd2 = NDDataArithmetic(300, unit='cm/m',
                            uncertainty=StdDevUncertainty(200))

    nddres2 = ndd1.power(ndd2)
    np.testing.assert_almost_equal(nddres.data,
                                   nddres2.unit.to('', nddres2.data))
    np.testing.assert_almost_equal(nddres.uncertainty.data,
                                   nddres2.unit.to('',
                                                   nddres2.uncertainty.data))

    # and to only first nominally unitless operands
    ndd1 = NDDataArithmetic(2,
                            uncertainty=StdDevUncertainty(200, unit='cm/m'))
    ndd2 = NDDataArithmetic(3, uncertainty=StdDevUncertainty(2))

    nddres2 = ndd1.power(ndd2)
    np.testing.assert_almost_equal(nddres.data, nddres2.data)
    np.testing.assert_almost_equal(nddres.uncertainty.data,
                                   nddres2.uncertainty.data)

    # and to only nominally unitless operands
    ndd1 = NDDataArithmetic(2,
                            uncertainty=StdDevUncertainty(200, unit='cm/m'))
    ndd2 = NDDataArithmetic(300, unit='cm/m',
                            uncertainty=StdDevUncertainty(200))

    nddres2 = ndd1.power(ndd2)
    np.testing.assert_almost_equal(nddres.data, nddres2.data)
    np.testing.assert_almost_equal(nddres.uncertainty.data,
                                   nddres2.uncertainty.data)


def test_compare_power_to_multiply_with_correlation():
    # First lets check if correlation for uncertainty works correctly
    ndd1 = NDDataArithmetic(7, uncertainty=StdDevUncertainty(3))
    ndd2 = NDDataArithmetic(2, uncertainty=StdDevUncertainty(6))

    ndd_power = ndd1.power(2)
    ndd_square = ndd1.multiply(ndd1, uncertainty_correlation=1)
    np.testing.assert_almost_equal(ndd_power.data, ndd_square.data)
    np.testing.assert_almost_equal(ndd_power.uncertainty.data,
                                   ndd_square.uncertainty.data)


def test_power_both_uncertainty_correlation():
    # First lets check if correlation for uncertainty works correctly
    ndd1 = NDDataArithmetic(7, uncertainty=StdDevUncertainty(3))
    ndd2 = NDDataArithmetic(2, uncertainty=StdDevUncertainty(6))

    for cor in [-1, -0.7, -0.2, 0, 0.3, 0.67, 0.75, 1]:
        ndd_power = ndd1.power(ndd2, uncertainty_correlation=cor)
        ref = 49*np.sqrt((2*3/7)**2 +
                         (np.log(7)*6)**2 +
                         (cor*2*2*np.log(7)*3*6/7))
        np.testing.assert_almost_equal(ndd_power.uncertainty.data, ref)


# Unfortunatly #4770 of astropy is probably not backported so will only
# be avaiable for astropy 1.2. So this test is marked as skipped.
@pytest.mark.xfail(not astropy_1_2, strict=True,
                   reason="dimensionless_scaled base or exponent are only "
                          "allowed from 1.2 on.")
def test_power_equivalent_units():
    # These tests are thought to ensure that the result doesn't depend on the
    # units. So 2cm yields the same result as 0.02m.

    def compare_results(result1, result2):
        if result1.unit is not None:
            data1 = result1.data * result1.unit
        else:
            data1 = result1.data * u.dimensionless_unscaled

        if result2.unit is not None:
            data2 = result2.data * result2.unit
        else:
            data2 = result2.data * u.dimensionless_unscaled

        data1 = data1.to(data2.unit)
        np.testing.assert_array_almost_equal(data1.value, data2.value)

        if result1.uncertainty.effective_unit is not None:
            data1 = (result1.uncertainty.data *
                     result1.uncertainty.effective_unit)
        else:
            data1 = result1.uncertainty.data * u.dimensionless_unscaled

        if result2.uncertainty.effective_unit is not None:
            data2 = (result2.uncertainty.data *
                     result2.uncertainty.effective_unit)
        else:
            data2 = result2.uncertainty.data * u.dimensionless_unscaled

        data1 = data1.to(data2.unit)
        np.testing.assert_array_almost_equal(data1.value, data2.value)

    # TESTS 1
    ndd1 = NDDataArithmetic(2, unit='m', uncertainty=StdDevUncertainty(0.02))
    ndd2 = NDDataArithmetic(3)
    res1 = ndd1.power(ndd2)

    ndd1 = NDDataArithmetic(2, unit='m',
                            uncertainty=StdDevUncertainty(2, unit='cm'))
    ndd2 = NDDataArithmetic(3)
    res2 = ndd1.power(ndd2)

    ndd1 = NDDataArithmetic(200., unit='cm', uncertainty=StdDevUncertainty(2.))
    ndd2 = NDDataArithmetic(3)
    res3 = ndd1.power(ndd2)

    compare_results(res1, res2)
    compare_results(res1, res3)

    # TESTS 2
    ndd1 = NDDataArithmetic(2, uncertainty=StdDevUncertainty(2))
    ndd2 = NDDataArithmetic(3, uncertainty=StdDevUncertainty(2))
    res1 = ndd1.power(ndd2)

    ndd1 = NDDataArithmetic(2, uncertainty=StdDevUncertainty(2))
    ndd2 = NDDataArithmetic(0.03, unit='m/cm',
                            uncertainty=StdDevUncertainty(0.02))
    res2 = ndd1.power(ndd2)

    ndd1 = NDDataArithmetic(0.02, unit='m/cm',
                            uncertainty=StdDevUncertainty(0.02))
    ndd2 = NDDataArithmetic(3, uncertainty=StdDevUncertainty(2))
    res3 = ndd1.power(ndd2)

    ndd1 = NDDataArithmetic(2, uncertainty=StdDevUncertainty(2))
    ndd2 = NDDataArithmetic(3,
                            uncertainty=StdDevUncertainty(0.02, unit='m/cm'))
    res4 = ndd1.power(ndd2)

    ndd1 = NDDataArithmetic(2,
                            uncertainty=StdDevUncertainty(0.02, unit='m/cm'))
    ndd2 = NDDataArithmetic(3,
                            uncertainty=StdDevUncertainty(2))
    res5 = ndd1.power(ndd2)

    ndd1 = NDDataArithmetic(200, unit='cm/m',
                            uncertainty=StdDevUncertainty(0.02, unit='m/cm'))
    ndd2 = NDDataArithmetic(3,
                            uncertainty=StdDevUncertainty(2))
    res6 = ndd1.power(ndd2)

    compare_results(res1, res2)
    compare_results(res1, res3)
    compare_results(res1, res4)
    compare_results(res1, res5)
    compare_results(res1, res6)

    # TESTS 3
    ndd1 = NDDataArithmetic(3)
    ndd2 = NDDataArithmetic(1.2, uncertainty=StdDevUncertainty(4))
    res1 = ndd1.power(ndd2)

    ndd2 = NDDataArithmetic(0.012, unit='m/cm',
                            uncertainty=StdDevUncertainty(0.04))
    res2 = ndd1.power(ndd2)

    ndd2 = NDDataArithmetic(1.2,
                            uncertainty=StdDevUncertainty(0.04, unit='m/cm'))
    res3 = ndd1.power(ndd2)

    ndd2 = NDDataArithmetic(0.012, unit='m/cm',
                            uncertainty=StdDevUncertainty(4, unit=''))
    res4 = ndd1.power(ndd2)

    ndd1 = NDDataArithmetic(300, unit='cm/m')
    ndd2 = NDDataArithmetic(0.012, unit='m/cm',
                            uncertainty=StdDevUncertainty(4, unit=''))
    res5 = ndd1.power(ndd2)

    compare_results(res1, res2)
    compare_results(res1, res3)
    compare_results(res1, res4)
    compare_results(res1, res5)

    # TESTS 4
    ndd1 = NDDataArithmetic(8, uncertainty=StdDevUncertainty(5))
    ndd2 = NDDataArithmetic(2.3)
    res1 = ndd1.power(ndd2)

    ndd1 = NDDataArithmetic(0.08, unit='m/cm',
                            uncertainty=StdDevUncertainty(0.05))
    res2 = ndd1.power(ndd2)

    ndd1 = NDDataArithmetic(8,
                            uncertainty=StdDevUncertainty(0.05, unit='m/cm'))
    res3 = ndd1.power(ndd2)

    ndd1 = NDDataArithmetic(0.08, unit='m/cm',
                            uncertainty=StdDevUncertainty(5, unit=''))
    res4 = ndd1.power(ndd2)

    compare_results(res1, res2)
    compare_results(res1, res3)
    compare_results(res1, res4)


def test_power_not_allowed_things():
    # Even with the exponent being a scalar it must not have an uncertainty
    # if the base has a unit that differs from dimensionless
    ndd1 = NDDataArithmetic(2, unit='m')
    ndd2 = NDDataArithmetic(3, uncertainty=StdDevUncertainty(0.1))
    with pytest.raises(u.UnitConversionError):
        ndd1.power(ndd2)

    ndd1 = NDDataArithmetic(2, unit='m')
    ndd2 = NDDataArithmetic(3, uncertainty=StdDevUncertainty(0.1))
    with pytest.raises(u.UnitConversionError):
        ndd1.power(ndd2)

    ndd1 = NDDataArithmetic(200., unit='cm')
    ndd2 = NDDataArithmetic(3, uncertainty=StdDevUncertainty(0.1))
    with pytest.raises(u.UnitConversionError):
        ndd1.power(ndd2)


def test_var_uncertainty_correctness():

    # Without units
    ndd1 = NDDataArithmetic(2, uncertainty=VarianceUncertainty(1))
    ndd2 = NDDataArithmetic(3, uncertainty=VarianceUncertainty(2))
    ndd_res = ndd1.add(ndd2)
    assert ndd_res.uncertainty.data == 3  # simply uncertainties added

    ndd_res = ndd1.subtract(ndd2)
    assert ndd_res.uncertainty.data == 3  # simply uncertainties added

    ndd_res = ndd1.multiply(ndd2)
    assert ndd_res.uncertainty.data == (1*3**2 + 2*2**2)

    ndd_res = ndd1.divide(ndd2)
    assert ndd_res.uncertainty.data == (1/3**2 + 2*2**2/3**4)

    ndd_res = ndd1.power(ndd2)
    ref_val = 1*(2**3*3/2)**2 + 2*(2**3*np.log(2))**2
    np.testing.assert_almost_equal(ndd_res.uncertainty.data, ref_val)

    # Correlation test for power here because power is not possible with this
    # combination of units I chose later.
    ndd_res = ndd1.power(ndd2, uncertainty_correlation=1)
    ref_val = (1*(2**3*3/2)**2 + 2*(2**3*np.log(2))**2 + 2 *
               np.sqrt(1*(2**3*3/2)**2 * 2*(2**3*np.log(2))**2))
    np.testing.assert_almost_equal(ndd_res.uncertainty.data, ref_val)

    # With correlation and parent_unit
    ndd1 = NDDataArithmetic(2, unit='m', uncertainty=VarianceUncertainty(1))
    ndd2 = NDDataArithmetic(3, unit='m', uncertainty=VarianceUncertainty(2))
    ndd_res = ndd1.add(ndd2, uncertainty_correlation=1)
    np.testing.assert_almost_equal(ndd_res.uncertainty.data,
                                   3 + 2 * np.sqrt(2))

    ndd_res = ndd1.subtract(ndd2, uncertainty_correlation=1)
    np.testing.assert_almost_equal(ndd_res.uncertainty.data,
                                   3 - 2 * np.sqrt(2))

    ndd_res = ndd1.multiply(ndd2, uncertainty_correlation=1)
    np.testing.assert_almost_equal(ndd_res.uncertainty.data,
                                   1*3**2 + 2*2**2 + 2*np.sqrt(9*8))

    ndd_res = ndd1.divide(ndd2, uncertainty_correlation=1)
    np.testing.assert_almost_equal(ndd_res.uncertainty.data,
                                   1/3**2 + 2*2**2/3**4 -
                                   2*np.sqrt(1/3**2 * 2*2**2/3**4))


@pytest.mark.parametrize(('op'), ['add', 'subtract', 'multiply', 'divide'])
@pytest.mark.parametrize(('corr'), [-1, 0.2, 0])
@pytest.mark.parametrize(('unit_uncert1'), [None, 'km*km'])
@pytest.mark.parametrize(('unit_uncert2'), [None, 'm*m'])
def test_var_compare_with_std(op, corr, unit_uncert1, unit_uncert2):
    # StdDevUncertainty is fully tested as is the converter. So it should
    # be enough to compare the result with Variance to the same calculation
    # with standard deviation.

    uncert1 = VarianceUncertainty(np.arange(3), unit=unit_uncert1)
    uncert2 = VarianceUncertainty(np.arange(2, 5), unit=unit_uncert2)
    ndd1 = NDDataArithmetic(np.array([7, 2, 6]), unit='m',
                            uncertainty=uncert1)
    ndd2 = NDDataArithmetic(np.array([5, 2, 8]), unit='cm',
                            uncertainty=uncert2)

    # control group
    ndd3 = NDDataArithmetic(ndd1, uncertainty=StdDevUncertainty(uncert1))
    ndd4 = NDDataArithmetic(ndd2, uncertainty=StdDevUncertainty(uncert2))

    ndd_var = getattr(ndd1, op)(ndd2, uncertainty_correlation=corr)
    ndd_std = getattr(ndd3, op)(ndd4, uncertainty_correlation=corr)
    var_from_var = ndd_var.uncertainty
    var_from_std = VarianceUncertainty(ndd_std.uncertainty)
    np.testing.assert_almost_equal(var_from_var.data, var_from_std.data)


# Unfortunatly #4770 of astropy is probably not backported so will only
# be avaiable for astropy 1.2. So this test is marked as skipped.
#@pytest.mark.xfail(not astropy_1_2,
#                   reason="dimensionless_scaled base or exponent are only "
#                         "allowed from 1.2 on.")
@pytest.mark.parametrize(('corr'), [-0.3, 0])
@pytest.mark.parametrize(('unit_base'), [None, u.m/u.cm])
@pytest.mark.parametrize(('unit_base_uncert'), [None, ''])
@pytest.mark.parametrize(('unit_exp'), [None, u.cm/u.mm])
@pytest.mark.parametrize(('unit_exp_uncert'), [None, u.cm/u.m])
@pytest.mark.parametrize(('exponent_has_uncert'), [False, True])
def test_var_compare_with_std_power(corr, unit_base, unit_base_uncert,
                                    unit_exp, unit_exp_uncert,
                                    exponent_has_uncert):

    # some cases only work for astropy >= 1.2. xfail them otherwise
    if ((unit_base == u.m/u.cm) and
            unit_exp is None and
            not exponent_has_uncert and
            not astropy_1_2):
        pytest.xfail("failing configuration (but should work)")

    uncert1 = VarianceUncertainty(2, unit=unit_base_uncert)
    if exponent_has_uncert:
        uncert2 = VarianceUncertainty(4, unit=unit_exp_uncert)
    else:
        uncert2 = None
    ndd1 = NDDataArithmetic(3.5, unit=unit_base, uncertainty=uncert1)
    ndd2 = NDDataArithmetic(2.7, unit=unit_exp, uncertainty=uncert2)

    # control group
    ndd3 = NDDataArithmetic(ndd1, uncertainty=StdDevUncertainty(uncert1))
    ndd4 = NDDataArithmetic(ndd2, uncertainty=StdDevUncertainty(uncert2))

    ndd_var = ndd1.power(ndd2, uncertainty_correlation=corr)
    ndd_std = ndd3.power(ndd4, uncertainty_correlation=corr)
    var_from_var = ndd_var.uncertainty
    var_from_std = VarianceUncertainty(ndd_std.uncertainty)
    np.testing.assert_allclose(var_from_var.data, var_from_std.data)
